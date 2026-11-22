"""Online pixel-wise DQN for the rl/pick corrector (pixel_Q_RL.md sec. 3), FCN Q-maps over the object crop.

Episode = one correction decision, learned online against the simulator (corrector_env.py): from the
handoff pose (gripper 7 cm above the centre of the object's segmentation box, pointing down) choose
(cell, rotation bin) on the object's 80x80 crop of the WRIST camera view (first person; the handoff itself
uses agentview), move there horizontally and rotate, hand off to pi0;
success if the pick is judged done within --timeout-s (5 s), otherwise a fail. Every episode ends after
its one decision, so the DQN target is the episode's reward and Q(s, a) estimates the success probability
of that handoff.

Chains: after a failed episode the gripper goes back to the handoff pose and a new episode starts from the
object's current state (possibly pushed or knocked over by the failed attempt), on the same scene, up to
--max-chain episodes; on success / irrecoverable state / the cap, the next chain starts from a fresh
upright start. Disturbed object states therefore come from the rollouts themselves.

Actor processes run chains with the current weights (epsilon-greedy over the valid cells); the main process
keeps the replay buffer and does the DQN updates (MSE loss on the executed entry only). Training stays on
the training split; held-out tasks are refused.

Modes
  train     online DQN (default)
  eval      greedy chains of a checkpoint on the eval starts
  baseline  no correction: pi0 from the handoff pose for the same timeout, chained the same way
  inspect   LIBERO only, no pi0: reset each start, write wrist-view mask / crop / cell back-projection PNGs
            (with the cells' 3D points projected into agentview as a cross-camera check)

Steps: see qlearning_code_instruction.md (toy check, inspect, smoke test, baseline, training).

Outputs (OUT_DIR): config.json, episodes.jsonl (one line per chain, with its episodes), train.jsonl
(per round: loss, Q, eps, success rates, buffer, object tilts), eval.jsonl, ckpt_latest.pt /
ckpt_best.pt, replay.pkl, qmaps/*.png, videos/*.mp4, inspect/*.png. With --wandb-project, the same
train / eval records (plus eval Q-maps and videos) also go to Weights & Biases (wandb/, wandb_id.txt).
"""

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import qnet as Q  # noqa: E402
from corrector_env import DEFAULT_TASKS, IneligibleStart, ResetFailed  # noqa: E402  (imports LIBERO lazily)
from replay import ReplayBuffer  # noqa: E402

FPS = 20


# ------------------------------------------------------------------ actors -------------------------

_W = {}   # per process: env, net


def _make_env(cfg):
    if cfg["env"] == "toy":
        from toy_env import ToyCorrectorEnv

        return ToyCorrectorEnv(cfg)
    from corrector_env import PickCorrectorEnv

    return PickCorrectorEnv(cfg)


def _worker(cfg):
    if "env" not in _W:
        import torch

        torch.set_num_threads(1)
        _W["env"], _W["net"] = _make_env(cfg), Q.build_qnet(cfg["width"]).eval()
    return _W["env"], _W["net"]


def run_chain(job):
    """One chain of episodes on one scene, in an actor. job: start, mode (train|eval|baseline), eps, weights,
    seed, record, save_qmap, cfg.

    Episode = one decision from the handoff pose, judged within the timeout; it always ends there (stored
    as a terminal transition, so the DQN target is the episode's reward). After a failed episode the
    chain goes on from the object's current state, until success, an irrecoverable state or max_chain
    episodes; the next chain starts from a fresh upright start."""
    cfg, start, mode = job["cfg"], job["start"], job["mode"]
    env, net = _worker(cfg)
    out = {"id": job["id"], "tag": start["tag"], "mode": mode, "eps": job["eps"], "transitions": [],
           "episodes": [], "qmap": None, "frames": None}
    t0 = time.time()
    try:
        obs = env.reset(start, record=job["record"])
    except IneligibleStart as e:          # never usable: the start is dropped
        out["skipped"] = str(e)
        return out
    except ResetFailed as e:              # this reset only: the chain is not counted
        out["reset_failed"] = str(e)
        return out
    if mode != "baseline" and job.get("weights") is not None:
        Q.load_state_dict_numpy(net, job["weights"])
    rng = np.random.default_rng(job["seed"])
    for t in range(cfg["max_chain"]):
        if mode == "baseline":
            nxt, r, info = env.baseline_step()
            out["episodes"].append(info)
        else:
            a, q, greedy = Q.select_action(net, obs, job["eps"], rng)
            nxt, r, info = env.step(a)
            info = dict(info, greedy=greedy, q_sa=float(q.reshape(-1)[a]),
                        q_max=float(q.reshape(-1)[Q.masked_argmax(q, obs["valid"])]))
            out["episodes"].append(info)
            out["transitions"].append({"obs": obs, "a": a, "r": r})
            if t == 0 and job.get("save_qmap"):
                out["qmap"] = (obs, q, a)
        if nxt is None:
            break
        obs = nxt
    rs = [e["success"] for e in out["episodes"]]
    out.update(success=any(rs), first_success=bool(rs and rs[0]), frames=env.frames, wall_s=time.time() - t0)
    return out


class Actors(object):
    """workers=0 runs episodes in this process (toy env, debugging); otherwise a spawn pool."""

    def __init__(self, cfg, workers):
        self.pool = None
        if workers > 0:
            self.pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(cfg,))

    def run(self, jobs):
        if self.pool is None:
            return [run_chain(j) for j in jobs]
        return list(self.pool.imap_unordered(run_chain, jobs, chunksize=1))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def _init_worker(cfg):
    _worker(dict(cfg, worker_id=os.getpid()))   # toy env: different random stream per worker


# ------------------------------------------------------------------ learner ------------------------

def dqn_update(net, target, opt, batch, a):
    import torch
    import torch.nn.functional as F

    dev = a.device
    x = Q.obs_batch_to_tensor(batch["rgb"], batch["depth"], batch["mask"], batch["proprio"], dev)
    act = torch.from_numpy(batch["a"]).to(dev)
    r = torch.from_numpy(batch["r"]).to(dev)
    done = torch.from_numpy(batch["done"]).to(dev)
    q_sa = net(x).flatten(1).gather(1, act[:, None]).squeeze(1)    # only the executed entry gets gradient
    if bool(batch["done"].all()):     # every episode is one decision: target = reward, no bootstrap needed
        y = r
    else:
        xn = Q.obs_batch_to_tensor(batch["next_rgb"], batch["next_depth"], batch["next_mask"],
                                   batch["next_proprio"], dev)
        with torch.no_grad():
            valid = torch.from_numpy(np.stack([Q.valid_actions(v) for v in batch["next_valid"]])).to(dev)
            qt = target(xn).flatten(1).masked_fill(~valid, float("-inf"))
            if a.double:
                an = net(xn).flatten(1).masked_fill(~valid, float("-inf")).argmax(1)
                qn = qt.gather(1, an[:, None]).squeeze(1)
            else:
                qn = qt.max(1)[0]                                       # max over 9 maps x cells in the mask
            qn = torch.where(valid.any(1), qn, torch.zeros_like(qn))
            y = r + a.gamma * (1.0 - done) * qn
    loss = F.smooth_l1_loss(q_sa, y) if a.huber else F.mse_loss(q_sa, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()
    return float(loss.detach()), float(q_sa.detach().mean()), float(y.mean())


# ------------------------------------------------------------------ helpers ------------------------

def build_starts(a):
    """(train starts, eval starts)."""
    if a.env == "toy":
        from toy_env import toy_starts

        s = toy_starts()
        return s, s
    import corrector_env as CE

    train, ev = [], []
    if "library" in a.sources:
        lib, skipped = CE.library_starts("rl/pick")
        for tag, why in skipped:
            print("skip {}: {}".format(tag, why))
        train += lib
        ev += lib
    if "tasks" in a.sources:
        free = a.mode != "train"               # eval / baseline / inspect may use any task, held-out included
        train += CE.task_starts(a.tasks, a.train_inits, allow_held_out=free)
        ev += CE.task_starts(a.tasks, a.eval_inits, allow_held_out=free)
    if not train:
        raise SystemExit("no start states")
    return train, ev


def env_cfg(a):
    return {"env": a.env, "width": a.width, "max_chain": a.max_chain, "timeout_s": a.timeout_s,
            "handoff_dz": a.handoff_dz, "handoff_jitter_cm": a.handoff_jitter_cm, "min_cell_frac": a.min_cell_frac,
            "wrist_box": a.wrist_box,
            "min_mask_px": a.min_mask_px, "env_seed": a.env_seed, "host": a.host, "port": a.port, "seed": a.seed}


def epsilon(a, episodes):
    f = min(1.0, episodes / max(1, a.eps_decay_episodes))
    return a.eps_start + f * (a.eps_end - a.eps_start)


def log_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


EPISODE_KEYS = ("k", "row", "col", "yaw_deg", "obj_tilt_deg", "success", "timeout", "chain_end", "move_steps",
                "move_reached", "move_err_m", "pi0_steps", "episode_steps", "handoff", "total_steps", "greedy",
                "q_sa", "q_max", "on_rim", "facing", "rot_ok")


def chain_record(res, extra=None):
    """One line of episodes.jsonl: a chain and its episodes."""
    rec = {"tag": res["tag"], "mode": res["mode"], "eps": res["eps"], "chain_success": res.get("success"),
           "first_success": res.get("first_success"), "n_episodes": len(res["episodes"]),
           "wall_s": round(res.get("wall_s", 0.0), 1), "skipped": res.get("skipped"),
           "reset_failed": res.get("reset_failed"),
           "episodes": [{k: e[k] for k in EPISODE_KEYS if k in e} for e in res["episodes"]]}
    rec.update(extra or {})
    return rec


def rates(results):
    """Per-episode success, first-attempt success (upright starts), chain success."""
    used = [r for r in results if not r.get("skipped") and not r.get("reset_failed")]
    eps = [e["success"] for r in used for e in r["episodes"]]
    mean = lambda v: float(np.mean(v)) if len(v) else float("nan")  # noqa: E731
    return {"episode_success": mean(eps), "first_success": mean([r["first_success"] for r in used]),
            "chain_success": mean([r["success"] for r in used]), "n_chains": len(used), "n_episodes": len(eps)}


def save_media(res, out, prefix):
    """Write the chain's Q-map PNG / video, if any; returns {"qmap": path, "video": path} of what was written."""
    paths = {}
    if res.get("qmap") is not None:
        import viz

        obs, q, act = res["qmap"]
        paths["qmap"] = out / "qmaps" / "{}_{}.png".format(prefix, res["tag"])
        viz.save_qmap(obs, q, act, str(paths["qmap"]), title="{} success={}".format(res["tag"], res.get("success")))
    if res.get("frames"):
        import imageio

        paths["video"] = out / "videos" / "{}_{}.mp4".format(prefix, res["tag"])
        imageio.mimwrite(str(paths["video"]), res["frames"], fps=FPS)
    return paths


# ------------------------------------------------------------------ wandb --------------------------

_WB = None   # the wandb module while a run is open (main process only); None = W&B logging off


def wb_init(a, out):
    """Open a W&B run when --wandb-project is set. Training runs keep their id in OUT_DIR/wandb_id.txt, so
    --resume continues the same W&B run; eval / baseline runs are separate runs in the same group."""
    global _WB
    if not a.wandb_project:
        return
    import wandb

    run_id, id_file = None, out / "wandb_id.txt"
    if a.mode == "train":
        run_id = id_file.read_text().strip() if a.resume and id_file.exists() else wandb.util.generate_id()
        id_file.write_text(run_id)
    name = a.wandb_name or (out.name if a.mode == "train" else "{}-{}".format(out.name, a.mode))
    wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=name, group=out.name, job_type=a.mode,
               tags=a.wandb_tags, config=vars(a), dir=str(out), id=run_id, resume="allow" if run_id else None,
               mode=a.wandb_mode)
    wandb.define_metric("episodes")
    for prefix in ("train/*", "eval/*"):
        wandb.define_metric(prefix, step_metric="episodes")
    wandb.define_metric("eval/episode_success", step_metric="episodes", summary="max")
    _WB = wandb


def wb_log(rec, prefix, episodes=None, extra=None):
    """Log the numeric fields of rec (nested dicts flattened to prefix/key/sub) plus extra, at x = episodes."""
    if _WB is None:
        return
    d = {}

    def flat(p, v):
        if isinstance(v, dict):
            for k, x in v.items():
                flat("{}/{}".format(p, k), x)
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            d[p] = float(v)

    for k, v in rec.items():
        flat("{}/{}".format(prefix, k), v)
    d.update(extra or {})
    if episodes is not None:
        d["episodes"] = episodes
    _WB.log(d)


def wb_finish():
    global _WB
    if _WB is not None:
        _WB.finish()
        _WB = None


def save_ckpt(path, net, target, opt, state):
    import torch

    torch.save({"net": net.state_dict(), "target": target.state_dict(), "opt": opt.state_dict(), "state": state},
               str(path))


# ------------------------------------------------------------------ modes --------------------------

def evaluate(a, actors, cfg, net, starts, out, label, episodes=None):
    """Greedy chains (or baseline chains when net is None) from the eval starts. episodes = training episodes
    so far (the W&B x value; None outside training)."""
    weights = Q.state_dict_numpy(net) if net is not None else None
    mode = "baseline" if net is None else "eval"
    jobs = []
    for rep in range(a.eval_repeats):
        for i, s in enumerate(starts):
            jobs.append({"id": len(jobs), "start": s, "mode": mode, "eps": a.eval_eps, "weights": weights, "cfg": cfg,
                         "seed": a.seed + 7919 * (len(jobs) + 1), "record": a.video and rep == 0 and i < a.media_n,
                         "save_qmap": rep == 0 and i < a.media_n})
    results = actors.run(jobs)
    media = []
    for r in results:
        log_jsonl(out / "episodes.jsonl", chain_record(r, {"label": label}))
        media.append((r, save_media(r, out, label)))
    rec = dict(rates(results), label=label, mode=mode,
               skipped=sorted({r["tag"]: r["skipped"] for r in results if r.get("skipped")}.items()),
               reset_failed=sum(bool(r.get("reset_failed")) for r in results))
    per = {}
    for r in results:
        if not r.get("skipped") and not r.get("reset_failed"):
            per.setdefault(r["tag"], []).extend(e["success"] for e in r["episodes"])
    rec["per_start_episode_success"] = {k: float(np.mean(v)) for k, v in per.items()}
    log_jsonl(out / "eval.jsonl", rec)
    if _WB is not None:
        # one key per start: wandb 0.13 queries the server for lists of images, even offline
        extra = {}
        for r, p in media:
            cap = "{} success={}".format(r["tag"], r.get("success"))
            if "qmap" in p:
                extra["eval/qmap/" + r["tag"]] = _WB.Image(str(p["qmap"]), caption=cap)
            if "video" in p:
                extra["eval/video/" + r["tag"]] = _WB.Video(str(p["video"]), caption=cap, fps=FPS, format="mp4")
        wb_log(rec, "eval", episodes, extra)      # per-start rates land in eval/per_start_episode_success/<tag>
    print("[{}] {}: episode success {:.3f} ({} episodes), first attempt {:.3f}, chain {:.3f} ({} chains)".format(
        label, mode, rec["episode_success"], rec["n_episodes"], rec["first_success"], rec["chain_success"],
        rec["n_chains"]), flush=True)
    return rec["episode_success"], results


def inspect(a, out):
    import viz
    from corrector_env import PickCorrectorEnv

    (out / "inspect").mkdir(parents=True, exist_ok=True)
    train, ev = build_starts(a)
    seen, env, rows = set(), PickCorrectorEnv(env_cfg(a)), []
    for s in train + ev:
        if s["tag"] in seen:
            continue
        seen.add(s["tag"])
        try:
            obs = env.reset(s)
        except (IneligibleStart, ResetFailed) as e:
            print("{:45s} SKIP {}: {}".format(s["tag"], type(e).__name__, e))
            rows.append({"tag": s["tag"], "skipped": str(e)})
            continue
        info = env.inspect_info()
        err = np.linalg.norm(np.array(info["cell_centres_px"]) - np.array(info["cell_points_px"]), axis=1)
        # cross-camera check: wrist cells that are >= 50 % object, projected into agentview, should hit the
        # agentview object mask (2 px tolerance; a few may be hidden from agentview, e.g. inside a mug)
        am, hits, n_obj = info["agent_mask"], 0, 0
        for (up, vp), f in zip(info["cell_points_agent_px"], info["cell_obj_frac"]):
            if f >= 0.5:
                n_obj += 1
                ui, vi = int(round(up)), int(round(vp))
                hits += bool(am[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3].any())
        agent_hit = hits / max(1, n_obj)
        viz.save_inspect(info, str(out / "inspect" / "{}.png".format(s["tag"])))
        rows.append({"tag": s["tag"], "object": info["object"], "mask_px": int(info["mask_full"].sum()),
                     "valid_cells": int(obs["valid"].sum()), "reproj_err_px_max": float(err.max()),
                     "object_cells": n_obj, "agentview_hit": round(agent_hit, 3),
                     "start_tilt_deg": round(info["start_tilt_deg"], 1), "handoff": info["handoff"],
                     "obj_tilt_deg": round(info["tilt_deg"], 1),
                     "proprio": [round(float(x), 3) for x in obs["proprio"]]})
        h = info["handoff"]
        print("{:45s} {:24s} wrist mask {:5d} px  valid cells {:3d}  object cells {:2d}  agentview hit {:.2f}  "
              "reproj err max {:.2f} px  tilt start {:.0f} -> handoff {:.0f} deg  handoff reached {} ({:.1f} cm)".format(
                  s["tag"], info["object"], rows[-1]["mask_px"], rows[-1]["valid_cells"], n_obj, agent_hit,
                  rows[-1]["reproj_err_px_max"], info["start_tilt_deg"], h["tilt_deg"], h["reached"],
                  100 * (h["pos_err_m"] or 0.0)))
    env.close()
    (out / "inspect" / "starts.json").write_text(json.dumps(rows, indent=2))


def train(a, out):
    """Online DQN. Each round, the actors run one chain each with the current weights; every episode of a
    chain is a terminal transition (s, a, r). Counts (--episodes, epsilon, eval/ckpt) are in episodes."""
    import torch

    torch.manual_seed(a.seed)
    cfg = env_cfg(a)
    train_starts, eval_starts = build_starts(a)
    net, target = Q.build_qnet(a.width).to(a.device), Q.build_qnet(a.width).to(a.device)
    target.load_state_dict(net.state_dict())
    target.eval()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    state = {"episodes": 0, "chains": 0, "grad_steps": 0, "rounds": 0, "best_eval": -1.0, "bad_tags": [],
             "width": a.width}
    buf = ReplayBuffer(a.buffer)
    if a.resume:
        ck = torch.load(str(out / "ckpt_latest.pt"), map_location=a.device)
        net.load_state_dict(ck["net"])
        target.load_state_dict(ck["target"])
        opt.load_state_dict(ck["opt"])
        state.update(ck["state"])
        if (out / "replay.pkl").exists():
            buf = ReplayBuffer.load(str(out / "replay.pkl"))
        print("resumed at episode {}, {} transitions".format(state["episodes"], len(buf)))
    bad = set(state["bad_tags"])
    rng = np.random.default_rng(a.seed + state["episodes"])
    actors = Actors(cfg, a.workers)
    t_start = time.time()
    try:
        while state["episodes"] < a.episodes:
            pool = [s for s in train_starts if s["tag"] not in bad]
            if not pool:
                raise SystemExit("every start state was skipped as ineligible")
            weights = Q.state_dict_numpy(net)
            eps_now = epsilon(a, state["episodes"])
            jobs = [{"id": i, "start": pool[rng.integers(len(pool))], "mode": "train", "eps": eps_now,
                     "weights": weights, "cfg": cfg, "seed": int(rng.integers(2 ** 31)), "record": False,
                     "save_qmap": False} for i in range(max(1, a.workers))]
            results = actors.run(jobs)
            ep_before, n_new = state["episodes"], 0
            for r in results:
                if r.get("reset_failed"):
                    print("reset failed {}: {}".format(r["tag"], r["reset_failed"]), flush=True)
                    log_jsonl(out / "episodes.jsonl", chain_record(r, {"chain": None}))
                    continue
                if r.get("skipped"):
                    bad.add(r["tag"])
                    print("skip {}: {}".format(r["tag"], r["skipped"]), flush=True)
                    log_jsonl(out / "episodes.jsonl", chain_record(r, {"chain": None}))
                    continue
                for tr in r["transitions"]:            # every episode ends after its decision: terminal
                    buf.add(tr["obs"], tr["a"], tr["r"], None, True, {"tag": r["tag"]})
                    n_new += 1
                state["episodes"] += len(r["episodes"])
                state["chains"] += 1
                log_jsonl(out / "episodes.jsonl", chain_record(r, {"chain": state["chains"],
                                                                   "episode": state["episodes"]}))
            state["bad_tags"] = sorted(bad)

            losses = []
            if len(buf) >= max(a.warmup, a.batch):
                net.train()
                for _ in range(int(round(a.utd * n_new))):
                    losses.append(dqn_update(net, target, opt, buf.sample(a.batch, rng), a))
                    state["grad_steps"] += 1
                    if state["grad_steps"] % a.target_every == 0:
                        target.load_state_dict(net.state_dict())
                net.eval()
            state["rounds"] += 1
            done_res = [r for r in results if not r.get("skipped") and not r.get("reset_failed")]
            rec = dict(rates(results), round=state["rounds"], episodes=state["episodes"], chains=state["chains"],
                       grad_steps=state["grad_steps"], eps=round(eps_now, 4),
                       elapsed_min=round((time.time() - t_start) / 60, 2), buffer=buf.stats(),
                       decision_tilt_deg=[round(e.get("obj_tilt_deg", 0.0), 1) for r in done_res
                                          for e in r["episodes"]])
            if losses:
                L = np.array(losses)
                rec.update(loss=float(L[:, 0].mean()), q_sa=float(L[:, 1].mean()), target=float(L[:, 2].mean()))
            log_jsonl(out / "train.jsonl", rec)
            if _WB is not None:
                tilts = rec["decision_tilt_deg"]
                wb_log(rec, "train", state["episodes"],
                       {"train/decision_tilt_hist": _WB.Histogram(tilts)} if tilts else None)
            if state["rounds"] % a.print_every == 0:
                print("round {round} ep {episodes} chains {chains} grad {grad_steps} eps {eps} loss {l} q {q} "
                      "buf {b} success(episodes in buffer) {s:.3f} {elapsed_min} min".format(
                          l="{:.4f}".format(rec["loss"]) if losses else "-",
                          q="{:.3f}".format(rec["q_sa"]) if losses else "-", b=len(buf),
                          s=rec["buffer"].get("reward_rate", 0.0), **rec), flush=True)

            ep = state["episodes"]
            if a.eval_every and ep // a.eval_every > ep_before // a.eval_every:
                rate, _ = evaluate(a, actors, cfg, net, [s for s in eval_starts if s["tag"] not in bad], out,
                                   "eval_ep{}".format(ep), episodes=ep)
                if rate > state["best_eval"]:
                    state["best_eval"] = rate
                    save_ckpt(out / "ckpt_best.pt", net, target, opt, state)
                wb_log({"best_episode_success": state["best_eval"]}, "eval", ep)
            if ep // a.ckpt_every > ep_before // a.ckpt_every or ep >= a.episodes:
                save_ckpt(out / "ckpt_latest.pt", net, target, opt, state)
                if a.save_buffer:
                    buf.save(str(out / "replay.pkl"))
    finally:
        save_ckpt(out / "ckpt_latest.pt", net, target, opt, state)
        actors.close()


def eval_mode(a, out, baseline):
    import torch

    cfg = env_cfg(a)
    _, eval_starts = build_starts(a)
    net = None
    if not baseline:
        ck = torch.load(str(a.ckpt or out / "ckpt_best.pt"), map_location=a.device)
        cfg["width"] = ck["state"].get("width", a.width)          # the network width the checkpoint was trained with
        net = Q.build_qnet(cfg["width"]).to(a.device)
        net.load_state_dict(ck["net"])
        net.eval()
    actors = Actors(cfg, a.workers)
    try:
        evaluate(a, actors, cfg, net, eval_starts, out, "baseline" if baseline else "eval")
    finally:
        actors.close()


# ------------------------------------------------------------------ main ---------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", default="train", choices=["train", "eval", "baseline", "inspect"])
    p.add_argument("--env", default="libero", choices=["libero", "toy"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ckpt", default=None, help="eval mode: checkpoint (default OUT_DIR/ckpt_best.pt)")
    # starts
    p.add_argument("--sources", nargs="+", default=["library", "tasks"], choices=["library", "tasks"])
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="<suite>:<ids>, ids like 9,10 or 0-9")
    p.add_argument("--train-inits", type=int, nargs="+", default=list(range(0, 40)))
    p.add_argument("--eval-inits", type=int, nargs="+", default=[40, 41])
    # corrector step
    p.add_argument("--timeout-s", type=float, default=5.0,
                   help="judge window per episode, from the decision (move + pi0); no success by then = fail")
    p.add_argument("--max-chain", type=int, default=5,
                   help="episodes per scene: after a failed episode the next starts from the object's current "
                        "state (re-handoff); after this many, or on success / irrecoverable, a fresh start")
    p.add_argument("--handoff-dz", type=float, default=0.07,
                   help="handoff pose: gripper this far above the object top, pointing down (m)")
    p.add_argument("--handoff-jitter-cm", type=float, default=0.0,
                   help="optional uniform xy jitter of the handoff pose (the VLM agent's placement error)")
    p.add_argument("--min-cell-frac", type=float, default=0.0,
                   help="object-pixel fraction a cell needs to be a valid action (0 = every cell of the 80x80 crop)")
    p.add_argument("--min-mask-px", type=int, default=20)
    p.add_argument("--wrist-box", type=float, default=150,
                   help="side (px of the 256 px wrist image) of the object-centred box the 80x80 Q-map crop spans")
    p.add_argument("--env-seed", type=int, default=7)
    # DQN
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.5,
                   help="unused while every episode is terminal (one decision per episode); kept for TD variants")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--buffer", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=64, help="transitions before the first update")
    p.add_argument("--utd", type=float, default=1.0,
                   help="gradient steps per new transition (CPU: ~0.04 s per step at width 32, batch 32)")
    p.add_argument("--target-every", type=int, default=200, help="gradient steps between target copies")
    p.add_argument("--double", action="store_true", help="double-DQN target (pixel_Q_RL.md uses the plain max)")
    p.add_argument("--huber", action="store_true", help="Huber instead of squared TD loss")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--eps-decay-episodes", type=int, default=1000)
    p.add_argument("--device", default="cpu", help="learner device (the LIBERO venv's torch 1.11+cu113 has no kernels for the RTX 5090, sm_120)")
    # runtime
    p.add_argument("--workers", type=int, default=None, help="actor processes (default: 0 for toy, 6 for libero)")
    p.add_argument("--eval-every", type=int, default=100, help="training episodes between evals (0 = never)")
    p.add_argument("--eval-repeats", type=int, default=2)
    p.add_argument("--eval-eps", type=float, default=0.0)
    p.add_argument("--media-n", type=int, default=3, help="eval starts that get a Q-map PNG (and a video)")
    p.add_argument("--video", action="store_true", help="record wrist | agentview mp4s of the first eval starts")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--save-buffer", action="store_true", default=True)
    p.add_argument("--no-save-buffer", dest="save_buffer", action="store_false")
    p.add_argument("--print-every", type=int, default=1)
    # Weights & Biases (off unless --wandb-project is given; the JSONL logs are written either way)
    p.add_argument("--wandb-project", default=None, help="W&B project; enables W&B logging")
    p.add_argument("--wandb-entity", default=None, help="W&B team / user (default: your default entity)")
    p.add_argument("--wandb-name", default=None, help="run name (default: OUT_DIR's name, + -eval / -baseline)")
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"],
                   help="offline: write to OUT_DIR/wandb and upload later with `wandb sync`")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.workers is None:
        a.workers = 0 if a.env == "toy" else 6
    return a


def main(argv=None):
    a = parse_args(argv)
    out = pathlib.Path(a.out_dir)
    for sub in ("qmaps", "videos"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if a.mode != "inspect":
        (out / ("config.json" if a.mode == "train" else "config_{}.json".format(a.mode))).write_text(
            json.dumps(vars(a), indent=2))
    if a.mode == "inspect":
        if a.env != "libero":
            raise SystemExit("inspect needs --env libero")
        inspect(a, out)
        return
    wb_init(a, out)
    try:
        if a.mode == "train":
            train(a, out)
        else:
            eval_mode(a, out, baseline=a.mode == "baseline")
    finally:
        wb_finish()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
