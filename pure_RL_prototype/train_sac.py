"""Train the always-on residual policy on top of frozen pi0 with SAC (bowl / mug pick).

Reward 1 when the target bowl or mug is grasped and lifted 5 cm (fa_core PickCheck), which ends the episode.
One residual decision per pi0 chunk (residual_env.ResidualPickEnv.chunk_episode): the policy sees wrist +
agentview (64 px), pi0 state and pi0's 5 actions, and offsets those actions by eps * u, u in [-1, 1]^6.

Training starts (sampled per episode):
  fail   library.json rl/pick reset states with a bowl / mug target (pre-failure states), probability --p-fail
  task   task init states of the bowl / mug tasks (pixel-Q's list: LIBERO-90 training split + libero_spatial),
         --train-inits; most of these pi0 already solves, which teaches the residual to stay near 0 there
Evaluation (deterministic policy, tanh(mean)), every --eval-every episodes, compared with plain pi0 (u = 0)
on exactly the same episodes:
  fail   the fail starts, --eval-fail-n episodes each (the same states as training: tracks the target failures)
  task   the task list at --eval-inits (layouts never trained on): generalization and do-no-harm

Actor processes run episodes with the newest weights (a spawn pool, as in pixel-Q); the learner (CPU, see
sac.py) updates while they run, at most --utd updates per collected decision.

Output (--out-dir): config.json, train.jsonl (per episode), learner.jsonl, eval.jsonl, ckpt_latest.pt,
ckpt_best.pt, videos/eval<k>/... Run in the container's LIBERO venv with the pi0 server up:

  cd /home/DockerShared/ECE780/pure_RL_prototype
  MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python train_sac.py --out-dir runs/sac_v0
Smoke test:
  ... train_sac.py --out-dir runs/sac_smoke --episodes 24 --warmup 50 --eval-every 12 --eval-fail-n 1 \
      --eval-tasks libero_spatial:0-1 --eval-inits 40 --workers 4
"""

import argparse
import json
import multiprocessing as mp
import pathlib
import time
from collections import deque

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_TASKS = "libero_90:9,10,14,17,25,31,34,37,63,64,67,71,72,85;libero_spatial:0-9"

_W = {}


# ------------------------------------------------------------------ actor processes -----------------

def _init_worker(cfg, counter):
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    import residual_env as RE
    import sac

    cfg = dict(cfg, worker_id=wid)
    _W.update(RE=RE, cfg=cfg, env=RE.ResidualPickEnv(cfg), policy=sac.Policy(RE.LOW_DIM, RE.POLICY_ACT_DIM))


def run_episode(job):
    """job: kind (train | eval | pi0), start, group, seed, weights (None for pi0), record, video (path or None)."""
    import torch

    RE, cfg, env, policy = _W["RE"], _W["cfg"], _W["env"], _W["policy"]
    torch.manual_seed(job["seed"])          # every spawned process starts from the same torch seed
    if job["kind"] == "pi0":
        act = lambda pobs: np.zeros(RE.POLICY_ACT_DIM)      # noqa: E731  (u = 0: exactly pi0)
    else:
        policy.load(job["weights"])
        det = job["kind"] == "eval"
        act = lambda pobs: policy.act(pobs, deterministic=det)  # noqa: E731
    out = {"kind": job["kind"], "group": job["group"], "tag": job["start"]["tag"], "eval_id": job.get("eval_id"),
           "key": _start_key(job["start"])}
    try:
        trans, info = env.chunk_episode(job["start"], act, cfg["eps_vec"], cfg["horizon"], record=bool(job["video"]))
    except (RE.IneligibleStart, RE.ResetFailed) as e:
        out["ineligible"] = "{}: {}".format(type(e).__name__, e)
        return out
    except Exception:  # keep training alive; the episode is lost
        import traceback

        out["error"] = traceback.format_exc(limit=4)
        return out
    frames = info.pop("frames")
    if job["video"] and frames:
        import imageio

        path = pathlib.Path(job["video"] + ("_success" if info["success"] else "_fail") + ".mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(path), frames, fps=20)
    out.update(info=info, transitions=trans if job["kind"] == "train" else None)
    return out


# ------------------------------------------------------------------ helpers -------------------------

def _start_key(start):
    """Ineligibility is a property of the task (its pick target) for task inits, of the state otherwise."""
    if start["kind"] == "task_init":
        return "{}_task{}".format(start["suite"], start["task_id"])
    return start["tag"]


def _ids(text):
    return [int(x) for part in str(text).split(",") if part.strip()
            for x in (range(int(part.split("-")[0]), int(part.split("-")[1]) + 1) if "-" in part else [part])]


def _log(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _rates(results):
    """{group: {"n", "success", "rate"}} over episode results (ineligible / errors excluded)."""
    out = {}
    for r in results:
        if "info" not in r:
            continue
        g = out.setdefault(r["group"], {"n": 0, "success": 0})
        g["n"] += 1
        g["success"] += int(r["info"]["success"])
    for g in out.values():
        g["rate"] = round(g["success"] / g["n"], 3) if g["n"] else None
    return out


def build_starts(a, RE):
    targets = [w.strip().lower() for w in a.targets.split(",") if w.strip()]
    fail, _ = RE.SC.library_starts(a.policy)
    fail = [s for s in fail if any(w in str(s.get("word")).lower() for w in targets)]
    tasks = [t for t in a.tasks.split(";") if t.strip()]
    train_tasks = RE.SC.task_starts(tasks, _ids(a.train_inits))
    eval_tasks = RE.SC.task_starts([t for t in (a.eval_tasks or a.tasks).split(";") if t.strip()], _ids(a.eval_inits))
    return targets, fail, train_tasks, eval_tasks


# ------------------------------------------------------------------ main ----------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume", action="store_true", help="continue from ckpt_latest.pt (the replay restarts empty)")
    # residual / env
    p.add_argument("--eps", type=float, default=0.1, help="residual bound, OSC input units (1.0 = 5 cm / step)")
    p.add_argument("--rot-scale", type=float, default=0.5)
    p.add_argument("--horizon", type=int, default=200, help="control steps per episode (20 Hz)")
    p.add_argument("--targets", default="bowl,mug")
    p.add_argument("--policy", default="rl/pick")
    p.add_argument("--tasks", default=DEFAULT_TASKS, help="';'-separated suite:ids for task-init starts")
    p.add_argument("--train-inits", default="0-39")
    p.add_argument("--p-fail", type=float, default=0.3, help="probability a training episode starts from a fail state")
    # SAC
    p.add_argument("--episodes", type=int, default=4000, help="training episodes")
    p.add_argument("--utd", type=float, default=0.5, help="max learner updates per collected decision")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2000, help="decisions in the replay before updates start")
    p.add_argument("--capacity", type=int, default=60000)
    p.add_argument("--gamma", type=float, default=0.97, help="per decision (5 control steps)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--init-alpha", type=float, default=0.1)
    p.add_argument("--target-entropy", type=float, default=None, help="default -6")
    p.add_argument("--act-penalty", type=float, default=0.005, help="reward -= act_penalty * mean(u^2)")
    p.add_argument("--init-log-std", type=float, default=0.0)
    p.add_argument("--threads", type=int, default=8, help="learner torch threads")
    # eval
    p.add_argument("--eval-every", type=int, default=300, help="training episodes between evaluations")
    p.add_argument("--eval-fail-n", type=int, default=8)
    p.add_argument("--eval-tasks", default=None, help="default: --tasks")
    p.add_argument("--eval-inits", default="40,41")
    # infra
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=7)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    import torch
    import residual_env as RE
    import sac

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    targets, fail_starts, train_tasks, eval_tasks = build_starts(a, RE)
    eps_vec = RE.eps_vector(a.eps, rot_scale=a.rot_scale)
    cfg = {"eps_vec": eps_vec.tolist(), "horizon": a.horizon, "targets": targets, "env_seed": a.env_seed,
           "seed": a.seed, "host": a.host, "port": a.port}
    (out / "config.json").write_text(json.dumps(dict(vars(a), eps_vec=eps_vec.tolist(),
                                                     fail_starts=[s["tag"] for s in fail_starts],
                                                     n_train_tasks=len(train_tasks), n_eval_tasks=len(eval_tasks)),
                                                indent=2))
    print("starts: fail {}, train task inits {}, eval task inits {}; eps {}".format(
        len(fail_starts), len(train_tasks), len(eval_tasks), eps_vec.round(3).tolist()), flush=True)

    L = sac.Learner(RE.LOW_DIM, RE.POLICY_ACT_DIM, lr=a.lr, gamma=a.gamma, tau=a.tau, init_alpha=a.init_alpha,
                    target_entropy=a.target_entropy, act_penalty=a.act_penalty, threads=a.threads,
                    init_log_std=a.init_log_std)
    R = sac.Replay(a.capacity, RE.LOW_DIM, RE.POLICY_ACT_DIM, (6, RE.IMG, RE.IMG))
    ep, decisions, n_eval, best = 0, 0, 0, -1.0
    if a.resume and (out / "ckpt_latest.pt").exists():
        ck = torch.load(out / "ckpt_latest.pt")
        L.load(ck["learner"])
        ep, decisions, n_eval, best = ck["episodes"], ck["decisions"], ck["n_eval"], ck["best"]
        print("resumed at episode {} ({} updates); replay starts empty".format(ep, L.updates), flush=True)

    dead = set()                     # _start_key of ineligible starts (never sampled again)
    train_pool = {"fail": fail_starts, "task": train_tasks}

    def sample_start():
        for _ in range(1000):
            g = "fail" if (rng.random() < a.p_fail and train_pool["fail"]) else "task"
            s = train_pool[g][rng.integers(len(train_pool[g]))]
            if _start_key(s) not in dead:
                return g, s
        raise SystemExit("no eligible training starts left")

    counter = mp.get_context("spawn").Value("i", 0)
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(cfg, counter))
    try:
        _loop(a, out, rng, L, R, pool, fail_starts, eval_tasks, dead, sample_start, ep, decisions,
              n_eval, best, baseline_path=out / "eval_pi0.json")
    finally:
        pool.terminate()     # also on Ctrl-C / errors: no orphaned workers
        pool.join()


def _loop(a, out, rng, L, R, pool, fail_starts, eval_tasks, dead, sample_start, ep, decisions,
          n_eval, best, baseline_path):
    import torch

    inflight, evals = [], {}                     # evals: eval_id -> {"pending", "results", "episodes"}
    job_seed = [int(rng.integers(1 << 30))]

    def submit(job):
        job_seed[0] += 1
        job["seed"] = job_seed[0]
        inflight.append(pool.apply_async(run_episode, (job,)))

    def submit_eval(kind, eval_id, weights):
        jobs = [("fail", s, k) for s in fail_starts for k in range(a.eval_fail_n)] + [("task", s, 0) for s in eval_tasks]
        for g, s, k in jobs:
            if _start_key(s) in dead:
                continue
            video = None
            if g == "fail" and k == 0:
                video = str(out / "videos" / "eval{}_{}".format(eval_id, kind) / s["tag"])
            submit({"kind": kind, "group": g, "start": s, "weights": weights, "eval_id": eval_id, "video": video})
            evals.setdefault(eval_id, {"pending": 0, "results": [], "episodes": ep, "kind": kind})["pending"] += 1

    # plain pi0 on the eval episodes first (once): the baseline every evaluation is compared with
    base_path = baseline_path
    baseline = json.loads(base_path.read_text()) if base_path.exists() else None
    if baseline is None:
        submit_eval("pi0", "pi0", None)

    t0, last_print = time.time(), time.time()
    recent = deque(maxlen=100)
    stats, last_avg = [], {}
    submitted = ep
    budget_done = False
    while True:
        # keep the workers busy with training episodes (evaluations queue in between)
        while len(inflight) < a.workers and submitted < a.episodes:
            g, s = sample_start()
            submit({"kind": "train", "group": g, "start": s, "weights": L.weights(), "video": None})
            submitted += 1
        ready = [r for r in inflight if r.ready()]
        for r in ready:
            inflight.remove(r)
            res = r.get()
            if "ineligible" in res or "error" in res:
                if "ineligible" in res:
                    dead.add(res["key"])
                print("{} {}: {}".format(res["kind"], res["tag"], res.get("ineligible") or res.get("error")), flush=True)
            if res["kind"] == "train":
                if "info" in res:
                    ep += 1
                    for t in res["transitions"]:
                        R.add(t)
                    decisions += len(res["transitions"])
                    recent.append((res["group"], res["info"]["success"]))
                    _log(out / "train.jsonl", dict(ep=ep, group=res["group"], start=res["tag"], updates=L.updates,
                                                   **res["info"]))
                else:
                    submitted -= 1           # replace the lost episode
            else:
                e = evals[res["eval_id"]]
                e["pending"] -= 1
                e["results"].append(res)
                if e["pending"] == 0:
                    rates = _rates(e["results"])
                    if e["kind"] == "pi0":
                        baseline = rates
                        base_path.write_text(json.dumps(baseline, indent=2))
                        print("pi0 baseline:", json.dumps(baseline), flush=True)
                    else:
                        rec = {"eval": res["eval_id"], "episodes": e["episodes"], "updates": L.updates,
                               "residual": rates, "pi0": baseline,
                               "per_start": {}}
                        for x in e["results"]:
                            if "info" in x and x["group"] == "fail":
                                d = rec["per_start"].setdefault(x["tag"], [0, 0])
                                d[0] += 1
                                d[1] += int(x["info"]["success"])
                        tot = sum(v["success"] for v in rates.values()) / max(1, sum(v["n"] for v in rates.values()))
                        rec["overall"] = round(tot, 3)
                        _log(out / "eval.jsonl", rec)
                        print("EVAL {} @ep {}: residual {} | pi0 {}".format(
                            res["eval_id"], e["episodes"], {g: v["rate"] for g, v in rates.items()},
                            {g: v["rate"] for g, v in (baseline or {}).items()}), flush=True)
                        state = {"learner": L.state(), "episodes": ep, "decisions": decisions, "n_eval": n_eval,
                                 "best": best, "args": vars(a)}
                        if tot > best:
                            best = tot
                            state["best"] = best
                            torch.save(state, out / "ckpt_best.pt")
                        torch.save(state, out / "ckpt_latest.pt")
                    del evals[res["eval_id"]]
            # evaluation schedule (by finished training episodes)
            if res["kind"] == "train" and "info" in res and ep % a.eval_every == 0 and ep < a.episodes:
                n_eval += 1
                submit_eval("eval", n_eval, L.weights())
        # learner: updates while the actors run, within the UTD budget
        did = False
        if R.n >= a.warmup and L.updates < a.utd * decisions:
            stats.append(L.update(R.sample(a.batch, rng)))
            did = True
            if len(stats) == 100:
                last_avg = {k: round(float(np.mean([s[k] for s in stats])), 4) for k in stats[0]}
                _log(out / "learner.jsonl", dict(updates=L.updates, episodes=ep, replay=R.n, **last_avg))
                stats = []
        if time.time() - last_print > 60:
            last_print = time.time()
            by = {g: [s for gg, s in recent if gg == g] for g in ("fail", "task")}
            print("[{:.0f} min] ep {}/{} updates {} replay {} | last100 success fail {} task {} | {}".format(
                (time.time() - t0) / 60, ep, a.episodes, L.updates, R.n,
                "{:.2f}".format(np.mean(by["fail"])) if by["fail"] else "-",
                "{:.2f}".format(np.mean(by["task"])) if by["task"] else "-",
                " ".join("{}={}".format(k, v) for k, v in last_avg.items())), flush=True)
        if ep >= a.episodes and not inflight and not budget_done:
            budget_done = True
            n_eval += 1
            submit_eval("eval", n_eval, L.weights())      # final evaluation
        if budget_done and not inflight:
            break
        if not did and not ready:
            time.sleep(0.05)
    print("done: {} episodes, {} updates, best eval overall {:.3f}".format(ep, L.updates, best), flush=True)


if __name__ == "__main__":
    main()
