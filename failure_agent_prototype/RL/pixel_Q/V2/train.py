"""Pixel-wise Q-learning V2: a learned trigger for the low-level corrector (pixel_Q_RL_V2.md).

The Q-map (qnet.py) values small local corrections from a gripper-centred heightmap, plus the no-op entry Q0
(let pi0 continue). Data come from BRANCH POINTS on pi0 main rollouts from task init states (rollout.py): at a
random gated tick the sim is snapshotted, a no-op branch (pi0 alone, H_max) labels Q0, and an intervention branch
(correction, then pi0 alone, H_max) labels that correction; both from the same snapshot. Labels are
gamma^(dt in s) for a pick within H_max, else 0; all terminal, so the loss is MSE on the labelled entry per head.

Modes (--mode)
  train --phase 0   levelling check (sec. 9.5): the intervention branch levels only (A0), branches run 10 s to log
                    success times. probe_log.jsonl, phase0_report.json, replay_phase0.pkl (no-op labels: Q0 warm
                    start for phase 1). No network is trained.
  train --phase 1   train the Q-map from branch points (epsilon-greedy corrections); replay prefilled from phase 0
  train --phase 2   fine-tune with the learned trigger acting on the main rollout (tau from calibrate or --tau)
  eval-branches     greedy branch points of a checkpoint on the eval starts (what training evaluates)
  calibrate         Q0 reliability, noise floor -> tau_noise, tau sweep (success vs interventions) -> tau
  evaluate          conditions pi0 | hl | trigger | random on the eval starts (held-out tasks allowed here)
  timing            A(s) along pi0 rollouts that the trigger does not act on: false-trigger rate, lead time
  phase0-report     recompute phase0_report.json from probe_log.jsonl (e.g. with another --h-max-s)
  inspect           LIBERO, no pi0: scripted poses near the target -> heightmap PNGs + geometry stats
  restore-check     LIBERO, no pi0: snapshot / restore of a running rollout must be exact
  finger-mask       LIBERO, no pi0: fixed wrist-image finger mask for --robot-mask fixed

Steps and outputs: see qlearning_code_instruction_V2.md.
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for _p in (HERE.parent, HERE):              # pixel_Q/ (the shared package), then this folder (V2/) first
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import evaluation as EV  # noqa: E402
import qnet as Q  # noqa: E402
import rollout as RO  # noqa: E402
from replay import ReplayBuffer  # noqa: E402
from shared import runlog  # noqa: E402
from shared.runlog import wb_log  # noqa: E402
from shared.scene import DEFAULT_TASKS  # noqa: E402

FPS = 20


def steps(sec):
    return int(round(float(sec) * RO.CTRL_HZ))


def clean(x):
    """JSON-safe copy (numpy scalars / arrays -> Python)."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def log_jsonl(path, rec):
    runlog.log_jsonl(path, clean(rec))


# ------------------------------------------------------------------ actors -------------------------

_W = {}   # per process: env (backend), scorer


def _make_backend(cfg):
    if cfg["env"] == "toy":
        from toy_env import ToyBranchEnv

        return ToyBranchEnv(cfg)
    from libero_env import BranchingPickEnv

    return BranchingPickEnv(cfg)


def _worker(cfg):
    if "env" not in _W:
        _W["env"], _W["scorer"] = _make_backend(cfg), Q.Scorer(cfg["width"], cfg["heads"], cfg["kappa"])
    return _W["env"], _W["scorer"]


def _init_worker(cfg):
    import torch

    torch.set_num_threads(1)                # actor processes only; in-process actors (--workers 0) share the learner's
    _worker(dict(cfg, worker_id=os.getpid() % 100000))


def run_job(job):
    """One main rollout in an actor. job: id, start, cfg, rc (rollout config), weights (None = no Q-map), kappa,
    seed, record, keep_obs, hl (HL module for this job, None = the cfg's), label."""
    env, scorer = _worker(job["cfg"])
    if job.get("hl"):
        env.set_hl(job["hl"])
    scorer.kappa = job["kappa"]
    scorer.loaded = False
    scorer.load(job.get("weights"))
    rng = np.random.default_rng(job["seed"])
    res = RO.main_rollout(env, job["start"], job["rc"], scorer, rng, record=job.get("record", False))
    res.pop("start", None)
    res["id"], res["label"], res["media"] = job["id"], job.get("label"), bool(job.get("media"))
    if not job.get("keep_obs"):
        res["samples"] = res["samples"][:1] if job.get("media") else []
    return res


def make_actors(cfg, workers):
    return runlog.ActorPool(run_job, workers, _init_worker, (cfg,))


def make_jobs(a, starts, repeats, cfg, rc, weights, label, media=0, hl=None, seed_base=0):
    jobs = []
    for rep in range(repeats):
        for i, s in enumerate(starts):
            m = rep == 0 and i < media
            jobs.append({"id": len(jobs), "start": s, "cfg": cfg, "rc": dict(rc, push_samples=True) if m else rc,
                         "weights": weights, "kappa": a.kappa, "seed": a.seed + seed_base + 7919 * (len(jobs) + 1),
                         "record": bool(a.video and m), "media": m, "label": label, "hl": hl, "keep_obs": False})
    return jobs


# ------------------------------------------------------------------ config -------------------------

def build_starts(a):
    """(train starts, eval starts). Training never loads held-out tasks; the other modes may (zero-shot eval)."""
    if a.env == "toy":
        from toy_env import toy_starts

        return toy_starts(16, "toy"), toy_starts(8, "toyeval")
    from shared import scene as SC

    free = a.mode != "train"
    train = SC.task_starts(a.tasks, a.train_inits, allow_held_out=free)
    ev = SC.task_starts(a.eval_tasks or a.tasks, a.eval_inits, allow_held_out=free)
    return train, ev


def env_cfg(a, width=None, heads=None):
    return {"env": a.env, "width": width or a.width, "heads": heads or a.heads, "kappa": a.kappa, "reach": a.reach,
            "gate_margin_m": a.gate_margin_cm / 100.0, "gate_min_gap": a.gate_min_gap, "min_mask_px": a.min_mask_px,
            "agent_px": a.agent_px, "wrist_px": a.wrist_px, "fill_holes": a.fill_holes, "robot_mask": a.robot_mask,
            "finger_mask": a.finger_mask, "mask_source": a.mask_source, "mix_p": a.mix_p, "box_jitter": a.box_jitter,
            "clear_min": a.clear_min_cm / 100.0, "clear_to": a.clear_to_cm / 100.0, "hl": a.hl,
            "vlm_model": a.vlm_model, "env_seed": a.env_seed, "seed": a.seed, "host": a.host, "port": a.port,
            "same_noise": a.same_noise}


def gate_m(a):
    return a.reach * Q.CELL_M + a.gate_margin_cm / 100.0


def rollout_cfg(a, kind, eps=None, tau=None, random_p=None, until=None):
    h = steps(a.h_max_s)
    rc = {"h_max_steps": h, "branch_cap_steps": h, "gamma": a.gamma, "prim_cap": a.prim_cap,
          "cooldown_steps": steps(a.cooldown_s), "n_persist": a.persist, "branch_kappa": a.branch_kappa,
          "until": until or "pick"}
    if kind == "phase0":
        rc.update(p_branch=a.p_branch, branch_action="probe", branch_cap_steps=steps(a.phase0_cap_s))
    elif kind in ("phase1", "phase2"):
        rc.update(p_branch=a.p_branch, branch_action="eps", eps=eps)
        if kind == "phase2":
            rc.update(trigger="learned", tau=tau)
    elif kind == "eval_branches":
        rc.update(p_branch=a.eval_p_branch, branch_action="greedy", push_samples=False)
    elif kind == "deploy":
        rc.update(p_branch=0.0, push_samples=False)
        if random_p is not None:
            rc.update(trigger="random", random_p=random_p)
        elif tau is not None and np.isfinite(tau):
            rc.update(trigger="learned", tau=tau)
    elif kind == "timing":
        rc.update(p_branch=0.0, log_ticks=True, push_samples=False)
    else:
        raise ValueError(kind)
    return rc


def epsilon(a, n):
    f = min(1.0, n / float(max(1, a.eps_decay_branches)))
    return a.eps_start + f * (a.eps_end - a.eps_start)


def load_tau(a, out):
    if a.tau is not None:
        return float(a.tau)
    p = out / "calibration.json"
    if p.exists():
        t = json.loads(p.read_text()).get("tau_recommended")
        if t is not None:
            return float(t)
    return None


def load_net(a, out):
    """Network of --ckpt (default OUT_DIR/ckpt_best.pt, else ckpt_latest.pt); width / heads from the checkpoint."""
    import torch

    path = pathlib.Path(a.ckpt) if a.ckpt else (out / "ckpt_best.pt" if (out / "ckpt_best.pt").exists()
                                                 else out / "ckpt_latest.pt")
    ck = torch.load(str(path), map_location="cpu")
    st = ck["state"]
    net = Q.build_qnet(st.get("width", a.width), st.get("heads", a.heads))
    net.load_state_dict(ck["net"])
    net.eval()
    return net, st, str(path)


def save_ckpt(path, net, opt, state):
    import torch

    torch.save({"net": net.state_dict(), "opt": None if opt is None else opt.state_dict(), "state": state}, str(path))


# ------------------------------------------------------------------ learner ------------------------

def q_update(net, opt, batch, a, rng):
    """One gradient step: MSE (or Huber) on the labelled entry of every head (bootstrap masks with --bootstrap-p
    < 1); gradient norm clipped at 10. Returns stats."""
    import torch
    import torch.nn.functional as F

    x = Q.obs_batch_to_tensor(batch["rgb"], batch["height"], batch["mask"], batch["observed"], batch["proprio"],
                              a.device)
    act = torch.from_numpy(batch["a"]).to(a.device)
    y = torch.from_numpy(batch["y"]).to(a.device)
    q = net(x)                                                     # (B, K, 9, 11, 11)
    B, K = q.shape[0], q.shape[1]
    qa = q.reshape(B, K, -1).gather(2, act[:, None, None].expand(B, K, 1)).squeeze(2)   # (B, K)
    err = F.smooth_l1_loss(qa, y[:, None].expand_as(qa), reduction="none") if a.huber else (qa - y[:, None]) ** 2
    if a.bootstrap_p < 1.0:
        m = torch.from_numpy((rng.random((B, K)) < a.bootstrap_p).astype(np.float32)).to(a.device)
        loss = (err * m).sum() / m.sum().clamp(min=1.0)
    else:
        loss = err.mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()
    noop = torch.from_numpy(batch["is_noop"]).to(a.device)
    qm = qa.detach().mean(1)
    out = {"loss": float(loss.detach()), "q_sa": float(qm.mean()), "y": float(y.mean())}
    if bool(noop.any()):
        out.update(q0_pred=float(qm[noop].mean()), q0_y=float(y[noop].mean()))
    if bool((~noop).any()):
        out.update(qa_pred=float(qm[~noop].mean()), qa_y=float(y[~noop].mean()))
    return out


# ------------------------------------------------------------------ media / logs -------------------

def rollout_record(r, extra=None):
    rec = {k: r.get(k) for k in ("tag", "label", "end", "steps", "pick_success", "pick_step", "task_success", "counts",
                                 "interventions", "hl", "info", "wall_s", "skipped", "reset_failed")}
    rec["n_branches"] = len(r.get("branches") or [])
    rec.update(extra or {})
    return rec


def save_media(r, out, prefix, net, kappa):
    paths = {}
    if r.get("samples") and r.get("branches") and net is not None:
        import viz

        obs = r["samples"][0][0]
        q = Q.ensemble_q(net, obs)
        paths["qmap"] = out / "qmaps" / "{}_{}.png".format(prefix, r["tag"])
        viz.save_qmap(obs, q, r["branches"][0]["a"], str(paths["qmap"]), kappa=kappa,
                      title="{} y_noop {:.2f} y_a {:.2f}".format(r["tag"], r["branches"][0]["y_noop"],
                                                                r["branches"][0]["y_a"]))
    if r.get("frames"):
        import imageio

        paths["video"] = out / "videos" / "{}_{}.mp4".format(prefix, r["tag"])
        imageio.mimwrite(str(paths["video"]), r["frames"], fps=FPS)
    return paths


def wb_media(paths_by_tag, prefix):
    WB = runlog.wb()
    if WB is None:
        return {}
    extra = {}
    for tag, p in paths_by_tag.items():        # one key per start: wandb 0.13 queries lists of images
        if "qmap" in p:
            extra["{}/qmap/{}".format(prefix, tag)] = WB.Image(str(p["qmap"]))
        if "video" in p:
            extra["{}/video/{}".format(prefix, tag)] = WB.Video(str(p["video"]), fps=FPS, format="mp4")
    return extra


# ------------------------------------------------------------------ evaluation of branch points ----

def evaluate_branches(a, actors, cfg, net, starts, out, label, step=None, tau=None, bad=()):
    """Greedy branch points (the deployed choice, mu - kappa sigma) on pi0 + HL main rollouts of the eval starts."""
    rc = rollout_cfg(a, "eval_branches")
    jobs = make_jobs(a, [s for s in starts if s["tag"] not in bad], a.eval_repeats, cfg, rc, Q.state_dict_numpy(net),
                     label, media=a.media_n, seed_base=104729)
    results = actors.run(jobs)
    media = {}
    for r in results:
        for b in r["branches"]:
            log_jsonl(out / "branches.jsonl", dict(b, tag=r["tag"], label=label))
        log_jsonl(out / "rollouts.jsonl", rollout_record(r))
        if r.get("media"):
            media[r["tag"]] = save_media(r, out, label, net, a.kappa)
    rec = dict(EV.branch_metrics(results, tau), label=label, rollouts=EV.rollout_metrics(results))
    if step is not None:
        rec["branch_points_total"] = step
    log_jsonl(out / "eval.jsonl", rec)
    wb_log(clean(rec), "eval", step, wb_media(media, "eval"))
    print("[{}] eval: {} branch points  y_noop {}  y_greedy {}  adv {}  policy gain {}  frac A>0 {}  Q0 bias {}  "
          "pick success (pi0+HL) {}".format(label, rec.get("n_branch_points"), _f(rec.get("y_noop")),
                                            _f(rec.get("y_a")), _f(rec.get("adv")), _f(rec.get("policy_gain")),
                                            _f(rec.get("frac_A_pos")), _f(rec.get("q0_bias")),
                                            _f(rec["rollouts"].get("pick_success"))), flush=True)
    return rec, results


def _f(x, fmt="{:.3f}"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


# ------------------------------------------------------------------ train (phases 0-2) -------------

def train(a, out):
    import torch

    torch.manual_seed(a.seed)
    phase = a.phase
    cfg = env_cfg(a)
    train_starts, eval_starts = build_starts(a)
    net = Q.build_qnet(a.width, a.heads).to(a.device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    state = {"phase": phase, "bp": {"0": 0, "1": 0, "2": 0}, "bp_total": 0, "rollouts": 0, "rounds": 0,
             "grad_steps": 0, "best_eval": None, "bad_tags": [], "width": a.width, "heads": a.heads, "tau": None}
    buf = ReplayBuffer(a.buffer)
    init = None
    if a.resume and (out / "ckpt_latest.pt").exists():
        init = out / "ckpt_latest.pt"
    elif phase == 2:                          # phase 2 starts from the trained (phase 1) Q-map
        init = pathlib.Path(a.ckpt) if a.ckpt else out / "ckpt_best.pt"
        if not init.exists():
            init = out / "ckpt_latest.pt"
        if not init.exists():
            raise SystemExit("phase 2 needs a phase 1 checkpoint in OUT_DIR (or --ckpt)")
    if init is not None:
        ck = torch.load(str(init), map_location=a.device)
        st = ck["state"]
        if (st.get("width"), st.get("heads")) != (a.width, a.heads):
            raise SystemExit("checkpoint has width {} / heads {}; pass the same --width / --heads".format(
                st.get("width"), st.get("heads")))
        net.load_state_dict(ck["net"])
        if ck.get("opt") is not None:
            opt.load_state_dict(ck["opt"])
        state.update(st)
        state["phase"] = phase
        if (out / "replay.pkl").exists():
            buf = ReplayBuffer.load(str(out / "replay.pkl"), a.buffer)
        print("loaded {}: {} branch points so far ({}), {} samples in the replay buffer".format(
            init, state["bp_total"], state["bp"], len(buf)), flush=True)
    if phase >= 1 and len(buf) == 0:
        pre = pathlib.Path(a.prefill) if a.prefill else out / "replay_phase0.pkl"
        if pre.exists():
            buf.extend(ReplayBuffer.load(str(pre)))
            print("prefilled the replay buffer with {} no-op labels from {}".format(len(buf), pre), flush=True)
    tau = None
    if phase == 2:
        tau = load_tau(a, out)
        if tau is None:
            raise SystemExit("phase 2 needs tau: run --mode calibrate first (calibration.json) or pass --tau")
        state["tau"] = tau
    # phase 0: probe counts per tilt bin (from the log, so --resume continues), and the no-op labels it collects
    probes = [0] * len(RO.TILT_BINS)
    pbuf = ReplayBuffer(a.buffer)
    if phase == 0:
        if a.resume and (out / "probe_log.jsonl").exists():
            for line in open(out / "probe_log.jsonl"):
                probes[RO.tilt_bin(json.loads(line)["tilt_deg"])] += 1
            if (out / "replay_phase0.pkl").exists():
                pbuf = ReplayBuffer.load(str(out / "replay_phase0.pkl"), a.buffer)
    per_bin = a.phase0_probes_per_bin

    def bin_counts():
        c = list(probes)
        if a.phase0_merge_top:
            c = c[:2] + [c[2] + c[3]] * 2
        return c

    def finished():
        if phase == 0:
            return all(c >= per_bin for c in bin_counts()) or sum(probes) >= a.phase0_max_probes
        return state["bp"][str(phase)] >= a.branch_points

    bad = set(state["bad_tags"])
    rng = np.random.default_rng(a.seed + 1000 * state["rounds"] + phase)
    actors = make_actors(cfg, a.workers)
    t_start = time.time()
    try:
        while not finished():
            pool = [s for s in train_starts if s["tag"] not in bad]
            if not pool:
                raise SystemExit("every start state was skipped as ineligible")
            if phase == 0:
                rc = rollout_cfg(a, "phase0")
                if a.phase0_skip_full_bins:
                    rc["skip_tilt_bins"] = tuple(k for k, c in enumerate(bin_counts()) if c >= per_bin)
                weights, eps_now = None, None
            else:
                eps_now = epsilon(a, state["bp"]["1"]) if phase == 1 else a.eps_end
                rc = rollout_cfg(a, "phase{}".format(phase), eps=eps_now, tau=tau)
                weights = Q.state_dict_numpy(net) if state["grad_steps"] > 0 or init is not None else None
            jobs = [{"id": i, "start": pool[rng.integers(len(pool))], "cfg": cfg, "rc": rc, "weights": weights,
                     "kappa": a.kappa, "seed": int(rng.integers(2 ** 31)), "record": False, "keep_obs": True,
                     "label": "train_p{}".format(phase)} for i in range(a.rollouts_per_round)]
            results = actors.run(jobs)
            bp_before, n_new = state["bp_total"], 0
            for r in results:
                if r.get("skipped"):
                    bad.add(r["tag"])
                    print("skip {}: {}".format(r["tag"], r["skipped"]), flush=True)
                    log_jsonl(out / "rollouts.jsonl", rollout_record(r, {"phase": phase}))
                    continue
                if r.get("reset_failed"):
                    print("reset failed {}: {}".format(r["tag"], r["reset_failed"]), flush=True)
                    log_jsonl(out / "rollouts.jsonl", rollout_record(r, {"phase": phase}))
                    continue
                state["rollouts"] += 1
                for b in r["branches"]:
                    log_jsonl(out / "branches.jsonl", dict(b, tag=r["tag"], phase=phase, label="train"))
                    if phase == 0:
                        probes[RO.tilt_bin(b["tilt_deg"])] += 1
                        log_jsonl(out / "probe_log.jsonl", dict({k: b.get(k) for k in (
                            "tilt_deg", "y_noop", "y_a", "t_noop", "t_a", "end_noop", "end_a", "steps_noop", "steps_a",
                            "prim", "target_dxy", "step")}, tag=r["tag"]))
                for obs, act, y, info in r["samples"]:
                    if phase == 0:
                        pbuf.add(obs, act, y, dict(info, tag=r["tag"], phase=0))
                    else:
                        buf.add(obs, act, y, dict(info, tag=r["tag"], phase=phase))
                        n_new += 1
                state["bp"][str(phase)] += len(r["branches"])
                state["bp_total"] += len(r["branches"])
                log_jsonl(out / "rollouts.jsonl", rollout_record(r, {"phase": phase, "bp_total": state["bp_total"]}))
            state["bad_tags"] = sorted(bad)

            losses = []
            if phase >= 1 and len(buf) >= max(a.warmup, a.batch):
                net.train()
                for _ in range(int(round(a.utd * n_new))):
                    losses.append(q_update(net, opt, buf.sample(a.batch, rng), a, rng))
                    state["grad_steps"] += 1
                net.eval()
            state["rounds"] += 1
            rec = {"round": state["rounds"], "phase": phase, "bp_total": state["bp_total"],
                   "bp_phase": state["bp"][str(phase)], "rollouts": state["rollouts"], "grad_steps": state["grad_steps"],
                   "eps": eps_now, "elapsed_min": round((time.time() - t_start) / 60, 2),
                   "buffer": (pbuf if phase == 0 else buf).stats(), "branches": EV.branch_metrics(results),
                   "rollout": EV.rollout_metrics(results)}
            if phase == 0:
                rec["probes_per_bin"] = list(probes)
            if losses:
                for k in losses[0]:
                    rec[k] = float(np.mean([L[k] for L in losses if k in L]))
            log_jsonl(out / "train.jsonl", rec)
            wb_log(clean(rec), "train", state["bp_total"])
            if state["rounds"] % a.print_every == 0:
                br = rec["branches"]
                print("round {} phase {} bp {} ({} this phase) rollouts {} grad {} eps {} | branch y_noop {} y_a {} "
                      "| loss {} Q0 {}/{} Qa {}/{} | buf {} | pick {} | {} min{}".format(
                          rec["round"], phase, rec["bp_total"], rec["bp_phase"], rec["rollouts"], rec["grad_steps"],
                          _f(eps_now, "{:.3f}"), _f(br.get("y_noop")), _f(br.get("y_a")), _f(rec.get("loss"), "{:.4f}"),
                          _f(rec.get("q0_pred")), _f(rec.get("q0_y")), _f(rec.get("qa_pred")), _f(rec.get("qa_y")),
                          rec["buffer"].get("size"), _f(rec["rollout"].get("pick_success")), rec["elapsed_min"],
                          "  probes/bin {}".format(probes) if phase == 0 else ""), flush=True)

            bp = state["bp_total"]
            if phase >= 1 and a.eval_every and bp // a.eval_every > bp_before // a.eval_every:
                erec, _ = evaluate_branches(a, actors, cfg, net, eval_starts, out, "eval_bp{}".format(bp), bp, tau, bad)
                g = erec.get("policy_gain")
                if g is not None and np.isfinite(g) and (state["best_eval"] is None or g > state["best_eval"]):
                    state["best_eval"] = float(g)
                    save_ckpt(out / "ckpt_best.pt", net, opt, state)
                wb_log({"best_policy_gain": state["best_eval"]}, "eval", bp)
            if a.ckpt_every and bp // a.ckpt_every > bp_before // a.ckpt_every:
                _save_all(a, out, net, opt, state, buf, pbuf, phase)
    finally:
        _save_all(a, out, net, opt, state, buf, pbuf, phase)
        actors.close()
    if phase == 0:
        phase0_report(a, out)


def _save_all(a, out, net, opt, state, buf, pbuf, phase):
    if phase == 0:
        pbuf.save(str(out / "replay_phase0.pkl"))
        (out / "phase0_state.json").write_text(json.dumps(clean(state), indent=2))
        return
    save_ckpt(out / "ckpt_latest.pt", net, opt, state)
    if a.save_buffer:
        buf.save(str(out / "replay.pkl"))


# ------------------------------------------------------------------ other modes --------------------

def phase0_report(a, out):
    p = out / "probe_log.jsonl"
    if not p.exists():
        raise SystemExit("no probe_log.jsonl in {} (run --mode train --phase 0 first)".format(out))
    rows = [json.loads(line) for line in open(p)]
    rep = EV.phase0_report(rows, steps(a.h_max_s), a.gamma, a.phase0_probes_per_bin)
    (out / "phase0_report.json").write_text(json.dumps(clean(rep), indent=2))
    print("phase 0: {} levelling probes; paired difference (level only - no-op) of labels at H_max = {:.1f} s:".format(
        rep["n_probes"], rep["h_max_s"]))
    for b in rep["bins"] + [rep["merged_20_plus"]]:
        print("  tilt {:>6s} deg  n {:4d}  diff {} +- {}".format(b["tilt"], b["n"], _f(b["diff"]), _f(b["se"])))
    t = rep["time_to_success_s"]
    if t.get("n"):
        print("  no-op time to success: n {} median {:.2f} s, 90th pct {:.2f} s; {:.0%} beyond H_max".format(
            t["n"], t["p50"], t["p90"], rep["frac_successes_beyond_h_max"]))
    print("  -> " + rep["recommendation"])
    return rep


def eval_branches_mode(a, out):
    net, st, path = load_net(a, out)
    cfg = env_cfg(a, st.get("width"), st.get("heads"))
    _, eval_starts = build_starts(a)
    actors = make_actors(cfg, a.workers)
    try:
        evaluate_branches(a, actors, cfg, net, eval_starts, out, "eval", tau=load_tau(a, out))
    finally:
        actors.close()


def _deploy(a, actors, cfg, net, starts, rc, label, hl=None, repeats=1, media=0, out=None):
    jobs = make_jobs(a, starts, repeats, cfg, rc, None if net is None else Q.state_dict_numpy(net), label,
                     media=media, hl=hl, seed_base=15485863)
    results = actors.run(jobs)
    if out is not None:
        for r in results:
            log_jsonl(out / "rollouts_{}.jsonl".format(a.mode), rollout_record(r))
            if r.get("media"):
                save_media(r, out, label, net, a.kappa)
    return results


def calibrate(a, out):
    """sec. 11: (1) greedy branch points on the eval starts -> Q0 reliability, noise floor, tau_noise;
    (2) tau sweep with the trigger acting on the main rollout (no branching) -> success vs interventions -> knee."""
    import viz

    net, st, path = load_net(a, out)
    cfg = env_cfg(a, st.get("width"), st.get("heads"))
    _, eval_starts = build_starts(a)
    actors = make_actors(cfg, a.workers)
    try:
        rc = rollout_cfg(a, "eval_branches")
        rc["p_branch"] = a.calib_p_branch
        results = actors.run(make_jobs(a, eval_starts, a.calib_repeats, cfg, rc, Q.state_dict_numpy(net), "calib",
                                       seed_base=32452843))
        brs = []
        for r in results:
            for b in r["branches"]:
                brs.append(dict(b, tag=r["tag"]))
                log_jsonl(out / "calib_points.jsonl", dict(b, tag=r["tag"]))
        rel = EV.reliability(brs)
        nf = EV.noise_floor(brs, gate_m(a), fast_steps=steps(1.0), edge_band_m=a.edge_band_cm / 100.0)
        bm = EV.branch_metrics(results)
        viz.save_reliability(rel, str(out / "reliability.png"))
        tau_noise = nf["tau_noise"]
        print("calibration: {} branch points; Q0 bias {} MSE {}; noise floor: fast n={} edge n={} -> tau_noise {}".format(
            len(brs), _f(bm.get("q0_bias")), _f(bm.get("q0_mse")), nf["fast"].get("n"), nf["edge"].get("n"),
            _f(tau_noise)), flush=True)
        for b in rel:
            if b["n"]:
                print("  Q0 in [{:.1f}, {:.1f}): n {:4d}  mean Q0 {:.2f}  mean label {:.2f}  success {:.2f}".format(
                    b["lo"], b["hi"], b["n"], b["q0_mean"], b["y_mean"], b["success_frac"]))
        base = tau_noise if tau_noise is not None else 0.0
        taus = list(a.tau_sweep) if a.tau_sweep else [round(base + d, 4) for d in (0.0, 0.05, 0.1, 0.2, 0.3)]
        sweep = []
        for tau in [float("inf")] + taus:
            rcd = rollout_cfg(a, "deploy", tau=tau)
            res = _deploy(a, actors, cfg, net, eval_starts, rcd, "sweep_tau{}".format(tau), repeats=a.sweep_repeats)
            m = EV.rollout_metrics(res)
            row = {"tau": tau if np.isfinite(tau) else "inf", "success": m["pick_success"],
                   "interventions_per_rollout": m["interventions_per_rollout"], "steps_to_pick_s": m["steps_to_pick_s"],
                   "n_rollouts": m["n_rollouts"], "fire_rate_per_eligible_tick": m["fire_rate_per_eligible_tick"]}
            sweep.append(row)
            log_jsonl(out / "sweep.jsonl", row)
            print("  tau {:>8}: pick success {}  interventions / rollout {}".format(
                str(row["tau"]), _f(row["success"]), _f(row["interventions_per_rollout"], "{:.2f}")), flush=True)
        pts = [(r["interventions_per_rollout"], r["success"]) for r in sweep]
        k = EV.knee(pts)
        knee_tau = sweep[k]["tau"] if k is not None else None
        rec = None if knee_tau in (None, "inf") else float(knee_tau)
        if rec is not None and tau_noise is not None and rec < tau_noise:
            rec = float(tau_noise)
        note = None
        if knee_tau == "inf":
            note = "the knee is 'no trigger': no tau in the sweep beat pi0 + HL; inspect the sweep before phase 2"
        viz.save_tau_sweep(sweep, str(out / "tau_sweep.png"))
        cal = {"ckpt": path, "tau_noise": tau_noise, "tau_knee": knee_tau, "tau_recommended": rec, "note": note,
               "kappa": a.kappa, "persist": a.persist, "cooldown_s": a.cooldown_s, "noise_floor": nf,
               "reliability": rel, "branch_metrics": bm, "sweep": sweep}
        (out / "calibration.json").write_text(json.dumps(clean(cal), indent=2))
        print("tau_noise {}  knee {}  -> tau {}{}".format(_f(tau_noise), knee_tau, _f(rec),
                                                          "  ({})".format(note) if note else ""))
    finally:
        actors.close()


def evaluate_mode(a, out):
    """sec. 11 evaluation: success rate and interventions per rollout for each condition."""
    conds = a.conditions
    net, st, tau, cfg = None, {}, None, None
    if any(c in ("trigger", "random") for c in conds):
        net, st, _ = load_net(a, out)
        tau = load_tau(a, out)
        if tau is None and "trigger" in conds:
            raise SystemExit("the trigger condition needs tau (--tau or OUT_DIR/calibration.json)")
    cfg = env_cfg(a, st.get("width"), st.get("heads"))
    _, eval_starts = build_starts(a)
    until = a.until or "task"
    actors = make_actors(cfg, a.workers)
    summary, rate = {}, a.random_rate
    try:
        for c in conds:
            hl = "none" if c == "pi0" else a.hl
            if c == "random":
                if rate is None:
                    rate = summary.get("trigger", {}).get("fire_rate_per_eligible_tick")
                if rate is None or not np.isfinite(rate):
                    raise SystemExit("the random condition needs the trigger condition first (same run) or --random-rate")
                rc = rollout_cfg(a, "deploy", random_p=rate, until=until)
            else:
                rc = rollout_cfg(a, "deploy", tau=tau if c == "trigger" else None, until=until)
            res = _deploy(a, actors, cfg, net if c in ("trigger", "random") else None, eval_starts, rc, c, hl=hl,
                          repeats=a.eval_repeats, media=a.media_n, out=out)
            m = EV.rollout_metrics(res)
            m.update(condition=c, hl=hl, until=until, tau=tau if c == "trigger" else None,
                     random_p=rate if c == "random" else None, per_start_pick=EV.per_start(res),
                     per_start_task=EV.per_start(res, "task_success"))
            summary[c] = m
            log_jsonl(out / "eval_conditions.jsonl", m)
            print("[{}] pick success {}  task success {}  interventions / rollout {}  HL corrections / rollout {}  "
                  "({} rollouts)".format(c, _f(m["pick_success"]), _f(m["task_success"]),
                                         _f(m["interventions_per_rollout"], "{:.2f}"), _f(m["hl_per_rollout"], "{:.2f}"),
                                         m["n_rollouts"]), flush=True)
    finally:
        actors.close()
    (out / "eval_conditions.json").write_text(json.dumps(clean(summary), indent=2))


def timing_mode(a, out):
    """sec. 11 trigger timing: score A(s) at every gated tick of pi0 + HL rollouts (no interventions), replay the
    trigger's persistence + cooldown offline -> false-trigger rate on rollouts pi0 completes alone."""
    import viz

    net, st, _ = load_net(a, out)
    tau = load_tau(a, out)
    if tau is None:
        raise SystemExit("timing needs tau (--tau or OUT_DIR/calibration.json)")
    cfg = env_cfg(a, st.get("width"), st.get("heads"))
    _, eval_starts = build_starts(a)
    actors = make_actors(cfg, a.workers)
    try:
        res = _deploy(a, actors, cfg, net, eval_starts, rollout_cfg(a, "timing"), "timing", repeats=a.eval_repeats)
    finally:
        actors.close()
    for r in res:
        log_jsonl(out / "timing.jsonl", {"tag": r["tag"], "pick_success": r.get("pick_success"), "end": r.get("end"),
                                         "steps": r.get("steps"), "ticks": r.get("ticks")})
    rep = EV.timing_report(res, tau, a.persist, steps(a.cooldown_s))
    (out / "timing_report.json").write_text(json.dumps(clean(rep), indent=2))
    viz.save_timelines(res, str(out / "timelines.png"), tau)
    print("timing (tau {:.3f}): false-trigger rate on {} rollouts pi0 completed alone {}; trigger rate on {} failures {}"
          .format(tau, rep["n_success"], _f(rep["false_trigger_rate"]), rep["n_fail"],
                  _f(rep["trigger_rate_on_failures"])))


def inspect_mode(a, out):
    """LIBERO, no pi0: per start, scripted poses near the target; heightmap PNGs and geometry stats."""
    import heightmap as HMAP
    import viz
    from libero_env import BranchingPickEnv
    from shared.scene import IneligibleStart, ResetFailed

    (out / "inspect").mkdir(parents=True, exist_ok=True)
    train, ev = build_starts(a)
    starts, seen = [], set()
    for s in train + ev:
        if s["tag"] not in seen:
            seen.add(s["tag"])
            starts.append(s)
    if a.inspect_max:
        starts = starts[:a.inspect_max]
    poses = [(0.0, 0.0, 0.07, 0.0, 0.0), (0.03, 0.0, 0.05, 0.0, 20.0), (0.0, -0.03, 0.04, 15.0, -30.0),
             (-0.02, 0.02, 0.10, 30.0, 0.0)]
    env = BranchingPickEnv(dict(env_cfg(a), hl="none"))
    rows = []
    for s in starts:
        try:
            env.reset(s)
        except (IneligibleStart, ResetFailed) as e:
            print("{:40s} SKIP {}".format(s["tag"], e))
            rows.append({"tag": s["tag"], "skipped": str(e)})
            continue
        for n, pose in enumerate(poses):
            mv = env.scripted_pose(*pose)
            obs = env.observe(force=True)
            info = env.inspect_info()
            viz.save_inspect(info, obs, str(out / "inspect" / "{}_pose{}.png".format(s["tag"], n)))
            ti = env.tick_info()
            gate, ginfo = env.gate()
            hc = HMAP.clearance_height(info["z"], Q.C, Q.C)
            row = {"tag": s["tag"], "pose": n, "object": info["object"], "robot_geoms": info["robot_geoms"],
                   "observed_frac": float(np.mean(obs["observed"])), "target_px_map": int(np.sum(obs["mask"])),
                   "agent_target_px": info["agent_target_px"], "robot_px_dropped": info["robot_px"],
                   "z_top": round(info["z_top"], 4), "above_top": round(ti["above_top"], 4),
                   "tilt_deg": round(ti["tilt_deg"], 1), "target_dxy": round(ti["target_dxy"], 4),
                   "h_clear_centre": None if hc is None else round(hc, 4), "gate_open": bool(gate),
                   "t_observe_ms": round(1000 * info["t_obs_s"], 1), "move_reached": mv.get("reached"),
                   "proprio": [round(float(x), 3) for x in obs["proprio"]]}
            rows.append(row)
            print("{:40s} pose {} {:20s} observed {:.2f}  target px {:4d}  robot px dropped {:6d}  above top {:+.3f} m  "
                  "tilt {:4.1f}  gate {}  h_clear {}  observe {:.0f} ms".format(
                      s["tag"], n, info["object"], row["observed_frac"], row["target_px_map"], row["robot_px_dropped"],
                      row["above_top"], row["tilt_deg"], row["gate_open"], row["h_clear_centre"], row["t_observe_ms"]),
                  flush=True)
    env.close()
    (out / "inspect" / "stats.json").write_text(json.dumps(clean(rows), indent=2))


def restore_check_mode(a, out):
    from libero_env import restore_check

    train, ev = build_starts(a)
    reps = []
    for s in (train + ev)[:max(1, a.inspect_max or 3)]:
        rep = restore_check(env_cfg(a), s)
        reps.append(rep)
        print(json.dumps(clean(rep)), flush=True)
    (out / "restore_check.json").write_text(json.dumps(clean(reps), indent=2))
    bad = [r for r in reps if r["verdict"] != "PASS"]
    print("restore check: {} / {} PASS".format(len(reps) - len(bad), len(reps)))
    if bad:
        raise SystemExit(1)


def finger_mask_mode(a, out):
    from libero_env import build_finger_mask

    train, _ = build_starts(a)
    path = out / "finger_mask.npy"
    m = build_finger_mask(env_cfg(a), train[:max(1, a.inspect_max or 8)], str(path))
    print("finger mask: {} of {} wrist pixels -> {}".format(int(m.sum()), m.size, path))


# ------------------------------------------------------------------ main ---------------------------

MODES = ("train", "eval-branches", "calibrate", "evaluate", "timing", "phase0-report", "inspect", "restore-check",
         "finger-mask")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", default="train", choices=MODES)
    p.add_argument("--phase", type=int, default=1, choices=[0, 1, 2], help="train mode: 0 levelling check, 1 train, "
                   "2 fine-tune with the trigger in the loop")
    p.add_argument("--env", default="libero", choices=["libero", "toy"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume", action="store_true", help="continue from OUT_DIR (checkpoint, replay, probe log)")
    p.add_argument("--ckpt", default=None, help="evaluation modes: checkpoint (default OUT_DIR/ckpt_best.pt)")
    p.add_argument("--prefill", default=None, help="phase 1: replay file to prefill (default OUT_DIR/replay_phase0.pkl)")
    # starts (task init states only)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="<suite>:<ids>, ids like 9,10 or 0-9")
    p.add_argument("--eval-tasks", nargs="+", default=None, help="eval starts' tasks (default --tasks); held-out "
                   "libero_90 tasks are accepted outside --mode train")
    p.add_argument("--train-inits", type=int, nargs="+", default=list(range(0, 40)))
    p.add_argument("--eval-inits", type=int, nargs="+", default=[40, 41])
    # observation / action / gate
    p.add_argument("--reach", type=int, default=3, help="reach r in cells (2 cm): (2r+1)^2 cells x 9 bins")
    p.add_argument("--gate-margin-cm", type=float, default=4.0, help="gate: target xy within r*2 cm + this")
    p.add_argument("--gate-min-gap", type=float, default=0.05, help="gate: finger gap (m) counted as open")
    p.add_argument("--min-mask-px", type=int, default=20, help="agentview target pixels for 'grounded'")
    p.add_argument("--agent-px", type=int, default=None, help="agentview render size for the heightmap (default env)")
    p.add_argument("--wrist-px", type=int, default=None, help="wrist render size for the heightmap (default env)")
    p.add_argument("--no-fill-holes", dest="fill_holes", action="store_false")
    p.add_argument("--robot-mask", default="seg", choices=["seg", "fixed"],
                   help="robot points: sim segmentation (training) or a fixed wrist finger mask (test time)")
    p.add_argument("--finger-mask", default=None, help="--robot-mask fixed: .npy from --mode finger-mask")
    p.add_argument("--mask-source", default="seg", choices=["seg", "box", "mix"],
                   help="target mask channel: sim segmentation, an agentview box (test time: detect), or a mix")
    p.add_argument("--mix-p", type=float, default=0.5, help="--mask-source mix: probability of the box mask")
    p.add_argument("--box-jitter", type=float, default=0.05, help="box mask: jitter of each side (fraction of size)")
    p.add_argument("--clear-min-cm", type=float, default=2.0, help="primitive: rise if below h_clear + this")
    p.add_argument("--clear-to-cm", type=float, default=3.0, help="primitive: ... to h_clear + this")
    p.add_argument("--prim-cap", type=int, default=30, help="primitive cap (control steps, all legs)")
    # labels / timing (control steps at 20 Hz)
    p.add_argument("--h-max-s", type=float, default=3.0, help="label window H_max (s)")
    p.add_argument("--gamma", type=float, default=0.85, help="label discount per second")
    p.add_argument("--phase0-cap-s", type=float, default=10.0, help="phase 0 branch length (s), to log success times")
    p.add_argument("--cooldown-s", type=float, default=3.0, help="after an intervention or an HL correction")
    p.add_argument("--persist", type=int, default=3, help="trigger: consecutive gated ticks with A > tau")
    p.add_argument("--p-branch", type=float, default=0.8, help="branch probability per gated tick (training)")
    p.add_argument("--eval-p-branch", type=float, default=0.8, help="branch probability in eval-branches")
    p.add_argument("--same-noise", action="store_true", help="toy only: same pi0 noise in both branches (the "
                   "openpi server does not accept a noise seed, see the instructions)")
    # phase 0
    p.add_argument("--phase0-probes-per-bin", type=int, default=100)
    p.add_argument("--phase0-max-probes", type=int, default=800)
    p.add_argument("--phase0-merge-top", action="store_true", help="count the 20-30 and 30+ tilt bins together")
    p.add_argument("--no-phase0-skip-full-bins", dest="phase0_skip_full_bins", action="store_false",
                   help="keep placing probes in tilt bins that already have enough")
    # network / learner
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--heads", type=int, default=5, help="ensemble heads K")
    p.add_argument("--kappa", type=float, default=1.0, help="uncertainty penalty of the trigger / deployed argmax")
    p.add_argument("--branch-kappa", type=float, default=0.0, help="greedy score of training branches mu - k*sigma")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--buffer", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=500, help="samples before the first update")
    p.add_argument("--utd", type=float, default=2.0, help="gradient steps per new sample")
    p.add_argument("--bootstrap-p", type=float, default=1.0, help="< 1: each head sees each sample with this p")
    p.add_argument("--huber", action="store_true")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--eps-decay-branches", type=int, default=1000, help="phase 1 branch points for eps 1 -> 0.1")
    p.add_argument("--branch-points", type=int, default=3000, help="phase 1 / 2 length in branch points")
    p.add_argument("--device", default="cpu")
    # evaluation / calibration
    p.add_argument("--tau", type=float, default=None, help="trigger margin (default OUT_DIR/calibration.json)")
    p.add_argument("--hl", default="privileged", choices=["privileged", "none", "moondream"],
                   help="HL module in the main rollout: privileged stand-in, none, or the Moondream2 agent")
    p.add_argument("--vlm-model", default=None, help="--hl moondream: model for vlm_backends.make_vlm")
    p.add_argument("--until", default=None, choices=["pick", "task"],
                   help="main rollouts end at pick success or task success (default: task in evaluate, else pick)")
    p.add_argument("--conditions", nargs="+", default=["pi0", "hl", "trigger", "random"],
                   choices=["pi0", "hl", "trigger", "random"])
    p.add_argument("--random-rate", type=float, default=None, help="random trigger: fire probability per eligible "
                   "gated tick (default: the trigger condition's rate in the same run)")
    p.add_argument("--calib-p-branch", type=float, default=0.8)
    p.add_argument("--calib-repeats", type=int, default=2)
    p.add_argument("--sweep-repeats", type=int, default=1)
    p.add_argument("--tau-sweep", type=float, nargs="*", default=None, help="tau values (default tau_noise + 0, "
                   ".05, .1, .2, .3)")
    p.add_argument("--edge-band-cm", type=float, default=1.0, help="noise floor: targets this close to the gate edge")
    # runtime
    p.add_argument("--workers", type=int, default=None, help="actor processes (default: 0 for toy, 6 for libero)")
    p.add_argument("--rollouts-per-round", type=int, default=None,
                   help="main rollouts per round before the learner updates (default: workers, 4 in-process)")
    p.add_argument("--eval-every", type=int, default=250, help="branch points between evals (0 = never)")
    p.add_argument("--eval-repeats", type=int, default=1)
    p.add_argument("--media-n", type=int, default=3, help="eval starts that get a Q-map PNG (and a video)")
    p.add_argument("--video", action="store_true")
    p.add_argument("--ckpt-every", type=int, default=100, help="branch points between checkpoints")
    p.add_argument("--save-buffer", action="store_true", default=True)
    p.add_argument("--no-save-buffer", dest="save_buffer", action="store_false")
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--inspect-max", type=int, default=0, help="inspect / restore-check / finger-mask: max starts")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=7)
    a = p.parse_args(argv)
    if a.workers is None:
        a.workers = 0 if a.env == "toy" else 6
    if a.rollouts_per_round is None:
        a.rollouts_per_round = a.workers if a.workers > 0 else 4
    if a.hl == "moondream" and a.mode == "train":
        raise SystemExit("--hl moondream is for evaluation (its agent keeps history a branch restore cannot rewind)")
    return a


def main(argv=None):
    a = parse_args(argv)
    out = pathlib.Path(a.out_dir)
    for sub in ("qmaps", "videos"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    name = "config_{}{}.json".format(a.mode, "_phase{}".format(a.phase) if a.mode == "train" else "")
    (out / name).write_text(json.dumps(vars(a), indent=2))
    if a.mode in ("inspect", "restore-check", "finger-mask") and a.env != "libero":
        raise SystemExit("--mode {} needs --env libero".format(a.mode))
    if a.mode == "phase0-report":
        phase0_report(a, out)
        return
    if a.mode == "inspect":
        inspect_mode(a, out)
        return
    if a.mode == "restore-check":
        restore_check_mode(a, out)
        return
    if a.mode == "finger-mask":
        finger_mask_mode(a, out)
        return
    runlog.wb_init(a, out, step_metric="branch_points", summary_max=("eval/policy_gain",),
                   prefixes=("train/*", "eval/*"))
    try:
        {"train": train, "eval-branches": eval_branches_mode, "calibrate": calibrate, "evaluate": evaluate_mode,
         "timing": timing_mode}[a.mode](a, out)
    finally:
        runlog.wb_finish()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
