"""Run a trained V along pi0's main rollouts: V_est at every gated tick, shown continuously in the video. No branch
episodes and no perturbations, so pi0 (+ the HL stand-in, as in collection) runs uninterrupted.

The loop, the gate and the HL cooldown are as in episodes.collect_rollout. Video frames (wrist | agentview, 2x)
carry a banner:
  - step / time since the first gated tick;
  - the current tick's gate status: GATED #k with its V_est (green), not gated with the reason (red), HL cooldown
    (magenta), or an HL correction (cyan);
  - the last V_est and how long ago it was made;
  - a strip chart over the rollout: V_est +- sd at the gated ticks, a gate bar per tick (green = gated,
    red = not gated, magenta = cooldown, cyan = HL correction), and the current step.

Outputs (--out-dir): videos/<group>_<tag>.mp4, ticks.csv (one row per gated tick), summary.json (per rollout).

  PY=/opt/openpi/examples/libero/.venv/bin/python
  MUJOCO_GL=egl $PY eval_v_rollouts.py --ckpt runs/v4_train/v_attn.pt --out-dir runs/v4_train/rollouts
"""

import argparse
import csv
import json
import os
import pathlib
import sys
import time

import numpy as np

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EVAL_TASKS = "libero_90:13,15,24,36,65,76"
TRAIN_TASKS = "libero_90:9,71,63,25"
CHART_H = 110

_W = {}


# ------------------------------------------------------------------ one rollout --------------------

def not_gated_why(env, gate_open, ginfo, cooling_s):
    why = []
    if not gate_open:
        if ginfo.get("pick") is False:
            why.append("pick done")
        if ginfo.get("open") is False:
            why.append("gripper not open")
        if ginfo.get("grounded") is False:
            why.append("target not grounded (HL)")
        elif ginfo.get("near") is False:
            why.append("{:.0f} cm from target xy (max {:.0f})".format(100 * ginfo["target_dxy"], 100 * env.gate_m))
        if ginfo.get("low") is False:
            why.append("{:.0f} cm above target top (max {:.0f})".format(
                100 * ginfo["above_top"], 100 * env.cfg["gate_max_above_m"]))
        if ginfo.get("empty") is False:
            why.append("holding an object")
    if cooling_s > 0:
        why.append("HL cooldown ({:.1f} s left)".format(cooling_s))
    elif gate_open:
        why.append("target not grounded in agentview")
    return ", ".join(why) or "?"


def strip_chart(width, x_max, cur, ticks, vs):
    """V_est over the rollout. ticks: [(step, kind)], kind in gated / not / cool / hl; vs: [(step, mean, sd)]."""
    import cv2
    import episodes as E

    img = np.full((CHART_H, width, 3), 25, np.uint8)
    x0, x1, y0, y1 = 46, width - 12, 8, CHART_H - 26           # plot area; gate bar below it
    sx = lambda s: int(x0 + (x1 - x0) * min(s, x_max) / float(max(1, x_max)))  # noqa: E731
    sy = lambda v: int(y1 - (y1 - y0) * v)                                     # noqa: E731
    for v in (0.0, 0.5, 1.0):
        cv2.line(img, (x0, sy(v)), (x1, sy(v)), (60, 60, 60), 1)
        cv2.putText(img, "{:.1f}".format(v), (8, sy(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, E.GREY, 1, cv2.LINE_AA)
    col = {"gated": E.GREEN, "not": E.RED, "cool": E.MAGENTA, "hl": E.CYAN}
    for i, (s, k) in enumerate(ticks):
        nxt = ticks[i + 1][0] if i + 1 < len(ticks) else cur
        cv2.rectangle(img, (sx(s), CHART_H - 18), (max(sx(s) + 1, sx(nxt) - 1), CHART_H - 8), col[k], -1)
    pts = [(sx(s), sy(m)) for s, m, _ in vs]
    for s, m, sd in vs:
        cv2.line(img, (sx(s), sy(min(1, m + sd))), (sx(s), sy(max(0, m - sd))), E.YELLOW, 1)
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, E.YELLOW, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, E.YELLOW, -1, cv2.LINE_AA)
    cv2.line(img, (sx(cur), y0), (sx(cur), CHART_H - 6), E.WHITE, 1)
    cv2.putText(img, "V_est", (x0 + 4, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, E.YELLOW, 1, cv2.LINE_AA)
    return img


def v_rollout(env, start, rc, net, video_path):
    """pi0 main rollout; V_est at every gated tick. Returns (stats, ticks)."""
    import episodes as E
    import vnet
    from shared import scene as SC

    t_wall = time.time()
    env.reset(start, record=True)
    head = "{} | {}".format(start["tag"], env.prompt[:60])
    hz, cooldown = float(E.CTRL_HZ), int(E.COOLDOWN_S * E.CTRL_HZ)
    video, chart_ticks, vs, rows = [], [], [], []
    st = {"gated_ticks": 0, "hl_corrections": 0}
    first, last, end = None, None, None          # last: (step, mean, sd, k) of the latest gated tick
    status = ("gate: no tick yet", E.GREY)
    last_tick, cooldown_until = None, -10 ** 9

    def flush():
        n, end_step = len(env.frames), env.step_count
        for i, f in enumerate(env.frames):
            s = end_step - n + 1 + i
            tt = "step {} ({:.2f} s) | ".format(s, s / hz) + ("no gated tick yet" if first is None else
                                                             "{:.2f} s since first gated tick".format((s - first) / hz))
            lv = "last V_est -" if last is None else "last V_est {:.2f} +- {:.2f}  (tick #{}, {:.1f} s ago) | gated {}".format(
                last[1], last[2], last[3], (s - last[0]) / hz, st["gated_ticks"])
            img = E.annotate(f, [head, tt, status[0], lv], [E.WHITE, E.WHITE, status[1], E.YELLOW])
            video.append(np.concatenate([img, strip_chart(img.shape[1], env.budget, s, chart_ticks, vs)], 0))
        env.frames = []

    while True:
        step = env.step_count
        if env.pick_done():
            end = "success"
        elif env.failure():
            end = env.failure()
        elif step >= env.budget:
            end = "budget"
        if end:
            break
        if step % rc["tick_steps"] == 0 and step != last_tick:
            last_tick = step
            flush()
            hl = env.hl_tick()
            if hl is not None:
                st["hl_corrections"] += 1
                cooldown_until = env.step_count + cooldown
                chart_ticks.append((step, "hl"))
                status = ("HL CORRECTION #{}: {} -> {}".format(st["hl_corrections"], hl.get("mode"), hl.get("action")),
                          E.CYAN)
                flush()
                continue
            gate_open, ginfo = env.gate()
            cooling_s = max(0, cooldown_until - step) / hz
            obs = env.observe() if gate_open and cooling_s == 0 else None
            if obs is not None:
                first = step if first is None else first
                st["gated_ticks"] += 1
                k = st["gated_ticks"]
                m, sd = vnet.predict(net, obs)
                last = (step, m, sd, k)
                vs.append((step, m, sd))
                chart_ticks.append((step, "gated"))
                ti = env.tick_info()
                rows.append({"tick": k, "tick_step": step, "t_s": step / hz, "v_est": m, "v_sd": sd,
                             "tilt_deg": float(ti["tilt_deg"]), "target_dxy": float(ti["target_dxy"]),
                             "proprio_gap": float(obs["proprio"][0]), "proprio_above": float(obs["proprio"][3])})
                status = ("GATED tick #{}:  V_est {:.2f} +- {:.2f}".format(k, m, sd), E.GREEN)
            else:
                why = not_gated_why(env, gate_open, ginfo, cooling_s)
                chart_ticks.append((step, "cool" if cooling_s > 0 else "not"))
                status = ("not gated: " + why, E.MAGENTA if cooling_s > 0 else E.RED)
        env.pi0_step()
    status = ("END: {}  (gated ticks {}, HL corrections {})".format(end, st["gated_ticks"], st["hl_corrections"]),
              E.GREEN if end == "success" else E.RED)
    flush()
    video += [video[-1]] * 40 if video else []
    st.update(end=end, main_picked=bool(env.pick_done()), steps=int(env.step_count),
              pick_step=int(env.pick_step()) if env.pick_done() else None, wall_s=round(time.time() - t_wall, 1))
    import imageio

    pathlib.Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(video_path), video, fps=E.FPS, macro_block_size=1)
    return st, rows


# ------------------------------------------------------------------ workers ------------------------

def _init_worker(cfg, ckpt):
    import torch

    torch.set_num_threads(1)
    import episodes  # noqa: F401  (puts ../V2 on sys.path)
    import vnet

    _W["cfg"] = dict(cfg, worker_id=os.getpid() % 100000)
    _W["net"], _ = vnet.load_ckpt(ckpt, "cpu")


def run_job(job):
    from libero_env import BranchingPickEnv
    from shared.scene import IneligibleStart, ResetFailed

    if "env" not in _W:
        _W["env"] = BranchingPickEnv(_W["cfg"])
    try:
        st, rows = v_rollout(_W["env"], job["start"], job["rc"], _W["net"], job["video"])
    except (IneligibleStart, ResetFailed) as e:
        return {"job": job["id"], "tag": job["start"]["tag"], "group": job["group"],
                "skip": "{}: {}".format(type(e).__name__, e)}
    return {"job": job["id"], "tag": job["start"]["tag"], "group": job["group"], "stats": st, "ticks": rows,
            "video": job["video"]}


# ------------------------------------------------------------------ main ---------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(HERE / "runs/v4_train/v_attn.pt"))
    p.add_argument("--out-dir", default=str(HERE / "runs/v4_train/rollouts"))
    p.add_argument("--eval-tasks", default=EVAL_TASKS)
    p.add_argument("--train-tasks", default=TRAIN_TASKS)
    p.add_argument("--inits", default="0-2")
    p.add_argument("--gate-max-above-cm", type=float, default=10.0, help="as collection")
    p.add_argument("--tick-steps", type=int, default=4)
    p.add_argument("--hl", default="privileged", choices=["privileged", "none"])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    import episodes as E
    from shared import runlog, scene as SC

    out = pathlib.Path(a.out_dir)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    rc = {"tick_steps": a.tick_steps}
    cfg = E.env_cfg(a.hl, a.host, a.port,
                    gate_max_above_m=a.gate_max_above_cm / 100.0 if a.gate_max_above_cm >= 0 else None)
    inits = SC.parse_ids(a.inits)
    starts = [("eval", s) for s in SC.task_starts([a.eval_tasks], inits, allow_held_out=True)] + \
             [("train", s) for s in SC.task_starts([a.train_tasks], inits)]
    jobs = [{"id": i, "start": s, "group": g, "rc": rc,
             "video": str(out / "videos" / "{}_{}.mp4".format(g, s["tag"]))} for i, (g, s) in enumerate(starts)]
    print("{} rollouts ({} eval-task, {} train-task), V = {}".format(
        len(jobs), sum(j["group"] == "eval" for j in jobs), sum(j["group"] == "train" for j in jobs), a.ckpt),
        flush=True)

    pool = runlog.ActorPool(run_job, a.workers, _init_worker, (cfg, a.ckpt))
    if pool.pool is None:
        _init_worker(cfg, a.ckpt)
    results = []
    try:
        it = pool.pool.imap_unordered(run_job, jobs, chunksize=1) if pool.pool else map(run_job, jobs)
        for res in it:
            results.append(res)
            if "skip" in res:
                print("  skip {}: {}".format(res["tag"], res["skip"]), flush=True)
                continue
            st, v = res["stats"], [t["v_est"] for t in res["ticks"]]
            print("[{}/{}] {:5s} {:24s} end {:8s} gated {:2d} | V_est first {} last {} | {:.0f} s".format(
                len(results), len(jobs), res["group"], res["tag"], st["end"], st["gated_ticks"],
                "{:.2f}".format(v[0]) if v else "-", "{:.2f}".format(v[-1]) if v else "-", st["wall_s"]), flush=True)
    finally:
        pool.close()

    results.sort(key=lambda r: r["job"])
    with open(str(out / "ticks.csv"), "w") as f:
        w = csv.writer(f)
        cols = ["tick", "tick_step", "t_s", "v_est", "v_sd", "tilt_deg", "target_dxy", "proprio_gap", "proprio_above"]
        w.writerow(["group", "tag", "end", "pick_step"] + cols)
        for r in results:
            for t in r.get("ticks", []):
                w.writerow([r["group"], r["tag"], r["stats"]["end"], r["stats"]["pick_step"]] +
                           [round(t[c], 4) if isinstance(t[c], float) else t[c] for c in cols])
    summary = []
    for r in results:
        s = {k: r.get(k) for k in ("tag", "group", "video", "skip")}
        if "stats" in r:
            v = [t["v_est"] for t in r["ticks"]]
            s.update(r["stats"], v_est=[round(x, 3) for x in v], v_mean=float(np.mean(v)) if v else None)
        summary.append(s)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for g in ("eval", "train"):
        rs = [s for s in summary if s["group"] == g and "end" in s]
        pk = [s for s in rs if s["main_picked"] and s["v_mean"] is not None]
        npk = [s for s in rs if not s["main_picked"] and s["v_mean"] is not None]
        print("{} tasks: {} rollouts, picked {} | mean V_est over gated ticks: picked rollouts {}, not picked {}".format(
            g, len(rs), sum(s["main_picked"] for s in rs),
            "{:.2f}".format(np.mean([s["v_mean"] for s in pk])) if pk else "-",
            "{:.2f}".format(np.mean([s["v_mean"] for s in npk])) if npk else "-"), flush=True)
    print("videos, ticks.csv, summary.json under {}".format(out), flush=True)


if __name__ == "__main__":
    main()
