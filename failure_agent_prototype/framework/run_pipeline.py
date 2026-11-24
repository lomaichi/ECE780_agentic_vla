"""Run the combined pipeline (psuedocode.txt) on LIBERO episodes: pi0 + Moondream2 failure agent + V trigger +
pixelQ V1 Q-map correction, with skill-library-tagged corrections. See README.md.

Inside the container (LIBERO client venv), from this folder:
  PY=/opt/openpi/examples/libero/.venv/bin/python
  # real run: pi0 server up (./serve_pi0.sh), Moondream2 on the GPU, trained V
  MUJOCO_GL=egl $PY run_pipeline.py --tasks 13 15 24 --inits 0 1 --out runs/eval_v0 --video
  # offline smoke test: no pi0 server, no GPU (scripted pi0, sim-box VLM, constant V)
  MUJOCO_GL=egl $PY run_pipeline.py --tasks 13 --pi0 stub-hover --vlm oracle --v-const 0.2 --out runs/smoke --video

Per episode, in --out/<suite>_task<T>_init<I>/:
  <name>.mp4             agentview (20 fps), as every fa_core.loop episode
  <name>_annotated.mp4   (--video) wrist | agentview with the supervisor's state per frame: VLM subtask, timer2,
                         V_est and the low-V count, gate, the running correction, privileged ground truth
  events.json            loop events + the framework report, every supervisor call, every correction (Q: trigger,
                         Q_INITIAL_POSITION, Q-map, chosen cell / yaw; HL: failure mode + library entry), VLM stats
  trace/*.npy            fa_core.loop trace (restorable with tools/scene_restore.py); --no-trace skips it
--out/summary.json and summary.md: one row per episode.
"""

import argparse
import bisect
import json
import logging
import pathlib
import time

import numpy as np

import config as C

C.ensure_paths()
from fa_core import common, loop, predicates as P  # noqa: E402

import components as K  # noqa: E402
from supervisor import FrameworkAgent  # noqa: E402

S = common.CONTROL_FREQ
log = logging.getLogger("framework")


# ------------------------------------------------------------------ one episode ----------------------

def pi0_client(a, env, obj):
    if a.pi0 == "server":
        m = K.load_modules()
        log.info("pi0: websocket server at %s:%d", a.host, a.port)
        return m["scene"].connect_pi0(a.host, a.port)
    import stubs

    if obj is None:
        raise RuntimeError("the scripted pi0 stub needs a pick target")
    return stubs.ScriptedPi0(env, obj, mode=a.pi0.split("-", 1)[1])


def run_one(a, suite, task_id, init_idx, models, library, vlm_backend, out):
    name = "{}_task{}_init{}".format(suite, task_id, init_idx)
    t_wall = time.time()
    np.random.seed(a.seed)
    env, task, _ = common.make_env(suite, task_id, a.env_seed)
    env.env.horizon = 100000            # robosuite's default 1000-step horizon would end long runs
    try:
        env.reset()
        init = common.init_state(suite, task_id, init_idx)
        obs = env.set_init_state(init)
        language = str(task.language)
        # privileged ground truth: logged and drawn, never shown to the agent
        checks = [c for c in P.goal_checks(env) if c.kind in ("pick", "place")] + [P.GoalCheck()]
        picks = [c.obj for c in checks if c.kind == "pick"]
        obj = picks[0] if picks else None       # the target's sim name (the privileged mask / grounding source)
        every = max(1, int(round(S / a.monitor_fps)))
        if a.vlm == "oracle":
            import stubs

            vlm_backend = stubs.OracleDetectVLM(env)
        calls0, lat0 = getattr(vlm_backend, "calls", 0), getattr(vlm_backend, "latency_s", 0.0)
        vlm = K.make_vlm_monitor(vlm_backend, env, every, log=log)
        per = None
        if obj is not None:
            per = K.Perception(env, obj, vlm, target_mask=a.target_mask, approach_dz=a.approach_dz,
                               q_cfg=models["q"].train_cfg, seed=a.seed)
        else:
            log.warning("%s: no pick-and-place goal: VLM high-level monitoring only", name)
        agent = FrameworkAgent(vlm, per, models["v"], models["q"], library=library, threshold=a.threshold,
                               threshold_count=a.threshold_count, timeout_s=a.timeout_s, monitor_fps=a.monitor_fps,
                               cooldown_s=a.cooldown_s, held_grace_s=a.held_grace_s, log=log)
        client = pi0_client(a, env, obj)
        wrist = {}

        def hook(step, o):
            if a.video:
                wrist[step] = common.upright(o["robot0_eye_in_hand_image"])

        res = loop.run_episode(env, obs, language, int(round(a.full_timeout_s * S)), client=client, agent=agent,
                               checks=checks, task_language=language, init_state=init, stop_on_success=True,
                               step_hook=hook, log=log)
        held_out = suite == "libero_90" and task_id in library.held_out_tasks()
        report = agent.report()
        info = {"name": name, "suite": suite, "task_id": task_id, "init_state_idx": init_idx, "language": language,
                "split": "held_out" if held_out else "survey", "target_object": obj, "success": res.success,
                "num_steps": res.num_steps, "sim_s": round(res.num_steps / float(S), 2),
                "wall_s": round(time.time() - t_wall, 1), "report": report,
                "gt": {c["spec"]: c.get("done_step") for c in res.checks}}
        extra = {"framework": info, "config": config_dict(a, models),
                 "vlm": vlm.stats(calls0, lat0), "corrections": agent.corrections, "calls": agent.calls}
        if a.no_trace:
            res.trace = {}
        d = loop.write_episode(res, out, name, extra=extra)
        if a.video:
            annotate_video(res, agent, wrist, info, d / "{}_annotated.mp4".format(name), a)
        info["vlm_calls"] = extra["vlm"]["vlm_calls"]
        return info
    finally:
        env.close()


def config_dict(a, models):
    c = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(a).items()}
    c["q_model"] = {"ckpt": models["q"].ckpt, "width": models["q"].width}
    c["v_model"] = {"ckpt": models["v"].ckpt, "meta": getattr(models["v"], "meta", None)}
    return c


# ------------------------------------------------------------------ video ----------------------------

WHITE, GREY, GREEN, RED, YELLOW, CYAN, ORANGE = ((235, 235, 235), (150, 150, 150), (90, 220, 90), (240, 80, 70),
                                                 (250, 220, 60), (90, 210, 240), (250, 160, 50))   # RGB


def _gate_text(c):
    g = c.get("gate") or {}
    if "cooldown_s" in g:
        return "cooldown {:.1f} s".format(g["cooldown_s"])
    bad = [k for k in ("pick", "open", "empty", "near", "low", "grounded") if g.get(k) is False]
    return ("CLOSE" if c.get("close") else "closed: " + ", ".join(bad)) if g else "-"


def annotate_video(res, agent, wrist, info, path, a, scale=2):
    """wrist | agentview, 2x, with a banner of the supervisor's state at every frame (20 fps = sim time)."""
    import cv2
    import imageio

    calls = sorted(agent.calls, key=lambda c: c["step"])
    steps = [c["step"] for c in calls]
    spans, cur = [], None
    for e in res.events:
        if e["type"] == "override_start":
            cur = [e["step"], None, e.get("reason", "")]
        elif e["type"] == "override_end" and cur is not None:
            cur[1] = e["step"]
            spans.append(cur)
            cur = None
    if cur is not None:
        spans.append([cur[0], 10 ** 9, cur[2]])
    gt = info["gt"]
    banner_h, lh = 150, 22
    frames = []
    for t, img in enumerate(res.frames):
        w = wrist.get(t, np.zeros_like(img))
        view = np.concatenate([w, img], axis=1)
        view = cv2.resize(view, (view.shape[1] * scale, view.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        k = bisect.bisect_right(steps, t) - 1
        c = calls[k] if k >= 0 else {}
        ov = next((s for s in spans if s[0] <= t < s[1]), None)
        now, col = ("pi0", WHITE) if ov is None else (ov[2], YELLOW if ov[2].startswith("Q[") else CYAN)
        v = "V_est {:.2f} +- {:.2f}".format(c["v_est"], c["v_std"]) if "v_est" in c else "V_est -"
        low = c.get("low_count", 0 if c.get("branch") in ("v_ok", "v_low") else None)
        gt_txt = "  ".join("{}@{}".format(sp.split("(")[0], ds) for sp, ds in gt.items() if ds is not None and ds <= t)
        lines = [
            ("{} | step {} ({:.1f} s) | {}".format(info["name"], t, max(0, t - common.NUM_STEPS_WAIT) / float(S),
                                                  info["language"][:70]), WHITE),
            ("subtask {} | timer2 {} / {:g} s | branch {}".format(
                c.get("subtask"), "-" if c.get("timer2_s") is None else "{:.1f}".format(c["timer2_s"]),
                a.timeout_s, c.get("branch", "-")), WHITE),
            ("{} (thr {:g}, low {}/{}) | close_to_object: {}".format(
                v, a.threshold, "-" if low is None else low, a.threshold_count, _gate_text(c)),
             RED if "v_est" in c and c["v_est"] < a.threshold else GREEN if "v_est" in c else GREY),
            ("NOW: " + now[:88], col),
            ("VLM: {} | GT (privileged): {}".format(
                c.get("vlm_failure_mode") or c.get("vlm_failure_suspected") or "no failure", gt_txt or "-"), GREY),
        ]
        banner = np.zeros((banner_h, view.shape[1], 3), np.uint8)
        for i, (txt, color) in enumerate(lines):
            cv2.putText(banner, txt, (8, 24 + i * (lh + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        frames.append(np.concatenate([banner, view], axis=0))
    imageio.mimwrite(str(path), frames, fps=S)


# ------------------------------------------------------------------ main -----------------------------

def write_summary(rows, out):
    (out / "summary.json").write_text(json.dumps(rows, indent=2, default=str))
    head = ("| episode | split | success | sim s | Q (V-low / timeout / skipped) | HL (library) | V evals (low) | "
            "VLM calls | wall s |\n|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in rows:
        if "error" in r:
            body += "| {} | | ERROR: {} | | | | | | |\n".format(r["name"], r["error"][:80])
            continue
        p = r["report"]
        hl = ", ".join("{} x{}".format(k, v) for k, v in sorted(p["hl_by_library"].items())) or "-"
        body += "| {} | {} | {} | {} | {} / {} / {} | {} | {} ({}) | {} | {} |\n".format(
            r["name"], r["split"], "yes" if r["success"] else "no", r["sim_s"], p["q_v_low"], p["q_timeout"],
            p["q_skipped"], hl, p["v_evals"], p["v_low_frames"], r.get("vlm_calls"), r["wall_s"])
    ok = [r for r in rows if "error" not in r]
    foot = "\nsuccess {}/{}\n".format(sum(r["success"] for r in ok), len(ok))
    (out / "summary.md").write_text(head + body + foot)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_90")
    p.add_argument("--tasks", type=int, nargs="+", required=True)
    p.add_argument("--inits", type=int, nargs="+", default=[0])
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/pipeline"))
    p.add_argument("--pi0", default="server", choices=["server", "stub-hover", "stub-wander", "stub-idle"],
                   help="server = the pi0 websocket server; stub-* = stubs.ScriptedPi0 (offline tests only)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--vlm", default="moondream2", choices=["moondream2", "oracle"],
                   help="moondream2 = local GPU (~5 GB); oracle = stubs.OracleDetectVLM (sim boxes, offline tests)")
    p.add_argument("--target-mask", default="seg", choices=["seg", "vlm"],
                   help="target mask for the V / Q observations: sim segmentation (as in training) or the VLM box")
    p.add_argument("--q-ckpt", type=pathlib.Path, default=C.DEFAULT_Q_CKPT)
    p.add_argument("--v-ckpt", type=pathlib.Path, default=C.DEFAULT_V_CKPT)
    p.add_argument("--v-const", type=float, default=None,
                   help="stand-in V: constant value instead of --v-ckpt (until V4 is trained)")
    p.add_argument("--threshold", type=float, default=C.THRESHOLD)
    p.add_argument("--threshold-count", type=int, default=C.THRESHOLD_COUNT)
    p.add_argument("--timeout-s", type=float, default=C.TIMEOUT_S)
    p.add_argument("--full-timeout-s", type=float, default=C.FULL_TASK_TIMEOUT_S)
    p.add_argument("--approach-dz", type=float, default=C.APPROACH_DZ)
    p.add_argument("--monitor-fps", type=float, default=C.MONITOR_FPS)
    p.add_argument("--cooldown-s", type=float, default=C.CORRECTION_COOLDOWN_S)
    p.add_argument("--held-grace-s", type=float, default=C.TIMEOUT_HELD_GRACE_S)
    p.add_argument("--video", action="store_true", help="also write the annotated video")
    p.add_argument("--no-trace", action="store_true", help="do not write trace/*.npy")
    p.add_argument("--env-seed", type=int, default=7)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu", help="torch device for Q and V (LIBERO venv: torch 1.11, CPU)")
    p.add_argument("--torch-threads", type=int, default=2)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    import torch

    torch.set_num_threads(a.torch_threads)
    if a.v_const is None and not a.v_ckpt.exists():
        raise SystemExit("no V checkpoint at {} (train it with V4/train_v.py, pass --v-ckpt, or use --v-const X for a "
                         "stand-in)".format(a.v_ckpt))
    models = {"q": K.QModel(a.q_ckpt, a.device),
              "v": K.ConstantV(a.v_const) if a.v_const is not None else K.VModel(a.v_ckpt, a.device)}
    log.info("Q: %s (width %d) | V: %s", models["q"].ckpt, models["q"].width, models["v"].ckpt)
    library = K.SkillLibrary()
    vlm_backend = K.make_vlm_backend(a.vlm, log=log)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tid in a.tasks:
        for init in a.inits:
            try:
                r = run_one(a, a.suite, tid, init, models, library, vlm_backend, out)
            except Exception as e:
                log.exception("%s task %d init %d failed", a.suite, tid, init)
                r = {"name": "{}_task{}_init{}".format(a.suite, tid, init), "error": repr(e)}
            rows.append(r)
            if "error" not in r:
                p = r["report"]
                log.info("%s: success %s in %.1f s sim | Q %d (V-low %d, timeout %d, skipped %d) | HL %d %s | "
                         "V evals %d (low %d) | %.0f s wall", r["name"], r["success"], r["sim_s"], p["q_corrections"],
                         p["q_v_low"], p["q_timeout"], p["q_skipped"], p["hl_corrections"], p["hl_by_library"],
                         p["v_evals"], p["v_low_frames"], r["wall_s"])
            write_summary(rows, out)
    log.info("summary: %s", out / "summary.md")


if __name__ == "__main__":
    main()
