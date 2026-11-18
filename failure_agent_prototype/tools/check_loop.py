"""Phase 1 checks of the control loop (fa_core/loop.py) that need NO pi0 server.

  replay  Feed recorded episodes' actions through the loop (ReplayAgent). The loop must reproduce
          the recorded states step for step (same seed, same init state, same actions), and the
          trace it writes must be restorable: scene_restore.reset_to at the middle step, replay
          20 steps, compare. Checks the loop's stepping, override path and logging.
  motion  From an init state, run scripted MoveTo legs (up 10 cm, +5 cm x, back, down) and report
          how close each leg gets. Checks the OSC frame/gain assumption behind fa_core/motion.py
          before Phase 2 builds RETREAT / POINT_APPROACH on it. Watch the saved video.

Run in the LIBERO client venv (MUJOCO_GL=egl):
  python check_loop.py replay [--episodes libero_90/task1_fail_init0 ...] [--tol 1e-6] [--restore-tol 1e-3]
  python check_loop.py motion [--suite libero_90 --task 1 --init 0]
Output goes to Documentation/phase1/check_loop/ (override with --out). Exit code 0 = PASS.
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fa_core import agents, common, loop, motion  # noqa: E402

DEFAULT_OUT = common.REPO / "Documentation" / "phase1" / "check_loop"
DEFAULT_EPISODES = ["libero_spatial/task0_success_init0", "libero_10/task3_success_init0",
                    "libero_90/task1_fail_init0", "libero_90/task19_success_init0"]


# ------------------------------------------------------------------ replay -------------------------

def replay_one(episode, tol, restore_steps, restore_tol):
    sr = common.tools_module("scene_restore")
    meta, entry, trace_dir = sr.episode_info(episode)
    if entry.get("error"):
        return {"episode": episode, "verdict": "SKIP", "reason": "recorded episode ended in a sim error"}
    rec = {k: np.load(trace_dir / f"{k}.npy") for k in ("states", "actions", "init_state", "dones")}
    n, wait = len(rec["actions"]), int(meta["num_steps_wait"])

    np.random.seed(meta["env_seed"])  # as record_suites.py
    env, task, _ = common.make_env(meta["task_suite"], entry["task_id"], meta["env_seed"], meta["camera_resolution"])
    try:
        env.reset()
        obs = env.set_init_state(rec["init_state"])
        res = loop.run_episode(env, obs, str(task.language), max_steps=n - wait, client=None,
                               agent=agents.ReplayAgent(rec["actions"]), num_steps_wait=wait,
                               init_state=rec["init_state"])
    finally:
        env.close()

    got = res.trace["states"]
    m = min(len(got), n)
    err = np.abs(got[:m] - rec["states"][:m])[:, 1:]  # skip sim time
    max_err = float(err.max()) if m else float("nan")
    first_bad = int(np.argmax(err.max(1) > tol)) if m and max_err > tol else None
    same_len = len(got) == n
    same_success = res.success == bool(entry["success"])

    # restorability of the loop's own trace: restore mid-episode, replay, compare with the loop's states
    k = m // 2
    k_end = min(k + restore_steps, m - 1)
    env2, _ = sr.env_for_episode(episode)
    try:
        sr.reset_to(env2, res.trace, k)
        for t in range(k, k_end):
            env2.step(res.trace["actions"][t].tolist())
        nq = int(env2.env.sim.model.nq)
        restore_err = float(np.abs(env2.get_sim_state()[1:1 + nq] - got[k_end][1:1 + nq]).max())  # qpos
    finally:
        env2.close()

    ok = same_len and same_success and max_err <= tol and restore_err <= restore_tol
    return {"episode": episode, "verdict": "PASS" if ok else "FAIL", "steps_recorded": n, "steps_loop": len(got),
            "success_recorded": bool(entry["success"]), "success_loop": res.success,
            "max_state_err": max_err, "first_step_over_tol": first_bad,
            "restore_from_loop_trace": {"step": k, "to": k_end, "qpos_max_err": restore_err},
            "events": [e["type"] for e in res.events]}


def cmd_replay(a):
    rows = []
    for ep in a.episodes:
        try:
            row = replay_one(ep, a.tol, a.restore_steps, a.restore_tol)
        except Exception as e:
            row = {"episode": ep, "verdict": "ERROR", "error": repr(e)}
        rows.append(row)
        print(json.dumps(row))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "replay.json").write_text(json.dumps(rows, indent=2))
    bad = [r for r in rows if r["verdict"] not in ("PASS", "SKIP")]
    print(f"\nreplay: {len(rows) - len(bad)}/{len(rows)} PASS/SKIP (tol {a.tol:g}); report {a.out / 'replay.json'}")
    return 1 if bad else 0


# ------------------------------------------------------------------ motion -------------------------

class MotionTestAgent(agents.Agent):
    """At its first call: scripted legs relative to the current EE position, then stop."""

    name = "motion_test"

    def __init__(self, legs):
        self.legs, self.motion, self.started = legs, None, False

    def act(self, view, priv=None):
        if self.started:
            return loop.Stop("motion test finished")
        self.started = True
        p0 = view.ee_pos
        self.motion = motion.Sequence([motion.MoveTo(p0 + np.asarray(d), common.GRIPPER_OPEN, quat=view.ee_quat)
                                       for d in self.legs])
        return loop.Override(self.motion, reason="motion test")


def cmd_motion(a):
    legs = [(0, 0, 0.10), (0.05, 0, 0.10), (-0.05, 0, 0.10), (0, 0, 0)]
    np.random.seed(7)
    env, task, _ = common.make_env(a.suite, a.task)
    agent = MotionTestAgent(legs)
    try:
        env.reset()
        obs = env.set_init_state(common.init_state(a.suite, a.task, a.init))
        start = np.array(obs["robot0_eef_pos"], copy=True)
        res = loop.run_episode(env, obs, str(task.language), max_steps=4 * motion.LEG_STEPS + 5,
                               client=None, agent=agent)
    finally:
        env.close()
    a.out.mkdir(parents=True, exist_ok=True)
    loop.write_episode(res, a.out, "motion_test")
    parts = agent.motion.describe()["parts"]
    rows = []
    for d, part in zip(legs, parts):
        rows.append({"leg_offset_m": d, "target": part["pos"], "reached": part["reached"], "steps": part["steps"],
                     "pos_err_m": part["pos_err_m"], "rot_err_rad": part["rot_err_rad"]})
        print(json.dumps(rows[-1]))
    ok = all(r["reached"] for r in rows) and not res.error
    print(f"\nstart EE {np.round(start, 3).tolist()}; motion: {'PASS' if ok else 'FAIL'} "
          f"(each leg must reach {motion.POS_TOL * 100:.1f} cm / {motion.ROT_TOL} rad within {motion.LEG_STEPS} steps)")
    print(f"video: {a.out / 'motion_test' / 'motion_test.mp4'}: the gripper should go straight up, right, "
          f"left, then straight down, without tilting")
    (a.out / "motion.json").write_text(json.dumps({"ok": ok, "legs": rows, "error": res.error}, indent=2))
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--episodes", nargs="+", default=DEFAULT_EPISODES, help="<suite>/<folder or episode id>")
    r.add_argument("--tol", type=float, default=1e-6, help="max |state difference| allowed")
    r.add_argument("--restore-steps", type=int, default=20)
    r.add_argument("--restore-tol", type=float, default=1e-3,
                   help="max qpos error after restoring the loop's trace mid-episode (same as scene_restore.py)")
    r.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    m = sub.add_parser("motion")
    m.add_argument("--suite", default="libero_90")
    m.add_argument("--task", type=int, default=1)
    m.add_argument("--init", type=int, default=0)
    m.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    a = p.parse_args()
    sys.exit(cmd_replay(a) if a.cmd == "replay" else cmd_motion(a))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
