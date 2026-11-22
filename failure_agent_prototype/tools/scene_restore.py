"""Restore a LIBERO scene from a recorded pi0 rollout trace, to start RL episodes at LL failure states.

Where the inputs live:
  reset states   failure_survey/skill_library/library.json -> rl_policies[<policy>].reset_states
                 = [{episode: "<suite>/<folder>", step, onset_step}, ...]; `step` is the reset point
                 picked before the LL failure onset (noise_steering_recovery_skills.md, Step 0)
  traces         Documentation/pi0_LIBERO_rollouts/<suite>/<folder>/trace/*.npy
                 (recorded by record_suites.py in this folder)

Step convention (meta.json): trace/states.npy[t] is the sim state before actions[t], and video frame t
is the agentview image of that state. Steps count the 10 wait steps, so survey frame numbers apply as-is.

Restoring only with env.set_init_state(states[t]) renders the right scene (that is all
Documentation/scene_reconstruction_test checked), but stepping on from it drifts. To continue from t
the controller, the gripper's last action and MuJoCo's warm start must be restored too (reset_to below).

Check mode: for every reset state, compare the restored render with the video frame, then replay the
recorded actions for --replay-steps and compare the reached state with the trace. Run in the LIBERO
client venv (MUJOCO_GL=egl):
  python scene_restore.py --check [--policy rl/pick] [--replay-steps 20] [--out DIR]

Each reset state gets a verdict: PASS (qpos within --max-qpos-err after the replay), LOOK (state OK
but the picture differs from the video more than --warn-pix: check the PNG), FAIL (state drifted) or
ERROR. The exit code is 0 only if every state is PASS. Velocities (qvel) are reported, not judged.
Also check every PNG by eye: is the restored scene really the moment BEFORE the failing attempt?

In-memory snapshots of a RUNNING rollout (capture_snapshot / restore_snapshot; pixel-Q V2 branch points) also
keep the controller's state and robosuite's step counter. Check them (no recordings or pi0 needed):
  python scene_restore.py --snapshot-check [--suite libero_90 --task 10 --init 0]
"""

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROLLOUTS = REPO / "Documentation" / "pi0_LIBERO_rollouts"
LIBRARY = HERE.parent / "failure_survey" / "skill_library" / "library.json"


def load_reset_states(policy=None):
    """Reset states from library.json, flattened to one dict per state, optionally for one policy."""
    lib = json.loads(LIBRARY.read_text())
    out = []
    for name, pol in lib["rl_policies"].items():
        if policy is not None and name != policy:
            continue
        for rs in pol["reset_states"]:
            out.append({"policy": name, "subtask": pol["subtask"], "horizon_steps": pol["horizon_steps"], **rs})
    if policy is not None and not out:
        raise KeyError(f"no reset states for {policy!r}; policies: {list(lib['rl_policies'])}")
    return out


def episode_info(episode):
    """(suite meta.json settings, this episode's meta entry, trace dir) for "<suite>/<folder>"."""
    suite, folder = episode.split("/")
    meta = json.loads((ROLLOUTS / suite / "meta.json").read_text())
    # accept the full folder name or the short id ("task0_fail_init0") used in survey/library files
    entry = next(e for e in meta["episodes"] if folder in (e["folder"], e.get("episode_id")))
    return meta, entry, ROLLOUTS / suite / entry["folder"] / "trace"


def load_trace(trace_dir, fields=("states", "gripper_current_action", "qacc_warmstart", "actions")):
    return {f: np.load(pathlib.Path(trace_dir) / f"{f}.npy") for f in fields}


def make_env(suite, task_id, env_seed, resolution=256):
    """Same env construction as record_suites.py."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
    env.seed(env_seed)
    return env, task


def env_for_episode(episode):
    """Env for a recorded episode, built with the recording's suite, task and seed."""
    meta, entry, _ = episode_info(episode)
    return make_env(meta["task_suite"], entry["task_id"], meta["env_seed"], meta["camera_resolution"])


def reset_to(env, trace, step):
    """Put env in the exact state before actions[step] so stepping continues as the recording did.

    trace: dict from load_trace(). Follows meta.json state_convention. Returns the observation.
    """
    env.reset()
    obs = env.set_init_state(trace["states"][step])
    inner = env.env  # robosuite env under LIBERO's ControlEnv
    robot = inner.robots[0]
    robot.controller.update(force=True)
    robot.gripper.current_action = trace["gripper_current_action"][step].copy()
    inner.sim.data.qacc_warmstart[:] = trace["qacc_warmstart"][step]
    return obs


def upright(img):
    """LIBERO renders upside down; rotate 180 deg (what pi0 and the videos see)."""
    return np.ascontiguousarray(img[::-1, ::-1])


# ------------------------------------------------------------------ in-memory snapshots ------------
# For pixel-Q V2 branch points (RL/pixel_Q/V2): rewind a RUNNING rollout in memory, without env.reset() (which
# rebuilds the scene). Unlike reset_to() (restoring a recorded trace step), this also keeps the controller's
# internal state and robosuite's step counter. Callers keep their own bookkeeping (pi0's queued actions,
# subtask checks, step count of their rollout) next to it.

def _plain_attrs(obj):
    """Copies of an object's numpy / scalar attributes (an OSC controller's goals, cached state and flags)."""
    out = {}
    for k, v in vars(obj).items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif v is None or isinstance(v, (np.generic, bool, int, float)):
            out[k] = v
    return out


def capture_snapshot(env):
    """Everything stepping on from the current state depends on: flattened sim state (time, qpos, qvel),
    MuJoCo's warm start, actuator controls (and activations), the gripper's last action, the controller's
    numpy / scalar attributes and robosuite's timestep / cur_time / done."""
    inner = env.env
    sim, robot = inner.sim, inner.robots[0]
    d = sim.data
    snap = {"sim_state": np.array(env.get_sim_state(), copy=True),
            "qacc_warmstart": np.array(d.qacc_warmstart, copy=True),
            "ctrl": np.array(d.ctrl, copy=True),
            "gripper_current_action": np.array(robot.gripper.current_action, copy=True),
            "controller": _plain_attrs(robot.controller),
            "env": {k: getattr(inner, k) for k in ("timestep", "cur_time", "done") if hasattr(inner, k)}}
    act = getattr(d, "act", None)
    if act is not None and np.size(act):
        snap["act"] = np.array(act, copy=True)
    return snap


def restore_snapshot(env, snap):
    """Put a running env back into a capture_snapshot() state (the snapshot itself is not modified)."""
    inner = env.env
    sim, robot = inner.sim, inner.robots[0]
    sim.set_state_from_flattened(np.asarray(snap["sim_state"]))
    sim.data.qacc_warmstart[:] = snap["qacc_warmstart"]
    sim.data.ctrl[:] = snap["ctrl"]
    if "act" in snap:
        sim.data.act[:] = snap["act"]
    sim.forward()
    robot.gripper.current_action = np.array(snap["gripper_current_action"], copy=True)
    robot.controller.update(force=True)
    for k, v in snap["controller"].items():          # after update(): flags such as new_update as they were
        setattr(robot.controller, k, v.copy() if isinstance(v, np.ndarray) else v)
    for k, v in snap["env"].items():
        setattr(inner, k, v)


def snapshot_check(suite="libero_90", task_id=10, init_idx=0, before=30, after=40, seed=0, env_seed=7):
    """Run `before` random actions, snapshot, run `after` more (run A), restore, run the same `after` (run B).
    Deterministic MuJoCo + a complete snapshot -> A and B identical (qpos, qvel, images, controller goal,
    timestep). Restore twice to show the snapshot is not consumed. Returns a report dict with a verdict."""
    from libero.libero import benchmark

    env, _ = make_env(suite, task_id, env_seed)
    rng = np.random.default_rng(seed)
    try:
        env.reset()
        env.set_init_state(benchmark.get_benchmark_dict()[suite]().get_task_init_states(task_id)[init_idx])
        for _ in range(10):
            env.step([0.0] * 6 + [-1.0])
        acts = np.c_[rng.uniform(-0.4, 0.4, (before + after, 3)), rng.uniform(-0.15, 0.15, (before + after, 3)),
                     np.where(rng.random(before + after) < 0.8, -1.0, 1.0)]
        for a in acts[:before]:
            env.step(a.tolist())
        snap = capture_snapshot(env)
        nq = int(env.env.sim.model.nq)

        def run():
            rec = {"state": [], "img": [], "goal": [], "timestep": []}
            for a in acts[before:]:
                obs, *_ = env.step(a.tolist())
                rec["state"].append(np.array(env.get_sim_state(), copy=True))
                rec["img"].append(obs["agentview_image"].astype(np.int16))
                rec["goal"].append(np.array(getattr(env.env.robots[0].controller, "goal_pos", np.zeros(3)), copy=True))
                rec["timestep"].append(int(getattr(env.env, "timestep", -1)))
            return rec

        A = run()
        restore_snapshot(env, snap)
        B = run()
        restore_snapshot(env, snap)
        C = run()
    finally:
        env.close()
    rows = {}
    for name, X in (("restore_1", B), ("restore_2", C)):
        sA, sX = np.array(A["state"]), np.array(X["state"])
        rows[name] = {"qpos_max_abs_err": float(np.abs(sA[:, 1:1 + nq] - sX[:, 1:1 + nq]).max()),
                      "qvel_max_abs_err": float(np.abs(sA[:, 1 + nq:] - sX[:, 1 + nq:]).max()),
                      "pix_max_abs_err": int(max(np.abs(a - b).max() for a, b in zip(A["img"], X["img"]))),
                      "goal_max_abs_err": float(np.abs(np.array(A["goal"]) - np.array(X["goal"])).max()),
                      "timestep_equal": A["timestep"] == X["timestep"]}
    ok = all(r["qpos_max_abs_err"] <= 1e-9 and r["qvel_max_abs_err"] <= 1e-6 and r["pix_max_abs_err"] == 0
             and r["timestep_equal"] for r in rows.values())
    return {"verdict": "PASS" if ok else "FAIL", "suite": suite, "task_id": task_id, "init": init_idx,
            "before": before, "after": after, **rows}


def check_one(rs, replay_steps, out, max_qpos_err, warn_pix):
    """Restore one reset state, replay the recorded actions, compare with the recording. Returns a report row."""
    import imageio.v2 as imageio

    meta, entry, trace_dir = episode_info(rs["episode"])
    trace = load_trace(trace_dir)
    n_states = len(trace["states"])
    step = rs["step"]
    if not 0 <= step < n_states:
        raise ValueError(f"reset step {step} outside the trace (0..{n_states - 1})")
    end = min(step + replay_steps, n_states - 1)
    onset = min(int(rs.get("onset_step", step)), n_states - 1)

    env, _ = env_for_episode(rs["episode"])
    try:
        obs = reset_to(env, trace, step)
        restored = upright(obs["agentview_image"])
        for t in range(step, end):
            obs, *_ = env.step(trace["actions"][t].tolist())
        replayed = upright(obs["agentview_image"])
        reached = env.get_sim_state()
        nq = int(env.env.sim.model.nq)
    finally:
        env.close()

    reader = imageio.get_reader(trace_dir.parent / entry["video"])
    v_step, v_end, v_onset = [reader.get_data(s)[..., :3] for s in (step, end, onset)]
    reader.close()

    # flattened sim state = [time, qpos (nq), qvel]; positions decide whether the reset is usable
    diff = np.abs(reached - trace["states"][end])
    qpos_err = float(diff[1:1 + nq].max())
    qvel_err = float(diff[1 + nq:].max()) if diff.size > 1 + nq else 0.0
    # the video is H.264-compressed, so a perfect restore still differs by a few grey levels
    pix_restore = float(np.abs(v_step.astype(int) - restored).mean())
    pix_replay = float(np.abs(v_end.astype(int) - replayed).mean())

    # top row: video @reset | restored @reset | video @onset (what the failure looks like)
    # bottom row: video @reset+replay | replayed | blank
    name = f"{rs['policy'].replace('/', '_')}__{rs['episode'].replace('/', '_')}_step{step:04d}"
    blank = np.zeros_like(v_step)
    imageio.imwrite(out / f"{name}.png", np.concatenate(
        [np.concatenate([v_step, restored, v_onset], 1), np.concatenate([v_end, replayed, blank], 1)], 0))

    verdict = "PASS" if qpos_err <= max_qpos_err else "FAIL"
    if verdict == "PASS" and max(pix_restore, pix_replay) > warn_pix:
        verdict = "LOOK"  # state matches but the picture differs: check the PNG by eye
    return {"verdict": verdict, "policy": rs["policy"], "episode": rs["episode"], "reset_step": step,
            "onset_step": rs.get("onset_step"), "replay_to": end,
            "qpos_max_abs_err": qpos_err, "qvel_max_abs_err": qvel_err,
            "pix_mae_restore": round(pix_restore, 2), "pix_mae_replay": round(pix_replay, 2),
            "image": f"{name}.png"}


def check(policy, replay_steps, out, max_qpos_err=1e-3, warn_pix=8.0):
    """Check every reset state; returns the number of non-PASS rows (FAIL, LOOK or ERROR)."""
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for rs in load_reset_states(policy):
        try:
            row = check_one(rs, replay_steps, out, max_qpos_err, warn_pix)
        except Exception as e:  # keep checking the other states
            row = {"verdict": "ERROR", "policy": rs["policy"], "episode": rs["episode"],
                   "reset_step": rs["step"], "error": repr(e)}
        report.append(row)
        print(json.dumps(row))
    (out / "report.json").write_text(json.dumps(report, indent=2))

    counts = {v: sum(r["verdict"] == v for r in report) for v in ("PASS", "LOOK", "FAIL", "ERROR")}
    print(f"\nthresholds: qpos <= {max_qpos_err:g} (m or rad) after {replay_steps} replayed steps; "
          f"LOOK if the mean pixel difference > {warn_pix:g}")
    print(f"{counts}  ({len(report)} reset states)")
    for r in report:
        if r["verdict"] != "PASS":
            print(f"  {r['verdict']:5s} {r['policy']:16s} {r['episode']} @ {r['reset_step']}"
                  + (f"  qpos err {r['qpos_max_abs_err']:.2e}" if "qpos_max_abs_err" in r else f"  {r.get('error')}"))
    print(f"images and report.json in {out}\n(each PNG, top: video at reset | restored | video at onset; "
          f"bottom: video after replay | replayed)")
    return len(report) - counts["PASS"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="verify restore + replay against the recordings")
    p.add_argument("--policy", default=None, help="e.g. rl/pick; default all policies")
    p.add_argument("--replay-steps", type=int, default=20)
    p.add_argument("--out", type=pathlib.Path, default=HERE.parent / "RL" / "restore_check")
    p.add_argument("--max-qpos-err", type=float, default=1e-3,
                   help="PASS if every qpos entry is within this after the replay (1e-3 = 1 mm / 1 mrad)")
    p.add_argument("--warn-pix", type=float, default=8.0,
                   help="mark LOOK if the mean pixel difference to the video exceeds this (0-255 scale)")
    p.add_argument("--snapshot-check", action="store_true",
                   help="in-memory snapshot / restore of a running rollout (pixel-Q V2 branch points): run random "
                        "actions, snapshot, continue, restore, repeat; the two continuations must be identical")
    p.add_argument("--suite", default="libero_90")
    p.add_argument("--task", type=int, default=10)
    p.add_argument("--init", type=int, default=0)
    args = p.parse_args()
    if args.snapshot_check:
        rep = snapshot_check(args.suite, args.task, args.init)
        print(json.dumps(rep, indent=2))
        sys.exit(0 if rep["verdict"] == "PASS" else 1)
    if args.check:
        sys.exit(1 if check(args.policy, args.replay_steps, args.out, args.max_qpos_err, args.warn_pix) else 0)
    else:
        for rs in load_reset_states(args.policy):
            print(json.dumps(rs))


if __name__ == "__main__":
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
