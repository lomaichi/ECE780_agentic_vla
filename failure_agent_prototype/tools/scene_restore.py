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
"""

import argparse
import json
import pathlib

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
    entry = next(e for e in meta["episodes"] if e["folder"] == folder)
    return meta, entry, ROLLOUTS / suite / folder / "trace"


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


def check(policy, replay_steps, out):
    import imageio.v2 as imageio

    out.mkdir(parents=True, exist_ok=True)
    report = []
    for rs in load_reset_states(policy):
        meta, entry, trace_dir = episode_info(rs["episode"])
        trace = load_trace(trace_dir)
        step = rs["step"]
        end = min(step + replay_steps, len(trace["states"]) - 1)

        env, _ = env_for_episode(rs["episode"])
        obs = reset_to(env, trace, step)
        restored = upright(obs["agentview_image"])
        for t in range(step, end):
            obs, *_ = env.step(trace["actions"][t].tolist())
        replayed = upright(obs["agentview_image"])
        reached = env.get_sim_state()
        env.close()

        reader = imageio.get_reader(trace_dir.parent / entry["video"])
        frames = [reader.get_data(s)[..., :3] for s in (step, end)]
        reader.close()

        # flattened sim state = [time, qpos, qvel]; skip time
        state_err = float(np.abs(reached[1:] - trace["states"][end][1:]).max())
        pix_restore = float(np.abs(frames[0].astype(int) - restored).mean())
        pix_replay = float(np.abs(frames[1].astype(int) - replayed).mean())
        name = f"{rs['episode'].replace('/', '_')}_step{step:04d}"
        imageio.imwrite(out / f"{name}.png", np.concatenate(
            [np.concatenate([frames[0], restored], 1), np.concatenate([frames[1], replayed], 1)], 0))
        row = {"policy": rs["policy"], "episode": rs["episode"], "reset_step": step, "onset_step": rs["onset_step"],
               "replay_to": end, "pix_mae_restore": round(pix_restore, 2), "pix_mae_replay": round(pix_replay, 2),
               "state_max_abs_err_after_replay": state_err}
        report.append(row)
        print(json.dumps(row))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"saved {len(report)} comparisons to {out} (rows: video | restored; top = reset step, bottom = after replay)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="verify restore + replay against the recordings")
    p.add_argument("--policy", default=None, help="e.g. rl/pick; default all policies")
    p.add_argument("--replay-steps", type=int, default=20)
    p.add_argument("--out", type=pathlib.Path, default=HERE.parent / "RL" / "restore_check")
    args = p.parse_args()
    if args.check:
        check(args.policy, args.replay_steps, args.out)
    else:
        for rs in load_reset_states(args.policy):
            print(json.dumps(rs))


if __name__ == "__main__":
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
