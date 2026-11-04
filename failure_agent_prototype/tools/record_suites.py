"""Record pi0 rollouts on whole LIBERO suites (no reconstruction, just recording).

Same rollout loop as openpi examples/libero/main.py and the same trace fields
as the Documentation/scene_reconstruction_test recording, minus the
random-step screenshots and the per-step exact renders. Layout:

  OUT_ROOT/<suite>/meta.json                          suite settings + one entry per episode
  OUT_ROOT/<suite>/task<id>_<success|fail>_init<k>/
      task<id>_<success|fail>_init<k>.mp4             agentview as pi0 saw it, 1 frame/step, 20 fps
      trace/<field>.npy                               states, actions, action_chunks, ...

Episodes already listed in a suite's meta.json are skipped, so the script can
be rerun after an interruption. Workers share one pi0_libero server.

Run with the LIBERO client venv while the pi0_libero server is up:
  MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python record_suites.py --out-root DIR
"""

import argparse
import collections
import json
import logging
import math
import multiprocessing as mp
import os
import pathlib
import shutil
import time

import numpy as np

POLICY = "pi0_libero (gs://openpi-assets/checkpoints/pi0_libero), stochastic"
LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
CONTROL_FREQ = 20  # LIBERO / robosuite default control frequency (Hz)
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def make_env(task, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)  # seed affects object placement even with a fixed init state
    return env, bddl_file


def upright(img):
    """LIBERO renders upside down; rotate 180 deg (what pi0 and the videos see)."""
    return np.ascontiguousarray(img[::-1, ::-1])


def capture_hidden(env):
    """Per-step state that env.get_sim_state() does not contain."""
    return {
        # zeros(1) right after reset, (2,) once format_action ran; same effect
        "gripper_current_action": np.broadcast_to(env.env.robots[0].gripper.current_action, (2,)).copy(),
        "qacc_warmstart": env.env.sim.data.qacc_warmstart.copy(),
    }


def quat2axisangle(quat):
    """Copied from openpi examples/libero/main.py (robosuite transform_utils)."""
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def proprio(obs):
    return {
        "ee_pos": obs["robot0_eef_pos"].copy(),
        "ee_quat": obs["robot0_eef_quat"].copy(),
        "gripper_states": obs["robot0_gripper_qpos"].copy(),
        "joint_states": obs["robot0_joint_pos"].copy(),
    }


def run_episode(job):
    """One rollout; writes the episode folder and returns (suite, meta entry)."""
    import imageio
    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as _websocket_client_policy

    a = job["args"]
    suite_name, task_id, init_idx = job["suite"], job["task_id"], job["init_idx"]
    np.random.seed(a["env_seed"])  # as in main.py
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    init_state = suite.get_task_init_states(task_id)[init_idx]
    max_steps = MAX_STEPS[suite_name]

    env, bddl_file = make_env(task, a["env_seed"])
    client = _websocket_client_policy.WebsocketClientPolicy(a["host"], a["port"])
    env.reset()
    obs = env.set_init_state(init_state)

    states, actions, is_wait, rewards, dones = [], [], [], [], []
    prop = collections.defaultdict(list)
    hidden = collections.defaultdict(list)
    pi0_states, chunk_steps, chunks, agent_imgs = [], [], [], []
    action_plan = collections.deque()
    done = False
    error = None
    t = 0
    t0 = time.time()
    while t < max_steps + a["num_steps_wait"]:
        # Everything appended in this iteration describes the state BEFORE actions[t]
        state_t = env.get_sim_state().copy()
        img = upright(obs["agentview_image"])
        wrist = upright(obs["robot0_eye_in_hand_image"])
        pi0_state = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        )
        wait = t < a["num_steps_wait"]
        if wait:
            action = np.asarray(LIBERO_DUMMY_ACTION, dtype=np.float64)
        else:
            if not action_plan:
                element = {
                    "observation/image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, a["resize_size"], a["resize_size"])),
                    "observation/wrist_image": image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist, a["resize_size"], a["resize_size"])),
                    "observation/state": pi0_state,
                    "prompt": str(task.language),
                }
                chunk = np.asarray(client.infer(element)["actions"])
                assert len(chunk) >= a["replan_steps"]
                chunk_steps.append(t)
                chunks.append(chunk)
                action_plan.extend(chunk[: a["replan_steps"]])
            action = np.asarray(action_plan.popleft(), dtype=np.float64)

        states.append(state_t)
        for k, v in capture_hidden(env).items():
            hidden[k].append(v)
        agent_imgs.append(img)
        pi0_states.append(pi0_state)
        for k, v in proprio(obs).items():
            prop[k].append(v)
        actions.append(action)
        is_wait.append(wait)

        try:
            obs, reward, done, info = env.step(action.tolist())
        except Exception as e:  # main.py also aborts the episode on sim errors
            logging.error(f"{suite_name} task {task_id} init {init_idx}: env.step failed at step {t}: {e}")
            error = repr(e)
            for buf in (states, agent_imgs, pi0_states, actions, is_wait, *prop.values(), *hidden.values()):
                buf.pop()
            break
        rewards.append(reward)
        dones.append(done)
        t += 1
        if done:
            break
    final_state = env.get_sim_state().copy()
    wall = time.time() - t0
    model_file = env.sim.model.get_xml()
    env.close()

    name = f"task{task_id}_{'success' if done else 'fail'}_init{init_idx}"
    suite_dir = pathlib.Path(a["out_root"]) / suite_name
    tmp = suite_dir / f".tmp_{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "trace").mkdir(parents=True)
    imageio.mimwrite(tmp / f"{name}.mp4", agent_imgs, fps=CONTROL_FREQ)
    trace = {
        "states": np.asarray(states),
        "final_state": final_state,
        "actions": np.asarray(actions),
        "is_wait": np.asarray(is_wait),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "dones": np.asarray(dones),
        "pi0_state": np.asarray(pi0_states),
        "chunk_steps": np.asarray(chunk_steps, dtype=np.int64),
        "action_chunks": np.asarray(chunks),
        "init_state": np.asarray(init_state),
        "model_file": np.asarray(model_file),
        **{f"obs_{k}": np.asarray(v) for k, v in prop.items()},
        **{k: np.asarray(v) for k, v in hidden.items()},
    }
    for k, v in trace.items():
        np.save(tmp / "trace" / f"{k}.npy", v)
    final = suite_dir / name
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    logging.info(f"{suite_name}/{name}: {len(actions)} steps, {wall:.0f} s")

    return suite_name, {
        "folder": name,
        "task_id": task_id,
        "init_state_idx": init_idx,
        "task_language": task.language,
        "task_name": task.name,
        "bddl_file": str(bddl_file),
        "success": bool(done),
        "num_steps": len(actions),
        "error": error,
        "wall_time_s": round(wall, 1),
        "video": f"{name}.mp4",
    }


def suite_meta(suite_name, a, episodes):
    episodes = sorted(episodes, key=lambda e: (e["task_id"], e["init_state_idx"]))
    return {
        "task_suite": suite_name,
        "policy": POLICY,
        "env_seed": a["env_seed"],
        "replan_steps": a["replan_steps"],
        "num_steps_wait": a["num_steps_wait"],
        "max_steps": MAX_STEPS[suite_name],
        "camera_resolution": LIBERO_ENV_RESOLUTION,
        "control_freq_hz": CONTROL_FREQ,
        "video_fps": CONTROL_FREQ,
        "video_note": "one frame per control step incl. wait steps: video time t s <-> step round(20 t)",
        "state_convention": "trace/states.npy[t] = sim state before actions[t]; final_state follows actions[-1]. "
                            "To continue from step t, after env.set_init_state(states[t]) also call "
                            "robot.controller.update(force=True) and restore gripper.current_action and "
                            "sim.data.qacc_warmstart from the trace",
        "num_episodes": len(episodes),
        "num_success": sum(e["success"] for e in episodes),
        "episodes": episodes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", required=True)
    p.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_10", "libero_90"])
    p.add_argument("--init-states", type=int, nargs="+", default=[0, 1])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--env-seed", type=int, default=7, help="env.seed(), same default as main.py")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--num-steps-wait", type=int, default=10)
    args = p.parse_args()
    a = vars(args)

    from libero.libero import benchmark

    done_eps = {}
    jobs = []
    for suite_name in args.suites:
        suite_dir = pathlib.Path(args.out_root) / suite_name
        suite_dir.mkdir(parents=True, exist_ok=True)
        meta_path = suite_dir / "meta.json"
        eps = json.loads(meta_path.read_text())["episodes"] if meta_path.exists() else []
        eps = [e for e in eps if (suite_dir / e["folder"]).is_dir()]
        done_eps[suite_name] = eps
        have = {(e["task_id"], e["init_state_idx"]) for e in eps}
        for task_id in range(benchmark.get_benchmark_dict()[suite_name]().n_tasks):
            for k in args.init_states:
                if (task_id, k) not in have:
                    jobs.append({"suite": suite_name, "task_id": task_id, "init_idx": k, "args": a})
    logging.info(f"{len(jobs)} episodes to run")

    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (suite_name, entry) in enumerate(pool.imap_unordered(run_episode, jobs), 1):
            done_eps[suite_name].append(entry)
            meta = suite_meta(suite_name, a, done_eps[suite_name])
            (pathlib.Path(args.out_root) / suite_name / "meta.json").write_text(json.dumps(meta, indent=2))
            logging.info(f"[{i}/{len(jobs)}] {suite_name}/{entry['folder']} ({time.time() - t0:.0f} s elapsed)")

    for suite_name in args.suites:
        eps = done_eps[suite_name]
        logging.info(f"{suite_name}: {sum(e['success'] for e in eps)}/{len(eps)} success")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
