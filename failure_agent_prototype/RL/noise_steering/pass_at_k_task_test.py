"""Task-level pass@k test: how often does plain pi0 succeed on a fixed LIBERO scene
when only its flow-matching noise changes between rollouts?

Every trial of a task starts from the same init state (fixed env seed and init
state index), so the only source of variation is the N(0, I) noise the pi0_libero
server draws on each infer() call (its JAX rng is split per call). Rollout loop
and settings match tools/record_suites.py / openpi examples/libero/main.py.

Output (OUT_DIR):
  metadata.json                                       per-task trials, summary, suggested RL mode
  task<id>/task<id>_trial<k>_<success|fail>.mp4       first 5 successes and first 5 failures

Run in the LIBERO client venv while the pi0_libero server is up:
  MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python pass_at_k_task_test.py --out-dir DIR
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import pathlib
import shutil
from collections import deque
from math import comb

import numpy as np

SUITE = "libero_90"
MAX_STEPS = 400  # default policy steps (20 s at 20 Hz); override with --timeout-s
NUM_STEPS_WAIT = 10
REPLAN_STEPS = 5
RESIZE = 224
ENV_RES = 256
DUMMY_ACTION = [0.0] * 6 + [-1.0]
FPS = 20
N_VIDEOS = 5
STEER_K, STEER_TAU = 8, 0.5  # suggest noise steering if pass@8 >= 0.5, else residual


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def run_trial(job):
    import imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    task_id, trial, a = job["task_id"], job["trial"], job["args"]
    np.random.seed(a["env_seed"])
    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(task_id)
    init_state = suite.get_task_init_states(task_id)[a["init_state"]]
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=ENV_RES, camera_widths=ENV_RES)
    env.seed(a["env_seed"])
    client = websocket_client_policy.WebsocketClientPolicy(a["host"], a["port"])
    env.reset()
    obs = env.set_init_state(init_state)

    frames, plan, done, t = [], deque(), False, 0
    max_steps = round(a["timeout_s"] * FPS) if a["timeout_s"] else MAX_STEPS
    while t < max_steps + NUM_STEPS_WAIT:
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        frames.append(img)
        if t < NUM_STEPS_WAIT:
            action = DUMMY_ACTION
        else:
            if not plan:
                state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"]))
                chunk = client.infer({
                    "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE)),
                    "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE)),
                    "observation/state": state,
                    "prompt": str(task.language),
                })["actions"]
                plan.extend(chunk[:REPLAN_STEPS])
            action = np.asarray(plan.popleft()).tolist()
        try:
            obs, _, done, _ = env.step(action)
        except Exception as e:  # main.py also treats sim errors as failure
            print(f"task{task_id} trial{trial}: env.step failed at {t}: {e}", flush=True)
            break
        t += 1
        if done:
            break
    env.close()

    tmp = pathlib.Path(a["out_dir"]) / ".tmp" / f"task{task_id}_trial{trial}.mp4"
    imageio.mimwrite(tmp, frames, fps=FPS)
    print(f"task{task_id} trial{trial}: {'success' if done else 'fail'} ({t} steps)", flush=True)
    return task_id, trial, task.name, task.language, bool(done)


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tasks", type=int, nargs="+", default=[1, 25, 27])
    p.add_argument("--n-trials", type=int, default=32)
    p.add_argument("--init-state", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=7)
    p.add_argument("--timeout-s", type=float, default=None, help="policy time limit in seconds at 20 Hz (excl. 10 wait steps)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    a = vars(p.parse_args())
    out = pathlib.Path(a["out_dir"])
    (out / ".tmp").mkdir(parents=True, exist_ok=True)

    jobs = [{"task_id": tid, "trial": k, "args": a} for tid in a["tasks"] for k in range(a["n_trials"])]
    with mp.get_context("spawn").Pool(a["workers"]) as pool:
        results = pool.map(run_trial, jobs)

    meta = {}
    for tid in a["tasks"]:
        rs = sorted((r for r in results if r[0] == tid), key=lambda r: r[1])
        trials = [{"trial": r[1], "result": "success" if r[4] else "fail"} for r in rs]
        n, c = len(rs), sum(r[4] for r in rs)
        (out / f"task{tid}").mkdir(exist_ok=True)
        kept = [r for r in rs if r[4]][:N_VIDEOS] + [r for r in rs if not r[4]][:N_VIDEOS]
        for r in kept:
            shutil.move(out / ".tmp" / f"task{tid}_trial{r[1]}.mp4",
                        out / f"task{tid}" / f"task{tid}_trial{r[1]}_{'success' if r[4] else 'fail'}.mp4")
        meta[f"task{tid}"] = {
            "task_name": rs[0][2],
            "task_language": rs[0][3],
            "trials": trials,
            "summary": {
                "n_trials": n, "n_success": c, "n_fail": n - c,
                **{f"pass@{k}": round(pass_at_k(n, c, k), 4) for k in (1, 4, 8, 16, 32) if k <= n},
            },
            "suggested_rl_mode": "noise_steering" if c > 0 and pass_at_k(n, c, STEER_K) >= STEER_TAU else "residual_rl",
        }
    shutil.rmtree(out / ".tmp")
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    for k, v in meta.items():
        print(k, v["summary"], v["suggested_rl_mode"])


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
