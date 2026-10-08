"""Collect native pi0 rollouts on libero_90, same loop as openpi's
examples/libero/main.py (seed 7, 10 settle steps, replan every 5 steps,
400-step budget), with the task id in the video name so tasks that share an
instruction (e.g. 75 and 80) don't overwrite each other.

Run inside the LIBERO client venv with the policy server up. Shard the 90
tasks over several processes to overlap simulation with inference:

  for i in 0 1 2 3; do MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python \
      tools/collect_native.py --shard $i/4 & done

Writes rollout_task<id>_<instruction>_<success|failure>.mp4 and
results_shard<i>.json into --out.
"""
import argparse
import collections
import json
import math
import pathlib
import time

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy

DUMMY = [0.0] * 6 + [-1.0]


def quat2axisangle(quat):
    quat = np.array(quat, dtype=float)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/DockerShared/ECE780/Documentation/native_pi0_libero90")
    ap.add_argument("--tasks", nargs="*", type=int, default=None)
    ap.add_argument("--shard", default="0/1", help="i/N: take tasks with id %% N == i")
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    i, n = (int(x) for x in args.shard.split("/"))
    tasks = args.tasks if args.tasks else list(range(suite.n_tasks))
    tasks = [t for t in tasks if t % n == i]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = websocket_client_policy.WebsocketClientPolicy("0.0.0.0", args.port)

    results = []
    for tid in tasks:
        task = suite.get_task(tid)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(tid)[args.init])
        t0, t, done, frames, plan = time.time(), 0, False, [], collections.deque()
        while t < args.max_steps + 10:
            if t < 10:
                obs, _, done, _ = env.step(DUMMY)
                t += 1
                continue
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
            wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
            frames.append(img)
            if not plan:
                chunk = client.infer({
                    "observation/image": img, "observation/wrist_image": wrist,
                    "observation/state": np.concatenate((obs["robot0_eef_pos"],
                                                         quat2axisangle(obs["robot0_eef_quat"]),
                                                         obs["robot0_gripper_qpos"])),
                    "prompt": str(task.language)})["actions"]
                plan.extend(chunk[:5])
            obs, _, done, _ = env.step(plan.popleft().tolist())
            if done:
                break
            t += 1
        env.close()
        suffix = "success" if done else "failure"
        name = f"rollout_task{tid}_{task.language.replace(' ', '_')}_{suffix}.mp4"
        imageio.mimwrite(out / name, frames, fps=10)
        results.append({"task": tid, "instruction": task.language, "scene": task.name,
                        "success": bool(done), "steps": t - 10, "video": name,
                        "wall_s": round(time.time() - t0, 1)})
        print(f"task {tid:2d} {suffix:7s} steps={t - 10:3d} {time.time() - t0:5.1f}s  {task.language}",
              flush=True)
        (out / f"results_shard{i}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
