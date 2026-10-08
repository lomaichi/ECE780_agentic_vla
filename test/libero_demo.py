"""Minimal LIBERO demo: load a task, run random actions, and either show a live
window (--gui) or save an mp4 of the agent-view camera.

Run with the LIBERO client venv:
    source /opt/openpi/examples/libero/.venv/bin/activate
    python libero_demo.py --suite libero_spatial --task 0            # headless -> mp4
    MUJOCO_GL=glx python libero_demo.py --suite libero_spatial --gui  # live window
"""
import argparse
import os
import pathlib

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv

parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="libero_spatial",
                    choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
parser.add_argument("--task", type=int, default=0)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--gui", action="store_true", help="open an on-screen viewer (needs MUJOCO_GL=glx)")
parser.add_argument("--out", default="videos")
args = parser.parse_args()

suite = benchmark.get_benchmark_dict()[args.suite]()
task = suite.get_task(args.task)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[{args.suite} #{args.task}] {task.language}")

env_kwargs = dict(bddl_file_name=bddl, camera_heights=256, camera_widths=256, horizon=args.steps + 1)
if args.gui:
    env = ControlEnv(**env_kwargs, has_renderer=True, has_offscreen_renderer=False,
                     use_camera_obs=False, render_camera="frontview")
else:
    env = OffScreenRenderEnv(**env_kwargs)
env.seed(0)
env.reset()
# Each task ships 50 fixed initial states; the benchmark evaluates on these.
obs = env.set_init_state(suite.get_task_init_states(args.task)[0])

frames = []
for t in range(args.steps):
    action = np.random.uniform(-1, 1, 7) * [0.5] * 6 + [0]  # 6-DoF delta EEF + gripper
    action[-1] = 1.0 if t % 40 < 20 else -1.0
    obs, reward, done, info = env.step(action.tolist())
    if args.gui:
        env.env.render()
    else:
        frames.append(obs["agentview_image"][::-1, ::-1])  # LIBERO images are upside-down
    if done:
        print("task success!")
        break
env.close()

if frames:
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)
    path = out / f"{args.suite}_task{args.task}.mp4"
    imageio.mimwrite(path, frames, fps=20)
    print(f"saved {path}")
