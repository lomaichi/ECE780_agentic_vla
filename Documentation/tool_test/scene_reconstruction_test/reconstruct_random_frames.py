"""Reconstruct 5 random frames of the rollout video from the recorded LIBERO trace.

Video frame i is the agentview obs image the policy saw before actions[i], i.e.
the scene in trace states[i] (meta.json: one frame per control step, 20 fps,
wait steps included). For each picked frame: fresh env from the same task +
env seed, env.reset(), env.set_init_state(states[i]), take the agentview obs,
and save original | reconstructed side by side in reconstruction/.

Run from this folder with the LIBERO client venv (MUJOCO_GL=egl):
  /opt/openpi/examples/libero/.venv/bin/python reconstruct_random_frames.py [--seed N]
"""

import argparse
import json
import pathlib

import imageio.v2 as imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent


def label(img, text):
    """Black header strip with text above img."""
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + 18), "black")
    canvas.paste(Image.fromarray(img), (0, 18))
    ImageDraw.Draw(canvas).text((4, 3), text, fill="white")
    return np.asarray(canvas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None, help="frame selection seed; random if omitted")
    p.add_argument("--n-frames", type=int, default=5)
    args = p.parse_args()

    meta = json.loads((HERE / "meta.json").read_text())
    trace = np.load(HERE / "trace.npz")
    states = trace["states"]
    video = imageio.mimread(HERE / meta["video"], memtest=False)
    assert len(video) == len(states), (len(video), len(states))

    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 2**32)
    rng = np.random.default_rng(seed)
    steps = np.sort(rng.choice(len(states), size=args.n_frames, replace=False))
    print(f"selection seed {seed}, frames {steps.tolist()}")

    task = benchmark.get_benchmark_dict()[meta["task_suite"]]().get_task(meta["task_id"])
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    res = meta["camera_resolution"]
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=res, camera_widths=res)
    env.seed(meta["env_seed"])

    out = HERE / "reconstruction"
    out.mkdir(exist_ok=True)
    for s in steps:
        env.reset()
        obs = env.set_init_state(states[s])
        recon = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # LIBERO renders upside down
        orig = video[s][..., :3]
        t = s / meta["video_fps"]
        img = np.concatenate([label(orig, f"video frame {s} (t={t:.2f} s)"),
                              label(recon, f"reconstructed from states[{s}]")], axis=1)
        imageio.imwrite(out / f"frame{s:04d}_compare.png", img)
    env.close()
    (out / "selection.json").write_text(json.dumps({"selection_seed": seed, "frames": steps.tolist()}, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
