"""Sample frames from the failed pi0 rollout videos for the full failure survey.

Scope: every failure in libero_spatial and libero_10, plus the failures of a random 50% of the
libero_90 tasks (split by task so the other 45 tasks stay unseen for zero-shot testing).
Video only (no trace): 20 fps video, one frame per control step, so step = round(20 * t).
A random 20% of the surveyed episodes are sampled at 3 fps, the rest at 1 fps. Frames are tiled
into labelled contact sheets (step number on each tile) for the VLM surveyor to read.

usage: python extract_frames.py [seed]
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent  # failure_agent_prototype/tools/frame_manipulation
ROLLOUTS = HERE.parents[2] / "Documentation" / "pi0_LIBERO_rollouts"
SURVEY = HERE.parents[1] / "failure_survey"
SAMPLING = SURVEY / "documentation" / "sampling"  # sampling_plan.json, libero90_split.json
SUITES = ["libero_spatial", "libero_10", "libero_90"]
VIDEO_FPS = 20
COLS, ROWS = 5, 4  # tiles per contact sheet
DENSE_FRACTION = 0.2

seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.SystemRandom().randrange(2**31)
rng = random.Random(seed)

# libero_90 train/test split by task id
train_tasks = sorted(rng.sample(range(90), 45))
split = {"suite": "libero_90", "seed": seed, "method": "random.Random(seed).sample(range(90), 45)",
         "survey_tasks": train_tasks, "held_out_tasks": sorted(set(range(90)) - set(train_tasks))}

episodes = []
for suite in SUITES:
    meta = json.load(open(ROLLOUTS / suite / "meta.json"))
    for ep in meta["episodes"]:
        if ep["success"]:
            continue
        if suite == "libero_90" and ep["task_id"] not in train_tasks:
            continue
        episodes.append((suite, ep))

dense = set(rng.sample(range(len(episodes)), round(DENSE_FRACTION * len(episodes))))

plan = []
for i, (suite, ep) in enumerate(episodes):
    fps = 3 if i in dense else 1
    cap = cv2.VideoCapture(str(ROLLOUTS / suite / ep["folder"] / ep["video"]))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    n = len(frames)
    steps = sorted({s for s in (round(k * VIDEO_FPS / fps) for k in range(n)) if s < n})  # 3 fps: every 6-7 steps
    if steps[-1] != n - 1:
        steps.append(n - 1)  # always include the final frame

    ep_dir = SURVEY / "episodes" / suite / ep["folder"]
    out = ep_dir / "frames"
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for s in steps:
        f = frames[s].copy()
        cv2.imwrite(str(out / f"step{s:04d}.png"), f)
        label = f"step {s}  t={s / VIDEO_FPS:.2f}s"
        cv2.rectangle(f, (0, 0), (f.shape[1], 18), (0, 0, 0), -1)
        cv2.putText(f, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(f)
    h, w = tiles[0].shape[:2]
    per = COLS * ROWS
    for k in range(0, len(tiles), per):
        chunk = tiles[k:k + per]
        sheet = np.zeros((h * ROWS, w * COLS, 3), np.uint8)
        for j, t in enumerate(chunk):
            r, c = divmod(j, COLS)
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
        used_rows = (len(chunk) - 1) // COLS + 1
        cv2.imwrite(str(ep_dir / f"sheet{k // per:02d}.png"), sheet[:used_rows * h])
    plan.append({"suite": suite, "folder": ep["folder"], "task_id": ep["task_id"],
                 "init_state_idx": ep["init_state_idx"], "task_language": ep["task_language"],
                 "survey_fps": fps, "video_frames": n, "num_sheets": (len(tiles) - 1) // per + 1,
                 "sampled_steps": steps})

SAMPLING.mkdir(parents=True, exist_ok=True)
counts = {s: sum(p["suite"] == s for p in plan) for s in SUITES}
json.dump({"random_seed": seed, "dense_fps": 3, "sparse_fps": 1, "counts": counts,
           "three_fps_episodes": [f'{p["suite"]}/{p["folder"]}' for p in plan if p["survey_fps"] == 3],
           "episodes": plan}, open(SAMPLING / "sampling_plan.json", "w"), indent=2)
json.dump(split, open(SAMPLING / "libero90_split.json", "w"), indent=2)
print(json.dumps({"seed": seed, "counts": counts, "n_3fps": len(dense)}))
