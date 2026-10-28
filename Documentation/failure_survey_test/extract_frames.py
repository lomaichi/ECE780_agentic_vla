"""Sample frames from the first 5 failed libero_90 rollout videos for the failure-survey PoC.

Video only (no trace): 20 fps video, one frame per control step, so step = round(20 * t).
One randomly chosen failure is sampled at 5 fps, the rest at 1 fps. Frames are tiled into
labelled contact sheets (step number on each tile) for the VLM surveyor to read.
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ROLLOUTS = ROOT.parent / "pi0_LIBERO_rollouts" / "libero_90"
VIDEO_FPS = 20
COLS, ROWS = 5, 4  # tiles per contact sheet

meta = json.load(open(ROLLOUTS / "meta.json"))
failures = [e for e in meta["episodes"] if not e["success"]][:5]

seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.SystemRandom().randrange(2**31)
pick = random.Random(seed).randrange(len(failures))

plan = []
for i, ep in enumerate(failures):
    fps = 5 if i == pick else 1
    stride = VIDEO_FPS // fps
    cap = cv2.VideoCapture(str(ROLLOUTS / ep["folder"] / ep["video"]))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    n = len(frames)
    steps = list(range(0, n, stride))
    if steps[-1] != n - 1:
        steps.append(n - 1)  # always include the final frame

    out = ROOT / ep["folder"] / "frames"
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
        cv2.imwrite(str(ROOT / ep["folder"] / f"sheet{k // per:02d}.png"), sheet[:used_rows * h])
    plan.append({"folder": ep["folder"], "task_language": ep["task_language"],
                 "survey_fps": fps, "video_frames": n, "sampled_steps": steps})

json.dump({"random_seed": seed, "five_fps_episode": failures[pick]["folder"], "episodes": plan},
          open(ROOT / "sampling_plan.json", "w"), indent=2)
print(json.dumps({"seed": seed, "5fps": failures[pick]["folder"]}))
for p in plan:
    print(p["folder"], p["survey_fps"], p["video_frames"], len(p["sampled_steps"]))
