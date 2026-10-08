"""Offline stall statistics from rollout videos (no sim state needed).

Reproduces the motion table in Documentation/failure_mode_sumary.txt:
mean absolute grey-level difference between frames 1 s apart (10 frames at
10 fps); a 1 s window is "still" when that difference is below --still.

  python3 tools/video_motion_stats.py /home/DockerShared/ECE780/Documentation/native_pi0_libero90
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np


def motion_profile(path, lag=10):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32))
    d = np.array([np.abs(frames[i] - frames[i - lag]).mean() for i in range(lag, len(frames))])
    return len(frames), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--still", type=float, default=1.0, help="grey-level threshold for 'still'")
    args = ap.parse_args()

    def key(p):
        m = re.search(r"task(\d+)", p)
        return int(m.group(1)) if m else 0

    print("%-40s %-3s %7s %7s %13s %14s" % ("video", "res", "len_s", "still%", "longest_still", "motion_last10s"))
    for f in sorted(glob.glob(os.path.join(args.folder, "**", "*.mp4"), recursive=True), key=key):
        n, d = motion_profile(f, lag=int(args.fps))
        still = d < args.still
        run = best = 0
        for s in still:
            run = run + 1 if s else 0
            best = max(best, run)
        res = "S" if "success" in f else "F"
        name = os.path.relpath(f, args.folder)[:40]
        print("%-40s %-3s %7.1f %6.0f%% %12.1fs %14.2f" % (
            name, res, n / args.fps, 100 * still.mean(), best / args.fps, d[-int(10 * args.fps):].mean()))


if __name__ == "__main__":
    main()
