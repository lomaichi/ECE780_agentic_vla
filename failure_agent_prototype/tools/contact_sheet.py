"""Timestamped contact sheets of rollout videos, for the video failure survey.

  python3 tools/contact_sheet.py <folder of mp4s> <out folder> [--frames 24 --cols 6]

Each sheet is a grid of evenly spaced frames, labelled with the time in
seconds (10 fps videos: 1 s = 10 env steps after the settle steps).
"""
import argparse
import glob
import os

import cv2
import numpy as np


def sheet(path, n=24, cols=6, scale=1.0):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    tiles = []
    for i in idx:
        im = frames[i].copy()
        if scale != 1.0:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(im, (0, 0), (52, 14), (0, 0, 0), -1)
        cv2.putText(im, f"{i / 10:.1f}s", (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(args.folder, "*.mp4"))):
        s = sheet(f, args.frames, args.cols, args.scale)
        if s is not None:
            cv2.imwrite(os.path.join(args.out, os.path.basename(f)[:-4] + ".jpg"), s,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
