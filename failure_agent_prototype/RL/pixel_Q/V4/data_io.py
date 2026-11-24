"""V4 sample shards (data/shard_*.npz), shared by collect.py (writes, resumes) and train_v.py (trains V).
Same format as V3's shards.

One row per kept branch episode: the observation at the takeover pose (vnet.OBS_FIELDS) + raw outcome
(t_success: control steps from takeover to pick, -1 = none within cap_steps; end) + meta (tag, split
train / val / eval, rollout, init, tick, episode, natural, perturbation offset, move steps, tilt, target distance).
Labels are made from the raw outcome at training time: label(t_success, h_steps, gamma).
"""

import pathlib
import zlib

import numpy as np

import vnet

CTRL_HZ = 20
OBS = vnet.OBS_FIELDS
META_STR = ("end", "tag", "split")
META_NUM = ("t_success", "cap_steps", "natural", "tick_step", "tick", "episode", "move_steps", "tilt_deg",
            "target_dxy", "rollout", "init")
META_VEC = ("d_cm", "rot_deg")


def label(t_success, h_steps, gamma):
    """gamma^(seconds to pick) if picked within h_steps, else 0."""
    return float(gamma ** (t_success / float(CTRL_HZ))) if 0 <= t_success <= h_steps else 0.0


def split_of(tag, val_frac):
    """Fixed tag-hash split of training (task, init) tags into train / val."""
    return "val" if (zlib.crc32(tag.encode()) % 1000) < val_frac * 1000 else "train"


def write_shard(path, rows):
    out = {f: np.stack([r[f] for r in rows]) for f in OBS}
    for k in META_STR:
        out[k] = np.array([r[k] for r in rows])
    for k in META_NUM:
        out[k] = np.array([r[k] for r in rows], np.float32)
    for k in META_VEC:
        out[k] = np.array([r[k] for r in rows], np.float32)
    np.savez_compressed(path, **out)


def read_shards(data_dir):
    """All shards of a run as one dict of arrays (columns), or {} if there are none."""
    parts = [dict(np.load(p)) for p in sorted(pathlib.Path(data_dir).glob("shard_*.npz"))]
    if not parts:
        return {}
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    d["t_success"] = d["t_success"].astype(np.int32)
    d["cap_steps"] = d["cap_steps"].astype(np.int32)
    d["natural"] = d["natural"].astype(bool)
    d["task"] = np.array([t.rsplit("_init", 1)[0] for t in d["tag"].astype(str)])
    return d


def shard_paths(data_dir):
    return sorted(pathlib.Path(data_dir).glob("shard_*.npz"))


def read_meta(data_dir, keys=("split", "rollout", "natural", "t_success")):
    """Only the given meta columns of all shards (no observations), or {} if there are none. Used by collect.py
    --resume to restore the sample counts without loading the heightmaps."""
    paths = shard_paths(data_dir)
    if not paths:
        return {}
    parts = []
    for p in paths:
        with np.load(p) as z:
            parts.append({k: z[k] for k in keys})
    return {k: np.concatenate([q[k] for q in parts]) for k in keys}
