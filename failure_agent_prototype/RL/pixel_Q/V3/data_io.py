"""V3 sample shards (data/shard_*.npz), shared by train.py (writes, resumes) and train_v.py (offline refits).

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


def rows_of(d):
    """Column dict -> list of row dicts (for re-ingesting on --resume)."""
    n = len(d["t_success"]) if d else 0
    return [{k: d[k][i] for k in d} for i in range(n)]
