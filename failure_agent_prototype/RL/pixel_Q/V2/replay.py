"""Replay buffer for V2: (state, action, label) samples, all terminal (pixel_Q_RL_V2.md sec. 7).

Every branch point adds two samples that share one observation dict: the no-op label (action A0) and the
intervention label (the executed correction). Pickle keeps that sharing, so storage is about one map per branch
point: rgb (88, 88, 3) uint8, height (88, 88) float16, mask / observed bool, proprio (4,) float32, valid
(11, 11) bool -> ~55 kB, i.e. ~0.6 GB for 20 000 samples. FIFO over a fixed capacity, never cleared.
"""

import pickle

import numpy as np

from qnet import A0

FIELDS = ("rgb", "height", "mask", "observed", "proprio", "valid")


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = []
        self.pos = 0
        self.n_added = 0

    def __len__(self):
        return len(self.data)

    def add(self, obs, action, label, info=None):
        s = {"obs": obs, "a": int(action), "y": float(label), "info": info or {}}
        if len(self.data) < self.capacity:
            self.data.append(s)
        else:
            self.data[self.pos] = s
        self.pos = (self.pos + 1) % self.capacity
        self.n_added += 1

    def extend(self, other):
        for s in other.data:
            self.add(s["obs"], s["a"], s["y"], s["info"])

    def sample(self, batch_size, rng):
        idx = rng.integers(len(self.data), size=batch_size)
        ss = [self.data[i] for i in idx]
        out = {"a": np.array([s["a"] for s in ss], dtype=np.int64),
               "y": np.array([s["y"] for s in ss], dtype=np.float32)}
        for f in FIELDS:
            out[f] = np.stack([s["obs"][f] for s in ss])
        out["is_noop"] = out["a"] == A0
        return out

    def stats(self):
        if not self.data:
            return {"size": 0}
        a = np.array([s["a"] for s in self.data])
        y = np.array([s["y"] for s in self.data])
        noop = a == A0
        mean = lambda v: float(v.mean()) if v.size else float("nan")  # noqa: E731
        return {"size": len(self.data), "added": self.n_added, "noop_frac": float(noop.mean()),
                "y_noop": mean(y[noop]), "y_corr": mean(y[~noop]),
                "success_noop": mean(y[noop] > 0), "success_corr": mean(y[~noop] > 0)}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"capacity": self.capacity, "data": self.data, "pos": self.pos, "n_added": self.n_added}, f,
                        protocol=4)

    @classmethod
    def load(cls, path, capacity=None):
        with open(path, "rb") as f:
            d = pickle.load(f)
        buf = cls(capacity or d["capacity"])
        if capacity is None or capacity == d["capacity"]:
            buf.data, buf.pos, buf.n_added = d["data"], d["pos"], d["n_added"]
        else:                      # re-add in insertion order under the new capacity
            old = d["data"][d["pos"]:] + d["data"][:d["pos"]] if len(d["data"]) == d["capacity"] else d["data"]
            for s in old:
                buf.add(s["obs"], s["a"], s["y"], s["info"])
        return buf
