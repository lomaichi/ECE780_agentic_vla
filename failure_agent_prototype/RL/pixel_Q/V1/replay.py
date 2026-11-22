"""Replay buffer for the pixel-Q DQN (pixel_Q_RL.md step 4).

A transition is (s, a, r, s', done). Observations are the compact dicts made by the envs
(rgb uint8 crop, depth float16, mask bool, proprio float32, valid (10, 10) bool), so 20k transitions
take roughly 1.5 GB. For done = 1 the next observation is not needed and is stored as None.
"""

import pickle

import numpy as np

from qnet import CROP, GRID, PROPRIO_DIM

_FIELDS = ("rgb", "depth", "mask", "proprio", "valid")


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = []
        self.pos = 0
        self.n_added = 0

    def __len__(self):
        return len(self.data)

    def add(self, obs, action, reward, next_obs, done, info=None):
        t = {"obs": obs, "a": int(action), "r": float(reward), "next": None if done else next_obs,
             "done": bool(done), "info": info or {}}
        if len(self.data) < self.capacity:
            self.data.append(t)
        else:
            self.data[self.pos] = t
        self.pos = (self.pos + 1) % self.capacity
        self.n_added += 1

    def sample(self, batch_size, rng):
        idx = rng.integers(len(self.data), size=batch_size)
        ts = [self.data[i] for i in idx]
        blank = _blank_obs()
        out = {"a": np.array([t["a"] for t in ts], dtype=np.int64),
               "r": np.array([t["r"] for t in ts], dtype=np.float32),
               "done": np.array([t["done"] for t in ts], dtype=np.float32)}
        for f in _FIELDS:
            out[f] = np.stack([t["obs"][f] for t in ts])
            out["next_" + f] = np.stack([(t["next"] or blank)[f] for t in ts])
        return out

    def stats(self):
        if not self.data:
            return {"size": 0}
        r = np.array([t["r"] for t in self.data])
        d = np.array([t["done"] for t in self.data])
        return {"size": len(self.data), "added": self.n_added, "reward_rate": float(r.mean()),
                "done_rate": float(d.mean())}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"capacity": self.capacity, "data": self.data, "pos": self.pos, "n_added": self.n_added}, f,
                        protocol=4)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        buf = cls(d["capacity"])
        buf.data, buf.pos, buf.n_added = d["data"], d["pos"], d["n_added"]
        return buf


def _blank_obs():
    return {"rgb": np.zeros((CROP, CROP, 3), np.uint8), "depth": np.zeros((CROP, CROP), np.float16),
            "mask": np.zeros((CROP, CROP), bool), "proprio": np.zeros(PROPRIO_DIM, np.float32),
            "valid": np.zeros((GRID, GRID), bool)}
