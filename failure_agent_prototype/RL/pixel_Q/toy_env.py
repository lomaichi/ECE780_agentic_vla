"""Synthetic stand-in for PickCorrectorEnv: same observation / action / step interface, no LIBERO, no pi0.

Use it to check the DQN code (train_dqn.py --env toy) before spending simulator time. The "object" is
a disc of random radius and position in the 80x80 crop, seen from above with a raised rim. A correction
succeeds with high probability only if
  - the chosen cell is on the rim (within RIM_PX of the disc edge),
  - on the side facing the gripper (the EE direction is given in proprio[0:2]), and
  - the rotation bin cancels the gripper's yaw error (the yaw error is given in proprio[4:6]).
So the optimal action depends on the image (where the rim is) and on proprio (which side, which bin),
like the real corrector. A uniformly random valid action succeeds about 5-10% of the time.
"""

import numpy as np

from qnet import CELL, CROP, GRID, PROPRIO_DIM, ROT_DEG, decode_action

RIM_PX = 6.0
P_GOOD, P_SIDE, P_BASE, P_LOST = 0.9, 0.25, 0.03, 0.05


class ToyCorrectorEnv(object):
    def __init__(self, cfg):
        self.rng = np.random.default_rng(cfg.get("seed", 0) + 1000 * cfg.get("worker_id", 0))
        self.frames = None

    def reset(self, start, record=False):
        r = self.rng
        self.center = CROP / 2 + r.uniform(-8, 8, 2)                 # (x, y) pixels
        self.radius = r.uniform(14, 24)
        ang = r.uniform(0, 2 * np.pi)
        self.ee_dir = np.array([np.cos(ang), np.sin(ang)])           # gripper side, image xy
        self.yaw_err = float(r.choice(ROT_DEG))                       # needs bin -yaw_err
        self.shade = r.uniform(40, 110)
        return self._observe()

    def _observe(self):
        yy, xx = np.mgrid[0:CROP, 0:CROP] + 0.5
        dist = np.hypot(xx - self.center[0], yy - self.center[1])
        mask = dist <= self.radius
        rim = mask & (dist >= self.radius - 3)
        rgb = np.full((CROP, CROP, 3), 128, np.uint8)
        rgb[mask] = int(self.shade)
        rgb[rim] = int(self.shade) + 60
        depth = np.where(rim, -1.0, np.where(mask, -0.4, 0.0)) + self.rng.normal(0, 0.03, (CROP, CROP))
        valid = mask.reshape(GRID, CELL, GRID, CELL).mean(axis=(1, 3)) >= 0.25
        yaw = np.radians(self.yaw_err)          # sin / cos, like the LIBERO env's EE yaw feature
        proprio = np.r_[1.5 * self.ee_dir, 1.0, 1.0, np.sin(yaw), np.cos(yaw)].astype(np.float32)
        assert proprio.shape == (PROPRIO_DIM,)
        return {"rgb": rgb, "depth": depth.astype(np.float16), "mask": mask, "proprio": proprio, "valid": valid}

    def step(self, action):
        """One episode (same contract as PickCorrectorEnv.step): returns (next_obs or None, reward, info)."""
        k, row, col = decode_action(action)
        cell = np.array([col * CELL + CELL / 2, row * CELL + CELL / 2])
        rel = cell - self.center
        d = np.linalg.norm(rel)
        on_rim = abs(d - self.radius) <= RIM_PX
        facing = d > 1e-6 and float(rel @ self.ee_dir) / d > 0.5
        rot_ok = abs(ROT_DEG[k] + self.yaw_err) < 1e-6
        p = P_GOOD if (on_rim and facing and rot_ok) else P_SIDE if (on_rim and facing) else P_BASE
        return self._finish(p, {"k": k, "row": row, "col": col, "on_rim": bool(on_rim), "facing": bool(facing),
                                "rot_ok": bool(rot_ok)})

    def baseline_step(self):
        return self._finish(P_BASE, {})

    def _finish(self, p, info):
        success = bool(self.rng.random() < p)
        info.update(success=success, obj_tilt_deg=0.0)
        if success:
            info["chain_end"] = "success"
            return None, 1.0, info
        if self.rng.random() < P_LOST:
            info["chain_end"] = "dropped"
            return None, 0.0, info
        info["chain_end"] = None                     # failed attempt nudged the object; next episode goes on
        self.center = np.clip(self.center + self.rng.normal(0, 2, 2), 22, CROP - 22)
        return self._observe(), 0.0, info

    def inspect_info(self):
        raise NotImplementedError("inspect is for the LIBERO env")

    def close(self):
        pass


def toy_starts(n=8):
    return [{"kind": "toy", "tag": "toy{}".format(i)} for i in range(n)]
