"""Scripted recovery primitives on LIBERO's OSC_POSE delta controller.

Action = [dx, dy, dz, drx, dry, drz, gripper] in [-1, 1]; the controller maps
1.0 to 0.05 m / 0.5 rad of target offset (robosuite osc_pose.json).
Gripper: -1 open, +1 close.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from robosuite.utils import transform_utils as T

POS_SCALE = 0.05
ROT_SCALE = 0.5


class Primitives:
    def __init__(self, step_fn: Callable, get_obs: Callable, home_pos, home_quat, cfg: dict,
                 joint_home: Callable = None):
        """step_fn(action) -> (obs, done); get_obs() -> latest obs; joint_home() ->
        done, an optional joint-space move back to the start configuration."""
        self.joint_home = joint_home
        self.step = step_fn
        self.get_obs = get_obs
        self.home_pos = np.array(home_pos, dtype=float)
        self.home_quat = np.array(home_quat, dtype=float)
        self.cfg = cfg
        self.gripper_cmd = -1.0
        self.incomplete = []    # residual errors of servo moves that got stuck

    def _servo(self, target_pos, target_quat, tol, max_steps, gripper) -> bool:
        """Drive the EEF to a pose. Returns True if the task succeeded meanwhile.
        Gives up early if the error stops shrinking (e.g. the arm is stretched to
        a singularity), recording it in self.incomplete."""
        best, since_best = np.inf, 0
        for _ in range(max_steps):
            obs = self.get_obs()
            pos, quat = obs["robot0_eef_pos"], obs["robot0_eef_quat"]
            dp = np.asarray(target_pos) - pos
            dr = T.quat2axisangle(T.quat_multiply(target_quat, T.quat_inverse(quat)))
            err = np.linalg.norm(dp) + 0.05 * np.linalg.norm(dr)
            if np.linalg.norm(dp) < tol and np.linalg.norm(dr) < 0.1:
                return False
            if err < best - 1e-3:
                best, since_best = err, 0
            else:
                since_best += 1
                if since_best >= 20:
                    self.incomplete.append(round(float(np.linalg.norm(dp)), 3))
                    return False
            a = np.concatenate([np.clip(dp / POS_SCALE, -1, 1), np.clip(dr / ROT_SCALE, -1, 1), [gripper]])
            _, done = self.step(a)
            if done:
                return True
        return False

    def release(self) -> bool:
        self.gripper_cmd = -1.0
        for _ in range(self.cfg["release_steps"]):
            _, done = self.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
            if done:
                return True
        return False

    def lift(self) -> bool:
        obs = self.get_obs()
        tgt = np.array(obs["robot0_eef_pos"]) + [0, 0, self.cfg["lift_dz_m"]]
        return self._servo(tgt, obs["robot0_eef_quat"], self.cfg["retract_tol_m"], 40, self.gripper_cmd)

    def approach(self, target_pos, dz=None) -> bool:
        """Hover the open gripper above a target (rise to home height, translate,
        descend), in the home orientation, so pi0 takes over next to the right
        object. Target comes from the coordinator (sim ground truth here; VLM
        pointing + depth on a real robot)."""
        self.gripper_cmd = -1.0
        obs = self.get_obs()
        pos = np.array(obs["robot0_eef_pos"])
        z_travel = max(pos[2], self.home_pos[2])
        hover = np.asarray(target_pos, dtype=float) + [0, 0, self.cfg["approach_dz_m"] if dz is None else dz]
        hover[2] = min(hover[2], z_travel)
        for wp, tol in (([pos[0], pos[1], z_travel], 0.03),
                        ([hover[0], hover[1], z_travel], 0.02),
                        (hover, self.cfg["retract_tol_m"])):
            if self._servo(np.array(wp), self.home_quat, tol, self.cfg["retract_max_steps"], -1.0):
                return True
        return False

    def carry_to(self, dest_pos, obj_pos_fn) -> bool:
        """Carry the held object over a destination and let go (the "where" of a
        place that pi0 gets wrong): rise so the object clears the destination,
        translate, descend until the object is carry_clear_m above the
        destination point, open, and back off upward. The gripper
        orientation is brought back to home on the way (pi0's twisted grasp
        poses can't reach the workspace edge). Destination from the coordinator
        (sim ground truth here)."""
        obs = self.get_obs()
        pos, quat = np.array(obs["robot0_eef_pos"]), self.home_quat
        dest = np.asarray(dest_pos, dtype=float)
        # lift a little and turn to the home orientation in place (no full rise
        # at the current xy: see retract_home)
        z1 = pos[2] + self.cfg["lift_dz_m"]
        if self._servo(np.array([pos[0], pos[1], z1]), quat, 0.03, 60, 1.0):
            return True
        # objects are often held off-centre (bowls by the rim): aim the object, not the EEF
        offset = np.array(self.get_obs()["robot0_eef_pos"]) - np.asarray(obj_pos_fn(), dtype=float)
        z_travel = max(z1, dest[2] + offset[2] + self.cfg["carry_lift_m"])
        drop = dest + offset + [0, 0, self.cfg["carry_clear_m"]]
        for wp, tol, steps in (([drop[0], drop[1], z_travel], 0.02, 120),
                               (drop, 0.015, 80)):
            if self._servo(np.array(wp), quat, tol, steps, 1.0):
                return True
        if self.release():
            return True
        obs = self.get_obs()
        up = np.array(obs["robot0_eef_pos"]) + [0, 0, self.cfg["lift_dz_m"]]
        return self._servo(up, quat, 0.03, 40, -1.0)

    def finish_slide(self, contact_pos, contact_quat, direction, remaining_m) -> bool:
        """Finish a drawer push: come to the push point (the handle) from above
        and outside, then push along the slide axis by the remaining distance
        plus a margin, with a closed gripper."""
        d = np.asarray(direction, dtype=float)
        d /= np.linalg.norm(d)
        c = np.asarray(contact_pos, dtype=float)
        pre = c - self.cfg["push_backoff_m"] * d
        end = c + (remaining_m + self.cfg["push_margin_m"]) * d
        obs = self.get_obs()
        pos = np.array(obs["robot0_eef_pos"])
        z_safe = max(pos[2], c[2] + 0.08)
        for wp, tol, steps in (([pos[0], pos[1], z_safe], 0.03, 60),
                               ([pre[0], pre[1], z_safe], 0.02, 80),
                               (pre, 0.01, 60),
                               (end, 0.005, 80)):
            if self._servo(np.array(wp), contact_quat, tol, steps, 1.0):
                return True
        return False

    def retract_home(self) -> bool:
        """Lift a little (clear the clutter), come back over the home position,
        then rise to it. Rising to home height first with the arm reaching
        forward straightens the elbow into a singularity (joint 4 at its limit)
        from which OSC cannot pull back."""
        obs = self.get_obs()
        pos = np.array(obs["robot0_eef_pos"])
        z1 = min(pos[2] + self.cfg["lift_dz_m"], max(pos[2], self.home_pos[2]))
        for wp, quat, tol, steps in (([pos[0], pos[1], z1], obs["robot0_eef_quat"], 0.03, 40),
                                     ([self.home_pos[0], self.home_pos[1], z1], self.home_quat, 0.03, 80)):
            if self._servo(np.array(wp), quat, tol, steps, self.gripper_cmd):
                return True
        if self._servo(self.home_pos, self.home_quat, self.cfg["retract_tol_m"],
                       self.cfg["retract_max_steps"], self.gripper_cmd):
            return True
        # still far from home: pi0 left the arm stretched (elbow at its joint
        # limit), where Cartesian control has no radial authority. Go back in
        # joint space instead.
        err = float(np.linalg.norm(np.array(self.get_obs()["robot0_eef_pos"]) - self.home_pos))
        if err > self.cfg["joint_home_trigger_m"] and self.joint_home is not None:
            self.incomplete.append(round(err, 3))
            return self.joint_home(self.gripper_cmd)
        return False
