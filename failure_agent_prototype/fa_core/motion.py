"""Scripted motions that the control loop can run instead of pi0 (an agent's Override decision).

A Motion is stepped by the loop: next_action(obs) returns the next 7-D LIBERO action
(dpos 3, drot 3, gripper 1; each in [-1, 1]; gripper -1 = open, +1 = close), or None when finished.
Motions only read the observation the loop passes in (EE pose, gripper), never privileged sim state.

MoveTo is the closed-loop OSC move from RL/pixel_Q/oracle_handoff_test.py (move_to). It assumes the
LIBERO OSC_POSE controller applies position and rotation deltas in the world frame, with 1 action unit
= 5 cm and 0.5 rad. That assumption has NOT been checked on the real simulator yet: run
`tools/check_loop.py motion` first and look at its numbers and video (PHASE1 step 4).
"""

import numpy as np

POS_GAIN = 10.0     # action units per metre of position error (1 unit = 5 cm OSC delta)
ROT_GAIN = 1.0      # action units per radian of rotation error (1 unit = 0.5 rad)
POS_TOL = 0.005     # m
ROT_TOL = 0.05      # rad
LEG_STEPS = 60      # max steps for one MoveTo


def _transform_utils():
    from robosuite.utils import transform_utils as T

    return T


def rotation_error(target_quat, current_quat, T=None):
    """Axis-angle of R_target @ R_current^T (robosuite OSC applies R_goal = R_delta @ R_now)."""
    T = T or _transform_utils()
    return np.asarray(T.quat2axisangle(T.mat2quat(T.quat2mat(target_quat) @ T.quat2mat(current_quat).T)))


class Motion(object):
    label = "motion"

    def next_action(self, obs):
        raise NotImplementedError

    def describe(self):
        return {"type": self.label}


class ActionSequence(Motion):
    """Fixed list of actions (e.g. replaying a recording)."""

    label = "actions"

    def __init__(self, actions):
        self.actions, self.i = [np.asarray(a, dtype=np.float64) for a in actions], 0

    def next_action(self, obs):
        if self.i >= len(self.actions):
            return None
        a = self.actions[self.i]
        self.i += 1
        return a

    def describe(self):
        return {"type": self.label, "n": len(self.actions)}


class Hold(Motion):
    """Zero EE delta for `steps` steps with a fixed gripper command (HALT is Hold(current grip, n))."""

    label = "hold"

    def __init__(self, grip, steps):
        self.grip, self.steps, self.i = float(grip), int(steps), 0

    def next_action(self, obs):
        if self.i >= self.steps:
            return None
        self.i += 1
        return np.r_[np.zeros(6), self.grip]

    def describe(self):
        return {"type": self.label, "grip": self.grip, "steps": self.steps}


class MoveTo(Motion):
    """Closed-loop move of the EE to (pos, quat); quat=None keeps the orientation at the first step."""

    label = "move_to"

    def __init__(self, pos, grip, quat=None, pos_tol=POS_TOL, rot_tol=ROT_TOL, max_steps=LEG_STEPS,
                 pos_gain=POS_GAIN, rot_gain=ROT_GAIN, transform_utils=None):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.quat = None if quat is None else np.asarray(quat, dtype=np.float64)
        self.grip = float(grip)
        self.pos_tol, self.rot_tol, self.max_steps = pos_tol, rot_tol, max_steps
        self.pos_gain, self.rot_gain = pos_gain, rot_gain
        self.T = transform_utils
        self.i = 0
        self.reached = False
        self.final_pos_err = self.final_rot_err = None

    def next_action(self, obs):
        if self.quat is None:
            self.quat = np.array(obs["robot0_eef_quat"], dtype=np.float64, copy=True)
        err = self.pos - np.asarray(obs["robot0_eef_pos"])
        rot = rotation_error(self.quat, obs["robot0_eef_quat"], self.T)
        self.final_pos_err, self.final_rot_err = float(np.linalg.norm(err)), float(np.linalg.norm(rot))
        if self.final_pos_err < self.pos_tol and self.final_rot_err < self.rot_tol:
            self.reached = True
            return None
        if self.i >= self.max_steps:
            return None
        self.i += 1
        return np.concatenate([np.clip(self.pos_gain * err, -1, 1), np.clip(self.rot_gain * rot, -1, 1),
                               [self.grip]])

    def describe(self):
        return {"type": self.label, "pos": self.pos.tolist(), "grip": self.grip, "steps": self.i,
                "reached": self.reached, "pos_err_m": self.final_pos_err, "rot_err_rad": self.final_rot_err}


class Sequence(Motion):
    """Run motions one after another."""

    label = "sequence"

    def __init__(self, motions):
        self.motions, self.k = list(motions), 0

    def next_action(self, obs):
        while self.k < len(self.motions):
            a = self.motions[self.k].next_action(obs)
            if a is not None:
                return a
            self.k += 1
        return None

    def describe(self):
        return {"type": self.label, "parts": [m.describe() for m in self.motions]}
