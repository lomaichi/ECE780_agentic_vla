"""High-level (HL) recovery modules for the LIBERO backend: the VLM agent's slot in the main rollout.

  PrivilegedHL   training stand-in for the Moondream2 grounded recovery agent's HL rules (pixel_Q_RL_V2.md
                 sec. 9.4). Same rules and thresholds as Documentation/VLM_inference_speed_test/moondreamV2/
                 recovery_agent.py (pick subtask), but measured from the sim state (target position, grasp
                 contacts) instead of detect + depth. Corrections: point_approach to the true target, or
                 release_and_retreat + point_approach after grasping the wrong object.
  MoondreamHL    the real GroundedRecoveryAgent (Moondream2 on the GPU, moondream_server.py) behind the same
                 interface, for the deployment evaluation (train.py --mode evaluate --hl moondream). It also
                 supplies the gate's inputs (current subtask, grounded target point, held) as at test time.
  NoHL           nothing (condition "pi0 alone").

Interface used by libero_env.BranchingPickEnv: reset(), tick(step) -> (motion, info) or None, gate_inputs()
-> dict or None (None = the backend uses privileged gate inputs). The backend runs the motion synchronously
(the gate is closed while it runs), so a correction is never "running" at a tick. Branches never call tick().
"""

import sys

import numpy as np

from fa_core import common, motion, predicates as P  # noqa: E402  (sys.path set by libero_env / train)

S = common.CONTROL_FREQ
# recovery_agent.py constants (pick subtask)
START_GRACE_S, GRACE_S, WINDOW_S, CONFIRM_S = 1.0, 1.0, 3.0, 0.6
PROGRESS_M, FAR_M, GRASP_FAR_M = 0.02, 0.08, 0.12
LOW_DZ, BELOW_DEST_DZ, DESCENT_M, DESCENT_S, STALL_M = 0.06, -0.04, 0.02, 1.0, 0.01
TARGET_MOVING_M = 0.03
COOLDOWN_S, MAX_CORRECTIONS = 3.0, 6
APPROACH_DZ = 0.07               # point_approach: open gripper this far above the target's top
MOONDREAM_DIR = common.REPO / "Documentation" / "VLM_inference_speed_test" / "moondreamV2"


class NoHL(object):
    name = "none"

    def __init__(self, backend, cfg=None):
        self.b = backend

    def reset(self):
        pass

    def tick(self, step):
        return None

    def gate_inputs(self):
        return None


def point_approach_legs(ee, quat, goal, raise_first=0.0):
    """judge_agent point_approach: rise to a travel height, travel over the goal, descend to it (gripper open,
    orientation kept)."""
    z0 = ee[2] + raise_first
    travel = max(z0, goal[2]) + 0.05
    return [motion.MoveTo([ee[0], ee[1], travel], common.GRIPPER_OPEN, quat=quat),
            motion.MoveTo([goal[0], goal[1], travel], common.GRIPPER_OPEN, quat=quat),
            motion.MoveTo(goal, common.GRIPPER_OPEN, quat=quat)]


class PrivilegedHL(object):
    """recovery_agent.py's pick-subtask HL rules on privileged measurements:
      wrong_grasp            closed on another movable object, target not moving -> release_and_retreat + approach
      wrong_approach         descending / closing within LOW_DZ of the target top, > GRASP_FAR_M from it (xy)
                             -> point_approach(target)
      target_not_approached  gripper empty, > FAR_M from the pre-grasp point (APPROACH_DZ above the target top),
      / stall                < PROGRESS_M closer over WINDOW_S -> point_approach(target)
    A rule must hold for CONFIRM_S on consecutive ticks; START_GRACE_S at the start, GRACE_S after a correction,
    COOLDOWN_S between corrections, at most MAX_CORRECTIONS per rollout. The backend's gate adds its own
    cooldown after every correction."""

    name = "privileged"

    def __init__(self, backend, cfg=None):
        self.b = backend

    def reset(self):
        self.samples, self.pending, self.corrections = [], None, []
        self.watch_from = int(START_GRACE_S * S)
        self.last_end = None
        self.t_start = None

    def gate_inputs(self):
        return None

    def _measure(self):
        b = self.b
        ee = np.array(b.obs["robot0_eef_pos"], dtype=np.float64)
        tgt = P.obj_pos(b.env, b.obj)
        top = tgt[2] + b.top_offset
        others = [o for o in b.movable if o != b.obj and P.grasped(b.env, o)]
        return {"ee": ee, "dxy": float(np.hypot(*(ee[:2] - tgt[:2]))), "dz": float(ee[2] - top),
                "a": float(np.linalg.norm(ee - np.r_[tgt[:2], top + APPROACH_DZ])), "tgt": tgt,
                "held_target": P.grasped(b.env, b.obj), "held_other": others,
                "closing": b.last_action is not None and b.last_action[-1] > 0}

    def _ago(self, step, secs):
        if not self.samples or step - self.samples[0]["step"] < secs * S:
            return None
        return [s for s in self.samples if s["step"] <= step - secs * S][-1]

    def _detect(self, step, m):
        old = self._ago(step, WINDOW_S)
        if m["held_other"]:
            moved = None if self.t_start is None else float(np.linalg.norm(m["tgt"] - self.t_start))
            if moved is None or moved < TARGET_MOVING_M:
                return "wrong_grasp", "release_and_retreat"
        if m["held_target"]:
            return None
        before = self._ago(step, DESCENT_S)
        descending = before is not None and before["ee"][2] - m["ee"][2] > DESCENT_M
        if BELOW_DEST_DZ < m["dz"] < LOW_DZ and m["dxy"] > GRASP_FAR_M and (descending or m["closing"]):
            return "wrong_approach", "point_approach"
        if old is not None and m["a"] > FAR_M and old["a"] - m["a"] < PROGRESS_M:
            stall = float(np.linalg.norm(m["ee"] - old["ee"])) < STALL_M
            return ("stall" if stall else "target_not_approached"), "point_approach"
        return None

    def tick(self, step):
        if self.t_start is None:
            self.t_start = P.obj_pos(self.b.env, self.b.obj)
        if self.b.check.done or step < self.watch_from:
            self.pending = None
            return None
        m = self._measure()
        self.samples.append({"step": step, "ee": m["ee"], "a": m["a"]})
        det = self._detect(step, m)
        if det is None:
            self.pending = None
            return None
        mode, action = det
        if self.pending is None or self.pending[0] != mode:
            self.pending = (mode, step)
        if (step - self.pending[1]) / float(S) < CONFIRM_S:
            return None
        if len(self.corrections) >= MAX_CORRECTIONS or (
                self.last_end is not None and step - self.last_end < COOLDOWN_S * S):
            return None
        goal = self.b.target_approach_point()
        if goal is None:
            return None
        ee = m["ee"]
        quat = np.array(self.b.obs["robot0_eef_quat"], dtype=np.float64)
        if action == "release_and_retreat":
            legs = [motion.Hold(common.GRIPPER_OPEN, 10),
                    motion.MoveTo(ee + [0, 0, 0.10], common.GRIPPER_OPEN, quat=quat)]
            legs += point_approach_legs(ee, quat, goal, raise_first=0.10)
        else:
            legs = point_approach_legs(ee, quat, goal)
        info = {"action": action, "mode": mode, "goal": [round(float(x), 4) for x in goal], "dxy": round(m["dxy"], 4)}
        self.corrections.append(info)
        self.pending = None
        return motion.Sequence(legs), info

    def correction_done(self, step):
        self.last_end = step
        self.samples, self.pending = [], None
        self.watch_from = step + int(GRACE_S * S)


class MoondreamHL(object):
    """GroundedRecoveryAgent (Moondream2) in the HL slot. Deployment evaluation only (no branch points: the agent
    keeps its own history, which a snapshot restore would not rewind). Needs the Moondream server
    (moondream_server.py, started on demand by vlm_backends.MoondreamLocal) on the GPU next to pi0."""

    name = "moondream"

    def __init__(self, backend, cfg=None):
        self.b, self.cfg = backend, cfg or {}
        self.agent, self.vlm = None, None

    def _build(self):
        if str(MOONDREAM_DIR) not in sys.path:
            sys.path.insert(0, str(MOONDREAM_DIR))
        import recovery_agent  # noqa: E402
        import vlm_backends  # noqa: E402

        if self.vlm is None:
            self.vlm = vlm_backends.make_vlm("moondream2", self.cfg.get("vlm_model"))
        self.agent = recovery_agent.GroundedRecoveryAgent(self.vlm, recovery_agent.GroundedCamera(self.b.env))
        self.agent.every_steps = 4               # called at every tick (5 fps), as live_judge.py --judge-fps 5

    def reset(self):
        self._build()                            # the camera wraps the current task's env
        self.agent.reset(self.b.task_language, self.b.prompt)

    def tick(self, step):
        from fa_core import loop as L

        view = L.AgentView(step, step, self.b.obs, self.b.last_action, self.b.prompt, self.b.task_language)
        d = self.agent.act(view, None)
        if isinstance(d, L.Override):
            return d.motion, {"action": "override", "mode": d.reason}
        if isinstance(d, L.SetPrompt):
            self.b.prompt = d.prompt
            self.b.plan.clear()
        elif isinstance(d, L.Stop):
            self.b.stop_reason = d.reason or "agent stop"
        return None

    def correction_done(self, step):
        pass                                     # the agent notices at its next call (motion_active)

    def gate_inputs(self):
        a = self.agent
        if a is None or getattr(a, "plan", None) is None or a.plan.get("object") is None:
            return {"pick": False, "target": None, "held": False, "box": None}
        cur = a.current
        meas = getattr(a, "meas", None) or {}
        geo = getattr(a, "geo", None) or {}
        return {"pick": cur.get("kind") == "pick", "target": geo.get("target"), "held": bool(meas.get("held")),
                "box": (getattr(a, "boxes", None) or {}).get("object")}


def make_hl(name, backend, cfg=None):
    return {"none": NoHL, "privileged": PrivilegedHL, "moondream": MoondreamHL}[name](backend, cfg)
