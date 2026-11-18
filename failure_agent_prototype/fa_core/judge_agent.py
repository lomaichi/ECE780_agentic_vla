"""VLM subtask judge + high-level corrector (test-time budget: images, proprioception, RGB-D camera).

Every 20 control steps (1 fps) the VLM sees the agentview image (with a pixel grid for pointing), the wrist
image and the last two seconds of agentview frames, and returns:
  - whether the CURRENT subtask (pick(X) or place(X, D)) has succeeded,
  - whether there is a failure, high-level (wrong target / destination, not approaching, stall) or
    low-level (fine motor: slip, misaligned grasp, knock),
  - for high-level failures only, a correction:
      point_approach(u, v)   move the open gripper 7 cm above the pointed target, then pi0 grasps
      carry_to(u, v)         carry the held object over the pointed destination, lower and release
      release_and_retreat    open the gripper and lift (wrong object grasped)
      reprompt(text)         give pi0 a different instruction
Low-level failures are reported but not corrected (pi0 keeps trying): that is the RL skills' job.

The subtask plan comes from the VLM too (one call at the first step). Simulator state is never read:
pointing uses only the camera's depth image and calibration (fa_core.camera).
"""

import numpy as np

from . import common, motion
from .agents import Agent
from .loop import CONTINUE, Override, SetPrompt, Stop

GRID = 512                   # the pointing image is the upright agentview upscaled to GRID x GRID
APPROACH_DZ = 0.07           # pre-grasp height above the pointed surface point (POINT_APPROACH)
RELEASE_DZ = 0.10            # EE height above the pointed destination surface when releasing
TRAVEL_DZ = 0.18             # travel height above the higher of the start / goal points
COOLDOWN_STEPS = 60          # no new correction within 3 s after the last one ended
MAX_CORRECTIONS = 6

SYSTEM = """You are the monitoring agent of a robot arm (Franka, parallel gripper) in a LIBERO tabletop \
simulation. A learned policy (pi0) controls the arm; you watch it once per second, judge whether the \
current subtask has succeeded, and correct HIGH-LEVEL failures. Be precise and literal about what you see. \
Answer only through the structured output."""

SCENE_NOTES = """Camera and wording conventions (fixed for every scene):
- Image 1 is the front camera, facing the robot. The robot base is at the far (top) side of the image.
- "back" / "behind" = closer to the robot = higher in the image; "front" = closer to the camera = lower in the image.
- "left" / "right" in instructions are from the ROBOT's point of view: the robot's left appears on the IMAGE RIGHT.
- "middle" = between the others along the relevant direction.
- Image 1 has a pixel grid: coordinates run 0..512, grid lines every 64 px, labels in px. u = column (x, \
left->right), v = row (y, top->bottom)."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string", "description": "the object to pick, with its distinguishing qualifier, e.g. 'black bowl at the back'"},
        "object_label": {"type": "string", "description": "short snake_case label, e.g. black_bowl_back"},
        "destination": {"type": "string", "description": "where to put it, e.g. 'plate', 'inside the top drawer of the cabinet'"},
        "destination_label": {"type": "string", "description": "short snake_case label, e.g. plate, top_drawer"},
        "relation": {"type": "string", "enum": ["on", "in"]},
        "object_location": {"type": "string", "description": "where that object is in image 1 (to tell it from similar ones)"},
        "destination_location": {"type": "string", "description": "where the destination is in image 1"},
    },
    "required": ["object", "object_label", "destination", "destination_label", "relation", "object_location",
                 "destination_location"],
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string", "description": "one or two sentences: what the arm and the objects are doing now"},
        "subtask_success": {"type": "boolean", "description": "the CURRENT subtask's success criterion holds in the current images"},
        "object_in_gripper": {"type": "boolean", "description": "the TARGET object is held between the gripper fingers right now"},
        "object_lost": {"type": "boolean", "description": "only during place: the target object is no longer held AND not at the destination (dropped, so it must be picked again)"},
        "failure": {"type": "string", "enum": ["none", "high_level", "low_level"]},
        "failure_mode": {"type": "string", "description": "short name, e.g. wrong_object, wrong_destination, target_not_approached, stall, grasp_slip, misaligned_grasp, knocked_object; empty if none"},
        "action": {"type": "string", "enum": ["continue", "point_approach", "carry_to", "release_and_retreat", "reprompt"]},
        "point_u": {"type": "integer", "description": "image-1 grid pixel u of the point for point_approach / carry_to, else -1"},
        "point_v": {"type": "integer", "description": "image-1 grid pixel v, else -1"},
        "new_prompt": {"type": "string", "description": "for reprompt (or optionally after point_approach): the new instruction for pi0, else empty"},
        "reason": {"type": "string", "description": "why this judgement and this action (one or two sentences)"},
    },
    "required": ["observation", "subtask_success", "object_in_gripper", "object_lost", "failure", "failure_mode",
                 "action", "point_u", "point_v", "new_prompt", "reason"],
}


def grid_image(img, size=GRID, step=64):
    """Upscaled upright image with a labelled pixel grid (coordinates in the upscaled frame)."""
    import cv2

    g = cv2.resize(np.asarray(img), (size, size), interpolation=cv2.INTER_CUBIC)
    over = g.copy()
    for k in range(step, size, step):
        cv2.line(over, (k, 0), (k, size - 1), (255, 255, 0), 1)
        cv2.line(over, (0, k), (size - 1, k), (255, 255, 0), 1)
    g = cv2.addWeighted(over, 0.35, g, 0.65, 0)
    for k in range(0, size, step):
        cv2.putText(g, str(k), (k + 2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        if k:
            cv2.putText(g, str(k), (2, k - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    return g


def strip(frames):
    return np.concatenate(list(frames), axis=1)


class SubtaskJudgeAgent(Agent):
    name = "vlm_judge"
    privileged = False
    every_steps = 20

    def __init__(self, vlm, unproject, img_size=common.ENV_RESOLUTION, cooldown_steps=COOLDOWN_STEPS,
                 max_corrections=MAX_CORRECTIONS, log=None):
        """vlm: fa_core.vlm.ClaudeCLI; unproject(u, v) -> world xyz of an upright agentview pixel (RGB-D)."""
        self.vlm, self.unproject, self.img_size = vlm, unproject, img_size
        self.cooldown_steps, self.max_corrections = cooldown_steps, max_corrections
        self.log = log
        self.vlm.system = SYSTEM

    # ---------------------------------------------------------------- state
    def reset(self, task_language, prompt):
        self.task = task_language
        self.plan = None
        self.subtasks = []          # [{"label", "kind", "done_step"}]
        self.idx = 0
        self.history = []           # agentview frames at past calls
        self.decisions = []         # one record per call (also VLM failures)
        self.corrections = []
        self.last_correction_end = None
        self.pending_prompt = None
        self.confirm_final = False
        self.motion_active = False

    @property
    def current(self):
        return self.subtasks[min(self.idx, len(self.subtasks) - 1)] if self.subtasks else None

    def _plan(self, view):
        g = grid_image(view.image)
        out = self.vlm.ask([SCENE_NOTES, g, f'Task instruction: "{self.task}".\nDecompose it into pick(object) '
                            'then place(destination). Identify exactly which object instance the instruction '
                            'means (use the conventions above) and describe where it and the destination are '
                            'in image 1.'], PLAN_SCHEMA)
        self.plan = out
        self.subtasks = [
            {"kind": "pick", "label": f"pick({out['object_label']})", "done_step": None},
            {"kind": "place", "label": f"place({out['object_label']}, {out['destination_label']}, {out['relation']})",
             "done_step": None},
        ]
        return out

    def _criteria(self):
        p = self.plan
        if self.current["kind"] == "pick":
            return (f"CURRENT SUBTASK: {self.current['label']} = pick up the {p['object']}.\nSuccess criterion: the "
                    f"{p['object']} (the right instance!) is held between the gripper fingers AND lifted clearly off "
                    f"its support (at least ~5 cm). Touching, pushing or a closed gripper next to it is NOT success.")
        return (f"CURRENT SUBTASK: {self.current['label']} = place the {p['object']} {p['relation']} the "
                f"{p['destination']}.\nSuccess criterion: the {p['object']} rests {p['relation']} the "
                f"{p['destination']} AND the gripper has released it (it is no longer held). Hovering above the "
                f"destination while still holding it is NOT success.")

    def _prompt_text(self, view):
        p = self.plan
        width = float(view.gripper_qpos[0] - view.gripper_qpos[1])
        cmd = None if view.last_action is None else ("close" if view.last_action[-1] > 0 else "open")
        done = [f"{s['label']} judged done at step {s['done_step']}" for s in self.subtasks if s["done_step"] is not None]
        recent = [f"  step {d['step']}: [{d.get('subtask')}] success={d.get('subtask_success')} "
                  f"failure={d.get('failure')} action={d.get('action')} | {d.get('observation', '')}"
                  for d in self.decisions[-4:] if "error" not in d]
        corr = [f"  step {c['step']}: {c['action']} ({c['reason']})" for c in self.corrections[-3:]]
        since = None if self.last_correction_end is None else view.step - self.last_correction_end
        can_correct = len(self.corrections) < self.max_corrections and (since is None or since >= self.cooldown_steps)
        lines = [
            f'Task instruction: "{self.task}"',
            f"Plan: 1) {self.subtasks[0]['label']}  2) {self.subtasks[1]['label']}",
            f"Target object: {p['object']} (initially: {p['object_location']}). Destination: {p['destination']} "
            f"({p['destination_location']}).",
            self._criteria(),
            "Completed: " + ("; ".join(done) if done else "none"),
            f"Now: control step {view.step} (20 steps = 1 s), pi0 instruction: \"{view.prompt}\".",
            f"Proprioception: gripper finger opening {width * 100:.1f} cm (about 8 cm = fully open, under 1 cm = "
            f"closed on nothing), last gripper command: {cmd}, end-effector height {view.ee_pos[2]:.3f} m "
            f"(table top is about 0.90 m).",
            "Your recent judgements:\n" + ("\n".join(recent) if recent else "  none"),
            "Corrections so far:\n" + ("\n".join(corr) if corr else "  none"),
            ("You MAY correct now." if can_correct else
             "You may NOT correct now (cooldown after a correction, or the correction budget is used): action must "
             "be continue."),
            """Images: 1) front camera now, with pixel grid; 2) wrist camera now (looking down from the gripper); \
3) front camera 2 s ago, 1 s ago, now (left to right).

Decide:
- subtask_success: does the CURRENT subtask's criterion hold now? Judge only the current subtask.
- failure: high_level = reasoning/targeting error that pi0 will not fix by itself: heading to or grasping the \
WRONG object/instance, carrying the object to the WRONG place, not approaching the target at all, or stalled \
(arm idle or oscillating far from the target for several seconds). low_level = fine motor problem near the right \
target (misaligned or slipping grasp, knocked the object, jitter while grasping): pi0 keeps trying, report only. \
pi0 needs a few seconds to start moving; do not call a failure on ambiguous evidence.
- action (only for high_level failures, when allowed; otherwise continue):
  point_approach: gripper empty and not going to the right target -> point (point_u, point_v) at the CENTRE of \
the correct target object in image 1; the open gripper is moved 7 cm above it and pi0 then grasps.
  carry_to: the correct object is held but is being taken somewhere wrong, or pi0 stalls while holding it -> \
point at the CENTRE of the destination surface/opening in image 1; it is carried there, lowered and released.
  release_and_retreat: the WRONG object is held (or the gripper is closing on it) -> open the gripper and lift; if you also point (point_u, point_v) at the CENTRE of the correct target, the open gripper then goes 7 cm above it (otherwise give -1, -1).
  reprompt: give pi0 a clearer instruction (new_prompt) when the problem is what pi0 is trying to do.
  continue: otherwise.""",
        ]
        return "\n".join(lines)

    # ---------------------------------------------------------------- correction motions
    def _motion(self, action, u, v, view):
        ee = np.array(view.ee_pos, dtype=np.float64)
        quat = np.array(view.ee_quat, dtype=np.float64)
        if action == "release_and_retreat":
            legs = [motion.Hold(common.GRIPPER_OPEN, 10), motion.MoveTo(ee + [0, 0, 0.10], common.GRIPPER_OPEN, quat=quat)]
            if u is None:
                return motion.Sequence(legs), None
            target = np.asarray(self.unproject(u * self.img_size / GRID, v * self.img_size / GRID), dtype=np.float64)
            goal = target + [0, 0, APPROACH_DZ]
            travel = max(ee[2] + 0.10, goal[2] + 0.05)
            legs += [motion.MoveTo([goal[0], goal[1], travel], common.GRIPPER_OPEN, quat=quat),
                     motion.MoveTo(goal, common.GRIPPER_OPEN, quat=quat)]
            return motion.Sequence(legs), target
        su, sv = u * self.img_size / GRID, v * self.img_size / GRID
        target = np.asarray(self.unproject(su, sv), dtype=np.float64)
        if action == "point_approach":
            grip, goal = common.GRIPPER_OPEN, target + [0, 0, APPROACH_DZ]
        else:  # carry_to
            grip, goal = common.GRIPPER_CLOSE, target + [0, 0, RELEASE_DZ]
        travel = max(ee[2], goal[2]) + 0.05 if action == "point_approach" else max(ee[2], target[2] + TRAVEL_DZ)
        legs = [motion.MoveTo([ee[0], ee[1], travel], grip, quat=quat),
                motion.MoveTo([goal[0], goal[1], travel], grip, quat=quat),
                motion.MoveTo(goal, grip, quat=quat)]
        if action == "carry_to":
            legs += [motion.Hold(common.GRIPPER_OPEN, 15),
                     motion.MoveTo(goal + [0, 0, 0.10], common.GRIPPER_OPEN, quat=quat)]
        return motion.Sequence(legs), target

    # ---------------------------------------------------------------- main
    def snapshot(self):
        return {"idx": self.idx, "subtasks": [dict(s) for s in self.subtasks],
                "confirm_final": self.confirm_final}

    def act(self, view, priv=None):
        decision = self._act(view)
        if self.decisions and self.decisions[-1]["step"] == view.step:
            self.decisions[-1]["after"] = self.snapshot()
        return decision

    def _act(self, view):
        if self.motion_active:  # the loop calls the agent again only once the override has finished
            self.motion_active, self.last_correction_end = False, view.step
            if self.corrections:
                self.corrections[-1]["end_step"] = view.step
        self.history.append(view.image)
        if self.pending_prompt:
            p, self.pending_prompt = self.pending_prompt, None
            self.decisions.append({"step": view.step, "subtask": self.current["label"], "action": "reprompt",
                                   "new_prompt": p, "note": "prompt queued with the previous point_approach"})
            return SetPrompt(p, reason="after point_approach")
        try:
            if self.plan is None:
                plan = self._plan(view)
                self.decisions.append({"step": view.step, "type": "plan", **plan})
            prev = self.history[-3:]
            while len(prev) < 3:
                prev.insert(0, prev[0])
            out = self.vlm.ask([grid_image(view.image), view.wrist, strip(prev), self._prompt_text(view)],
                               JUDGE_SCHEMA)
        except Exception as e:
            self.decisions.append({"step": view.step, "subtask": self.current and self.current["label"],
                                   "error": repr(e)[:300]})
            return CONTINUE
        cur = self.current
        rec = {"step": view.step, "subtask": cur["label"], "subtask_idx": self.idx, **out}
        self.decisions.append(rec)

        # subtask bookkeeping
        if self.idx >= len(self.subtasks) - 1 and self.confirm_final:
            if out["subtask_success"]:
                rec["decision"] = "task complete (place confirmed twice)"
                return Stop("VLM: task complete")
            self.confirm_final = False
            cur["done_step"] = None
        if out["subtask_success"] and cur["done_step"] is None:
            cur["done_step"] = view.step
            if self.idx < len(self.subtasks) - 1:
                self.idx += 1
                rec["decision"] = f"advance to {self.current['label']}"
                return CONTINUE
            self.confirm_final = True
            rec["decision"] = "final subtask success, confirming at the next call"
            return CONTINUE
        if cur["kind"] == "place" and out["object_lost"]:
            self.idx = 0
            for s in self.subtasks:
                s["done_step"] = None
            rec["decision"] = "object lost: back to pick"
            return CONTINUE

        # high-level correction
        act = out["action"]
        since = None if self.last_correction_end is None else view.step - self.last_correction_end
        allowed = len(self.corrections) < self.max_corrections and (since is None or since >= self.cooldown_steps)
        if out["failure"] != "high_level" or act == "continue":
            return CONTINUE
        if not allowed:
            rec["decision"] = "correction suppressed (cooldown / budget)"
            return CONTINUE
        corr = {"step": view.step, "action": act, "reason": out["reason"], "failure_mode": out["failure_mode"],
                "subtask": cur["label"]}
        if act == "reprompt":
            if not out["new_prompt"].strip():
                rec["decision"] = "reprompt without a prompt: ignored"
                return CONTINUE
            corr["new_prompt"] = out["new_prompt"].strip()
            self.corrections.append(corr)
            self.last_correction_end = view.step
            rec["decision"] = "correction: reprompt"
            return SetPrompt(corr["new_prompt"], reason=out["reason"])
        valid = 0 <= out["point_u"] < GRID and 0 <= out["point_v"] < GRID
        if act in ("point_approach", "carry_to") and not valid:
            rec["decision"] = f"{act} without a valid point: ignored"
            return CONTINUE
        u, v = (out["point_u"], out["point_v"]) if valid else (None, None)
        try:
            m, target = self._motion(act, u, v, view)
        except Exception as e:
            rec["decision"] = f"{act} failed to build: {e!r}"
            return CONTINUE
        if u is not None:
            corr.update({"point_grid": [u, v], "point_img": [u * self.img_size / GRID, v * self.img_size / GRID],
                         "target_world": None if target is None else np.round(target, 3).tolist()})
        if act == "point_approach" and out["new_prompt"].strip():
            self.pending_prompt = corr["new_prompt"] = out["new_prompt"].strip()
        self.corrections.append(corr)
        self.motion_active = True
        rec["decision"] = f"correction: {act}"
        return Override(m, reason=f"{act}: {out['reason']}")
