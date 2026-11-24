"""Copy of failure_agent_prototype/fa_core/judge_agent.py for the VLM speed test (edit freely; the
original, used by tools/run_judge.py, is untouched). Needs failure_agent_prototype on sys.path.

VLM subtask judge + high-level corrector (test-time budget: images, proprioception, RGB-D camera).

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

from fa_core import common, motion
from fa_core.agents import Agent
from fa_core.loop import CONTINUE, Override, SetPrompt, Stop

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
        target = None if u is None else np.asarray(
            self.unproject(u * self.img_size / GRID, v * self.img_size / GRID), dtype=np.float64)
        return self._motion_world(action, target, view), target

    @staticmethod
    def _motion_world(action, target, view):
        """Correction motion to a world point (target surface point; None for a plain release_and_retreat)."""
        ee = np.array(view.ee_pos, dtype=np.float64)
        quat = np.array(view.ee_quat, dtype=np.float64)
        if action == "release_and_retreat":
            legs = [motion.Hold(common.GRIPPER_OPEN, 10), motion.MoveTo(ee + [0, 0, 0.10], common.GRIPPER_OPEN, quat=quat)]
            if target is None:
                return motion.Sequence(legs)
            goal = target + [0, 0, APPROACH_DZ]
            travel = max(ee[2] + 0.10, goal[2] + 0.05)
            legs += [motion.MoveTo([goal[0], goal[1], travel], common.GRIPPER_OPEN, quat=quat),
                     motion.MoveTo(goal, common.GRIPPER_OPEN, quat=quat)]
            return motion.Sequence(legs)
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
        return motion.Sequence(legs)

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


# ==================================================================== success-only judge (small VLMs) ==

import re  # noqa: E402

# LIBERO instructions with one pick-and-place: "put the X on/in Y", "pick up the X and place it in Y",
# "stack the X on Y", "put the X to the right of Y", optionally followed by "and close it" etc.
_PICK_PLACE = re.compile(r"^(?:pick up|put|place|stack) the (?P<obj>.+?) (?:and (?:put|place) it )?"
                         r"(?P<rel>on top of|on|in front of|in|inside|into|to the (?:right|left|front) of|behind) "
                         r"(?:the )?(?P<dest>.+?)(?: and (?P<rest>.+))?$")

# Spatial relations: the object goes on the table next to a reference object (the "destination"), on one side of
# the reference's detect box in the upright front image. left / right are the ROBOT's (as fa_core judge_agent
# SCENE_NOTES): the robot's right is the IMAGE LEFT; front = closer to the camera = lower in the image. Checked
# against the BDDL goal regions projected into the image (libero_90 tasks 34, 69, 70, 76).
#   relation phrase -> (label, image side of the reference box)
SIDE_RELATIONS = {"to the right of": ("right_of", "left"), "to the left of": ("left_of", "right"),
                  "to the front of": ("front_of", "down"), "in front of": ("front_of", "down"),
                  "behind": ("behind", "up")}
SIDE_AT = 0.25           # place success: the object's box centre within this much (image fraction) of the
                         # reference box's edge, on the relation's side
SIDE_SLACK = 0.05        # ... and inside the reference box's extent along the other axis, give or take this much


def side_region(d, side, depth, overlap=0.0):
    """Normalised box next to detect box d on `side` of the image ('left' / 'right' / 'up' / 'down'): `depth` deep
    away from d's edge (reaching `overlap` back into d), as wide as d along the edge."""
    x0, x1, y0, y1 = d["x_min"], d["x_max"], d["y_min"], d["y_max"]
    if side == "left":
        x0, x1 = x0 - depth, x0 + overlap
    elif side == "right":
        x0, x1 = x1 - overlap, x1 + depth
    elif side == "up":
        y0, y1 = y0 - depth, y0 + overlap
    else:
        y0, y1 = y1 - overlap, y1 + depth
    return {"x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1}

SUCCESS_QUESTIONS = {
    "pick": "Is the {object} held in the robot gripper and lifted up off the surface it was on? Answer yes or no.",
    "place": "Is the {object} resting {relation} the {destination}, released from the robot gripper? Answer yes or no.",
    "finish": 'Has the robot fully completed this instruction: "{task}"? Answer yes or no.',
}


def parse_instruction(task):
    """Rule-based plan (stand-in for the coordinator): pick / place / finish subtasks with their questions."""
    m = _PICK_PLACE.match(task.strip().lower())
    if not m:
        return {"object": None, "destination": None, "relation": None, "on_top": False, "side": None,
                "subtasks": [{"kind": "finish", "label": "finish(task)"}]}
    obj, dest, r = m.group("obj"), m.group("dest"), m.group("rel")
    side = SIDE_RELATIONS.get(r)
    if side is not None:
        rel, rel_label = r, side[0]         # relation keeps the phrase ("resting to the right of the caddy")
    else:
        rel = rel_label = "in" if r in ("in", "inside", "into") else "on"
    top = r == "on top of"
    lab = lambda s: re.sub(r"\W+", "_", s).strip("_")  # noqa: E731
    subs = [{"kind": "pick", "label": f"pick({lab(obj)})"},
            {"kind": "place", "label": f"place({lab(obj)}, {lab(dest)}, {rel_label})"}]
    if m.group("rest"):
        subs.append({"kind": "finish", "label": f"finish({lab(m.group('rest'))})"})
    return {"object": obj, "destination": dest, "relation": rel, "on_top": top,
            "side": None if side is None else side[1], "subtasks": subs}


class SubtaskSuccessJudge(SubtaskJudgeAgent):
    """Subtask-success judge for small, fast VLMs (Moondream2): every 20 steps (1 fps) ONE short yes/no
    question about the current subtask on the front + wrist camera images. No failure typing, no corrections.

    The plan is rule-based (parse_instruction); only the success judgement is the VLM's. Bookkeeping as in
    SubtaskJudgeAgent: a yes advances to the next subtask; the last one needs two yes in a row to stop."""

    name = "vlm_success_judge"

    def __init__(self, vlm, log=None, wrist=True):
        self.vlm, self.log, self.wrist = vlm, log, wrist
        self.vlm.system = ""

    def _plan(self, view):
        self.plan = parse_instruction(self.task)
        # history: earlier completions that were later contradicted ({"done_step", "lost_step", "reason"})
        self.subtasks = [dict(s, done_step=None, history=[]) for s in self.plan["subtasks"]]
        return {k: v for k, v in self.plan.items() if k != "subtasks"}

    def question(self):
        return SUCCESS_QUESTIONS[self.current["kind"]].format(task=self.task, **self.plan)

    def _act(self, view):
        if self.plan is None:
            self.decisions.append({"step": view.step, "type": "plan", **self._plan(view),
                                   "subtasks": [s["label"] for s in self.subtasks]})
        cur, q = self.current, self.question()
        parts = [view.image] + ([view.wrist] if self.wrist else []) + [q]
        try:
            ans = self.vlm.ask(parts, None)["answer"]
        except Exception as e:
            self.decisions.append({"step": view.step, "subtask": cur["label"], "error": repr(e)[:300]})
            return CONTINUE
        yes = ans.strip().lower().startswith("yes")
        rec = {"step": view.step, "subtask": cur["label"], "subtask_idx": self.idx, "question": q,
               "observation": f"{cur['label']}: {ans.strip()[:60]}", "subtask_success": yes, "failure": "none"}
        self.decisions.append(rec)
        if self.idx >= len(self.subtasks) - 1 and self.confirm_final:
            if yes:
                rec["decision"] = "task complete (confirmed twice)"
                return Stop("VLM: task complete")
            self.confirm_final, cur["done_step"] = False, None
        if yes and cur["done_step"] is None:
            cur["done_step"] = view.step
            if self.idx < len(self.subtasks) - 1:
                self.idx += 1
                rec["decision"] = f"advance to {self.current['label']}"
            else:
                self.confirm_final = True
                rec["decision"] = "final subtask success, confirming at the next call"
        return CONTINUE


# ==================================================================== grounded success judge (Moondream2) ==

_QUALIFIERS = ("back", "front", "middle", "left", "right")
PICK_RISE = 0.04         # target box centre must rise this much (fraction of image height) above its start
                         # (a lift moves the object UP in the front image; pushing it along the table moves it
                         # down/sideways, so plain displacement is not used: it fired on pushes)
GRIP_MIN_M = 0.003       # finger opening above this with a close command = something between the fingers (bowl rims: ~0.5 cm)
STABLE = 0.02            # place: target box centre moved less than this since the previous call
OPEN_M = 0.06            # place: finger opening above this = released
LOST_CONFIRM = 2         # a completed subtask is undone after it is contradicted on this many consecutive calls
                         # (one call is not enough: a bowl released just above the plate is briefly "not at" it)
MAX_JUMP = 0.10          # the tracked box may move at most MAX_JUMP + MAX_JUMP_PER_S * (seconds since it was last
MAX_JUMP_PER_S = 1.0     # seen) between calls; a farther "nearest" box is another object (detect switched to it at
                         # 5 fps: "rose +0.52" in one call, false pick 28 steps early), so the object counts as unseen


def split_qualifier(phrase):
    """'black bowl at the back' -> ('black bowl', 'back'); 'middle black bowl' -> ('black bowl', 'middle')."""
    words, qual = phrase.split(), None
    for q in _QUALIFIERS:
        if q in words:
            qual = q
    core = re.sub(r"\b(?:at|on|in) the (?:back|front|middle|left|right)\b", "", phrase)
    core = " ".join(w for w in core.split() if w not in _QUALIFIERS + ("side",))
    return core.strip(), qual


def centre(b):
    return (0.5 * (b["x_min"] + b["x_max"]), 0.5 * (b["y_min"] + b["y_max"]))


def pick_instance(boxes, qual, world=None):
    """The box the qualifier means (front camera; the robot's left is the image right).

    world: optional world points of the boxes (RGB-D back-projection). "middle" is then taken along the world axis
    with the larger spread (front/back = x, left/right = y): in the image, perspective shrinks the front/back
    spread, so the image axis picks left/right orderings of front/middle/back rows (task 17: the front bowl)."""
    if not boxes:
        return None
    if qual == "middle" and world is not None and len(boxes) >= 3:
        w = np.asarray(world)
        ax = 0 if np.ptp(w[:, 0]) >= np.ptp(w[:, 1]) else 1
        return boxes[sorted(range(len(boxes)), key=lambda i: w[i, ax])[len(boxes) // 2]]
    c = [centre(b) for b in boxes]
    if qual == "back":
        return boxes[min(range(len(c)), key=lambda i: c[i][1])]
    if qual == "front":
        return boxes[max(range(len(c)), key=lambda i: c[i][1])]
    if qual == "left":
        return boxes[max(range(len(c)), key=lambda i: c[i][0])]
    if qual == "right":
        return boxes[min(range(len(c)), key=lambda i: c[i][0])]
    if qual == "middle" and len(boxes) >= 3:
        ax = 1 if np.ptp([p[1] for p in c]) >= np.ptp([p[0] for p in c]) else 0
        order = sorted(range(len(c)), key=lambda i: c[i][ax])
        return boxes[order[len(order) // 2]]
    return boxes[0]


class GroundedSuccessJudge(SubtaskSuccessJudge):
    """Subtask-success judge built on Moondream2's grounded skill (detect) instead of yes/no VQA, which a
    2B model answers "yes" to regardless of the image.

    At the first call: rule-based plan, detect(object) picks the instance the qualifier means, and
    detect(destination) gives the destination box (static fixtures). Every call (1 fps): ONE detect(object)
    on the front camera, tracked to the box nearest the previous one, and
      pick  success = gripper closed on something (proprioception) AND the object box rose >= PICK_RISE
                      above its start (lifted with the gripper)
      place success = gripper open AND the object box is at the destination box (on: its centre inside the
                      destination box, on top of: in the box's upper half; in: inside; to the right of / left of
                      / front of / behind: within SIDE_AT of the box's edge on that side, SIDE_RELATIONS) AND it
                      did not move since the previous call (resting, not carried).
    Non pick-and-place instructions fall back to the yes/no question (SubtaskSuccessJudge).

    Monitoring is continuous: a success is not final. Every call also re-checks the completed subtasks with
    the same detect call and proprioception:
      pick  (while the place is not done) the object must stay carried: gripper closed on something, or the
            object already at the destination (being released there). Gripper open, or closed on nothing,
            with the object detected away from the destination contradicts it (dropped / slipped).
      place (while a later subtask runs) the object, when detected, must still be at the destination.
    Only positive evidence counts (an undetected object contradicts nothing: it may be in a drawer).
    A subtask contradicted on LOST_CONFIRM consecutive calls is undone: it and every later subtask are reset
    and the judge goes back to it (the lost completion is kept in the subtask's "history"). After a lost
    pick, the lift reference is re-based to where the object comes to rest (updated while the gripper is
    open), so the re-grasp is judged against the object's new position, not its initial one. The final
    subtask is not monitored: its success ends the episode after the confirm-twice rule.

    task_done_stop=False (the framework, failure_agent_prototype/framework): the judge never decides the task is
    complete and never returns Stop; the environment's success check ends the episode. The final subtask is still
    judged at every call like the others: its done_step is set when it is judged done and cleared when a later
    call judges it not done (its latest verdict)."""

    name = "vlm_grounded_judge"

    def __init__(self, vlm, log=None, lost_confirm=LOST_CONFIRM, task_done_stop=True):
        super().__init__(vlm, log=log, wrist=False)
        self.phase = "plan"
        self.lost_confirm = lost_confirm
        self.task_done_stop = task_done_stop

    def reset(self, task_language, prompt):
        super().reset(task_language, prompt)
        self.meas = None            # measurements of the latest call (see _measure)
        self.lost = None            # (subtask index, consecutive contradicted calls) of a pending contradiction
        self.rebase = False         # after a lost pick: the lift reference follows the object while released

    def _plan(self, view):
        out = super()._plan(view)
        self.boxes = {"object": None, "start": None, "prev": None, "destination": None}
        if self.plan["object"] is None:
            return out
        self.obj_phrase, self.obj_qual = split_qualifier(self.plan["object"])
        self.dest_phrase, self.dest_qual = split_qualifier(self.plan["destination"])
        self.phase = "plan"
        try:
            b = self._instance(self.vlm.detect(view.image, self.obj_phrase), self.obj_qual)
            d = self._instance(self.vlm.detect(view.image, self.dest_phrase), self.dest_qual)
        finally:
            self.phase = "judge"
        self.boxes.update({"object": b, "start": b, "prev": b, "destination": d})
        self.last_seen = view.step
        out.update({"detect_object": self.obj_phrase, "object_qualifier": self.obj_qual, "object_box": b,
                    "detect_destination": self.dest_phrase, "destination_box": d})
        return out

    def _instance(self, boxes, qual):
        return pick_instance(boxes, qual, world=self._box_world(boxes) if qual == "middle" else None)

    def _box_world(self, boxes):
        """World points of detect boxes, if the agent has a depth camera (recovery_agent does); else None."""
        return None

    def _measure(self, view):
        """ONE detect(object) call + proprioception -> measurements for the success and monitoring checks.

        rise / move / still / at are None when the object (or, for at, the destination) is not detected."""
        B = self.boxes
        width = float(view.gripper_qpos[0] - view.gripper_qpos[1])
        closing = view.last_action is not None and view.last_action[-1] > 0
        found = self.vlm.detect(view.image, self.obj_phrase)
        prev = B["object"]
        jumped = False
        if found and prev is not None:
            pc = centre(prev)
            b = min(found, key=lambda x: np.hypot(centre(x)[0] - pc[0], centre(x)[1] - pc[1]))
            limit = MAX_JUMP + MAX_JUMP_PER_S * (view.step - self.last_seen) / common.CONTROL_FREQ
            if np.hypot(centre(b)[0] - pc[0], centre(b)[1] - pc[1]) > limit:
                b, jumped = None, True
        else:
            b = pick_instance(found, self.obj_qual) if found else None
        if b is not None:
            self.last_seen = view.step
        B["prev"], B["object"] = prev, (b or prev)
        if self.rebase and not closing and b is not None:
            B["start"] = b  # lost pick: lift is measured from where the released object lies now
        m = {"found": b is not None, "jumped": jumped, "width": width, "closing": closing,
             "held": closing and width > GRIP_MIN_M, "released": not closing or width > OPEN_M,
             "grip": f"gripper {'closed' if closing else 'open'} {width * 100:.1f} cm",
             "rise": None, "move": None, "still": None, "at": None}
        if b is None or B["start"] is None:
            return m
        (cx, cy), (sx, sy) = centre(b), centre(B["start"])
        m["rise"], m["move"] = sy - cy, float(np.hypot(cx - sx, cy - sy))
        pc = centre(B["prev"]) if B["prev"] is not None else (cx, cy)
        m["still"] = float(np.hypot(cx - pc[0], cy - pc[1])) < STABLE
        d = B["destination"]
        side = self.plan.get("side")
        if d is not None and side is not None:
            # next to the reference object, on the relation's side of its box (SIDE_RELATIONS)
            r = side_region(d, side, SIDE_AT)
            if side in ("left", "right"):
                r["y_min"], r["y_max"] = r["y_min"] - SIDE_SLACK, r["y_max"] + SIDE_SLACK
            else:
                r["x_min"], r["x_max"] = r["x_min"] - SIDE_SLACK, r["x_max"] + SIDE_SLACK
            m["at"] = r["x_min"] <= cx <= r["x_max"] and r["y_min"] <= cy <= r["y_max"]
        elif d is not None:
            inside_x = d["x_min"] <= cx <= d["x_max"]
            if self.plan["on_top"]:
                m["at"] = inside_x and d["y_min"] - 0.1 <= cy <= 0.5 * (d["y_min"] + d["y_max"])
            else:
                m["at"] = inside_x and d["y_min"] - 0.03 <= cy <= d["y_max"] + 0.03
        return m

    def _judge(self, view):
        """(success, observation text) for the current pick / place subtask from one detect call."""
        m = self.meas = self._measure(view)
        if m["rise"] is None:
            return False, f"{self.obj_phrase} not detected" + (" (nearest box jumped: another object)" if m["jumped"] else "")
        if self.current["kind"] == "pick":
            ok = m["held"] and m["rise"] >= PICK_RISE
            return ok, f"{self.obj_phrase} rose {m['rise']:+.2f} moved {m['move']:.2f}, {m['grip']}"
        if m["at"] is None:
            return False, f"{self.dest_phrase} not detected"
        ok = m["at"] and m["released"] and m["still"]
        return ok, (f"{self.obj_phrase} {'at' if m['at'] else 'not at'} {self.dest_phrase}, "
                    f"{'still' if m['still'] else 'moving'}, {m['grip']}")

    def _contradiction(self):
        """(index, reason) of a completed, non-final subtask the latest measurements contradict, else None."""
        m = self.meas
        if m is None or not m["found"]:
            return None
        for k, s in enumerate(self.subtasks[:-1]):
            if s["done_step"] is None:
                continue
            nxt_done = self.subtasks[k + 1]["done_step"] is not None
            if s["kind"] == "pick" and not nxt_done and not m["held"] and m["at"] is False:
                why = "closed on nothing" if m["closing"] else "released"
                return k, f"{self.obj_phrase} not in the gripper ({why}, {m['width'] * 100:.1f} cm) " \
                          f"and not at {self.dest_phrase}"
            if s["kind"] == "place" and m["at"] is False:
                return k, f"{self.obj_phrase} no longer at {self.dest_phrase}"
        return None

    def _monitor(self, view, rec):
        """Continuous monitoring of completed subtasks. Returns True if one was undone at this call."""
        c = self._contradiction()
        if c is None:
            self.lost = None
            return False
        k, reason = c
        n = self.lost[1] + 1 if self.lost is not None and self.lost[0] == k else 1
        self.lost = (k, n)
        s = self.subtasks[k]
        rec["monitor"] = f"{s['label']} contradicted {n}/{self.lost_confirm}: {reason}"
        if n < self.lost_confirm:
            return False
        for t in self.subtasks[k:]:
            if t["done_step"] is not None:
                t["history"].append({"done_step": t["done_step"], "lost_step": view.step, "reason": reason})
            t["done_step"] = None
        self.idx, self.confirm_final, self.lost = k, False, None
        if s["kind"] == "pick":
            self.rebase = True
        rec["decision"] = f"{s['label']} LOST: back to it"
        if self.log:
            self.log.info("step %d: %s", view.step, rec["decision"])
        return True

    def _act(self, view):
        if self.plan is None:
            try:
                plan = self._plan(view)
            except Exception as e:
                self.plan = None
                self.decisions.append({"step": view.step, "error": f"plan: {e!r}"[:300]})
                return CONTINUE
            self.decisions.append({"step": view.step, "type": "plan", **plan,
                                   "subtasks": [s["label"] for s in self.subtasks]})
        if self.plan["object"] is None:
            return super()._act(view)
        cur = self.current
        try:
            if cur["kind"] == "finish":  # yes/no question below; detect only to keep monitoring pick / place
                self.meas = self._measure(view)
            else:
                yes, obs_text = self._judge(view)
        except Exception as e:
            self.decisions.append({"step": view.step, "subtask": cur["label"], "error": repr(e)[:300]})
            return CONTINUE
        if cur["kind"] == "finish":
            rec = {"step": view.step, "subtask": cur["label"], "subtask_idx": self.idx, "failure": "none"}
            if self._monitor(view, rec):
                rec["observation"] = rec["monitor"]
                self.decisions.append(rec)
                return CONTINUE
            out = super()._act(view)
            if rec.get("monitor") and self.decisions and self.decisions[-1]["step"] == view.step:
                self.decisions[-1]["monitor"] = rec["monitor"]
                self.decisions[-1]["observation"] += f" | {rec['monitor']}"
            return out
        rec = {"step": view.step, "subtask": cur["label"], "subtask_idx": self.idx, "subtask_success": yes,
               "failure": "none", "observation": f"{cur['label']}: {'SUCCESS' if yes else 'not yet'} ({obs_text})",
               "object_box": self.boxes["object"]}
        self.decisions.append(rec)
        regressed = self._monitor(view, rec)
        if rec.get("monitor"):
            rec["observation"] += f" | {rec['monitor']}"
        if regressed:
            return CONTINUE
        final = self.idx >= len(self.subtasks) - 1
        if final and not self.task_done_stop:      # subtask verdict only; task success is the environment's
            if yes and cur["done_step"] is None:
                cur["done_step"] = view.step
                rec["decision"] = "final subtask judged done (task success left to the environment)"
            elif not yes and cur["done_step"] is not None:
                cur["done_step"] = None
                rec["decision"] = "final subtask judged not done again"
            return CONTINUE
        if final and self.confirm_final:
            if yes:
                rec["decision"] = "task complete (confirmed twice)"
                return Stop("VLM: task complete")
            self.confirm_final, cur["done_step"] = False, None
        if yes and cur["done_step"] is None:
            cur["done_step"] = view.step
            if self.idx < len(self.subtasks) - 1:
                self.idx += 1
                rec["decision"] = f"advance to {self.current['label']}"
            else:
                self.confirm_final = True
                rec["decision"] = "final subtask success, confirming at the next call"
        return CONTINUE
