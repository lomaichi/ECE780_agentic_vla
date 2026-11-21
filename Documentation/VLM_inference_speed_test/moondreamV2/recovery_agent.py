"""Grounded failure detection + high-level recovery for Moondream2, on top of the grounded success judge.

Moondream2 cannot do what fa_core.judge_agent.SubtaskJudgeAgent asks of Opus (a JSON judgement with failure
type, action and a pointed pixel): its yes/no answers carry no signal (README). So failures are not asked
about; they are measured, from the same single detect(object) call per judge call that GroundedSuccessJudge
already makes, plus proprioception and the agentview RGB-D camera (depth image + calibration, fa_core.camera):

  object / destination box (detect)  --depth-->  world surface point of the target and of the destination
  EE position (proprio)               vs those points: horizontal distance, height above them, progress

Failure rules (library ids from failure_survey/skill_library/library.json). A rule must hold on consecutive
calls for CONFIRM_S before it counts; windowed rules need WINDOW_S of calls since the last subtask change or
correction (GRACE_S after it).

  pick (gripper not yet carrying the target)
    wrong_grasp            HL-3/HL-4  gripper closed on something GRASP_FAR_M from the target, target not moving
                                      -> release_and_retreat, then approach the target
    wrong_approach         HL-2/3/4   EE descending (or closing the gripper) within LOW_DZ of the target height,
                                      GRASP_FAR_M away from it      -> point_approach(target)
    target_not_approached  HL-7       gripper empty, EE FAR_M from the pre-grasp point (APPROACH_DZ above the
                                      target, 3-D: hovering high above it counts) and less than PROGRESS_M
                                      closer to it over WINDOW_S
    stall                  HL-1/S1    same, and the EE moved less than STALL_M      -> point_approach(target)
    grasp_loss             HL-8       the judge undid a completed pick (object dropped away from the
                                      destination)                -> release_and_retreat + approach the object
    missed_grasp           LL-4/LL-5  gripper closed on nothing next to the target   (reported only)
    pregrasp_hesitation    LL-8       open gripper next to the target, EE still over WINDOW_S (reported only)
  Destination point: inside the destination box (its highest surface for "on top of"); for "in the ... drawer",
  point("open drawer") instead (detect boxes the whole cabinet).

  place (target carried)
    wrong_destination      HL-6       carried object descending, EE within RELEASE_LOW_DZ of the destination
                                      height but FAR_M away from it -> carry_to(destination)
    destination_not_approached HL-6   FAR_M from the release point (RELEASE_DZ above the destination, 3-D),
                                      not getting PROGRESS_M closer to it over WINDOW_S
    handoff_stall          HL-1       same, EE still                -> carry_to(destination)
    release_stall          LL-7       carried object over the destination, EE still (reported only)
    placement_instability  LL-6       the object was released next to the destination but is not on/in it
                                      (judge undid the pick)       (reported only)

High-level failures get a scripted correction (the motions of SubtaskJudgeAgent, aimed at the measured world
points instead of a VLM-pointed pixel), with the same cooldown and budget. Low-level failures are reported,
not corrected: pi0 keeps trying (that is the RL skills' job). Instructions that are not one pick-and-place
fall back to the success judge without recovery. Simulator state is never read by the agent.
"""

import numpy as np

import judge_agent as J
from fa_core import camera, common
from fa_core.loop import CONTINUE, Override, Stop

S = common.CONTROL_FREQ          # control steps per second
START_GRACE_S = 1.0              # no failure checks in the first second (pi0 starts moving)
GRACE_S = 1.0                    # ... and for this long after a subtask change or a correction
WINDOW_S = 3.0                   # progress window (the library's "3 frames at 1 fps")
CONFIRM_S = 0.6                  # a failure condition must hold this long on consecutive calls
PROGRESS_M = 0.02                # getting closer by less than this over WINDOW_S = not approaching
NEAR_M = 0.06                    # EE within this horizontal distance of the target / destination point = at it
FAR_M = 0.08                     # beyond this from the pre-grasp / release point = not there yet; a release is at
                                 # the wrong place
GRASP_FAR_M = 0.12               # a grasp / descent this far (xy) from the target point is at another object (a bowl
                                 # held by its rim puts the EE ~7 cm from its centre, + ~3 cm depth-point offset:
                                 # 8 cm released a correct grasp, task 25; the wrong grasps seen were 21-29 cm away)
LOW_DZ = 0.06                    # EE this little above the target surface point = descending to grasp
RELEASE_LOW_DZ = 0.08            # carrying: EE this little above the destination surface = lowering to release
BELOW_DEST_DZ = -0.04            # ... and not this far below it (en route under a tall destination, e.g. a cabinet top;
                                 # also the lower bound of the wrong_approach height band)
DESCENT_M = 0.02                 # EE z dropped this much over DESCENT_S = descending
DESCENT_S = 1.0
STALL_M = 0.01                   # EE moved less than this over WINDOW_S = stalled
TARGET_MOVING = 0.03             # target box moved this much (normalised) since the start = it is the one held
COOLDOWN_S = 3.0
MAX_CORRECTIONS = 6


class GroundedCamera(object):
    """The agentview RGB-D camera of the test-time budget: depth image + calibration. No object poses."""

    def __init__(self, env, size=common.ENV_RESOLUTION):
        self.env, self.size = env, size

    def depth(self):
        return camera.depth_image(self.env, self.size)

    def unproject(self, u, v, depth):
        """World point at upright pixel (u, v) in px."""
        return camera.unproject(self.env, u, v, self.size, depth=depth, window=1)

    def project(self, xyz):
        """World point -> upright pixel, normalised to 0..1 (like the detect boxes)."""
        u, v = camera.project(self.env, xyz, self.size)
        return float(u) / self.size, float(v) / self.size


def surface_point(cam, depth, box, xs=(0.3, 0.7), ys=(0.3, 0.7), n=5, keep="low", band=0.06):
    """World point of the surface seen inside a detect box (normalised): median of an n x n grid of depth samples
    in the inner part of the box. keep="low": without samples more than `band` m above the lowest one (the arm or
    gripper in front of an object); "high": only samples within `band` of the highest one (the top surface of a
    fixture, for "on top of"); "all": every sample."""
    pts = []
    for fy in np.linspace(ys[0], ys[1], n):
        for fx in np.linspace(xs[0], xs[1], n):
            u = (box["x_min"] + fx * (box["x_max"] - box["x_min"])) * cam.size
            v = (box["y_min"] + fy * (box["y_max"] - box["y_min"])) * cam.size
            pts.append(cam.unproject(u, v, depth))
    pts = np.asarray(pts)
    if keep == "low":
        pts = pts[pts[:, 2] <= pts[:, 2].min() + band]
    elif keep == "high":
        pts = pts[pts[:, 2] >= pts[:, 2].max() - band]
    return np.median(pts, axis=0)


def xy_dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class GroundedRecoveryAgent(J.GroundedSuccessJudge):
    name = "vlm_grounded_recovery"

    def __init__(self, vlm, cam, log=None, cooldown_s=COOLDOWN_S, max_corrections=MAX_CORRECTIONS):
        super().__init__(vlm, log=log)
        self.cam, self.cooldown_steps, self.max_corrections = cam, int(cooldown_s * S), max_corrections

    def reset(self, task_language, prompt):
        super().reset(task_language, prompt)
        self.geo = {"target": None, "dest": None, "target_px": None, "dest_px": None, "ee_px": None}
        self.samples = []           # one per judge call since watch_from: {"step", "ee", "d_target", "d_dest"}
        self.watch_from = None      # failure checks start at this step
        self.pending = None         # (mode, first step) of a failure condition not yet confirmed
        self.failures = []          # confirmed failures (one entry per onset): step, level, mode, library, action
        self.lost_event = None      # set by _monitor when the judge undoes a completed pick

    # ---------------------------------------------------------------- geometry
    def _box_world(self, boxes):
        depth = self.cam.depth()
        return [surface_point(self.cam, depth, b) for b in boxes]

    def _plan(self, view):
        out = super()._plan(view)
        if self.plan["object"] is None:
            return out
        depth = self.cam.depth()
        B = self.boxes
        if B["object"] is not None:
            self.geo["target"] = surface_point(self.cam, depth, B["object"])
        if self.plan["relation"] == "in" and "drawer" in self.plan["destination"]:
            # detect boxes a drawer as the whole cabinet; point("open drawer") lands inside the open drawer
            self.phase = "plan"
            try:
                pts = self.vlm.point(view.image, "open drawer")
            finally:
                self.phase = "judge"
            if pts:
                u, v = pts[0]["x"] * self.cam.size, pts[0]["y"] * self.cam.size
                self.geo["dest"] = np.asarray(camera.unproject(self.cam.env, u, v, self.cam.size, depth=depth))
                out["dest_point"] = pts[0]
        if self.geo["dest"] is None and B["destination"] is not None:
            # on top of a fixture: its highest surface (the upper part of the box)
            top = self.plan["on_top"]
            self.geo["dest"] = surface_point(self.cam, depth, B["destination"], ys=(0.05, 0.4) if top else (0.3, 0.7),
                                             keep="high" if top else "all", band=0.03)
        for k in ("target", "dest"):
            if self.geo[k] is not None:
                self.geo[k + "_px"] = self.cam.project(self.geo[k])
        out.update({k: None if self.geo[k] is None else np.round(self.geo[k], 3).tolist() for k in ("target", "dest")})
        return out

    def _measure(self, view):
        m = super()._measure(view)
        g, ee = self.geo, np.asarray(view.ee_pos, dtype=np.float64)
        b = self.boxes["object"]
        # the target point follows the object while it is not in the gripper (it may be pushed or dropped)
        if m["found"] and not m["held"] and b is not None:
            g["target"] = surface_point(self.cam, self.cam.depth(), b)
            g["target_px"] = self.cam.project(g["target"])
        g["ee_px"] = self.cam.project(ee)
        m["ee"] = ee
        m["d_target"] = None if g["target"] is None else xy_dist(ee, g["target"])
        m["d_dest"] = None if g["dest"] is None else xy_dist(ee, g["dest"])
        m["dz_target"] = None if g["target"] is None else float(ee[2] - g["target"][2])
        m["dz_dest"] = None if g["dest"] is None else float(ee[2] - g["dest"][2])
        # 3-D distance to where the EE has to get: the pre-grasp point / the release point
        m["a_target"] = None if g["target"] is None else float(np.linalg.norm(ee - g["target"] - [0, 0, J.APPROACH_DZ]))
        m["a_dest"] = None if g["dest"] is None else float(np.linalg.norm(ee - g["dest"] - [0, 0, J.RELEASE_DZ]))
        return m

    def _monitor(self, view, rec):
        undone = super()._monitor(view, rec)
        if undone and self.subtasks[self.idx]["kind"] == "pick":
            self.lost_event = {"step": view.step, "reason": self.subtasks[self.idx]["history"][-1]["reason"]}
        return undone

    def _rewatch(self, step, grace_s=GRACE_S):
        self.watch_from, self.samples, self.pending = step + int(grace_s * S), [], None

    # ---------------------------------------------------------------- main
    def _act(self, view):
        if self.watch_from is None:
            self._rewatch(view.step, START_GRACE_S)
        if self.motion_active:  # the loop calls the agent again only once the override has finished
            self.motion_active, self.last_correction_end = False, view.step
            self.corrections[-1]["end_step"] = view.step
            self._rewatch(view.step)
        idx0, n0 = self.idx, len(self.decisions)
        out = super()._act(view)
        rec = self.decisions[-1] if len(self.decisions) > n0 else None
        if isinstance(out, Stop) or rec is None or "error" in rec or self.plan is None or \
                self.plan["object"] is None or self.current["kind"] == "finish":
            return out
        if self.idx != idx0:
            self._rewatch(view.step)
        return self._recover(view, rec)

    def _recover(self, view, rec):
        m, g = self.meas, self.geo
        rec["signals"] = {k: None if m.get(k) is None else round(m[k], 3)
                          for k in ("d_target", "d_dest", "dz_target", "dz_dest", "a_target", "a_dest")}
        rec["ee_px"], rec["target_px"], rec["dest_px"] = g["ee_px"], g["target_px"], g["dest_px"]
        if self.lost_event is not None:
            ev, self.lost_event = self.lost_event, None
            self._rewatch(view.step)
            near_dest = m["d_dest"] is not None and m["d_dest"] < FAR_M
            if near_dest:
                return self._report(view, rec, ("low_level", "placement_instability", "LL-6", None,
                                                f"released next to the {self.plan['destination']} but not "
                                                f"{self.plan['relation']} it ({ev['reason']})"), confirmed=True)
            det = ("high_level", "grasp_loss", "HL-8", "release_and_retreat",
                   f"object lost on the way ({ev['reason']}): re-approach it")
            return self._report(view, rec, det, confirmed=True)
        if view.step < self.watch_from:
            self.pending = None
            return CONTINUE
        self.samples.append({"step": view.step, "ee": m["ee"], "a_target": m["a_target"], "a_dest": m["a_dest"]})
        return self._report(view, rec, self._detect(view, m))

    def _ago(self, view, secs):
        """Sample about `secs` ago (None until the samples since watch_from span that long)."""
        if not self.samples or view.step - self.samples[0]["step"] < secs * S:
            return None
        return [s for s in self.samples if s["step"] <= view.step - secs * S][-1]

    def _detect(self, view, m):
        """(level, mode, library id, action, reason) of the failure the latest measurements show, else None."""
        cur, p, old = self.current, self.plan, self._ago(view, WINDOW_S)
        moved = None if old is None else float(np.linalg.norm(m["ee"] - old["ee"]))
        if cur["kind"] == "pick":
            dT = m["d_target"]
            if dT is None:
                return None
            target_moving = m["move"] is not None and m["move"] > TARGET_MOVING
            if m["held"] and dT > GRASP_FAR_M and not target_moving:
                return ("high_level", "wrong_grasp", "HL-3/HL-4", "release_and_retreat",
                        f"gripper closed on something {dT * 100:.0f} cm from the {p['object']}")
            before = self._ago(view, DESCENT_S)
            descending = before is not None and before["ee"][2] - m["ee"][2] > DESCENT_M
            # height band + descending: travelling below an elevated target (task 36: a bowl at 1.12 m) is not a grasp
            if not m["held"] and BELOW_DEST_DZ < m["dz_target"] < LOW_DZ and dT > GRASP_FAR_M and \
                    (descending or m["closing"]):
                return ("high_level", "wrong_approach", "HL-2/HL-3/HL-4", "point_approach",
                        f"descended to {m['dz_target'] * 100:.0f} cm above the target height but "
                        f"{dT * 100:.0f} cm away from the {p['object']}")
            aT = m["a_target"]
            if old is not None and not m["held"] and aT > FAR_M and old["a_target"] - aT < PROGRESS_M:
                stall = moved < STALL_M
                return ("high_level", "stall" if stall else "target_not_approached", "HL-1/S1" if stall else "HL-7",
                        "point_approach",
                        f"{aT * 100:.0f} cm from the pre-grasp point over the {p['object']}, "
                        f"{(old['a_target'] - aT) * 100:+.1f} cm closer in {WINDOW_S:.0f} s, EE moved {moved * 100:.1f} cm")
            if m["closing"] and not m["held"] and dT < NEAR_M:
                return ("low_level", "missed_grasp", "LL-4/LL-5", None,
                        f"gripper closed on nothing {dT * 100:.0f} cm from the {p['object']}")
            if old is not None and not m["closing"] and aT <= FAR_M and moved < STALL_M:
                return ("low_level", "pregrasp_hesitation", "LL-8", None,
                        f"open gripper {dT * 100:.0f} cm from the {p['object']}, EE moved {moved * 100:.1f} cm "
                        f"in {WINDOW_S:.0f} s")
            return None
        dD = m["d_dest"]  # place
        if dD is None or not m["held"]:
            return None
        before = self._ago(view, DESCENT_S)
        descending = before is not None and before["ee"][2] - m["ee"][2] > DESCENT_M
        if descending and BELOW_DEST_DZ < m["dz_dest"] < RELEASE_LOW_DZ and dD > FAR_M:
            return ("high_level", "wrong_destination", "HL-6", "carry_to",
                    f"lowering the {p['object']} {dD * 100:.0f} cm away from the {p['destination']}")
        aD = m["a_dest"]
        if old is not None and aD > FAR_M and old["a_dest"] - aD < PROGRESS_M:
            stall = moved < STALL_M
            return ("high_level", "handoff_stall" if stall else "destination_not_approached",
                    "HL-1" if stall else "HL-6", "carry_to",
                    f"carrying, {aD * 100:.0f} cm from the release point over the {p['destination']}, "
                    f"{(old['a_dest'] - aD) * 100:+.1f} cm closer in {WINDOW_S:.0f} s, EE moved {moved * 100:.1f} cm")
        if old is not None and dD < NEAR_M and moved < STALL_M:
            return ("low_level", "release_stall", "LL-7", None,
                    f"holding over the {p['destination']} ({dD * 100:.0f} cm), EE moved {moved * 100:.1f} cm "
                    f"in {WINDOW_S:.0f} s")
        return None

    def _report(self, view, rec, det, confirmed=False):
        """Debounce a detected failure, log it, and correct it if it is high-level and a correction is allowed."""
        if det is None:
            self.pending = None
            return CONTINUE
        level, mode, lib, action, reason = det
        if not confirmed:
            if self.pending is None or self.pending[0] != mode:
                self.pending = (mode, view.step)
            held = (view.step - self.pending[1]) / S
            if held < CONFIRM_S:
                rec.update(failure_suspected=mode, failure_reason=reason)
                return CONTINUE
        rec.update(failure=level, failure_mode=mode, library=lib, reason=reason, action="continue")
        onset = not self.failures or self.failures[-1]["mode"] != mode or self.failures[-1]["last_step"] < \
            view.step - 2 * self.every_steps
        if onset:
            self.failures.append({"step": view.step, "last_step": view.step, "level": level, "mode": mode,
                                  "library": lib, "reason": reason, "subtask": self.current["label"], "action": None})
        self.failures[-1]["last_step"] = view.step
        if level == "low_level":
            rec["decision"] = "low-level: reported only, pi0 keeps trying"
            return CONTINUE
        since = None if self.last_correction_end is None else view.step - self.last_correction_end
        if len(self.corrections) >= self.max_corrections or (since is not None and since < self.cooldown_steps):
            rec["decision"] = "correction suppressed (cooldown / budget)"
            return CONTINUE
        target = self.geo["dest"] if action == "carry_to" else self.geo["target"]
        if target is None:
            rec["decision"] = f"{action}: no target point"
            return CONTINUE
        motion = self._motion_world(action, target, view)
        px = self.cam.project(target)
        corr = {"step": view.step, "action": action, "failure_mode": mode, "library": lib, "reason": reason,
                "subtask": self.current["label"], "target_world": np.round(target, 3).tolist(),
                "point_img": [px[0] * self.cam.size, px[1] * self.cam.size]}
        self.corrections.append(corr)
        self.failures[-1]["action"] = action
        self.motion_active, self.pending = True, None
        rec.update(action=action, target_world=corr["target_world"], decision=f"correction: {action}")
        if self.log:
            self.log.info("step %d: %s [%s] -> %s %s", view.step, mode, lib, action, corr["target_world"])
        return Override(motion, reason=f"{action}: {mode} ({lib}): {reason}")
