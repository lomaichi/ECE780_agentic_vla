"""Online symptom monitor.

Called once per env step. Turns ground-truth sim state into
  * events   (grasp, release, predicate change, articulation change, topple, push)
  * triggers (stall, no_progress, disturbance, out_of_view, wrist_windup,
              wrong_grasp, unplanned_articulation, premature_goal)
Triggers are what interrupt pi0 and hand control to the diagnosis agent.
"""
from __future__ import annotations

import collections
import dataclasses
from typing import Deque, Dict, List, Optional

import numpy as np

from .scene import SceneState


@dataclasses.dataclass
class Event:
    t: int
    kind: str
    data: dict

    def to_dict(self):
        return {"t": self.t, "kind": self.kind, **self.data}


def _approach_tilt_deg(q: np.ndarray, q_home: np.ndarray) -> float:
    """Angle between the gripper's approach (z) axis now and at home. Ignores
    yaw about that axis, which pi0 uses legitimately (e.g. mug handles).
    Quaternions in robosuite order (x, y, z, w)."""
    def z_axis(q):
        x, y, z, w = q / np.linalg.norm(q)
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    c = float(np.clip(z_axis(q) @ z_axis(q_home), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


class SymptomMonitor:
    def __init__(self, scene: SceneState, cfg: dict):
        self.scene = scene
        self.cfg = cfg
        self.events: List[Event] = []
        self.ever_grasped: Dict[str, int] = {}         # obj -> first confirmed grasp t
        self.min_eef_dist: Dict[str, float] = {}        # goal arg -> closest EEF approach
        self.max_goal_joint_progress: Dict[str, float] = {}
        self._start_joints = scene.joint_qpos()
        self._start_tilt = {o: scene.obj_tilt_deg(o) for o in scene.movable}
        self._start_pos = {o: scene.obj_pos(o) for o in scene.movable}
        self._goal_status = scene.eval_goal()
        self.goal_start = list(self._goal_status)
        self.min_place_dist: Dict[int, float] = {}       # placement goal index -> closest obj-dest xy
        self.artic_targets = scene.articulation_targets()
        # goal joint -> (eef offset from the moving body, eef_quat, t) the last time
        # the EEF moved it toward its target (diagnostic record of pi0's contact)
        self.push_point: Dict[str, tuple] = {}
        # joints of close/turn-on predicates whose region still has a placement into it
        self._ordered_joints: Dict[str, tuple] = {}
        for g in scene.goal:
            if g.name in ("close", "turnon"):
                for p in scene.goal:
                    if p.is_placement and p.args[1] == g.args[0]:
                        for j in scene._site_joints(g.args[0]):
                            self._ordered_joints[j] = (g, p)
        self._prev_q = dict(self._start_joints)
        self.home_quat: Optional[np.ndarray] = None   # set by the runner (EEF quat after settling)
        self.max_tilt_deg = 0.0
        self.focus: Optional[str] = None   # object a recovery grasp step is after (set by the runner)
        self.reset_phase(0)

    # ------------------------------------------------------------------ phases
    def reset_phase(self, t: int):
        """Called at the start of each pi0 phase (initial run and every recovery).
        Re-baselines windows and already-reported changes so they don't re-trigger."""
        if getattr(self, "_held", None) and self._held not in self.scene.grasped():
            self._emit(t, "release", obj=self._held, pos=self.scene.obj_pos(self._held).round(3).tolist())
        self.phase_start = t
        self._eef_win: Deque[np.ndarray] = collections.deque(maxlen=self.cfg["stall_window"])
        self._grip_win: Deque[float] = collections.deque(maxlen=self.cfg["stall_window"])
        self._last_progress_t = t
        self._out_count = 0
        self._windup_count = 0
        self._grasp_count: Dict[str, int] = collections.defaultdict(int)
        self._carry_count: Dict[str, int] = collections.defaultdict(int)
        self._held: Optional[str] = None
        self._phase_joints = self.scene.joint_qpos()
        self._phase_tilt = {o: self.scene.obj_tilt_deg(o) for o in self.scene.movable}
        self._phase_pos = {o: self.scene.obj_pos(o) for o in self.scene.movable}
        self._reported = set()
        self.triggers: List[Event] = []

    def _emit(self, t, kind, trigger=False, **data):
        ev = Event(t, kind, data)
        self.events.append(ev)
        if trigger:
            self.triggers.append(ev)
        return ev

    # -------------------------------------------------------------------- step
    def update(self, t: int, obs: dict) -> List[Event]:
        """Process one env step. Returns triggers raised at this step."""
        self.triggers = []
        sc, cfg = self.scene, self.cfg
        eef = np.array(obs["robot0_eef_pos"], dtype=float)
        grip_w = float(obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1])
        self._eef_win.append(eef)
        self._grip_win.append(grip_w)

        # --- goal predicates
        status = sc.eval_goal()
        for g, old, new in zip(sc.goal, self._goal_status, status):
            if old != new:
                self._emit(t, "predicate", index=g.index, predicate=g.raw, value=new)
                self._last_progress_t = t
        self._goal_status = status

        # --- closest approach to goal arguments (for R3 "reached but missed")
        for g in sc.goal:
            for a in g.args:
                try:
                    d = float(np.linalg.norm(sc.obj_pos(a) - eef))
                except Exception:
                    continue
                self.min_eef_dist[a] = min(self.min_eef_dist.get(a, 1e9), d)

        # --- closest the moved object got to its destination (R4 vs R6)
        for g in sc.goal:
            if g.is_placement:
                try:
                    d = float(np.linalg.norm(sc.obj_pos(g.args[0])[:2] - sc.obj_pos(g.args[1])[:2]))
                except Exception:
                    continue
                self.min_place_dist[g.index] = min(self.min_place_dist.get(g.index, 1e9), d)

        # --- grasps
        grasped = set(sc.grasped())
        for o in sc.movable:
            self._grasp_count[o] = self._grasp_count[o] + 1 if o in grasped else 0
        held = None
        for o in grasped:
            lifted = sc.obj_pos(o)[2] - self._start_pos[o][2] > cfg["lift_confirm_m"]
            if self._grasp_count[o] >= cfg["grasp_confirm_steps"] and lifted:
                held = o
        # objects held with a single pad in contact (bowls by the rim) never pass
        # the contact check: also accept being carried well off the table, next
        # to a closed gripper, for a few steps
        if held is None:
            for o in sc.movable:
                p = sc.obj_pos(o)
                carried = p[2] - self._start_pos[o][2] > cfg["carried_lift_m"] and grip_w < cfg["grip_open_w"] \
                    and float(np.linalg.norm(p - eef)) < cfg["carry_near_m"]
                self._carry_count[o] = self._carry_count[o] + 1 if carried else 0
                if self._carry_count[o] >= cfg["grasp_confirm_steps"]:
                    held = o
                    break
        # MuJoCo's pad-contact grasp check flickers (e.g. a bowl held by its rim);
        # keep a confirmed grasp while the object stays lifted next to a closed gripper
        if held is None and self._held is not None:
            o = self._held
            p = sc.obj_pos(o)
            if p[2] - self._start_pos[o][2] > cfg["lift_confirm_m"] and grip_w < cfg["grip_open_w"] \
                    and float(np.linalg.norm(p - eef)) < cfg["carry_dist_m"]:
                held = o
        if held and held != self._held:
            self.ever_grasped.setdefault(held, t)
            # judged against the targets of still-pending placements, so grabbing
            # the base bowl before the stacking step counts as a wrong grasp
            pending = [g.args[0] for g, ok in zip(sc.goal, status) if g.is_placement and not ok]
            if self.focus:                      # a recovery step asked for this object
                pending = [self.focus]
            is_target = held in pending
            self._emit(t, "grasp", obj=held, is_target=is_target)
            self._last_progress_t = t
            if sc.targets and not is_target and ("wrong_grasp", held) not in self._reported:
                self._reported.add(("wrong_grasp", held))
                self._emit(t, "wrong_grasp", trigger=True, obj=held, targets=pending)
        if self._held and held != self._held:
            self._emit(t, "release", obj=self._held, pos=sc.obj_pos(self._held).round(3).tolist())
            self._last_progress_t = t
        self._held = held

        # --- articulation
        q = sc.joint_qpos()
        for j, tgt in self.artic_targets.items():
            dq = q[j] - self._prev_q[j]
            if abs(dq) > 2e-4 and np.sign(dq) == np.sign(tgt - self._prev_q[j]):
                self.push_point[j] = (eef - sc.joint_body_pos(j), np.array(obs["robot0_eef_quat"], dtype=float), t)
        self._prev_q = q
        for j, info in sc.joints.items():
            thr = cfg["slide_change_m"] if info["type"] == "slide" else cfg["hinge_change_rad"]
            if j in sc.goal_joints:
                self.max_goal_joint_progress[j] = max(self.max_goal_joint_progress.get(j, 0.0),
                                                      abs(q[j] - self._start_joints[j]))
            if abs(q[j] - self._phase_joints[j]) > thr and ("artic", j) not in self._reported:
                self._reported.add(("artic", j))
                planned = j in sc.goal_joints
                self._emit(t, "articulation" if planned else "unplanned_articulation",
                           trigger=not planned, joint=j, owner=info["owner"], planned=planned,
                           start=round(self._phase_joints[j], 4), now=round(q[j], 4))
                self._last_progress_t = t

        # --- premature goal: a close/turn-on joint moved toward its target while
        #     the placement into that region is still pending (wrong order), or
        #     the close/turn-on predicate became true first
        for j, (g, p) in self._ordered_joints.items():
            if status[p.index] or ("premature", g.index) in self._reported:
                continue
            tgt = self.artic_targets.get(j)
            moved = q[j] - self._phase_joints[j]
            thr = cfg["slide_change_m"] if sc.joints[j]["type"] == "slide" else cfg["hinge_change_rad"]
            if status[g.index] or (tgt is not None and abs(moved) > thr
                                   and np.sign(moved) == np.sign(tgt - self._phase_joints[j])):
                self._reported.add(("premature", g.index))
                self._emit(t, "premature_goal", trigger=True, satisfied=g.raw, pending=p.raw,
                           joint=j, moved=round(float(moved), 4))

        # --- disturbance: topple or push of objects that were never grasped
        for o in sc.movable:
            if o in self.ever_grasped or o == self._held:
                continue
            dtilt = abs(sc.obj_tilt_deg(o) - self._phase_tilt[o])
            if dtilt > cfg["topple_deg"] and ("topple", o) not in self._reported:
                self._reported.add(("topple", o))
                self._emit(t, "disturbance", trigger=True, obj=o, how="topple",
                           tilt_change=round(dtilt, 1), is_target=o in sc.targets)
            disp = float(np.linalg.norm(sc.obj_pos(o)[:2] - self._phase_pos[o][:2]))
            if disp > cfg["push_dist_m"] and ("push", o) not in self._reported and o not in sc.targets:
                self._reported.add(("push", o))
                self._emit(t, "disturbance", trigger=True, obj=o, how="push",
                           dist=round(disp, 3), is_target=False)

        # --- out of view
        if sc.in_view(eef, cfg["view_margin_px"]):
            self._out_count = 0
        else:
            self._out_count += 1
            if self._out_count == cfg["out_of_view_steps"]:
                self._emit(t, "out_of_view", trigger=True, eef=eef.round(3).tolist())

        # --- wrist wind-up: gripper tilted far from the top-down home pose (demos
        #     stay roughly top-down; once contorted, pi0 stops acting purposefully)
        if self.home_quat is not None:
            dq = _approach_tilt_deg(np.array(obs["robot0_eef_quat"], dtype=float), self.home_quat)
            self.max_tilt_deg = max(self.max_tilt_deg, dq)
            self._windup_count = self._windup_count + 1 if dq > cfg["windup_deg"] else 0
            if self._windup_count == cfg["windup_steps"]:
                self._emit(t, "wrist_windup", trigger=True, angle_deg=round(dq, 1))

        # --- no progress: no grasp/release/predicate/articulation event for a long
        #     window although the arm keeps moving (dithering that the stall box misses)
        acting = t - self.phase_start
        if acting >= cfg["progress_window"] and t - self._last_progress_t >= cfg["progress_window"] \
                and "no_progress" not in self._reported and "stall" not in self._reported:
            self._reported.add("no_progress")
            self._emit(t, "no_progress", trigger=True, since=self._last_progress_t,
                       eef=eef.round(3).tolist(), held=self._held)

        # --- stall
        if acting >= cfg["stall_grace"] and len(self._eef_win) == self._eef_win.maxlen:
            pts = np.stack(self._eef_win)
            extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
            grip_change = max(self._grip_win) - min(self._grip_win)
            no_progress = t - self._last_progress_t >= cfg["stall_window"]
            if extent < cfg["stall_extent_m"] and grip_change < 0.01 and no_progress \
                    and "stall" not in self._reported:
                self._reported.add("stall")
                self._emit(t, "stall", trigger=True, extent_m=round(extent, 4),
                           eef=eef.round(3).tolist(), held=self._held)

        return list(self.triggers)

    # ---------------------------------------------------------------- summary
    def snapshot(self) -> dict:
        sc = self.scene
        return {
            "goal_status": self._goal_status,
            "goal_start": self.goal_start,
            "max_tilt_deg": round(self.max_tilt_deg, 1),
            "min_place_dist": {k: round(v, 3) for k, v in self.min_place_dist.items()},
            "held": self._held,
            "ever_grasped": dict(self.ever_grasped),
            "min_eef_dist": {k: round(v, 3) for k, v in self.min_eef_dist.items()},
            "goal_joint_progress": {k: round(v, 4) for k, v in self.max_goal_joint_progress.items()},
            "obj_disp": {o: round(float(np.linalg.norm(sc.obj_pos(o)[:2] - self._start_pos[o][:2])), 3)
                         for o in sc.movable},
        }
