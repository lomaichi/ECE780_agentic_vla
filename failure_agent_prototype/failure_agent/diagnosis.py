"""Rule-based diagnosis agent.

Maps (trigger, monitor trace, scene state) onto the fixed taxonomy:
root causes R1-R4 first, symptom-only S1-S3 as the fallback. Every
diagnosis carries human-readable evidence so a VLM coordinator (or a person)
can audit it, and the slots the recovery prompts need.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np

from . import language as lang
from .monitor import Event, SymptomMonitor
from .scene import SceneState

SYMPTOM_ENTRY = {
    "stall": "S1_stall",
    "no_progress": "S1_stall",
    "timeout": "S1_stall",
    "wrist_windup": "S4_wrist_windup",
    "disturbance": "S2_disturbance",
    "out_of_view": "S3_out_of_view",
}


@dataclasses.dataclass
class Diagnosis:
    code: str                 # library entry id, e.g. "R1_wrong_object"
    trigger: str              # what interrupted pi0 (stall, wrong_grasp, timeout, ...)
    t: int
    evidence: List[str]
    slots: Dict[str, str]

    def to_dict(self):
        return dataclasses.asdict(self)


class DiagnosisAgent:
    def __init__(self, scene: SceneState, cfg: dict):
        self.scene = scene
        self.cfg = cfg

    # ------------------------------------------------------------------ slots
    def base_slots(self) -> Dict[str, str]:
        """Prompt slots derived from the goal: first unsatisfied placement and
        first unsatisfied articulation predicate."""
        sc = self.scene
        status = sc.eval_goal()
        slots = {"instruction": sc.instruction}
        pos = {o: sc.obj_pos(o) for o in sc.movable}
        for g, ok in zip(sc.goal, status):
            if g.is_placement and not ok and "target" not in slots:
                obj, dest = g.args
                slots["target_obj"] = obj   # BDDL name, used by the approach primitive
                slots["target"] = lang.obj_phrase(obj)
                desc = lang.spatial_descriptor(pos, obj)
                slots["target_desc"] = f"{slots['target']} {desc}" if desc else slots["target"]
                slots["dest_obj"] = dest   # BDDL name, used by carry_to
                slots["dest"] = lang.region_phrase(dest, sc._owner_of(dest))
                slots["place"] = lang.place_phrase(g.name, dest, sc._owner_of(dest), pos)
            if g.is_articulation and not ok and "verb" not in slots:
                a = g.args[0]
                slots["verb"] = lang.VERB_MAP[g.name]
                slots["region"] = lang.region_phrase(a, sc._owner_of(a))
                js = sc.articulation_joints(a)
                if js:
                    slots["artic_joint"] = js[0]
        return slots

    # -------------------------------------------------------------- diagnose
    def diagnose(self, trigger: str, t: int, mon: SymptomMonitor) -> Diagnosis:
        sc = self.scene
        snap = mon.snapshot()
        slots = self.base_slots()
        recent = [e for e in mon.events if e.t >= mon.phase_start]
        ev: List[str] = []

        # R1 wrong object: grasped / carried a movable object outside the target set
        pending = [g.args[0] for g, ok in zip(sc.goal, snap["goal_status"]) if g.is_placement and not ok]
        if sc.targets:
            wrong = [e.data["obj"] for e in recent if e.kind == "wrong_grasp"]
            if not wrong and snap["held"] and snap["held"] not in pending:
                wrong = [snap["held"]]
            if not wrong:
                wrong = [o for o in snap["ever_grasped"]
                         if o not in sc.targets and snap["obj_disp"][o] > self.cfg["push_dist_m"]
                         and not any(tg in snap["ever_grasped"] for tg in sc.targets)]
            if wrong:
                o = wrong[0]
                slots["wrong_obj"] = lang.obj_phrase(o)
                ev.append(f"grasped '{o}' but the pending goal targets are {pending}")
                ev.append(f"'{o}' displaced {snap['obj_disp'][o]:.3f} m from its start")
                return Diagnosis("R1_wrong_object", trigger, t, ev, slots)

        # R2 scene habit: unplanned articulation, or a later sub-step done first
        habit = [e for e in recent if e.kind in ("unplanned_articulation", "premature_goal")]
        if habit:
            e = habit[0]
            if e.kind == "unplanned_articulation":
                ev.append(f"joint '{e.data['joint']}' of '{e.data['owner']}' moved "
                          f"{e.data['start']} -> {e.data['now']}; it is not part of the goal")
            else:
                ev.append(f"worked on {e.data['satisfied']} (joint moved {e.data['moved']}) "
                          f"while {e.data['pending']} is still false: sub-steps out of order")
            return Diagnosis("R2_scene_habit", trigger, t, ev, slots)

        status = snap["goal_status"]

        # R3 stopped short on an articulation predicate
        for g, ok in zip(sc.goal, status):
            if not g.is_articulation or ok:
                continue
            a = g.args[0]
            joints = sc.articulation_joints(a)
            prog = max([snap["goal_joint_progress"].get(j, 0.0) for j in joints] or [0.0])
            near = min(snap["min_eef_dist"].get(a, 9.0), snap["min_eef_dist"].get(sc._owner_of(a), 9.0))
            if prog > 0.005 or near < self.cfg["reach_dist_m"]:
                if joints:
                    slots["artic_joint"] = joints[0]
                slots["verb"] = lang.VERB_MAP[g.name]
                slots["region"] = lang.region_phrase(a, sc._owner_of(a))
                ev.append(f"goal {g.raw} unsatisfied; goal joint moved {prog:.4f}, "
                          f"closest EEF approach {near:.3f} m")
                return Diagnosis("R3_stopped_short", trigger, t, ev, slots)

        # R4 placement: right object brought to the destination, imprecise placement
        for g, ok in zip(sc.goal, status):
            if not g.is_placement or ok:
                continue
            obj, dest = g.args
            if obj not in snap["ever_grasped"]:
                continue
            d = float(np.linalg.norm(sc.obj_pos(obj)[:2] - sc.obj_pos(dest)[:2]))
            sib = [s for s in sc.sibling_regions(dest) if sc.contained_in(obj, s)]
            if sib or d < self.cfg["place_near_m"]:
                ev.append(f"'{obj}' was grasped; {d:.3f} m from '{dest}' (xy) but {g.raw} is false")
                if sib:
                    ev.append(f"'{obj}' is inside sibling region(s) {sib}")
                return Diagnosis("R4_placement", trigger, t, ev, slots)

        # R7 partial compound task: some sub-goals achieved, pi0 stopped before the rest
        achieved = [ok and not was for ok, was in zip(status, snap["goal_start"])]
        if len(sc.goal) > 1 and any(achieved) and not all(status):
            done = [g.raw for g, a in zip(sc.goal, achieved) if a]
            todo = [g.raw for g, ok in zip(sc.goal, status) if not ok]
            ev.append(f"sub-goals {done} hold but {todo} were never completed")
            return Diagnosis("R7_partial_task", trigger, t, ev, slots)

        # R6 destination not reached: right object grasped, carried elsewhere or dropped
        for g, ok in zip(sc.goal, status):
            if not g.is_placement or ok or g.args[0] not in snap["ever_grasped"]:
                continue
            obj, dest = g.args
            best = snap["min_place_dist"].get(g.index, 9.0)
            held = snap["held"] == obj
            slots["target_held"] = "yes" if held else ""
            ev.append(f"'{obj}' was grasped but is not at '{dest}' (closest it got: {best:.3f} m); "
                      f"now {'held' if held else 'dropped'} "
                      f"{float(np.linalg.norm(sc.obj_pos(obj)[:2] - sc.obj_pos(dest)[:2])):.3f} m away")
            return Diagnosis("R6_destination_missed", trigger, t, ev, slots)

        # R5 target not reached: the goal object / fixture was never grasped or engaged
        for g, ok in zip(sc.goal, status):
            if ok:
                continue
            a = g.args[0]
            near = snap["min_eef_dist"].get(a, 9.0)
            if g.is_placement and a in sc.movable:
                slots["focus_obj"] = "yes"
                kind = "never approached" if near > self.cfg["reach_dist_m"] else "approached but never grasped"
                ev.append(f"target '{a}' {kind} (closest EEF approach {near:.3f} m); goal {g.raw}")
                return Diagnosis("R5_target_not_reached", trigger, t, ev, slots)
            if g.is_articulation:
                slots["focus_fixture"] = "yes"
                slots["verb"] = lang.VERB_MAP[g.name]
                slots["region"] = lang.region_phrase(a, sc._owner_of(a))
                js = sc.articulation_joints(a)
                if js:
                    slots["artic_joint"] = js[0]
                ev.append(f"goal fixture '{a}' never engaged (closest EEF approach {near:.3f} m, "
                          f"joint unmoved); goal {g.raw}")
                return Diagnosis("R5_target_not_reached", trigger, t, ev, slots)

        # Symptom-only fallback
        code = SYMPTOM_ENTRY.get(trigger, "S1_stall")
        last = [e for e in recent if e.kind == trigger]
        if last:
            ev.append(f"{trigger}: {last[-1].data}")
        ev.append("no root cause matched; using the symptom's generic recovery")
        return Diagnosis(code, trigger, t, ev, slots)
