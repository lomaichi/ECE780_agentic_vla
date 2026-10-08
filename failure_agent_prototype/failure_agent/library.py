"""Skill library: load entries, turn a diagnosis into a concrete recovery plan,
and keep per-entry success statistics (stored separately from the YAML)."""
from __future__ import annotations

import json
import pathlib
from typing import Dict, List

import yaml

from . import language as lang
from .diagnosis import Diagnosis
from .scene import SceneState


class SkillLibrary:
    def __init__(self, path: str, stats_path: str):
        self.path = pathlib.Path(path)
        self.entries: Dict[str, dict] = yaml.safe_load(self.path.read_text())["entries"]
        self.stats_path = pathlib.Path(stats_path)
        self.stats = json.loads(self.stats_path.read_text()) if self.stats_path.exists() else {}

    def lookup(self, code: str) -> dict:
        if code not in self.entries:
            raise KeyError(f"no library entry for '{code}'")
        return self.entries[code]

    # ------------------------------------------------------------------ plans
    def build_plan(self, diag: Diagnosis, scene: SceneState, attempt: int, cfg: dict) -> List[dict]:
        """Concrete steps for this diagnosis. `attempt` = how many times this
        entry was already used in the episode (selects the prompt variant)."""
        entry = self.lookup(diag.code)
        plan: List[dict] = []
        slots = diag.slots
        for step in entry["recover"]:
            do = step["do"]
            if "when" in step and not slots.get(step["when"]):
                continue
            if do in ("release", "lift", "retract_home"):
                plan.append({"do": do})
            elif do == "finish_articulation":
                if slots.get("artic_joint"):
                    plan.append({"do": "finish_articulation", "joint": slots["artic_joint"]})
            elif do == "approach":
                if step.get("joint") == "artic":
                    if slots.get("artic_joint"):
                        plan.append({"do": "approach", "joint": slots["artic_joint"]})
                elif slots.get("target_obj"):   # skipped when the goal has no movable target
                    plan.append({"do": "approach", "obj": slots["target_obj"]})
            elif do == "carry_to":
                if slots.get("target_obj") and slots.get("dest_obj"):
                    plan.append({"do": "carry_to", "obj": slots["target_obj"], "dest": slots["dest_obj"]})
            elif do == "pi0":
                prompt = lang.choose_prompt(step["prompts"], slots, attempt)
                if prompt is None:
                    prompt = slots["instruction"]
                until = step.get("until", "goal")
                if until == "grasp_target":
                    if not slots.get("target_obj"):
                        continue
                    until = {"grasp": slots["target_obj"]}
                plan.append({"do": "pi0", "prompt": prompt, "until": until,
                             "budget": cfg["grasp_budget"] if step.get("until") == "grasp_target"
                             else step.get("budget", cfg["pi0_step_budget"])})
            elif do == "decompose":
                plan += self.decompose(scene, cfg)
            else:
                raise ValueError(f"unknown recovery step '{do}' in {diag.code}")
        return plan

    @staticmethod
    def decompose(scene: SceneState, cfg: dict) -> List[dict]:
        """BDDL goal -> ordered single-step pi0 sub-tasks: open, then placements
        (opening a destination drawer first if it is mostly closed),
        each placement preceded by an approach over the object, then close / turn on / off."""
        status = scene.eval_goal()
        pos = {o: scene.obj_pos(o) for o in scene.movable}
        opens, places, finals = [], [], []
        for g, ok in zip(scene.goal, status):
            if ok and g.name not in ("close", "turnon", "turnoff"):
                continue
            if g.is_placement:
                obj, dest = g.args
                place = lang.place_phrase(g.name, dest, scene._owner_of(dest), pos)
                dest_np = lang.region_phrase(dest, scene._owner_of(dest))
                desc = lang.spatial_descriptor(pos, obj)
                obj_np = lang.obj_phrase(obj) + (f" {desc}" if desc else "")
                frac = scene.open_fraction(dest)
                if frac is not None and frac < cfg["min_open_fraction"]:
                    opens += SkillLibrary._articulate(scene, "open", dest, ["open", dest], cfg)
                # pi0 grasps; the coordinator supplies where (approach) and where to (carry_to)
                places.append({"do": "approach", "obj": obj})
                places.append({"do": "pi0",
                               "prompt": f"pick up the {obj_np} and place it {place}",
                               "until": {"grasp": obj}, "budget": cfg["grasp_budget"]})
                places.append({"do": "carry_to", "obj": obj, "dest": dest})
            elif g.name == "open":
                opens += SkillLibrary._articulate(scene, "open", g.args[0], g.raw, cfg)
            elif g.is_articulation:
                finals += SkillLibrary._articulate(scene, g.name, g.args[0], g.raw, cfg)
        # an open-goal and a destination drawer can be the same drawer
        seen, uniq = set(), []
        for st in opens:
            key = (st["do"], st.get("prompt"), st.get("joint"))
            if key not in seen:
                seen.add(key)
                uniq.append(st)
        steps = uniq + places + finals
        # release + retract between sub-tasks so each starts from the
        # in-distribution home pose (an approach replaces the retract)
        out, prev_end = [], False
        for st in steps:
            if prev_end and st["do"] != "carry_to":
                out.append({"do": "release"})
                if st["do"] != "approach":
                    out.append({"do": "retract_home"})
            out.append(st)
            prev_end = st["do"] in ("pi0", "carry_to", "finish_articulation")
        return out

    @staticmethod
    def _articulate(scene: SceneState, verb: str, arg: str, predicate: list, cfg: dict) -> List[dict]:
        """Sub-task for one articulation: hover over the handle/knob, pi0 with the
        single-step prompt, then (drawers being closed) the scripted push."""
        joints = scene.articulation_joints(arg)
        out = []
        if joints:
            out.append({"do": "approach", "joint": joints[0]})
        out.append({"do": "pi0", "prompt": f"{lang.VERB_MAP[verb]} {lang.region_phrase(arg, scene._owner_of(arg))}",
                    "until": {"predicate": predicate}, "budget": cfg["pi0_step_budget"]})
        if verb == "close" and joints and scene.joints[joints[0]]["type"] == "slide":
            out.append({"do": "finish_articulation", "joint": joints[0]})
        return out

    # ------------------------------------------------------------------ stats
    def record(self, code: str, recovered: bool):
        s = self.stats.setdefault(code, {"attempts": 0, "recovered": 0})
        s["attempts"] += 1
        s["recovered"] += int(recovered)

    def save_stats(self):
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats_path.write_text(json.dumps(self.stats, indent=2))
