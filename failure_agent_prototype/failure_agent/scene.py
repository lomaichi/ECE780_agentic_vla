"""Ground-truth scene access for a LIBERO env.

Everything the monitor and the diagnosis agent need is read from the MuJoCo
state and the task's BDDL goal: which objects the goal is about, which joints
the goal involves, what is grasped, object poses, and where the EEF projects
in the agentview camera.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np

UNARY_ARTICULATION = {"open", "close", "turnon", "turnoff"}
BINARY_PLACEMENT = {"in", "on"}


@dataclasses.dataclass
class GoalPredicate:
    index: int
    name: str            # lower-case predicate name, e.g. "in", "close"
    args: List[str]      # BDDL argument names
    raw: list            # as stored in parsed_problem["goal_state"]

    @property
    def is_placement(self) -> bool:
        return self.name in BINARY_PLACEMENT

    @property
    def is_articulation(self) -> bool:
        return self.name in UNARY_ARTICULATION


class SceneState:
    def __init__(self, env, camera: str = "agentview"):
        self.env = env            # OffScreenRenderEnv
        self.e = env.env          # LIBERO problem instance
        self.sim = self.e.sim
        self.camera = camera
        self.img_h = self.img_w = int(env.env.camera_heights[0]) if hasattr(env.env, "camera_heights") else 256

        pp = self.e.parsed_problem
        self.instruction: str = pp["language_instruction"] if isinstance(pp["language_instruction"], str) \
            else " ".join(pp["language_instruction"])
        self.goal: List[GoalPredicate] = [
            GoalPredicate(i, s[0].lower(), list(s[1:]), s) for i, s in enumerate(pp["goal_state"])
        ]
        self.movable = list(self.e.objects_dict.keys())
        self.fixtures = list(self.e.fixtures_dict.keys())

        # Movable objects the goal asks to move (first arg of in/on)
        self.targets: List[str] = [g.args[0] for g in self.goal
                                   if g.is_placement and g.args[0] in self.e.objects_dict]
        # Destinations of placement predicates (site or object names)
        self.destinations: List[str] = [g.args[1] for g in self.goal if g.is_placement]

        self.joints: Dict[str, dict] = self._collect_articulated_joints()
        self.goal_joints: List[str] = self._goal_joints()

    # ------------------------------------------------------------------ joints
    def _collect_articulated_joints(self) -> Dict[str, dict]:
        """All hinge/slide joints on fixtures and objects (drawers, knobs, doors)."""
        out = {}
        model = self.sim.model
        for owner, d in list(self.e.fixtures_dict.items()) + list(self.e.objects_dict.items()):
            for j in getattr(d, "joints", []) or []:
                try:
                    jid = model.joint_name2id(j)
                except Exception:
                    continue
                jtype = int(model.jnt_type[jid])  # 0 free, 1 ball, 2 slide, 3 hinge
                if jtype not in (2, 3):
                    continue
                out[j] = {"owner": owner, "type": "slide" if jtype == 2 else "hinge",
                          "qpos_addr": model.get_joint_qpos_addr(j)}
        return out

    def _site_joints(self, name: str) -> List[str]:
        site = self.e.object_sites_dict.get(name)
        if site is not None and getattr(site, "joints", None):
            return [j for j in site.joints if j in self.joints]
        return []

    def _owner_of(self, name: str) -> str:
        """Parent object of a site name (e.g. wooden_cabinet_1_top_region -> wooden_cabinet_1)."""
        st = self.e.object_states_dict.get(name)
        if st is not None and getattr(st, "object_state_type", "") == "site" and st.parent_name:
            return st.parent_name
        return name     # objects, and table regions (which have no parent)

    def _goal_joints(self) -> List[str]:
        """Joints the goal legitimately involves: articulation predicates, plus the
        drawer of any placement destination that is inside an articulated fixture."""
        js = []
        for g in self.goal:
            for a in g.args:
                sj = self._site_joints(a)
                if sj:
                    js += sj
                elif g.is_articulation:
                    owner = self._owner_of(a)
                    js += [j for j, info in self.joints.items() if info["owner"] == owner]
        return sorted(set(js))

    def articulation_joints(self, name: str) -> List[str]:
        """Goal joints that an articulation argument (drawer region, stove, microwave) moves."""
        return [j for j in self.goal_joints if j in self._site_joints(name)] or \
               [j for j in self.goal_joints if self.joints[j]["owner"] == self._owner_of(name)]

    def handle_point(self, joint: str) -> np.ndarray:
        """Where to engage an articulated part: for a slide joint (drawer) the
        outermost collision geom along the axis (the handle); for a hinge (knob,
        microwave door) the collision geom farthest from the hinge axis."""
        m, d = self.sim.model, self.sim.data
        jid = m.joint_name2id(joint)
        body = m.jnt_bodyid[jid]
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_contype[g]] or \
                [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
        if not geoms:
            return np.array(d.xpos[body], dtype=float)
        if self.joints[joint]["type"] == "slide":
            axis = self.joint_axis_world(joint)
            owner_body = self.e.get_object(self.joints[joint]["owner"]).root_body
            center = d.body_xpos[m.body_name2id(owner_body)]
            # the handle is on the face away from the fixture's centre
            sign = np.sign(float((d.xpos[body] - center) @ axis)) or 1.0
            g = max(geoms, key=lambda g: float((d.geom_xpos[g] - d.xpos[body]) @ (sign * axis)))
        else:
            anchor, ax = d.xanchor[jid], d.xaxis[jid]

            def r(g):
                v = d.geom_xpos[g] - anchor
                return float(np.linalg.norm(v - (v @ ax) * ax))
            g = max(geoms, key=r)
        return np.array(d.geom_xpos[g], dtype=float)

    def articulation_targets(self) -> Dict[str, float]:
        """Goal joint -> qpos inside the range its articulation predicate needs
        (midpoint of the object's default_<open|close|turnon|turnoff>_ranges)."""
        out = {}
        for g in self.goal:
            if not g.is_articulation:
                continue
            a = g.args[0]
            owner = self._owner_of(a)
            props = getattr(self.e.get_object(owner), "object_properties", {}).get("articulation", {})
            rng = props.get(f"default_{g.name}_ranges")
            if not rng:
                continue
            joints = self._site_joints(a) or [j for j, i in self.joints.items() if i["owner"] == owner]
            for j in joints:
                out[j] = float(np.mean(rng))
        return out

    def joint_axis_world(self, joint: str) -> np.ndarray:
        """Direction a slide joint's body moves (world frame) as qpos increases."""
        m = self.sim.model
        jid = m.joint_name2id(joint)
        body = m.jnt_bodyid[jid]
        return self.sim.data.xmat[body].reshape(3, 3) @ m.jnt_axis[jid]

    def joint_body_pos(self, joint: str) -> np.ndarray:
        """World position of the body a joint moves (e.g. the drawer)."""
        m = self.sim.model
        return np.array(self.sim.data.xpos[m.jnt_bodyid[m.joint_name2id(joint)]], dtype=float)

    def slide_push_point(self, joint: str, direction: np.ndarray) -> np.ndarray:
        """Outermost collision geom of the sliding body against `direction` (the
        handle, for LIBERO drawers): pushing there along `direction` closes it."""
        m, d = self.sim.model, self.sim.data
        body = m.jnt_bodyid[m.joint_name2id(joint)]
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_contype[g]]
        g = min(geoms, key=lambda g: float((d.geom_xpos[g] - d.xpos[body]) @ direction))
        return np.array(d.geom_xpos[g], dtype=float)

    def joint_qpos(self) -> Dict[str, float]:
        return {j: float(self.sim.data.qpos[info["qpos_addr"]]) for j, info in self.joints.items()}

    # --------------------------------------------------------------- predicates
    def eval_goal(self) -> List[bool]:
        out = []
        for g in self.goal:
            try:
                out.append(bool(self.e._eval_predicate(g.raw)))
            except Exception:
                out.append(False)
        return out

    def eval_predicate(self, raw) -> bool:
        try:
            return bool(self.e._eval_predicate(raw))
        except Exception:
            return False

    def is_closed(self, name: str) -> Optional[bool]:
        st = self.e.object_states_dict.get(name)
        if st is None or not hasattr(st, "is_close"):
            return None
        try:
            return bool(st.is_close())
        except Exception:
            return None

    def is_open(self, name: str) -> Optional[bool]:
        st = self.e.object_states_dict.get(name)
        if st is None or not hasattr(st, "is_open"):
            return None
        try:
            return bool(st.is_open())
        except Exception:
            return None

    def open_fraction(self, region: str) -> Optional[float]:
        """0 = closed, 1 = open, for a drawer region (from the object's ranges)."""
        js = self._site_joints(region)
        if not js:
            return None
        props = getattr(self.e.get_object(self._owner_of(region)), "object_properties", {}).get("articulation", {})
        if not props.get("default_open_ranges") or not props.get("default_close_ranges"):
            return None
        q_open, q_close = float(np.mean(props["default_open_ranges"])), float(np.mean(props["default_close_ranges"]))
        q = self.joint_qpos()[js[0]]
        return float(np.clip((q - q_close) / (q_open - q_close), 0.0, 1.0))

    def sibling_regions(self, region: str) -> List[str]:
        """Other contain regions of the same parent (e.g. caddy compartments)."""
        owner = self._owner_of(region)
        return [s for s in self.e.object_sites_dict
                if s != region and s.startswith(owner) and "region" in s]

    def contained_in(self, obj: str, region: str) -> bool:
        return self.eval_predicate(["in", obj, region])

    # ----------------------------------------------------------- objects / eef
    def obj_pos(self, name: str) -> np.ndarray:
        st = self.e.object_states_dict[name]
        return np.array(st.get_geom_state()["pos"], dtype=float)

    def obj_tilt_deg(self, name: str) -> float:
        """Angle between the object's z axis and world z."""
        q = self.e.object_states_dict[name].get_geom_state()["quat"]  # MuJoCo order w,x,y,z
        w, x, y, z = [float(v) for v in q]
        r22 = 1.0 - 2.0 * (x * x + y * y)
        return float(np.degrees(np.arccos(np.clip(r22, -1.0, 1.0))))

    def grasped(self) -> List[str]:
        gripper = self.e.robots[0].gripper
        out = []
        for name, obj in self.e.objects_dict.items():
            try:
                if self.e._check_grasp(gripper=gripper, object_geoms=obj):
                    out.append(name)
            except Exception:
                pass
        return out

    def project(self, p: np.ndarray) -> np.ndarray:
        """World point -> (u=col, v=row) pixel in the raw agentview render (before
        LIBERO's 180-degree rotation). Pinhole model from MuJoCo camera params;
        same math as robosuite.utils.camera_utils, which needs h5py."""
        model, data = self.sim.model, self.sim.data
        cid = model.camera_name2id(self.camera)
        f = 0.5 * self.img_h / np.tan(np.radians(model.cam_fovy[cid]) / 2.0)
        rot = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])  # MuJoCo cam looks down -z
        pc = rot.T @ (np.asarray(p, dtype=float) - data.cam_xpos[cid])
        x, y = f * pc[0] / pc[2] + self.img_w / 2.0, f * pc[1] / pc[2] + self.img_h / 2.0
        # MuJoCo renders bottom-up; flip rows to match obs["agentview_image"]
        return np.array([x, self.img_h - 1 - y])

    def in_view(self, p: np.ndarray, margin: int) -> bool:
        u, v = self.project(p)
        return margin <= u < self.img_w - margin and margin <= v < self.img_h - margin
