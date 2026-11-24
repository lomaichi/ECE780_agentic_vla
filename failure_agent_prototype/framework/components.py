"""The pipeline's components, built from the existing code (nothing here re-implements a trained model's input):

  load_modules()   imports pixelQ V1 (Q-map), V2 (heightmap backend V4 uses), V4 (V net) and the Moondream2
                   grounded agent. V1 and V2 both have a top-level `qnet` module, so V1 is imported in
                   isolation (its qnet is kept as sys.modules["pixelq_v1_qnet"]).
  SkillLibrary     failure_survey/skill_library/library.json: entry lookup for corrections, rl/pick coverage
  QModel           V1 Q-map checkpoint: masked argmax over (rotation bin, cell)
  VModel           V4 V(s) checkpoint (vnet.load_ckpt); ConstantV is a stand-in until V is trained
  VLMMonitor       wraps moondreamV2/recovery_agent.GroundedRecoveryAgent (subtask tracking, HL failure
                   detection + correction, grounded target point / box) behind the supervisor's interface
  Perception       close_to_object() gate + V observation (V2 BranchingPickEnv.gate / observe, the exact
                   code that produced V4's training samples), Q_INITIAL_POSITION + Q observation + Q move
                   (V1 PickCorrectorEnv geometry, the exact code Q was trained with)

Privileged stand-ins (as in Q and V training; see README.md): the heightmap drops robot pixels with the sim
segmentation; the target's sim identity (its geoms) decides whether it is "grounded" in agentview. The target
MASK is the sim segmentation (target_mask="seg", how Q and V were trained) or the VLM's detect box
(target_mask="vlm": V2's box mask for V; a box-derived depth mask for Q_INITIAL_POSITION and the Q crop).

Python 3.8 compatible. LIBERO / robosuite / torch are imported when a component is built, not at import.
"""

import json
import re
import sys

import numpy as np

import config as C

C.ensure_paths()
from fa_core import camera, common, motion  # noqa: E402

_MODS = {}


# ------------------------------------------------------------------ module loading -------------------

def load_modules():
    """Import the pixel-Q and Moondream code once. Returns a dict of modules."""
    if _MODS:
        return _MODS
    for d in (C.V2_DIR, C.V4_DIR, C.MOONDREAM_DIR):
        if str(d) not in sys.path:
            sys.path.append(str(d))       # appended: never shadows fa_core / shared / this folder
    # V2 first: its `qnet` (heightmap constants) is the one sys.modules["qnet"] keeps
    import heightmap as v2_heightmap  # noqa: F401
    import libero_env as v2_env
    import qnet as v2_qnet
    import vnet as v4_vnet
    from shared import scene as SC

    if v2_qnet.__file__ is None or pathlib_parent(v2_qnet.__file__) != C.V2_DIR:
        raise ImportError("sys.modules['qnet'] is not V2's qnet ({}); import order broken".format(v2_qnet.__file__))
    # V1 in isolation: its corrector_env does `from qnet import ...` and must get V1's qnet
    saved = sys.modules.pop("qnet")
    sys.path.insert(0, str(C.V1_DIR))
    try:
        import corrector_env as v1_env
        v1_qnet = sys.modules["qnet"]
    finally:
        sys.path.remove(str(C.V1_DIR))
        sys.modules["qnet"] = saved
    sys.modules["pixelq_v1_qnet"] = v1_qnet
    if pathlib_parent(v1_qnet.__file__) != C.V1_DIR:
        raise ImportError("V1 corrector_env got the wrong qnet: {}".format(v1_qnet.__file__))
    _MODS.update(v1_env=v1_env, v1_qnet=v1_qnet, v2_env=v2_env, v2_qnet=v2_qnet, v4_vnet=v4_vnet, scene=SC)
    return _MODS


def pathlib_parent(path):
    import pathlib

    return pathlib.Path(path).resolve().parent


def load_moondream():
    """recovery_agent + vlm_backends from Documentation/VLM_inference_speed_test/moondreamV2."""
    load_modules()
    import recovery_agent
    import vlm_backends

    return recovery_agent, vlm_backends


# ------------------------------------------------------------------ skill library --------------------

class SkillLibrary(object):
    """library.json (failure survey): which entry a correction answers, and what rl/pick covers."""

    def __init__(self, path=C.LIBRARY):
        self.path = str(path)
        self.data = json.loads(open(self.path).read())
        self.entries = {e["id"]: e for e in self.data["entries"]}
        self.version = self.data.get("version")

    def lookup(self, ids):
        """'HL-3/HL-4' (recovery_agent's library field) -> [{id, name, level, remedy}]."""
        out = []
        for i in re.split(r"[/,\s]+", str(ids or "")):
            e = self.entries.get(i)
            if e is not None:
                out.append({"id": i, "name": e["name"], "level": e["level"],
                            "remedy": e.get("remedy_brief") or _first_step(e)})
        return out

    def rl_policy(self, policy_id):
        """INVOKE_RL(policy_id): the low-level entries it covers (rl/pick: LL-4, LL-5, LL-8)."""
        p = self.data.get("rl_policies", {}).get(policy_id, {})
        covers = p.get("covers", [])
        return {"invoke_rl": policy_id, "implemented_by": "pixelQ V1 Q-map (V4 V as the trigger)",
                "covers": [{"id": i, "name": self.entries[i]["name"]} for i in covers if i in self.entries]}

    def held_out_tasks(self):
        return set(self.data["split_guard"]["held_out_tasks"])


def _first_step(e):
    r = e.get("recovery") or []
    return r[0].get("do") if r and isinstance(r[0], dict) else None


# ------------------------------------------------------------------ models ---------------------------

class QModel(object):
    """pixelQ V1 Q-map (9 rotation bins x 10 x 10 cells of the wrist crop at the handoff pose)."""

    def __init__(self, ckpt=C.DEFAULT_Q_CKPT, device="cpu"):
        import torch

        m = load_modules()
        self.q = m["v1_qnet"]
        ck = torch.load(str(ckpt), map_location=device)
        self.width = int(ck.get("state", {}).get("width", 32))
        self.net = self.q.build_qnet(self.width).to(device)
        self.net.load_state_dict(ck["net"])
        self.net.eval()
        self.device, self.ckpt = device, str(ckpt)
        self.train_cfg = _read_json(pathlib_parent(ckpt) / "config.json")

    def select(self, obs):
        """Greedy (masked argmax) action of a V1 observation. Returns (flat action, info)."""
        qm = self.q.q_map(self.net, obs, self.device)
        a = self.q.masked_argmax(qm, obs["valid"])
        k, row, col = self.q.decode_action(a)
        return a, {"action": int(a), "k": int(k), "row": int(row), "col": int(col),
                   "yaw_deg": float(self.q.ROT_DEG[k]), "q": float(qm[k, row, col]),
                   "q_max_per_rot": [float(x) for x in qm.reshape(len(self.q.ROT_DEG), -1).max(axis=1)],
                   "qmap": np.round(qm, 3).tolist()}


class VModel(object):
    """pixelQ V4 V(s): ensemble mean and spread of the discounted chance pi0 picks from here."""

    def __init__(self, ckpt=C.DEFAULT_V_CKPT, device="cpu"):
        m = load_modules()
        self.vnet = m["v4_vnet"]
        self.net, self.meta = self.vnet.load_ckpt(str(ckpt), device)
        self.device, self.ckpt = device, str(ckpt)

    def predict(self, obs):
        return self.vnet.predict(self.net, obs, self.device)


class ConstantV(object):
    """Stand-in V until V4 is trained: always `value` (value < THRESHOLD triggers Q on every gated streak)."""

    def __init__(self, value):
        self.value, self.ckpt, self.meta = float(value), "constant:{}".format(value), {"arch": "constant"}

    def predict(self, obs):
        return self.value, 0.0


def _read_json(path):
    try:
        return json.loads(open(str(path)).read())
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------ VLM ------------------------------

class VLMMonitor(object):
    """GroundedRecoveryAgent (Moondream2 detect + depth + proprio) behind the supervisor's interface.

    It does the pseudocode's two VLM jobs: failure_detection (subtask success judge, HL failure rules mapped to
    library ids, HL corrections) and estimate_object_location (the tracked detect box of the target and its
    back-projected surface point, refreshed while the target is not in the gripper)."""

    def __init__(self, agent, every_steps):
        self.agent = agent
        agent.every_steps = every_steps          # the agent's onset bookkeeping uses its own call period
        self.rec = None

    def reset(self, task_language, prompt):
        self.agent.reset(task_language, prompt)
        self.rec = None

    def act(self, view):
        a, n0 = self.agent, len(self.agent.decisions)
        d = a.act(view, None)
        self.rec = a.decisions[-1] if len(a.decisions) > n0 else None
        return d

    def subtask(self):
        a = self.agent
        if a.plan is None or not a.subtasks:
            return None, None, None
        return a.idx, a.current["kind"], a.plan.get("object")

    def hl_failure(self):
        return self.rec is not None and self.rec.get("failure") == "high_level"

    def failure(self):
        r = self.rec or {}
        if not r.get("failure") or r.get("failure") == "none":
            return None
        return {k: r.get(k) for k in ("failure", "failure_mode", "library", "reason", "action", "target_world")}

    def held(self):
        return bool((getattr(self.agent, "meas", None) or {}).get("held"))

    def box(self):
        return (getattr(self.agent, "boxes", None) or {}).get("object")

    def target_point(self):
        return (getattr(self.agent, "geo", None) or {}).get("target")

    def gate_inputs(self):
        """What V2's gate() reads from an agent (= hl.MoondreamHL.gate_inputs)."""
        a = self.agent
        if a.plan is None or a.plan.get("object") is None or not a.subtasks:
            return {"pick": False, "target": None, "held": False, "box": None}
        return {"pick": a.current["kind"] == "pick", "target": self.target_point(), "held": self.held(),
                "box": self.box()}

    def external_correction_done(self, step):
        """A Q correction (not the agent's own) just ended: restart its failure windows and its cooldown, as after
        one of its own corrections."""
        a = self.agent
        if hasattr(a, "_rewatch"):
            a._rewatch(step)
        a.last_correction_end = step

    def discard_correction(self):
        """The supervisor dropped the correction the agent returned at this call (timeout branch first)."""
        a = self.agent
        if not getattr(a, "motion_active", False):
            return None
        a.motion_active = False
        c = a.corrections.pop() if a.corrections else None
        if getattr(a, "failures", None):
            a.failures[-1]["action"] = "preempted by timeout Q"
        if self.rec is not None:
            self.rec["decision"] = "correction preempted by the timeout Q correction"
        return c

    def summary(self):
        a, r = self.agent, self.rec or {}
        out = {"subtask": a.current["label"] if a.subtasks else None}
        for k in ("observation", "failure", "failure_mode", "library", "failure_suspected", "monitor", "error"):
            if r.get(k) not in (None, "", "none"):
                out["vlm_" + k] = r[k]
        return out

    def stats(self, calls0=0, latency0=0.0):
        """Per-episode VLM numbers (the backend's counters run over the whole process: pass their start values)."""
        v = self.agent.vlm
        return {"vlm_calls": getattr(v, "calls", 0) - calls0,
                "vlm_latency_s": round(getattr(v, "latency_s", 0.0) - latency0, 2),
                "plan": {k: self.agent.plan.get(k) for k in ("object", "destination", "relation")}
                if self.agent.plan else None,
                "subtasks": [dict(s) for s in self.agent.subtasks],
                "hl_failures": list(getattr(self.agent, "failures", [])),
                "hl_corrections": list(getattr(self.agent, "corrections", []))}


def make_vlm_backend(backend, model=None, log=None):
    """'moondream2': vlm_backends.MoondreamLocal (local GPU; starts moondream_server.py if none answers on port
    8100), created once per run. 'oracle': None here; the runner builds a stubs.OracleDetectVLM per episode."""
    if backend == "oracle":
        return None
    _, vlm_backends = load_moondream()
    return vlm_backends.make_vlm(backend, model, log=log)


def make_vlm_monitor(vlm, env, every_steps, log=None):
    """A fresh GroundedRecoveryAgent for one episode (its camera wraps this episode's env)."""
    recovery_agent, _ = load_moondream()
    agent = recovery_agent.GroundedRecoveryAgent(vlm, recovery_agent.GroundedCamera(env), log=log)
    return VLMMonitor(agent, every_steps)


# ------------------------------------------------------------------ perception -----------------------

def full_obs(view):
    """The env observation the pixel-Q code reads, from what the agent sees: AgentView keeps the robot0_* keys
    (proprio + wrist image) and the upright agentview image; the raw render is the upright image rotated back
    (common.upright is its own inverse)."""
    obs = dict(view.obs)
    obs["agentview_image"] = common.upright(view.image)
    return obs


def box_mask_agent(env, box, res, robot_ids=None, floor_dz=0.005):
    """Target mask in the upright agentview image from a detect box (normalised, upright): the box pixels at least
    floor_dz above the box's 10th-percentile height (V2's box mask rule: drops the table), without robot pixels
    (robot_ids: the robot's geoms, the same robot self-filter V's heightmap uses). The arm often sits inside the
    target's box; without the filter it became the target's "top" (Q_INITIAL_POSITION 5-8 cm too high in 4 of 9
    smoke-test corrections, also with V2's proximity rule). Returns (mask, floor z, top z) or None."""
    m = load_modules()
    SC = m["scene"]
    if box is None:
        return None
    depth = common.upright(camera.depth_image(env, res))
    u0, u1 = int(np.floor(box["x_min"] * res)), int(np.ceil(box["x_max"] * res))
    v0, v1 = int(np.floor(box["y_min"] * res)), int(np.ceil(box["y_max"] * res))
    u0, v0, u1, v1 = max(0, u0), max(0, v0), min(res, u1), min(res, v1)
    if u1 - u0 < 2 or v1 - v0 < 2:
        return None
    vv, uu = np.mgrid[v0:v1, u0:u1]
    pts = SC.pixels_to_world(env, uu, vv, depth[v0:v1, u0:u1], res)
    robot = np.zeros(pts.shape[:2], bool)
    if robot_ids is not None:
        seg = common.upright(SC.segmentation_geom_ids(env, res))[v0:v1, u0:u1]
        robot = np.isin(seg, robot_ids)
    if robot.all():
        return None
    floor = float(np.percentile(pts[..., 2][~robot], 10))
    keep = (pts[..., 2] >= floor + floor_dz) & ~robot
    if not keep.any():
        return None
    mask = np.zeros((res, res), bool)
    mask[v0:v1, u0:u1] = keep
    return mask, floor, float(np.percentile(pts[..., 2][keep], 95))


def _v1_observer_cls():
    m = load_modules()
    v1 = m["v1_env"]

    class QObserver(v1.PickCorrectorEnv):
        """V1 PickCorrectorEnv wrapped around the pipeline's env (no reset, no stepping): only its geometry and
        observation code run (_render, _observe), so the Q-map gets exactly its training inputs."""

        def __init__(self, env, obj, cfg, target_mask, box_fn):
            from robosuite.utils import transform_utils as T

            self.cfg, self.env, self.T, self.obj = dict(cfg), env, T, obj
            self.res = common.ENV_RESOLUTION
            self.geom_ids = m["scene"].object_geom_ids(env, obj)
            self.robot_ids = m["v2_env"].robot_geom_ids(env) if target_mask == "vlm" else None
            self.target_mask, self.box_fn = target_mask, box_fn
            self.rng = np.random.default_rng(cfg.get("seed", 0))
            self.obs, self._frame, self.up0, self.frames, self.client = None, None, None, None, None

        def sync(self, obs):
            self.obs = obs
            if self.up0 is None:
                self.up0 = self._obj_up()      # object tilt is logged relative to the first call (privileged)

        def _render(self, cam=camera.CAMERA, size=None):
            if self.target_mask == "seg":
                return super(QObserver, self)._render(cam, size)
            env, size = self.env, size or self.res
            if cam == camera.CAMERA and size == self.res:
                rgb = common.upright(self.obs["agentview_image"])
            else:
                rgb = common.upright(np.asarray(env.env.sim.render(width=size, height=size, camera_name=cam)))
            depth = common.upright(camera.depth_image(env, size, cam))
            return rgb, depth, self._box_mask(cam, size, depth)

        def _box_mask(self, cam, size, depth):
            """VLM box -> mask in this camera: agentview = the box mask; wrist = the non-robot wrist pixels whose
            world point projects into the agentview box, between the box floor (+5 mm) and its top (+1 cm)."""
            am = box_mask_agent(self.env, self.box_fn(), self.res, robot_ids=self.robot_ids)
            if am is None:
                return np.zeros((size, size), bool)
            mask_a, floor, top = am
            if cam == camera.CAMERA and size == self.res:
                return mask_a
            SC = m["scene"]
            robot = np.isin(common.upright(SC.segmentation_geom_ids(self.env, size, cam)), self.robot_ids).reshape(-1)
            vv, uu = np.mgrid[0:size, 0:size]
            pts = SC.pixels_to_world(self.env, uu, vv, depth, size, cam).reshape(-1, 3)
            c, r, front = m["v2_env"].project_raw(self.env, pts, self.res)
            ua, va = self.res - 1 - c, self.res - 1 - r                  # raw -> upright agentview pixel
            ui, vi = np.round(ua).astype(int), np.round(va).astype(int)
            inside = front & (ui >= 0) & (ui < self.res) & (vi >= 0) & (vi < self.res)
            hit = np.zeros(len(pts), bool)
            hit[inside] = mask_a[vi[inside], ui[inside]]
            z = pts[:, 2]
            return (hit & ~robot & (z >= floor + 0.005) & (z <= top + 0.01)).reshape(size, size)

    return QObserver


def _v2_observer_cls():
    m = load_modules()
    v2 = m["v2_env"]

    class _AgentGate(object):
        """Stands in V2's HL slot: gate_inputs() comes from the VLM monitor (as hl.MoondreamHL)."""

        name = "vlm"

        def __init__(self, fn):
            self.fn = fn

        def gate_inputs(self):
            return self.fn()

    class VObserver(v2.BranchingPickEnv):
        """V2 BranchingPickEnv wrapped around the pipeline's env (no reset, no stepping): only gate() and
        observe() run, i.e. the code that produced V4's training samples."""

        def __init__(self, env, obj, cfg, gate_fn):
            from robosuite.utils import transform_utils as T

            SC = m["scene"]
            self.cfg, self.env, self.T, self.obj = dict(cfg), env, T, obj
            self.res = common.ENV_RESOLUTION
            self.reach = int(self.cfg.get("reach", 3))
            self.gate_m = self.reach * m["v2_qnet"].CELL_M + self.cfg.get("gate_margin_m", 0.04)
            self.finger_mask, self.robot_ids = None, v2.robot_geom_ids(env)
            self.geom_ids = SC.object_geom_ids(env, obj)
            self.hl = _AgentGate(gate_fn)
            self.rng = np.random.default_rng(self.cfg.get("seed", 0))
            self.obs, self.last_action, self.top_offset, self._frame, self.frames = None, None, None, None, None

        def sync(self, obs, last_action):
            self.obs = obs
            self.last_action = np.asarray(common.DUMMY_ACTION if last_action is None else last_action, dtype=np.float64)
            if self.top_offset is None:    # target top above its body origin: observe()'s fallback z_top only
                from fa_core import predicates as P

                SC = m["scene"]
                depth = common.upright(camera.depth_image(self.env, self.res))
                mask = np.isin(common.upright(SC.segmentation_geom_ids(self.env, self.res)), self.geom_ids)
                top = SC.mask_box_centre_top(self.env, depth, mask, self.res)[1] if mask.sum() >= 5 else None
                self.top_offset = 0.0 if top is None else top - P.obj_pos(self.env, self.obj)[2]

    return VObserver


class Perception(object):
    """Camera-side work of the supervisor for one episode and one target object (its sim name)."""

    def __init__(self, env, obj, vlm, target_mask="seg", approach_dz=C.APPROACH_DZ, q_cfg=None,
                 gate_max_above_m=C.V4_GATE_MAX_ABOVE_M, seed=0):
        if target_mask not in ("seg", "vlm"):
            raise ValueError("target_mask must be 'seg' or 'vlm'")
        m = load_modules()
        sys.path.insert(0, str(C.V4_DIR))
        try:
            import episodes as v4_episodes          # V4's own backend settings (collect.py env_cfg)
        finally:
            sys.path.remove(str(C.V4_DIR))
        vcfg = v4_episodes.env_cfg("none", None, None, seed=seed, gate_max_above_m=gate_max_above_m)
        vcfg["mask_source"] = "box" if target_mask == "vlm" else "seg"
        qcfg = {"wrist_box": 150, "min_mask_px": 20, "min_cell_frac": 0.0, "handoff_dz": approach_dz, "seed": seed}
        qcfg.update({k: v for k, v in (q_cfg or {}).items() if k in ("wrist_box", "min_mask_px", "min_cell_frac")})
        self.env, self.obj, self.vlm, self.target_mask, self.approach_dz = env, obj, vlm, target_mask, approach_dz
        self.SC, self.v1q = m["scene"], m["v1_qnet"]
        self.vo = _v2_observer_cls()(env, obj, vcfg, vlm.gate_inputs)
        self.qo = _v1_observer_cls()(env, obj, qcfg, target_mask, vlm.box)
        self.vcfg, self.qcfg = vcfg, qcfg

    # ---- close_to_object + V observation (V2 / V4)
    def gate(self, view):
        self.vo.sync(full_obs(view), view.last_action)
        return self.vo.gate()

    def v_obs(self, view):
        self.vo.sync(full_obs(view), view.last_action)
        return self.vo.observe()

    # ---- Q correction (V1)
    def handoff(self, view):
        """approach(Q_INITIAL_POSITION): V1's handoff move. Target = above the centre of the target's agentview mask
        box, approach_dz above its top (95th percentile height), gripper open, approach axis straight down
        (current yaw kept). Path: rise to that height if below it, travel, descend. Returns (motion, info) or
        None if the target is not seen."""
        obs = full_obs(view)
        self.qo.sync(obs)
        _, depth, mask = self.qo._render()
        if mask.sum() < self.qcfg["min_mask_px"]:
            return None
        centre, top = self.SC.mask_box_centre_top(self.env, depth, mask, common.ENV_RESOLUTION)
        target = np.r_[centre[:2], top + self.approach_dz]
        q_now = np.array(obs["robot0_eef_quat"], dtype=np.float64)
        quat = self.SC.top_down_quat(q_now, self.qo.T)
        ee = np.array(obs["robot0_eef_pos"], dtype=np.float64)
        z = max(ee[2], target[2])
        legs = [motion.MoveTo([ee[0], ee[1], z], common.GRIPPER_OPEN, q_now),
                motion.MoveTo([target[0], target[1], z], common.GRIPPER_OPEN, quat)]
        if z > target[2]:     # descend; when the EE started below the handoff height the travel leg already ends
            legs.append(motion.MoveTo(target, common.GRIPPER_OPEN, quat))   # there (V1 repeats it: a stalled
        m = motion.Sequence(legs)                                           # travel leg then cost 60 more steps)
        return m, {"q_initial_position": target, "object_centre": centre, "object_top_z": top,
                   "mask_px": int(mask.sum()), "target_mask": self.target_mask}

    def q_obs(self, view):
        """V1 observation (80x80 wrist crop on the target, depth, mask, proprio, valid cells) at the current pose."""
        self.qo.sync(full_obs(view))
        return self.qo._observe()

    def q_move(self, view, action):
        """V1's correction move for `action`: horizontal move (same height) above the cell's surface point,
        yaw rotated by the bin's offset, gripper open, at most 60 steps. Uses the frame of the last q_obs()."""
        fr, T = self.qo._frame, self.qo.T
        k, row, col = self.v1q.decode_action(action)
        if not fr["valid"][row, col]:
            raise ValueError("action {} is not a valid cell".format(action))
        target = fr["cell_points"][row, col]
        obs = view.obs
        q_now = np.array(obs["robot0_eef_quat"], dtype=np.float64)
        ee = np.array(obs["robot0_eef_pos"], dtype=np.float64)
        quat = T.mat2quat(self.SC.rotz(self.v1q.ROT_DEG[k]) @ T.quat2mat(q_now))   # OSC rotations: world frame
        mv = motion.MoveTo([target[0], target[1], ee[2]], common.GRIPPER_OPEN, quat, max_steps=60)
        return mv, {"cell_point": target, "goal": [target[0], target[1], ee[2]], "yaw_deg": self.v1q.ROT_DEG[k],
                    "obj_tilt_deg": fr["tilt_deg"], "valid_cells": int(fr["valid"].sum())}
