"""LIBERO environment for the rl/pick pixel-Q corrector. One EPISODE = one correction decision.

The Q-map values handing off to pi0. Every episode starts from the same HANDOFF POSE: the VLM agent
brings the open gripper 7 cm (handoff_dz) above the target object, over the centre of its segmentation
box, pointing straight down (-z), and hands over to the Q-map. The Q-map only chooses a horizontal move
and a yaw (z is never chosen); then pi0 takes over and the subtask is judged within a timeout.

Episodes come in CHAINS on one scene:
  reset(start)  restore an upright start (a library rl/pick reset state, or a task's init state after the
                10 wait steps), HANDOFF move, observe -> obs of the chain's first episode
  step(a)       one episode. a = (rotation bin k, cell row, cell col) as a flat index (qnet.encode_action):
                  1. back-project the cell to a 3D point on the object (masked depth pixels of the cell)
                  2. move horizontally (same height) above that point, rotating the yaw by ROT_DEG[k],
                     gripper open (fa_core.motion.MoveTo)
                  3. pi0 until the subtask succeeds or the judge window ends (timeout_steps control steps
                     counted from the decision, 5 s = 100 steps by default, including the move)
                reward 1 = success, 0 = timeout (or irrecoverable). The episode always ends here.
                returns (next_obs, reward, info). next_obs is the first observation of the NEXT episode
                of the chain: after a timeout the VLM agent re-does the HANDOFF move and learning goes on
                from the object's current state (pushed, tilted, knocked over: whatever the failed
                attempt left). next_obs is None when the chain ends: success, irrecoverable (off the
                table / out of reach / not visible / sim error) or a failed re-handoff. The caller caps
                the chain length and then resets to a fresh upright start.

Judge: the sim pick predicate (grasped and lifted >= 5 cm for 10 steps) stands in for the VLM judge
during training (pixel_Q_RL.md "Reward source"). fa_core.judge_agent.SubtaskJudgeAgent is the VLM judge
for test time; plugging it in here would cost ~5 VLM calls (1 fps) per episode.

HANDOFF move (_handoff): above the centre of the object's segmentation box, handoff_dz above the
object's top, approach axis straight down, current yaw kept, gripper open (optional xy jitter,
handoff_jitter_cm, default 0). Only mask + depth + camera geometry are used. Path: rise to the target
height (if below it), travel, descend.

Observation (dict, compact, what the Q-net and the replay buffer see):
  rgb      (80, 80, 3) uint8   crop of the upright agentview around the object; outside the mask grey
  depth    (80, 80) float16    (depth - median object depth) / 0.1 m, clipped to [-1.5, 1.5]
  mask     (80, 80) bool       target object
  proprio  (6,) float32        EE - object centre (dm, world xyz), gripper opening (0..1), EE yaw sin/cos
  valid    (10, 10) bool       8x8 cells with >= min_cell_frac of their pixels on the object

PRIVILEGED (training only): the object mask comes from MuJoCo's segmentation render (stand-in for the
segmentation model of pixel_Q_RL.md sec. 2), and reward / termination read the sim state through
fa_core.predicates (as pixel_Q_RL.md "Reward source" allows). Depth and camera geometry are what a
test-time agent has (fa_core.camera). The Q-net input itself is RGB-D + mask + proprio only.
"""

import json
import pathlib
import re
import sys
from collections import deque

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FAP = HERE.parents[1]                      # failure_agent_prototype/
if str(FAP) not in sys.path:
    sys.path.insert(0, str(FAP))
from fa_core import camera, common, motion, predicates as P  # noqa: E402

from qnet import CELL, CROP, GRID, PROPRIO_DIM, ROT_DEG, decode_action  # noqa: E402

ROUND_RIM = ("bowl", "mug", "cup")
# pixel_Q_RL.md appendix: LIBERO-90 training ("survey") split. The held-out split is never loaded.
TRAIN_TASKS_90 = (9, 10, 14, 17, 19, 25, 31, 34, 37, 38, 41, 42, 49, 53, 57, 58, 59, 61, 62, 63, 64, 67, 69, 71,
                  72, 73, 77, 78, 80, 85, 86, 87, 88, 89)
# the bowl / mug pick-and-place tasks among them (from the recorded task language)
DEFAULT_TASKS = ("libero_90:9,10,14,17,25,31,34,37,63,64,67,71,72,85", "libero_spatial:0-9")

MJ_OBJ_GEOM = 5                 # mjtObj.mjOBJ_GEOM in MuJoCo's segmentation render
# Irrecoverable = the object left the table or the robot's reach. A tilted or knocked-over object is NOT
# terminal: it is one of the states the Q-map must handle.
DROP_M = 0.15                   # object fell this far below its start height (off the table)
WORKSPACE_M = 0.30              # object pushed this far (xy) from its start
DEPTH_SCALE = 0.1               # m per unit of the depth channel


class IneligibleStart(Exception):
    """The start state cannot be used (not a round-rim target, held already, object not visible)."""


class ResetFailed(Exception):
    """This reset did not reach a usable handoff state (e.g. the handoff move knocked the object over).
    Stochastic: skip the episode, but keep the start state."""


# ------------------------------------------------------------------ start states -------------------

def _parse_ids(text):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def library_starts(policy="rl/pick"):
    """rl/pick reset states from library.json (pre-failure states pi0 actually reached)."""
    sr = common.tools_module("scene_restore")
    held_out = common.held_out_tasks()
    lib = json.loads(common.LIBRARY.read_text())
    words = {}
    for e in lib["entries"]:
        for inst in e.get("instances", []):
            m = re.match(r"pick\((\w+)\)", inst.get("subtask", ""))
            if m and "rl_reset_step" in inst:
                words[(inst["episode"], inst["rl_reset_step"])] = m.group(1)
    starts, skipped = [], []
    for rs in sr.load_reset_states(policy):
        suite, folder = rs["episode"].split("/")
        tid = int(re.match(r"task(\d+)", folder).group(1))
        tag = "{}@{}".format(rs["episode"].replace("/", "_"), rs["step"])
        if suite == "libero_90" and tid in held_out:
            raise SystemExit("{}: libero_90 task {} is in the held-out split".format(tag, tid))
        if suite == "libero_90" and tid not in TRAIN_TASKS_90:
            skipped.append((tag, "libero_90 task {} is not in the pixel-Q training split".format(tid)))
            continue
        starts.append({"kind": "reset_state", "tag": tag, "episode": rs["episode"], "step": rs["step"],
                       "word": words.get((rs["episode"], rs["step"]))})
    return starts, skipped


def task_starts(task_specs, init_states):
    """Task init states, e.g. task_specs=["libero_90:9,10", "libero_spatial:0-9"], init_states=[0, 1, 2]."""
    held_out = common.held_out_tasks()
    starts = []
    for spec in task_specs:
        suite, ids = spec.split(":")
        for tid in _parse_ids(ids):
            if suite == "libero_90" and tid in held_out:
                raise SystemExit("libero_90 task {} is in the held-out split".format(tid))
            for k in init_states:
                starts.append({"kind": "task_init", "tag": "{}_task{}_init{}".format(suite, tid, k), "suite": suite,
                               "task_id": tid, "init_idx": int(k)})
    return starts


def is_round_rim(name):
    return any(w in name.lower() for w in ROUND_RIM)


# ------------------------------------------------------------------ geometry -----------------------

def segmentation_geom_ids(env, size, cam=camera.CAMERA):
    """Per-pixel MuJoCo geom id (-1 = not a geom) of the raw (not upright) render, like depth_image."""
    sim = env.env.sim
    try:
        seg = sim.render(width=size, height=size, camera_name=cam, segmentation=True)
    except TypeError as e:
        raise RuntimeError("MjSim.render has no segmentation option in this robosuite ({}); the corrector needs "
                           "a mask source".format(e))
    seg = np.asarray(seg)
    if seg.ndim == 3 and seg.shape[-1] == 2:      # robosuite: (objtype, objid) per pixel
        return np.where(seg[..., 0] == MJ_OBJ_GEOM, seg[..., 1], -1)
    raise RuntimeError("unexpected segmentation render shape {}".format(seg.shape))


def object_geom_ids(env, name):
    m = env.env.sim.model
    obj = env.env.objects_dict[name]
    ids = set()
    for g in list(obj.contact_geoms) + list(obj.visual_geoms):
        try:
            ids.add(m.geom_name2id(g))
        except Exception:
            pass
    if not ids:
        raise IneligibleStart("no geoms found for {}".format(name))
    return np.array(sorted(ids))


def pixels_to_world(env, us, vs, depth, size, cam=camera.CAMERA):
    """Vectorised fa_core.camera.unproject: upright pixels (u, v) with metric depth -> world points (..., 3)."""
    pos, mat, fovy = camera._cam(env, cam)
    f = 0.5 * size / np.tan(np.deg2rad(fovy) / 2)
    c, r = size - 1 - np.asarray(us, float), size - 1 - np.asarray(vs, float)
    d = np.asarray(depth, float)
    pc = np.stack([(c + 0.5 - size / 2) / f * d, (r + 0.5 - size / 2) / f * d, -d], axis=-1)
    return pos + pc @ mat.T


def rotz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def top_down_quat(quat, T):
    """Same yaw as `quat`, approach axis straight down: EE z axis = world -z (robosuite Panda: the EE z
    axis points out between the fingers; check with --mode inspect, approach_tilt_deg)."""
    R = T.quat2mat(quat)
    x = np.array([R[0, 0], R[1, 0], 0.0])
    if np.linalg.norm(x) < 1e-6:                   # x axis vertical: fall back to the y axis for the yaw
        x = np.cross(np.array([R[0, 1], R[1, 1], 0.0]), [0.0, 0.0, -1.0])
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, -1.0])
    return T.mat2quat(np.stack([x, np.cross(z, x), z], axis=1))


def approach_tilt_deg(quat, T):
    """Angle between the EE approach axis (z) and straight down."""
    return float(np.degrees(np.arccos(np.clip(-T.quat2mat(quat)[2, 2], -1, 1))))


def proprio_features(obs, center, T):
    ee = np.asarray(obs["robot0_eef_pos"], float)
    q = np.asarray(obs["robot0_gripper_qpos"], float)
    R = T.quat2mat(obs["robot0_eef_quat"])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    feat = np.r_[(ee - center) * 10.0, np.clip((q[0] - q[1]) / 0.08, 0, 1.5), np.sin(yaw), np.cos(yaw)]
    assert feat.shape == (PROPRIO_DIM,)
    return feat.astype(np.float32)


# ------------------------------------------------------------------ env ----------------------------

class PickCorrectorEnv(object):
    def __init__(self, cfg, client=None):
        """cfg: timeout_s, handoff_dz, handoff_jitter_cm, min_cell_frac, min_mask_px, env_seed, seed, host, port."""
        self.cfg = cfg
        self.client = client
        self.env, self._env_key, self._trace_key, self._trace = None, None, None, None
        self.frames = None
        self.rng = np.random.default_rng(cfg.get("seed", 0) + cfg.get("worker_id", 0))
        from robosuite.utils import transform_utils as T

        self.T = T

    # ---- setup
    def _pi0(self):
        if self.client is None:
            from openpi_client import websocket_client_policy

            self.client = websocket_client_policy.WebsocketClientPolicy(self.cfg["host"], self.cfg["port"])
        return self.client

    def _ensure_env(self, start):
        trace_dir = None
        if start["kind"] == "reset_state":
            sr = common.tools_module("scene_restore")
            meta, entry, trace_dir = sr.episode_info(start["episode"])
            key = (meta["task_suite"], entry["task_id"], meta["env_seed"], meta["camera_resolution"])
        else:
            key = (start["suite"], start["task_id"], self.cfg.get("env_seed", 7), common.ENV_RESOLUTION)
        if key != self._env_key:
            if self.env is not None:
                self.env.close()
            self.env, self.task, _ = common.make_env(key[0], key[1], key[2], key[3])
            self._env_key, self.res = key, key[3]
        if trace_dir is not None and self._trace_key != start["episode"]:
            sr = common.tools_module("scene_restore")
            self._trace, self._trace_key = sr.load_trace(trace_dir), start["episode"]

    def _resolve_target(self, start, ee):
        if start["kind"] == "task_init":
            picks = [c for c in P.goal_checks(self.env) if c.kind == "pick"]
            if not picks:
                raise IneligibleStart("task goal has no pick-and-place of a movable object")
            return picks[0].obj
        word = start.get("word")
        if not word:
            raise IneligibleStart("no pick(...) label in library.json for this reset state")
        names = P.object_names(self.env)
        toks = P._tokens(word)
        score = {n: len(toks & P._tokens(n)) for n in names}
        best = max(score.values()) if score else 0
        if best == 0:
            raise IneligibleStart("no sim object matches {!r}".format(word))
        cands = [n for n in names if score[n] == best]
        return min(cands, key=lambda n: np.linalg.norm(P.obj_pos(self.env, n) - ee))   # tie: the one pi0 went for

    def reset(self, start, record=False):
        """Restore an (upright) start, HANDOFF move, observe. Returns the first episode's observation."""
        self._ensure_env(start)
        env = self.env
        if start["kind"] == "reset_state":
            obs = common.tools_module("scene_restore").reset_to(env, self._trace, start["step"])
        else:
            env.reset()
            obs = env.set_init_state(common.init_state(start["suite"], start["task_id"], start["init_idx"]))
            for _ in range(common.NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(list(common.DUMMY_ACTION))
        self.obs, self.prompt = obs, str(self.task.language)
        self.obj = start.get("object") or self._resolve_target(start, obs["robot0_eef_pos"])
        if not is_round_rim(self.obj):
            raise IneligibleStart("target {} is not a round-rim object".format(self.obj))
        if P.grasped(env, self.obj):
            raise IneligibleStart("{} is already grasped at the start".format(self.obj))
        self.geom_ids = object_geom_ids(env, self.obj)
        self.check = P.PickCheck(self.obj)
        self.check.start(env)
        self.p0 = P.obj_pos(env, self.obj)           # scene reference: irrecoverable + tilt logging
        self.up0 = self._obj_up()
        self.steps, self.sim_error, self.fail_reason = 0, None, None
        self.frames = [common.upright(obs["agentview_image"])] if record else None
        self.start_tilt_deg = approach_tilt_deg(obs["robot0_eef_quat"], self.T)
        if self._render()[2].sum() < self.cfg.get("min_mask_px", 20):
            raise IneligibleStart("{} is not visible in agentview".format(self.obj))
        stop, self.handoff_info = self._handoff()
        if stop or self.check.done:
            raise ResetFailed("handoff move: " + str(self._failure() or self.handoff_info.get("error")
                                                      or "pick predicate fired"))
        self.check.start(env)            # lift is measured from the state the Q-map is handed
        o = self._observe()
        if o is None:
            raise ResetFailed("{} not visible at the handoff pose".format(self.obj))
        return o

    def _obj_up(self):
        bid = P.body_id(self.env, self.obj)
        return np.array(self.env.env.sim.data.body_xmat[bid]).reshape(3, 3)[:, 2].copy()

    def obj_tilt_deg(self):
        """Object tilt from its pose at the start (upright): 0 standing, ~90 on its side."""
        return float(np.degrees(np.arccos(np.clip(self._obj_up() @ self.up0, -1, 1))))

    # ---- observation
    def _render(self):
        """(upright rgb, upright metric depth, upright object mask) of the current state."""
        env, size = self.env, self.res
        rgb = common.upright(self.obs["agentview_image"])
        depth = common.upright(camera.depth_image(env, size))
        mask_full = np.isin(common.upright(segmentation_geom_ids(env, size)), self.geom_ids)
        return rgb, depth, mask_full

    def _handoff(self):
        """Gripper open, pointing straight down (current yaw kept), above the centre of the object's
        segmentation box, handoff_dz (7 cm) above the object's top. Returns (stop, info).

        Box centre -> world: the ray through the box-centre pixel, cut at the object's median height (the
        box centre need not lie on the object, e.g. a bowl on its side). Top = 95th percentile height of
        the object's pixels. Only mask + depth + camera geometry are used, as at test time."""
        _, depth, mask_full = self._render()
        if mask_full.sum() < self.cfg.get("min_mask_px", 20):
            return True, {"error": "object not visible"}
        vs, us = np.nonzero(mask_full)
        pts = pixels_to_world(self.env, us, vs, depth[vs, us], self.res)
        uc, vc = 0.5 * (us.min() + us.max()), 0.5 * (vs.min() + vs.max())
        cam_pos = camera._cam(self.env, camera.CAMERA)[0]
        ray = pixels_to_world(self.env, uc, vc, 1.0, self.res) - cam_pos       # point at unit depth - camera
        z_mid = float(np.median(pts[:, 2]))
        centre = cam_pos + ray * (z_mid - cam_pos[2]) / ray[2]
        jit = self.rng.uniform(-1, 1, 2) * self.cfg.get("handoff_jitter_cm", 0.0) / 100.0
        top = float(np.percentile(pts[:, 2], 95))
        target = np.r_[centre[:2] + jit, top + self.cfg.get("handoff_dz", 0.07)]
        q_now = np.array(self.obs["robot0_eef_quat"], copy=True)
        quat = top_down_quat(q_now, self.T)
        ee = np.array(self.obs["robot0_eef_pos"], copy=True)
        z = max(ee[2], target[2])
        m = motion.Sequence([
            motion.MoveTo([ee[0], ee[1], z], common.GRIPPER_OPEN, q_now),
            motion.MoveTo([target[0], target[1], z], common.GRIPPER_OPEN, quat),
            motion.MoveTo(target, common.GRIPPER_OPEN, quat),
        ])
        stop, n = self._run_motion(m)
        last = m.motions[-1]
        return stop, {"target": [float(x) for x in target], "steps": n, "reached": bool(last.reached),
                      "pos_err_m": last.final_pos_err,
                      "tilt_deg": approach_tilt_deg(self.obs["robot0_eef_quat"], self.T)}

    def _failure(self):
        return ("sim_error: " + self.sim_error) if self.sim_error else self.fail_reason

    def _observe(self):
        env, size = self.env, self.res
        rgb, depth, mask_full = self._render()
        if mask_full.sum() < self.cfg.get("min_mask_px", 20):
            return None
        vs, us = np.nonzero(mask_full)                     # crop centred on the segmentation box
        u0 = int(np.clip(round(0.5 * (us.min() + us.max()) - CROP / 2), 0, size - CROP))
        v0 = int(np.clip(round(0.5 * (vs.min() + vs.max()) - CROP / 2), 0, size - CROP))
        win = (slice(v0, v0 + CROP), slice(u0, u0 + CROP))
        m, d = mask_full[win], depth[win]
        vv, uu = np.mgrid[v0:v0 + CROP, u0:u0 + CROP]
        world = pixels_to_world(env, uu, vv, d, size)
        cells_m = m.reshape(GRID, CELL, GRID, CELL)
        frac = cells_m.mean(axis=(1, 3))
        valid = frac >= self.cfg.get("min_cell_frac", 0.25)
        if not valid.any():
            return None
        cnt = cells_m.sum(axis=(1, 3))[..., None]
        wsum = (world * m[..., None]).reshape(GRID, CELL, GRID, CELL, 3).sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            cell_points = wsum / cnt                       # mean surface point of the cell's object pixels
        center = world[m].mean(axis=0)
        crop_rgb = rgb[win].copy()
        crop_rgb[~m] = 128
        dfeat = np.clip((d - np.median(d[m])) / DEPTH_SCALE, -1.5, 1.5)
        self._frame = {"u0": u0, "v0": v0, "cell_points": cell_points, "center": center, "valid": valid,
                       "rgb_full": rgb, "mask_full": mask_full, "tilt_deg": self.obj_tilt_deg()}
        return {"rgb": crop_rgb, "depth": dfeat.astype(np.float16), "mask": m.copy(),
                "proprio": proprio_features(self.obs, center, self.T), "valid": valid}

    def inspect_info(self):
        """Current frame + every valid cell's 3D point projected back to the image (should hit the cell)."""
        fr = dict(self._frame)
        rows, cols = np.nonzero(fr["valid"])
        fr["cells"] = [(int(r), int(c)) for r, c in zip(rows, cols)]
        fr["cell_centres_px"] = [(fr["u0"] + c * CELL + CELL / 2 - 0.5, fr["v0"] + r * CELL + CELL / 2 - 0.5)
                                 for r, c in fr["cells"]]
        fr["cell_points_px"] = [tuple(float(x) for x in camera.project(self.env, fr["cell_points"][r, c], self.res))
                                for r, c in fr["cells"]]
        fr["object"], fr["prompt"] = self.obj, self.prompt
        fr["handoff"], fr["start_tilt_deg"] = self.handoff_info, self.start_tilt_deg
        return fr

    # ---- stepping
    def _irrecoverable(self):
        p = P.obj_pos(self.env, self.obj)
        if p[2] < self.p0[2] - DROP_M:
            return "dropped"
        if np.linalg.norm(p[:2] - self.p0[:2]) > WORKSPACE_M:
            return "out_of_workspace"
        return None

    def _env_step(self, action):
        """One control step. Returns True when the correction step should end (success / failure)."""
        try:
            self.obs, _, _, _ = self.env.step([float(x) for x in action])
        except Exception as e:      # main.py also counts sim errors as failures
            self.sim_error = str(e)
            return True
        self.steps += 1
        if self.frames is not None:
            self.frames.append(common.upright(self.obs["agentview_image"]))
        if self.check.update(self.env, self.steps):
            return True
        self.fail_reason = self._irrecoverable()
        return self.fail_reason is not None

    def _run_motion(self, m):
        n = 0
        while True:
            a = m.next_action(self.obs)
            if a is None:
                return False, n
            n += 1
            if self._env_step(a):
                return True, n

    def _run_pi0(self, n):
        client, plan = self._pi0(), deque()
        for i in range(n):
            if not plan:
                chunk = np.asarray(client.infer(common.pi0_request(self.obs, self.prompt))["actions"])
                plan.extend(chunk[:common.REPLAN_STEPS])
            if self._env_step(plan.popleft()):
                return True, i + 1
        return False, n

    def _timeout_steps(self):
        return int(round(self.cfg.get("timeout_s", 5.0) * common.CONTROL_FREQ))

    def _next_episode(self):
        """After a failed episode: the VLM agent re-does the HANDOFF move; observe the next episode's state.
        Returns (next_obs or None, chain end reason or None)."""
        failed = self._failure()
        if failed:                                     # irrecoverable: the chain ends
            return None, failed
        stop, handoff = self._handoff()
        self.last_handoff = handoff
        if stop or self.check.done:
            return None, "handoff: " + str(self._failure() or handoff.get("error") or "pick predicate fired")
        self.check.start(self.env)                     # lift measured from the new state
        nxt = self._observe()
        return (nxt, None) if nxt is not None else (None, "not_visible")

    def step(self, action):
        """One episode: 2D move + yaw, then pi0 until success or the judge timeout. See module docstring."""
        k, row, col = decode_action(action)
        fr = self._frame
        if not fr["valid"][row, col]:
            raise ValueError("action {} is outside the object mask".format(action))
        T = self.T
        target = fr["cell_points"][row, col]
        q_now = np.array(self.obs["robot0_eef_quat"], copy=True)
        ee = np.array(self.obs["robot0_eef_pos"], copy=True)
        quat = T.mat2quat(rotz(ROT_DEG[k]) @ T.quat2mat(q_now))   # OSC rotations are world-frame: yaw about z
        # 2D only: move above the cell's surface point at the handoff height (z is not part of the action)
        budget = self._timeout_steps()
        move = motion.MoveTo([target[0], target[1], ee[2]], common.GRIPPER_OPEN, quat, max_steps=min(60, budget))
        stop, n_move = self._run_motion(move)
        n_pi0 = 0
        if not stop and budget - n_move > 0:
            stop, n_pi0 = self._run_pi0(budget - n_move)
        success = bool(self.check.done)
        self.last_handoff = None
        if success:
            nxt, end = None, "success"
        else:
            nxt, end = self._next_episode()
        info = {"k": k, "row": row, "col": col, "yaw_deg": ROT_DEG[k], "obj_tilt_deg": fr["tilt_deg"],
                "target": [float(x) for x in target], "move_steps": n_move, "move_reached": bool(move.reached),
                "move_err_m": move.final_pos_err, "pi0_steps": n_pi0, "episode_steps": n_move + n_pi0,
                "success": success, "timeout": not success and n_move + n_pi0 >= budget,
                "chain_end": end, "handoff": self.last_handoff, "total_steps": self.steps}
        return nxt, float(success), info

    def baseline_step(self):
        """One episode without the Q-map: from the handoff pose, pi0 until success or the judge timeout.
        Returns (next_obs or None, reward, info), chained like step()."""
        stop, n_pi0 = self._run_pi0(self._timeout_steps())
        success = bool(self.check.done)
        self.last_handoff = None
        nxt, end = (None, "success") if success else self._next_episode()
        return nxt, float(success), {"pi0_steps": n_pi0, "success": success, "chain_end": end,
                                     "obj_tilt_deg": self._frame["tilt_deg"], "total_steps": self.steps}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
