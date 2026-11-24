"""LIBERO backend for V2 (the rollout.py interface): pi0 main rollouts from task init states, the gate, the
gripper-centred heightmap, the local correction primitive, in-memory snapshots and the HL slot.

pixel_Q_RL_V2.md sections:
  sec. 4  gate(): pick subtask (PickCheck not latched), gripper open (last command open and finger gap >=
          gate_min_gap) and empty (no movable object grasped), target within r*2 cm + 4 cm (xy) of the grasp point,
          and (only if cfg gate_max_above_m is set; V3) the grasp point at most that far above the target's top.
          Privileged sim quantities stand in for the VLM's judgements during training; with --hl moondream the
          gate reads the agent's own subtask / grounded target / held measurements instead.
  sec. 5  observe(): wrist + agentview depth unprojected to world points (camera pose at this tick), robot points
          dropped (sim segmentation, or --robot-mask fixed: a fixed finger mask on the wrist image + a height cut
          on agentview), rasterized into the 88 x 88 yaw-aligned map (heightmap.py). Target mask: sim segmentation
          (default), or derived from an agentview box (--mask-source box | mix; at test time Moondream's detect
          box). "Grounded" = the target is seen by agentview (>= min_mask_px pixels).
  sec. 6  execute_primitive(): clearance rise, level (rotate about the grasp site to point down, yaw kept), move
          to the cell's xy at the current height adding the bin's yaw; one step budget (prim_cap) for all legs;
          then pi0's queue is cleared so it replans.
  sec. 9  snapshot()/restore(): tools/scene_restore.capture_snapshot (sim, warm start, controls, gripper,
          controller, robosuite step counter) + this env's step counter, pi0's queued actions, the PickCheck state,
          the last action and the failure flags. Branch frames are cut from the video on restore.
  sec. 9.4 hl_tick(): HL module (hl.py): PrivilegedHL stand-in (training), MoondreamHL (deployment), NoHL.

Irrecoverable (V1's rules, from the target's sim position, only while the pick is not done): fell > 15 cm below
its start height, or pushed > 30 cm (xy) from its start. A tilted / knocked-over object is NOT terminal. Sim
errors end the rollout too. V1's "not visible" rule is not used (the target may leave the wrist view during
normal pi0 motion; sec. 9.3).

Pixels: the depth / segmentation renders and the obs images share MuJoCo's raw (upside-down) layout; world points
are unprojected in that layout, so nothing is flipped here.
"""

import pathlib
import sys
import time
from collections import deque

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for _p in (HERE.parent, HERE.parents[2]):   # pixel_Q/ (the shared package), failure_agent_prototype/ (fa_core)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from fa_core import camera, common, motion, predicates as P  # noqa: E402
from shared import scene as SC  # noqa: E402
from shared.scene import WRIST_CAMERA, IneligibleStart, LiberoScene  # noqa: E402

import heightmap as HMAP  # noqa: E402
import hl as HL  # noqa: E402
from qnet import CELL_M, GRID, ROT_DEG, decode_action, reach_mask  # noqa: E402

AGENT_CAMERA = camera.CAMERA
FULL_GAP_M = 0.08                      # Panda finger gap when fully open
ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mount0_")


# ------------------------------------------------------------------ geometry helpers ---------------

def _body_name(model, b):
    try:
        return model.body_id2name(b) or ""
    except Exception:                  # plain mujoco model
        import mujoco

        return mujoco.mj_id2name(getattr(model, "_model", model), mujoco.mjtObj.mjOBJ_BODY, b) or ""


def robot_geom_ids(env):
    """Geoms of the robot (arm, gripper, mount): every geom whose body, or an ancestor body, has a robosuite robot
    prefix. Used to drop robot points from the heightmap during training (sec. 5 step 2)."""
    m = env.env.sim.model
    nb = int(m.nbody)
    names = [_body_name(m, b) for b in range(nb)]
    parent = np.asarray(m.body_parentid)
    is_robot = np.zeros(nb, bool)
    for b in range(nb):
        k = b
        while k > 0:
            if names[k].startswith(ROBOT_BODY_PREFIXES):
                is_robot[b] = True
                break
            k = int(parent[k])
    return np.flatnonzero(is_robot[np.asarray(m.geom_bodyid)])


def camera_cloud(env, cam, size, rgb_raw=None):
    """World points of every pixel of one camera at the current state (raw layout, row-major).
    Returns dict: pts (N, 3), rgb (N, 3) uint8, seg (N,) geom id (-1 = none), size."""
    depth = camera.depth_image(env, size, cam)
    seg = SC.segmentation_geom_ids(env, size, cam)
    if rgb_raw is None:
        rgb_raw = np.asarray(env.env.sim.render(width=size, height=size, camera_name=cam))
    pos, mat, fovy = camera._cam(env, cam)
    f = 0.5 * size / np.tan(np.deg2rad(fovy) / 2)
    rr, cc = np.mgrid[0:size, 0:size]
    pc = np.stack([(cc + 0.5 - size / 2) / f * depth, (rr + 0.5 - size / 2) / f * depth, -depth], axis=-1)
    return {"pts": (pos + pc @ mat.T).reshape(-1, 3), "rgb": np.asarray(rgb_raw)[..., :3].reshape(-1, 3),
            "seg": np.asarray(seg).reshape(-1), "size": size}


def project_raw(env, pts, size, cam=AGENT_CAMERA):
    """World points (N, 3) -> raw-layout pixel coordinates (col, row) of `cam`, and in-front mask."""
    pos, mat, fovy = camera._cam(env, cam)
    p = (np.asarray(pts) - pos) @ mat
    d = -p[:, 2]
    f = 0.5 * size / np.tan(np.deg2rad(fovy) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = p[:, 0] / d * f + size / 2 - 0.5
        r = p[:, 1] / d * f + size / 2 - 0.5
    return c, r, d > 1e-6


def box_to_raw(box, size):
    """Normalized UPRIGHT box {x_min, y_min, x_max, y_max} (Moondream detect) -> raw (c0, r0, c1, r1)."""
    return (size - 1 - box["x_max"] * size, size - 1 - box["y_max"] * size,
            size - 1 - box["x_min"] * size, size - 1 - box["y_min"] * size)


def rotx(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# ------------------------------------------------------------------ env ----------------------------

class BranchingPickEnv(LiberoScene):
    """See module docstring. cfg: reach, gate_margin_m, gate_min_gap, gate_max_above_m, min_mask_px, agent_px, wrist_px, fill_holes,
    robot_mask (seg | fixed), finger_mask (path), mask_source (seg | box | mix), mix_p, box_jitter, clear_min,
    clear_to, hl (privileged | none | moondream), vlm_model, env_seed, seed, worker_id, host, port."""

    def __init__(self, cfg, client=None):
        super(BranchingPickEnv, self).__init__(cfg, client)
        self.reach = int(cfg.get("reach", 3))
        self.gate_m = self.reach * CELL_M + cfg.get("gate_margin_m", 0.04)
        self.frames, self._frame, self._robot_key = None, None, None
        self.hl = HL.make_hl(cfg.get("hl", "privileged"), self, cfg)
        self.finger_mask = None
        if cfg.get("robot_mask", "seg") == "fixed":
            if not cfg.get("finger_mask"):
                raise SystemExit("--robot-mask fixed needs --finger-mask (build one with --mode finger-mask)")
            self.finger_mask = np.load(cfg["finger_mask"]).astype(bool)

    # ---- rollout
    def reset(self, start, record=False):
        if start["kind"] != "task_init":
            raise IneligibleStart("V2 starts from task init states only (--sources tasks)")
        self._ensure_env(start)
        env = self.env
        if self._robot_key != self._env_key:
            self.robot_ids, self._robot_key = robot_geom_ids(env), self._env_key
        env.reset()
        obs = env.set_init_state(common.init_state(start["suite"], start["task_id"], start["init_idx"]))
        for _ in range(common.NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(list(common.DUMMY_ACTION))
        self.obs, self.prompt = obs, str(self.task.language)
        self.task_language = self.prompt
        self.obj = start.get("object") or self._resolve_target(start, obs["robot0_eef_pos"])
        if not SC.is_round_rim(self.obj):
            raise IneligibleStart("target {} is not a round-rim object".format(self.obj))
        if P.grasped(env, self.obj):
            raise IneligibleStart("{} is already grasped at the start".format(self.obj))
        self.geom_ids = SC.object_geom_ids(env, self.obj)
        self.movable = P.object_names(env)
        self.check = P.PickCheck(self.obj)
        self.check.start(env)
        self.p0 = P.obj_pos(env, self.obj)              # irrecoverable rules + tilt logging
        self.up0 = self._obj_up()
        self.steps, self.plan = 0, deque()
        self.last_action = np.asarray(common.DUMMY_ACTION, dtype=np.float64)
        self.sim_error, self.fail_reason, self.stop_reason, self._task_done = None, None, None, False
        self.budget = int(common.MAX_STEPS[start["suite"]])
        self.suite, self.task_id = start["suite"], start["task_id"]
        self.frames = [SC.video_frame(obs)] if record else None
        self._frame = None
        depth = common.upright(camera.depth_image(env, self.res))
        mask = np.isin(common.upright(SC.segmentation_geom_ids(env, self.res)), self.geom_ids)
        if mask.sum() < self.cfg.get("min_mask_px", 20):
            raise IneligibleStart("{} is not visible in agentview".format(self.obj))
        _, top = SC.mask_box_centre_top(env, depth, mask, self.res)
        self.top_offset = top - self.p0[2]               # target top above its body origin (HL stand-in)
        self.hl.reset()

    @property
    def step_count(self):
        return self.steps

    def set_hl(self, name):
        """Switch the HL module (evaluation conditions run different ones on the same workers)."""
        if name != self.hl.name:
            self.hl = HL.make_hl(name, self, self.cfg)

    def _irrecoverable(self):
        if self.check.done:                              # placing moves the object anywhere
            return None
        p = P.obj_pos(self.env, self.obj)
        if p[2] < self.p0[2] - SC.DROP_M:
            return "dropped"
        if np.linalg.norm(p[:2] - self.p0[:2]) > SC.WORKSPACE_M:
            return "out_of_workspace"
        return None

    def _env_step(self, action):
        """One control step. Returns True when a motion should stop (pick done, irrecoverable, sim error)."""
        try:
            self.obs, _, done, _ = self.env.step([float(x) for x in action])
        except Exception as e:                           # main.py also counts sim errors as failures
            self.sim_error = str(e)
            return True
        self.steps += 1
        self.last_action = np.asarray(action, dtype=np.float64)
        if done:
            self._task_done = True
        if self.frames is not None:
            self.frames.append(SC.video_frame(self.obs))
        self.check.update(self.env, self.steps)
        self.fail_reason = self._irrecoverable()
        return bool(self.check.done or self.fail_reason is not None)

    def _run_motion(self, m):
        n = 0
        while True:
            a = m.next_action(self.obs)
            if a is None:
                return False, n
            n += 1
            if self._env_step(a):
                return True, n

    def pi0_step(self):
        if self.sim_error or self.fail_reason:
            return
        if not self.plan:
            chunk = np.asarray(self._pi0().infer(common.pi0_request(self.obs, self.prompt))["actions"])
            self.plan.extend(chunk[:common.REPLAN_STEPS])
        self._env_step(self.plan.popleft())

    # ---- primitive (sec. 6)
    def execute_primitive(self, action, max_steps=30):
        """Correction `action` from the geometry of the last observe(): clearance rise, level, move + yaw, within
        max_steps control steps in total; then pi0's queue is cleared (it replans). Returns an info dict."""
        fr, T = self._frame, self.T
        if fr is None:
            raise RuntimeError("execute_primitive needs an observe() first")
        k, i, j = decode_action(action)
        txy = HMAP.cell_to_world(i, j, fr["p_xy"], fr["psi"])
        ee = np.array(self.obs["robot0_eef_pos"], dtype=np.float64)
        q_now = np.array(self.obs["robot0_eef_quat"], dtype=np.float64)
        h_clear = HMAP.clearance_height(fr["z"], i, j)
        z = ee[2]
        rise = h_clear is not None and z < h_clear + self.cfg.get("clear_min", 0.02)
        if rise:
            z = h_clear + self.cfg.get("clear_to", 0.03)
        q_level = SC.top_down_quat(q_now, T)
        q_goal = T.mat2quat(SC.rotz(ROT_DEG[k]) @ T.quat2mat(q_level))   # OSC rotations are world-frame
        legs = ([("rise", [ee[0], ee[1], z], q_now)] if rise else []) + [
            ("level", [ee[0], ee[1], z], q_level), ("move", [txy[0], txy[1], z], q_goal)]
        used, info_legs, stop = 0, {}, False
        for name, pos, quat in legs:
            left = int(max_steps) - used
            if left <= 0 or stop:
                info_legs[name] = {"steps": 0, "reached": False, "skipped": True}
                continue
            m = motion.MoveTo(pos, common.GRIPPER_OPEN, quat, max_steps=left)
            stop, n = self._run_motion(m)
            used += n
            info_legs[name] = {"steps": n, "reached": bool(m.reached), "pos_err_m": m.final_pos_err,
                               "rot_err_rad": m.final_rot_err}
        self.plan.clear()                                # hand back: pi0 replans from the new pose
        return {"steps": used, "legs": info_legs, "h_clear": h_clear, "rise": bool(rise), "z": round(float(z), 4),
                "target_xy": [round(float(x), 4) for x in txy], "yaw_deg": ROT_DEG[k], "k": k, "i": i, "j": j,
                "reached": all(v.get("reached") for v in info_legs.values()), "stopped": bool(stop)}

    # ---- HL slot (sec. 9.4)
    def hl_tick(self):
        out = self.hl.tick(self.steps)
        if out is None:
            return None
        m, info = out
        stop, n = self._run_motion(m)
        self.plan.clear()
        self.hl.correction_done(self.steps)
        info.update(steps=n, stopped=bool(stop), hl=self.hl.name)
        return info

    def target_approach_point(self):
        """point_approach goal: 7 cm above the top of the target, over the centre of its agentview mask box (the
        sim position if it is not visible)."""
        depth = common.upright(camera.depth_image(self.env, self.res))
        mask = np.isin(common.upright(SC.segmentation_geom_ids(self.env, self.res)), self.geom_ids)
        if mask.sum() >= self.cfg.get("min_mask_px", 20):
            centre, top = SC.mask_box_centre_top(self.env, depth, mask, self.res)
        else:
            p = P.obj_pos(self.env, self.obj)
            centre, top = p, p[2] + self.top_offset
        return np.r_[centre[:2], top + HL.APPROACH_DZ]

    # ---- gate (sec. 4)
    def gate(self):
        obs = self.obs
        ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        q = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
        gap = float(q[0] - q[1])
        opened = bool(self.last_action[-1] < 0 and gap >= self.cfg.get("gate_min_gap", 0.05))
        max_above = self.cfg.get("gate_max_above_m")    # None = no height condition (V2)
        gi = self.hl.gate_inputs()
        if gi is None:                                   # privileged stand-ins for the VLM's judgements
            tpos = P.obj_pos(self.env, self.obj)
            dxy = float(np.hypot(*(ee[:2] - tpos[:2])))
            above = float(ee[2] - tpos[2] - self.top_offset)
            info = {"pick": not self.check.done, "open": opened, "target_dxy": dxy, "near": dxy <= self.gate_m,
                    "above_top": above, "low": max_above is None or above <= max_above}
            if not (info["pick"] and opened and info["near"] and info["low"]):
                return False, info
            info["empty"] = not any(P.grasped(self.env, o) for o in self.movable)
            return info["empty"], info
        tgt = gi["target"]                               # the agent's grounded surface point on the target
        dxy = float("inf") if tgt is None else float(np.hypot(ee[0] - tgt[0], ee[1] - tgt[1]))
        above = float("inf") if tgt is None else float(ee[2] - tgt[2])
        info = {"pick": bool(gi["pick"]), "open": opened, "empty": not gi["held"], "target_dxy": dxy,
                "near": dxy <= self.gate_m, "above_top": above, "low": max_above is None or above <= max_above,
                "grounded": tgt is not None}
        return bool(info["pick"] and opened and info["empty"] and info["near"] and info["low"]), info

    # ---- observation (sec. 5)
    def _robot_points(self, cl, is_wrist, p):
        if self.finger_mask is None:                     # training: sim segmentation
            return np.isin(cl["seg"], self.robot_ids)
        if is_wrist:                                     # test time: fixed finger mask on the wrist image
            return self.finger_mask.reshape(-1) if self.finger_mask.size == cl["seg"].size else \
                np.zeros(cl["seg"].size, bool)
        pts = cl["pts"]                                  # agentview: the hand and the arm above the grasp point
        near = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]) < 0.06
        return (near & (pts[:, 2] > p[2] - 0.015)) | (pts[:, 2] > p[2] + 0.05)

    def _box_raw(self, tgt_agent, size):
        """Agentview box of the target (raw layout): the agent's detect box (moondream), else the target's
        segmentation box jittered by box_jitter of its size (a stand-in for a detector's box)."""
        gi = self.hl.gate_inputs()
        if gi is not None and gi.get("box") is not None:
            return box_to_raw(gi["box"], size)
        rr, cc = np.divmod(np.flatnonzero(tgt_agent), size)
        c0, c1, r0, r1 = cc.min(), cc.max(), rr.min(), rr.max()
        j = self.cfg.get("box_jitter", 0.05)
        w, h = c1 - c0 + 1, r1 - r0 + 1
        d = self.rng.uniform(-j, j, 4) * np.array([w, h, w, h])
        return c0 + d[0], r0 + d[1], c1 + d[2], r1 + d[3]

    def observe(self, force=False):
        """Heightmap observation at the current state, or None if the target is not grounded (agentview sees fewer
        than min_mask_px of its pixels; force=True observes anyway, for inspection)."""
        t0 = time.time()
        env, obs, T = self.env, self.obs, self.T
        p = np.array(obs["robot0_eef_pos"], dtype=np.float64)
        R = T.quat2mat(obs["robot0_eef_quat"])
        _, psi = HMAP.level_frame(R)
        a_px, w_px = int(self.cfg.get("agent_px") or self.res), int(self.cfg.get("wrist_px") or self.res)
        cl_a = camera_cloud(env, AGENT_CAMERA, a_px, obs["agentview_image"] if a_px == self.res else None)
        tgt_a = np.isin(cl_a["seg"], self.geom_ids)
        if tgt_a.sum() < self.cfg.get("min_mask_px", 20) and not force:
            return None
        cl_w = camera_cloud(env, WRIST_CAMERA, w_px, obs["robot0_eye_in_hand_image"] if w_px == self.res else None)
        pts, rgb, tgt, wrist, n_robot = [], [], [], [], 0
        for cl, is_wrist in ((cl_a, False), (cl_w, True)):
            robot = self._robot_points(cl, is_wrist, p)
            keep = ~robot & (cl["seg"] >= 0)
            n_robot += int(robot.sum())
            pts.append(cl["pts"][keep])
            rgb.append(cl["rgb"][keep])
            tgt.append(np.isin(cl["seg"][keep], self.geom_ids))
            wrist.append(np.full(int(keep.sum()), is_wrist))
        pts, rgb, tgt_seg, wrist = np.concatenate(pts), np.concatenate(rgb), np.concatenate(tgt), np.concatenate(wrist)
        z_top = float(np.percentile(pts[tgt_seg, 2], 95)) if tgt_seg.any() else \
            float(P.obj_pos(env, self.obj)[2] + self.top_offset)
        src = self.cfg.get("mask_source", "seg")
        if src == "mix":
            src = "box" if self.rng.random() < self.cfg.get("mix_p", 0.5) else "seg"
        target = tgt_seg
        if src == "box" and tgt_a.any():
            c0, r0, c1, r1 = self._box_raw(tgt_a, a_px)
            c, r, front = project_raw(env, pts, a_px)
            inbox = front & (c >= c0) & (c <= c1) & (r >= r0) & (r <= r1)
            floor = float(np.percentile(pts[inbox, 2], 10)) if inbox.any() else -np.inf
            target = inbox & (pts[:, 2] >= floor + 0.005)
        raster = HMAP.rasterize(pts, rgb, target, wrist, p[:2], psi)
        if self.cfg.get("fill_holes", True):
            raster = HMAP.fill_holes(raster)
        ch = HMAP.channels(raster, p[2])
        q = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
        tilt = HMAP.tilt_rad(R)
        ch["proprio"] = np.array([np.clip((q[0] - q[1]) / FULL_GAP_M, 0, 1.5), np.sin(tilt), np.cos(tilt),
                                  np.clip((p[2] - z_top) / 0.1, -1.5, 3.0)], np.float32)
        ch["valid"] = reach_mask(self.reach)
        self._frame = {"p_xy": p[:2].copy(), "psi": psi, "z": raster["z"], "z_grasp": float(p[2]), "z_top": z_top,
                       "raster": raster, "mask_source": src, "agent_target_px": int(tgt_a.sum()),
                       "robot_px": n_robot, "t_obs_s": time.time() - t0}
        return ch

    def tick_info(self):
        obs = self.obs
        ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        tpos = P.obj_pos(self.env, self.obj)
        q = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
        return {"tilt_deg": SC.approach_tilt_deg(obs["robot0_eef_quat"], self.T),
                "target_dxy": float(np.hypot(*(ee[:2] - tpos[:2]))), "gap": float(q[0] - q[1]),
                "above_top": float(ee[2] - tpos[2] - self.top_offset), "obj_tilt_deg": self.obj_tilt_deg()}

    # ---- snapshot (sec. 9.3)
    def snapshot(self):
        sr = common.tools_module("scene_restore")
        c = self.check
        return {"env": sr.capture_snapshot(self.env), "obs": {k: np.array(v, copy=True) for k, v in self.obs.items()},
                "steps": self.steps, "plan": [np.array(a, copy=True) for a in self.plan],
                "last_action": self.last_action.copy(), "check": (c.count, c.done, c.done_step, c.z0),
                "task_done": self._task_done, "sim_error": self.sim_error, "fail_reason": self.fail_reason,
                "stop_reason": self.stop_reason, "prompt": self.prompt,
                "n_frames": None if self.frames is None else len(self.frames)}

    def restore(self, snap):
        sr = common.tools_module("scene_restore")
        sr.restore_snapshot(self.env, snap["env"])
        self.obs = {k: np.array(v, copy=True) for k, v in snap["obs"].items()}
        self.steps = snap["steps"]
        self.plan = deque(np.array(a, copy=True) for a in snap["plan"])
        self.last_action = snap["last_action"].copy()
        self.check.count, self.check.done, self.check.done_step, self.check.z0 = snap["check"]
        self._task_done, self.sim_error, self.fail_reason = snap["task_done"], snap["sim_error"], snap["fail_reason"]
        self.stop_reason, self.prompt = snap["stop_reason"], snap["prompt"]
        if self.frames is not None and snap["n_frames"] is not None:
            del self.frames[snap["n_frames"]:]

    # ---- status
    def pick_done(self):
        return bool(self.check.done)

    def pick_step(self):
        return self.check.done_step

    def task_done(self):
        return self._task_done

    def failure(self):
        return ("sim_error: " + self.sim_error) if self.sim_error else self.fail_reason

    def agent_stop(self):
        return self.stop_reason

    def summary(self):
        return {"object": self.obj, "prompt": self.prompt, "suite": self.suite, "task_id": self.task_id,
                "budget": self.budget, "hl": self.hl.name, "obj_tilt_deg": round(self.obj_tilt_deg(), 1)}

    # ---- inspection (train.py --mode inspect)
    def scripted_pose(self, dx=0.0, dy=0.0, dz=0.07, tilt_deg=0.0, yaw_deg=0.0):
        """Move the open gripper (no pi0) to dz above the target's top + (dx, dy), levelled then tilted by tilt_deg
        about the world x axis and turned by yaw_deg. Returns the MoveTo's describe()."""
        goal = self.target_approach_point() + np.array([dx, dy, dz - HL.APPROACH_DZ])
        q_level = SC.top_down_quat(np.array(self.obs["robot0_eef_quat"], dtype=np.float64), self.T)
        R = SC.rotz(yaw_deg) @ rotx(tilt_deg) @ self.T.quat2mat(q_level)
        ee = np.array(self.obs["robot0_eef_pos"], dtype=np.float64)
        legs = [motion.MoveTo([ee[0], ee[1], max(ee[2], goal[2])], common.GRIPPER_OPEN, q_level),
                motion.MoveTo([goal[0], goal[1], max(ee[2], goal[2])], common.GRIPPER_OPEN, q_level),
                motion.MoveTo(goal, common.GRIPPER_OPEN, self.T.mat2quat(R))]
        for m in legs:
            self._run_motion(m)
        return legs[-1].describe()

    def inspect_info(self):
        """The last observation's frame + its cell centres projected into the upright agentview image."""
        fr = dict(self._frame)
        cells = HMAP.cell_world_grid(fr["p_xy"], fr["psi"]).reshape(-1, 2)
        z = np.full((len(cells), 1), fr["z_top"])
        uv = [camera.project(self.env, np.r_[xy, zz], self.res) for xy, zz in zip(cells, z)]
        fr["cells_agent_px"] = np.array(uv).reshape(GRID, GRID, 2)
        fr["agent_rgb"] = common.upright(self.obs["agentview_image"])
        fr["wrist_rgb"] = common.upright(self.obs["robot0_eye_in_hand_image"])
        fr["grasp_agent_px"] = camera.project(self.env, np.r_[fr["p_xy"], fr["z_grasp"]], self.res)
        fr["object"], fr["prompt"], fr["robot_geoms"] = self.obj, self.prompt, int(len(self.robot_ids))
        return fr


# ------------------------------------------------------------------ checks / tools -----------------

def _random_actions(rng, n, open_frac=0.9):
    return np.c_[rng.uniform(-0.4, 0.4, (n, 3)), rng.uniform(-0.15, 0.15, (n, 3)),
                 np.where(rng.random(n) < open_frac, -1.0, 1.0)]


def restore_check(cfg, start, before=40, after=60, seed=0):
    """Backend-level snapshot check (no pi0): scripted random actions, snapshot, continue (A), restore, continue
    with the same actions (B), restore again and observe. A and B must match exactly: sim state, step counter,
    PickCheck state, observation channels. Returns a report dict."""
    env = BranchingPickEnv(dict(cfg, hl="none"))
    rng = np.random.default_rng(seed)
    try:
        env.reset(start)
        acts = _random_actions(rng, before + after)
        for a in acts[:before]:
            env._env_step(a)
        o0 = env.observe(force=True)
        snap = env.snapshot()
        env.plan.extend([np.zeros(7)] * 3)               # junk in the queue: restore must remove it

        def run():
            rec = []
            for a in acts[before:]:
                env._env_step(a)
                rec.append((env.steps, np.array(env.env.get_sim_state(), copy=True), env.check.count,
                            env.check.z0, env.obs["agentview_image"].astype(np.int16)))
            return rec

        A = run()
        env.restore(snap)
        queue_ok = len(env.plan) == 0
        B = run()
        env.restore(snap)
        o1 = env.observe(force=True)
    finally:
        env.close()
    rep = {"steps_equal": all(a[0] == b[0] for a, b in zip(A, B)),
           "state_max_abs_err": float(max(np.abs(a[1] - b[1]).max() for a, b in zip(A, B))),
           "check_equal": all(a[2] == b[2] and a[3] == b[3] for a, b in zip(A, B)),
           "pix_max_abs_err": int(max(np.abs(a[4] - b[4]).max() for a, b in zip(A, B))),
           "queue_restored": queue_ok,
           "obs_equal": all(np.array_equal(o0[k], o1[k]) for k in ("rgb", "height", "mask", "observed", "proprio"))}
    ok = rep["steps_equal"] and rep["state_max_abs_err"] <= 1e-9 and rep["check_equal"] and \
        rep["pix_max_abs_err"] == 0 and rep["queue_restored"] and rep["obs_equal"]
    rep.update(verdict="PASS" if ok else "FAIL", tag=start["tag"])
    return rep


def build_finger_mask(cfg, starts, out_path, steps=60, every=4, min_frac=0.5):
    """Fixed wrist-image finger mask for --robot-mask fixed (sec. 5 step 2, test time): the wrist pixels that show
    the robot in >= min_frac of frames, with the gripper open, over scripted random motions from each start.
    Saved as a raw-layout (wrist_px, wrist_px) bool .npy. Returns the mask."""
    env = BranchingPickEnv(dict(cfg, hl="none"))
    rng = np.random.default_rng(cfg.get("seed", 0))
    counts, n = None, 0
    try:
        for s in starts:
            try:
                env.reset(s)
            except IneligibleStart:
                continue
            size = int(cfg.get("wrist_px") or env.res)
            for t, a in enumerate(_random_actions(rng, steps, open_frac=1.0)):
                env._env_step(a)
                if t % every == 0:
                    seg = SC.segmentation_geom_ids(env.env, size, WRIST_CAMERA)
                    hit = np.isin(seg, env.robot_ids)
                    counts = hit.astype(np.int64) if counts is None else counts + hit
                    n += 1
    finally:
        env.close()
    if not n:
        raise SystemExit("no frames collected")
    mask = counts / float(n) >= min_frac
    np.save(out_path, mask)
    return mask
