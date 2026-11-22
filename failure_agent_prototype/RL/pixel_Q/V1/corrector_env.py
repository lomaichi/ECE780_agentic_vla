"""LIBERO environment for the rl/pick pixel-Q corrector. One EPISODE = one correction decision.

The Q-map values handing off to pi0. Every episode starts from the same HANDOFF POSE: the VLM agent
brings the open gripper 7 cm (handoff_dz) above the target object, over the centre of its segmentation
box, pointing straight down (-z), and hands over to the Q-map. The Q-map only chooses a horizontal move
and a yaw (z is never chosen); then pi0 takes over and the subtask is judged within a timeout.

Episodes come in CHAINS on one scene:
  reset(start)  restore an upright start (a library rl/pick reset state, or a task's init state after the
                10 wait steps), HANDOFF move, observe -> obs of the chain's first episode
  step(a)       one episode. a = (rotation bin k, cell row, cell col) as a flat index (qnet.encode_action):
                  1. back-project the cell to a 3D point: the mean of its object pixels, or for a cell
                     with none, the ray through its centre cut at the object's median height
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

Views: the HANDOFF move (the VLM agent's stand-in) reads the third-person agentview; the Q-map
observation, its mask and its cells are on the first-person WRIST view taken at the handoff pose
(camera looking straight down, ~17 cm above the object's top). pi0 still gets both views as always.

HANDOFF move (_handoff): above the centre of the object's segmentation box, handoff_dz above the
object's top, approach axis straight down, current yaw kept, gripper open (optional xy jitter,
handoff_jitter_cm, default 0). Only mask + depth + camera geometry are used. Path: rise to the target
height (if below it), travel, descend.

Observation (dict, compact, what the Q-net and the replay buffer see):
  rgb      (80, 80, 3) uint8   crop of the upright WRIST camera image (robot0_eye_in_hand) centred on
                               the object's box: a wrist_box (150) px box of the 256 px image, rendered
                               at 137 px so the crop is 80x80 (one cell ~2 cm on the object); outside
                               the mask grey
  depth    (80, 80) float16    (depth - median object depth) / 0.1 m, clipped to [-1.5, 1.5]
  mask     (80, 80) bool       target object
  proprio  (6,) float32        EE - object centre (dm, world xyz), gripper opening (0..1), EE yaw sin/cos
  valid    (10, 10) bool       8x8 cells the Q-map may pick: the whole crop by default (min_cell_frac 0);
                               min_cell_frac > 0 keeps only cells with that fraction of object pixels

PRIVILEGED (training only): the object mask comes from MuJoCo's segmentation render (stand-in for the
segmentation model of pixel_Q_RL.md sec. 2), and reward / termination read the sim state through
fa_core.predicates (as pixel_Q_RL.md "Reward source" allows). Depth and camera geometry are what a
test-time agent has (fa_core.camera). The Q-net input itself is RGB-D + mask + proprio only.

Start states, geometry, the pi0 client and the env base class (LiberoScene) live in ../shared/scene.py
(shared with V2) and are re-exported here under their V1 names.
"""

import pathlib
import sys
from collections import deque

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for _p in (HERE.parent, HERE.parents[2]):  # pixel_Q/ (the shared package), failure_agent_prototype/ (fa_core)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from fa_core import camera, common, motion, predicates as P  # noqa: E402
from shared.scene import (  # noqa: E402
    DEFAULT_TASKS, DROP_M, WORKSPACE_M, WRIST_CAMERA, IneligibleStart, LiberoScene, ResetFailed, approach_tilt_deg,
    is_round_rim, library_starts, mask_box_centre_top, object_geom_ids, pixels_to_world, ray_at_height, rotz,
    segmentation_geom_ids, task_starts, top_down_quat, video_frame)

from qnet import CELL, CROP, GRID, PROPRIO_DIM, ROT_DEG, decode_action  # noqa: E402

_REEXPORTED = (DEFAULT_TASKS, library_starts, task_starts)   # train_dqn.py imports these from here

DEPTH_SCALE = 0.1               # m per unit of the depth channel
WRIST_BOX = 150                 # Q-map region: a 150x150 box of the 256 px wrist image (object box ~110x100)


def proprio_features(obs, center, T):
    ee = np.asarray(obs["robot0_eef_pos"], float)
    q = np.asarray(obs["robot0_gripper_qpos"], float)
    R = T.quat2mat(obs["robot0_eef_quat"])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    feat = np.r_[(ee - center) * 10.0, np.clip((q[0] - q[1]) / 0.08, 0, 1.5), np.sin(yaw), np.cos(yaw)]
    assert feat.shape == (PROPRIO_DIM,)
    return feat.astype(np.float32)


# ------------------------------------------------------------------ env ----------------------------

class PickCorrectorEnv(LiberoScene):
    """LiberoScene (shared/scene.py) gives _pi0, _ensure_env, _resolve_target, _obj_up, obj_tilt_deg, close."""

    def __init__(self, cfg, client=None):
        """cfg: timeout_s, handoff_dz, handoff_jitter_cm, min_cell_frac, min_mask_px, env_seed, seed, host, port."""
        super(PickCorrectorEnv, self).__init__(cfg, client)
        self.frames = None

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
        self.frames = [video_frame(obs)] if record else None
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

    # ---- observation
    def _render(self, cam=camera.CAMERA, size=None):
        """(upright rgb, upright metric depth, upright object mask) of the current state seen by `cam` at
        size x size px (default: agentview at the env resolution)."""
        env, size = self.env, size or self.res
        if cam == camera.CAMERA and size == self.res:
            rgb = common.upright(self.obs["agentview_image"])
        else:   # sim.render returns the same (raw, upside-down) layout as the obs images
            rgb = common.upright(np.asarray(env.env.sim.render(width=size, height=size, camera_name=cam)))
        depth = common.upright(camera.depth_image(env, size, cam))
        mask_full = np.isin(common.upright(segmentation_geom_ids(env, size, cam)), self.geom_ids)
        return rgb, depth, mask_full

    def _wrist_res(self):
        """Wrist render size whose 80x80 crop spans a wrist_box x wrist_box box of the full-res wrist image
        (256 * 80 / 150 -> 137 px). Rendering small keeps the pixel <-> 3D geometry exact (no resampling)."""
        box = float(self.cfg.get("wrist_box", WRIST_BOX))
        n = int(round(self.res * CROP / box))
        if n < CROP:
            raise ValueError("wrist_box {} is larger than the {} px wrist image".format(box, self.res))
        return n

    def _handoff(self):
        """Gripper open, pointing straight down (current yaw kept), above the centre of the object's
        segmentation box, handoff_dz (7 cm) above the object's top. Returns (stop, info).

        Box centre -> world: the ray through the box-centre pixel, cut at the object's median height (the
        box centre need not lie on the object, e.g. a bowl on its side). Top = 95th percentile height of
        the object's pixels. Only mask + depth + camera geometry are used, as at test time."""
        _, depth, mask_full = self._render()
        if mask_full.sum() < self.cfg.get("min_mask_px", 20):
            return True, {"error": "object not visible"}
        centre, top = mask_box_centre_top(self.env, depth, mask_full, self.res)
        jit = self.rng.uniform(-1, 1, 2) * self.cfg.get("handoff_jitter_cm", 0.0) / 100.0
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
        """Q-map observation from the WRIST camera (first person) at the handoff pose."""
        env, size, cam = self.env, self._wrist_res(), WRIST_CAMERA
        rgb, depth, mask_full = self._render(cam, size)
        if mask_full.sum() < self.cfg.get("min_mask_px", 20):
            return None
        vs, us = np.nonzero(mask_full)                     # crop centred on the segmentation box
        u0 = int(np.clip(round(0.5 * (us.min() + us.max()) - CROP / 2), 0, size - CROP))
        v0 = int(np.clip(round(0.5 * (vs.min() + vs.max()) - CROP / 2), 0, size - CROP))
        win = (slice(v0, v0 + CROP), slice(u0, u0 + CROP))
        m, d = mask_full[win], depth[win]
        vv, uu = np.mgrid[v0:v0 + CROP, u0:u0 + CROP]
        world = pixels_to_world(env, uu, vv, d, size, cam)
        cells_m = m.reshape(GRID, CELL, GRID, CELL)
        frac = cells_m.mean(axis=(1, 3))
        valid = frac >= self.cfg.get("min_cell_frac", 0.0)
        if not valid.any():
            return None
        cnt = cells_m.sum(axis=(1, 3))[..., None]
        wsum = (world * m[..., None]).reshape(GRID, CELL, GRID, CELL, 3).sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            cell_points = wsum / cnt                       # mean surface point of the cell's object pixels
        # cells with no object pixels: the ray through the cell centre cut at the object's median height
        # (same rule as the handoff's box centre). Their own depth may hit the table, the fingers or a
        # neighbouring object, none of which says where the gripper should go.
        empty = cnt[..., 0] == 0
        if empty.any():
            rr, cc = np.nonzero(empty)
            cell_points[rr, cc] = ray_at_height(env, u0 + cc * CELL + CELL / 2 - 0.5, v0 + rr * CELL + CELL / 2 - 0.5,
                                                float(np.median(world[m][:, 2])), size, cam)
        center = world[m].mean(axis=0)
        crop_rgb = rgb[win].copy()
        crop_rgb[~m] = 128
        dfeat = np.clip((d - np.median(d[m])) / DEPTH_SCALE, -1.5, 1.5)
        self._frame = {"u0": u0, "v0": v0, "cell_points": cell_points, "center": center, "valid": valid,
                       "cell_frac": frac,
                       "rgb_full": rgb, "mask_full": mask_full, "tilt_deg": self.obj_tilt_deg()}
        return {"rgb": crop_rgb, "depth": dfeat.astype(np.float16), "mask": m.copy(),
                "proprio": proprio_features(self.obs, center, self.T), "valid": valid}

    def inspect_info(self):
        """Current frame + every valid cell's 3D point projected back to the wrist image (should hit the
        cell) and into agentview (cells on the object should land on the agentview object mask)."""
        fr = dict(self._frame)
        rows, cols = np.nonzero(fr["valid"])
        fr["cells"] = [(int(r), int(c)) for r, c in zip(rows, cols)]
        fr["cell_centres_px"] = [(fr["u0"] + c * CELL + CELL / 2 - 0.5, fr["v0"] + r * CELL + CELL / 2 - 0.5)
                                 for r, c in fr["cells"]]
        fr["cell_points_px"] = [tuple(float(x) for x in camera.project(self.env, fr["cell_points"][r, c],
                                                                        self._wrist_res(), WRIST_CAMERA))
                                for r, c in fr["cells"]]
        fr["agent_rgb"], _, fr["agent_mask"] = self._render()
        fr["cell_points_agent_px"] = [tuple(float(x) for x in camera.project(self.env, fr["cell_points"][r, c],
                                                                              self.res))
                                      for r, c in fr["cells"]]
        fr["cell_obj_frac"] = [float(fr["cell_frac"][r, c]) for r, c in fr["cells"]]
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
            self.frames.append(video_frame(self.obs))
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
            raise ValueError("action {} is not a valid cell (min_cell_frac)".format(action))
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
