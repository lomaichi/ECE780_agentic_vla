"""Synthetic stand-in for libero_env.BranchingPickEnv: same backend interface (rollout.py), no LIBERO, no pi0.

Use it to check the V2 code on CPU (train.py --env toy) before spending simulator time. Units are metres and
20 Hz control steps, like the real env, so the label window, ticks, cooldowns and caps mean the same thing.

The world: a round bowl (radius 4-6 cm, rim 5 cm high) on a table, plus a distractor block. A scripted "pi0"
approaches a rim point at hover height, aligns its yaw, descends, closes, lifts and holds (the pick predicate
fires after the 10-step hold, as PickCheck does). The grasp works when the gripper is on the rim with the
finger axis (the gripper's x axis, i.e. the map's columns) radial to the bowl. Per rollout pi0 has one mode:
  good        right rim point, aligned yaw                         -> succeeds on its own
  offset      rim point offset by 2.5-4 cm                         -> misses, retries the same way
  yaw         yaw error of 15-25 deg                               -> grasp works ~20-55 % of the time
  hesitate    stalls 1-3 s above the rim before descending          -> slow
  distractor  heads for the distractor (an HL failure)             -> the HL stand-in re-aims it at the bowl
A correction (cell + yaw) moves the gripper; pi0 then replans: from a rim point it descends right there with its
current yaw, otherwise it re-approaches its own (biased) rim point. So a correction onto the rim with a yaw that
makes the fingers radial fixes offset / yaw / hesitate states, and interrupting a good approach only costs time:
A(s) > 0, ~0 and < 0 states all occur, as in sec. 9.1. The right cell and yaw are visible in the yaw-aligned
heightmap (where the rim is relative to the gripper). The approach tilt does not affect the grasp, but levelling
takes time (3 deg per step) and the hand-back makes pi0 replan, so the levelling probe of phase 0 is slightly worse
than the no-op, more so at high tilt: the phase 0 report should flag that, which is what it is for.

Video (reset(record=True), train.py --video): one top-down frame per control step of the main rollout
(render_frame); branch frames are cut on restore, as in the LIBERO env.
"""

import copy

import numpy as np

import heightmap as HMAP
from qnet import CELL_M, HM, ROT_DEG, decode_action, reach_mask

RIM_H = 0.05                  # bowl rim height above the table
FLOOR_H = 0.01                # bowl interior floor
HOVER_DZ = 0.06               # pi0 hovers this far above the rim before descending
GRASP_Z = RIM_H - 0.015       # grasp point height when closing on the rim
LIFT_M, HOLD = 0.05, 10       # PickCheck: lifted 5 cm, held 10 steps
V_XY, V_Z, V_YAW, V_TILT = 0.008, 0.006, np.radians(4.0), np.radians(3.0)   # per control step
V_XY_PRIM = 0.010             # primitive (MoveTo) speed
RIM_TOL = 0.012               # "on the rim" for pi0's replanning and for a good grasp
MODES = (("good", 0.45), ("offset", 0.2), ("yaw", 0.15), ("hesitate", 0.1), ("distractor", 0.1))
BUDGET = 220                  # main rollout step budget (libero_spatial's)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _axis_err(yaw, radial):
    """Angle between a finger axis (undirected) and a radial direction, in [0, pi/2]."""
    return abs(_wrap(2 * (yaw - radial)) / 2.0)


def _toward(x, target, v):
    d = target - x
    n = float(np.linalg.norm(d))
    return target.copy() if n <= v else x + d * (v / n)


def _toward_angle(a, target, v):
    d = _wrap(target - a)
    return target if abs(d) <= v else a + np.sign(d) * v


class ToyBranchEnv(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.get("seed", 0) + 1000 * cfg.get("worker_id", 0))
        self.reach = int(cfg.get("reach", 3))
        self.gate_m = self.reach * CELL_M + cfg.get("gate_margin_m", 0.04)
        self.hl_mode = cfg.get("hl", "privileged")
        self.same_noise = bool(cfg.get("same_noise", False))
        self.frames = None
        self.budget = BUDGET
        self.s = None

    # ---- rollout
    def reset(self, start, record=False):
        r = self.rng
        mode = MODES[r.choice(len(MODES), p=[p for _, p in MODES])][0]
        ang = r.uniform(-np.pi, np.pi)
        dist = r.uniform(0.15, 0.25)
        dang = ang + r.choice([-1, 1]) * r.uniform(0.6, 1.2)
        self.s = {
            "step": 0, "mode": mode, "R": r.uniform(0.04, 0.06), "b": np.zeros(2),
            "dist_xy": np.array([np.cos(dang), np.sin(dang)]) * r.uniform(0.18, 0.25),
            "g": np.r_[np.cos(ang) * dist, np.sin(ang) * dist, 0.20], "yaw": r.uniform(-np.pi, np.pi),
            "tilt": 0.0, "tilt_target": np.radians(r.uniform(0, 35)), "grip": -1.0, "phase": "approach",
            "timer": 0, "hesitate": int(r.integers(20, 60)) if mode == "hesitate" else 0,
            "offset": np.zeros(2), "yaw_err": 0.0, "grasped": False, "z_lift0": None, "pick_step": None,
            "heading_distractor": mode == "distractor", "fail": None, "hl_count": 0, "hl_pending": 0,
            "hl_grace_until": 20, "colour": r.integers(40, 220, 3), "last_grip": -1.0,
        }
        s = self.s
        if mode == "offset":
            s["offset"] = r.uniform(0.025, 0.04) * np.array([np.cos(r.uniform(0, 2 * np.pi)),
                                                             np.sin(r.uniform(0, 2 * np.pi))])
        if mode == "yaw":
            s["yaw_err"] = np.radians(r.uniform(15, 25)) * r.choice([-1, 1])
        self._replan(after_correction=False)
        self.frames = [] if record else None
        self._frame = None
        self._record("pi0")

    @property
    def step_count(self):
        return self.s["step"]

    def set_hl(self, name):
        self.hl_mode = name

    def _replan(self, after_correction):
        s, b, R = self.s, self.s["b"], self.s["R"]
        gxy = s["g"][:2]
        if s["heading_distractor"]:
            s["plan_xy"], s["plan_yaw"], s["phase"] = s["dist_xy"].copy(), s["yaw"], "approach"
            return
        rad = float(np.arctan2(gxy[1] - b[1], gxy[0] - b[0]))
        if after_correction and abs(np.linalg.norm(gxy - b) - R) < RIM_TOL and s["g"][2] < RIM_H + 2 * HOVER_DZ:
            s["plan_xy"], s["plan_yaw"], s["phase"] = gxy.copy(), s["yaw"], "descend"   # descend right here
            return
        base = b + R * np.array([np.cos(rad), np.sin(rad)])
        s["plan_xy"] = base + s["offset"]
        s["plan_yaw"] = rad + s["yaw_err"]
        s["phase"] = "approach"

    def _jitter(self):
        self.s["g"][:2] += self.rng.normal(0, 0.0007, 2)

    def _advance(self):
        """One control step of the scripted pi0."""
        s = self.s
        ph, hz = s["phase"], RIM_H + HOVER_DZ
        if ph == "approach":
            s["g"][:2] = _toward(s["g"][:2], s["plan_xy"], V_XY)
            s["g"][2] = float(_toward(np.array([s["g"][2]]), np.array([hz]), V_Z)[0])
            s["yaw"] = _toward_angle(s["yaw"], s["plan_yaw"], V_YAW)
            s["tilt"] = _toward_angle(s["tilt"], s["tilt_target"], V_TILT)
            self._jitter()
            if np.linalg.norm(s["g"][:2] - s["plan_xy"]) < 0.004 and abs(s["g"][2] - hz) < 0.004:
                if s["heading_distractor"]:
                    s["phase"] = "descend"
                else:
                    s["phase"] = "hesitate" if s["hesitate"] > 0 else "descend"
        elif ph == "hesitate":
            s["hesitate"] -= 1
            self._jitter()
            if s["hesitate"] <= 0:
                s["phase"] = "descend"
        elif ph == "descend":
            s["g"][2] = float(_toward(np.array([s["g"][2]]), np.array([GRASP_Z]), V_Z)[0])
            s["yaw"] = _toward_angle(s["yaw"], s["plan_yaw"], V_YAW)
            if abs(s["g"][2] - GRASP_Z) < 1e-6:
                s["phase"], s["timer"], s["grip"] = "close", 5, 1.0
        elif ph == "close":
            s["timer"] -= 1
            if s["timer"] <= 0:
                if not s["heading_distractor"] and self.rng.random() < self._grasp_p():
                    s["grasped"], s["z_lift0"], s["phase"] = True, s["g"][2], "lift"
                else:
                    s["phase"], s["timer"], s["grip"] = "open", 5, -1.0
        elif ph == "lift":
            s["g"][2] += 0.005
            if s["g"][2] - s["z_lift0"] >= LIFT_M:
                s["phase"], s["timer"] = "hold", HOLD
        elif ph == "hold":
            s["timer"] -= 1
            if s["timer"] <= 0 and s["pick_step"] is None:
                s["pick_step"], s["phase"] = s["step"] + 1, "done"
        elif ph == "open":
            s["timer"] -= 1
            if s["timer"] <= 0:
                s["phase"] = "retreat"
        elif ph == "retreat":
            s["g"][2] = float(_toward(np.array([s["g"][2]]), np.array([hz]), V_Z)[0])
            if abs(s["g"][2] - hz) < 1e-6:
                self._replan(after_correction=False)
        s["step"] += 1
        self._record("pi0")

    def _grasp_p(self):
        s, b, R = self.s, self.s["b"], self.s["R"]
        gxy = s["g"][:2]
        d_r = abs(np.linalg.norm(gxy - b) - R)
        e = np.degrees(_axis_err(s["yaw"], np.arctan2(gxy[1] - b[1], gxy[0] - b[0])))
        if d_r < RIM_TOL and e < 10:
            return 0.92
        if d_r < RIM_TOL and e < 22:
            return 0.55
        if d_r < 0.02 and e < 22:
            return 0.2
        return 0.02

    def pi0_step(self):
        if self.s["fail"] is None:
            self._advance()

    # ---- primitive
    def execute_primitive(self, action, max_steps=30):
        """Clearance rise, level, then move to the cell's xy at the current height adding the bin's yaw (one
        motion), all within max_steps; then pi0 replans (sec. 6)."""
        s, fr = self.s, self._frame
        k, i, j = decode_action(action)
        txy = HMAP.cell_to_world(i, j, fr["p_xy"], fr["psi"])
        tyaw = fr["psi"] + np.radians(ROT_DEG[k])
        h_clear = HMAP.clearance_height(fr["z"], i, j)
        z_goal = s["g"][2]
        if h_clear is not None and s["g"][2] < h_clear + 0.02:
            z_goal = h_clear + 0.03
        n = {"rise": 0, "level": 0, "move": 0}
        steps, reached = 0, False
        while steps < max_steps:
            if s["g"][2] < z_goal - 1e-6:
                s["g"][2] = min(z_goal, s["g"][2] + V_Z)
                n["rise"] += 1
            elif s["tilt"] > 1e-6:
                s["tilt"] = max(0.0, s["tilt"] - V_TILT)
                n["level"] += 1
            else:
                before = np.linalg.norm(s["g"][:2] - s["b"]) - s["R"]
                s["g"][:2] = _toward(s["g"][:2], txy, V_XY_PRIM)
                s["yaw"] = _toward_angle(s["yaw"], tyaw, V_YAW)
                after = np.linalg.norm(s["g"][:2] - s["b"]) - s["R"]
                if np.sign(before) != np.sign(after) and s["g"][2] < RIM_H + 0.005 and self.rng.random() < 0.5:
                    s["fail"] = "knocked_over"                # dragged the fingers through the rim
                n["move"] += 1
                if np.linalg.norm(s["g"][:2] - txy) < 1e-6 and abs(_wrap(s["yaw"] - tyaw)) < 1e-6:
                    reached = True
            s["grip"] = -1.0
            s["step"] += 1
            self._record("prim")
            steps += 1
            if reached or s["fail"]:
                break
        s["hesitate"] = 0
        s["phase"] = "approach"
        self._replan(after_correction=True)
        return {"steps": steps, "legs": n, "reached": reached, "h_clear": h_clear, "z_goal": round(z_goal, 4),
                "target_xy": [round(float(x), 4) for x in txy], "yaw_deg": ROT_DEG[k]}

    # ---- HL stand-in (pi0 heading for the distractor)
    def hl_tick(self):
        s = self.s
        if self.hl_mode == "none" or not s["heading_distractor"] or s["hl_count"] >= 6 or \
                s["step"] < s["hl_grace_until"]:
            s["hl_pending"] = 0
            return None
        near_wrong = np.linalg.norm(s["g"][:2] - s["dist_xy"]) < 0.10
        s["hl_pending"] = s["hl_pending"] + 1 if near_wrong else 0
        if s["hl_pending"] < 3:                               # confirmed on 3 ticks (0.6 s)
            return None
        goal = np.r_[s["b"] + 0.0, RIM_H + 0.07]              # point_approach: 7 cm above the target
        n0 = s["step"]
        while np.linalg.norm(s["g"] - goal) > 1e-6 and s["step"] - n0 < 60:
            s["g"][:2] = _toward(s["g"][:2], goal[:2], V_XY_PRIM)
            s["g"][2] = float(_toward(np.array([s["g"][2]]), np.array([goal[2]]), V_Z)[0])
            s["grip"] = -1.0
            s["step"] += 1
            self._record("hl")
        s["heading_distractor"], s["hl_count"], s["hl_pending"] = False, s["hl_count"] + 1, 0
        s["hl_grace_until"] = s["step"] + 20
        if self.rng.random() < 0.3:                           # where pi0 goes from here: biased or not
            s["offset"] = 0.03 * np.array([np.cos(self.rng.uniform(0, 6.3)), np.sin(self.rng.uniform(0, 6.3))])
        s["phase"] = "approach"
        self._replan(after_correction=False)
        return {"action": "point_approach", "mode": "wrong_approach", "steps": s["step"] - n0}

    # ---- gate and observation
    def gate(self):
        s = self.s
        dxy = float(np.linalg.norm(s["g"][:2] - s["b"]))
        info = {"pick": s["pick_step"] is None, "open": s["grip"] < 0 and s["phase"] not in ("close", "open"),
                "empty": not s["grasped"], "near": dxy <= self.gate_m, "target_dxy": dxy}
        return bool(info["pick"] and info["open"] and info["empty"] and info["near"]), info

    def observe(self):
        s = self.s
        p_xy, psi, zg = s["g"][:2].copy(), float(s["yaw"]), float(s["g"][2])
        vv, uu = np.mgrid[0:HM, 0:HM] + 0.5
        xy = HMAP.map_to_world(uu, vv, p_xy, psi)
        d = np.linalg.norm(xy - s["b"], axis=-1)
        R = s["R"]
        z = np.zeros((HM, HM))
        rgb = np.empty((HM, HM, 3), np.uint8)
        rgb[:] = (175, 160, 140)
        interior, rim = d < R - 0.004, (d >= R - 0.004) & (d <= R + 0.002)
        z[interior], z[rim] = FLOOR_H, RIM_H
        rgb[interior] = np.clip(s["colour"] * 0.7, 0, 255).astype(np.uint8)
        rgb[rim] = s["colour"].astype(np.uint8)
        blk = np.max(np.abs(xy - s["dist_xy"]), axis=-1) < 0.025
        z[blk] = 0.04
        rgb[blk] = (60, 90, 200)
        observed = self.rng.random((HM, HM)) > 0.03
        raster = {"z": np.where(observed, z, np.nan), "rgb": rgb, "target": d <= R + 0.002, "observed": observed,
                  "wrist": np.ones((HM, HM), bool)}
        ch = HMAP.channels(raster, zg)
        top = RIM_H
        ch["proprio"] = np.array([1.0, np.sin(s["tilt"]), np.cos(s["tilt"]),
                                  np.clip((zg - top) / 0.1, -1.5, 3.0)], np.float32)
        ch["valid"] = reach_mask(self.reach)
        self._frame = {"p_xy": p_xy, "psi": psi, "z": raster["z"], "z_grasp": zg}
        return ch

    def tick_info(self):
        s = self.s
        return {"tilt_deg": float(np.degrees(s["tilt"])), "target_dxy": float(np.linalg.norm(s["g"][:2] - s["b"])),
                "mode": s["mode"], "phase": s["phase"]}

    # ---- snapshot
    def snapshot(self):
        snap = {"s": copy.deepcopy(self.s), "frame": copy.deepcopy(self._frame),
                "n_frames": None if self.frames is None else len(self.frames)}
        if self.same_noise:
            snap["rng"] = copy.deepcopy(self.rng.bit_generator.state)
        return snap

    def restore(self, snap):
        self.s, self._frame = copy.deepcopy(snap["s"]), copy.deepcopy(snap["frame"])
        if "rng" in snap:
            self.rng.bit_generator.state = copy.deepcopy(snap["rng"])
        if self.frames is not None and snap.get("n_frames") is not None:
            del self.frames[snap["n_frames"]:]                # branch frames are not part of the main rollout

    # ---- status
    def pick_done(self):
        return self.s["pick_step"] is not None

    def pick_step(self):
        return self.s["pick_step"]

    def task_done(self):
        return False

    def failure(self):
        return self.s["fail"]

    def agent_stop(self):
        return None

    def summary(self):
        return {"mode": self.s["mode"], "R": round(self.s["R"], 4), "hl_count": self.s["hl_count"]}

    def close(self):
        pass

    # ---- video (record=True): one frame per control step of the main rollout
    def _record(self, src):
        if self.frames is not None:
            self.frames.append(render_frame(self.s, src))


VIEW_M, VIEW_PX, BAR_PX, VIEW_Z = 0.30, 128, 16, 0.25
_VY, _VX = (np.mgrid[0:VIEW_PX, 0:VIEW_PX] + 0.5) * (2 * VIEW_M / VIEW_PX) - VIEW_M
_VY = -_VY                                                    # image row 0 = world +y
SRC_BORDER = {"pi0": None, "prim": (255, 200, 0), "hl": (230, 0, 230)}


def render_frame(s, src="pi0", scale=3):
    """Top-down toy frame (60 x 60 cm around the bowl) | gripper height bar. Bowl rim in its colour, distractor
    blue, the two fingertips along the gripper's x axis (green open, red closed, white once picked); the bar shows
    the grasp height (white) against the rim height (grey). Border: yellow = correction primitive, magenta = HL
    correction."""
    img = np.empty((VIEW_PX, VIEW_PX + BAR_PX, 3), np.uint8)
    img[:] = (205, 195, 180)
    d = np.hypot(_VX - s["b"][0], _VY - s["b"][1])
    view = img[:, :VIEW_PX]
    view[d < s["R"] - 0.004] = np.clip(s["colour"] * 0.7, 0, 255).astype(np.uint8)
    view[(d >= s["R"] - 0.004) & (d <= s["R"] + 0.002)] = s["colour"].astype(np.uint8)
    view[np.maximum(np.abs(_VX - s["dist_xy"][0]), np.abs(_VY - s["dist_xy"][1])) < 0.025] = (60, 90, 200)
    col = (255, 255, 255) if s["pick_step"] is not None else (40, 170, 40) if s["grip"] < 0 else (210, 40, 40)
    half = 0.015 if s["grip"] < 0 else 0.008
    for sgn in (-1, 1):
        f = s["g"][:2] + sgn * half * np.array([np.cos(s["yaw"]), np.sin(s["yaw"])])
        view[np.hypot(_VX - f[0], _VY - f[1]) < 0.008] = col
    bar = img[:, VIEW_PX:]
    bar[:] = (60, 60, 60)
    row = lambda z: int(np.clip(VIEW_PX - 1 - z / VIEW_Z * VIEW_PX, 0, VIEW_PX - 1))  # noqa: E731
    bar[row(RIM_H)] = (150, 150, 150)
    r = row(s["g"][2])
    bar[max(0, r - 1):r + 2, 3:-3] = (255, 255, 255)
    if SRC_BORDER.get(src):
        img[:3], img[-3:], img[:, :3], img[:, VIEW_PX - 3:VIEW_PX] = [SRC_BORDER[src]] * 4
    return np.repeat(np.repeat(img, scale, 0), scale, 1)


def toy_starts(n=8, prefix="toy"):
    return [{"kind": "toy", "tag": "{}{}".format(prefix, i)} for i in range(n)]
