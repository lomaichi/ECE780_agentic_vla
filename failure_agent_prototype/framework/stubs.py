"""Offline stand-ins, for testing the pipeline without the pi0 server or the GPU. PRIVILEGED (they read the sim
state); never use them to report results.

  OracleDetectVLM   replaces vlm_backends.MoondreamLocal: detect(phrase) returns the agentview segmentation boxes
                    of the sim objects whose names best match the phrase. The GroundedRecoveryAgent on top of it is
                    unchanged (same rules, depth back-projection, tracking).
  ScriptedPi0       replaces the pi0 websocket client: infer() returns a chunk of P-control actions toward a
                    waypoint relative to the target object, chosen to exercise the pipeline's branches:
                      hover   open gripper 3-4 cm beside / 8 cm above the target (pregrasp hesitation, LL-8):
                              V4's gate opens -> V branch -> Q corrections; timer2 timeout
                      wander  heads 25 cm away from the target: HL target_not_approached -> point_approach
                      idle    zero motion: HL stall -> point_approach, then the V branch near the target
"""

import time

import numpy as np

import config as C

C.ensure_paths()
from fa_core import common, predicates as P  # noqa: E402

_STOP = {"the", "of", "a", "an", "on", "in", "at", "to", "top", "side", "and"}


class OracleDetectVLM(object):
    """detect / point / ask with MoondreamLocal's signatures, answered from the sim (segmentation render)."""

    def __init__(self, env, size=common.ENV_RESOLUTION):
        self.env, self.size = env, size
        self.system, self.model = "", "oracle"
        self.calls, self.cost_usd, self.latency_s = 0, 0.0, 0.0
        self.last_server = {}

    def _geoms(self, name):
        e = self.env.env
        obj = dict(getattr(e, "fixtures_dict", {}), **getattr(e, "objects_dict", {})).get(name)
        ids = set()
        for g in list(getattr(obj, "contact_geoms", [])) + list(getattr(obj, "visual_geoms", [])):
            try:
                ids.add(e.sim.model.geom_name2id(g))
            except Exception:
                pass
        return np.array(sorted(ids))

    def detect(self, img, phrase):
        from shared import scene as SC

        t0 = time.time()
        e = self.env.env
        names = list(getattr(e, "objects_dict", {})) + list(getattr(e, "fixtures_dict", {}))
        toks = P._tokens(phrase) - _STOP
        score = {n: len(toks & P._tokens(n)) for n in names}
        best = max(score.values()) if score else 0
        boxes = []
        if best > 0:
            seg = common.upright(SC.segmentation_geom_ids(self.env, self.size))
            for n in sorted(k for k, s in score.items() if s == best):
                ids = self._geoms(n)
                if not len(ids):
                    continue
                vs, us = np.nonzero(np.isin(seg, ids))
                if len(us) < 5:
                    continue
                boxes.append({"x_min": us.min() / float(self.size), "y_min": vs.min() / float(self.size),
                              "x_max": (us.max() + 1) / float(self.size), "y_max": (vs.max() + 1) / float(self.size)})
        self.calls += 1
        self.latency_s += time.time() - t0
        return boxes

    def point(self, img, phrase):
        self.calls += 1
        return []          # "open drawer": no point -> the agent falls back to the destination box

    def ask(self, parts, schema):
        self.calls += 1
        return {"answer": "no"}


class ScriptedPi0(object):
    """pi0 client stand-in: infer(request) -> {"actions": (10, 7)} toward a waypoint near the target."""

    OFFSETS = {"hover": (0.03, 0.025, 0.08), "wander": (0.0, 0.25, 0.15)}

    def __init__(self, env, obj, mode="hover", chunk=10):
        if mode not in ("hover", "wander", "idle"):
            raise ValueError("ScriptedPi0 mode must be hover, wander or idle")
        self.env, self.obj, self.mode, self.chunk = env, obj, mode, chunk
        self.calls = 0

    def infer(self, req):
        self.calls += 1
        ee = np.asarray(req["observation/state"][:3], dtype=np.float64)
        if self.mode == "idle":
            a = np.r_[np.zeros(6), common.GRIPPER_OPEN]
        else:
            wp = P.obj_pos(self.env, self.obj) + np.asarray(self.OFFSETS[self.mode])
            a = np.r_[np.clip(10.0 * (wp - ee), -1, 1), np.zeros(3), common.GRIPPER_OPEN]
        return {"actions": np.tile(a, (self.chunk, 1))}
