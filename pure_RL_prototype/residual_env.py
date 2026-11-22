"""pi0 (frozen, black box) + a bounded per-step residual, on LIBERO pick start states.

Every executed action is

    a_t = a_pi0_t + clip(delta_t, -eps, eps)      on the dims where eps > 0, then clipped to [-1, 1]
    a_t = a_pi0_t                                 on the other dims (eps = 0 is exactly plain pi0)

eps is enforced HERE, whatever produced delta (random noise in noise_test.py, a learned policy later), so
the bound is a property of the env, not of the policy. Action units are LIBERO's OSC_POSE delta inputs:
1.0 = 5 cm (dims 0-2) or 0.5 rad (dims 3-5) of commanded motion per control step; dim 6 is the gripper.

pi0 is queried exactly as in record_suites.py / fa_core.loop (common.pi0_request, task language as the
prompt, 5 actions executed per chunk). The residual acts at every control step (20 Hz) and is given the
current observation and pi0's action for this step.

Start states (failure_agent_prototype/RL/pixel_Q/shared/scene.py):
  reset_state  library.json rl/pick reset states: pre-failure states pi0 actually reached (no wait steps)
  task_init    a task's init state, after the 10 settle steps (for the "do no harm" check)
Reward: fa_core.predicates.PickCheck on the target (grasped and lifted >= 5 cm for 10 states). An episode
also ends early when the target leaves the table or the workspace (pixel-Q's irrecoverable rule).

Run in the LIBERO client venv inside the container (MUJOCO_GL=egl), with the pi0 server up (./serve_pi0.sh).
"""

import pathlib
import sys
import time
from collections import deque

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FAP = HERE.parent / "failure_agent_prototype"
PIXEL_Q = FAP / "RL" / "pixel_Q"
for _p in (FAP, PIXEL_Q):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from fa_core import common, predicates as P  # noqa: E402
from shared import scene as SC  # noqa: E402
from shared.scene import IneligibleStart, LiberoScene, ResetFailed  # noqa: E402,F401  (re-exported)

ACTION_DIM = 7
POS_DIMS, ROT_DIMS, GRIPPER_DIM = (0, 1, 2), (3, 4, 5), 6


def eps_vector(eps, dims=(0, 1, 2, 3, 4, 5), rot_scale=0.5):
    """Per-dim residual bound from one scalar: eps on position dims, eps * rot_scale on rotation dims,
    eps on the gripper; 0 on dims not in `dims`."""
    v = np.zeros(ACTION_DIM)
    for d in dims:
        v[d] = eps * (rot_scale if d in ROT_DIMS else 1.0)
    return v


# ------------------------------------------------------------------ policy observation -------------

IMG = 64                                   # policy image side (256 renders, block-averaged 4x)
POLICY_ACT_DIM = 6                         # the residual policy changes the 6 pose dims, never the gripper
LOW_DIM = 8 + common.REPLAN_STEPS * ACTION_DIM   # pi0 state (8) + pi0's executed actions of this chunk (5 x 7)


def downsample(img, size=IMG):
    """(H, W, 3) uint8 -> (size, size, 3) uint8 by block mean (H, W multiples of size)."""
    k = img.shape[0] // size
    return img.reshape(size, k, size, k, 3).mean(axis=(1, 3)).astype(np.uint8)


def policy_obs(obs, chunk, size=IMG):
    """What the residual policy sees at a decision: wrist + agentview (upright, as pi0 sees them) stacked
    channel-first (6, size, size) uint8, and low (LOW_DIM,) float32 = pi0 state + pi0's next actions."""
    img = np.concatenate([downsample(common.upright(obs["robot0_eye_in_hand_image"]), size),
                          downsample(common.upright(obs["agentview_image"]), size)], axis=-1)
    low = np.concatenate([common.pi0_state(obs), np.asarray(chunk, dtype=np.float64).reshape(-1)])
    return {"img": np.ascontiguousarray(img.transpose(2, 0, 1)), "low": low.astype(np.float32)}


class ResidualPickEnv(LiberoScene):
    """cfg keys: env_seed, seed, worker_id, host, port (LiberoScene); targets: optional list of words, the
    pick target's sim name must contain one (else IneligibleStart)."""

    def reset(self, start):
        """Restore a start state and set up the pick check. Returns the observation.
        Raises IneligibleStart when the start cannot be used (no target, target already grasped)."""
        self._ensure_env(start)
        env = self.env
        if start["kind"] == "reset_state":
            obs = common.tools_module("scene_restore").reset_to(env, self._trace, start["step"])
        else:
            env.reset()
            obs = env.set_init_state(common.init_state(start["suite"], start["task_id"], start["init_idx"]))
            for _ in range(common.NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(list(common.DUMMY_ACTION))
        self.prompt = str(self.task.language)
        self.obj = start.get("object") or self._resolve_target(start, obs["robot0_eef_pos"])
        targets = self.cfg.get("targets")
        if targets and not any(w in self.obj.lower() for w in targets):
            raise IneligibleStart("target {} is not one of {}".format(self.obj, targets))
        if P.grasped(env, self.obj):
            raise IneligibleStart("{} is already grasped at the start".format(self.obj))
        self.check = P.PickCheck(self.obj)
        self.check.start(env)
        self.p0 = P.obj_pos(env, self.obj)
        return obs

    def _irrecoverable(self):
        p = P.obj_pos(self.env, self.obj)
        if p[2] < self.p0[2] - SC.DROP_M:
            return "dropped"
        if np.linalg.norm(p[:2] - self.p0[:2]) > SC.WORKSPACE_M:
            return "out_of_workspace"
        return None

    def rollout(self, start, residual=None, eps=None, horizon=200, record=False):
        """One episode from `start`: pi0 + clip(residual, -eps, eps) until pick success, an irrecoverable
        state, a sim error or `horizon` control steps.

        residual: None (plain pi0) or a callable residual(obs, a_pi0) -> (7,) delta, with an optional
            reset() called once at the start of the episode.
        eps: (7,) per-dim bound (see eps_vector); None or zeros = plain pi0.
        Returns a dict: success, end (success | timeout | dropped | out_of_workspace | sim_error), steps,
            obj, prompt, pi0_calls, mean_abs_delta (7,), wall_s, frames (list or None).
        """
        t0 = time.time()
        obs = self.reset(start)
        eps = np.zeros(ACTION_DIM) if eps is None else np.asarray(eps, dtype=np.float64)
        active = eps > 0
        if residual is not None and hasattr(residual, "reset"):
            residual.reset()
        plan = deque()
        frames = [SC.video_frame(obs)] if record else None
        end, t, calls, abs_delta = "timeout", 0, 0, np.zeros(ACTION_DIM)
        while t < horizon:
            if not plan:
                chunk = np.asarray(self._pi0().infer(common.pi0_request(obs, self.prompt))["actions"])
                plan.extend(chunk[:common.REPLAN_STEPS])
                calls += 1
            a_pi0 = np.asarray(plan.popleft(), dtype=np.float64)
            a = a_pi0.copy()
            if residual is not None and active.any():
                delta = np.clip(np.asarray(residual(obs, a_pi0), dtype=np.float64), -eps, eps)
                a[active] = np.clip(a_pi0[active] + delta[active], -1.0, 1.0)
            abs_delta += np.abs(a - a_pi0)
            try:
                obs, _, _, _ = self.env.step(a.tolist())
            except Exception as e:  # main.py also counts sim errors as failures
                end, self.sim_error = "sim_error", repr(e)
                break
            t += 1
            if record:
                frames.append(SC.video_frame(obs))
            if self.check.update(self.env, t):
                end = "success"
                break
            fail = self._irrecoverable()
            if fail is not None:
                end = fail
                break
        return {"success": end == "success", "end": end, "steps": t, "obj": self.obj, "prompt": self.prompt,
                "pi0_calls": calls, "mean_abs_delta": (abs_delta / max(t, 1)).round(4).tolist(),
                "wall_s": round(time.time() - t0, 1), "frames": frames}

    # ---- one decision per pi0 chunk (the RL trainer, train_sac.py)
    def _pi0_chunk(self, obs):
        chunk = np.asarray(self._pi0().infer(common.pi0_request(obs, self.prompt))["actions"], dtype=np.float64)
        return chunk[:common.REPLAN_STEPS]

    def chunk_episode(self, start, act_fn, eps, horizon=200, record=False, size=IMG):
        """One episode with one residual decision per pi0 chunk.

        At each decision pi0 is queried (as in rollout(): every REPLAN_STEPS control steps), the policy sees
        policy_obs(obs, pi0's REPLAN_STEPS actions) and returns u in [-1, 1]^6; the executed actions are
        a_pi0 + eps[:6] * u (the same offset on each of this chunk's actions; gripper untouched), clipped
        to the action box as in rollout().

        act_fn(pobs) -> u (6,). Returns (transitions, info):
          transitions  dicts img, low, u, r, done, img2, low2. r = 1 on pick success. done = terminal
                       (success or irrecoverable); a timeout is a truncation (done False, next obs kept).
          info         success, end, steps, decisions, mean_abs_u, wall_s, obj, frames (or None)
        """
        t0 = time.time()
        obs = self.reset(start)
        eps = np.asarray(eps, dtype=np.float64)
        active = eps > 0
        frames = [SC.video_frame(obs)] if record else None
        chunk = self._pi0_chunk(obs)
        pobs = policy_obs(obs, chunk, size)
        trans, end, t, abs_u = [], "timeout", 0, []
        while t < horizon:
            u = np.clip(np.asarray(act_fn(pobs), dtype=np.float64).reshape(POLICY_ACT_DIM), -1.0, 1.0)
            delta = np.zeros(ACTION_DIM)
            delta[:POLICY_ACT_DIM] = np.clip(eps[:POLICY_ACT_DIM] * u, -eps[:POLICY_ACT_DIM], eps[:POLICY_ACT_DIM])
            abs_u.append(np.abs(u))
            r, done = 0.0, False
            for a_pi0 in chunk:
                a = a_pi0.copy()
                a[active] = np.clip(a_pi0[active] + delta[active], -1.0, 1.0)
                try:
                    obs, _, _, _ = self.env.step(a.tolist())
                except Exception as e:  # the unfinished decision is dropped
                    self.sim_error = repr(e)
                    end = "sim_error"
                    break
                t += 1
                if record:
                    frames.append(SC.video_frame(obs))
                if self.check.update(self.env, t):
                    r, done, end = 1.0, True, "success"
                    break
                fail = self._irrecoverable()
                if fail is not None:
                    done, end = True, fail
                    break
                if t >= horizon:
                    break
            if end == "sim_error":
                break
            if done:
                pobs2 = pobs          # not used (terminal), stored so every transition has the same fields
            else:
                chunk = self._pi0_chunk(obs)      # also on timeout: the critic bootstraps from it
                pobs2 = policy_obs(obs, chunk, size)
            trans.append({"img": pobs["img"], "low": pobs["low"], "u": u.astype(np.float32), "r": r,
                          "done": done, "img2": pobs2["img"], "low2": pobs2["low"]})
            pobs = pobs2
            if done:
                break
        info = {"success": end == "success", "end": end, "steps": t, "decisions": len(trans),
                "mean_abs_u": np.mean(abs_u, axis=0).round(3).tolist() if abs_u else None,
                "wall_s": round(time.time() - t0, 1), "obj": self.obj, "frames": frames}
        return trans, info
