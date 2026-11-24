"""FrameworkAgent: psuedocode.txt as an fa_core agent (fa_core.loop.run_episode drives pi0 and calls it).

The loop runs pi0 and calls the agent every `every_steps` control steps (5 fps) and right after an override
(scripted motion) finishes; it is not called while an override runs. Each call is one iteration of the
pseudocode's `While` loop:

  VLM monitor call (subtask tracking + high-level failure detection, Moondream2 grounded agent)
  if subtask == pick(bowl/mug):
      if timer2 < TIMEOUT:
          if not HL and Qmax already issued in this pick subtask: pi0 (V is not checked again)
          elif not HL and close_to_object():           (V4's gate, from the VLM's measurements)
              V_est = V(obs)                           (V2/V4 gripper-centred heightmap + proprio)
              V_est < THRESHOLD on THRESHOLD_COUNT consecutive frames
                  -> approach(Q_INITIAL_POSITION) -> argmax Q(obs) -> pi0
              else pi0
          elif HL: the VLM's high-level correction (skill library)
          else pi0
      else (timeout): approach(Q_INITIAL_POSITION) -> argmax Q(obs) -> pi0, reset timer2
  else: HL correction if HL else pi0
The whole-episode timeout (timer1) is the loop's step budget (run_pipeline.py: FULL_TASK_TIMEOUT_S * 20).

The Q correction is two overrides: (1) approach the handoff pose (V1's training start pose: open gripper
7 cm above the target's top, pointing down); when it ends the loop calls the agent again, which (2) builds
the Q observation there (V1's wrist crop), takes the masked argmax and moves to that cell with that yaw.
Then pi0 resumes.

How the pseudocode was read where it is ambiguous (see README.md):
  - timer2 starts when a pick(bowl/mug) subtask becomes current (also when the VLM goes back to the pick after
    a lost grasp) and restarts after a timeout correction; low_success_count starts at 0 there. (The literal
    pseudocode re-begins timer2 and zeroes the count on every iteration, so neither could ever fire.)
  - "consecutive" frames: the count also resets on frames where V is not evaluated (not close to the object,
    HL failure, correction).
  - close_to_object() includes V4's 1 s cooldown after any correction (CORRECTION_COOLDOWN_S).
  - once a Q correction issues its argmax(Q) move (V-low or timeout trigger), V is no longer checked for the
    rest of that pick subtask: pi0 carries on from the Qmax pose. A new pick subtask (incl. going back to the pick
    after a lost grasp) re-arms V. The timeout and HL branches are unaffected.
  - the timeout branch has priority over an HL correction found at the same call (it sits outside the HL
    check in the pseudocode); the VLM agent is told its correction was dropped.
  - the timeout correction waits up to TIMEOUT_HELD_GRACE_S while the gripper is closed on something.
  - the VLM judges subtasks only, never the whole task: LIBERO's success check (the env's `done`) ends the
    episode (run_episode stop_on_success). The VLM agent is built with task_done_stop=False; a Stop from it
    anyway is logged and ignored.

Collaborators (duck-typed; components.py has the real ones, test_supervisor.py fakes):
  vlm         reset(task, prompt); act(view) -> decision; subtask() -> (idx, kind, object phrase);
              hl_failure() -> bool; failure() -> dict | None; held() -> bool;
              external_correction_done(step); discard_correction() -> dict | None; summary() -> dict
  perception  gate(view) -> (open, info); v_obs(view) -> obs | None; handoff(view) -> (motion, info) | None;
              q_obs(view) -> obs | None; q_move(view, action) -> (motion, info)
  v_model     predict(obs) -> (mean, std)
  q_model     select(obs) -> (action, info)
  library     rl_policy(policy_id) -> dict; lookup(ids) -> [entry summaries]       (optional)
"""

import logging
import traceback

import config as C

C.ensure_paths()
from fa_core import common  # noqa: E402
from fa_core.agents import Agent  # noqa: E402
from fa_core.loop import CONTINUE, Continue, Override, SetPrompt, Stop  # noqa: E402

S = common.CONTROL_FREQ


def describe(decision):
    if decision is None or isinstance(decision, Continue):
        return "pi0"
    if isinstance(decision, Override):
        return "override: " + decision.reason
    if isinstance(decision, SetPrompt):
        return "set_prompt: " + decision.prompt
    if isinstance(decision, Stop):
        return "stop: " + decision.reason
    return repr(decision)


class FrameworkAgent(Agent):
    name = "framework"
    privileged = False

    def __init__(self, vlm, perception, v_model, q_model, library=None, threshold=C.THRESHOLD,
                 threshold_count=C.THRESHOLD_COUNT, timeout_s=C.TIMEOUT_S, monitor_fps=C.MONITOR_FPS,
                 cooldown_s=C.CORRECTION_COOLDOWN_S, held_grace_s=C.TIMEOUT_HELD_GRACE_S, log=None):
        self.vlm, self.per, self.V, self.Q, self.library = vlm, perception, v_model, q_model, library
        self.threshold, self.threshold_count = float(threshold), int(threshold_count)
        self.timeout_steps = int(round(timeout_s * S))
        self.cooldown_steps = int(round(cooldown_s * S))
        self.held_grace_steps = int(round(held_grace_s * S))
        self.every_steps = max(1, int(round(S / float(monitor_fps))))
        self.log = log or logging.getLogger(__name__)
        self.q_enabled = perception is not None and v_model is not None and q_model is not None

    # ---------------------------------------------------------------- state
    def reset(self, task_language, prompt):
        self.vlm.reset(task_language, prompt)
        self.sub_key = None          # (idx, kind) of the VLM's current subtask
        self.pick_start = None       # timer2: step the current pick(bowl/mug) subtask (re)started; None = other
        self.low_count = 0           # low_success_count
        self.q_issued = False        # argmax(Q) move issued in the current pick subtask: V no longer checked
        self.defer_start = None      # step the timeout first fired while the gripper held something
        self.pending = None          # running correction: {"kind": q_approach | q_move | hl, ...}
        self.last_correction_end = None
        self.calls = []              # one record per call
        self.corrections = []        # one record per correction (Q or HL)

    # ---------------------------------------------------------------- main
    def act(self, view, priv=None):
        rec = {"step": int(view.step), "t_s": round(view.policy_step / float(S), 2)}
        self.calls.append(rec)
        try:
            d = self._act(view, rec)
        except Exception as e:      # a perception / model error must not end the episode: pi0 continues
            rec["error"] = repr(e)[:300]
            self.log.warning("step %d: framework error, pi0 continues:\n%s", view.step, traceback.format_exc())
            if self.pending is not None and self.pending["kind"] != "hl":
                self._end_correction(view.step, rec, aborted=True)
            d = CONTINUE
        rec["decision"] = describe(d)
        return d

    def _act(self, view, rec):
        step = view.step
        if self.pending is not None:              # the loop calls right after an override ends
            if self.pending["kind"] == "q_approach":
                return self._q_select(view, rec)  # at the handoff pose: Q obs -> argmax -> move
            self._end_correction(step, rec)

        d = self.vlm.act(view)                    # subtask tracking + HL failure detection at 5 fps
        hl = self.vlm.hl_failure()
        rec.update(self.vlm.summary())
        if isinstance(d, Stop):                   # task success is LIBERO's call, never the VLM's
            rec["vlm_stop_ignored"] = d.reason
            d = CONTINUE
        self._track_subtask(step, rec)

        if self.pick_start is None:               # not pick(bowl/mug): HL correction only
            rec["branch"] = "hl" if hl else "other_subtask"
            self.low_count = 0
            return self._hl(d, view, rec) if hl else CONTINUE

        t2 = step - self.pick_start
        rec["timer2_s"] = round(t2 / float(S), 2)
        if t2 >= self.timeout_steps and not self._defer_timeout(step, rec):
            rec["branch"] = "timeout"
            self.low_count = 0
            if hl:
                dropped = self.vlm.discard_correction()
                if dropped is not None:
                    rec["hl_preempted"] = dropped.get("action")
            return self._q_start(view, rec, "timeout")

        if not hl:
            if self.q_issued:                     # Qmax already issued: carry on with it, V is not checked
                rec["branch"] = "q_issued"
                self.low_count = 0
                return CONTINUE
            if self._close_to_object(view, rec):
                v, sd = self.V.predict(self._vobs)
                rec["v_est"], rec["v_std"] = round(float(v), 4), round(float(sd), 4)
                if v < self.threshold:
                    self.low_count += 1
                    rec["low_count"] = self.low_count
                    if self.low_count >= self.threshold_count:
                        rec["branch"] = "v_low"
                        self.low_count = 0
                        return self._q_start(view, rec, "v_low")
                    rec["branch"] = "v_low_counting"
                    return CONTINUE
                rec["branch"] = "v_ok"
                self.low_count = 0
                return CONTINUE
            rec["branch"] = "not_close"
            self.low_count = 0
            return CONTINUE
        rec["branch"] = "hl"
        self.low_count = 0
        return self._hl(d, view, rec)

    # ---------------------------------------------------------------- pieces
    def _track_subtask(self, step, rec):
        idx, kind, obj = self.vlm.subtask()
        key = (idx, kind)
        if key == self.sub_key:
            return
        self.sub_key = key
        self.low_count, self.defer_start, self.q_issued = 0, None, False
        pick = kind == "pick" and C.is_round_rim(obj) and self.q_enabled
        self.pick_start = step if pick else None
        rec["subtask_start"] = {"idx": idx, "kind": kind, "object": obj, "timer2": bool(pick)}

    def _defer_timeout(self, step, rec):
        """True while the timeout correction waits for a closed gripper (at most held_grace_steps)."""
        if not self.vlm.held():
            self.defer_start = None
            return False
        if self.defer_start is None:
            self.defer_start = step
        if step - self.defer_start < self.held_grace_steps:
            rec["timeout_deferred_s"] = round((step - self.defer_start) / float(S), 2)
            return True
        return False

    def _close_to_object(self, view, rec):
        """VLM.close_to_object(): V4's gated tick (pick subtask, gripper open and empty, target within 10 cm (xy)
        of the grasp point and at most 10 cm below it, target grounded in agentview, no correction in the last
        1 s). V is only evaluated on such states: they are the only ones it was trained on."""
        self._vobs = None
        step = view.step
        if self.last_correction_end is not None and step - self.last_correction_end < self.cooldown_steps:
            rec["close"] = False
            rec["gate"] = {"cooldown_s": round((self.cooldown_steps - (step - self.last_correction_end)) / float(S), 2)}
            return False
        ok, info = self.per.gate(view)
        rec["gate"] = _round(info)
        if ok:
            self._vobs = self.per.v_obs(view)
            if self._vobs is None:
                rec["gate"]["grounded"] = False
                ok = False
        rec["close"] = bool(ok)
        return bool(ok)

    def _hl(self, d, view, rec):
        """execute(VLM.HL_correction): pass the VLM agent's correction through, tagged with its library entry."""
        if not isinstance(d, Override):
            if self.vlm.hl_failure():
                rec["hl_note"] = "correction suppressed by the VLM agent (cooldown / budget / no target point)"
            return d
        f = self.vlm.failure() or {}
        corr = {"type": "HL", "step": int(view.step), "subtask": rec.get("subtask"), "mode": f.get("failure_mode"),
                "library_id": f.get("library"), "action": f.get("action"), "reason": f.get("reason"),
                "target_world": f.get("target_world")}
        if self.library is not None:
            corr["library"] = self.library.lookup(f.get("library"))
        self.corrections.append(corr)
        self.pending = {"kind": "hl", "motion": d.motion}
        return d

    def _q_start(self, view, rec, trigger):
        """execute(approach(Q_INITIAL_POSITION)): first half of the Q correction."""
        step = view.step
        corr = {"type": "Q", "trigger": trigger, "step": int(step), "subtask": rec.get("subtask"),
                "timer2_s": rec.get("timer2_s"), "v_est": rec.get("v_est")}
        if self.library is not None:
            corr["library"] = self.library.rl_policy("rl/pick")
        self.corrections.append(corr)
        h = self.per.handoff(view)
        if h is None:
            corr["skipped"] = "target not located (Q_INITIAL_POSITION unknown)"
            rec["q"] = corr["skipped"]
            self._after_q(step, trigger)
            return CONTINUE
        motion, info = h
        corr["handoff"] = _round(info)
        self.pending = {"kind": "q_approach", "trigger": trigger, "motion": motion}
        return Override(motion, reason="Q[{}]: approach Q_INITIAL_POSITION".format(trigger))

    def _q_select(self, view, rec):
        """execute(argmax(Q(obs))) at the handoff pose: second half of the Q correction."""
        p, corr = self.pending, self.corrections[-1]
        corr["approach"] = _round(p["motion"].describe())
        corr["approach_end_step"] = int(view.step)
        rec["branch"] = "q_select"
        obs = self.per.q_obs(view)
        if obs is None:
            corr["skipped"] = "target not visible in the wrist view at the handoff pose"
            rec["q"] = corr["skipped"]
            self._end_correction(view.step, rec)
            return CONTINUE
        a, qinfo = self.Q.select(obs)
        motion, minfo = self.per.q_move(view, a)
        corr["q"] = _round(qinfo)
        corr["move"] = _round(minfo)
        rec["q"] = {k: corr["q"][k] for k in ("k", "row", "col", "yaw_deg", "q") if k in corr["q"]}
        self.pending = {"kind": "q_move", "trigger": p["trigger"], "motion": motion}
        self.q_issued = True                      # V is not checked again in this pick subtask
        return Override(motion, reason="Q[{}]: argmax cell ({}, {}) yaw {:+.0f} deg, Q {:.2f}".format(
            p["trigger"], qinfo["row"], qinfo["col"], qinfo["yaw_deg"], qinfo["q"]))

    def _end_correction(self, step, rec, aborted=False):
        p, self.pending = self.pending, None
        self.last_correction_end = step
        self.low_count = 0
        if self.corrections:
            c = self.corrections[-1]
            c["end_step"] = int(step)
            if p is not None and "motion" in p and p["kind"] != "q_approach":
                c["motion_end"] = _round(p["motion"].describe())
            if aborted:
                c["aborted"] = True
        if p is not None and p["kind"] in ("q_approach", "q_move"):
            self.vlm.external_correction_done(step)   # restart the VLM's failure windows, as after its own
            self._after_q(step, p["trigger"])
        rec["correction_end"] = p["kind"] if p is not None else None

    def _after_q(self, step, trigger):
        if trigger == "timeout":
            self.pick_start, self.defer_start = step, None     # reset(timer2)

    # ---------------------------------------------------------------- report
    def report(self):
        n = lambda f: sum(1 for c in self.corrections if f(c))  # noqa: E731
        v = [c["v_est"] for c in self.calls if "v_est" in c]
        return {
            "calls": len(self.calls), "v_evals": len(v), "v_low_frames": sum(1 for x in v if x < self.threshold),
            "v_mean": None if not v else round(sum(v) / len(v), 4),
            "q_corrections": n(lambda c: c["type"] == "Q" and "skipped" not in c),
            "q_v_low": n(lambda c: c["type"] == "Q" and c["trigger"] == "v_low" and "skipped" not in c),
            "q_timeout": n(lambda c: c["type"] == "Q" and c["trigger"] == "timeout" and "skipped" not in c),
            "q_skipped": n(lambda c: c["type"] == "Q" and "skipped" in c),
            "hl_corrections": n(lambda c: c["type"] == "HL"),
            "hl_by_library": _count(c.get("library_id") for c in self.corrections if c["type"] == "HL"),
            "errors": sum(1 for c in self.calls if "error" in c),
        }


def _count(xs):
    out = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def _round(x, nd=4):
    """JSON-friendly copy: numpy -> python, floats rounded."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(x, dict):
        return {str(k): _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    if np is not None and isinstance(x, np.ndarray):
        return _round(x.tolist(), nd)
    if np is not None and isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return round(x, nd)
    return x
