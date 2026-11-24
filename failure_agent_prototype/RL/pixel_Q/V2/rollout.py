"""Main rollouts with branch points, the learned trigger and the deployed loop (pixel_Q_RL_V2.md sec. 8-10).

Backend-agnostic: the LIBERO env (libero_env.BranchingPickEnv) and the toy world (toy_env.ToyBranchEnv) implement
the same small interface, so the scheduling below is exactly what runs in the simulator:

  reset(start, record)            new main rollout (raises shared.scene.IneligibleStart / ResetFailed)
  step_count, budget              control steps since the rollout start (20 Hz; restored by restore()), budget
  pi0_step()                      one control step of pi0 (from its queued chunk; asks for a new chunk when empty)
  execute_primitive(a, max_steps) run correction a synchronously (clearance rise, level, move + yaw), from the
                                  geometry of the last observe(); hands back to pi0 (its queue is cleared)
  hl_tick()                       HL module (VLM agent or its privileged stand-in): if it corrects now, runs the
                                  correction synchronously and returns its info dict, else None
  gate()                          (open, info): gate conditions 1-3 (pick subtask, gripper open and empty, target
                                  grounded within r + 4 cm); condition 4 (no correction running, cooldown) is here
  observe()                       heightmap observation, or None if the target is not grounded (not visible)
  snapshot() / restore(snap)      in-memory state of the running rollout: sim, step counter, pi0's queue, checks
  pick_done(), pick_step()        pick predicate latched / control step at which it first fired
  task_done(), failure(), agent_stop()   task success, irrecoverable reason (None if fine), VLM agent's Stop
  tick_info()                     dict with at least tilt_deg (approach tilt) and target_dxy (m), for logs
  frames, summary(), close()

Timing is counted in control steps, never wall-clock: a tick is every TICK_STEPS = 4 control steps (5 fps).

Main rollout (sec. 9.2 / 10.3), at every tick:
  1. HL first: if the HL module corrects, the gate stays closed (correction running), then a cooldown starts.
  2. Gate (+ cooldown) open and target grounded -> a gated tick.
  3. Branch point with probability p_branch: snapshot -> no-op branch (pi0 alone) -> restore -> intervention
     branch (correction, then pi0 alone) -> restore. Each branch decides nothing after its first tick and gives
     one label: gamma^(dt in s) if the pick predicate first fires dt <= H_max after the handoff to pi0 (the end
     of the correction's primitive; the branch tick for the no-op branch), else 0.
  4. Trigger (phase 2, evaluation, deployment): A(s) > tau on N consecutive gated ticks -> run the best
     correction on the main rollout, then a cooldown. Gate closed / cooldown / not grounded resets the streak.
  Otherwise pi0 runs. The main rollout ends at pick success (until="pick") or task success (until="task"), an
  irrecoverable state, the VLM agent's Stop, or the step budget.
"""

import time

import numpy as np

from qnet import A0, decode_action
from shared.scene import IneligibleStart, ResetFailed

CTRL_HZ = 20                    # control steps per second
TICK_STEPS = 4                  # one tick every 4 control steps = 5 fps
TILT_BINS = ((0, 10), (10, 20), (20, 30), (30, 180))   # phase 0 tilt bins (deg)


def label_at(h_steps, t_success, gamma):
    """Label of a branch whose pick first fired t_success control steps after its branch tick (None = never),
    for a window of h_steps: gamma^(t / 20) if t <= h_steps, else 0."""
    if t_success is None or t_success > h_steps:
        return 0.0
    return float(gamma ** (t_success / float(CTRL_HZ)))


def tilt_bin(tilt_deg):
    for k, (lo, hi) in enumerate(TILT_BINS):
        if lo <= tilt_deg < hi:
            return k
    return len(TILT_BINS) - 1


# ------------------------------------------------------------------ trigger ------------------------

class Trigger(object):
    """Learned trigger (sec. 8): fires when A(s) > tau on n_persist consecutive gated ticks."""

    kind = "learned"

    def __init__(self, tau, n_persist=3):
        self.tau, self.n_persist, self.streak = float(tau), int(n_persist), 0

    def observe(self, A):
        self.streak = self.streak + 1 if A > self.tau else 0
        return self.streak >= self.n_persist

    def reset(self):
        self.streak = 0


class RandomTrigger(object):
    """Random timing at a given rate per eligible gated tick (evaluation condition 4); the correction is still
    the Q-map's greedy one, so only the timing differs from the learned trigger."""

    kind = "random"

    def __init__(self, p, rng):
        self.p, self.rng, self.streak = float(p), rng, 0

    def observe(self, A):
        return bool(self.rng.random() < self.p)

    def reset(self):
        pass


def make_trigger(rc, rng):
    kind = rc.get("trigger")
    if kind in (None, "none"):
        return None
    if kind == "learned":
        return Trigger(rc["tau"], rc.get("n_persist", 3))
    if kind == "random":
        return RandomTrigger(rc["random_p"], rng)
    raise ValueError("unknown trigger {!r}".format(kind))


# ------------------------------------------------------------------ branches -----------------------

def run_branch(backend, action, cap_steps, h_max_steps, gamma, prim_cap):
    """One branch from the current (just restored) state: optional correction, then pi0 alone, until success,
    an irrecoverable state, or cap_steps after the pi0 handoff (sec. 10.2). Nothing is decided after the first
    tick. The timer (window, label dt, steps) starts at the handoff to pi0: after the correction's primitive
    (rise, level, move + yaw) has finished, or at the branch tick for the no-op branch. The primitive's own
    duration is in prim["steps"] and does not count against the window or the discount.
    Returns dict(label, t (steps from handoff to success or None), end, steps (from handoff), prim)."""
    prim = None
    if action is not None:
        prim = backend.execute_primitive(action, max_steps=prim_cap)
    t0 = backend.step_count                       # handoff to pi0: the timer starts here
    while True:
        if backend.pick_done():
            dt = max(0, int(backend.pick_step() - t0))
            return {"label": label_at(h_max_steps, dt, gamma), "t": dt, "end": "success",
                    "steps": backend.step_count - t0, "prim": prim}
        fail = backend.failure()
        if fail or backend.step_count - t0 >= cap_steps:
            return {"label": 0.0, "t": None, "end": fail or "window", "steps": backend.step_count - t0, "prim": prim}
        backend.pi0_step()


def branch_point(backend, obs, rc, scorer, rng, scored):
    """sec. 9.3: snapshot, no-op branch, restore, intervention branch, restore. Returns the branch record."""
    cap, h, g, pc = rc["branch_cap_steps"], rc["h_max_steps"], rc["gamma"], rc["prim_cap"]
    snap = backend.snapshot()
    t_wall = time.time()
    noop = run_branch(backend, None, cap, h, g, pc)
    backend.restore(snap)
    mode = rc["branch_action"]
    greedy = None
    if mode == "probe":                                   # phase 0: level only (the centre cell at 0 deg)
        a = A0
    elif mode == "greedy":                                # evaluation / calibration: the deployed choice
        a, greedy = int(scored["a_best"]), True
    else:                                                 # training: epsilon-greedy over the corrections
        a, greedy = scorer.branch_action(obs, rc["eps"], rng, kappa=rc.get("branch_kappa", 0.0), scored=scored)
    corr = run_branch(backend, a, cap, h, g, pc)
    backend.restore(snap)
    k, i, j = decode_action(a)
    rec = {"step": int(backend.step_count), "a": int(a), "k": k, "i": i, "j": j, "greedy": greedy,
           "y_noop": noop["label"], "t_noop": noop["t"], "end_noop": noop["end"], "steps_noop": noop["steps"],
           "y_a": corr["label"], "t_a": corr["t"], "end_a": corr["end"], "steps_a": corr["steps"],
           "prim": corr["prim"], "wall_s": round(time.time() - t_wall, 2)}
    if scored is not None:
        mu = scored["q"].mean(axis=0).reshape(-1)
        sd = scored["q"].std(axis=0).reshape(-1)
        rec.update(q0=scored["q0"], A=scored["A"], a_best=scored["a_best"], q_a=float(mu[a]), sigma_a=float(sd[a]),
                   q_best=scored["mu_best"], sigma_best=scored["sigma_best"])
    return rec


# ------------------------------------------------------------------ main rollout -------------------

DEFAULT_RC = {
    "p_branch": 0.0,             # branch probability per gated tick
    "branch_action": "eps",      # probe (phase 0) | eps (training) | greedy (evaluation)
    "eps": 1.0,
    "branch_kappa": 0.0,         # greedy score of training branches: mu - branch_kappa * sigma
    "branch_cap_steps": 60,      # branch length (H_max; 10 s in phase 0 to log long success times)
    "h_max_steps": 60,           # label window H_max
    "gamma": 0.85,               # discount per second
    "prim_cap": 30,              # primitive cap (control steps)
    "cooldown_steps": 60,        # after an intervention or an HL correction (3 s)
    "trigger": None,             # None | learned | random
    "tau": 0.0, "n_persist": 3, "random_p": 0.0,
    "until": "pick",             # pick | task
    "log_ticks": False,          # score A(s) at every gated tick (timing analysis)
    "skip_tilt_bins": (),        # phase 0: tilt bins that are already full (no branch points there)
    "push_samples": True,        # keep the observations of branch points (training)
}


def main_rollout(backend, start, rc, scorer=None, rng=None, record=False):
    """One main rollout from `start` (see module docstring). rc: rollout config (DEFAULT_RC keys).

    Returns a dict: tag, end, steps, pick_success, pick_step, task_success, branches (records), samples
    [(obs, action, label, info)], interventions, hl, ticks (A(s) per gated tick when scored), counts, frames."""
    rc = dict(DEFAULT_RC, **rc)
    rng = rng if rng is not None else np.random.default_rng()
    out = {"tag": start["tag"], "start": start, "branches": [], "samples": [], "interventions": [], "hl": [],
           "ticks": [], "frames": None}
    t_wall = time.time()
    try:
        backend.reset(start, record=record)
    except IneligibleStart as e:          # never usable: the start is dropped for the run
        out["skipped"] = str(e)
        return out
    except ResetFailed as e:              # this reset only
        out["reset_failed"] = str(e)
        return out
    trig = make_trigger(rc, rng)
    if (trig is not None or rc["log_ticks"] or rc["branch_action"] != "probe" and rc["p_branch"] > 0) \
            and scorer is None:
        raise ValueError("this rollout needs a Q-map (scorer)")
    cooldown_until = -10 ** 9
    last_tick = None
    n = {"ticks": 0, "gate_open": 0, "gated": 0, "eligible": 0, "not_grounded": 0}
    while True:
        end = _end_reason(backend, rc["until"])
        if end:
            break
        step = backend.step_count
        if step % TICK_STEPS == 0 and step != last_tick:
            last_tick = step
            n["ticks"] += 1
            hl = backend.hl_tick()                                   # 1. HL has priority
            if hl is not None:
                hl["step"] = step
                out["hl"].append(hl)
                cooldown_until = backend.step_count + rc["cooldown_steps"]
                if trig is not None:
                    trig.reset()
                continue
            if backend.agent_stop():
                continue
            gate_open, ginfo = backend.gate()                       # 2. gate + cooldown
            if not gate_open or step < cooldown_until:
                if trig is not None:
                    trig.reset()
            else:
                n["gate_open"] += 1
                info = backend.tick_info()                           # cheap (no rendering)
                want_branch = rc["p_branch"] > 0 and rng.random() < rc["p_branch"] and \
                    tilt_bin(info["tilt_deg"]) not in rc["skip_tilt_bins"]
                need_obs = want_branch or trig is not None or rc["log_ticks"]
                obs = backend.observe() if need_obs else None
                if need_obs and obs is None:                          # target not grounded: not a gated tick
                    n["not_grounded"] += 1
                    if trig is not None:
                        trig.reset()
                elif obs is not None:
                    n["gated"] += 1
                    scored = scorer.score(obs) if scorer is not None and scorer.loaded else None
                    if want_branch:                                 # 3. branch point
                        br = branch_point(backend, obs, rc, scorer, rng, scored)
                        br.update(tilt_deg=round(info["tilt_deg"], 2), target_dxy=round(info["target_dxy"], 4),
                                  tick_info=info)
                        out["branches"].append(br)
                        if rc["push_samples"]:
                            out["samples"].append((obs, A0, br["y_noop"], {"noop": True, "step": step}))
                            if rc["branch_action"] != "probe":
                                out["samples"].append((obs, br["a"], br["y_a"], {"noop": False, "step": step}))
                    if scored is not None and (trig is not None or rc["log_ticks"]):
                        fire = False
                        if trig is not None:                         # 4. trigger
                            n["eligible"] += 1
                            fire = trig.observe(scored["A"])
                        out["ticks"].append({"step": step, "A": round(scored["A"], 4), "q0": round(scored["q0"], 4),
                                             "best": round(scored["best"], 4), "a_best": scored["a_best"],
                                             "streak": getattr(trig, "streak", 0), "fire": bool(fire),
                                             "tilt_deg": round(info["tilt_deg"], 2),
                                             "target_dxy": round(info["target_dxy"], 4)})
                        if fire:
                            prim = backend.execute_primitive(scored["a_best"], max_steps=rc["prim_cap"])
                            k, i, j = decode_action(scored["a_best"])
                            out["interventions"].append({"step": step, "a": scored["a_best"], "k": k, "i": i, "j": j,
                                                         "A": scored["A"], "q0": scored["q0"], "prim": prim,
                                                         "trigger": trig.kind})
                            cooldown_until = backend.step_count + rc["cooldown_steps"]
                            trig.reset()
                            continue
        backend.pi0_step()
    out.update(end=end, steps=int(backend.step_count), pick_success=bool(backend.pick_done()),
               pick_step=backend.pick_step(), task_success=bool(backend.task_done()), counts=n,
               frames=backend.frames, wall_s=round(time.time() - t_wall, 1), info=backend.summary())
    return out


def _end_reason(backend, until):
    if until == "pick" and backend.pick_done():
        return "success"
    if backend.task_done():
        return "task_success"
    fail = backend.failure()
    if fail:
        return fail
    stop = backend.agent_stop()
    if stop:
        return "agent_stop: " + stop
    if backend.step_count >= backend.budget:
        return "budget"
    return None
