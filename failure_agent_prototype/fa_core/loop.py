"""EpisodeRunner: the control loop between pi0 and LIBERO, with an agent hook.

Without an agent (or with NoOpAgent) it is exactly the rollout loop of tools/record_suites.py /
openpi examples/libero/main.py: 10 dummy wait steps, then pi0 is asked for an action chunk, the first
5 actions are executed, and it is asked again.

With an agent, the loop calls agent.act(view) at step num_steps_wait, then every agent.every_steps
steps (20 = 1 fps), and right after an override finishes. The agent is not called while an override is
running. It returns one of:

  CONTINUE / None          let pi0 keep going
  SetPrompt(prompt)        pi0 gets a new instruction; pi0's queued actions are dropped, so the new
                           prompt takes effect at this step
  Override(motion)         run a scripted Motion (fa_core.motion) instead of pi0 until it finishes;
                           pi0's queued actions are dropped; afterwards pi0 resumes with a fresh chunk
  Stop(reason)             end the episode (counted as a failure unless the task already succeeded)

What the agent sees (AgentView) is limited to the agentview and wrist images, proprioception, the last
action and the prompt: the same budget as at test time. Only an agent with `privileged = True` also
gets a PrivilegedView (the env and the subtask check status); use that for oracle experiments only, and
it is recorded in the episode's events.json.

Everything is logged in the record_suites.py trace format (states[t] = sim state before actions[t],
plus gripper_current_action and qacc_warmstart), so tools/scene_restore.py can restore any step of an
episode run through this loop. Extra field: source[t] = 0 wait, 1 pi0, 2 override.
"""

import collections
import json
import logging
import pathlib
import shutil
import time

import numpy as np

from . import common

SOURCE_WAIT, SOURCE_PI0, SOURCE_OVERRIDE = 0, 1, 2
SOURCE_NAMES = {SOURCE_WAIT: "wait", SOURCE_PI0: "pi0", SOURCE_OVERRIDE: "override"}


# ------------------------------------------------------------------ agent interface ----------------

class Continue(object):
    def __repr__(self):
        return "CONTINUE"


CONTINUE = Continue()


class SetPrompt(object):
    def __init__(self, prompt, reason=""):
        self.prompt, self.reason = str(prompt), reason


class Override(object):
    def __init__(self, motion, reason=""):
        self.motion, self.reason = motion, reason


class Stop(object):
    def __init__(self, reason=""):
        self.reason = reason


class AgentView(object):
    """What a (non-privileged) agent may look at. Arrays are copies."""

    def __init__(self, step, policy_step, obs, last_action, prompt, task_language):
        self.step = step                      # control step = index into the episode trace
        self.policy_step = policy_step        # steps since the wait steps ended
        self.image = common.upright(obs["agentview_image"]).copy()
        self.wrist = common.upright(obs["robot0_eye_in_hand_image"]).copy()
        self.ee_pos = np.array(obs["robot0_eef_pos"], copy=True)
        self.ee_quat = np.array(obs["robot0_eef_quat"], copy=True)
        self.gripper_qpos = np.array(obs["robot0_gripper_qpos"], copy=True)
        # robot0_* keys only (proprio + wrist camera); object poses ("<obj>_pos", "object-state") are left out
        self.obs = {k: np.array(v, copy=True) for k, v in obs.items() if k.startswith("robot0_")}
        self.last_action = None if last_action is None else np.array(last_action, copy=True)
        self.prompt = prompt
        self.task_language = task_language


class PrivilegedView(object):
    """Oracle access: the env itself and the status of the subtask checks. Never for test-time agents."""

    def __init__(self, env, checks):
        self.env = env
        self.checks = {c.spec: c.status() for c in checks}


# ------------------------------------------------------------------ result -------------------------

class EpisodeResult(object):
    def __init__(self):
        self.success = False
        self.error = None
        self.stop_reason = None
        self.num_steps = 0
        self.wall_time_s = 0.0
        self.trace = {}
        self.frames = []
        self.events = []
        self.checks = []
        self.agent = None
        self.agent_calls = 0
        self.interventions = 0

    def summary(self):
        return {"success": self.success, "num_steps": self.num_steps, "error": self.error,
                "stop_reason": self.stop_reason, "agent": self.agent, "agent_calls": self.agent_calls,
                "interventions": self.interventions, "checks": self.checks,
                "wall_time_s": round(self.wall_time_s, 1)}


# ------------------------------------------------------------------ loop ---------------------------

def run_episode(env, obs, prompt, max_steps, client=None, agent=None, checks=(), task_language=None,
                num_steps_wait=common.NUM_STEPS_WAIT, replan_steps=common.REPLAN_STEPS, resize=common.RESIZE,
                stop_on_success=True, record=True, start_step=0, init_state=None, log=None, step_hook=None,
                stop_checks=()):
    """Run one episode from the env's current state; `obs` is the observation of that state.

    max_steps: policy steps after the wait steps (record_suites.py uses MAX_STEPS[suite]).
    num_steps_wait: dummy steps first (use 0 when continuing from a restored trace step).
    start_step: step number of the first state (non-zero when continuing a recorded episode), used
        for step numbers in views, events and check results.
    checks: fa_core.predicates checks, updated after every step (logged; privileged).
    step_hook: optional callable(step, obs) for logging, called for the first state and after every step
        (after the checks are updated), so step indexes the state like states[step].
    stop_checks: specs of checks (e.g. "pick(akita_black_bowl_1)") whose completion ends the episode
        (stop_reason "check <spec> done"); res.success stays the env's own success.
    """
    log = log or logging.getLogger(__name__)
    task_language = prompt if task_language is None else task_language
    res = EpisodeResult()
    res.agent = None if agent is None else {"name": getattr(agent, "name", type(agent).__name__),
                                            "privileged": bool(getattr(agent, "privileged", False)),
                                            "every_steps": int(getattr(agent, "every_steps", 20))}
    every = res.agent["every_steps"] if agent is not None else None
    if agent is not None and hasattr(agent, "reset"):
        agent.reset(task_language, prompt)

    checks = list(checks)
    check_errors = []

    def update_checks(step):
        """Update every check; a check that raises is dropped and logged (checks never stop an episode)."""
        for c in list(checks):
            try:
                c.update(env, step)
            except Exception as e:
                checks.remove(c)
                check_errors.append({"spec": c.spec, "step": step, "error": repr(e)})
                log.warning("check %s disabled at step %d: %r", c.spec, step, e)

    for c in list(checks):
        try:
            c.start(env)
        except Exception as e:
            checks.remove(c)
            check_errors.append({"spec": c.spec, "step": start_step, "error": repr(e)})
            log.warning("check %s disabled at start: %r", c.spec, e)
    update_checks(start_step)
    if step_hook is not None:
        step_hook(start_step, obs)

    buf = collections.defaultdict(list)
    plan = collections.deque()
    motion, motion_start, motion_reason = None, None, ""
    motion_just_ended = False
    last_call, last_action = None, None
    done = False
    t0 = time.time()
    t = 0
    total = num_steps_wait + max_steps

    def event(kind, step, **kw):
        e = {"step": step, "type": kind}
        e.update(kw)
        res.events.append(e)

    event("start", start_step, prompt=prompt)
    while t < total:
        step = start_step + t
        action, source = None, None
        if t < num_steps_wait:
            action, source = np.asarray(common.DUMMY_ACTION, dtype=np.float64), SOURCE_WAIT
        else:
            if motion is not None:
                action = motion.next_action(obs)
                if action is None:
                    event("override_end", step, started=motion_start, reason=motion_reason,
                          motion=motion.describe())
                    motion, motion_just_ended = None, True
                else:
                    source = SOURCE_OVERRIDE
            if action is None and agent is not None and (
                    last_call is None or motion_just_ended or t - last_call >= every):
                last_call, motion_just_ended = t, False
                res.agent_calls += 1
                view = AgentView(step, t - num_steps_wait, obs, last_action, prompt, task_language)
                priv = PrivilegedView(env, checks) if res.agent["privileged"] else None
                decision = agent.act(view, priv)
                if isinstance(decision, Stop):
                    res.stop_reason = decision.reason or "agent stop"
                    event("stop", step, reason=res.stop_reason)
                    break
                if isinstance(decision, SetPrompt):
                    event("set_prompt", step, old=prompt, new=decision.prompt, reason=decision.reason)
                    prompt = decision.prompt
                    plan.clear()
                    res.interventions += 1
                elif isinstance(decision, Override):
                    motion, motion_start, motion_reason = decision.motion, step, decision.reason
                    plan.clear()
                    res.interventions += 1
                    event("override_start", step, reason=decision.reason)
                    action = motion.next_action(obs)
                    if action is None:
                        event("override_end", step, started=step, reason=motion_reason, motion=motion.describe())
                        motion = None
                    else:
                        source = SOURCE_OVERRIDE
                elif decision is not None and not isinstance(decision, Continue):
                    raise TypeError(f"agent returned {decision!r}; expected CONTINUE, SetPrompt, Override or Stop")
            if action is None:
                if not plan:
                    if client is None:
                        raise RuntimeError("pi0 is needed at step %d but no pi0 client was given" % step)
                    chunk = np.asarray(client.infer(common.pi0_request(obs, prompt, resize))["actions"])
                    if len(chunk) < replan_steps:
                        raise RuntimeError(f"pi0 returned {len(chunk)} actions, need >= {replan_steps}")
                    buf["chunk_steps"].append(step)
                    buf["action_chunks"].append(chunk)
                    plan.extend(chunk[:replan_steps])
                action, source = np.asarray(plan.popleft(), dtype=np.float64), SOURCE_PI0

        # everything appended here describes the state BEFORE actions[t]
        if record:
            buf["states"].append(env.get_sim_state().copy())
            for k, v in common.capture_hidden(env).items():
                buf[k].append(v)
            for k, v in common.proprio(obs).items():
                buf["obs_" + k].append(v)
            buf["pi0_state"].append(common.pi0_state(obs))
            res.frames.append(common.upright(obs["agentview_image"]))
        buf["actions"].append(action)
        buf["is_wait"].append(source == SOURCE_WAIT)
        buf["source"].append(source)
        try:
            obs, reward, done, _ = env.step(action.tolist())
        except Exception as e:  # main.py also aborts the episode on sim errors
            log.error("env.step failed at step %d: %s", step, e)
            res.error = repr(e)
            keys = ["actions", "is_wait", "source"]
            if record:
                keys += ["states", "gripper_current_action", "qacc_warmstart", "pi0_state",
                         "obs_ee_pos", "obs_ee_quat", "obs_gripper_states", "obs_joint_states"]
                res.frames.pop()
            for k in keys:
                buf[k].pop()
            break
        buf["rewards"].append(float(reward))
        buf["dones"].append(bool(done))
        last_action = action
        t += 1
        update_checks(start_step + t)  # states[start_step + t] is the state after this action
        if step_hook is not None:
            step_hook(start_step + t, obs)
        if done and stop_on_success:
            break
        hit = [c.spec for c in checks if c.spec in stop_checks and c.done]
        if hit:
            res.stop_reason = "check {} done".format(hit[0])
            event("stop", start_step + t, reason=res.stop_reason)
            break

    if motion is not None:
        event("override_end", start_step + t, started=motion_start, reason=motion_reason,
              motion=motion.describe(), cut_short=True)
    res.success = bool(done)
    res.num_steps = len(buf["actions"])
    res.wall_time_s = time.time() - t0
    res.checks = [c.status() for c in checks] + [dict(e, done=None) for e in check_errors]
    event("end", start_step + t, success=res.success, error=res.error, stop_reason=res.stop_reason)
    if record:
        res.trace = {
            "states": np.asarray(buf["states"]),
            "final_state": env.get_sim_state().copy(),
            "actions": np.asarray(buf["actions"]),
            "is_wait": np.asarray(buf["is_wait"], dtype=bool),
            "source": np.asarray(buf["source"], dtype=np.int8),
            "rewards": np.asarray(buf["rewards"], dtype=np.float64),
            "dones": np.asarray(buf["dones"], dtype=bool),
            "pi0_state": np.asarray(buf["pi0_state"]),
            "chunk_steps": np.asarray(buf["chunk_steps"], dtype=np.int64),
            "action_chunks": np.asarray(buf["action_chunks"]),
            "gripper_current_action": np.asarray(buf["gripper_current_action"]),
            "qacc_warmstart": np.asarray(buf["qacc_warmstart"]),
            "step_offset": np.asarray(start_step, dtype=np.int64),
            **{k: np.asarray(buf[k]) for k in ("obs_ee_pos", "obs_ee_quat", "obs_gripper_states", "obs_joint_states")},
        }
        if init_state is not None:
            res.trace["init_state"] = np.asarray(init_state)
        try:
            res.trace["model_file"] = np.asarray(env.sim.model.get_xml())
        except Exception as e:  # not needed for restoring; keep the episode
            log.warning("could not save model xml: %s", e)
    return res


def write_episode(res, out_dir, name, fps=common.CONTROL_FREQ, extra=None):
    """Write <out_dir>/<name>/{<name>.mp4, trace/*.npy, events.json} atomically (tmp dir + rename)."""
    import imageio

    out_dir = pathlib.Path(out_dir)
    tmp = out_dir / f".tmp_{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "trace").mkdir(parents=True)
    if res.frames:
        imageio.mimwrite(tmp / f"{name}.mp4", res.frames, fps=fps)
    for k, v in res.trace.items():
        np.save(tmp / "trace" / f"{k}.npy", v)
    info = res.summary()
    info["events"] = res.events
    if extra:
        info.update(extra)
    (tmp / "events.json").write_text(json.dumps(info, indent=2, default=_json_default))
    final = out_dir / name
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    return final


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
