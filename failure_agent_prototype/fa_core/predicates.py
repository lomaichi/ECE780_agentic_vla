"""Subtask success checks over the simulator state (privileged: training reward and evaluation only).

Specs are short strings with simulator names (see the scene's objects with tools/probe_libero_api.py):

  pick(akita_black_bowl_1)                        grasped AND lifted >= 5 cm above its start height
  place(akita_black_bowl_1, plate_1, on)          LIBERO On/In(obj, dest) AND obj not grasped
  open(wooden_cabinet_1_top_region)               LIBERO Open
  close(wooden_cabinet_1_top_region)              LIBERO Close
  turn_on(flat_stove_1_cook_region)               LIBERO TurnOn
  turn_off(flat_stove_1_cook_region)              LIBERO TurnOff
  goal()                                          the task's own LIBERO success (all BDDL goal predicates)

Every check holds for `hold` consecutive evaluated states before it counts (default 10 for pick and
place, 1 otherwise), and then latches: done stays True, done_step records when it first counted.

The checks call LIBERO's own predicate code (BDDLBaseDomain._eval_predicate), so open/close/on/in mean
exactly what they mean in the BDDL task goals. Destinations can be objects, fixtures or regions/sites
(e.g. wooden_cabinet_1_top_side, basket_1_contain_region), as in the BDDL files.

Never give these to a test-time agent: at test time success must come from the agent's own view.
"""

import re

import numpy as np

LIFT_M = 0.05
PICK_HOLD = 10
PLACE_HOLD = 10
STATE_PREDICATES = {"open": "open", "close": "close", "turn_on": "turnon", "turn_off": "turnoff"}


# ------------------------------------------------------------------ LIBERO access -------------------

class LiberoApiError(RuntimeError):
    pass


def _inner(env):
    return env.env  # the BDDL problem env under LIBERO's ControlEnv / OffScreenRenderEnv


def _require(obj, attr, what):
    if not hasattr(obj, attr):
        raise LiberoApiError(f"LIBERO env has no `{attr}` ({what}); run tools/probe_libero_api.py and "
                             f"adapt fa_core/predicates.py to your LIBERO version")
    return getattr(obj, attr)


def object_names(env):
    """Movable objects (BDDL :objects)."""
    return list(_require(_inner(env), "objects_dict", "movable objects"))


def state_names(env):
    """Everything LIBERO predicates accept: objects, fixtures and regions/sites."""
    return list(_require(_inner(env), "object_states_dict", "predicate arguments"))


def body_id(env, name):
    e = _inner(env)
    ids = getattr(e, "obj_body_id", None)
    if ids is not None and name in ids:
        return ids[name]
    return e.sim.model.body_name2id(f"{name}_main")  # LIBERO object root bodies are <name>_main


def obj_pos(env, name):
    return np.array(_inner(env).sim.data.body_xpos[body_id(env, name)], copy=True)


def grasped(env, name):
    e = _inner(env)
    objects = _require(e, "objects_dict", "movable objects")
    if name not in objects:
        raise KeyError(f"{name!r} is not a movable object; objects: {sorted(objects)}")
    check = _require(e, "_check_grasp", "robosuite grasp check")
    return bool(check(gripper=e.robots[0].gripper, object_geoms=objects[name]))


def libero_predicate(env, predicate, *names):
    """Evaluate one LIBERO predicate (on, in, open, close, turnon, turnoff, up) by simulator names."""
    e = _inner(env)
    known = _require(e, "object_states_dict", "predicate arguments")
    missing = [n for n in names if n not in known]
    if missing:
        raise KeyError(f"unknown name(s) {missing}; predicate arguments in this scene: {sorted(known)}")
    evaluate = _require(e, "_eval_predicate", "BDDL predicate evaluation")
    return bool(evaluate([predicate.lower(), *names]))


def task_success(env):
    return bool(_require(_inner(env), "_check_success", "task success")())


def goal_predicates(env):
    """The task's BDDL goal as [[predicate, arg, ...], ...] (lower-case predicate names)."""
    problem = _require(_inner(env), "parsed_problem", "parsed BDDL problem")
    return [[str(g[0]).lower()] + [str(a) for a in g[1:]] for g in problem["goal_state"]]


def sync(env):
    """Recompute derived quantities (poses, contacts) after writing the sim state directly."""
    _inner(env).sim.forward()


def set_state(env, flat_state):
    """Write a flattened sim state ([time, qpos, qvel], as in trace/states.npy) without rendering.

    Enough for evaluating predicates; to continue stepping from a state use scene_restore.reset_to.
    """
    sim = _inner(env).sim
    if hasattr(sim, "set_state_from_flattened"):
        sim.set_state_from_flattened(np.asarray(flat_state))
        sim.forward()
    else:  # slower: also regenerates (renders) observations
        env.set_init_state(np.asarray(flat_state))
        sim.forward()


# ------------------------------------------------------------------ checks --------------------------

class Check(object):
    """Base: subclasses implement holds(env). Call start(env) once, then update(env, step) per state."""

    kind = "base"
    hold = 1

    def __init__(self):
        self.count, self.done, self.done_step = 0, False, None

    @property
    def spec(self):
        raise NotImplementedError

    def start(self, env):
        self.count, self.done, self.done_step = 0, False, None

    def holds(self, env):
        raise NotImplementedError

    def update(self, env, step=None):
        if self.done:
            return True
        self.count = self.count + 1 if self.holds(env) else 0
        if self.count >= self.hold:
            self.done, self.done_step = True, step
        return self.done

    def status(self):
        return {"spec": self.spec, "done": self.done, "done_step": self.done_step}


class PickCheck(Check):
    kind = "pick"

    def __init__(self, obj, lift_m=LIFT_M, hold=PICK_HOLD):
        super().__init__()
        self.obj, self.lift_m, self.hold, self.z0 = obj, lift_m, hold, None

    @property
    def spec(self):
        return f"pick({self.obj})"

    def start(self, env):
        super().start(env)
        self.z0 = obj_pos(env, self.obj)[2]  # height at the start of the evaluated segment

    def holds(self, env):
        return obj_pos(env, self.obj)[2] - self.z0 >= self.lift_m and grasped(env, self.obj)


class PlaceCheck(Check):
    """Strict by default (released, stable for 10 states): the RL reward for rl/place.

    require_release=False, hold=1 is exactly LIBERO's own On/In goal predicate. Recorded success
    episodes stop the moment LIBERO's goal holds, often with the object still in the gripper, so only
    that lenient form can be checked against them (tools/validate_predicates.py does both).
    """

    kind = "place"

    def __init__(self, obj, dest, relation="on", hold=PLACE_HOLD, require_release=True):
        super().__init__()
        if relation not in ("on", "in"):
            raise ValueError(f"place relation must be on or in (LIBERO has no other), got {relation!r}")
        self.obj, self.dest, self.relation, self.hold = obj, dest, relation, hold
        self.require_release = require_release
        if not require_release:
            self.kind = "place_libero"

    @property
    def spec(self):
        return f"place({self.obj}, {self.dest}, {self.relation})" + ("" if self.require_release else "[libero]")

    def holds(self, env):
        if not libero_predicate(env, self.relation, self.obj, self.dest):
            return False
        return not (self.require_release and grasped(env, self.obj))


class StateCheck(Check):
    """open / close / turn_on / turn_off of a fixture or region."""

    def __init__(self, verb, target, hold=1):
        super().__init__()
        if verb not in STATE_PREDICATES:
            raise ValueError(f"verb must be one of {sorted(STATE_PREDICATES)}, got {verb!r}")
        self.kind, self.verb, self.target, self.hold = verb, verb, target, hold

    @property
    def spec(self):
        return f"{self.verb}({self.target})"

    def holds(self, env):
        return libero_predicate(env, STATE_PREDICATES[self.verb], self.target)


class GoalCheck(Check):
    kind = "goal"

    def __init__(self, hold=1):
        super().__init__()
        self.hold = hold

    @property
    def spec(self):
        return "goal()"

    def holds(self, env):
        return task_success(env)


_SPEC = re.compile(r"^\s*(\w+)\s*\(\s*(.*?)\s*\)\s*$")


def parse_spec(spec, hold=None):
    """'place(a, b, on)' -> PlaceCheck('a', 'b', 'on'). hold overrides the default."""
    m = _SPEC.match(spec)
    if not m:
        raise ValueError(f"bad check spec {spec!r}; expected e.g. pick(akita_black_bowl_1)")
    verb, args = m.group(1).lower(), [a.strip() for a in m.group(2).split(",") if a.strip()]
    kw = {} if hold is None else {"hold": hold}
    if verb == "pick" and len(args) == 1:
        return PickCheck(args[0], **kw)
    if verb == "place" and len(args) in (2, 3):
        return PlaceCheck(args[0], args[1], args[2].lower() if len(args) == 3 else "on", **kw)
    if verb in STATE_PREDICATES and len(args) == 1:
        return StateCheck(verb, args[0], **kw)
    if verb == "goal" and not args:
        return GoalCheck(**kw)
    raise ValueError(f"bad check spec {spec!r}: pick(obj) | place(obj, dest[, on|in]) | "
                     f"open/close/turn_on/turn_off(target) | goal()")


def goal_checks(env, strict_place=True):
    """Checks implied by the task goal: pick+place for every On/In, a state check for the rest.

    strict_place=False gives LIBERO-equivalent place checks (see PlaceCheck).
    """
    inverse = {v: k for k, v in STATE_PREDICATES.items()}
    movable = set(object_names(env))
    checks = []
    for g in goal_predicates(env):
        if g[0] in ("on", "in") and len(g) == 3:
            if g[1] in movable:
                checks.append(PickCheck(g[1]))
            if strict_place:
                checks.append(PlaceCheck(g[1], g[2], g[0]))
            else:
                checks.append(PlaceCheck(g[1], g[2], g[0], hold=1, require_release=False))
        elif g[0] in inverse and len(g) == 2:
            checks.append(StateCheck(inverse[g[0]], g[1]))
    return checks


# ------------------------------------------------------------------ survey label -> check -----------

def _tokens(name):
    return {t for t in re.split(r"[_\s]+", str(name).lower()) if t and not t.isdigit()}


def best_match(word, names):
    """The single name sharing the most tokens with `word` (e.g. black_bowl -> akita_black_bowl_1).

    Returns None when nothing matches or the best score is tied (e.g. two black bowls): ambiguous
    labels are skipped rather than guessed.
    """
    scored = sorted(((len(_tokens(word) & _tokens(n)), n) for n in set(names)), reverse=True)
    if not scored or scored[0][0] == 0 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
        return None
    return scored[0][1]


def subtask_to_spec(subtask, goals, movable):
    """Map a survey subtask label (pick(black_bowl), close(top_drawer), ...) to a check spec.

    Only the task's own goal objects are candidates, so the mapping never guesses a simulator name.
    Returns (spec or None, reason).
    """
    m = _SPEC.match(subtask)
    if not m:
        return None, "unparsed label"
    verb, args = m.group(1).lower(), [a.strip() for a in m.group(2).split(",") if a.strip()]
    if not args:
        return None, "no argument"
    if verb in ("pick", "place"):
        cands = [g for g in goals if g[0] in ("on", "in") and len(g) == 3 and g[1] in movable]
        obj = best_match(args[0], [g[1] for g in cands])
        if obj is None:
            return None, f"{args[0]!r} matches no single goal object in {[g[1] for g in cands]}"
        if verb == "pick":
            return f"pick({obj})", "ok"
        g = [g for g in cands if g[1] == obj]
        if len(g) != 1:
            return None, f"{obj} has {len(g)} goal placements"
        return f"place({obj}, {g[0][2]}, {g[0][0]})", "ok"
    if verb in STATE_PREDICATES:
        cands = [g for g in goals if g[0] == STATE_PREDICATES[verb] and len(g) == 2]
        if len(cands) == 1:
            return f"{verb}({cands[0][1]})", "ok"
        target = best_match(args[0], [g[1] for g in cands])
        if target is None:
            return None, f"{len(cands)} matching goal predicates for {verb}"
        return f"{verb}({target})", "ok"
    return None, f"verb {verb!r} has no check"


def outcome_expectation(outcome):
    """Survey outcome text -> expected check result: True (success), False (failed / never attempted),
    None (partial, undone, unclear: not used)."""
    o = str(outcome).strip().lower()
    if o.startswith("success"):
        return True
    if o.startswith("failed") or o.startswith("never attempted"):
        return False
    return None
