# Subtask Success Criteria

A summary of `fa_core/predicates.py`, the code that decides whether the robot finished each small step of a LIBERO task.

> **Privileged information.** These checks read the simulator's internal state, such as exact object positions and gripper contacts. Use them only for **training rewards and evaluation**. A test-time agent must never get them; at test time, success must come from the agent's own view.

## The criteria

| Spec | Succeeds when | Must hold for |
|---|---|---|
| `pick(obj)` | `obj` is grasped **and** lifted ≥ 5 cm above its resting height (the lowest height seen since `start`; objects spawn ~7 cm up and settle during the wait steps) | 10 states in a row |
| `place(obj, dest, on\|in)` | LIBERO's `On`/`In(obj, dest)` is true **and** `obj` is no longer grasped | 10 states in a row |
| `open(target)` | LIBERO's `Open` is true | 1 state |
| `close(target)` | LIBERO's `Close` is true | 1 state |
| `turn_on(target)` | LIBERO's `TurnOn` is true | 1 state |
| `turn_off(target)` | LIBERO's `TurnOff` is true | 1 state |
| `goal()` | LIBERO's own task success (all BDDL goal predicates) | 1 state |

- Names are simulator names, e.g. `akita_black_bowl_1`, `plate_1`, `wooden_cabinet_1_top_region`. Destinations and targets can be objects, fixtures or regions/sites, as in the BDDL files.
- `on`, `in`, `open` and the others use LIBERO's own predicate code (`_eval_predicate`), so they mean exactly what they mean in the BDDL task goals.
- Pass `hold=` to `parse_spec` to override how long a check must hold.

## How a check behaves over time

1. **Must stay true.** A check counts only after its condition is true for `hold` evaluated states in a row. If the condition breaks, the counter resets to 0, so a brief lift followed by a drop doesn't count.
2. **Latches.** Once a check succeeds, `done` stays `True`, and `done_step` records the step where it first counted.

Usage: `start(env)` once → `update(env, step)` on every state → `status()` returns `{spec, done, done_step}`.

## Strict vs. lenient place

| Mode | Settings | Use |
|---|---|---|
| Strict (default) | `require_release=True`, `hold=10` | RL reward for placing. Object released and stable. |
| Lenient (`[libero]`) | `require_release=False`, `hold=1` | Exactly LIBERO's own `On`/`In` goal. Needed to check recorded demos, which stop the moment LIBERO reports success, often while the object is still in the gripper. |

`tools/validate_predicates.py` runs both modes.

## Checks generated from the task goal

`goal_checks(env)` reads the task's BDDL goal and builds:

- For every `On`/`In(obj, dest)`: `pick(obj)` (if `obj` is movable) followed by `place(obj, dest, on|in)`. Strict by default; `strict_place=False` gives the lenient version.
- For every `Open`/`Close`/`TurnOn`/`TurnOff(target)`: the matching state check.

## Mapping survey labels to checks

Survey labels use short names, e.g. `pick(black_bowl)` or `close(top_drawer)`. `subtask_to_spec` maps them to specs:

- Only objects and targets that appear in the task's goal are candidates, so the mapping never invents a simulator name.
- `best_match` splits names into words (ignoring numbers) and picks the candidate that shares the most words with the label. If nothing matches, or two candidates tie (e.g. two black bowls), it returns nothing, so **ambiguous labels are skipped rather than guessed**.
- `place(...)` requires exactly one goal placement for the matched object.
- For state verbs, if the goal has exactly one predicate of that type, it is used directly.

Survey outcomes map to an expected check result through `outcome_expectation`:

| Survey outcome | Expected |
|---|---|
| `success…` | `True` |
| `failed…`, `never attempted…` | `False` |
| partial, undone, unclear | `None` (not used) |
