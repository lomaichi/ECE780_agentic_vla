"""Phase 1 step 1: check that every LIBERO / robosuite call fa_core relies on exists in YOUR install.

fa_core/predicates.py uses a few LIBERO internals that no recorded run has exercised yet
(objects_dict, object_states_dict, obj_body_id, _check_grasp, _eval_predicate, _check_success,
parsed_problem, sim.set_state_from_flattened). This script builds one env, calls each of them, and
prints PASS/FAIL per item plus the scene's simulator names (what check specs refer to).
No pi0 server needed. Run in the LIBERO client venv:

  MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python probe_libero_api.py [--suite libero_90 --task 1]

Exit code 0 = everything Phase 1 needs works. If anything FAILs, send the output to whoever maintains
fa_core: the fix is a small change in fa_core/predicates.py.
"""

import argparse
import importlib
import os
import pathlib
import sys
import traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fa_core import common, predicates as P  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_90")
    p.add_argument("--task", type=int, default=1, help="default: libero_90 task 1 (drawer + bowl, survey split)")
    p.add_argument("--init", type=int, default=0)
    a = p.parse_args()

    results = []

    def item(name, fn, required=True):
        """Run one probe; optional items print WARN instead of FAIL and do not affect the exit code."""
        try:
            out = fn()
            results.append((name, True, required))
            print(f"PASS  {name}" + (f": {out}" if out not in (None, "") else ""))
            return out
        except Exception as e:
            results.append((name, False, required))
            print(f"{'FAIL' if required else 'WARN'}  {name}: {e!r}")
            if required:
                traceback.print_exc(limit=2)
            return None

    item("import robosuite.utils.transform_utils", lambda: importlib.import_module("robosuite.utils.transform_utils") and "")
    item("import openpi_client", lambda: importlib.import_module("openpi_client") and "")
    env, task, _ = common.make_env(a.suite, a.task)
    env.reset()
    obs = env.set_init_state(common.init_state(a.suite, a.task, a.init))
    print(f"\n{a.suite} task {a.task}: {task.language}\n")

    objs = item("env.env.objects_dict (movable objects)", lambda: sorted(P.object_names(env)))
    item("env.env.fixtures_dict (fixtures; informational)", lambda: sorted(getattr(env.env, "fixtures_dict")),
         required=False)
    item("env.env.object_states_dict (predicate arguments)", lambda: sorted(P.state_names(env)))
    goals = item("env.env.parsed_problem['goal_state']", lambda: P.goal_predicates(env))
    item("env.env._check_success()", lambda: P.task_success(env))
    if goals:
        for g in goals:
            item(f"_eval_predicate {g}", lambda g=g: P.libero_predicate(env, g[0], *g[1:]))
    for o in objs or []:
        item(f"body id + position of {o}", lambda o=o: np.round(P.obj_pos(env, o), 3).tolist())
        item(f"_check_grasp({o})", lambda o=o: P.grasped(env, o))
    item("goal_checks(env) specs", lambda: [c.spec for c in P.goal_checks(env)])

    def roundtrip():
        s0 = env.get_sim_state().copy()
        env.step(list(common.DUMMY_ACTION))
        moved = env.get_sim_state().copy()
        P.set_state(env, s0)
        back = env.get_sim_state().copy()
        if not np.allclose(back, s0):
            raise AssertionError(f"state not restored, max err {np.abs(back - s0).max():.3g}")
        return f"restored (max err {np.abs(back - s0).max():.1e}; after one step it had moved {np.abs(moved - s0).max():.1e})"

    item("sim.set_state_from_flattened roundtrip", roundtrip)
    def fast_path():
        if not hasattr(env.env.sim, "set_state_from_flattened"):
            raise AssertionError("missing: predicates.set_state falls back to set_init_state (slower, still correct)")
        return True

    item("sim has set_state_from_flattened (fast path)", fast_path, required=False)
    item("env.sim.model.get_xml()", lambda: f"{len(env.sim.model.get_xml())} chars")
    item("pi0 state from obs", lambda: np.round(common.pi0_state(obs), 3).tolist())
    env.close()

    n_fail = sum(1 for _, ok, req in results if req and not ok)
    n_warn = sum(1 for _, ok, req in results if not req and not ok)
    print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} PASS, {n_fail} FAIL, {n_warn} WARN")
    print("RESULT:", "OK, Phase 1 code can run" if n_fail == 0 else "FAILED, see the FAIL lines above")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
