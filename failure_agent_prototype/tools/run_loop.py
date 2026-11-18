"""Run pi0 through the fa_core control loop with an agent, over LIBERO tasks and init states.

Same task selection, env settings and output layout as record_suites.py, so baseline_summary.py reads
the output directly and any step of any episode can be restored with scene_restore.reset_to:

  OUT_ROOT/<suite>/meta.json
  OUT_ROOT/<suite>/task<id>_<success|fail>_init<k>/{task..mp4, trace/*.npy, events.json}

events.json lists every agent decision (re-prompts, overrides) and the subtask check results.
Episodes already in meta.json are skipped, so an interrupted run can simply be restarted.

Phase 1 step 6 (sanity check: the loop must not change pi0's behaviour), pi0 server up:
  MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python run_loop.py --agent noop \\
      --out-root ../../Documentation/phase1/noop_heldout --suites libero_90 --split held_out \\
      --init-states 0 1 2 3 4 5 6 7 8 9
  python baseline_summary.py ../../Documentation/phase1/noop_heldout \\
      --compare ../../Documentation/pi0_baseline_heldout
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fa_core import agents, common, loop, predicates as P  # noqa: E402


def run_job(job):
    from openpi_client import websocket_client_policy

    a = job["args"]
    suite, task_id, init_idx = job["suite"], job["task_id"], job["init_idx"]
    np.random.seed(a["env_seed"])  # as in main.py / record_suites.py
    env, task, bddl = common.make_env(suite, task_id, a["env_seed"])
    client = websocket_client_policy.WebsocketClientPolicy(a["host"], a["port"])
    agent = agents.make_agent(a["agent"])
    init = common.init_state(suite, task_id, init_idx)
    env.reset()
    obs = env.set_init_state(init)
    checks = []
    if a["goal_checks"]:
        try:
            checks = P.goal_checks(env) + [P.GoalCheck()]
        except Exception as e:  # checks are logging only; never let them stop the run
            logging.warning("subtask checks disabled for %s task %d: %r", suite, task_id, e)
    try:
        res = loop.run_episode(env, obs, str(task.language), common.MAX_STEPS[suite], client=client, agent=agent,
                               checks=checks, num_steps_wait=a["num_steps_wait"], replan_steps=a["replan_steps"],
                               init_state=init)
    finally:
        env.close()
    name = f"task{task_id}_{'success' if res.success else 'fail'}_init{init_idx}"
    loop.write_episode(res, pathlib.Path(a["out_root"]) / suite, name,
                       extra={"suite": suite, "task_id": task_id, "init_state_idx": init_idx,
                              "task_language": task.language})
    logging.info("%s/%s: %d steps, %d agent calls, %d interventions, %.0f s", suite, name, res.num_steps,
                 res.agent_calls, res.interventions, res.wall_time_s)
    return suite, {
        "folder": name, "task_id": task_id, "init_state_idx": init_idx, "task_language": task.language,
        "task_name": task.name, "bddl_file": str(bddl), "success": res.success, "num_steps": res.num_steps,
        "error": res.error, "stop_reason": res.stop_reason, "agent_calls": res.agent_calls,
        "interventions": res.interventions, "checks": res.checks, "wall_time_s": round(res.wall_time_s, 1),
        "video": f"{name}.mp4",
    }


def suite_meta(suite, a, episodes):
    episodes = sorted(episodes, key=lambda e: (e["task_id"], e["init_state_idx"]))
    return {
        "task_suite": suite,
        "policy": common.POLICY,
        "loop": "fa_core.loop.run_episode",
        "agent": a["agent"],
        "env_seed": a["env_seed"],
        "replan_steps": a["replan_steps"],
        "num_steps_wait": a["num_steps_wait"],
        "max_steps": common.MAX_STEPS[suite],
        "camera_resolution": common.ENV_RESOLUTION,
        "control_freq_hz": common.CONTROL_FREQ,
        "video_fps": common.CONTROL_FREQ,
        "state_convention": "trace/states.npy[t] = sim state before actions[t]; restore with "
                            "tools/scene_restore.reset_to (controller, gripper action, qacc_warmstart)",
        "last_run_selection": {"tasks": a["tasks"], "split": a["split"], "init_states": a["init_states"]},
        "num_episodes": len(episodes),
        "num_success": sum(e["success"] for e in episodes),
        "episodes": episodes,
    }


def main():
    record_suites = common.tools_module("record_suites")  # task selection shared with record_suites.py
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", required=True)
    p.add_argument("--agent", default="noop", help=f"none | {' | '.join(sorted(agents.AGENTS))}")
    p.add_argument("--suites", nargs="+", default=["libero_90"])
    p.add_argument("--tasks", type=int, nargs="+", default=None)
    p.add_argument("--split", choices=record_suites.SPLITS, default="all")
    p.add_argument("--init-states", type=int, nargs="+", default=[0, 1])
    p.add_argument("--no-goal-checks", dest="goal_checks", action="store_false",
                   help="do not log the goal-derived subtask checks")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--env-seed", type=int, default=7)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=common.REPLAN_STEPS)
    p.add_argument("--num-steps-wait", type=int, default=common.NUM_STEPS_WAIT)
    args = p.parse_args()
    a = vars(args)
    try:
        agents.make_agent(args.agent)
    except KeyError as e:
        p.error(str(e))

    from libero.libero import benchmark

    jobs, done = [], {}
    for suite in args.suites:
        s = benchmark.get_benchmark_dict()[suite]()
        try:
            tasks = record_suites.select_tasks(suite, s.n_tasks, args.tasks, args.split)
        except ValueError as e:
            p.error(str(e))
        for tid in tasks:
            n_init = len(s.get_task_init_states(tid))
            bad = [k for k in args.init_states if not 0 <= k < n_init]
            if bad:
                p.error(f"{suite} task {tid} has init states 0..{n_init - 1}; invalid: {bad}")
        suite_dir = pathlib.Path(args.out_root) / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        meta_path = suite_dir / "meta.json"
        eps = json.loads(meta_path.read_text())["episodes"] if meta_path.exists() else []
        eps = [e for e in eps if (suite_dir / e["folder"]).is_dir()]
        if meta_path.exists() and json.loads(meta_path.read_text()).get("agent") not in (None, args.agent):
            p.error(f"{meta_path} was recorded with another agent; use a new --out-root")
        done[suite] = eps
        have = {(e["task_id"], e["init_state_idx"]) for e in eps}
        jobs += [{"suite": suite, "task_id": t, "init_idx": k, "args": a}
                 for t in tasks for k in args.init_states if (t, k) not in have]
    logging.info("%d episodes to run with agent %r", len(jobs), args.agent)

    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (suite, entry) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            done[suite].append(entry)
            (pathlib.Path(args.out_root) / suite / "meta.json").write_text(
                json.dumps(suite_meta(suite, a, done[suite]), indent=2, default=loop._json_default))
            logging.info("[%d/%d] %s/%s (%.0f s elapsed)", i, len(jobs), suite, entry["folder"], time.time() - t0)
    for suite in args.suites:
        eps = done[suite]
        logging.info("%s: %d/%d success", suite, sum(e["success"] for e in eps), len(eps))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
