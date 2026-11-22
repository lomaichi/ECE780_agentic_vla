"""Phase 1 step 3: validate the subtask checks (fa_core/predicates.py) on the recorded rollouts.

No pi0 needed: every recorded state (trace/states.npy, plus final_state) is written back into the
simulator and the checks are evaluated on it, in order.

Part A, plumbing: LIBERO's task success recomputed on each episode's final_state must equal the
  recorded result. This checks state restore + LIBERO predicate access on all 220 episodes.

Part B, subtask checks, two sources of expected answers:
  goal    success episodes: for every goal On/In(obj, dest) the pick(obj) check and the LIBERO-
          equivalent place check must fire; open/close/turn_on/turn_off goals must fire.
          (Recorded successes stop the instant LIBERO's goal holds, often before the release, so the
          strict place check, released + stable for 10 states, is reported but not judged here.)
  survey  surveyed failures (failure_survey/episodes/*/survey.json): each subtask label is mapped to a
          check through the task's goal objects (fa_core.predicates.subtask_to_spec); "success" -> must
          fire, "failed" / "never attempted" -> must not fire; partial/unclear labels and ambiguous
          names (e.g. two black bowls) are skipped and listed. Place uses the strict check here.
  The survey labels are visual judgements at 1-3 fps, so some disagreement is expected. Every
  disagreement is listed with the step at which the check fired: look at the video there.

Run in the LIBERO client venv (MUJOCO_GL=egl), after probe_libero_api.py passes:
  python validate_predicates.py [--suites libero_spatial libero_10 libero_90] [--workers 6]
                                [--episodes libero_90/task10_fail_init0 ...] [--out DIR]
Writes DIR/report.md and DIR/report.json (default Documentation/phase1/validate_predicates).
"""

import argparse
import collections
import json
import multiprocessing as mp
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fa_core import common, predicates as P  # noqa: E402

DEFAULT_OUT = common.REPO / "Documentation" / "phase1" / "validate_predicates"
_ENV = {}


def worker_env(suite, task_id):
    key = (suite, task_id)
    if _ENV.get("key") != key:
        if "env" in _ENV:
            _ENV["env"].close()
        env, _, _ = common.make_env(suite, task_id)
        env.reset()
        _ENV.update(key=key, env=env)
    return _ENV["env"]


def survey_file(suite, entry):
    return common.SURVEY_EPISODES / suite / entry.get("episode_id", entry["folder"]) / "survey.json"


def build_checks(env, entry, survey):
    """[(check, source, expected or None, note)]"""
    goals = P.goal_predicates(env)
    movable = set(P.object_names(env))
    out = []
    if entry["success"]:
        for c in P.goal_checks(env, strict_place=False):
            out.append((c, "goal", True, ""))
        for c in P.goal_checks(env, strict_place=True):
            if c.kind == "place":
                out.append((c, "goal", None, "strict place: reported only (recording stops at LIBERO success)"))
    if survey is not None:
        for st in survey["subtasks"]:
            spec, reason = P.subtask_to_spec(st["name"], goals, movable)
            expected = P.outcome_expectation(st["outcome"])
            note = f"{st['name']}: {st['outcome']}"
            if spec is None:
                out.append((None, "survey", None, f"skipped ({reason}); {note}"))
            elif expected is None:
                out.append((P.parse_spec(spec), "survey", None, f"label not used; {note}"))
            else:
                out.append((P.parse_spec(spec), "survey", expected, note))
    return goals, out


def run_job(job):
    suite, entry = job["suite"], job["entry"]
    name = f"{suite}/{entry.get('episode_id', entry['folder'])}"
    try:
        env = worker_env(suite, entry["task_id"])
        trace_dir = common.ROLLOUTS / suite / entry["folder"] / "trace"
        states = np.load(trace_dir / "states.npy")
        final = np.load(trace_dir / "final_state.npy")
        sf = survey_file(suite, entry)
        survey = json.loads(sf.read_text()) if sf.exists() else None

        # Part A: LIBERO's verdict on the recorded final state
        P.set_state(env, final)
        final_success = P.task_success(env)

        # Part B: checks over all states in order (index len(states) = final_state)
        P.set_state(env, states[0])
        goals, items = build_checks(env, entry, survey)
        checks = [c for c, *_ in items if c is not None]
        for c in checks:
            c.start(env)
        seq = list(states) + [final]
        for k, s in enumerate(seq):
            if k:
                P.set_state(env, s)1-3 fps
            for c in checks:
                c.update(env, k)
        rows = []
        for c, source, expected, note in items:
            rows.append({"episode": name, "source": source, "kind": None if c is None else c.kind,
                         "spec": None if c is None else c.spec, "expected": expected,
                         "fired": None if c is None else c.done, "fired_at_step": None if c is None else c.done_step,
                         "agree": None if (c is None or expected is None) else (c.done == expected), "note": note})
        return {"episode": name, "recorded_success": bool(entry["success"]), "final_state_success": final_success,
                "goals": goals, "n_states": len(seq), "rows": rows, "error": None}
    except Exception as e:
        return {"episode": name, "recorded_success": bool(entry["success"]), "final_state_success": None,
                "goals": None, "n_states": 0, "rows": [], "error": repr(e)}


def report_md(results):
    lines = ["# Phase 1: subtask check validation", ""]
    errs = [r for r in results if r["error"]]
    a_ok = [r for r in results if not r["error"] and r["final_state_success"] == r["recorded_success"]]
    a_bad = [r for r in results if not r["error"] and r["final_state_success"] != r["recorded_success"]]
    lines += ["## Part A: LIBERO success recomputed on the recorded final state", "",
              f"- agree: {len(a_ok)} / {len(results) - len(errs)} episodes",
              f"- disagree: {len(a_bad)}" + ("".join(f"\n  - {r['episode']}: recorded {r['recorded_success']}, "
                                                    f"recomputed {r['final_state_success']}" for r in a_bad)),
              f"- errors: {len(errs)}" + "".join(f"\n  - {r['episode']}: {r['error']}" for r in errs), ""]
    rows = [row for r in results for row in r["rows"]]
    judged = [x for x in rows if x["agree"] is not None]
    lines += ["## Part B: subtask checks", "", "| source | check | judged | agree | disagree |", "|---|---|---|---|---|"]
    by = collections.defaultdict(list)
    for x in judged:
        by[(x["source"], x["kind"])].append(x)
    for (src, kind), xs in sorted(by.items()):
        n_ok = sum(x["agree"] for x in xs)
        lines.append(f"| {src} | {kind} | {len(xs)} | {n_ok} | {len(xs) - n_ok} |")
    lines += ["", "### Disagreements (look at the video around the step shown)", "",
              "| episode | source | check | expected | fired at step | label |", "|---|---|---|---|---|---|"]
    for x in judged:
        if not x["agree"]:
            lines.append(f"| {x['episode']} | {x['source']} | `{x['spec']}` | {'fire' if x['expected'] else 'no fire'} | "
                         f"{x['fired_at_step'] if x['fired'] else '-'} | {x['note']} |")
    strict = [x for x in rows if x["source"] == "goal" and x["kind"] == "place" and x["expected"] is None]
    lines += ["", f"### Strict place on success episodes (informational): fired in {sum(bool(x['fired']) for x in strict)}"
              f" / {len(strict)}", "",
              "### Skipped survey labels", ""]
    lines += [f"- {x['episode']}: {x['note']}" for x in rows if x["source"] == "survey" and x["expected"] is None]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_10", "libero_90"])
    p.add_argument("--episodes", nargs="+", default=None, help="only these, e.g. libero_90/task10_fail_init0")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    a = p.parse_args()

    jobs = []
    for suite in a.suites:
        meta = json.loads((common.ROLLOUTS / suite / "meta.json").read_text())
        for e in meta["episodes"]:
            name = f"{suite}/{e.get('episode_id', e['folder'])}"
            if a.episodes is None or name in a.episodes or f"{suite}/{e['folder']}" in a.episodes:
                jobs.append({"suite": suite, "entry": e})
    jobs.sort(key=lambda j: (j["suite"], j["entry"]["task_id"]))  # consecutive jobs reuse a worker's env
    print(f"{len(jobs)} episodes")
    if not jobs:
        sys.exit("no episodes selected")

    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap(run_job, jobs, chunksize=4), 1):
            results.append(r)
            judged = [x for x in r["rows"] if x["agree"] is not None]
            print(f"[{i}/{len(jobs)}] {r['episode']}: A {'ERROR ' + r['error'] if r['error'] else ('ok' if r['final_state_success'] == r['recorded_success'] else 'MISMATCH')}"
                  f", B {sum(x['agree'] for x in judged)}/{len(judged)} agree", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(results, indent=2, default=str))
    md = report_md(results)
    (a.out / "report.md").write_text(md)
    print("\n" + md.split("### Disagreements")[0])
    print(f"full report: {a.out / 'report.md'}")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
