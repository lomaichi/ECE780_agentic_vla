"""Summarize pi0 success rates from a record_suites.py output folder.

Reads OUT_ROOT/<suite>/meta.json for every suite present and reports, per suite
(and per library.json split for libero_90): episodes, successes, success rate, and
two 95% confidence intervals:

  task-bootstrap  resamples whole tasks (with replacement). Use this one: episodes of
                  the same task are strongly correlated (a task tends to go 0/10 or
                  10/10), so treating all episodes as independent is too optimistic.
  wilson          binomial interval over episodes, assuming independence (for reference).

Also prints a per-task table and warns about incomplete runs (tasks with fewer init
states than the others), sim errors, and a meta.json whose policy is not pi0_libero.

Pure standard library; runs in any Python >= 3.8 (no LIBERO needed):
  python baseline_summary.py ../../Documentation/pi0_baseline_heldout
Writes OUT_ROOT/summary.json and OUT_ROOT/summary.md.

Comparing two runs (e.g. the Phase 1 no-op loop vs the baseline, or later a method vs the baseline):
  python baseline_summary.py NEW_OUT_ROOT --compare BASELINE_OUT_ROOT
For every suite/split in both, on the tasks both contain: success rate of each run, the difference
(new - baseline) and a paired task-bootstrap 95% CI of the difference (tasks resampled together for
both runs). "no detectable difference" = the CI contains 0. Also written to NEW_OUT_ROOT/compare.json.
"""

import argparse
import collections
import json
import math
import pathlib
import random

LIBRARY = pathlib.Path(__file__).resolve().parents[1] / "failure_survey" / "skill_library" / "library.json"
Z95 = 1.959963984540054
N_BOOT = 10000


def wilson(c, n, z=Z95):
    """95% Wilson score interval for c successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = c / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def task_bootstrap(per_task, n_boot=N_BOOT, seed=0):
    """95% percentile CI of the pooled success rate, resampling tasks.

    per_task: list of (successes, episodes), one per task.
    """
    if not per_task:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(per_task)
    rates = []
    for _ in range(n_boot):
        draw = [per_task[rng.randrange(k)] for _ in range(k)]
        n = sum(e for _, e in draw)
        rates.append(sum(c for c, _ in draw) / n)
    rates.sort()
    return (rates[int(0.025 * n_boot)], rates[int(0.975 * n_boot) - 1])


def group_stats(episodes):
    """Pooled stats for a list of meta.json episode entries."""
    by_task = collections.defaultdict(list)
    for e in episodes:
        by_task[e["task_id"]].append(e)
    per_task = [(sum(x["success"] for x in eps), len(eps)) for _, eps in sorted(by_task.items())]
    n = len(episodes)
    c = sum(e["success"] for e in episodes)
    return {
        "tasks": len(by_task),
        "episodes": n,
        "successes": c,
        "success_rate": c / n if n else float("nan"),
        "ci95_task_bootstrap": task_bootstrap(per_task),
        "ci95_wilson": wilson(c, n),
        "tasks_with_any_success": sum(1 for s, _ in per_task if s > 0),
        "sim_errors": sum(1 for e in episodes if e.get("error")),
    }


def per_task_rows(episodes):
    by_task = collections.defaultdict(list)
    for e in episodes:
        by_task[e["task_id"]].append(e)
    rows = []
    for tid, eps in sorted(by_task.items()):
        rows.append({
            "task_id": tid,
            "task_language": eps[0]["task_language"],
            "episodes": len(eps),
            "successes": sum(e["success"] for e in eps),
            "init_states": sorted(e["init_state_idx"] for e in eps),
        })
    return rows


def pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"  # x != x is True for NaN


def fmt_ci(ci):
    return f"[{pct(ci[0])}, {pct(ci[1])}]"


def load_groups(out_root, library=LIBRARY):
    """{suite: {group: [episodes]}} with groups all (+ survey, held_out for libero_90)."""
    held_out = set(json.loads(pathlib.Path(library).read_text())["split_guard"]["held_out_tasks"])
    out = {}
    for meta_path in sorted(pathlib.Path(out_root).glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        eps = meta["episodes"]
        groups = {"all": eps}
        if meta["task_suite"] == "libero_90":
            groups["survey"] = [e for e in eps if e["task_id"] not in held_out]
            groups["held_out"] = [e for e in eps if e["task_id"] in held_out]
        out[meta["task_suite"]] = {g: v for g, v in groups.items() if v}
    return out


def paired_diff(eps_new, eps_base, n_boot=N_BOOT, seed=0):
    """Difference in pooled success rate (new - base) over the tasks both runs contain."""
    def per_task(eps):
        d = collections.defaultdict(lambda: [0, 0])
        for e in eps:
            d[e["task_id"]][0] += bool(e["success"])
            d[e["task_id"]][1] += 1
        return d

    a, b = per_task(eps_new), per_task(eps_base)
    tasks = sorted(set(a) & set(b))
    if not tasks:
        return None
    pairs = [(a[t], b[t]) for t in tasks]

    def rates(sample):
        na, nb = sum(x[0][1] for x in sample), sum(x[1][1] for x in sample)
        return sum(x[0][0] for x in sample) / na, sum(x[1][0] for x in sample) / nb

    r_new, r_base = rates(pairs)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        ra, rb = rates([pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))])
        diffs.append(ra - rb)
    diffs.sort()
    ci = (diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1])
    return {"tasks": len(tasks), "episodes_new": sum(p[0][1] for p in pairs),
            "episodes_base": sum(p[1][1] for p in pairs), "rate_new": r_new, "rate_base": r_base,
            "diff": r_new - r_base, "ci95_diff_task_bootstrap": ci,
            "verdict": "no detectable difference" if ci[0] <= 0 <= ci[1] else
                       ("new is higher" if ci[0] > 0 else "new is lower")}


def compare(new_root, base_root, library=LIBRARY):
    new, base = load_groups(new_root, library), load_groups(base_root, library)
    out = {}
    for suite in sorted(set(new) & set(base)):
        for g in sorted(set(new[suite]) & set(base[suite])):
            d = paired_diff(new[suite][g], base[suite][g])
            if d:
                out[f"{suite}/{g}"] = d
    return out


def summarize(out_root, library=LIBRARY):
    out_root = pathlib.Path(out_root)
    metas = sorted(out_root.glob("*/meta.json"))
    if not metas:
        raise SystemExit(f"no <suite>/meta.json under {out_root}")
    held_out = set(json.loads(pathlib.Path(library).read_text())["split_guard"]["held_out_tasks"])

    report, warnings = {}, []
    for meta_path in metas:
        meta = json.loads(meta_path.read_text())
        suite = meta["task_suite"]
        eps = meta["episodes"]
        if "pi0_libero" not in meta.get("policy", ""):
            warnings.append(f"{suite}: meta.json policy is {meta.get('policy')!r}, not pi0_libero")
        groups = {"all": eps}
        if suite == "libero_90":
            groups["survey"] = [e for e in eps if e["task_id"] not in held_out]
            groups["held_out"] = [e for e in eps if e["task_id"] in held_out]
        rows = per_task_rows(eps)
        counts = {r["episodes"] for r in rows}
        if len(counts) > 1:
            short = [r["task_id"] for r in rows if r["episodes"] < max(counts)]
            warnings.append(f"{suite}: uneven episodes per task ({sorted(counts)}); short tasks: {short} "
                            f"(rerun record_suites.py with the same arguments to finish them)")
        n_err = sum(1 for e in eps if e.get("error"))
        if n_err:
            warnings.append(f"{suite}: {n_err} episodes ended in a sim error (counted as failures, as in main.py)")
        report[suite] = {
            "meta": str(meta_path),
            "max_steps": meta.get("max_steps"),
            "env_seed": meta.get("env_seed"),
            "groups": {g: group_stats(v) for g, v in groups.items() if v},
            "per_task": rows,
        }
    return {"out_root": str(out_root), "results": report, "warnings": warnings}


def to_markdown(summary):
    lines = [f"# pi0 baseline summary: `{summary['out_root']}`", ""]
    lines += ["| suite | split | tasks | episodes | successes | success rate | 95% CI (task bootstrap) "
              "| 95% CI (Wilson, episodes iid) | tasks with any success |",
              "|---|---|---|---|---|---|---|---|---|"]
    for suite, r in summary["results"].items():
        for g, s in r["groups"].items():
            lines.append(f"| {suite} | {g} | {s['tasks']} | {s['episodes']} | {s['successes']} | "
                         f"{pct(s['success_rate'])} | {fmt_ci(s['ci95_task_bootstrap'])} | "
                         f"{fmt_ci(s['ci95_wilson'])} | {s['tasks_with_any_success']}/{s['tasks']} |")
    for suite, r in summary["results"].items():
        lines += ["", f"## {suite}: per task", "", "| task | successes / episodes | instruction |", "|---|---|---|"]
        for row in r["per_task"]:
            lines.append(f"| {row['task_id']} | {row['successes']}/{row['episodes']} | {row['task_language']} |")
    if summary["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in summary["warnings"]]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_root", help="folder given to record_suites.py --out-root")
    p.add_argument("--library", default=str(LIBRARY), help="library.json with split_guard.held_out_tasks")
    p.add_argument("--compare", default=None, metavar="BASELINE_OUT_ROOT",
                   help="also compare with another run (paired by task)")
    a = p.parse_args()
    summary = summarize(a.out_root, a.library)
    md = to_markdown(summary)
    out = pathlib.Path(a.out_root)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(md)
    for suite, r in summary["results"].items():
        for g, s in r["groups"].items():
            print(f"{suite:15s} {g:9s} {s['successes']:4d}/{s['episodes']:<4d} = {pct(s['success_rate']):>6s}  "
                  f"95% CI {fmt_ci(s['ci95_task_bootstrap'])} (task bootstrap)  "
                  f"{s['tasks_with_any_success']}/{s['tasks']} tasks with any success")
    for w in summary["warnings"]:
        print("WARNING:", w)
    print(f"wrote {out / 'summary.md'} and {out / 'summary.json'}")
    if a.compare:
        cmp = compare(a.out_root, a.compare, a.library)
        if not cmp:
            print(f"\ncompare: no suite/split with common tasks in {a.compare}")
        for key, d in cmp.items():
            print(f"\ncompare {key} on {d['tasks']} common tasks: new {pct(d['rate_new'])} ({d['episodes_new']} eps) "
                  f"vs baseline {pct(d['rate_base'])} ({d['episodes_base']} eps): diff {100 * d['diff']:+.1f} pts, "
                  f"95% CI [{100 * d['ci95_diff_task_bootstrap'][0]:+.1f}, {100 * d['ci95_diff_task_bootstrap'][1]:+.1f}]"
                  f" -> {d['verdict']}")
        (out / "compare.json").write_text(json.dumps({"baseline": str(a.compare), "results": cmp}, indent=2))


if __name__ == "__main__":
    main()
