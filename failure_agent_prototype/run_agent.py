"""Run pi0 on LIBERO with the failure agent.

Examples (inside the LIBERO client venv, policy server running):
  python run_agent.py --tasks 62 71 --mode both
  python run_agent.py --tasks survey --init 0 1 2 --workers 6
  python run_agent.py --tasks all --mode observe --no-video

With --workers N, tasks are pulled from a shared queue by N processes (one
websocket client each; the policy server serializes inference). Console
output is printed one finished episode at a time.
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import pathlib
import sys
import traceback

import yaml

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from failure_agent.library import SkillLibrary  # noqa: E402

NATIVE_RESULTS = ROOT.parent / "Documentation" / "native_pi0_libero90" / "results.json"


def survey_failures():
    """Failed libero_90 tasks in the native pi0 rollouts (Documentation/native_pi0_libero90)."""
    rows = json.loads(NATIVE_RESULTS.read_text())
    return sorted(r["task"] for r in rows if not r["success"])


def run_tasks(cfg, tasks, inits, modes, out, save_video, stats_path, emit):
    """Run every (task, init, mode) for tasks taken from `tasks` (a list or a
    queue). `emit(row, lines)` receives each finished episode."""
    from openpi_client import websocket_client_policy
    from failure_agent.runner import EpisodeRunner

    client = websocket_client_policy.WebsocketClientPolicy(cfg["policy"]["host"], cfg["policy"]["port"])
    library = SkillLibrary(str(ROOT / cfg["agent"]["library"]), str(stats_path))
    lines = []
    runner = EpisodeRunner(cfg, client, library, out, save_video=save_video, log=lines.append)
    it = iter(tasks.get, None) if hasattr(tasks, "get") else iter(tasks)
    try:
        for tid in it:
            for idx in inits:
                for mode in modes:
                    lines.clear()
                    lines.append(f"task {tid} init {idx} [{mode}]")
                    try:
                        tr = runner.run(tid, idx, mode)
                    except Exception:
                        lines.append(traceback.format_exc())
                        emit({"task": tid, "init": idx, "mode": mode, "success": False, "success_t": None,
                              "steps": 0, "chain": "ERROR", "instruction": ""}, list(lines))
                        runner.release_env(tid)
                        continue
                    chain = " > ".join(f"{d['trigger']}->{d['code']}" for d in tr["diagnoses"]) or "-"
                    lines.append(f"    success={tr['success']} steps={tr['steps']} diagnoses: {chain}")
                    emit({"task": tid, "init": idx, "mode": mode, "success": tr["success"],
                          "success_t": tr["success_t"], "steps": tr["steps"], "chain": chain,
                          "instruction": tr["instruction"]}, list(lines))
                    library.save_stats()
            runner.release_env(tid)
    finally:
        runner.close()
        library.save_stats()


def _worker(i, cfg, queue, inits, modes, out, save_video, results):
    def emit(row, lines):
        results.put((row, lines))
    try:
        run_tasks(cfg, queue, inits, modes, out, save_video, out / f"library_stats_w{i}.json", emit)
    finally:
        results.put(None)


def merge_stats(parts, dest: pathlib.Path):
    total = json.loads(dest.read_text()) if dest.exists() else {}
    for p in parts:
        for code, s in json.loads(p.read_text()).items():
            t = total.setdefault(code, {"attempts": 0, "recovered": 0})
            t["attempts"] += s["attempts"]
            t["recovered"] += s["recovered"]
    dest.write_text(json.dumps(total, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["survey"],
                    help="task ids, 'survey' (failures in the native pi0 rollouts) or 'all'")
    ap.add_argument("--init", nargs="+", type=int, default=[0], help="initial-state indices (0-49)")
    ap.add_argument("--mode", choices=["observe", "agent", "both"], default="both")
    ap.add_argument("--config", default=str(ROOT / "config" / "agent.yaml"))
    ap.add_argument("--out", default=None, help="output dir (default logs/run_<timestamp>)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    if args.tasks == ["survey"]:
        tasks = survey_failures()
    elif args.tasks == ["all"]:
        tasks = list(range(90))
    else:
        tasks = [int(t) for t in args.tasks]
    modes = ["observe", "agent"] if args.mode == "both" else [args.mode]
    out = pathlib.Path(args.out) if args.out else \
        ROOT / "logs" / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    rows = []

    def emit(row, lines):
        rows.append(row)
        print("\n".join(lines), flush=True)
        (out / "summary.json").write_text(json.dumps(
            sorted(rows, key=lambda r: (r["task"], r["init"], modes.index(r["mode"]))), indent=1))

    for old in out.glob("library_stats_w*.json"):
        old.unlink()
    n = min(args.workers, len(tasks))
    if n <= 1:
        run_tasks(cfg, tasks, args.init, modes, out, not args.no_video, out / "library_stats_w0.json", emit)
    else:
        ctx = mp.get_context("fork")
        queue, results = ctx.Queue(), ctx.Queue()
        for t in tasks:
            queue.put(t)
        for _ in range(n):
            queue.put(None)
        procs = [ctx.Process(target=_worker, args=(i, cfg, queue, args.init, modes, out,
                                                   not args.no_video, results)) for i in range(n)]
        for p in procs:
            p.start()
        done = 0
        while done < n:
            item = results.get()
            if item is None:
                done += 1
            else:
                emit(*item)
        for p in procs:
            p.join()
    parts = sorted(out.glob("library_stats_w*.json"))
    merge_stats(parts, ROOT / "logs" / "library_stats.json")
    merge_stats(parts, out / "library_stats.json")
    for p in parts:
        p.unlink()

    rows.sort(key=lambda r: (r["task"], r["init"], modes.index(r["mode"])))
    print("\n%-5s %-4s %-8s %-7s %-6s %s" % ("task", "init", "mode", "success", "t", "diagnoses"))
    for r in rows:
        print("%-5d %-4d %-8s %-7s %-6s %s" % (r["task"], r["init"], r["mode"], r["success"],
                                               r["success_t"] or "-", r["chain"]))
    for m in modes:
        sel = [r for r in rows if r["mode"] == m]
        print(f"{m}: {sum(r['success'] for r in sel)}/{len(sel)} succeeded")
    print(f"logs: {out}")


if __name__ == "__main__":
    main()
