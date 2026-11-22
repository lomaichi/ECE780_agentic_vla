"""Go/no-go for residual RL on pick: can a bounded RANDOM residual on top of pi0 turn pi0's pick failures
into successes, without breaking the picks pi0 already makes?

RL can only reinforce corrections that sometimes succeed. Before training anything, this sweeps the
residual bound eps and the noise's time correlation rho with no learning at all:

  fail starts   library.json rl/pick reset states (pre-failure states pi0 reached; pixel-Q training split)
  ok starts     task init states where pi0 usually picks fine (default libero_spatial 0-9, init 0)
Both are limited to --targets (default bowl and mug pick-and-place); other targets are skipped.

Every (start, eps, rho) condition runs --n episodes of pi0 + clip(noise, -eps, eps) (residual_env.py).
eps = 0 is plain pi0 from the same start: the baseline. The noise is Gaussian AR(1) per action dim,

    x_t = rho * x_{t-1} + sqrt(1 - rho^2) * sigma * xi_t,      sigma = --sigma-frac * eps (per dim)

rho = 0 is white noise (mostly averages out through the controller); rho = 0.95 (~1 s correlation) is a
slowly drifting offset, closer to what a learned state-dependent residual can produce.

Reading the result (summary.json, printed table):
  fail rate     pick success on fail starts. Compare with the eps = 0 row
  rescued       fail starts where pi0 alone got 0/n but this condition got >= 1: success pi0 cannot reach
                on its own but a bounded residual can
  ok rate       pick success on ok starts (do no harm). Should stay near the eps = 0 row
GO if some (eps, rho) lifts the fail rate or rescues starts while the ok rate holds; the largest such eps
is the residual bound to train with. NO-GO for pick if no eps ever beats the baseline.

Output (--out-dir): config.json, results.jsonl (one line per episode, appended as jobs finish; rerun with
--resume to continue), summary.json, videos/<group>/<start>/eps<e>_rho<r>_trial<k>_<success|fail>.mp4
(first --videos trials of each condition; wrist | agentview).

Run in the container's LIBERO client venv while the pi0_libero server is up (./serve_pi0.sh):
  cd /home/DockerShared/ECE780/pure_RL_prototype
  MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python noise_test.py --out-dir runs/noise_test
Smoke test:
  ... noise_test.py --out-dir runs/noise_smoke --max-starts 1 --n 2 --eps 0,0.2 --rho 0.95 --workers 2
Summary of an existing run only:
  ... noise_test.py --out-dir runs/noise_test --summarize
"""

import argparse
import json
import multiprocessing as mp
import pathlib
import time
from math import comb

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

_W = {}   # per-worker state: env (ResidualPickEnv), cfg


# ------------------------------------------------------------------ noise ---------------------------

class ARNoise(object):
    """Zero-mean Gaussian AR(1) per action dim with stationary std `sigma` (7,) and correlation `rho`."""

    def __init__(self, sigma, rho, rng):
        self.sigma, self.rho, self.rng = np.asarray(sigma, dtype=np.float64), float(rho), rng
        self.x = None

    def reset(self):
        self.x = self.sigma * self.rng.standard_normal(self.sigma.shape)

    def __call__(self, obs, a_pi0):
        out = self.x.copy()
        self.x = self.rho * self.x + np.sqrt(1.0 - self.rho ** 2) * self.sigma * self.rng.standard_normal(
            self.sigma.shape)
        return out


# ------------------------------------------------------------------ worker --------------------------

def _init_worker(cfg, counter):
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    import residual_env as RE

    cfg = dict(cfg, worker_id=wid)
    _W["RE"], _W["cfg"] = RE, cfg
    _W["env"] = RE.ResidualPickEnv(cfg)


def run_job(job):
    """All --n trials of one (start, eps, rho). Returns {"job": ..., "rows": [...]} or an ineligible note."""
    RE, cfg, env = _W["RE"], _W["cfg"], _W["env"]
    start, eps, rho = job["start"], job["eps"], job["rho"]
    eps_vec = RE.eps_vector(eps, cfg["dims"], cfg["rot_scale"])
    rows = []
    for k in range(cfg["n"]):
        rng = np.random.default_rng([cfg["seed"], job["start_idx"], job["cond_idx"], k])
        noise = ARNoise(cfg["sigma_frac"] * eps_vec, rho, rng) if eps > 0 else None
        record = k < cfg["videos"]
        try:
            r = env.rollout(start, noise, eps_vec, cfg["horizon"], record=record)
        except (RE.IneligibleStart, RE.ResetFailed) as e:
            return {"job": _job_key(job), "group": job["group"], "ineligible": "{}: {}".format(type(e).__name__, e)}
        except Exception as e:  # keep the sweep alive; the job is not recorded, so --resume retries it
            import traceback

            return {"job": _job_key(job), "group": job["group"], "error": traceback.format_exc(limit=3)}
        frames = r.pop("frames")
        if record and frames:
            import imageio

            vdir = pathlib.Path(cfg["out_dir"]) / "videos" / job["group"] / start["tag"]
            vdir.mkdir(parents=True, exist_ok=True)
            name = "eps{:g}_rho{:g}_trial{}_{}.mp4".format(eps, rho, k, "success" if r["success"] else "fail")
            imageio.mimwrite(str(vdir / name), frames, fps=20)
        rows.append(dict(group=job["group"], start=start["tag"], eps=eps, rho=rho, trial=k, **r))
    return {"job": _job_key(job), "group": job["group"], "rows": rows}


def _job_key(job):
    return "{}|{}|{:g}|{:g}".format(job["group"], job["start"]["tag"], job["eps"], job["rho"])


# ------------------------------------------------------------------ summary -------------------------

def _pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def summarize(rows, ineligible=()):
    """Per-condition and per-start success, rescued starts, and a suggested verdict."""
    cells = {}                                       # (group, eps, rho, start) -> [n, c]
    for r in rows:
        key = (r["group"], r["eps"], r["rho"], r["start"])
        n_c = cells.setdefault(key, [0, 0])
        n_c[0] += 1
        n_c[1] += int(r["success"])
    conds = sorted({(k[1], k[2]) for k in cells})
    groups = sorted({k[0] for k in cells})
    starts = {g: sorted({k[3] for k in cells if k[0] == g}) for g in groups}

    def cell(g, e, rh, s):
        return cells.get((g, e, rh, s))

    baseline = {(g, s): cell(g, 0.0, 0.0, s) for g in groups for s in starts[g]}
    table = []
    for e, rh in conds:
        row = {"eps": e, "rho": rh}
        for g in groups:
            got = [(s, cell(g, e, rh, s)) for s in starts[g] if cell(g, e, rh, s)]
            n = sum(nc[0] for _, nc in got)
            c = sum(nc[1] for _, nc in got)
            row[g] = {"n": n, "success": c, "rate": round(c / n, 3) if n else None,
                      "starts": len(got), "starts_with_success": sum(1 for _, nc in got if nc[1] > 0),
                      "mean_pass@1": round(float(np.mean([_pass_at_k(nc[0], nc[1], 1) for _, nc in got])), 3)
                      if got else None}
            if e > 0:
                row[g]["rescued"] = sorted(s for s, nc in got if nc[1] > 0 and baseline.get((g, s))
                                           and baseline[(g, s)][1] == 0)
                # paired with pi0: this condition's rate minus the eps = 0 rate, over the same starts only
                paired = [(nc, baseline[(g, s)]) for s, nc in got if baseline.get((g, s))]
                row[g]["paired_starts"] = len(paired)
                row[g]["vs_pi0"] = round(sum(nc[1] for nc, _ in paired) / sum(nc[0] for nc, _ in paired)
                                         - sum(b[1] for _, b in paired) / sum(b[0] for _, b in paired), 3) \
                    if paired else None
        table.append(row)
    per_start = {g: {s: {"{:g}/{:g}".format(e, rh): cell(g, e, rh, s) for e, rh in conds if cell(g, e, rh, s)}
                     for s in starts[g]} for g in groups}
    return {"conditions": table, "per_start": per_start, "ineligible": list(ineligible),
            "verdict": _verdict(table)}


def _verdict(table, max_harm=0.05, min_lift=0.10, min_rescued=2):
    """Suggested GO / NO-GO (a heuristic; read the table): some eps > 0 must lift the fail-start rate by
    >= min_lift or rescue >= min_rescued starts, while the ok-start rate drops by <= max_harm. Lift and
    drop are paired with pi0 (vs_pi0: same starts only), so a partial run is not compared across starts."""
    good = []
    for r in table:
        if r["eps"] == 0 or "fail" not in r or r["fail"].get("vs_pi0") is None:
            continue
        lift = r["fail"]["vs_pi0"]
        harm = None
        if "ok" in r and r["ok"].get("vs_pi0") is not None:
            harm = -r["ok"]["vs_pi0"]
        if (lift >= min_lift or len(r["fail"]["rescued"]) >= min_rescued) and (harm is None or harm <= max_harm):
            good.append((r["eps"], r["rho"], round(lift, 3), harm))
    if not any(r["eps"] == 0 for r in table):
        return "no eps = 0 baseline yet"
    if not good:
        return ("NO-GO (suggested): no eps > 0 lifted the fail-start rate by >= {} or rescued >= {} starts "
                "with ok-start harm <= {}".format(min_lift, min_rescued, max_harm))
    best = max(good, key=lambda g: (g[0], g[2]))
    return ("GO (suggested): {} condition(s) pass; largest passing eps = {:g} (rho {:g}, fail-rate lift {:+.2f}, "
            "ok-rate drop {})".format(len(good), best[0], best[1], best[2],
                                      "n/a" if best[3] is None else "{:+.2f}".format(best[3])))


def print_summary(s):
    groups = [g for g in ("fail", "ok") if any(g in r for r in s["conditions"])]
    block = "| {:>5} {:>9} {:>8} {:>10} {:>7} "
    top = "{:>11} ".format("") + "".join("| {:<44}".format(g + " starts") for g in groups)
    head = "{:>6} {:>4} ".format("eps", "rho") + "".join(
        block.format("rate", "succ/n", "starts>0", "vs pi0", "rescued") for _ in groups)
    print(top)
    print(head)
    print("-" * len(head))
    for r in s["conditions"]:
        line = "{:>6g} {:>4g} ".format(r["eps"], r["rho"])
        for g in groups:
            x = r.get(g)
            if not x or x["rate"] is None:
                line += block.format("-", "-", "-", "-", "-")
                continue
            vs = "-"
            if r["eps"] > 0:
                vs = "n/a" if x["vs_pi0"] is None else "{:+.2f} ({})".format(x["vs_pi0"], x["paired_starts"])
            line += block.format("{:.2f}".format(x["rate"]), "{}/{}".format(x["success"], x["n"]),
                                 "{}/{}".format(x["starts_with_success"], x["starts"]), vs,
                                 len(x.get("rescued", [])) if r["eps"] > 0 else "-")
        print(line)
    print("vs pi0 = rate minus pi0's (eps = 0) rate over the same starts; (k) = number of such starts")
    if s["ineligible"]:
        print("ineligible starts:", *s["ineligible"], sep="\n  ")
    print(s["verdict"])


# ------------------------------------------------------------------ main ----------------------------

def _floats(text):
    return [float(x) for x in str(text).split(",") if x.strip()]


def _load_rows(path):
    rows, inel = [], []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                (inel if "ineligible" in d else rows).append(d)
    return rows, inel


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eps", default="0,0.05,0.1,0.2,0.4",
                   help="residual bounds in OSC input units (1.0 = 5 cm / step); 0 = plain pi0 baseline")
    p.add_argument("--rho", default="0,0.95", help="AR(1) noise correlation per control step")
    p.add_argument("--sigma-frac", type=float, default=1.0, help="noise std as a fraction of eps")
    p.add_argument("--dims", default="0,1,2,3,4,5", help="action dims the residual may change (6 = gripper)")
    p.add_argument("--rot-scale", type=float, default=0.5, help="eps on rotation dims = eps * rot-scale")
    p.add_argument("--n", type=int, default=16, help="episodes per (start, eps, rho)")
    p.add_argument("--horizon", type=int, default=200, help="control steps per episode (20 Hz)")
    p.add_argument("--policy", default="rl/pick", help="library.json rl_policies entry for the fail starts")
    p.add_argument("--ok-tasks", default="libero_spatial:0-9",
                   help="task-init starts for the do-no-harm check ('' to skip), e.g. libero_spatial:0-9")
    p.add_argument("--ok-inits", default="0", help="init state indexes for --ok-tasks")
    p.add_argument("--targets", default="bowl,mug",
                   help="only starts whose pick target's name contains one of these words ('' = all)")
    p.add_argument("--max-starts", type=int, default=None, help="use only the first N starts per group (smoke)")
    p.add_argument("--videos", type=int, default=1, help="record the first N trials of each condition")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-seed", type=int, default=7)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--resume", action="store_true", help="skip conditions already in results.jsonl")
    p.add_argument("--summarize", action="store_true", help="only summarize an existing results.jsonl")
    a = p.parse_args()

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"

    if a.summarize:
        rows, inel = _load_rows(res_path)
        s = summarize(rows, [d["job"] + ": " + d["ineligible"] for d in inel])
        (out / "summary.json").write_text(json.dumps(s, indent=2))
        print_summary(s)
        return

    import residual_env as RE   # after argparse so --help works without LIBERO

    eps_list, rho_list = _floats(a.eps), _floats(a.rho)
    conds = [(0.0, 0.0)] if 0.0 in eps_list else []
    conds += [(e, r) for e in eps_list if e > 0 for r in rho_list]

    targets = [w.strip().lower() for w in a.targets.split(",") if w.strip()]
    fail_starts, skipped = RE.SC.library_starts(a.policy)
    if targets:   # library label (e.g. black_bowl, white_mug); the env checks the sim object again
        skipped += [(s["tag"], "target {!r} not in --targets".format(s.get("word"))) for s in fail_starts
                    if not any(w in str(s.get("word")).lower() for w in targets)]
        fail_starts = [s for s in fail_starts if any(w in str(s.get("word")).lower() for w in targets)]
    ok_starts = RE.SC.task_starts([t for t in a.ok_tasks.split(";") if t.strip()],
                                  [int(x) for x in a.ok_inits.split(",") if x.strip()]) if a.ok_tasks else []
    if a.max_starts is not None:
        fail_starts, ok_starts = fail_starts[:a.max_starts], ok_starts[:a.max_starts]
    for tag, why in skipped:
        print("skip {}: {}".format(tag, why))

    cfg = {"out_dir": str(out), "n": a.n, "horizon": a.horizon, "videos": a.videos, "seed": a.seed,
           "sigma_frac": a.sigma_frac, "rot_scale": a.rot_scale, "dims": [int(x) for x in a.dims.split(",")],
           "env_seed": a.env_seed, "host": a.host, "port": a.port, "targets": targets}
    (out / "config.json").write_text(json.dumps(dict(vars(a), conditions=conds,
                                                     fail_starts=[s["tag"] for s in fail_starts],
                                                     ok_starts=[s["tag"] for s in ok_starts]), indent=2))

    jobs = []
    for si, (group, start) in enumerate([("fail", s) for s in fail_starts] + [("ok", s) for s in ok_starts]):
        for ci, (e, r) in enumerate(conds):
            jobs.append({"group": group, "start": start, "start_idx": si, "eps": e, "rho": r, "cond_idx": ci})

    done = set()
    if a.resume:
        rows, inel = _load_rows(res_path)
        counts = {}
        for r in rows:
            k = "{}|{}|{:g}|{:g}".format(r["group"], r["start"], r["eps"], r["rho"])
            counts[k] = counts.get(k, 0) + 1
        done = {k for k, n in counts.items() if n >= a.n} | {d["job"] for d in inel}
    elif res_path.exists():
        res_path.unlink()
    todo = [j for j in jobs if _job_key(j) not in done]
    print("{} starts (fail {}, ok {}), {} conditions, {} jobs to run ({} done), {} episodes each".format(
        len(fail_starts) + len(ok_starts), len(fail_starts), len(ok_starts), len(conds), len(todo),
        len(jobs) - len(todo), a.n), flush=True)

    t0 = time.time()
    counter = mp.get_context("spawn").Value("i", 0)
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(cfg, counter)) as pool, \
            open(res_path, "a") as f:
        for i, res in enumerate(pool.imap_unordered(run_job, todo), 1):
            if "error" in res:
                msg = "ERROR (not recorded; rerun with --resume):\n" + res["error"]
            elif "ineligible" in res:
                f.write(json.dumps({"job": res["job"], "ineligible": res["ineligible"]}) + "\n")
                msg = "ineligible: " + res["ineligible"]
            else:
                for r in res["rows"]:
                    f.write(json.dumps(r) + "\n")
                c = sum(r["success"] for r in res["rows"])
                msg = "{}/{} success, {:.0f} s/episode".format(
                    c, len(res["rows"]), np.mean([r["wall_s"] for r in res["rows"]]))
            f.flush()
            print("[{}/{}] {:.0f} min  {}  {}".format(i, len(todo), (time.time() - t0) / 60, res["job"], msg),
                  flush=True)

    rows, inel = _load_rows(res_path)
    s = summarize(rows, [d["job"] + ": " + d["ineligible"] for d in inel])
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print_summary(s)


if __name__ == "__main__":
    main()
