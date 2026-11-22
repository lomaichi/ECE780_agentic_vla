"""Metrics and reports for V2 (pixel_Q_RL_V2.md sec. 9.5, 10.5, 11). Pure functions over rollout results
(rollout.main_rollout dicts); train.py runs the rollouts and writes the files.

  branch_metrics     at branch points: no-op vs intervention labels (paired), Q0 / Q(a) calibration, the
                     policy gain of following the Q-map (correct where A(s) > 0, else let pi0 continue)
  rollout_metrics    main rollouts: pick / task success, end reasons, interventions and HL corrections
  reliability        Q0 reliability table (predicted Q0 bins -> mean label, success fraction)
  noise_floor        A(s) on states where a correction cannot matter -> tau_noise (95th percentile)
  knee               the knee of a (interventions, success) curve
  phase0_report      levelling probes vs no-op by tilt bin (paired difference +- SE), time to success
  virtual_trigger / timing_report   when the trigger would fire on rollouts it does not act on
"""

import numpy as np

from rollout import TILT_BINS, label_at


def _mean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def _se(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def used(results):
    return [r for r in results if not r.get("skipped") and not r.get("reset_failed")]


def branch_metrics(results, tau=None):
    brs = [b for r in used(results) for b in r["branches"]]
    out = {"n_branch_points": len(brs)}
    if not brs:
        return out
    y0 = np.array([b["y_noop"] for b in brs])
    ya = np.array([b["y_a"] for b in brs])
    adv = ya - y0
    out.update(y_noop=float(y0.mean()), y_a=float(ya.mean()), adv=float(adv.mean()), adv_se=_se(adv),
               success_noop=float((y0 > 0).mean()), success_a=float((ya > 0).mean()),
               prim_steps=_mean([(b.get("prim") or {}).get("steps") for b in brs]),
               prim_reached=_mean([float(bool((b.get("prim") or {}).get("reached"))) for b in brs if b.get("prim")]),
               branch_wall_s=_mean([b.get("wall_s") for b in brs]))
    scored = [b for b in brs if b.get("q0") is not None]
    if scored:
        A = np.array([b["A"] for b in scored])
        q0 = np.array([b["q0"] for b in scored])
        qa = np.array([b["q_a"] for b in scored])
        s0 = np.array([b["y_noop"] for b in scored])
        sa = np.array([b["y_a"] for b in scored])
        sadv = sa - s0
        out.update(n_scored=len(scored), A_mean=float(A.mean()), frac_A_pos=float((A > 0).mean()),
                   policy_gain=float(np.where(A > 0, sadv, 0.0).mean()),
                   policy_gain_se=_se(np.where(A > 0, sadv, 0.0)),
                   adv_when_A_pos=_mean(sadv[A > 0]), adv_when_A_neg=_mean(sadv[A <= 0]),
                   q0_bias=float((q0 - s0).mean()), q0_mse=float(((q0 - s0) ** 2).mean()),
                   qa_bias=float((qa - sa).mean()), q_corr_minus_q0=float((qa - q0).mean()),
                   corr_A_adv=float(np.corrcoef(A, sadv)[0, 1]) if len(A) > 2 and A.std() > 0 and sadv.std() > 0
                   else float("nan"))
        if tau is not None:
            out.update(frac_A_tau=float((A > tau).mean()), policy_gain_tau=float(np.where(A > tau, sadv, 0.0).mean()))
    return out


def rollout_metrics(results):
    u = used(results)
    ends = {}
    for r in u:
        k = str(r.get("end")).split(":")[0]
        ends[k] = ends.get(k, 0) + 1
    gated = sum(r["counts"]["gated"] for r in u) if u else 0
    eligible = sum(r["counts"]["eligible"] for r in u) if u else 0
    n_int = sum(len(r["interventions"]) for r in u)
    return {"n_rollouts": len(u), "pick_success": _mean([float(r["pick_success"]) for r in u]),
            "task_success": _mean([float(r["task_success"]) for r in u]),
            "interventions_per_rollout": _mean([len(r["interventions"]) for r in u]),
            "hl_per_rollout": _mean([len(r["hl"]) for r in u]),
            "steps_to_pick_s": _mean([r["pick_step"] / 20.0 for r in u if r.get("pick_step") is not None]),
            "gated_ticks_per_rollout": _mean([r["counts"]["gated"] for r in u]),
            "gated_ticks": gated, "eligible_ticks": eligible, "interventions": n_int,
            "fire_rate_per_eligible_tick": n_int / float(eligible) if eligible else float("nan"),
            "ends": ends, "skipped": len([r for r in results if r.get("skipped")]),
            "reset_failed": len([r for r in results if r.get("reset_failed")])}


def per_start(results, key="pick_success"):
    per = {}
    for r in used(results):
        per.setdefault(r["tag"], []).append(float(r[key]))
    return {k: float(np.mean(v)) for k, v in sorted(per.items())}


def reliability(brs, n_bins=10):
    """Q0 reliability over branch points: bins of predicted Q0 -> mean y_noop and fraction of no-op successes."""
    sel = [b for b in brs if b.get("q0") is not None]
    q0 = np.array([b["q0"] for b in sel], float)
    y0 = np.array([b["y_noop"] for b in sel], float)
    idx = np.clip(np.floor(q0 * n_bins).astype(int), 0, n_bins - 1) if q0.size else np.zeros(0, int)
    rows = []
    for k in range(n_bins):
        lo, hi = k / float(n_bins), (k + 1) / float(n_bins)
        m = idx == k
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()), "q0_mean": _mean(q0[m]) if m.any() else float("nan"),
                     "y_mean": _mean(y0[m]) if m.any() else float("nan"),
                     "success_frac": _mean((y0[m] > 0).astype(float)) if m.any() else float("nan")})
    return rows


def _pcts(v):
    v = np.asarray([x for x in v if x is not None], float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def noise_floor(brs, gate_m, fast_steps=20, edge_band_m=0.01):
    """sec. 11: states where the true advantage is ~0 -- the no-op branch succeeds within 1 s (fast), or the
    target is at the edge of the gate (within edge_band_m of r + 4 cm). tau_noise = 95th pct of A there."""
    sb = [b for b in brs if b.get("A") is not None]
    fast = [b["A"] for b in sb if b.get("t_noop") is not None and b["t_noop"] <= fast_steps]
    edge = [b["A"] for b in sb if b.get("target_dxy") is not None and b["target_dxy"] >= gate_m - edge_band_m]
    union = [b["A"] for b in sb if (b.get("t_noop") is not None and b["t_noop"] <= fast_steps) or
             (b.get("target_dxy") is not None and b["target_dxy"] >= gate_m - edge_band_m)]
    rep = {"fast": _pcts(fast), "edge": _pcts(edge), "union": _pcts(union), "all": _pcts([b["A"] for b in sb])}
    rep["tau_noise"] = rep["union"].get("p95") if union else None
    return rep


def knee(points):
    """Index of the knee of points [(x, y)] (x = interventions per rollout, y = success): the point farthest from
    the chord between the lowest-x and highest-x points, after scaling both axes to [0, 1]."""
    P = np.asarray(points, float)
    if len(P) < 3:
        return int(np.argmax(P[:, 1])) if len(P) else None
    order = np.argsort(P[:, 0], kind="stable")
    Q = P[order]
    span = np.where(np.ptp(Q, axis=0) > 0, np.ptp(Q, axis=0), 1.0)
    Q = (Q - Q.min(axis=0)) / span
    a, b = Q[0], Q[-1]
    e, w = b - a, Q - a
    d = np.abs(e[0] * w[:, 1] - e[1] * w[:, 0]) / max(np.linalg.norm(e), 1e-9)   # 2-D cross product
    return int(order[int(np.argmax(d))])


def phase0_report(rows, h_steps, gamma, per_bin=100):
    """sec. 10.5. rows: probe log dicts with tilt_deg, t_noop, t_a (long-cap success times, None = none)."""
    bins = []
    for k, (lo, hi) in enumerate(TILT_BINS):
        rs = [r for r in rows if lo <= r["tilt_deg"] < hi]
        d = [label_at(h_steps, r["t_a"], gamma) - label_at(h_steps, r["t_noop"], gamma) for r in rs]
        m, s = _mean(d), _se(d)
        bins.append({"tilt": "{}-{}".format(lo, hi if hi < 180 else "+"), "n": len(rs), "diff": m, "se": s,
                     "z": m / s if s and np.isfinite(s) and s > 0 else float("nan"),
                     "y_noop": _mean([label_at(h_steps, r["t_noop"], gamma) for r in rs]),
                     "y_level": _mean([label_at(h_steps, r["t_a"], gamma) for r in rs])})
    rs = [r for r in rows if r["tilt_deg"] >= TILT_BINS[2][0]]
    d = [label_at(h_steps, r["t_a"], gamma) - label_at(h_steps, r["t_noop"], gamma) for r in rs]
    merged = {"tilt": "20+", "n": len(rs), "diff": _mean(d), "se": _se(d)}
    succ = [r["t_noop"] for r in rows if r.get("t_noop") is not None]
    t = _pcts(np.array(succ, float) / 20.0) if succ else {"n": 0}
    beyond = float(np.mean([s > h_steps for s in succ])) if succ else float("nan")
    flagged = [b["tilt"] for b in bins if b["n"] >= 10 and np.isfinite(b["z"]) and abs(b["z"]) > 2 and abs(b["diff"]) > 0.05]
    rec = ("no real difference at any tilt: go ahead with Q0" if not flagged else
           "levelling changes the outcome at tilt {}: train Q0 only below that tilt, or switch to the separate V "
           "network (sec. 15)".format(", ".join(flagged)))
    if t.get("n") and t.get("p90") is not None and t["p90"] > h_steps / 20.0:
        rec += "; the 90th percentile of no-op success times ({:.1f} s) exceeds H_max ({:.1f} s): consider raising " \
               "H_max".format(t["p90"], h_steps / 20.0)
    return {"bins": bins, "merged_20_plus": merged, "n_probes": len(rows), "per_bin_target": per_bin,
            "time_to_success_s": t, "frac_successes_beyond_h_max": beyond, "h_max_s": h_steps / 20.0,
            "gamma": gamma, "recommendation": rec}


def virtual_trigger(ticks, tau, n_persist, cooldown_steps):
    """Steps at which the trigger WOULD fire on a tick log of a rollout it did not act on (gated ticks only;
    a gap between gated ticks longer than one tick resets the streak, as a closed gate does)."""
    fires, streak, last, cool = [], 0, None, -10 ** 9
    for t in ticks:
        if last is not None and t["step"] - last > 4:
            streak = 0
        last = t["step"]
        if t["step"] < cool:
            streak = 0
            continue
        streak = streak + 1 if t["A"] > tau else 0
        if streak >= n_persist:
            fires.append(t["step"])
            streak, cool = 0, t["step"] + cooldown_steps
    return fires


def timing_report(results, tau, n_persist, cooldown_steps):
    u = [r for r in used(results)]
    ok = [r for r in u if r["pick_success"]]
    bad = [r for r in u if not r["pick_success"]]
    rows = []
    for r in u:
        f = virtual_trigger(r["ticks"], tau, n_persist, cooldown_steps)
        rows.append({"tag": r["tag"], "success": r["pick_success"], "fires": f, "end_step": r["steps"],
                     "first_fire_before_end_s": None if not f else (r["steps"] - f[0]) / 20.0,
                     "A_max": max([t["A"] for t in r["ticks"]] or [float("nan")])})
    fire_ok = [bool(x["fires"]) for x in rows if x["success"]]
    fire_bad = [bool(x["fires"]) for x in rows if not x["success"]]
    return {"tau": tau, "n_rollouts": len(u), "n_success": len(ok), "n_fail": len(bad),
            "false_trigger_rate": _mean([float(x) for x in fire_ok]),
            "fires_per_successful_rollout": _mean([len(x["fires"]) for x in rows if x["success"]]),
            "trigger_rate_on_failures": _mean([float(x) for x in fire_bad]),
            "first_fire_before_end_s_failures": _pcts([x["first_fire_before_end_s"] for x in rows
                                                       if not x["success"] and x["fires"]]),
            "rows": rows}
