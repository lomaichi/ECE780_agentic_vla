"""Summarise a run directory: detection vs. the survey's video labels
(observe mode), recovery outcomes (agent mode), and library statistics.

  python3 tools/summarize_run.py logs/survey90
"""
import argparse
import collections
import glob
import json
import os

# Video-survey labels per failed task (Documentation/failure_mode_sumary.txt),
# mapped onto the agent's codes: (primary, all labels incl. secondary).
# Survey-only names map as: subtask hand-off / partial compound -> R7;
# phantom target, dithering, grasp failure, target never approached -> R5;
# destination ignored, drop in transit -> R6; no-release adjust loop -> R4;
# wrist wind-up -> S4.
_R1, _R2, _R3, _R4, _R5, _R6, _R7 = ("R1_wrong_object", "R2_scene_habit", "R3_stopped_short", "R4_placement",
                                     "R5_target_not_reached", "R6_destination_missed", "R7_partial_task")
_S1, _S2, _S3, _S4 = "S1_stall", "S2_disturbance", "S3_out_of_view", "S4_wrist_windup"
SURVEY_LABELS = {
    0: [_R3], 1: [_R7, _R2], 2: [_R2], 3: [_R2], 4: [_R2], 5: [_R2, _R3], 6: [_R1, _R2], 7: [_R2, _S3],
    8: [_R2], 9: [_R2, _R4, _R6], 11: [_R1, _R2], 12: [_R1, _R2], 13: [_R1, _R2],
    14: [_R1, _S2], 15: [_R1], 16: [_R1, _S2], 17: [_R2, _R4, _R6], 18: [_R2, _R5], 20: [_R3, _R5],
    21: [_R1, _R5, _R2, _R7], 22: [_R2], 23: [_R2, _R3, _R7], 25: [_R2, _R6], 26: [_R1, _R2], 27: [_R5],
    28: [_R2, _R3],
    29: [_R4, _R1, _R5], 30: [_R2], 31: [_R2], 32: [_R5, _R1, _R6], 34: [_R2, _R6], 35: [_R3, _S4],
    36: [_R5, _R2, _S4], 37: [_R5, _R1], 39: [_R2, _S4, _S2, _S3], 40: [_R1], 41: [_R1], 42: [_R1, _R2],
    43: [_R6, _S4],
    44: [_R3], 45: [_R3], 46: [_R1], 48: [_R1, _R6], 49: [_R1], 50: [_R1], 51: [_R1], 52: [_R1],
    53: [_R5, _R1], 54: [_R1, _R4], 55: [_R1, _R4], 56: [_R1], 58: [_R1],
    59: [_R1, _R5], 61: [_R1, _R2], 62: [_R1, _R2], 63: [_R7, _R2], 64: [_R7, _R2], 65: [_R2, _R1],
    66: [_R2, _R1], 69: [_R2, _R4], 71: [_R2, _R1], 72: [_R4, _R2], 73: [_R2, _R4], 74: [_R2, _R4],
    75: [_R2, _R4],
    76: [_R1, _R2], 77: [_R1, _S2], 78: [_R4], 80: [_R1], 81: [_R5, _R2], 82: [_R5, _R2], 83: [_R1],
    84: [_R1, _R5], 85: [_R4, _R2], 86: [_R1, _R6], 87: [_R1, _R6], 88: [_R6], 89: [_R6],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()

    traces = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.run_dir, "*", "trace.json")))]
    by = collections.defaultdict(dict)
    for t in traces:
        by[(t["task_id"], t["init_idx"])][t["mode"]] = t

    print("%-5s %-4s | %-8s %-20s %-6s | %-8s %-6s %s" % (
        "task", "init", "native", "observe diagnosis", "match", "agent", "t", "agent diagnosis chain"))
    match = match_primary = total = 0
    nat_ok = ag_ok = ag_native_budget = n = 0
    for (tid, idx), m in sorted(by.items()):
        ob, ag = m.get("observe"), m.get("agent")
        od, mt = "-", ""
        if ob:
            if ob["diagnoses"]:
                od = ob["diagnoses"][0]["code"]
                total += 1
                labels = SURVEY_LABELS.get(tid, [])
                hit = od in labels
                match += hit
                match_primary += bool(labels) and od == labels[0]
                mt = "yes" if hit else "no"
            nat_ok += ob["success"]
        chain = " > ".join(d["code"].split("_")[0] for d in ag["diagnoses"]) if ag else "-"
        if ag:
            ag_ok += ag["success"]
            ag_native_budget += ag["native_budget_success"]
        n += 1
        print("%-5d %-4d | %-8s %-20s %-6s | %-8s %-6s %s" % (
            tid, idx, ob["success"] if ob else "-", od, mt, ag["success"] if ag else "-",
            (ag["success_t"] or "-") if ag else "-", chain))

    print(f"\nnative pi0 (observe): {nat_ok}/{n} succeeded")
    n_ag = sum(1 for m in by.values() if m.get("agent"))
    if n_ag:
        print(f"agent:                {ag_ok}/{n_ag} succeeded ({ag_native_budget} within the native 400-step budget)")
    if total:
        print(f"observe diagnosis matches a survey label on {match}/{total} failed episodes "
              f"(the primary label on {match_primary}); survey labels are from one init-0 video per task, "
              "and pi0 is stochastic, so other runs may fail differently")
        # first observe diagnosis per failed episode
        dist = collections.Counter(m["observe"]["diagnoses"][0]["code"] for m in by.values()
                                   if m.get("observe") and m["observe"]["diagnoses"])
        print("observe diagnoses: " + ", ".join(f"{k} {v}" for k, v in sorted(dist.items())))

    # library statistics from this run's agent episodes
    stats = collections.defaultdict(lambda: [0, 0])
    for t in traces:
        if t["mode"] != "agent":
            continue
        for d in t["diagnoses"]:
            stats[d["code"]][0] += 1
            stats[d["code"]][1] += d.get("outcome") == "recovered"
    print("\nlibrary entry          used  recovered")
    for code, (u, r) in sorted(stats.items()):
        print("%-22s %4d  %4d" % (code, u, r))


if __name__ == "__main__":
    main()
