"""Table of recovery-agent runs (live_judge.py --judge recovery --out X.json): per episode the ground-truth
subtask steps, the judge's success steps, the failures it reported and the corrections it made (with the
ground truth when each correction ended).

  python3 summarize_recovery.py video/recovery/*.json
"""

import json
import sys


def first(gt, prefix):
    for c in gt:
        if c["spec"].startswith(prefix):
            return c["done_step"]
    return None


def judged(subtasks, kind):
    s = [x for x in subtasks if x["kind"] == kind]
    if not s:
        return None
    s = s[0]
    steps = [h["done_step"] for h in s.get("history", [])] + ([s["done_step"]] if s["done_step"] is not None else [])
    return ",".join(map(str, steps)) or None


def main(paths):
    print("| task | instruction | pick GT / judge | place GT / judge | LIBERO success | stop | failures (onset step) | "
          "corrections (start-end, GT at end) |")
    print("|---|---|---|---|---|---|---|---|")
    for path in paths:
        r = json.load(open(path))
        gt = r["ground_truth"]
        fails = ", ".join(f"{f['mode']} [{f['library']}] @{f['step']}" + (f" -> {f['action']}" if f["action"] else "")
                          for f in r.get("failures", [])) or "-"
        corr = []
        for c in r.get("corrections", []):
            end = c.get("gt_end")
            held = "" if end is None else " (" + ", ".join(k.split("(")[0] for k, v in end.items() if v) + ")"
            corr.append(f"{c['action']} @{c['step']}-{c.get('end_step', '?')}" + (held if held != " ()" else " (none)"))
        print(f"| {r['task_id']} init {r['init_state_idx']} | {r['task_language']} | "
              f"{first(gt, 'pick') or '-'} / {judged(r['subtasks'], 'pick') or '-'} | "
              f"{first(gt, 'place') or '-'} / {judged(r['subtasks'], 'place') or '-'} | "
              f"{'yes @' + str(first(gt, 'goal')) if first(gt, 'goal') else 'no'} | {r['stop_reason'] or 'max steps'} | "
              f"{fails} | {'; '.join(corr) or '-'} |")


if __name__ == "__main__":
    main(sys.argv[1:])
