"""Merge the per-episode survey.json files into survey_results.json.

Surveyors name new modes independently, so mode_map.json maps every surveyor mode_name onto the
unified taxonomy ID (HL-n / LL-n). Run after all batches are surveyed:
    python aggregate.py
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
plan = json.load(open(ROOT / "sampling_plan.json"))
taxonomy = json.load(open(ROOT / "mode_map.json"))
name_to_id = {name: mid for mid, m in taxonomy["modes"].items() for name in m["aliases"]}

episodes, missing, unmapped = [], [], set()
for p in plan["episodes"]:
    f = ROOT / "episodes" / p["suite"] / p["folder"] / "survey.json"
    if not f.exists():
        missing.append(f"{p['suite']}/{p['folder']}")
        continue
    s = json.load(open(f))
    for fl in s["failures"]:
        fl["surveyor_mode_id"], fl["surveyor_mode_name"] = fl.pop("mode_id"), fl.pop("mode_name")
        mid = taxonomy["overrides"].get(f'{s["episode"]}::{fl["surveyor_mode_name"]}',
                                        name_to_id.get(fl["surveyor_mode_name"]))
        if mid is None:
            unmapped.add(fl["surveyor_mode_name"])
        fl["mode_id"] = mid
        fl["mode_name"] = taxonomy["modes"][mid]["name"] if mid else None
        if fl["level"] == "low" and fl.get("rl_reset_frame"):
            fl["rl_reset_frame"] = f"episodes/{p['suite']}/{p['folder']}/{fl['rl_reset_frame']}"
    s.update(suite=p["suite"], folder=p["folder"], task_id=p["task_id"],
             init_state_idx=p["init_state_idx"], sampled_steps=len(p["sampled_steps"]))
    episodes.append(s)

if missing or unmapped:
    raise SystemExit(f"missing surveys: {missing}\nunmapped mode names: {sorted(unmapped)}")

primary = collections.Counter(fl["mode_id"] for e in episodes for fl in e["failures"] if fl["primary"])
allf = collections.Counter(fl["mode_id"] for e in episodes for fl in e["failures"])
stats = {
    "episodes_surveyed": len(episodes),
    "by_suite": dict(collections.Counter(e["suite"] for e in episodes)),
    "by_fps": dict(collections.Counter(e["survey_fps"] for e in episodes)),
    "primary_level": dict(collections.Counter(fl["level"] for e in episodes for fl in e["failures"] if fl["primary"])),
    "primary_mode_counts": dict(primary.most_common()),
    "all_failure_mode_counts": dict(allf.most_common()),
}
ll_instances = [{"episode": e["episode"], "mode_id": fl["mode_id"], "subtask": fl["rl_subtask"],
                 "onset_step": fl["onset_step"], "onset_window": fl["onset_window"],
                 "rl_reset_step": fl["rl_reset_step"], "rl_reset_frame": fl["rl_reset_frame"],
                 "rl_success_check": fl["rl_success_check"], "step_resolution": e["step_resolution"],
                 "confidence": fl["confidence"],
                 # LL failures seen while pi0 handled a non-instructed object: evidence for the mode, not a task reset
                 "on_instructed_subtask": "[no" not in fl["subtask"]}
                for e in episodes for fl in e["failures"] if fl["level"] == "low"]

json.dump({
    "survey": "full failure survey: libero_spatial + libero_10 failures, libero_90 failures of a random 50% task split",
    "inputs_used": "rollout video frames only (sampled at 1 or 3 fps); trace/ never opened",
    "step_convention": "video is 20 fps, one frame per control step incl. 10 wait steps: step = round(20 * t_video)",
    "sampling": {"seed": plan["random_seed"], "three_fps_fraction": 0.2,
                 "three_fps_episodes": plan["three_fps_episodes"], "libero90_split": "libero90_split.json",
                 "note": "libero_spatial had 0 failed rollouts (20/20 success), so it contributes no episodes"},
    "taxonomy": taxonomy["modes"],
    "new_primitives": taxonomy["new_primitives"],
    "stats": stats,
    "ll_rl_reset_index": ll_instances,
    "episodes": episodes,
}, open(ROOT / "survey_results.json", "w"), indent=2)
print(json.dumps(stats, indent=1))
