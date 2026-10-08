# Failure agent: prototyping log (v1, all 90 libero_90 tasks)

Development log for `failure_agent_prototype/` (ECE780), 2026-10-08. This
redoes the earlier 19-rollout prototype on all 90 libero_90 tasks: native
rollouts, video survey, generalising the agent, development runs, and the
final evaluation. The old prototype's logs were deleted; its findings are
summarised where they matter. Background on the failure modes:
`Documentation/failure_mode_sumary.txt`.

## 1. Setup and native rollouts

- Policy server started with the prebuilt venv as before (`uv run --offline`
  still fails without network):
  `cd /opt/openpi && OPENPI_DATA_HOME=/home/DockerShared/.cache/openpi .venv/bin/python scripts/serve_policy.py policy:checkpoint --policy.config=pi0_libero --policy.dir=gs://openpi-assets/checkpoints/pi0_libero`.
- Wrote `tools/collect_native.py`: the loop of openpi's
  `examples/libero/main.py` (seed 7, 10 settle steps, replan every 5 steps,
  400-step cap, init state 0), with the task id in the video name (tasks 75
  and 80, 78 and 81 etc. share an instruction) and a `results.json`.
- Deleted the old 19 rollouts and collected all 90 tasks in 6 parallel
  shards (one websocket client each; GPU at ~91%): about 25 minutes.
- Result: **12/90 successes** (10, 19, 24, 33, 38, 47, 57, 60, 67, 68, 70,
  79). All 5 successes of the old sample (24, 60, 68, 70, 79) succeeded again.
  78 failures.

## 2. Video survey

- `tools/video_motion_stats.py` on all 90 videos (`logs/native_motion_stats.txt`)
  and `tools/contact_sheet.py` (24-frame timestamped sheets).
- Six reviewer subagents labelled 13 failures each from the sheets and
  enlarged frames, against R1-R4/S1-S3, free to propose new labels. Merged
  into `Documentation/native_pi0_libero90_survey_labels.md`; the summary
  `Documentation/failure_mode_sumary.txt` was rewritten.
- New modes the reviewers proposed, merged into the taxonomy as
  **R5 target not reached** (phantom target, dithering, grasp failure, target
  in a raised spot), **R6 destination missed** (carried away, dropped in
  transit), **R7 partial task** (one sub-goal done, next never started) and
  **S4 wrist wind-up**.
- Main finding: pi0's grasps are usually clean; it binds the instruction to
  the wrong target or destination, often running the libero_10 task of the
  same scene. libero_10 shares 9 of the 20 libero_90 scenes; native success
  is 8/35 there vs 4/55 elsewhere.
- Process note: the first reviewer was launched with unfilled `{TASKS}` /
  `{GROUP}` placeholders and stopped without labelling; it was resumed with
  the values.

## 3. Generalising the agent to 90 tasks

`tests/smoke_scene.py` now runs on all 90 tasks by default (scene parsing,
prompt slots, decomposition, camera projection, `retract_home`).

| Issue | Cause | Fix |
|---|---|---|
| Phrases for new destinations ("the kitchen table", "in the wooden two layer shelf") | name/region maps covered only the 14 old tasks | added ketchup, milk, wine bottle/rack, white bowl, shelf; table regions relative to an object ("to the right of the plate", "to the front of the white mug"); shelf regions ("on/under the cabinet shelf"); a single `{place}` slot replaces `{rel} {dest}` |
| Spatial descriptors mirrored: "black bowl on the right" for task 60's "on the left", same for 38, 65-68, 87-89 | descriptors used left/right in the image pi0 sees, but the camera faces the robot and LIBERO says left/right from the robot's side | descriptors from world coordinates (left = -y, front = +x); "in the middle" for three look-alikes; peers grouped by noun phrase (black + yellow book). All 20 ambiguous tasks now reproduce LIBERO's own wording |
| Task 8 decomposition: "open the top drawer" twice | open-goal and destination-drawer pre-step are the same drawer | de-duplicate |
| `TypeError: startswith ... NoneType` in R4 diagnosis, crashing 3 workers of the first observe pass (tasks 34, 85 lost) | table regions (`kitchen_table_plate_right_region`) have no parent object | `_owner_of` falls back to the name; per-episode exceptions are now logged and the worker continues |
| Final evaluation would take most of a day sequentially | one client | `run_agent.py --workers N`: processes pull tasks from a queue; console output is printed one finished episode at a time; per-worker library stats are merged |

## 4. Sim-ground-truth pass (`logs/dev_observe_all`)

Observe mode on all 78 failures at init 0 with the v0 diagnosis rules (run
while the survey was being done). Natively 2/78 succeeded this time. The
v0 rules gave: R1 27, R3 10, R2 6, R4 4, and **29 symptom-only fallbacks**
(S1 20, S2 8, S3 1). Listing the fallbacks with closest EEF approach and
grasp history showed the same modes the reviewers proposed:

| Pattern in the fallbacks | Tasks | Became |
|---|---|---|
| no grasp, EEF never within ~10 cm of the target | 2, 48, 54, 59, 80, 81, 82, 84, 36, 37 | R5 |
| no articulation progress, EEF never near the fixture | 7, 11, 20, 39 | R5 (fixture) |
| target grasped, never near the destination | 9, 25, 34, 88, 89 | R6 |
| one predicate achieved, the other pending | 63 | R7 |

## 5. Development runs with pi0

All agent mode, init 0 unless noted; `logs/dev_v*_agent`.

| Run | Change being tested | Outcome |
|---|---|---|
| `dev_v1_agent` (15 tasks) | R5/R6/R7/S4 diagnoses; `no_progress` and `wrist_windup` triggers; recipe "approach → pi0 until the target is held → `carry_to`" for R1/R4/R5/R6 and inside `decompose`; approach to handles/knobs | 4/15. Stacking (63): pi0 grabbed the other bowl, which is also a goal target, so no trigger. Task 65: the carry crossed the white mug and stopped 5 cm short at the workspace edge. Shelf (88): the region is *inside* the upper shelf layer, so a drop from above lands on top |
| `dev_v2_agent` (15) | carry at home height and in the home orientation (all drop points reachable in a policy-free test; pi0's twisted grasp orientation was not); wrong grasp judged against pending targets and the object a recovery grasp step asks for (`focus`) | 7/15 (0, 4, 35, 43, 46, 63, 65). Task 9: `retract_home` stuck 0.25 m from home every time |
| debug (snapshot replay) | task 9 failure snapshot restored, retract traced | elbow (joint 4) at its -0.07 rad limit: pi0 left the arm stretched; Cartesian OSC has no radial authority there. Also found `_check_grasp` drops a bowl held by its rim (pi0 carried it to the plate while the monitor said "released") |
| `dev_v3_agent` (12 R1-heavy tasks) | approach hover 15 cm → 7 cm, so the target fills the wrist view | 3/12, but 59 and 62 (salad dressing, never recovered before) and 53 now succeed. pi0 grasps what is directly below the gripper |
| `dev_v4_agent` (6 tasks × inits 0-1) | retract: lift 10 cm, move over home, then rise; fallback to a joint-space move (swap in robosuite `JOINT_POSITION`, restore OSC after) | 4/12; joint fallback triggered but default gains (kp 50, 0.05 rad/step) left it 0.8 rad short |
| debug | joint-controller gains on the task 9 snapshot | kp 150, 0.1 rad/step: home within 0.008 rad in 93 steps, OSC resumes normally |
| `dev_v5_agent` (9, 84 × inits 0-1) | tuned joint fallback | joint_home residual 0.013-0.019 rad, but 0/4: pi0 "never grasps" the bowl |
| `dev_v6_agent` (9, 84, 17 × inits 0-1) | grasp also confirmed when the object is carried ≥ 5 cm up, within 10 cm of a closed gripper, for 5 steps; grasp kept while lifted next to a closed gripper | 3/6: task 9 2/2 (the bowl had been grasped by the rim all along), 17 1/2; 84 (red mug) 0/2 |

Process notes:
- A merge command deleted the rerun traces of tasks 34 and 85 after a failed
  `mv` (the `rm` was chained with `;`); they were rerun.
- `pgrep -f "[r]un_agent.py --tasks survey"` (bracket trick) to wait on the
  final run without matching the waiting shell itself.

## 6. Final evaluation (`logs/survey90`)

All 78 native failures × init states 0–2 × {observe, agent}, 6 workers,
`logs/library_stats.json` reset first. Command:
`./run_agent.sh --tasks survey --init 0 1 2 --mode both --workers 6 --out logs/survey90`
(about 35 minutes; 468 episodes, no errors).

Full console output (one block per episode, warnings and EGL teardown
tracebacks removed): `logs/survey90.log`. Per-scene and credit breakdown:
`logs/survey90_breakdown.txt`.

### Headline

```
native pi0 (observe): 5/234 succeeded
agent:                102/234 succeeded (77 within the native 400-step budget)
tasks solved at least once by the agent: 42/78 (3/3: 27, 2/3: 6, 1/3: 9)
old 14 survey tasks: native 1/42, agent 22/42 (previous prototype: native 1/42, agent 8/42)
successful agent episodes by credited entry (last diagnosis):
   35  R5 [plan has carry_to]        32  R1 [carry_to]
   11  R3 [finish_articulation]       6  no recovery needed (pi0 alone)
    5  R6 [carry_to]                  4  R2 [carry_to]
    6  R7 [carry_to] (3 also with the drawer push)
    3  R4 [carry_to]
```

### Library statistics (`logs/survey90/library_stats.json`)

```json
{
  "R3_stopped_short": {
    "attempts": 86,
    "recovered": 11
  },
  "R5_target_not_reached": {
    "attempts": 185,
    "recovered": 35
  },
  "R2_scene_habit": {
    "attempts": 28,
    "recovered": 4
  },
  "R1_wrong_object": {
    "attempts": 71,
    "recovered": 32
  },
  "R4_placement": {
    "attempts": 86,
    "recovered": 3
  },
  "R6_destination_missed": {
    "attempts": 48,
    "recovered": 5
  },
  "R7_partial_task": {
    "attempts": 20,
    "recovered": 6
  }
}
```

### Summary (`tools/summarize_run.py logs/survey90`)

```
task  init | native   observe diagnosis    match  | agent    t      agent diagnosis chain
0     0    | False    R3_stopped_short     yes    | True     338    R3
0     1    | False    R3_stopped_short     yes    | True     186    R3
0     2    | False    R3_stopped_short     yes    | True     355    R3
1     0    | False    R3_stopped_short     no     | True     594    R3 > R7
1     1    | False    R3_stopped_short     no     | True     634    R3 > R7
1     2    | False    R3_stopped_short     no     | True     505    R3 > R7
2     0    | False    R5_target_not_reached no     | False    -      R5 > R4 > R4
2     1    | False    R5_target_not_reached no     | False    -      R5 > R5 > R4
2     2    | False    R5_target_not_reached no     | False    -      R5 > R4 > R4
3     0    | False    R2_scene_habit       yes    | True     554    R2 > R3 > R3
3     1    | False    R2_scene_habit       yes    | True     451    R2 > R3
3     2    | False    R2_scene_habit       yes    | True     522    R2 > R3 > R3
4     0    | False    R2_scene_habit       yes    | False    -      R3 > R2 > R3
4     1    | False    R2_scene_habit       yes    | True     430    R2 > R3
4     2    | False    R2_scene_habit       yes    | True     487    R2 > R3
5     0    | False    R2_scene_habit       yes    | True     658    R2 > R3 > R3
5     1    | False    R2_scene_habit       yes    | True     587    R2 > R3 > R3
5     2    | False    R2_scene_habit       yes    | False    -      R2 > R3 > R2
6     0    | False    R5_target_not_reached no     | False    -      R5 > R5 > R5
6     1    | False    R5_target_not_reached no     | False    -      R2 > R5 > R5
6     2    | False    R2_scene_habit       yes    | False    -      R2 > R5 > R5
7     0    | False    R3_stopped_short     no     | False    -      R5 > R3 > R3
7     1    | False    R3_stopped_short     no     | False    -      R5 > R3 > R3
7     2    | False    R5_target_not_reached no     | False    -      R5 > R3 > R3
8     0    | False    R6_destination_missed no     | False    -      R6 > R3
8     1    | False    R3_stopped_short     no     | False    -      R6 > R6 > R4
8     2    | False    R4_placement         no     | False    -      R6 > R6 > R6
9     0    | False    R6_destination_missed yes    | False    -      R6 > R4
9     1    | False    R6_destination_missed yes    | True     453    R6
9     2    | False    R6_destination_missed yes    | False    -      R6 > R6 > R6
11    0    | False    R5_target_not_reached no     | False    -      R5 > R3 > R3
11    1    | True     -                           | False    -      R3 > R3
11    2    | False    R3_stopped_short     no     | False    -      R5 > R3 > R3
12    0    | False    R1_wrong_object      yes    | True     271    R1
12    1    | False    R1_wrong_object      yes    | True     246    R1
12    2    | False    R1_wrong_object      yes    | True     235    R1
13    0    | False    R1_wrong_object      yes    | False    -      R1 > R5 > R5
13    1    | False    R1_wrong_object      yes    | False    -      R1 > R1 > R1
13    2    | False    R5_target_not_reached no     | False    -      R5 > R5 > R5
14    0    | False    R1_wrong_object      yes    | True     302    R4
14    1    | False    R1_wrong_object      yes    | True     375    R5
14    2    | False    R4_placement         no     | True     836    R5 > R5 > R5
15    0    | False    R1_wrong_object      yes    | True     338    R5
15    1    | False    R1_wrong_object      yes    | True     300    R5
15    2    | False    R1_wrong_object      yes    | True     337    R1
16    0    | False    R6_destination_missed no     | False    -      R1 > R5 > R5
16    1    | False    R1_wrong_object      yes    | False    -      R5 > R5 > R5
16    2    | False    R6_destination_missed no     | False    -      R5 > R5 > R5
17    0    | False    R5_target_not_reached no     | True     506    R5 > R5
17    1    | False    R4_placement         yes    | True     187    R6
17    2    | False    R6_destination_missed yes    | True     294    R6
18    0    | False    R2_scene_habit       yes    | False    -      R2 > R5 > R5
18    1    | False    R2_scene_habit       yes    | False    -      R5 > R5 > R5
18    2    | False    R2_scene_habit       yes    | False    -      R2 > R5 > R5
20    0    | False    R5_target_not_reached yes    | False    -      R5 > R3 > R3
20    1    | False    R3_stopped_short     yes    | True     788    R3 > R3 > R3
20    2    | False    R5_target_not_reached yes    | False    -      R3 > R3 > R3
21    0    | False    R7_partial_task      yes    | False    -      R7 > R7 > R7
21    1    | False    R1_wrong_object      yes    | False    -      R1 > R7 > R7
21    2    | False    R1_wrong_object      yes    | False    -      R7 > R7 > R7
22    0    | False    R3_stopped_short     no     | False    -      R3 > R3 > R3
22    1    | False    R3_stopped_short     no     | False    -      R5 > R3 > R3
22    2    | False    R3_stopped_short     no     | False    -      R3 > R3 > R3
23    0    | False    R7_partial_task      yes    | False    -      R7 > R3 > R3
23    1    | False    R7_partial_task      yes    | False    -      R7 > R3 > R3
23    2    | False    R7_partial_task      yes    | False    -      R3 > R3 > R3
25    0    | False    R6_destination_missed yes    | True     302    R6
25    1    | False    R6_destination_missed yes    | True     273    R2
25    2    | False    R2_scene_habit       yes    | False    -      R2 > R2 > R6
26    0    | False    R1_wrong_object      yes    | True     283    R1
26    1    | False    R1_wrong_object      yes    | True     278    R1
26    2    | False    R1_wrong_object      yes    | True     426    R1 > R5
27    0    | False    R5_target_not_reached yes    | False    -      R5 > R5 > R5
27    1    | False    R6_destination_missed no     | False    -      R6 > R4 > R4
27    2    | False    R5_target_not_reached yes    | False    -      R5 > R5 > R5
28    0    | False    R3_stopped_short     yes    | False    -      R5 > R3 > R3
28    1    | True     -                           | False    -      R5 > R3 > R3
28    2    | False    R3_stopped_short     yes    | False    -      R5 > R3 > R3
29    0    | False    R5_target_not_reached yes    | False    -      R5 > R5 > R5
29    1    | False    R4_placement         yes    | False    -      R6 > R4 > R4
29    2    | False    R6_destination_missed no     | False    -      R5 > R4 > R4
30    0    | False    R2_scene_habit       yes    | True     473    R5 > R2
30    1    | False    R2_scene_habit       yes    | True     617    R5 > R2 > R2
30    2    | False    R2_scene_habit       yes    | True     425    R5 > R2
31    0    | False    R2_scene_habit       yes    | False    -      R5 > R5 > R2
31    1    | False    R2_scene_habit       yes    | False    -      R5 > R5 > R2
31    2    | False    R2_scene_habit       yes    | False    -      R5 > R2 > R5
32    0    | False    R1_wrong_object      yes    | False    -      R5 > R4 > R4
32    1    | False    R5_target_not_reached yes    | False    -      R5 > R4 > R1
32    2    | False    R5_target_not_reached yes    | True     374    R1 > R4
34    0    | False    R2_scene_habit       yes    | False    -      R2 > R6 > R6
34    1    | False    R6_destination_missed yes    | False    -      R2 > R6 > R6
34    2    | False    R6_destination_missed yes    | True     350    R6
35    0    | False    R3_stopped_short     yes    | False    -      R3 > R3 > R3
35    1    | False    R3_stopped_short     yes    | False    -      R3 > R3 > R3
35    2    | False    R3_stopped_short     yes    | False    -      R5 > R3 > R3
36    0    | False    R5_target_not_reached yes    | False    -      R5 > R5 > R5
36    1    | False    R5_target_not_reached yes    | False    -      R5 > R5 > R2
36    2    | False    R5_target_not_reached yes    | False    -      R5 > R5 > R5
37    0    | False    R5_target_not_reached yes    | False    -      R5 > R5 > R5
37    1    | False    R5_target_not_reached yes    | False    -      R5 > R5 > R5
37    2    | False    R5_target_not_reached yes    | False    -      R5 > R5 > R2
39    0    | False    R5_target_not_reached no     | False    -      R5 > R3 > R3
39    1    | False    R5_target_not_reached no     | False    -      R5 > R3 > R3
39    2    | False    R5_target_not_reached no     | False    -      R5 > R3 > R3
40    0    | False    R1_wrong_object      yes    | False    -      R1 > R1 > R5
40    1    | False    R1_wrong_object      yes    | False    -      R5 > R5 > R5
40    2    | False    R1_wrong_object      yes    | False    -      R1 > R5 > R5
41    0    | False    R1_wrong_object      yes    | False    -      R1 > R5 > R1
41    1    | False    R1_wrong_object      yes    | False    -      R5 > R5 > R5
41    2    | False    R1_wrong_object      yes    | False    -      R1 > R5 > R1
42    0    | False    R1_wrong_object      yes    | False    -      R5 > R5 > R5
42    1    | False    R6_destination_missed no     | False    -      R5 > R5 > R1
42    2    | False    R1_wrong_object      yes    | False    -      R1 > R5 > R5
43    0    | False    R6_destination_missed yes    | True     121    
43    1    | False    R4_placement         no     | True     132    
43    2    | False    R6_destination_missed yes    | True     134    
44    0    | False    R5_target_not_reached no     | False    -      R5 > R3 > R3
44    1    | False    R3_stopped_short     yes    | False    -      R5 > R3 > R3
44    2    | False    R3_stopped_short     yes    | False    -      R3 > R3 > R3
45    0    | False    R3_stopped_short     yes    | False    -      R7 > R1 > R1
45    1    | False    R1_wrong_object      no     | False    -      R3 > R3 > R3
45    2    | False    R7_partial_task      no     | False    -      R7 > R7 > R1
46    0    | False    R5_target_not_reached no     | True     234    R5
46    1    | False    R1_wrong_object      yes    | True     291    R5
46    2    | True     -                           | True     297    R5
48    0    | False    R5_target_not_reached no     | False    -      R5 > R5 > R5
48    1    | False    R5_target_not_reached no     | False    -      R5 > R5 > R5
48    2    | False    R5_target_not_reached no     | False    -      R5 > R5 > R5
49    0    | False    R1_wrong_object      yes    | True     205    R1
49    1    | False    R1_wrong_object      yes    | True     203    R1
49    2    | False    R1_wrong_object      yes    | True     254    R1
50    0    | False    R5_target_not_reached no     | True     222    R5
50    1    | False    R1_wrong_object      yes    | True     206    R1
50    2    | False    R5_target_not_reached no     | True     208    R5
51    0    | False    R1_wrong_object      yes    | True     214    R1
51    1    | True     -                           | True     221    R5
51    2    | False    R1_wrong_object      yes    | True     210    R1
52    0    | False    R5_target_not_reached no     | False    -      R5 > R4 > R4
52    1    | False    R1_wrong_object      yes    | True     247    R5
52    2    | False    R1_wrong_object      yes    | True     279    R1
53    0    | False    R5_target_not_reached yes    | True     263    R5
53    1    | False    R5_target_not_reached yes    | True     232    R5
53    2    | False    R5_target_not_reached yes    | True     123    R5
54    0    | False    R1_wrong_object      yes    | True     90     R5
54    1    | False    R5_target_not_reached no     | True     273    R5
54    2    | False    R5_target_not_reached no     | True     241    R5
55    0    | False    R1_wrong_object      yes    | True     219    R1
55    1    | False    R1_wrong_object      yes    | True     266    R1
55    2    | False    R1_wrong_object      yes    | True     264    R5
56    0    | False    R1_wrong_object      yes    | True     272    R1
56    1    | False    R1_wrong_object      yes    | True     244    R1
56    2    | False    R1_wrong_object      yes    | True     265    R1
58    0    | False    R1_wrong_object      yes    | False    -      R1 > R5 > R5
58    1    | False    R1_wrong_object      yes    | True     399    R5 > R5
58    2    | False    R1_wrong_object      yes    | False    -      R5 > R5 > R5
59    0    | False    R5_target_not_reached yes    | True     233    R5
59    1    | False    R1_wrong_object      yes    | True     280    R5
59    2    | False    R5_target_not_reached yes    | True     244    R5
61    0    | False    R5_target_not_reached no     | True     332    R5
61    1    | False    R1_wrong_object      yes    | True     488    R5 > R5
61    2    | False    R1_wrong_object      yes    | True     269    R5
62    0    | False    R1_wrong_object      yes    | True     619    R1 > R5 > R5
62    1    | False    R1_wrong_object      yes    | True     372    R1 > R5
62    2    | False    R1_wrong_object      yes    | True     291    R1
63    0    | False    R6_destination_missed no     | True     415    R7
63    1    | False    R6_destination_missed no     | False    -      R6 > R6 > R6
63    2    | False    R6_destination_missed no     | False    -      R6 > R6 > R6
64    0    | False    R7_partial_task      yes    | True     507    R7 > R7
64    1    | False    R7_partial_task      yes    | True     277    R4
64    2    | False    R7_partial_task      yes    | True     313    R7
65    0    | False    R1_wrong_object      yes    | True     205    R1
65    1    | False    R1_wrong_object      yes    | True     759    R1 > R5 > R5
65    2    | False    R1_wrong_object      yes    | False    -      R1 > R5 > R4
66    0    | False    R1_wrong_object      yes    | False    -      R1 > R5 > R5
66    1    | False    R1_wrong_object      yes    | True     430    R1 > R5
66    2    | False    R1_wrong_object      yes    | False    -      R1 > R6 > R6
69    0    | False    R1_wrong_object      no     | True     203    R1
69    1    | False    R1_wrong_object      no     | True     194    R1
69    2    | False    R5_target_not_reached no     | True     224    R5
71    0    | False    R1_wrong_object      yes    | False    -      R1 > R5 > R5
71    1    | False    R1_wrong_object      yes    | False    -      R1 > R5 > R5
71    2    | False    R1_wrong_object      yes    | True     224    R1
72    0    | False    R4_placement         yes    | True     130    
72    1    | False    R1_wrong_object      no     | True     127    
72    2    | True     -                           | True     167    
73    0    | False    R4_placement         yes    | False    -      R4 > R4 > R4
73    1    | False    R4_placement         yes    | False    -      R4 > R4 > R4
73    2    | False    R4_placement         yes    | False    -      R4 > R4 > R4
74    0    | False    R4_placement         yes    | False    -      R4 > R4 > R6
74    1    | False    R4_placement         yes    | False    -      R4 > R4 > R4
74    2    | False    R6_destination_missed no     | False    -      R4 > R4 > R4
75    0    | False    R6_destination_missed no     | False    -      R4 > R4 > R6
75    1    | False    R4_placement         yes    | False    -      R4 > R4 > R4
75    2    | False    R6_destination_missed no     | False    -      R4 > R6 > R4
76    0    | False    R1_wrong_object      yes    | True     241    R1
76    1    | False    R1_wrong_object      yes    | True     234    R1
76    2    | False    R1_wrong_object      yes    | True     229    R1
77    0    | False    R1_wrong_object      yes    | True     322    R1
77    1    | False    R5_target_not_reached no     | True     289    R1
77    2    | False    R4_placement         no     | False    -      R4 > R4 > R4
78    0    | False    R6_destination_missed no     | False    -      R5 > R4 > R6
78    1    | False    R5_target_not_reached no     | False    -      R5 > R4 > R4
78    2    | False    R5_target_not_reached no     | False    -      R5 > R4 > R4
80    0    | False    R5_target_not_reached no     | False    -      R1 > R6 > R6
80    1    | False    R6_destination_missed no     | False    -      R6 > R4 > R6
80    2    | False    R4_placement         no     | False    -      R5 > R4 > R4
81    0    | False    R5_target_not_reached yes    | False    -      R5 > R4 > R4
81    1    | False    R5_target_not_reached yes    | False    -      R5 > R4 > R4
81    2    | False    R5_target_not_reached yes    | False    -      R5 > R4 > R4
82    0    | False    R5_target_not_reached yes    | True     659    R5 > R5 > R5
82    1    | False    R5_target_not_reached yes    | True     343    R5
82    2    | False    R5_target_not_reached yes    | True     337    R5
83    0    | False    R1_wrong_object      yes    | False    -      R1 > R4 > R6
83    1    | False    R5_target_not_reached no     | False    -      R1 > R4 > R4
83    2    | False    R1_wrong_object      yes    | True     261    R1
84    0    | False    R5_target_not_reached yes    | False    -      R5 > R5 > R5
84    1    | False    R5_target_not_reached yes    | False    -      R1 > R5 > R5
84    2    | False    R1_wrong_object      yes    | False    -      R1 > R5 > R5
85    0    | False    R6_destination_missed no     | False    -      R5 > R4 > R4
85    1    | False    R6_destination_missed no     | False    -      R5 > R5 > R6
85    2    | False    R5_target_not_reached no     | False    -      R6 > R6
86    0    | False    R1_wrong_object      yes    | False    -      R1 > R4 > R4
86    1    | False    R1_wrong_object      yes    | False    -      R1 > R4 > R6
86    2    | False    R1_wrong_object      yes    | False    -      R1 > R4 > R4
87    0    | False    R1_wrong_object      yes    | True     258    R1
87    1    | False    R1_wrong_object      yes    | True     251    R1
87    2    | False    R1_wrong_object      yes    | True     273    R1
88    0    | False    R6_destination_missed yes    | False    -      R6 > R4 > R4
88    1    | False    R6_destination_missed yes    | False    -      R6 > R4 > R4
88    2    | False    R6_destination_missed yes    | False    -      R6 > R4 > R4
89    0    | False    R6_destination_missed yes    | False    -      R6 > R4 > R4
89    1    | False    R4_placement         no     | False    -      R4 > R4 > R4
89    2    | False    R6_destination_missed yes    | False    -      R6 > R4 > R4

native pi0 (observe): 5/234 succeeded
agent:                102/234 succeeded (77 within the native 400-step budget)
observe diagnosis matches a survey label on 162/229 failed episodes (the primary label on 123); survey labels are from one init-0 video per task, and pi0 is stochastic, so other runs may fail differently
observe diagnoses: R1_wrong_object 76, R2_scene_habit 21, R3_stopped_short 22, R4_placement 15, R5_target_not_reached 56, R6_destination_missed 31, R7_partial_task 8

library entry          used  recovered
R1_wrong_object          71    32
R2_scene_habit           28     4
R3_stopped_short         86    11
R4_placement             86     3
R5_target_not_reached   185    35
R6_destination_missed    48     5
R7_partial_task          20     6
```

### Per scene (`logs/survey90_breakdown.txt`)

```
by scene (agent / episodes, native):
  KITCHEN_SCENE1       agent  1/12  native 0/12
  KITCHEN_SCENE10      agent 13/18  native 0/18
  KITCHEN_SCENE2       agent 12/21  native 1/21
  KITCHEN_SCENE3       agent  1/ 9  native 0/9
  KITCHEN_SCENE4       agent  5/15  native 0/15
  KITCHEN_SCENE5       agent  4/15  native 1/15
  KITCHEN_SCENE6       agent  1/ 3  native 0/3
  KITCHEN_SCENE7       agent  0/ 9  native 0/9
  KITCHEN_SCENE8       agent  0/ 3  native 0/3
  KITCHEN_SCENE9       agent  3/18  native 0/18
  LIVING_ROOM_SCENE1   agent  6/ 9  native 1/9
  LIVING_ROOM_SCENE2   agent 14/15  native 1/15
  LIVING_ROOM_SCENE3   agent 10/12  native 0/12
  LIVING_ROOM_SCENE4   agent 10/12  native 0/12
  LIVING_ROOM_SCENE5   agent  3/ 6  native 0/6
  LIVING_ROOM_SCENE6   agent  7/ 9  native 1/9
  STUDY_SCENE1         agent  3/12  native 0/12
  STUDY_SCENE2         agent  2/ 9  native 0/9
  STUDY_SCENE3         agent  4/15  native 0/15
  STUDY_SCENE4         agent  3/12  native 0/12
never solved by the agent: [2, 6, 7, 8, 11, 13, 16, 18, 21, 22, 23, 27, 28, 29, 31, 35, 36, 37, 39, 40, 41, 42, 44, 45, 48, 73, 74, 75, 78, 80, 81, 84, 85, 86, 88, 89]
```

## 7. Development run details

Per-episode results (`episode  success  success_t  steps  trigger->diagnosis chain`).

### `logs/dev_v1_agent` (4/15)

```
task0_init0_agent        True   377    377  stall->R3_stopped_short > stall->R3_stopped_short
task4_init0_agent        False  -      642  premature_goal->R2_scene_habit > premature_goal->R2_scene_habit > disturbance->R3_stopped_short
task9_init0_agent        False  -      900  stall->R6_destination_missed > timeout->R4_placement > timeout->R4_placement
task20_init0_agent       True   467    467  no_progress->R5_target_not_reached > wrist_windup->R3_stopped_short
task30_init0_agent       False  -      493  stall->R5_target_not_reached > unplanned_articulation->R2_scene_habit > disturbance->R5_target_not_reached
task35_init0_agent       False  -      864  no_progress->R3_stopped_short > no_progress->R3_stopped_short > no_progress->R3_stopped_short
task39_init0_agent       False  -      721  stall->R5_target_not_reached > stall->R3_stopped_short > no_progress->R3_stopped_short
task43_init0_agent       True   277    277  stall->R5_target_not_reached
task46_init0_agent       True   240    240  stall->R5_target_not_reached
task59_init0_agent       False  -      481  stall->R5_target_not_reached > wrong_grasp->R1_wrong_object > wrong_grasp->R1_wrong_object
task62_init0_agent       False  -      542  wrong_grasp->R1_wrong_object > disturbance->R5_target_not_reached > wrong_grasp->R1_wrong_object
task63_init0_agent       False  -      722  out_of_view->R6_destination_missed > timeout->R6_destination_missed > timeout->R6_destination_missed
task65_init0_agent       False  -      642  disturbance->R5_target_not_reached > timeout->R6_destination_missed > timeout->R4_placement
task81_init0_agent       False  -      652  no_progress->R5_target_not_reached > timeout->R4_placement > timeout->R4_placement
task88_init0_agent       False  -      710  no_progress->R6_destination_missed > out_of_view->R4_placement > out_of_view->R4_placement
```

### `logs/dev_v2_agent` (7/15)

```
task0_init0_agent        True   426    426  stall->R3_stopped_short > no_progress->R3_stopped_short
task4_init0_agent        True   468    468  premature_goal->R2_scene_habit > stall->R3_stopped_short
task9_init0_agent        False  -      900  no_progress->R6_destination_missed > timeout->R6_destination_missed > timeout->R6_destination_missed
task20_init0_agent       False  -      762  stall->R5_target_not_reached > wrist_windup->R3_stopped_short > no_progress->R3_stopped_short
task30_init0_agent       False  -      485  stall->R5_target_not_reached > unplanned_articulation->R2_scene_habit > disturbance->R5_target_not_reached
task35_init0_agent       True   490    490  no_progress->R3_stopped_short
task39_init0_agent       False  -      879  no_progress->R5_target_not_reached > no_progress->R3_stopped_short > no_progress->R3_stopped_short
task43_init0_agent       True   136    136  -
task46_init0_agent       True   250    250  stall->R5_target_not_reached
task59_init0_agent       False  -      475  stall->R5_target_not_reached > wrong_grasp->R1_wrong_object > wrong_grasp->R1_wrong_object
task62_init0_agent       False  -      532  wrong_grasp->R1_wrong_object > disturbance->R5_target_not_reached > wrong_grasp->R1_wrong_object
task63_init0_agent       True   406    406  stall->R7_partial_task
task65_init0_agent       True   543    543  wrong_grasp->R1_wrong_object > disturbance->R5_target_not_reached > timeout->R5_target_not_reached
task81_init0_agent       False  -      734  no_progress->R5_target_not_reached > timeout->R4_placement > timeout->R6_destination_missed
task88_init0_agent       False  -      710  no_progress->R6_destination_missed > out_of_view->R4_placement > out_of_view->R4_placement
```

### `logs/dev_v3_agent` (3/12)

```
task9_init0_agent        False  -      900  no_progress->R6_destination_missed > timeout->R6_destination_missed > timeout->R6_destination_missed
task30_init0_agent       False  -      900  stall->R5_target_not_reached > unplanned_articulation->R2_scene_habit > timeout->R5_target_not_reached
task48_init0_agent       False  -      647  stall->R5_target_not_reached > disturbance->R5_target_not_reached > stall->R5_target_not_reached
task52_init0_agent       False  -      539  disturbance->R5_target_not_reached > timeout->R4_placement > stall->R4_placement
task53_init0_agent       True   304    304  no_progress->R5_target_not_reached
task59_init0_agent       True   237    237  stall->R5_target_not_reached
task62_init0_agent       True   644    644  no_progress->R5_target_not_reached > timeout->R5_target_not_reached > disturbance->R5_target_not_reached
task71_init0_agent       False  -      569  wrong_grasp->R1_wrong_object > disturbance->R5_target_not_reached > timeout->R5_target_not_reached
task80_init0_agent       False  -      819  no_progress->R5_target_not_reached > stall->R5_target_not_reached > timeout->R5_target_not_reached
task81_init0_agent       False  -      720  no_progress->R5_target_not_reached > timeout->R4_placement > timeout->R4_placement
task84_init0_agent       False  -      900  no_progress->R5_target_not_reached > disturbance->R5_target_not_reached > stall->R5_target_not_reached
task86_init0_agent       False  -      628  wrong_grasp->R1_wrong_object > timeout->R4_placement > timeout->R4_placement
```

### `logs/dev_v4_agent` (4/12)

```
task9_init0_agent        False  -      900  no_progress->R6_destination_missed > timeout->R6_destination_missed
task9_init1_agent        False  -      900  no_progress->R6_destination_missed > timeout->R4_placement
task30_init0_agent       False  -      759  stall->R5_target_not_reached > unplanned_articulation->R2_scene_habit > timeout->R6_destination_missed
task30_init1_agent       True   585    585  stall->R5_target_not_reached > unplanned_articulation->R2_scene_habit > unplanned_articulation->R2_scene_habit
task52_init0_agent       True   254    254  disturbance->R5_target_not_reached
task52_init1_agent       True   269    269  out_of_view->R5_target_not_reached
task71_init0_agent       True   361    361  wrong_grasp->R1_wrong_object > disturbance->R5_target_not_reached
task71_init1_agent       False  -      894  wrong_grasp->R1_wrong_object > stall->R5_target_not_reached > timeout->R6_destination_missed
task80_init0_agent       False  -      717  disturbance->R5_target_not_reached > timeout->R5_target_not_reached > timeout->R5_target_not_reached
task80_init1_agent       False  -      778  stall->R6_destination_missed > timeout->R4_placement > timeout->R6_destination_missed
task84_init0_agent       False  -      900  wrong_grasp->R1_wrong_object > stall->R5_target_not_reached > timeout->R5_target_not_reached
task84_init1_agent       False  -      774  no_progress->R5_target_not_reached > timeout->R5_target_not_reached > disturbance->R5_target_not_reached
```

### `logs/dev_v5_agent` (0/4)

```
task9_init0_agent        False  -      900  no_progress->R6_destination_missed > timeout->R6_destination_missed > timeout->R6_destination_missed
task9_init1_agent        False  -      894  stall->R6_destination_missed > stall->R6_destination_missed > stall->R6_destination_missed
task84_init0_agent       False  -      900  wrong_grasp->R1_wrong_object > timeout->R5_target_not_reached > disturbance->R5_target_not_reached
task84_init1_agent       False  -      900  wrong_grasp->R1_wrong_object > timeout->R5_target_not_reached > timeout->R5_target_not_reached
```

### `logs/dev_v6_agent` (3/6)

```
task9_init0_agent        True   879    879  no_progress->R6_destination_missed > timeout->R6_destination_missed
task9_init1_agent        True   635    635  no_progress->R6_destination_missed
task17_init0_agent       False  -      805  disturbance->R5_target_not_reached > timeout->R5_target_not_reached > timeout->R5_target_not_reached
task17_init1_agent       True   178    178  disturbance->R6_destination_missed
task84_init0_agent       False  -      900  disturbance->R5_target_not_reached > disturbance->R5_target_not_reached > timeout->R5_target_not_reached
task84_init1_agent       False  -      900  no_progress->R5_target_not_reached > timeout->R5_target_not_reached > timeout->R5_target_not_reached
```

