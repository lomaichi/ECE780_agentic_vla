# VLM inference speed test

Live closed loop (pi0 + VLM subtask judge at 1 fps) with per-component timing, to find a VLM fast enough to
replace Opus as the live judge. Files:

- `live_judge.py` – runs one libero_90 episode, live window and/or `--video` (HUD with model name and the
  inference time of every VLM call; frames where the sim is frozen waiting for the VLM are included and marked)
- `judge_agent.py` – `SubtaskJudgeAgent` (full JSON judge + corrections, for Opus), `SubtaskSuccessJudge`
  (yes/no question per call), `GroundedSuccessJudge` (Moondream2 `detect` + proprioception)
- `vlm_backends.py` – Claude CLI, Moondream cloud API, `MoondreamLocal` (Moondream2 on the local GPU)
- `moondream_server.py` – HTTP server for Moondream2 (own venv; started automatically by `MoondreamLocal`);
  `/query`, `/detect`, `/point`
- `recovery_agent.py` – `GroundedRecoveryAgent`: grounded judge + grounded failure detection and high-level
  recovery (section below)
- `summarize_recovery.py` – results table of `--judge recovery` runs

## Moondream2 (local)

- Weights: `ECE780/model/moondream2` (`vikhyatk/moondream2`, revision 2025-06-21), tokenizer cached in
  `ECE780/model/hf_cache`, venv `ECE780/model/.venv-moondream` (py3.11, torch 2.11 cu128: the container's torch
  cu126 builds have no sm_120 kernels for the RTX 5090). ~5 GB of GPU memory next to pi0's ~24 GB.
- Run (pi0 server up):
  ```
  MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python live_judge.py --task 10 --init 0 --vlm moondream2 \
      --no-window --speed 0 --run-past-success --video video/moondream2_grounded_task10_init0.mp4 \
      --out video/moondream2_grounded_task10_init0.json
  ```
  `--judge success` for the yes/no judge; drop `--no-window` for the live window.

## Results (2026-11-20, RTX 5090, 1 judge call per second of sim time)

| | per call (round trip) | GPU part | episode pipeline |
|---|---|---|---|
| Moondream2 `detect` (grounded judge) | 76 ms (p95 78 ms) | 61 ms | 2.2–2.3x real time, pi0 57 ms/chunk |
| Moondream2 yes/no `query` | 74 ms | ~50 ms | |
| Moondream2 full JSON judge prompt (judge_agent.SubtaskJudgeAgent) | 0.8–0.9 s | | prompt ~1100 tokens + 730 image tokens of the 2048 context |

Judge quality vs the privileged sim checks (drawn in the videos as ground truth, never shown to the agent):

- Yes/no VQA: unusable. Moondream2 answers "yes" (or always "no") whatever the image; 6 phrasings x 3 views
  over 97 labelled frames, none separated pick from no-pick. `video/moondream2_vqa_task10_init0.mp4`: "task
  complete" at step 50, before the arm moved.
- Full JSON judge prompt: same failure (pick "success" at step 10) and it nearly overflows the context.
- Grounded (detect + gripper proprioception, final version, 5 episodes): pick judged within one call (±20 steps) of ground truth in 4 of 5
  episodes, the miss was a lift shorter than 1 s; no false pick after requiring the box to rise (plain
  displacement fired on a push, fixed). No false place. pi0 never completed a place in these runs, so place
  successes are untested; one place false positive (bowl at the cabinet edge) was caught by the
  confirm-twice rule in an earlier version.

Continuous monitoring (grounded judge, 2026-11-20): a subtask success is no longer final. Every call also
re-checks the completed subtasks from the same `detect` call + proprioception (pick: the object stays in the
gripper or is already at the destination; place: the object, when detected, is still at the destination). A
subtask contradicted on 2 consecutive calls (`LOST_CONFIRM`) is undone and the judge goes back to it; the lost
completion stays in `subtasks[k]["history"]` and the HUD shows `SUCCESS @90 LOST @170`. After a lost pick the lift
reference is re-based to where the object comes to rest. Only positive evidence counts (undetected object = no
contradiction). The ground-truth row now also shows whether a latched check still holds (`now held / DROPPED`).
Task 9 init 2 (bowl dropped next to the plate at ~step 142): before, `pick` stayed SUCCESS for the rest of the
episode (`video/moondreamV2_1fps/moondream2_grounded_task9_init2.mp4`); now contradicted at 150, LOST at 170
(`..._task9_init2_continuous.mp4`). Task 10 init 0 (bowl carried throughout): no false loss.

Videos: `video/moondream2_grounded_task10_init0.mp4`, `video/moondream2_grounded_task9_init2.mp4`,
`video/moondream2_vqa_task10_init0.mp4`; other runs in `video/attempts/`. Each has a json timing report.

## Failure detection and recovery (Moondream2 grounded judge, 5 fps)

`live_judge.py --vlm moondream2 --judge recovery` (judge at 5 fps by default for the moondream2 judges) runs
`recovery_agent.GroundedRecoveryAgent`: the grounded success judge (unchanged rules, plus the fixes below) and,
on the same single `detect(object)` call per judge call, failure detection and correction. Moondream2 cannot
answer the fa_core `SubtaskJudgeAgent` JSON (failure type, action, pointed pixel), so nothing is asked: the
object / destination boxes are back-projected with the agentview depth image + calibration (`fa_core.camera`)
to world points and compared with the EE position (proprio). Rules and thresholds are in the module docstring;
each maps to a skill-library id (`failure_survey/skill_library/library.json`):

| failure (library id) | measured as | action |
|---|---|---|
| wrong_grasp (HL-3/HL-4) | gripper closed on something >= 12 cm (xy) from the target, target box not moving | release_and_retreat + approach target |
| wrong_approach (HL-2/3/4) | EE descending / closing within 6 cm of the target height, >= 12 cm away | point_approach(target) |
| target_not_approached (HL-7), stall (HL-1/S1) | empty gripper > 8 cm from the pre-grasp point (7 cm above target, 3-D) and < 2 cm closer in 3 s | point_approach(target) |
| grasp_loss (HL-8) | judge undoes a completed pick away from the destination | release_and_retreat + approach object |
| wrong_destination (HL-6) | carried object descending near the destination height, > 8 cm (xy) from it | carry_to(destination) |
| destination_not_approached (HL-6), handoff_stall (HL-1) | carrying, > 8 cm from the release point (3-D), < 2 cm closer in 3 s | carry_to(destination) |
| missed_grasp (LL-4/5), pregrasp_hesitation (LL-8), release_stall (LL-7), placement_instability (LL-6) | near target / destination, closed on nothing / still / released off it | reported only (RL's job) |

A rule must hold for 0.6 s of consecutive calls; checks pause 1 s after a subtask change or a correction (3 s
cooldown, 6 corrections max, as in fa_core). Correction motions are fa_core's (`_motion_world`, aimed at world
points instead of a VLM pixel). Destination point: inside the destination box (highest surface for "on top
of"); for "in the ... drawer" `point("open drawer")`, because `detect` boxes the whole cabinet for every drawer
phrase. Instructions that are not one pick-and-place: success judge only, no recovery.

Spatial relations (2026-11-24): `parse_instruction` also reads "to the right of / to the left of / to the front of /
in front of / behind X" (libero_90 tasks 34, 37, 69, 70, 76, 84, 85; all other LIBERO instructions parse as
before). left / right are the robot's: "right of X" is the IMAGE LEFT of X's detect box (checked against the BDDL
goal regions projected into the image). Place success: the object's box centre within 0.25 (image fraction) of X's
box edge on that side. Destination point: offset along that image axis from the table in a strip beside X's box,
X's own position along the other axis. With real `detect` boxes it falls inside the BDDL goal region in 6 of 7
tasks (0.9-2.4 cm from its centre; `failure_agent_prototype/framework/runs/_test/check_side_dest.py`); in task 34
`detect("white mug")` boxes the yellow-and-white mug, so the reference is wrong. "under" (task 42) still falls back.

Live window (pi0 server up, a display):
```
MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python live_judge.py --task 2 --init 0 --vlm moondream2 \
    --judge recovery --run-past-success
```
The terminal prints one line per judge call as it happens (success detection, EE distances, failure verdict,
decision; `--print events` only for calls where something happens), correction ends with the ground truth at
that moment, and privileged ground-truth changes (display only). The window's HUD adds the measured target /
destination / EE points, a failure-monitor line (suspect / HIGH_LEVEL / low_level, grace) and the corrections.
`--no-window --speed 0 --video X.mp4 --out X.json` for headless runs; per-call ground truth is logged as
`gt_now` in the json (added after the agent decided).

Judge fixes made for this (they also apply to `--judge grounded`):
- tracking: the nearest detect box is rejected when it moved more than 0.1 + 1.0/s x (time since the object was
  last seen) (image fraction): at 5 fps `detect` switched to another box ("rose +0.52") and judged pick 28 steps
  before the ground truth.
- "middle": chosen along the WORLD axis with the larger spread when the agent has depth (recovery agent): in
  the image, perspective shrinks front/back, so task 17 picked the front bowl as the middle one.
- paths: the files moved one level down into `moondreamV2/`; `sys.path` / model paths fixed.

### Results (2026-11-20, libero_90 init 0 and 1 of the six tasks of the Opus judge run, `video/recovery/`)

Native pi0 rollouts (`Documentation/pi0_LIBERO_rollouts`) solve only task 24 of these six.

| task | instruction | pick GT / judge | place GT / judge | LIBERO success | failures (onset step) | corrections (start-end, GT at end) |
|---|---|---|---|---|---|---|
| 2 init 0 | put the black bowl in the top drawer of the cabinet | 235 / 229 | - / - | no | target_not_approached @102, wrong_destination @281 | point_approach @102-197; carry_to @281-406 (not placed) |
| 2 init 1 | same | 286 / 281 | - / - | no | target_not_approached @102, destination_not_approached @373 | point_approach @102-245; carry_to @373-(episode end) |
| 17 init 0 | stack the middle black bowl on the back black bowl | - / - | - / - | no | target_not_approached @226, @370 | point_approach @226-278; @370-(end) |
| 17 init 1 | same | 378 / - | - / - | no | target_not_approached @190 | point_approach @190-269 |
| 24 init 0 | put the black bowl in the bottom drawer of the cabinet | - / 118 | 169 / 190 | yes @160 | - | - |
| 24 init 1 | same | 193 / 166 | 229 / 226 | yes @210 | - | - |
| 25 init 0 | put the black bowl on top of the cabinet | - / - | - / - | no | target_not_approached @222, wrong_approach @382 | point_approach @222-286; @382-(end) |
| 25 init 1 | same | 228 / 220 | - / - | **yes @408** | target_not_approached @126, wrong_destination @292 | point_approach @126-180; **carry_to @292** (released on the cabinet) |
| 30 init 0 | put the black bowl on the plate | 233 / - | - / - | no | target_not_approached @122 | point_approach @122-179 |
| 30 init 1 | same | - / - | - / - | no | stall @130, target_not_approached @346 | point_approach @130-238; @346-(end) |
| 36 init 0 | put the white bowl on the plate | - / - | - / - | no | wrong_approach @202 | point_approach @202-291 |
| 36 init 1 | same | - / - | - / - | no | wrong_grasp @78, target_not_approached @325 | release_and_retreat @78-189; point_approach @325-390 |

Findings (this batch + the previous valid batch, `video/attempts/recovery_v2_far8cm/`, 24 episodes):
- Grounding: target points within ~3 cm (xy) of the sim object on all six tasks, destination points within
  0.6 cm for plates; cabinet top and drawers land on the right surface (checked once per task).
- Detection vs ground truth: one harmful false positive, in the earlier batch: `wrong_grasp` released a correct
  rim grasp of the bowl (task 25: 8 cm from the target point). Fixed by the 12 cm grasp threshold; the true wrong
  grasps were 21-29 cm away. No correction in the final batch fired while the ground truth held the target
  during pick. `target_not_approached` needs >= 4.6 s of evidence (first at step 102; Opus pointed at step 50).
- Recovery: `carry_to` after `wrong_destination` produced 2 LIBERO successes pi0 does not reach natively (task 25
  init 1 here; task 2 init 0 in the earlier batch, bowl into the top drawer). `point_approach` /
  `release_and_retreat` were followed by a real grasp in 11 of 20 cases; never on task 36 (white bowl raised on
  a shelf).
- Limits: the scripted vertical descent of `point_approach` stalls ~8 cm high in task 30 (the hand hits the
  white cabinet next to the bowl; the Opus run had the same ~95-step correction). Detection waits 1 s grace +
  3 s window, and `carry_to` takes ~110 steps, so late place corrections run out of the 400-step budget (task 2
  init 1). The pick judge still misses some lifts (task 30 init 0, task 17 init 1). Low-level failures are only
  reported. The ground-truth `pick` check misses bowls held by the rim (MuJoCo `_check_grasp`), so it is
  conservative.
- Batch runs: start `moondream_server.py` yourself before running episodes in parallel. A server started by
  `MoondreamLocal` is stopped when THAT process exits, and parallel episodes then get "connection refused"
  (an earlier batch lost 33-39 calls per episode this way and was discarded).
