# Failure survey: libero_spatial, libero_10, libero_90 (50% task split)

This is the full-scale run of the failure survey prototyped in `Documentation/failure_survey_test/`. Every failed pi0 rollout in scope was surveyed **from the rollout video only**, and each failure is classified high-level (HL) or low-level (LL). LL failures have their onset and RL-reset steps recorded so the scene can be rebuilt for RL later. **No RL was run.** Per-episode metadata is in `survey_results.json`.

## Scope and sampling

| Suite | Rollouts | Failed | Surveyed |
|---|---|---|---|
| libero_spatial | 20 | **0** | 0 (pi0 succeeded on all 20, so there is nothing to survey) |
| libero_10 | 20 | 5 | 5 |
| libero_90 | 180 | 161 | **82** (all failures of 45 randomly chosen tasks) |
| **Total** | | | **87** |

- **libero_90 split.** The 50% is drawn at the **task** level: 45 of 90 tasks, `libero90_split.json`, seed `2018922090`. The other 45 tasks stay fully unseen for the zero-shot test in the README architecture. Those 45 tasks hold 82 of the 161 failures (51%).
- **Frame rate.** A random 20% of the 87 episodes (17) were surveyed at **3 fps**, one frame every 6-7 steps, so onset is known to ±7 steps. The other 70 were surveyed at **1 fps**, one frame every 20 steps (±20). The list is in `sampling_plan.json`.
- **Video only.** No `trace/` file was opened. Frames come from the .mp4 at the sampled rate only. The instruction text was the only metadata given to the surveyors.
  - One surveyor (LIVING_ROOM_SCENE3) looked at LIBERO's object texture assets to tell the cans and boxes apart. That is not rollout data.
  - The left/right convention was checked on a *successful* libero_10 video. "Robot's left" shows up on the **image right**, because the camera faces the robot.
- **Step convention.** step = round(20 × t_video), counting the 10 wait steps. To rebuild a scene, use `trace/states.npy[step]`, following `state_convention` in the rollout `meta.json`.
- **Process.**
  1. 8 surveyor agents each took one batch, grouped by scene so labels within a scene stay consistent, and followed `SURVEY_PROTOCOL.md`.
  2. Each wrote `episodes/<suite>/<folder>/survey.json` with zoomed `evidence_*.png` strips.
  3. They proposed 39 mode names between them. `mode_map.json` merges those into the 16 modes below, using per-episode overrides where two batches used the same name for different things.
  4. `aggregate.py` builds `survey_results.json`.

## Headline findings

1. **Most failures are high-level: 68 of 87 primary failures.** pi0's grasping and carrying are mostly fine on unseen tasks. What fails is *grounding the instruction*: which object, which instance, which destination.
2. **The scene drives behaviour more than the language.** In 44 of 87 episodes the surveyor saw pi0 running a *trained* libero_10 or libero_spatial task of the same or a similar scene, whatever the instruction said. Examples:
   - In KITCHEN_SCENE4 it puts the bowl in the bottom drawer and closes it.
   - In LIVING_ROOM_SCENE1/2/3 it puts the soup or cream cheese in the basket or tray.
   - In LIVING_ROOM_SCENE6 it puts the white mug on the plate.
   - In STUDY_SCENE1/2 it puts something in the back compartment.
   - In KITCHEN_SCENE8 it grabs a moka pot, even for "turn off the stove".
   
   Qualifiers (left/right/middle/back/front, colour, top/bottom drawer) and relation words (left of, front of, under, front compartment) are mostly ignored.
3. **Low-level failures cluster in a few motor skills:**
   - unstable rim, edge or handle grasps that slip or pivot (10 instances);
   - placement or release (7);
   - knocking things over before the grasp (4);
   - articulated fixtures: drawer push, door pull, stove knob (5).
4. **On its own finetuning tasks (libero_10), pi0 fails mostly low-level: 4 of 5.** The task order is right there; the motor execution is not. On libero_90 it is the other way round: 67 of 82 primaries are high-level.
5. **3 fps vs 1 fps.**
   - Within libero_90, the LL share of primaries is about the same: 3/14 at 3 fps, 12/68 at 1 fps.
   - The overall 3-fps LL share looks higher only because 3 of the 5 libero_10 episodes, which are mostly LL, landed in the 3-fps sample.
   - 1 fps was enough to *classify* every episode. 3 fps mainly tightens the RL reset step.

## Failure modes

Counts: **P** = episodes where this mode is the primary (root-cause) failure; **All** = every recorded instance, including secondary ones.

### High-level

| ID | Name | P | All | Definition | Seen in (libero_90 task/init unless marked) |
|---|---|---|---|---|---|
| HL-1 | subtask_handoff_stall | 6 | 6 | Finishes a subtask (a fixture action or a grasp), then idles; the next subtask never starts | L10 task8/0, 1/1, 9/0, 9/1, 23/0, 23/1 |
| HL-2 | scene_default_skill_override | 9 | 14 | Runs the scene's habitual skill or fixture interaction instead of the instructed skill (pushes the drawer shut, grabs a pot instead of the knob, skips the drawer) | 3/0-1, 8/0-1, 11/1, 31/1, 39/0-1, 85/1 (+ secondary in 6/0, 22/1, 23/0-1, 85/0) |
| HL-3 | wrong_object_category | 25 | 25 | Grasps or moves an object of a different category than the target, usually one from a trained task in the scene | 14/0, 21/1, 31/0, 41/0-1, 42/0-1, 49/0-1, 53/0-1, 57/1, 58/0-1, 59/0-1, 61/0-1, 62/0-1, 69/0-1, 77/1, 78/1, 80/0 |
| HL-4 | referring_expression_instance_error | 10 | 10 | Right kind of object or fixture, wrong instance; the qualifier (left/middle/right, colour, top/bottom drawer) is ignored | 6/0-1 & 11/0 (middle drawer opened), 38/1 (wrong moka pot), 71/0-1 (white mug, not red), 86/0-1 & 87/0-1 (always the same book) |
| HL-5 | intermediate_subtask_skipped | 4 | 4 | In "stack A on B and place them in C", skips the stack and carries one bowl straight to C | 63/0-1, 64/0-1 |
| HL-6 | destination_misgrounding | 11 | 18 | Right object, wrong destination or relation: the trained destination, a mirrored relation, or hovering without committing | 17/1, 25/0-1, 34/0-1, 73/0-1, 88/0-1, 89/0-1 (+ secondary in 9/0-1, 69/0, 86/0-1, 87/0-1) |
| HL-7 | target_never_approached | 3 | 3 | Wanders among other objects or fixtures all episode and never approaches the target | 35/1, 37/0-1 |
| HL-8 | grasp_loss_unnoticed | 0 | 3 | After a missed or slipped grasp, keeps transporting or placing with an empty gripper | 10/0-1, 80/1 (all secondary to LL-4/LL-5) |

### Low-level (RL candidates)

| ID | Name | P | All | Definition | RL subtask |
|---|---|---|---|---|---|
| LL-1 | drawer_push_stall | 2 | 2 | The drawer push stalls 1/2-2/3 closed while the hand rests on the drawer front | `rl/close_drawer` |
| LL-2 | hinged_door_pull_stall | 1 | 1 | The door swings partly open, then the pull stops short of the open threshold | `rl/open_door` |
| LL-3 | knob_approach_contortion | 2 | 2 | The wrist and elbow fold over the stove knob, the wrist body touches it, and the arm swings away without turning it | `rl/turn_knob` |
| LL-4 | unstable_grasp_slip | 7 | 10 | A rim, edge or handle grasp; the object tilts, pivots or slips out during lift or transport | `rl/pick` |
| LL-5 | pregrasp_contact_disturbance | 2 | 4 | A misaligned descent knocks over or pushes the target, or tips a nearby receptacle, before the grasp | `rl/pick` |
| LL-6 | placement_instability | 1 | 4 | Released off-centre, at an edge or tilted, so the object falls off, tips over or overhangs | `rl/place` |
| LL-7 | place_release_stall | 3 | 3 | Brought to (or just beside) the right destination, but never lowered or released | `rl/place` |
| LL-8 | pregrasp_hesitation | 1 | 3 | Hovers next to the right target with small adjustments for many seconds, using up the step budget | `rl/pick` (time-penalised) |

The 8 LL modes come down to **5 RL subtasks**. `rl/pick` covers LL-4, LL-5 and LL-8 and is the largest RL target (17 instances). For `rl/pick` and `rl/place`, the binary reward needs a per-subtask predicate (Grasped+lifted, or On/In), not the whole-task LIBERO success flag.

### LL reset index (for scene reconstruction)

`reset` is the last sampled step **before** the failing attempt starts; restore `trace/states.npy[reset]`. `onset` is the first sampled step where the failure is visible. Resolution is ±20 steps at 1 fps and ±7 at 3 fps (marked †). Rows marked ‡ happened while pi0 was handling a *non-instructed* object. They are evidence for the mode but are not task-relevant reset states.

| RL subtask | Mode | Episode | reset | onset | conf. |
|---|---|---|---|---|---|
| rl/close_drawer | LL-1 | libero_90/task1_fail_init0 (top drawer) | 60 | 160 | high |
| rl/close_drawer | LL-1 | libero_90/task22_fail_init1 (bottom drawer, bowl inside) | 280 | 340 | medium |
| rl/open_door | LL-2 | libero_90/task35_fail_init0 (microwave) | 160 | 260 | medium |
| rl/turn_knob | LL-3 | libero_90/task44_fail_init0 | 167 | 180 | medium |
| rl/turn_knob | LL-3 | libero_90/task44_fail_init1 | 180 | 200 | medium |
| rl/pick | LL-4 | libero_90/task10_fail_init0 † (black bowl) | 53 | 67 | high |
| rl/pick | LL-4 | libero_90/task10_fail_init1 (black bowl) | 40 | 80 | high |
| rl/pick | LL-4 | libero_90/task25_fail_init1 (black bowl, drops in transport) | 160 | 340 | high |
| rl/pick | LL-4 | libero_90/task77_fail_init0 (book pivots) | 80 | 140 | medium |
| rl/pick | LL-4 | libero_90/task78_fail_init0 (book pivots) | 80 | 140 | medium |
| rl/pick | LL-4 | libero_90/task85_fail_init0 † (white mug) | 180 | 207 | medium |
| rl/pick | LL-4 | libero_10/task2_fail_init1 (moka pot dragged) | 160 | 300 | medium |
| rl/pick | LL-4 | libero_90/task21_fail_init0 (frying pan swings) | 280 | 300 | low |
| rl/pick | LL-4 ‡ | libero_90/task41_fail_init0, task42_fail_init1 (white bowl) | 60, 160 | 140, 260 | low, medium |
| rl/pick | LL-5 | libero_90/task80_fail_init1 (book knocked over) | 60 | 100 | high |
| rl/pick | LL-5 | libero_10/task6_fail_init0 † (pudding toppled) | 253 | 267 | medium |
| rl/pick | LL-5 | libero_90/task31_fail_init0 (bowl pushed) | 280 | 300 | low |
| rl/pick | LL-5 ‡ | libero_90/task49_fail_init0 (basket tipped) | 180 | 220 | medium |
| rl/pick | LL-8 | libero_10/task8_fail_init1 † (2nd moka pot) | 260 | 340 | medium |
| rl/pick | LL-8 | libero_90/task21_fail_init0, task21_fail_init1 † (frying pan) | 160, 200 | 200, 227 | low |
| rl/place | LL-7 | libero_90/task14_fail_init1 (bowl → plate) | 120 | 220 | medium |
| rl/place | LL-7 | libero_90/task17_fail_init0 (bowl → bowl stack) | 120 | 240 | medium |
| rl/place | LL-7 | libero_90/task72_fail_init1 (mug → plate) | 160 | 220 | medium |
| rl/place | LL-6 | libero_90/task38_fail_init1 (moka pot tips on stove) | 260 | 320 | high |
| rl/place | LL-6 | libero_10/task4_fail_init1 † (mug overhangs plate) | 333 | 373 | low |
| rl/place | LL-6 ‡ | libero_90/task41_fail_init1, task42_fail_init0 (bowl off cabinet edge) | 207, 200 | 220, 240 | medium |

Guide demos for RL, using successful rollouts of the same skill:
- `rl/turn_knob`: libero_10 task2 successes, KITCHEN_SCENE3, where pi0 turns the knob reliably.
- `rl/pick`: the libero_spatial bowl picks (bowls) and the libero_10 task5 successes (books).
- `rl/close_drawer`: libero_10 task3 successes, as in the prototype, plus libero_90 task22_success_init0. The PoC reward env, libero_90 task0, is now a held-out test task, so task22 (in the survey split) replaces it.
- `rl/open_door`: no in-distribution door-*opening* demo exists; libero_10 task9 only closes the microwave.

## Proposed remedies (high-level only)

The prototype's primitives are used throughout: HALT, RETREAT, REPROMPT, INSERT_SUBTASK, VERIFY, WATCH, INVOKE_RL, ESCALATE. Re-prompts are full LIBERO-style sentences.

**One new primitive is needed: `POINT_APPROACH(obj)`.** HL-2, HL-3, HL-4 and HL-6 account for 55 of the 68 HL primaries. In all of them, re-prompting fights a scene prior that already beat the original prompt. The failure-agent v1 prototype found that "coordinator says WHERE, pi0 does the grasp" was the winning recipe. `POINT_APPROACH` makes that available without privileged state:
1. The VLM points at the object in the 1 fps agentview frame.
2. The pixel is back-projected onto the table or fixture plane using the known camera calibration.
3. A scripted OSC move brings the EE 7 cm above that point.
4. pi0 is then handed the pick or place sentence.

Once the gripper is over the right object, pi0's local grasp prior takes over, and the ignored qualifier no longer matters.

| ID | Detect (1 fps image + proprio) | Recovery | Prevention (coordinator) |
|---|---|---|---|
| HL-1 subtask_handoff_stall | EE moves < 1 cm over 3 frames, or leaves the camera view; the previous subtask passes VERIFY | VERIFY(prev done) → HALT → RETREAT 10 cm (back into view if needed) → REPROMPT next subtask sentence → WATCH 3 s; retry once → ESCALATE. For a *held* object (task9): REPROMPT the place sentence while it is still held | Issue one subtask at a time and advance only on VERIFY; never rely on pi0 to sequence a compound sentence |
| HL-2 scene_default_skill_override | The empty gripper heads for a non-target fixture over 2 frames, or a protected state flips (e.g. "destination drawer open") | HALT → RETREAT 10 cm + 5 cm away from the attractor → if a protected state broke, INSERT_SUBTASK to restore it (`open the top drawer...`) → POINT_APPROACH(target) → REPROMPT object-first sentence → WATCH 5 s; retry with 15 cm → ESCALATE | Declare protected states before running pi0. For fixture tasks (knob, drawer), POINT_APPROACH the fixture *before* the first pi0 step, because trained-object attractors win in the first second |
| HL-3 wrong_object_category | **Pre-grasp check:** when the EE descends below about 5 cm above the table or the gripper starts closing, VERIFY "the object between the fingers is {target}" on a 2x crop | HALT → if lifted, set the wrong object back down → RETREAT 10 cm → POINT_APPROACH(target) → REPROMPT `pick up the {target} and place it {rel} the {dest}` → VERIFY(target grasped); retry once → ESCALATE | List the scene's trained-task objects as known distractors, and run the pre-grasp VERIFY on every pick. For these scenes, start with POINT_APPROACH instead of plain prompting |
| HL-4 referring_expression_instance_error | Same pre-grasp or pre-pull VERIFY, phrased around the qualifier: "is the gripper on the *lowest* handle?" or "is this the *red* mug?" on a 3x zoom | HALT → RETREAT → (drawer case: optionally close the wrongly opened drawer) → POINT_APPROACH(the resolved instance) → REPROMPT → VERIFY; retry → ESCALATE | The coordinator resolves the qualifier itself (in the robot frame: robot-left = image-right; front = lower in the image) and passes pi0 a pointed target, never the qualifier alone. pi0 ignores left/middle/right and top/bottom |
| HL-5 intermediate_subtask_skipped | The gripper carries a single object toward the final container before VERIFY(stacked) | HALT → if the base bowl is already in the tray, stack in place: REPROMPT `pick up the {top} and place it on the {base}`; else put it down, POINT_APPROACH(top) → REPROMPT stack sentence → VERIFY(stacked) → REPROMPT `pick up the {base} and place it in the {container}` | Split "stack and place" into two prompts gated by VERIFY. pi0 never stacked in any of the 4 episodes |
| HL-6 destination_misgrounding | The loaded gripper moves toward a non-destination fixture over 2 frames, or hovers more than 3 s without nearing the destination | HALT (keep the grasp) → RETREAT up 10 cm → POINT_APPROACH(destination point) → REPROMPT the place sentence (or a scripted lower-and-release if pi0 still won't release) → VERIFY(on/in); retry → ESCALATE | The coordinator turns relation and compartment words ("left of the plate", "front compartment", "under the shelf") into a target *point* before the carry; pi0 is not trusted with relations. Shelf placements may need a horizontal-insertion skill (v1 open gap) |
| HL-7 target_never_approached | After 3 frames, the EE is not heading toward the target (VLM heading check) | HALT → RETREAT → POINT_APPROACH(target), especially for raised or unusual start poses like a bowl on top of the microwave → REPROMPT → WATCH 3 s → ESCALATE | Check the heading early, within 2-3 frames, rather than waiting for the timeout |
| HL-8 grasp_loss_unnoticed | **Post-grasp check:** the gripper is fully closed (width ≈ 0), or a crop shows nothing between the fingers, while the EE is moving to the destination | HALT → RETREAT 10 cm → POINT_APPROACH(object at its new pose) → INVOKE_RL(`rl/pick`) once trained, else REPROMPT the pick sentence → VERIFY(grasped); retry → ESCALATE | VERIFY the grasp after every pick before allowing transport. This turns every LL-4/LL-5 into a recoverable event |

General FR-agent rules that came out of the survey:
- **Treat "arm out of camera view for 2 frames" as a stall.** It was seen in 4 episodes: libero_10 task8/0, libero_90 task42/0 and both task44 episodes.
- **Fire early.** Median HL onset is about step 60 (3 s). Most episodes then waste 300+ steps, so a check at 2-3 frames leaves most of the budget for recovery.
- **The step budget matters.** libero_90 runs 400 steps. LL-7 and LL-8 hesitations, and HL-1 idles, often take 150-250 steps. A WATCH timeout of about 5 s on any no-progress state is needed.

## Scene-level observations

| Scene (tasks) | What pi0 does regardless of instruction |
|---|---|
| KITCHEN_SCENE10 (1, 3) | Pushes the open top drawer shut (HL-2/LL-1, same as the PoC) |
| KITCHEN_SCENE4 (22, 23, 25) | Runs libero_10 "black bowl in the bottom drawer and close it", then idles |
| KITCHEN_SCENE1/2 (6, 8-11, 14, 17) | Runs the libero_spatial bowl→plate routine; opens the *middle* drawer for "top" and "bottom" |
| KITCHEN_SCENE9 (41, 42, 44) | Never touches the frying pan; carries the white bowl to the cabinet top. The knob approach contorts (LL-3) |
| KITCHEN_SCENE8 (38, 39, L10 8) | Grabs a moka pot even for "turn off the stove"; the second pot is slow or never started |
| KITCHEN_SCENE3 (21, L10 2) | Turns the stove on reliably; the pan grasp hesitates, or it falls back to the moka pot |
| KITCHEN_SCENE6/7 (34, 35, 37) | Drawn to the microwave and plate; never reaches the bowl on top of the microwave |
| LIVING_ROOM_SCENE1/2/3 (49, 53, 57-59) | Replays "put both X and Y in the basket" (soup/cream cheese/butter); the tray is treated as the basket |
| LIVING_ROOM_SCENE4 (61-64) | Generic "one object into the tray"; the object name and left/right are ignored; never stacks |
| LIVING_ROOM_SCENE5/6 (69, 71, 72, L10 4/6) | White mug onto the plate first (trained); pudding goes "right of" even when told "left of" |
| STUDY_SCENE1/2 (73, 77, 78, 80) | Everything goes to the middle-back compartment (trained); 3 of 6 grab the mug instead of the book |
| STUDY_SCENE3 (85) | Drifts to the book→caddy behaviour instead of the mug |
| STUDY_SCENE4 (86-89) | Always grabs the same book (robot-right) and carries it to the same spot; rarely reaches the shelf |

## Caveats

- The labels are visual judgements from 256 px frames, with no predicate check. The low-confidence calls are listed with `confidence: low` in `survey_results.json`. Notable ones:
  - task4/1 (L10): the overhang cause is not visible.
  - task21/0: pan swing vs. hesitation.
  - task89/1: possibly an LL insertion failure under the shelf rather than HL-6.
  - task6/0: possibly an LL miss on the bottom handle.
- Object identities in the living-room scenes, especially the cans, come from appearance.
- HL-3 and HL-4 are separated by object category. In practice they share a cause (trained-task bleed-through) and a remedy (POINT_APPROACH).
- Onset and reset steps carry the sampling resolution (±20 or ±7 steps). Before RL, check each `rl_reset_step` against the trace: does the restored state show the pre-attempt configuration described in the observation?
- All remedies are **proposed**. None has been replayed in closed loop.

## Files

- `README.md`: this file.
- **`skill_library/library.json` + `skill_library/README.md`**: the failure recovery skill library v0.2, in the PoC schema. It has 16 entries, each with detect / recovery / prevention / success_check / instances; LL entries also carry an `rl_spec`, and `rl_policies` groups the reset states per RL policy. It replaces the PoC v0.1 library. Generated by `build_library.py`.
- `survey_results.json`: everything machine-readable:
  - sampling metadata and the unified taxonomy (with remedies and RL subtasks);
  - stats;
  - `ll_rl_reset_index`: every LL failure with its reset and onset steps, frame path, and `on_instructed_subtask` flag;
  - the full per-episode records: subtasks, behaviour summary, `runs_trained_task`, and failures with both the surveyor's and the unified mode.
- `sampling_plan.json`, `libero90_split.json`: the random draws (one seed).
- `SURVEY_PROTOCOL.md`: the surveyor instructions and the per-episode schema.
- `mode_map.json`: surveyor mode name → unified ID, plus per-episode overrides.
- `build_library.py`: builds `skill_library/library.json` from `survey_results.json` (run after `aggregate.py`).
- `extract_frames.py`: makes the split, the fps draw, the frames and the contact sheets. `zoom_frames.py` makes the zoom strips. `aggregate.py` builds `survey_results.json`.
- `episodes/<suite>/<folder>/`: `survey.json`, `sheetNN.png`, `frames/stepNNNN.png` and `evidence_*.png`, about 440 MB in total.
