# Failure survey PoC: libero_90, first 5 failures

**Scope.** This covers the first 5 failed episodes in `pi0_LIBERO_rollouts/libero_90/meta.json` order: task0 init0/1, task1 init0/1 and task2 init0. Only the rollout videos were used. `trace/` was never opened, and `meta.json` was read only for the task instruction and the video file name. No RL was run.

**Sampling.** The 5 fps episode was chosen at random (seed `664752799`, see `sampling_plan.json`) and came out as **task2_fail_init0**: 104 frames, one every 4 steps. The other four were surveyed at 1 fps: 22 frames each, one every 20 steps, plus the last frame. Each step maps to video time as step = round(20 x t). Step 0-9 are the env wait steps.

## Per-episode result

| Episode | Instruction | fps | Failure | Level | Subtask | Onset step |
|---|---|---|---|---|---|---|
| task0_fail_init0 | close the top drawer | 1 | LL-1 push stalls, drawer ~2/3 closed | low | close(top_drawer) | ~160 (160-180) |
| task0_fail_init1 | close the top drawer | 1 | LL-1 push stalls just short of flush | low | close(top_drawer) | ~180 (160-180) |
| task1_fail_init0 | close drawer + bowl on top | 1 | LL-1 push stalls; bowl subtasks never start | low | close(top_drawer) | ~160 (140-160) |
| task1_fail_init1 | close drawer + bowl on top | 1 | HL-1 drawer closed (~step 100), then hovers, never picks bowl | high | pick(black_bowl) | ~100 (80-100) |
| task2_fail_init0 | bowl **into** top drawer | 5 | HL-2 closes the destination drawer instead of picking the bowl | high | pick(black_bowl) | ~44 (40-44) |
| (same) | | | LL-1 that wrong push also stalls ~2/3 closed | low | close(top_drawer)* | ~136 (136-148) |

\* Not part of task2. Recorded only as more evidence for LL-1.

The zoomed key frames behind each call are in `<episode>/evidence_*.png`. All sampled frames are in `<episode>/frames/step####.png`, and the labelled contact sheets are in `<episode>/sheet##.png`.

## Failure recovery skill library

The survey's findings are organized as a reusable library in **`skill_library/`**. `library.json` is the version the FR agent loads; `README.md` is the same content in readable form.

Each entry is one failure mode:
- `id` / `level`: `HL-n` (high-level) or `LL-n` (low-level).
- `applies_when`, `slots`: the general pattern. Slots (target object, destination, finished subtask, ...) are filled from the coordinator's plan at test time.
- `detect`: a cheap proprio cue checked every 1 fps frame, a VLM question about the image, and `distinguish_from` for look-alike entries.
- `recovery`: ordered steps built only from the shared primitives (`HALT`, `RETREAT`, `REPROMPT`, `INSERT_SUBTASK`, `VERIFY`, `WATCH`, `INVOKE_RL`, `ESCALATE`). They use images and proprio only, with no privileged sim state. Each recovery ends with a `success_check` and a retry/escalate branch.
- `prevention`: what the coordinator should do so the failure doesn't happen in the first place.
- `rl_spec` (LL only): the subtask, the binary LIBERO reward source, reset states (episode + onset step) and guide demos from LIBERO-spatial / LIBERO-10 successes.
- `instances`, `status`: every surveyed episode and onset step where the mode was seen. Status goes `proposed` → `validated` once the recovery works when replayed from a recorded onset state.

**Prompt rule:** pi0 re-prompts use full LIBERO-style sentences, e.g. `pick up the {obj} and place it on the {dest}`. pi0 has never seen bare atomic verbs ("pick up the bowl") in its finetuning data.

## Failure modes found

All recoveries below are **proposed**; none has been run in closed loop yet.

| ID | Failure | Seen in (onset step) | Quick recovery |
|---|---|---|---|
| **HL-1** subtask_handoff_stall | Finishes subtask k of a compound instruction, then idles; subtask k+1 never starts | task1 init1 (~100) | VERIFY k done → HALT → RETREAT up 10 cm → REPROMPT `pick up the {next_obj} and place it on the {next_dest}` → WATCH 3 s; retry once, then ESCALATE |
| **HL-2** scene_default_skill_override | Runs the scene's habitual skill (pushing the open drawer shut) instead of the instructed one, wrecking the destination | task2 init0 (~44) | HALT on a protected-state violation or a wrong-fixture approach → RETREAT up 10 cm and 5 cm away → if the destination got closed, INSERT_SUBTASK `open the {destination}` → REPROMPT object-first `pick up the {obj} and place it in the {destination}` → WATCH 5 s; retry with 15 cm, then ESCALATE |
| **LL-1** drawer_push_stall | Pushes the drawer from above; it stops 1/2-2/3 closed while the hand rests on the drawer front | task0 init0 (160), task0 init1 (180), task1 init0 (160), task2 init0 (136, inside HL-2) | `INVOKE_RL(rl/close_drawer)` once trained. Until then: RETREAT 5 cm, REPROMPT `close the {drawer}` (expected to be weak) |

**LL-1: drawer push stalls before closure (low-level).** Seen in 4 of 5 episodes. pi0 pushes the open top drawer from above with its fingertips and the underside of the hand. The drawer travels about 1/2 to 2/3 of the way. pi0 then pitches the fingers down and rests against the drawer front near the handle, applying no further push, until the 400-step timeout.
- **Detect:** no motion along the closing direction for 2 frames, and a 2x zoom VERIFY says the drawer front is not flush.
- **RL subtask:** `close(top_drawer)`. libero_90 task0 *is* that subtask, so its LIBERO success predicate can be the binary reward directly.
- **RL reset states:** the four recorded onset steps plus task0 inits 0/1. Restoring sim state at those steps needs the trace; see `state_convention` in meta.json.
- **Guide demo:** libero_10/task3_success_init0 ("put the black bowl in the bottom drawer of the cabinet and close it"). Around steps 210-238 pi0 closes the drawer with its hand down beside the drawer front, pushing straight in along the closing direction. That is the motion missing in LL-1. Candidates not yet watched: task3_success_init1 and task9_success_init0 (microwave close).

**HL-1: no switch to the next subtask after a success (high-level).** Seen in task1_fail_init1. pi0 closes the drawer by step ~100, then hovers over the cabinet with the gripper open for about 300 steps. It never approaches the bowl, which is in plain view.
- **Detect:** EE moves < 1 cm over 3 frames, the gripper is open, the previous subtask passes VERIFY, and the EE is not moving toward the next target.
- **Prevent:** the coordinator gives pi0 one subtask at a time and advances only on VERIFY, never relying on pi0 to sequence a compound sentence.
- **Instance binding:** next_obj = black bowl, next_dest = top of the cabinet.

**HL-2: runs the scene's default skill instead of the instruction (high-level).** Seen in task2_fail_init0. The opening scene is the same as in task0/task1 (open drawer, bowl in front). pi0 runs the close-drawer behaviour even though the drawer is the place destination.
- **Detect:** with the gripper empty, the EE heads for a non-target fixture over 2 frames, or a protected state ("top drawer is open") flips to false.
- **Prevent:** before running pi0, the coordinator declares protected states, and the FR agent HALTs on the first violation instead of waiting for the timeout.
- **Why retreat first:** pi0's choice of skill seems driven by what is near the gripper. Re-prompting while the hand is still at the drawer would likely restart the push.

## Notes for the full survey

- **Video-only takeaway.** In all five episodes, pi0 does almost the same thing in this scene whatever the instruction: it goes to the open drawer and pushes. Sometimes that matches the instruction (task0/1) and sometimes it doesn't (task2). This hints that the scene drives pi0 more than the language does here. It is a hypothesis from 3 tasks in a single scene.
- **1 fps was enough to classify every episode.** Pinning down the low-level onset at 1 fps leaves ±20 steps of uncertainty. 5 fps on task2 narrowed that to ±4 steps but did not change the diagnosis, at about 5x the frames.
- **Zoom is required.** At 256 px, "drawer 2/3 closed" and "drawer flush" look almost the same in a full frame. Each call here needed a 2x crop of the cabinet (`zoom_frames.py`). The surveyor agent needs a crop/zoom tool, or a higher-resolution render.
- **Uncertainty.** Without the trace, "closed" in task1_fail_init1 is a visual judgement (all three handles in line), not the LIBERO predicate.
- **Library maintenance.** Before adding a new entry, check whether the failure is an existing entry with different slot values. If so, add an `instance` instead, so the library stays small and general.

## Next steps
1. Validate each recovery by replaying it from its recorded onset state (needs trace states + sim). Mark entries `validated` or revise them.
2. Train `rl/close_drawer` (LL-1) with SimpleVLA-RL using the `rl_spec` in `library.json`.
3. Extend the survey to LIBERO-spatial, LIBERO-10 and the 50% LIBERO-90 training split. Grow the library by adding instances first and new entries only when needed.

## Files
- `extract_frames.py`: samples the frames, makes the 5 fps / 1 fps random choice, writes `sampling_plan.json` and the contact sheets.
- `zoom_frames.py`: makes the zoomed evidence strips.
- `skill_library/library.json`, `skill_library/README.md`: the failure recovery skill library.
- `survey_results.json`: per-episode results (subtasks, failures, level, onset step/window, RL candidates); each failure links to its `library_entry`.
- `<episode>/`: sampled frames, contact sheets, `evidence_*.png`.
