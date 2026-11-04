# Failure survey protocol (surveyor instructions)

This is the protocol every surveyor follows, one batch of failed pi0 rollouts at a time. It extends the 5-episode proof of concept in `Documentation/failure_survey_test/`. Read that folder's `failure_survey_report.md` and `skill_library/README.md` first; they define the seed taxonomy (HL-1, HL-2, LL-1) and the recovery primitives.

## Context
- The policy is pi0 (`pi0_libero`). It was finetuned on libero_spatial and libero_10 but has never seen libero_90. libero_90 shares scenes with libero_10, and pi0 often runs the libero_10 task of the same scene instead of the instruction. The trained libero_10 tasks are:
  - LIVING_ROOM_SCENE2: put both the alphabet soup and the tomato sauce in the basket / put both the cream cheese box and the butter in the basket
  - KITCHEN_SCENE3: turn on the stove and put the moka pot on it
  - KITCHEN_SCENE4: put the black bowl in the bottom drawer of the cabinet and close it
  - LIVING_ROOM_SCENE5: put the white mug on the left plate and put the yellow and white mug on the right plate
  - STUDY_SCENE1: pick up the book and place it in the back compartment of the caddy
  - LIVING_ROOM_SCENE6: put the white mug on the plate and put the chocolate pudding to the right of the plate
  - LIVING_ROOM_SCENE1: put both the alphabet soup and the cream cheese box in the basket
  - KITCHEN_SCENE8: put both moka pots on the stove
  - KITCHEN_SCENE6: put the yellow and white mug in the microwave and close it
  - libero_spatial (all): pick up the black bowl ... and place it on the plate
- The survey feeds a failure recovery skill library. **High-level (HL)** failures are reasoning errors: the wrong object, wrong destination, wrong skill or fixture, a wrong spatial relation, wrong subtask order, or idling/never starting the next subtask. **Low-level (LL)** failures are fine motor control or fine spatial perception: grasp misalignment, a missed or slipped grasp, a drop, a push or pull that stalls, a knob not turned far enough, an imprecise placement that falls off or lands outside the region, or knocking objects over. HL failures get a proposed remedy. LL failures get an RL subtask later, so **their steps must be recorded precisely** so the scene can be rebuilt from the trace.

## Inputs: video only
- Use only `episodes/<suite>/<folder>/sheetNN.png` (contact sheets, step labelled on each tile) and `episodes/<suite>/<folder>/frames/stepNNNN.png`. These are the sampled frames of the rollout video.
- **Never open anything under `Documentation/pi0_LIBERO_rollouts/*/<folder>/trace/`**. Do not read meta.json either; your batch list gives the instruction. Do not decode extra frames from the .mp4, because the sampling rate (1 fps or 3 fps) is part of the experiment.
- Zoom: `python3 failure_agent_prototype/tools/frame_manipulation/zoom_frames.py <episode_dir> <s1,s2,...> x0,y0,x1,y1 <scale> <out.png>`. It crops the same box from several sampled frames (256x256 px each) and stacks them side by side. Run it from the repo root `/home/DockerShared/ECE780`. Use it whenever a call depends on small detail: gripper contact, drawer flush, knob position, an object on vs. beside its destination. Save the strips you base a call on as `<episode_dir>/evidence_<short_name>.png`.

## Conventions
- step = control step, including 10 wait steps (steps 0-9 are static). The video is 20 fps, one frame per step.
- At 1 fps the frames are 20 steps apart (±20 resolution). At 3 fps they are every 6-7 steps (±7).
- **Camera orientation:** the camera faces the robot, so the robot base is at the top of the image. LIBERO's left/right are from the robot's side: the robot's **left appears on the image right**. "front" (away from the robot) is lower in the image and "back" is toward the robot (top). Keep this in mind for "left plate", "right of the caddy", "back compartment" and similar.
- Subtask vocabulary: `pick(obj)`, `place(obj, dest, relation)` where relation is on/in/under/left_of/right_of/front_of, `open(fixture)`, `close(fixture)`, `turn_on(stove)`, `turn_off(stove)`, `stack(obj, base)`.

## Per episode
1. Look at every contact sheet for the episode. Split the instruction into subtasks and give each an outcome: success, partial, failed, never attempted, or undone.
2. Find each distinct failure. Mark one as `primary`: the earliest root cause of the task failure. Record other independent failures too. Do not record pure downstream consequences, such as "bowl never placed because it was never picked"; note those in the subtask outcome instead.
3. For each failure:
   - `level`: high or low.
   - `mode_id`: an existing seed ID (HL-1, HL-2, LL-1) if it truly fits; otherwise `NEW`.
   - `mode_name`: a generic snake_case name, with no episode-specific objects (e.g. `grasp_slip_during_lift`, not `ketchup_drops`).
   - `mode_definition`: one generic sentence.
   - `onset_step` and `onset_window`: [last sampled step where things still look fine, first sampled step where the failure is evident].
   - For **LL** failures:
     - `rl_reset_step`: the last sampled step *before* the failing attempt begins. This is the scene to restore for RL, e.g. gripper positioned above the object before the bad grasp.
     - `rl_reset_frame` (path of that frame).
     - `rl_subtask`: the subtask the RL policy should learn.
     - `rl_success_check`: a short text predicate.
   - For **HL** failures: `hl_remedy`, a short ordered recovery using the primitives HALT, RETREAT, REPROMPT, INSERT_SUBTASK, VERIFY, WATCH, INVOKE_RL, ESCALATE, plus a one-line `prevention` for the coordinator. pi0 re-prompts must be full LIBERO-style sentences such as "pick up the X and place it on the Y", never bare verbs.
   - `observation`: what you saw, with step numbers.
   - `confidence`: high, medium or low.
   - `evidence`: the evidence file name, or null.
4. Write `episodes/<suite>/<folder>/survey.json` in exactly this shape:

```json
{
  "episode": "libero_90/task6_fail_init0",
  "task_language": "...",
  "survey_fps": 1,
  "step_resolution": 20,
  "subtasks": [{"name": "open(bottom_drawer)", "outcome": "failed: ..."}],
  "behaviour_summary": "one or two sentences: what pi0 actually did over the episode",
  "runs_trained_task": "the libero_10/libero_spatial task pi0 appears to be executing instead, or null",
  "failures": [{
    "primary": true,
    "level": "low",
    "mode_id": "NEW",
    "mode_name": "drawer_pull_slip",
    "mode_definition": "...",
    "subtask": "open(bottom_drawer)",
    "onset_step": 120,
    "onset_window": [100, 120],
    "rl_reset_step": 80,
    "rl_reset_frame": "frames/step0080.png",
    "rl_subtask": "open(bottom_drawer)",
    "rl_success_check": "bottom drawer open by more than half its travel",
    "hl_remedy": null,
    "prevention": null,
    "observation": "...",
    "confidence": "medium",
    "evidence": "evidence_drawer_pull.png"
  }],
  "notes": "uncertainties, anything odd"
}
```

Stay faithful to what is visible. If you can't tell whether something was a grasp miss or a never-attempted grasp, say so and lower the confidence; don't invent detail.
