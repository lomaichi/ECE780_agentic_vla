# Failure recovery skill library (v0.1, from the 5-failure PoC)

`library.json` is the machine-readable library. The FR agent loads it at test time. This file is the human-readable version.

## How the FR agent uses it
1. Every 1 fps frame: compute the cheap proprio cues (EE displacement, gripper width) and check the coordinator's **protected states** (`VERIFY`).
2. When an entry's `detect.proprio` cue fires, ask the VLM the matching `detect.visual` question. Use `distinguish_from` to separate look-alike entries.
3. Bind the entry's `slots` from the coordinator's current plan (target object, destination, finished subtask).
4. Run `recovery` top to bottom using only the primitives below, then check `success_check`.
5. On failure, take the entry's `on_not_fired` / `on_fired` branch; when those run out, `ESCALATE`.
6. Log the episode, onset step and outcome under `instances`. An entry becomes `validated` once its recovery works when replayed from a recorded onset state.

Primitives: `HALT`, `RETREAT(dz, dx, dy)`, `REPROMPT(text)`, `INSERT_SUBTASK`, `VERIFY(predicate, roi)`, `WATCH(cue, window)`, `INVOKE_RL(policy)`, `ESCALATE`. All of them use images and proprio only, with no privileged sim positions.

**Prompt rule:** never give pi0 a bare verb ("pick up the bowl"). Use a full LIBERO-style sentence, e.g. `pick up the {obj} and place it on the {dest}`. That is the only phrasing pi0 has seen in its finetuning data.

## Entries

### HL-1 subtask_handoff_stall (high-level)
pi0 finishes one subtask of a compound instruction, then idles.
- **Detect:** EE moves < 1 cm over 3 frames, gripper open, the previous subtask passes VERIFY, and the EE is not moving toward the next target.
- **Recover:** HALT → RETREAT up 10 cm → REPROMPT `pick up the {next_obj} and place it on the {next_dest}` → WATCH 3 s for an approach. Retry once, then ESCALATE.
- **Prevent:** the coordinator issues one subtask at a time.
- **Seen in:** task1_fail_init1 at step ~100.

### HL-2 scene_default_skill_override (high-level)
pi0 runs the scene's habitual skill (pushing the open drawer shut) instead of the instructed one, and may destroy a precondition such as an open destination.
- **Detect:** with the gripper empty, the EE heads for a non-target fixture over 2 frames, or a protected state (destination drawer open) flips to false.
- **Recover:** HALT → RETREAT up 10 cm and 5 cm away from the fixture → if the destination got closed, INSERT_SUBTASK `open the {destination}` → REPROMPT object-first `pick up the {obj} and place it in the {destination}` → WATCH 5 s. Retry once with a 15 cm retreat, then ESCALATE.
- **Prevent:** the coordinator declares protected states, and the FR agent HALTs on the first violation.
- **Seen in:** task2_fail_init0 at step ~44.

### LL-1 drawer_push_stall (low-level → RL)
pi0 pushes the drawer from above. It gets 1/2-2/3 of the way, then rests with fingers down on the drawer front.
- **Detect:** no motion along the closing direction for 2 frames, and a 2x zoom VERIFY says the drawer front is not flush.
- **Recover:** `INVOKE_RL(rl/close_drawer)` once it is trained. Until then: retreat 5 cm, re-prompt `close the {drawer}`. This fallback is expected to be weak because it re-runs the same pi0 behaviour.
- **RL spec:**
  - Reward: the binary LIBERO predicate of libero_90 task0.
  - Reset states: the 4 recorded onset steps plus task0 inits 0/1.
  - Guide demo: libero_10/task3_success_init*. There pi0 closes a drawer with a horizontal push, hand beside the drawer front, at steps ~210-238.
- **Seen in:** task0 init0 (step 160), task0 init1 (180), task1 init0 (160), task2 init0 (136, inside the HL-2 behaviour).

## Adding an entry
Copy an entry and keep the fields: `id, level, name, summary, applies_when, slots, detect{proprio, visual, distinguish_from}, recovery[], prevention, success_check, instances[], status`. Low-level entries also need an `rl_spec`. Before adding a new entry, check whether the failure is an existing entry with different slot values; if so, add an `instance` instead.
