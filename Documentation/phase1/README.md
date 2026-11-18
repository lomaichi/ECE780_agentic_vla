# Phase 1: infrastructure (runbook)

Phase 1 builds what Phase 2 (VLM recovery) and Phase 3 (RL skills) both need:

1. **Verified RL reset states**: the 24 "just before the failure" states in `library.json` restore and replay exactly.
2. **Subtask success checks** (`fa_core/predicates.py`): pick, place, open/close, turn on/off, whole task.
3. **The control loop** (`fa_core/loop.py`): pi0 runs as in `record_suites.py`, and an agent can watch at 1 fps and
   override pi0 with scripted motions or change its prompt. Everything is logged in the restorable trace format.

All scripts run in the container's LIBERO client venv. Only step 5 needs the pi0 server.

```bash
cd /home/DockerShared/ECE780/failure_agent_prototype/tools
PY=/opt/openpi/examples/libero/.venv/bin/python
export MUJOCO_GL=egl
```

| Step | Command | Time (rough) | Pass when |
|---|---|---|---|
| 1. API probe | `$PY probe_libero_api.py` | < 1 min | last line `RESULT: OK` |
| 2. Reset states | `$PY scene_restore.py --check --out ../../Documentation/phase1/restore_check` | ~5 min | all `PASS`, and every PNG shows the moment *before* the attempt |
| 3. Subtask checks | `$PY validate_predicates.py` | 10-30 min | Part A 220/220; Part B goal rows agree; survey disagreements explained |
| 4. Loop checks | `$PY check_loop.py replay` then `$PY check_loop.py motion` | ~2 min | both `PASS`; motion video looks right |
| 5. No-op sanity | `run_loop.py --agent noop ...` then `baseline_summary.py --compare` | 30-60 min | `no detectable difference` vs the Phase 0 baseline |

## Step 1: API probe

`fa_core/predicates.py` uses LIBERO internals that no recorded run has exercised yet (`objects_dict`,
`object_states_dict`, `obj_body_id`, `_check_grasp`, `_eval_predicate`, `_check_success`, `parsed_problem`).
The probe calls each one on a real scene and prints the scene's simulator names.
Any `FAIL` stops Phase 1: send the output, and the fix is usually one line in `fa_core/predicates.py`.

## Step 2: reset states (`scene_restore.py --check`)

For each RL reset state: restore it, replay the recorded actions for 20 steps, compare with the recording.

- `PASS`: every joint/object position is within 1 mm / 1 mrad after the replay (`--max-qpos-err`).
- `LOOK`: state OK but the image differs from the video more than usual (`--warn-pix`): open the PNG.
- `FAIL` / `ERROR`: drop or fix that state.

**Also open every PNG** (top row: video at reset | restored | video at the failure onset). The reset steps were read
from video at ±20 steps; if the restored scene is already mid-failure (bowl already slipping, drawer already stalled),
move the step earlier. To change a step, edit `library.json` in **both** places: `rl_policies.<policy>.reset_states`
and the matching `entries[].instances[].rl_reset_step` (the oracle test matches on it). Note what you changed.

If nearly every state FAILs with similar small errors (say 1e-3 to 1e-2), restore itself is imperfect, not the states;
report the numbers instead of loosening the threshold.

## Step 3: subtask checks (`validate_predicates.py`)

No pi0 needed; it writes recorded states back into the sim and evaluates the checks.

- **Part A** recomputes LIBERO's own success on every recorded final state. Must agree on all 220 episodes; any
  mismatch means state restore or predicate access is wrong.
- **Part B, goal rows** (success episodes): `pick` and the LIBERO-equivalent `place` must fire. They should agree
  almost everywhere; a disagreement here is a check bug.
- **Part B, survey rows** (87 surveyed failures): the survey labels are visual, at 1-3 fps, so some disagreement is
  normal. For each one, watch the video around the listed step and decide whether the label or the check is wrong.
  Labels it can't map unambiguously (e.g. two black bowls) are skipped and listed.

Report: `Documentation/phase1/validate_predicates/report.md`.

## Step 4: loop checks (`check_loop.py`)

- `replay`: recorded actions through the loop must reproduce the recorded states exactly (default tolerance 1e-6),
  and the loop's own trace must restore mid-episode. If this fails, the loop is not faithful to `record_suites.py`.
- `motion`: scripted `MoveTo` legs (up 10 cm, +5 cm x, back, down). This is the first real test of the OSC-frame
  assumption copied from `oracle_handoff_test.py`. Watch `Documentation/phase1/check_loop/motion_test/motion_test.mp4`:
  straight up, right, left, down, no tilting. If it drifts sideways or tilts, fix `fa_core/motion.py`
  (POS_GAIN / ROT_GAIN or the rotation frame) before Phase 2.

## Step 5: no-op sanity check

The loop with an agent that never intervenes must perform like plain pi0. Start the server (`./serve_pi0.sh` in
another shell), then:

```bash
$PY run_loop.py --agent noop --out-root ../../Documentation/phase1/noop_heldout \
    --suites libero_90 --split held_out --init-states 0 1 2 3 4 5 6 7 8 9
python3 baseline_summary.py ../../Documentation/phase1/noop_heldout \
    --compare ../../Documentation/pi0_baseline_heldout
```

Pass: `no detectable difference` on `libero_90/held_out`. pi0 samples random noise, so the two rates won't be
identical; the CI on the difference must contain 0. If the loop is lower with a CI excluding 0, something in the loop
changes pi0's inputs: stop and report.

## What to commit

Code is already in the branch. From the runs, commit the small text results:
`Documentation/phase1/restore_check/report.json`, `Documentation/phase1/validate_predicates/report.{md,json}`,
`Documentation/phase1/check_loop/{replay,motion}.json`, `Documentation/phase1/noop_heldout/libero_90/meta.json`,
`Documentation/phase1/noop_heldout/{summary,compare}.*`, and `library.json` if you edited reset states.
Decide as a group about PNGs, videos and traces (the no-op run is ~0.5 GB).

## Code map

| File | What |
|---|---|
| `fa_core/common.py` | constants, paths, pi0 request building, env construction (identical to `record_suites.py`) |
| `fa_core/predicates.py` | `pick(obj)`, `place(obj, dest, on/in)`, `open/close/turn_on/turn_off(target)`, `goal()` |
| `fa_core/motion.py` | `MoveTo`, `Hold`, `ActionSequence`, `Sequence` scripted motions |
| `fa_core/loop.py` | `run_episode` (pi0 loop + agent hook), `CONTINUE / SetPrompt / Override / Stop`, `write_episode` |
| `fa_core/agents.py` | `Agent` base, `NoOpAgent`, `ReplayAgent`, registry (`--agent`) |
| `tools/probe_libero_api.py` | step 1 |
| `tools/scene_restore.py` | step 2 (now with PASS / LOOK / FAIL verdicts and onset frames) |
| `tools/validate_predicates.py` | step 3 |
| `tools/check_loop.py` | step 4 |
| `tools/run_loop.py` | step 5, and later every Phase 2 agent run |
| `tools/baseline_summary.py` | now with `--compare` (paired by task, bootstrap CI of the difference) |

Writing a Phase 2 agent: subclass `fa_core.agents.Agent`, implement `act(view)` returning `CONTINUE`,
`SetPrompt(...)`, `Override(motion)` or `Stop(...)`, and add it to `AGENTS`. `view` holds only the camera images,
proprioception, last action and prompt (the test-time budget). Set `privileged = True` only for oracle experiments.
