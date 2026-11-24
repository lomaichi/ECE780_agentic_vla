# Framework: pi0 + VLM failure agent + V trigger + Q-map correction

This folder implements [psuedocode.txt](psuedocode.txt). It combines four existing pieces in one closed loop:

| Piece | From | Role in the loop |
|---|---|---|
| pi0 | pi0 server (`serve_pi0.sh`), `fa_core.loop.run_episode` | base policy, runs whenever nothing overrides it |
| VLM (Moondream2) | `Documentation/VLM_inference_speed_test/moondreamV2/recovery_agent.py` (`GroundedRecoveryAgent`) | subtask tracking (pick / place), high-level (HL) failure detection and correction, target location (detect box + depth) |
| V (pixelQ V4) | `RL/pixel_Q/V4/vnet.py` checkpoint (`train_v.py`) | low-level trigger: chance that pi0 picks from the current state |
| Q-map (pixelQ V1) | `RL/pixel_Q/V1/runs/pick_v0/ckpt_best.pt` | low-level correction: where (cell) and at what yaw to hand back to pi0 |
| Failure survey | `failure_survey/skill_library/library.json` | every correction is tagged with its library entry: HL-x for VLM corrections, `INVOKE_RL(rl/pick)` (covers LL-4, LL-5, LL-8) for Q corrections |

Nothing that feeds a trained model is re-implemented. V's observation comes from V2's `BranchingPickEnv.gate()` /
`observe()` (the code that produced V4's training samples). Q's comes from V1's `PickCorrectorEnv` handoff geometry and
`_observe()`. Both are wrapped around the pipeline's env, with no reset and no stepping.

## Pseudocode → code

`FrameworkAgent` (`supervisor.py`) is an `fa_core` agent. The loop runs pi0 and calls the agent at 5 fps, and again right
after a scripted motion (override) ends. Each call is one iteration of the pseudocode's `While`:

| Pseudocode | Code |
|---|---|
| `Begin(pi0)` | `loop.run_episode(..., client=pi0)` |
| `Begin(VLM, --mode failure_detection, 5 fps)` | `VLMMonitor.act()` every call (`MONITOR_FPS = 5`) |
| `Begin(VLM, --mode estimate_object_location)` | the VLM's tracked detect box + depth surface point (`VLMMonitor.box()` / `target_point()`) |
| `FULL_TASK_TIMEOUT` (timer1) | the loop's step budget: `--full-timeout-s 30` × 20 Hz |
| `subtask == pick(bowl/mug)` | VLM's current subtask is `pick` and its object is a bowl / mug / cup |
| timer2, `TIMEOUT` | `pick_start`; sim time since the pick subtask (re)started |
| `obs = [l, proprio, height_map]` | V2/V4 observation: 88×88 gripper-centred heightmap (rgb, height, target mask, observed) + 4 proprio. The language `l` enters only through the target mask: V4's net has no text input |
| `VLM.close_to_object()` | V4's gated tick: VLM says pick, gripper open and empty, VLM target point within 10 cm (xy) of the grasp point and at most 10 cm below it, target grounded in agentview, no correction in the last 1 s |
| `V(obs) < THRESHOLD`, `THRESHOLD_COUNT` | `V.predict(obs)` (ensemble mean) < 0.5 on 3 consecutive gated frames |
| `approach(Q_INITIAL_POSITION)` | V1's handoff: open gripper 7 cm above the target's top, over the centre of its mask box, approach axis −z |
| `argmax(Q(obs))` | at the handoff pose: V1 wrist crop → Q-map (9 yaw bins × 10×10 cells) → masked argmax → horizontal move to that cell's surface point with that yaw |
| `continue(pi0)` | `CONTINUE`; pi0 replans from a fresh chunk after every override |
| `VLM.HL`, `execute(VLM.HL_correction)` | a confirmed high-level failure at this call; the VLM agent's `Override` (point_approach / carry_to / release_and_retreat) passed through |

## How the ambiguous parts were read

- **timer2 and `low_success_count`.** Taken literally, the pseudocode restarts timer2 and zeroes the count on every
  iteration, so neither could ever fire. Here timer2 starts when a pick(bowl/mug) subtask becomes current. That includes
  the VLM going back to the pick after a lost grasp. timer2 restarts after a timeout correction. The count starts at 0 at
  the same points.
- **"Consecutive" frames.** The count resets on any frame where V is not evaluated: not close to the object, an HL
  failure, or a correction.
- **Cooldown.** `close_to_object()` includes V4's 1 s cooldown after any correction. V never saw states inside it in
  training.
- **Timeout vs HL.** At the same call, the timeout branch wins over an HL correction, because it sits outside the HL
  check. The VLM agent is told its correction was dropped.
- **Timeout while holding.** The timeout correction opens the gripper. If the gripper is closed on something when timer2
  runs out (pi0 may be lifting the bowl while the pick judge lags), it waits up to 2 s (`TIMEOUT_HELD_GRACE_S`).
- **`target_object_z + 7cm`.** Read as 7 cm above the object's **top**, which is how V1 was trained (its wrist crop
  scale depends on that height).
- **Correction time.** timer2 keeps running during corrections, as in the pseudocode. A slow correction can therefore
  use up most of the 10 s (see the smoke tests below).
- **VLM "task complete".** It is logged, not obeyed, since the pseudocode stops only at the timeout. LIBERO success
  still ends the episode. `--stop-on-vlm-done` obeys it instead.
- **Approach path.** V1's three handoff legs are kept. A third leg that repeats the second leg's goal is dropped (same
  end pose). A stalled travel leg otherwise cost another 60 steps.

## Privileged stand-ins (as in Q and V training)

- Robot pixels are dropped using the sim segmentation of the robot's own geoms (V2 `robot_mask=seg`). On a real robot
  this is the self-filter rendered from the robot's joint state. It covers V's heightmap and the `vlm` target masks
  below.
- The target's sim identity is taken from the BDDL goal. It decides whether the target is "grounded" in agentview and
  supplies the heightmap's `z_top` (both V2 behaviours).
- The target **mask** for V and Q is set by `--target-mask`:
  - `seg` (default): the sim segmentation, exactly as Q and V were trained.
  - `vlm`: the VLM's detect box. For V this is V2's box mode. For `Q_INITIAL_POSITION` and Q's wrist crop it is a
    box-derived depth mask: non-robot box pixels above the box floor; on the wrist view, non-robot points that project
    into the agentview box, between the floor and the object's top + 1 cm.
  - Check against `seg` (`runs/_test/check_masks.py`, 3 scenes, gripper hovering over the bowl):
    `Q_INITIAL_POSITION` agrees within 0.4–1.6 mm, and the wrist-crop masks reach IoU 0.84–0.88 (`vlm` about 8 %
    larger). **Q's argmax differed in all 3 scenes**, so the Q-map is sensitive to the mask shape. Keep `seg` until Q
    is trained or tested on box masks.
- The VLM itself uses only images, proprioception and the RGB-D camera (`GroundedRecoveryAgent`).

## Run

Inside the container (`podman exec -it robomomo bash`), from this folder:

```bash
PY=/opt/openpi/examples/libero/.venv/bin/python

# unit tests of the decision logic (fakes; no sim, no torch)
$PY test_supervisor.py

# offline smoke test: no pi0 server, no GPU. Scripted pi0, sim-box "VLM", constant V. CPU rendering.
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES="" \
  $PY run_pipeline.py --tasks 15 --pi0 stub-hover --vlm oracle --v-const 0.2 --out runs/smoke --video

# real run: pi0 server up, Moondream2 on the GPU (~5 GB next to pi0), trained V
MUJOCO_GL=egl $PY run_pipeline.py --tasks 13 15 24 36 65 76 --inits 0 1 --out runs/eval_v0 --video \
  --v-ckpt ../RL/pixel_Q/V4/runs/v4_train/v_attn.pt
```

- `--pi0 stub-hover | stub-wander | stub-idle` and `--vlm oracle` are privileged test doubles (`stubs.py`). Never
  report results from them.
- `--v-const X` stands in for V until V4 is trained (`X < 0.5` makes every gated streak trigger Q).
- Start `moondream_server.py` yourself before parallel runs (see the moondreamV2 README). Check `nvidia-smi` first:
  while the pi0 server and V4 collection are running, only ~3.5 GB of the GPU is free.
- Other flags: `--threshold`, `--threshold-count`, `--timeout-s`, `--full-timeout-s`, `--approach-dz`,
  `--monitor-fps`, `--cooldown-s`, `--held-grace-s`, `--target-mask`, `--q-ckpt`, `--no-trace`. See `--help`.
- The default 30 s budget (pseudocode) is longer than the 20 s LIBERO-90 budget used for the pi0 baselines. Pass
  `--full-timeout-s 20` for a like-for-like comparison.

## Outputs (`--out`)

| Path | Content |
|---|---|
| `<suite>_task<T>_init<I>/<name>.mp4` | agentview, 20 fps |
| `<name>/<name>_annotated.mp4` | `--video`: wrist + agentview with VLM subtask, timer2, V_est and the low-V count, gate status, running correction, privileged ground truth |
| `<name>/events.json` | loop events; `framework` (success, report); `corrections` (Q: trigger, Q_INITIAL_POSITION, approach result, Q-map, cell / yaw, move result, library tag; HL: mode, library entry, action); `calls` (every supervisor call: branch, gate, V, timer2, VLM verdict); `vlm` (plan, subtasks, HL failures); `config` |
| `<name>/trace/*.npy` | `fa_core.loop` trace, restorable with `tools/scene_restore.py` |
| `summary.md`, `summary.json` | one row per episode: success, Q corrections (V-low / timeout / skipped), HL corrections by library entry, V evaluations |

## Files

| File | Role |
|---|---|
| `supervisor.py` | `FrameworkAgent`: the pseudocode's decision logic (no torch / LIBERO imports) |
| `components.py` | module loading (V1's `qnet` isolated from V2's), `QModel`, `VModel` / `ConstantV`, `VLMMonitor`, `Perception` (V2 / V1 observers), `SkillLibrary` |
| `run_pipeline.py` | entry point: episodes, outputs, annotated video, summary |
| `stubs.py` | offline test doubles: `ScriptedPi0`, `OracleDetectVLM` |
| `config.py` | pseudocode constants, additions, default paths |
| `test_supervisor.py` | unit tests of every branch |

## Status

- V is not trained yet. A V4 checkpoint saved with torch 2.7 (`train_v.py`'s venv) was checked to load and predict in
  the LIBERO venv (torch 1.11) with a randomly initialised net (`runs/_test/v_random_attn.pt`).
- The pipeline has not been run with the real pi0 or Moondream2. Only the offline smoke tests below have run.
- `test_supervisor.py`: 18 tests of every branch pass.

### Offline smoke tests (2026-11-23, `runs/_test/`)

Scripted pi0, sim-box VLM, CPU rendering. Each episode used the full 30 s with no task success, which is expected: the
stubs never grasp.

| Run | Task | What happened |
|---|---|---|
| `hover15` | 15 middle bowl → cabinet top, V = 0.2 | 7 V-low + 2 timeout Q corrections; approach ~30 steps, Q move reached in ~15 steps (4–5 mm), as in V1 training (median 17 steps, 4.5 mm) |
| `hover15_vlmmask` | same, random-init V, `--target-mask vlm` | 8 V-low + 2 timeout Q corrections, all `Q_INITIAL_POSITION`s at the correct height |
| `wander15` | 15, pi0 heads away | 2× HL-7 `target_never_approached` → point_approach, 2 timeout Q corrections |
| `idle24` | 24 bowl → bottom drawer, pi0 still | 2× HL-1 stall → point_approach, then 1 V-low + 2 timeout Q corrections |
| `smoke_hover` | 13 front bowl → plate, V = 0.2 | 3 V-low + 2 timeout Q corrections. The front bowl is at the edge of the Panda's reach (joint 4 at its limit): approach legs and Q moves stalled 0.7–1.8 cm short, so the first correction took 12 s and the timeout fired right after it |

Found and fixed by these runs:
- V1's repeated third handoff leg wasted 60 steps after a stalled travel leg (dropped).
- In `vlm` mask mode the arm inside the detect box became the object's "top" (`Q_INITIAL_POSITION` 5–8 cm too high).
  Fixed by the robot self-filter.

### Open questions for the pseudocode

- **timer2 during corrections.** timer2 keeps running through V-low and HL corrections. A slow correction (task 13:
  12 s) leaves pi0 no time before the timeout correction. Pausing timer2 during corrections, or restarting it after
  them, is a one-line change in `supervisor.py` (`_end_correction`).
- **V-low frequency.** With a pessimistic V, a Q correction can repeat every ~1.4 s of pi0 time (1 s cooldown, then
  3 frames at 5 fps). V1 judged each correction over 5 s of pi0. `--cooldown-s` controls the gap.
