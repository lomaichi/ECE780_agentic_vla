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
| `Qmax issued` | `q_issued`: set once the argmax(Q) move is issued (V-low or timeout); V is then not checked again in that pick subtask (branch `q_issued`) and pi0 carries on from the Qmax pose. Reset when a new pick subtask starts |
| `continue(pi0)` | `CONTINUE`; pi0 replans from a fresh chunk after every override |
| `VLM.HL`, `execute(VLM.HL_correction)` | a confirmed high-level failure at this call; the VLM agent's `Override` (point_approach / carry_to / release_and_retreat) passed through |

## How the ambiguous parts were read

- **timer2 and `low_success_count`.** Taken literally, the pseudocode restarts timer2 and zeroes the count on every
  iteration, so neither could ever fire. Here timer2 starts when a pick(bowl/mug) subtask becomes current. That includes
  the VLM going back to the pick after a lost grasp. timer2 restarts after a timeout correction. The count starts at 0 at
  the same points.
- **"Consecutive" frames.** The count resets on any frame where V is not evaluated: not close to the object, an HL
  failure, or a correction.
- **After Qmax.** Once the argmax(Q) move is issued, V is not evaluated again for the rest of that pick subtask: pi0
  carries on from the Qmax pose. A new pick subtask (including going back to the pick after a lost grasp) re-arms V.
  The timeout and HL branches still run. A Q correction skipped before its argmax move (target not located / not
  visible at the handoff) does not count as issued.
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
- **Task success.** LIBERO's success check (the env's `done`, i.e. the BDDL goal) ends the episode and is the
  reported success. The VLM never judges the whole task: it judges subtasks only (pick, place). Its agent is built
  with `task_done_stop=False` (`judge_agent.GroundedSuccessJudge`), so a place judged done is recorded as the
  subtask's `done_step` (cleared again if a later call judges it not done) and pi0 continues until LIBERO succeeds or
  the 30 s budget runs out. (Before 2026-11-24 the VLM still made a "task complete" call after two consecutive
  place successes. The framework only logged it, but the recovery agent skipped HL detection on those calls.)
- **Approach path.** V1's three handoff legs are kept. A third leg that repeats the second leg's goal is dropped (same
  end pose). A stalled travel leg otherwise cost another 60 steps.
- **HL correction travel height** (2026-11-25, `components.ClearanceRecoveryAgent`, constants `HL_*` in `config.py`).
  The VLM agent's corrections (point_approach, release_and_retreat, carry_to) go straight up, across and straight
  down. moondreamV2 set the travel height from the start and goal points only (carry_to: destination + 18 cm). That
  ignored what stands in between, and the carried object hanging below the gripper. In `runs/eval` task 65 (init 0
  and init 2), carry_to swept the red mug (13.5 cm tall) through the other two mugs. The travel height is now also at
  least `obstacle top + 5 cm + what hangs below the EE`, capped at robot base + 45 cm for reach:
  - obstacle top: the highest agentview depth point within 8 cm (11 cm when carrying) of the straight path, taken
    over the current frame and the plan-time frame. The plan-time frame covers what the arm hides now. Robot pixels
    are dropped (self-filter dilated by 2 px), and so are non-geom pixels (the hand's site markers).
  - below the EE: 2 cm for the fingertips. When carrying, the target's height, measured at the plan call from depth
    in its detect box (within 1 cm of the mesh height in 13 held-out scenes). When carrying, current points around
    the EE down to the object's bottom are skipped, because they are the carried object.
  - Replays of the recorded task 65 carry_to states (init 0 and 2): travel raised 11 cm, the other two mugs moved
    0.0–1.4 cm. With the old height in init 2 they moved 5.1 and 14.1 cm (knocked over). Corrections with nothing
    taller on the way keep the old height.

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
- `--raw-pi0`: for each episode, also roll out pi0 alone from the same init, with the same budget and stop rule
  (`<name>_pi0.mp4`, `framework.pi0_raw` in events.json, a summary column with a per-task table).
  `--no-plain-video` deletes the un-annotated framework video. `--label-dirs` names the episode folders
  `<name>_success` / `<name>_fail` after the framework rollout's success.
- `--success pick` (default `task`): the privileged `pick(<target>)` check (both fingers on the target, lifted >= 5 cm,
  for 10 steps) ends the episode and is its success; otherwise it runs to `--full-timeout-s`. LIBERO's goal no longer
  stops it (`task_success` in events.json still records it). It applies to the framework and the `--raw-pi0` rollout
  (`fa_core.loop.run_episode(stop_checks=...)`). Use it while there is no Q / V for place().
- The default 30 s budget (pseudocode) is longer than the 20 s LIBERO-90 budget used for the pi0 baselines. Pass
  `--full-timeout-s 20` for a like-for-like comparison.

## Outputs (`--out`)

| Path | Content |
|---|---|
| `<suite>_task<T>_init<I>/<name>.mp4` | agentview, 20 fps |
| `<name>/<name>_annotated.mp4` | `--video`: wrist + agentview with VLM subtask, timer2, V_est and the low-V count, gate status, running correction (HL: with its travel z), privileged ground truth |
| `<name>/<name>_pi0.mp4` | `--raw-pi0`: agentview of pi0 alone from the same init |
| `<name>/events.json` | loop events; `framework` (success, report); `corrections` (Q: trigger, Q_INITIAL_POSITION, approach result, Q-map, cell / yaw, move result, library tag; HL: mode, library entry, action, `travel`: z, obstacle top, below-EE, cap); `calls` (every supervisor call: branch, gate, V, timer2, VLM verdict); `vlm` (plan, subtasks, HL failures); `config` |
| `<name>/trace/*.npy` | `fa_core.loop` trace, restorable with `tools/scene_restore.py` |
| `summary.md`, `summary.json` | one row per episode: success (by `--success`), Q corrections (V-low / timeout / skipped), HL corrections by library entry, V evaluations |

## Files

| File | Role |
|---|---|
| `supervisor.py` | `FrameworkAgent`: the pseudocode's decision logic (no torch / LIBERO imports) |
| `components.py` | module loading (V1's `qnet` isolated from V2's), `QModel`, `VModel` / `ConstantV`, `VLMMonitor`, `ClearanceRecoveryAgent` (obstacle-aware HL travel), `Perception` (V2 / V1 observers), `SkillLibrary` |
| `run_pipeline.py` | entry point: episodes, outputs, annotated video, summary |
| `stubs.py` | offline test doubles: `ScriptedPi0`, `OracleDetectVLM` |
| `config.py` | pseudocode constants, additions, default paths |
| `test_supervisor.py` | unit tests of every branch |

## Status

- V4 is trained (`../RL/pixel_Q/V4/runs/v4_train/v_attn.pt`).
- First real run (2026-11-24, pi0 server + Moondream2 + V4 + Q-map, libero_90 task 76 init 0,
  `runs/full_v4_task76_fixed/`): no success. pi0 grasped the book; `wrong_grasp` (HL-3/4) released it and approached
  the mug at step 86 (~5.7 s); timer2 ran out 1 s after that, so the timeout Q fired (step 212); pi0 then picked the
  mug (GT step 331). In the place phase `carry_to` (HL-6, step 402) started from pi0's 1.29 m carry height, could
  not hold the orientation travelling there toward the base (0.78 rad error) and the mug slipped out. V4 was never
  evaluated: before the HL correction pi0 was 10-17 cm from the mug, after the Q correction the gripper closed at
  once. An earlier run (`runs/full_v4_task76/`) had no pick subtask at all: Moondream2's parser did not read "to
  the right of" (fixed, see the moondreamV2 README).
- `runs/eval_v1/` (2026-11-24, tasks 13 15 24 36 65 76 × inits 0 1, after the VLM stopped judging task
  success): 3/12 (15 init0/1, 24 init0), against 4/12 in `runs/eval_v0/`. Every success ended at LIBERO's goal step.
  The VLM did not judge any place done in either run, so its old "task complete" call never fired. The difference
  comes from pi0's stochasticity: task 24 init1 succeeded in v0, while in v1 the bowl was released outside the drawer
  after the pick (VLM pick lost at step 441). Then a `stall` point_approach (HL-1/S1) went to a bad target point
  142 cm away (step 533): a depth/box localisation error to look into.
- `runs/heldout_bowl_mug/` (2026-11-25): every held-out pick-and-place task with a bowl or mug (libero_90 2 12 13
  15 16 24 29 30 36 43 60 65 66 68 76 84), inits 0-5, default parameters, obstacle-aware HL travel, `--raw-pi0`.
  Framework 31/96, pi0 alone (same init, 30 s budget, stop rule) 22/96: the framework fixed 16 and broke 7. By task
  (framework / pi0): 2 5/0, 12 1/0, 15 4/0, 24 6/6, 29 1/0, 43 3/6, 60 2/4, 66 3/0, 68 5/6, 76 1/0, and 0/0 for 13
  16 30 36 65 84.
  - HL travel: 130 corrections, 39 raised by more than 1 cm, 3 capped. Over the 119 that completed, no non-held
    object moved more than 2 cm, except the book released by a task 76 wrong_grasp.
  - Task 65 never reached the place phase, so no carry_to ran. In all 6 inits pi0 first grasped the white mug
    (wrong_grasp, a correct call). The red mug was picked in 2 inits only, once lying on its side; the VLM judged
    neither pick done.
  - Of the 57 wrong_grasp calls, 49 were real and 8 were false positives, all in task 60 ("black bowl on the
    left"): the VLM target point sits on the other bowl, so pi0's correct grasp is released. That starts 3 of the 7
    breaks (60 init 0/3/5). In task 43 (3 breaks) pi0 alone finishes in 7-9 s; the framework episodes had a
    correction by step 54-186 (V-low Q, carry_to, grasp_loss release) and never placed the bowl. 68 init 0 began
    with a real wrong grasp.
- `test_supervisor.py`: 21 tests pass.

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
- **V-low frequency.** Since 2026-11-24 a V-low Q correction fires at most once per pick subtask (V is not checked
  after Qmax is issued). Before that, a pessimistic V could repeat it every ~1.4 s of pi0 time. A further Q correction
  in the same subtask can still come from the timeout.
