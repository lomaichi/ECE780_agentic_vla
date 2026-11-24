# Pixel-Q V2: Code Instructions

How to run the code for [pixel_Q_RL_V2.md](pixel_Q_RL_V2.md) and how to read what it produces. Design reasons
are in pixel_Q_RL_V2.md ("§" refers to it).

## 1. What V2 does

pi0 drives the robot from a task's init state. Every **tick** (4 control steps at 20 Hz = 5 fps) the code checks
a **gate**: the subtask is still `pick`, the gripper is open and empty, and the target is within reach
(3 cells = 6 cm, + 4 cm). Outside the gate pi0 just runs.

At a gated tick the **Q-map** reads an 88×88 heightmap centred on the gripper (22 cm, rotated with its yaw) and
gives, for each of 5 ensemble heads, one value per action:

- **action**: move to one of the 7×7 cells (2 cm) within reach at the current height, plus a yaw change of
  −20° … +20° (9 bins). 441 actions.
- **no-op**: the centre cell at 0°. Its value **Q0** = value of letting pi0 continue.
- **value** = discounted pick success within H_max = 3 s (label 0.85^seconds if picked, else 0).

**Trigger**: `A(s) = max_corrections [mean − κ·std] − Q0` (κ = 1). Fire when A(s) > **τ** on 3 consecutive gated
ticks, then a 3 s cooldown. τ is how much better than "let pi0 continue" a correction must look; we use
**τ = 0.2** for now (`--tau 0.2`).

**Labels come from branch points.** At a random gated tick (p = 0.1) the sim is snapshotted and two futures run
from the same state; then the main rollout continues as if nothing happened:

```
main rollout ──tick──[branch point]──tick── ...
                        ├─ no-op branch:        pi0 alone, 3 s          → y_noop  (trains Q0)
                        └─ intervention branch: correction a, then pi0  → y_a     (trains Q(a))
```

| Phase | What it does |
|---|---|
| 0 | Levelling check: is "do nothing" ≈ "only level the gripper"? Collects no-op labels for phase 1 |
| 1 | Trains the Q-map from branch points (ε-greedy corrections, ε 1.0 → 0.1). Trigger off |
| 2 | Fine-tunes with the trigger acting on the main rollout; branching continues. Needs τ |
| evaluate | pi0 / pi0+HL / pi0+HL+trigger / pi0+HL+random trigger on the eval starts |

**HL** = high-level recovery (e.g. pi0 heading for the wrong object). Training uses a privileged stand-in from
sim state; `--hl moondream` uses the real Moondream2 agent (evaluation only).

## 2. Files

| File | Role |
|---|---|
| `train.py` | **Entry point**: all modes (`--mode`), actors + learner, logs, plots |
| `rollout.py` | One main rollout: ticks, gate, HL, branch points, trigger, cooldowns |
| `libero_env.py` | LIBERO backend: pi0 client, gate, heightmap, correction primitive, snapshot/restore, videos |
| `hl.py` | HL modules: `PrivilegedHL` (training), `MoondreamHL` (real agent), `NoHL` |
| `heightmap.py` | Map geometry, rasterization, clearance height |
| `qnet.py` | Ensemble FCN, action encoding, A(s) |
| `replay.py`, `evaluation.py`, `viz.py` | Replay buffer; metrics and reports; PNGs and plots |
| `test_v2.py` | Unit tests (no LIBERO, no pi0) |

## 3. Setup

- **Python**: `/opt/openpi/examples/libero/.venv/bin/python` (not the container's default `python`, which has no torch).
- **Device**: leave `--device cpu` (default). This venv's torch can't use the RTX 5090; simulation and pi0
  dominate the time anyway.
- **pi0 server** (needed by everything except `inspect` / `restore-check`):

```bash
cd /opt/openpi && OPENPI_DATA_HOME=/home/DockerShared/.cache/openpi .venv/bin/python scripts/serve_policy.py \
    policy:checkpoint --policy.config=pi0_libero --policy.dir=gs://openpi-assets/checkpoints/pi0_libero
```

Ready in ~1 min; one server serves all workers.

## 4. How to run

V2 has **not been run in LIBERO yet**: do steps 1–2 first, they check the simulator side without pi0.

```bash
cd /home/DockerShared/ECE780/failure_agent_prototype/RL/pixel_Q/V2
PY=/opt/openpi/examples/libero/.venv/bin/python
OUT=runs/pick_v2

# 1. Heightmap geometry, no pi0 -> inspect/*.png + printed table (section 5.6)
$PY train.py --mode inspect --out-dir runs/inspect --train-inits 0 --eval-inits 40 --inspect-max 6

# 2. Snapshot / restore must be exact, no pi0 (PASS / FAIL per start)
$PY train.py --mode restore-check --out-dir runs/restore --train-inits 0 --eval-inits 40

# 3. Phase 0 (~400 probes); read phase0_report.json before going on
$PY train.py --mode train --phase 0 --out-dir $OUT

# 4. Phase 1 (3000 branch points; add --resume to continue after a crash / Ctrl-C)
$PY train.py --mode train --phase 1 --out-dir $OUT --video --wandb-project pixelQ_V2_p1

# 5. Phase 2
$PY train.py --mode train --phase 2 --out-dir $OUT --tau 0.2 --branch-points 1000 --video --wandb-project pixelQ_V2_p2

# 6. Evaluate on chosen tasks / inits, with videos
$PY train.py --mode evaluate --out-dir $OUT --tau 0.2 --video --media-n 4 \
    --eval-tasks libero_90:10,14 libero_spatial:3 --eval-inits 40 41 42

# 7. Trigger timing on the same starts
$PY train.py --mode timing --out-dir $OUT --tau 0.2 --eval-tasks libero_90:10,14 --eval-inits 40 41 42

# optional: Q0 reliability, noise floor, tau sweep
$PY train.py --mode calibrate --out-dir $OUT --calib-repeats 4 --sweep-repeats 2
```

Re-running a mode into the same `--out-dir` appends to its `.jsonl` logs.

### Tasks and init states

| Flag | Meaning | Default |
|---|---|---|
| `--tasks` | Training tasks, `<suite>:<ids>` (ids `9,10`, `0-9`, `1,3-5`); several specs allowed | `libero_90:9,10,14,17,25,31,34,37,63,64,67,71,72,85` `libero_spatial:0-9` |
| `--eval-tasks` | Eval tasks, same syntax | same as `--tasks` |
| `--train-inits` / `--eval-inits` | Init-state indices | `0 … 39` / `40 41` |
| `--eval-repeats` | Rollouts per eval start (pi0 is stochastic) | 1 |

- Eval starts = `--eval-tasks` × `--eval-inits`, tagged like `libero_90_task10_init40`.
- Held-out LIBERO-90 tasks are refused in `train` and allowed in `evaluate`, `eval-branches`, `calibrate`, `timing`.
- Only bowl / mug / cup targets are eligible; others print `skip <tag>: ...`.

### Videos and Q-map images

`--video` records the first `--media-n` eval starts (default 3). Videos are wrist | agentview at 20 fps; only the
main rollout is shown (branch frames are cut).

| Mode | `videos/` | `qmaps/` |
|---|---|---|
| `train` (each eval, every `--eval-every` = 250 branch points) | `eval_bp<N>_<start>.mp4` | `eval_bp<N>_<start>.png` |
| `eval-branches` | `eval_<start>.mp4` | `eval_<start>.png` |
| `evaluate` | `<condition>_<start>.mp4` | — |

Intervention steps are in `rollouts*.jsonl` (`interventions[].step`, 20 steps per second).

### Modes

| Mode | Runs | Writes |
|---|---|---|
| `train --phase 0` | Intervention branch only levels; branches run 10 s to log success times | `probe_log.jsonl`, `phase0_report.json`, `replay_phase0.pkl` |
| `train --phase 1` | Prefills replay from `replay_phase0.pkl`; ε-greedy branch points; evals every 250 | `train.jsonl`, `eval.jsonl`, `branches.jsonl`, `rollouts.jsonl`, `ckpt_latest.pt`, `ckpt_best.pt`, `replay.pkl` |
| `train --phase 2` | As phase 1 plus the trigger on the main rollout; starts from `ckpt_best.pt` | same files |
| `eval-branches` | Greedy branch points of a checkpoint on the eval starts | `eval.jsonl`, Q-maps, videos |
| `calibrate` | Reliability, noise floor, τ sweep | `calibration.json`, `reliability.png`, `tau_sweep.png` |
| `evaluate` | `--conditions pi0 hl trigger random`, no branching | `eval_conditions.json`, `rollouts_evaluate.jsonl`, videos |
| `timing` | A(s) along pi0 + HL rollouts; trigger replayed offline | `timing_report.json`, `timelines.png` |
| `inspect`, `restore-check`, `finger-mask` | Simulator checks, no pi0 | `inspect/`, `restore_check.json`, `finger_mask.npy` |

- **Checkpoint**: evaluation modes use `--ckpt`, else `ckpt_best.pt` (highest policy gain over phases 1–2), else
  `ckpt_latest.pt`.
- **τ**: `--tau`, else `tau_recommended` from `calibration.json`. Phase 1 doesn't use it.
- Other useful flags: `--branch-points` (3000), `--p-branch` (0.1), `--workers` (6), `--h-max-s` (3.0),
  `--reach` (3), `--persist` / `--cooldown-s` (3 / 3.0), `--hl` (`privileged` | `none` | `moondream`),
  `--wandb-project`. `$PY train.py --help` lists all.
- W&B in this venv needs `click` first: `uv pip install --python /opt/openpi/examples/libero/.venv/bin/python click`.

## 5. How to read the outputs

### 5.1 Training printout (one line per round)

```
round 535 phase 1 bp 2996 (2996 this phase) rollouts 2140 grad 11804 eps 0.100 | branch y_noop 0.532 y_a 0.508
  | loss 0.0049 Q0 0.204/0.200 Qa 0.358/0.355 | buf 6394 | pick 1.000 | 1.15 min
```

| Field | Meaning |
|---|---|
| `bp` | Branch points so far (`this phase` counts toward `--branch-points`) |
| `eps` | Chance an intervention branch tries a random correction |
| `branch y_noop / y_a` | This round's mean labels: pi0 alone vs. with the correction |
| `Q0 pred/label`, `Qa pred/label` | Prediction vs. label on the training batches; should track each other |
| `pick` | Pick success of this round's main rollouts |

### 5.2 Eval printout (every 250 branch points, and `eval-branches`)

```
[eval_bp2500] eval: 33 branch points  y_noop 0.113  y_greedy 0.496  adv 0.383  policy gain 0.413  frac A>0 0.939
  Q0 bias -0.039  pick success (pi0+HL) 0.625
```

At each eval branch point the greedy correction (what the trigger would run) is compared with the no-op.

| Field | Meaning | Good |
|---|---|---|
| `y_noop` / `y_greedy` | Mean label: pi0 alone / after the greedy correction | y_greedy > y_noop |
| `adv` | Mean (y_greedy − y_noop): "always correct" | > 0 |
| `policy gain` | Mean gain where A(s) > 0, else 0: "correct only where the Q-map says so". Selects `ckpt_best.pt` | > 0, ≥ adv |
| `frac A>0` | Share of states where the Q-map wants to act | not ~1 |
| `Q0 bias` | Mean (Q0 − y_noop) | ≈ 0 |

With 10–40 branch points per eval these swing a lot; read the trend in `eval.jsonl` (which also has standard
errors and `corr_A_adv`: does a higher A(s) mean a bigger real gain?).

### 5.3 Phase 0 report

Per approach-tilt bin, `diff ± se` = (label after levelling only) − (label of pi0 alone), same snapshot. Near 0
means the no-op assumption holds. A bin with |diff| > 0.05 and > 2 se is flagged: train Q0 only below that tilt,
or use a separate V network (§15). If many no-op successes come after H_max, raise `--h-max-s`.

### 5.4 Calibration

- **`reliability.png`**: branch points binned by predicted Q0; blue = mean actual y_noop, orange = success
  fraction. Calibrated = blue on the diagonal. Bins with few points (counts printed) are noise.
- **Noise floor** (`tau_noise`): 95th percentile of A(s) where a correction can't matter (pi0 succeeds within 1 s,
  or the target is at the gate's edge).
- **`tau_sweep.png`**: success vs. interventions per rollout at τ = ∞ and `tau_noise` + {0 … 0.3}; the knee becomes
  `tau_recommended`. Points with zero interventions are the same condition, so their spread shows the noise.

### 5.5 Q-map PNG

1. **Left**: heightmap colours with mean − κ·std (max over yaw bins) on the reachable cells, red low → green high.
   White ring = gripper's cell (no-op); blue box = the executed correction; black dots = unseen pixels. Columns
   run along the gripper's finger axis; up = the gripper's +y.
2. **Middle**: height (darker = further below the fingers), target mask magenta.
3. **Right**: mean Q for each of the 9 yaw bins; the executed bin's label in blue.

Caption: the branch point's real outcomes (y_noop, y_a), predicted Q0 and A, chosen cell and yaw, colour range.

### 5.6 Inspect

Four scripted poses per start. Each PNG: map channels (rgb | height | mask | observed), agentview with the cell
centres (cyan = reachable, red = gripper's cell), wrist view. Check the cyan dots sit on the target, the mask is on
the bowl / mug, no arm in the heightmap (`robot px dropped` > 0), `observed` > ~0.8.

### 5.7 Timing

- `timelines.png`: A(s) over time per rollout (`ok` / `FAIL`), black line = τ. Ideally below τ on `ok`, above it
  before a `FAIL`.
- `timing_report.json`: `false_trigger_rate` (would fire on rollouts pi0 completes alone),
  `trigger_rate_on_failures`, lead time before failures. Needs both successes and failures among the rollouts.

### 5.8 Evaluate

One line per condition: `pi0` (alone), `hl` (pi0 + HL, the current system), `trigger` (+ learned trigger),
`random` (+ a trigger firing at random at the same rate, same greedy correction: tests whether the timing
matters). Columns: pick / task success, interventions and HL corrections per rollout. Per-start results are in
`eval_conditions.json`.
