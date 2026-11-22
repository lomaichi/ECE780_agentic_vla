# Pixel-Q V3: V-learning code instructions

How to run the V3 value-learning code and how to read what it produces. Design decisions are in
[plan_the_algo.txt](plan_the_algo.txt) (sections 6-7); the discussion behind them is in `../V2/run_recap.txt`.

## 1. What it does

**V(s)** = the value of letting pi0 continue from state s: `gamma^(seconds to pick)` if pi0 picks the target
within H = 3 s, else 0 (gamma = 0.85, so a pick after 1 s is worth 0.85, after 3 s 0.61). s is the 88×88
gripper-centred heightmap (rgb, height, target mask, observed) + 4 proprio numbers (finger gap, sin / cos tilt,
height above the target's top). V is a 5-head ensemble (mean = estimate, spread = uncertainty).

**Data comes from perturbed branches around pi0's own rollouts:**

```
pi0 main rollout ──tick──tick──[selected tick]──tick── ... ──> pick success / 20 s budget / irrecoverable failure
                                 ├─ ep 0  natural:    pi0 alone from the snapshot, up to 3 s     -> 1 sample
                                 ├─ ep 1  perturbed:  small move, then pi0 alone, up to 3 s     -> 1 sample
                                 ├─ ...
                                 └─ ep 5  perturbed                                              -> 1 sample
                                 (state restored after each episode; the main rollout then continues)
```

- **Tick**: every 4 control steps (5 fps).
- **Gated tick**: the V2 gate holds: (1) the pick isn't done, (2) the gripper is open (last command open,
  finger gap ≥ 5 cm), (3) the target is within 10 cm (xy) of the grasp point, (4) the gripper holds nothing;
  plus no HL correction in the last 3 s and ≥ 20 target pixels in agentview. No height or tilt condition.
  Gating usually lasts ~5-9 ticks per rollout, until pi0 starts closing the gripper.
- **Selected tick**: a gated tick picked with probability `--p-select` (0.5).
- **Perturbation**: an OSC move to pi0's pose + offset (xy ±1.5 cm, z −0.5 … +1.5 cm, rotation ±10° about each
  gripper axis), pi0's action queue cleared, then pi0 alone. The 3 s clock starts on arrival.
- **Sample**: the heightmap where pi0 takes over + the raw outcome (steps to pick, or none). Perturbed episodes
  that didn't reach the pose (30 steps) or moved the target > 5 mm are not saved.

**Actor / learner rounds:** each round, `--workers` actors run one rollout each in parallel; then the learner
takes `--utd` gradient steps per new training sample and the next round starts with the new weights.

**Splits:**

| Split | What | Used for |
|---|---|---|
| train | 14 training-split LIBERO-90 bowl/mug tasks, 85% of their (task, init) tags | training V |
| val | the other 15% of those tags (fixed hash, ~87 of 560) | picking the best V (`v.pt`) |
| eval | 6 held-out bowl/mug tasks: 13, 15, 24, 36, 65, 76 (unseen tasks) | test metrics only |

Held-out tasks can never be training tasks (the start loader refuses them). Eval rollouts are scheduled so eval
samples stay at about 3/7 of all samples (7:3 train : eval tasks) until `--eval-target-samples`.

## 2. Files

| File | Role |
|---|---|
| `train.py` | **Entry point**: actors + online learner, evaluation, wandb, outputs |
| `episodes.py` | One actor rollout: ticks, gate, selection, natural + perturbed episodes, video rendering |
| `data_io.py` | Sample shards (`data/shard_*.npz`), labels, train/val split |
| `vnet.py` | V network (`cnn`; `proprio` baseline), checkpoint save / load |
| `train_v.py` | Offline refit of V from a run's shards (other H ≤ 3 s, gamma, weights, epochs) |

V3 uses V2's LIBERO backend unchanged (`../V2/libero_env.py`, `heightmap.py`, `hl.py`, `qnet.py` constants).

## 3. Setup

- **Python**:
  - `train.py`: the LIBERO venv `/opt/openpi/examples/libero/.venv/bin/python`. Its torch can't use the RTX 5090,
    so the learner runs on CPU (`--device cpu`, default).
  - `train_v.py`: `/opt/simplevla/.venv/bin/python` with `--device cuda` (torch 2.7, GPU).
- **pi0 server** (needed by `train.py`; one server serves all actors; ready in ~1 min):

```bash
cd /opt/openpi && OPENPI_DATA_HOME=/home/DockerShared/.cache/openpi .venv/bin/python scripts/serve_policy.py \
    policy:checkpoint --policy.config=pi0_libero --policy.dir=gs://openpi-assets/checkpoints/pi0_libero
```

- **wandb** (optional): online mode needs `wandb login` once; or pass `--wandb-mode offline`.

## 4. How to run

```bash
cd /home/DockerShared/ECE780/failure_agent_prototype/RL/pixel_Q/V3
PY=/opt/openpi/examples/libero/.venv/bin/python

# smoke test first: a few rollouts, every second training rollout recorded
$PY train.py --out-dir runs/v3_smoke --target-samples 60 --eval-target-samples 30 --workers 2 --video-every 2 \
    --eval-every 2

# full run (10000 training-task samples + 4300 eval-task samples)
$PY train.py --out-dir runs/v3 --wandb-project pixelQ_V3

# continue after a crash / Ctrl-C (reloads the shards and v.pt)
$PY train.py --out-dir runs/v3 --wandb-project pixelQ_V3 --resume

# optional: offline refit from the collected shards (GPU)
/opt/simplevla/.venv/bin/python train_v.py --data runs/v3/data --out-dir runs/v3_offline --device cuda
```

**Time estimate** (from the pilots: ~1.2 s per episode, ~20 kept samples per rollout): ~700 rollouts for both
targets; 6 actors share one pi0 server (inference is serialized), so expect roughly 1.5-2 h. Check
`collect/rollouts_per_hour` in wandb after the first evaluation.

### Flags

Collection:

| Flag | Default | Meaning |
|---|---|---|
| `--tasks` | `libero_90:9,10,14,17,25,31,34,37,63,64,67,71,72,85` | training tasks |
| `--eval-tasks` | `libero_90:13,15,24,36,65,76` | unseen eval tasks (held-out split) |
| `--train-inits` / `--eval-inits` | `0-39` / `0-39` | init states |
| `--tick-steps` | 4 | control steps per tick (4 = 5 fps) |
| `--p-select` | 0.5 | chance a gated tick is selected |
| `--n-perturb` | 5 | perturbed episodes per selected tick (+ 1 natural) |
| `--pos-cm` | 1.5 | xy offset ± (cm) |
| `--dz-cm` | −0.5 1.5 | z offset range (cm) |
| `--rot-deg` | 10 | rotation ± about each gripper axis (deg) |
| `--move-cap` | 30 | max control steps for the perturbation move |
| `--h-max-s` | 3.0 | episode cap and label window (s) |
| `--target-samples` | 10000 | kept training-task samples (train + val) |
| `--eval-target-samples` | 4300 | kept eval-task samples |
| `--workers` | 6 | actors |
| `--hl` | privileged | HL stand-in on the main rollout (`none` to disable) |
| `--video-every` | 50 | record every n-th training rollout (0 = never) |

Learner:

| Flag | Default | Meaning |
|---|---|---|
| `--gamma` | 0.85 | time discount per second in the label |
| `--natural-weight` | 1.0 | loss weight of natural vs perturbed samples |
| `--width` / `--heads` | 32 / 5 | network width / ensemble heads |
| `--bootstrap-p` | 0.8 | chance a sample trains a given head |
| `--lr` / `--weight-decay` / `--batch` | 3e-4 / 1e-4 / 64 | optimizer |
| `--utd` | 0.5 | gradient steps per new training sample |
| `--warmup` | 300 | training samples before the first update |
| `--val-frac` | 0.15 | share of training (task, init) tags held out for val |
| `--eval-every` | 50 | rollouts between evaluations |
| `--device` | cpu | learner device |

Infra: `--out-dir`, `--resume`, `--host` / `--port` (pi0 server, `0.0.0.0` / 8000), `--wandb-project`,
`--wandb-entity`, `--wandb-name`, `--wandb-mode`.

## 5. Outputs

| Path | Content |
|---|---|
| `data/shard_*.npz` | the samples, ~200 per shard (written every ~200 samples and at exit) |
| `config.json` | the run's flags |
| `v.pt` | best V so far (lowest val MSE); `vnet.load_ckpt("v.pt")` |
| `metrics.json` | final test metrics of `v.pt` on the eval tasks + calibration bins |
| `reliability.png` | predicted V vs actual label on the eval tasks (binned) |
| `curves.png` | val MSE and test MSE over rollouts |
| `videos/rollout<n>_<tag>.mp4` | every `--video-every`-th training rollout |
| `wandb/` | only with `--wandb-project` |

`train_v.py` writes `v_cnn.pt`, `v_proprio.pt` (baseline), `metrics.json`, `reliability.png`, `curves.png`,
`config_train_v.json` to its own `--out-dir`.

### 5.1 Shard fields

Per sample: `rgb, height, mask, observed, proprio` (the observation at takeover); `t_success` (control steps from
takeover to pick, −1 = none within the cap), `cap_steps`, `end` (`success`, `timeout`, or a failure name);
`tag`, `split` (train / val / eval), `rollout`, `init`, `tick`, `tick_step`, `episode` (0 = natural),
`natural`, `d_cm`, `rot_deg` (offset; 0 for natural), `move_steps`, `tilt_deg`, `target_dxy`.
Label: `data_io.label(t_success, h_steps, gamma)`.

## 6. How to read the outputs

### 6.1 Printout

One line per round:

```
rollouts 342 | samples train 4120 val 690 eval 2050 | grad 2410 loss 0.0451 | 0.92 h
```

Every `--eval-every` rollouts:

```
  [eval] val MSE 0.0480 (best 0.0471) | test {'all': 'MSE 0.081 R2 0.31 AUC 0.79', 'natural': ..., 'perturbed': ...}
  | natural success 0.41 perturbed 0.29 | gated 7.1 selected 3.5 per rollout | arrival 0.88 bump 0.03
```

| Field | Meaning | Look for |
|---|---|---|
| val MSE | V's error on held-out inits of the training tasks | falls, then flattens |
| test MSE / R² / AUC | V on the unseen eval tasks; R² vs predicting the training mean, AUC = ranks successes above failures | R² > 0, AUC well above 0.5 |
| natural / perturbed success | pick rate of the episodes (perturbed: kept ones) | perturbed somewhat below natural |
| gated / selected per rollout | ticks in the gate / selected for branching | ~5-9 / ~half of it |
| arrival / bump | share of perturbation moves that reached the pose / moved the target | arrival high, bump low (both are dropped) |

### 6.2 metrics.json

`test.all / natural / perturbed`: `n, mse, r2, auc_success, bias` (bias = mean V − mean label; > 0 = optimistic).
`test_calibration`: per predicted-V bin, the mean prediction vs the mean actual label (`reliability.png`).

### 6.3 Videos

Each video is one training rollout, wrist | agentview at 2×, 20 fps, with a banner:

- **Main rollout**: time since the first gated tick (budget 20 s), last V_est, gated / selected counts.
- **Every tick once gating starts**: a short pause with `TICK #k SELECTED -> 1 natural + 5 perturbed episodes`
  (green), `not selected` (grey) or `not gated: <reason>` (red, e.g. `gate closed (open)` = pi0 is closing the
  gripper), and V_est at gated ticks.
- **Selected tick**: the 6 episodes in sequence: offset (PERTURBED d = … rot = …, `[not reached: dropped]` /
  `[bumped: dropped]`), MOVING phase, pi0 timer up to 3 s, V_est at takeover, and the true V when the episode ends;
  then `back to the natural rollout` with the episodes' true V.
- **V_est** comes from the learner's weights at the time the rollout started (untrained before the first update,
  i.e. before `--warmup` training samples).
