# Pixel-Q V4: V-learning code instructions

V4 learns the same value function as V3, but in two separate steps:

1. **`collect.py`** gathers the samples on CPU worker processes and writes them to shard files.
2. **`train_v.py`** trains V from those shards on the GPU.

There is no actor-learner loop. V never steers collection: the data come from pi0 plus random perturbations. So
an online learner adds nothing to the data, and on CPU it slowed collection down. You can train V while
collection is still running, on the shards written so far, and retrain later with another H, gamma or
architecture without new simulation.

Design background: [../V3/plan_the_algo.txt](../V3/plan_the_algo.txt) (sections 6-7) and `../V2/run_recap.txt`.

## 1. What it does

**V(s)** is the value of letting pi0 continue from state s:

- `gamma^(seconds to pick)` if pi0 picks the target within H, else 0.
- gamma = 0.85, so a pick after 1 s is worth 0.85, after 3 s 0.61, after 5 s 0.44.
- H is chosen at training time and can be at most the collection's episode cap (5 s).

s is the 88×88 gripper-centred heightmap (rgb, height, target mask, observed) + 4 proprio numbers (finger gap,
sin / cos tilt, height above the target's top). V is a 5-head ensemble: the mean is the estimate, the spread is
the uncertainty.

**Data come from perturbed branches around pi0's own rollouts:**

```
pi0 main rollout ──tick──tick──[selected tick]──tick── ... ──> pick success / 20 s budget / irrecoverable failure
                                 ├─ ep 0  natural:    pi0 alone from the snapshot, up to 5 s     -> 1 sample
                                 ├─ ep 1  perturbed:  small move, then pi0 alone, up to 5 s     -> 1 sample
                                 ├─ ...
                                 └─ ep 8  perturbed                                              -> 1 sample
                                 (state restored after each episode; the main rollout then continues)
```

- **Tick**: every 4 control steps (5 fps).
- **Gated tick**: a tick where all of these hold:
  - the pick isn't done yet;
  - the gripper is open (last command open, finger gap ≥ 5 cm) and holds nothing;
  - the target is within 10 cm (xy) of the grasp point;
  - the grasp point is at most 10 cm above the target's top (`--gate-max-above-cm`);
  - there was no HL correction in the last 1 s;
  - the target covers ≥ 20 pixels in agentview.
- **Selected tick**: a gated tick picked with probability `--p-select` (0.5).
- **Perturbation**: the robot does an OSC move to pi0's pose + an offset (xy ±3 cm, z −1 … +2 cm, rotation ±10°
  about each gripper axis). pi0's action queue is cleared, then pi0 runs alone. The 5 s clock starts on arrival.
- **Pick success** (`fa_core/predicates.py` `PickCheck`), checked every control step. The pick counts once all
  of these hold for 10 consecutive steps:
  - both fingers touch the target, anywhere on the finger (finger body or pad);
  - the target is lifted ≥ 5 cm above its resting height.
- **Sample**: the heightmap where pi0 takes over + the raw outcome (steps to pick, or none within 5 s). A
  perturbed episode is not saved if the move didn't reach its pose within 40 steps or moved the target > 5 mm.

**Splits** (stored with every sample at collection time):

| Split | What | Used for |
|---|---|---|
| train | 14 training-split LIBERO-90 bowl/mug tasks, 85% of their (task, init) tags | training V |
| val | the other 15% of those tags (fixed hash) | picking the best epoch |
| eval | 6 held-out bowl/mug tasks: 13, 15, 24, 36, 65, 76 (unseen tasks) | test metrics only |

Held-out tasks can never be training tasks (the start loader refuses them). Eval rollouts are scheduled so that
eval samples stay at about 3/7 of all samples (7:3 train : eval tasks) until `--eval-target-samples`.

## 2. Files

| File | Role |
|---|---|
| `collect.py` | **Step 1, entry point**: worker processes run rollouts and write shards; progress to stdout + W&B |
| `train_v.py` | **Step 2, entry point**: trains V from a shard folder on the GPU, plus baselines; metrics + plots |
| `episodes.py` | One rollout: ticks, gate, selection, natural + perturbed episodes, video rendering |
| `data_io.py` | Shard format (`data/shard_*.npz`), labels, train/val split |
| `vnet.py` | V network (`attn` attention pooling, `cnn` mean + max pooling, `proprio` baseline), checkpoint save / load |

V4 uses V2's LIBERO backend unchanged (`../V2/libero_env.py`, `heightmap.py`, `hl.py`, `qnet.py` constants) and
the shared pick check in `fa_core/predicates.py`.

## 3. Setup

Everything runs inside the LIBERO container (`podman exec -it robomomo bash`). The repo is mounted at
`/home/DockerShared`. Each step has its own Python:

| Step | Python | Why |
|---|---|---|
| `collect.py` | `/opt/openpi/examples/libero/.venv/bin/python` (py3.8) | has LIBERO + openpi-client. Its torch (1.11 cu113) can't use the RTX 5090, but collection needs no GPU torch: physics is CPU, rendering is EGL. |
| `train_v.py` | `/opt/simplevla/.venv/bin/python` (py3.10) | torch 2.7.1 cu128: runs on the RTX 5090 |

**pi0 server** (needed by `collect.py` only; one server serves all workers; ready in about a minute). Capping JAX
at 60% of the GPU leaves room for `train_v.py` to run alongside. With JAX's default 75%, the server holds about
24.7 GB of the 32 GB:

```bash
cd /opt/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 OPENPI_DATA_HOME=/home/DockerShared/.cache/openpi .venv/bin/python scripts/serve_policy.py \
    policy:checkpoint --policy.config=pi0_libero --policy.dir=gs://openpi-assets/checkpoints/pi0_libero
```

**W&B**: on by default (project `pixelQ_V4`). Online mode needs `wandb login` once. Use `--wandb-mode offline`
to log locally, or `--wandb-project ""` to turn it off.

## 4. How to run

```bash
cd /home/DockerShared/ECE780/failure_agent_prototype/RL/pixel_Q/V4
```

### 4.1 Collect (CPU workers + pi0 server)

```bash
PY=/opt/openpi/examples/libero/.venv/bin/python
export MUJOCO_GL=egl

# smoke test first: a few rollouts, every second training rollout recorded, no W&B
$PY collect.py --out-dir runs/v4_smoke --target-samples 30 --eval-target-samples 15 --video-every 5 \
    --wandb-project "pixelQ_V4_smoke"

# full run (12000 training-task samples + 5000 eval-task samples)
$PY collect.py --out-dir runs/v4 --wandb-project pixelQ_V4

# continue after a crash / Ctrl-C / kill (counts come back from the shards; same W&B run)
$PY collect.py --out-dir runs/v4 --wandb-project pixelQ_V4 --resume
```

Long runs: start it under `nohup ... > runs/v4.log 2>&1 &`, or inside `tmux`, so it survives closing the
terminal. Stop it with Ctrl-C or `kill <pid>`: the buffered samples (fewer than ~200) are written to a last
shard, and only the rollouts of the round in progress are lost.

Safety checks:

- A new run refuses an `--out-dir` that already has shards (no silent overwrite). Use `--resume` or a new folder.
- `--resume` refuses collection flags that differ from the run's `config.json` (`--tasks`, inits, tick, gate,
  `--p-select`, perturbation ranges, `--move-cap`, `--h-max-s`, `--hl`, `--val-frac`), so one folder never mixes
  data collected with different settings. The amount flags (`--target-samples`, `--workers`, `--video-every`, …)
  may change.

### 4.2 Train V (GPU)

```bash
PY=/opt/simplevla/.venv/bin/python

# attn V (default) + the proprio-only baseline, labels with H = 5 s
$PY train_v.py --data runs/v4/data --out-dir runs/v4_train

# other settings on the same data, no new simulation
$PY train_v.py --data runs/v4/data --out-dir runs/v4_train_cnn_H3 --arch cnn --h-max-s 3
```

`train_v.py` can run while `collect.py` is still going: it reads the shards that exist when it starts (the last
<200 samples are still in the collector's buffer). It loads all samples onto the GPU, so at about 21k samples
expect roughly 1-1.5 GB plus the model. If it runs out of GPU memory next to the pi0 server, restart the server with
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.6` (sec. 3) or train after collection.

### Flags

`collect.py`, what is collected (fixed for a run, checked by `--resume`):

| Flag | Default | Meaning |
|---|---|---|
| `--tasks` | `libero_90:9,10,14,17,25,31,34,37,63,64,67,71,72,85` | training tasks |
| `--eval-tasks` | `libero_90:13,15,24,36,65,76` | unseen eval tasks (held-out split) |
| `--train-inits` / `--eval-inits` | `0-39` / `0-39` | init states |
| `--tick-steps` | 4 | control steps per tick (4 = 5 fps) |
| `--gate-max-above-cm` | 10 | gate closed while the grasp point is more than this above the target's top (< 0 = off) |
| `--p-select` | 0.5 | chance a gated tick is selected |
| `--n-perturb` | 8 | perturbed episodes per selected tick (+ 1 natural) |
| `--pos-cm` | 3 | xy offset ± (cm) |
| `--dz-cm` | −1 2 | z offset range (cm) |
| `--rot-deg` | 10 | rotation ± about each gripper axis (deg) |
| `--move-cap` | 40 | max control steps for the perturbation move |
| `--h-max-s` | 5.0 | episode cap (s) = the longest label window `train_v.py` can use |
| `--hl` | privileged | HL stand-in on the main rollout (`none` to disable) |
| `--val-frac` | 0.15 | share of training (task, init) tags held out for val |

`collect.py`, how much / infra:

| Flag | Default | Meaning |
|---|---|---|
| `--out-dir` | `runs/v4` | run folder |
| `--resume` | off | continue the run in `--out-dir` |
| `--target-samples` | 15000 | kept training-task samples (train + val) |
| `--eval-target-samples` | 6000 | kept eval-task samples |
| `--workers` | 6 | parallel rollouts (one process each; 0 = in the main process, for debugging) |
| `--video-every` | 10 | record every n-th training rollout (0 = never) |
| `--gamma` | 0.85 | only for the "true V" numbers printed in the videos |
| `--host` / `--port` | `0.0.0.0` / 8000 | pi0 server |
| `--wandb-project` | `pixelQ_V4` | W&B project (`""` = off) |
| `--wandb-entity` / `--wandb-name` / `--wandb-mode` | – / out-dir name / online | W&B |

`train_v.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `runs/v4/data` | shard folder |
| `--out-dir` | `runs/v4_train` | output folder |
| `--h-max-s` | 5.0 | label window H (s), at most the collection's `--h-max-s` |
| `--gamma` | 0.85 | time discount per second |
| `--arch` | attn | `attn`, `cnn` or `proprio` |
| `--no-baselines` | off | skip the proprio-only baseline |
| `--natural-weight` | 1.0 | loss weight of natural vs perturbed samples |
| `--width` / `--heads` | 32 / 5 | network width / ensemble heads |
| `--bootstrap-p` | 0.8 | chance a sample trains a given head |
| `--epochs` / `--batch` | 60 / 64 | training length |
| `--lr` / `--weight-decay` | 3e-4 / 1e-4 | optimizer |
| `--flip-aug` | off | random mirror across the finger axis |
| `--device` | cuda | |

## 5. Outputs

`collect.py` (`--out-dir`):

| Path | Content |
|---|---|
| `data/shard_*.npz` | the samples, ~200+ per shard (written every round once ≥ 200 are buffered, and at exit) |
| `config.json` | the run's flags (`--resume` checks against it) |
| `videos/rollout<n>_<tag>.mp4` | every `--video-every`-th training rollout |
| `wandb/`, `wandb_id.txt` | W&B files; the id lets `--resume` continue the same W&B run |

`train_v.py` (`--out-dir`):

| Path | Content |
|---|---|
| `v_<arch>.pt`, `v_proprio.pt` | best-epoch (lowest val MSE) checkpoints; `vnet.load_ckpt(path)` |
| `metrics.json` | val / test metrics per model (and the constant baseline), test breakdowns (natural / perturbed, tilt bin), calibration |
| `reliability.png` | predicted V vs actual label on test (binned) |
| `curves.png` | train / val MSE per epoch |
| `config_train_v.json` | the flags |

### 5.1 Shard fields

Same format as V3. Per sample:

- **observation at takeover**: `rgb, height, mask, observed, proprio`;
- **outcome**: `t_success` (control steps from takeover to pick, −1 = none within the cap), `cap_steps`, `end`
  (`success`, `timeout`, or a failure name);
- **meta**: `tag`, `split` (train / val / eval), `rollout`, `init`, `tick`, `tick_step`, `episode` (0 = natural),
  `natural`, `d_cm`, `rot_deg` (offset; 0 for natural), `move_steps`, `tilt_deg`, `target_dxy`.

Label: `data_io.label(t_success, h_steps, gamma)`; `train_v.py` recomputes it for its own `--h-max-s` / `--gamma`.

Don't mix in V3 shards. They were labelled with the old pad-only grasp check, which missed deep grips (bowl in
the air, label 0). `V3/runs/v3` also used other collection settings (H 3 s, ±1.5 cm, 5 perturbations).

## 6. How to read the outputs

### 6.1 collect.py printout

One line per round (a round = `--workers` rollouts):

```
rollouts 78 | samples 3360 (train 1650 val 280 eval 1430) | picked natural 0.24 perturbed 0.27 | 37 rollouts/h 1600 samples/h | 2.10 h, ETA 11.0 h
```

(The numbers are illustrative.)

`samples` are counted since the start of the run (restored from the shards on `--resume`); rates, hours and ETA
are since this (re)start.

### 6.2 W&B (`collect/*`, x-axis `rollouts`)

| Metric | Meaning |
|---|---|
| `samples_total` | all kept samples so far (train + val + eval) |
| `samples_train` / `samples_val` / `samples_eval` | per split |
| `samples_train_tasks` | train + val, the count `--target-samples` refers to |
| `progress_train_tasks` / `progress_eval_tasks` | fraction of each target reached |
| `samples_natural` / `samples_perturbed` | by episode type |
| `picked_frac_natural` / `picked_frac_perturbed` | share of kept samples where pi0 picked within the cap |
| `shards` | shard files written |
| `rollouts_per_hour`, `samples_per_hour`, `eta_hours`, `hours` | speed and time left (since this (re)start) |
| `samples_per_rollout`, `gated_per_rollout`, `selected_per_rollout` | yield per rollout |
| `arrival_rate` / `bump_rate` | share of perturbation moves that reached the pose / moved the target (both are dropped) |
| `main_pick_rate` | share of main rollouts where pi0 picked |
| `rollouts_skipped` | starts skipped as ineligible (e.g. target not visible) |

Look for: `arrival_rate` high, `bump_rate` low, `picked_frac_perturbed` somewhat below `picked_frac_natural`,
`eta_hours` settling after the first rounds.

### 6.3 train_v.py

Per model, it prints the split sizes with label means, then `epoch … train … val … best …` every
`--print-every` epochs, and finally a test table (eval tasks):

| Column | Meaning | Look for |
|---|---|---|
| MSE | error of the ensemble mean vs the label | below `constant` and `proprio` |
| R2 | 1 − MSE / MSE of predicting the train mean | > 0, above `proprio` |
| bias | mean V − mean label (> 0 = optimistic) | near 0 |
| AUC | ranks samples with a pick above those without | well above 0.5 |
| spearman | rank correlation of V and label | |
| sd, corr(sd,\|e\|) | ensemble spread, and whether it tracks the error | corr > 0 |

`metrics.json` has the same numbers for val and test, the natural / perturbed and tilt breakdowns, and the
calibration bins behind `reliability.png`.

### 6.4 Videos

Each video is one training rollout, wrist | agentview at 2×, 20 fps, with a banner:

- **Main rollout**: time since the first gated tick (budget 20 s), gated / selected counts.
- **Every tick once gating starts**: a short pause with `TICK #k SELECTED -> 1 natural + 8 perturbed episodes`
  (green), `not selected` (grey), or `not gated: <reason>` (red, e.g. `gripper not open` = pi0 is closing the
  gripper).
- **Selected tick**: the 9 episodes in sequence. Each shows the offset (`PERTURBED d = … rot = …`, plus
  `[not reached: dropped]` / `[bumped: dropped]`), the MOVING phase, the pi0 timer up to 5 s, and the true V when
  the episode ends. Then `back to the natural rollout` with the episodes' true V.
- The `V_est` lines read `n/a`: collection has no V.
