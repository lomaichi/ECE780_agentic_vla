# Pixel-Q Online DQN Prototype: Code Summary and Instructions

Prototype of the `rl/pick` corrector in [pixel_Q_RL.md](pixel_Q_RL.md). An online DQN, interacting with
the LIBERO simulator, learns Q-maps with a fully convolutional network (FCN) over the target object.
pi0 stays frozen.

**What the Q-map values.** Each correction starts from a fixed **handoff pose**. The VLM agent brings
the open gripper 7 cm above the target object, over the centre of its segmentation box, pointing
straight down (−z), and hands over to the Q-map. The Q-map then chooses only:
- a horizontal move above one cell of the object, and
- a yaw rotation (9 bins: 0, or ±5 … ±20° in 5° steps).

z is never part of the action. The single motion primitive is **handing off to pi0**, so `Q(s, cell, k)`
is the value of handing the task to pi0 after moving above that cell and rotating by bin k.

**Episodes and chains.**
- **Episode = one decision.** From the handoff pose, the Q-map acts, pi0 takes over, and the subtask is
  judged within a **5 s timeout**. Success gives reward 1. Timeout means the bowl was not picked up:
  reward 0, and the episode ends.
- **Chain = consecutive episodes on one scene.** After a failed episode, the VLM brings the gripper back
  to the handoff pose and a **new episode starts from the bowl's current state**. That state may be
  pushed, tilted or knocked over by the failed attempt.
- A chain starts from an **upright bowl**. Disturbed bowl states come from the rollouts themselves, with
  no artificial randomization.
- A chain ends on success, on an irrecoverable state, or after `--max-chain` episodes. The next chain
  starts from a fresh upright start.

Since every episode is one decision that ends with its judgement, `Q(s, a)` estimates **the probability
that this handoff succeeds within 5 s**, and the DQN target is the episode's reward.

## 1. Files

| File | Role |
|---|---|
| `qnet.py` | FCN Q-network, action encoding, masked argmax, epsilon-greedy selection |
| `replay.py` | Replay buffer of `(s, a, r, s', done)` with compact observations; save/load |
| `corrector_env.py` | LIBERO env: start states, handoff pose, segmentation, crop, back-projection, 2D move + yaw, pi0 handoff, timeout judge, chaining |
| `toy_env.py` | Synthetic env with the same interface. Tests the DQN code without LIBERO or pi0 |
| `viz.py` | PNGs: Q-map overlays, and inspect views (wrist view: mask, crop box, cell back-projection; agentview: cross-camera check) |
| `train_dqn.py` | Entry point. Modes `train`, `eval`, `baseline`, `inspect`. Actor pool + learner |
| `interactive_handoff.py` | Standalone tool, not used by training: pi0 drives a LIBERO-90 task, and you can take over the gripper from the keyboard in a browser page (`http://localhost:8765`) and hand it back |
| `archive/` | The earlier oracle handoff test (`oracle_handoff_test.py`), its outputs and the go/no-go proposal |

Reused from `fa_core/`:
- `common`: env construction, task init states, `pi0_request`, `upright`, held-out split, control constants
- `camera`: depth render, intrinsics, `project`
- `motion.MoveTo` / `motion.Sequence`: closed-loop OSC moves (each `MoveTo` capped at 60 steps)
- `predicates`: `PickCheck` (pick success), goal checks and object lookup for the target
- `tools/scene_restore`: restore library reset states

## 2. State, action and network

**Observation** (`corrector_env._observe`, taken at the handoff pose): one dict, which the replay buffer also stores.

| Key | Shape / type | Content |
|---|---|---|
| `rgb` | 80×80×3 uint8 | Crop of the upright **wrist camera** (`robot0_eye_in_hand`, first person), centred on the object's segmentation box in that view. It spans a 150×150 box of the 256 px wrist image (`--wrist-box`): the wrist view is rendered at 137 px and an 80×80 window is cut, so pixel ↔ 3D geometry stays exact. Pixels outside the mask are grey (128) |
| `depth` | 80×80 float16 | (depth − median object depth) / 0.1 m, clipped to ±1.5 |
| `mask` | 80×80 bool | Target object |
| `proprio` | 6 float32 | EE − object centre (world xyz, in dm), gripper opening (finger gap / 8 cm, clipped to 0..1.5), EE yaw sin/cos |
| `valid` | 10×10 bool | Cells the Q-map may pick: all 100 cells of the crop by default. `--min-cell-frac f > 0` keeps only cells with ≥ f object pixels |

**Views.** The Q-map's mask, crop and cells are all on the wrist view, taken at the handoff pose. Only
the handoff move (the VLM agent's stand-in) and the start's visibility check use the third-person
agentview, since before the handoff the object need not be in the wrist view. pi0 gets both views as always.
At the handoff the wrist camera looks straight down from ~17 cm above the object's top (it sits 4.9 cm
beside and 9.7 cm above the gripper point), so the object appears below the image centre, next to the
hand. Its box is ~110×100 px of the 256 px image; one 8 px cell of the crop is ~2 cm on the object.

The 80×80 crop is **object-centred**, and Q is learnt over **all of it**, not only the cells on the object.
Cell indices therefore mean "where relative to the object"; cells off the object are legal actions too.

**Action:** flat index `a = k·100 + row·10 + col`.
- `k` ∈ 0..8 selects a rotation bin, yaw offset `ROT_DEG = −20, −15, …, +20°`.
- `(row, col)` selects one of the 10×10 cells, each an 8×8-pixel window of the crop.
- Every cell is valid by default, with every rotation bin allowed (900 actions in total).

**FCN** (`qnet.build_qnet`, ~0.76 M parameters at `--width 32`):
- Input: 11 channels (rgb 3 + depth 1 + mask 1 + proprio 6 tiled over the crop).
- Three stride-2 conv blocks reduce 80 → 10, so each output cell lines up with exactly one 8×8 window.
- Two dilated conv blocks follow, so the receptive field covers the whole crop.
- A 1×1 head concatenates per-cell features with globally pooled context.
- Output: **(9, 10, 10)**, one Q-map per rotation bin.
- Normalization is GroupNorm, so single-sample actor inference behaves the same as batched training.

## 3. Handoff pose, episode and chain

**Handoff move** (`PickCorrectorEnv._handoff`). This stands in for the VLM agent. It runs at the start
of a chain and again before every later episode.
1. Take the centre pixel of the object's 2D segmentation box. Find its (x, y) by cutting the camera ray
   through that pixel at the object's median height. The box centre need not lie on the object (e.g. a
   bowl on its side), so the depth at that pixel can't be used directly.
2. The object's top z is the 95th percentile height of its masked depth points. Steps 1–2 use only
   mask, depth and camera geometry, which is what a test-time agent has.
3. Target pose:
   - position: `(box-centre xy, top + --handoff-dz)`, with `--handoff-dz 0.07`. Optional xy jitter
     `--handoff-jitter-cm` (default 0) can stand in for the VLM agent's placement error.
   - orientation: approach axis straight down, current yaw kept.
4. Gripper open. Path (one `MoveTo` per leg, up to 60 steps each): rise to the target height (if the
   gripper is lower) with the current orientation, travel while levelling the gripper, descend.

The handoff fails if it hits an irrecoverable state or a sim error, or if the pick predicate fires
during it. After every successful handoff the pick check is restarted, and the observation is taken.

**One episode** (`PickCorrectorEnv.step`), starting from the handoff pose:
1. Back-project the chosen cell: the mean 3D surface point of its object pixels; for a cell with no
   object pixels, the ray through the cell centre cut at the object's median height (the same rule as
   the handoff's box centre; the cell's own depth may hit the table, the fingers or another object).
   An invalid cell (only possible with `--min-cell-frac > 0`) raises `ValueError`.
2. **2D move + yaw:** one `MoveTo` to that point's (x, y) at the **handoff height**, rotating yaw by
   `ROT_DEG[k]` about world z. Gripper open. The move is capped at 60 steps.
3. **pi0** takes over (5 actions per chunk) until the subtask succeeds or the **judge window** ends. The
   window is `--timeout-s 5` = 100 control steps at 20 Hz, counted from the decision (move + pi0).
4. **Judge:** success = `PickCheck` (grasped and lifted ≥ 5 cm for 10 consecutive steps). Lift is
   measured from the object's resting height, the lowest height seen since this episode's handoff, so
   it also works for a bowl on its side.
   - Success → reward 1, and the chain ends.
   - Timeout → reward 0. The VLM re-does the handoff move, and the next episode starts from the
     object's current state.
   - Irrecoverable → reward 0, and the chain ends. Irrecoverable means fell off the table (> 15 cm below
     its chain-start height), out of reach (> 30 cm in xy from its chain-start position), not visible, a
     failed re-handoff, or a sim error.
   - **Tilted or knocked over is not irrecoverable.** It is a state for the next episode.
5. The actor caps a chain at `--max-chain` episodes (default 5), then starts a fresh upright start.

**Judge stand-in.** At test time a VLM judges the subtask (`fa_core.judge_agent.SubtaskJudgeAgent`, one
`claude -p` call per second). During training the sim predicate stands in for it, as pixel_Q_RL.md's
"Reward source" note allows. Using the VLM here would cost ~5 calls per episode and add its judging
errors to the reward.

**Baseline** (`--mode baseline`): the same chains with no Q-map, on the eval starts. From each handoff
pose, pi0 runs straight away for the same 5 s window. It measures what the Q-map adds over the VLM's
handoff alone.

## 4. Start states (training split only, upright)

Chosen with `--sources` (default: both).

- **`library`**: `rl/pick` reset states from `library.json` (pre-failure states pi0 actually reached;
  the bowl is still standing).
  - The target object comes from the survey's `pick(...)` label, matched to a sim object by name
    tokens. On a tie, the object closest to the gripper wins.
  - LIBERO-90 tasks outside the pixel-Q training list are skipped.
  - Library starts are used for both training and evaluation.
- **`tasks`** (`--tasks`): task init states after the 10 wait steps.
  - The target is the object of the first pick-and-place goal check.
  - Default tasks are the bowl/mug ones: LIBERO-90 `9, 10, 14, 17, 25, 31, 34, 37, 63, 64, 67, 71, 72, 85` and LIBERO-spatial `0–9`.
  - Training uses init states 0–39 (`--train-inits`); evaluation uses 40–41 (`--eval-inits`).
- Each chain starts from one of these, with the bowl upright, followed by the handoff move.
- **Held-out guard:** a LIBERO-90 held-out task stops the run (`SystemExit`).
- **Skips:**
  - `IneligibleStart` drops the start for the rest of the run (the list is kept in the checkpoint): a
    non-round-rim target (name without bowl/mug/cup), an object already grasped, an object not visible
    before the handoff, or no target found (no `pick(...)` label, no matching sim object, no pick goal).
  - `ResetFailed` skips only that chain, when the first handoff move fails or the object is not
    visible at the handoff pose. Such chains are logged and not counted.
- The object's tilt relative to its chain-start (upright) orientation is logged at every decision:
  `obj_tilt_deg` per episode in `episodes.jsonl`, and `decision_tilt_deg` per round in `train.jsonl`. It
  shows how often disturbed states occur.

## 5. Training loop (`train_dqn.train`)

The loop maps to the steps in pixel_Q_RL.md:

```
repeat until --episodes (counted in episodes = decisions):
  one round: each actor (spawn pool, --workers) runs one chain with the current weights,   ❶–❸
      from a training start drawn uniformly (ineligible starts excluded)
      every episode: ε-greedy over all valid cells, ε linear 1.0 → 0.1 over --eps-decay-episodes
  add every episode to the replay buffer as a terminal transition (s, a, r)                ❹
  if |D| ≥ max(--warmup, --batch): do (--utd × new episodes) gradient steps:              ❺–❼
      y = r   (each episode is one decision; the γ / target-network / --double path stays
               in dqn_update for transitions that are not terminal, and is unused here)
      loss = MSE(Q(s, a), y) on the executed entry only (--huber: smooth L1)
      Adam with weight decay, gradient norm clipped at 10
  every --eval-every episodes: greedy chains (--eval-eps 0) on the eval starts, --eval-repeats times
      → eval.jsonl, ckpt_best.pt, Q-map PNGs
  every --ckpt-every episodes, and on exit: ckpt_latest.pt (+ replay.pkl), resume with --resume
```

**Success rates** (in `train.jsonl`, `eval.jsonl` and the eval printout):

| Rate | Meaning |
|---|---|
| `episode_success` | Fraction of all episodes that succeed. Includes disturbed states; the main number (selects `ckpt_best.pt`) |
| `first_success` | First episode of each chain, i.e. from the upright start |
| `chain_success` | Success within `--max-chain` episodes |

Defaults:

| Setting | Default |
|---|---|
| timeout-s | 5 (100 control steps) |
| max-chain | 5 episodes |
| lr | 1e-4 (Adam, weight decay 1e-5) |
| batch | 32 |
| buffer | 20 000 |
| warmup | 64 |
| utd | 1 |
| eps-decay-episodes | 1000 (ε 1.0 → 0.1) |
| handoff-dz | 0.07 m |
| handoff-jitter-cm | 0 |
| min-cell-frac | 0.25 |
| episodes | 3000 |
| eval-every / eval-repeats | 100 episodes / 2 |
| ckpt-every | 50 episodes |
| workers | 6 (LIBERO) / 0 (toy, in-process) |

The learner runs on CPU. The LIBERO venv's torch (1.11+cu113) has no kernels for the RTX 5090 (sm_120),
and the pi0 server holds the GPU anyway (~26 GB with 6 clients).

**Learner cost.** One gradient step (width 32, batch 32, terminal-only batch) takes about 0.04 s on this
machine's 32-core CPU (torch's default 16 threads). A round of 6 chains adds ~10–15 episodes, so at
`--utd 1` the learner needs well under a second per round, against ~5–10 s of simulation. The loop is
synchronous (actors wait while the learner updates), so `--utd` above 1 is affordable here; it was
lowered to 1 on a slower CPU (~1.1 s per step). Watch `elapsed_min` in `train.jsonl`. Actors use one
torch thread each; don't run two learners at once without `OMP_NUM_THREADS`.

## 6. How to run

Use the LIBERO client venv in the container (Python 3.8, torch 1.11, robosuite 1.4.1, MuJoCo 3.2.3), with
`MUJOCO_GL=egl` (`train_dqn.py` sets it by default). Start the pi0_libero server for every step except 0
and 1, from the repo root with `./serve_pi0.sh`, or without `uv` syncing:
`cd /opt/openpi && OPENPI_DATA_HOME=/home/DockerShared/.cache/openpi .venv/bin/python scripts/serve_policy.py policy:checkpoint --policy.config=pi0_libero --policy.dir=gs://openpi-assets/checkpoints/pi0_libero`
(ready in ~1 min; it compiles on the first request, and the actors retry their connection meanwhile).

```bash
cd failure_agent_prototype/RL/pixel_Q
PY=/opt/openpi/examples/libero/.venv/bin/python

# 0. Code check, no simulator (~2 min on CPU). See section 8 for the expected numbers
$PY train_dqn.py --env toy --out-dir runs/toy --episodes 1500 --width 16 --lr 3e-4 \
    --eval-every 500 --eval-repeats 10 --eps-decay-episodes 750 --print-every 100 --no-save-buffer

# 1. Geometry + handoff check, no pi0. Check in the PNGs and the printed table:
#    left = wrist view (what the Q-map sees): magenta tint = the object mask (visualisation only),
#    yellow box = the 80x80 crop, white dots = the 100 cell centres, cyan = their back-projections
#    (on the white dots off the object; up to ~5 px off on edge cells, where the mean object pixel is
#    not the cell centre). right = agentview with the same 3D cell points: cyan (>= 50 % object)
#    should land on the object ("agentview hit" ~1; a few can be hidden from agentview).
#    Also "tilt ... -> handoff ~0 deg", "handoff reached True", valid cells 100.
#    The defaults reset all ~1000 task starts (~20 min); one init per task takes ~1.5 min:
$PY train_dqn.py --mode inspect --out-dir runs/inspect --train-inits 0 --eval-inits 40

# 2. Smoke test (pi0 up, ~2.5 min: the eval at episode 10 runs ~50 chains)
$PY train_dqn.py --out-dir runs/smoke --episodes 10 --workers 2 --eval-every 5 --eval-repeats 1 --video

# 3. Baseline to beat: handoff pose + pi0, no correction, same chains, eval starts (~4.5 min)
$PY train_dqn.py --mode baseline --out-dir runs/baseline

# 4. Train, then evaluate the best checkpoint (the network width is read from the checkpoint;
#    --ckpt picks another one)
$PY train_dqn.py --out-dir runs/pick_v0 --workers 6 --wandb-project online         # add --resume to continue
$PY train_dqn.py --mode eval --out-dir runs/pick_v0 --video

# 5. Eval on specific task


$PY train_dqn.py --mode baseline --out-dir runs/baseline \
      --sources tasks \
      --tasks libero_90:10 \
      --eval-inits 0 1 2 3\
      --video --media-n 4
```

**Weights & Biases (optional).** Add `--wandb-project <project>` to any train / eval / baseline command
(optional: `--wandb-entity`, `--wandb-name`, `--wandb-tags`, `--wandb-mode offline`). One-time login in
the LIBERO venv: `/opt/openpi/examples/libero/.venv/bin/wandb login`, or set `WANDB_API_KEY`. The x-axis is
`episodes`. Logged keys: `train/*` (everything in a `train.jsonl` line: loss, q_sa, target, eps, success
rates, `train/buffer/*`, plus a `train/decision_tilt_hist` histogram), `eval/*` (episode / first / chain
success, `eval/best_episode_success`, `eval/per_start_episode_success/<tag>`, and the Q-map PNGs / videos
as `eval/qmap/<tag>` / `eval/video/<tag>`). `--resume` continues the same W&B run (its id is in
`OUT_DIR/wandb_id.txt`); eval / baseline runs are separate runs grouped under OUT_DIR's name. The venv has
wandb 0.13.1 (LIBERO's pin; its missing deps were added, nothing upgraded). That version asks the server
about lists of images even in offline mode, so log images one key at a time.

```bash
$PY train_dqn.py --out-dir runs/pick_v1 --workers 6 --wandb-project pixel-q-pick
$PY train_dqn.py --out-dir runs/pick_v1 --workers 6 --wandb-project pixel-q-pick --wandb-mode offline
$PY -m wandb sync runs/pick_v1/wandb/offline-run-*      # upload an offline run later
```

**Outputs** (`OUT_DIR`):

| File | Content |
|---|---|
| `config.json` | Run settings (`config_eval.json` / `config_baseline.json` in those modes) |
| `episodes.jsonl` | One line per chain: start, chain/first success, and per episode the cell, yaw, object tilt, success, timeout, chain-end reason, move/pi0 steps, re-handoff result, Q values. Skipped and reset-failed chains too |
| `train.jsonl` | Per round: success rates, loss, mean Q, ε, buffer stats, object tilts at the decisions |
| `eval.jsonl` | Episode / first-attempt / chain success, per-start episode success, failed resets |
| `ckpt_latest.pt`, `ckpt_best.pt`, `replay.pkl` | Checkpoints (with the network width and the ineligible starts) and replay buffer (skipped with `--no-save-buffer`) |
| `qmaps/<label>_<start>.png` | First decision of the first eval chain of the first `--media-n` (3) eval starts: crop with max-over-rotation Q, chosen cell boxed, plus the 9 rotation maps. Label is `eval_ep<N>` during training, `eval` / `baseline` in those modes |
| `videos/<label>_<start>.mp4` | Wrist \| agentview recordings of the same eval chains (with `--video`) |
| `inspect/*.png`, `inspect/starts.json` | Inspect-mode output |

## 7. Deviations from pixel_Q_RL.md and assumptions

`pixel_Q_RL.md` itself was not edited. Where the code differs from it:

- **Handoff pose and 2D action.** The plan's inference step 3 says "run a `POINT_APPROACH`" to the
  back-projected point, which would also move in z. The code follows the clarified design instead:
  fixed handoff pose at 7 cm, then a horizontal move at that height plus yaw.
- **Episode = one decision with a 5 s judge window**, replacing the plan's "pi0 runs 3 chunks" and
  `T_max` corrections per episode. Failed episodes chain on the same scene, from the bowl's current
  state.
- **Reward source.** The sim pick predicate stands in for the VLM judge during training (section 3).
- **Cell size.** 8×8 windows, non-overlapping (stride 8): 100 cells × 9 bins = 900 actions. The plan's
  Notation section still says 5×5.
- **Segmentation is privileged.** The mask comes from MuJoCo's segmentation render, standing in for a
  segmentation model. The network itself sees only RGB-D, mask and proprio.
- **No language input.** One corrector per subtask on round-rim objects. The crop already shows the object.
- **Only `rl/pick` is implemented.** For `rl/place`, segment the destination, keep the gripper closed
  during the handoff and the move, and use `PlaceCheck` as the judge.
- **Rotation as 9 output channels**, not VPG-style input rotation (see section 8 on rotation).

## 8. What has been tested, and what has not

Steps 0–3 of section 6, eval mode and `--resume` all ran in the container on 2026-11-20 (RTX 5090 for
the pi0 server, learner on CPU). No full training run yet.

**Tested on CPU without the simulator:**
- Q-net shapes, action encoding and decoding, masked argmax, replay buffer, DQN update (terminal and
  bootstrapped paths), and the Q-map PNGs.
- Chain training on the toy env end to end, plus resume, baseline and eval modes. Eval reads the
  network width from the checkpoint.
- `top_down_quat`: it levels a tilted gripper to pointing straight down and keeps the yaw. Tested with a
  stand-in for robosuite's quaternion functions (x, y, z, w order).
- pyflakes is clean, and all files parse as Python 3.8.

**Toy results** (chained episodes, 1500 episodes ≈ 450 chains, width 16, ~7 min on the first CPU, ~2 min
in the container; a rerun here gave episode / first / chain success 0.22 / 0.29 / 0.63 at 1500). Random valid
actions succeed ~8 % per episode, and ~34 % per chain of up to 5 episodes.

| Eval after | Episode success | First attempt | Chain success | Rim cell | Facing side | Right rotation bin |
|---|---|---|---|---|---|---|
| 500 episodes | 0.13 | 0.21 | 0.43 | 84 % | 38 % | 9 % |
| 1000 episodes | 0.19 | 0.19 | 0.60 | 92 % | 55 % | 9 % |
| 1500 episodes | 0.23 | 0.28 | 0.65 | 87 % | 57 % | 11 % |

So the online loop learns: episode success is 3× random, and chain success is about double. The ceiling
without the rotation part is ~0.33 per episode (rim and facing side, but a wrong bin), which is where
it is heading. The rotation bin stays at chance (1/9), as in earlier runs.

**Rotation is the hard part** (from earlier toy runs with the same network):

| Part of the decision | Depends on | Learned? |
|---|---|---|
| Pick a rim cell | image | yes: 85–92 % of greedy picks |
| Pick the rim side facing the gripper | proprio direction × image | partly: ~65 % |
| Pick the rotation bin that cancels a yaw error | proprio yaw only | **no**: ~10 %, chance is 11 % |

Diagnosis:
- The data contains the signal: the empirical success table picks the right bin for every yaw.
- The learners can't resolve it from the sin/cos yaw feature, where neighbouring bins differ by only ~0.09.
  - An MLP on proprio alone reaches 0.44 bin accuracy.
  - The FCN reaches 0.20 even offline on balanced data.
  - Neither late fusion, FiLM conditioning, nor a ±1-scaled yaw feature changed this, so none were kept.

What this means for LIBERO:
- If the right rotation mainly depends on something visible (a mug handle relative to the gripper, a
  bowl lying on its side), the FCN should cope, since it learns image-dependent choices well.
- If it depends on the gripper's absolute yaw, consider:
  - a higher-resolution yaw encoding, e.g. radial-basis (RBF) features over the yaw angle; or
  - VPG-style rotation of the input crop, so the yaw shows up in the image rather than only in proprio.

**Verified in LIBERO** (inspect on the library starts plus init 0 and 40 of every default task, 59 starts;
smoke test; baseline):
- **Segmentation render.** robosuite 1.4.1 returns (objtype, objid) per pixel; the masks sit on the
  bowl/mug in every inspect PNG.
- **Image orientation.** Cell back-projections land within 1.9–3.0 px of the cell centres (the gap is
  the cell's mean object pixel versus its geometric centre). (Agentview crop; superseded below.)
- **Wrist-view Q-map** (2026-11-20, inspect → `runs/inspect_wrist`, 53 usable starts): every start
  reaches a handoff with 100 valid cells, 13–41 of them ≥ 50 % object (mugs 13–30, bowls 19–41). The
  wrist cells' 3D points land on the object in agentview (agentview hit 1.00 on 47 starts, 0.86–0.97 on
  the rest, where the gripper hides part of the object from agentview). The bottom 1–2 cell rows of
  the crop are the gripper body in the wrist view; they back-project under it at object height.
  The baseline and timing numbers below were measured before the switch; pi0 itself is unchanged.
- **EE axis convention.** Approach tilt is ~3° at the task starts and 5–32° at the library states,
  0–3° at the handoff.
- **Starts.** 6 library starts are skipped as not round-rim (moka pot, book, pudding), and the 3 task-21
  library starts as outside the training split. Every task start reaches a handoff. 11–31 cells with ≥ 25 % object pixels (all 100 are actions now).
- **Baseline to beat** (eval starts, 2 repeats, 104 chains): episode success **0.36**, first attempt
  **0.44**, chain **0.86**.
- **Is 5 s enough?** Mostly. Successful episodes needed 38–90 pi0 steps (median ~55–59, 90th percentile
  ~75–82), and the 2D move takes a median of 16 steps (60 at most). A few successes may still be cut
  off at the window's end.
- **Disturbed states occur.** In episodes after a chain's first, 14–15 % have the object tilted > 30°
  (many ~86°, i.e. on its side).
- **Occlusion.** At the handoff pose the fingers cut a thin unmasked seam through the object (visible
  in the inspect PNGs). This only shifts where on-object cells back-project; every cell stays an action.
- **Simulation time.** ~5.5 s per chain with 2 workers (smoke) and ~15 s with 6 workers (baseline;
  they share one pi0 server). 6 spawn workers ran without the EGL hang seen in `validate_predicates`.

**Open issues:**
- **libero_90 task 31 (bowl onto the cabinet).** From the task init states the handoff's descent leg
  stalls 4.5–7 cm above its target (60 steps, `reached False`), and re-handoffs take ~50–220 steps. The
  reset is not rejected, so the Q-map decides from ~12–14 cm above the bowl instead of 7 cm. No first
  episode there succeeded in the baseline; later episodes in the chain sometimes did. The library start
  of the same task reaches its handoff.
- The handoff counts as done even when `reached` is False; only a failure or a fired pick predicate
  raises `ResetFailed`.

## 9. Changes since the first version

| Change | Why |
|---|---|
| Handoff pose (7 cm above the segmentation-box centre, pointing down, open) before every decision | The VLM agent sets this pose before every Q decision |
| Action = horizontal move at handoff height + yaw (was rise, travel, descend to 7 cm above the cell point) | z is not part of the learned action |
| Episode = one decision, judged within a 5 s timeout (`--timeout-s`); was up to `t_max` bootstrapped corrections with a fixed 100-step pi0 segment | Clarified episode definition; Q = success probability of the handoff |
| Failed episodes chain on the same scene from the object's current state (`--max-chain`); fresh upright start after success / irrecoverable / cap | Disturbed bowl states come from the rollouts, not from randomizing starts |
| Tipped / knocked-over object is not terminal; off-table 5 → 15 cm, out-of-reach 25 → 30 cm | A disturbed bowl is a state the Q-map must handle |
| Pick lift measured from the object's resting height (lowest since each episode's handoff) | Works for objects lying on their side |
| Baseline = the same chains with pi0 right after each handoff | Measures what the Q-map adds over the handoff alone |
| Success rates per episode, first attempt and chain; `ckpt_best.pt` by episode success | Matches the chain structure |
| `ResetFailed` (skip one chain) separate from `IneligibleStart` (drop the start) | A failed first handoff is chance, not a property of the start |
| Target-network forward skipped when a batch is all terminal; eval reads the width from the checkpoint | Speed; usability |
| Default `--utd` 4 → 1 | ~1.1 s per CPU gradient step; utd 4 made the learner the bottleneck |
| Tried and removed: late fusion, FiLM, a scaled toy yaw feature (rotation learning), a random pi0 prefix and a state archive (state diversity) | No measurable gain / superseded by chaining from the bowl's current state |
| robosuite horizon raised to 100 000 steps per `env.reset()` (`corrector_env.HORIZON`) | Long chains passed the default 1000 and ended with "executing action in terminated episode" (2 of 104 baseline chains) |
| Actors retry the pi0 connection on a handshake timeout | A cold server compiles on its first request and blocks the other actors' handshakes |
| `interactive_handoff.py` imports `pi0_request` / `REPLAN_STEPS` from `fa_core.common` | `oracle_handoff_test.py` moved to `archive/` |
