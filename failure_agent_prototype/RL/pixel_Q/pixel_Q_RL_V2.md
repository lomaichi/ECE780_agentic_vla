# Pixel-wise Q-learning V2: A Learned Trigger for the Low-Level Corrector (ECE780)

V2 extends [pixel_Q_RL.md](pixel_Q_RL.md) and [qlearning_code_instruction.md](qlearning_code_instruction.md).

In V1, the corrector acted only after the VLM agent brought the gripper to a fixed handoff pose. In V2, the Q-map also decides **when** to intervene. At every gated tick of the rollout (5 fps), the value of the best small correction is compared with the value of letting pi0 continue, which is read from the Q-map's own no-op entry, **Q0**. A correction runs only when it is worth enough more.

pi0 stays frozen. The VLM agent keeps priority for high-level (HL) failures.

## 0. Why V2

**V1 result** (`runs/pick_v0`, eval starts, from the canonical handoff pose):

- The greedy Q-map reached a best eval episode success of 0.65. Its last eval gave 0.54 episode, 0.77 first-attempt and 0.87 chain success.
- The baseline (handoff pose, then pi0, no correction) gave 0.36, 0.44 and 0.86.

**What is missing is a trigger.** The Moondream2 grounded recovery agent detects and corrects HL failures. Low-level (LL) modes are only reported: missed_grasp, pregrasp_hesitation, release_stall, placement_instability.

LL failure is a continuous visual judgement. It is hard to put into words or rules, and a standalone classifier would need a lot of labelled data.

**V2 learns when a correction pays off.** Intervene only if the estimated value of the best correction is higher than the value of letting pi0 continue, by a margin.

**Design principle: widen the trigger, not the corrector.**

- The trigger sees every state of the rollout.
- The correction stays local and small.
- pi0 keeps the approach and the grasp, and the VLM keeps HL recovery.

## 1. What changes from V1

| | V1 | V2 |
|---|---|---|
| When the corrector acts | After the VLM's handoff move | Any gated tick at 5 fps, chosen by the learned trigger |
| State at a decision | 7 cm above the object, facing −z | Wherever pi0 has brought the gripper |
| Observation | Object-centred 80×80 wrist crop; fixed scale only because the pose was fixed | Gripper-centred world-frame heightmap, 88×88 px at 2.5 mm/px (22 cm) |
| Action | Move above any of 100 cells at handoff height, plus yaw | Level the gripper, then move to a cell within reach r at the current height, plus yaw |
| Value of doing nothing | Not modelled | Q0: the centre cell at 0°, a pseudo no-op |
| Label | Success within 5 s (0/1) | Time-discounted success within H_max = 3 s |
| Start states | Library failure-onset states and task init states | Task init states only |
| Data | On-policy chains from the handoff pose | Branch points on main rollouts; each gives a no-op branch and an intervention branch from the same snapshot |

## 2. Scope (unchanged)

- `rl/pick` only. `rl/place` follows the same recipe later.
- Round-rim targets only (bowls, mugs).
- Same training and held-out splits as V1, with the same held-out guard.

## 3. Terms

| Term | Meaning |
|---|---|
| Control step | One 20 Hz step of the sim. **All timing is counted in control steps**, never wall-clock time |
| Tick | A 5 fps decision opportunity: every 4 control steps |
| Gated tick | A tick at which the gate (§4) is open |
| Main rollout | A run from a task init state until pick success, an irrecoverable state, or the step budget. Contains many ticks. Hosts the branch points |
| Branch point | A gated tick where the sim is snapshotted and two branches are run from the snapshot |
| Branch | An H_max window run from a branch point's snapshot: one decision at its start, then pi0 alone. Produces exactly one label |
| No-op branch | Branch with no action: pi0 alone. Its label trains Q0 |
| Intervention branch | Branch with a correction (primitive), then pi0 alone. Its label trains that correction's entry |
| Levelling probe | A phase 0 intervention branch that executes the centre cell (level only) |
| Q0 | Q(s, a_0), the value of letting pi0 continue |
| A(s) | Best penalized correction value minus Q0. The trigger fires when it exceeds τ |

The word "episode" is avoided on purpose: a **main rollout** has many ticks and several branch points; a **branch** has one decision.

## 4. Roles and the gate

| Component | Job |
|---|---|
| pi0 | Approach, descent, grasp. Runs whenever nothing else is acting |
| VLM agent (Moondream2 grounded, 5 fps) | Subtask success, HL failure detection and correction, target grounding (world point from agentview detect + depth). **Has priority**: the trigger is off while an HL correction runs |
| Q-map Q(s, a) | Value of each local correction, plus the no-op entry Q0. Decides **when** to intervene (the trigger) and **which** correction to run |

**Gate.** The trigger is evaluated, and branch points can be placed, only when all of the following hold:

1. The current subtask is `pick`.
2. The gripper is open and holds nothing.
3. The target is grounded, and its xy is within r + 4 cm of the grasp point.
4. No VLM correction is running, and the cooldown since the last intervention or VLM correction has passed.

The gate defines the corrector's scope. It is not failure detection. Outside the gate, pi0 runs and the Q-map is not evaluated.

During training, privileged sim quantities stand in for the VLM's judgements, as in V1: the pick predicate, the target's sim position and visibility, and the sim segmentation.

### Who decides what

| Stage | When to intervene | Which correction |
|---|---|---|
| Phase 0: levelling check (§9.5) | Random branch points | Always the centre cell (level only) |
| Phase 1: training with branching | Random branch points | Q-map, ε-greedy |
| Phase 2: fine-tuning with the trigger in the loop | Main rollout: learned trigger. Branches: random branch points | Main rollout: Q-map, greedy. Branches: ε-greedy |
| Deployment | Learned trigger | Q-map, greedy |

**The Q-map does control timing.** At deployment, the trigger is computed from the Q-map: intervene when the best correction beats Q0 by a margin (§8).

**But timing is not learned by letting the trigger fire.** It follows from the Q-map's values, which are learned at branch points spread across many moments of many rollouts. At each branch point the data records two outcomes from the same snapshot: what happened after a correction, and what happened when pi0 simply continued. If an untrained trigger chose those moments, the data would contain only the moments it happened to fire on.

## 5. Observation: gripper-centred heightmap

**Construction**, at each gated tick:

1. Render wrist and agentview depth (`fa_core.camera`). Unproject both into world points, using each camera's pose at that tick.
2. Drop points that belong to the robot. During training, use the sim segmentation. At test time, use a fixed finger mask on the wrist image.
3. Keep the points inside a 22×22 cm square centred on the grasp site's xy. Align the square with the **gripper's yaw**, so the map is egocentric in yaw like V1's wrist crop. Rotation then stays visible in the image instead of only in proprio.
4. Rasterize at 2.5 mm/px into 88×88. Each pixel keeps the highest point and its colour. Where both cameras see a pixel, the wrist point wins (higher resolution).

**Channels:**

| Channel | Size | Content |
|---|---|---|
| `rgb` | 3 | Colour of the top surface |
| `height` | 1 | (h − z_grasp) / 0.1 m, clipped to ±1.5: how far below the fingers each surface is |
| `mask` | 1 | Target object |
| `observed` | 1 | 1 where any camera saw the pixel, 0 for holes |

**Proprio** (tiled over the map as in V1): gripper opening; tilt of the approach axis from −z (sin, cos); height of the grasp point above the target's top. EE yaw is not needed, because the map is yaw-aligned.

**Cell grid.** 11×11 cells of 8×8 px (2 cm each). The grasp point sits at the centre of cell (c, c), with c = 5. The FCN's stride of 8 maps 88 → 11. The grid must be odd so the gripper sits in a cell, not on a corner between four: Q0 needs a centre cell.

**Cell → world target**, with s = 2 cm and ψ the gripper's yaw:

$$
x_{\text{target}} = p_{\text{grasp},xy} + R_z(\psi)\begin{bmatrix}(j - c)\,s\\ -(i - c)\,s\end{bmatrix},\qquad z_{\text{target}} = z_{\text{current}}
$$

No depth lookup is needed to place a cell. Height is an input the Q-map conditions on, not part of the action.

## 6. Action, no-op and primitive

**Action** $a = (i, j, k)$, with rotation bin $k \in \{0, \dots, 8\}$, $\theta_k \in \{-20°, -15°, \dots, +20°\}$; $k = 4$ is 0°.

**Valid actions** $\mathcal{A}_r = \{(i, j, k) : \max(|i - c|, |j - c|) \le r\}$, set through the existing `valid` mask. The map still shows the full 22 cm for context.

- Default r = 3 cells (6 cm): 7×7 = 49 cells × 9 bins = 441 actions, including the no-op.
- Tune r from data: in `pick_v0/episodes.jsonl`, measure how far the chosen cell was from the gripper's cell in successful episodes.

**No-op** $a_0 = (c, c, 4)$: the centre cell at 0°, i.e. no lateral move and no yaw change. Its value is **Q0(s) := Q(s, a_0)**.

- **Pseudo no-op.** Strictly, the primitive at $a_0$ would still level the gripper. Levelling is assumed small enough to ignore, so $a_0$ ≈ "let pi0 continue". Phase 0 (§9.5) tests this assumption before any training.
- **Q0 is trained from no-op branches**, where pi0 simply continues and nothing is executed. At deployment, when no correction wins, nothing is run either: pi0 keeps its current chunk, and the gripper is not levelled.
- **Corrections** are $\mathcal{A}_r \setminus \{a_0\}$: 440 actions.

**Primitive**, run for a correction:

1. **Clearance.** Let h_clear be the maximum heightmap height over a gripper-sized window (6×6 cm) around the current cell, the destination cell and the straight path between them. If z_grasp < h_clear + 2 cm, rise to h_clear + 3 cm.
2. **Level.** Rotate about the grasp site until the approach axis points along −z, keeping the yaw (`top_down_quat`). The grasp point's xy does not move.
3. **Move.** Bring the grasp site to the target xy at the current height, adding θ_k to the yaw. Gripper open. One `MoveTo`, capped at 30 control steps (1.5 s).
4. **Hand back** to pi0, which replans.

The cap matters because the primitive's time comes out of the 3 s branch window. Local moves of at most 6 cm and 20° should take far less; log the actual durations.

**Z is not learned.** It is set by the primitive: the current height, after the clearance rise.

## 7. Values and labels

**Label** of a branch started at tick $t$:

$$
y =
\begin{cases}
\gamma^{\Delta t}, & \text{if the pick predicate first fires at } t + \Delta t,\ \Delta t \le H_{\max},\\
0, & \text{otherwise.}
\end{cases}
$$

- Δt is in seconds of sim time (control steps / 20) and counts from the branch tick, so the primitive's execution time is included.
- Defaults: γ = 0.85 per second, H_max = 3 s = 60 control steps.
- The discount rewards faster success and charges each correction for the time it takes.

**Choosing H_max.** 3 s is the default. V1 measured a median of about 3.7 s from decision to pick, starting 7 cm above the object, including the move. V2's branch points can be closer to the grasp, but the window must still fit the primitive plus pi0's grasp and the 10-step hold that `PickCheck` requires. Phase 0 logs time-to-success with a long cap to confirm the value (§9.5). If correction branches rarely succeed within the window, the window is the cause, not the corrector: raise H_max, for example to 5 s.

**Q(s, a)**, for a correction, is the expected y from executing $a$, then pi0 alone until the window ends.

**Q0(s)** is the expected y if pi0 simply continues from s for the same window.

Every intervention is a **one-step improvement over pi0**. There is no bootstrapping. All samples are terminal, as in V1, so the existing terminal-only learner path still applies.

**Two kinds of label, one network:**

| Label | Comes from | Trains |
|---|---|---|
| y_noop | The no-op branch at a branch point (pi0 alone, H_max) | The $a_0$ entry (Q0) |
| y_a | The intervention branch at that branch point (correction, then pi0 alone, H_max) | The executed correction's entry |

**Every branch point gives exactly one of each, from the same snapshot and over windows of the same length.** The data is therefore 1:1 by construction, and no batch balancing is needed.

Q0 is still by far the best-supervised entry. It gets one label per branch point, while each of the 440 correction entries gets about 1/440 of them. Because the convolution weights are shared across cells, Q0's labels also train the trunk that every other entry uses.

**Network.**

- V1's FCN, with 6 map channels plus tiled proprio as input and (9, 11, 11) as output.
- An ensemble of K = 5 heads on a shared trunk gives an uncertainty estimate for each entry.

**Loss.** MSE on the entry each sample labels, all other entries masked out, as in V1.

## 8. Trigger

With μ and σ the ensemble mean and standard deviation:

$$
A(s) = \max_{a \in \mathcal{A}_r \setminus \{a_0\}} \big[\mu_a(s) - \kappa\,\sigma_a(s)\big] \;-\; \mu_{a_0}(s)
$$

- Intervene with the argmax when A(s) > τ on N consecutive gated ticks (N = 3, i.e. 0.6 s), then apply a cooldown of 3 s.
- The trigger is **signed**: only a positive advantage counts.
- Q0 uses its mean. It is the best-supervised entry.

**Why the margin and the uncertainty penalty.** Suppose a correction is truly useless in some state, so every action's true value equals Q0. The network's estimates still carry random errors. Q0 is a single estimate, but the max over 440 estimates picks the luckiest error, about 3 standard errors above a typical one. The trigger would fire with zero real benefit.

The coin version: flip each of 440 fair coins 10 times, and the best coin usually shows 9 heads, without being special.

Each measure cuts this bias differently:

- The local reach reduces the number of candidates.
- κσ removes cells that look good by luck, since ensemble members disagree on them.
- Persistence removes frame-to-frame luck.
- τ is set above the measured noise floor (§11).

The deployment loop is in §10.6.

## 9. Training

### 9.1 Start states: task init states only

- Main rollouts start from the tasks' normal init states: training split, inits 0–39, after the 10 wait steps. Use `--sources tasks`.
- The library's failure-onset states are **not** used. They were collected as moments where the corrector should act, so they would teach the trigger to fire.
- Full rollouts from the init state contain every kind of moment, and the trigger must learn all of them:
  - moments where a correction helps (A > 0),
  - moments where pi0 is doing fine (A ≈ 0),
  - moments where a correction hurts (A < 0), for example interrupting a grasp that was about to succeed.

  The third kind is what teaches the trigger to stay quiet.
- The gate already focuses data on the phase near the target, so there is no need to bias sampling toward failure moments.

### 9.2 One main rollout at a glance

```
main rollout (phases 0–1: pi0 + HL stand-in, never intervenes)
─tick─tick─tick─[branch point]─tick─tick─tick─ ... ─tick─[branch point]─tick─ ...
                      │
                      ├─ 1. snapshot: sim + step count + pi0's queued actions
                      ├─ 2. no-op branch:        pi0 alone for H_max          → y_noop  (Q0)
                      ├─ 3. restore snapshot
                      ├─ 4. intervention branch: correction a, then pi0 alone → y_a     (Q pixel a)
                      ├─ 5. restore snapshot
                      └─ 6. the main rollout continues from the branch point
```

**Counts**, per 2 s of gated time at 5 fps:

| | Count |
|---|---|
| Gated ticks on the main rollout | 10 |
| Branch points (P_BRANCH = 0.1 per gated tick) | ~1 |
| Labels per branch point | 2: one y_noop, one y_a |
| Ticks that are not branch points | ~9: the main rollout just moves on, no labels |

Branching is not a coin flip between "no-op" and "intervene". At a branch point **both** branches always run.

### 9.3 Branch points

**Selection.** Each gated tick of the main rollout becomes a branch point with probability P_BRANCH = 0.1, i.e. about one per 2 s of gated time.

- **Why not every tick.** A branch point costs up to 2 × H_max of extra simulation plus the primitive: up to 150 control steps, against 40 steps for 2 s of main rollout. Branches that succeed stop early. Ticks 0.2 s apart are also nearly the same state, so branching at every tick would mostly buy duplicates. The same compute spread over more rollouts, inits and tasks gives more varied states.
- **Why random.** A fixed spacing can keep landing at the same point in pi0's chunk cycle: ticks fall every 4 control steps and pi0 replans every 5. Random selection samples every point in that cycle. A stratified version, with one random tick in each 2 s window, gives more even coverage if wanted.

P_BRANCH is a compute knob. Raise it as far as the simulation budget allows, and lower it if the replay buffer fills with near-duplicate states.

**Procedure at a branch point:**

1. **Snapshot.** Save the sim state, the env's step counter and pi0's queued actions in memory, using the save/restore code in `tools/scene_restore`. This is a temporary snapshot of the running rollout, not a survey or library state. Restoring it must also put back pi0's queued actions; otherwise clear the queue and let pi0 replan.
2. **No-op branch.** pi0 alone from the snapshot. Stop at success, an irrecoverable state, or H_max. The result is y_noop.
3. **Restore.**
4. **Intervention branch.** Choose a correction ε-greedily over $\mathcal{A}_r \setminus \{a_0\}$ (ε from 1.0 to 0.1, as in V1). Run the primitive, then pi0 alone. Same stop rules. The result is y_a.
5. **Restore**, and the main rollout continues from the branch point.

Inside a branch, nothing is decided after its first tick: no new branch points, no trigger, no HL stand-in. Each branch runs its own step loop with its own clock, measured from the branch tick.

**Optional: same pi0 noise.** If the pi0 client lets the sampling noise seed be fixed, use the same seed in both branches. The two labels then differ mainly because of the action.

**Cheaper alternative.** Skip the no-op branch and read y_noop from the main rollout instead. After the restore, the main rollout continues with pi0, and its first H_max steps after the branch tick give the same label. This saves one window per branch point, but it needs bookkeeping for open windows. The label must be dropped if an HL correction or the step budget cuts that window. With H_max = 3 s, the explicit no-op branch is cheap enough to keep as the default.

**Main rollout limits.** The main rollout is not limited by H_max. It ends at pick success, an irrecoverable state (V1's rules), or the episode step budget. There are no pose resets inside a rollout, and then the scene resets to a new init state. V1's "not visible" irrecoverable rule should use the agentview or the sim position, since the target can leave the wrist view during normal pi0 motion.

**Why not intervene at random without snapshots.** Each label means "this action, then pi0 alone for H_max." Intervening at random on, say, half the ticks would interrupt the robot about every 0.4 s, so no label window would stay clean.

A restore-free version works if interventions are rare and blocked for H_max after each one. It still has three downsides:

- one label per H_max of rollout;
- no paired comparison, so the advantage estimate is noisier;
- many visited states come after an intervention, unlike the pi0-driven states the deployed trigger will mostly see.

### 9.4 HL failures during training

If pi0 heads for the wrong object, the target never comes within reach, the gate stays closed, and the rollout yields no data. That is harmless, but it wastes the rollout.

At deployment, the VLM agent would correct such failures and bring the gripper toward the right target. Those post-correction states are ones the trigger will see.

To include them, run a privileged stand-in for the VLM's HL rules in the main rollout, as V1 did for the handoff. It uses `point_approach` toward the true target from the sim state, when the gripper approaches or grasps the wrong object. Branches never run it: they are pi0 alone by definition, so labels are unaffected.

### 9.5 Phases

**Phase 0: levelling check, before any training.**

Q0 rests on one assumption: "do nothing" ≈ "level the gripper only". Phase 0 tests it. It needs nothing from the Q-map, so it runs first; its result decides whether the Q0 approach is valid at all.

1. Run main rollouts with branch points exactly as in §9.3.
2. At every branch point, the intervention branch executes $a_0$ (level only, then pi0 alone): a levelling probe.
3. Compare each probe's label with y_noop from the same snapshot, binned by tilt: 0–10°, 10–20°, 20–30°, over 30°.

About 100 probes per tilt bin is enough to spot a difference of roughly 0.15. That is about 400 probes, roughly the cost of a V1 baseline run. High-tilt states are rarer: run until those bins fill, or merge the top two bins.

| Result | Next step |
|---|---|
| No real difference at any tilt | Go ahead with Q0. Optionally keep a ~1% probe rate in phase 1 as a monitor |
| A difference above some tilt | Train Q0 only on branch points below that tilt, or switch to the separate-V option (§15) before training starts |

Phase 0 also sets H_max. Run its branches with a long cap (10 s) and log the time to success. Labels are computed by truncating at H_max, and H_max is checked against about the 90th percentile of the successes. The y_noop labels from phase 0 are ordinary Q0 labels and pre-fill the replay buffer for phase 1. The probes' own labels are not used for training.

**Phase 1: train the map with branching.**

- Actors run main rollouts with branch points (§9.3). The learner updates the network online from the replay buffer, as in V1's actors-plus-learner loop.
- Optional extra warm-start for Q0: no-op labels at random gated ticks of the saved rollouts in `Documentation/pi0_LIBERO_rollouts`, replayed from `states.npy`. Training-split tasks only: that folder also holds held-out tasks.
- At the end, check Q0's calibration with a reliability plot on init states 40–41: of the ticks predicted at about 0.6, do about 60% succeed?

**Phase 2: calibrate τ, then fine-tune with the trigger in the loop.**

- Set τ (§11).
- The main rollout now follows the deployed behaviour: HL stand-in first, then the learned trigger, otherwise pi0. This visits the states the deployed system will see, including states after an intervention.
- Branching continues unchanged, with ε at its floor, so exploration still comes from the intervention branches. The labels still come only from branches, so they stay clean even though the main rollout now intervenes.

### 9.6 Defaults

| Setting | Default |
|---|---|
| Heightmap | 88×88 px, 2.5 mm/px (22 cm), yaw-aligned |
| Cells | 11×11, 2 cm; no-op at centre cell (5, 5), 0° |
| Reach r | 3 cells (6 cm): 441 actions, 440 corrections |
| Rotation bins | 9, ±20° in 5° steps |
| Primitive cap | 30 control steps (1.5 s) |
| H_max | 3 s = 60 control steps (confirmed in phase 0) |
| γ | 0.85 per second |
| Ensemble K / κ | 5 / 1.0 |
| Persistence N / cooldown | 3 ticks / 3 s |
| P_BRANCH | 0.1 per gated tick (~1 branch point per 2 s of gated time) |
| Labels per branch point | 1 no-op + 1 intervention (1:1, no batch balancing) |
| Phase 0 probes | ~100 per tilt bin (~400 total) |
| ε (intervention branches) | 1.0 → 0.1, as in V1 |
| Replay buffer | 20,000 samples, never cleared |
| Minibatch / warm-up | 32 / 500 samples before the first update |
| Start states | Task init states only, training split, inits 0–39 (eval: 40–41) |
| lr, optimizer | As in V1 (1e-4, Adam with weight decay 1e-5) |

## 10. Pseudocode

### 10.1 Constants and labels

```python
CTRL_HZ        = 20                  # control steps per second (all timing in steps)
TICK_STEPS     = 4                   # one tick every 4 control steps = 5 fps
H_MAX_STEPS    = 3 * CTRL_HZ         # branch window: 3 s = 60 control steps
PRIM_CAP       = 30                  # primitive cap: 1.5 s
P_BRANCH       = 0.1                 # branch probability per gated tick
EPISODE_BUDGET = TASK_HORIZON        # main rollout step budget
GAMMA          = 0.85                # discount per second
A0             = (5, 5, 4)           # no-op: centre cell, 0°

def label(dt_steps):
    """Label of a branch whose pick succeeded dt_steps after its branch tick."""
    return GAMMA ** (dt_steps / CTRL_HZ) if dt_steps <= H_MAX_STEPS else 0.0
```

### 10.2 One branch

```python
def run_branch(action, cap_steps=H_MAX_STEPS):
    """Run one branch from the current (just-restored) snapshot.
    Returns (label, steps_to_success or None). Decides nothing after its first tick."""
    t0 = env.step_count
    if action != NOOP:
        execute_primitive(action, max_steps=PRIM_CAP)  # clearance rise, level, move + yaw
    while True:
        if pick_success():                             # PickCheck: lifted 5 cm, held 10 steps
            dt = env.step_count - t0
            return label(dt), dt
        if irrecoverable() or env.step_count - t0 >= cap_steps:
            return 0.0, None
        pi0_step()                                     # pi0 alone: no trigger, no HL stand-in
```

### 10.3 Actor: main rollouts with branch points (phases 0–2)

```python
def actor(phase):
    while training:
        env.reset(random_training_split_init())        # new scene, after the 10 wait steps
        while not (pick_success() or irrecoverable() or env.step_count >= EPISODE_BUDGET):

            # ---- branch point -------------------------------------------------------
            if is_tick() and gate_open() and random() < P_BRANCH:
                s    = observe()                       # heightmap + proprio at this tick
                snap = save_state()                    # sim + step count + pi0 action queue

                cap = 10 * CTRL_HZ if phase == 0 else H_MAX_STEPS   # phase 0 logs long times
                y_noop, t_noop = run_branch(NOOP, cap)
                restore_state(snap)

                a = A0 if phase == 0 else eps_greedy(Q(s), over=A_r_minus_A0)
                y_a, t_a = run_branch(a, cap)
                restore_state(snap)

                if phase == 0:
                    probe_log.append((tilt(s), y_noop, y_a, t_noop, t_a))
                    buffer.push(s, A0, y_noop)        # pre-fills Q0; probe labels not pushed
                else:
                    buffer.push(s, A0, y_noop)        # one no-op sample
                    buffer.push(s, a,  y_a)           # one correction sample

            # ---- main rollout continues ---------------------------------------------
            if hl_failure():                           # privileged stand-in for the VLM's HL rules
                hl_correction()                        # e.g. point_approach to the true target
            elif phase == 2 and is_tick() and gate_open() and trigger_fires(observe()):
                execute_primitive(best_correction(), max_steps=PRIM_CAP)   # deployed behaviour
            else:
                pi0_step()                             # one control step
```

`trigger_fires` holds the persistence counter and the cooldown (§10.6). In phases 0–1 the main rollout never intervenes.

### 10.4 Learner

```python
def learner():
    while training:
        if len(buffer) < 500:                          # warm-up
            wait(); continue
        s, a, y = buffer.sample(32)                    # buffer is never cleared
        pred = Q_ensemble(s)[:, a]                     # only the entry each sample labels
        loss = mse(pred, y)                            # per head; bootstrap masks optional
        optimizer.step(loss)
        every N updates: eval on inits 40–41, checkpoint, log
```

### 10.5 Phase 0 analysis

```python
for bin in [(0, 10), (10, 20), (20, 30), (30, 180)]:   # tilt, degrees
    rows = [r for r in probe_log if bin[0] <= r.tilt < bin[1]]
    d = [label_at(H_MAX_STEPS, r.t_a) - label_at(H_MAX_STEPS, r.t_noop) for r in rows]
    report(bin, len(rows), mean(d), std(d) / sqrt(len(rows)))   # paired difference ± SE

successes = [r.t_noop for r in probe_log if r.t_noop is not None]
report("time to success, 90th percentile (s)", percentile(successes, 90) / CTRL_HZ)
```

`label_at(h, t)` recomputes a label truncated at window h from a logged success time t (0 if t is None or t > h).

### 10.6 Deployment

```python
streak, cooldown_until = 0, 0
every tick (5 fps), after the VLM agent's HL check:
    if vlm_correction_running or not gate_open() or now < cooldown_until:
        streak = 0; pi0 continues; continue            # pick subtask, gripper open & empty,
                                                       # target grounded within reach
    s = observe()                                      # gripper-centred, yaw-aligned heightmap
    mu, sigma = Q_ensemble(s)                          # (9, 11, 11) each, masked to A_r
    A = max(mu - KAPPA * sigma, over=A_r_minus_A0) - mu[A0]
    streak = streak + 1 if A > TAU else 0
    if streak >= 3:
        execute_primitive(argmax(mu - KAPPA * sigma, over=A_r_minus_A0), max_steps=PRIM_CAP)
        hand_back_to_pi0()
        streak, cooldown_until = 0, now + 3 s
    else:
        pi0 continues                                  # the no-op: nothing executed
```

## 11. Calibration, validation and evaluation

**Q0 calibration.** Run a reliability plot on held-out inits. At branch points, also compare Q0 with y_noop, and each executed correction's Q with y_a.

**Noise floor and τ.**

- In some states a correction cannot matter, so the true advantage is about 0: gated ticks where the no-op branch succeeds within 1 s, and gated ticks at the edge of reach.
- The measured distribution of A(s) on those states is the noise floor. Set τ at or above its 95th percentile.
- Then sweep τ on the eval starts, plot success rate against interventions per rollout, and pick the knee.

**Pseudo-no-op assumption and H_max.** Both are checked in phase 0 (§9.5, §10.5). If a monitor probe rate is kept, re-check the tilt bins at the end of phase 1.

**Window health.** During phase 1, watch two numbers:

- the fraction of intervention branches that succeed within H_max,
- the mean Q of corrections against Q0.

If corrections rarely succeed within the window, raise H_max before blaming the corrector.

**Trigger timing.**

- Plot A(s) over time on rollouts the survey labelled LL-5 (pre-grasp disturbance) and LL-8 (pre-grasp hesitation), and on pi0 successes. It should rise before the LL failures and stay low on the successes.
- Report the false-trigger rate on rollouts pi0 completes alone.

**Evaluation** (held-out split, zero-shot). Report success rate and interventions per rollout for:

1. pi0 alone
2. pi0 + VLM agent (HL only): the current system
3. pi0 + VLM agent + learned trigger (V2)
4. pi0 + VLM agent + a random trigger at the same intervention rate as 3. This tests whether the learned timing matters, not just the corrections.

## 12. Implementation notes (mapping to the current code)

| File | Change |
|---|---|
| `corrector_env.py` | `_observe` → heightmap builder. A rollout-driven env: run pi0 with the gate; branch point selection; `run_branch`; the HL stand-in in the main rollout; a phase 0 mode where the intervention branch executes $a_0$. The local primitive replaces `_handoff` + `step`. Keep `PickCheck`, the irrecoverable rules and the `HORIZON` fix |
| `tools/scene_restore.py` | In-memory snapshot / restore of a running rollout: MuJoCo state, the env's step counter, and the controller's internal state. Extend the restore check to cover the controller and step counter, not only pixels |
| pi0 client | Snapshot / restore of the queued action chunk, or clear the queue on restore so pi0 replans. Optional fixed noise seed per branch |
| `qnet.py` | 6 map channels + tiled proprio, output (9, 11, 11), ensemble heads, masked argmax over $\mathcal{A}_r \setminus \{a_0\}$, accessor for Q0 |
| `replay.py` | One buffer of (state, action, label) samples, capacity 20,000, never cleared. Compact storage: uint8 RGB, float16 height |
| `train_dqn.py` | Actor and learner processes as in V1; modes for phases 0–2; phase 0 report (§10.5); calibration mode (reliability, noise floor, τ sweep); trigger-in-the-loop eval mode with the VLM agent; `--sources tasks` only |
| `fa_core/camera.py` | Wrist camera pose at each tick, for unprojection |
| `recovery_agent.py` | Expose a "correction running" flag (to suppress the trigger) and the grounded target world point |

## 13. Open questions

- **H_max.** Is 3 s long enough for primitive + grasp + hold? Confirmed or revised by phase 0.
- **Reach and cell size.** Is r = 3 cells enough, and are 2 cm cells fine enough for mugs?
- **Height at hand-back.** Corrections run at the current height. Check whether pi0's success depends on the height it takes over from: run V1's baseline with `--handoff-dz` 0.04 / 0.07 / 0.10 / 0.13. If it depends strongly, add a fixed canonical height to the primitive (still not learned).
- **Test-time mask.** Training uses the sim's segmentation, but at test time the mask comes from Moondream's detect box. Train with box-derived masks too, or mix the two, to avoid a domain gap.
- **Holes.** Check how often the area under the hand is unobserved, and whether the `observed` channel is enough.
- **Compute at 5 fps.** Measure heightmap construction plus the ensemble forward pass on CPU, next to the pi0 server and Moondream on the GPU.
- **P_BRANCH.** Find the rate that fits the simulation budget, and watch for near-duplicate states in the replay buffer.

## 14. FAQ

**Does the Q-map decide when to intervene?**
Yes, at deployment and in the phase 2 main rollout: the trigger is A(s) > τ, computed from the Q-map. The branch points themselves are always chosen at random, so the Q-map learns values at every kind of moment, including the ones where it shouldn't fire (§4).

**If Q0 is never executed at deployment, how do the Q pixels get data?**
From the intervention branches. At every branch point, the intervention branch executes one correction chosen ε-greedily, which gives a label for that pixel. The no-op branch from the same snapshot gives Q0's label.

**What is the "saved state"?**
An in-memory snapshot of the running rollout at the branch point: sim state, step counter and pi0's queued actions. It lets the sim be rewound between branches. It has nothing to do with the failure survey's library states, which V2 does not use.

**Is each tick a coin flip between no-op and intervene?**
No. About 1 in 10 gated ticks becomes a branch point, and at a branch point both branches always run. The other ticks are just the main rollout moving on.

**Are no-op and correction samples 50/50?**
Yes, by construction: each branch point gives one of each. No batch balancing is needed.

**Is the no-op label taken from the whole rollout?**
No. The no-op branch has the same length and stop rules as the intervention branch: H_max from the branch tick.

**Does one episode contain many ticks?**
A main rollout contains many ticks and several branch points. A branch has many control steps and about 15 ticks at H_max = 3 s, but exactly one decision, at its start, and produces exactly one label.

**Is the timeout measured in wall-clock time?**
No. All timing is in control steps (20 per second), so pi0's inference speed doesn't change the labels. Restoring a snapshot also restores the step counter.

**Is the replay buffer cleared after each training step?**
No. Branches cost real simulation, so samples are kept (capacity 20,000) and the learner trains continuously on minibatches of 32.

**When are the levelling probes run?**
Once, before training (phase 0). They check whether treating the centre cell as a no-op is valid. Their labels are not used for training.

## 15. Alternative: a separate V network

Instead of reading the no-op value from the map's centre entry, a separate network **V(s)** can model it. The trigger becomes:

$$
A(s) = \max_{a \in \mathcal{A}_r} \big[\mu_a(s) - \kappa\,\sigma_a(s)\big] - V(s)
$$

**What changes.**

- V is a small CNN on the same heightmap and proprio. It outputs a scalar and is trained on the no-op labels only.
- It is trained first, then frozen while the Q-map trains.
- The centre cell at 0° becomes an ordinary correction ("level only") that can be executed. Its value against V shows what levelling alone is worth, so the phase 0 check is no longer a precondition.
- V can also use denser no-op labels, for example from every gated tick of the main rollout, without starving the Q-map, since the two networks don't share batches.

**When to switch.**

- Phase 0 shows that levelling alone changes the outcome at the tilts pi0 reaches.
- Q0's calibration suffers from sharing the trunk with the correction entries, for example if it drifts while the corrections train.
- Debugging the trigger: with two networks, it is clear which side was wrong when the trigger misfires.

**Cost.** The two networks no longer share errors. If the Q0 approach misjudges a scene's difficulty, Q0 and the corrections shift together and the error cancels in A; with separate networks it does not. Check the V-vs-Q offset at branch points: a constant offset is absorbed by τ, but one that varies from state to state is not.

**Merged form.** If the two need to share a trunk again, use a dueling architecture, Q(s, a) = V(s) + Adv(s, a). The scalar V head is trained on no-op labels, V + Adv(a) on intervention labels, and the trigger thresholds Adv directly.
