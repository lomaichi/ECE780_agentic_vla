# Residual RL Plan for Pi0 Low-Level Failure Recovery (ECE780)

Companion to `../noise_steering/noise_steering_recovery_skills.md` (Section 7 there is the one-paragraph sketch this plan expands). Pi0 stays completely frozen. Each low-level (LL) skill is a small residual policy that adds a bounded correction to pi0's actions. Training uses only binary success predicates from the simulator, with no learned reward model. At test time the FR agent loads a skill through `INVOKE_RL(policy_id)`.

## 0. Why residual, and what the evidence says

**Task-level pass@k test** (`Documentation/noise_steering_test`, 2026-11-10). Plain pi0 was run from a fixed scene (init 0, env seed 7) 32 times per task, varying only the flow noise, with a 40 s limit:

| libero_90 task | successes | pass@32 | routing |
|---|---|---|---|
| 1: close top drawer, bowl on top | 0/32 | 0 | residual |
| 25: black bowl on top of cabinet | 0/32 | 0 | residual |
| 27: wine bottle on wine rack (**held out**) | 0/32 | 0 | test only |

From the start state, success is outside pi0's noise distribution, so noise steering has nothing to amplify. Residual RL can move the actions outside pi0's support.

**Caveat.** This test covered the whole task. The skills here are per subtask and start from a state just before the failure. Step 1 re-runs pass@k at that level, and a subtask with pass@k > 0 can still be routed back to noise steering, which is cheaper and safer.

**Split guard.** RL environments, reset states and demos come only from the 45 survey-split libero_90 tasks (`library.json -> split_guard`), plus LIBERO-spatial and LIBERO-10. **Task 27 is held out.** It was used only for the plain-pi0 diagnostic above and must not contribute reset states, demos or tuning.

## 1. Skills to train (from `library.json -> rl_policies`)

| policy_id | covers | reward predicate (training only) | horizon | reset states |
|---|---|---|---|---|
| `rl/close_drawer` | LL-1 drawer_push_stall | LIBERO predicate of task22 (drawer closed) | 120 | task1_fail_init0 @60, task22_fail_init1 @280 |
| `rl/pick` | LL-4 slip, LL-5 disturbance, LL-8 hesitation | Grasped(obj) AND lifted ≥ 5 cm for 10 steps | 100 | 14 states (incl. task25_fail_init1 @160) |
| `rl/place` | LL-6 instability, LL-7 release stall | On/In(obj, dest) AND gripper open AND EE retreated | 100 | 5 states + new task25 "bowl held above cabinet" states (Step 0) |
| `rl/turn_knob` | LL-3 | LIBERO predicate of task44 / task39 | 120 | task44 init0 @167, init1 @180 |
| `rl/open_door` | LL-2 | LIBERO predicate of task35 | 200 | task35_fail_init0 @160 |

**Order:** `rl/close_drawer` first (one clean LIBERO predicate, and it is task 1's first subtask), then `rl/place` (task 25's observed failure: the bowl is carried to the cabinet but never placed), then `rl/pick`. `rl/turn_knob` and `rl/open_door` come last; they have the fewest reset states.

These specs currently say `method: "SimpleVLA-RL on pi0"`. Section 9 explains how the two relate. Update each entry's `method` once a residual skill is validated.

## 2. Residual interface

The correction is applied per control step, but decided once per executed chunk, so it updates at pi0's replan rate. Every 5 steps pi0 proposes a chunk; only `base = chunk[:5]` (5 x 7) is executed, after adding the correction:

```
a_t = clip(base_t + alpha * tanh(delta_t), -1, 1)        t = 0..4
```

- The LIBERO action is a 7-D OSC delta (dpos 3, drot 3, gripper 1) in [-1, 1], where 1 is about 5 cm per step of translation. With `alpha = 0.2`, the residual can move the EE up to about 1 cm per step, which is enough to re-aim a grasp or push and not enough to fling the arm.
- **Gripper:** this channel gets its own scale `alpha_g = 1.0`, so the residual can flip open/close. LL-7 (never releases) and LL-8 (never closes) are gripper-timing failures that a small residual cannot fix.
- One residual decision per chunk keeps the RL horizon at about 20–40 decisions instead of 100–200, which helps credit assignment with a binary reward.

### Conditioning (black-box, works with the current websocket server)

The current server only returns actions, so start black-box:

- agentview and wrist images, 128 x 128, through a **frozen** pretrained encoder (ResNet-18 ImageNet, or DINOv2-S), cached per decision;
- 8-D proprio (the same state pi0 receives);
- pi0's proposed `base` chunk (5 x 7, flattened). This is what lets the residual act as a correction rather than a replacement policy;
- the subtask instruction as a frozen sentence embedding (computed once per episode).

Upgrade path: pi0's mean-pooled prefix features (2048-d), as in the steering doc Section 1. This needs a server hook that also returns them, and it ties skills to the pi0 checkpoint (see Section 10).

```python
class ResidualPolicy(nn.Module):
    """Gaussian policy over a bounded correction to pi0's executed chunk."""
    def __init__(self, img_dim=512, txt_dim=384, proprio_dim=8, n_exec=5, act_dim=7, hidden=512):
        self.n_exec, self.act_dim = n_exec, act_dim
        in_dim = 2 * img_dim + txt_dim + proprio_dim + n_exec * act_dim
        self.trunk   = MLP(in_dim, [hidden, hidden], act=GELU, layernorm=True)
        self.mu_head = Linear(hidden, n_exec * act_dim)
        zero_init(self.mu_head)                                   # mean correction = 0 -> pi0 at init
        self.log_std = Parameter(full((n_exec * act_dim,), log(0.3)))   # exploration std (pre-tanh)

    def forward(self, f_agent, f_wrist, f_txt, proprio, base):
        h   = self.trunk(concat(f_agent, f_wrist, f_txt, proprio, base.flatten()))
        mu  = self.mu_head(h).view(self.n_exec, self.act_dim)
        std = exp(clamp(self.log_std, -5.0, 0.0)).view(self.n_exec, self.act_dim)
        return Normal(mu, std)

ALPHA = tensor([a, a, a, a, a, a, 1.0])  # a ramped 0 -> 0.2 (Section 6); gripper channel unscaled

def apply_residual(base, delta):
    return clip(base + ALPHA * tanh(delta), -1, 1)
```

Unlike steering, the zero-mean init equals pi0 only when sampling deterministically. A stochastic residual with std 0.3 is exploration around pi0. That exploration is the point: it can reach actions pi0 never proposes.

## 3. Step 0: Infrastructure and the reset-state bank

1. **Scene restore.** Use `tools/scene_restore.py:reset_to(env, trace, step)`. It restores the sim state plus the controller, `gripper.current_action` and `qacc_warmstart`; `set_init_state` alone drifts. Run `scene_restore.py --check --policy <id>` for each policy before training, and reject any reset state that fails the replay check.
2. **New reset states for `rl/place` (task 25).** The diagnostic kept only videos. Re-record about 16 task25 init0/init1 rollouts with traces (`tools/record_suites.py` settings). Pick the steps where the bowl is grasped and above or beside the cabinet top, using the predicate Grasped(bowl) and EE within 10 cm of the cabinet top. Do the same for any survey-split rollout that reaches "object held above destination".
3. **Subtask predicates.** Add `pick_success`, `place_success` and `drawer_closed` as functions over privileged sim state: object poses, the contact-based `Grasped` from LIBERO, and drawer joint qpos. They are used only for reward and evaluation in training, never at test time.
4. **Vectorized rollout.** Use N LIBERO subprocess envs (start with 6, as in the pass@k test). Pi0 calls are batched to the server. Image features are computed client-side on the GPU in one batch per decision step.
5. **Augmentation.** Around each reset state, jitter the EE pose (±1 cm, ±3°) by a short scripted OSC move after restore. Optionally jitter the object pose (±1 cm) when it is not grasped. This turns 2–14 states into a usable training distribution and prevents memorizing one trajectory.
6. **Validation split.** Per policy, hold out the reset states from one task, e.g. task22 for `rl/close_drawer` (train on task1), so validation measures transfer across scenes.

```python
def rl_rollout(envs, resets, policy, horizon, predicate, deterministic=False):
    obs = [reset_to_jittered(e, r) for e, r in zip(envs, resets)]
    traj = [[] for _ in envs]; done = [False] * len(envs)
    for _ in range(horizon // N_EXEC):
        base  = pi0.infer_batch(obs, instruction)[:, :N_EXEC]        # frozen, fresh noise
        ctx   = encode(obs, instruction, base)                        # frozen features
        dist  = policy(*ctx)
        delta = dist.mean if deterministic else dist.sample()
        for i, e in enumerate(envs):
            if done[i]: continue
            traj[i].append((ctx[i], delta[i], dist.log_prob(delta).sum((-2, -1))[i].detach()))
            for a in apply_residual(base[i], delta[i]):
                obs[i] = e.step(a)
                if predicate(e): done[i] = True; break
    return traj, [float(d) for d in done]
```

## 4. Step 1: Subtask-level explorability check (routing)

For each policy and reset state, run three samplers. n = 32 rollouts each, at the policy's horizon:

| sampler | what it measures | if pass@k > 0 … |
|---|---|---|
| plain pi0 (`delta = 0`) | is success in pi0's distribution from this state? | route to **noise steering** (cheaper; see the steering doc) |
| pi0 + random residual, std in {0.3, 0.6}, alpha = 0.2 | can undirected exploration around pi0 find success? | **on-policy residual RL** (Step 2a) |
| neither succeeds | signal too sparse for critic-free RL | **off-policy residual RL with demos and a curriculum** (Step 2b + Section 5) |

Log pass@1, pass@8, pass@32 and coverage (the fraction of reset states with c > 0) per sampler. The table is a reportable result: where each LL failure mode sits relative to pi0's support. It is the subtask-level version of the 0/96 result.

## 5. Exploration aids for zero-success modes

Use these only when the Step 1 random-residual pass@k is about 0. Each keeps the reward binary.

- **Reverse curriculum in time.** Collect reset states closer to the goal and move them backward once success exceeds 50%. Sources: later steps of the same failed trace, or states where the subtask is half done (drawer 2/3 closed, bowl already above the cabinet).
- **Staged binary predicates** for `rl/pick`. Train a short-horizon "reach to pre-grasp pose" stage first, with the predicate "EE within 2 cm of the grasp region, gripper open". Then chain the full pick from that stage's success states. These are still sim predicates, not a learned reward.
- **Demo seeding (RLPD-style).** Many guide demos are pi0 successes in *other* scenes (libero_10 task3 drawer push, libero_spatial bowl picks). They are not residual demos, but scripted or teleop-style demos in the *training* scene can be. Use `POINT_APPROACH` and the scripted OSC moves from the library to write 5–10 scripted successes per policy. Convert each one to residual labels with `delta = atanh(clip((a_demo - base) / ALPHA, -0.99, 0.99))` and put them in a 50/50 demo/online replay.
- **Larger alpha early.** Allow alpha up to 0.4 during the first iterations of a zero-success mode, then anneal it toward 0.2.

## 6. Step 2: Training

### Step 2a: Group-relative PPO (critic-free, on-policy)

This is the same recipe as steering doc Section 5b. Rollouts from the same jittered reset state form a group; the advantage is the binary reward standardized within the group. The action is only 35-D per decision, compared with 1600-D for noise, so importance ratios are well-behaved and no per-dimension normalization is needed.

```python
policy_old = copy(policy)
for it in range(n_iters):
    alpha_schedule(it)                                   # ramp 0 -> 0.2 over the first ~20% of iterations
    batch = []
    for r in sample(reset_bank[policy_id], B):           # B = 32 states
        trajs, rs = rl_rollout(envs_for(r, G), [r] * G, policy_old, H, pred)   # G = 8
        rs = tensor(rs)
        if rs.std() == 0: continue
        adv = (rs - rs.mean()) / (rs.std() + 1e-6)
        for traj, a in zip(trajs, adv):
            batch += [(ctx, d, lp, a) for (ctx, d, lp) in traj]
    for _ in range(n_epochs):                            # 4
        for ctx, d, lp_old, a in minibatches(batch):
            dist  = policy(*ctx)
            ratio = exp(dist.log_prob(d).sum((-2, -1)) - lp_old)
            pg    = -min(ratio * a, clip(ratio, 1 - eps, 1 + eps) * a).mean()
            mag   = (ALPHA * tanh(dist.mean)).pow(2).sum((-2, -1)).mean()    # pull correction to 0
            step(opt, pg + lam * mag)
    policy_old = copy(policy)
    evaluate(policy, val_resets, deterministic=True)
```

### Step 2b: Residual SAC with demos (off-policy, value-based)

Use this when 2a is too sample-hungry or Step 1 says exploration is too sparse. It follows the Policy Decorator / RLPD pattern:

- the MDP state is (ctx), the action is delta (tanh-squashed), the reward is the binary predicate at episode end, and gamma = 0.99 per decision;
- critics `Q(ctx, delta)`, with an ensemble of 2–10 and LayerNorm; the update-to-data ratio is about 4;
- **controlled exploration**: for the first K steps, act with plain pi0 plus a small Gaussian on delta, not the actor, so the replay starts near pi0 behavior; then hand over to the actor;
- 50/50 replay of online data and demos (Section 5).

The critic is trained on the true binary reward, so this is value learning, not a learned reward model, which keeps the project constraint. Reference implementations: Policy Decorator (residual SAC on frozen base policies) and RLPD.

## 7. Regularization and safety

- **Zero-init mean and alpha ramp.** The skill starts as pi0. If success drops while alpha ramps, alpha is too large.
- **Correction-magnitude penalty** `lam * ||alpha * tanh(mu)||^2` (`lam = 1e-2`). This is the residual analogue of the noise-steering KL to N(0, I). It keeps corrections small unless they pay off.
- **std floor** (`log_std >= -5`) to avoid early collapse, and **std ceiling** (`log_std <= 0`) so exploration never dominates pi0.
- **Termination guards** (training only): end the episode with r = 0 if the object falls off the table, the EE leaves the workspace box, or contact forces spike. This is cheaper than running out the horizon.
- **Subtask scope.** A skill acts only for its horizon (100–200 steps), then hands control back to plain pi0, or ESCALATE if `success_check` fails. It never runs a whole task.

## 8. Evaluation

Per policy, on train and validation reset states, 32 deterministic and 32 stochastic rollouts each:

1. Plain pi0 success (the Step 1 baseline), random-residual success, and trained-residual success.
2. **Closed-loop subtask success.** In survey-split rollouts, the FR agent detects the LL mode from 1 fps frames and proprio and invokes the skill. Success is judged by the FR agent, not the sim predicate, as in steering doc Section 9.
3. **Held-out zero-shot (final only).** Measure whole-task success on the held-out libero_90 half, including task 27, with the full system. These tasks are never used for training or tuning.
4. **Sanity checks before training.** A zero-mean residual run deterministically must reproduce plain pi0's success on the reset bank within noise. `scene_restore.py --check` must pass for every reset state.

## 9. Relation to SimpleVLA-RL (LoRA)

`SimpleVLA-RL/` (LoRA, because GPU memory is limited) fine-tunes pi0 itself. That is the **comparison baseline and the fallback**, not the main method:

| | residual RL (this plan) | SimpleVLA-RL LoRA |
|---|---|---|
| pi0 weights | frozen | LoRA-updated |
| skill artefact | about 2–5 M-param MLP per mode, swappable | one adapter per mode, or a shared one |
| GPU | pi0 inference plus a small policy fits on the 5090 | the backward pass through 3B parameters is tight even with LoRA |
| reach outside pi0's support | yes (bounded by alpha) | yes |
| explainability / library fit | explicit per-mode skill, loaded by `INVOKE_RL` | weaker, since it changes base behavior |

Run SimpleVLA-RL on the first policy (`rl/close_drawer`) with the same reset bank and predicate, for a like-for-like comparison of sample efficiency and success.

## 10. Deployment and post-training

- **Library entry.** Each validated skill updates its `rl_policies` entry: `method: residual_ppo | residual_sac`, `available: true`, checkpoint path, `alpha`, conditioning, and Step 1/8 diagnostics.

```yaml
policy_id: rl/close_drawer
skill_type: residual
trained_with: residual_ppo          # residual_ppo | residual_sac
pi0_checkpoint: pi0_libero
conditioning: [resnet18_agent_512, resnet18_wrist_512, minilm_instr_384, proprio_8, pi0_base_chunk_35]
alpha: {pos_rot: 0.2, gripper: 1.0}
horizon_steps: 120
checkpoint: skills/rl_close_drawer_residual_v1.pt
train_resets: [libero_90/task1_fail_init0@60, ...]
val_resets:   [libero_90/task22_fail_init1@280]
diagnostics: {pi0_pass32: 0.0, random_residual_pass32: ..., success_after: ..., val_success_after: ...}
```

- **Closed loop.** On an LL detection the FR agent runs HALT, then `INVOKE_RL(policy_id)`. The skill runs deterministically for at most its horizon, then `success_check`, then a retry or ESCALATE, as in the existing `fallback_until_rl` structure.
- **After pi0 post-training (stage 6).** The residual conditions on pi0's base chunk, so it is tied to the pi0 checkpoint even in the black-box design. Either re-validate each skill against the new checkpoint (a Step 8 rerun), or keep the original pi0 as the base under recovery skills.

## 11. Milestones

| # | deliverable | exit criterion |
|---|---|---|
| M0 | reset bank, predicates, jitter, vectorized env | `scene_restore --check` passes; zero-residual run reproduces pi0 |
| M1 | Step 1 table for all 5 policies | pass@k for pi0 / random residual per reset state |
| M2 | `rl/close_drawer` residual (2a) | val success ≥ 50% from task22 states |
| M3 | `rl/place` (task25 states), `rl/pick` | train success ≥ 60%, val > pi0 baseline |
| M4 | SimpleVLA-RL comparison on `rl/close_drawer` | success and env steps reported side by side |
| M5 | closed-loop FR integration on the survey split | subtask recovery rate reported |
| M6 | held-out zero-shot eval (incl. task 27) | final whole-task success vs. plain pi0 |

## 12. Starting hyperparameters

| parameter | value |
|---|---|
| executed actions per decision `n_exec` | 5 (pi0 replan rate) |
| horizon | per policy, 100–200 steps (20–40 decisions) |
| alpha (pos/rot) | ramp 0 → 0.2; up to 0.4 for zero-success modes (Section 5) |
| alpha (gripper) | 1.0 |
| initial residual std (pre-tanh) | 0.3 |
| B states x G rollouts per iteration | 32 x 8 |
| PPO clip `eps` / epochs / lr | 0.2 / 4 / 3e-4 |
| magnitude penalty `lam` | 1e-2 |
| SAC: critics / UTD / lr / gamma | 2–10 (LayerNorm) / 4 / 3e-4 / 0.99 |
| demo:online replay ratio (2b) | 50:50 |
| image encoder | frozen ResNet-18 (or DINOv2-S), 128 x 128 input |

## 13. Risks

- **Too few reset states** (1–2 for door and knob). Mitigate with jitter and re-recorded survey rollouts on survey-split tasks to harvest more onsets.
- **Residual cannot fix a high-level failure.** If pi0 heads for the wrong target, as seen in the task 1 and task 27 videos, a bounded residual will not redirect it. Those cases belong to the HL primitives (POINT_APPROACH, REPROMPT); the FR agent must separate them before INVOKE_RL.
- **Throughput.** Short horizons (100–200 steps) run at about 6 s per episode per worker, so about 1 episode/s with 6 workers. 10k episodes per policy is about 3 h. Increase the worker count if CPU-bound; pi0 inference is batched.
- **Sim-predicate leakage.** Privileged predicates are for training reward only. Test-time success must come from the FR agent's 1 fps and proprio view.
