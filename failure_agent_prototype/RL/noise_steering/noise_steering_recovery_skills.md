# Noise-Space Steering Skills for Pi0 Failure Recovery (ECE780)

Low-level failure recovery for the ECE780 agentic-control project. Pi0 stays completely frozen. Each recovery skill is a small, stand-alone policy stored in the failure recovery skill library. Training uses only a binary subtask-success reward, with no learned reward model.

## 0. Core idea

Pi0 generates an action chunk by integrating a flow from an initial Gaussian noise tensor `z ~ N(0, I)` of shape `(H, D)` (chunk length x padded action dim, e.g. 50 x 32). A steering skill replaces that random noise with noise chosen by a small policy `pi_skill(z | context)`. The RL "action" is the noise, not the robot action. Its log-likelihood is just a Gaussian log-density, so flow matching causes no difficulty, and gradients never pass through pi0.

Initializing the steering policy as `N(0, I)` makes it exactly equal to pi0. Every skill therefore starts from pi0's behavior and can only move away from it as far as the data justifies.

Working hypothesis: LIBERO-90 is only slightly out of distribution for pi0, so successful behaviors already exist in pi0's output distribution. Step 1 tests this hypothesis per failure mode.

## 1. What the steering policy is conditioned on

The steering policy is conditioned on everything pi0 itself sees: camera images, the subtask instruction, and robot proprioception. Conditioning on the instruction and visual context is what makes a skill reusable across tasks and objects, rather than memorizing one scene.

The recommended feature source is pi0's own VLM prefix. Pi0 already runs its PaliGemma backbone on the images plus instruction to build the KV cache for the action expert. Mean-pooling the final-layer prefix hidden states gives a ~2048-d vector that encodes both scene and instruction, at no extra compute. In pi0, proprioception enters the action expert (suffix), not the prefix, so it is concatenated separately. This requires a small hook into pi0 (grey-box access). It also ties skills to a specific pi0 checkpoint, which matters for the post-training stage (see Section 8).

The black-box alternative is the original DSRL design: a small CNN on downsampled images (e.g. 64x64), an MLP on proprioception, plus a frozen sentence embedding of the instruction. It is fully decoupled from pi0's internals, at the cost of weaker features.

There is no failure-mode ID input. Each failure mode gets its own policy, and the FR agent selects which one to load.

```python
class SteeringPolicy(nn.Module):
    """Gaussian policy over pi0's initial flow noise z in R^{H x D}."""
    def __init__(self, feat_dim=2048, proprio_dim=8, H=50, D=32, hidden=512):
        self.H, self.D = H, D
        self.trunk   = MLP(feat_dim + proprio_dim, [hidden, hidden], act=GELU)
        self.mu_head = Linear(hidden, H * D)
        zero_init(self.mu_head)                     # mean = 0 at init
        self.log_std = Parameter(zeros(H * D))      # std = 1 at init -> policy == pi0

    def forward(self, feat, proprio):
        h   = self.trunk(concat(stop_grad(feat), proprio))
        mu  = self.mu_head(h).view(self.H, self.D)
        std = exp(clamp(self.log_std, -4.0, 0.5)).view(self.H, self.D)
        return Normal(mu, std)                       # independent per element

GaussianPrior = lambda feat, proprio: Normal(zeros(H, D), ones(H, D))   # = plain pi0
```

Pi0 wrapper used everywhere below. Recent openpi versions let `sample_actions` take an explicit noise argument; if yours does not, patch it in.

```python
def pi0_encode(pi0, obs, instruction):
    kv_cache, prefix_hidden = pi0.embed_and_run_prefix(obs.images, instruction)  # frozen VLM
    feat = mean_pool(prefix_hidden)                                              # (2048,)
    return kv_cache, feat

def pi0_decode(pi0, kv_cache, proprio, z):
    chunk = pi0.integrate_flow(kv_cache, proprio, init_noise=z)   # frozen action expert
    return chunk[:, :7]                                            # LIBERO uses 7-d actions
```

Shared rollout function. One policy decision is made per chunk, and only the first `n_exec` actions of each chunk are executed before replanning. The subtask success check is the binary reward from the LIBERO predicates.

```python
def rollout(env, start_state, subtask, policy, max_chunks=10, n_exec=5, deterministic=False):
    obs  = env.reset_to(start_state)
    traj = []                                    # (feat, proprio, z, logp_old)
    for _ in range(max_chunks):
        kv, feat = pi0_encode(pi0, obs, subtask.instruction)
        dist     = policy(feat, obs.proprio)
        z        = dist.mean if deterministic else dist.sample()
        traj.append((feat, obs.proprio, z, dist.log_prob(z).sum().detach()))
        for a in pi0_decode(pi0, kv, obs.proprio, z)[:n_exec]:
            obs = env.step(a)
            if subtask.success(env):             # binary reward
                return traj, 1
    return traj, 0
```

In practice, run several LIBERO environments in subprocesses and batch the pi0 calls. The single-environment form above is for clarity.

## 2. Step 0: Build the failure-state bank

During the failure survey (LIBERO-spatial, LIBERO-10, training half of LIBERO-90), save simulator states in a short ring buffer. When the FR agent flags a low-level failure, store a state from slightly *before* the failure. The exact moment of failure is often unrecoverable, e.g. the object has already been knocked off the table.

```python
failure_bank = defaultdict(list)                 # failure_mode_id -> [FailureState]

for task in train_tasks:
    for ep in range(n_episodes):
        env.reset(task.init_state(ep))
        for subtask in coordinator.decompose(task):
            ring = RingBuffer(max_seconds=3)
            outcome = run_pi0_monitored(env, subtask, ring, fr_agent)   # plain pi0, N(0,I) noise
            if outcome.failed and outcome.mode.level == "low":
                s = ring.state_at(outcome.t_fail - backoff)            # e.g. backoff = 0.5-1.0 s
                failure_bank[outcome.mode.id].append(
                    FailureState(sim_state=s, task=task, subtask=subtask))
            if outcome.failed:
                break

# Optional augmentation: jitter object poses / EE pose slightly around each saved state.
```

Hold out the failure states from a few training tasks as a validation split. This measures transfer to unseen tasks within the training half before touching the held-out LIBERO-90 tasks.

## 3. Step 1: pass@k diagnostic and routing

For each failure mode, sample plain pi0 `n` times from each saved state and count successes. If pass@1 is low but pass@k is high, success is within pi0's distribution, so steering will work. If pass@k is also near zero, pi0 cannot solve it from that state, so route the mode to a residual skill (Section 7).

```python
def pass_at_k(n, c, k):                          # unbiased estimator
    return 1.0 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0

route = {}
for mode, states in failure_bank.items():
    p1, pk, coverage = [], [], []
    for s in sample(states, M):                  # e.g. M = 20 states
        c = sum(rollout(env, s.sim_state, s.subtask, GaussianPrior)[1] for _ in range(n))  # n = 32
        p1.append(pass_at_k(n, c, 1)); pk.append(pass_at_k(n, c, k)); coverage.append(c > 0)
    route[mode] = "steer" if mean(pk) >= tau and mean(coverage) >= tau_cov else "residual"
    log(mode, pass1=mean(p1), passk=mean(pk), coverage=mean(coverage))
```

This step uses only the binary reward and is a reportable result in itself: it measures how out of distribution each failure mode actually is.

## 4. Step 2a: Success-filtered noise regression (simplest; no critic)

This is iterated rejection sampling plus maximum likelihood, the ReST/RAFT recipe applied in noise space. Sample noise from the current policy, keep the (context, noise) pairs from successful episodes, and fit the policy to them. The first iteration is pure rejection sampling from pi0.

```python
policy  = SteeringPolicy()                       # starts identical to pi0
success = ReplayBuffer(max_size=200_000)

for it in range(n_iters):
    for s in sample(failure_bank[mode], B):      # e.g. B = 32 states
        for _ in range(G):                       # e.g. G = 8 samples per state
            traj, r = rollout(env, s.sim_state, s.subtask, policy)
            if r == 1:
                success.extend([(f, p, z) for (f, p, z, _) in traj])

    for _ in range(n_epochs):
        f, p, z = success.sample(batch_size)
        dist = policy(f, p)
        nll  = -dist.log_prob(z).sum(dim=(-2, -1)).mean()
        kl   = kl_divergence(dist, Normal(0, 1)).sum(dim=(-2, -1)).mean()
        loss = nll + beta * kl                    # beta small, e.g. 1e-3
        step(optimizer, loss)

    evaluate(policy, val_states, deterministic=True)
```

The KL term toward `N(0, I)` keeps the noise in the region the flow model was trained on. Noise far from the Gaussian shell can produce garbage actions. The `log_std` floor prevents premature collapse.

## 5. Step 2b: Group-relative policy gradient (critic-free)

Saved failure states give natural groups: `G` rollouts from the same state. The advantage is each rollout's binary reward standardized within its group, so failed samples actively push probability away from bad noise. No value function is learned.

```python
policy_old = copy(policy)

for it in range(n_iters):
    batch = []
    for s in sample(failure_bank[mode], B):
        group = [rollout(env, s.sim_state, s.subtask, policy_old) for _ in range(G)]
        rs = tensor([r for _, r in group], dtype=float)
        if rs.std() == 0:                         # all succeed or all fail: no signal
            continue
        adv = (rs - rs.mean()) / (rs.std() + 1e-6)
        for (traj, _), a in zip(group, adv):
            batch += [(f, p, z, logp_old, a) for (f, p, z, logp_old) in traj]

    for _ in range(n_epochs):
        for f, p, z, logp_old, a in minibatches(batch):
            dist  = policy(f, p)
            logp  = dist.log_prob(z).sum(dim=(-2, -1))
            ratio = exp((logp - logp_old) / (H * D))     # per-element normalization for stability
            pg    = -min(ratio * a, clip(ratio, 1 - eps, 1 + eps) * a).mean()
            kl    = kl_divergence(dist, Normal(0, 1)).sum(dim=(-2, -1)).mean()
            step(optimizer, pg + beta * kl)

    policy_old = copy(policy)
```

Importance ratios over a 1600-dimensional Gaussian are sensitive. Normalizing the log-ratio by the dimension (as above), keeping the learning rate small, and using a tight clip (`eps` around 0.1 to 0.2) keep updates stable. If it is still noisy, reduce the steered dimensionality, for example by predicting the mean in a low-rank subspace.

## 6. Step 2c (optional): DSRL with SAC

Use this only if 2a and 2b are too sample-hungry. The latent-action MDP has state = (feat, proprio), action = z (tanh-bounded), and reward = binary subtask success. The critic `Q(feat, proprio, z)` is trained on the true binary reward, so this is value learning, not reward modelling. Reference implementations: `nakamotoo/dsrl_pi0` (JAX) and the DSRL example in RLinf (PyTorch).

```python
for env_step in range(N):
    kv, feat = pi0_encode(pi0, obs, instr)
    z        = actor.sample(feat, obs.proprio)                # tanh-squashed Gaussian
    obs2, r, done = execute_chunk(env, pi0_decode(pi0, kv, obs.proprio, z)[:n_exec])
    replay.add(feat, obs.proprio, z, r, feat2, obs2.proprio, done)
    sac_update(actor, critics, replay)                         # standard SAC
```

## 7. Fallback: residual skills for out-of-support failures

For modes routed to `"residual"`, the policy outputs a bounded correction added to pi0's actions. It additionally conditions on pi0's proposed chunk. Training reuses exactly the losses from 2a or 2b, with `delta` in place of `z`.

```python
class ResidualPolicy(nn.Module):
    def forward(self, feat, proprio, base_chunk):              # base_chunk: (n_exec, 7)
        h  = self.trunk(concat(stop_grad(feat), proprio, base_chunk.flatten()))
        return Normal(self.mu_head(h).view(n_exec, 7), exp(self.log_std))   # mu zero-init

# inside rollout:
base  = pi0_decode(pi0, kv, obs.proprio, randn(H, D))[:n_exec]
delta = residual(feat, obs.proprio, base).sample()
for a in base + alpha * tanh(delta):                           # alpha ramped from 0 to ~0.1-0.2
    obs = env.step(a)
```

The KL regularizer becomes a penalty pulling `delta` toward zero, which keeps the skill close to pi0 by default.

## 8. High-level lever: instruction rephrasing (no training)

Pi0 was fine-tuned on full LIBERO task instructions, not atomic commands like `pick_up(red_block)`. The coordinator should phrase each subtask in LIBERO-spatial/10 style. Phrasings are selected with the same binary reward on training states.

```python
examples = sample(libero_spatial_and_10_instructions, 20)

def candidate_phrasings(subtask, n=5):
    return vlm(f"Rewrite '{subtask}' as a robot instruction in the style of: {examples}", n=n)

for subtask_type in subtask_types:
    scores = {ph: mean(rollout(env, s, with_instr(s.subtask, ph), GaussianPrior)[1]
                       for s in states_for(subtask_type))
              for ph in candidate_phrasings(subtask_type)}
    phrasing_table[subtask_type] = argmax(scores)
```

Run this *before* collecting the failure bank, so that skills are trained on failures that remain after the cheap fix.

## 9. Deployment in the closed loop

At test time no simulator reward is used. Subtask completion and failure are judged by the FR agent and coordinator, which keeps the zero-shot evaluation fair.

```python
for subtask in coordinator.plan(task):
    instr = rephrase(subtask)
    outcome = run_pi0_monitored(env, instr, fr_agent)          # 1 fps screenshots + proprio
    retries = 0
    while outcome.failed and retries < max_retries:
        mode = fr_agent.classify(outcome)
        if mode.level == "high":
            outcome = apply_high_level_recovery(mode, library)
        else:
            skill   = library.load(mode.id)                    # SteeringPolicy or ResidualPolicy
            outcome = run_skill_monitored(env, instr, skill, fr_agent,
                                          max_chunks=T_rec, deterministic=True)
        retries += 1
```

On post-training: once pi0 is fine-tuned on successful trajectories (stage 6), skills that use pi0's prefix features or steer its noise were trained against the old checkpoint. Either re-validate them with the pass@k check, or keep the original pi0 frozen as the skill backbone.

## 10. Skill library entry

```yaml
skill_id: grasp_misalign_bowl_v1
failure_mode: "gripper closes off-center on bowl rim; bowl slips during lift"
level: low
detection_signature: "gripper closed, object height unchanged after lift, EE offset from rim"
skill_type: steer            # steer | residual
trained_with: filtered_regression   # filtered_regression | group_pg | dsrl_sac
pi0_checkpoint: pi0_libero_base
conditioning: [pi0_prefix_pool_2048, proprio_8]
checkpoint: skills/grasp_misalign_bowl_v1.pt
training_tasks: [libero_spatial_*, libero_10_3, libero_90_17, ...]
diagnostics: {pass1_before: 0.12, passk16_before: 0.71, success_after: 0.83, val_success_after: 0.64}
```

## 11. Starting hyperparameters and sanity checks

| Parameter | Starting value |
|---|---|
| Chunk length H / noise dim D | 50 / 32 (match your pi0 config) |
| Executed actions per chunk `n_exec` | 5 |
| Max chunks per recovery episode | 10 |
| States per iteration B / samples per state G | 32 / 8 |
| KL weight `beta` | 1e-3 |
| PPO clip `eps` | 0.1 to 0.2 |
| Learning rate | 3e-4 (2a), 1e-4 (2b) |
| Residual scale `alpha` | ramp 0 to 0.1-0.2 |

Sanity checks before any training: the zero-initialized steering policy should reproduce plain pi0's success rate on the failure bank (within noise), and deterministic evaluation with `z = 0` should be measured as a separate baseline, since the mean noise alone sometimes changes success rate noticeably.
