# Pixel-wise Q-learning for Pi0 Pick-and-Place Failure Recovery: Proposal Review and Go/No-Go Test (ECE780)

This solution is a middle ground between noise-steering (no new skill, just warm start) and residual learning (new skill). Same as all the proposed solutions, Pi0 stays frozen. A small correction picks *where* the gripper goes on the object (pick), or *where* the held object goes on the destination (place), then control returns to pi0, based on a pixel Q-map. We will call this part "RL corrector"

## 0. Scope

As a proof of concept, the project only apply pixel-wise Q-learning on pick-and-place tasks on bowls or mugs, during which low level failure can be corrected by better gripper alignment. Theoretically, if this framework is successful, it can be applied to other task/failure modes and be stored in a failure recovery skill library. The scope is defined as follows:

For brevity, "Eligible task" refers to those that satisfy:
- **Pick-and-place failures only.** Push/pull (drawers, doors),  and knob turning are out of scope. In-scope modes: LL-5 pre-grasp disturbance, LL-8 pre-grasp hesitation (`rl/pick`); LL-6 placement instability, LL-7 release stall (`rl/place`).
- **Round-rim object only** E.g. bowls, mugs, no wine bottle, pan, etc. differently shaped object should have different RL corrector, and thus we only focus on bowl and mug for this scope
- **LIBERO-90 pick-and-place-only tasks** To avoid too many factors in the final task success/failure, a task is only in scope if its instruction contains none of "open", "close", "turn on", "turn off". Judgment call such as "put X in the top/bottom drawer" tasks are kept, assuming the drawer starts open. Compound tasks such as "put X in Y and do Z" are included

This is how the data will be split. See the specific task list from appendix
  - training split (for online RL on LIBERO): 70% split of LIBERO-90 (unseen, mostly failed rollouts) and all LIBERO-spatial (seen, provides success rollouts), eligible tasks only
  - held-out split (final zero-shot eval only): pick-and-place only tasks from 
  - Compound tasks with a fixture step (e.g. 1, 3, 8, 21, 23, 45) are excluded.

## 1. The idea

Pixel-wise Q-learning, as in Zeng et al. (VPG) and Transporter Networks. A CNN outputs a Q-map over the image.

In our implementation, we only implement 3DOF operational-space end-effector pose adjustment (in-plane translation and rotation) instead of 7DOF arm pose. At each pixel, there is only one motion primitive, that is to execute pi0. There are 9 rotation bin, spaced by 5deg up to 20deg in both cw and ccw direction. Therefore there are 1x9=9 Q maps in total. During inference, all 9 maps will be computed at each frame, and the robot will be commanded to move to the max Q pixel plus rotate by the corresponding rotation bin, then execute pi0.

Model input is language instruction, RGB image (256x256 resolution), and robot pose.

## 2. Inference

1. A segmentation mask (80x80) selects the pick object. grey out everything outside this mask
2. The network outputs Q(u, v, θ) for each 5×5-pixel cell and each of 9 rotation bins, where (u,v) is the pixel coordinate in the image's local frame, and θ is the rotation bin
3. Take the argmax pixel. Back-project pixel + depth + camera intrinsics to a 3D point, then run a `POINT_APPROACH` to it
4. Hand control back to pi0. Corrections are issued less often than pi0 replans; pi0 runs 3 chunks after each correction.
5. After each correction + pi0 3-chunk episode, the coordinator VLM agent will determine whether the current subtask has been succeeded

## 3. Training

Each corrector (`rl/pick`, `rl/place`) is trained separately with the same DQN loop below; only the segmented object and the subtask being judged differ. One time step $t$ is one **correction decision**, not one pi0 action: the corrector picks $(u, v, \theta)$, the robot runs `POINT_APPROACH`, pi0 runs 3 chunks, and the VLM coordinator judges the subtask. Only training-split tasks are used; the held-out split is never touched during training.

**Notation**
- State $s_t = (I_t, D_t, \ell, p_t)$: RGB image $I_t$ (256×256, everything outside the mask $M_t$ greyed out), depth $D_t$, language instruction $\ell$, end-effector pose $p_t$.
- Action $a_t = (u, v, k)$: a 5×5-pixel cell $(u, v)$ and a rotation bin $k \in \{0, \dots, 8\}$, with $\theta_k \in \{-20°, -15°, \dots, +15°, +20°\}$.
- Valid actions $\mathcal{A}(s_t) = \{(u, v, k) : (u, v) \in M_t\}$, i.e. only cells on the segmented object.
- $Q_\theta(s_t)$ is a $9 \times H \times W$ tensor (one Q-map per rotation bin); $Q_\theta(s_t, a_t)$ is a single entry of it.

---

### Pixel-wise DQN Corrector: Training Algorithm

**Initialization**
- Initialize replay buffer $\mathcal{D}$ with capacity $N$.
- Initialize online Q-network $Q_\theta(s, a)$ (pixel-wise CNN, 9 output maps) with random weights $\theta$.
- Initialize target network $Q_{\theta^-}$ with $\theta^- \leftarrow \theta$.
- Load frozen pi0 $\pi_0$, the segmentation model, and the VLM coordinator.

**For each episode:**
- Sample a task and a reset state from the training split.
- Run $\pi_0$ until the coordinator hands off: at the start of the pick subtask (`rl/pick`) or once the object is held (`rl/place`). Observe $s_0$.

**For each correction step $t = 0, 1, \dots, T_{\max}-1$:**

❶ **Select an action ($\epsilon$-greedy, inside the mask):**

$$
a_t =
\begin{cases}
\text{uniform random } (u, v, k) \in \mathcal{A}(s_t), & \text{with probability } \epsilon, \\
\arg\max_{a \in \mathcal{A}(s_t)} Q_\theta(s_t, a), & \text{otherwise.}
\end{cases}
$$

❷ **Execute the correction, then hand back to pi0:**
- Back-project the cell centre to a 3D target point:

$$
x_t = T^{\text{world}}_{\text{cam}} \; D_t(u, v)\, K^{-1} [\,u,\ v,\ 1\,]^\top
$$

- Rotate gripper yaw by $\theta_k$ and run `POINT_APPROACH` to $x_t$.
- Run $\pi_0$ for 3 chunks.

❸ **Interact with the environment:**
Observe reward $r_t$ and next state $s_{t+1}$ (with new mask $M_{t+1}$).

$$
r_t =
\begin{cases}
1, & \text{if the VLM coordinator judges the subtask succeeded}, \\
0, & \text{otherwise.}
\end{cases}
$$

$$
\text{done}_t =
\begin{cases}
1, & \text{if subtask succeeded, or irrecoverable failure (object dropped / out of workspace)}, \\
0, & \text{otherwise.}
\end{cases}
$$

Reaching $T_{\max}$ ends the episode but is **not** terminal: $\text{done}_t = 0$, so the target still bootstraps.

❹ **Store the transition in replay buffer:**

$$
(s_t,\ a_t,\ r_t,\ s_{t+1},\ \text{done}_t) \in \mathcal{D}.
$$

❺ **Sample a minibatch** of $B$ transitions from $\mathcal{D}$ (only once $|\mathcal{D}| \ge B_{\text{warmup}}$).

---

### Pixel-wise DQN Corrector: Training Algorithm (Continued)

**For each sampled transition $(s_j, a_j, r_j, s'_j, \text{done}_j)$:**

❻ **Compute the target value:**

$$
y_j =
\begin{cases}
r_j, & \text{if } \text{done}_j = 1, \\
r_j + \gamma \displaystyle\max_{a' \in \mathcal{A}(s'_j)} Q_{\theta^-}(s'_j, a'), & \text{otherwise.}
\end{cases}
$$

The max runs over all 9 rotation maps and all cells inside the mask of $s'_j$.

❼ **Update the online Q-network:**
Minimize the squared temporal-difference error:

$$
\mathcal{L}(\theta) = \frac{1}{B} \sum_{j=1}^{B} \big[\, y_j - Q_\theta(s_j, a_j) \,\big]^2 .
$$

Only the executed entry $a_j = (u_j, v_j, k_j)$ receives gradient; every other pixel and rotation bin of the $9 \times H \times W$ output is masked out of the loss (as in VPG).

Update the parameters using gradient descent:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta).
$$

❽ **Update the target network periodically:**
Every $C$ steps, copy the online network parameters:

$$
\theta^- \leftarrow \theta.
$$

❾ **Advance:**
If $\text{done}_t = 1$ or $t = T_{\max}-1$, end the episode; otherwise $s_t \leftarrow s_{t+1}$ and go to ❶.

---

**Training notes**
- **Two correctors, two buffers.** `rl/pick` and `rl/place` each have their own $Q_\theta$, $Q_{\theta^-}$ and $\mathcal{D}$; a transition goes only into the buffer of the corrector that produced it.
- **Sparse reward.** Most LIBERO-90 rollouts fail, so LIBERO-spatial tasks are mixed into the task sampler to supply positive rewards early. $\epsilon$ is annealed from high to low over training.
- **Reward source.** VLM judging errors go straight into $y_j$. During training, LIBERO's ground-truth success predicates can be used as the reward (or to audit the VLM), keeping the VLM for inference only.
- **Hyperparameters to set:** $N$, $B$, $B_{\text{warmup}}$, $C$, $\gamma$, $\alpha$, $\epsilon$ schedule, $T_{\max}$. Since a subtask needs only a few corrections, a small $\gamma$ (VPG used 0.5) is a reasonable start.

## 4. Apendix
  - survey split (training / RL / reset states), 
        34 tasks from LIBERO90: 9, 10, 14, 17, 19, 25, 31, 34, 37, 38, 41, 42, 49, 53, 57, 58, 59, 61, 62, 63, 64, 67, 69, 71, 72, 73, 77, 78, 80, 85, 86, 87, 88, 89
        pick and place only tasks from LIBERO-spatial (Q-learning on this helps provide more successful trials)
  - held-out split (final zero-shot eval only), 37 tasks: 2, 12, 13, 15, 16, 18, 24, 26, 27, 29, 30, 32, 36, 40, 43, 46, 47, 48, 50, 51, 52, 54, 55, 56, 60, 65, 66, 68, 70, 74, 75, 76, 79, 81, 82, 83, 84
  - Judgment call: "put X in the top/bottom drawer" tasks (2, 24, 26, 29, 32) are kept, assuming the drawer starts open. They are all held out.
  - Compound tasks with a fixture step (e.g. 1, 3, 8, 21, 23, 45) are excluded.

