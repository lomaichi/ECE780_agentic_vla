# Residual RL on frozen pi0 (low-level failures only)

pi0 stays a frozen black box. A small residual policy runs **always on, in series** with it, and corrects
every action pi0 sends to the robot:

```
VLM       (1 Hz)           what to do   : CONTINUE / SetPrompt / Override / Stop   (high-level failures)
pi0       (every 5 steps)  how to do it : action chunk
residual  (20 Hz)          fine-tuning  : a = a_pi0 + clip(Δa, -ε, ε)  ->  robot    (low-level failures)
```

- **Scope.** Low-level failures only, such as a grasp that is a bit off. Wrong-object failures need corrections far outside the ±ε bound, and fixing them means understanding the language and the scene. The VLM agent handles those.
- **No failure detection.** The residual sees the wrist camera and pi0's action at every step, so it never needs the VLM to flag a low-level failure. To make always-on safe:
  - ε caps the harm in any state, even ones the residual never trained on;
  - a penalty on |Δa| plus training on states where pi0 already succeeds teaches it to output Δa ≈ 0 when nothing is wrong.
- **With the VLM loop** (`fa_core/loop.py`):
  - the residual corrects pi0's actions under CONTINUE and SetPrompt;
  - it is off during Override (scripted motions);
  - the VLM stays out of the loop during residual training; the reward comes from the sim's pick predicate.

## Files

| File | What |
|---|---|
| `residual_env.py` | `ResidualPickEnv`: restores a pick start state, runs pi0 + bounded per-step residual, reward = `PickCheck` (grasped and lifted 5 cm for 10 states). ε is enforced in the env, not the policy. Reused by the trainer |
| `noise_test.py` | Step 1 go/no-go: pi0 + random bounded residual, no learning |

Reused, not copied:
- `failure_agent_prototype/fa_core` (`pi0_request`, predicates, `scene_restore.reset_to`);
- `failure_agent_prototype/RL/pixel_Q/shared/scene.py` (start states with the held-out guard, target resolution, env cache, pi0 client).

## Step 1: go/no-go (`noise_test.py`)

RL can only reinforce corrections that sometimes work. Before training, check that a bounded random residual on top of pi0 ever turns a pick failure into a success.

- **Fail starts.** The 11 `rl/pick` reset states in `library.json` that fall in the pixel-Q training split. These are states just before a recorded pi0 failure. The three task-21 states are excluded by that split.
- **Ok starts.** `libero_spatial` tasks 0-9, init 0, where pi0 usually picks fine. These are the do-no-harm check.
- **Conditions.** ε ∈ {0, 0.05, 0.1, 0.2, 0.4} × noise correlation ρ ∈ {0, 0.95}. ε = 0 is plain pi0, the baseline.
- **Units.** OSC input units: 1.0 = 5 cm or 0.5 rad of commanded motion per step. Rotation uses ε/2. The gripper is untouched by default. For scale, pi0's own mean |a| is about 0.2 on position dims and about 0.03 on rotation dims.
- **Run size.** 16 episodes per (start, condition), 200 steps (10 s) each.

Inside the container (`./libero_pi0.sh`), with `./serve_pi0.sh` running in another shell:

```bash
cd /home/DockerShared/ECE780/pure_RL_prototype
PY=/opt/openpi/examples/libero/.venv/bin/python
MUJOCO_GL=egl $PY noise_test.py --out-dir runs/noise_test                 # full sweep
MUJOCO_GL=egl $PY noise_test.py --out-dir runs/noise_test --resume        # continue after a stop
MUJOCO_GL=egl $PY noise_test.py --out-dir runs/noise_test --summarize     # table from results.jsonl
```

How to read the table:

- **fail rate vs. the ε = 0 row:** does a bounded residual help from pre-failure states?
- **rescued:** fail starts where pi0 alone scored 0/16 but this condition scored at least once. These are successes pi0 can't reach alone but a residual inside ±ε can.
- **ok rate:** must stay near the ε = 0 row.

**GO** if some ε lifts the fail rate or rescues starts while the ok rate holds. The largest such ε becomes the residual bound for training. **NO-GO** for pick if no ε beats the baseline. The printed verdict is a heuristic: ≥ +0.10 lift or ≥ 2 rescued starts, with ≤ 0.05 ok drop. Check the per-start numbers in `summary.json` and the videos too.

## Next (after a GO)

Step 2 is the residual policy and trainer:
- **Policy input:** wrist image, proprio, and pi0's action.
- **Policy output:** Gaussian Δa through tanh·ε, with μ initialized to 0.
- **Training:** GRPO over groups of rollouts from the same start state, with the following additions:
  - drop groups where every rollout got the same reward;
  - penalize |Δa|;
  - ramp ε up from 0;
  - use a mix of fail and ok starts.
