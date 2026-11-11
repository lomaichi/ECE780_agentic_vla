# Pixel-wise Q-learning for Pi0 Low-Level Failure Recovery: Proposal Review and Go/No-Go Test (ECE780)

An alternative to `../residual_RL/residule_RL_plan.md`. Pi0 stays frozen. A small correction picks *where* on the object the gripper should go, then control returns to pi0. Written 2026-11-11. The test script `oracle_handoff_test.py` in this folder has been written but **not run yet**: no LIBERO or pi0 outside the container.

## 1. The idea

Pixel-wise Q-learning, as in Zeng et al. (VPG: push-grasp synergies) and Transporter Networks. A fully convolutional network outputs a Q-map over the image. The action is a pixel, plus a primitive (grasp / push / place) and a rotation bin.

1. A segmentation mask selects the target object. Crop around it (agentview RGB plus depth, which robosuite can render).
2. The network outputs Q(u, v, θ) for each 3×3-pixel cell and each of k rotation bins (8–16 rotated copies of the crop).
3. Take the argmax. Back-project pixel + depth + camera intrinsics to a 3D point, then run a scripted OSC primitive to it. This is a learned version of the existing `POINT_APPROACH`.
4. Hand control back to pi0. Corrections are issued less often than pi0 replans; pi0 runs n chunks after each correction.

The crop keeps the correction small, playing the role that `alpha` plays in the residual plan. The Q heatmap is directly explainable, and the FR agent (a VLM) can look at it. That suits the README's explainability goal better than a residual MLP's 35-D delta.

## 2. Which LL modes it fits

| LL mode | fit | why |
|---|---|---|
| LL-4 slip, LL-8 hesitation (`rl/pick`) | strong | the failure is aiming at the wrong grasp point |
| LL-1 drawer push stall | good | push point and direction on the drawer front is a pixel action |
| LL-3 knob | good | grasp point plus rotation bin |
| `rl/place` destination | good | Transporter-style choice of where to place |
| LL-6 instability, LL-7 release stall | poor | timing failures; a "where" action can't fix "when to open" |
| LL-2 open door | poor | the motion follows an arc over time, not a single point |

The two methods are complementary: pixel-Q for spatial misalignment, and the residual (or a scripted gripper primitive) for timing.

## 3. Issues to resolve

1. **Existing rollouts have no pixel actions.** Pi0 acts in 7-D deltas. You'd need hindsight labels: project the EE position at contact into the image and label that pixel with the subtask outcome. That gives about one labelled pixel per attempt. On disk now: libero_10 has 15 successes and 5 failures, libero_spatial 20 successes and 0 failures, libero_90 19 successes and 161 failures (before removing held-out and unseen-skill tasks). That is tens of positives per skill, in other scenes, while VPG needed thousands of attempts. Traces store the full sim state (`states.npy`, 77-D), so RGB, depth and segmentation can be re-rendered at any step. Most data must still come from **online sim collection**: restore a reset state, try a pixel action, check the predicate. This is still off-policy, and cheap because one primitive is one attempt. Use the pi0 successes to warm-start.
2. **With one primitive and a binary reward, this is a bandit problem** (γ = 0, binary classification per pixel). That's fine and easier to train, but describe it accurately in the report. It becomes true Q-learning (γ > 0) only with chained corrections, such as push then grasp, or re-grasp.
3. **The reward must include the handoff.** Reward = the subtask predicate holds within m pi0 chunks *after* the correction. The Q-map then learns to put the robot in a state from which pi0 succeeds. The risk is that pi0 fails even from a good state; it was 0/32 from the start states. That risk is what the go/no-go test (Section 4) checks.
4. **Segmentation at test time.** Sim masks are privileged, under the same rule as "sim-predicate leakage" in the residual plan. At test time the mask has to come from the FR agent plus an open-vocabulary segmenter (Grounded-SAM / SAM2). Train on noisy or jittered crops.
5. **Resolution.** At 128–256 px, a 3×3 cell is about 1 cm, roughly the scale of the misalignment being fixed. The wrist camera has better resolution, but its frame moves with the arm.
6. **Rotation.** Without rotation bins, the knob and grasp-orientation failures aren't covered.

## 4. Go/No-Go test: `oracle_handoff_test.py`

**Question:** if a perfect Q-map put the gripper at the best spot on the object, would that rescue the failure? The test brute-forces the Q-map with privileged sim object poses instead of learning it.

For each of the 14 `rl/pick` reset states in `library.json`, it lays a 5×5 grid over the target object's footprint (world AABB from mesh vertices, +1 cm margin). Three arms:

| arm | what runs | trials | tells you |
|---|---|---|---|
| baseline | plain pi0 from the reset state, budget 2 × horizon | 32 / state | subtask-level pass@k (also the residual plan's Step 1 for pick) |
| handoff | gripper open: rise, travel, descend to 7 cm above the point, then pi0 for `horizon` steps | 4 / cell | the proposed design: correct, then hand back |
| scripted | same approach, descend to object top − 2 cm, close, lift 10 cm; no pi0 | 4 / cell | whether grasping at that point works at all |

- **Success** is the `rl/pick` predicate: Grasped(obj) AND obj ≥ 5 cm above its reset height for 10 consecutive steps.
- Each trial jitters the target by ±0.5 cm, about one pixel cell.
- **Phase 2 (confirm):** each arm's best cell per state is re-run 16 times, to remove the max-of-25-noisy-estimates bias.
- The split guard is enforced: the script exits if a reset state comes from a held-out libero_90 task.

### Verdict (printed at the end; thresholds are constants at the top of the script)

- **GO (handoff):** on ≥ 50% of states, the confirmed best handoff cell is ≥ 0.5 **and** ≥ baseline + 0.25. Pixel-Q with handoff is viable, and the per-cell map is a training target / sanity check.
- **GO (primitive):** handoff fails but scripted ≥ 0.5. The primitive must finish the grasp itself, and pi0 takes over only at the next subtask. Test that handoff separately.
- **NO-GO:** neither. Even the best point doesn't rescue these states.
- **location_matters** (per state): best cell rate − median cell rate ≥ 0.25. If this rarely holds, plain `POINT_APPROACH` to the object centre is enough and pixel-Q adds nothing. Pixel-Q is worth building only when the heatmaps show a clear peak.

### How to run (inside the podman container, `./libero_pi0.sh`)

Shell 1, pi0 server:
```
cd /opt/openpi && uv run scripts/serve_policy.py --env LIBERO
```

Shell 2 (`./libero_pi0.sh` again opens a second shell):
```
cd <repo>/failure_agent_prototype/RL/pixel_Q
PY=/opt/openpi/examples/libero/.venv/bin/python

# 0. reset states restore and replay cleanly
$PY ../../tools/scene_restore.py --check --policy rl/pick

# 1. inspect, no pi0: check each PNG shows the grid on the right object
$PY oracle_handoff_test.py --out-dir out_oracle --inspect

# 2. smoke test on one state; watch out_oracle_smoke/videos/*.mp4
$PY oracle_handoff_test.py --out-dir out_oracle_smoke --states 0 --grid 3 --n 2 --n-base 4 --confirm-n 4

# 3. full run (estimated 1–1.5 h with 6 workers)
$PY oracle_handoff_test.py --out-dir out_oracle
```

Useful flags: `--states i j ...` (indices into the rl/pick reset list), `--object <episode>@<step>=<sim name>`, `--arms handoff scripted`, `--grid`, `--yaws 0 45 -45` (gripper yaw bins), `--dz`, `--grasp-depth`, `--jitter-cm`, `--n`, `--n-base`, `--confirm-n`, `--baseline-steps`, `--workers`.

### Output (`--out-dir`)

- `metadata.json`: settings, resolved objects, per-cell rates, confirm rates, verdict.
- `heatmaps/<state>_<arm>.png`: upright reset frame (3×), each cell coloured red (0) to green (1).
- `videos/<state>_<arm>*.mp4`: trial 0 of the baseline and of each confirm run.
- `inspect/*.png`, `inspect/states.json`: from `--inspect`.

### Untested; check these during steps 1–2

- **Wrong object.** Objects are matched from `pick(black_bowl)`-style subtask words by name tokens; ties go to the object nearest the EE. `moka_pot_left_in_image` (libero_10/task8_fail_init1@260) is the most likely to be wrong. Override with `--object`; the inspect output lists the scene's object names.
- **Scripted motion.** Assumes the LIBERO OSC takes position and rotation deltas in the world frame (gains `POS_GAIN = 10`/m, `ROT_GAIN = 1`/rad). If the smoke-test video shows the arm drifting sideways instead of rising, travelling and descending, fix `move_to`.
- **Heatmap orientation.** A mirrored overlay affects only the drawing; the test runs in world coordinates.
- **Object already grasped at reset.** The script prints a warning, because the open-gripper approach drops the object. Skip those states with `--states`.
- **Possible API mismatches** (robosuite/LIBERO version): `env.env.obj_body_id`, `_check_grasp`, `model.mesh_vert`, `camera_utils.project_points_from_world_to_camera`. `--inspect` exercises all of these except `_check_grasp`.

## 5. Next steps

- **If GO:** write the full pixel-Q plan, with the same structure as the residual plan. Planned contents:
  - data engine: online sim pixel trials from the reset bank, plus hindsight labels from the pi0 successes;
  - an FCN over segmented crops with rotation bins;
  - reward = subtask predicate after handoff;
  - the `INVOKE_PIXELQ` library entry;
  - a comparison against the residual and SimpleVLA-RL.
- **Extend the test** to `rl/close_drawer` (push point on the drawer front) and `rl/turn_knob` (grasp point plus yaw bins).
- **If NO-GO:** keep the residual plan as the main method for pick, and report the oracle heatmaps as evidence that aiming alone does not rescue these failures.
