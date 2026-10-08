# Failure agent prototype (ECE780)

A diagnosis + recovery agent around native pi0 on LIBERO libero_90, built
from the failure survey of all 90 tasks in
`Documentation/failure_mode_sumary.txt` (per-task labels:
`Documentation/native_pi0_libero90_survey_labels.md`). It implements three of
the four parts of the project architecture:

| Project part        | Here                                                                 |
|---------------------|----------------------------------------------------------------------|
| Action client       | pi0_libero via the openpi websocket server                           |
| Diagnosis agent     | `SymptomMonitor` (online triggers) + `DiagnosisAgent` (root cause)   |
| Library             | `config/skill_library.yaml` + `SkillLibrary` (plans, stats)          |
| Coordinator agent   | minimal rule-based stand-in: library lookup, prompt slot filling, and the object/handle/destination positions the scripted primitives need |

```
           +--------------------- pi0 action chunks ----------------------+
           v                                                              |
   LIBERO env --obs--> SymptomMonitor --trigger--> DiagnosisAgent --code--> SkillLibrary
       ^                (every step)               (R1-R7, S1-S4)          (recovery plan)
       |                                                                      |
       +-- scripted primitives (release / retract_home / approach / carry_to / <+
           finish_articulation) + pi0 sub-tasks with re-worded prompts
```

## Design in one line

The survey showed pi0's grasps on unseen libero_90 tasks are usually clean,
but it binds the instruction to the wrong target or destination (it runs the
training task of a similar scene). So the recovery recipe is: the
coordinator supplies **where** (hover the open gripper over the right object,
then carry the grasped object to its destination), and pi0 supplies **how**
(the grasp itself).

## Taxonomy

Root causes (decided by the diagnosis agent, checked in this order) take
precedence over symptoms (detected online). Detection uses simulator ground
truth: BDDL goal predicates, object poses, joint values, gripper contacts,
and the EEF projected into the agentview camera.

| Code | Name | Online trigger | Rule |
|------|------|----------------|------|
| R1 | wrong object | `wrong_grasp` | a movable object outside the pending placement targets (or the object a recovery grasp step asked for) is held, or was carried away |
| R2 | scene habit | `unplanned_articulation`, `premature_goal` | a joint the goal doesn't involve moved, or a close/turn-on joint moved toward its target while the placement into that region is pending |
| R3 | stopped short | (stall / no_progress / timeout) | articulation goal false, but the goal joint moved or the EEF came within 12 cm of the fixture |
| R4 | placement | (stall / no_progress / timeout) | target was grasped and is within 15 cm (xy) of the destination, or in a sibling region |
| R7 | partial task | (stall / no_progress / timeout) | some goal predicate became true during the episode, another is still false |
| R6 | destination missed | (stall / no_progress / timeout) | target was grasped but is not near the destination (carried elsewhere or dropped) |
| R5 | target not reached | (stall / no_progress / timeout) | target never grasped (closest approach reported), or the goal joint never moved and the EEF never came near the fixture |
| S1 | stall | `stall`, `no_progress` | EEF in a 3 cm box for 60 steps; or no grasp/release/predicate/articulation event for 160 steps |
| S2 | disturbance | `disturbance` | a never-grasped object tilted > 45 deg, or a non-target object pushed > 8 cm |
| S3 | out of view | `out_of_view` | EEF outside the agentview image for 15 steps |
| S4 | wrist wind-up | `wrist_windup` | gripper approach axis tilted > 75 deg from the home pose for 15 steps |

Grasps are confirmed from MuJoCo pad contacts, or (bowls held by the rim
never pass that check) when an object is carried ≥ 5 cm above its start,
within 10 cm of a closed gripper, for 5 steps.

## Recovery steps

| Step | What it does |
|------|--------------|
| `release` | open the gripper in place |
| `retract_home` | lift 10 cm, move back over the home position, rise to the home pose. If still > 5 cm off (pi0 left the arm stretched, elbow at its joint limit, where OSC has no radial authority), swap in robosuite's `JOINT_POSITION` controller and drive back to the start joint configuration |
| `approach` | hover the open gripper 7 cm above the target object (or 8 cm above the goal fixture's handle/knob) in the home orientation, then hand over to pi0 |
| `pi0` | run pi0 with a prompt until the goal / a sub-goal predicate holds / the target is held (`until: grasp_target`), a trigger fires, or its budget ends |
| `carry_to` | carry the held target over its destination (compensating for an off-centre grasp), descend, release, back off |
| `finish_articulation` | push a drawer at its handle along the slide axis by the remaining distance (slide joints only) |
| `decompose` | expand the unsatisfied BDDL goal into ordered sub-tasks: open (approach handle + pi0) → each placement (approach + pi0 grasp + carry_to) → close / turn on / off (approach + pi0, plus the push for drawers) |

Library entries (`config/skill_library.yaml`): R1, R4, R5, R6 use the
assisted pick-and-place (approach → pi0 grasp → carry_to); R2 and R7
decompose; R3 pushes the drawer and re-prompts; S1-S4 retract and re-prompt.

`approach`, `carry_to` and `finish_articulation` read object, handle and
destination positions from the simulator. On hardware a VLM coordinator
(pointing + depth) would have to supply them.

## Running

Policy server (shell 1). Use the prebuilt venv directly; `uv run --offline`
tries to rebuild `openpi-client` and fails without network:

```bash
cd /opt/openpi && OPENPI_DATA_HOME=/home/DockerShared/.cache/openpi \
  .venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_libero --policy.dir=gs://openpi-assets/checkpoints/pi0_libero
```

Agent (shell 2):

```bash
cd /home/DockerShared/ECE780/failure_agent_prototype
./run_agent.sh --tasks 62 71 --mode both                     # observe (native pi0) + agent
./run_agent.sh --tasks survey --init 0 1 2 --workers 6       # all 78 native failures
./run_agent.sh --tasks 0 --mode agent --no-video
python3 tools/summarize_run.py logs/<run>          # detection vs survey, recovery, library stats
```

Native rollouts and survey tools:

```bash
for i in 0 1 2 3 4 5; do MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python \
    tools/collect_native.py --shard $i/6 & done            # -> Documentation/native_pi0_libero90
python3 tools/video_motion_stats.py <folder of mp4s>       # offline stall stats from video
python3 tools/contact_sheet.py <folder of mp4s> <out dir>  # 24-frame sheets for labelling
```

`--tasks survey` reads the failed tasks from
`Documentation/native_pi0_libero90/results.json`. `--workers N` runs N
processes pulling tasks from a shared queue (one websocket client each; the
server serializes inference; 6 workers keep an RTX 5090 at ~90%).

Modes:
- `observe`: native pi0, exactly like openpi's `examples/libero/main.py`
  (400 steps). The monitor records triggers without interrupting, and the
  diagnosis agent labels the failed episode once at the end.
- `agent`: triggers interrupt pi0, then diagnose, build a library plan, and
  recover, up to `agent.max_recoveries` (3) times, with a hard cap of
  `env.max_total_steps` (900).

## Outputs (`logs/<run>/task<id>_init<k>_<mode>/`)

- `trace.json`: goal, targets, every event (grasp, release, predicate,
  articulation, triggers, primitive residuals, joint_home), each diagnosis
  with evidence, slots and the executed plan, and the final scene summary.
- `rollout_{success,failure}.mp4`: 448 px agentview with the current
  prompt or recovery step overlaid.
- `snapshots/t<step>_<code>.npz`: full MuJoCo state at each diagnosis.
  Restore with `env.set_init_state(np.load(f)["state"])` to replay recovery
  candidates from the exact failure state (the counterfactual / RL hook).
- `logs/<run>/summary.json`, `logs/<run>/library_stats.json`, and
  `logs/library_stats.json` (accumulated across runs).

## Files

```
config/agent.yaml            thresholds, budgets, which triggers interrupt pi0
config/skill_library.yaml    the library (hand-curated; stats kept separately)
failure_agent/scene.py       sim ground truth: goal, targets, joints, handles, grasp, projection
failure_agent/monitor.py     online events and triggers
failure_agent/diagnosis.py   rule-based root-cause diagnosis + prompt slots
failure_agent/library.py     plan building (incl. BDDL goal decomposition), stats
failure_agent/recovery.py    scripted primitives on the OSC_POSE controller
failure_agent/language.py    BDDL names -> LIBERO-style phrases (incl. left/right/middle), prompts
failure_agent/runner.py      closed-loop episode runner (+ joint-space retract fallback)
run_agent.py / run_agent.sh  CLI (parallel workers)
tests/smoke_scene.py         policy-free checks on all 90 tasks (parsing, slots, decomposition, retract)
tools/collect_native.py      native pi0 rollouts (main.py loop) with task ids in the file names
tools/contact_sheet.py       contact sheets for the video survey
tools/video_motion_stats.py  offline stall stats from video only
tools/summarize_run.py       detection vs survey labels, recovery, library stats
```

## Results (logs/survey90, 2026-10-08)

All 78 libero_90 tasks that failed in the native collection × init states
0–2, pi0_libero, `config/agent.yaml` as committed. Native = observe mode
(400 steps); agent = up to 3 recoveries, 900 steps total.

| | success |
|---|---|
| native pi0 | 5/234 (2%) |
| agent | 102/234 (44%); 77 within the native 400-step budget |
| tasks solved at least once by the agent | 42/78 (27 of them 3/3) |
| the 14 tasks of the earlier prototype | 22/42 (earlier prototype: 8/42) |

Per library entry (`logs/survey90/library_stats.json`; "recovered" = goal
reached before the next diagnosis):

| entry | used | recovered | notes |
|---|---|---|---|
| R1 wrong object | 71 | 32 | approach + pi0 grasp + carry; the lower hover makes pi0 grasp what is under the gripper |
| R5 target not reached | 185 | 35 | same recipe; the most common diagnosis |
| R3 stopped short | 86 | 11 | the drawer push finishes KITCHEN_SCENE10 (tasks 0, 1, 3, 4, 5: 13/15); hinges have no scripted finish (stove knob 20: 1/3, 44: 0/3; microwave 35: 0/3) |
| R7 partial task | 20 | 6 | decomposition of the remaining sub-goals |
| R6 destination missed | 48 | 5 | carry_to cannot place *inside* the two-layer shelf (needs a horizontal insertion) |
| R2 scene habit | 28 | 4 | |
| R4 placement | 86 | 3 | mostly the shelf and the caddy compartments (books must be inserted upright) |

Who finished the successful agent episodes: 85/102 ended with a plan that
included the scripted `carry_to`, 11 with the scripted drawer push, and 6
needed no recovery (pi0 alone, by chance).

Detection: on observe runs every failed episode got a root cause (no
symptom-only fallbacks): R1 76, R5 56, R6 31, R3 22, R2 21, R4 15, R7 8. The
label matches a video-survey label in 162/229 episodes (the primary label in
123/229).

Takeaways:
1. Grounding, not motor skill, is what pi0 lacks on libero_90. Supplying the
   target location (approach) and the destination (carry_to) turns 2% into
   44%, with pi0 still doing every grasp. The coordinator VLM's job is
   therefore pointing (which object, where to), not motion.
2. These numbers depend on privileged sim positions for approach/carry and
   on scripted placement, so they are an upper bound for what a VLM
   coordinator with perfect pointing could achieve with this library.
3. What's left: placing inside the shelf (40, 42, 86, 88, 89) and upright
   into caddy compartments (73-75, 78, 80); a closed destination drawer
   (task 2: pi0 shuts the drawer first, and only `decompose` re-opens
   drawers, not the R1/R5 recipe); hinged articulations (stove knob
   44/45, microwave 35-37) with no scripted finish; pi0 refusing some targets
   even from right above them (frying pan, red mug in 84); tall objects
   tipping out of the basket (48). These, plus the 185 R5 attempts that did
   not recover, are the natural targets for learned recovery (SimpleVLA-RL
   from the saved failure snapshots).
4. Caveats: 3 inits per task; pi0 is stochastic; "recovered" credits only
   the last diagnosis of a successful episode.
