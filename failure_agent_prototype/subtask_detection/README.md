# Subtask detection: implementation scope

Goal: provide reusable, simulator-evidence pick and on-support place detection for the pixel-Q correction loop. This implements procedures 1-3 only. It does not train a model, run a VLM, detect arbitrary failure categories, or launch simulator evaluation.

Create one detector per attempt. Call update after robot actions, with simulation time in seconds. The detector returns a Result; to_dict() produces JSON-compatible fields. The controller chooses retry/advance/stop; the trainer maps the result to reward. The detector itself never moves the robot.

## Files and changes
All files in this directory are new. Existing project files were not edited.
- __init__.py: public exports.
- detectors.py: observation/result contracts, temporal state machine, pick and place implementations.
- libero_adapter.py: simulator evidence extraction and explicit predicate callbacks.
- smoke_checks.py: bounded synthetic implementation checks, not a simulator-validation suite.
- CONTRACT.md: inputs, outputs, assumptions and configuration.
- README.md: scope, change inventory and integration instructions.

## Example
```python
from failure_agent_prototype.subtask_detection import PickDetector, Observation
pick = PickDetector('bowl', start_time_s=0, initial_height_m=0.8)
for i in range(12):
    result = pick.update(Observation(time_s=i*0.05, target='bowl',
        target_z_m=0.86, target_grasped=True))
print(result.to_dict())  # succeeded; condition_held
```

Run locally from repository root:
`PYTHONDONTWRITEBYTECODE=1 python3 -m failure_agent_prototype.subtask_detection.smoke_checks`

## LIBERO integration
Create LiberoAdapter(env, target, destination, on_support, released). Its observe(time_s) result can be passed to either detector. Target and destination are exact simulator identifiers, not language descriptions. The caller must resolve them.

on_support(inner, target, destination) must return a Python bool from the installed environment's verified object-on-support relation. released(inner, target) must return a Python bool establishing intentional release (e.g. validated open-gripper state). Neither callback is guessed from proximity. Missing callbacks leave place status unknown. API errors propagate to the caller and must be logged as infrastructure errors, not policy failures.

The adapter follows existing repo grasp/body lookup code; velocity access uses MuJoCo body velocity APIs. These calls have NOT been verified in the lab. No complete LIBERO place predicate binding is supplied because the dependency source/runtime is unavailable in this checkout. This is an explicit integration boundary, not a completed end-to-end detector deployment.

Before lab use: verify object identifiers, grasp predicate, velocity units, on-support semantics and open-gripper mapping. Then review trajectories and tune thresholds. Replay validation and broad detector performance evaluation are deferred beyond tonight's scope.
