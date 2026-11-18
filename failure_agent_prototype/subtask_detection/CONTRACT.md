# Inputs, outputs and clarification decisions

## Goal and boundary
Detect completion of a specified pick or on-support place attempt. Caller provides task decomposition, target identity, destination identity and observations. In-container placement, insertion and stacking-specific constraints are not included.

## Inputs
Observation: time_s (monotonic simulation seconds), target (exact ID), target_z_m (world height), target_grasped (bool), destination (exact ID), on_support (bool), released (bool), linear_speed_m_s (nonnegative magnitude), angular_speed_rad_s (nonnegative magnitude), irrecoverable (optional external bool).

Unknown values are None. Numeric units are metres, seconds and radians. Pick uses height and grasp. Place uses relation, release, grasp and speeds. Wrong object IDs raise ValueError instead of silently measuring another object. Simulator exceptions are not swallowed. The optional irrecoverable flag is externally established; this module does not infer workspace bounds.

## Pick defaults
Reference height is measured at attempt start. Lift >= 0.05 m and target grasp must persist for 0.5 s. Temporary grasp loss resets confirmation and reports in_progress/grasp_lost, permitting recovery. An already-held object requires a meaningful pre-pick reference height; do not initialize a new pick at an elevated height and expect completion.

## Place defaults
Named target must satisfy on_support for the named destination, be released, not grasped, have speed <= 0.01 m/s and angular speed <= 0.1 rad/s for 0.5 s. This is a stability proxy; it cannot promise future stability. A stationary wrong-object placement is not accepted because target and destination bindings are explicit.

## Shared timing
Default timeout: 10 s. Maximum observation gap: 0.15 s; a larger gap resets confirmation. Missing evidence also resets confirmation. Designed for control-rate observations; 1 fps VLM observations require a different sampling configuration and validation. Timeout takes precedence at the exact deadline, even with missing evidence. Successful or failed results latch: create a fresh detector for a retry. Later slips after a successful pick require a new transport monitor, not this completed attempt detector.

## Output
Result fields: subtask, status, reason, evidence. Status is in_progress, succeeded, failed, unknown. Failed is returned for timeout or explicit irrecoverable evidence. Unknown means required evidence is unavailable. Evidence includes elapsed time and measured quantities supporting the decision. A missed grasp alone is not irrecoverable failure.

Desired pick output: {"subtask":"pick","status":"succeeded","reason":"condition_held","evidence":{"target":"bowl","lift_m":0.06,"target_grasped":true,"valid_duration_s":0.5,"elapsed_s":0.5}}

Desired place output: succeeded only after a continuous supported, released, stable window. A held-over-plate observation returns in_progress; missing support evidence returns unknown.

## Decisions requiring lab confirmation
Thresholds are engineering defaults, not measured optimal settings. Destination predicates must describe the intended spatial relation and actual support, not just bounding-box overlap. Release mapping and simulator API compatibility remain unverified. No claims of detector accuracy or full-task recovery success are made.
