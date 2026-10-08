# ECE780

Session convention: when the user brings up "ECE780" at the start of a session, the whole session is dedicated to this project. Work within the context below without asking them to restate it.

## Project premise

Preliminary research in **agentic robot control**, using **pi0** (and potentially **pi0.5**) as benchmarks and the base VLA models. Pi0 uses a 3B VLM backbone plus a flow-matching action expert for action-chunk generation.

## System architecture (4 parts, closed loop)

1. **Coordinator agent**: a stronger VLM that plans subtasks and assigns individual short-horizon tasks to the action clients.
2. **Action client**: performs pi0 inference to execute the subtasks.
3. **Diagnosis agent**: analyzes why a task failed and patches common failure modes into the library.
4. **Library**: a skill library containing a failure-recovery method for each common failure mode.

The loop closes because the diagnosis agent passes its diagnostics to the coordinator, which looks up the library for a plausible solution and reassigns work to the action client.

## Tentative directions

- Use RL (referencing **SimpleVLA-RL**) to develop failure-recovery strategies.
- Search successful episodes of native, standalone pi0 for motion primitives / mini-tasks that pi0 achieves reliably and repeatably, forming a **motion-primitives library** the coordinator agent can use.

## Paths

- Host path: `/home/e36luo/DockerShared/ECE780`, mounted in the container as `/home/DockerShared/ECE780`.
