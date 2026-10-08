# ECE780

## Project premise

Preliminary research in **agentic robot control**, using **pi0** (and potentially **pi0.5**) as benchmarks and the base VLA models. 

**Pi0** uses a 3B VLM backbone plus a flow-matching action expert for action-chunk generation. While pi0 was originally developed as a stand-alone action policy, this project adopts it for action clients, utilizing its ability to perform atomic short-horizon tasks, while delegating the long-horizon reasoning or failure analysis to other VLM agents. 

**Pi0.5** will be used for comparison, as it implements an opposite philosophy -- integrating the VLM high-level reasoning and low-level VLA action expert in one utilized network, instead of decentralizing them as we propose.

## System architecture (2 modules, 5 parts, closed loop)

**Module 1 : Closed-Loop Execution** 
1. **Coordinator agent**: a stronger VLM that plans subtasks and assigns individual short-horizon tasks to the action clients, drawing from the patched motion primitive
2. **Action client**: performs pi0 inference to execute the subtasks.
3. **Diagnosis agent**: analyzes why a task failed and patches common failure modes into the library.

**Module 2: Skill Library**
4. **Motion Primitive Library**: contains motion primitives distilled from successful short-horizon task runs
5. **Failure Recovery Library**: contains a failure-recovery method for each common failure mode.

## Tentative directions

- Use RL (referencing **SimpleVLA-RL**) to develop failure-recovery strategies.
- Search successful episodes of native, standalone pi0 for motion primitives / mini-tasks that pi0 achieves reliably and repeatably, forming a **motion-primitives library** the coordinator agent can use.

## Paths

- Host path: `/home/e36luo/DockerShared/ECE780`, mounted in the container as `/home/DockerShared/ECE780`.
