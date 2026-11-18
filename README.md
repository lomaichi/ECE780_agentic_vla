# ECE780

## Project premise

Preliminary research in **agentic robot control**, using **pi0** as benchmarks and the base VLA models. 

**Pi0** uses a 3B VLM backbone plus a flow-matching action expert for action-chunk generation. Pi0 struggles with generalizability, as it has very low zero-shot success when tested against LIBERO-90. Note that the `pi0_libero` checkpoint is finetuned on four LIBERO suites (LIBERO-spatial, LIBERO-object, LIBERO-goal and LIBERO-10; openpi's `physical-intelligence/libero` dataset), but not on LIBERO-90. Serve it with `./serve_pi0.sh`; openpi's `serve_policy.py --env LIBERO` shortcut serves pi0.5 instead. To improve zero-shot success in a data-efficient and explanable manner, we propose combining VLM agent and RL. The goal is: 
- To avoid data-intensive fine-tuning
- To explicitly accumulate reusable failure recovery strategy

## System architecture

**Training: Failure Recovery Skill Library**

1. **Failure Survey**: An VLM agent surveys the failed Pi0 rollouts on LIBERO-spatial, LIBERO-10, and a random 50% of LIBERO-90. During the survey, the agent outputs:
- Common failure modes: high-level and low-level. High-level modes refer to those caused by reasoning error (e.g. identified the wrong object), low-level control modes refer to those related to fine motor control or fine spatial perception (e.g. gripper misaligning with the grippable area)
- For high-level failure mode: Agent propose solutions
- For low-level failure mode: Agent initiate an RL for that corresponding subtask (e.g. if gripper jitters and fail to pick up a bowl due to mislignment is a low-level failure mode, initiate RL for the pickup() task). 
2. **RL for low-level failure recovery**: The RL method will following SimpleVLA-RL (https://github.com/PRIME-RL/SimpleVLA-RL), LIBERO will provide binary pass/fail during the online RL for the corresponding subtask. Tentatively an agent will be called to dispatch the most similar LIBERO-spatial or LIBERO-10 subtask success to guide the RL

**Testing : Closed-Loop Execution** 

This uses the remaining 50% LIBERO-90 tasks for zero-shot success evaluation.

3. **Coordinator agent**: a VLM that plans subtasks and assigns individual short-horizon tasks to the action clients. For example, with the task instruction "pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf", it breaks it down into atomic tasks "navigate_to(red_block), pick_up(red_block), navigate_to(blue_bowl), place(cabinet shelf, on)"
4. **Action client**: performs pi0 inference to execute the subtasks.
5. **Failure Recovery Agent (FR Agent)**: Monitor the task by looking at the screenshot at 1 fps as well as proprioception (i.e. robot pose), and detects failure, while deciding whether failure is high-level or low-level:
- High-level failure: FR Agent use self-proposed high-level recovery method in failure recovery skill library
- Low-level failure: FR Agent use low-level RL in failure recovery library

**Post Training: Finetune Pi0 with Successful Trajectory**

6. Use the test-time success trajectory to fine tune pi0 after each rollout, so that finetuned pi0 can be used for next test time
