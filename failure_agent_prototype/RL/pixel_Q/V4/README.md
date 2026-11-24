# Pixel-Q V4: value function V(s), collection and training split apart

V4 learns the same **V(s)** as V3: the discounted chance that pi0, left alone from the pose where it takes over,
picks the target within H. The difference is the pipeline. There is no actor-learner loop:

1. `collect.py` runs pi0 rollouts with perturbed branches on CPU worker processes and writes sample shards. It
   logs the sample counts to W&B.
2. `train_v.py` trains V from the shards on the GPU, during collection or after it.

V3's online learner added nothing to the data, because V never steers collection, and on CPU it slowed collection
down. V4 data are labelled with the updated pick check (both fingers anywhere, not only the fingertip pads), so
V3 shards should not be mixed in.

- How to run and read the outputs: [Vlearning_code_instruction.md](Vlearning_code_instruction.md)
- Design decisions: [../V3/plan_the_algo.txt](../V3/plan_the_algo.txt) (sections 6-7)

| File | Role |
|---|---|
| `collect.py` | Step 1: workers collect perturbed-branch samples into shards (no learner) |
| `train_v.py` | Step 2: trains V (and the proprio-only baseline) from a shard folder on the GPU |
| `episodes.py` | One rollout (ticks, selection, natural + perturbed episodes, video) |
| `data_io.py` | Sample shards and labels |
| `vnet.py` | V network (attention or CNN pooling on the gripper-centred heightmap + proprio; proprio-only baseline) |
