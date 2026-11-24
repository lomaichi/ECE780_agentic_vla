# Pixel-Q V3: value function V(s) for the low-level trigger

V3 replaces V2's Q-map with a value network **V(s)**: the discounted chance that pi0, left alone from the pose
where it takes over, picks the target within H = 3 s. Corrections are later chosen by searching candidate poses
with V on synthetic observations (separate stand-alone test, not in this folder).

- How to run and read the outputs: [Vlearning_code_instruction.md](Vlearning_code_instruction.md)
- Design decisions: [plan_the_algo.txt](plan_the_algo.txt) (sections 6-7)
- Discussion history and earlier results (V on V2 no-op data, pilots): `../V2/run_recap.txt`

| File | Role |
|---|---|
| `train.py` | Entry point: actors collect perturbed-branch samples, the learner trains V online |
| `episodes.py` | One actor rollout (ticks, selection, natural + perturbed episodes, video) |
| `data_io.py` | Sample shards and labels |
| `vnet.py` | V network (CNN on the gripper-centred heightmap + proprio; proprio-only baseline) |
| `train_v.py` | Offline refit of V from a run's shards |

`runs/V_pilot/` holds the pilot videos from the design phase (3 s window).
