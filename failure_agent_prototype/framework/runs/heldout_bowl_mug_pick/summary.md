| episode | split | success | pi0 raw (same budget) | pi0 alone (baseline dir) | sim s | Q (V-low / timeout / skipped) | HL (library) | V evals (low) | VLM calls | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| libero_90_task2_init0 | held_out | yes | no (30.5 s) | - | 14.9 | 0 / 1 / 0 | HL-7 x1 | 3 (3) | 44 | 8.2 |
| libero_90_task2_init1 | held_out | yes | no (30.5 s) | - | 7.2 | 0 / 0 / 0 | - | 4 (0) | 37 | 5.8 |
| libero_90_task2_init2 | held_out | yes | no (30.5 s) | - | 15.9 | 1 / 1 / 0 | HL-7 x1 | 3 (3) | 46 | 7.8 |
| libero_90_task2_init3 | held_out | yes | no (30.5 s) | - | 15.8 | 0 / 1 / 0 | HL-7 x1 | 1 (0) | 52 | 8.4 |
| libero_90_task2_init4 | held_out | yes | no (30.5 s) | - | 11.75 | 0 / 0 / 0 | HL-7 x1 | 2 (0) | 43 | 7.0 |
| libero_90_task2_init5 | held_out | yes | no (30.5 s) | - | 16.25 | 1 / 1 / 0 | HL-7 x1 | 3 (3) | 44 | 7.7 |
| libero_90_task12_init0 | held_out | yes | no (30.5 s) | - | 11.25 | 0 / 0 / 0 | HL-3/HL-4 x1 | 1 (0) | 39 | 8.6 |
| libero_90_task12_init1 | held_out | yes | yes (27.3 s) | - | 15.45 | 0 / 1 / 0 | HL-3/HL-4 x1 | 2 (2) | 51 | 10.9 |
| libero_90_task12_init2 | held_out | yes | no (30.5 s) | - | 9.5 | 0 / 0 / 0 | HL-3/HL-4 x1 | 4 (2) | 31 | 7.1 |
| libero_90_task12_init3 | held_out | yes | no (30.5 s) | - | 19.5 | 0 / 1 / 0 | HL-7 x1 | 23 (4) | 69 | 14.0 |
| libero_90_task12_init4 | held_out | yes | no (30.5 s) | - | 15.4 | 1 / 1 / 0 | - | 3 (3) | 50 | 10.2 |
| libero_90_task12_init5 | held_out | yes | yes (21.75 s) | - | 15.9 | 0 / 1 / 0 | - | 0 (0) | 66 | 13.2 |

success 12/12 (criterion: pick(target): grasped and lifted >= 5 cm for 10 steps; the episode stops there, else runs to the budget)

pi0 raw, same init / budget / stop rule as the framework (12 episodes): framework 12/12, pi0 alone 2/12 (framework fixed 10, broke 0)

| task | framework | pi0 raw |
|---|---|---|
| 2 | 6/6 | 0/6 |
| 12 | 6/6 | 2/6 |

no recorded pi0 baseline (pi0_LIBERO_rollouts, LIBERO goal only) for these episodes
