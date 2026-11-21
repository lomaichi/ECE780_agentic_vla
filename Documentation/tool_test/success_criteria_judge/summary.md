# VLM subtask judge vs. predicates

Per-call agreement (VLM's judgement of the current subtask at each 1 fps call vs. the predicate at the same step). `latched` = predicate as defined in fa_core/predicates.py (held 10 steps, then latches); `now` = the predicate's condition at that instant.

| subtask / reference | n | agree | VLM yes & pred yes | VLM yes & pred no | VLM no & pred yes | VLM no & pred no |
|---|---|---|---|---|---|---|
| pick/pred_latched | 67 | 0.955 | 1 | 3 | 0 | 63 |
| pick/pred_now | 67 | 0.97 | 2 | 2 | 0 | 63 |
| place/pred_latched | 16 | 0.812 | 4 | 2 | 1 | 9 |
| place/pred_now | 16 | 0.812 | 4 | 2 | 1 | 9 |

Per episode: first step at which each judged success (`-` = never).

| task | pick pred | pick VLM | place pred | place VLM | LIBERO (any / at end) | stop | steps | corrections |
|---|---|---|---|---|---|---|---|---|
| 2: put the black bowl in the top drawer of the cabinet | 229 | 217 | 269 | 277 | True / True | VLM: task complete | 297 | point_approach@50 |
| 17: stack the middle black bowl on the back black bowl | - | 150 | - | 331 | False / False | VLM: task complete | 351 | carry_to@230 |
| 24: put the black bowl in the bottom drawer of the cabinet | 136 | 130 | 187 | 210 | True / True | VLM: task complete | 230 | - |
| 25: put the black bowl on top of the cabinet | 332 | 334 | - | - | False / False | max steps | 410 | point_approach@190, carry_to@354 |
| 30: put the black bowl on the plate | - | - | - | - | False / False | max steps | 410 | point_approach@110, point_approach@265 |
| 36: put the white bowl on the plate | - | - | - | - | False / False | max steps | 410 | point_approach@50, reprompt@259, point_approach@319 |
