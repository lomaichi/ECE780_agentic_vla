# Phase 1: subtask check validation

## Part A: LIBERO success recomputed on the recorded final state

- agree: 218 / 220 episodes
- disagree: 2
  - libero_spatial/task3_success_init1: recorded True, recomputed False
  - libero_spatial/task6_success_init0: recorded True, recomputed False
- errors: 0

## Part B: subtask checks

| source | check | judged | agree | disagree |
|---|---|---|---|---|
| goal | close | 5 | 5 | 0 |
| goal | pick | 60 | 44 | 16 |
| goal | place_libero | 60 | 58 | 2 |
| goal | turn_on | 3 | 3 | 0 |
| survey | close | 6 | 5 | 1 |
| survey | open | 9 | 9 | 0 |
| survey | pick | 57 | 52 | 5 |
| survey | place | 61 | 61 | 0 |
| survey | turn_off | 2 | 2 | 0 |
| survey | turn_on | 5 | 5 | 0 |

### Disagreements (look at the video around the step shown)

| episode | source | check | expected | fired at step | label |
|---|---|---|---|---|---|
| libero_10/task4_success_init0 | goal | `pick(white_yellow_mug_1)` | fire | - |  |
| libero_10/task4_fail_init1 | survey | `pick(white_yellow_mug_1)` | fire | - | pick(white_mug): success: grasped ~67-93 |
| libero_10/task4_fail_init1 | survey | `pick(white_yellow_mug_1)` | fire | - | pick(yellow_and_white_mug): success: grasped ~300-313 |
| libero_10/task6_success_init1 | goal | `pick(chocolate_pudding_1)` | fire | - |  |
| libero_10/task9_success_init0 | goal | `pick(white_yellow_mug_1)` | fire | - |  |
| libero_10/task9_success_init1 | goal | `pick(white_yellow_mug_1)` | fire | - |  |
| libero_90/task1_fail_init1 | survey | `close(wooden_cabinet_1_top_region)` | fire | - | close(top_drawer): success: handles in line by step 100 |
| libero_90/task8_fail_init0 | survey | `pick(akita_black_bowl_1)` | fire | - | pick(black_bowl): success (out of order): bowl grasped and lifted by ~140 |
| libero_90/task8_fail_init1 | survey | `pick(akita_black_bowl_1)` | fire | - | pick(black_bowl): success (out of order): bowl grasped and lifted by ~140 |
| libero_90/task14_fail_init1 | survey | `pick(akita_black_bowl_2)` | fire | - | pick(middle_black_bowl): success: middle (image-left) bowl grasped ~80-100 and lifted by 120 |
| libero_90/task38_success_init0 | goal | `pick(moka_pot_1)` | fire | - |  |
| libero_90/task67_success_init1 | goal | `pick(porcelain_mug_1)` | fire | - |  |
| libero_spatial/task0_success_init1 | goal | `pick(akita_black_bowl_1)` | fire | - |  |
| libero_spatial/task3_success_init0 | goal | `pick(akita_black_bowl_1)` | fire | - |  |
| libero_spatial/task3_success_init1 | goal | `pick(akita_black_bowl_1)` | fire | - |  |
| libero_spatial/task3_success_init1 | goal | `place(akita_black_bowl_1, plate_1, on)[libero]` | fire | - |  |
| libero_spatial/task5_success_init0 | goal | `pick(akita_black_bowl_1)` | fire | - |  |
| libero_spatial/task5_success_init1 | goal | `pick(akita_black_bowl_1)` | fire | - |  |
| libero_spatial/task6_success_init0 | goal | `pick(akita_black_bowl_1)` | fire | - |  |
| libero_spatial/task6_success_init0 | goal | `place(akita_black_bowl_1, plate_1, on)[libero]` | fire | - |  |
| libero_spatial/task8_success_init0 | goal | `pick(akita_black_bowl_1)` | fire | - |  |
| libero_spatial/task8_success_init1 | goal | `pick(akita_black_bowl_1)` | fire | - |  |
| libero_spatial/task9_success_init0 | goal | `pick(akita_black_bowl_1)` | fire | - |  |
| libero_spatial/task9_success_init1 | goal | `pick(akita_black_bowl_1)` | fire | - |  |

### Strict place on success episodes (informational): fired in 12 / 60

### Skipped survey labels

- libero_10/task4_fail_init1: label not used; place(yellow_and_white_mug, right_plate, on): partial (visually): set on the robot-right (image left) plate ~353-393, but the plate is partly outside the frame and the mug sits off-centre toward that edge
- libero_10/task8_fail_init0: skipped ('moka_pot_center' matches no single goal object in ['moka_pot_1', 'moka_pot_2']); pick(moka_pot_center): success: grasped ~step 120-140
- libero_10/task8_fail_init0: skipped ('moka_pot_center' matches no single goal object in ['moka_pot_1', 'moka_pot_2']); place(moka_pot_center, stove, on): success (visual): set down ~step 240-260 on the back-right part of the stove; could not confirm it is inside the cook region
- libero_10/task8_fail_init0: skipped ('moka_pot_left_in_image' matches no single goal object in ['moka_pot_1', 'moka_pot_2']); pick(moka_pot_left_in_image): never attempted: arm sweeps up to the image-top-left and leaves the camera view by step 400
- libero_10/task8_fail_init0: skipped ('moka_pot_left_in_image' matches no single goal object in ['moka_pot_1', 'moka_pot_2']); place(moka_pot_left_in_image, stove, on): never attempted
- libero_10/task8_fail_init1: skipped ('moka_pot_center' matches no single goal object in ['moka_pot_1', 'moka_pot_2']); pick(moka_pot_center): success: grasped ~step 107-127
- libero_10/task8_fail_init1: skipped ('moka_pot_center' matches no single goal object in ['moka_pot_1', 'moka_pot_2']); place(moka_pot_center, stove, on): success (visual): on the stove at step 200, released by ~227
- libero_10/task8_fail_init1: skipped ('moka_pot_left_in_image' matches no single goal object in ['moka_pot_1', 'moka_pot_2']); pick(moka_pot_left_in_image): partial/late: arm reaches the pot by ~267 but hovers and adjusts for ~170 steps; grasp closes ~433-447, lift at 460
- libero_10/task8_fail_init1: skipped ('moka_pot_left_in_image' matches no single goal object in ['moka_pot_1', 'moka_pot_2']); place(moka_pot_left_in_image, stove, on): failed (timeout): pot is being lowered onto the crowded stove against pot 1 from step 487; still held in the gripper at 529
- libero_90/task1_fail_init0: label not used; close(top_drawer): partial: pushed from ~80, motion stops by 160 with the drawer roughly 1/2-2/3 closed, not flush
- libero_90/task9_fail_init0: label not used; pick(black_bowl): partial: rim grasp by ~80; the bowl is held but not clearly lifted until late
- libero_90/task9_fail_init1: label not used; pick(black_bowl): partial: rim grasp by ~80; the bowl is held but not clearly lifted until late
- libero_90/task17_fail_init0: skipped (verb 'stack' has no check); stack(middle_black_bowl, back_black_bowl): failed: held bowl brought beside the back bowl (140-200), then drifts left and hovers holding it to the end; never placed on top
- libero_90/task17_fail_init1: skipped (verb 'stack' has no check); stack(middle_black_bowl, back_black_bowl): failed: bowl carried past the back bowl to the cabinet top (127) and held there tilted until 409
- libero_90/task21_fail_init0: skipped ('frying_pan' matches no single goal object in ['chefmate_8_frypan_1']); pick(frying_pan): partial: gripper hovers at the pan/moka handles 160-280, then grasps the pan near its handle ~300; pan swings up toward vertical while lifted
- libero_90/task21_fail_init0: skipped ('frying_pan' matches no single goal object in ['chefmate_8_frypan_1']); place(frying_pan, stove, on): failed: never transported toward the stove; arm swings away (right/up) and times out at 409
- libero_90/task21_fail_init1: skipped ('frying_pan' matches no single goal object in ['chefmate_8_frypan_1']); pick(frying_pan): failed: gripper hovers over the pan rim 207-333 without grasping, then picks the moka pot instead (340)
- libero_90/task21_fail_init1: skipped ('frying_pan' matches no single goal object in ['chefmate_8_frypan_1']); place(frying_pan, stove, on): never attempted; the moka pot is placed on the stove instead and tips over (380-409)
- libero_90/task25_fail_init1: label not used; pick(black_bowl): partial: rim grasp at ~180, lifted ~200, then bowl tilts and drops ~320-340
- libero_90/task35_fail_init0: label not used; open(microwave): partial: door swung partly open (~step 200-220) then stalled short of open
- libero_90/task41_fail_init0: skipped ('frying_pan' matches no single goal object in ['chefmate_8_frypan_1']); pick(frying_pan): never attempted: gripper never approaches the pan; it picks the white bowl instead
- libero_90/task41_fail_init0: skipped ('frying_pan' matches no single goal object in ['chefmate_8_frypan_1']); place(frying_pan, cabinet, on): never attempted
- libero_90/task41_fail_init1: skipped ('frying_pan' matches no single goal object in ['chefmate_8_frypan_1']); pick(frying_pan): never attempted: the white bowl is picked instead
- libero_90/task41_fail_init1: skipped ('frying_pan' matches no single goal object in ['chefmate_8_frypan_1']); place(frying_pan, cabinet, on): never attempted (the wrong object, the bowl, is carried to the cabinet top and falls off its edge)
- libero_90/task42_fail_init0: skipped ('frying_pan' matches no single goal object in ['chefmate_8_frypan_1']); pick(frying_pan): never attempted: the white bowl is picked instead
- libero_90/task42_fail_init0: skipped ('frying_pan' matches no single goal object in ['chefmate_8_frypan_1']); place(frying_pan, cabinet_shelf, under): never attempted (the bowl is taken to the top of the cabinet, not under the shelf)
- libero_90/task42_fail_init1: skipped ('frying_pan' matches no single goal object in ['chefmate_8_frypan_1']); pick(frying_pan): never attempted: gripper briefly hovers over the pan's back rim (80-100) without descending or closing, then picks the white bowl
- libero_90/task42_fail_init1: skipped ('frying_pan' matches no single goal object in ['chefmate_8_frypan_1']); place(frying_pan, cabinet_shelf, under): never attempted
- libero_90/task59_fail_init1: label not used; pick(tomato_sauce): partial: grasped and lifted only at ~400-409, after first moving the alphabet soup
- libero_90/task63_fail_init0: skipped ('left_bowl' matches no single goal object in ['akita_black_bowl_1', 'akita_black_bowl_2']); pick(left_bowl): never attempted: pi0 picked the right bowl (the base) instead
- libero_90/task63_fail_init0: skipped (verb 'stack' has no check); stack(left_bowl, right_bowl): never attempted
- libero_90/task63_fail_init0: skipped ('stacked_bowls' matches no single goal object in ['akita_black_bowl_1', 'akita_black_bowl_2']); place(stacked_bowls, tray, in): partial: the right (base) bowl alone was placed in the tray
- libero_90/task63_fail_init1: skipped ('left_bowl' matches no single goal object in ['akita_black_bowl_1', 'akita_black_bowl_2']); pick(left_bowl): success: robot's-left bowl (image right) grasped ~step 60-80
- libero_90/task63_fail_init1: skipped (verb 'stack' has no check); stack(left_bowl, right_bowl): never attempted: carried past the right bowl straight to the tray
- libero_90/task63_fail_init1: skipped ('stacked_bowls' matches no single goal object in ['akita_black_bowl_1', 'akita_black_bowl_2']); place(stacked_bowls, tray, in): partial: only the left (top) bowl placed in the tray; right bowl stays on the table
- libero_90/task64_fail_init0: skipped ('right_bowl' matches no single goal object in ['akita_black_bowl_2', 'akita_black_bowl_1']); pick(right_bowl): never attempted: pi0 picked the left bowl (the base) instead
- libero_90/task64_fail_init0: skipped (verb 'stack' has no check); stack(right_bowl, left_bowl): never attempted
- libero_90/task64_fail_init0: skipped ('stacked_bowls' matches no single goal object in ['akita_black_bowl_2', 'akita_black_bowl_1']); place(stacked_bowls, tray, in): partial: left (base) bowl alone placed in the tray
- libero_90/task64_fail_init1: skipped ('right_bowl' matches no single goal object in ['akita_black_bowl_2', 'akita_black_bowl_1']); pick(right_bowl): never attempted: pi0 picked the left bowl (the base) instead
- libero_90/task64_fail_init1: skipped (verb 'stack' has no check); stack(right_bowl, left_bowl): never attempted
- libero_90/task64_fail_init1: skipped ('stacked_bowls' matches no single goal object in ['akita_black_bowl_2', 'akita_black_bowl_1']); place(stacked_bowls, tray, in): partial: left (base) bowl alone placed in the tray
- libero_90/task69_fail_init0: label not used; pick(chocolate_pudding): partial: delayed ~200 steps by the wrong-object detour; pudding apparently grasped ~300-320
- libero_90/task77_fail_init0: label not used; pick(book): partial: book lifted at ~120 but grasped at its top edge and swings to horizontal in the gripper by ~140
- libero_90/task78_fail_init0: label not used; pick(book): partial: book lifted at ~120 but grasped at its top edge and swings to horizontal in the gripper by ~140
