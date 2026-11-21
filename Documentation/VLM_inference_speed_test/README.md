# VLM inference speed test

Live closed loop (pi0 + VLM subtask judge at 1 fps) with per-component timing, to find a VLM fast enough to
replace Opus as the live judge. Files:

- `live_judge.py` – runs one libero_90 episode, live window and/or `--video` (HUD with model name and the
  inference time of every VLM call; frames where the sim is frozen waiting for the VLM are included and marked)
- `judge_agent.py` – `SubtaskJudgeAgent` (full JSON judge + corrections, for Opus), `SubtaskSuccessJudge`
  (yes/no question per call), `GroundedSuccessJudge` (Moondream2 `detect` + proprioception)
- `vlm_backends.py` – Claude CLI, Moondream cloud API, `MoondreamLocal` (Moondream2 on the local GPU)
- `moondream_server.py` – HTTP server for Moondream2 (own venv; started automatically by `MoondreamLocal`)

## Moondream2 (local)

- Weights: `ECE780/model/moondream2` (`vikhyatk/moondream2`, revision 2025-06-21), tokenizer cached in
  `ECE780/model/hf_cache`, venv `ECE780/model/.venv-moondream` (py3.11, torch 2.11 cu128: the container's torch
  cu126 builds have no sm_120 kernels for the RTX 5090). ~5 GB of GPU memory next to pi0's ~24 GB.
- Run (pi0 server up):
  ```
  MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python live_judge.py --task 10 --init 0 --vlm moondream2 \
      --no-window --speed 0 --run-past-success --video video/moondream2_grounded_task10_init0.mp4 \
      --out video/moondream2_grounded_task10_init0.json
  ```
  `--judge success` for the yes/no judge; drop `--no-window` for the live window.

## Results (2026-11-20, RTX 5090, 1 judge call per second of sim time)

| | per call (round trip) | GPU part | episode pipeline |
|---|---|---|---|
| Moondream2 `detect` (grounded judge) | 76 ms (p95 78 ms) | 61 ms | 2.2–2.3x real time, pi0 57 ms/chunk |
| Moondream2 yes/no `query` | 74 ms | ~50 ms | |
| Moondream2 full JSON judge prompt (judge_agent.SubtaskJudgeAgent) | 0.8–0.9 s | | prompt ~1100 tokens + 730 image tokens of the 2048 context |

Judge quality vs the privileged sim checks (drawn in the videos as ground truth, never shown to the agent):

- Yes/no VQA: unusable. Moondream2 answers "yes" (or always "no") whatever the image; 6 phrasings x 3 views
  over 97 labelled frames, none separated pick from no-pick. `video/moondream2_vqa_task10_init0.mp4`: "task
  complete" at step 50, before the arm moved.
- Full JSON judge prompt: same failure (pick "success" at step 10) and it nearly overflows the context.
- Grounded (detect + gripper proprioception, final version, 5 episodes): pick judged within one call (±20 steps) of ground truth in 4 of 5
  episodes, the miss was a lift shorter than 1 s; no false pick after requiring the box to rise (plain
  displacement fired on a push, fixed). No false place. pi0 never completed a place in these runs, so place
  successes are untested; one place false positive (bowl at the cabinet edge) was caught by the
  confirm-twice rule in an earlier version.

Continuous monitoring (grounded judge, 2026-11-20): a subtask success is no longer final. Every call also
re-checks the completed subtasks from the same `detect` call + proprioception (pick: the object stays in the
gripper or is already at the destination; place: the object, when detected, is still at the destination). A
subtask contradicted on 2 consecutive calls (`LOST_CONFIRM`) is undone and the judge goes back to it; the lost
completion stays in `subtasks[k]["history"]` and the HUD shows `SUCCESS @90 LOST @170`. After a lost pick the lift
reference is re-based to where the object comes to rest. Only positive evidence counts (undetected object = no
contradiction). The ground-truth row now also shows whether a latched check still holds (`now held / DROPPED`).
Task 9 init 2 (bowl dropped next to the plate at ~step 142): before, `pick` stayed SUCCESS for the rest of the
episode (`video/moondreamV2_1fps/moondream2_grounded_task9_init2.mp4`); now contradicted at 150, LOST at 170
(`..._task9_init2_continuous.mp4`). Task 10 init 0 (bowl carried throughout): no false loss.

Videos: `video/moondream2_grounded_task10_init0.mp4`, `video/moondream2_grounded_task9_init2.mp4`,
`video/moondream2_vqa_task10_init0.mp4`; other runs in `video/attempts/`. Each has a json timing report.
