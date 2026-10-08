"""Policy-free smoke test: scene parsing, prompt slots, decomposition, camera
projection and the retract_home primitive (all 90 libero_90 tasks by default).

  cd /opt/openpi/examples/libero && MUJOCO_GL=egl .venv/bin/python \
      /home/DockerShared/ECE780/failure_agent_prototype/tests/smoke_scene.py
"""
import pathlib
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from failure_agent.diagnosis import DiagnosisAgent  # noqa: E402
from failure_agent.library import SkillLibrary  # noqa: E402
from failure_agent.recovery import Primitives  # noqa: E402
from failure_agent.scene import SceneState  # noqa: E402

cfg = yaml.safe_load((ROOT / "config" / "agent.yaml").read_text())
suite = benchmark.get_benchmark_dict()["libero_90"]()
tasks = [int(a) for a in sys.argv[1:]] or list(range(suite.n_tasks))

for tid in tasks:
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(7)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(tid)[0])
    for _ in range(10):
        obs, *_ = env.step([0.0] * 6 + [-1.0])
    sc = SceneState(env)
    slots = DiagnosisAgent(sc, cfg["monitor"]).base_slots()
    print(f"=== task {tid}: {sc.instruction}")
    print("  goal:", [g.raw for g in sc.goal], "| targets:", sc.targets, "| goal_joints:", sc.goal_joints)
    print("  slots:", {k: v for k, v in slots.items() if k != "instruction"})
    print("  decompose:", [s["prompt"] for s in SkillLibrary.decompose(sc, cfg["recovery"]) if s["do"] == "pi0"])
    eef = obs["robot0_eef_pos"]
    print("  eef px (raw u,v):", sc.project(eef).round(1), "in_view:", sc.in_view(eef, 8),
          "| tilts:", {o: round(sc.obj_tilt_deg(o)) for o in sc.movable})

    # retract test: push the arm away with scripted actions, then retract_home
    state = {"obs": obs}

    def step(a):
        state["obs"], _, done, _ = env.step(list(a))
        return state["obs"], done
    prims = Primitives(step, lambda: state["obs"], eef.copy(), obs["robot0_eef_quat"].copy(), cfg["recovery"])
    for _ in range(30):
        step([0.3, -0.4, -0.3, 0.1, 0.0, 0.1, -1.0])
    moved = np.linalg.norm(state["obs"]["robot0_eef_pos"] - eef)
    prims.retract_home()
    err = np.linalg.norm(state["obs"]["robot0_eef_pos"] - eef)
    print(f"  retract_home: displaced {moved:.3f} m -> residual {err:.3f} m")
    env.close()
