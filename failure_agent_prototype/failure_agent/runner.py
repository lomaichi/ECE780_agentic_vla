"""Closed-loop episode runner: pi0 action client + symptom monitor + diagnosis
agent + skill library.

mode="observe": native pi0 (same loop as openpi's examples/libero/main.py);
                the monitor records triggers without interrupting, and the
                diagnosis agent labels the episode once at the end.
mode="agent":   triggers interrupt pi0 -> diagnose -> library plan -> recover,
                up to agent.max_recoveries times.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import time
from typing import List, Optional

import cv2
import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from robosuite.controllers import load_controller_config

from .diagnosis import DiagnosisAgent
from .library import SkillLibrary
from .monitor import SymptomMonitor
from .recovery import Primitives
from .scene import SceneState

DUMMY_ACTION = [0.0] * 6 + [-1.0]


class BudgetExceeded(Exception):
    """Raised by _step when env.max_total_steps is reached (ends the episode)."""


def _quat2axisangle(quat):
    quat = np.array(quat, dtype=float)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


class EpisodeRunner:
    def __init__(self, cfg: dict, client, library: SkillLibrary, out_dir: pathlib.Path,
                 save_video: bool = True, log=print):
        self.log = log            # console lines (run_agent buffers them per episode)
        self.cfg = cfg
        self.client = client
        self.library = library
        self.out_dir = out_dir
        self.save_video = save_video
        self.suite = benchmark.get_benchmark_dict()[cfg["env"]["suite"]]()
        self._envs = {}

    # ------------------------------------------------------------------ env
    def _get_env(self, task_id: int):
        if task_id not in self._envs:
            task = self.suite.get_task(task_id)
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            res = self.cfg["env"]["resolution"]
            # robosuite refuses to step past `horizon`; keep it above our own cap
            horizon = self.cfg["env"]["max_total_steps"] + self.cfg["env"]["num_steps_wait"] + 100
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=res, camera_widths=res,
                                     horizon=horizon)
            env.seed(self.cfg["env"]["seed"])
            self._envs[task_id] = (env, task)
        return self._envs[task_id]

    def release_env(self, task_id: int):
        if task_id in self._envs:
            self._envs.pop(task_id)[0].close()

    def close(self):
        for tid in list(self._envs):
            self.release_env(tid)

    # -------------------------------------------------------------- episode
    def run(self, task_id: int, init_idx: int, mode: str) -> dict:
        cfg = self.cfg
        env, task = self._get_env(task_id)
        env.reset()
        obs = env.set_init_state(self.suite.get_task_init_states(task_id)[init_idx])
        for _ in range(cfg["env"]["num_steps_wait"]):
            obs, _, _, _ = env.step(DUMMY_ACTION)

        ep_dir = self.out_dir / f"task{task_id}_init{init_idx}_{mode}"
        (ep_dir / "snapshots").mkdir(parents=True, exist_ok=True)

        self.env, self.obs, self.t, self.done = env, obs, 0, False
        self.success_t: Optional[int] = None
        self.frames: List[np.ndarray] = []
        self.label = "pi0: " + task.language
        scene = SceneState(env)
        self.scene = scene
        mon = SymptomMonitor(scene, cfg["monitor"])
        self.mon = mon
        self.q_home = np.array(env.env.robots[0]._joint_positions)
        prims = Primitives(self._step, lambda: self.obs, obs["robot0_eef_pos"].copy(),
                           obs["robot0_eef_quat"].copy(), cfg["recovery"], joint_home=self._joint_home)
        self.prims = prims
        mon.home_quat = obs["robot0_eef_quat"].copy()

        trace = {"task_id": task_id, "init_idx": init_idx, "mode": mode, "instruction": task.language,
                 "goal": [g.raw for g in scene.goal], "targets": scene.targets,
                 "goal_joints": scene.goal_joints, "diagnoses": []}
        t0 = time.time()

        try:
            self._episode(mode, task, trace, ep_dir)
        except BudgetExceeded:
            trace["budget_exhausted"] = True
        # library statistics: a recovery counts if the goal was reached before the next diagnosis
        if mode == "agent":
            for i, d in enumerate(trace["diagnoses"]):
                recovered = self.done and i == len(trace["diagnoses"]) - 1
                d["outcome"] = "recovered" if recovered else "not_recovered"
                self.library.record(d["code"], recovered)

        trace.update({
            "success": bool(self.done), "success_t": self.success_t, "steps": self.t,
            "native_budget_success": bool(self.done and self.success_t <= cfg["env"]["max_steps"]),
            "wall_s": round(time.time() - t0, 1),
            "events": [e.to_dict() for e in mon.events],
            "final": mon.snapshot(),
        })
        (ep_dir / "trace.json").write_text(json.dumps(trace, indent=1, default=_jsonable))
        if self.save_video and self.frames:
            imageio.mimwrite(ep_dir / f"rollout_{'success' if self.done else 'failure'}.mp4",
                             self.frames, fps=10)
        return trace

    def _episode(self, mode, task, trace, ep_dir):
        cfg, mon, scene, env = self.cfg, self.mon, self.scene, self.env
        diag_agent = DiagnosisAgent(scene, cfg["monitor"])
        if mode == "observe":
            self._run_pi0(task.language, "goal", cfg["env"]["max_steps"], interrupt=False)
            if not self.done:
                # label with the first trigger that would have interrupted pi0
                first = next((e.kind for e in mon.events if e.kind in self._enabled()), "timeout")
                d = diag_agent.diagnose(first, self.t, mon)
                trace["diagnoses"].append({**d.to_dict(), "plan": None, "outcome": "posthoc"})
        else:
            agenda = [{"do": "pi0", "prompt": task.language, "until": "goal",
                       "budget": cfg["env"]["max_steps"]}]
            n_rec, entry_uses = 0, collections.Counter()
            while not self.done and self.t < cfg["env"]["max_total_steps"]:
                outcome, trig = "agenda_empty", None
                while agenda and not self.done:
                    step = agenda.pop(0)
                    outcome, trig = self._exec(step)
                    if outcome == "trigger":
                        break
                    if outcome == "budget" and step.get("until") == "goal":
                        trig = "timeout"
                        break
                if self.done:
                    break
                if n_rec >= cfg["agent"]["max_recoveries"]:
                    break
                trig = trig or "timeout"
                d = diag_agent.diagnose(trig, self.t, mon)
                np.savez_compressed(ep_dir / "snapshots" / f"t{self.t:04d}_{d.code}.npz",
                                    state=env.get_sim_state(), t=self.t)
                plan = self.library.build_plan(d, scene, entry_uses[d.code], cfg["recovery"])
                entry_uses[d.code] += 1
                n_rec += 1
                trace["diagnoses"].append({**d.to_dict(), "plan": [dict(s) for s in plan]})
                self.log(f"    [t={self.t}] {trig} -> {d.code}: {d.evidence[0]}")
                for s in plan:
                    if s["do"] == "pi0":
                        self.log(f"        pi0: \"{s['prompt']}\"")
                agenda = list(plan)

    # -------------------------------------------------------------- helpers
    def _enabled(self):
        return {k for k, v in self.cfg["agent"]["trigger_on"].items() if v}

    def _step(self, action):
        """One env step; used by pi0 phases and scripted primitives alike."""
        if self.t >= self.cfg["env"]["max_total_steps"]:
            raise BudgetExceeded()
        self.obs, _, done, _ = self.env.step(list(np.asarray(action, dtype=float)))
        self.t += 1
        if done and not self.done:
            self.done, self.success_t = True, self.t
        if self.save_video:
            self.frames.append(self._frame())
        return self.obs, self.done

    def _frame(self):
        img = np.ascontiguousarray(self.obs["agentview_image"][::-1, ::-1])
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(img, (0, 0), (448, 34), (0, 0, 0), -1)
        cv2.putText(img, f"t={self.t}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(img, self.label[:70], (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return img

    def _exec(self, step):
        do = step["do"]
        if do == "pi0":
            return self._run_pi0(step["prompt"], step["until"], step["budget"], interrupt=True)
        self.label = f"recovery: {do}" + "".join(f" {step[k]}" for k in ("obj", "joint", "dest") if k in step)
        if do == "approach" and "joint" in step:
            self.prims.approach(self.scene.handle_point(step["joint"]), dz=self.cfg["recovery"]["handle_dz_m"])
        elif do == "approach":
            self.prims.approach(self.scene.obj_pos(step["obj"]))
        elif do == "carry_to":
            if self.mon._held == step["obj"] or step["obj"] in self.scene.grasped():
                self.prims.carry_to(self.scene.obj_pos(step["dest"]), lambda: self.scene.obj_pos(step["obj"]))
            else:
                self.mon._emit(self.t, "primitive_skipped", primitive="carry_to", reason=f"{step['obj']} not held")
        elif do == "finish_articulation":
            self._finish_articulation(step["joint"])
        else:
            getattr(self.prims, do)()
        if self.prims.incomplete:
            self.mon._emit(self.t, "primitive_incomplete", primitive=do, residual_m=self.prims.incomplete)
            self.prims.incomplete = []
        self.mon.reset_phase(self.t)
        return ("success" if self.done else "done"), None

    def _joint_home(self, gripper: float) -> bool:
        """Drive the arm back to its start joint configuration with robosuite's
        JOINT_POSITION controller, then restore OSC_POSE (which re-targets the
        current pose). Used when a Cartesian retract is stuck at the elbow limit."""
        robot, base = self.env.env.robots[0], self.env.env
        osc_cfg = dict(robot.controller_config)
        rc = self.cfg["recovery"]
        jcfg = load_controller_config(default_controller="JOINT_POSITION")
        jcfg.update(kp=rc["joint_home_kp"], output_max=rc["joint_home_step_rad"],
                    output_min=-rc["joint_home_step_rad"])
        robot.controller_config = jcfg
        robot._load_controller()
        base._action_dim = robot.action_dim
        try:
            for _ in range(rc["joint_home_max_steps"]):
                dq = self.q_home - np.array(robot._joint_positions)
                if np.abs(dq).max() < 0.02:
                    break
                self._step(np.concatenate([np.clip(dq / rc["joint_home_step_rad"], -1, 1), [gripper]]))
                if self.done:
                    return True
        finally:
            robot.controller_config = osc_cfg
            robot._load_controller()
            base._action_dim = robot.action_dim
        self.mon._emit(self.t, "joint_home", residual_rad=round(float(np.abs(self.q_home - np.array(
            robot._joint_positions)).max()), 3))
        return self.done

    def _finish_articulation(self, joint: str):
        """Scripted push of a slide joint (drawer) at its handle, along the axis,
        by the remaining distance. Skipped (and logged) for hinges/knobs."""
        tgt = self.mon.artic_targets.get(joint)
        info = self.scene.joints.get(joint, {})
        if tgt is None or info.get("type") != "slide":
            self.mon._emit(self.t, "primitive_skipped", primitive="finish_articulation", joint=joint,
                           reason=f"joint type {info.get('type')}" if tgt is not None else "no target range")
            return
        q = self.scene.joint_qpos()[joint]
        direction = self.scene.joint_axis_world(joint) * np.sign(tgt - q)
        handle = self.scene.slide_push_point(joint, direction)
        self.prims.finish_slide(handle, self.prims.home_quat, direction, abs(tgt - q))
        self.mon._emit(self.t, "finish_articulation", joint=joint, q_before=round(q, 4),
                       q_after=round(self.scene.joint_qpos()[joint], 4), target=round(tgt, 4))

    def _until_met(self, until) -> bool:
        if until == "goal":
            return self.done
        if isinstance(until, dict) and "predicate" in until:
            return self.scene.eval_predicate(until["predicate"])
        if isinstance(until, dict) and "grasp" in until:
            # held and lifted, as confirmed by the monitor (updated after this check)
            return self.mon._held == until["grasp"]
        return False

    def _run_pi0(self, prompt: str, until, budget: int, interrupt: bool):
        cfg = self.cfg
        self.label = "pi0: " + prompt
        self.mon.reset_phase(self.t)
        self.mon.focus = until["grasp"] if isinstance(until, dict) and "grasp" in until else None
        plan = collections.deque()
        enabled = self._enabled()
        start = self.t
        while self.t - start < budget and self.t < cfg["env"]["max_total_steps"]:
            if not plan:
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                    np.ascontiguousarray(self.obs["agentview_image"][::-1, ::-1]),
                    cfg["policy"]["resize_size"], cfg["policy"]["resize_size"]))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                    np.ascontiguousarray(self.obs["robot0_eye_in_hand_image"][::-1, ::-1]),
                    cfg["policy"]["resize_size"], cfg["policy"]["resize_size"]))
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/state": np.concatenate((self.obs["robot0_eef_pos"],
                                                         _quat2axisangle(self.obs["robot0_eef_quat"]),
                                                         self.obs["robot0_gripper_qpos"])),
                    "prompt": prompt,
                }
                chunk = self.client.infer(element)["actions"]
                plan.extend(chunk[: cfg["policy"]["replan_steps"]])
            action = plan.popleft()
            self.prims.gripper_cmd = float(action[-1])
            self._step(action)
            if self.done:
                return "success", None
            if self._until_met(until):
                return "until_met", None
            trigs = [e for e in self.mon.update(self.t, self.obs) if e.kind in enabled]
            if trigs and interrupt:
                return "trigger", trigs[0].kind
        return "budget", None
