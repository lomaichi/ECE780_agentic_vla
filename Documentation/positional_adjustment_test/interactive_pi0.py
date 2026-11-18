"""Interactive LIBERO: pi0 drives, you can interrupt and nudge the gripper, then hand back to pi0.

This is the simulation side (LIBERO client venv, Python 3.8). The window is window.py (Python 3.11 Tk;
the 3.8 venv's Tk aborts on this X server), which launches this script and talks to it over a socketpair.

The window shows the first-person (eye-in-hand / wrist) camera as the camera really sees it
(no overlays, no mirroring). The frame border tells who is in control: blue = pi0, red = you,
green = task success.

Keys (the window must have keyboard focus):
  space              start / stop the interruption. Stopping hands control back to pi0, which drops
                     its queued action chunk and replans from the current observation.
  arrow keys         translate the gripper in the horizontal plane, in the wrist view's frame
                     (up = towards the top of the image, right = towards the image's right side)
  ctrl + up/down     move the gripper vertically (world z)
  ctrl + left/right  rotate the gripper about the vertical axis (left = turn left, i.e. CCW seen from above)
  shift (hold)       fine motion
  r                  restart the episode from the same init state      n   next init state
  q / esc            quit
While interrupted, height is held unless ctrl+up/down is pressed, tilt is always held, the arm holds
still when no key is pressed, and the gripper keeps pi0's last open/close command.

pi0 gets exactly what openpi's examples/libero/main.py sends (256 px renders, rotated 180 deg,
resize_with_pad to 224, eef pos + axis-angle + gripper qpos, replan every 5 steps). The window view is
a separate, higher-resolution render of the same camera, so it does not change pi0's input.

Run (needs the pi0_libero server on port 8000, see README.md):
  ./run.sh [--task 2] [--init 0] [--agentview]     (all options: ./run.sh --help)
Headless check of the control mapping (no window, no keyboard):
  ./run.sh --selftest
"""

import argparse
import os
import sys
import threading
import time
from collections import deque

import numpy as np

SUITE = "libero_90"
RESIZE, REPLAN_STEPS, NUM_WAIT = 224, 5, 10  # as openpi examples/libero/main.py
DUMMY = [0.0] * 6 + [-1.0]
POS_GAIN = 10.0  # action units per metre of pose error (1 unit = 5 cm OSC delta)
ROT_GAIN = 1.0  # action units per radian of error (1 unit = 0.5 rad)
LEASH = 0.03  # m: the teleop target never runs further than this ahead of the gripper
WRIST, AGENT = "robot0_eye_in_hand", "agentview"


# ---------------------------------------------------------------- helpers ---------------------------

def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def upright(img):
    """What pi0 is trained on: LIBERO's (upside-down) render rotated 180 deg."""
    return np.ascontiguousarray(img[::-1, ::-1])


def camera_view(img):
    """The camera image as the camera sees it (MuJoCo renders bottom row first)."""
    return np.ascontiguousarray(img[::-1])


def pi0_request(obs, prompt):
    from openpi_client import image_tools

    def prep(img):
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(upright(img), RESIZE, RESIZE))

    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    return {"observation/image": prep(obs["agentview_image"]), "observation/wrist_image": prep(obs["robot0_eye_in_hand_image"]),
            "observation/state": state, "prompt": prompt}


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def flat(v):
    v = np.array([v[0], v[1], 0.0])
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def make_env(task_id, seed, resolution):
    import pathlib

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # big horizon: interruptions add steps, and robosuite refuses to step past it
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution, horizon=100000)
    env.seed(seed)
    return env, task, suite.get_task_init_states(task_id)


# ---------------------------------------------------------------- shared state (sim thread <-> window)

class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.keys = set()  # held: up down left right ctrl shift
        self.human = False
        self.commands = deque()  # "reset", "next", "quit"
        self.frame = None  # (wrist rgb, agent rgb or None, status dict)
        self.frame_id = 0

    def snapshot(self):
        with self.lock:
            return self.human, set(self.keys), (self.commands.popleft() if self.commands else None)


# ---------------------------------------------------------------- simulation -----------------------

class Teleop:
    """Integrates held keys into a target pose and servoes the OSC controller towards it."""

    def __init__(self, a, fps):
        self.a, self.dt = a, 1.0 / fps
        self.target_pos = self.target_R = None

    def start(self, env, obs):
        import robosuite.utils.transform_utils as T

        self.target_pos = obs["robot0_eef_pos"].copy()
        self.target_R = T.quat2mat(obs["robot0_eef_quat"])
        self.start_pos, self.start_R = self.target_pos.copy(), self.target_R.copy()

    def wrist_axes(self, env):
        """World-frame horizontal unit vectors for 'image up' and 'image right' of the wrist view.

        MuJoCo camera frame: +x = image right, +y = image up, looking along -z. For a camera tilted down,
        the ground direction that moves up the image is the horizontal part of (+y) + (forward = -z).
        """
        sim = env.env.sim
        R = sim.data.cam_xmat[sim.model.camera_name2id(WRIST)].reshape(3, 3)
        return flat(R[:, 1] - R[:, 2]), flat(R[:, 0])

    def action(self, env, obs, keys, grip):
        import robosuite.utils.transform_utils as T

        a, k = self.a, keys
        scale = a.fine if "shift" in k else 1.0
        up, right = self.wrist_axes(env)
        pos, R = obs["robot0_eef_pos"], T.quat2mat(obs["robot0_eef_quat"])

        if "ctrl" in k:  # ctrl: up/down = world vertical, left/right = yaw
            move = np.array([0.0, 0.0, 1.0]) * (("up" in k) - ("down" in k))
            self.target_R = rotz(np.deg2rad(a.rot_speed) * scale * self.dt * (("left" in k) - ("right" in k))) @ self.target_R
        else:  # horizontal plane, in the wrist image's frame
            move = up * (("up" in k) - ("down" in k)) + right * (("right" in k) - ("left" in k))
        if np.linalg.norm(move) > 0:
            self.target_pos = self.target_pos + a.trans_speed * scale * self.dt * move / np.linalg.norm(move)
        lag = self.target_pos - pos  # don't let the target run away if the arm is blocked (e.g. by the table)
        if np.linalg.norm(lag) > LEASH:
            self.target_pos = pos + LEASH * lag / np.linalg.norm(lag)

        act = np.zeros(7)
        act[:3] = np.clip(POS_GAIN * (self.target_pos - pos), -1, 1)
        act[3:6] = np.clip(ROT_GAIN * quat2axisangle(T.mat2quat(self.target_R @ R.T)), -1, 1)
        act[6] = grip
        return act

    def summary(self, obs):
        import robosuite.utils.transform_utils as T

        d = 100 * (obs["robot0_eef_pos"] - self.start_pos)
        Rrel = T.quat2mat(obs["robot0_eef_quat"]) @ self.start_R.T
        yaw = np.rad2deg(np.arctan2(Rrel[1, 0], Rrel[0, 0]))
        return f"moved dx {d[0]:+.1f} dy {d[1]:+.1f} dz {d[2]:+.1f} cm (|xy| {np.hypot(d[0], d[1]):.1f}), yaw {yaw:+.1f} deg"


class Sim:
    def __init__(self, a, shared):
        self.a, self.S = a, shared
        self.teleop = Teleop(a, a.fps)
        self.sink = None  # set by serve(): sends each frame to the window

    def render_views(self, obs):
        sim, a = self.env.env.sim, self.a
        if a.view_res > 0:
            wrist = camera_view(sim.render(camera_name=WRIST, width=a.view_res, height=a.view_res))
            agent = camera_view(sim.render(camera_name=AGENT, width=a.view_res, height=a.view_res)) if a.agentview else None
        else:
            wrist = camera_view(obs["robot0_eye_in_hand_image"])
            agent = camera_view(obs["agentview_image"]) if a.agentview else None
        return wrist, agent

    def publish(self, obs, status):
        wrist, agent = self.render_views(obs)
        with self.S.lock:
            self.S.frame = (wrist, agent, status)
            self.S.frame_id += 1
        if self.sink is not None:
            self.sink(wrist, agent, status)

    def run(self, controller=None):
        """Episode loop. `controller(t)` -> (human, keys, cmd) replaces the keyboard (self-test)."""
        from openpi_client import websocket_client_policy

        a, S = self.a, self.S
        np.random.seed(a.seed)
        client = websocket_client_policy.WebsocketClientPolicy(a.host, a.port)
        self.env, task, inits = make_env(a.task, a.seed, a.resolution)
        prompt, init = str(task.language), a.init
        print(f"{SUITE} task {a.task}: {prompt!r}  ({len(inits)} init states)", flush=True)

        while True:
            self.env.reset()
            obs = self.env.set_init_state(inits[init])
            with S.lock:
                S.human = False
                S.commands.clear()
            plan, grip, t, success, prev_human, takeovers, human_steps, t0 = deque(), -1.0, 0, False, False, 0, 0, 0
            print(f"--- episode: init {init}", flush=True)
            while True:
                tick = time.time()
                human, keys, cmd = controller(t) if controller else S.snapshot()
                if cmd == "quit":
                    return obs
                if cmd in ("reset", "next"):
                    init = (init + 1) % len(inits) if cmd == "next" else init
                    break

                if human and not prev_human and not success:
                    takeovers, t0 = takeovers + 1, t
                    self.teleop.start(self.env, obs)
                    print(f"[step {t}] interrupt #{takeovers}", flush=True)
                if prev_human and not human and not success:
                    plan.clear()  # pi0 replans from the current observation
                    print(f"[step {t}] hand back to pi0 after {t - t0} steps: {self.teleop.summary(obs)}", flush=True)
                prev_human = human

                if success:
                    action = None  # freeze; r / n to continue
                elif t < NUM_WAIT:
                    action = DUMMY
                elif human:
                    action = self.teleop.action(self.env, obs, keys, grip)
                    human_steps += 1
                else:
                    if not plan:
                        chunk = np.asarray(client.infer(pi0_request(obs, prompt))["actions"])
                        plan.extend(chunk[:REPLAN_STEPS])
                    action = plan.popleft()
                    grip = 1.0 if action[6] > 0 else -1.0

                if action is not None:
                    obs, _, done, _ = self.env.step([float(x) for x in action])
                    t += 1
                    if done and not success:
                        success = True
                        print(f"[step {t}] SUCCESS (takeovers {takeovers}, human steps {human_steps})", flush=True)

                mode = "SUCCESS" if success else ("HUMAN" if human else "PI0")
                self.publish(obs, {"mode": mode, "step": t, "init": init, "takeovers": takeovers,
                                   "gripper": "closed" if grip > 0 else "open", "prompt": prompt})
                if controller is None:
                    time.sleep(max(0.0, 1.0 / a.fps - (time.time() - tick)))


# ---------------------------------------------------------------- link to window.py ----------------
# window.py (Python 3.11, Tk) starts this script with --ipc-fd: one end of a socketpair.
#   window -> sim: JSON lines {"keys": [...held...]} | {"toggle": 1} | {"cmd": "reset"|"next"|"quit"}
#   sim -> window: per frame, !II (header length, payload length) + JSON header + raw RGB payload
#                  (wrist image, then the agentview image if shown)

def serve(a):
    import json
    import socket
    import struct

    sock = socket.socket(fileno=a.ipc_fd)
    S = Shared()

    def reader():
        for line in sock.makefile("rb"):
            m = json.loads(line)
            with S.lock:
                if "keys" in m:
                    S.keys = set(m["keys"])
                if m.get("toggle"):
                    S.human = not S.human
                if "cmd" in m:
                    S.commands.append(m["cmd"])
        with S.lock:  # window closed
            S.commands.append("quit")

    def sink(wrist, agent, status):
        views = [wrist] + ([agent] if agent is not None else [])
        head = json.dumps(dict(status, sizes=[v.shape[:2] for v in views])).encode()
        payload = b"".join(v.tobytes() for v in views)
        try:
            sock.sendall(struct.pack("!II", len(head), len(payload)) + head + payload)
        except OSError:
            with S.lock:
                S.commands.append("quit")

    threading.Thread(target=reader, daemon=True).start()
    sim = Sim(a, S)
    sim.sink = sink
    sim.run()


# ---------------------------------------------------------------- self-test ------------------------

def selftest(a):
    """Scripted keys (no window): let pi0 run, interrupt, move right / up / rotate left / raise / lower, hand back.

    Checks that each key moves the gripper the right way in the wrist camera's frame, and saves the
    view before/after each move to selftest/ for an eyeball check.
    """
    import cv2
    import robosuite.utils.transform_utils as T

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest")
    os.makedirs(out, exist_ok=True)
    S = Shared()
    sim = Sim(a, S)
    plan = [(0, False, set()), (40, True, set()), (45, True, {"right"}), (75, True, set()), (80, True, {"up"}),
            (110, True, set()), (115, True, {"ctrl", "left"}), (145, True, set()), (150, True, {"ctrl", "up"}),
            (180, True, set()), (185, True, {"ctrl", "down"}), (215, True, set()), (220, False, set()), (250, None, None)]
    marks, results, n = {}, [], [0]

    def controller(_):
        t = n[0]  # count calls, not env steps, so the script also ends if the episode freezes on success
        n[0] += 1
        if t in (45, 75, 80, 110, 115, 145, 150, 180, 185, 215):  # pose + camera axes + view at segment boundaries
            obs = sim.last_obs
            up, right = sim.teleop.wrist_axes(sim.env)
            marks[t] = (obs["robot0_eef_pos"].copy(), T.quat2mat(obs["robot0_eef_quat"]), up, right)
            cv2.imwrite(f"{out}/t{t:03d}.png", cv2.cvtColor(S.frame[0], cv2.COLOR_RGB2BGR))
        human, keys = next((h, k) for s, h, k in reversed(plan) if t >= s)
        return (bool(human), keys, "quit" if human is None else None)

    orig_publish = sim.publish

    def publish(obs, status):
        sim.last_obs = obs
        orig_publish(obs, status)

    sim.publish = publish
    sim.run(controller)

    for (s0, s1, axis, label) in [(45, 75, 3, "right -> image right"), (80, 110, 2, "up -> image up")]:
        p0, _, up, right = marks[s0]
        d = marks[s1][0] - p0
        along = d @ (right if axis == 3 else up)
        results.append((label, along > 0.02 and abs(d[2]) < 0.01, f"{100 * along:+.1f} cm along, dz {100 * d[2]:+.1f} cm"))
    Rrel = marks[145][1] @ marks[115][1].T
    yaw = np.rad2deg(np.arctan2(Rrel[1, 0], Rrel[0, 0]))
    results.append(("ctrl+left -> CCW from above", yaw > 5, f"yaw {yaw:+.1f} deg"))
    for (s0, s1, sign, label) in [(150, 180, 1, "ctrl+up -> world up"), (185, 215, -1, "ctrl+down -> world down")]:
        d = marks[s1][0] - marks[s0][0]
        results.append((label, sign * d[2] > 0.02 and np.hypot(d[0], d[1]) < 0.01,
                        f"dz {100 * d[2]:+.1f} cm, |dxy| {100 * np.hypot(d[0], d[1]):.1f} cm"))
    for label, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {label}: {msg}")
    print(f"views saved in {out}")
    return all(ok for _, ok, _ in results)


# ---------------------------------------------------------------- main -----------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=int, default=14, help="LIBERO-90 task id")
    p.add_argument("--init", type=int, default=0, help="index into the task's 50 init states")
    p.add_argument("--seed", type=int, default=7, help="env.seed() and np.random.seed(), as main.py")
    p.add_argument("--resolution", type=int, default=256, help="pi0 camera resolution (keep 256)")
    p.add_argument("--view-res", type=int, default=640, help="window render resolution (0 = show pi0's 256 px frames)")
    p.add_argument("--agentview", action="store_true", help="also show the third-person agentview camera")
    p.add_argument("--trans-speed", type=float, default=0.08, help="m/s while an arrow key is held")
    p.add_argument("--rot-speed", type=float, default=30.0, help="deg/s while ctrl+left/right is held")
    p.add_argument("--fine", type=float, default=0.25, help="speed factor while shift is held")
    p.add_argument("--fps", type=float, default=20.0, help="control steps per second (LIBERO data is 20 Hz)")
    p.add_argument("--host", default="127.0.0.1", help="pi0 server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--selftest", action="store_true", help="headless scripted-key check, see selftest()")
    p.add_argument("--ipc-fd", type=int, default=None, help=argparse.SUPPRESS)  # set by window.py
    a = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    if a.selftest:
        code = 0 if selftest(a) else 1
    elif a.ipc_fd is not None:
        serve(a)
        code = 0
    else:
        sys.exit("start the window with ./run.sh (window.py launches this script); --selftest runs headless")
    sys.stdout.flush()
    os._exit(code)  # skip robosuite's EGL teardown, which raises noisy errors at interpreter exit


if __name__ == "__main__":
    main()
