"""Watch the closed loop (pi0 + VLM subtask judge) live, and measure where the wall-clock time goes.

Same episode as failure_agent_prototype/tools/run_judge.py (fa_core.loop.run_episode + the judge agent;
here the local copy judge_agent.py), plus:
  - a live window (OpenCV/Qt): front + wrist camera, current subtask, latest VLM judgement, corrections,
    and a timing HUD. While the VLM or pi0 is thinking the window stays responsive and shows a timer.
  - optionally (--mujoco-viewer) the interactive MuJoCo viewer on the same sim (rotate / zoom with the mouse).
  - per-component timing: pi0 inference (per chunk), VLM calls (plan + judge), env step + render,
    and the overall real-time factor. Printed at the end and written to --out (json) if given.

The VLM call blocks the control loop (as in run_judge.py), so the sim is frozen while the VLM thinks;
that stall is exactly what "VLM" time in the report measures.

With --video the same window (camera views + HUD, model name, per-call VLM inference times) is written to an
mp4, one frame per control step (20 fps = sim time). While the VLM thinks the sim is frozen; those seconds
are written too (wall-clock frames marked "sim frozen"), so the video shows each judge call's stall. The
episode's privileged subtask checks (fa_core.predicates) are drawn as ground truth (display only: the agent
never sees them). --no-window --video renders headless.

Keys in the camera window:  space = pause/resume,  f = toggle real-time pacing,  q / Esc = quit.

Inside the container (pi0 server up via ./serve_pi0.sh; open a second shell with ./libero_pi0.sh):
  cd Documentation/VLM_inference_speed_test
  MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python live_judge.py --task 10 --init 0 --mujoco-viewer
  ... --no-vlm         pi0 only (baseline latency, no VLM monitoring)
  ... --speed 0        run as fast as the pipeline allows (no real-time pacing)
  ... --vlm moondream2 --no-window --speed 0 --video video/moondream2_task10_init0.mp4
                       Moondream2 on the local GPU (model/moondream2, moondream_server.py started if needed),
                       grounded success judge (--judge grounded, the default for moondream2) at 5 fps
  ... --vlm moondream2 --judge recovery
                       grounded success judge + grounded failure detection and high-level recovery
                       (recovery_agent.py). Every judge call is printed to the terminal as it happens:
                       success detection, failure verdict and recovery decision (--print events: only the
                       calls where something happens); ground-truth predicate changes are printed too.
"""

import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "failure_agent_prototype"))
sys.path.insert(0, str(HERE))
from fa_core import camera, common, loop, predicates  # noqa: E402
import judge_agent  # noqa: E402  (local copy, see judge_agent.py)
import recovery_agent  # noqa: E402
import vlm_backends as V  # noqa: E402

SUITE = "libero_90"
WIN = "closed loop (live)"
VIEW = 448                      # displayed size of each camera view
PANEL_H = 272
PANEL_H_RECOVERY = 330          # two more HUD lines: failure monitor and corrections
DT = 1.0 / common.CONTROL_FREQ

WHITE, GREY, GREEN, RED, YELLOW, ORANGE, CYAN = ((235, 235, 235), (150, 150, 150), (90, 200, 60), (60, 70, 230),
                                                 (60, 210, 250), (40, 150, 250), (230, 200, 90))  # BGR


class UserQuit(Exception):
    pass


# ------------------------------------------------------------------ terminal log ---------------------

TTY = sys.stdout.isatty()
ANSI = {"green": 32, "red": 31, "yellow": 33, "orange": "38;5;208", "cyan": 36, "grey": 90, "bold": 1}


def paint(text, *styles):
    if not TTY or not styles:
        return text
    return "".join(f"\033[{ANSI[c]}m" for c in styles) + text + "\033[0m"


def print_decision(d, every_call=True):
    """One terminal line per judge call: success detection | failure verdict | decision."""
    if d.get("type") == "plan":
        print(paint(f"[step {d['step']:4d}] plan: {d.get('subtasks')}  object '{d.get('detect_object')}' "
                    f"({d.get('object_qualifier')}) -> {d.get('target')}  destination '{d.get('detect_destination')}' "
                    f"-> {d.get('dest')}", "cyan"), flush=True)
        return
    dec, fail = d.get("decision", ""), d.get("failure", "none")
    event = bool(dec) or fail not in (None, "none") or d.get("failure_suspected") or d.get("monitor") or "error" in d
    if not (every_call or event):
        return
    head = f"[step {d['step']:4d} | {d['step'] * DT:5.1f} s]"
    if "error" in d:
        print(paint(f"{head} VLM error: {d['error']}", "red"), flush=True)
        return
    obs = d.get("observation") or d.get("note") or ""
    ok = d.get("subtask_success")
    parts = [paint(obs, "green" if ok else "grey" if ok is False else "")]
    sig = d.get("signals") or {}
    if sig.get("d_target") is not None or sig.get("d_dest") is not None:
        kind = "place" if str(d.get("subtask", "")).startswith("place") else "pick"
        k = "d_dest" if kind == "place" else "d_target"
        if sig.get(k) is not None:
            z, ap = sig["dz_dest" if kind == "place" else "dz_target"], sig["a_dest" if kind == "place" else "a_target"]
            parts.append(paint(f"EE-{'dest' if kind == 'place' else 'target'} {sig[k] * 100:4.1f} cm xy "
                               f"{z * 100:+5.1f} cm z, {ap * 100:4.1f} cm to the "
                               f"{'release' if kind == 'place' else 'pre-grasp'} point", "grey"))
    if fail == "high_level":
        parts.append(paint(f"HIGH-LEVEL FAILURE {d.get('failure_mode')} [{d.get('library')}]: {d.get('reason')}",
                           "yellow", "bold"))
    elif fail == "low_level":
        parts.append(paint(f"low-level failure {d.get('failure_mode')} [{d.get('library')}]: {d.get('reason')}",
                           "orange"))
    elif d.get("failure_suspected"):
        parts.append(paint(f"suspect {d['failure_suspected']}: {d.get('failure_reason')}", "grey"))
    if dec:
        col = ("cyan", "bold") if dec.startswith("correction:") else ("red", "bold") if "LOST" in dec else \
            ("green", "bold") if ("advance" in dec or "complete" in dec or "final" in dec) else ("grey",)
        txt = dec + (f" -> target {d['target_world']}" if d.get("target_world") else "")
        parts.append(paint("=> " + txt, *col))
    print(head + " " + " | ".join(parts), flush=True)


# ------------------------------------------------------------------ timing ---------------------------

class Timing(object):
    def __init__(self):
        self.samples = {"pi0": [], "vlm_plan": [], "vlm_judge": [], "step": [], "ui": [], "pacing": []}
        self.t0 = time.time()

    def add(self, key, dt):
        self.samples[key].append(dt)

    def total(self, key):
        return float(sum(self.samples[key]))

    @staticmethod
    def stats(xs):
        if not xs:
            return {"n": 0}
        a = np.asarray(xs) * 1000
        return {"n": len(xs), "mean_ms": round(float(a.mean()), 1), "p50_ms": round(float(np.median(a)), 1),
                "p95_ms": round(float(np.percentile(a, 95)), 1), "max_ms": round(float(a.max()), 1),
                "total_s": round(float(a.sum()) / 1000, 2)}

    def report(self, sim_steps):
        wall = time.time() - self.t0
        overhead = self.total("ui") + self.total("pacing")
        compute = wall - overhead           # what the pipeline itself needed (no drawing, no real-time sleep)
        sim_time = sim_steps * DT
        parts = {k: self.total(k) for k in ("pi0", "vlm_plan", "vlm_judge", "step")}
        return {
            "sim_steps": sim_steps, "sim_time_s": round(sim_time, 2),
            "wall_time_s": round(wall, 2), "pipeline_time_s": round(compute, 2),
            "ui_and_pacing_s": round(overhead, 2),
            "real_time_factor": round(sim_time / compute, 3) if compute > 0 else None,
            "share_of_pipeline": {k: round(v / compute, 3) if compute > 0 else None for k, v in parts.items()},
            "per_call": {k: self.stats(v) for k, v in self.samples.items() if k not in ("ui", "pacing")},
        }


# ------------------------------------------------------------------ window ---------------------------

class LiveWindow(object):
    def __init__(self, title, speed, show=True, model_label="", panel_h=PANEL_H):
        import cv2

        self.cv2, self.title, self.speed, self.show, self.model_label = cv2, title, speed, show, model_label
        self.paused, self.frame, self.hud, self.last, self.panel_h = False, None, {}, None, panel_h
        if show:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(WIN, 2 * VIEW, VIEW + panel_h)

    def set_obs(self, obs):
        cv2 = self.cv2
        views = []
        for k in ("agentview_image", "robot0_eye_in_hand_image"):
            img = cv2.resize(common.upright(obs[k]), (VIEW, VIEW), interpolation=cv2.INTER_LINEAR)
            views.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame = np.concatenate(views, axis=1)

    def draw(self, busy=None):
        """Render the frame (kept in self.last for the video); show it if there is a window."""
        cv2 = self.cv2
        if self.frame is None:
            return None
        W = 2 * VIEW
        img = np.full((VIEW + self.panel_h, W, 3), 22, np.uint8)
        img[:VIEW] = self.frame
        h = self.hud

        def put(text, x, y, col=WHITE, s=0.45, th=1):
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)

        put("front camera", 8, 18)
        put("wrist camera", VIEW + 8, 18)
        for box, col, lab in h.get("boxes", []):  # VLM detections (normalised) on the front view
            if box:
                p0 = (int(box["x_min"] * VIEW), int(box["y_min"] * VIEW))
                cv2.rectangle(img, p0, (int(box["x_max"] * VIEW), int(box["y_max"] * VIEW)), col, 2)
                put(lab, p0[0], max(p0[1] - 5, 12), col, 0.45, 1)
        for pt, col, lab in h.get("points", []):  # measured world points (target, destination, EE), projected
            if pt is not None:
                q = (int(pt[0] * VIEW), int(pt[1] * VIEW))
                cv2.circle(img, q, 5, col, -1, cv2.LINE_AA)
                cv2.circle(img, q, 6, (0, 0, 0), 1, cv2.LINE_AA)
                put(lab, q[0] + 8, q[1] + 4, col, 0.4, 1)
        corr = h.get("point")
        if corr is not None:
            u, v = int(corr[0] * VIEW / common.ENV_RESOLUTION), int(corr[1] * VIEW / common.ENV_RESOLUTION)
            cv2.drawMarker(img, (u, v), YELLOW, cv2.MARKER_CROSS, 24, 2)
            cv2.circle(img, (u, v), 16, YELLOW, 2)
        calls = h.get("calls", [])
        if calls:  # latest VLM calls, newest first, bottom-right of the wrist view
            lines = [f"{c['kind']:5s} @{c['step']:4d}  {c['dt'] * 1000:5.0f} ms" +
                     (f" (GPU {c['gpu'] * 1000:4.0f})" if c.get("gpu") is not None else "") +
                     (f"  {c['answer']}" if c.get("answer") else "") for c in calls[-6:][::-1]]
            x0, y0 = VIEW + 8, VIEW - 30 - 17 * len(lines)
            cv2.rectangle(img, (x0 - 4, y0 - 16), (W - 6, VIEW - 22), (0, 0, 0), -1)
            put("VLM inference time per call", x0, y0, CYAN, 0.45, 1)
            for j, line in enumerate(lines):
                put(line, x0, y0 + 17 * (j + 1), WHITE if j else YELLOW, 0.45, 1)
        if busy:
            put(busy, 8, VIEW - 14, CYAN, 0.7, 2)
        if self.paused:
            put("PAUSED (space)", W - 190, VIEW - 14, ORANGE, 0.7, 2)

        y = VIEW + 22
        if self.model_label:
            put(self.model_label, 8, y, YELLOW, 0.6, 2)
            y += 22
        put(self.title[:110], 8, y, WHITE, 0.5, 1)
        y += 22
        put(f"step {h.get('step', 0):4d}   sim t = {h.get('step', 0) * DT:6.2f} s   wall = {h.get('wall', 0):6.1f} s"
            f"   real-time factor (pipeline) = {h.get('rtf', 0):.2f}   pacing: "
            f"{'%.1fx' % self.speed if self.speed > 0 else 'off'}", 8, y, GREY)
        y += 20
        put(f"pi0  last {h.get('pi0_last', 0) * 1000:6.0f} ms  mean {h.get('pi0_mean', 0) * 1000:6.0f} ms  "
            f"(n={h.get('pi0_n', 0)})    VLM  last {h.get('vlm_last', 0) * 1000:6.0f} ms  mean "
            f"{h.get('vlm_mean', 0) * 1000:6.0f} ms  (n={h.get('vlm_n', 0)})    step+render {h.get('step_mean', 0) * 1000:4.1f} ms",
            8, y, CYAN)
        y += 22
        put(f"pi0 prompt: \"{h.get('prompt', '')}\""[:120], 8, y, GREY)
        y += 20
        for k, (lab, done, cur, hist) in enumerate(h.get("subtasks", [])):
            col = GREEN if done is not None else (ORANGE if hist else (YELLOW if cur else WHITE))
            lost = "   ".join(f"SUCCESS @{e['done_step']} LOST @{e['lost_step']}" for e in hist[-2:])
            put(("> " if cur else "  ") + lab + (f"   {lost}" if lost else "") +
                (f"   SUCCESS @{done}" if done is not None else ""), 8, y, col)
            y += 18
        gt = h.get("gt")
        if gt:  # latched first success, plus whether it still holds now (pick: held, place: on/in)
            put("ground truth (sim, display only): " + "   ".join(
                f"{spec} " + (f"@{st}" if st is not None else "-") +
                ("" if st is None or now is None else (" now " + ("held" if now else "DROPPED") if spec.startswith("pick")
                                                       else (" now ok" if now else " now NOT")))
                for spec, st, now in gt), 8, y, GREY, 0.42)
            y += 18
        for text, col in h.get("extra", []):
            put(text[:125], 8, y, col, 0.45)
            y += 18
        dec = h.get("decision")
        if dec:
            col = {"none": GREY, "low_level": ORANGE, "high_level": YELLOW}.get(dec.get("failure"), GREY)
            tag = "" if dec.get("failure") in (None, "none") else f" [{dec['failure']}: {dec.get('failure_mode', '')}]"
            text = f"VLM @{dec['step']}{tag}" + (f" ({dec['decision']})" if dec.get("decision") else "") + \
                   ": " + (dec.get("observation") or dec.get("error") or dec.get("note") or "") + \
                   (f" | {dec['reason']}" if dec.get("reason") and dec.get("failure") not in (None, "none") else "")
            for j, line in enumerate(textwrap.wrap(text, 125)[:3]):
                put(line, 8, y + 18 * j, col if j == 0 else WHITE)
        self.last = img
        if self.show:
            cv2.imshow(WIN, img)
            self.keys()
        return img

    def keys(self, wait_ms=1):
        if not self.show:
            if wait_ms > 1:
                time.sleep(wait_ms / 1000)
            return
        k = self.cv2.waitKey(wait_ms) & 0xFF
        if k in (ord("q"), 27):
            raise UserQuit()
        if k == ord(" "):
            self.paused = not self.paused
        if k == ord("f"):
            self.speed = 0.0 if self.speed > 0 else 1.0
        try:
            if self.cv2.getWindowProperty(WIN, self.cv2.WND_PROP_VISIBLE) < 1:
                raise UserQuit()
        except self.cv2.error:
            raise UserQuit()


class VideoRecorder(object):
    """mp4 of the rendered window at the control rate (20 fps); re-encoded to H.264 with ffmpeg at close."""

    def __init__(self, path, fps=common.CONTROL_FREQ):
        self.path, self.fps, self.writer, self.n = pathlib.Path(path), fps, None, 0
        self.tmp = self.path.with_suffix(".raw.mp4")

    def write(self, img, k=1):
        import cv2

        if img is None:
            return
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                          (img.shape[1], img.shape[0]))
        for _ in range(k):
            self.writer.write(img)
        self.n += k

    def close(self):
        if self.writer is None:
            return None
        self.writer.release()
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.tmp), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "23", str(self.path)])
        if r.returncode == 0:
            self.tmp.unlink()
        else:
            self.tmp.rename(self.path)
        return self.path


def gt_now(env, check):
    """Whether a privileged check's condition holds NOW (the checks latch at their first success); display only.
    pick: object grasped (lifting is not required while it is carried down to the destination)."""
    try:
        if check.kind == "pick":
            return bool(predicates.grasped(env, check.obj))
        return bool(check.holds(env))
    except Exception:
        return None


def recovery_hud(agent):
    """Two HUD lines for recovery_agent.GroundedRecoveryAgent: failure monitor, corrections."""
    m = agent.meas or {}
    sig, place = [], bool(agent.subtasks) and agent.current["kind"] == "place"
    if m.get("d_target") is not None and not place:
        sig.append(f"EE-target {m['d_target'] * 100:.0f} cm xy {m['dz_target'] * 100:+.0f} cm z "
                   f"(pre-grasp pt {m['a_target'] * 100:.0f} cm)")
    if m.get("d_dest") is not None and place:
        sig.append(f"EE-dest {m['d_dest'] * 100:.0f} cm xy {m['dz_dest'] * 100:+.0f} cm z "
                   f"(release pt {m['a_dest'] * 100:.0f} cm)")
    last_step = agent.decisions[-1]["step"] if agent.decisions else None
    if agent.pending is not None:
        st, col = f"suspect {agent.pending[0]}", ORANGE
    elif agent.failures and agent.failures[-1]["last_step"] == last_step:
        f = agent.failures[-1]
        st, col = f"{f['level'].upper()} {f['mode']} [{f['library']}]", YELLOW if f["level"] == "high_level" else ORANGE
    else:
        st, col = "no failure", GREEN
    if agent.watch_from is not None and last_step is not None and last_step < agent.watch_from:
        st += f" (grace until @{agent.watch_from})"
    corr = "   ".join(f"{c['action']}@{c['step']}" + (f"-{c['end_step']}" if c.get("end_step") else
                                                     (" RUNNING" if agent.motion_active else ""))
                      + f" [{c['failure_mode']}]" for c in agent.corrections[-3:])
    return [(f"failure monitor: {st}   |   " + "   ".join(sig), col),
            (f"corrections {len(agent.corrections)}/{agent.max_corrections}: " + (corr or "none") +
             f"   |   failures seen: {len(agent.failures)}", CYAN if agent.motion_active else WHITE)]


def blocking_with_ui(fn, ui, label, viewer=None, rec=None):
    """Run fn() in a thread while keeping the window responsive. Returns (result, seconds).

    rec: also write the waiting frames to this VideoRecorder at the wall-clock rate (sim frozen)."""
    out = {}

    def run():
        try:
            out["r"] = fn()
        except BaseException as e:  # re-raised in the caller
            out["e"] = e

    t0, written = time.time(), 0
    th = threading.Thread(target=run, daemon=True)
    th.start()
    while th.is_alive():
        th.join(0.01 if rec is not None else 0.03)
        if ui is not None:
            el = time.time() - t0
            img = ui.draw(busy=f"{label} ... {el:4.2f} s" + ("  (sim frozen)" if rec is not None else ""))
            if rec is not None and int(el * rec.fps) > written:
                rec.write(img, int(el * rec.fps) - written)
                written = int(el * rec.fps)
        if viewer is not None and viewer.is_running():
            viewer.sync()
    dt = time.time() - t0
    if "e" in out:
        raise out["e"]
    return out.get("r"), dt


# ------------------------------------------------------------------ main -----------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=int, default=10, help="libero_90 task id")
    p.add_argument("--init", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=common.MAX_STEPS[SUITE])
    p.add_argument("--speed", type=float, default=1.0,
                   help="pace the sim at this multiple of real time (20 Hz); 0 = as fast as possible")
    p.add_argument("--no-vlm", action="store_true", help="pi0 only (no VLM monitoring), for a latency baseline")
    p.add_argument("--mujoco-viewer", action="store_true", help="also open the interactive MuJoCo viewer")
    p.add_argument("--no-window", action="store_true", help="no camera window (timing only)")
    p.add_argument("--warmup", type=int, default=2, help="untimed pi0 calls before the episode (JAX compile)")
    p.add_argument("--vlm", choices=["moondream", "moondream2", "claude"], default="moondream",
                   help="moondream: Moondream cloud API (key in $MOONDREAM_API_KEY or "
                        "/home/DockerShared/.moondream_api_key); moondream2: Moondream2 on the local GPU "
                        "(model/moondream2); claude: Claude Code CLI")
    p.add_argument("--judge", choices=["full", "success", "grounded", "recovery"], default=None,
                   help="full: judge_agent.SubtaskJudgeAgent (JSON judgement + corrections); success: "
                        "SubtaskSuccessJudge (one yes/no question per call); grounded: GroundedSuccessJudge "
                        "(one detect call per call + proprioception; moondream2 only); recovery: "
                        "recovery_agent.GroundedRecoveryAgent (grounded judge + grounded failure detection and "
                        "high-level corrections; moondream2 only). Default: grounded for moondream2, full otherwise")
    p.add_argument("--judge-fps", type=float, default=None,
                   help="VLM judge calls per second of sim time (default 5 for the moondream2 grounded / recovery "
                        "judges, 1 otherwise; the sim runs at 20 Hz, so the judge runs every round(20 / fps) control "
                        "steps, max 20 fps)")
    p.add_argument("--print", dest="print_mode", choices=["calls", "events", "none"], default="calls",
                   help="terminal log of the judge: every call (calls), only calls where something happens "
                        "(events), or nothing")
    p.add_argument("--video", type=pathlib.Path, default=None, help="write the rendered window to this mp4")
    p.add_argument("--run-past-success", action="store_true",
                   help="do not end the episode at LIBERO's own success; only the VLM judge (or max steps) ends it")
    p.add_argument("--no-gt", action="store_true", help="do not draw the privileged ground-truth subtask checks")
    p.add_argument("--model", default=None,
                   help=f"VLM model (default {V.MOONDREAM_DEFAULT} / {V.DEFAULT_MODEL})")
    p.add_argument("--effort", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", type=pathlib.Path, default=None, help="write the timing report (json) here")
    a = p.parse_args()
    a.judge = a.judge or ("grounded" if a.vlm == "moondream2" else "full")
    if a.judge_fps is None:
        a.judge_fps = 5.0 if a.judge in ("grounded", "recovery") else 1.0
    every_steps = max(1, int(round(common.CONTROL_FREQ / a.judge_fps)))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("live")

    from openpi_client import websocket_client_policy

    np.random.seed(7)
    env, task, _ = common.make_env(SUITE, a.task)
    env.reset()
    init = common.init_state(SUITE, a.task, a.init)
    obs = env.set_init_state(init)
    title = f"libero_90 task {a.task} init {a.init}: \"{task.language}\"" + ("   [pi0 only]" if a.no_vlm else f"   [VLM: {a.vlm}]")

    timing = Timing()
    vlm = None if a.no_vlm else V.make_vlm(a.vlm, a.model, a.effort, log=log)
    model_label = "pi0 only (no VLM)" if vlm is None else f"VLM judge: {vlm.model}" + (
        f" (local, {vlm.info.get('device', '').replace('NVIDIA GeForce ', '')})" if hasattr(vlm, "info") else "")
    rec = None if a.video is None else VideoRecorder(a.video)
    ui = None if a.no_window and rec is None else LiveWindow(
        title, a.speed, show=not a.no_window, model_label=model_label,
        panel_h=PANEL_H_RECOVERY if a.judge == "recovery" and not a.no_vlm else PANEL_H)
    checks = [] if a.no_gt else predicates.goal_checks(env) + [predicates.GoalCheck()]
    viewer = None
    if a.mujoco_viewer:
        import mujoco.viewer

        sim = env.env.sim
        viewer = mujoco.viewer.launch_passive(sim.model._model, sim.data._data)

    # pi0 client, timed
    client = websocket_client_policy.WebsocketClientPolicy(a.host, a.port)
    raw_infer = client.infer
    # the server's first requests include JAX compilation (~25 s); keep them out of the measurement
    for k in range(a.warmup):
        _, dt = blocking_with_ui(lambda: raw_infer(common.pi0_request(obs, str(task.language))), ui,
                                 f"pi0 warm-up {k + 1}/{a.warmup}", viewer)
        log.info("pi0 warm-up call %d: %.2f s (not counted)", k + 1, dt)

    def infer(req):
        r, dt = blocking_with_ui(lambda: raw_infer(req), ui, "pi0 inference", viewer)
        timing.add("pi0", dt)
        return r

    client.infer = infer

    # VLM agent, timed
    agent, calls = None, []
    state = {"last": None, "steps": 0, "prompt": str(task.language)}
    if vlm is not None:
        raw_ask = vlm.ask

        def ask(parts, schema):
            kind = "vlm_plan" if schema is judge_agent.PLAN_SCHEMA else "vlm_judge"
            r, dt = blocking_with_ui(lambda: raw_ask(parts, schema), ui,
                                     f"{vlm.model} planning" if kind == "vlm_plan" else f"{vlm.model} judging",
                                     viewer, rec)
            timing.add(kind, dt)
            ans = r.get("answer") if schema is None else r.get("subtask_success", "plan")
            calls.append({"step": state["steps"], "kind": "plan" if kind == "vlm_plan" else "judge", "dt": dt,
                          "gpu": getattr(vlm, "last_server", {}).get("latency_s"),
                          "answer": str(ans).strip().split("\n")[0][:24] if ans is not None else ""})
            return r

        vlm.ask = ask
        def timed_skill(skill, raw, unit):
            def call(img, phrase):
                kind = "vlm_plan" if getattr(agent, "phase", "judge") == "plan" else "vlm_judge"
                r, dt = blocking_with_ui(lambda: raw(img, phrase), ui, f"{vlm.model} {skill}('{phrase}')", viewer, rec)
                timing.add(kind, dt)
                calls.append({"step": state["steps"], "kind": "plan" if kind == "vlm_plan" else "judge", "dt": dt,
                              "gpu": vlm.last_server.get("latency_s"), "answer": f"{skill}: {len(r)} {unit}"})
                return r
            return call

        if hasattr(vlm, "detect"):
            vlm.detect = timed_skill("detect", vlm.detect, "box")
        if hasattr(vlm, "point"):
            vlm.point = timed_skill("point", vlm.point, "pt")
        if a.judge == "recovery":
            agent = recovery_agent.GroundedRecoveryAgent(vlm, recovery_agent.GroundedCamera(env), log=log)
        elif a.judge == "grounded":
            agent = judge_agent.GroundedSuccessJudge(vlm, log=log)
        elif a.judge == "success":
            agent = judge_agent.SubtaskSuccessJudge(vlm, log=log)
        else:
            agent = judge_agent.SubtaskJudgeAgent(vlm, lambda u, v: camera.unproject(env, u, v, common.ENV_RESOLUTION),
                                                  log=log)
        agent.every_steps = every_steps  # loop.run_episode calls the agent every this many control steps
        log.info("VLM judge every %d control steps (%.2f fps of sim time)", every_steps,
                 common.CONTROL_FREQ / every_steps)

        # real-time terminal log of every judge call; ground-truth snapshots around each correction
        raw_act = agent.act

        def act(view, priv=None):
            was_moving, n_dec, n_corr = getattr(agent, "motion_active", False), len(agent.decisions), \
                len(agent.corrections)
            out = raw_act(view, priv)
            gt = {c.spec: gt_now(env, c) for c in checks}
            if was_moving and agent.corrections and agent.corrections[-1].get("end_step") == view.step:
                c = agent.corrections[-1]
                c["gt_end"] = gt
                if a.print_mode != "none":
                    print(paint(f"[step {view.step:4d}] correction {c['action']} (started @{c['step']}) finished "
                                f"after {view.step - c['step']} steps; ground truth now: {gt}", "cyan"), flush=True)
            if len(agent.corrections) > n_corr:
                agent.corrections[-1]["gt_start"] = gt
            for d in agent.decisions[n_dec:]:
                d["gt_now"] = gt  # evaluation only: added after the agent decided
                if a.print_mode != "none":
                    print_decision(d, every_call=a.print_mode == "calls")
            return out

        agent.act = act


    gt_seen = {}

    def hook(step, o):
        now = time.time()
        if a.print_mode != "none":
            for c in checks:  # privileged predicates latch: print when one first holds (display only)
                if c.done_step is not None and gt_seen.get(c.spec) is None:
                    print(paint(f"[step {step:4d}] GROUND TRUTH (sim, not seen by the agent): {c.spec} holds "
                                f"from step {c.done_step}", "bold"), flush=True)
                gt_seen[c.spec] = c.done_step
        if state["last"] is not None:
            # time since the previous hook, minus pi0 / VLM calls made in between = env.step + render + bookkeeping
            between = now - state["last"] - (timing.total("pi0") + timing.total("vlm_plan") + timing.total("vlm_judge")
                                             - state["model_time"])
            timing.add("step", max(between, 0.0))
        state["steps"] = step
        if viewer is not None and viewer.is_running():
            viewer.sync()
        if ui is not None:
            t_ui = time.time()
            ui.set_obs(o)
            h = ui.hud
            pi0s, vlms = timing.samples["pi0"], timing.samples["vlm_plan"] + timing.samples["vlm_judge"]
            rep_wall = now - timing.t0 - timing.total("ui") - timing.total("pacing")
            h.update({"step": step, "wall": now - timing.t0, "rtf": step * DT / rep_wall if rep_wall > 0 else 0,
                      "pi0_last": pi0s[-1] if pi0s else 0, "pi0_mean": float(np.mean(pi0s)) if pi0s else 0,
                      "pi0_n": len(pi0s), "vlm_last": vlms[-1] if vlms else 0,
                      "vlm_mean": float(np.mean(vlms)) if vlms else 0, "vlm_n": len(vlms),
                      "step_mean": float(np.mean(timing.samples["step"])) if timing.samples["step"] else 0,
                      "calls": calls, "gt": [(c.spec, c.done_step, gt_now(env, c)) for c in checks]})
            if agent is not None:
                h["subtasks"] = [(s["label"], s["done_step"], k == agent.idx, s.get("history", []))
                                 for k, s in enumerate(agent.subtasks)]
                decs = [d for d in agent.decisions if d.get("type") != "plan"]
                h["decision"] = decs[-1] if decs else None
                last_corr = agent.corrections[-1] if agent.corrections else None
                h["point"] = last_corr.get("point_img") if last_corr and agent.motion_active else None
                bx = getattr(agent, "boxes", None) or {}
                h["boxes"] = [(bx.get("object"), YELLOW, getattr(agent, "obj_phrase", "")),
                              (bx.get("destination"), GREEN, getattr(agent, "dest_phrase", ""))]
                if hasattr(agent, "geo"):
                    g = agent.geo
                    h["points"] = [(g.get("target_px"), YELLOW, "target"), (g.get("dest_px"), GREEN, "dest"),
                                   (g.get("ee_px"), CYAN, "EE")]
                    h["extra"] = recovery_hud(agent)
                prompts = [c["new_prompt"] for c in agent.corrections if c.get("new_prompt")]
                h["prompt"] = prompts[-1] if prompts else state["prompt"]
            else:
                h["prompt"] = state["prompt"]
            ui.draw()
            if rec is not None:
                rec.write(ui.last)
            while ui.paused:
                ui.draw()
                time.sleep(0.03)
            timing.add("ui", time.time() - t_ui)
        # real-time pacing: never let a step take less than DT / speed of wall time
        speed = ui.speed if ui is not None else a.speed
        if speed > 0 and state["last"] is not None:
            t_p = time.time()
            target = state["last"] + DT / speed
            while time.time() < target:
                if ui is not None:
                    ui.keys(1)
                else:
                    time.sleep(0.002)
            timing.add("pacing", time.time() - t_p)
        state["model_time"] = timing.total("pi0") + timing.total("vlm_plan") + timing.total("vlm_judge")
        state["last"] = time.time()

    state["model_time"] = 0.0
    res, quit_early = None, False
    try:
        res = loop.run_episode(env, obs, str(task.language), a.max_steps, client=client, agent=agent,
                               checks=checks, init_state=init, stop_on_success=not a.run_past_success, step_hook=hook, log=log, record=False)
    except UserQuit:
        quit_early = True
        log.info("quit by user at step %d", state["steps"])
    finally:
        rep = timing.report(state["steps"])
        rep.update({"task_id": a.task, "init_state_idx": a.init, "task_language": str(task.language),
                    "vlm": None if a.no_vlm else {"backend": a.vlm, "model": vlm.model, "judge": a.judge,
                                                 "every_steps": every_steps,
                                                 "server": getattr(vlm, "info", None),
                                                 "cost_usd": round(vlm.cost_usd, 3) if vlm else None,
                                                 "answers_repaired": getattr(vlm, "repairs", None),
                                                 "answers_unparseable": getattr(vlm, "parse_failures", None)},
                    "vlm_decisions": [] if agent is None else [
                        {k: v for k, v in d.items() if k != "after"} for d in agent.decisions],
                    "quit_early": quit_early,
                    "success": None if res is None else res.success,
                    "stop_reason": None if res is None else res.stop_reason,
                    "agent_calls": None if res is None else res.agent_calls,
                    "corrections": [] if agent is None else agent.corrections,
                    "failures": getattr(agent, "failures", []),
                    "events": None if res is None else res.events,
                    "subtasks": [] if agent is None else agent.subtasks,
                    "vlm_calls": calls, "ground_truth": [c.status() for c in checks]})
        print(json.dumps({k: v for k, v in rep.items() if k not in ("corrections", "vlm_decisions")}, indent=1, default=loop._json_default))
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(rep, indent=1, default=loop._json_default))
            log.info("timing report -> %s", a.out)
        if ui is not None:
            ui.hud["decision"] = {"step": state["steps"], "failure": "none",
                                  "observation": f"episode over: success={rep['success']} "
                                                 f"stop={rep['stop_reason']}." + ("" if a.no_window else
                                                                                  " Press q to close.")}
        if rec is not None:
            ui.hud["calls"] = calls
            rec.write(ui.draw(), 2 * common.CONTROL_FREQ)  # hold the last frame 2 s
            log.info("video (%d frames) -> %s", rec.n, rec.close())
        if ui is not None and ui.show and not quit_early:
            try:
                while True:
                    ui.draw()
                    time.sleep(0.03)
            except UserQuit:
                pass
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("QT_X11_NO_MITSHM", "1")
    main()
