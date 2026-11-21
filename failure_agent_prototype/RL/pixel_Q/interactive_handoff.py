"""Interactive LIBERO: pi0 drives, you can take over the gripper from the keyboard and hand it back.

The "window" is a local web page served by this script (open http://localhost:PORT), so it works
headless and over ssh. The main view is the wrist (eye-in-hand) camera; agentview is inset.

Keys (in the browser tab):
  space              start / stop the interruption. Stopping hands control back to pi0, which replans
                     from the current observation (its queued action chunk is dropped).
  arrows             translate the gripper in the horizontal plane, relative to the wrist view
                     (up = towards the top of the wrist image)
  ctrl + left/right  rotate the gripper about the vertical axis (left = counter-clockwise seen from above)
  shift              hold for fine motion (x0.3)
  r                  restart the episode from the same init state;  n  next init state
While interrupted with no key held the arm holds still. The gripper keeps pi0's last open/close command.

Both views are shown as the camera actually sees them (not mirrored). pi0 still gets the usual
180-degree-rotated images, as in openpi's examples/libero/main.py.

Run in the LIBERO client venv while the pi0_libero server is up:
  shell 1:  ./serve_pi0.sh   (repo root; `serve_policy.py --env LIBERO` would serve pi0.5, not pi0)
  shell 2:  MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python interactive_handoff.py [--init 0]
  then open http://localhost:8765 (from another machine: ssh -L 8765:localhost:8765 <host>)
"""

import argparse
import json
import os
import pathlib
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
sys.path.insert(0, str(HERE.parents[1]))
import scene_restore as sr  # noqa: E402
from fa_core.common import REPLAN_STEPS, pi0_request  # noqa: E402

SUITE, MAX_STEPS, NUM_WAIT, FPS = "libero_90", 400, 10, 20
DUMMY = [0.0] * 6 + [-1.0]


# ---------------------------------------------------------------- shared state (sim <-> web) -------

class Shared:
    def __init__(self):
        self.lock = threading.Condition()
        self.frame_id = 0
        self.jpeg = {}  # view name -> latest jpeg bytes
        self.keys = {"up": False, "down": False, "left": False, "right": False, "ctrl": False, "shift": False}
        self.human = False
        self.commands = deque()  # "reset", "next"
        self.status = {}

    def publish(self, frames, status):
        import cv2

        enc = {k: cv2.imencode(".jpg", cv2.cvtColor(v, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
               for k, v in frames.items()}
        with self.lock:
            self.jpeg, self.status = enc, status
            self.frame_id += 1
            self.lock.notify_all()


S = Shared()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>LIBERO interactive</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:14px/1.4 monospace;display:flex;gap:16px;padding:16px}
 #main{position:relative} #wrist{width:640px;height:640px;display:block;border:3px solid #444}
 #agent{position:absolute;right:8px;bottom:8px;width:200px;height:200px;border:2px solid #888}
 #mode{font-size:22px;font-weight:bold;padding:6px 10px;margin-bottom:8px;display:inline-block}
 .pi0{background:#1d4ed8;color:#fff} .human{background:#dc2626;color:#fff} .done{background:#16a34a;color:#fff}
 kbd{background:#333;padding:1px 5px;border-radius:3px} td{padding:1px 8px 1px 0}
</style></head><body>
<div id="main"><img id="wrist" src="/stream/wrist"><img id="agent" src="/stream/agent"></div>
<div><div id="mode" class="pi0">PI0</div><table id="st"></table><p>
<kbd>space</kbd> interrupt / hand back<br><kbd>&larr;&uarr;&rarr;&darr;</kbd> translate (wrist-view frame)<br>
<kbd>ctrl</kbd>+<kbd>&larr;</kbd>/<kbd>&rarr;</kbd> rotate<br><kbd>shift</kbd> fine<br>
<kbd>r</kbd> restart &nbsp; <kbd>n</kbd> next init</p><p id="focus" style="color:#f59e0b"></p></div>
<script>
const K={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right'};
const keys={up:false,down:false,left:false,right:false,ctrl:false,shift:false};
function send(){fetch('/keys',{method:'POST',body:JSON.stringify(keys)});}
function cmd(c){fetch('/cmd',{method:'POST',body:c});}
document.addEventListener('keydown',e=>{
  if(e.key in K||e.key===' ')e.preventDefault();
  if(e.key===' '){if(!e.repeat)cmd('toggle');return;}
  if(e.key==='r'&&!e.repeat)cmd('reset'); if(e.key==='n'&&!e.repeat)cmd('next');
  let ch=false; if(e.key in K&&!keys[K[e.key]]){keys[K[e.key]]=true;ch=true;}
  if(keys.ctrl!==e.ctrlKey||keys.shift!==e.shiftKey){keys.ctrl=e.ctrlKey;keys.shift=e.shiftKey;ch=true;}
  if(ch)send();});
document.addEventListener('keyup',e=>{
  if(e.key in K){e.preventDefault();keys[K[e.key]]=false;}
  keys.ctrl=e.ctrlKey;keys.shift=e.shiftKey;send();});
window.addEventListener('blur',()=>{for(const k in keys)keys[k]=false;send();
  document.getElementById('focus').textContent='click the page to give it keyboard focus';});
window.addEventListener('focus',()=>document.getElementById('focus').textContent='');
setInterval(async()=>{try{const s=await (await fetch('/status')).json();
  const m=document.getElementById('mode');
  m.textContent=s.success?'SUCCESS':(s.human?'HUMAN':'PI0'); m.className=s.success?'done':(s.human?'human':'pi0');
  document.getElementById('wrist').style.borderColor=s.human?'#dc2626':'#444';
  document.getElementById('st').innerHTML=Object.entries(s).filter(([k])=>!['human','success'].includes(k))
   .map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');}catch(e){}},150);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/status":
            with S.lock:
                st = dict(S.status, human=S.human)
            return self._send(json.dumps(st).encode(), "application/json")
        if self.path.startswith("/stream/"):
            view = self.path.rsplit("/", 1)[1]
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while True:
                    with S.lock:
                        S.lock.wait_for(lambda: S.frame_id != last, timeout=1.0)
                        last, jpg = S.frame_id, S.jpeg.get(view)
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_error(404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        with S.lock:
            if self.path == "/keys":
                S.keys.update({k: bool(v) for k, v in json.loads(body).items() if k in S.keys})
            elif self.path == "/cmd":
                if body == "toggle":
                    S.human = not S.human
                    if not S.human:
                        S.keys.update(dict.fromkeys(S.keys, False))
                else:
                    S.commands.append(body)
        self._send(b"ok", "text/plain")


# ---------------------------------------------------------------- sim loop -------------------------

def wrist_axes(env, cam="robot0_eye_in_hand"):
    """World-frame horizontal unit vectors for 'image up' and 'image right' of the (unmirrored) wrist view.

    MuJoCo camera frame: +x = image right, +y = image up, looking along -z.
    """
    sim = env.env.sim
    R = sim.data.cam_xmat[sim.model.camera_name2id(cam)].reshape(3, 3)

    def flat(v):
        v = np.array([v[0], v[1], 0.0])
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v

    return flat(R[:, 1]), flat(R[:, 0])


def human_action(env, keys, grip, a):
    up, right = wrist_axes(env)
    scale = a.fine if keys["shift"] else 1.0
    act = np.zeros(7)
    if keys["ctrl"]:
        act[5] = a.rot_speed * scale * (keys["left"] - keys["right"])  # OSC rotation delta is in the world frame
        move = up * (keys["up"] - keys["down"])
    else:
        move = up * (keys["up"] - keys["down"]) + right * (keys["right"] - keys["left"])
    n = np.linalg.norm(move)
    if n > 0:
        act[:3] = a.trans_speed * scale * move / n
    act[6] = grip
    return act


def view(img):
    """Unmirrored camera image (LIBERO renders upside down, i.e. OpenGL row order)."""
    return np.ascontiguousarray(img[::-1])


def make_env(a):
    from libero.libero import benchmark

    env, task = sr.make_env(SUITE, a.task, a.seed, a.resolution)
    inits = benchmark.get_benchmark_dict()[SUITE]().get_task_init_states(a.task)
    return env, task, inits


def sim_loop(a):
    from openpi_client import websocket_client_policy

    np.random.seed(a.seed)
    client = websocket_client_policy.WebsocketClientPolicy(a.host, a.port)
    env, task, inits = make_env(a)
    prompt = str(task.language)
    init = a.init
    print(f"task {a.task}: {prompt!r}  ({len(inits)} init states)", flush=True)

    while True:  # one iteration per episode
        env.reset()
        obs = env.set_init_state(inits[init])
        with S.lock:
            S.human = False
            S.commands.clear()
        plan, grip, t, success, human_steps, takeovers, prev_human = deque(), -1.0, 0, False, 0, 0, False
        infer_ms = 0.0
        while True:
            tick = time.time()
            with S.lock:
                cmd = S.commands.popleft() if S.commands else None
                human, keys = S.human, dict(S.keys)
            if cmd in ("reset", "next"):
                init = (init + 1) % len(inits) if cmd == "next" else init
                break

            if success:
                action = None  # freeze on success; r / n to continue
            elif t < NUM_WAIT:
                action = DUMMY
            elif human:
                if not prev_human:
                    takeovers += 1
                plan.clear()  # pi0 replans from scratch when control comes back
                action = human_action(env, keys, grip, a)
                human_steps += 1
            else:
                if not plan:
                    t0 = time.time()
                    chunk = np.asarray(client.infer(pi0_request(obs, prompt))["actions"])
                    infer_ms = 1000 * (time.time() - t0)
                    plan.extend(chunk[:REPLAN_STEPS])
                action = plan.popleft()
                grip = float(np.sign(action[6])) if action[6] != 0 else grip
            prev_human = human

            if action is not None:
                try:
                    obs, _, done, _ = env.step([float(x) for x in action])
                except Exception as e:
                    print(f"env.step failed: {e}; restarting episode", flush=True)
                    break
                t += 1
                success = success or bool(done)
                if success:
                    print(f"SUCCESS at step {t} (human steps {human_steps}, takeovers {takeovers})", flush=True)

            ee = obs["robot0_eef_pos"]
            S.publish({"wrist": view(obs["robot0_eye_in_hand_image"]), "agent": view(obs["agentview_image"])},
                      {"success": success, "task": f"libero_90/{a.task}", "init": init, "step": f"{t}/{MAX_STEPS + NUM_WAIT}",
                       "gripper": "closed" if grip > 0 else "open", "ee_xyz_cm": " ".join(f"{100 * v:.1f}" for v in ee),
                       "human_steps": human_steps, "takeovers": takeovers, "pi0_infer_ms": f"{infer_ms:.0f}",
                       "prompt": prompt})
            time.sleep(max(0.0, 1.0 / a.fps - (time.time() - tick)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", type=int, default=2, help="LIBERO-90 task id")
    p.add_argument("--init", type=int, default=0, help="index into the task's init states")
    p.add_argument("--seed", type=int, default=7, help="env.seed() and np.random.seed(), as main.py")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--trans-speed", type=float, default=0.4, help="OSC action units (1 = max delta) while a key is held")
    p.add_argument("--rot-speed", type=float, default=0.3)
    p.add_argument("--fine", type=float, default=0.3, help="speed factor while shift is held")
    p.add_argument("--fps", type=float, default=FPS, help="control rate shown in real time")
    p.add_argument("--host", default="0.0.0.0", help="pi0 server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--web-bind", default="127.0.0.1")
    p.add_argument("--web-port", type=int, default=8765)
    a = p.parse_args()

    srv = ThreadingHTTPServer((a.web_bind, a.web_port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"open http://localhost:{a.web_port}", flush=True)
    sim_loop(a)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
