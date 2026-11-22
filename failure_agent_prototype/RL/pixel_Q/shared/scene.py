"""LIBERO plumbing shared by the pixel-Q correctors (V1 corrector_env.py, V2 libero_env.py).

Moved here unchanged from V1's corrector_env.py:
  start states    library_starts (rl/pick reset states of library.json), task_starts (task init states),
                  the pixel-Q training split of LIBERO-90 and the held-out guard (SystemExit)
  geometry        segmentation render, object geoms, upright pixel <-> world, ray / plane cut, yaw rotation,
                  top-down orientation, approach tilt
  pi0 client      websocket client that waits silently for a busy / cold server
  LiberoScene     base class: env construction (one env per task, robosuite horizon raised), target
                  lookup, object tilt

Pixels named (u, v) are in the UPRIGHT image (common.upright, what pi0 sees): u = column, v = row.
"""

import json
import pathlib
import re
import sys

import numpy as np

FAP = pathlib.Path(__file__).resolve().parents[3]       # failure_agent_prototype/ (shared -> pixel_Q -> RL -> it)
if str(FAP) not in sys.path:
    sys.path.insert(0, str(FAP))
from fa_core import camera, common, predicates as P  # noqa: E402

ROUND_RIM = ("bowl", "mug", "cup")
# pixel_Q_RL.md appendix: LIBERO-90 training ("survey") split. The held-out split is never loaded for training.
TRAIN_TASKS_90 = (9, 10, 14, 17, 19, 25, 31, 34, 37, 38, 41, 42, 49, 53, 57, 58, 59, 61, 62, 63, 64, 67, 69, 71,
                  72, 73, 77, 78, 80, 85, 86, 87, 88, 89)
# the bowl / mug pick-and-place tasks among them (from the recorded task language)
DEFAULT_TASKS = ("libero_90:9,10,14,17,25,31,34,37,63,64,67,71,72,85", "libero_spatial:0-9")

MJ_OBJ_GEOM = 5                 # mjtObj.mjOBJ_GEOM in MuJoCo's segmentation render
# Irrecoverable = the object left the table or the robot's reach. A tilted or knocked-over object is NOT
# terminal: it is one of the states the corrector must handle.
DROP_M = 0.15                   # object fell this far below its start height (off the table)
WORKSPACE_M = 0.30              # object pushed this far (xy) from its start
WRIST_CAMERA = "robot0_eye_in_hand"
HORIZON = 100000                # robosuite steps per env.reset(); rollouts end on their own budgets, not this


class IneligibleStart(Exception):
    """The start state cannot be used (not a round-rim target, held already, object not visible)."""


class ResetFailed(Exception):
    """This reset did not reach a usable state. Stochastic: skip the episode, but keep the start state."""


# ------------------------------------------------------------------ start states -------------------

def parse_ids(text):
    """'9,10,14' / '0-9' / '1,3-5' -> list of ints."""
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


_parse_ids = parse_ids     # V1 name


def library_starts(policy="rl/pick"):
    """rl/pick reset states from library.json (pre-failure states pi0 actually reached)."""
    sr = common.tools_module("scene_restore")
    held_out = common.held_out_tasks()
    lib = json.loads(common.LIBRARY.read_text())
    words = {}
    for e in lib["entries"]:
        for inst in e.get("instances", []):
            m = re.match(r"pick\((\w+)\)", inst.get("subtask", ""))
            if m and "rl_reset_step" in inst:
                words[(inst["episode"], inst["rl_reset_step"])] = m.group(1)
    starts, skipped = [], []
    for rs in sr.load_reset_states(policy):
        suite, folder = rs["episode"].split("/")
        tid = int(re.match(r"task(\d+)", folder).group(1))
        tag = "{}@{}".format(rs["episode"].replace("/", "_"), rs["step"])
        if suite == "libero_90" and tid in held_out:
            raise SystemExit("{}: libero_90 task {} is in the held-out split".format(tag, tid))
        if suite == "libero_90" and tid not in TRAIN_TASKS_90:
            skipped.append((tag, "libero_90 task {} is not in the pixel-Q training split".format(tid)))
            continue
        starts.append({"kind": "reset_state", "tag": tag, "episode": rs["episode"], "step": rs["step"],
                       "word": words.get((rs["episode"], rs["step"]))})
    return starts, skipped


def task_starts(task_specs, init_states, allow_held_out=False):
    """Task init states, e.g. task_specs=["libero_90:9,10", "libero_spatial:0-9"], init_states=[0, 1, 2].

    allow_held_out: accept held-out libero_90 tasks (eval / baseline / inspect only, never training)."""
    held_out = set() if allow_held_out else common.held_out_tasks()
    starts = []
    for spec in task_specs:
        suite, ids = spec.split(":")
        for tid in parse_ids(ids):
            if suite == "libero_90" and tid in held_out:
                raise SystemExit("libero_90 task {} is in the held-out split".format(tid))
            for k in init_states:
                starts.append({"kind": "task_init", "tag": "{}_task{}_init{}".format(suite, tid, k), "suite": suite,
                               "task_id": tid, "init_idx": int(k)})
    return starts


def is_round_rim(name):
    return any(w in name.lower() for w in ROUND_RIM)


# ------------------------------------------------------------------ geometry -----------------------

def segmentation_geom_ids(env, size, cam=camera.CAMERA):
    """Per-pixel MuJoCo geom id (-1 = not a geom) of the raw (not upright) render, like depth_image."""
    sim = env.env.sim
    try:
        seg = sim.render(width=size, height=size, camera_name=cam, segmentation=True)
    except TypeError as e:
        raise RuntimeError("MjSim.render has no segmentation option in this robosuite ({}); the corrector needs "
                           "a mask source".format(e))
    seg = np.asarray(seg)
    if seg.ndim == 3 and seg.shape[-1] == 2:      # robosuite: (objtype, objid) per pixel
        return np.where(seg[..., 0] == MJ_OBJ_GEOM, seg[..., 1], -1)
    raise RuntimeError("unexpected segmentation render shape {}".format(seg.shape))


def object_geom_ids(env, name):
    m = env.env.sim.model
    obj = env.env.objects_dict[name]
    ids = set()
    for g in list(obj.contact_geoms) + list(obj.visual_geoms):
        try:
            ids.add(m.geom_name2id(g))
        except Exception:
            pass
    if not ids:
        raise IneligibleStart("no geoms found for {}".format(name))
    return np.array(sorted(ids))


def pixels_to_world(env, us, vs, depth, size, cam=camera.CAMERA):
    """Vectorised fa_core.camera.unproject: upright pixels (u, v) with metric depth -> world points (..., 3)."""
    pos, mat, fovy = camera._cam(env, cam)
    f = 0.5 * size / np.tan(np.deg2rad(fovy) / 2)
    c, r = size - 1 - np.asarray(us, float), size - 1 - np.asarray(vs, float)
    d = np.asarray(depth, float)
    pc = np.stack([(c + 0.5 - size / 2) / f * d, (r + 0.5 - size / 2) / f * d, -d], axis=-1)
    return pos + pc @ mat.T


def ray_at_height(env, us, vs, z, size, cam=camera.CAMERA):
    """World points where the camera rays through upright pixels (u, v) cross the plane world-z = z."""
    pos = camera._cam(env, cam)[0]
    ray = pixels_to_world(env, us, vs, np.ones(np.shape(us)), size, cam) - pos     # point at unit depth - camera
    return pos + ray * ((z - pos[2]) / ray[..., 2])[..., None]


def mask_box_centre_top(env, depth, mask, size, cam=camera.CAMERA):
    """World point above the centre of a mask's 2D box, and the masked surface's top height.

    depth, mask: UPRIGHT (size, size) metric depth and bool mask of one object. The box centre need not lie on
    the object (e.g. a bowl on its side), so its depth is not used: the camera ray through the box-centre pixel
    is cut at the object's median height. Top = 95th percentile height of the masked points. Only mask + depth
    + camera geometry are used (what a test-time agent has). Returns (centre (3,), top z)."""
    vs, us = np.nonzero(mask)
    pts = pixels_to_world(env, us, vs, depth[vs, us], size, cam)
    uc, vc = 0.5 * (us.min() + us.max()), 0.5 * (vs.min() + vs.max())
    centre = ray_at_height(env, uc, vc, float(np.median(pts[:, 2])), size, cam)
    return centre, float(np.percentile(pts[:, 2], 95))


def video_frame(obs):
    """Wrist view (first person) | agentview, both upright as pi0 gets them."""
    return np.concatenate([common.upright(obs["robot0_eye_in_hand_image"]), common.upright(obs["agentview_image"])],
                          axis=1)


def rotz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def top_down_quat(quat, T):
    """Same yaw as `quat`, approach axis straight down: EE z axis = world -z (robosuite Panda: the EE z
    axis points out between the fingers; check with --mode inspect, approach_tilt_deg)."""
    R = T.quat2mat(quat)
    x = np.array([R[0, 0], R[1, 0], 0.0])
    if np.linalg.norm(x) < 1e-6:                   # x axis vertical: fall back to the y axis for the yaw
        x = np.cross(np.array([R[0, 1], R[1, 1], 0.0]), [0.0, 0.0, -1.0])
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, -1.0])
    return T.mat2quat(np.stack([x, np.cross(z, x), z], axis=1))


def approach_tilt_deg(quat, T):
    """Angle between the EE approach axis (z) and straight down."""
    return float(np.degrees(np.arccos(np.clip(-T.quat2mat(quat)[2, 2], -1, 1))))


# ------------------------------------------------------------------ pi0 client ---------------------

def _patient_client_cls():
    from openpi_client import msgpack_numpy, websocket_client_policy
    import websockets.sync.client

    class PatientClient(websocket_client_policy.WebsocketClientPolicy):
        """WebsocketClientPolicy without the 10 s websocket open timeout. The openpi server runs
        policy.infer() inside its asyncio loop, so while one worker's request runs (the first one
        JIT-compiles for a minute or more) no other handshake is answered. The stock client then gives
        up after 10 s and drops the socket, which the server logs as an "opening handshake failed ...
        no close frame received or sent" traceback per attempt. Waiting instead is silent."""

        def _wait_for_server(self):
            import time

            while True:
                try:
                    conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None,
                                                          open_timeout=None)
                    return conn, msgpack_numpy.unpackb(conn.recv())
                except ConnectionRefusedError:
                    time.sleep(5)

    return PatientClient


def _PatientClient(host, port):
    return _patient_client_cls()(host, port)


patient_client = _PatientClient


def connect_pi0(host, port, worker_id=0):
    """pi0 websocket client. The client waits forever and silently on a refused connection, so say so once.
    The server's /healthz is probed over HTTP: a bare TCP connect+close makes the websockets server log an
    "opening handshake failed ... did not receive a valid HTTP request" traceback."""
    import urllib.request

    try:
        urllib.request.urlopen("http://{}:{}/healthz".format(host, port), timeout=2).close()
    except OSError as e:
        print("[worker {}] pi0 server not reachable at {}:{} ({}); waiting for it (start ./serve_pi0.sh)".format(
            worker_id, host, port, e), flush=True)
    return _PatientClient(host, port)


# ------------------------------------------------------------------ env base -----------------------

class LiberoScene(object):
    """One LIBERO env per (suite, task, seed, resolution), rebuilt only when the start's task changes.

    Subclasses set self.obs / self.obj / ... in their reset(). cfg keys used here: env_seed, seed, worker_id,
    host, port."""

    def __init__(self, cfg, client=None):
        self.cfg = cfg
        self.client = client
        self.env, self._env_key, self._trace_key, self._trace = None, None, None, None
        self.rng = np.random.default_rng(cfg.get("seed", 0) + cfg.get("worker_id", 0))
        from robosuite.utils import transform_utils as T

        self.T = T

    def _pi0(self):
        if self.client is None:
            self.client = connect_pi0(self.cfg["host"], self.cfg["port"], self.cfg.get("worker_id", 0))
        return self.client

    def _ensure_env(self, start):
        trace_dir = None
        if start["kind"] == "reset_state":
            sr = common.tools_module("scene_restore")
            meta, entry, trace_dir = sr.episode_info(start["episode"])
            key = (meta["task_suite"], entry["task_id"], meta["env_seed"], meta["camera_resolution"])
        else:
            key = (start["suite"], start["task_id"], self.cfg.get("env_seed", 7), common.ENV_RESOLUTION)
        if key != self._env_key:
            if self.env is not None:
                self.env.close()
            self.env, self.task, _ = common.make_env(key[0], key[1], key[2], key[3])
            # one env.reset() serves a whole chain / main rollout, which can pass robosuite's default horizon
            # of 1000 steps ("executing action in terminated episode")
            self.env.env.horizon = HORIZON
            self._env_key, self.res = key, key[3]
        if trace_dir is not None and self._trace_key != start["episode"]:
            sr = common.tools_module("scene_restore")
            self._trace, self._trace_key = sr.load_trace(trace_dir), start["episode"]

    def _resolve_target(self, start, ee):
        if start["kind"] == "task_init":
            picks = [c for c in P.goal_checks(self.env) if c.kind == "pick"]
            if not picks:
                raise IneligibleStart("task goal has no pick-and-place of a movable object")
            return picks[0].obj
        word = start.get("word")
        if not word:
            raise IneligibleStart("no pick(...) label in library.json for this reset state")
        names = P.object_names(self.env)
        toks = P._tokens(word)
        score = {n: len(toks & P._tokens(n)) for n in names}
        best = max(score.values()) if score else 0
        if best == 0:
            raise IneligibleStart("no sim object matches {!r}".format(word))
        cands = [n for n in names if score[n] == best]
        return min(cands, key=lambda n: np.linalg.norm(P.obj_pos(self.env, n) - ee))   # tie: the one pi0 went for

    def _obj_up(self):
        bid = P.body_id(self.env, self.obj)
        return np.array(self.env.env.sim.data.body_xmat[bid]).reshape(3, 3)[:, 2].copy()

    def obj_tilt_deg(self):
        """Object tilt from its pose at the start (upright): 0 standing, ~90 on its side."""
        return float(np.degrees(np.arccos(np.clip(self._obj_up() @ self.up0, -1, 1))))

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
