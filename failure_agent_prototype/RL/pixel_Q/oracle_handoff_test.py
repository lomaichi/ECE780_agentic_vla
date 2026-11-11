"""Go/no-go test for pixel-wise Q-learning on rl/pick: does pi0 succeed once the gripper is put at
the right spot on the object?

Pixel-wise Q-learning picks one point on the segmented object and runs a short scripted move there.
This test replaces the learned Q-map with brute force over privileged sim poses. For every rl/pick
reset state in library.json it lays a grid of candidate points over the target object's footprint
and measures, per point, how often each arm succeeds:

  baseline  plain pi0 from the reset state, no correction                         n_base trials
  handoff   scripted approach to <point> + dz with the gripper open, then pi0     n trials per cell
  scripted  scripted approach, descend, close, lift 10 cm; no pi0                 n trials per cell

Success = the rl/pick predicate: Grasped(obj) AND obj >= 5 cm above its reset height for 10
consecutive steps. The success rate per cell is a ground-truth (one-step, bandit) Q-map. The best
cell of each arm is then re-run with confirm_n trials, because the max over many noisy cells is
biased upward.

Verdict (printed at the end; thresholds are the GO_* constants below):
  GO (handoff)    on >= half the states the confirmed best handoff cell has success >= 0.5 and
                  >= baseline + 0.25. Pixel-Q -> short approach -> pi0 is viable, and the per-cell
                  map is a training target / sanity check for the learned Q-map.
  GO (primitive)  handoff fails but the scripted grasp works: the primitive has to finish the
                  grasp itself, and pi0 takes over only at the next subtask (check that separately).
  NO-GO           neither: even the best point on the object does not rescue these states.
  location_matters (per state) = best cell rate - median cell rate >= 0.25. If it rarely holds,
  a plain POINT_APPROACH to the object centre already does the job and pixel-Q adds nothing.

Privileged sim state (object poses, contacts) is used here only as an oracle. None of it is
available at test time.

Steps:
  0. Confirm the reset states restore and replay cleanly:
       python ../../tools/scene_restore.py --check --policy rl/pick
  1. Inspect (no pi0 needed): resolve each target object and draw the grid on its reset frame.
       python oracle_handoff_test.py --out-dir DIR --inspect
     Check DIR/inspect/*.png and DIR/inspect/states.json. Fix a wrong object with
       --object libero_90/task10_fail_init0@53=akita_black_bowl_1
  2. Smoke test, one state (pi0 server up):
       python oracle_handoff_test.py --out-dir DIR --states 0 --grid 3 --n 2 --n-base 4 --confirm-n 4
     Watch DIR/videos/*: the scripted approach should rise, travel and descend smoothly.
  3. Full run (14 states x 25 cells; roughly 1-1.5 h with 6 workers):
       python oracle_handoff_test.py --out-dir DIR

Run in the LIBERO client venv (MUJOCO_GL=egl) while the pi0_libero server is up:
  shell 1:  cd /opt/openpi && uv run scripts/serve_policy.py --env LIBERO
  shell 2:  /opt/openpi/examples/libero/.venv/bin/python oracle_handoff_test.py --out-dir DIR ...

Output (DIR):
  metadata.json               settings, per-state objects, per-cell rates, confirm results, verdict
  heatmaps/<state>_<arm>.png  upright reset frame (3x), each cell coloured red (0) -> green (1)
  videos/<state>_<arm>.mp4    trial 0 of the baseline and of each confirm run
The heatmap dots are a visualisation only (world -> agentview projection). The test itself runs in
world coordinates, so a mirrored overlay does not affect the numbers.
"""

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import re
import sys
from collections import deque

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import scene_restore as sr  # noqa: E402

POLICY = "rl/pick"
REPLAN_STEPS, RESIZE, FPS = 5, 224, 20  # same as record_suites.py / pass_at_k_task_test.py
LIFT_M, LIFT_HOLD = 0.05, 10  # rl/pick predicate
OPEN, CLOSE = -1.0, 1.0  # LIBERO gripper action
POS_GAIN = 10.0  # action units per metre of error (1 unit = 5 cm OSC delta)
ROT_GAIN = 1.0  # action units per radian of error (1 unit = 0.5 rad)
LEG_STEPS = 60  # max steps per scripted leg
SAFE_DZ = 0.12  # travel height above the object top
CLOSE_STEPS = 15
GO_RATE, GO_MARGIN, GO_FRAC, PEAK = 0.5, 0.25, 0.5, 0.25


# ---------------------------------------------------------------- pi0 -----------------------------

def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def pi0_request(obs, prompt):
    from openpi_client import image_tools

    img, wrist = sr.upright(obs["agentview_image"]), sr.upright(obs["robot0_eye_in_hand_image"])
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    return {
        "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE)),
        "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE)),
        "observation/state": state,
        "prompt": prompt,
    }


# ---------------------------------------------------------------- scene (privileged) --------------

def obj_pos(env, name):
    inner = env.env
    return inner.sim.data.body_xpos[inner.obj_body_id[name]].copy()


def grasped(env, name):
    inner = env.env
    return bool(inner._check_grasp(gripper=inner.robots[0].gripper, object_geoms=inner.objects_dict[name]))


def object_aabb(env, name):
    """World-frame axis-aligned bounding box of an object, from its geoms (mesh vertices or primitive sizes)."""
    sim = env.env.sim
    m, d = sim.model, sim.data
    obj = env.env.objects_dict[name]
    ids = set()
    for g in list(obj.contact_geoms) + list(obj.visual_geoms):
        try:
            ids.add(m.geom_name2id(g))
        except Exception:
            pass
    if not ids:
        raise RuntimeError(f"no geoms found for {name}")
    pts = []
    for g in ids:
        R, p, t, s = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], int(m.geom_type[g]), m.geom_size[g]
        if t == 7:  # mesh: MuJoCo stores the vertices in the geom frame
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = np.asarray(m.mesh_vert[a:a + n])
        else:  # 6 box, 2 sphere, 5 cylinder, 3 capsule; anything else: bounding sphere
            h = {6: s[:3], 2: np.full(3, s[0]), 5: np.array([s[0], s[0], s[1]]),
                 3: np.array([s[0], s[0], s[1] + s[0]])}.get(t, np.full(3, m.geom_rbound[g]))
            v = np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])])
        pts.append(v @ R.T + p)
    pts = np.concatenate(pts)
    return pts.min(0), pts.max(0)


def target_words():
    """(episode, reset step) -> object word from the pick(...) subtask of the library's LL instances."""
    lib = json.loads(sr.LIBRARY.read_text())
    out = {}
    for e in lib["entries"]:
        for inst in e.get("instances", []):
            m = re.match(r"pick\((\w+)\)", inst.get("subtask", ""))
            if m and "rl_reset_step" in inst:
                out[(inst["episode"], inst["rl_reset_step"])] = m.group(1)
    return out


def resolve_object(env, word, ee):
    """Sim object whose name shares the most tokens with `word`; ties go to the one nearest the EE
    (the object pi0 was going for)."""
    names = list(env.env.objects_dict)
    toks = [t for t in word.lower().split("_") if t]
    score = {n: sum(t in n.lower() for t in toks) for n in names}
    best = max(score.values(), default=0)
    if best == 0:
        raise KeyError(f"no sim object matches {word!r}; objects: {names}")
    cands = [n for n in names if score[n] == best]
    return min(cands, key=lambda n: np.linalg.norm(obj_pos(env, n) - ee))


def rotz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---------------------------------------------------------------- rollout -------------------------

class PickPredicate:
    def __init__(self, env, name):
        self.env, self.name = env, name
        self.z0 = obj_pos(env, name)[2]
        self.count, self.done = 0, False

    def update(self):
        lifted = obj_pos(self.env, self.name)[2] - self.z0 >= LIFT_M
        self.count = self.count + 1 if (lifted and grasped(self.env, self.name)) else 0
        self.done |= self.count >= LIFT_HOLD
        return self.done


class Episode:
    """Steps the env, updates the predicate, optionally records upright agentview frames."""

    def __init__(self, env, obs, pred, record):
        self.env, self.obs, self.pred = env, obs, pred
        self.frames = [sr.upright(obs["agentview_image"])] if record else None
        self.steps, self.sim_error = 0, False

    def step(self, action):
        """Returns True when the rollout should stop (success or sim error)."""
        try:
            self.obs, *_ = self.env.step([float(x) for x in action])
        except Exception as e:  # main.py also treats sim errors as failure
            print(f"env.step failed: {e}", flush=True)
            self.sim_error = True
            return True
        self.steps += 1
        if self.frames is not None:
            self.frames.append(sr.upright(self.obs["agentview_image"]))
        return self.pred.update()


def move_to(ep, pos, quat, grip, tol=0.005, max_steps=LEG_STEPS):
    """Closed-loop OSC move of the EE to (pos, quat). Returns True if the rollout should stop."""
    from robosuite.utils import transform_utils as T

    for _ in range(max_steps):
        err = pos - ep.obs["robot0_eef_pos"]
        # robosuite OSC applies the rotation delta in the world frame: R_goal = R_delta @ R_now
        rot = T.quat2axisangle(T.mat2quat(T.quat2mat(quat) @ T.quat2mat(ep.obs["robot0_eef_quat"]).T))
        if np.linalg.norm(err) < tol and np.linalg.norm(rot) < 0.05:
            return False
        a = np.concatenate([np.clip(POS_GAIN * err, -1, 1), np.clip(ROT_GAIN * rot, -1, 1), [grip]])
        if ep.step(a):
            return True
    return False


def hold(ep, grip, steps):
    for _ in range(steps):
        if ep.step(np.r_[np.zeros(6), grip]):
            return True
    return False


def approach(ep, pre, quat, travel_z):
    """Gripper open: rise to travel height, move above `pre` while rotating to `quat`, descend to `pre`."""
    ee, q_now = ep.obs["robot0_eef_pos"].copy(), ep.obs["robot0_eef_quat"].copy()
    return (move_to(ep, np.array([ee[0], ee[1], travel_z]), q_now, OPEN)
            or move_to(ep, np.array([pre[0], pre[1], travel_z]), quat, OPEN)
            or move_to(ep, pre, quat, OPEN))


def run_pi0(ep, client, prompt, steps):
    plan = deque()
    for _ in range(steps):
        if not plan:
            plan.extend(np.asarray(client.infer(pi0_request(ep.obs, prompt))["actions"])[:REPLAN_STEPS])
        if ep.step(plan.popleft()):
            return


def scripted_grasp(ep, grasp, quat):
    (move_to(ep, grasp, quat, OPEN)
     or hold(ep, CLOSE, CLOSE_STEPS)
     or move_to(ep, grasp + np.array([0, 0, 0.10]), quat, CLOSE)
     or hold(ep, CLOSE, LIFT_HOLD + 5))


_CACHE = {}  # per worker process: the env and trace of the last episode, and the pi0 client


def worker_env(episode):
    if _CACHE.get("episode") != episode:
        if "env" in _CACHE:
            _CACHE["env"].close()
        env, _ = sr.env_for_episode(episode)
        _, _, trace_dir = sr.episode_info(episode)
        _CACHE.update(episode=episode, env=env, trace=sr.load_trace(trace_dir))
    return _CACHE["env"], _CACHE["trace"]


def pi0_client(a):
    if "client" not in _CACHE:
        from openpi_client import websocket_client_policy

        _CACHE["client"] = websocket_client_policy.WebsocketClientPolicy(a["host"], a["port"])
    return _CACHE["client"]


def run_job(job):
    from robosuite.utils import transform_utils as T

    s, a, arm = job["state"], job["args"], job["arm"]
    env, trace = worker_env(s["episode"])
    client = pi0_client(a) if arm != "scripted" else None
    rng = np.random.default_rng(job["seed"])
    results = []
    for k in range(job["n"]):
        obs = sr.reset_to(env, trace, s["step"])
        ep = Episode(env, obs, PickPredicate(env, s["object"]), record=job["video"] and k == 0)
        if arm == "baseline":
            run_pi0(ep, client, s["prompt"], a["baseline_steps"] or 2 * s["horizon"])
        else:
            x, y = np.asarray(job["cell"]) + rng.uniform(-1, 1, 2) * a["jitter_cm"] / 100  # ~ one 3x3-pixel cell
            quat = T.mat2quat(rotz(job["yaw"]) @ T.quat2mat(obs["robot0_eef_quat"]))
            top = s["top_z"]
            if not approach(ep, np.array([x, y, top + a["dz"]]), quat, max(obs["robot0_eef_pos"][2], top + SAFE_DZ)):
                if arm == "handoff":
                    run_pi0(ep, client, s["prompt"], s["horizon"])
                else:
                    gz = max(top - a["grasp_depth"], s["bottom_z"] + 0.01)
                    scripted_grasp(ep, np.array([x, y, gz]), quat)
        results.append({"success": bool(ep.pred.done), "steps": ep.steps, "sim_error": ep.sim_error})
        if ep.frames is not None:
            import imageio

            name = f"{s['tag']}_{arm}" + ("" if arm == "baseline" else f"_cell{job['cell_idx']}_yaw{job['yaw']:g}")
            imageio.mimwrite(pathlib.Path(a["out_dir"]) / "videos" / f"{name}.mp4", ep.frames, fps=FPS)
    print(f"{s['tag']} {arm:8s} cell={job.get('cell_idx', '-')} yaw={job.get('yaw', '-')}: "
          f"{sum(r['success'] for r in results)}/{len(results)}", flush=True)
    return {"id": job["id"], "trials": results}


# ---------------------------------------------------------------- setup / report ------------------

def prepare_states(a):
    """Resolve each rl/pick reset state: target object, its AABB, the grid and the grid's agentview pixels."""
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

    words = target_words()
    overrides = dict(o.split("=", 1) for o in a["object"])
    held_out = set(json.loads(sr.LIBRARY.read_text())["split_guard"]["held_out_tasks"])
    states, images = [], {}
    for i, rs in enumerate(sr.load_reset_states(POLICY)):
        if a["states"] is not None and i not in a["states"]:
            continue
        key = f"{rs['episode']}@{rs['step']}"
        suite, folder = rs["episode"].split("/")
        tid = int(re.match(r"task(\d+)", folder).group(1))
        if suite == "libero_90" and tid in held_out:
            raise SystemExit(f"{key}: libero_90 task {tid} is in the held-out split")
        meta, _, trace_dir = sr.episode_info(rs["episode"])
        env, task = sr.env_for_episode(rs["episode"])
        obs = sr.reset_to(env, sr.load_trace(trace_dir), rs["step"])
        ee = obs["robot0_eef_pos"]
        word = words.get((rs["episode"], rs["step"]))
        if key in overrides:
            name = overrides[key]
        elif word is None:
            raise SystemExit(f"{key}: no pick(...) instance in library.json; pass --object {key}=<sim name> "
                             f"(objects: {list(env.env.objects_dict)})")
        else:
            name = resolve_object(env, word, ee)
        lo, hi = object_aabb(env, name)
        mg = a["margin_cm"] / 100
        xs = np.linspace(lo[0] - mg, hi[0] + mg, a["grid"])
        ys = np.linspace(lo[1] - mg, hi[1] + mg, a["grid"])
        cells = [(float(x), float(y)) for y in ys for x in xs]
        res = meta["camera_resolution"]
        M = get_camera_transform_matrix(env.env.sim, "agentview", res, res)
        pix = project_points_from_world_to_camera(np.array([[x, y, hi[2]] for x, y in cells]), M, res, res)
        pix = (res - 1) - np.asarray(pix)  # same 180 deg rotation as sr.upright
        s = {"idx": i, "tag": f"{rs['episode'].replace('/', '_')}@{rs['step']}", "episode": rs["episode"],
             "step": rs["step"], "onset_step": rs["onset_step"], "horizon": rs["horizon_steps"],
             "prompt": str(task.language), "word": word, "object": name, "objects_in_scene": list(env.env.objects_dict),
             "grasped_at_reset": grasped(env, name),
             "ee_to_obj_cm": round(float(np.linalg.norm(obj_pos(env, name) - ee)) * 100, 1),
             "aabb_lo": lo.tolist(), "aabb_hi": hi.tolist(), "top_z": float(hi[2]), "bottom_z": float(lo[2]),
             "cells": cells, "pixels": pix.tolist()}
        if s["grasped_at_reset"]:
            print(f"WARNING {key}: {name} is already grasped at the reset step; the open-gripper approach drops it")
        print(f"[{i}] {key}: {word} -> {name}, footprint {100 * (hi[0] - lo[0]):.1f} x {100 * (hi[1] - lo[1]):.1f} cm, "
              f"EE {s['ee_to_obj_cm']} cm away", flush=True)
        states.append(s)
        images[s["tag"]] = sr.upright(obs["agentview_image"])
        env.close()
    return states, images


def draw_cells(img, pixels, values, path, scale=3):
    from PIL import Image, ImageDraw

    im = Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for i, ((r, c), v) in enumerate(zip(pixels, values)):
        x, y = c * scale, r * scale
        fill = (255, 255, 255) if v is None else (int(255 * (1 - v)), int(255 * v), 0)
        d.rectangle([x - 5, y - 5, x + 5, y + 5], fill=fill, outline=(0, 0, 0))
        if v is None:
            d.text((x + 7, y - 6), str(i), fill=(255, 255, 0))
    im.save(path)


def rate(trials):
    return sum(t["success"] for t in trials) / len(trials) if trials else float("nan")


def run_jobs(jobs, a):
    with mp.get_context("spawn").Pool(a["workers"]) as pool:
        return {r["id"]: r["trials"] for r in pool.imap(run_job, jobs, chunksize=1)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--inspect", action="store_true", help="resolve objects and draw grids only; no pi0")
    p.add_argument("--states", type=int, nargs="+", default=None, help="indices into the rl/pick reset list")
    p.add_argument("--object", action="append", default=[], help="override: <episode>@<step>=<sim object name>")
    p.add_argument("--arms", nargs="+", default=["handoff", "scripted"], choices=["handoff", "scripted"])
    p.add_argument("--grid", type=int, default=5, help="grid x grid points over the object footprint")
    p.add_argument("--margin-cm", type=float, default=1.0, help="extend the footprint by this on each side")
    p.add_argument("--yaws", type=float, nargs="+", default=[0.0], help="gripper yaw offsets (deg) from the reset pose")
    p.add_argument("--dz", type=float, default=0.07, help="pre-grasp height above the object top (POINT_APPROACH default)")
    p.add_argument("--grasp-depth", type=float, default=0.02, help="scripted arm: close this far below the object top")
    p.add_argument("--jitter-cm", type=float, default=0.5, help="uniform xy jitter per trial (pixel-cell size)")
    p.add_argument("--n", type=int, default=4, help="trials per cell and arm")
    p.add_argument("--n-base", type=int, default=32, help="plain pi0 trials per state")
    p.add_argument("--confirm-n", type=int, default=16, help="trials to re-run each arm's best cell")
    p.add_argument("--baseline-steps", type=int, default=None, help="default 2 x horizon (approach + pi0 budget)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    a = vars(p.parse_args())
    out = pathlib.Path(a["out_dir"])
    for sub in ("inspect", "heatmaps", "videos"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    states, images = prepare_states(a)
    if a["inspect"]:
        for s in states:
            draw_cells(images[s["tag"]], s["pixels"], [None] * len(s["pixels"]), out / "inspect" / f"{s['tag']}.png")
        (out / "inspect" / "states.json").write_text(json.dumps(states, indent=2))
        print(f"wrote {len(states)} grids to {out / 'inspect'}")
        return

    # phase 1: baseline + every (cell, yaw) for each arm
    jobs = []
    for s in states:
        jobs.append({"state": s, "arm": "baseline", "n": a["n_base"], "video": True})
        for arm in a["arms"]:
            for ci, cell in enumerate(s["cells"]):
                for yaw in a["yaws"]:
                    jobs.append({"state": s, "arm": arm, "cell": cell, "cell_idx": ci, "yaw": yaw, "n": a["n"], "video": False})
    for i, j in enumerate(jobs):
        j.update(id=i, seed=a["seed"] + i, args=a)
    print(f"phase 1: {len(jobs)} jobs", flush=True)
    res1 = run_jobs(jobs, a)

    # phase 2: re-run each arm's best cell per state
    summary, confirm_jobs = {}, []
    for s in states:
        sj = [j for j in jobs if j["state"]["tag"] == s["tag"]]
        base = next(j for j in sj if j["arm"] == "baseline")
        summary[s["tag"]] = {"baseline": {"rate": rate(res1[base["id"]]), "n": base["n"]}}
        for arm in a["arms"]:
            cells = [{"cell_idx": j["cell_idx"], "yaw": j["yaw"], "rate": rate(res1[j["id"]]),
                      "mean_steps": float(np.mean([t["steps"] for t in res1[j["id"]]]))} for j in sj if j["arm"] == arm]
            best = max(cells, key=lambda c: c["rate"])
            summary[s["tag"]][arm] = {"cells": cells, "best": best,
                                      "median": float(np.median([c["rate"] for c in cells]))}
            confirm_jobs.append({"state": s, "arm": arm, "cell": s["cells"][best["cell_idx"]], "cell_idx": best["cell_idx"],
                                 "yaw": best["yaw"], "n": a["confirm_n"], "video": True})
    for i, j in enumerate(confirm_jobs):
        j.update(id=len(jobs) + i, seed=a["seed"] + len(jobs) + i, args=a)
    print(f"phase 2: {len(confirm_jobs)} confirm jobs", flush=True)
    res2 = run_jobs(confirm_jobs, a)
    for j in confirm_jobs:
        summary[j["state"]["tag"]][j["arm"]]["confirm_rate"] = rate(res2[j["id"]])

    # heatmaps + verdict
    rows = []
    for s in states:
        r = summary[s["tag"]]
        base = r["baseline"]["rate"]
        row = {"state": s["tag"], "object": s["object"], "baseline": base}
        for arm in a["arms"]:
            cells = r[arm]["cells"]
            vals = [max(c["rate"] for c in cells if c["cell_idx"] == ci) for ci in range(len(s["cells"]))]
            draw_cells(images[s["tag"]], s["pixels"], vals, out / "heatmaps" / f"{s['tag']}_{arm}.png")
            r[arm]["location_matters"] = r[arm]["best"]["rate"] - r[arm]["median"] >= PEAK
            row[f"{arm}_best"] = r[arm]["confirm_rate"]
            row[f"{arm}_loc"] = r[arm]["location_matters"]
        r["go_handoff"] = "handoff" in a["arms"] and row["handoff_best"] >= GO_RATE and row["handoff_best"] >= base + GO_MARGIN
        r["go_primitive"] = "scripted" in a["arms"] and row["scripted_best"] >= GO_RATE
        rows.append(row)

    n = len(states)
    frac_h = sum(summary[s["tag"]]["go_handoff"] for s in states) / n
    frac_p = sum(summary[s["tag"]]["go_primitive"] for s in states) / n
    verdict = "GO (handoff)" if frac_h >= GO_FRAC else "GO (primitive)" if frac_p >= GO_FRAC else "NO-GO"
    loc = {arm: sum(summary[s["tag"]][arm]["location_matters"] for s in states) / n for arm in a["arms"]}
    report = {"settings": {k: v for k, v in a.items()}, "states": states, "results": summary,
              "verdict": {"verdict": verdict, "frac_states_go_handoff": frac_h, "frac_states_go_primitive": frac_p,
                          "frac_states_location_matters": loc,
                          "thresholds": {"GO_RATE": GO_RATE, "GO_MARGIN": GO_MARGIN, "GO_FRAC": GO_FRAC, "PEAK": PEAK}}}
    (out / "metadata.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'state':45s} {'object':22s} base  " + "  ".join(f"{arm}(best, loc)" for arm in a["arms"]))
    for row in rows:
        print(f"{row['state']:45s} {row['object']:22s} {row['baseline']:.2f}  "
              + "  ".join(f"{row[f'{arm}_best']:.2f} {'Y' if row[f'{arm}_loc'] else 'n'}      " for arm in a["arms"]))
    print(f"\nVERDICT: {verdict}  (handoff GO on {frac_h:.0%} of states, primitive GO on {frac_p:.0%}; "
          f"location matters: {', '.join(f'{k} {v:.0%}' for k, v in loc.items())})")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
