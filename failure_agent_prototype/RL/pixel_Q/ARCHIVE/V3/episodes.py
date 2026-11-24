"""V3 actor side: one pi0 main rollout with perturbed branch episodes at randomly selected gated ticks.

The method of the tick-loop pilot (plan_the_algo.txt sec. 7):
  - pi0 (+ the HL stand-in, as V2) runs from a task init state; a tick every rc["tick_steps"] control steps.
  - at a gate-open tick (V2 gate + grasp point <= gate_max_above_m above the target's top, no HL correction in
    the last 3 s) the heightmap is observed; target grounded =
    a gated tick. With probability rc["p_select"] the tick is selected: snapshot -> episode 0 natural (pi0 alone
    from the snapshot) + n_perturb perturbed episodes (small OSC move around pi0's own pose, pi0's queue cleared,
    pi0 alone) -> restore -> the natural main rollout continues.
  - every episode observes the heightmap where pi0 takes over and runs pi0 until pick success or h_steps; the
    raw outcome (t_success, end) is stored, the label gamma^(t/20) is made by the learner.
  - the main rollout ends on pick success, the suite budget (20 s for LIBERO-90), or an irrecoverable failure.
  - kept samples: natural episodes, and perturbed episodes that reached the pose without bumping the target;
    the target must be grounded at takeover.

Video (record=True): main-rollout frames with a banner, a pause at every tick once gating starts (SELECTED /
not selected / not gated, V_est of the current V), the branch episodes inline (V_est at takeover, timer, true
outcome). Written by the actor itself (frames never travel back to the learner).

Uses V2's LIBERO backend (../V2/libero_env.py) unchanged.
"""

import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "V2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vnet  # noqa: E402

CTRL_HZ = 20
FPS = 20
COOLDOWN_S = 1.          # no gated tick within this long after an HL correction (V2)
BUMP_M = 0.005            # target moved more than this during the perturbation move = bumped
OBS_FIELDS = vnet.OBS_FIELDS

WHITE, YELLOW, GREEN, RED, CYAN = (235, 235, 235), (255, 220, 60), (80, 230, 80), (255, 80, 80), (90, 220, 255)
GREY, MAGENTA = (150, 150, 150), (255, 120, 255)


def env_cfg(hl, host, port, seed=0, gate_max_above_m=0.10):
    """V2 phase 1 backend settings, plus V3's gate height condition (grasp point at most gate_max_above_m above
    the target's top; None = off, as V2)."""
    return {"env": "libero", "reach": 3, "gate_margin_m": 0.04, "gate_min_gap": 0.05,
            "gate_max_above_m": gate_max_above_m, "min_mask_px": 20,
            "agent_px": None, "wrist_px": None, "fill_holes": True, "robot_mask": "seg", "finger_mask": None,
            "mask_source": "seg", "mix_p": 0.5, "box_jitter": 0.05, "clear_min": 0.02, "clear_to": 0.03,
            "hl": hl, "vlm_model": None, "env_seed": 7, "seed": seed, "host": host, "port": port, "same_noise": False}


def rot_xyz(ax, ay, az):
    """Rx(ax) @ Ry(ay) @ Rz(az), degrees."""
    a, b, c = np.radians([ax, ay, az])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rx @ ry @ rz


def sample_offset(rng, rc):
    d = np.r_[rng.uniform(-rc["pos_cm"], rc["pos_cm"], 2), rng.uniform(rc["dz_cm"][0], rc["dz_cm"][1])] / 100.0
    return d, rng.uniform(-rc["rot_deg"], rc["rot_deg"], 3)


# ------------------------------------------------------------------ one episode --------------------

def run_episode(env, snap, offset, rc, tpos0, record):
    """Restore, optional perturbation move, observe at takeover, pi0 until pick or h_steps.
    Returns (info, obs or None, frames or None, n_move)."""
    from fa_core import common, motion, predicates as P
    from shared import scene as SC

    T = env.T
    env.restore(snap)
    env.frames = [SC.video_frame(env.obs)] if record else None
    info = {"natural": offset is None, "arrived": True, "bumped": False, "move_steps": 0,
            "d_cm": [0.0, 0.0, 0.0], "rot_deg": [0.0, 0.0, 0.0]}
    if offset is not None:
        d, r = offset
        ee0 = np.array(env.obs["robot0_eef_pos"], dtype=np.float64)
        q0 = np.array(env.obs["robot0_eef_quat"], dtype=np.float64)
        m = motion.MoveTo(ee0 + d, common.GRIPPER_OPEN, T.mat2quat(T.quat2mat(q0) @ rot_xyz(*r)),
                          max_steps=rc["move_cap"])
        env.plan.clear()
        _, n = env._run_motion(m)
        env.plan.clear()                                        # pi0 replans from the new pose
        shift = float(np.linalg.norm(P.obj_pos(env.env, env.obj) - tpos0))
        info.update(arrived=bool(m.reached), bumped=shift > BUMP_M, move_steps=int(n),
                    d_cm=(d * 100).round(2).tolist(), rot_deg=r.round(1).tolist())
    obs = env.observe()
    ti = env.tick_info()
    info.update(grounded=obs is not None, gate_open=bool(env.gate()[0]), tilt_deg=float(ti["tilt_deg"]),
                target_dxy=float(ti["target_dxy"]))
    t0, h = env.step_count, rc["h_steps"]
    while True:
        if env.pick_done():
            info.update(end="success", t_success=max(0, int(env.pick_step() - t0)))
            break
        fail = env.failure()
        if fail or env.step_count - t0 >= h:
            info.update(end=fail or "timeout", t_success=-1)
            break
        env.pi0_step()
    info["pi0_steps"] = int(env.step_count - t0)
    frames = env.frames
    env.frames = None
    return info, obs, frames, info["move_steps"]


def keep_episode(info, obs):
    return obs is not None and (info["natural"] or (info["arrived"] and not info["bumped"]))


# ------------------------------------------------------------------ video helpers ------------------

def annotate(frame, lines, colors, scale=2):
    import cv2

    img = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_LINEAR)
    lh = 26
    banner = np.zeros((lh * len(lines) + 10, img.shape[1], 3), np.uint8)
    for i, (txt, col) in enumerate(zip(lines, colors)):
        cv2.putText(banner, txt, (10, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 1, cv2.LINE_AA)
    return np.concatenate([img, banner], axis=0)


def label_of(info, rc):
    t = info["t_success"]
    return float(rc["gamma"] ** (t / float(CTRL_HZ))) if 0 <= t <= rc["h_steps"] else 0.0


def render_episode(frames, info, v, head, ep, n_ep, rc, hold_start=10, hold_end=25):
    if info["natural"]:
        desc = "ep {}/{} NATURAL (pi0 pose at the tick)".format(ep, n_ep)
    else:
        d, r = info["d_cm"], info["rot_deg"]
        desc = "ep {}/{} PERTURBED d=({:+.1f},{:+.1f},{:+.1f})cm rot=({:+.0f},{:+.0f},{:+.0f})deg".format(
            ep, n_ep, d[0], d[1], d[2], r[0], r[1], r[2])
        if not info["arrived"]:
            desc += " [not reached: dropped]"
        if info["bumped"]:
            desc += " [bumped: dropped]"
    vt = "V_est {:.2f} +- {:.2f}".format(*v) if v is not None else "V_est n/a"
    H, nm, out = rc["h_steps"] / float(CTRL_HZ), info["move_steps"], []
    for i, f in enumerate(frames):
        if 0 < i <= nm:
            out.append(annotate(f, [head, desc, "MOVING to perturbed pose  {:.2f} s".format(i / float(CTRL_HZ)),
                                    "V_est (at arrival)"], [WHITE, YELLOW, CYAN, WHITE]))
        else:
            out.append(annotate(f, [head, desc, "pi0  timer {:.2f} / {:.2f} s".format((i - nm) / float(CTRL_HZ), H),
                                    vt + "   true V: running"], [WHITE, YELLOW, WHITE, WHITE]))
    y = label_of(info, rc)
    if info["end"] == "success":
        fin, col = "{}   true V {:.2f}  (picked at {:.2f} s)".format(vt, y, info["t_success"] / float(CTRL_HZ)), \
            (GREEN if y > 0 else RED)
    else:
        fin, col = "{}   true V 0.00  ({})".format(vt, info["end"]), RED
    last = annotate(frames[-1], [head, desc, "END  timer {:.2f} / {:.2f} s".format(info["pi0_steps"] / float(CTRL_HZ), H),
                                 fin], [WHITE, YELLOW, col, col])
    return [out[0]] * hold_start + out + [last] * hold_end


# ------------------------------------------------------------------ one rollout --------------------

def collect_rollout(env, start, rc, rng, net=None, video_path=None, hold_tick=12):
    """One main rollout. rc: tick_steps, p_select, n_perturb, pos_cm, dz_cm, rot_deg, move_cap, h_steps, gamma.
    net: V for the video's V_est (only with video_path). Returns dict(samples, stats)."""
    from fa_core import predicates as P
    from shared import scene as SC

    record = video_path is not None
    t_wall = time.time()
    env.reset(start, record=record)
    tag = start["tag"]
    samples = []
    st = {"gated_ticks": 0, "selected_ticks": 0, "natural_run": 0, "natural_success": 0, "perturbed_run": 0,
          "perturbed_arrived": 0, "perturbed_bumped": 0, "perturbed_kept": 0, "perturbed_success": 0,
          "not_grounded": 0, "hl_corrections": 0}
    video, first, last_v = [], None, None
    head = "{} | {} | MAIN natural rollout".format(tag, env.prompt[:44])

    def vpred(obs):
        return vnet.predict(net, obs) if (net is not None and obs is not None) else None

    def flush_main():
        if not record:
            return
        n, end_step = len(env.frames), env.step_count
        for i, f in enumerate(env.frames):
            s = end_step - n + 1 + i
            tt = "step {} | no gated tick yet".format(s) if first is None else \
                "step {} | {:.2f} s since first gated tick (budget {:.0f} s)".format(
                    s, (s - first) / float(CTRL_HZ), env.budget / float(CTRL_HZ))
            vt = "last V_est {:.2f} +- {:.2f}".format(*last_v) if last_v else "last V_est -"
            video.append(annotate(f, [head, tt, "{} | gated {} | selected {} | P(select) {:g}".format(
                vt, st["gated_ticks"], st["selected_ticks"], rc["p_select"]), status[0]],
                [WHITE, WHITE, WHITE, status[1]]))
        env.frames = []

    def not_gated_why(gate_open, ginfo, cooling_s):
        """Why a tick was not gated: the failed gate conditions, the HL cooldown, or (gate open, no cooldown) the
        target not grounded in agentview."""
        why = []
        if not gate_open:
            if ginfo.get("pick") is False:
                why.append("pick done")
            if ginfo.get("open") is False:
                why.append("gripper not open")
            if ginfo.get("grounded") is False:
                why.append("target not grounded (HL)")
            elif ginfo.get("near") is False:
                why.append("{:.0f} cm from target xy (max {:.0f})".format(100 * ginfo["target_dxy"], 100 * env.gate_m))
            if ginfo.get("low") is False:
                why.append("{:.0f} cm above target top (max {:.0f})".format(
                    100 * ginfo["above_top"], 100 * env.cfg["gate_max_above_m"]))
            if ginfo.get("empty") is False:
                why.append("holding an object")
        if cooling_s > 0:
            why.append("HL cooldown ({:.1f} s left)".format(cooling_s))
        elif gate_open:
            why.append("target not grounded in agentview")
        return ", ".join(why) or "?"

    def pause(lines, colors, n):
        if record:
            video.extend([annotate(SC.video_frame(env.obs), lines, colors)] * n)

    last_tick, cooldown_until, end = None, -10 ** 9, None
    cooldown = int(COOLDOWN_S * CTRL_HZ)
    status = ("gate: no tick yet", GREY)       # 4th banner line of the main-rollout frames: the last tick's gate
    while True:
        step = env.step_count
        if env.pick_done():
            end = "success"
        elif env.failure():
            end = env.failure()
        elif step >= env.budget:
            end = "budget"
        if end:
            break
        if step % rc["tick_steps"] == 0 and step != last_tick:
            last_tick = step
            flush_main()                       # frames since the previous tick carry that tick's gate status
            hl = env.hl_tick()
            if hl is not None:
                st["hl_corrections"] += 1
                cooldown_until = env.step_count + cooldown
                status = ("HL CORRECTION #{}: {} -> {} (scripted motion, no ticks)".format(
                    st["hl_corrections"], hl.get("mode"), hl.get("action")), CYAN)
                flush_main()                   # the correction's own frames
                pause([head, "steps {}-{}: HL correction #{} ({} -> {})".format(
                    step, env.step_count, st["hl_corrections"], hl.get("mode"), hl.get("action")),
                       "HL CORRECTION done: no gated tick for the next {:g} s (HL cooldown)".format(COOLDOWN_S), ""],
                      [WHITE, WHITE, CYAN, WHITE], hold_tick)
                status = ("not gated: HL cooldown ({:.1f} s left)".format(COOLDOWN_S), MAGENTA)
                continue
            gate_open, ginfo = env.gate()
            cooling_s = max(0, cooldown_until - step) / float(CTRL_HZ)
            obs = env.observe() if gate_open and cooling_s == 0 else None
            if obs is not None:
                status = ("gate OPEN at step {} (tick #{})".format(step, st["gated_ticks"] + 1), GREEN)
                if first is None:
                    first = step
                st["gated_ticks"] += 1
                k = st["gated_ticks"]
                sel = bool(rng.random() < rc["p_select"])
                st["selected_ticks"] += int(sel)
                last_v = vpred(obs)
                tsec = "step {} ({:.2f} s since first gated tick)".format(step, (step - first) / float(CTRL_HZ))
                flush_main()
                lab = ("TICK #{}  SELECTED -> 1 natural + {} perturbed episodes".format(k, rc["n_perturb"]), GREEN) \
                    if sel else ("TICK #{}  not selected (pi0 continues)".format(k), GREY)
                pause([head, tsec + " | gated", lab[0], "V_est {:.2f} +- {:.2f}".format(*last_v) if last_v else ""],
                      [WHITE, WHITE, lab[1], MAGENTA], hold_tick)
                if sel:
                    snap = env.snapshot()
                    tpos0 = P.obj_pos(env.env, env.obj)
                    offs = [None] + [sample_offset(rng, rc) for _ in range(rc["n_perturb"])]
                    bhead = "{} | tick #{} {} | H {:g} s".format(tag, k, tsec, rc["h_steps"] / float(CTRL_HZ))
                    ys = []
                    for ep, off in enumerate(offs):
                        info, eobs, frames, _ = run_episode(env, snap, off, rc, tpos0, record)
                        if info["natural"]:
                            st["natural_run"] += 1
                            st["natural_success"] += int(info["end"] == "success")
                        else:
                            st["perturbed_run"] += 1
                            st["perturbed_arrived"] += int(info["arrived"])
                            st["perturbed_bumped"] += int(info["bumped"])
                        if eobs is None:
                            st["not_grounded"] += 1
                        if keep_episode(info, eobs):
                            if not info["natural"]:
                                st["perturbed_kept"] += 1
                                st["perturbed_success"] += int(info["end"] == "success")
                            samples.append(dict({f: eobs[f] for f in OBS_FIELDS}, t_success=info["t_success"],
                                                cap_steps=rc["h_steps"], end=info["end"], natural=info["natural"],
                                                tick_step=step, tick=k, episode=ep, d_cm=info["d_cm"],
                                                rot_deg=info["rot_deg"], move_steps=info["move_steps"],
                                                tilt_deg=info["tilt_deg"], target_dxy=info["target_dxy"]))
                        ys.append(label_of(info, rc))
                        if record:
                            video += render_episode(frames, info, vpred(eobs), bhead, ep, len(offs) - 1, rc)
                    env.restore(snap)
                    if record:
                        env.frames = []
                    pause([head, tsec, "back to the natural rollout (original state)",
                           "branch true V: " + " ".join("{:.2f}".format(y) for y in ys)], [WHITE, WHITE, CYAN, WHITE],
                          hold_tick)
            else:
                if record:
                    why = not_gated_why(gate_open, ginfo, cooling_s)
                    status = ("not gated: " + why, MAGENTA if cooling_s > 0 else RED)
                if first is not None and record:
                    pause([head, "step {} ({:.2f} s since first gated tick)".format(
                        step, (step - first) / float(CTRL_HZ)), "TICK  not gated: {}".format(why), ""],
                          [WHITE, WHITE, RED, WHITE], hold_tick)
        env.pi0_step()
    st.update(end=end, main_picked=bool(env.pick_done()), steps=int(env.step_count), wall_s=round(time.time() - t_wall, 1))
    if record:
        flush_main()
        pause([head, "MAIN rollout end: {}".format(end), "gated {} | selected {} | samples kept {} | HL corrections {}".format(
            st["gated_ticks"], st["selected_ticks"], len(samples), st["hl_corrections"]), ""],
              [WHITE, GREEN if end == "success" else RED, WHITE, WHITE], 40)
        import imageio

        pathlib.Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(video_path), video, fps=FPS, macro_block_size=1)
    return {"samples": samples, "stats": st}
