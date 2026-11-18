"""Roll out pi0 + the VLM subtask judge (fa_core.judge_agent) and score its judgements against the predicates.

For every libero_90 task given: pi0 runs the task; the VLM agent watches at 1 fps, judges the current subtask
(pick(object) then place(destination)) and corrects high-level failures. The privileged subtask predicates
(fa_core.predicates: pick, strict place, LIBERO goal) are evaluated at every step as ground truth; the agent
never sees them.

Output in --out:
  task<T>_init<I>.mp4    real-time video (20 fps), labelled every frame with the current subtask, the VLM's
                         judgement and the predicates' judgement, plus each high-level correction and why
  task<T>_init<I>.json   VLM decisions, corrections, predicate timeline, per-call agreement
  summary.{md,json}      agreement over all episodes

  cd failure_agent_prototype/tools
  MUJOCO_GL=egl /opt/openpi/examples/libero/.venv/bin/python run_judge.py --tasks 2 9 10 \\
      --out ../../Documentation/success_criteria_judge --workers 5
Needs the pi0 server (serve_pi0.sh) and the authenticated `claude` CLI.
"""

import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
import textwrap
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fa_core import camera, common, judge_agent, loop, predicates as P, vlm as V  # noqa: E402

SUITE = "libero_90"
SCALE = 2                       # agentview / wrist upscaled 256 -> 512 in the video
PANEL_H = 296
SHOW_CORRECTION_AFTER = 40      # frames a finished correction stays on screen

GREEN, RED, GREY, YELLOW, ORANGE, WHITE, CYAN = ((60, 200, 90), (230, 70, 60), (150, 150, 150), (250, 210, 60),
                                                 (250, 150, 40), (235, 235, 235), (90, 200, 230))


# ------------------------------------------------------------------ one episode ---------------------

def run_one(task_id, init_idx, a):
    from openpi_client import websocket_client_policy

    log = logging.getLogger(f"task{task_id}")
    np.random.seed(7)
    env, task, _ = common.make_env(SUITE, task_id)
    client = websocket_client_policy.WebsocketClientPolicy(a.host, a.port)
    env.reset()
    init = common.init_state(SUITE, task_id, init_idx)
    obs = env.set_init_state(init)

    pick_place = [c for c in P.goal_checks(env) if c.kind in ("pick", "place")]
    goal = P.GoalCheck()
    checks = pick_place + [goal]
    size = common.ENV_RESOLUTION

    def unproject(u, v):  # RGB-D camera only (depth render + calibration), no object poses
        return camera.unproject(env, u, v, size)

    vlm = V.ClaudeCLI(model=a.model, effort=a.effort)
    agent = judge_agent.SubtaskJudgeAgent(vlm, unproject)
    timeline, wrist = {}, {}

    def hook(step, o):
        rec = {}
        for c in checks:
            try:
                now = bool(c.holds(env))
            except Exception:
                now = None
            rec[c.spec] = [bool(c.done), now]
        timeline[step] = rec
        wrist[step] = common.upright(o["robot0_eye_in_hand_image"]).copy()

    t0 = time.time()
    try:
        res = loop.run_episode(env, obs, str(task.language), a.max_steps, client=client, agent=agent,
                               checks=checks, init_state=init, stop_on_success=False, step_hook=hook, log=log)
    finally:
        env.close()
    ends = {e["started"]: e["step"] for e in res.events if e["type"] == "override_end"}
    for c in agent.corrections:  # when each scripted correction actually finished (or was cut by the episode end)
        if c["step"] in ends:
            c["end_step"] = ends[c["step"]]
    specs = [c.spec for c in pick_place]
    result = score_episode(task_id, init_idx, str(task.language), res, agent, timeline, specs, goal.spec)
    result.update({"wall_time_s": round(time.time() - t0, 1), "vlm_model": a.model, "vlm_calls": vlm.calls,
                   "vlm_cost_usd": round(vlm.cost_usd, 3),
                   "vlm_mean_latency_s": round(vlm.latency_s / max(vlm.calls, 1), 2)})
    name = f"task{task_id}_init{init_idx}"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    render_video(out / f"{name}.mp4", res, wrist, timeline, agent, specs, goal.spec, result)
    (out / f"{name}.json").write_text(json.dumps(result, indent=1, default=loop._json_default))
    log.info("%s: libero=%s pred=%s vlm=%s, %d calls, %d corrections, $%.2f, %.0f s", name,
             result["libero_success_any"], result["predicate_done_step"], result["vlm_done_step"],
             vlm.calls, len(agent.corrections), vlm.cost_usd, time.time() - t0)
    return result


def score_episode(task_id, init_idx, language, res, agent, timeline, specs, goal_spec):
    calls = []
    for d in agent.decisions:
        if "subtask_success" not in d:
            continue
        k, s = d["subtask_idx"], d["step"]
        pred = timeline.get(s, {}).get(specs[k], [None, None])
        calls.append({"step": s, "subtask_idx": k, "kind": ["pick", "place"][k], "vlm": bool(d["subtask_success"]),
                      "pred_latched": pred[0], "pred_now": pred[1], "failure": d["failure"],
                      "action": d["action"]})
    steps = sorted(timeline)
    pred_done = {}
    for spec in specs + [goal_spec]:
        first = next((s for s in steps if timeline[s][spec][0]), None)
        pred_done[spec] = first
    vlm_first = {}
    for k in range(2):
        vlm_first[["pick", "place"][k]] = next((c["step"] for c in calls if c["subtask_idx"] == k and c["vlm"]), None)
    return {
        "suite": SUITE, "task_id": task_id, "init_state_idx": init_idx, "task_language": language,
        "plan": agent.plan, "subtasks": [s["label"] for s in agent.subtasks], "predicate_specs": specs,
        "num_steps": res.num_steps, "stop_reason": res.stop_reason, "error": res.error,
        "libero_success_any": pred_done[goal_spec] is not None,
        "libero_success_final": bool(timeline[steps[-1]][goal_spec][1]) if steps else False,
        "predicate_done_step": {"pick": pred_done[specs[0]], "place": pred_done[specs[1]], "goal": pred_done[goal_spec]},
        "vlm_done_step": vlm_first,
        "corrections": agent.corrections, "calls": calls, "decisions": agent.decisions, "events": res.events,
    }


# ------------------------------------------------------------------ video ---------------------------

def _font(size, bold=False):
    from PIL import ImageFont

    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def render_video(path, res, wrist, timeline, agent, specs, goal_spec, result):
    import imageio
    from PIL import Image, ImageDraw

    f, fb, fs = _font(15), _font(15, True), _font(13)
    decisions = [d for d in agent.decisions if "after" in d]
    labels = [s["label"] for s in agent.subtasks] or ["pick(?)", "place(?)"]
    corrections = agent.corrections
    W = 2 * 256 * SCALE
    writer = imageio.get_writer(str(path), fps=common.CONTROL_FREQ, macro_block_size=8)
    di = -1
    for i, frame in enumerate(res.frames):
        while di + 1 < len(decisions) and decisions[di + 1]["step"] <= i:
            di += 1
        dec = decisions[di] if di >= 0 else None
        top = Image.new("RGB", (W, 256 * SCALE + PANEL_H), (20, 20, 24))
        top.paste(Image.fromarray(frame).resize((256 * SCALE, 256 * SCALE), Image.BICUBIC), (0, 0))
        wr = wrist.get(i, wrist.get(i - 1))
        if wr is not None:
            top.paste(Image.fromarray(wr).resize((256 * SCALE, 256 * SCALE), Image.BICUBIC), (256 * SCALE, 0))
        g = ImageDraw.Draw(top)
        g.text((8, 6), "front camera", fill=WHITE, font=fs)
        g.text((256 * SCALE + 8, 6), "wrist camera", fill=WHITE, font=fs)
        if dec is not None and dec["step"] == i and "subtask_success" in dec:
            g.ellipse((W - 26, 8, W - 12, 22), fill=CYAN)
            g.text((W - 120, 7), "VLM call", fill=CYAN, font=fs)

        # active correction (shown while it runs and SHOW_CORRECTION_AFTER frames after)
        corr = None
        for c in corrections:
            end = c.get("end_step", c["step"]) + SHOW_CORRECTION_AFTER
            if c["step"] <= i <= end:
                corr = c
        if corr is not None and corr.get("point_img") is not None:
            u, v = (np.asarray(corr["point_img"]) * SCALE).tolist()
            g.line((u - 12, v, u + 12, v), fill=YELLOW, width=3)
            g.line((u, v - 12, u, v + 12), fill=YELLOW, width=3)
            g.ellipse((u - 16, v - 16, u + 16, v + 16), outline=YELLOW, width=2)

        y = 256 * SCALE + 6
        state = dec["after"] if dec else {"idx": 0, "subtasks": [{"label": l, "done_step": None} for l in labels]}
        g.text((8, y), f"libero_90 task {result['task_id']}  init {result['init_state_idx']}   step {i}  "
                       f"(t = {i / common.CONTROL_FREQ:5.2f} s)   \"{result['task_language']}\"", fill=WHITE, font=fb)
        y += 24
        g.text((8, y), "subtask", fill=GREY, font=fs)
        g.text((400, y), "VLM judgement", fill=GREY, font=fs)
        g.text((640, y), "predicate judgement", fill=GREY, font=fs)
        y += 18
        tl = timeline.get(i, {})
        for k, lab in enumerate(labels):
            cur = k == state["idx"]
            g.text((8, y), ("> " if cur else "   ") + lab, fill=YELLOW if cur else WHITE, font=fb if cur else f)
            st = state["subtasks"][k] if k < len(state["subtasks"]) else {"done_step": None}
            if st.get("done_step") is not None:
                txt, col = f"SUCCESS (@{st['done_step']})", GREEN
            elif cur and dec is not None and "subtask_success" in dec and dec["subtask_idx"] == k:
                txt, col = "no success", RED
            elif cur and dec is not None and "subtask_success" in dec:
                txt, col = "no success (not judged yet)", RED
            else:
                txt, col = "-", GREY
            g.text((400, y), txt, fill=col, font=f)
            done, now = tl.get(specs[k], [None, None]) if k < len(specs) else (None, None)
            first = result["predicate_done_step"]["pick" if k == 0 else "place"]
            ptxt = (f"SUCCESS (@{first})" if done else "no success") + ("   [holds now]" if now else "")
            g.text((640, y), ptxt, fill=GREEN if done else RED, font=f)
            y += 20
        gd, gn = tl.get(goal_spec, [None, None])
        g.text((8, y), "   whole task (LIBERO goal)", fill=WHITE, font=f)
        g.text((640, y), "SUCCESS" if gn else ("succeeded earlier, not now" if gd else "no success"),
               fill=GREEN if gn else (ORANGE if gd else RED), font=f)
        y += 22
        g.text((8, y), f"pi0 prompt: \"{_prompt_at(res.events, i)}\"", fill=GREY, font=fs)
        y += 18
        if corr is not None:
            running = i <= corr.get("end_step", corr["step"])
            pt = ("" if corr.get("point_grid") is None else
                  f" -> pointed px {corr['point_grid']} (world {corr.get('target_world')})")
            head = (f"HIGH-LEVEL CORRECTION ({'running' if running else 'done'}) @ step {corr['step']}: "
                    f"{corr['action']}{pt}   [{corr['failure_mode']}]")
            g.rectangle((4, y - 2, W - 4, y + 64), outline=YELLOW, width=2)
            g.text((10, y), head, fill=YELLOW, font=fb)
            why = "why: " + corr["reason"] + (f"  | new pi0 prompt: \"{corr['new_prompt']}\"" if corr.get("new_prompt") else "")
            for line in textwrap.wrap(why, 125)[:2]:
                y += 20
                g.text((10, y), line, fill=WHITE, font=f)
        elif dec is not None and "subtask_success" in dec:
            fail = dec["failure"]
            col = {"none": GREY, "low_level": ORANGE, "high_level": YELLOW}.get(fail, GREY)
            tag = "" if fail == "none" else f"  [{fail}: {dec['failure_mode']}]"
            extra = f"  ({dec['decision']})" if dec.get("decision") else ""
            for j, line in enumerate(textwrap.wrap(f"VLM @{dec['step']}{tag}{extra}: {dec['observation']}", 130)[:3]):
                g.text((8, y + 20 * j), line, fill=col if j == 0 else WHITE, font=fs)
        elif dec is not None and dec.get("error"):
            g.text((8, y), f"VLM @{dec['step']}: call failed ({dec['error'][:90]})", fill=ORANGE, font=fs)
        writer.append_data(np.asarray(top))
    writer.close()


def _prompt_at(events, step):
    prompt = None
    for e in events:
        if e["step"] > step:
            break
        if e["type"] == "start":
            prompt = e["prompt"]
        elif e["type"] == "set_prompt":
            prompt = e["new"]
    return prompt


# ------------------------------------------------------------------ summary -------------------------

def summarize(out):
    eps = [json.loads(p.read_text()) for p in sorted(pathlib.Path(out).glob("task*_init*.json"))]
    rows, conf = [], {}
    for kind in ("pick", "place"):
        for ref in ("pred_latched", "pred_now"):
            c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for e in eps:
                for x in e["calls"]:
                    if x["kind"] != kind or x[ref] is None:
                        continue
                    c[("t" if x["vlm"] == x[ref] else "f") + ("p" if x["vlm"] else "n")] += 1
            n = sum(c.values())
            c["n"], c["agreement"] = n, round((c["tp"] + c["tn"]) / n, 3) if n else None
            conf[f"{kind}/{ref}"] = c
    for e in sorted(eps, key=lambda e: e["task_id"]):
        p, v = e["predicate_done_step"], e["vlm_done_step"]
        rows.append({"task": e["task_id"], "init": e["init_state_idx"], "language": e["task_language"],
                     "subtasks": e["subtasks"], "pred_pick": p["pick"], "vlm_pick": v["pick"],
                     "pred_place": p["place"], "vlm_place": v["place"], "libero_any": e["libero_success_any"],
                     "libero_final": e["libero_success_final"], "stop": e["stop_reason"], "steps": e["num_steps"],
                     "corrections": [f"{c['action']}@{c['step']}" for c in e["corrections"]],
                     "calls": len(e["calls"]), "cost": e.get("vlm_cost_usd")})
    summary = {"episodes": rows, "per_call_confusion": conf}
    (pathlib.Path(out) / "summary.json").write_text(json.dumps(summary, indent=1))

    def cell(x):
        return "-" if x is None else str(x)

    md = ["# VLM subtask judge vs. predicates", "",
          "Per-call agreement (VLM's judgement of the current subtask at each 1 fps call vs. the predicate at the "
          "same step). `latched` = predicate as defined in fa_core/predicates.py (held 10 steps, then latches); "
          "`now` = the predicate's condition at that instant.", "",
          "| subtask / reference | n | agree | VLM yes & pred yes | VLM yes & pred no | VLM no & pred yes | VLM no & pred no |",
          "|---|---|---|---|---|---|---|"]
    for k, c in conf.items():
        md.append(f"| {k} | {c['n']} | {cell(c['agreement'])} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} |")
    md += ["", "Per episode: first step at which each judged success (`-` = never).", "",
           "| task | pick pred | pick VLM | place pred | place VLM | LIBERO (any / at end) | stop | steps | corrections |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['task']}: {r['language']} | {cell(r['pred_pick'])} | {cell(r['vlm_pick'])} | "
                  f"{cell(r['pred_place'])} | {cell(r['vlm_place'])} | {r['libero_any']} / {r['libero_final']} | "
                  f"{r['stop'] or 'max steps'} | {r['steps']} | {', '.join(r['corrections']) or '-'} |")
    (pathlib.Path(out) / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ------------------------------------------------------------------ main ----------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", type=int, nargs="+", required=True)
    p.add_argument("--init", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=common.REPO / "Documentation" / "success_criteria_judge")
    p.add_argument("--max-steps", type=int, default=common.MAX_STEPS[SUITE])
    p.add_argument("--model", default=V.DEFAULT_MODEL)
    p.add_argument("--effort", default=None)
    p.add_argument("--workers", type=int, default=1, help="parallel episode processes")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--summarize-only", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if a.summarize_only:
        summarize(a.out)
        return
    if len(a.tasks) == 1 or a.workers <= 1:
        for t in a.tasks:
            try:
                run_one(t, a.init, a)
            except Exception:
                logging.exception("task %d failed", t)
    else:
        base = [sys.executable, __file__, "--init", str(a.init), "--out", str(a.out), "--max-steps",
                str(a.max_steps), "--model", a.model, "--host", a.host, "--port", str(a.port)]
        if a.effort:
            base += ["--effort", a.effort]
        queue, running = list(a.tasks), []
        while queue or running:
            while queue and len(running) < a.workers:
                t = queue.pop(0)
                running.append((t, subprocess.Popen(base + ["--tasks", str(t)])))
                time.sleep(5)  # stagger EGL context creation
            for t, proc in list(running):
                if proc.poll() is not None:
                    running.remove((t, proc))
                    logging.info("task %d exited with %d", t, proc.returncode)
            time.sleep(2)
    summarize(a.out)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
