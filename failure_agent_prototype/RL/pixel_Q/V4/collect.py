"""V4 data collection: worker processes run pi0 rollouts with perturbed branch episodes and write sample shards.

No learner and no V in here. V never steers collection (pi0 + random perturbations), so V is trained separately
by train_v.py on the GPU, after collection or alongside it on the shards written so far.
Vlearning_code_instruction.md has the details.

Rounds:
  1. --workers rollouts run in parallel (one per worker process). A job is an eval rollout (held-out task,
     --eval-tasks) while eval samples are short of both 3/7 of all samples and --eval-target-samples; otherwise a
     training rollout (--tasks). Starts come from shuffled cycles over (task, init).
  2. Every kept sample gets its split: eval = held-out tasks; val = --val-frac of the training (task, init) tags
     by a fixed hash; train = the rest. Samples are buffered and written to data/shard_NNNN.npz every ~200.
  3. After every round: one stdout line and the collect/* W&B metrics (sample counts so far, rates, ETA).
  4. Every --video-every training rollouts, that job records a video (episodes.py).
Stops when both sample targets are reached, or on Ctrl-C / SIGTERM (the buffered samples are written; the
rollouts of the interrupted round are lost). --resume continues a run: the counts come back from the shards and
the collection flags must match the run's config.json.

Outputs (--out-dir): data/shard_*.npz, config.json, videos/*.mp4, wandb/ (with --wandb-project).

Runs in the LIBERO client venv (py3.8, has openpi-client) and needs the pi0 server:
  PY=/opt/openpi/examples/libero/.venv/bin/python
  MUJOCO_GL=egl $PY collect.py --out-dir runs/v4 --wandb-project pixelQ_V4
"""

import argparse
import json
import os
import pathlib
import signal
import sys
import time

import numpy as np

# LIBERO's get_task_init_states torch.load()s its own numpy init-state files; torch >= 2.6 defaults to
# weights_only=True and refuses them. Set before torch is imported; the spawned workers inherit it.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_io import CTRL_HZ, read_meta, shard_paths, split_of, write_shard  # noqa: E402

TRAIN_TASKS = "libero_90:9,10,14,17,25,31,34,37,63,64,67,71,72,85"
EVAL_TASKS = "libero_90:13,15,24,36,65,76"
SHARD_ROWS = 200
# flags that define the data: --resume refuses a run whose config.json differs in any of them
DATA_FLAGS = ("tasks", "eval_tasks", "train_inits", "eval_inits", "tick_steps", "gate_max_above_cm", "p_select",
              "n_perturb", "pos_cm", "dz_cm", "rot_deg", "move_cap", "h_max_s", "hl", "val_frac")
STAT_KEYS = ("gated_ticks", "selected_ticks", "natural_run", "natural_success", "perturbed_run",
             "perturbed_arrived", "perturbed_bumped", "perturbed_kept", "perturbed_success", "not_grounded",
             "hl_corrections")


# ------------------------------------------------------------------ workers ------------------------

_W = {}


def _init_worker(cfg):
    import torch

    torch.set_num_threads(1)
    import episodes  # noqa: F401  (puts ../V2 on sys.path)
    from libero_env import BranchingPickEnv  # noqa: F401

    _W["cfg"] = dict(cfg, worker_id=os.getpid() % 100000)


def run_job(job):
    """One rollout in a worker. job: id, start, rc, cfg, seed, video (path or None)."""
    import episodes as E
    from shared.scene import IneligibleStart, ResetFailed

    if "env" not in _W:
        from libero_env import BranchingPickEnv

        _W.setdefault("cfg", job["cfg"])
        _W["env"] = BranchingPickEnv(_W["cfg"])
    rng = np.random.default_rng(job["seed"])
    try:
        res = E.collect_rollout(_W["env"], job["start"], job["rc"], rng, video_path=job["video"])
    except (IneligibleStart, ResetFailed) as e:
        return {"job": job["id"], "skip": "{}: {}".format(type(e).__name__, e)}
    res["job"] = job["id"]
    return res


def _sigterm(signum, frame):
    raise KeyboardInterrupt    # `kill <pid>` stops like Ctrl-C: the buffered samples are still written


# ------------------------------------------------------------------ main ---------------------------

def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(HERE / "runs/v4"))
    p.add_argument("--resume", action="store_true", help="continue the run in --out-dir (same collection flags)")
    # what is collected
    p.add_argument("--tasks", nargs="+", default=[TRAIN_TASKS], help="training tasks (held-out tasks refused)")
    p.add_argument("--eval-tasks", nargs="+", default=[EVAL_TASKS], help="unseen eval tasks (held-out split)")
    p.add_argument("--train-inits", default="0-39")
    p.add_argument("--eval-inits", default="0-39")
    p.add_argument("--tick-steps", type=int, default=4, help="control steps per tick (4 = 5 fps)")
    p.add_argument("--gate-max-above-cm", type=float, default=10.0,
                   help="gate closed while the grasp point is more than this far above the target's top (<0 = off)")
    p.add_argument("--p-select", type=float, default=0.5, help="chance a gated tick is selected for branching")
    p.add_argument("--n-perturb", type=int, default=8, help="perturbed episodes per selected tick (+ 1 natural)")
    p.add_argument("--pos-cm", type=float, default=3., help="xy offset +- (cm)")
    p.add_argument("--dz-cm", type=float, nargs=2, default=[-1., 2.], help="z offset range (cm)")
    p.add_argument("--rot-deg", type=float, default=10.0, help="+- about each gripper axis (deg)")
    p.add_argument("--move-cap", type=int, default=40, help="max control steps of the perturbation move")
    p.add_argument("--h-max-s", type=float, default=5.0, help="episode cap (s): the longest label window H")
    p.add_argument("--hl", default="privileged", choices=["privileged", "none"])
    p.add_argument("--val-frac", type=float, default=0.15, help="share of training (task, init) tags kept for val")
    # how much, how fast
    p.add_argument("--target-samples", type=int, default=12000, help="kept training-task samples (train + val)")
    p.add_argument("--eval-target-samples", type=int, default=5000, help="kept eval-task samples")
    p.add_argument("--workers", type=int, default=6, help="parallel rollouts (0 = in this process, debugging)")
    p.add_argument("--video-every", type=int, default=10, help="record every n-th training rollout (0 = never)")
    p.add_argument("--gamma", type=float, default=0.85,
                   help="discount per second, only for the true-V numbers in the videos (train_v.py makes the labels)")
    # infra
    p.add_argument("--host", default="0.0.0.0", help="pi0 server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--wandb-project", default="pixelQ_V4", help='W&B project ("" = no W&B)')
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    a = p.parse_args()
    a.mode, a.wandb_tags = "train", None   # runlog: "train" = a resumable run (W&B id kept in wandb_id.txt)
    return a


def main():
    a = parse()
    from shared import runlog, scene as SC
    import episodes as E

    out = pathlib.Path(a.out_dir)
    data, cfg_path = out / "data", out / "config.json"
    existing = shard_paths(data)
    if existing and not a.resume:
        raise SystemExit("{} already has {} shards: pass --resume to continue that run, or use a new --out-dir"
                         .format(data, len(existing)))
    if a.resume and cfg_path.exists():
        prev = json.loads(cfg_path.read_text())
        diff = {k: (prev.get(k), getattr(a, k)) for k in DATA_FLAGS if prev.get(k) != getattr(a, k)}
        if diff:
            raise SystemExit("--resume with other collection flags than {} has, {{flag: (old, new)}}: {}".format(
                cfg_path, diff))
    data.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(vars(a), indent=2))

    h_steps = int(round(a.h_max_s * CTRL_HZ))
    rc = {"tick_steps": a.tick_steps, "p_select": a.p_select, "n_perturb": a.n_perturb, "pos_cm": a.pos_cm,
          "dz_cm": a.dz_cm, "rot_deg": a.rot_deg, "move_cap": a.move_cap, "h_steps": h_steps, "gamma": a.gamma}
    cfg = E.env_cfg(a.hl, a.host, a.port,
                    gate_max_above_m=a.gate_max_above_cm / 100.0 if a.gate_max_above_cm >= 0 else None)

    train_starts = SC.task_starts(a.tasks, SC.parse_ids(a.train_inits))                       # refuses held-out
    eval_starts = SC.task_starts(a.eval_tasks, SC.parse_ids(a.eval_inits), allow_held_out=True)
    clash = {s["task_id"] for s in train_starts} & {s["task_id"] for s in eval_starts}
    if clash:
        raise SystemExit("tasks {} are in both --tasks and --eval-tasks".format(sorted(clash)))
    print("training starts {} ({} tasks), eval starts {} ({} tasks)".format(
        len(train_starts), len({s["task_id"] for s in train_starts}), len(eval_starts),
        len({s["task_id"] for s in eval_starts})), flush=True)

    # sample counts so far: restored from the shards on --resume
    count = {"train": 0, "val": 0, "eval": 0}
    kept = {"natural": 0, "perturbed": 0, "picked_natural": 0, "picked_perturbed": 0}
    n_rollouts, n_train_rollouts, n_shards = 0, 0, 0
    if a.resume:
        m = read_meta(data)
        if m:
            sp = m["split"].astype(str)
            for k in count:
                count[k] = int((sp == k).sum())
            nat, picked = m["natural"] > 0, m["t_success"] >= 0
            kept.update(natural=int(nat.sum()), perturbed=int((~nat).sum()),
                        picked_natural=int((nat & picked).sum()), picked_perturbed=int((~nat & picked).sum()))
            n_rollouts = int(m["rollout"].max()) + 1
            n_train_rollouts = len(set(m["rollout"][sp != "eval"].tolist()))
            n_shards = max(int(p.stem.split("_")[1]) for p in shard_paths(data)) + 1
        print("resumed: {} samples (train {} val {} eval {}) from {} shards, {} rollouts".format(
            sum(count.values()), count["train"], count["val"], count["eval"], n_shards, n_rollouts), flush=True)

    signal.signal(signal.SIGTERM, _sigterm)
    runlog.wb_init(a, out, step_metric="rollouts", prefixes=("collect/*",))
    pool = runlog.ActorPool(run_job, a.workers, _init_worker, (cfg,))
    rng = np.random.default_rng(n_rollouts)        # a resumed run does not replay the same start order
    cycles = {"train": [], "eval": []}
    bad = set()

    def next_start(kind):
        src = train_starts if kind == "train" else eval_starts
        for _ in range(2 * len(src) + 1):
            if not cycles[kind]:
                cycles[kind] = [src[i] for i in rng.permutation(len(src))]
            s = cycles[kind].pop()
            if s["tag"] not in bad:
                return s
        raise SystemExit("every {} start is ineligible".format(kind))

    def n_kept():
        return count["train"] + count["val"], count["eval"]

    agg = dict.fromkeys(STAT_KEYS + ("rollouts", "main_picked", "skipped", "samples"), 0)   # since this (re)start
    yield_est = {"train": 20.0, "eval": 20.0}            # running mean of kept samples per rollout, per kind
    shard_rows, interrupted, t0 = [], False, time.time()
    try:
        while True:
            n_tr, n_ev = n_kept()
            if n_tr >= a.target_samples and n_ev >= a.eval_target_samples:
                print("both sample targets reached", flush=True)
                break
            jobs = []
            pend = {"train": 0.0, "eval": 0.0}           # expected samples of the jobs already queued this round
            for _ in range(max(1, a.workers)):
                tr, ev = n_tr + pend["train"], n_ev + pend["eval"]
                want_eval = ev < a.eval_target_samples and (tr >= a.target_samples or ev < 3.0 / 7 * (tr + ev + 1))
                kind = "eval" if want_eval else "train"
                pend[kind] += yield_est[kind]
                s = next_start(kind)
                jid = n_rollouts + len(jobs)
                video = None
                if kind == "train":
                    n_train_rollouts += 1
                    if a.video_every and n_train_rollouts % a.video_every == 0:
                        video = str(out / "videos" / "rollout{:05d}_{}.mp4".format(jid, s["tag"]))
                jobs.append({"id": jid, "start": s, "kind": kind, "rc": rc, "cfg": cfg, "seed": 7919 * (jid + 1),
                             "video": video})
            results = pool.run(jobs)
            for res in sorted(results, key=lambda r: r["job"]):
                job = jobs[res["job"] - jobs[0]["id"]]
                tag = job["start"]["tag"]
                if "skip" in res:
                    bad.add(tag)
                    agg["skipped"] += 1
                    print("  skip {}: {}".format(tag, res["skip"]), flush=True)
                    continue
                split = "eval" if job["kind"] == "eval" else split_of(tag, a.val_frac)
                for smp in res["samples"]:
                    smp.update(tag=tag, split=split, rollout=job["id"], init=job["start"]["init_idx"])
                    shard_rows.append(smp)
                    count[split] += 1
                    key = "natural" if smp["natural"] else "perturbed"
                    kept[key] += 1
                    kept["picked_" + key] += int(smp["t_success"] >= 0)
                agg["samples"] += len(res["samples"])
                yield_est[job["kind"]] = 0.8 * yield_est[job["kind"]] + 0.2 * len(res["samples"])
                st = res["stats"]
                for k in STAT_KEYS:
                    agg[k] += st[k]
                agg["rollouts"] += 1
                agg["main_picked"] += int(st["main_picked"])
            n_rollouts += len(jobs)
            if len(shard_rows) >= SHARD_ROWS:
                write_shard(data / "shard_{:04d}.npz".format(n_shards), shard_rows)
                n_shards, shard_rows = n_shards + 1, []

            # ---- progress: stdout + W&B (x-axis: rollouts)
            n_tr, n_ev = n_kept()
            hours = (time.time() - t0) / 3600.0
            sph = agg["samples"] / max(hours, 1e-6)
            left = max(0, a.target_samples - n_tr) + max(0, a.eval_target_samples - n_ev)
            R = max(1, agg["rollouts"])
            rec = {"samples_total": n_tr + n_ev, "samples_train": count["train"], "samples_val": count["val"],
                   "samples_eval": n_ev, "samples_train_tasks": n_tr,
                   "progress_train_tasks": n_tr / float(max(1, a.target_samples)),
                   "progress_eval_tasks": n_ev / float(max(1, a.eval_target_samples)),
                   "samples_natural": kept["natural"], "samples_perturbed": kept["perturbed"],
                   "picked_frac_natural": kept["picked_natural"] / float(max(1, kept["natural"])),
                   "picked_frac_perturbed": kept["picked_perturbed"] / float(max(1, kept["perturbed"])),
                   "shards": n_shards,
                   "hours": hours, "rollouts_per_hour": agg["rollouts"] / max(hours, 1e-6),
                   "samples_per_hour": sph, "eta_hours": left / sph if sph > 0 else float("nan"),
                   "samples_per_rollout": agg["samples"] / float(R),
                   "gated_per_rollout": agg["gated_ticks"] / float(R),
                   "selected_per_rollout": agg["selected_ticks"] / float(R),
                   "arrival_rate": agg["perturbed_arrived"] / float(max(1, agg["perturbed_run"])),
                   "bump_rate": agg["perturbed_bumped"] / float(max(1, agg["perturbed_run"])),
                   "main_pick_rate": agg["main_picked"] / float(R), "rollouts_skipped": agg["skipped"]}
            runlog.wb_log(rec, "collect", step=n_rollouts)
            print("rollouts {} | samples {} (train {} val {} eval {}) | picked natural {:.2f} perturbed {:.2f} | "
                  "{:.0f} rollouts/h {:.0f} samples/h | {:.2f} h, ETA {:.1f} h".format(
                      n_rollouts, n_tr + n_ev, count["train"], count["val"], n_ev, rec["picked_frac_natural"],
                      rec["picked_frac_perturbed"], rec["rollouts_per_hour"], sph, hours, rec["eta_hours"]),
                  flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted: stopping the workers (the current round's rollouts are lost)", flush=True)
        if pool.pool is not None:
            pool.pool.terminate()
    finally:
        if not interrupted:
            pool.close()
        if shard_rows:
            write_shard(data / "shard_{:04d}.npz".format(n_shards), shard_rows)
            n_shards += 1
        runlog.wb_finish()
    n_tr, n_ev = n_kept()
    print("{} samples (train {} val {} eval {}) in {} shards under {}".format(
        n_tr + n_ev, count["train"], count["val"], n_ev, n_shards, data), flush=True)
    print("train V on the GPU:  /opt/simplevla/.venv/bin/python train_v.py --data {} --out-dir {}".format(
        data, out.parent / (out.name + "_train")), flush=True)


if __name__ == "__main__":
    main()
