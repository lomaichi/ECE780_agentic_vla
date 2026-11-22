"""V3 training: actors collect perturbed-branch V samples (episodes.py), the learner trains V online.

plan_the_algo.txt sec. 6-7. Rounds, as in V2:
  1. --workers rollouts run in parallel (one per actor). A job is an eval rollout (held-out task, --eval-tasks)
     while eval samples are short of both 3/7 of all samples and --eval-target-samples; otherwise a training
     rollout (--tasks). Starts come from shuffled cycles over (task, init).
  2. Samples go to the train / val / eval sets: eval = held-out tasks (test metrics only); val = --val-frac of the
     training (task, init) tags by a fixed hash (picks the best V, never trained on); train = the rest.
  3. The learner takes utd x (new training samples) gradient steps once --warmup samples are in.
  4. Every --eval-every rollouts: val MSE (v.pt saved when it improves), test metrics on the eval tasks
     (all / natural / perturbed), stdout + wandb.
  5. Every --video-every training rollouts, that job records a video with the current V (episodes.py).
Stops when both sample targets are reached. Then metrics.json / reliability.png / curves.png from the best v.pt.

Outputs (--out-dir): data/shard_*.npz, config.json, v.pt, metrics.json, reliability.png, curves.png,
videos/*.mp4, wandb/ (with --wandb-project). --resume reloads the shards and v.pt.

Runs in the LIBERO venv; needs the pi0 server (../V2/qlearning_code_instruction_V2.md sec. 3):
  PY=/opt/openpi/examples/libero/.venv/bin/python
  $PY train.py --out-dir runs/v3 --wandb-project pixelQ_V3
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vnet  # noqa: E402
from data_io import CTRL_HZ, OBS, label, read_shards, rows_of, split_of, write_shard  # noqa: E402
from train_v import calibration, metrics  # noqa: E402

TRAIN_TASKS = "libero_90:9,10,14,17,25,31,34,37,63,64,67,71,72,85"
EVAL_TASKS = "libero_90:13,15,24,36,65,76"


# ------------------------------------------------------------------ actors -------------------------

_W = {}


def _init_worker(cfg):
    import torch

    torch.set_num_threads(1)
    import episodes  # noqa: F401  (puts ../V2 on sys.path)
    from libero_env import BranchingPickEnv  # noqa: F401

    _W["cfg"] = dict(cfg, worker_id=os.getpid() % 100000)


def run_job(job):
    """One rollout in an actor. job: start, rc, seed, video (path or None), weights (V state dict or None)."""
    import episodes as E
    from shared.scene import IneligibleStart, ResetFailed

    if "env" not in _W:
        from libero_env import BranchingPickEnv

        _W.setdefault("cfg", job["cfg"])
        _W["env"] = BranchingPickEnv(_W["cfg"])
    net = None
    if job.get("video") and job.get("weights") is not None:
        import torch

        net = vnet.build_vnet("cnn", job["width"], job["heads"])
        net.load_state_dict({k: torch.from_numpy(v) for k, v in job["weights"].items()})
        net.eval()
    rng = np.random.default_rng(job["seed"])
    try:
        res = E.collect_rollout(_W["env"], job["start"], job["rc"], rng, net=net, video_path=job.get("video"))
    except (IneligibleStart, ResetFailed) as e:
        return {"job": job["id"], "skip": "{}: {}".format(type(e).__name__, e)}
    res["job"] = job["id"]
    return res


# ------------------------------------------------------------------ data -------------------------

class SampleSet(object):
    """Compact samples (obs fields + label + meta) of one split, with a fixed per-head bootstrap mask."""

    def __init__(self, heads, boot_p, rng):
        self.heads, self.boot_p, self.rng = heads, boot_p, rng
        self.obs = {f: [] for f in OBS}
        self.y, self.w, self.natural, self.boot = [], [], [], []

    def __len__(self):
        return len(self.y)

    def add(self, s, y, w):
        for f in OBS:
            self.obs[f].append(np.asarray(s[f]))
        self.y.append(float(y))
        self.w.append(float(w))
        self.natural.append(bool(s["natural"]))
        self.boot.append((self.rng.random(self.heads) < self.boot_p).astype(np.float32))

    def batch(self, idx):
        return {f: np.stack([self.obs[f][i] for i in idx]) for f in OBS}

    def arrays(self):
        return np.array(self.y, np.float32), np.array(self.natural, bool)


# ------------------------------------------------------------------ learner ------------------------

class Learner(object):
    def __init__(self, a):
        import torch

        self.torch, self.a = torch, a
        torch.manual_seed(0)
        self.dev = torch.device(a.device)
        self.net = vnet.build_vnet("cnn", a.width, a.heads).to(self.dev)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        self.rng = np.random.default_rng(0)
        self.steps, self.loss_ema = 0, None

    def _tensors(self, b):
        T = self.torch
        return vnet.batch_to_tensor(*(T.from_numpy(np.ascontiguousarray(b[f])).to(self.dev) for f in OBS))

    def train_steps(self, data, n):
        T = self.torch
        self.net.train()
        for _ in range(n):
            idx = self.rng.integers(len(data), size=self.a.batch)
            x, p = self._tensors(data.batch(idx))
            y = T.tensor([data.y[i] for i in idx], device=self.dev)[:, None]
            w = T.tensor(np.stack([data.boot[i] * data.w[i] for i in idx]), device=self.dev)
            v = self.net(x, p)
            loss = (w * (v - y) ** 2).sum() / w.sum().clamp(min=1e-6)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.steps += 1
            self.loss_ema = float(loss) if self.loss_ema is None else 0.98 * self.loss_ema + 0.02 * float(loss)

    def predict(self, data):
        """(N, heads) predictions for every sample of a SampleSet."""
        T = self.torch
        self.net.eval()
        out = []
        with T.no_grad():
            for s in range(0, len(data), 128):
                x, p = self._tensors(data.batch(range(s, min(s + 128, len(data)))))
                out.append(self.net(x, p).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.a.heads))

    def weights(self):
        return {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}

    def save(self, path, meta):
        vnet.save_ckpt(path, self.net, dict(meta, arch="cnn", width=self.a.width, heads=self.a.heads))

    def load(self, path):
        net, _ = vnet.load_ckpt(path, self.a.device)
        self.net.load_state_dict(net.state_dict())


def test_metrics(learner, ev, y_ref):
    """MSE / R2 / AUC / bias on the eval set: all, natural only, perturbed only."""
    if len(ev) < 2:
        return None
    q = learner.predict(ev)
    mu, sd = q.mean(1), q.std(1)
    y, nat = ev.arrays()
    out = {}
    for name, m in (("all", np.ones(len(y), bool)), ("natural", nat), ("perturbed", ~nat)):
        if m.sum() >= 2:
            r = metrics(mu[m], sd[m], y[m], y[m] > 0, y_ref)
            out[name] = {k: r[k] for k in ("n", "mse", "r2", "auc_success", "bias")}
    return out, mu, y


def val_mse(learner, val):
    if len(val) == 0:
        return float("nan")
    y, _ = val.arrays()
    return float(np.mean((learner.predict(val).mean(1) - y) ** 2))


# ------------------------------------------------------------------ plots --------------------------

def save_plots(out, mu, y, hist):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="0.7", lw=1, ls="--")
    cal = calibration(mu, y, y > 0)
    ax.plot([c["pred"] for c in cal], [c["y"] for c in cal], marker="o")
    for c in cal:
        ax.annotate(str(c["n"]), (c["pred"], c["y"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("predicted V (bin mean)")
    ax.set_ylabel("actual label (bin mean)")
    ax.set_title("V reliability on eval tasks (counts per bin)")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    r = [h["rollouts"] for h in hist]
    ax.plot(r, [h["val_mse"] for h in hist], label="val MSE (train tasks, held-out inits)")
    ax.plot(r, [h.get("test_mse", np.nan) for h in hist], label="test MSE (eval tasks)")
    ax.set_xlabel("rollouts")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------ main ---------------------------

def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(HERE / "runs/v3"))
    p.add_argument("--resume", action="store_true")
    # collection
    p.add_argument("--tasks", nargs="+", default=[TRAIN_TASKS], help="training tasks (held-out tasks refused)")
    p.add_argument("--eval-tasks", nargs="+", default=[EVAL_TASKS], help="unseen eval tasks (held-out split)")
    p.add_argument("--train-inits", default="0-39")
    p.add_argument("--eval-inits", default="0-39")
    p.add_argument("--tick-steps", type=int, default=4, help="control steps per tick (4 = 5 fps)")
    p.add_argument("--p-select", type=float, default=0.5, help="chance a gated tick is selected for branching")
    p.add_argument("--n-perturb", type=int, default=5)
    p.add_argument("--pos-cm", type=float, default=1.5, help="xy offset +- (cm)")
    p.add_argument("--dz-cm", type=float, nargs=2, default=[-0.5, 1.5], help="z offset range (cm)")
    p.add_argument("--rot-deg", type=float, default=10.0, help="+- about each gripper axis (deg)")
    p.add_argument("--move-cap", type=int, default=30, help="max control steps of the perturbation move")
    p.add_argument("--h-max-s", type=float, default=3.0, help="episode cap / label window H (s)")
    p.add_argument("--target-samples", type=int, default=10000, help="kept training-task samples (train + val)")
    p.add_argument("--eval-target-samples", type=int, default=4300, help="kept eval-task samples")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--hl", default="privileged", choices=["privileged", "none"])
    p.add_argument("--video-every", type=int, default=50, help="record every n-th training rollout (0 = never)")
    # learner
    p.add_argument("--gamma", type=float, default=0.85, help="time discount per second of the label")
    p.add_argument("--natural-weight", type=float, default=1.0, help="loss weight of natural vs perturbed samples")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--heads", type=int, default=5)
    p.add_argument("--bootstrap-p", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--utd", type=float, default=0.5, help="gradient steps per new training sample")
    p.add_argument("--warmup", type=int, default=300, help="training samples before the first update")
    p.add_argument("--val-frac", type=float, default=0.15, help="share of training (task, init) tags kept for val")
    p.add_argument("--eval-every", type=int, default=50, help="rollouts between val / test evaluations")
    p.add_argument("--device", default="cpu")
    # infra
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-mode", default="online")
    a = p.parse_args()
    a.mode, a.wandb_tags = "train", None
    return a


def main():
    a = parse()
    import torch
    from shared import runlog, scene as SC
    import episodes as E

    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
    out = pathlib.Path(a.out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    h_steps = int(round(a.h_max_s * CTRL_HZ))
    rc = {"tick_steps": a.tick_steps, "p_select": a.p_select, "n_perturb": a.n_perturb, "pos_cm": a.pos_cm,
          "dz_cm": a.dz_cm, "rot_deg": a.rot_deg, "move_cap": a.move_cap, "h_steps": h_steps, "gamma": a.gamma}
    cfg = E.env_cfg(a.hl, a.host, a.port)

    train_starts = SC.task_starts(a.tasks, SC.parse_ids(a.train_inits))                       # refuses held-out
    eval_starts = SC.task_starts(a.eval_tasks, SC.parse_ids(a.eval_inits), allow_held_out=True)
    clash = {s["task_id"] for s in train_starts} & {s["task_id"] for s in eval_starts}
    if clash:
        raise SystemExit("tasks {} are in both --tasks and --eval-tasks".format(sorted(clash)))
    print("training starts {} ({} tasks), eval starts {} ({} tasks)".format(
        len(train_starts), len({s["task_id"] for s in train_starts}), len(eval_starts),
        len({s["task_id"] for s in eval_starts})), flush=True)

    rng = np.random.default_rng(0)
    sets = {k: SampleSet(a.heads, a.bootstrap_p, np.random.default_rng(i)) for i, k in enumerate(("train", "val", "eval"))}
    learner = Learner(a)
    y_ref = [0.0]                          # constant baseline for R2: running mean of training labels
    n_rollouts, n_train_rollouts, shard_rows, n_shards, best_val, hist = 0, 0, [], 0, float("inf"), []

    def ingest(r):
        y = label(r["t_success"], h_steps, a.gamma)
        sets[r["split"]].add(r, y, a.natural_weight if r["natural"] else 1.0)

    if a.resume:
        rows = rows_of(read_shards(out / "data"))
        for r in rows:
            r["natural"], r["t_success"], r["split"] = bool(r["natural"]), int(r["t_success"]), str(r["split"])
            ingest(r)
        n_shards = len(list((out / "data").glob("shard_*.npz")))
        n_rollouts = int(max([float(r["rollout"]) for r in rows], default=-1)) + 1
        n_train_rollouts = len({float(r["rollout"]) for r in rows if r["split"] != "eval"})
        if (out / "v.pt").exists():
            learner.load(out / "v.pt")
        print("resumed: {} samples from {} shards, {} rollouts".format(len(rows), n_shards, n_rollouts), flush=True)

    runlog.wb_init(a, out, step_metric="rollouts")
    pool = runlog.ActorPool(run_job, a.workers, _init_worker, (cfg,))
    cycles = {"train": [], "eval": []}

    def next_start(kind):
        if not cycles[kind]:
            src = train_starts if kind == "train" else eval_starts
            cycles[kind] = [src[i] for i in rng.permutation(len(src))]
        return cycles[kind].pop()

    def n_kept():
        return len(sets["train"]) + len(sets["val"]), len(sets["eval"])

    bad = set()
    agg = {}
    yield_est = {"train": 20.0, "eval": 20.0}            # running mean of kept samples per rollout, per kind
    t0 = time.time()
    try:
        while True:
            n_tr, n_ev = n_kept()
            if n_tr >= a.target_samples and n_ev >= a.eval_target_samples:
                break
            jobs = []
            n_tr, n_ev = n_kept()
            pend = {"train": 0.0, "eval": 0.0}           # expected samples of the jobs already queued this round
            for w in range(max(1, a.workers)):
                tr, ev = n_tr + pend["train"], n_ev + pend["eval"]
                want_eval = ev < a.eval_target_samples and (tr >= a.target_samples or ev < 3.0 / 7 * (tr + ev + 1))
                kind = "eval" if want_eval else "train"
                pend[kind] += yield_est[kind]
                s = next_start(kind)
                while s["tag"] in bad:
                    s = next_start(kind)
                video = None
                if kind == "train":
                    n_train_rollouts += 1
                    if a.video_every and n_train_rollouts % a.video_every == 0:
                        video = str(out / "videos" / "rollout{:05d}_{}.mp4".format(n_rollouts + len(jobs), s["tag"]))
                jobs.append({"id": n_rollouts + len(jobs), "start": s, "kind": kind, "rc": rc, "cfg": cfg,
                             "seed": 7919 * (n_rollouts + len(jobs) + 1), "video": video,
                             "weights": learner.weights() if video else None, "width": a.width, "heads": a.heads})
            results = pool.run(jobs)
            n_new_train = 0
            for res in sorted(results, key=lambda r: r["job"]):
                job = jobs[res["job"] - jobs[0]["id"]]
                if "skip" in res:
                    bad.add(job["start"]["tag"])
                    print("  skip {}: {}".format(job["start"]["tag"], res["skip"]), flush=True)
                    continue
                tag = job["start"]["tag"]
                split = "eval" if job["kind"] == "eval" else split_of(tag, a.val_frac)
                for smp in res["samples"]:
                    smp.update(tag=tag, split=split, rollout=job["id"], init=job["start"]["init_idx"])
                    ingest(smp)
                    shard_rows.append(smp)
                    n_new_train += int(split == "train")
                yield_est[job["kind"]] = 0.8 * yield_est[job["kind"]] + 0.2 * len(res["samples"])
                st = res["stats"]
                for k in ("gated_ticks", "selected_ticks", "natural_run", "natural_success", "perturbed_run",
                          "perturbed_arrived", "perturbed_bumped", "perturbed_kept", "perturbed_success"):
                    agg[k] = agg.get(k, 0) + st[k]
                agg["rollouts"] = agg.get("rollouts", 0) + 1
                agg["main_picked"] = agg.get("main_picked", 0) + int(st["main_picked"])
            n_rollouts += len(jobs)
            if len(sets["train"]):
                y_ref[0] = float(np.mean(sets["train"].y))
            if len(shard_rows) >= 200:
                write_shard(out / "data" / "shard_{:04d}.npz".format(n_shards), shard_rows)
                n_shards, shard_rows = n_shards + 1, []
            if len(sets["train"]) >= a.warmup and n_new_train:
                learner.train_steps(sets["train"], int(np.ceil(a.utd * n_new_train)))
            n_tr, n_ev = n_kept()
            el = (time.time() - t0) / 3600.0
            print("rollouts {} | samples train {} val {} eval {} | grad {} loss {:.4f} | {:.2f} h".format(
                n_rollouts, len(sets["train"]), len(sets["val"]), n_ev, learner.steps, learner.loss_ema or float("nan"),
                el), flush=True)
            if n_rollouts // a.eval_every != (n_rollouts - len(jobs)) // a.eval_every:
                vm = val_mse(learner, sets["val"])
                tm = test_metrics(learner, sets["eval"], y_ref[0])
                rec = {"rollouts": n_rollouts, "val_mse": vm}
                if tm:
                    rec["test_mse"] = tm[0].get("all", {}).get("mse", np.nan)
                hist.append(rec)
                if np.isfinite(vm) and vm < best_val:
                    best_val = vm
                    learner.save(out / "v.pt", {"h_max_s": a.h_max_s, "gamma": a.gamma, "rollouts": n_rollouts,
                                                "val_mse": vm})
                R = max(1, agg.get("rollouts", 0))
                coll = {"samples_train": len(sets["train"]), "samples_val": len(sets["val"]), "samples_eval": n_ev,
                        "success_natural": agg.get("natural_success", 0) / max(1, agg.get("natural_run", 0)),
                        "success_perturbed": agg.get("perturbed_success", 0) / max(1, agg.get("perturbed_kept", 0)),
                        "gated_per_rollout": agg.get("gated_ticks", 0) / R,
                        "selected_per_rollout": agg.get("selected_ticks", 0) / R,
                        "arrival_rate": agg.get("perturbed_arrived", 0) / max(1, agg.get("perturbed_run", 0)),
                        "bump_rate": agg.get("perturbed_bumped", 0) / max(1, agg.get("perturbed_run", 0)),
                        "rollouts_per_hour": n_rollouts / max(el, 1e-6)}
                runlog.wb_log(coll, "collect", step=n_rollouts)
                runlog.wb_log({"loss": learner.loss_ema or float("nan"), "val_mse": vm,
                               "grad_steps": learner.steps}, "train", step=n_rollouts)
                if tm:
                    runlog.wb_log(tm[0], "test", step=n_rollouts)
                print("  [eval] val MSE {:.4f} (best {:.4f}) | test {} | natural success {:.2f} perturbed {:.2f} | "
                      "gated {:.1f} selected {:.1f} per rollout | arrival {:.2f} bump {:.2f}".format(
                          vm, best_val, {k: "MSE {:.3f} R2 {:.2f} AUC {:.2f}".format(v["mse"], v["r2"], v["auc_success"])
                                         for k, v in (tm[0] if tm else {}).items()},
                          coll["success_natural"], coll["success_perturbed"], coll["gated_per_rollout"],
                          coll["selected_per_rollout"], coll["arrival_rate"], coll["bump_rate"]), flush=True)
    finally:
        pool.close()
        if shard_rows:
            write_shard(out / "data" / "shard_{:04d}.npz".format(n_shards), shard_rows)

    # final: best checkpoint on the eval tasks
    if (out / "v.pt").exists():
        learner.load(out / "v.pt")
    tm = test_metrics(learner, sets["eval"], y_ref[0])
    if tm:
        res, mu, y = tm
        (out / "metrics.json").write_text(json.dumps(
            {"best_val_mse": best_val, "rollouts": n_rollouts, "samples": {k: len(v) for k, v in sets.items()},
             "test": res, "test_calibration": calibration(mu, y, y > 0)}, indent=2))
        save_plots(out, mu, y, hist)
        WB = runlog.wb()
        if WB is not None:
            WB.log({"rollouts": n_rollouts, "final/reliability": WB.Image(str(out / "reliability.png"))})
            runlog.wb_log(res, "final", step=n_rollouts)
        print("final (best v.pt) on eval tasks: {}".format(res), flush=True)
    runlog.wb_finish()


if __name__ == "__main__":
    main()
