"""V4: offline training of the value network V(s) (vnet.py) from the shards collect.py wrote.

Data: a folder of shards (collect.py's OUT_DIR/data, data_io.py). The split stored with the samples is used:
train / val (training tasks; val picks the best epoch) / test = eval (the held-out eval tasks).
Labels are recomputed from the raw outcome for the window --h-max-s (<= the episode cap the shards were collected
with, collect.py --h-max-s, 5 s by default) and --gamma:

    y = gamma^(t / 20)  if the pick happened t <= H control steps after takeover, else 0

Models: --arch (attn by default; cnn) and, unless --no-baselines, the proprio-only MLP and the constant train mean,
so the report shows how much the heightmap adds.

Can run while collect.py is still writing shards: it trains on the shards present when it starts.

Outputs in --out-dir:
  v_<arch>.pt          checkpoint (vnet.load_ckpt)
  metrics.json         test / val metrics per model, test breakdowns (natural / perturbed, tilt bin), calibration
  reliability.png      binned predicted V vs actual label on test
  curves.png           train / val MSE per epoch
  config_train_v.json

Usage (Vlearning_code_instruction.md sec. 4.2):
  PY=/opt/simplevla/.venv/bin/python        # torch 2.7 + cu128: runs on the RTX 5090
  $PY train_v.py --data runs/v4/data --out-dir runs/v4_train --device cuda
"""

import argparse
import collections
import json
import pathlib
import time

import numpy as np

import data_io
import vnet

CTRL_HZ = 20
TILT_BINS = ((0, 10), (10, 20), (20, 30), (30, 1e9))


# ------------------------------------------------------------------ data ---------------------------

def relabel(d, h_max_s, gamma):
    """-> (y, success, known): labels for window h_max_s, pick-within-H flags, and the non-censored mask."""
    h = int(round(h_max_s * CTRL_HZ))
    t, cap, end = d["t_success"], d["cap_steps"], d["end"]
    picked = t >= 0
    known = picked | (end != "window") | (cap >= h)
    success = picked & (t <= h)
    y = np.where(success, gamma ** (np.maximum(t, 0) / float(CTRL_HZ)), 0.0).astype(np.float32)
    return y, success, known


# ------------------------------------------------------------------ metrics ------------------------

def rankdata(x):
    from scipy.stats import rankdata as rd

    return rd(x)


def auc(score, positive):
    """Mann-Whitney AUC of score for positive vs negative (nan if one class is empty)."""
    pos, n_pos = np.asarray(positive, bool), int(np.sum(positive))
    n_neg = len(pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def metrics(mu, sd, y, success, y_train_mean):
    """mu / sd: ensemble mean / std per sample. R2 is against the constant train mean."""
    if len(y) == 0:
        return {"n": 0}
    mse = float(np.mean((mu - y) ** 2))
    mse_const = float(np.mean((y_train_mean - y) ** 2))
    err = np.abs(mu - y)
    return {"n": int(len(y)), "mse": mse, "mse_const": mse_const, "r2": 1.0 - mse / mse_const if mse_const > 0 else float("nan"),
            "mae": float(err.mean()), "bias": float(np.mean(mu - y)), "auc_success": auc(mu, success),
            "spearman": spearman(mu, y), "y_mean": float(y.mean()), "pred_mean": float(mu.mean()),
            "sd_mean": float(sd.mean()), "corr_sd_abs_err": float(np.corrcoef(sd, err)[0, 1]) if sd.std() > 0 else float("nan")}


def calibration(mu, y, success, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mu >= lo) & ((mu < hi) | (hi == 1.0))
        if m.any():
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "pred": float(mu[m].mean()),
                        "y": float(y[m].mean()), "success": float(success[m].mean())})
    return out


def tilt_bin(t):
    for k, (lo, hi) in enumerate(TILT_BINS):
        if lo <= t < hi:
            return k
    return len(TILT_BINS) - 1


# ------------------------------------------------------------------ training -----------------------

def train_model(arch, a, D, y, idx, device, sw=None):
    """Train one ensemble; returns (net, best-epoch val MSE, history)."""
    import torch

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    net = vnet.build_vnet(arch, a.width, a.heads).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    yt = torch.from_numpy(y).to(device)
    tr, va = idx["train"], idx["val"]
    # per (sample, head) bootstrap mask, fixed for the run: heads see different subsets -> ensemble spread
    boot = (rng.random((len(y), a.heads)) < a.bootstrap_p).astype(np.float32)
    if sw is not None:                                       # per-sample loss weight (--natural-weight)
        boot = boot * np.asarray(sw, np.float32)[:, None]
    boot = torch.from_numpy(boot).to(device)

    def forward(ii):
        x, p = vnet.batch_to_tensor(*(D[k][ii] for k in vnet.OBS_FIELDS))
        if a.flip_aug and net.training:                     # optional: mirror across the finger axis
            flip = torch.rand(len(ii), device=device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        return net(x, p)

    def predict(ii):
        net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(ii), 256):
                out.append(forward(ii[s:s + 256]))
        net.train()
        return torch.cat(out).cpu().numpy()

    best, best_state, hist = float("inf"), None, []
    tr_t = torch.from_numpy(tr).to(device)
    steps_per_epoch = max(1, len(tr) // a.batch)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t), device=device)]
        tot = 0.0
        for s in range(steps_per_epoch):
            ii = perm[s * a.batch:(s + 1) * a.batch]
            v = forward(ii)
            w = boot[ii]
            loss = (w * (v - yt[ii, None]) ** 2).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        pv = predict(va).mean(axis=1)
        val_mse = float(np.mean((pv - y[va]) ** 2))
        rec = {"model": arch, "epoch": ep, "train_mse": tot / steps_per_epoch, "val_mse": val_mse,
               "elapsed_s": round(time.time() - t0, 1)}
        hist.append(rec)
        if val_mse < best:
            best = val_mse
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            rec["best"] = True
        if ep == 1 or ep % a.print_every == 0 or ep == a.epochs:
            print("  [{}] epoch {:3d}  train {:.4f}  val {:.4f}  best {:.4f}  {:.0f}s".format(
                arch, ep, rec["train_mse"], val_mse, best, time.time() - t0), flush=True)
    net.load_state_dict(best_state)
    net.eval()
    return net, best, hist, predict


# ------------------------------------------------------------------ plots --------------------------

def plot_reliability(path, curves, n_bins=10):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="0.7", lw=1, ls="--")
    for name, (mu, y) in curves.items():
        cal = calibration(mu, y, y > 0, n_bins)
        ax.plot([c["pred"] for c in cal], [c["y"] for c in cal], marker="o", label=name)
        for c in cal:
            ax.annotate(str(c["n"]), (c["pred"], c["y"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("predicted V (bin mean)")
    ax.set_ylabel("actual no-op label (bin mean)")
    ax.set_title("V reliability on test (counts per bin)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(path, hists):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, h in hists.items():
        e = [r["epoch"] for r in h]
        l = ax.plot(e, [r["val_mse"] for r in h], label=name + " val")[0]
        ax.plot(e, [r["train_mse"] for r in h], color=l.get_color(), ls=":", label=name + " train")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ main ---------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="runs/v4/data", help="shard folder (collect.py OUT_DIR/data)")
    p.add_argument("--out-dir", default="runs/v4_train")
    p.add_argument("--h-max-s", type=float, default=5.0,
                   help="label window H (s), at most the episode cap of the shards (collect.py --h-max-s)")
    p.add_argument("--gamma", type=float, default=0.85, help="discount per second")
    p.add_argument("--natural-weight", type=float, default=1.0, help="loss weight of natural vs perturbed samples")
    p.add_argument("--arch", default="attn", choices=["attn", "cnn", "proprio"])
    p.add_argument("--no-baselines", action="store_true", help="skip the proprio-only baseline")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--heads", type=int, default=5)
    p.add_argument("--bootstrap-p", type=float, default=0.8, help="chance a sample trains a given head")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--flip-aug", action="store_true", help="random mirror across the finger axis (assumes symmetry)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--print-every", type=int, default=10)
    a = p.parse_args()
    a.seed = 0

    import torch

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_train_v.json").write_text(json.dumps(vars(a), indent=2))
    raw = data_io.read_shards(a.data)
    if not raw:
        raise SystemExit("no shards in {}".format(a.data))
    if a.h_max_s * CTRL_HZ > raw["cap_steps"].min():
        raise SystemExit("--h-max-s {} exceeds the episode cap ({} s)".format(a.h_max_s, raw["cap_steps"].min() / CTRL_HZ))
    y, success, _ = relabel(raw, a.h_max_s, a.gamma)
    sel = raw
    groups = sel["tag"]
    print("data {}: {} samples ({} natural, {} perturbed); H = {:.1f} s, gamma {}".format(
        a.data, len(y), sel["natural"].sum(), (~sel["natural"]).sum(), a.h_max_s, a.gamma))
    sp = sel["split"].astype(str)
    masks = {"train": sp == "train", "val": sp == "val", "test": sp == "eval"}
    idx = {k: np.flatnonzero(m) for k, m in masks.items()}
    for k, ii in idx.items():
        print("  {:5s}: n {:4d}  groups {:3d}  y mean {:.3f}  success {:.3f}".format(
            k, len(ii), len(set(groups[ii].tolist())), y[ii].mean(), success[ii].mean()))

    device = torch.device(a.device)
    D = {k: torch.from_numpy(np.ascontiguousarray(sel[k])).to(device) for k in vnet.OBS_FIELDS}
    y_tr_mean = float(y[idx["train"]].mean())

    archs = [a.arch] + ([] if a.no_baselines or a.arch == "proprio" else ["proprio"])
    results, hists, preds = {}, {}, {}
    for arch in archs:
        print("training {} ...".format(arch), flush=True)
        sw = np.where(sel["natural"], a.natural_weight, 1.0)
        net, best, hist, predict = train_model(arch, a, D, y, idx, device, sw)
        hists[arch] = hist
        vnet.save_ckpt(out / "v_{}.pt".format(arch), net,
                       {"arch": arch, "width": a.width, "heads": a.heads, "h_max_s": a.h_max_s, "gamma": a.gamma,
                        "best_val_mse": best, "data": a.data})
        res = {}
        for split in ("val", "test"):
            q = predict(idx[split])
            preds[(arch, split)] = q
            res[split] = metrics(q.mean(1), q.std(1), y[idx[split]], success[idx[split]], y_tr_mean)
        results[arch] = res
    results["constant"] = {s: metrics(np.full(len(idx[s]), y_tr_mean), np.zeros(len(idx[s])), y[idx[s]],
                                      success[idx[s]], y_tr_mean) for s in ("val", "test")}

    # breakdowns on test for the main model
    te = idx["test"]
    mu = preds[(a.arch, "test")].mean(1)
    br = collections.OrderedDict()
    split_key = ("type", np.where(sel["natural"][te], 1, 0))
    for name, key in (split_key, ("tilt_bin", np.array([tilt_bin(t) for t in sel["tilt_deg"][te]]))):
        br[name] = {}
        for v in sorted(set(key.tolist())):
            m = key == v
            label = "{}-{}".format(*TILT_BINS[v]) if name == "tilt_bin" else \
                {0: "perturbed", 1: "natural"}[v]
            br[name][label.replace("-1000000000.0", "+")] = metrics(mu[m], np.zeros(m.sum()), y[te][m], success[te][m], y_tr_mean)

    report = {"config": vars(a), "n_samples": int(len(y)),
              "split_sizes": {k: int(len(v)) for k, v in idx.items()}, "y_train_mean": y_tr_mean,
              "models": results, "test_breakdown_{}".format(a.arch): br,
              "test_calibration_{}".format(a.arch): calibration(mu, y[te], success[te])}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    plot_reliability(out / "reliability.png", {k: (preds[(k, "test")].mean(1), y[te]) for k in archs})
    plot_curves(out / "curves.png", hists)

    print("\ntest set = eval tasks ({} samples):".format(len(te)))
    print("  {:9s} {:>7s} {:>7s} {:>7s} {:>8s} {:>8s} {:>7s} {:>9s}".format(
        "model", "MSE", "R2", "bias", "AUC", "spearman", "sd", "corr(sd,|e|)"))
    for k, r in results.items():
        t = r["test"]
        print("  {:9s} {:7.4f} {:7.3f} {:+7.3f} {:8.3f} {:8.3f} {:7.3f} {:9.3f}".format(
            k, t["mse"], t["r2"], t["bias"], t["auc_success"], t["spearman"], t["sd_mean"], t["corr_sd_abs_err"]))
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
