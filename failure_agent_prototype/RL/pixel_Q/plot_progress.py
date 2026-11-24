"""Plot train_dqn.py progress from OUT_DIR/train.jsonl and OUT_DIR/eval.jsonl into OUT_DIR/progress.png.

  python3 plot_progress.py OUT_DIR            # one-off
  python3 plot_progress.py OUT_DIR --watch 30 # re-render every 30 s while training runs
"""

import argparse
import json
import math
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:  # partially written last line
            pass
    return rows


def col(rows, key, sub=None):
    x, y = [], []
    for r in rows:
        v = r.get(key) if sub is None else (r.get(key) or {}).get(sub)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            x.append(r.get("episodes", 0))
            y.append(v)
    return np.array(x), np.array(y, dtype=float)


def smooth(y, k):
    if len(y) < k or k <= 1:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def render(out, k):
    tr, ev = load(out / "train.jsonl"), load(out / "eval.jsonl")
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    for key, lab in [("episode_success", "episode"), ("first_success", "first attempt"), ("chain_success", "chain")]:
        x, y = col(tr, key)
        if len(y):
            ax[0, 0].plot(x[len(x) - len(smooth(y, k)):], smooth(y, k), label=lab)
    x, y = col(tr, "buffer", "reward_rate")
    ax[0, 0].plot(x, y, "k--", lw=1, label="buffer reward rate")
    ax[0, 0].set_title("training success (rolling {} rounds)".format(k))

    x, y = col(tr, "loss")
    ax[0, 1].plot(x, y, alpha=0.3)
    ax[0, 1].plot(x[len(x) - len(smooth(y, k)):], smooth(y, k), label="TD loss")
    ax[0, 1].set_title("loss")

    for key in ["q_sa", "target"]:
        x, y = col(tr, key)
        ax[1, 0].plot(x[len(x) - len(smooth(y, k)):], smooth(y, k), label=key)
    x, y = col(tr, "eps")
    ax[1, 0].plot(x, y, ":", label="epsilon")
    ax[1, 0].set_title("Q(s,a) vs target, epsilon")

    ev = [r for r in ev if r.get("mode") in (None, "train", "eval") and str(r.get("label", "")).startswith("eval_ep")]
    ex = [int(r["label"][len("eval_ep"):]) for r in ev]
    for key, lab in [("episode_success", "episode"), ("first_success", "first attempt"), ("chain_success", "chain")]:
        ax[1, 1].plot(ex, [r.get(key) for r in ev], "o-", label=lab)
    ax[1, 1].set_title("held-out eval (greedy)")

    for a in ax.flat:
        a.set_xlabel("training episodes")
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    last = tr[-1] if tr else {}
    fig.suptitle("{}  |  round {}  ep {}  {:.1f} min".format(out.name, last.get("round", 0), last.get("episodes", 0),
                                                         last.get("elapsed_min", 0)))
    fig.tight_layout()
    fig.savefig(out / "progress.png", dpi=90)
    plt.close(fig)
    print("wrote {} ({} train rounds, {} evals)".format(out / "progress.png", len(tr), len(ev)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--smooth", type=int, default=10, help="rolling window in rounds")
    p.add_argument("--watch", type=float, default=0, help="re-render every N seconds")
    a = p.parse_args()
    out = pathlib.Path(a.out_dir)
    while True:
        render(out, a.smooth)
        if not a.watch:
            break
        time.sleep(a.watch)
