"""Run logging and actor processes shared by V1 train_dqn.py and V2 train.py.

  log_jsonl(path, rec)          append one JSON line
  wb_init / wb_log / wb_finish  Weights & Biases (off unless --wandb-project is given). Training runs keep their
                                W&B id in OUT_DIR/wandb_id.txt so --resume continues the same run; other modes
                                are separate runs grouped under OUT_DIR's name. wb() is the wandb module while a
                                run is open, else None.
  ActorPool                     spawn process pool running one job function (in-process when workers = 0)

wandb 0.13.1 (LIBERO's pin) asks the server about lists of images even in offline mode: log images one key at
a time.
"""

import json
import multiprocessing as mp

import numpy as np


def log_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------ wandb --------------------------

_WB = None            # the wandb module while a run is open (main process only)
_STEP = "episodes"    # x-axis metric of the open run


def wb():
    return _WB


def wb_init(a, out, step_metric="episodes", summary_max=(), prefixes=("train/*", "eval/*")):
    """Open a W&B run when a.wandb_project is set (a: argparse namespace with mode, resume, wandb_* fields)."""
    global _WB, _STEP
    if not a.wandb_project:
        return
    import wandb

    run_id, id_file = None, out / "wandb_id.txt"
    if a.mode == "train":
        run_id = id_file.read_text().strip() if a.resume and id_file.exists() else wandb.util.generate_id()
        id_file.write_text(run_id)
    name = a.wandb_name or (out.name if a.mode == "train" else "{}-{}".format(out.name, a.mode))
    wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=name, group=out.name, job_type=a.mode,
               tags=a.wandb_tags, config=vars(a), dir=str(out), id=run_id, resume="allow" if run_id else None,
               mode=a.wandb_mode)
    wandb.define_metric(step_metric)
    for prefix in prefixes:
        wandb.define_metric(prefix, step_metric=step_metric)
    for key in summary_max:
        wandb.define_metric(key, step_metric=step_metric, summary="max")
    _WB, _STEP = wandb, step_metric


def wb_log(rec, prefix, step=None, extra=None):
    """Log the numeric fields of rec (nested dicts flattened to prefix/key/sub) plus extra, at x = step."""
    if _WB is None:
        return
    d = {}

    def flat(p, v):
        if isinstance(v, dict):
            for k, x in v.items():
                flat("{}/{}".format(p, k), x)
        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            d[p] = float(v)

    for k, v in rec.items():
        flat("{}/{}".format(prefix, k), v)
    d.update(extra or {})
    if step is not None:
        d[_STEP] = step
    _WB.log(d)


def wb_finish():
    global _WB
    if _WB is not None:
        _WB.finish()
        _WB = None


# ------------------------------------------------------------------ actors -------------------------

class ActorPool(object):
    """workers = 0 runs the jobs in this process (toy envs, debugging); otherwise a spawn pool whose workers run
    init_fn(*init_args) once. job_fn must be a module-level function (it is pickled by reference)."""

    def __init__(self, job_fn, workers, init_fn=None, init_args=()):
        self.job_fn, self.pool = job_fn, None
        if workers > 0:
            self.pool = mp.get_context("spawn").Pool(workers, initializer=init_fn, initargs=init_args)

    def run(self, jobs):
        if self.pool is None:
            return [self.job_fn(j) for j in jobs]
        return list(self.pool.imap_unordered(self.job_fn, jobs, chunksize=1))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
