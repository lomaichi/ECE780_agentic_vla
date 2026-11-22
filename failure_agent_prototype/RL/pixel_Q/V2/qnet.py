"""V2 Q-map: gripper-centred heightmap in, one Q entry per (rotation bin, cell) out, K ensemble heads.

Input (pixel_Q_RL_V2.md sec. 5): an 88x88 gripper-centred, yaw-aligned heightmap at 2.5 mm/px (22 cm) with
  rgb (3), height (1), target mask (1), observed (1)  +  proprio (4) tiled over the map  = 10 channels.
Output: (K, 9, 11, 11) = K ensemble heads x 9 rotation bins x 11 x 11 cells of 8x8 px (2 cm). The FCN's stride
of 8 maps 88 -> 11; the grasp point sits at the centre of cell (C, C) = (5, 5).

Actions (sec. 6): a = (k, i, j) with rotation bin k (yaw offset ROT_DEG[k]; K0 = 4 is 0 deg), cell row i, cell
column j; flat index a = k * 121 + i * 11 + j. The no-op A0 = (K0, C, C) is the centre cell at 0 deg; its value
Q0(s) = Q(s, A0) is the value of letting pi0 continue. Valid actions are the cells within Chebyshev reach r of
the centre (obs["valid"], an (11, 11) bool mask), every rotation bin. Corrections = valid actions minus A0.

Trigger score (sec. 8): with mu / sigma the ensemble mean / std (over heads),
    A(s) = max_{a in corrections} [mu_a - kappa * sigma_a] - mu_A0.

Python 3.8 / torch 1.11 compatible (the LIBERO client venv).
"""

import pathlib
import sys

import numpy as np

_PIXQ = str(pathlib.Path(__file__).resolve().parents[1])   # pixel_Q/: the shared package
if _PIXQ not in sys.path:
    sys.path.insert(0, _PIXQ)

HM = 88                        # heightmap side (px)
RES_M = 0.0025                 # m per heightmap pixel
CELL = 8                       # cell side (px) = 2 cm
CELL_M = CELL * RES_M
GRID = HM // CELL              # 11 x 11 cells
C = GRID // 2                  # centre cell index (5): the grasp point
ROT_DEG = tuple(-20.0 + 5.0 * k for k in range(9))   # yaw offsets of the 9 rotation bins (deg)
N_ROT = len(ROT_DEG)
K0 = N_ROT // 2                # the 0 deg bin
N_ACTIONS = N_ROT * GRID * GRID
MAP_CH = 6                     # rgb 3 + height + mask + observed
PROPRIO_DIM = 4                # gripper opening, tilt sin, tilt cos, height above the target top
IN_CH = MAP_CH + PROPRIO_DIM


def encode_action(k, i, j):
    return int(k) * GRID * GRID + int(i) * GRID + int(j)


def decode_action(a):
    a = int(a)
    return a // (GRID * GRID), (a // GRID) % GRID, a % GRID


A0 = encode_action(K0, C, C)   # the pseudo no-op


def reach_mask(r):
    """(11, 11) bool: cells within Chebyshev distance r (cells) of the centre cell."""
    ii, jj = np.mgrid[0:GRID, 0:GRID]
    return np.maximum(np.abs(ii - C), np.abs(jj - C)) <= int(r)


def valid_actions(valid):
    """(11, 11) bool cell mask -> (1089,) bool action mask (same cells for every rotation bin)."""
    return np.tile(np.asarray(valid, dtype=bool).reshape(1, -1), (N_ROT, 1)).reshape(-1)


def correction_actions(valid):
    """Valid actions without the no-op A0 (the trigger and the intervention branches choose among these)."""
    m = valid_actions(valid)
    m[A0] = False
    return m


# ------------------------------------------------------------------ network ------------------------

def _torch():
    import torch

    return torch


def build_qnet(width=32, heads=5):
    """Shared trunk (../shared/fcn.py, as V1) + `heads` independent 1x1 heads. forward(x): (B, heads, 9, 11, 11)."""
    torch = _torch()
    nn = torch.nn
    from shared import fcn

    class EnsembleQNet(nn.Module):
        def __init__(self, w, k):
            super(EnsembleQNet, self).__init__()
            self.encoder = fcn.encoder(IN_CH, w)                        # 88 -> 11
            self.heads = nn.ModuleList([fcn.head(w, N_ROT) for _ in range(k)])
            self.n_heads = k

        def forward(self, x):
            f = fcn.with_context(self.encoder(x))
            return torch.stack([h(f) for h in self.heads], dim=1)

    return EnsembleQNet(width, heads)


def obs_batch_to_tensor(rgb, height, mask, observed, proprio, device="cpu"):
    """Stacked observation arrays -> (B, IN_CH, 88, 88) float tensor.

    rgb (B, 88, 88, 3) uint8, height (B, 88, 88) float16/32, mask / observed (B, 88, 88) bool, proprio (B, 4)."""
    torch = _torch()
    rgb = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).float().permute(0, 3, 1, 2) / 255.0 - 0.5
    height = torch.from_numpy(np.asarray(height, dtype=np.float32)).to(device).unsqueeze(1)
    mask = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(device).unsqueeze(1)
    observed = torch.from_numpy(np.asarray(observed, dtype=np.float32)).to(device).unsqueeze(1)
    p = torch.from_numpy(np.asarray(proprio, dtype=np.float32)).to(device)
    p = p[:, :, None, None].expand(-1, -1, height.shape[2], height.shape[3])
    return torch.cat([rgb, height, mask, observed, p], dim=1)


def obs_to_tensor(obs_list, device="cpu"):
    st = lambda k: np.stack([o[k] for o in obs_list])  # noqa: E731
    return obs_batch_to_tensor(st("rgb"), st("height"), st("mask"), st("observed"), st("proprio"), device)


def ensemble_q(net, obs, device=None):
    """All heads' Q-maps for one observation: (K, 9, 11, 11) numpy. device: default the network's."""
    torch = _torch()
    if device is None:
        device = next(net.parameters()).device
    with torch.no_grad():
        return net(obs_to_tensor([obs], device))[0].cpu().numpy()


def ensemble_stats(q):
    """(K, 9, 11, 11) -> mean, std over heads (population std; 0 for a single head)."""
    q = np.asarray(q, dtype=np.float64)
    return q.mean(axis=0), q.std(axis=0)


def score(q, valid, kappa):
    """Trigger quantities from one observation's ensemble Q (K, 9, 11, 11) and its (11, 11) valid mask.

    Returns dict: A (advantage of the best penalized correction over Q0), a_best (flat action), q0 (mean),
    best (mu - kappa sigma of a_best), mu_best, sigma_best, sigma_q0, a_mean (argmax of mu alone)."""
    mu, sd = ensemble_stats(q)
    corr = correction_actions(valid)
    pen = np.where(corr, (mu - kappa * sd).reshape(-1), -np.inf)
    a_best = int(np.argmax(pen))
    a_mean = int(np.argmax(np.where(corr, mu.reshape(-1), -np.inf)))
    q0 = float(mu.reshape(-1)[A0])
    return {"A": float(pen[a_best] - q0), "a_best": a_best, "q0": q0, "best": float(pen[a_best]),
            "mu_best": float(mu.reshape(-1)[a_best]), "sigma_best": float(sd.reshape(-1)[a_best]),
            "sigma_q0": float(sd.reshape(-1)[A0]), "a_mean": a_mean}


def random_correction(valid, rng):
    idx = np.flatnonzero(correction_actions(valid))
    return int(idx[rng.integers(len(idx))])


def state_dict_numpy(net):
    """CPU numpy copy of the weights (cheap to pickle to actor processes)."""
    return {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}


def load_state_dict_numpy(net, sd):
    torch = _torch()
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in sd.items()})


class Scorer(object):
    """Q-map inference in an actor: weights from the learner, trigger scores, epsilon-greedy branch actions."""

    def __init__(self, width, heads, kappa):
        self.net = build_qnet(width, heads).eval()
        self.kappa = float(kappa)
        self.loaded = False

    def load(self, weights):
        if weights is not None:
            load_state_dict_numpy(self.net, weights)
            self.loaded = True

    def q(self, obs):
        return ensemble_q(self.net, obs)

    def score(self, obs, kappa=None):
        q = self.q(obs)
        s = score(q, obs["valid"], self.kappa if kappa is None else kappa)
        s["q"] = q
        return s

    def branch_action(self, obs, eps, rng, kappa=0.0, scored=None):
        """Correction for an intervention branch: uniform over corrections with probability eps, else the argmax
        of mu - kappa * sigma (kappa 0 in training: greedy on the ensemble mean). Returns (action, greedy?)."""
        if rng.random() < eps or not self.loaded:
            return random_correction(obs["valid"], rng), False
        if scored is not None and kappa == 0.0:
            return int(scored["a_mean"]), True
        s = scored if scored is not None and kappa == self.kappa else self.score(obs, kappa)
        return int(s["a_best"]), True
