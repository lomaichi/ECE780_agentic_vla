"""Fully convolutional pixel-wise Q-network for the pixel-Q corrector (pixel_Q_RL.md sec. 1-3).

Input: an 80x80 crop of the upright agentview image centred on the segmented object, as
  rgb (3, outside the mask greyed out), relative depth (1), mask (1), proprio tiled over the crop (6)
Output: Q(s, k, row, col) of shape (9, 10, 10): one map per rotation bin k, one entry per 8x8-pixel
window (row, col) of the crop. Three stride-2 convolutions take 80 -> 10, so every output cell lines
up with exactly one 8x8 window and the network is fully convolutional (as in VPG).

An action is the flat index a = k * 100 + row * 10 + col. Only cells on the object are valid
(obs["valid"], a (10, 10) bool array); the rotation bin is always free.

Python 3.8 / torch 1.11 compatible (the LIBERO client venv).
"""

import numpy as np

CROP = 80                      # crop side in pixels (of the 256x256 upright agentview)
CELL = 8                       # Q-map window side in pixels
GRID = CROP // CELL            # 10 x 10 cells
ROT_DEG = tuple(-20.0 + 5.0 * k for k in range(9))   # yaw offsets of the 9 rotation bins (deg)
N_ROT = len(ROT_DEG)
N_ACTIONS = N_ROT * GRID * GRID
PROPRIO_DIM = 6                # see corrector_env.proprio_features
IN_CH = 3 + 1 + 1 + PROPRIO_DIM


def encode_action(k, row, col):
    return int(k) * GRID * GRID + int(row) * GRID + int(col)


def decode_action(a):
    a = int(a)
    return a // (GRID * GRID), (a // GRID) % GRID, a % GRID


def valid_actions(valid):
    """(10, 10) bool cell mask -> (900,) bool action mask (same cells for every rotation bin)."""
    return np.tile(np.asarray(valid, dtype=bool).reshape(1, -1), (N_ROT, 1)).reshape(-1)


# ------------------------------------------------------------------ network ------------------------

def _torch():
    import torch

    return torch


def build_qnet(width=32):
    torch = _torch()
    nn = torch.nn

    def block(cin, cout, stride=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, cout),
            nn.ReLU(inplace=True),
        )

    class PixelQNet(nn.Module):
        def __init__(self, w):
            super(PixelQNet, self).__init__()
            self.encoder = nn.Sequential(
                block(IN_CH, w),                    # 80
                block(w, 2 * w, stride=2),          # 40
                block(2 * w, 2 * w),
                block(2 * w, 4 * w, stride=2),      # 20
                block(4 * w, 4 * w),
                block(4 * w, 4 * w, stride=2),      # 10: one cell per 8x8 window
                block(4 * w, 4 * w, dilation=2),    # receptive field grows to cover the whole crop
                block(4 * w, 4 * w, dilation=4),
            )
            self.head = nn.Sequential(              # per-cell features + global (pooled) context
                nn.Conv2d(8 * w, 4 * w, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(4 * w, N_ROT, 1),
            )

        def forward(self, x):
            f = self.encoder(x)
            g = f.mean(dim=(2, 3), keepdim=True).expand_as(f)
            return self.head(torch.cat([f, g], dim=1))   # (B, 9, 10, 10)

    return PixelQNet(width)


def obs_batch_to_tensor(rgb, depth, mask, proprio, device="cpu"):
    """Stacked observation arrays -> (B, IN_CH, 80, 80) float tensor.

    rgb (B, 80, 80, 3) uint8, depth (B, 80, 80) float16/32, mask (B, 80, 80) bool, proprio (B, 6).
    """
    torch = _torch()
    rgb = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).float().permute(0, 3, 1, 2) / 255.0 - 0.5
    depth = torch.from_numpy(np.asarray(depth, dtype=np.float32)).to(device).unsqueeze(1)
    mask = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(device).unsqueeze(1)
    p = torch.from_numpy(np.asarray(proprio, dtype=np.float32)).to(device)
    p = p[:, :, None, None].expand(-1, -1, CROP, CROP)
    return torch.cat([rgb, depth, mask, p], dim=1)


def obs_to_tensor(obs_list, device="cpu"):
    return obs_batch_to_tensor(np.stack([o["rgb"] for o in obs_list]), np.stack([o["depth"] for o in obs_list]),
                               np.stack([o["mask"] for o in obs_list]), np.stack([o["proprio"] for o in obs_list]),
                               device)


def q_map(net, obs, device="cpu"):
    """Q-maps (9, 10, 10) for one observation, as numpy."""
    torch = _torch()
    with torch.no_grad():
        return net(obs_to_tensor([obs], device))[0].cpu().numpy()


def masked_argmax(q, valid):
    """Best valid action of a (9, 10, 10) Q tensor. Returns the flat action index."""
    flat = np.where(valid_actions(valid), np.asarray(q).reshape(-1), -np.inf)
    return int(np.argmax(flat))


def select_action(net, obs, eps, rng, device="cpu"):
    """epsilon-greedy inside the mask (pixel_Q_RL.md step 1). Returns (action, qmap, greedy?)."""
    q = q_map(net, obs, device)
    if rng.random() < eps:
        cells = np.argwhere(obs["valid"])
        r, c = cells[rng.integers(len(cells))]
        return encode_action(rng.integers(N_ROT), r, c), q, False
    return masked_argmax(q, obs["valid"]), q, True


def state_dict_numpy(net):
    """CPU numpy copy of the weights (cheap to pickle to actor processes)."""
    return {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}


def load_state_dict_numpy(net, sd):
    torch = _torch()
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in sd.items()})
