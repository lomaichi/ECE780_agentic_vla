"""V3 no-op value network V(s): the value of letting pi0 continue from s (discounted pick success within H).

Replaces V2's Q0 = Q(s, A0) entry of the Q-map (pixel_Q_RL_V2.md sec. 15). The trigger becomes
    A(s) = max_{a in corrections} [mu_a(s) - kappa * sigma_a(s)] - V(s)
with every cell (the centre cell at 0 deg included) an ordinary, executable correction.

Input: V2's observation, unchanged: 88x88 gripper-centred, yaw-aligned heightmap (rgb 3, height 1, target mask 1,
observed 1) + proprio (4) tiled over the map = 10 channels. proprio = [finger gap, sin tilt, cos tilt, height of
the grasp point above the target's top].

Architectures (build_vnet(arch=...)):
  cnn       shared FCN encoder (../shared/fcn.py, as the Q-map) -> global mean + max pool -> concat proprio ->
            K independent MLP heads -> sigmoid. K heads = ensemble (mean, std), trained on bootstrap masks.
  attn      the same encoder, but attention pooling instead of mean + max: the 11x11 cell features get a learned
            2D position embedding (the map is gripper-centred, so a cell's position relative to the grasp point
            matters and mean / max pooling throws it away), then ATTN_MAPS learned softmax maps over the 121
            cells each pool a feature vector -> concat proprio -> K MLP heads. The maps are shared by the heads;
            net.attention(x) returns them (B, ATTN_MAPS, 11, 11) for plotting.
  proprio   K MLP heads on proprio alone: the baseline that says how much the map adds.

Output: (B, K) values in [0, 1].
"""

import pathlib
import sys

import numpy as np

_PIXQ = str(pathlib.Path(__file__).resolve().parents[1])   # pixel_Q/: the shared package
if _PIXQ not in sys.path:
    sys.path.insert(0, _PIXQ)

HM = 88
MAP_CH = 6
PROPRIO_DIM = 4
IN_CH = MAP_CH + PROPRIO_DIM
OBS_FIELDS = ("rgb", "height", "mask", "observed", "proprio")
ATTN_MAPS = 4


def _torch():
    import torch

    return torch


def build_vnet(arch="cnn", width=32, heads=5):
    torch = _torch()
    nn = torch.nn
    from shared import fcn

    def mlp_head(cin, hidden):
        return nn.Sequential(nn.Linear(cin, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                             nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    class CnnV(nn.Module):
        def __init__(self, w, k):
            super(CnnV, self).__init__()
            self.encoder = fcn.encoder(IN_CH, w)                   # 88 -> 11, 4w channels
            self.heads = nn.ModuleList([mlp_head(8 * w + PROPRIO_DIM, 2 * w) for _ in range(k)])
            self.n_heads = k

        def forward(self, x, proprio):
            f = self.encoder(x)
            z = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3)), proprio], dim=1)
            return torch.sigmoid(torch.cat([h(z) for h in self.heads], dim=1))

    class AttnV(nn.Module):
        def __init__(self, w, k, m=ATTN_MAPS):
            super(AttnV, self).__init__()
            g = HM // 8
            self.encoder = fcn.encoder(IN_CH, w)                   # 88 -> 11, 4w channels
            self.pos = nn.Parameter(0.02 * torch.randn(1, 4 * w, g, g))
            self.logits = nn.Conv2d(4 * w, m, 1)
            self.heads = nn.ModuleList([mlp_head(m * 4 * w + PROPRIO_DIM, 2 * w) for _ in range(k)])
            self.n_heads = k

        def _pool(self, x):
            f = self.encoder(x) + self.pos                          # (B, C, g, g)
            a = torch.softmax(self.logits(f).flatten(2), dim=2)     # (B, m, g*g)
            return torch.einsum("bmn,bcn->bmc", a, f.flatten(2)).flatten(1), a.view(a.shape[0], a.shape[1],
                                                                                    *f.shape[2:])

        def forward(self, x, proprio):
            z = torch.cat([self._pool(x)[0], proprio], dim=1)
            return torch.sigmoid(torch.cat([h(z) for h in self.heads], dim=1))

        def attention(self, x):
            return self._pool(x)[1]

    class ProprioV(nn.Module):
        def __init__(self, w, k):
            super(ProprioV, self).__init__()
            self.heads = nn.ModuleList([mlp_head(PROPRIO_DIM, 2 * w) for _ in range(k)])
            self.n_heads = k

        def forward(self, x, proprio):
            return torch.sigmoid(torch.cat([h(proprio) for h in self.heads], dim=1))

    if arch == "cnn":
        return CnnV(width, heads)
    if arch == "attn":
        return AttnV(width, heads)
    if arch == "proprio":
        return ProprioV(width, heads)
    raise ValueError("unknown arch {!r}".format(arch))


def batch_to_tensor(rgb, height, mask, observed, proprio):
    """Compact torch tensors already on the device -> (x (B, 10, 88, 88) float, proprio (B, 4) float).

    rgb (B, 88, 88, 3) uint8, height (B, 88, 88) float16, mask / observed (B, 88, 88) bool, proprio (B, 4)."""
    torch = _torch()
    rgb = rgb.float().permute(0, 3, 1, 2) / 255.0 - 0.5
    p = proprio.float()
    pt = p[:, :, None, None].expand(-1, -1, height.shape[1], height.shape[2])
    x = torch.cat([rgb, height.float().unsqueeze(1), mask.float().unsqueeze(1), observed.float().unsqueeze(1), pt], 1)
    return x, p


def obs_to_tensor(obs_list, device="cpu"):
    """List of V2 observation dicts -> network input on device."""
    torch = _torch()
    st = lambda k: torch.from_numpy(np.ascontiguousarray(np.stack([o[k] for o in obs_list]))).to(device)  # noqa: E731
    return batch_to_tensor(*(st(k) for k in OBS_FIELDS))


def predict(net, obs, device=None):
    """One observation -> (mean, std) over the ensemble heads."""
    torch = _torch()
    if device is None:
        device = next(net.parameters()).device
    with torch.no_grad():
        v = net(*obs_to_tensor([obs], device))[0].cpu().numpy()
    return float(v.mean()), float(v.std())


def save_ckpt(path, net, meta):
    torch = _torch()
    torch.save({"state_dict": net.state_dict(), "meta": meta}, str(path))


def load_ckpt(path, device="cpu"):
    torch = _torch()
    d = torch.load(str(path), map_location=device)
    m = d["meta"]
    net = build_vnet(m["arch"], m["width"], m["heads"]).to(device)
    net.load_state_dict(d["state_dict"])
    net.eval()
    return net, m
