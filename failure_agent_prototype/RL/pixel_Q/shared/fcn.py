"""Fully convolutional Q-map trunk shared by V1 (qnet.build_qnet) and V2 (qnet.build_qnet, ensemble heads).

encoder(in_ch, w)   three stride-2 conv blocks (input side / 8: one output cell per 8x8 pixel window), then
                    two dilated blocks so the receptive field covers the whole map. GroupNorm, so single-sample
                    actor inference behaves like batched training.
with_context(f)     per-cell features concatenated with globally pooled features (8w channels)
head(w, n_out)      1x1 conv head on with_context(f): (B, n_out, H/8, W/8)

The module layout (encoder.<i>.<j>, head.<i>) is V1's, so V1 checkpoints load unchanged.
Python 3.8 / torch 1.11 compatible.
"""


def _nn():
    import torch

    return torch.nn


def conv_block(cin, cout, stride=1, dilation=1):
    nn = _nn()
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
        nn.GroupNorm(8, cout),
        nn.ReLU(inplace=True),
    )


def encoder(in_ch, w):
    nn = _nn()
    return nn.Sequential(
        conv_block(in_ch, w),                     # N
        conv_block(w, 2 * w, stride=2),           # N / 2
        conv_block(2 * w, 2 * w),
        conv_block(2 * w, 4 * w, stride=2),       # N / 4
        conv_block(4 * w, 4 * w),
        conv_block(4 * w, 4 * w, stride=2),       # N / 8: one cell per 8x8 window
        conv_block(4 * w, 4 * w, dilation=2),     # receptive field grows to cover the whole map
        conv_block(4 * w, 4 * w, dilation=4),
    )


def head(w, n_out):
    nn = _nn()
    return nn.Sequential(
        nn.Conv2d(8 * w, 4 * w, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(4 * w, n_out, 1),
    )


def with_context(f):
    """(B, C, H, W) -> (B, 2C, H, W): per-cell features + their global mean."""
    import torch

    g = f.mean(dim=(2, 3), keepdim=True).expand_as(f)
    return torch.cat([f, g], dim=1)
