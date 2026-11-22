"""Gripper-centred, yaw-aligned heightmap (pixel_Q_RL_V2.md sec. 5) and the geometry the primitive needs.

Pure numpy: the simulator side (rendering both cameras, unprojecting, robot / target masks) is in libero_env.py
and the toy world draws its own points; both end in rasterize() + channels().

Map frame. The map is a 22 x 22 cm square (88 px at 2.5 mm) centred on the grasp site's xy and rotated with the
gripper's yaw psi (the yaw of the levelled gripper frame, see level_frame). A world point w maps to local
coordinates (lx, ly) = Rz(psi)^T (w_xy - p_xy) and to continuous map coordinates
    u = lx / RES + 44   (column; increases along the gripper's x axis)
    v = -ly / RES + 44  (row; increases along the gripper's -y axis)
so pixel (row r, col q) covers [q, q+1) x [r, r+1) and cell (i, j) covers 8x8 pixels with its centre at
u = 8 j + 4, v = 8 i + 4. The grasp point (u = v = 44) is the centre of cell (5, 5), and
    cell (i, j) -> world xy = p_xy + Rz(psi) [(j - C) s, -(i - C) s],  s = 2 cm     (sec. 5, "Cell -> world").

Rasterization keeps, per pixel, the highest point and its colour; where both cameras see a pixel, the wrist
point wins (it is finer). Holes (pixels no camera saw) have observed = 0; optionally, a hole whose 3x3
neighbourhood has >= 3 observed pixels takes the median-height neighbour (agentview is coarser than 2.5 mm, so
regions only it sees are otherwise speckled with holes).
"""

import numpy as np

from qnet import C, CELL, CELL_M, GRID, HM, RES_M

HALF = HM / 2.0                 # continuous map coordinate of the grasp point
HEIGHT_SCALE = 0.1              # m per unit of the height channel
HEIGHT_CLIP = 1.5
GRIPPER_WIN_M = 0.06            # clearance window: gripper-sized 6 x 6 cm


def level_frame(R):
    """Levelled gripper frame of rotation matrix R (approach axis z -> world -z, yaw kept) and its yaw psi.

    Same construction as shared.scene.top_down_quat: x = R's x axis projected on the table plane (R's y axis
    crossed with -z if x is vertical). Returns (R_level 3x3, psi in rad)."""
    R = np.asarray(R, dtype=np.float64)
    x = np.array([R[0, 0], R[1, 0], 0.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([R[0, 1], R[1, 1], 0.0]), [0.0, 0.0, -1.0])
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, -1.0])
    return np.stack([x, np.cross(z, x), z], axis=1), float(np.arctan2(x[1], x[0]))


def tilt_rad(R):
    """Angle between the approach axis (R's z axis) and straight down."""
    return float(np.arccos(np.clip(-np.asarray(R)[2, 2], -1.0, 1.0)))


def world_to_map(xy, p_xy, psi):
    """World xy (..., 2) -> continuous map coordinates (u, v) (each (...,))."""
    d = np.asarray(xy, dtype=np.float64) - np.asarray(p_xy, dtype=np.float64)
    c, s = np.cos(psi), np.sin(psi)
    lx = c * d[..., 0] + s * d[..., 1]
    ly = -s * d[..., 0] + c * d[..., 1]
    return lx / RES_M + HALF, -ly / RES_M + HALF


def map_to_world(u, v, p_xy, psi):
    """Continuous map coordinates -> world xy (..., 2)."""
    lx = (np.asarray(u, dtype=np.float64) - HALF) * RES_M
    ly = -(np.asarray(v, dtype=np.float64) - HALF) * RES_M
    c, s = np.cos(psi), np.sin(psi)
    return np.stack([p_xy[0] + c * lx - s * ly, p_xy[1] + s * lx + c * ly], axis=-1)


def cell_to_world(i, j, p_xy, psi):
    """Cell (row i, column j) -> world xy of its centre (sec. 5)."""
    c, s = np.cos(psi), np.sin(psi)
    lx, ly = (j - C) * CELL_M, -(i - C) * CELL_M
    return np.array([p_xy[0] + c * lx - s * ly, p_xy[1] + s * lx + c * ly])


def cell_centre_uv(i, j):
    return CELL * j + CELL / 2.0, CELL * i + CELL / 2.0


def rasterize(points, colors, target, wrist, p_xy, psi):
    """Top surface per map pixel.

    points (N, 3) world, colors (N, 3) uint8, target (N,) bool, wrist (N,) bool (True = wrist camera point).
    Returns dict: z (HM, HM) float64 world height (nan = hole), rgb (HM, HM, 3) uint8, target (HM, HM) bool,
    observed (HM, HM) bool, wrist (HM, HM) bool (which camera won the pixel)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    u, v = world_to_map(pts[:, :2], p_xy, psi)
    q, r = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    inside = (q >= 0) & (q < HM) & (r >= 0) & (r < HM) & np.isfinite(pts[:, 2])
    pix = r * HM + q
    z = np.full(HM * HM, np.nan)
    rgb = np.zeros((HM * HM, 3), np.uint8)
    tgt = np.zeros(HM * HM, bool)
    src = np.zeros(HM * HM, bool)
    colors = np.asarray(colors).reshape(-1, 3)
    target = np.asarray(target, dtype=bool).reshape(-1)
    wrist = np.asarray(wrist, dtype=bool).reshape(-1)
    for is_wrist in (False, True):              # agentview first; the wrist overwrites the pixels it sees
        sel = np.flatnonzero(inside & (wrist == is_wrist))
        if sel.size == 0:
            continue
        ps, zs = pix[sel], pts[sel, 2]
        order = np.lexsort((zs, ps))            # by pixel, then height ascending
        ps_o = ps[order]
        last = np.r_[ps_o[1:] != ps_o[:-1], True]
        top = sel[order[last]]                  # the highest point of every pixel
        dst = ps_o[last]
        z[dst], rgb[dst], tgt[dst], src[dst] = pts[top, 2], colors[top], target[top], is_wrist
    return {"z": z.reshape(HM, HM), "rgb": rgb.reshape(HM, HM, 3), "target": tgt.reshape(HM, HM),
            "observed": np.isfinite(z).reshape(HM, HM), "wrist": src.reshape(HM, HM)}


def fill_holes(r, min_neighbours=3):
    """One pass: a hole with >= min_neighbours observed pixels in its 3x3 neighbourhood takes the neighbour of
    median height (its colour and target flag too) and counts as observed. Modifies and returns r."""
    z = r["z"]
    pad = np.pad(z, 1, constant_values=np.nan)
    offs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    Z = np.stack([pad[1 + dr:1 + dr + HM, 1 + dc:1 + dc + HM] for dr, dc in offs])      # (8, HM, HM)
    cnt = np.isfinite(Z).sum(axis=0)
    holes = ~np.isfinite(z) & (cnt >= min_neighbours)
    if not holes.any():
        return r
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(holes[None], Z, np.nan), axis=0)
    dist = np.abs(Z - med[None])
    dist[~np.isfinite(dist)] = np.inf
    k = np.argmin(dist, axis=0)
    rr, cc = np.nonzero(holes)
    kk = k[rr, cc]
    sr = np.clip(rr + np.array([o[0] for o in offs])[kk], 0, HM - 1)
    sc = np.clip(cc + np.array([o[1] for o in offs])[kk], 0, HM - 1)
    for key in ("z", "rgb", "target", "wrist"):
        r[key][rr, cc] = r[key][sr, sc]
    r["observed"] = np.isfinite(r["z"])
    return r


def channels(r, z_grasp):
    """Raster -> the Q-map's map channels: rgb uint8, height float16 ((h - z_grasp) / 0.1 m, clipped to
    +-1.5; 0 at holes), mask bool (target), observed bool."""
    obs = r["observed"]
    h = np.where(obs, np.clip((r["z"] - z_grasp) / HEIGHT_SCALE, -HEIGHT_CLIP, HEIGHT_CLIP), 0.0)
    rgb = np.where(obs[..., None], r["rgb"], 0).astype(np.uint8)
    return {"rgb": rgb, "height": h.astype(np.float16), "mask": (r["target"] & obs).copy(), "observed": obs.copy()}


def clearance_height(z, i, j, win_m=GRIPPER_WIN_M, step_px=2.0):
    """h_clear (sec. 6, primitive step 1): the highest observed surface (world z, m) under a gripper-sized window
    swept from the grasp point (map centre) to the centre of destination cell (i, j). None if nothing under the
    window was observed. z: the raster's (HM, HM) world heights (nan = hole)."""
    half = 0.5 * win_m / RES_M
    u1, v1 = cell_centre_uv(i, j)
    n = int(np.ceil(np.hypot(u1 - HALF, v1 - HALF) / step_px)) + 1
    m = np.zeros((HM, HM), bool)
    for t in np.linspace(0.0, 1.0, max(n, 1)):
        u, v = HALF + t * (u1 - HALF), HALF + t * (v1 - HALF)
        r0, r1 = int(max(0, np.floor(v - half))), int(min(HM, np.ceil(v + half)))
        c0, c1 = int(max(0, np.floor(u - half))), int(min(HM, np.ceil(u + half)))
        m[r0:r1, c0:c1] = True
    vals = z[m & np.isfinite(z)]
    return float(vals.max()) if vals.size else None


def cell_world_grid(p_xy, psi):
    """World xy of every cell centre: (GRID, GRID, 2)."""
    ii, jj = np.mgrid[0:GRID, 0:GRID]
    c, s = np.cos(psi), np.sin(psi)
    lx, ly = (jj - C) * CELL_M, -(ii - C) * CELL_M
    return np.stack([p_xy[0] + c * lx - s * ly, p_xy[1] + s * lx + c * ly], axis=-1)
