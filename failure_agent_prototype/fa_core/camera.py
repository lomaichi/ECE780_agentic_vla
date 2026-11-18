"""Agentview camera geometry: pixel + depth -> world point (for VLM pointing, POINT_APPROACH).

What a test-time agent may use: the RGB-D camera itself (intrinsics from fovy, extrinsics from the camera
mount, a depth image). No object poses are read here.

Pixels are in the UPRIGHT image (common.upright, what pi0 / the videos / the VLM see): u = column, v = row,
origin top-left. The raw MuJoCo render is the upright image rotated by 180 degrees.
"""

import numpy as np

CAMERA = "agentview"


def _sim(env):
    return env.env.sim


def depth_image(env, size, camera=CAMERA):
    """Metric depth (m, distance along the optical axis) of the raw (not upright) render."""
    sim = _sim(env)
    _, z = sim.render(width=size, height=size, camera_name=camera, depth=True)
    extent = sim.model.stat.extent
    near, far = sim.model.vis.map.znear * extent, sim.model.vis.map.zfar * extent
    return near / (1.0 - z * (1.0 - near / far))


def _cam(env, camera):
    sim = _sim(env)
    cid = sim.model.camera_name2id(camera)
    fovy = float(sim.model.cam_fovy[cid])
    return np.array(sim.data.cam_xpos[cid]), np.array(sim.data.cam_xmat[cid]).reshape(3, 3), fovy


def _raw_pixel(u, v, size):
    """Upright (u, v) -> raw render (col, row-from-bottom-of-array)."""
    return size - 1 - u, size - 1 - v


def unproject(env, u, v, size, depth=None, camera=CAMERA, window=2):
    """World point seen at upright pixel (u, v). Uses the median depth in a small window (robust to edges)."""
    if depth is None:
        depth = depth_image(env, size, camera)
    c, r = _raw_pixel(float(u), float(v), size)
    ci, ri = int(round(np.clip(c, 0, size - 1))), int(round(np.clip(r, 0, size - 1)))
    patch = depth[max(0, ri - window):ri + window + 1, max(0, ci - window):ci + window + 1]
    d = float(np.median(patch))
    pos, mat, fovy = _cam(env, camera)
    f = 0.5 * size / np.tan(np.deg2rad(fovy) / 2)
    # raw render: array row 0 is the BOTTOM of the OpenGL image (y up), MuJoCo camera looks along -z
    x_cam = (c + 0.5 - size / 2) / f * d
    y_cam = (r + 0.5 - size / 2) / f * d
    return pos + mat @ np.array([x_cam, y_cam, -d])


def project(env, world, size, camera=CAMERA):
    """World point -> upright pixel (u, v). Inverse of unproject (used only for checks and overlays)."""
    pos, mat, fovy = _cam(env, camera)
    p = mat.T @ (np.asarray(world) - pos)
    d = -p[2]
    f = 0.5 * size / np.tan(np.deg2rad(fovy) / 2)
    c = p[0] / d * f + size / 2 - 0.5
    r = p[1] / d * f + size / 2 - 0.5
    return size - 1 - c, size - 1 - r
