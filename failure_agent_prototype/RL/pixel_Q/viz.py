"""PNG helpers: Q-map overlays (training / eval) and the inspect view of a LIBERO start state."""

import numpy as np

from qnet import CELL, GRID, N_ROT, ROT_DEG, decode_action


def _heat(v):
    """0..1 -> red..yellow..green."""
    v = float(np.clip(v, 0, 1))
    return (int(255 * min(1, 2 * (1 - v))), int(255 * min(1, 2 * v)), 0)


def save_qmap(obs, q, action, path, scale=4, title=""):
    """Left: the crop with max-over-rotation Q per valid cell and the chosen cell boxed.
    Right: the 9 rotation maps (shared colour scale over valid cells)."""
    from PIL import Image, ImageDraw

    q = np.asarray(q)
    valid = np.asarray(obs["valid"])
    qv = q[:, valid]
    lo, hi = (float(qv.min()), float(qv.max())) if qv.size else (0.0, 1.0)
    norm = (lambda x: (x - lo) / (hi - lo)) if hi > lo else (lambda x: 0.5)
    side = GRID * CELL * scale
    small = side // 3
    im = Image.new("RGB", (side + small * 3 + 10, side + 30), (20, 20, 20))
    crop = Image.fromarray(obs["rgb"]).resize((side, side), Image.NEAREST)
    over = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    qmax = q.max(axis=0)
    s = CELL * scale
    for r in range(GRID):
        for c in range(GRID):
            if valid[r, c]:
                d.rectangle([c * s, r * s, (c + 1) * s - 1, (r + 1) * s - 1], fill=_heat(norm(qmax[r, c])) + (110,))
    k, ra, ca = decode_action(action)
    d.rectangle([ca * s, ra * s, (ca + 1) * s - 1, (ra + 1) * s - 1], outline=(0, 160, 255, 255), width=3)
    im.paste(Image.alpha_composite(crop.convert("RGBA"), over).convert("RGB"), (0, 0))
    cs = small // GRID
    for kk in range(N_ROT):
        ox, oy = side + 10 + (kk % 3) * small, (kk // 3) * small
        for r in range(GRID):
            for c in range(GRID):
                col = _heat(norm(q[kk, r, c])) if valid[r, c] else (60, 60, 60)
                ImageDraw.Draw(im).rectangle([ox + c * cs, oy + r * cs, ox + (c + 1) * cs - 2, oy + (r + 1) * cs - 2],
                                             fill=col)
        ImageDraw.Draw(im).text((ox + 2, oy + 1), "{:+.0f}".format(ROT_DEG[kk]), fill=(0, 160, 255) if kk == k else
                                (255, 255, 255))
    ImageDraw.Draw(im).text((4, side + 8), "{}  chosen yaw {:+.0f} deg  Q in [{:.3f}, {:.3f}]".format(
        title, ROT_DEG[k], lo, hi), fill=(230, 230, 230))
    im.save(path)


def _tint(rgb, mask, scale):
    from PIL import Image

    rgb = rgb.astype(np.float32)
    rgb[mask] = 0.5 * rgb[mask] + 0.5 * np.array([255, 0, 255])
    return Image.fromarray(rgb.astype(np.uint8)).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)


def save_inspect(info, path, side=512):
    """Left: the WRIST view the Q-map sees, object mask tinted magenta (visualisation only), the 80x80 crop
    box (yellow), valid cell centres (white) and their back-projected 3D points projected again (cyan; should
    sit on the white dots). Right: agentview, with the same 3D cell points projected into it (cyan where the
    cell is >= 50 % object, white otherwise): the on-object ones should land on the object."""
    from PIL import Image, ImageDraw

    ws = side // info["rgb_full"].shape[0]
    wrist = _tint(info["rgb_full"], info["mask_full"], ws)
    d = ImageDraw.Draw(wrist)
    u0, v0, crop = info["u0"], info["v0"], GRID * CELL
    d.rectangle([u0 * ws, v0 * ws, (u0 + crop) * ws, (v0 + crop) * ws], outline=(255, 255, 0), width=2)
    for (uc, vc), (up, vp) in zip(info["cell_centres_px"], info["cell_points_px"]):
        d.ellipse([uc * ws - 3, vc * ws - 3, uc * ws + 3, vc * ws + 3], fill=(255, 255, 255))
        d.ellipse([up * ws - 2, vp * ws - 2, up * ws + 2, vp * ws + 2], fill=(0, 255, 255))
    d.text((4, 4), "wrist ({} px)  |  {}".format(info["rgb_full"].shape[0], info["object"]), fill=(255, 255, 0))

    a_s = side // info["agent_rgb"].shape[0]
    agent = _tint(info["agent_rgb"], info["agent_mask"], a_s)
    d = ImageDraw.Draw(agent)
    for (up, vp), f in zip(info["cell_points_agent_px"], info["cell_obj_frac"]):
        col = (0, 255, 255) if f >= 0.5 else (255, 255, 255)
        d.ellipse([up * a_s - 2, vp * a_s - 2, up * a_s + 2, vp * a_s + 2], fill=col)
    d.text((4, 4), "agentview  |  {}".format(info["prompt"]), fill=(255, 255, 0))

    im = Image.new("RGB", (wrist.width + agent.width + 6, max(wrist.height, agent.height)), (20, 20, 20))
    im.paste(wrist, (0, 0))
    im.paste(agent, (wrist.width + 6, 0))
    im.save(path)
