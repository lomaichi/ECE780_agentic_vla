"""PNG helpers for V2 (PIL only, like V1): Q-map overlays, inspect views, and small line plots (reliability,
tau sweep, A(s) timelines)."""

import numpy as np

from qnet import A0, C, CELL, GRID, HM, N_ROT, ROT_DEG, correction_actions, decode_action, ensemble_stats


def _heat(v):
    """0..1 -> red..yellow..green."""
    v = float(np.clip(v, 0, 1))
    return (int(255 * min(1, 2 * (1 - v))), int(255 * min(1, 2 * v)), 0)


def _font():
    from PIL import ImageFont

    return ImageFont.load_default()


def height_image(h):
    """Height channel ((h - z_grasp) / 0.1, -1.5..1.5) -> grey RGB (dark = far below the fingers)."""
    g = np.clip((np.asarray(h, dtype=np.float32) + 1.5) / 3.0, 0, 1)
    return np.repeat((255 * g).astype(np.uint8)[..., None], 3, axis=2)


def save_qmap(obs, q, action, path, kappa=1.0, scale=4, title=""):
    """Left: heightmap RGB with max over rotation bins of mu - kappa*sigma per valid cell; the chosen cell boxed
    (blue), the no-op cell ringed (white). Middle: height channel with the target mask (magenta).
    Right: the 9 rotation maps of mu (shared colour scale over valid cells), no-op entry outlined."""
    from PIL import Image, ImageDraw

    mu, sd = ensemble_stats(q)
    pen = mu - kappa * sd
    valid = np.asarray(obs["valid"])
    vals = pen[:, valid]
    lo, hi = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)
    norm = (lambda x: (x - lo) / (hi - lo)) if hi > lo else (lambda x: 0.5)
    side, s = HM * scale, CELL * scale
    small = side // 3
    im = Image.new("RGB", (2 * side + 3 * small + 30, side + 34), (20, 20, 20))
    base = Image.fromarray(obs["rgb"]).resize((side, side), Image.NEAREST).convert("RGBA")
    over = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    pmax = pen.max(axis=0)
    for r in range(GRID):
        for c in range(GRID):
            if valid[r, c]:
                d.rectangle([c * s, r * s, (c + 1) * s - 1, (r + 1) * s - 1], fill=_heat(norm(pmax[r, c])) + (100,))
    k, ra, ca = decode_action(action)
    d.rectangle([ca * s, ra * s, (ca + 1) * s - 1, (ra + 1) * s - 1], outline=(0, 160, 255, 255), width=3)
    d.ellipse([C * s + s // 2 - 5, C * s + s // 2 - 5, C * s + s // 2 + 5, C * s + s // 2 + 5],
              outline=(255, 255, 255, 255), width=2)
    im.paste(Image.alpha_composite(base, over).convert("RGB"), (0, 0))
    hgt = height_image(obs["height"])
    hgt[np.asarray(obs["mask"])] = (0.5 * hgt[np.asarray(obs["mask"])] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    hgt[~np.asarray(obs["observed"])] = (0, 0, 90)
    im.paste(Image.fromarray(hgt).resize((side, side), Image.NEAREST), (side + 10, 0))
    cs = small // GRID
    dr = ImageDraw.Draw(im)
    for kk in range(N_ROT):
        ox, oy = 2 * side + 20 + (kk % 3) * small, (kk // 3) * small
        for r in range(GRID):
            for c in range(GRID):
                col = _heat(norm(mu[kk, r, c])) if valid[r, c] else (60, 60, 60)
                dr.rectangle([ox + c * cs, oy + r * cs, ox + (c + 1) * cs - 2, oy + (r + 1) * cs - 2], fill=col)
        if kk == N_ROT // 2:
            dr.rectangle([ox + C * cs - 1, oy + C * cs - 1, ox + (C + 1) * cs - 1, oy + (C + 1) * cs - 1],
                         outline=(255, 255, 255))
        dr.text((ox + 2, oy + 1), "{:+.0f}".format(ROT_DEG[kk]), fill=(0, 160, 255) if kk == k else (255, 255, 255),
                font=_font())
    corr = correction_actions(valid)
    flat = pen.reshape(-1)
    A = float(np.max(np.where(corr, flat, -np.inf)) - mu.reshape(-1)[A0])
    dr.text((4, side + 6), "{}  Q0 {:.3f}  A {:+.3f} (kappa {})  chosen cell ({},{}) yaw {:+.0f}  mu-k*sigma in [{:.3f}, "
            "{:.3f}]".format(title, float(mu.reshape(-1)[A0]), A, kappa, ra, ca, ROT_DEG[k], lo, hi),
            fill=(230, 230, 230), font=_font())
    im.save(path)


def save_inspect(info, obs, path, scale=4):
    """Map channels (rgb | height | target mask | observed) and agentview with the 11 x 11 cell centres projected
    at the target's top height (cyan = within reach, red = the grasp point's cell) and the wrist view."""
    from PIL import Image, ImageDraw

    side = HM * scale
    tiles = [obs["rgb"], height_image(obs["height"]),
             np.repeat((255 * np.asarray(obs["mask"], np.uint8))[..., None], 3, axis=2),
             np.repeat((255 * np.asarray(obs["observed"], np.uint8))[..., None], 3, axis=2)]
    agent = Image.fromarray(info["agent_rgb"]).resize((2 * info["agent_rgb"].shape[0],) * 2, Image.NEAREST)
    a_s = agent.width / float(info["agent_rgb"].shape[0])
    d = ImageDraw.Draw(agent)
    valid = np.asarray(obs["valid"])
    for r in range(GRID):
        for c in range(GRID):
            u, v = info["cells_agent_px"][r, c] * a_s
            col = (255, 0, 0) if (r, c) == (C, C) else (0, 255, 255) if valid[r, c] else (200, 200, 200)
            d.ellipse([u - 2, v - 2, u + 2, v + 2], fill=col)
    wrist = Image.fromarray(info["wrist_rgb"]).resize((2 * info["wrist_rgb"].shape[0],) * 2, Image.NEAREST)
    W = 4 * (side + 6) + agent.width + wrist.width + 12
    im = Image.new("RGB", (W, max(side, agent.height) + 40), (20, 20, 20))
    for n, t in enumerate(tiles):
        tile = Image.fromarray(np.asarray(t, np.uint8)).resize((side, side), Image.NEAREST)
        dd = ImageDraw.Draw(tile)
        for g in range(1, GRID):
            dd.line([(g * CELL * scale, 0), (g * CELL * scale, side)], fill=(70, 70, 70))
            dd.line([(0, g * CELL * scale), (side, g * CELL * scale)], fill=(70, 70, 70))
        dd.rectangle([C * CELL * scale, C * CELL * scale, (C + 1) * CELL * scale, (C + 1) * CELL * scale],
                     outline=(255, 0, 0))
        im.paste(tile, (n * (side + 6), 0))
    im.paste(agent, (4 * (side + 6), 0))
    im.paste(wrist, (4 * (side + 6) + agent.width + 12, 0))
    ImageDraw.Draw(im).text((4, max(side, agent.height) + 6), "{} | {}  rgb | height | target | observed | agentview "
                            "(cells at the target top) | wrist".format(info.get("object"), info.get("prompt")),
                            fill=(230, 230, 230), font=_font())
    im.save(path)


def line_plot(series, path, title="", xlabel="", ylabel="", hlines=(), size=(640, 420), xlim=None, ylim=None):
    """series: [(label, xs, ys, (r, g, b), "line" | "dots")]. hlines: [(y, (r, g, b))]."""
    from PIL import Image, ImageDraw

    W, H = size
    L, R, T, B = 60, 20, 30, 50
    im = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(im)
    xs = np.concatenate([np.asarray(s[1], float) for s in series if len(s[1])] or [np.array([0.0, 1.0])])
    ys = np.concatenate([np.asarray(s[2], float) for s in series if len(s[2])] or [np.array([0.0, 1.0])])
    ys = np.concatenate([ys, [h[0] for h in hlines]]) if hlines else ys
    x0, x1 = xlim or (float(np.nanmin(xs)), float(np.nanmax(xs)))
    y0, y1 = ylim or (float(np.nanmin(ys)), float(np.nanmax(ys)))
    x1, y1 = (x1 + 1e-9 if x1 <= x0 else x1), (y1 + 1e-9 if y1 <= y0 else y1)
    px = lambda x: L + (x - x0) / (x1 - x0) * (W - L - R)  # noqa: E731
    py = lambda y: H - B - (y - y0) / (y1 - y0) * (H - T - B)  # noqa: E731
    f = _font()
    d.rectangle([L, T, W - R, H - B], outline=(0, 0, 0))
    for t in np.linspace(0, 1, 5):
        xv, yv = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        d.text((px(xv) - 12, H - B + 4), "{:.3g}".format(xv), fill=(0, 0, 0), font=f)
        d.text((4, py(yv) - 6), "{:.3g}".format(yv), fill=(0, 0, 0), font=f)
    for y, col in hlines:
        d.line([(L, py(y)), (W - R, py(y))], fill=col, width=1)
    for n, (label, sx, sy, col, style) in enumerate(series):
        pts = [(px(x), py(y)) for x, y in zip(sx, sy) if np.isfinite(x) and np.isfinite(y)]
        if style == "line" and len(pts) > 1:
            d.line(pts, fill=col, width=2)
        for p in pts:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=col)
        d.text((L + 6, T + 4 + 12 * n), label, fill=col, font=f)
    d.text((L, 8), title, fill=(0, 0, 0), font=f)
    d.text((W // 2 - 40, H - 20), xlabel, fill=(0, 0, 0), font=f)
    d.text((4, 12), ylabel, fill=(0, 0, 0), font=f)
    im.save(path)


def save_reliability(bins, path, title="Q0 reliability"):
    b = [x for x in bins if x["n"] > 0]
    line_plot([("ideal", [0, 1], [0, 1], (160, 160, 160), "line"),
               ("mean y_noop", [x["q0_mean"] for x in b], [x["y_mean"] for x in b], (30, 90, 220), "line"),
               ("success frac", [x["q0_mean"] for x in b], [x["success_frac"] for x in b], (220, 120, 30), "line")],
              path, title=title, xlabel="predicted Q0", ylabel="observed", xlim=(0, 1), ylim=(0, 1))


def save_tau_sweep(rows, path):
    r = [x for x in rows if x.get("success") is not None]
    line_plot([("pick success vs interventions / rollout", [x["interventions_per_rollout"] for x in r],
                [x["success"] for x in r], (30, 90, 220), "line")], path,
              title="tau sweep (labels: tau) " + ", ".join("{}".format(x["tau"]) for x in r),
              xlabel="interventions per rollout", ylabel="success")


def save_timelines(results, path, tau, max_rollouts=12):
    cols = [(30, 90, 220), (220, 60, 60), (40, 160, 60), (200, 120, 20), (120, 60, 180), (20, 160, 160)]
    series = []
    for n, r in enumerate([r for r in results if r.get("ticks")][:max_rollouts]):
        t = [x["step"] / 20.0 for x in r["ticks"]]
        series.append(("{} {}".format(r["tag"], "ok" if r.get("pick_success") else "FAIL"), t,
                       [x["A"] for x in r["ticks"]], cols[n % len(cols)], "line"))
    if series:
        line_plot(series, path, title="A(s) at gated ticks", xlabel="time (s)", ylabel="A(s)",
                  hlines=[(tau, (0, 0, 0))], size=(900, 500))
