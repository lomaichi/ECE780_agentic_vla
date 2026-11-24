"""Spatial-relation plans (to the right of / left of / front of): real Moondream2 detect on the first frame ->
GroundedRecoveryAgent destination point, vs the BDDL goal region (site position +- site half size). No pi0."""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/DockerShared/ECE780/Documentation/VLM_inference_speed_test/moondreamV2")
sys.path.insert(0, "/home/DockerShared/ECE780/failure_agent_prototype")
import imageio  # noqa: E402
import cv2  # noqa: E402
from fa_core import common, loop as L  # noqa: E402
import recovery_agent as R  # noqa: E402
import vlm_backends  # noqa: E402

vlm = vlm_backends.make_vlm("moondream2")
tiles, rows = [], []
for tid in [int(x) for x in sys.argv[1:]]:
    env, task, _ = common.make_env("libero_90", tid, 7)
    obs = env.reset()
    for _ in range(10):                                   # the loop's settle steps
        obs, _, _, _ = env.step(np.zeros(7).tolist()[:6] + [-1.0])
    sim, inner = env.env.sim, env.env
    reg = inner.parsed_problem["goal_state"][0][-1]
    sid = sim.model.site_name2id(reg)
    site, half = np.array(sim.data.site_xpos[sid]), np.array(sim.model.site_size[sid])
    a = R.GroundedRecoveryAgent(vlm, R.GroundedCamera(env))
    a.every_steps = 4
    a.reset(task.language, task.language)
    view = L.AgentView(10, 0, obs, None, task.language, task.language)
    a.act(view, None)
    d = a.geo["dest"]
    inside = None if d is None else bool(abs(d[0] - site[0]) <= half[0] and abs(d[1] - site[1]) <= half[1])
    row = {"task": tid, "plan": {k: a.plan.get(k) for k in ("object", "destination", "relation", "side")},
           "object_box": a.boxes["object"], "dest_box": a.boxes["destination"],
           "target": None if a.geo["target"] is None else np.round(a.geo["target"], 3).tolist(),
           "dest": None if d is None else np.round(d, 3).tolist(), "region_site": np.round(site, 3).tolist(),
           "region_half": np.round(half, 3).tolist(), "dest_in_region_xy": inside,
           "dest_to_site_xy_cm": None if d is None else round(float(np.hypot(*(d[:2] - site[:2]))) * 100, 1)}
    rows.append(row)
    print(json.dumps(row))
    img = view.image.copy()
    S = img.shape[0]
    for b, c in ((a.boxes["object"], (0, 255, 0)), (a.boxes["destination"], (255, 255, 0))):
        if b:
            cv2.rectangle(img, (int(b["x_min"] * S), int(b["y_min"] * S)), (int(b["x_max"] * S), int(b["y_max"] * S)), c, 1)
    for p, c in ((d, (255, 0, 0)), (site, (255, 0, 255))):
        if p is not None:
            u, v = a.cam.project(p)
            cv2.circle(img, (int(u * S), int(v * S)), 3, c, -1)
    tiles.append(cv2.resize(img, (320, 320)))
    env.close()
imageio.imwrite("/home/DockerShared/ECE780/failure_agent_prototype/framework/runs/_test/check_side_dest.png",
                np.concatenate(tiles, 1))
