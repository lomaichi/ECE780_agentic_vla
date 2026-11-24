"""One-off check: --target-mask vlm (detect box) vs seg (training) for Q_INITIAL_POSITION and the Q wrist crop."""
import sys

sys.path.insert(0, "/home/DockerShared/ECE780/failure_agent_prototype/framework")
import numpy as np  # noqa: E402

import components as K  # noqa: E402
import config as C  # noqa: E402
from fa_core import common, loop, motion, predicates as P  # noqa: E402

m = K.load_modules()
SC = m["scene"]


class BoxVLM(object):
    def __init__(self, env, obj):
        self.env, self.obj = env, obj

    def box(self):
        seg = common.upright(SC.segmentation_geom_ids(self.env, 256))
        vs, us = np.nonzero(np.isin(seg, SC.object_geom_ids(self.env, self.obj)))
        return {"x_min": us.min() / 256., "y_min": vs.min() / 256., "x_max": (us.max() + 1) / 256., "y_max": (vs.max() + 1) / 256.}

    def gate_inputs(self):
        return {"pick": True, "target": None, "held": False, "box": self.box()}


q = K.QModel()
for task, hover in ((15, (0.03, 0.025, 0.08)), (24, (0.0, 0.03, 0.06)), (13, (0.03, 0.0, 0.10))):
    env, t, _ = common.make_env("libero_90", task)
    env.reset()
    obs = env.set_init_state(common.init_state("libero_90", task, 0))
    for _ in range(10):
        obs, _, _, _ = env.step(list(common.DUMMY_ACTION))
    obj = [c.obj for c in P.goal_checks(env) if c.kind == "pick"][0]
    mv = motion.MoveTo(P.obj_pos(env, obj) + np.array(hover), common.GRIPPER_OPEN, max_steps=80)
    while True:
        a = mv.next_action(obs)
        if a is None:
            break
        obs, _, _, _ = env.step(list(a))
    vlm = BoxVLM(env, obj)
    per = {k: K.Perception(env, obj, vlm, target_mask=k) for k in ("seg", "vlm")}
    view = loop.AgentView(100, 90, obs, None, t.language, t.language)
    h = {k: p.handoff(view) for k, p in per.items()}
    print("task {} ({}) | Q_INIT seg {} | vlm {} | diff {:.1f} mm".format(
        task, obj, np.round(h["seg"][1]["q_initial_position"], 4), np.round(h["vlm"][1]["q_initial_position"], 4),
        1000 * np.linalg.norm(h["seg"][1]["q_initial_position"] - h["vlm"][1]["q_initial_position"])))
    mo = h["seg"][0]                                     # drive to the seg handoff pose
    while True:
        a = mo.next_action(obs)
        if a is None:
            break
        obs, _, _, _ = env.step(list(a))
    view = loop.AgentView(200, 190, obs, None, t.language, t.language)
    o = {k: p.q_obs(view) for k, p in per.items()}
    ms, mvm = o["seg"]["mask"], o["vlm"]["mask"]
    iou = (ms & mvm).sum() / float((ms | mvm).sum())
    sel = {k: q.select(o[k])[1] for k in o}
    print("   wrist crop mask px seg {} vlm {} IoU {:.2f} | Q argmax seg {} vlm {}".format(
        int(ms.sum()), int(mvm.sum()), iou, (sel["seg"]["k"], sel["seg"]["row"], sel["seg"]["col"]),
        (sel["vlm"]["k"], sel["vlm"]["row"], sel["vlm"]["col"])))
    env.close()
