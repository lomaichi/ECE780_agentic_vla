"""CPU unit tests for V2 (no LIBERO, no pi0): python test_v2.py  (numpy, torch, pillow).

Covers the action encoding and network shapes, the trigger score, the map geometry (cell <-> world under yaw,
rasterization priority, hole filling, clearance), the replay buffer, labels, the trigger's persistence, branch
points and snapshot / restore on the toy backend, the main rollout's sample structure and cooldowns, and the
report functions. The LIBERO backend is checked in the simulator (--mode inspect / restore-check).
"""

import pathlib
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for _p in (HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import evaluation as EV  # noqa: E402
import heightmap as HMAP  # noqa: E402
import qnet as Q  # noqa: E402
import rollout as RO  # noqa: E402
from replay import ReplayBuffer  # noqa: E402
from toy_env import ToyBranchEnv, toy_starts  # noqa: E402


def rotz(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rotx(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


class TestQNet(unittest.TestCase):
    def test_actions(self):
        self.assertEqual((Q.HM, Q.GRID, Q.C, Q.K0), (88, 11, 5, 4))
        self.assertEqual(Q.decode_action(Q.A0), (4, 5, 5))
        for a in (0, 17, Q.A0, Q.N_ACTIONS - 1):
            self.assertEqual(Q.encode_action(*Q.decode_action(a)), a)
        v = Q.reach_mask(3)
        self.assertEqual(int(v.sum()), 49)
        self.assertEqual(int(Q.valid_actions(v).sum()), 441)
        corr = Q.correction_actions(v)
        self.assertEqual(int(corr.sum()), 440)
        self.assertFalse(corr[Q.A0])
        self.assertTrue(corr[Q.encode_action(0, 5, 5)])          # yaw-only at the centre is a correction
        self.assertFalse(Q.valid_actions(v)[Q.encode_action(4, 0, 0)])   # corner: out of reach

    def test_network(self):
        import torch

        torch.manual_seed(0)
        net = Q.build_qnet(16, 3)
        obs = {"rgb": np.zeros((88, 88, 3), np.uint8), "height": np.zeros((88, 88), np.float16),
               "mask": np.zeros((88, 88), bool), "observed": np.ones((88, 88), bool),
               "proprio": np.zeros(4, np.float32), "valid": Q.reach_mask(3)}
        q = Q.ensemble_q(net, obs)
        self.assertEqual(q.shape, (3, 9, 11, 11))
        x = Q.obs_to_tensor([obs, obs])
        self.assertEqual(tuple(x.shape), (2, Q.IN_CH, 88, 88))
        self.assertEqual(tuple(net(x).shape), (2, 3, 9, 11, 11))
        try:                                                         # usable GPU? (LIBERO venv: no RTX 5090 kernels;
            torch.nn.Conv2d(1, 1, 1).cuda()(torch.zeros(1, 1, 2, 2).cuda())   # a shared GPU may be full)
        except (AssertionError, RuntimeError) as e:
            self.skipTest("no usable GPU: {}".format(str(e).splitlines()[0]))
        np.testing.assert_allclose(Q.ensemble_q(net.cuda(), obs), q, atol=3e-3)   # follows the net's device (TF32)

    def test_score(self):
        q = np.zeros((2, 9, 11, 11))
        q[:, Q.K0, Q.C, Q.C] = 0.5                                   # Q0 = 0.5
        q[0, 2, 4, 6], q[1, 2, 4, 6] = 0.9, 0.7                      # mean 0.8, std 0.1
        q[:, 3, 0, 0] = 5.0                                          # out of reach: ignored
        s = Q.score(q, Q.reach_mask(3), kappa=1.0)
        self.assertEqual(Q.decode_action(s["a_best"]), (2, 4, 6))
        self.assertAlmostEqual(s["q0"], 0.5)
        self.assertAlmostEqual(s["A"], 0.8 - 0.1 - 0.5)
        s0 = Q.score(q, Q.reach_mask(3), kappa=0.0)
        self.assertAlmostEqual(s0["A"], 0.3)


class TestGeometry(unittest.TestCase):
    def test_cells(self):
        p = np.array([0.3, -0.1])
        for psi in (0.0, 0.7, -2.0):
            for i, j in ((5, 5), (2, 8), (8, 1)):
                w = HMAP.cell_to_world(i, j, p, psi)
                u, v = HMAP.world_to_map(w, p, psi)
                self.assertAlmostEqual(float(u), 8 * j + 4, places=6)
                self.assertAlmostEqual(float(v), 8 * i + 4, places=6)
        np.testing.assert_allclose(HMAP.cell_to_world(5, 5, p, 1.0), p)
        # psi = 90 deg: one cell along the gripper's x (column + 1) is world +y; one row down is world +x
        np.testing.assert_allclose(HMAP.cell_to_world(5, 6, p, np.pi / 2), p + [0, 0.02], atol=1e-12)
        np.testing.assert_allclose(HMAP.cell_to_world(6, 5, p, np.pi / 2), p + [0.02, 0], atol=1e-12)
        np.testing.assert_allclose(HMAP.cell_world_grid(p, 0.4)[2, 7], HMAP.cell_to_world(2, 7, p, 0.4))

    def test_level_frame(self):
        R = rotz(0.6) @ rotx(np.pi) @ rotx(np.radians(25))           # pointing down, tilted 25 deg, yaw 0.6
        R_level, psi = HMAP.level_frame(R)
        np.testing.assert_allclose(R_level[:, 2], [0, 0, -1], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(R_level), 1.0, places=9)
        self.assertAlmostEqual(psi, 0.6, places=9)
        self.assertAlmostEqual(np.degrees(HMAP.tilt_rad(R)), 25.0, places=6)
        self.assertAlmostEqual(HMAP.tilt_rad(R_level), 0.0, places=6)

    def test_rasterize(self):
        p, psi = np.zeros(2), 0.3
        w = HMAP.map_to_world(np.array([10.5, 10.5, 10.5, 20.5]), np.array([30.5, 30.5, 30.5, 40.5]), p, psi)
        pts = np.c_[w, [0.01, 0.05, 0.03, 0.02]]
        col = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], np.uint8)
        tgt = np.array([False, True, False, False])
        wrist = np.array([False, False, True, False])
        r = HMAP.rasterize(pts, col, tgt, wrist, p, psi)
        self.assertAlmostEqual(r["z"][30, 10], 0.03)                 # the wrist point wins over a higher agent one
        self.assertEqual(int(r["rgb"][30, 10, 0]), 3)
        self.assertFalse(r["target"][30, 10])
        r2 = HMAP.rasterize(pts[:2], col[:2], tgt[:2], np.zeros(2, bool), p, psi)
        self.assertAlmostEqual(r2["z"][30, 10], 0.05)                # same camera: the highest point wins
        self.assertTrue(r2["target"][30, 10])
        self.assertEqual(int(r["observed"].sum()), 2)
        ch = HMAP.channels(r, z_grasp=0.1)
        self.assertAlmostEqual(float(ch["height"][30, 10]), (0.03 - 0.1) / 0.1, places=2)
        self.assertEqual(float(ch["height"][0, 0]), 0.0)
        self.assertEqual(ch["rgb"].dtype, np.uint8)

    def test_fill_holes(self):
        z = np.full((88, 88), 0.02)
        z[40, 40] = np.nan
        z[0:3, 0:3] = np.nan
        r = {"z": z, "rgb": np.zeros((88, 88, 3), np.uint8), "target": np.zeros((88, 88), bool),
             "observed": np.isfinite(z), "wrist": np.zeros((88, 88), bool)}
        r = HMAP.fill_holes(r)
        self.assertTrue(r["observed"][40, 40])
        self.assertAlmostEqual(r["z"][40, 40], 0.02)
        self.assertFalse(r["observed"][0, 0])                        # corner of a 3x3 hole: < 3 observed neighbours

    def test_clearance(self):
        z = np.full((88, 88), 0.0)
        u, v = HMAP.cell_centre_uv(5, 8)                             # destination 3 cells along +x
        z[int(v) - 2:int(v) + 2, 56:58] = 0.08                       # a wall between the gripper and the cell
        self.assertAlmostEqual(HMAP.clearance_height(z, 5, 8), 0.08)
        self.assertAlmostEqual(HMAP.clearance_height(z, 5, 2), 0.0)  # the other way: wall never under the window
        self.assertIsNone(HMAP.clearance_height(np.full((88, 88), np.nan), 5, 5))


class TestReplay(unittest.TestCase):
    def test_buffer(self):
        rng = np.random.default_rng(0)
        obs = {"rgb": np.zeros((88, 88, 3), np.uint8), "height": np.zeros((88, 88), np.float16),
               "mask": np.zeros((88, 88), bool), "observed": np.ones((88, 88), bool),
               "proprio": np.zeros(4, np.float32), "valid": Q.reach_mask(3)}
        buf = ReplayBuffer(5)
        for n in range(4):
            buf.add(obs, Q.A0, 0.5)
            buf.add(obs, 7, 0.0)
        self.assertEqual(len(buf), 5)
        b = buf.sample(6, rng)
        self.assertEqual(b["rgb"].shape, (6, 88, 88, 3))
        self.assertEqual(b["is_noop"].dtype, bool)
        with tempfile.TemporaryDirectory() as d:
            buf.save(d + "/r.pkl")
            b2 = ReplayBuffer.load(d + "/r.pkl")
            self.assertEqual(len(b2), 5)
            self.assertIs(b2.data[0]["obs"], b2.data[1]["obs"])      # shared observation survives pickling
            b3 = ReplayBuffer.load(d + "/r.pkl", capacity=3)
            self.assertEqual(len(b3), 3)


class TestRollout(unittest.TestCase):
    def test_label(self):
        self.assertAlmostEqual(RO.label_at(60, 20, 0.85), 0.85)
        self.assertAlmostEqual(RO.label_at(60, 60, 0.85), 0.85 ** 3)
        self.assertEqual(RO.label_at(60, 61, 0.85), 0.0)
        self.assertEqual(RO.label_at(60, None, 0.85), 0.0)

    def test_trigger(self):
        t = RO.Trigger(0.1, 3)
        self.assertEqual([t.observe(a) for a in (0.2, 0.2, 0.05, 0.2, 0.2, 0.2)], [False] * 5 + [True])

    def _env(self, **kw):
        cfg = dict({"seed": 3, "reach": 3}, **kw)
        return ToyBranchEnv(cfg)

    def test_snapshot_restore(self):
        env = self._env(same_noise=True)
        env.reset(toy_starts(1)[0])
        for _ in range(15):
            env.pi0_step()
        env.observe()
        snap = env.snapshot()
        r1 = RO.run_branch(env, None, 60, 60, 0.85, 30)
        env.restore(snap)
        r2 = RO.run_branch(env, None, 60, 60, 0.85, 30)
        self.assertEqual((r1["t"], r1["end"]), (r2["t"], r2["end"]))  # same noise -> identical branches
        env.restore(snap)
        self.assertEqual(env.step_count, 15)

    def test_main_rollout_samples(self):
        env = self._env()
        scorer = Q.Scorer(16, 2, 1.0)
        rng = np.random.default_rng(1)
        n_bp = n_s = 0
        for s in toy_starts(12):
            res = RO.main_rollout(env, s, {"p_branch": 0.5, "branch_action": "eps", "eps": 1.0}, scorer, rng)
            n_bp += len(res["branches"])
            n_s += len(res["samples"])
            for b in res["branches"]:
                self.assertTrue(b["y_noop"] == 0 or 0 < b["y_noop"] <= 1)
                self.assertIsNotNone(b["prim"])
            for obs, a, y, info in res["samples"]:
                self.assertEqual(obs["rgb"].shape, (88, 88, 3))
        self.assertGreater(n_bp, 0)
        self.assertEqual(n_s, 2 * n_bp)                              # one no-op + one correction per branch point

    def test_phase0_samples(self):
        env = self._env()
        rng = np.random.default_rng(2)
        res = [RO.main_rollout(env, s, {"p_branch": 0.5, "branch_action": "probe", "branch_cap_steps": 200}, None,
                               rng) for s in toy_starts(10)]
        brs = [b for r in res for b in r["branches"]]
        self.assertTrue(brs)
        self.assertTrue(all(b["a"] == Q.A0 for b in brs))
        self.assertEqual(sum(len(r["samples"]) for r in res), len(brs))   # probes' own labels are not pushed
        self.assertTrue(all(a == Q.A0 for r in res for _, a, _, _ in r["samples"]))

    def test_trigger_cooldown(self):
        env = self._env()

        class Always(object):                                        # A(s) = 1 everywhere: the trigger fires ASAP
            loaded, kappa = True, 1.0

            def score(self, obs, kappa=None):
                q = np.zeros((1, 9, 11, 11))
                q[0, 2, 5, 6] = 1.0
                s = Q.score(q, obs["valid"], 1.0)
                s["q"] = q
                return s

        rng = np.random.default_rng(0)
        rc = {"trigger": "learned", "tau": 0.5, "n_persist": 3, "cooldown_steps": 60}
        n_int = 0
        for s in toy_starts(8):
            res = RO.main_rollout(env, s, rc, Always(), rng)
            steps = [x["step"] for x in res["interventions"]]
            n_int += len(steps)
            self.assertTrue(all(b - a >= 60 for a, b in zip(steps, steps[1:])))
            for x in res["interventions"]:
                self.assertLessEqual(x["prim"]["steps"], 30)
        self.assertGreater(n_int, 0)


class TestEvaluation(unittest.TestCase):
    def test_knee(self):
        pts = [(0.0, 0.5), (0.5, 0.7), (1.0, 0.75), (2.0, 0.77), (4.0, 0.78)]
        self.assertEqual(EV.knee(pts), 2)            # scaled distances to the chord: 0, .59, .64, .46, 0
        self.assertEqual(EV.knee([(2.0, 0.77), (0.0, 0.5), (1.0, 0.75), (4.0, 0.78), (0.5, 0.7)]), 2)   # unsorted

    def test_phase0_report(self):
        rng = np.random.default_rng(0)
        rows = [{"tilt_deg": float(rng.uniform(0, 40)), "t_noop": int(rng.integers(20, 50)),
                 "t_a": int(rng.integers(20, 50))} for _ in range(400)]
        rep = EV.phase0_report(rows, 60, 0.85)
        self.assertEqual(sum(b["n"] for b in rep["bins"]), 400)
        self.assertIn("go ahead", rep["recommendation"])
        rows2 = [dict(r, t_a=None) if r["tilt_deg"] >= 30 else r for r in rows]
        self.assertIn("30-+", EV.phase0_report(rows2, 60, 0.85)["recommendation"])

    def test_virtual_trigger(self):
        ticks = [{"step": 4 * k, "A": a} for k, a in enumerate([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])]
        self.assertEqual(EV.virtual_trigger(ticks, 0.1, 3, 60), [8])
        gap = [{"step": 0, "A": 1}, {"step": 4, "A": 1}, {"step": 40, "A": 1}]
        self.assertEqual(EV.virtual_trigger(gap, 0.1, 3, 60), [])   # a closed gate in between resets the streak


if __name__ == "__main__":
    unittest.main(verbosity=2)
