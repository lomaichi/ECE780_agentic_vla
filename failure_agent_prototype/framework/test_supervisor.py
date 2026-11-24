"""Unit tests of FrameworkAgent's decision logic (psuedocode.txt branches) with fakes: no simulator, no torch.

  python test_supervisor.py        (any Python >= 3.8 with numpy)
"""

import unittest

import config as C

C.ensure_paths()
from fa_core.loop import CONTINUE, Continue, Override, Stop  # noqa: E402

from supervisor import FrameworkAgent  # noqa: E402

EVERY = 4                     # 5 fps
T0 = 10                       # first agent call (after the 10 wait steps)


class View(object):
    def __init__(self, step):
        self.step, self.policy_step = step, step - T0
        self.obs, self.image, self.last_action = {}, None, None


class Motion(object):
    def describe(self):
        return {"type": "fake"}


class FakeVLM(object):
    def __init__(self):
        self.kind, self.obj, self.idx = "pick", "black bowl", 0
        self.hl, self.decision, self._held = False, CONTINUE, False
        self.discarded, self.external, self.motion_active = 0, [], False

    def reset(self, task, prompt):
        pass

    def act(self, view):
        if isinstance(self.decision, Override):
            self.motion_active = True
        return self.decision

    def subtask(self):
        return self.idx, self.kind, self.obj

    def hl_failure(self):
        return self.hl

    def failure(self):
        return {"failure": "high_level", "failure_mode": "target_not_approached", "library": "HL-7",
                "action": "point_approach", "reason": "test"} if self.hl else None

    def held(self):
        return self._held

    def external_correction_done(self, step):
        self.external.append(step)

    def discard_correction(self):
        if not self.motion_active:
            return None
        self.motion_active = False
        self.discarded += 1
        return {"action": "point_approach"}

    def summary(self):
        return {"subtask": "{}({})".format(self.kind, self.obj)}


class FakePer(object):
    def __init__(self):
        self.gate_open, self.grounded, self.located, self.visible = True, True, True, True

    def gate(self, view):
        return self.gate_open, {"pick": True, "open": self.gate_open}

    def v_obs(self, view):
        return {"obs": view.step} if self.grounded else None

    def handoff(self, view):
        return (Motion(), {"q_initial_position": [0.0, 0.0, 1.0]}) if self.located else None

    def q_obs(self, view):
        return {"valid": True} if self.visible else None

    def q_move(self, view, action):
        return Motion(), {"goal": [0.0, 0.0, 1.0]}


class FakeV(object):
    def __init__(self, values):
        self.values, self.calls = list(values), 0

    def predict(self, obs):
        v = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return v, 0.0


class FakeQ(object):
    def select(self, obs):
        return 0, {"k": 5, "row": 1, "col": 2, "yaw_deg": 5.0, "q": 0.7}


def make(values=(0.8,), **kw):
    vlm, per, v = FakeVLM(), FakePer(), FakeV(values)
    ag = FrameworkAgent(vlm, per, v, FakeQ(), **kw)
    ag.reset("put the black bowl on the plate", "put the black bowl on the plate")
    return ag, vlm, per, v


def is_q(d, phase):
    return isinstance(d, Override) and d.reason.startswith("Q[") and phase in d.reason


class TestFramework(unittest.TestCase):
    def test_every_steps_is_5fps(self):
        ag, _, _, _ = make()
        self.assertEqual(ag.every_steps, EVERY)

    def test_v_low_three_frames_triggers_q(self):
        ag, vlm, per, v = make([0.2])
        self.assertIsInstance(ag.act(View(10)), Continue)
        self.assertIsInstance(ag.act(View(14)), Continue)
        d = ag.act(View(18))
        self.assertTrue(is_q(d, "approach Q_INITIAL_POSITION"), d)
        self.assertEqual(ag.corrections[-1]["trigger"], "v_low")
        d = ag.act(View(60))                     # approach finished -> argmax Q -> move
        self.assertTrue(is_q(d, "argmax cell (1, 2)"), d)
        n = v.calls
        self.assertIsInstance(ag.act(View(80)), Continue)     # move finished: correction ends, cooldown starts
        self.assertEqual(vlm.external, [80])
        for s in range(84, 200, EVERY):                      # Qmax issued: V never checked again, pi0 carries on
            self.assertIsInstance(ag.act(View(s)), Continue, s)
            self.assertEqual(ag.calls[-1]["branch"], "q_issued")
        self.assertEqual(v.calls, n)

    def test_v_rearmed_on_new_pick_subtask_after_q(self):
        ag, vlm, _, v = make([0.2])
        for s in (10, 14, 18):
            ag.act(View(s))
        ag.act(View(60))                         # argmax Q issued
        ag.act(View(80))                         # move done
        n = v.calls
        ag.act(View(84))
        self.assertEqual(v.calls, n)
        vlm.kind, vlm.idx = "place", 1
        ag.act(View(100))
        vlm.kind, vlm.idx = "pick", 0            # grasp lost: back to the pick -> V checked again
        ag.act(View(150))
        self.assertEqual(v.calls, n + 1)
        self.assertEqual(ag.calls[-1]["low_count"], 1)

    def test_timeout_still_fires_after_q_issued(self):
        ag, _, _, v = make([0.2])
        for s in (10, 14, 18):
            ag.act(View(s))
        ag.act(View(60))
        ag.act(View(80))
        n = v.calls
        for s in range(84, 210, EVERY):
            self.assertIsInstance(ag.act(View(s)), Continue)
        self.assertTrue(is_q(ag.act(View(210)), "Q[timeout]"))
        self.assertEqual(v.calls, n)

    def test_q_skipped_at_handoff_keeps_v_armed(self):
        ag, _, per, v = make([0.2])
        per.visible = False
        for s in (10, 14, 18):
            ag.act(View(s))
        ag.act(View(50))                         # skipped: no argmax issued
        self.assertFalse(ag.q_issued)

    def test_high_v_resets_count(self):
        ag, _, _, _ = make([0.2, 0.2, 0.9, 0.2, 0.2, 0.2])
        ds = [ag.act(View(T0 + EVERY * i)) for i in range(6)]
        self.assertTrue(all(isinstance(d, Continue) for d in ds[:5]))
        self.assertTrue(is_q(ds[5], "approach"))

    def test_not_close_breaks_the_streak(self):
        ag, _, per, v = make([0.2])
        ag.act(View(10))
        ag.act(View(14))
        per.gate_open = False
        self.assertIsInstance(ag.act(View(18)), Continue)
        self.assertEqual(ag.calls[-1]["branch"], "not_close")
        per.gate_open = True
        ag.act(View(22))
        ag.act(View(26))
        self.assertTrue(is_q(ag.act(View(30)), "approach"))
        self.assertEqual(v.calls, 5)

    def test_not_grounded_is_not_close(self):
        ag, _, per, v = make([0.2])
        per.grounded = False
        for i in range(5):
            self.assertIsInstance(ag.act(View(T0 + EVERY * i)), Continue)
        self.assertEqual(v.calls, 0)

    def test_hl_correction_passes_through_and_skips_v(self):
        ag, vlm, _, v = make([0.2])
        vlm.hl, vlm.decision = True, Override(Motion(), reason="point_approach: target_not_approached (HL-7)")
        d = ag.act(View(10))
        self.assertIs(d, vlm.decision)
        self.assertEqual(v.calls, 0)
        self.assertEqual(ag.corrections[-1]["type"], "HL")
        self.assertEqual(ag.corrections[-1]["library_id"], "HL-7")
        vlm.hl, vlm.decision = False, CONTINUE
        ag.act(View(110))                     # HL motion ended -> cooldown
        self.assertEqual(ag.corrections[-1]["end_step"], 110)
        self.assertEqual(vlm.external, [])    # the VLM tracks its own corrections

    def test_suppressed_hl_does_not_run_v(self):
        ag, vlm, _, v = make([0.2])
        vlm.hl = True                         # failure confirmed but no correction allowed (cooldown)
        self.assertIsInstance(ag.act(View(10)), Continue)
        self.assertEqual(v.calls, 0)
        self.assertEqual(ag.calls[-1]["branch"], "hl")

    def test_timeout_fires_q_and_resets_timer2(self):
        ag, _, _, v = make([0.9])
        for s in range(10, 210, EVERY):
            self.assertIsInstance(ag.act(View(s)), Continue)
        d = ag.act(View(210))                 # timer2 = 10 s
        self.assertTrue(is_q(d, "Q[timeout]"), d)
        ag.act(View(250))                     # approach done -> move
        ag.act(View(270))                     # move done -> timer2 restarts here
        self.assertEqual(ag.pick_start, 270)
        self.assertIsInstance(ag.act(View(274)), Continue)
        self.assertEqual(ag.calls[-1]["timer2_s"], 0.2)

    def test_timeout_preempts_hl(self):
        ag, vlm, _, _ = make([0.9])
        for s in range(10, 210, EVERY):
            ag.act(View(s))
        vlm.hl, vlm.decision = True, Override(Motion(), reason="point_approach: x")
        d = ag.act(View(210))
        self.assertTrue(is_q(d, "Q[timeout]"))
        self.assertEqual(vlm.discarded, 1)

    def test_timeout_waits_for_a_closed_gripper(self):
        ag, vlm, per, _ = make([0.9])
        for s in range(10, 210, EVERY):
            ag.act(View(s))
        vlm._held, per.gate_open = True, False
        for s in range(210, 250, EVERY):      # 2 s grace
            self.assertIsInstance(ag.act(View(s)), Continue, s)
        self.assertTrue(is_q(ag.act(View(250)), "Q[timeout]"))

    def test_other_subtask_hl_only(self):
        ag, vlm, _, v = make([0.2])
        vlm.kind, vlm.idx = "place", 1
        for s in range(10, 400, EVERY):
            self.assertIsInstance(ag.act(View(s)), Continue)
        self.assertEqual(v.calls, 0)
        vlm.hl, vlm.decision = True, Override(Motion(), reason="carry_to: wrong_destination (HL-6)")
        self.assertIs(ag.act(View(400)), vlm.decision)

    def test_pick_of_non_round_rim_object_is_other_subtask(self):
        ag, vlm, _, v = make([0.2])
        vlm.obj = "ketchup"
        for s in range(10, 300, EVERY):
            self.assertIsInstance(ag.act(View(s)), Continue)
        self.assertEqual(v.calls, 0)

    def test_back_to_pick_restarts_timer2(self):
        ag, vlm, _, _ = make([0.9])
        ag.act(View(10))
        vlm.kind, vlm.idx = "place", 1
        ag.act(View(100))
        self.assertIsNone(ag.pick_start)
        vlm.kind, vlm.idx = "pick", 0         # grasp lost: the VLM goes back to the pick
        ag.act(View(150))
        self.assertEqual(ag.pick_start, 150)

    def test_q_skipped_when_target_not_located(self):
        ag, _, per, _ = make([0.9])
        per.located = False
        for s in range(10, 210, EVERY):
            ag.act(View(s))
        self.assertIsInstance(ag.act(View(210)), Continue)
        self.assertIn("skipped", ag.corrections[-1])
        self.assertEqual(ag.pick_start, 210)  # timer2 restarts: no retry on every call

    def test_q_skipped_when_not_visible_at_handoff(self):
        ag, vlm, per, _ = make([0.2])
        per.visible = False
        ag.act(View(10))
        ag.act(View(14))
        self.assertTrue(is_q(ag.act(View(18)), "approach"))
        self.assertIsInstance(ag.act(View(50)), Continue)
        self.assertIn("skipped", ag.corrections[-1])
        self.assertEqual(vlm.external, [50])
        self.assertIsNone(ag.pending)

    def test_vlm_never_ends_the_episode(self):
        # task success is LIBERO's call: a VLM Stop is logged and ignored, pi0 continues
        ag, vlm, _, _ = make([0.9])
        vlm.kind = "place"
        vlm.decision = Stop("VLM: task complete")
        self.assertIsInstance(ag.act(View(10)), Continue)
        self.assertEqual(ag.calls[-1]["vlm_stop_ignored"], "VLM: task complete")
        self.assertNotIn("vlm_done_step", ag.report())

    def test_error_in_perception_lets_pi0_continue(self):
        ag, _, per, _ = make([0.2])

        def boom(view):
            raise RuntimeError("render failed")

        per.gate = boom
        self.assertIsInstance(ag.act(View(10)), Continue)
        self.assertIn("error", ag.calls[-1])

    def test_report(self):
        ag, _, _, _ = make([0.2])
        for s in (10, 14, 18):
            ag.act(View(s))
        ag.act(View(50))
        ag.act(View(70))
        r = ag.report()
        self.assertEqual((r["q_corrections"], r["q_v_low"], r["v_evals"], r["v_low_frames"]), (1, 1, 3, 3))


if __name__ == "__main__":
    unittest.main(verbosity=1)
