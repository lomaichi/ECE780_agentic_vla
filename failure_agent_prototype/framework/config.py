"""Constants of psuedocode.txt, the few additions the implementation needs, and default paths.

Times are SIMULATION time (control steps / 20 Hz), so a slow VLM call does not eat into a timeout.
Python 3.8 compatible (the LIBERO client venv).
"""

import pathlib
import sys

FRAMEWORK = pathlib.Path(__file__).resolve().parent
FAP = FRAMEWORK.parent                                    # failure_agent_prototype/
PIXQ = FAP / "RL" / "pixel_Q"
V1_DIR, V2_DIR, V4_DIR = PIXQ / "V1", PIXQ / "V2", PIXQ / "V4"
MOONDREAM_DIR = FAP.parent / "Documentation" / "VLM_inference_speed_test" / "moondreamV2"
LIBRARY = FAP / "failure_survey" / "skill_library" / "library.json"

# ------------------------------------------------------------------ psuedocode.txt -------------------
THRESHOLD = 0.5               # V_est below this = pi0 is unlikely to pick from here
THRESHOLD_COUNT = 3           # consecutive low-V frames before the Q correction runs
TIMEOUT_S = 10.0              # pick(bowl/mug) subtask timeout (timer2)
APPROACH_DZ = 0.07            # Q_INITIAL_POSITION: 7 cm above the target, approach axis along -z
FULL_TASK_TIMEOUT_S = 30.0    # whole episode (timer1)
MONITOR_FPS = 5.0             # the VLM samples the world view at 5 fps; V is evaluated on the same frames

# ------------------------------------------------------------------ additions -------------------------
# V4's gate: no tick within 1 s after a correction (V was never trained on those states). Applied after
# every correction (HL or Q) as part of close_to_object().
CORRECTION_COOLDOWN_S = 0.2
# The timeout correction opens the gripper. If the gripper is closed on something when timer2 runs out
# (pi0 may be lifting the bowl while the pick judge still lags), wait up to this long before firing.
TIMEOUT_HELD_GRACE_S = 2.0

ROUND_RIM = ("bowl", "mug", "cup")   # = shared.scene.ROUND_RIM: the objects the Q-map was trained on

# ------------------------------------------------------------------ models -----------------------------
DEFAULT_Q_CKPT = V1_DIR / "runs" / "pick_v0" / "ckpt_best.pt"
DEFAULT_V_CKPT = V4_DIR / "runs" / "v4_train" / "v_attn.pt"      # written by V4/train_v.py (not trained yet)
V4_GATE_MAX_ABOVE_M = 0.10    # V4 collect.py --gate-max-above-cm 10 (runs/v4/config.json)


def ensure_paths():
    """failure_agent_prototype/ (fa_core) and RL/pixel_Q/ (shared) on sys.path."""
    for p in (FAP, PIXQ):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def is_round_rim(name):
    return name is not None and any(w in str(name).lower() for w in ROUND_RIM)
