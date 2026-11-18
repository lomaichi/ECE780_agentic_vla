"""Constants, paths and helpers shared by the Phase 1 code.

Everything here matches tools/record_suites.py (which produced all recorded rollouts) and
openpi examples/libero/main.py: the same env construction, image flip, 8-D pi0 state, resize and
replan rate. If pi0 is fed inputs that differ from these it quietly performs worse, so new code
should build pi0 requests only through pi0_request().
"""

import json
import math
import pathlib
import sys

import numpy as np

FA_CORE = pathlib.Path(__file__).resolve().parent
FAP = FA_CORE.parent                                   # failure_agent_prototype/
REPO = FAP.parent
TOOLS = FAP / "tools"
ROLLOUTS = REPO / "Documentation" / "pi0_LIBERO_rollouts"
LIBRARY = FAP / "failure_survey" / "skill_library" / "library.json"
SURVEY_EPISODES = FAP / "failure_survey" / "episodes"

POLICY = "pi0_libero (gs://openpi-assets/checkpoints/pi0_libero), stochastic"
ENV_RESOLUTION = 256          # LIBERO camera resolution used for every recording
RESIZE = 224                  # pi0 input size (resize_with_pad)
REPLAN_STEPS = 5              # actions executed from each pi0 chunk before asking again
NUM_STEPS_WAIT = 10           # dummy steps after reset so objects settle
CONTROL_FREQ = 20             # Hz; one video frame per control step
DUMMY_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
GRIPPER_OPEN, GRIPPER_CLOSE = -1.0, 1.0
MAX_STEPS = {                 # policy steps per suite (excluding the wait steps), as in main.py
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def quat2axisangle(quat):
    """(x, y, z, w) quaternion -> axis-angle. Copied from openpi examples/libero/main.py."""
    quat = np.array(quat, dtype=np.float64, copy=True)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def upright(img):
    """LIBERO renders upside down; rotate 180 deg (what pi0 and the videos see)."""
    return np.ascontiguousarray(img[::-1, ::-1])


def pi0_state(obs):
    """8-D pi0 state: EE position (3), EE axis-angle (3), gripper finger qpos (2)."""
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]))


def pi0_request(obs, prompt, resize=RESIZE):
    """The exact request record_suites.py sends to the pi0 server."""
    from openpi_client import image_tools

    img, wrist = upright(obs["agentview_image"]), upright(obs["robot0_eye_in_hand_image"])
    return {
        "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize)),
        "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize)),
        "observation/state": pi0_state(obs),
        "prompt": str(prompt),
    }


def proprio(obs):
    """Robot proprioception, in the trace field names of record_suites.py (obs_<key>)."""
    return {
        "ee_pos": np.array(obs["robot0_eef_pos"], copy=True),
        "ee_quat": np.array(obs["robot0_eef_quat"], copy=True),
        "gripper_states": np.array(obs["robot0_gripper_qpos"], copy=True),
        "joint_states": np.array(obs["robot0_joint_pos"], copy=True),
    }


def capture_hidden(env):
    """Per-step state that env.get_sim_state() does not contain (needed to restore and continue)."""
    return {
        "gripper_current_action": np.broadcast_to(env.env.robots[0].gripper.current_action, (2,)).copy(),
        "qacc_warmstart": env.env.sim.data.qacc_warmstart.copy(),
    }


def make_env(suite, task_id, env_seed=7, resolution=ENV_RESOLUTION):
    """LIBERO env for one task, built exactly like record_suites.py. Returns (env, task, bddl_path)."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution)
    env.seed(env_seed)  # the seed affects object placement even with a fixed init state
    return env, task, bddl


def init_state(suite, task_id, init_idx):
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[suite]().get_task_init_states(task_id)[init_idx]


def held_out_tasks(library=LIBRARY):
    """libero_90 task ids reserved for the zero-shot test (library.json split_guard)."""
    return set(json.loads(pathlib.Path(library).read_text())["split_guard"]["held_out_tasks"])


def tools_module(name):
    """Import a script from failure_agent_prototype/tools (e.g. scene_restore, record_suites)."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    return __import__(name)
