"""Parse every LIBERO instruction with moondreamV2 judge_agent.parse_instruction; print one line per task."""
import json
import sys

sys.path.insert(0, "/home/DockerShared/ECE780/Documentation/VLM_inference_speed_test/moondreamV2")
sys.path.insert(0, "/home/DockerShared/ECE780/failure_agent_prototype")
from libero.libero import benchmark  # noqa: E402
import judge_agent as J  # noqa: E402

for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
    s = benchmark.get_benchmark_dict()[suite]()
    for i in range(s.n_tasks):
        t = s.get_task(i).language
        p = J.parse_instruction(t)
        print(suite, i, json.dumps({k: v for k, v in p.items()}, sort_keys=True), "|", t)
