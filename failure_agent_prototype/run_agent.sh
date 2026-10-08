#!/bin/bash
# Run the failure agent inside the LIBERO client venv.
# Requires the pi0 policy server on port 8000 (see README.md).
#   ./run_agent.sh --tasks 62 71 --mode both
HERE="$(cd "$(dirname "$0")" && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
exec /opt/openpi/examples/libero/.venv/bin/python "$HERE/run_agent.py" "$@"
