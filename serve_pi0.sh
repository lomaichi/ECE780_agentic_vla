#!/bin/bash
# Start the pi0 LIBERO policy server (pi0_libero), inside the libero_pi0 container.
#
# Use this instead of `uv run scripts/serve_policy.py --env LIBERO`: that shortcut
# serves pi0.5 (pi05_libero), NOT pi0. All rollouts in Documentation/pi0_LIBERO_rollouts
# and every test in failure_agent_prototype assume pi0_libero, so results from a
# pi0.5 server are not comparable with them.
#
# Usage (shell 1 of the container, see libero_pi0.sh):
#   ./serve_pi0.sh
# Wait for the log to show the server listening on 0.0.0.0:8000, then start the
# LIBERO client in a second shell. First run downloads ~12 GB to $OPENPI_DATA_HOME.

set -euo pipefail

cd /opt/openpi
exec uv run scripts/serve_policy.py policy:checkpoint \
	--policy.config=pi0_libero \
	--policy.dir=gs://openpi-assets/checkpoints/pi0_libero
