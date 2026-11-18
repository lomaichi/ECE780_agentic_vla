#!/usr/bin/env bash
# Start the interactive window. Arguments go to interactive_pi0.py (./run.sh --help lists them).
# window.py runs on Python 3.11 (its Tk works on this X server; the LIBERO venv's Tk aborts) and
# launches the simulation in the LIBERO client venv (Python 3.8).
HERE="$(cd "$(dirname "$0")" && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec /opt/uv-python/cpython-3.11.17-linux-x86_64-gnu/bin/python3 "$HERE/window.py" "$@"
