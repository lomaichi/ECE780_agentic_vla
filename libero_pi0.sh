#!/bin/bash
# Docker launcher for the libero_pi0:v0 image (pi0 + LIBERO).
# Based on mujoco_robosuite.sh: same X11 display, camera, and GPU access.
#
# Host: laptop, RTX 5070 Laptop GPU (8 GB, Blackwell sm_120) + Intel iGPU
# (hybrid graphics, so on-screen GL uses PRIME render offload), Docker with the
# nvidia CDI spec. Originally written for rootless podman on an RTX 5090 (32 GB).
# Same GPU architecture, so the image (JAX 0.5.3, torch cu128) runs unchanged.
#
# 8 GB VRAM: pi0 in bf16 needs about 7 GB, and JAX by default preallocates 75%
# of VRAM (~6 GB), which is too little. XLA_* below turn off preallocation and
# let JAX use up to 95%. Do not run another GPU model (VLM, SimpleVLA-RL, ...)
# alongside the pi0 server; it will not fit.
#
# Docker runs the container as real root (podman rootless mapped root to your
# uid), so files the container writes under ~/DockerShared are root-owned. Fix:
#   sudo chown -R "$USER": ~/DockerShared/ECE780_agentic_vla
#
# Usage: ./libero_pi0.sh [container-name]   (default: robomomo)
# First run starts the container; running it again while the container is up
# opens another shell in it instead.
#
# Inside the container:
#   shell 1 (pi0 policy server):
#     ./serve_pi0.sh
#     (do NOT use `serve_policy.py --env LIBERO`: that serves pi0.5, not pi0)
#   shell 2 (open by running this script again):
#     cd /opt/openpi && source examples/libero/.venv/bin/activate
#     python examples/libero/main.py --args.task-suite-name libero_spatial
#
# Rendering defaults to headless EGL (MUJOCO_GL=egl). For an on-screen
# MuJoCo viewer window, run with MUJOCO_GL=glx instead.

CONTAINER="${1:-robomomo}"

# The project code uses /home/DockerShared/ECE780 as the repo path. This checkout
# is in a different folder, so it is also mounted at that path (see REPO_ARGS).
REPO_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
REPO_ARGS=()
if [ "$REPO_DIR" != "$HOME/DockerShared/ECE780" ]; then
	# Create the mount point as you; otherwise docker creates it as root
	mkdir -p "$HOME/DockerShared/ECE780"
	REPO_ARGS+=(-v "$REPO_DIR:/home/DockerShared/ECE780")
fi

# Allow the container (running as root) to open windows on your X display
xhost +SI:localuser:root >/dev/null
XAUTH_ARGS=()
if [ -n "$XAUTHORITY" ] && [ -f "$XAUTHORITY" ]; then
	XAUTH_ARGS=(-v "$XAUTHORITY:/tmp/.docker.xauth:ro" --env=XAUTHORITY=/tmp/.docker.xauth)
fi

# Shell setup shared by the first and any additional shells
INIT_CMD="
	export XDG_RUNTIME_DIR=/tmp/runtime-root
	mkdir -p \$XDG_RUNTIME_DIR
	chmod 700 \$XDG_RUNTIME_DIR
	ldconfig

	cd /home/DockerShared/ECE780

	exec bash
"

# Container already running: just open a new shell in it
if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; then
	exec docker exec -it \
		--env=DISPLAY \
		"$CONTAINER" \
		bash -c "$INIT_CMD"
fi

CAM_DEVICES=()
for dev in /dev/video* /dev/media*; do
	[ -e "$dev" ] && CAM_DEVICES+=(--device="$dev:$dev")
done

# pi0 checkpoints are cached here (OPENPI_DATA_HOME in the image); create it
# on the host first so it is owned by you
mkdir -p "$HOME/DockerShared/.cache/openpi"

# GPU access: use the NVIDIA CDI spec (/var/run/cdi/nvidia.yaml on this host,
# which also passes the dGPU's /dev/dri nodes and EGL/GLX libraries). Otherwise
# fall back to the nvidia runtime.
GPU_ARGS=()
if ls /etc/cdi/nvidia*.yaml /var/run/cdi/nvidia*.yaml >/dev/null 2>&1; then
	GPU_ARGS+=(--device=nvidia.com/gpu=all)
else
	GPU_ARGS+=(--gpus all --runtime=nvidia --env=NVIDIA_VISIBLE_DEVICES=all)
fi

docker run -it \
	--rm \
	--network=host \
	--ipc=host \
	"${GPU_ARGS[@]}" \
	--device=/dev/dri:/dev/dri \
	"${CAM_DEVICES[@]}" \
	--env=DISPLAY \
	--env=QT_X11_NO_MITSHM=1 \
	--env=NVIDIA_DRIVER_CAPABILITIES=all \
	--env=__GLX_VENDOR_LIBRARY_NAME=nvidia \
	--env=__NV_PRIME_RENDER_OFFLOAD=1 \
	--env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
	--env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
	"${XAUTH_ARGS[@]}" \
	--env=WANDB_API_KEY=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\
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v terminator-config:/etc/xdg/terminator \
	-v claude-config:/root/.claude-config \
	-v "$HOME/DockerShared:/home/DockerShared" \
	"${REPO_ARGS[@]}" \
	--name="$CONTAINER" \
	libero_pi0:v0 \
	bash -c "$INIT_CMD"
