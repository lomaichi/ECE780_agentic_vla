#!/bin/bash
# Podman (rootless) launcher for the libero_pi0:v0 image (pi0 + LIBERO).
# Based on mujoco_robosuite.sh: same X11 display, camera, and GPU access.
#
# Inside the container:
#   shell 1 (policy server):
#     cd /opt/openpi && uv run scripts/serve_policy.py --env LIBERO
#   shell 2 (open with: podman exec -it libero_pi0 bash):
#     cd /opt/openpi && source examples/libero/.venv/bin/activate
#     python examples/libero/main.py --args.task-suite-name libero_spatial
#
# Rendering defaults to headless EGL (MUJOCO_GL=egl). For an on-screen
# MuJoCo viewer window, run with MUJOCO_GL=glx instead.

# Allow the container (running as your uid) to open windows on your X display
xhost +SI:localuser:"$USER" >/dev/null

CAM_DEVICES=()
for dev in /dev/video* /dev/media*; do
	[ -e "$dev" ] && CAM_DEVICES+=(--device="$dev")
done

# pi0 checkpoints are cached here (OPENPI_DATA_HOME in the image); create it
# on the host first so it is owned by you
mkdir -p "$HOME/DockerShared/.cache/openpi"

podman run -it \
	--rm \
	--network=host \
	--ipc=host \
	--device=nvidia.com/gpu=all \
	--device=/dev/dri \
	"${CAM_DEVICES[@]}" \
	--env=DISPLAY \
	--env=QT_X11_NO_MITSHM=1 \
	--env=NVIDIA_DRIVER_CAPABILITIES=all \
	--env=__GLX_VENDOR_LIBRARY_NAME=nvidia \
	--env=__NV_PRIME_RENDER_OFFLOAD=1 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v terminator-config:/etc/xdg/terminator \
	-v claude-config:/root/.claude-config \
	-v "$HOME/DockerShared:/home/DockerShared" \
	--name=robomomo \
	libero_pi0:v0 \
	bash -c "
		export XDG_RUNTIME_DIR=/tmp/runtime-root
		mkdir -p \$XDG_RUNTIME_DIR
		chmod 700 \$XDG_RUNTIME_DIR

		cd /home/DockerShared/ECE780

		exec bash
    "
