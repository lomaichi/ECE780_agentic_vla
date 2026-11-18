#!/bin/bash
# Podman (rootless) launcher for the libero_pi0:v0 image (pi0 + LIBERO).
# Based on mujoco_robosuite.sh: same X11 display, camera, and GPU access.
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

# Allow the container (running as your uid) to open windows on your X display
xhost +SI:localuser:"$USER" >/dev/null

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
if [ "$(podman inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; then
	exec podman exec -it \
		--env=DISPLAY \
		"$CONTAINER" \
		bash -c "$INIT_CMD"
fi

CAM_DEVICES=()
for dev in /dev/video* /dev/media*; do
	[ -e "$dev" ] && CAM_DEVICES+=(--device="$dev")
done

# pi0 checkpoints are cached here (OPENPI_DATA_HOME in the image); create it
# on the host first so it is owned by you
mkdir -p "$HOME/DockerShared/.cache/openpi"

# GPU access: use the NVIDIA CDI spec when the host has one. Otherwise
# (no nvidia-container-toolkit, /etc/cdi empty) pass the device nodes and
# bind-mount the host driver libraries read-only; INIT_CMD runs ldconfig.
GPU_ARGS=()
if ls /etc/cdi/nvidia*.yaml /var/run/cdi/nvidia*.yaml >/dev/null 2>&1; then
	GPU_ARGS+=(--device=nvidia.com/gpu=all)
else
	for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
		[ -e "$dev" ] && GPU_ARGS+=(--device="$dev")
	done
	LIB=/usr/lib/x86_64-linux-gnu
	for f in "$LIB"/libcuda.so* "$LIB"/libnvidia-ml.so* "$LIB"/libnvidia-ptxjitcompiler.so* \
		"$LIB"/libnvidia-nvvm*.so* "$LIB"/libEGL_nvidia.so* "$LIB"/libnvidia-eglcore.so* \
		"$LIB"/libnvidia-glcore.so* "$LIB"/libnvidia-glsi.so* "$LIB"/libnvidia-tls.so* \
		"$LIB"/libnvidia-gpucomp.so* "$LIB"/libGLX_nvidia.so* \
		/usr/share/glvnd/egl_vendor.d/10_nvidia.json /usr/bin/nvidia-smi; do
		[ -e "$f" ] && GPU_ARGS+=(-v "$f:$f:ro")
	done
fi

podman run -it \
	--rm \
	--network=host \
	--ipc=host \
	"${GPU_ARGS[@]}" \
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
	--name="$CONTAINER" \
	libero_pi0:v0 \
	bash -c "$INIT_CMD"
