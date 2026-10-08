#!/bin/bash


CONTAINER="${1:-robomomo}"

# Allow the container (running as your uid) to open windows on your X display
xhost +SI:localuser:"$USER" >/dev/null

podman exec -it \
	--env=DISPLAY \
	"$CONTAINER" \
	bash -c "
		export XDG_RUNTIME_DIR=/tmp/runtime-root
		mkdir -p \$XDG_RUNTIME_DIR
		chmod 700 \$XDG_RUNTIME_DIR

		cd /home/DockerShared/ECE780

		exec bash
    "
