FROM python:3.10-slim-bookworm

# System dependencies for MuJoCo / robosuite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    linux-libc-dev \
    libglib2.0-0 \
    libgl1 \
    libegl1 \
    libglfw3 \
    libosmesa6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install versions validated together.
# robosuite 1.5.2 requires MuJoCo < 3.10; MuJoCo 3.9.0 was tested successfully.
RUN python3 -m pip install --no-cache-dir \
    "mujoco==3.9.0" \
    "robosuite==1.5.2"

# Create robosuite's private macros.py configuration file
RUN python3 /usr/local/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py

# Claude Code (native installer -> /root/.local/bin/claude)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"
# Keep Claude's login/settings in one directory so it can be mounted as a volume
ENV CLAUDE_CONFIG_DIR=/root/.claude-config

# ---------------------------------------------------------------------------
# pi0 + LIBERO (Physical-Intelligence/openpi)
#
# openpi uses two isolated environments, so neither touches the system
# robosuite 1.5.2 install above:
#   - policy server: /opt/openpi/.venv                (Python 3.11, JAX, pi0)
#   - LIBERO client: /opt/openpi/examples/libero/.venv (Python 3.8, robosuite 1.4.1)
#
# Usage (two shells in the container):
#   /home/DockerShared/ECE780/serve_pi0.sh      (pi0 server, config pi0_libero)
#     NOT `uv run scripts/serve_policy.py --env LIBERO`: that shortcut serves
#     pi0.5 (config pi05_libero, gs://openpi-assets/checkpoints/pi05_libero).
#   cd /opt/openpi && source examples/libero/.venv/bin/activate && \
#       python examples/libero/main.py --args.task-suite-name libero_spatial
# ---------------------------------------------------------------------------

# cmake is needed to build egl_probe (a LIBERO/robomimic dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    ffmpeg \
    libsm6 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

ARG OPENPI_REF=main
RUN git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git /opt/openpi \
    && cd /opt/openpi \
    && git checkout "${OPENPI_REF}" \
    && git submodule update --init --recursive

WORKDIR /opt/openpi

# Policy server environment
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync \
    && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . \
    && uv cache clean

# LIBERO client environment (pins from openpi/examples/libero/requirements.txt)
RUN uv venv --python 3.8 examples/libero/.venv \
    && VIRTUAL_ENV=examples/libero/.venv uv pip sync \
        examples/libero/requirements.txt third_party/libero/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
        --index-strategy=unsafe-best-match \
    && VIRTUAL_ENV=examples/libero/.venv uv pip install -e packages/openpi-client \
    && VIRTUAL_ENV=examples/libero/.venv uv pip install -e third_party/libero \
    && uv cache clean

# Newer wandb in the LIBERO client venv: the pinned 0.13.1 rejects current
# (86-char) W&B API keys ("must be 40 characters long"). 0.24.2 is the last
# release supporting Python 3.8; keeps protobuf 3.20.3, bumps sentry-sdk to 2.x.
RUN VIRTUAL_ENV=examples/libero/.venv uv pip install "wandb==0.24.2" \
    && uv cache clean

# LIBERO asks for its paths interactively on first import unless a config exists
ENV LIBERO_CONFIG_PATH=/opt/libero-config
RUN mkdir -p "${LIBERO_CONFIG_PATH}" && B=/opt/openpi/third_party/libero/libero/libero && printf '%s\n' \
    "benchmark_root: ${B}" \
    "bddl_files: ${B}/bddl_files" \
    "init_states: ${B}/init_files" \
    "datasets: ${B}/../datasets" \
    "assets: ${B}/assets" \
    > "${LIBERO_CONFIG_PATH}/config.yaml"

# ---------------------------------------------------------------------------
# SimpleVLA-RL (OpenVLA-OFT RL on LIBERO), isolated venv /opt/simplevla/.venv:
#   - torch 2.7.1+cu128 (RTX 5090 / sm_120 needs CUDA >= 12.8)
#   - robosuite 1.4.1 for LIBERO (system Python keeps robosuite 1.5.2)
#   - no vllm (pins torch 2.4; LIBERO script uses rollout.name=hf)
# See SimpleVLA-RL/DEPENDENCY_CHANGES.md.
#
# Run the container with --pids-limit=-1: Ray exceeds the default 2048 pids
# (the RAY_*prestart* env vars in the run script are only a workaround).
# ---------------------------------------------------------------------------
COPY SimpleVLA-RL/requirements-5090.lock.txt /tmp/simplevla-requirements.txt
RUN uv venv --python 3.10 /opt/simplevla/.venv \
    && VIRTUAL_ENV=/opt/simplevla/.venv uv pip install -r /tmp/simplevla-requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match \
    && VIRTUAL_ENV=/opt/simplevla/.venv uv pip install --no-deps -e /opt/openpi/third_party/libero \
    && rm /tmp/simplevla-requirements.txt \
    && uv cache clean

# Checkpoints (several GB) go to the mounted share so they persist across --rm runs
ENV PYTHONPATH=/opt/openpi/third_party/libero \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    OPENPI_DATA_HOME=/home/DockerShared/.cache/openpi

WORKDIR /home/DockerShared

CMD ["bash"]
