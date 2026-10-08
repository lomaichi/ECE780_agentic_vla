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
#   cd /opt/openpi && uv run scripts/serve_policy.py --env LIBERO
    ## this by default loads pi0.5 instead of pi0. the default checkpoint is : 
    ## EnvMode.LIBERO: Checkpoint(
    ## config="pi05_libero",
    ## dir="gs://openpi-assets/checkpoints/pi05_libero"),
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

# LIBERO asks for its paths interactively on first import unless a config exists
ENV LIBERO_CONFIG_PATH=/opt/libero-config
RUN mkdir -p "${LIBERO_CONFIG_PATH}" && B=/opt/openpi/third_party/libero/libero/libero && printf '%s\n' \
    "benchmark_root: ${B}" \
    "bddl_files: ${B}/bddl_files" \
    "init_states: ${B}/init_files" \
    "datasets: ${B}/../datasets" \
    "assets: ${B}/assets" \
    > "${LIBERO_CONFIG_PATH}/config.yaml"

# Checkpoints (several GB) go to the mounted share so they persist across --rm runs
ENV PYTHONPATH=/opt/openpi/third_party/libero \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    OPENPI_DATA_HOME=/home/DockerShared/.cache/openpi

WORKDIR /home/DockerShared

CMD ["bash"]
