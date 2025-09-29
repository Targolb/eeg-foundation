# Build with:
#   CUDA 12.x drivers → BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
#   CUDA 11.x drivers → BASE=pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
ARG BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM ${BASE}

# Reproducible, non-interactive, headless plotting
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (expects requirements.txt in build context)
WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /root/.cache/pip

# Make your repo the default working directory
WORKDIR /workspace/eeg-foundation

# Optional: help Python find your package modules
ENV PYTHONPATH=/workspace/eeg-foundation

# Default shell
CMD ["/bin/bash"]
