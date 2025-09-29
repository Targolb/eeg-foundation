ARG BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM ${BASE}

# Avoid prompts, keep things reproducible
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

# Basic OS deps (unzip for Bonn zips, git if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git unzip ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Default workdir inside the container
WORKDIR /workspace/eeg-foundation
