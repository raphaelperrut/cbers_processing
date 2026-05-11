# docker/Dockerfile
# Multi-target: cpu and gpu

#############################
# Common: project copy
#############################
FROM python:3.11-slim AS base-py

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/cbers_colorize:/app

WORKDIR /app

# System deps for rasterio/gdal/proj
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    libproj-dev \
    libgeos-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install "numpy==1.26.4"

# Copy requirements and install (common)
COPY requirements.txt /app/requirements.txt


#############################
# CPU target
#############################
FROM base-py AS cpu

# Torch CPU
RUN python -m pip install \
    "torch==2.2.2" \
    --index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install -r /app/requirements.txt

COPY . /app
RUN python -m pip install -e .

ENTRYPOINT ["python", "-m", "cbers_colorize.cli"]


#############################
# GPU target
#############################
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 + pip + build deps + GDAL stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    python3-dev \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    libproj-dev \
    libgeos-dev \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make "python" command exist (your pipeline currently calls "python")
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/cbers_colorize:/app

WORKDIR /app

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Pin numpy before torch (avoid numpy 2.x)
RUN python -m pip install "numpy==1.26.4"

# Torch CUDA (ajuste o índice conforme tua versão desejada)
# Para torch 2.2.2 + CUDA 12.1 wheels:
RUN python -m pip install \
    "torch==2.2.2" \
    --index-url https://download.pytorch.org/whl/cu121

# Requirements
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# Project
COPY . /app
RUN python -m pip install -e .

ENTRYPOINT ["python", "-m", "cbers_colorize.cli"]