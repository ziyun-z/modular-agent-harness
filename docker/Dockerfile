FROM python:3.11-slim

# System tools needed for SWE-bench repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    patch \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Common Python test/build tools pre-installed to speed up task setup
RUN pip install \
    pytest \
    pytest-xdist \
    pytest-cov \
    tox \
    setuptools \
    wheel \
    build

WORKDIR /workspace

# Configure git (needed for git apply inside containers)
RUN git config --global user.email "agent@harness.local" && \
    git config --global user.name "Agent Harness" && \
    git config --global advice.detachedHead false

CMD ["sleep", "infinity"]
