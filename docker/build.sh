#!/usr/bin/env bash
# Build the sandbox Docker image.
# Run from the repo root: bash docker/build.sh

set -euo pipefail

IMAGE_NAME="swebench-sandbox:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building $IMAGE_NAME ..."
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
echo "Done. Image: $IMAGE_NAME"
