#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-avazu-serving:ci}"

docker build -t "$IMAGE_NAME" .
