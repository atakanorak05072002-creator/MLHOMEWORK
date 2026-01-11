#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-avazu-serving:ci}"
CONTAINER_NAME="${CONTAINER_NAME:-avazu-serving-ci}"
PORT="${PORT:-8000}"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "$CONTAINER_NAME" -p "$PORT:8000" "$IMAGE_NAME" >/dev/null

healthy=0
for _ in $(seq 1 20); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null; then
    healthy=1
    break
  fi
  sleep 1
done

if [ "$healthy" -ne 1 ]; then
  echo "Service did not become healthy in time" >&2
  exit 1
fi

curl -fsS -X POST "http://localhost:$PORT/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":{"site_id":"s1","app_id":"a1","site_domain":"sd1","app_domain":"ad1","device_type":"d1","device_conn_type":"dc1"}}' \
  >/dev/null

echo "Smoke test OK"
