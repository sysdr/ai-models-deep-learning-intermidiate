#!/usr/bin/env bash
# Prune local dev junk under this directory, then aggressively clean Docker resources.
# Docker steps stop ALL running containers and remove unused images/volumes (destructive).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "== Local cleanup: $SCRIPT_DIR =="

while IFS= read -r -d '' d; do
  echo "  removing: $d"
  rm -rf "$d"
done < <(find "$SCRIPT_DIR" -type d -name node_modules -print0 2>/dev/null)

while IFS= read -r -d '' d; do
  echo "  removing: $d"
  rm -rf "$d"
done < <(find "$SCRIPT_DIR" -type d \( -name venv -o -name .venv -o -name .pytest_cache \) -print0 2>/dev/null)

while IFS= read -r -d '' d; do
  echo "  removing: $d"
  rm -rf "$d"
done < <(find "$SCRIPT_DIR" -type d -name __pycache__ -print0 2>/dev/null)

find "$SCRIPT_DIR" -type f -name '*.pyc' -print -delete 2>/dev/null || true

while IFS= read -r -d '' p; do
  echo "  removing (istio-related): $p"
  rm -rf "$p"
done < <(find "$SCRIPT_DIR" \( -iname '*istio*' \) -print0 2>/dev/null)

echo "== Docker cleanup (stops all containers; prune --volumes removes unused volumes) =="

if ! command -v docker >/dev/null 2>&1; then
  echo "  docker not installed; skipping Docker steps."
  exit 0
fi

if docker info >/dev/null 2>&1; then
  RUNNING="$(docker ps -q 2>/dev/null || true)"
  if [[ -n "${RUNNING:-}" ]]; then
    echo "  stopping containers: $RUNNING"
    docker stop $RUNNING
  else
    echo "  no running containers"
  fi
  docker system prune -af --volumes
  docker builder prune -af 2>/dev/null || true
  echo "  Docker prune finished."
else
  echo "  Docker daemon not reachable (permission or not running); skipping Docker steps."
fi

echo "cleanup.sh finished."
