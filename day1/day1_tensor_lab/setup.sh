#!/usr/bin/env bash
set -euo pipefail
DAY1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$DAY1/setup.py" "$@"
