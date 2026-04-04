#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [[ ! -d "$HERE/.venv" ]]; then
    python3 -m venv "$HERE/.venv"
fi
PY="$HERE/.venv/bin/python3"
[[ -x "$PY" ]] || PY="$HERE/.venv/bin/python"
"$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$PY" -m pip install -q -r "$HERE/requirements.txt"
exec "$PY" -m streamlit run "$HERE/app.py"         --server.headless true         --browser.gatherUsageStats false