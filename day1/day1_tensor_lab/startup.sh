#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LESSON="$SCRIPT_DIR"
APP="$LESSON/app.py"
VENV_ST="$LESSON/.venv/bin/streamlit"

if [[ ! -f "$APP" ]]; then
  echo "error: Streamlit app missing: $APP (run ./build.py)" >&2
  exit 1
fi

cd "$LESSON"

if [[ -x "$VENV_ST" ]]; then
  exec "$VENV_ST" run "$APP" --server.headless true --browser.serverAddress localhost
fi

if command -v streamlit >/dev/null 2>&1; then
  echo "warn: using streamlit on PATH (venv missing or incomplete): $(command -v streamlit)" >&2
  exec streamlit run "$APP" --server.headless true --browser.serverAddress localhost
fi

echo "error: no streamlit. Create venv: python3 -m venv \"$LESSON/.venv\" && \"$LESSON/.venv/bin/pip\" install -r \"$LESSON/requirements.txt\"" >&2
exit 1
