#!/usr/bin/env python3
"""
start.py — launch the lesson_02 Streamlit dashboard.
Run from anywhere:
  python /abs/path/to/day2/lesson_02/start.py

If Streamlit is not installed in the lesson venv yet, this script runs
`pip install -r requirements.txt` once (can be slow on /mnt/c drives in WSL).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path


def _pick_port(start_port: int = 8502, end_port: int = 8510) -> int:
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range [{start_port}, {end_port}]")


def _streamlit_import_ok(python_exec: str, lesson_dir: Path) -> bool:
    r = subprocess.run(
        [python_exec, "-c", "import streamlit, plotly"],
        capture_output=True,
        text=True,
        cwd=str(lesson_dir),
        check=False,
    )
    return r.returncode == 0


def _ensure_dependencies(python_exec: str, lesson_dir: Path) -> int:
    if _streamlit_import_ok(python_exec, lesson_dir):
        return 0

    req = lesson_dir / "requirements.txt"
    if not req.is_file():
        print(f"ERROR: {req} not found — cannot install Streamlit.", file=sys.stderr)
        return 1

    print(
        "Streamlit/Plotly not found in this Python environment.\n"
        "Installing from requirements.txt …\n"
        "(On WSL, installs under /mnt/c/ can take several minutes — "
        "a Linux-native copy of the repo is faster.)",
        file=sys.stderr,
    )
    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    install = subprocess.run(
        [python_exec, "-m", "pip", "install", "-r", str(req)],
        cwd=str(lesson_dir),
        env=env,
        check=False,
    )
    if install.returncode != 0:
        return int(install.returncode)
    if not _streamlit_import_ok(python_exec, lesson_dir):
        print("ERROR: pip finished but `import streamlit` still fails.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    lesson_dir = Path(__file__).resolve().parent
    app_py = lesson_dir / "app.py"
    if not app_py.is_file():
        print(f"ERROR: app.py not found at {app_py}", file=sys.stderr)
        return 1

    py = lesson_dir / ".venv" / "bin" / "python"
    python_exec = str(py) if py.is_file() else sys.executable

    dep = _ensure_dependencies(python_exec, lesson_dir)
    if dep != 0:
        return dep

    port = _pick_port()
    cmd = [
        python_exec,
        "-m",
        "streamlit",
        "run",
        str(app_py),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.port",
        str(port),
    ]
    print("Starting dashboard:")
    print(" ".join(cmd))
    print(f"Open in browser: http://localhost:{port}")
    completed = subprocess.run(cmd, check=False, cwd=str(lesson_dir))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
