#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PY = HERE / ".venv" / "bin" / "python3"
if not PY.exists():
    PY = HERE / ".venv" / "bin" / "python"

def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=HERE)

def main() -> int:
    run([sys.executable, "-m", "venv", str(HERE / ".venv")])
    run([str(PY), "-m", "ensurepip", "--upgrade"])
    run([str(PY), "-m", "pip", "install", "-q", "-r", str(HERE / "requirements.txt")])
    run([
        str(PY), "-m", "streamlit", "run", str(HERE / "app.py"),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())