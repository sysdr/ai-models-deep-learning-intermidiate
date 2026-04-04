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
    run([str(PY), "-m", "pip", "install", "-r", str(HERE / "requirements.txt")])
    run([str(PY), str(HERE / "test_model.py")])
    print("\nBuild finished successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())