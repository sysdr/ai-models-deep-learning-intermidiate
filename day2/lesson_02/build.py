#!/usr/bin/env python3
"""
build.py — regenerate lesson_02 files from day2/setup.py.
Run from anywhere:
  python /abs/path/to/day2/lesson_02/build.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    lesson_dir = Path(__file__).resolve().parent
    setup_py = lesson_dir.parent / "setup.py"
    if not setup_py.is_file():
        print(f"ERROR: setup.py not found at {setup_py}", file=sys.stderr)
        return 1

    print(f"Running generator: {setup_py}")
    result = subprocess.run([sys.executable, str(setup_py)], check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
