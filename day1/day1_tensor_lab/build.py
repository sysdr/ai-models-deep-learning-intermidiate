#!/usr/bin/env python3
"""Run ../setup.py, verify generated files in this directory, optional venv, tests."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
DAY1 = LAB.parent
SETUP = DAY1 / "setup.py"
LESSON = LAB
EXPECTED = [
    "model.py",
    "train.py",
    "app.py",
    "requirements.txt",
    "test_model.py",
    "README.md",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Day 1 tensor lab (this directory).")
    ap.add_argument(
        "--venv",
        action="store_true",
        help="Create .venv here and pip install -r requirements.txt",
    )
    ap.add_argument("--skip-tests", action="store_true", help="Do not run test_model.py")
    args = ap.parse_args()

    if not SETUP.is_file():
        print(f"error: missing {SETUP} (expected one level up from this folder)", file=sys.stderr)
        return 1

    subprocess.check_call([sys.executable, str(SETUP)], cwd=DAY1)

    missing = [f for f in EXPECTED if not (LESSON / f).is_file()]
    if missing:
        print(f"error: generator did not produce: {missing}", file=sys.stderr)
        return 1

    test_py = sys.executable
    if args.venv:
        vdir = LESSON / ".venv"
        subprocess.check_call([sys.executable, "-m", "venv", str(vdir)], cwd=LESSON)
        pip = vdir / "bin" / "pip"
        subprocess.check_call(
            [
                str(pip),
                "install",
                "--default-timeout",
                "120",
                "-r",
                str(LESSON / "requirements.txt"),
            ],
            cwd=LESSON,
        )
        test_py = str(vdir / "bin" / "python")

    if not args.skip_tests:
        subprocess.check_call([test_py, str(LESSON / "test_model.py")], cwd=LESSON)

    print(f"✓  build ok — {LESSON.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
