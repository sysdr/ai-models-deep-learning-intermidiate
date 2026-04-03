#!/usr/bin/env python3
"""Start Streamlit on ./app.py. Replaces the current process."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
APP = LAB / "app.py"
VENV_ST = LAB / ".venv" / "bin" / "streamlit"


def main() -> None:
    if not APP.is_file():
        print(f"error: missing {APP} — run ./build.py", file=sys.stderr)
        raise SystemExit(1)

    app_path = str(APP.resolve())
    os.chdir(LAB)

    if os.access(VENV_ST, os.X_OK):
        os.execl(
            str(VENV_ST),
            str(VENV_ST),
            "run",
            app_path,
            "--server.headless",
            "true",
            "--browser.serverAddress",
            "localhost",
        )

    st = shutil.which("streamlit")
    if st:
        print(
            f"warn: using streamlit on PATH (venv missing or incomplete): {st}",
            file=sys.stderr,
        )
        os.execl(
            st,
            st,
            "run",
            app_path,
            "--server.headless",
            "true",
            "--browser.serverAddress",
            "localhost",
        )

    print(
        "error: streamlit not found. Run: ./build.py --venv\n"
        f"  or: python3 -m venv {LAB / '.venv'} && "
        f"{LAB / '.venv' / 'bin' / 'pip'} install -r {LAB / 'requirements.txt'}",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
