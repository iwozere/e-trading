#!/usr/bin/env python3
"""
Static type-check gate: run mypy and pyright, fail if either reports an error.

Both checkers must stay at **0 errors** (see the coding conventions and the
now-retired ``pyright-plan.md``). This script is the single enforcement point,
used both locally and by CI (``.github/workflows/typecheck.yml``).

Canonical environment
---------------------
Run this from the project's ``.venv`` so that mypy and pyright analyse the same
interpreter and the same installed packages::

    .venv/Scripts/python.exe scripts/typecheck.py   # Windows
    .venv/bin/python scripts/typecheck.py            # Linux / macOS

pyright resolves its analysis environment from ``pyrightconfig.json`` (``venv:
".venv"``); mypy resolves from whichever interpreter runs it. Running both
through the ``.venv`` interpreter keeps them in lockstep — the split-brain the
cleanup deliberately closed. ``pyright`` is pinned exactly in
``requirements-dev.txt`` so the error count is reproducible; do not substitute
``npx pyright`` (it floats to the latest build).

Usage:
    python scripts/typecheck.py            # run both checkers
    python scripts/typecheck.py --mypy     # mypy only
    python scripts/typecheck.py --pyright  # pyright only
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# mypy's checked surface: the whole application package. pyright's surface is
# governed by pyrightconfig.json (src + tests + config, minus the excluded
# legacy pipelines), so it takes no path argument here.
MYPY_TARGETS = ["src"]


def _run(label: str, cmd: list[str]) -> bool:
    """Run one checker, streaming its output; return True on success (exit 0)."""
    print(f"\n=== {label} ===", flush=True)
    print("  $ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    ok = result.returncode == 0
    print(f"  -> {label}: {'PASS' if ok else 'FAIL'} (exit {result.returncode})", flush=True)
    return ok


def run_mypy() -> bool:
    return _run("mypy", [sys.executable, "-m", "mypy", *MYPY_TARGETS])


def run_pyright() -> bool:
    return _run("pyright", [sys.executable, "-m", "pyright"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the static type-check gate (mypy + pyright).")
    parser.add_argument("--mypy", action="store_true", help="run mypy only")
    parser.add_argument("--pyright", action="store_true", help="run pyright only")
    args = parser.parse_args()

    run_both = not (args.mypy or args.pyright)
    results: list[bool] = []
    if args.mypy or run_both:
        results.append(run_mypy())
    if args.pyright or run_both:
        results.append(run_pyright())

    passed = all(results)
    print("\n=== type-check gate: " + ("PASS" if passed else "FAIL") + " ===", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
