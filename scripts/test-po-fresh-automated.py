#!/usr/bin/env python3
"""Full automated PO Fresh validation: health check + engine e2e + API contract tests."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
E2E = ROOT / "scripts" / "test-po-fresh-local-e2e.py"


def _get(url: str, timeout: float = 3.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["curl", "-sf", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", str(int(timeout)), url],
            capture_output=True,
            text=True,
            timeout=timeout + 2,
        )
        code = int(proc.stdout.strip() or "0") if proc.returncode == 0 else 0
        return code, proc.stdout.strip()
    except Exception as e:
        return 0, str(e)


def wait_for_stack(seconds: float = 60.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        bc, _ = _get("http://127.0.0.1:8000/api/health")
        fc, _ = _get("http://127.0.0.1:5173/")
        if bc == 200 and fc == 200:
            return True
        time.sleep(2)
    return False


def run(cmd: list[str], label: str) -> int:
    print(f"\n=== {label} ===")
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def main() -> int:
    print("=== PO Fresh automated test suite ===")
    if not wait_for_stack():
        print("FAIL: backend/frontend not reachable on :8000 / :5173")
        print("Start with: bash scripts/start-po-local-daemon.sh")
        return 1
    print("PASS: stack health (backend :8000, frontend :5173)")

    rc = 0
    rc |= run([str(PY), str(E2E)], "Engine e2e (warm cache + formula audit)")
    rc |= run(
        [
            str(PY),
            "-m",
            "pytest",
            "tests/test_po_fresh_api.py",
            "tests/test_po_session_hydrate.py",
            "tests/test_po_service.py::test_below_target_cover_gets_po_when_projected_within_lead_window",
            "tests/test_po_service.py::test_lead_gate_qty_targets_post_po_cover_not_lead_only",
            "tests/test_po_service.py::test_digit_token_only_lead_does_not_enable_sheet_po_gate",
            "-q",
        ],
        "Pytest PO contract",
    )

    print("\n=== SUMMARY ===")
    if rc == 0:
        print("ALL PASS — PO calculation validated")
        return 0
    print("FAIL — see sections above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
