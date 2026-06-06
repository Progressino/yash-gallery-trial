#!/usr/bin/env python3
"""Wrapper — implementation lives in backend/scripts/ (copied into the Docker image)."""
from backend.scripts.run_po_calculate_production import main

if __name__ == "__main__":
    raise SystemExit(main())
