#!/usr/bin/env python3
"""Prod probe: were Cut-to-Pack Cutting JOs created? Safe read-only diagnosis."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


def _copy_prod_db() -> Path:
    """Prefer docker cp of production.db from the running backend container."""
    out = Path(tempfile.mkdtemp(prefix="c2p-probe-")) / "production.db"
    names = [
        "app-backend-1",
        "backend",
        "yash-gallery-trial-backend-1",
        "app_backend_1",
    ]
    # Discover container with production.db
    try:
        ps = subprocess.check_output(
            ["docker", "ps", "--format", "{{.Names}}"],
            text=True,
            timeout=30,
        )
        running = [n.strip() for n in ps.splitlines() if n.strip()]
    except Exception as e:
        raise SystemExit(f"docker ps failed: {e}") from e
    candidates = [n for n in names if n in running] + running
    last_err = None
    for name in candidates:
        for src in (
            f"{name}:/app/data/production.db",
            f"{name}:/data/production.db",
            f"{name}:/root/app/data/production.db",
        ):
            try:
                subprocess.check_call(
                    ["docker", "cp", src, str(out)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
                if out.is_file() and out.stat().st_size > 0:
                    print("DB_SOURCE", src)
                    return out
            except Exception as e:
                last_err = e
                continue
    # Fallback host path
    for p in (
        Path("/root/app/data/production.db"),
        Path("/root/app/backend/data/production.db"),
    ):
        if p.is_file():
            print("DB_SOURCE", p)
            return p
    raise SystemExit(f"Could not locate production.db (last={last_err})")


def main() -> int:
    db = _copy_prod_db()
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    summary = {}
    summary["cutting_total"] = conn.execute(
        "SELECT COUNT(*) FROM job_orders WHERE process='Cutting' AND IFNULL(status,'')!='Cancelled'"
    ).fetchone()[0]
    summary["stitching_total"] = conn.execute(
        "SELECT COUNT(*) FROM job_orders WHERE process='Stitching' AND IFNULL(status,'')!='Cancelled'"
    ).fetchone()[0]

    by_mode = {}
    for r in conn.execute(
        """SELECT IFNULL(NULLIF(TRIM(production_mode),''),'(blank)') AS mode, process, COUNT(*) AS n
           FROM job_orders
           WHERE IFNULL(status,'')!='Cancelled'
           GROUP BY 1, 2
           ORDER BY n DESC"""
    ):
        by_mode.setdefault(r["mode"], {})[r["process"]] = r["n"]
    summary["by_mode_process"] = by_mode

    c2p_cutting = conn.execute(
        """SELECT COUNT(*) FROM job_orders
           WHERE process='Cutting' AND IFNULL(status,'')!='Cancelled'
             AND LOWER(REPLACE(IFNULL(production_mode,''),' ','_')) IN
                 ('cut_to_pack','cuttopack','cut_pack','c2p')"""
    ).fetchone()[0]
    summary["cut_to_pack_cutting_jos"] = c2p_cutting

    s2p_stitch = conn.execute(
        """SELECT COUNT(*) FROM job_orders
           WHERE process='Stitching' AND IFNULL(status,'')!='Cancelled'
             AND LOWER(REPLACE(IFNULL(production_mode,''),' ','_')) IN
                 ('stitch_to_pack','stitchtopack','stitch_pack','s2p')"""
    ).fetchone()[0]
    summary["stitch_to_pack_stitching_jos"] = s2p_stitch

    recent_c2p = [
        dict(r)
        for r in conn.execute(
            """SELECT jo_number, so_number, sku, planned_qty, production_mode, created_at, updated_at
               FROM job_orders
               WHERE process='Cutting'
                 AND LOWER(REPLACE(IFNULL(production_mode,''),' ','_')) IN
                     ('cut_to_pack','cuttopack','cut_pack','c2p')
               ORDER BY id DESC LIMIT 15"""
        )
    ]
    summary["recent_cut_to_pack_cutting"] = recent_c2p

    # Recent Cutting creates (any mode) — helps see if import landed without mode tag
    recent_cut = [
        dict(r)
        for r in conn.execute(
            """SELECT jo_number, so_number, sku, planned_qty, production_mode, created_at
               FROM job_orders WHERE process='Cutting'
               ORDER BY id DESC LIMIT 20"""
        )
    ]
    summary["recent_cutting_any"] = recent_cut

    # WIP ledger batches (if table exists)
    try:
        wip_batches = [
            dict(r)
            for r in conn.execute(
                """SELECT import_batch, ready_to_stage, COUNT(*) AS rows, SUM(qty) AS qty
                   FROM ready_to_wip_ledger
                   GROUP BY import_batch, ready_to_stage
                   ORDER BY MAX(id) DESC LIMIT 20"""
            )
        ]
        summary["recent_wip_batches"] = wip_batches
    except sqlite3.Error as e:
        summary["recent_wip_batches"] = f"n/a: {e}"

    print("CUT_TO_PACK_PROBE")
    print(json.dumps(summary, indent=2, default=str))

    if c2p_cutting <= 0:
        print(
            "VERDICT: No Cutting JOs with production_mode=cut_to_pack found. "
            "Do NOT assume the Cut-to-Pack file created Cutting JOs — "
            "re-upload only if the file was the JO template (so_number,sku,planned_qty,process,production_mode). "
            "If it was a Ready-To WIP template, it credited stock only (created=0 JOs)."
        )
        return 2
    print(f"VERDICT: Found {c2p_cutting} Cut-to-Pack Cutting JO(s). Safe to search/export; no re-upload needed for those.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        print("PROBE_FAIL", e)
        raise SystemExit(1) from e
