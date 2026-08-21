#!/usr/bin/env python3
"""Prod reconciliation: today's Stitching issues vs Ready-To Finishing/Kaj/Handwork."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import requests

BASE = os.environ.get("PROBE_BASE_URL", "http://127.0.0.1:8000")
PROD_DB = os.environ.get("PRODUCTION_DB_PATH", "/root/app/data/production.db")
TODAY = os.environ.get("RECON_DATE", date.today().isoformat())
TARGETS = ("Finishing", "Kaj Button", "Handwork")


def _env_auth() -> tuple[str, str]:
    user = os.environ.get("AUTH_USERNAME", "").strip()
    pw = os.environ.get("AUTH_PASSWORD", "").strip()
    if not user or not pw:
        env_path = Path("/root/app/.env")
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k == "AUTH_USERNAME":
                    user = user or v
                elif k == "AUTH_PASSWORD":
                    pw = pw or v
    if not user or not pw:
        raise SystemExit("AUTH_USERNAME/PASSWORD missing")
    return user, pw


def _find_production_db() -> str:
    env = os.environ.get("PRODUCTION_DB_PATH", "").strip()
    candidates = [
        env,
        "/root/app/data/production.db",
        "/root/app/backend/data/production.db",
        "/root/app/production.db",
        "/data/production.db",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    # Docker volume (compose prod mounts /data)
    try:
        import subprocess

        for cid_cmd in (
            ["docker", "ps", "-q", "-f", "name=backend"],
            ["docker", "ps", "-q", "-f", "ancestor=yash-gallery"],
        ):
            try:
                out = subprocess.check_output(cid_cmd, text=True, timeout=15).strip().splitlines()
            except Exception:
                continue
            for cid in out:
                cid = cid.strip()
                if not cid:
                    continue
                dest = f"/tmp/production_probe_{cid[:8]}.db"
                try:
                    subprocess.check_call(
                        ["docker", "cp", f"{cid}:/data/production.db", dest],
                        timeout=60,
                    )
                    if Path(dest).is_file():
                        return dest
                except Exception:
                    continue
    except Exception:
        pass
    raise SystemExit(
        "production.db not found — set PRODUCTION_DB_PATH or ensure docker backend has /data/production.db"
    )


def _connect():
    path = _find_production_db()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn, str(path)


def main() -> int:
    s = requests.Session()
    h = s.get(f"{BASE}/api/health", timeout=30)
    h.raise_for_status()
    print("HEALTH", json.dumps({k: h.json().get(k) for k in ("status", "git_sha", "warm_cache")}))

    user, pw = _env_auth()
    r = s.post(
        f"{BASE}/api/auth/login",
        json={"username": user, "password": pw},
        headers={"X-Device-Id": "gha-ready-recon"},
        timeout=60,
    )
    r.raise_for_status()
    print("LOGIN", r.status_code, "RECON_DATE", TODAY)

    # --- API Ready-To snapshots ---
    api_boards: dict[str, list] = {}
    for stage in TARGETS:
        resp = s.get(f"{BASE}/api/production/ready-to-process/{stage}", timeout=120)
        resp.raise_for_status()
        rows = resp.json() if isinstance(resp.json(), list) else []
        api_boards[stage] = rows
        print(
            f"API_READY {stage}: lines={len(rows)} "
            f"unique_so_sku={len({(x.get('so_number'), x.get('sku')) for x in rows})} "
            f"qty_sum={sum(int(x.get('available_qty') or 0) for x in rows)}"
        )

    # Cross-board overlap of SO+SKU
    keys = {st: {(str(x.get("so_number") or ""), str(x.get("sku") or "")) for x in api_boards[st]} for st in TARGETS}
    all_keys = set().union(*keys.values())
    multi = [k for k in all_keys if sum(1 for st in TARGETS if k in keys[st]) > 1]
    print(
        "CROSS_BOARD unique_so_sku_any_board=",
        len(all_keys),
        "on_2+_boards=",
        len(multi),
        "sum_of_board_lines=",
        sum(len(api_boards[st]) for st in TARGETS),
    )

    conn, db_path = _connect()
    print("DB", db_path)

    # Today's issues FROM Stitching — IST calendar day + UTC day + last 36h fallback
    issues = conn.execute(
        """
        SELECT i.id, i.jo_id, i.jo_line_id, i.sku, i.from_process, i.to_process,
               i.issued_qty, i.issue_date, i.created_at, j.jo_number, j.so_number, j.process AS jo_process
        FROM jo_piece_issues i
        LEFT JOIN job_orders j ON j.id = i.jo_id
        WHERE UPPER(REPLACE(TRIM(IFNULL(i.from_process,'')), ' ', '')) IN ('STITCHING')
          AND UPPER(TRIM(IFNULL(i.to_process,''))) IN ('FINISHING', 'KAJ BUTTON', 'HANDWORK')
          AND (
            substr(IFNULL(i.issue_date, ''), 1, 10) = ?
            OR substr(IFNULL(i.created_at, ''), 1, 10) = ?
            OR datetime(IFNULL(i.created_at, i.issue_date)) >= datetime('now', '-36 hours')
          )
        ORDER BY i.id
        """,
        (TODAY, TODAY),
    ).fetchall()

    # Diagnostics when empty
    if not issues:
        sample = conn.execute(
            """
            SELECT substr(IFNULL(issue_date, created_at),1,10) AS d, from_process, to_process, COUNT(*) AS n
            FROM jo_piece_issues
            WHERE UPPER(REPLACE(TRIM(IFNULL(from_process,'')), ' ', '')) = 'STITCHING'
            GROUP BY 1, 2, 3
            ORDER BY 1 DESC LIMIT 15
            """
        ).fetchall()
        print("ISSUE_DATE_DIAG", [dict(r) for r in sample])

    jo_ids = {int(r["jo_id"]) for r in issues if r["jo_id"] is not None}
    jo_numbers = {str(r["jo_number"] or "") for r in issues if r["jo_number"]}
    by_target: dict[str, list] = defaultdict(list)
    for r in issues:
        by_target[str(r["to_process"] or "")].append(r)

    # Duplicate issue events: same jo_id + to_process more than once today
    jo_target_counts: dict[tuple, int] = defaultdict(int)
    for r in issues:
        jo_target_counts[(r["jo_id"], str(r["to_process"] or ""))] += 1
    multi_issue = {k: v for k, v in jo_target_counts.items() if v > 1}

    print(
        "ISSUES_TODAY from_stitching=",
        len(issues),
        "unique_jo_ids=",
        len(jo_ids),
        "unique_jo_numbers=",
        len(jo_numbers),
        "multi_issue_same_jo_target=",
        len(multi_issue),
    )
    for t in TARGETS:
        rows = by_target.get(t, [])
        ujos = {r["jo_id"] for r in rows}
        print(
            f"  issued→{t}: events={len(rows)} unique_jos={len(ujos)} "
            f"qty={sum(int(r['issued_qty'] or 0) for r in rows)}"
        )

    # JOs issued to 2+ different targets today
    jo_targets: dict[int, set[str]] = defaultdict(set)
    for r in issues:
        if r["jo_id"] is not None:
            jo_targets[int(r["jo_id"])].add(str(r["to_process"] or ""))
    multi_route = {jid: tgts for jid, tgts in jo_targets.items() if len(tgts) > 1}
    print("LEGIT_MULTI_TARGET_JOS", len(multi_route))
    if multi_route:
        sample = list(multi_route.items())[:5]
        print("  sample", sample)

    # Stitching stock still available that appears as feeder on API boards
    stitch_stock = conn.execute(
        """
        SELECT so_number, sku, available_qty, jo_number
        FROM process_stock
        WHERE UPPER(TRIM(process))='STITCHING' AND available_qty > 0
        """
    ).fetchall()
    stitch_keys = {(str(r["so_number"] or ""), str(r["sku"] or "")): int(r["available_qty"] or 0) for r in stitch_stock}
    print("STITCHING_STOCK_ROWS", len(stitch_keys), "qty_sum", sum(stitch_keys.values()))

    overlap_boards = []
    for k in multi:
        so, sku = k
        boards = [st for st in TARGETS if k in keys[st]]
        overlap_boards.append(
            {
                "so": so,
                "sku": sku,
                "boards": boards,
                "stitch_avail": stitch_keys.get(k, 0),
                "api_qtys": {
                    st: next(
                        (int(x.get("available_qty") or 0) for x in api_boards[st] if (str(x.get("so_number") or ""), str(x.get("sku") or "")) == k),
                        0,
                    )
                    for st in boards
                },
            }
        )
    print("OVERLAP_SAMPLE", json.dumps(overlap_boards[:15], default=str))

    # WIP ledger rows that could force cross-board
    try:
        wip = conn.execute(
            """
            SELECT ready_to_stage, from_process, so_number, sku, quantity, created_at
            FROM ready_to_wip_imports
            WHERE UPPER(TRIM(ready_to_stage)) IN ('FINISHING','KAJ BUTTON','HANDWORK')
              AND UPPER(TRIM(from_process))='STITCHING'
            ORDER BY id DESC LIMIT 500
            """
        ).fetchall()
        print("WIP_LEDGER_STITCH_FEEDER_ROWS", len(wip))
    except Exception as e:
        print("WIP_LEDGER_ERR", e)
        wip = []

    # Issues without matching stock movement integrity: sum issued vs stock at target
    summary = {
        "recon_date": TODAY,
        "stitching_issue_events": len(issues),
        "unique_stitching_jos_issued": len(jo_ids),
        "unique_jo_numbers": len(jo_numbers),
        "issued_by_target_unique_jos": {t: len({r["jo_id"] for r in by_target.get(t, [])}) for t in TARGETS},
        "issued_by_target_events": {t: len(by_target.get(t, [])) for t in TARGETS},
        "legitimate_multi_target_jos": len(multi_route),
        "duplicate_issue_same_jo_target": len(multi_issue),
        "api_ready_lines": {t: len(api_boards[t]) for t in TARGETS},
        "api_ready_unique_so_sku": {t: len(keys[t]) for t in TARGETS},
        "sum_api_ready_lines": sum(len(api_boards[t]) for t in TARGETS),
        "unique_so_sku_any_ready_board": len(all_keys),
        "so_sku_on_multiple_ready_boards": len(multi),
        "overlap_with_stitching_stock": sum(1 for o in overlap_boards if o["stitch_avail"] > 0),
        "wip_ledger_stitch_feeder_sample": len(wip),
    }
    print("RECON_SUMMARY", json.dumps(summary))
    print("READY_RECON_OK")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
