"""Bulk opening stock import — target physical qty, reservation conflict flags."""
from __future__ import annotations

import pytest

from backend.db import grey_db, item_db
from backend.services.fabric_opening_import import (
    apply_opening_import,
    preview_opening_import,
    template_xlsx_bytes,
)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(item_db, "DB_PATH", str(tmp_path / "items.db"))
    monkeypatch.setattr(grey_db, "_DB", str(tmp_path / "grey.db"))
    monkeypatch.setenv("GREY_DB_PATH", str(tmp_path / "grey.db"))
    item_db.init_db()
    grey_db.init_db()
    yield


def _xlsx(rows: list[tuple]):
    from openpyxl import Workbook
    import io

    wb = Workbook()
    ws = wb.active
    ws.append(["item_code", "target_qty", "unit", "reason", "reference_no", "entry_date"])
    for r in rows:
        ws.append(list(r))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_template_and_target_adjust_idempotent():
    assert template_xlsx_bytes()[:2] == b"PK"
    sfg = item_db._connect().execute("SELECT id FROM item_types WHERE code='SFG'").fetchone()[0]
    iid = item_db.create_item("FAB-OPEN-1", "Fabric One", sfg, uom="MTR")
    item_db.adjust_item_stock(iid, 100, "IN", entry_date="2026-01-01", reason="Opening Stock")

    raw = _xlsx([("FAB-OPEN-1", 250, "MTR", "Opening Stock — count", "PHYS-1", "2026-11-01")])
    prev = preview_opening_import(raw)
    assert prev["actions"] == 1
    assert prev["rows"][0]["delta"] == 150

    out = apply_opening_import(raw, created_by="tester")
    assert out["applied_count"] == 1
    item = item_db.get_item_by_code_ci("FAB-OPEN-1")
    assert float(item["stock"]) == 250

    # Re-upload same target → no-op
    out2 = apply_opening_import(raw, created_by="tester")
    assert out2["applied_count"] == 0
    assert out2["skipped_count"] >= 1


def test_reservation_conflict_skipped():
    sfg = item_db._connect().execute("SELECT id FROM item_types WHERE code='SFG'").fetchone()[0]
    iid = item_db.create_item("FAB-OPEN-2", "Fabric Two", sfg, uom="MTR")
    item_db.adjust_item_stock(iid, 200, "IN", entry_date="2026-01-01", reason="Opening Stock")
    conn = grey_db._connect()
    conn.execute(
        """INSERT INTO printed_fabric_checked_stock(
            fabric_code, fabric_name, checked_qty, passed_qty, reserved_qty, available_qty)
           VALUES (?,?,?,?,?,?)""",
        ("FAB-OPEN-2", "Fabric Two", 200, 200, 50, 150),
    )
    conn.execute(
        """INSERT INTO printed_fabric_reservations(
            fabric_code, fabric_name, so_number, sku, qty, unit, status, remarks, reservation_kind)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        ("FAB-OPEN-2", "Fabric Two", "SO-9", "SKU-S", 50, "MTR", "Active", "old", "so"),
    )
    conn.commit()
    conn.close()

    raw = _xlsx([("FAB-OPEN-2", 180, "MTR", "Opening Stock", "PHYS-2", "2026-11-01")])
    prev = preview_opening_import(raw)
    assert prev["conflicts"] == 1
    out = apply_opening_import(raw, created_by="tester", skip_conflicts=True)
    assert out["applied_count"] == 0
    assert out["skipped_count"] >= 1
    assert float(item_db.get_item_by_code_ci("FAB-OPEN-2")["stock"]) == 200
