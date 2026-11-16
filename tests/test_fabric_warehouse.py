"""Fabric Warehouse — dashboard math + ledger aggregation over existing sources."""
from __future__ import annotations

import sqlite3

import pytest

from backend.db import grey_db, item_db, production_db, purchase_db
from backend.services import fabric_warehouse as fw

TODAY = fw._today()


@pytest.fixture(autouse=True)
def isolated_dbs(tmp_path, monkeypatch):
    grey = str(tmp_path / "grey.db")
    pur = str(tmp_path / "purchase.db")
    prod = str(tmp_path / "production.db")
    items = str(tmp_path / "items.db")
    monkeypatch.setenv("GREY_DB_PATH", grey)
    monkeypatch.setenv("PURCHASE_DB_PATH", pur)
    monkeypatch.setenv("PRODUCTION_DB_PATH", prod)
    monkeypatch.setenv("ITEM_DB_PATH", items)
    monkeypatch.setattr(grey_db, "_DB", grey)
    monkeypatch.setattr(grey_db, "_PRODUCTION_DB", prod)
    monkeypatch.setattr(purchase_db, "_DB", pur)
    monkeypatch.setattr(production_db, "_DB", prod)
    monkeypatch.setattr(item_db, "DB_PATH", items)
    grey_db.init_db()
    purchase_db.init_db()
    production_db.init_db()
    item_db.init_db()
    yield


def _receive_and_qc(code="P100", qty=100.0, passed=90.0, failed=10.0, day="2026-01-05"):
    grey_db.insert_printed_fabric_unchecked(
        code, qty, fabric_name="Floral", printer="PrintCo", jwo_ref="JWO-1",
        grn_ref="GRN-1", receive_date=day,
    )
    grey_db.do_printed_fabric_qc({
        "fabric_code": code, "jwo_ref": "JWO-1", "passed_qty": passed,
        "failed_qty": failed, "qc_by": "QC", "qc_date": "2026-01-06",
    })


def _seed_jo_issue(code="P100", qty=30.0, day="2026-01-10", ret=0.0):
    conn = sqlite3.connect(production_db._DB)
    conn.execute(
        "INSERT INTO job_orders(jo_number, jo_date, process, so_number, status) VALUES('JO-9','2026-01-09','Cutting','SO-1','Created')"
    )
    jo_id = conn.execute("SELECT id FROM job_orders WHERE jo_number='JO-9'").fetchone()[0]
    conn.execute(
        "INSERT INTO jo_fabric_issues(jo_id,issue_date,fabric_code,issued_qty,issued_by) VALUES(?,?,?,?,?)",
        (jo_id, day, code, qty, "cutter"),
    )
    if ret:
        conn.execute(
            "INSERT INTO jo_fabric_returns(jo_id,return_date,fabric_code,returned_qty) VALUES(?,?,?,?)",
            (jo_id, "2026-01-12", code, ret),
        )
    conn.commit()
    conn.close()


def _seed_min(code="P100", qty=5.0):
    conn = sqlite3.connect(purchase_db._DB)
    conn.execute(
        "INSERT INTO material_issue_notes(min_number,min_date,jwo_reference,to_vendor,status) "
        "VALUES('MIN-1','2026-01-11','JWO-2','Embroiderer','Confirmed')"
    )
    mid = conn.execute("SELECT id FROM material_issue_notes").fetchone()[0]
    conn.execute(
        "INSERT INTO min_lines(min_id,material_code,material_name,issue_qty) VALUES(?,?,?,?)",
        (mid, code, "Floral", qty),
    )
    # Draft MIN must be ignored
    conn.execute(
        "INSERT INTO material_issue_notes(min_number,min_date,status) VALUES('MIN-2','2026-01-11','Draft')"
    )
    mid2 = conn.execute("SELECT id FROM material_issue_notes WHERE min_number='MIN-2'").fetchone()[0]
    conn.execute(
        "INSERT INTO min_lines(min_id,material_code,issue_qty) VALUES(?,?,?)", (mid2, code, 999),
    )
    conn.commit()
    conn.close()


def _reserve(code="P100", qty=20.0, so="SO-1", sku="SKU-A"):
    return grey_db.reserve_printed_fabric({
        "fabric_code": code, "so_number": so, "sku": sku, "qty": qty,
    })


def _rows(**kw):
    return fw.build_ledger(**kw)["rows"]


def test_grn_and_qc_ledger_rows():
    _receive_and_qc()
    types = [r["txn_type"] for r in _rows(pcode="P100")]
    assert types == [fw.T_GRN, fw.T_QC_PASS, fw.T_QC_REJECT]
    res = fw.build_ledger(pcode="P100", warehouse="all")
    # GRN +100 unchecked; pass is a transfer (0); reject -10 => net 90 physical
    assert res["total_in"] == 100.0
    assert res["total_out"] == 10.0
    assert res["rows"][-1]["balance"] == 90.0
    chk = fw.build_ledger(pcode="P100", warehouse="checked")
    assert [r["txn_type"] for r in chk["rows"]] == [fw.T_QC_PASS]
    assert chk["rows"][0]["in_qty"] == 90.0 and chk["rows"][0]["balance"] == 90.0
    unc = fw.build_ledger(pcode="P100", warehouse="unchecked")
    assert unc["rows"][-1]["balance"] == 0.0  # fully QC'd


def test_partial_qc_leaves_pending_unchecked():
    _receive_and_qc(qty=100, passed=60, failed=0)
    st = {r["pcode"]: r for r in fw.stock_by_pcode()}
    assert st["P100"]["unchecked"] == 40.0
    assert st["P100"]["checked"] == 60.0
    assert st["P100"]["physical"] == 100.0
    d = fw.dashboard(from_date="2026-01-01", to_date="2026-12-31")
    assert d["pending_unchecked"] == 40.0
    assert d["pending_unchecked_codes"] == 1


def test_out_sources_min_and_jo_and_return():
    _receive_and_qc()  # checked 90
    _seed_jo_issue(qty=30, ret=5)
    _seed_min(qty=5)
    res = fw.build_ledger(pcode="P100", warehouse="checked")
    types = {r["txn_type"] for r in res["rows"]}
    assert {fw.T_JO_ISSUE, fw.T_JO_RETURN, fw.T_MIN} <= types
    # draft MIN (999) ignored
    assert res["total_out"] == 35.0
    assert res["total_in"] == 95.0
    assert res["rows"][-1]["balance"] == 90 - 30 + 5 - 5


def test_reservation_is_allocation_only_not_out():
    _receive_and_qc()  # checked 90 free
    _reserve(qty=20)
    rows = _rows(pcode="P100", warehouse="all")
    res_rows = [r for r in rows if r["txn_type"] == fw.T_RESERVE_SO]
    assert len(res_rows) == 1
    r = res_rows[0]
    assert r["physical_move"] is False
    assert r["out_qty"] == 0 and r["in_qty"] == 0
    assert r["alloc_qty"] == 20.0
    # physical running balance unchanged by reservation
    assert rows[-1]["balance"] == 90.0
    # direction=out never includes reservations
    assert not [x for x in _rows(pcode="P100", direction="out") if not x["physical_move"]]
    # alloc filter shows only reservation
    assert [x["txn_type"] for x in _rows(pcode="P100", direction="alloc")] == [fw.T_RESERVE_SO]
    d = fw.dashboard(from_date="2026-01-01", to_date="2026-12-31", pcode="P100")
    assert d["so_allocated"] == 20.0
    assert d["free"] == 70.0
    assert d["out_qty"] == 10.0  # only the QC reject; reservation not OUT
    st = fw.stock_by_pcode("P100")[0]
    assert st["physical"] == 90.0 and st["free"] == 70.0 and st["so_allocated"] == 20.0


def test_opening_conversion_and_adjustment(monkeypatch):
    item_id = item_db.create_item(item_code="P200", item_name="Opening Print", item_type_id=3)
    item_db.adjust_item_stock(item_id, 50, "IN", entry_date="2026-01-02",
                              reason="Opening Stock", created_by="admin")
    item_db.adjust_item_stock(item_id, 5, "OUT", entry_date="2026-01-03",
                              reason="Damaged roll", created_by="admin")
    res = fw.build_ledger(pcode="P200", warehouse="all")
    types = [r["txn_type"] for r in res["rows"]]
    assert fw.T_OPENING in types and fw.T_ADJ in types
    assert res["rows"][-1]["balance"] == 45.0
    # Opening-only fabric has no checked stock and is not Free until converted
    # (free math is owned by printed_stock_visibility; ledger just reports movement).


def test_period_opening_closing_and_date_filter():
    _receive_and_qc(day="2026-01-05")
    _seed_jo_issue(qty=30, day="2026-01-10")
    res = fw.build_ledger(pcode="P100", from_date="2026-01-08", to_date="2026-01-31", warehouse="checked")
    assert res["opening"] == 90.0
    assert res["total_out"] == 30.0
    assert res["closing"] == 60.0
    d = fw.dashboard(from_date="2026-01-08", to_date="2026-01-31", pcode="P100", warehouse="checked")
    assert (d["opening"], d["in_qty"], d["out_qty"], d["closing"]) == (90.0, 0.0, 30.0, 60.0)


def test_txn_type_and_search_filters_and_export():
    _receive_and_qc()
    _seed_jo_issue()
    only_jo = _rows(txn_type=fw.T_JO_ISSUE)
    assert len(only_jo) == 1 and only_jo[0]["ref_no"] == "JO-9"
    assert only_jo[0]["link"]["path"] == "/production"
    assert len(_rows(q="cutter")) == 1
    csv_bytes = fw.export_csv(_rows())
    assert csv_bytes.startswith(b"\xef\xbb\xbf") and b"JO Fabric Issue" in csv_bytes
    xlsx = fw.export_xlsx(_rows())
    assert xlsx[:2] == b"PK"


def test_recon_diff_flags_unposted_min():
    _receive_and_qc()
    _seed_min(qty=5)
    d = fw.dashboard(from_date="2026-01-01", to_date="2026-12-31", pcode="P100")
    # Purchase MIN is OUT in ledger but not applied to printed_fabric_checked_stock
    assert d["ledger_closing_all_time"] == 85.0
    assert d["physical_now"] == 90.0
    assert d["recon_diff"] == 5.0


def test_api_endpoints(client):
    _receive_and_qc()
    r = client.get("/api/fabric-warehouse/stock")
    assert r.status_code == 200, r.text
    assert client.get("/api/fabric-warehouse/dashboard").json()["pcode_count"] == 1
    led = client.get("/api/fabric-warehouse/ledger", params={"pcode": "P100", "order": "asc"}).json()
    assert led["total"] == 3 and led["rows"][0]["txn_type"] == fw.T_GRN
    csv_r = client.get("/api/fabric-warehouse/ledger/export", params={"format": "csv"})
    assert csv_r.status_code == 200 and "text/csv" in csv_r.headers["content-type"]
    x = client.get("/api/fabric-warehouse/ledger/export", params={"format": "xlsx"})
    assert x.status_code == 200 and x.content[:2] == b"PK"
    assert client.get("/api/fabric-warehouse/ledger", params={"warehouse": "bad"}).status_code == 400
