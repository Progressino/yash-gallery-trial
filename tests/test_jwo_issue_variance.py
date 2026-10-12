"""Printing JWO grey issue variance: reverse output calc, original vs revised, controlled corrections."""
from __future__ import annotations

import pytest

from backend.db import item_db, purchase_db
from backend.services import jwo_issue_variance as v
from backend.services.jwo_min_notes import get_min_by_jwo_id

GREY, PRINTED, THREAD = "GREY-COT-58", "P752FLORAL", "THREAD-1"
GROSS_PER_UNIT = 0.9  # 450 m grey → 500 m printed


@pytest.fixture(autouse=True)
def iso(tmp_path, monkeypatch):
    items = str(tmp_path / "items.db")
    monkeypatch.setattr(purchase_db, "_DB", str(tmp_path / "purchase.db"))
    monkeypatch.setenv("ITEM_DB_PATH", items)
    monkeypatch.setenv("GREY_DB_PATH", str(tmp_path / "grey.db"))
    monkeypatch.setattr(item_db, "DB_PATH", items)
    item_db.init_db()
    purchase_db.init_db()

    def fake_bom(code, name, qty):
        return [
            {"finished_item_code": code, "finished_item_name": name, "finished_planned_qty": qty,
             "material_code": GREY, "material_name": "Grey Cotton", "material_type": "GF",
             "bom_qty_per_unit": 0.85, "required_qty": round(GROSS_PER_UNIT * qty, 3), "unit": "MTR"},
            {"finished_item_code": code, "finished_item_name": name, "finished_planned_qty": qty,
             "material_code": THREAD, "material_name": "Thread", "material_type": "RM",
             "bom_qty_per_unit": 0.01, "required_qty": round(0.01 * qty, 3), "unit": "MTR"},
        ]

    monkeypatch.setattr("backend.services.jwo_min_notes.explode_bom_materials", fake_bom)
    monkeypatch.setattr("backend.services.jwo_min_notes._lookup_item_name", lambda c: c)
    yield


def _jwo(output_qty: float = 500, rate: float = 20) -> int:
    num = purchase_db.create_jwo({
        "processor_name": "Shree Printers",
        "so_reference": "SO-101",
        "pr_reference": "PR-0009",
        "lines": [{"input_material": GREY, "input_qty": GROSS_PER_UNIT * output_qty, "output_material": PRINTED,
                   "output_qty": output_qty, "rate": rate}],
    })
    conn = purchase_db._connect()
    jid = conn.execute("SELECT id FROM jwo_headers WHERE jwo_number=?", (num,)).fetchone()[0]
    conn.close()
    return int(jid)


def _lines(jid: int) -> dict[str, dict]:
    return {l["material_code"]: l for l in get_min_by_jwo_id(jid)["lines"]}


def _jwo_line(jid: int) -> dict:
    return purchase_db.get_jwo_by_id(jid)["lines"][0]


def _seed_stock(code: str, qty: float) -> int:
    conn = item_db._connect()
    gf = conn.execute("SELECT id FROM item_types WHERE code='GF'").fetchone()[0]
    conn.close()
    iid = item_db.create_item(code, code, gf, uom="MTR")
    item_db.apply_document_stock_delta(iid, qty, "IN")
    return iid


def _stock(iid: int) -> float:
    return float(item_db.get_item(iid)["stock"])


def _adjustments(min_id: int) -> list[dict]:
    return v.list_adjustments(min_id)


def test_default_bom_calculation_unchanged():
    jid = _jwo()
    g = _lines(jid)[GREY]
    assert g["issue_qty"] == 450 and g["bom_issue_qty"] == 450
    assert g["original_output_qty"] == 500 and g["expected_output_qty"] == 500
    assert g["variance_reason"] == ""
    jl = _jwo_line(jid)
    assert jl["original_input_qty"] == 450 and jl["original_output_qty"] == 500


def test_increase_issue_recalculates_output_and_related_min_lines():
    jid = _jwo()
    note = get_min_by_jwo_id(jid)
    gline = _lines(jid)[GREY]
    res = v.revise_issue_qty(note["id"], gline["id"], 500, reason="Bundle Size", user_name="stores")
    assert res["expected_output_qty"] == pytest.approx(555.556)

    lines = _lines(jid)
    g, t = lines[GREY], lines[THREAD]
    assert g["issue_qty"] == 500 and g["bom_issue_qty"] == 450
    assert g["original_output_qty"] == 500 and g["expected_output_qty"] == pytest.approx(555.556)
    assert g["variance_reason"] == "Bundle Size" and g["revised_by"] == "stores"
    assert t["issue_qty"] == pytest.approx(5.556) and t["bom_issue_qty"] == pytest.approx(5.0)

    jl = _jwo_line(jid)
    assert jl["input_qty"] == 500 and jl["output_qty"] == pytest.approx(555.556)
    assert jl["original_input_qty"] == 450 and jl["original_output_qty"] == 500
    assert jl["amount"] == pytest.approx(555.556 * 20)
    assert purchase_db.get_jwo_by_id(jid)["total"] == pytest.approx(555.556 * 20)

    adj = _adjustments(note["id"])
    assert adj[0]["phase"] == "PRE_ISSUE" and adj[0]["old_issue_qty"] == 450 and adj[0]["new_issue_qty"] == 500
    assert adj[0]["reason"] == "Bundle Size" and adj[0]["stock_posted"] == 0


def test_decrease_and_reset_to_bom():
    jid = _jwo()
    note = get_min_by_jwo_id(jid)
    v.revise_issue_qty(note["id"], _lines(jid)[GREY]["id"], 400, reason="Stock Shortage")
    assert _lines(jid)[GREY]["expected_output_qty"] == pytest.approx(444.444)
    v.revise_issue_qty(note["id"], _lines(jid)[GREY]["id"], 450)
    g = _lines(jid)[GREY]
    assert g["issue_qty"] == 450 and g["expected_output_qty"] == 500 and g["variance_reason"] == ""
    assert _jwo_line(jid)["variance_reason"] == ""


def test_reason_rules():
    jid = _jwo()
    note = get_min_by_jwo_id(jid)
    lid = _lines(jid)[GREY]["id"]
    with pytest.raises(ValueError, match="Reason is required"):
        v.revise_issue_qty(note["id"], lid, 500)
    with pytest.raises(ValueError, match="Remarks are required"):
        v.revise_issue_qty(note["id"], lid, 500, reason="Other")
    with pytest.raises(ValueError, match="Reason must be one of"):
        v.revise_issue_qty(note["id"], lid, 500, reason="Because")
    assert _lines(jid)[GREY]["issue_qty"] == 450


def test_jwo_form_input_change_uses_same_reverse_calc():
    jid = _jwo()
    base = _jwo_line(jid)
    line = {k: base[k] for k in ("input_material", "output_material", "output_qty", "rate", "process_type")}
    with pytest.raises(ValueError, match="Reason is required"):
        purchase_db.update_jwo(jid, {"lines": [{**line, "input_qty": 500}]})
    purchase_db.update_jwo(jid, {"lines": [{**line, "input_qty": 500, "variance_reason": "Vendor MOQ"}], "user_name": "planner"})
    jl = _jwo_line(jid)
    assert jl["output_qty"] == pytest.approx(555.556) and jl["variance_reason"] == "Vendor MOQ"
    g = _lines(jid)[GREY]
    assert g["issue_qty"] == 500 and g["bom_issue_qty"] == 450 and g["variance_reason"] == "Vendor MOQ"
    adj = _adjustments(get_min_by_jwo_id(jid)["id"])
    assert adj and adj[0]["new_issue_qty"] == 500 and adj[0]["user_name"] == "planner"

    # Re-saving the header keeps the revision and originals.
    purchase_db.update_jwo(jid, {"remarks": "x", "lines": [{**line, "input_qty": 500, "output_qty": jl["output_qty"]}]})
    jl2 = _jwo_line(jid)
    assert jl2["original_output_qty"] == 500 and jl2["variance_reason"] == "Vendor MOQ"


def test_confirmed_issue_blocks_direct_edits():
    _seed_stock(GREY, 1000)
    jid = _jwo()
    note = get_min_by_jwo_id(jid)
    purchase_db.update_min_status(note["id"], "Confirmed")
    lid = _lines(jid)[GREY]["id"]
    with pytest.raises(ValueError, match="confirmed"):
        v.revise_issue_qty(note["id"], lid, 500, reason="Bundle Size")
    base = _jwo_line(jid)
    line = {k: base[k] for k in ("input_material", "output_material", "output_qty", "rate", "process_type")}
    with pytest.raises(ValueError, match="confirmed"):
        purchase_db.update_jwo(jid, {"lines": [{**line, "input_qty": 500, "variance_reason": "Bundle Size"}]})
    assert _lines(jid)[GREY]["issue_qty"] == 450


def test_post_issue_correction_posts_stock_and_audits():
    iid = _seed_stock(GREY, 1000)
    jid = _jwo()
    note = get_min_by_jwo_id(jid)
    v.revise_issue_qty(note["id"], _lines(jid)[GREY]["id"], 500, reason="Bundle Size")
    purchase_db.update_min_status(note["id"], "Confirmed")
    assert _stock(iid) == 500

    lid = _lines(jid)[GREY]["id"]
    with pytest.raises(ValueError, match="Reason is required"):
        v.correct_confirmed_issue(note["id"], lid, 520, reason="")
    res = v.correct_confirmed_issue(note["id"], lid, 520, reason="Roll / Than Length", user_name="manager")
    assert res["stock_direction"] == "OUT" and res["delta_qty"] == 20 and res["stock_posted"]
    assert _stock(iid) == 480
    g = _lines(jid)[GREY]
    assert g["issue_qty"] == 520 and g["bom_issue_qty"] == 450
    assert g["expected_output_qty"] == pytest.approx(577.778)
    assert _jwo_line(jid)["output_qty"] == pytest.approx(577.778)

    v.correct_confirmed_issue(note["id"], lid, 490, reason="Planning Change", user_name="manager")
    assert _stock(iid) == 510
    post = [a for a in _adjustments(note["id"]) if a["phase"] == "POST_ISSUE"]
    assert [(a["old_issue_qty"], a["new_issue_qty"], a["stock_posted"]) for a in post] == [(520, 490, 1), (500, 520, 1)]

    # Cancelling the confirmed MIN reverses the corrected qty → stock back to start.
    purchase_db.update_min_status(note["id"], "Cancelled")
    assert _stock(iid) == 1000


def test_correction_requires_confirmed_issue():
    jid = _jwo()
    note = get_min_by_jwo_id(jid)
    with pytest.raises(ValueError, match="Only a confirmed"):
        v.correct_confirmed_issue(note["id"], _lines(jid)[GREY]["id"], 500, reason="Bundle Size")


def test_variance_report():
    conn = purchase_db._connect()
    conn.execute("INSERT INTO pr_headers(pr_number, pr_date, pr_type, source) VALUES('PR-0009','2026-10-01','Job Work','MRP')")
    pr_id = conn.execute("SELECT id FROM pr_headers WHERE pr_number='PR-0009'").fetchone()[0]
    conn.execute("INSERT INTO pr_lines(pr_id, material_code, required_qty, unit) VALUES(?,?,?,?)", (pr_id, PRINTED, 480, "MTR"))
    conn.commit()
    conn.close()

    jid = _jwo()
    jwo_number = purchase_db.get_jwo_by_id(jid)["jwo_number"]
    note = get_min_by_jwo_id(jid)
    v.revise_issue_qty(note["id"], _lines(jid)[GREY]["id"], 500, reason="Bundle Size")

    conn = purchase_db._connect()
    conn.execute(
        "INSERT INTO grn_headers(grn_number, grn_date, grn_type, reference_number, status) VALUES('GRN-9','2026-10-05','JWO Receipt',?,'Verified')",
        (jwo_number,),
    )
    gid = conn.execute("SELECT id FROM grn_headers WHERE grn_number='GRN-9'").fetchone()[0]
    conn.execute("INSERT INTO grn_lines(grn_id, material_code, received_qty, accepted_qty) VALUES(?,?,?,?)", (gid, PRINTED, 545, 540))
    conn.commit()
    conn.close()

    rep = v.printing_variance_report(only_variance=True)
    assert len(rep["rows"]) == 2  # grey + thread lines both moved with the revised output
    row = next(r for r in rep["rows"] if r["grey_code"] == GREY)
    assert row["mrp_requirement_qty"] == 480
    assert row["original_planned_output"] == 500 and row["bom_grey_requirement"] == 450
    assert row["revised_grey_issue"] == 500 and row["revised_expected_output"] == pytest.approx(555.556)
    assert row["actual_received_qty"] == 540
    assert row["excess_grey_qty"] == 50 and row["excess_grey_pct"] == pytest.approx(11.11)
    assert row["excess_output_qty"] == pytest.approx(55.556) and row["excess_output_pct"] == pytest.approx(11.11)
    assert row["received_vs_original_qty"] == 40 and row["reason"] == "Bundle Size"
    assert row["output_vs_mrp_qty"] == pytest.approx(75.556) and row["output_vs_mrp_pct"] == pytest.approx(15.74)
    s = rep["summary"]
    assert s["original_output"] == 500 and s["expected_output"] == pytest.approx(555.556) and s["received"] == 540
    assert s["mrp_requirement"] == 480 and s["output_vs_mrp_qty"] == pytest.approx(75.556)
    assert rep["by_reason"][0]["reason"] == "Bundle Size"
    assert v.printing_variance_report(reason="Vendor MOQ")["rows"] == []


def test_http_endpoints(client, monkeypatch):
    _seed_stock(GREY, 1000)
    jid = _jwo()
    note = get_min_by_jwo_id(jid)
    lid = _lines(jid)[GREY]["id"]
    bad = client.post(f"/api/purchase/min/{note['id']}/lines/{lid}/revise", json={"issue_qty": 500})
    assert bad.status_code == 400 and "Reason is required" in bad.json()["detail"]
    ok = client.post(f"/api/purchase/min/{note['id']}/lines/{lid}/revise", json={"issue_qty": 500, "reason": "Bundle Size"})
    assert ok.status_code == 200, ok.text
    assert ok.json()["expected_output_qty"] == pytest.approx(555.556)

    purchase_db.update_min_status(note["id"], "Confirmed")
    lid = _lines(jid)[GREY]["id"]
    monkeypatch.setattr(
        "backend.main.decode_token", lambda t: {"sub": "clerk", "role": "Employee", "permissions": []} if t == "test-token" else None
    )
    denied = client.post(f"/api/purchase/min/{note['id']}/lines/{lid}/correct", json={"issue_qty": 510, "reason": "Bundle Size"})
    assert denied.status_code == 403
    monkeypatch.setattr(
        "backend.main.decode_token", lambda t: {"sub": "mgr", "role": "Manager", "permissions": []} if t == "test-token" else None
    )
    fixed = client.post(f"/api/purchase/min/{note['id']}/lines/{lid}/correct", json={"issue_qty": 510, "reason": "Bundle Size"})
    assert fixed.status_code == 200, fixed.text
    assert client.get(f"/api/purchase/min/{note['id']}/adjustments").json()[0]["user_name"] == "mgr"
    rep = client.get("/api/purchase/reports/printing-variance", params={"only_variance": True}).json()
    assert any(r["grey_code"] == GREY and r["revised_grey_issue"] == 510 for r in rep["rows"])
    assert "Bundle Size" in client.get("/api/purchase/variance-reasons").json()
