"""Production JO MIN control: tolerance, approval, stock on confirm (actual qty), partial issues."""
from __future__ import annotations

import sqlite3

import pytest

from backend.services import min_tolerance


# ── tolerance helper ─────────────────────────────────────────────────────────

def test_tolerance_helper_default_and_override(monkeypatch):
    monkeypatch.delenv("MIN_ISSUE_TOLERANCE_PCT", raising=False)
    assert min_tolerance.get_tolerance_pct() == 10.0
    assert not min_tolerance.exceeds_tolerance(100, 100)
    assert not min_tolerance.exceeds_tolerance(100, 110)  # boundary = within
    assert not min_tolerance.exceeds_tolerance(100, 91)
    assert min_tolerance.exceeds_tolerance(100, 111)
    assert min_tolerance.exceeds_tolerance(100, 89)
    assert min_tolerance.exceeds_tolerance(0, 5)  # issue with no BOM requirement
    assert min_tolerance.variance_pct(100, 90) == -10.0
    assert min_tolerance.variance_pct(0, 5) is None

    monkeypatch.setenv("MIN_ISSUE_TOLERANCE_PCT", "5")
    assert min_tolerance.get_tolerance_pct() == 5.0
    assert min_tolerance.exceeds_tolerance(100, 106)
    assert not min_tolerance.exceeds_tolerance(100, 105)


def test_reason_validation():
    allowed = min_tolerance.PRODUCTION_VARIANCE_REASONS
    assert min_tolerance.validate_reason("", "", required=False, allowed=allowed) == ("", "")
    with pytest.raises(ValueError):
        min_tolerance.validate_reason("", "", required=True, allowed=allowed)
    with pytest.raises(ValueError):
        min_tolerance.validate_reason("Bogus", "", required=True, allowed=allowed)
    with pytest.raises(ValueError):
        min_tolerance.validate_reason("Other", "", required=True, allowed=allowed)
    assert min_tolerance.validate_reason("Wastage", "", required=True, allowed=allowed)[0] == "Wastage"


# ── fixtures (same isolation pattern as test_production_module_smoke) ────────

@pytest.fixture
def dbs(tmp_path, monkeypatch):
    paths = {
        "SALES_DB_PATH": str(tmp_path / "sales.db"),
        "PURCHASE_DB_PATH": str(tmp_path / "purchase.db"),
        "PRODUCTION_DB_PATH": str(tmp_path / "production.db"),
        "GREY_DB_PATH": str(tmp_path / "grey.db"),
        "ITEM_DB_PATH": str(tmp_path / "items.db"),
    }
    for k, v in paths.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MIN_ISSUE_TOLERANCE_PCT", raising=False)

    from backend.db import grey_db, item_db, production_db, purchase_db, sales_db
    from backend.routers import production as production_router
    from backend.services import jo_issue_notes

    monkeypatch.setattr(sales_db, "_DB", paths["SALES_DB_PATH"])
    monkeypatch.setattr(purchase_db, "_DB", paths["PURCHASE_DB_PATH"])
    monkeypatch.setattr(production_db, "_DB", paths["PRODUCTION_DB_PATH"])
    if hasattr(production_db, "_ITEM_DB"):
        monkeypatch.setattr(production_db, "_ITEM_DB", paths["ITEM_DB_PATH"])
    monkeypatch.setattr(grey_db, "_DB", paths["GREY_DB_PATH"])
    monkeypatch.setattr(item_db, "DB_PATH", paths["ITEM_DB_PATH"])
    monkeypatch.setattr(production_router, "_ITEM_DB_PATH", paths["ITEM_DB_PATH"])
    monkeypatch.setattr(jo_issue_notes, "_PROD_DB", paths["PRODUCTION_DB_PATH"])
    monkeypatch.setattr(jo_issue_notes, "_ITEM_DB", paths["ITEM_DB_PATH"])

    sales_db.init_db()
    purchase_db.init_db()
    production_db.init_db()
    grey_db.init_db()
    item_db.init_db()
    return paths


def _seed_bom(item_db_path: str, parent_code: str = "STYLE1") -> None:
    conn = sqlite3.connect(item_db_path)
    type_id = conn.execute("SELECT id FROM item_types WHERE code='FG' OR name LIKE 'Finished%' LIMIT 1").fetchone()
    if type_id is None:
        type_id = conn.execute("INSERT INTO item_types (name, code) VALUES ('Finished Good','FG')").lastrowid
    else:
        type_id = type_id[0]
    parent_id = conn.execute(
        "INSERT INTO items (item_code, item_name, item_type_id) VALUES (?,?,?)",
        (parent_code, f"{parent_code} parent", type_id),
    ).lastrowid
    fab_id = conn.execute(
        "INSERT INTO items (item_code, item_name, item_type_id) VALUES ('FAB-CTN','Cotton Fabric',?)",
        (type_id,),
    ).lastrowid
    bom_id = conn.execute(
        "INSERT INTO bom_headers (item_id, bom_name, applies_to, is_default) VALUES (?, 'Default', 'all', 1)",
        (parent_id,),
    ).lastrowid
    conn.execute(
        """INSERT INTO bom_lines (bom_id, component_item_id, component_name, component_type, quantity, unit)
           VALUES (?, ?, 'Cotton Fabric', 'RM', 1.5, 'MTR')""",
        (bom_id, fab_id),
    )
    conn.commit()
    conn.close()


def _set_item_stock(path: str, code: str, qty: float) -> None:
    conn = sqlite3.connect(path)
    conn.execute("UPDATE items SET stock=? WHERE item_code=?", (qty, code))
    conn.commit()
    conn.close()


def _item_stock(path: str, code: str) -> float:
    conn = sqlite3.connect(path)
    v = conn.execute("SELECT stock FROM items WHERE item_code=?", (code,)).fetchone()[0]
    conn.close()
    return float(v or 0)


def _set_checked_stock(grey_path: str, code: str, qty: float) -> None:
    conn = sqlite3.connect(grey_path)
    conn.execute(
        """INSERT INTO printed_fabric_checked_stock(fabric_code, fabric_name, checked_qty, passed_qty, available_qty)
           VALUES(?,?,?,?,?)""",
        (code, "Cotton Fabric", qty, qty, qty),
    )
    conn.commit()
    conn.close()


def _checked_available(grey_path: str, code: str) -> float:
    conn = sqlite3.connect(grey_path)
    v = conn.execute("SELECT available_qty FROM printed_fabric_checked_stock WHERE fabric_code=?", (code,)).fetchone()
    conn.close()
    return float(v[0]) if v else 0.0


def _make_jo(client, dbs, *, stock: float = 1000.0, planned: int = 100) -> dict:
    _seed_bom(dbs["ITEM_DB_PATH"])
    _set_item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN", stock)
    r = client.post(
        "/api/production/orders",
        json={
            "jo_date": "2026-05-15", "so_number": "SO-MIN-1", "so_source": "manual", "sku": "STYLE1", "sku_name": "Printed shirt",
            "process": "Cutting", "planned_qty": planned,
            "fabric_code": "FAB-CTN", "fabric_qty": 100, "fabric_unit": "MTR",
        },
    )
    assert r.status_code == 200, r.text
    jo_num = r.json()["jo_number"]
    jo = next(o for o in client.get("/api/production/orders").json() if o["jo_number"] == jo_num)
    note = client.get(f"/api/production/orders/{jo['id']}/issue-note").json()
    return {"jo": jo, "note": note, "line": next(l for l in note["lines"] if l["material_code"] == "FAB-CTN")}


def _as(monkeypatch, user: str, role: str = "Manager") -> None:
    """Switch the authenticated test user (token decode is patched in conftest)."""
    monkeypatch.setattr(
        "backend.main.decode_token",
        lambda t: {"sub": user, "role": role, "permissions": []} if t == "test-token" else None,
    )


def _patch(client, note_id, line_id, qty, reason="", remarks=""):
    return client.patch(
        f"/api/production/issue-notes/{note_id}/lines/{line_id}",
        json={"issue_qty": qty, "reason": reason, "remarks": remarks},
    )


# ── schema defaults ──────────────────────────────────────────────────────────

def test_new_note_is_draft_with_frozen_bom_and_default_issue(dbs, client):
    ctx = _make_jo(client, dbs)
    note, line = ctx["note"], ctx["line"]
    assert note["status"] == "Draft"
    assert line["required_qty"] == pytest.approx(150.0)
    assert line["bom_required_qty"] == pytest.approx(150.0)
    assert line["issue_qty"] == pytest.approx(150.0)
    settings = client.get("/api/production/issue-notes/settings").json()
    assert settings["tolerance_pct"] == 10.0
    assert "Mini CAD Average Change" in settings["reasons"]


# ── draft edit within tolerance / over tolerance needs approval ──────────────

def test_edit_within_tolerance_needs_reason_then_stays_draft(dbs, client):
    ctx = _make_jo(client, dbs)
    nid, lid = ctx["note"]["id"], ctx["line"]["id"]

    assert _patch(client, nid, lid, 140).status_code == 400  # reason required when ≠ BOM
    assert _patch(client, nid, lid, 140, "Other").status_code == 400  # Other needs remarks
    assert _patch(client, nid, lid, 140, "Bogus").status_code == 400
    assert _patch(client, nid, lid, 150, "Wastage").status_code == 400  # unchanged

    r = _patch(client, nid, lid, 140, "Wastage")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "Draft" and body["approval_required"] is False
    ln = next(l for l in body["issue_note"]["lines"] if l["id"] == lid)
    assert ln["issue_qty"] == pytest.approx(140.0)
    assert ln["bom_required_qty"] == pytest.approx(150.0)  # BOM stays frozen
    assert ln["variance_qty"] == pytest.approx(-10.0)
    assert ln["variance_pct"] == pytest.approx(-6.67, abs=0.01)
    assert ln["variance_reason"] == "Wastage"

    adj = client.get(f"/api/production/issue-notes/{nid}/adjustments").json()
    assert len(adj) == 1 and adj[0]["phase"] == "PRE_ISSUE" and adj[0]["new_issue_qty"] == pytest.approx(140.0)


def test_over_tolerance_requires_manager_approval_before_confirm(dbs, client, monkeypatch):
    ctx = _make_jo(client, dbs)
    nid, lid = ctx["note"]["id"], ctx["line"]["id"]

    r = _patch(client, nid, lid, 200, "Mini CAD Average Change")  # +33%
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "Pending Approval" and r.json()["approval_required"] is True

    c = client.post(f"/api/production/issue-notes/{nid}/confirm")
    assert c.status_code == 400 and "approve" in c.json()["detail"].lower()
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(1000.0)

    # Non-manager cannot approve
    monkeypatch.setattr(
        "backend.main.decode_token",
        lambda t: {"sub": "clerk1", "role": "Clerk", "permissions": []} if t == "test-token" else None,
    )
    assert client.post(f"/api/production/issue-notes/{nid}/approve").status_code == 403

    # Manager / Admin can
    monkeypatch.setattr(
        "backend.main.decode_token",
        lambda t: {"sub": "mgr1", "role": "Manager", "permissions": []} if t == "test-token" else None,
    )
    a = client.post(f"/api/production/issue-notes/{nid}/approve")
    assert a.status_code == 200, a.text
    assert a.json()["issue_note"]["status"] == "Draft"
    assert a.json()["issue_note"]["approved_by"] == "mgr1"

    # Editing the qty again voids the approval (back to Pending if still over).
    r2 = _patch(client, nid, lid, 210, "Mini CAD Average Change")
    assert r2.json()["status"] == "Pending Approval"
    assert client.get(f"/api/production/issue-notes/{nid}").json()["approved_by"] == ""

    # Back within tolerance → Draft, confirmable without approval.
    r3 = _patch(client, nid, lid, 160, "Wastage")
    assert r3.json()["status"] == "Draft" and r3.json()["approval_required"] is False
    assert client.post(f"/api/production/issue-notes/{nid}/confirm").status_code == 200


# ── stock on confirm uses the actual issue qty ───────────────────────────────

def test_confirm_deducts_actual_issue_qty_and_posts_printed_fabric(dbs, client):
    ctx = _make_jo(client, dbs)
    _set_checked_stock(dbs["GREY_DB_PATH"], "FAB-CTN", 500.0)
    nid, lid, joid = ctx["note"]["id"], ctx["line"]["id"], ctx["jo"]["id"]

    assert _patch(client, nid, lid, 140, "Roll/Bundle Adjustment").status_code == 200
    r = client.post(f"/api/production/issue-notes/{nid}/confirm", json={"issued_by": "Stores", "received_by": "Cutter"})
    assert r.status_code == 200, r.text
    note = r.json()["issue_note"]
    assert note["status"] == "Confirmed" and note["stock_posted"] == 1
    assert note["issued_by"] == "Stores" and note["received_by"] == "Cutter"
    assert note["confirmed_by"] == "tester"

    # Actual 140 — not BOM 150 — leaves Item Master and Checked stock.
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(860.0)
    assert _checked_available(dbs["GREY_DB_PATH"], "FAB-CTN") == pytest.approx(360.0)

    jo = client.get(f"/api/production/orders/{joid}").json()
    issues = jo["fabric_issues"]
    assert len(issues) == 1 and issues[0]["fabric_code"] == "FAB-CTN"
    assert issues[0]["issued_qty"] == pytest.approx(140.0)
    assert jo["fabric_issued_qty"] == pytest.approx(140.0)

    # Confirmed notes are read-only and cannot be confirmed twice.
    assert _patch(client, nid, lid, 130, "Wastage").status_code == 400
    assert client.post(f"/api/production/issue-notes/{nid}/confirm").status_code == 400
    # Refresh BOM must not rewrite lines under a posted MIN.
    regen = client.post(f"/api/production/orders/{joid}/regenerate-issue-note")
    assert regen.status_code == 200
    assert regen.json()["issue_note"]["status"] == "Confirmed"
    assert regen.json()["issue_note"]["lines"][0]["issue_qty"] == pytest.approx(140.0)


def test_confirm_non_printed_material_only_moves_item_master(dbs, client):
    ctx = _make_jo(client, dbs)
    nid = ctx["note"]["id"]
    assert client.post(f"/api/production/issue-notes/{nid}/confirm").status_code == 200
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(850.0)  # BOM 150 when unchanged
    assert client.get(f"/api/production/orders/{ctx['jo']['id']}").json()["fabric_issues"] == []


def test_negative_stock_blocks_confirm(dbs, client):
    ctx = _make_jo(client, dbs, stock=100.0)
    nid = ctx["note"]["id"]
    r = client.post(f"/api/production/issue-notes/{nid}/confirm")
    assert r.status_code == 400 and "insufficient" in r.json()["detail"].lower()
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(100.0)
    assert client.get(f"/api/production/issue-notes/{nid}").json()["status"] == "Draft"


def test_insufficient_checked_printed_fabric_blocks_confirm(dbs, client):
    ctx = _make_jo(client, dbs)
    _set_checked_stock(dbs["GREY_DB_PATH"], "FAB-CTN", 50.0)
    nid = ctx["note"]["id"]
    r = client.post(f"/api/production/issue-notes/{nid}/confirm")
    assert r.status_code == 400 and "checked" in r.json()["detail"].lower()
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(1000.0)
    assert _checked_available(dbs["GREY_DB_PATH"], "FAB-CTN") == pytest.approx(50.0)


# ── partial / additional MIN ─────────────────────────────────────────────────

def test_partial_issue_then_additional_for_remaining(dbs, client, monkeypatch):
    ctx = _make_jo(client, dbs)
    nid, lid, joid = ctx["note"]["id"], ctx["line"]["id"], ctx["jo"]["id"]

    # Additional only after the primary is confirmed.
    assert client.post(f"/api/production/orders/{joid}/issue-notes/additional").status_code == 400

    # Partial first issue (90 of 150 = -40%) → approval needed (Stock Shortage).
    assert _patch(client, nid, lid, 90, "Stock Shortage").json()["status"] == "Pending Approval"
    assert client.post(f"/api/production/issue-notes/{nid}/approve").status_code == 400  # issuer ≠ approver
    _as(monkeypatch, "mgr1")
    assert client.post(f"/api/production/issue-notes/{nid}/approve").status_code == 200
    _as(monkeypatch, "tester", "Admin")
    assert client.post(f"/api/production/issue-notes/{nid}/confirm").status_code == 200
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(910.0)

    add = client.post(f"/api/production/orders/{joid}/issue-notes/additional")
    assert add.status_code == 200, add.text
    extra = add.json()["issue_note"]
    assert extra["parent_in_id"] == nid and extra["in_number"] != ctx["note"]["in_number"]
    assert extra["status"] == "Draft"
    eline = extra["lines"][0]
    assert eline["bom_required_qty"] == pytest.approx(60.0)  # remaining = 150 − 90 confirmed
    assert eline["issue_qty"] == pytest.approx(60.0)
    cum = next(c for c in extra["cumulative"] if c["material_code"] == "FAB-CTN")
    assert cum["bom_required_qty"] == pytest.approx(150.0)
    assert cum["confirmed_issued_qty"] == pytest.approx(90.0)
    assert cum["pending_issue_qty"] == pytest.approx(60.0)

    # Only one open additional note at a time.
    assert client.post(f"/api/production/orders/{joid}/issue-notes/additional").status_code == 400

    assert client.post(f"/api/production/issue-notes/{extra['id']}/confirm").status_code == 200
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(850.0)  # 90 + 60 = BOM 150

    cum2 = client.get(f"/api/production/orders/{joid}/issue-note").json()
    c = next(c for c in cum2["cumulative"] if c["material_code"] == "FAB-CTN")
    assert c["confirmed_issued_qty"] == pytest.approx(150.0) and c["remaining_qty"] == pytest.approx(0.0)
    assert len(cum2["additional_notes"]) == 1

    # Nothing left against BOM → default additional rejected; explicit extra is judged on the cumulative.
    assert client.post(f"/api/production/orders/{joid}/issue-notes/additional").status_code == 400
    ex = client.post(
        f"/api/production/orders/{joid}/issue-notes/additional",
        json={"lines": [{"material_code": "FAB-CTN", "issue_qty": 30, "reason": "Wastage"}]},
    )
    assert ex.status_code == 200, ex.text
    assert ex.json()["status"] == "Pending Approval"  # 150 + 30 = 180 vs BOM 150 → +20% > 10%

    # "Extra issue" with no explicit lines: zero-qty Draft lines the issuer edits (cancel the open one first).
    assert client.post(f"/api/production/issue-notes/{ex.json()['issue_note']['id']}/cancel").status_code == 200
    ex2 = client.post(f"/api/production/orders/{joid}/issue-notes/additional", json={"extra": True})
    assert ex2.status_code == 200, ex2.text
    assert ex2.json()["status"] == "Draft"
    xl = ex2.json()["issue_note"]["lines"][0]
    assert xl["issue_qty"] == pytest.approx(0.0) and xl["bom_required_qty"] == pytest.approx(0.0)
    assert client.post(f"/api/production/issue-notes/{ex2.json()['issue_note']['id']}/confirm").status_code == 400  # nothing to issue


# ── authorized correction after confirm ──────────────────────────────────────

def test_confirmed_correction_posts_stock_delta_with_audit(dbs, client, monkeypatch):
    ctx = _make_jo(client, dbs)
    _set_checked_stock(dbs["GREY_DB_PATH"], "FAB-CTN", 500.0)
    nid, lid, joid = ctx["note"]["id"], ctx["line"]["id"], ctx["jo"]["id"]
    assert _patch(client, nid, lid, 140, "Wastage").status_code == 200
    assert client.post(f"/api/production/issue-notes/{nid}/confirm").status_code == 200

    # Draft-only: correct before confirm is invalid elsewhere; reason mandatory here.
    assert client.post(f"/api/production/issue-notes/{nid}/correct",
                       json={"line_id": lid, "issue_qty": 150}).status_code == 400

    r = client.post(f"/api/production/issue-notes/{nid}/correct",
                    json={"line_id": lid, "issue_qty": 150, "reason": "Rework"})
    assert r.status_code == 200, r.text
    assert r.json()["delta_qty"] == pytest.approx(10.0) and r.json()["stock_direction"] == "OUT"
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(850.0)
    assert _checked_available(dbs["GREY_DB_PATH"], "FAB-CTN") == pytest.approx(350.0)
    assert client.get(f"/api/production/orders/{joid}").json()["fabric_issued_qty"] == pytest.approx(150.0)

    r2 = client.post(f"/api/production/issue-notes/{nid}/correct",
                     json={"line_id": lid, "issue_qty": 130, "reason": "Wastage"})
    assert r2.status_code == 200, r2.text
    assert r2.json()["stock_direction"] == "IN"
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(870.0)
    assert _checked_available(dbs["GREY_DB_PATH"], "FAB-CTN") == pytest.approx(370.0)
    assert client.get(f"/api/production/orders/{joid}").json()["fabric_issued_qty"] == pytest.approx(130.0)

    audit = client.get(f"/api/production/issue-notes/{nid}/adjustments").json()
    post = [a for a in audit if a["phase"] == "POST_ISSUE"]
    assert len(post) == 2 and all(a["stock_posted"] == 1 for a in post)

    # Over-reducing below zero stock is blocked on an increase beyond stock.
    _set_item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN", 1.0)
    assert client.post(f"/api/production/issue-notes/{nid}/correct",
                       json={"line_id": lid, "issue_qty": 200, "reason": "Rework"}).status_code == 400

    # Non-managers cannot correct.
    monkeypatch.setattr(
        "backend.main.decode_token",
        lambda t: {"sub": "clerk1", "role": "Clerk", "permissions": []} if t == "test-token" else None,
    )
    assert client.post(f"/api/production/issue-notes/{nid}/correct",
                       json={"line_id": lid, "issue_qty": 120, "reason": "Rework"}).status_code == 403


# ── print payload ────────────────────────────────────────────────────────────

def test_print_payload_has_bom_issue_diff_percent(dbs, client):
    ctx = _make_jo(client, dbs)
    nid, lid = ctx["note"]["id"], ctx["line"]["id"]
    assert _patch(client, nid, lid, 135, "Wastage").status_code == 200
    p = client.get(f"/api/production/issue-notes/{nid}/print")
    assert p.status_code == 200, p.text
    body = p.json()
    assert body["in_number"] == ctx["note"]["in_number"] and body["jo_number"] == ctx["jo"]["jo_number"]
    ln = body["lines"][0]
    assert ln["bom_required_qty"] == pytest.approx(150.0)
    assert ln["issue_qty"] == pytest.approx(135.0)
    assert ln["diff_qty"] == pytest.approx(-15.0)
    assert ln["variance_pct"] == pytest.approx(-10.0)
    assert ln["variance_reason"] == "Wastage"
    assert body["tolerance_pct"] == 10.0


def test_cancel_additional_but_not_primary(dbs, client):
    ctx = _make_jo(client, dbs)
    nid, joid = ctx["note"]["id"], ctx["jo"]["id"]
    assert client.post(f"/api/production/issue-notes/{nid}/cancel").status_code == 400  # primary
    assert client.post(f"/api/production/issue-notes/{nid}/confirm").status_code == 200
    assert client.post(f"/api/production/orders/{joid}/issue-notes/additional",
                       json={"lines": [{"material_code": "FAB-CTN", "issue_qty": 5, "reason": "Wastage"}]}).status_code == 200
    extra = client.get(f"/api/production/orders/{joid}/issue-note").json()["additional_notes"][0]
    assert client.post(f"/api/production/issue-notes/{extra['id']}/cancel").status_code == 200
    assert client.get(f"/api/production/issue-notes/{extra['id']}").json()["status"] == "Cancelled"


# ── Purchase MIN tolerance (shared helper via jwo_issue_variance) ────────────

def test_purchase_min_over_tolerance_needs_approval(tmp_path, monkeypatch):
    from backend.db import purchase_db
    from backend.services import jwo_issue_variance as jv

    monkeypatch.setattr(purchase_db, "_DB", str(tmp_path / "purchase_tol.db"))
    purchase_db.init_db()
    monkeypatch.setattr(jv, "_assert_editable", lambda _id: None)
    monkeypatch.setattr(jv, "_audit_event", lambda *a, **k: None)
    monkeypatch.setattr("backend.db.document_audit_db.enroll_document", lambda *a, **k: None)
    monkeypatch.setattr(purchase_db, "_sync_item_master_stock", lambda *a, **k: None)
    monkeypatch.setattr(purchase_db, "_post_min_grey_ledger", lambda *a, **k: None)

    num = purchase_db.create_min({
        "jwo_reference": "JWO-T1",
        "lines": [{"material_code": "GREY-T", "issue_qty": 100, "unit": "MTR"}],
    })
    note = purchase_db.get_min_by_number(num)
    mid, lid = note["id"], note["lines"][0]["id"]

    # Within tolerance (+5%) → stays Draft, confirmable.
    r = jv.revise_issue_qty(mid, lid, 105, reason="Bundle Size", user_name="u1")
    assert r["status"] == "Draft" and r["approval_required"] is False and r["tolerance_pct"] == 10.0

    # +25% vs the frozen BOM → Pending Approval; confirm blocked.
    r = jv.revise_issue_qty(mid, lid, 125, reason="Vendor MOQ", user_name="u1")
    assert r["status"] == "Pending Approval" and r["approval_required"] is True
    with pytest.raises(ValueError, match="approve"):
        purchase_db.update_min_status(mid, "Confirmed")

    a = jv.approve_min(mid, user_name="mgr")
    assert a["issue_note"]["status"] == "Draft" and a["issue_note"]["approved_by"] == "mgr"
    purchase_db.update_min_status(mid, "Confirmed")
    done = purchase_db.get_min_by_number(num)
    assert done["status"] == "Confirmed" and done["stock_posted"] == 1


# ── reserved printed fabric is consumable on Confirm (SO-13 / 7081YKPINK regression) ─────────────

def _seed_reserved(grey_path, *, reserved, free=0.0, so="SO-MIN-1", sku="STYLE1", status="Active", code="FAB-CTN"):
    """Checked printed stock after an SO allocation: metres moved available → reserved."""
    conn = sqlite3.connect(grey_path)
    conn.execute(
        """INSERT INTO printed_fabric_checked_stock(fabric_code, fabric_name, checked_qty, passed_qty,
               reserved_qty, available_qty) VALUES(?,?,?,?,?,?)""",
        (code, "Cotton Fabric", reserved + free, reserved + free, reserved, free),
    )
    conn.execute(
        """INSERT INTO printed_fabric_reservations(fabric_code, fabric_name, so_number, sku, qty, unit, status)
           VALUES(?,?,?,?,?,'MTR',?)""",
        (code, "Cotton Fabric", so, sku, reserved, status),
    )
    conn.commit()
    conn.close()


def _stock_row(grey_path, code="FAB-CTN"):
    conn = sqlite3.connect(grey_path)
    r = conn.execute(
        "SELECT available_qty, reserved_qty FROM printed_fabric_checked_stock WHERE fabric_code=?", (code,)
    ).fetchone()
    conn.close()
    return float(r[0]), float(r[1])


def _reservations(grey_path, code="FAB-CTN"):
    conn = sqlite3.connect(grey_path)
    rows = conn.execute(
        "SELECT status, qty FROM printed_fabric_reservations WHERE fabric_code=? ORDER BY id", (code,)
    ).fetchall()
    conn.close()
    return [(r[0], float(r[1])) for r in rows]


@pytest.mark.parametrize("res_status", ["Active", "JO Created"])
def test_reserved_only_fabric_confirms_for_matching_so_jo(dbs, client, res_status):
    ctx = _make_jo(client, dbs)  # BOM 150, SO-MIN-1 / STYLE1
    _seed_reserved(dbs["GREY_DB_PATH"], reserved=150.0, free=0.0, status=res_status)
    nid, joid = ctx["note"]["id"], ctx["jo"]["id"]
    assert _stock_row(dbs["GREY_DB_PATH"]) == (0.0, 150.0)  # free = 0 — the old confirm refused here

    r = client.post(f"/api/production/issue-notes/{nid}/confirm")
    assert r.status_code == 200, r.text

    # Physical stock leaves once: Item Master −150, reservation consumed, free untouched (no double deduct).
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(850.0)
    assert _stock_row(dbs["GREY_DB_PATH"]) == (0.0, 0.0)
    assert _reservations(dbs["GREY_DB_PATH"]) == [("CUTTING_ISSUED", 150.0)]
    jo = client.get(f"/api/production/orders/{joid}").json()
    assert [i["issued_qty"] for i in jo["fabric_issues"]] == [150.0]


def test_reserved_plus_free_used_when_issue_exceeds_reservation(dbs, client):
    ctx = _make_jo(client, dbs)  # issue 150
    _seed_reserved(dbs["GREY_DB_PATH"], reserved=100.0, free=80.0)
    nid = ctx["note"]["id"]
    assert client.post(f"/api/production/issue-notes/{nid}/confirm").status_code == 200
    # 100 from the reservation + 50 from free; free 80 → 30, reserved fully consumed.
    assert _stock_row(dbs["GREY_DB_PATH"]) == (pytest.approx(30.0), 0.0)
    assert _reservations(dbs["GREY_DB_PATH"]) == [("CUTTING_ISSUED", 100.0)]
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(850.0)


def test_partial_issue_splits_reservation_and_keeps_the_rest_reserved(dbs, client):
    ctx = _make_jo(client, dbs)
    _seed_reserved(dbs["GREY_DB_PATH"], reserved=150.0, free=0.0)
    nid, lid = ctx["note"]["id"], ctx["line"]["id"]
    assert _patch(client, nid, lid, 140, "Wastage").status_code == 200
    assert client.post(f"/api/production/issue-notes/{nid}/confirm").status_code == 200
    assert _stock_row(dbs["GREY_DB_PATH"]) == (0.0, pytest.approx(10.0))
    assert sorted(_reservations(dbs["GREY_DB_PATH"])) == [("Active", 10.0), ("CUTTING_ISSUED", 140.0)]


def test_reserved_for_other_so_or_insufficient_total_still_blocked(dbs, client):
    ctx = _make_jo(client, dbs)
    nid = ctx["note"]["id"]
    # Another SO's reservation is not consumable by this JO.
    _seed_reserved(dbs["GREY_DB_PATH"], reserved=150.0, free=0.0, so="SO-OTHER", sku="STYLE1")
    r = client.post(f"/api/production/issue-notes/{nid}/confirm")
    assert r.status_code == 400 and "insufficient checked printed fabric" in r.json()["detail"].lower()
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(1000.0)
    assert _stock_row(dbs["GREY_DB_PATH"]) == (0.0, 150.0)
    assert client.get(f"/api/production/issue-notes/{nid}").json()["status"] == "Draft"


def test_own_reservation_short_of_issue_with_no_free_blocked(dbs, client):
    ctx = _make_jo(client, dbs)  # needs 150, only 100 reserved + 20 free
    _seed_reserved(dbs["GREY_DB_PATH"], reserved=100.0, free=20.0)
    nid = ctx["note"]["id"]
    r = client.post(f"/api/production/issue-notes/{nid}/confirm")
    assert r.status_code == 400 and "checked printed fabric" in r.json()["detail"].lower()
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(1000.0)
    assert _stock_row(dbs["GREY_DB_PATH"]) == (pytest.approx(20.0), pytest.approx(100.0))
    assert _reservations(dbs["GREY_DB_PATH"]) == [("Active", 100.0)]


# ── interim approval: maker-checker, reject, send back ───────────────────────

def _pending(client, dbs, monkeypatch):
    ctx = _make_jo(client, dbs)
    nid, lid = ctx["note"]["id"], ctx["line"]["id"]
    _as(monkeypatch, "issuer1", "Clerk")
    assert _patch(client, nid, lid, 200, "Mini CAD Average Change").json()["status"] == "Pending Approval"
    return ctx, nid, lid


def test_issuer_cannot_approve_own_pending_min(dbs, client, monkeypatch):
    ctx, nid, lid = _pending(client, dbs, monkeypatch)
    _as(monkeypatch, "issuer1", "Manager")  # even with a Manager role
    r = client.post(f"/api/production/issue-notes/{nid}/approve")
    assert r.status_code == 400 and "different" in r.json()["detail"].lower()
    assert client.get(f"/api/production/issue-notes/{nid}").json()["status"] == "Pending Approval"

    _as(monkeypatch, "mgr2", "Manager")
    a = client.post(f"/api/production/issue-notes/{nid}/approve", json={"remarks": "OK per Sir"})
    assert a.status_code == 200, a.text
    note = a.json()["issue_note"]
    assert note["approved_by"] == "mgr2" and note["approval_status"] == "Approved"
    assert note["approval_remarks"] == "OK per Sir"
    # Stock moves only on Confirm after approval.
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(1000.0)
    assert client.post(f"/api/production/issue-notes/{nid}/confirm").status_code == 200
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(800.0)


def test_send_back_requires_remarks_then_resubmit_and_approve(dbs, client, monkeypatch):
    ctx, nid, lid = _pending(client, dbs, monkeypatch)
    _as(monkeypatch, "mgr2", "Manager")
    assert client.post(f"/api/production/issue-notes/{nid}/send-back", json={}).status_code == 400
    sb = client.post(f"/api/production/issue-notes/{nid}/send-back", json={"remarks": "Why 200? Recheck marker"})
    assert sb.status_code == 200, sb.text
    note = sb.json()["issue_note"]
    assert note["status"] == "Draft" and note["approval_status"] == "Sent Back"
    assert note["approval_remarks"] == "Why 200? Recheck marker"
    # Cannot be confirmed while over tolerance and unapproved.
    assert client.post(f"/api/production/issue-notes/{nid}/confirm").status_code == 400

    _as(monkeypatch, "issuer1", "Clerk")
    rs = client.post(f"/api/production/issue-notes/{nid}/resubmit", json={"remarks": "Marker verified"})
    assert rs.status_code == 200 and rs.json()["issue_note"]["status"] == "Pending Approval"

    audit = client.get(f"/api/production/issue-notes/{nid}/adjustments").json()
    phases = [a["phase"] for a in audit]
    assert "SENT_BACK" in phases and "RESUBMITTED" in phases
    assert any(a["phase"] == "SENT_BACK" and a["remarks"] == "Why 200? Recheck marker" for a in audit)


def test_reject_primary_resets_to_bom_and_additional_is_closed(dbs, client, monkeypatch):
    ctx, nid, lid = _pending(client, dbs, monkeypatch)
    _as(monkeypatch, "mgr2", "Manager")
    assert client.post(f"/api/production/issue-notes/{nid}/reject", json={}).status_code == 400
    rj = client.post(f"/api/production/issue-notes/{nid}/reject", json={"remarks": "Not justified"})
    assert rj.status_code == 200, rj.text
    note = rj.json()["issue_note"]
    assert note["status"] == "Draft" and note["approval_status"] == "Rejected"
    assert note["lines"][0]["issue_qty"] == pytest.approx(150.0)  # reset to BOM
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(1000.0)
    assert any(a["phase"] == "REJECTED" for a in client.get(f"/api/production/issue-notes/{nid}/adjustments").json())

    # Clerks cannot decide.
    _as(monkeypatch, "clerk1", "Clerk")
    assert client.post(f"/api/production/issue-notes/{nid}/reject", json={"remarks": "x"}).status_code == 403
    assert client.post(f"/api/production/issue-notes/{nid}/send-back", json={"remarks": "x"}).status_code == 403

    # Additional MIN → Rejected (closed, no stock).
    _as(monkeypatch, "tester", "Admin")
    assert client.post(f"/api/production/issue-notes/{nid}/confirm").status_code == 200
    joid = ctx["jo"]["id"]
    ex = client.post(f"/api/production/orders/{joid}/issue-notes/additional",
                     json={"lines": [{"material_code": "FAB-CTN", "issue_qty": 40, "reason": "Wastage"}]})
    assert ex.json()["status"] == "Pending Approval"
    _as(monkeypatch, "mgr2", "Manager")
    r2 = client.post(f"/api/production/issue-notes/{ex.json()['issue_note']['id']}/reject", json={"remarks": "No"})
    assert r2.status_code == 200 and r2.json()["issue_note"]["status"] == "Rejected"
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(850.0)


# ── Additional MIN: cumulative tolerance vs BOM ──────────────────────────────

def test_additional_min_tolerance_is_cumulative_vs_bom(dbs, client, monkeypatch):
    ctx = _make_jo(client, dbs, planned=350)  # BOM 350 × 1.5 = 525
    nid, joid = ctx["note"]["id"], ctx["jo"]["id"]
    assert ctx["line"]["bom_required_qty"] == pytest.approx(525.0)
    assert client.post(f"/api/production/issue-notes/{nid}/confirm").status_code == 200

    # 525 + 20 = 545 vs BOM 525 → +3.81% (within 10%) → Draft, own number / date / user.
    add = client.post(f"/api/production/orders/{joid}/issue-notes/additional",
                      json={"lines": [{"material_code": "FAB-CTN", "issue_qty": 20, "reason": "Wastage"}]})
    assert add.status_code == 200, add.text
    extra = add.json()["issue_note"]
    assert add.json()["status"] == "Draft"
    assert extra["in_number"] != ctx["note"]["in_number"] and extra["parent_in_id"] == nid
    assert extra["created_by"] == "tester" and extra["in_date"]
    ln = extra["lines"][0]
    assert ln["cum_prior_issued_qty"] == pytest.approx(525.0) and ln["cum_total_qty"] == pytest.approx(545.0)
    assert ln["variance_pct"] == pytest.approx(3.81, abs=0.01)
    assert client.post(f"/api/production/issue-notes/{extra['id']}/confirm").status_code == 200
    assert _item_stock(dbs["ITEM_DB_PATH"], "FAB-CTN") == pytest.approx(1000.0 - 545.0)

    # Another +80 → 625 vs 525 = +19.05% → needs approval.
    again = client.post(f"/api/production/orders/{joid}/issue-notes/additional",
                        json={"lines": [{"material_code": "FAB-CTN", "issue_qty": 80, "reason": "Wastage"}]})
    assert again.status_code == 200 and again.json()["status"] == "Pending Approval"
    assert again.json()["issue_note"]["lines"][0]["variance_pct"] == pytest.approx(19.05, abs=0.01)
