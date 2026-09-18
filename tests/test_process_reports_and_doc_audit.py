"""Process balance reports + document audit registry."""
from __future__ import annotations

import uuid

import pytest

from backend.db import document_audit_db as audit
from backend.db import item_db, purchase_db
from backend.services.cutting_reports import build_cutting_report, _process_sql_names


@pytest.fixture()
def audit_env(tmp_path, monkeypatch):
    pdb = str(tmp_path / "purchase.db")
    idb = str(tmp_path / "items.db")
    monkeypatch.setenv("PURCHASE_DB_PATH", pdb)
    monkeypatch.setenv("DOCUMENT_AUDIT_DB_PATH", pdb)
    monkeypatch.setenv("ITEM_DB_PATH", idb)
    monkeypatch.setattr(purchase_db, "_DB", pdb)
    monkeypatch.setattr(audit, "DB_PATH", pdb)
    monkeypatch.setattr(item_db, "DB_PATH", idb)
    purchase_db.init_db()
    item_db.init_db()
    audit.init_db()
    return audit


def test_process_sql_names_kaj_and_shirring():
    assert "Kajh Button" in _process_sql_names("Kajh Button")
    assert "Kaj Button" in _process_sql_names("Kaj Button")
    assert "Shirring (Bobbin Elastic)" in _process_sql_names("Shirring (Bobbin Elastic)")


def test_shirring_seeded_in_routing(tmp_path, monkeypatch):
    idb = str(tmp_path / "items_r.db")
    monkeypatch.setenv("ITEM_DB_PATH", idb)
    monkeypatch.setattr(item_db, "DB_PATH", idb)
    item_db.init_db()
    steps = item_db.list_routing_steps()
    names = {s["name"] for s in steps}
    assert "Shirring (Bobbin Elastic)" in names


def test_document_audit_verify_lock_unverify(audit_env):
    audit.enroll_document(
        "PO",
        101,
        doc_number="PO-TEST-1",
        so_reference="SO1",
        party_name="Vendor",
        doc_date="2026-09-17",
        created_by="tester",
    )
    assert audit.is_verified("PO", 101) is False
    audit.verify_document("PO", 101, actor="Accounts")
    assert audit.is_verified("PO", 101) is True
    with pytest.raises(ValueError, match="Verified"):
        audit.assert_doc_editable("PO", 101)
    audit.unverify_document("PO", 101, actor="Accounts", reason="Typo in qty")
    assert audit.is_verified("PO", 101) is False
    audit.assert_doc_editable("PO", 101)  # should not raise
    events = audit.list_audit_events("PO", 101)
    types = [e["event_type"] for e in events]
    assert "created" in types and "verified" in types and "unverified" in types


def test_unverify_requires_reason(audit_env):
    audit.enroll_document("JO", 5, doc_number="JO-1")
    audit.verify_document("JO", 5, actor="A")
    with pytest.raises(ValueError, match="Reason"):
        audit.unverify_document("JO", 5, actor="A", reason="")


def test_list_documents_fast_without_date_filter(audit_env):
    for i in range(5):
        audit.enroll_document("JO", 1000 + i, doc_number=f"JO-{i}", process_name="Cutting")
    out = audit.list_documents(audit_status="Pending", limit=10)
    assert out["total"] >= 5
    assert out["pending"] >= 5
    assert len(out["rows"]) >= 5


def test_get_document_detail_skips_blockers_when_requested(audit_env):
    audit.enroll_document("GRN", 9, doc_number="GRN-9")
    d = audit.get_document_detail("GRN", 9, include_blockers=False)
    assert d["doc_number"] == "GRN-9"
    assert d["dependency_blockers"] == []
    assert d["events"]


def test_print_brand_uses_embedded_logo():
    from backend.services.print_brand import brand_header_html, logo_data_url

    url = logo_data_url()
    assert url.startswith("data:image/png;base64,")
    html = brand_header_html(doc_title="PURCHASE ORDER", doc_number="PO-1", department="Purchase")
    assert "Yash Gallery Pvt. Ltd." in html
    assert "Bhiwandi" in html
    assert url in html

