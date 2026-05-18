"""JWO creation must auto-generate linked Material Issue Note from BOM."""
import pandas as pd
import pytest

from backend.db import purchase_db
from backend.services.jwo_min_notes import create_min_for_jwo, get_min_by_jwo_id


@pytest.fixture
def purchase_db_tmp(tmp_path, monkeypatch):
    db = tmp_path / "purchase_test.db"
    monkeypatch.setattr(purchase_db, "_DB", str(db))
    purchase_db.init_db()
    return db


def test_create_min_for_jwo_from_bom_lines(purchase_db_tmp, monkeypatch):
    """Uses JWO line input/output when BOM DB unavailable in test."""
    jwo = {
        "jwo_date": "2026-05-16",
        "processor_name": "Test Printer",
        "so_reference": "SO-1",
        "issued_by": "Stores",
    }
    lines = [
        {
            "output_material": "PRINTED-FAB-01",
            "output_qty": 100,
            "input_material": "GREY-FAB-01",
            "input_qty": 200,
            "input_unit": "MTR",
        }
    ]

    monkeypatch.setattr(
        "backend.services.jwo_min_notes.explode_bom_materials",
        lambda code, name, qty: [],
    )
    monkeypatch.setattr(
        "backend.services.jwo_min_notes._lookup_item_name",
        lambda c: "Printed Fabric" if "PRINTED" in c else "Grey Fabric",
    )

    note = create_min_for_jwo(1, "JWO-0001", jwo, lines)
    assert note is not None
    assert note["jwo_reference"] == "JWO-0001"
    assert note["jwo_date"] == "2026-05-16"
    assert len(note["lines"]) >= 1
    ln = note["lines"][0]
    assert ln["material_code"] == "GREY-FAB-01"
    assert float(ln["issue_qty"]) == 200
    assert ln["output_material"] == "PRINTED-FAB-01"
    assert float(ln["output_qty"]) == 100

    again = get_min_by_jwo_id(1)
    assert again["min_number"] == note["min_number"]
