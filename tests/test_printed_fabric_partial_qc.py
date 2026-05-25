"""Partial printed-fabric QC must leave pending qty in Unchecked."""
from __future__ import annotations

import pytest

from backend.db import grey_db


@pytest.fixture(autouse=True)
def isolated_grey(tmp_path, monkeypatch):
    grey_path = str(tmp_path / "grey.db")
    monkeypatch.setenv("GREY_DB_PATH", grey_path)
    monkeypatch.setattr(grey_db, "_DB", grey_path)
    grey_db.init_db()
    yield


def test_partial_qc_leaves_remainder_in_unchecked():
    grey_db.insert_printed_fabric_unchecked(
        "P308", 100, fabric_name="Print A", jwo_ref="JWO-1", grn_ref="GRN-0100"
    )
    out = grey_db.do_printed_fabric_qc(
        {
            "fabric_code": "P308",
            "fabric_name": "Print A",
            "jwo_ref": "JWO-1",
            "passed_qty": 50,
            "failed_qty": 0,
            "qc_by": "QC1",
        }
    )
    assert out["processed_qty"] == 50
    assert out["pending_qty"] == pytest.approx(50)

    unchecked = grey_db.list_printed_fabric_unchecked()
    assert len(unchecked) == 1
    assert unchecked[0]["fabric_code"] == "P308"
    assert unchecked[0]["jwo_ref"] == "JWO-1"
    assert float(unchecked[0]["qty"]) == pytest.approx(50)

    checked = grey_db.list_printed_fabric_checked()
    assert len(checked) == 1
    assert float(checked[0]["passed_qty"]) == pytest.approx(50)
    assert float(checked[0]["available_qty"]) == pytest.approx(50)


def test_second_partial_qc_closes_line_when_complete():
    grey_db.insert_printed_fabric_unchecked("FAB-1", 80, jwo_ref="J1", grn_ref="GRN-1")
    grey_db.do_printed_fabric_qc(
        {"fabric_code": "FAB-1", "jwo_ref": "J1", "passed_qty": 30, "failed_qty": 0, "qc_by": "A"}
    )
    grey_db.do_printed_fabric_qc(
        {"fabric_code": "FAB-1", "jwo_ref": "J1", "passed_qty": 50, "failed_qty": 0, "qc_by": "B"}
    )
    assert grey_db.list_printed_fabric_unchecked() == []
    checked = grey_db.list_printed_fabric_checked()
    assert float(checked[0]["passed_qty"]) == pytest.approx(80)


def test_partial_qc_rejects_over_pending():
    grey_db.insert_printed_fabric_unchecked("FAB-2", 40, jwo_ref="J2")
    with pytest.raises(ValueError, match="only"):
        grey_db.do_printed_fabric_qc(
            {"fabric_code": "FAB-2", "jwo_ref": "J2", "passed_qty": 50, "failed_qty": 0}
        )
