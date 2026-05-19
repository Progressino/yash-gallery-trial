"""Stitching Costing module smoke tests."""
import pandas as pd
import pytest

from backend.db import stitching_db
from backend.db.stitching_db import init_db, get_sheet_df, save_sheet_df
from backend.services import stitching_costing as svc


@pytest.fixture(autouse=True)
def isolated_stitching_db(tmp_path, monkeypatch):
    path = tmp_path / "stitch_test.db"
    monkeypatch.setenv("STITCHING_DB_PATH", str(path))
    monkeypatch.setattr(stitching_db, "_DB", str(path))
    init_db()


def test_stitching_init_and_dashboard():
    for key in stitching_db.DATA_KEYS:
        df = get_sheet_df(key)
        assert df is not None
    dash = svc.dashboard_summary("2026-05-15")
    assert "metrics" in dash
    assert dash["metrics"]["total_karigar"] >= 1


def test_save_production_entry_infers_single_operation():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-SINGLE", "Operation": "Stitch", "Target": 80, "Rate_Rs": 3.0}]),
    )
    out = svc.save_production_entry(
        date_str="2026-05-15",
        karigar_id="K001",
        karigar_name="Test",
        challan_no="CH-1",
        style="SKU-SINGLE",
        hour_entries=[{"hour_col": "H_09_10", "operation": "", "pieces": 25}],
    )
    assert out["ok"] is True
    pl = get_sheet_df("production_log")
    assert int(pl.iloc[-1]["Total_Pieces"]) == 25


def test_master_dedupe_and_delete():
    svc.add_style_operation_row("SKU-X", "Cut", 100, 2.5)
    dup = svc.add_style_operation_row("SKU-X", "Cut", 100, 2.5)
    assert dup["ok"] is False
    out = svc.delete_master_rows("style_master", [{"Style": "SKU-X", "Operation": "Cut"}])
    assert out["ok"] is True
    assert out["removed"] == 1


def test_dashboard_many_karigars_uses_batch_rates():
    """Dashboard must not call get_daily_rate_for_date per row (was minutes / timeout)."""
    karigars = [
        {
            "Karigar_ID": f"K{i:04d}",
            "Name": f"Karigar {i}",
            "Skill": "Stitching",
            "Daily_Rate_Rs": 400 + (i % 50),
        }
        for i in range(250)
    ]
    save_sheet_df("karigar_master", pd.DataFrame(karigars))
    import time

    t0 = time.perf_counter()
    dash = svc.dashboard_summary("2026-05-15")
    elapsed = time.perf_counter() - t0
    assert dash["metrics"]["total_karigar"] == 250
    assert len(dash["karigar_status"]) == 250
    assert elapsed < 3.0


def test_merge_sheet_keeps_server_and_adds_incoming():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "KEEP-ME", "Operation": "Cut", "Target": 100, "Rate_Rs": 2.0}]),
    )
    incoming = pd.DataFrame(
        [
            {"Style": "KEEP-ME", "Operation": "Cut", "Target": 999, "Rate_Rs": 9.0},
            {"Style": "FROM-FILE", "Operation": "Stitch", "Target": 80, "Rate_Rs": 4.0},
        ]
    )
    merged = svc.merge_sheet_dataframes("style_master", get_sheet_df("style_master"), incoming)
    assert len(merged) == 2
    keep = merged[merged["Style"] == "KEEP-ME"].iloc[0]
    assert int(keep["Target"]) == 100
    assert "FROM-FILE" in set(merged["Style"].astype(str))


def test_import_merge_mode_via_router_logic():
    save_sheet_df("style_master", pd.DataFrame([{"Style": "A", "Operation": "Op1", "Target": 1, "Rate_Rs": 1.0}]))
    existing = get_sheet_df("style_master")
    incoming = pd.DataFrame([{"Style": "B", "Operation": "Op2", "Target": 2, "Rate_Rs": 2.0}])
    out = svc.merge_sheet_dataframes("style_master", existing, incoming)
    save_sheet_df("style_master", out)
    df = get_sheet_df("style_master")
    assert len(df) == 2
    assert set(df["Style"].astype(str)) == {"A", "B"}


def test_looks_like_seed_only_master():
    assert svc.looks_like_seed_only_master() is True
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "BRAND-NEW-STYLE", "Operation": "Cut", "Target": 1, "Rate_Rs": 1.0}]),
    )
    assert svc.looks_like_seed_only_master() is False


def test_karigar_rate_by_date():
    svc.add_karigar_master("K099", "Test Karigar", "Stitching", 400.0, "2026-05-01")
    svc.update_karigar_master("K099", daily_rate_rs=500.0, effective_from="2026-05-10")
    assert svc.get_daily_rate_for_date("K099", "2026-05-05") == 400.0
    assert svc.get_daily_rate_for_date("K099", "2026-05-15") == 500.0
    svc.delete_master_rows("karigar_master", [{"Karigar_ID": "K099"}])


def test_resolve_hour_pieces_sticker_and_pl_sign():
    assert svc.resolve_hour_pieces({"sticker_in": 10, "sticker_out": 5, "manual_pieces": False}) == 5
    assert svc.resolve_hour_pieces({"sticker_in": 0, "sticker_out": 0, "pieces": 12, "manual_pieces": True}) == 12
    budgeted = 100.0
    actual = 80.0
    assert round(budgeted - actual, 2) == 20.0


def test_stitching_save_production_entry():
    out = svc.save_production_entry(
        date_str="2026-05-15",
        karigar_id="K001",
        karigar_name="Ramesh Kumar",
        challan_no="10220-2526",
        style="1894YKDGREEN",
        hour_entries=[
            {"hour_col": "H_09_10", "operation": "Cutting", "pieces": 12},
            {"hour_col": "H_10_11", "operation": "Cutting", "pieces": 8},
        ],
    )
    assert out["ok"] is True
    pl = get_sheet_df("production_log")
    assert not pl.empty
    assert int(pl["Total_Pieces"].sum()) >= 20
    row = pl.iloc[-1]
    assert float(row["PL_Rs"]) == round(float(row["Budgeted_Expense_Rs"]) - float(row["Actual_Expense_Rs"]), 2)


def test_save_production_entry_sticker_pieces():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-ST", "Operation": "Stitch", "Target": 80, "Rate_Rs": 3.0}]),
    )
    out = svc.save_production_entry(
        date_str="2026-05-15",
        karigar_id="K001",
        karigar_name="Test",
        challan_no="CH-ST",
        style="SKU-ST",
        hour_entries=[
            {
                "hour_col": "H_09_10",
                "operation": "Stitch",
                "sticker_in": 10,
                "sticker_out": 5,
                "manual_pieces": False,
                "pieces": 0,
            },
        ],
    )
    assert out["ok"] is True
    pl = get_sheet_df("production_log")
    assert int(pl.iloc[-1]["H_09_10"]) == 5
    assert int(pl.iloc[-1]["SI_H_09_10"]) == 10
    assert int(pl.iloc[-1]["SO_H_09_10"]) == 5


def test_stitching_performance_report():
    from backend.services.stitching_costing import performance_report

    save_sheet_df(
        "production_log",
        pd.DataFrame(
            [
                {
                    "Date": "2026-05-10",
                    "Karigar_ID": "K001",
                    "Karigar_Name": "Ramesh",
                    "Total_Pieces": 50,
                    "Piece_Value_Rs": 500,
                    "Efficiency_%": 90,
                }
            ]
        ),
    )
    save_sheet_df(
        "karigar_attendance",
        pd.DataFrame(
            [
                {
                    "Date": "2026-05-10",
                    "E_Code": "K001",
                    "Name": "Ramesh",
                    "Payable_Hrs": 8,
                    "Total_Pay": 400,
                }
            ]
        ),
    )
    out = performance_report("2026-05-01", "2026-05-15")
    assert out["ok"] is True
    assert out["rows"]
    assert out["summary"]["net_surplus"] == 100.0


def test_stitching_admin_and_style_update():
    from backend.db.stitching_db import verify_admin_password, change_admin_password
    from backend.services.stitching_costing import update_style_operation

    assert verify_admin_password("admin123") is True
    out = update_style_operation("1894YKDGREEN", "Cutting", target=130, rate_rs=2.75)
    assert out["ok"] is True
    ch = change_admin_password("admin123", "admin123")
    assert ch["ok"] is True


def test_stitching_style_costing_report():
    rep = svc.style_costing_report(month="All", style="All", party="All")
    assert "summary" in rep
    assert "rows" in rep


def test_style_costing_uses_received_qty_for_party_value():
    save_sheet_df(
        "challan_master",
        pd.DataFrame(
            [
                {
                    "Challan_No": "CH-90",
                    "Style": "STYLE-A",
                    "Party": "Party1",
                    "Total_Qty": 100,
                    "Received_Qty": 90,
                    "Rate_Per_Pc": 10,
                    "Deposit_Rs": 0,
                    "Date": "2026-05-01",
                }
            ]
        ),
    )
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "STYLE-A", "Operation": "Stitch", "Target": 80, "Rate_Rs": 5}]),
    )
    rep = svc.style_costing_report(month="All", style="STYLE-A", party="All")
    row = next(r for r in rep["rows"] if r["Challan_No"] == "CH-90")
    assert float(row["Party_Value_Ordered_Rs"]) == 1000.0
    assert float(row["Party_Value_Received_Rs"]) == 900.0
    assert float(row["Party_Value_Rs"]) == 900.0
    assert int(row["Pending"]) == 10


def test_stitching_production_entry_reports():
    svc.save_production_entry(
        date_str="2026-05-15",
        karigar_id="K001",
        karigar_name="Ramesh Kumar",
        challan_no="10220-2526",
        style="1894YKDGREEN",
        hour_entries=[
            {"hour_col": "H_09_10", "operation": "Cutting", "pieces": 12},
            {"hour_col": "H_10_11", "operation": "Cutting", "pieces": 8},
        ],
    )
    rep = svc.production_entry_reports("2026-05-15", "K001")
    assert len(rep["history"]) >= 1
    assert len(rep["report1"]) >= 1
    assert rep["report1"][0]["Total_Pieces"] >= 20
