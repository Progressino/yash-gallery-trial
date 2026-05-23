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
    assert svc.resolve_hour_pieces({"sticker_in": 10, "sticker_out": 20, "manual_pieces": False}) == 10
    assert svc.resolve_hour_pieces({"sticker_in": 10, "sticker_out": 0, "manual_pieces": False}) == 10
    assert svc.resolve_hour_pieces({"sticker_in": 0, "sticker_out": 25, "manual_pieces": False}) == 25
    assert svc.resolve_hour_pieces({"sticker_in": 0, "sticker_out": 0, "pieces": 12, "manual_pieces": True}) == 12


def test_resolve_session_hour_pieces_cumulative_out():
    entries = [
        {"hour_col": "H_09_10", "operation": "Astin Attach", "sticker_out": 30, "manual_pieces": False},
        {"hour_col": "H_10_11", "operation": "Astin Attach", "sticker_out": 55, "manual_pieces": False},
        {"hour_col": "H_11_12", "operation": "Astin Attach", "sticker_out": 95, "manual_pieces": False},
    ]
    pcs = svc.resolve_session_hour_pieces(entries)
    assert pcs["H_09_10"] == 30
    assert pcs["H_10_11"] == 25
    assert pcs["H_11_12"] == 40
    assert sum(pcs.values()) == 95


def test_financial_audit_caps_day_budget():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-CAP", "Operation": "Op", "Target": 30, "Rate_Rs": 2.0}]),
    )
    out = svc.save_production_entry(
        date_str="2026-05-21",
        karigar_id="K200",
        karigar_name="Cap Test",
        challan_no="CH-CAP",
        style="SKU-CAP",
        hour_entries=[
            {"hour_col": "H_09_10", "operation": "Op", "pieces": 200, "manual_pieces": True},
            {"hour_col": "H_10_11", "operation": "Op", "pieces": 125, "manual_pieces": True},
        ],
    )
    assert out["ok"] is True
    pl = get_sheet_df("production_log")
    row = pl[pl["Karigar_ID"].astype(str) == "K200"].iloc[-1]
    assert float(row["Budgeted_Expense_Rs"]) <= 480.01
    assert float(row["Budgeted_Expense_Rs"]) > 400
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


def test_style_master_report_totals_and_views():
    save_sheet_df(
        "style_master",
        pd.DataFrame(
            [
                {"Style": "STYLE-R", "Operation": "Stitch", "Target": 80, "Rate_Rs": 5},
                {"Style": "STYLE-R", "Operation": "Finish", "Target": 60, "Rate_Rs": 3},
            ]
        ),
    )
    save_sheet_df(
        "production_log",
        pd.DataFrame(
            [
                {
                    "Date": "2026-05-10",
                    "Style": "STYLE-R",
                    "Operation": "Stitch",
                    "Karigar_ID": "K1",
                    "Karigar_Name": "Ali",
                    "Challan_No": "CH-1",
                    "Total_Pieces": 50,
                    "Efficiency_%": 90,
                    "Piece_Value_Rs": 250,
                    "Budgeted_Expense_Rs": 240,
                    "Actual_Expense_Rs": 250,
                    "PL_Rs": -10,
                }
            ]
        ),
    )
    save_sheet_df(
        "challan_master",
        pd.DataFrame(
            [
                {
                    "Challan_No": "CH-1",
                    "Style": "STYLE-R",
                    "Party": "P1",
                    "Total_Qty": 100,
                    "Received_Qty": 80,
                    "Rate_Per_Pc": 20,
                    "Deposit_Rs": 0,
                    "Date": "2026-05-01",
                }
            ]
        ),
    )
    full = svc.style_master_report("STYLE-R", date_from="2026-05-01", date_to="2026-05-31", view="full")
    assert full["ok"] is True
    assert full["totals"]["master_operations"] == 2
    assert full["totals"]["labour_rate_per_piece_rs"] == 8.0
    assert full["totals"]["production_pieces"] == 50
    assert full["master"]["totals"]["sum_rate_rs"] == 8.0
    assert len(full["production"]["by_operation"]) >= 1
    assert len(full["costing"]["challans"]) >= 1

    master_only = svc.style_master_report("STYLE-R", view="master")
    assert "master" in master_only
    assert "production" not in master_only


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


def test_compute_financial_audit_sop_examples():
    fin = svc.compute_financial_audit(100, 110, 520)
    assert fin["budget_rate_per_piece"] == 4.8
    assert fin["budgeted_amount"] == 528.0
    assert fin["actual_amount"] == 520.0
    assert fin["pl_rs"] == 8.0

    fin_low = svc.compute_financial_audit(100, 70, 520)
    assert fin_low["budgeted_amount"] == 336.0
    assert fin_low["pl_rs"] == -184.0

    fin_side = svc.compute_financial_audit(20, 20, 350)
    assert fin_side["budget_rate_per_piece"] == 24.0
    assert fin_side["budgeted_amount"] == 480.0
    assert fin_side["actual_amount"] == 350.0
    assert fin_side["pl_rs"] == 130.0


def test_compute_formula_ltl_sop_examples():
    assert svc.ltl_tolerance_pct_for_rate(250) == 35.0
    assert svc.ltl_tolerance_pct_for_rate(350) == 12.0
    assert svc.compute_formula_ltl(100, 520) == 95
    assert svc.compute_formula_ltl(100, 200) == 27
    assert svc.compute_formula_ltl(20, 350) == 13
    assert svc.compute_formula_ltl(20, 500) == 18


def test_delete_challan():
    save_sheet_df(
        "challan_master",
        pd.DataFrame([{"Challan_No": "DEL-1", "Style": "S1", "Party": "P", "Total_Qty": 10}]),
    )
    out = svc.delete_challan("DEL-1")
    assert out["ok"] is True
    assert get_sheet_df("challan_master").empty


def test_resolve_applied_ltl_manual_override_precedence():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "1065YKBLL", "Operation": "Gala Piping", "Target": 100, "Rate_Rs": 2.0}]),
    )
    save_sheet_df(
        "karigar_master",
        pd.DataFrame([{"Karigar_ID": "944", "Name": "Soniya", "Skill": "Stitching", "Daily_Rate_Rs": 200}]),
    )
    svc.upsert_ltl_override("1065YKBLL", "Gala Piping", "944", 40)
    out = svc.resolve_applied_ltl("1065YKBLL", "Gala Piping", "944", as_of_date="2026-05-21", base_target=100)
    assert out["formula_ltl"] == 27
    assert out["applied_ltl"] == 40
    assert out["target_type"] == "Manual Override"
    svc.upsert_ltl_override("1065YKBLL", "Gala Piping", "944", None)


def test_save_production_entry_persists_applied_ltl():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-LTL", "Operation": "Side", "Target": 20, "Rate_Rs": 3.0}]),
    )
    save_sheet_df(
        "karigar_master",
        pd.DataFrame([{"Karigar_ID": "823", "Name": "Suman", "Skill": "Stitching", "Daily_Rate_Rs": 350}]),
    )
    svc.save_production_entry(
        date_str="2026-05-21",
        karigar_id="823",
        karigar_name="Suman",
        challan_no="CH-LTL",
        style="SKU-LTL",
        hour_entries=[{"hour_col": "H_09_10", "operation": "Side", "pieces": 15}],
    )
    pl = get_sheet_df("production_log")
    row = pl.iloc[-1]
    assert int(row["Applied_LTL"]) == 13
    assert int(row["Base_Target"]) == 20
    assert int(row["Formula_LTL"]) == 13


def test_save_production_entry_replaces_same_session_on_resave():
    """Correcting and re-saving must update rows, not append duplicates."""
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-UP", "Operation": "Astin Attach", "Target": 30, "Rate_Rs": 2.0}]),
    )
    base = dict(
        date_str="2026-05-21",
        karigar_id="K100",
        karigar_name="Test Worker",
        challan_no="1107-2627",
        style="SKU-UP",
    )
    svc.save_production_entry(
        **base,
        hour_entries=[{"hour_col": "H_09_10", "operation": "Astin Attach", "pieces": 20}],
    )
    svc.save_production_entry(
        **base,
        hour_entries=[{"hour_col": "H_09_10", "operation": "Astin Attach", "pieces": 30}],
    )
    pl = get_sheet_df("production_log")
    mask = (
        (pl["Date"].astype(str) == "2026-05-21")
        & (pl["Karigar_ID"].astype(str) == "K100")
        & (pl["Challan_No"].astype(str) == "1107-2627")
        & (pl["Style"].astype(str) == "SKU-UP")
    )
    subset = pl[mask]
    assert len(subset) == 1, f"expected 1 row after resave, got {len(subset)}"
    assert int(subset.iloc[0]["Total_Pieces"]) == 30
    assert int(subset.iloc[0]["H_09_10"]) == 30


def test_save_production_entry_normalizes_operation_whitespace():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-WS", "Operation": "Cutting", "Target": 10, "Rate_Rs": 1.0}]),
    )
    svc.save_production_entry(
        date_str="2026-05-21",
        karigar_id="K101",
        karigar_name="T",
        challan_no="CH-1",
        style="SKU-WS",
        hour_entries=[{"hour_col": "H_09_10", "operation": " Cutting ", "pieces": 5}],
    )
    svc.save_production_entry(
        date_str="2026-05-21",
        karigar_id="K101",
        karigar_name="T",
        challan_no="CH-1",
        style="SKU-WS",
        hour_entries=[
            {"hour_col": "H_09_10", "operation": "cutting", "pieces": 5},
            {"hour_col": "H_10_11", "operation": "Cutting", "pieces": 7},
        ],
    )
    pl = get_sheet_df("production_log")
    mask = (pl["Karigar_ID"].astype(str) == "K101") & (pl["Style"].astype(str) == "SKU-WS")
    assert len(pl[mask]) == 1
    row = pl[mask].iloc[0]
    assert int(row["H_09_10"]) == 5
    assert int(row["H_10_11"]) == 7


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
    r1 = rep["report1"][0]
    assert r1["Date"] == "2026-05-15"
    assert r1.get("Save_Time")
    assert r1["Total_Pieces"] >= 20
    assert r1["Style"] == "1894YKDGREEN"
    assert float(r1["Daily_Salary_Rs"]) > 0
    assert float(r1["Hourly_Salary_Rs"]) > 0
    assert float(r1["Budget_Rate_Per_Piece"]) > 0
    assert float(r1["Budgeted_Expense_Rs"]) > 0
    assert float(r1["Actual_Expense_Rs"]) > 0
    assert "Profit_Loss" in r1
    if rep["report2_summary"]:
        assert rep["report2_summary"][0]["Style"] == "1894YKDGREEN"
        assert float(rep["report2_summary"][0]["Daily_Salary_Rs"]) > 0
    if rep["report2_hourly"]:
        assert rep["report2_hourly"][0]["Date"] == "2026-05-15"
        assert rep["report2_hourly"][0].get("Save_Time")
        assert rep["report2_hourly"][0]["Style"] == "1894YKDGREEN"
        assert float(rep["report2_hourly"][0]["Hourly_Salary_Rs"]) > 0


def test_save_production_entry_replaces_when_style_case_differs():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "1894YKDGREEN", "Operation": "Cutting", "Target": 30, "Rate_Rs": 2.0}]),
    )
    base = dict(
        date_str="2026-05-21",
        karigar_id="K200",
        karigar_name="Case Test",
        challan_no="CH-200",
    )
    svc.save_production_entry(
        **base,
        style="1894ykdgreen",
        hour_entries=[{"hour_col": "H_09_10", "operation": "Cutting", "pieces": 10}],
    )
    svc.save_production_entry(
        **base,
        style="1894YKDGREEN",
        hour_entries=[{"hour_col": "H_09_10", "operation": "Cutting", "pieces": 18}],
    )
    pl = get_sheet_df("production_log")
    mask = (pl["Karigar_ID"].astype(str) == "K200") & (
        pl["Style"].astype(str).str.lower() == "1894ykdgreen"
    )
    assert len(pl[mask]) == 1
    assert int(pl[mask].iloc[0]["Total_Pieces"]) == 18


def test_production_entry_reports_dedupes_legacy_duplicate_rows():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-DUP", "Operation": "Op1", "Target": 10, "Rate_Rs": 1.0}]),
    )
    save_sheet_df(
        "production_log",
        pd.DataFrame(
            [
                {
                    "Date": "2026-05-22",
                    "Karigar_ID": "K9",
                    "Karigar_Name": "Dup",
                    "Challan_No": "C1",
                    "Style": "SKU-DUP",
                    "Operation": "Op1",
                    "H_09_10": 5,
                    "Total_Pieces": 5,
                    "Rate_Rs": 1.0,
                    "Daily_Rate_Rs": 480.0,
                    "Save_Time": "2026-05-22 09:00:00",
                },
                {
                    "Date": "2026-05-22",
                    "Karigar_ID": "K9",
                    "Karigar_Name": "Dup",
                    "Challan_No": "C1",
                    "Style": "sku-dup",
                    "Operation": "Op1",
                    "H_09_10": 12,
                    "Total_Pieces": 12,
                    "Rate_Rs": 1.0,
                    "Daily_Rate_Rs": 480.0,
                    "Save_Time": "2026-05-22 11:00:00",
                },
            ]
        ),
    )
    rep = svc.production_entry_reports("2026-05-22", "K9")
    assert len(rep["report1"]) == 1
    assert int(rep["report1"][0]["Total_Pieces"]) == 12
    assert float(rep["report2_hourly"][0]["Hourly_Salary_Rs"]) == 60.0
