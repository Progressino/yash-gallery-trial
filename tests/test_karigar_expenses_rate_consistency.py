import pandas as pd


def test_list_karigar_expenses_recomputes_auto_amount_and_rates(tmp_path, monkeypatch):
    from backend.db import stitching_db
    from backend.db.stitching_db import init_db, save_sheet_df
    from backend.services import stitching_costing as svc

    path = tmp_path / "exp_test.db"
    monkeypatch.setenv("STITCHING_DB_PATH", str(path))
    monkeypatch.setattr(stitching_db, "_DB", str(path))
    init_db()

    # Daily rate history: date-specific should be used when listing.
    save_sheet_df(
        "karigar_rate_history",
        pd.DataFrame(
            [
                {"Karigar_ID": "900", "Effective_From": "2026-06-01", "Daily_Rate_Rs": 418},
                {"Karigar_ID": "900", "Effective_From": "2026-06-02", "Daily_Rate_Rs": 822},
            ]
        ),
    )

    # Two expenses same day, both 4 hours, but stored with wrong historical hourly/amount.
    save_sheet_df(
        "karigar_expenses",
        pd.DataFrame(
            [
                {
                    "Date": "2026-06-02",
                    "Karigar_ID": "900",
                    "Karigar_Name": "Bihari Lal",
                    "Work_Type": "Other Work",
                    "Challan_No": "CH-1",
                    "Style": "SKU-1",
                    "Hours": 4,
                    "Amount_Rs": 209,  # wrong
                    "Daily_Rate_Rs": 418,
                    "Hourly_Rate_Rs": 52.25,
                    "Auto_Amount": True,
                },
                {
                    "Date": "2026-06-02",
                    "Karigar_ID": "900",
                    "Karigar_Name": "Bihari Lal",
                    "Work_Type": "Other Work",
                    "Challan_No": "CH-2",
                    "Style": "SKU-2",
                    "Hours": 4,
                    "Amount_Rs": 411,  # wrong
                    "Daily_Rate_Rs": 822,
                    "Hourly_Rate_Rs": 102.75,
                    "Auto_Amount": True,
                },
            ]
        ),
    )

    out = svc.list_karigar_expenses("2026-06-02", "2026-06-02")
    assert len(out) == 2
    # Effective daily on 2026-06-02 is 822 => hourly 102.75, 4h => 411.0
    assert float(out[0]["Hourly_Rate_Rs"]) == 102.75
    assert float(out[1]["Hourly_Rate_Rs"]) == 102.75
    assert float(out[0]["Amount_Rs"]) == 411.0
    assert float(out[1]["Amount_Rs"]) == 411.0

