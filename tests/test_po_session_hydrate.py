"""PO calculate must load uploaded inventory / existing PO from disk warm cache."""

import json

import pandas as pd


def test_hydrate_po_session_from_disk_parquets(tmp_path, monkeypatch):
    import backend.main as main_mod
    from backend.session import AppSession
    from backend.services.po_session_hydrate import hydrate_po_session_for_calculate

    main_mod.clear_warm_cache()
    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))

    inv = pd.DataFrame({"OMS_SKU": ["SKU-A", "SKU-B"], "Total_Inventory": [10, 20]})
    ep = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-B"],
            "PO_Qty_Ordered": [1, 2],
            "Pending_Cutting": [0, 0],
            "Balance_to_Dispatch": [0, 0],
            "PO_Pipeline_Total": [5, 6],
        }
    )
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A"],
            "TxnDate": [pd.Timestamp("2026-06-01")],
            "Transaction Type": ["Shipment"],
            "Quantity": [3],
            "Units_Effective": [3],
            "Source": ["Meesho"],
        }
    )

    inv.to_parquet(tmp_path / "inventory_df_variant.parquet", index=False)
    ep.to_parquet(tmp_path / "existing_po_df.parquet", index=False)
    sales.to_parquet(tmp_path / "sales_df.parquet", index=False)
    (tmp_path / "inventory_session_meta.json").write_text(
        json.dumps({"inventory_snapshot_uploaded_at": "2026-06-05T05:42:10Z"}),
        encoding="utf-8",
    )
    (tmp_path / "existing_po_meta.json").write_text(
        json.dumps(
            {
                "existing_po_generation": 3,
                "existing_po_rows": 2,
                "existing_po_filename": "test.xlsx",
            }
        ),
        encoding="utf-8",
    )

    sess = AppSession()
    stats = hydrate_po_session_for_calculate(sess)

    assert stats["inventory_rows"] == 2
    assert stats["existing_po_rows"] == 2
    assert stats["sales_rows"] == 1
    assert len(sess.inventory_df_variant) == 2
    assert len(sess.existing_po_df) == 2
    assert len(sess.sales_df) == 1
    assert int(getattr(sess, "existing_po_generation", 0) or 0) == 3


def test_po_session_hydrate_loads_platform_parquets_when_po_session_only(tmp_path, monkeypatch):
    import backend.main as main_mod
    from backend.session import AppSession
    from backend.services.po_session_hydrate import hydrate_po_session_for_calculate

    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("WARM_CACHE_PO_SESSION_ONLY", "1")
    main_mod.clear_warm_cache()

    mtr = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01"]),
            "SKU": ["A1"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [500],
        }
    )
    mtr.to_parquet(tmp_path / "mtr_df.parquet", index=False)
    sales = pd.DataFrame(
        {
            "Sku": ["A1"],
            "TxnDate": [pd.Timestamp("2026-06-01")],
            "Transaction Type": ["Shipment"],
            "Quantity": [1],
            "Units_Effective": [1],
            "Source": ["Amazon"],
        }
    )
    sales.to_parquet(tmp_path / "sales_df.parquet", index=False)

    sess = AppSession()
    main_mod._warm_cache = {"sales_df": sales}
    sess.sales_df = sales.copy()
    hydrate_po_session_for_calculate(sess)
    assert len(sess.mtr_df) == 1


def test_placeholder_sidecars_restored_from_backup(tmp_path, monkeypatch):
    import backend.main as main_mod
    from backend.session import AppSession
    from backend.services.po_engine import calculate_po_base
    from backend.services.po_session_hydrate import (
        effective_sku_status_df_for_engine,
        ensure_po_sidecars_hydrated,
    )

    warm = tmp_path / "warm"
    backup = tmp_path / "github_cache" / "2026-05-29"
    warm.mkdir(parents=True)
    backup.mkdir(parents=True)
    monkeypatch.setenv("WARM_CACHE_DIR", str(warm))
    monkeypatch.setenv("GITHUB_BLOB_CACHE_DIR", str(tmp_path / "github_cache"))

    placeholder = pd.DataFrame(
        {
            "OMS_SKU": ["Z-1"],
            "SKU_Sheet_Status": ["Open"],
            "SKU_Sheet_Closed": [False],
            "Lead_Time_From_Sheet": [10.0],
        }
    )
    real_status = pd.DataFrame(
        {
            "OMS_SKU": ["REAL-SKU-1", "REAL-SKU-2"],
            "SKU_Sheet_Status": ["Open", "Open"],
            "SKU_Sheet_Closed": [False, False],
            "Lead_Time_From_Sheet": [30.0, 45.0],
        }
    )
    placeholder.to_parquet(warm / "sku_status_lead_df.parquet", index=False)
    real_status.to_parquet(backup / "sku_status_lead_df.parquet", index=False)

    main_mod.clear_warm_cache()
    main_mod._warm_cache["sku_status_lead_df"] = placeholder.copy()
    sess = AppSession()
    sess.sku_status_lead_df = placeholder.copy()

    stats = ensure_po_sidecars_hydrated(sess)
    assert stats["sku_status_lead_df"] == 2
    assert len(sess.sku_status_lead_df) == 2
    assert sess.sku_status_lead_df.iloc[0]["OMS_SKU"] == "REAL-SKU-1"

    assert effective_sku_status_df_for_engine(sess) is not None

    days = pd.date_range("2025-11-01", periods=30, freq="D")
    sales = pd.DataFrame(
        {
            "Sku": ["REAL-SKU-1"] * 30,
            "TxnDate": days,
            "Transaction Type": ["Shipment"] * 30,
            "Quantity": [2] * 30,
            "Units_Effective": [2] * 30,
            "Source": ["Myntra"] * 30,
        }
    )
    inv = pd.DataFrame({"OMS_SKU": ["REAL-SKU-1"], "Total_Inventory": [0]})
    po_placeholder = calculate_po_base(
        sales,
        inv,
        30,
        7,
        60,
        safety_pct=0.0,
        sku_status_df=placeholder,
    )
    po_real = calculate_po_base(
        sales,
        inv,
        30,
        7,
        60,
        safety_pct=0.0,
        sku_status_df=real_status,
    )
    assert int(po_placeholder.iloc[0]["PO_Qty"]) == 0
    assert int(po_real.iloc[0]["PO_Qty"]) > 0


def test_effective_sku_status_ignores_placeholder(tmp_path, monkeypatch):
    from backend.session import AppSession
    from backend.services.po_session_hydrate import effective_sku_status_df_for_engine

    sess = AppSession()
    sess.sku_status_lead_df = pd.DataFrame(
        {
            "OMS_SKU": ["Z-1"],
            "SKU_Sheet_Status": ["Open"],
            "SKU_Sheet_Closed": [False],
            "Lead_Time_From_Sheet": [10.0],
        }
    )
    assert effective_sku_status_df_for_engine(sess) is None


def test_effective_sku_status_ignores_sparse_fragment_on_large_catalog():
    from backend.session import AppSession
    from backend.services.po_session_hydrate import effective_sku_status_df_for_engine

    sess = AppSession()
    sess.sku_status_lead_df = pd.DataFrame(
        {
            "OMS_SKU": ["FOO-BAR", "BAZ-QUX"],
            "SKU_Sheet_Status": ["Open", "Open"],
            "SKU_Sheet_Closed": [False, False],
            "Lead_Time_From_Sheet": [45.0, 45.0],
        }
    )
    sess.inventory_df_variant = pd.DataFrame(
        {"OMS_SKU": [f"SKU-{i}" for i in range(200)], "Total_Inventory": [1] * 200}
    )
    assert effective_sku_status_df_for_engine(sess) is None
