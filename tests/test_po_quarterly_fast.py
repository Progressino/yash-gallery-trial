"""Streaming quarterly aggregator."""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import pandas as pd

from backend.services.po_quarterly_fast import _accumulate_shipment_frame


def test_accumulate_frame_counts_quarters():
    start_ts = pd.Timestamp("2024-06-01")
    end_ts = pd.Timestamp("2026-06-04")
    today = pd.Timestamp.today()
    q_label_map = {(2025, 4): "Jan-Mar 2025", (2025, 3): "Oct-Dec 2024"}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-11-15"]),
            "SKU": ["1001YKBEIGE-M"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [40],
        }
    )
    n = _accumulate_shipment_frame(
        df,
        "amazon",
        None,
        strip_pl=True,
        canonical_oms=False,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
    )
    assert n == 1
    assert quarter_sums[("1001YKBEIGE-M", "Oct-Dec 2024")] == 40


def test_accumulate_frame_skips_quarters_outside_label_map():
    """Tier-3 history can include FY quarters older than the requested n_quarters window."""
    start_ts = pd.Timestamp("2024-06-01")
    end_ts = pd.Timestamp("2026-06-04")
    today = pd.Timestamp.today()
    q_label_map = {(2025, 4): "Jan-Mar 2025", (2025, 3): "Oct-Dec 2024"}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-05-15", "2024-11-15"]),
            "SKU": ["OLD-SKU-M", "1001YKBEIGE-M"],
            "Transaction_Type": ["Shipment", "Shipment"],
            "Quantity": [99, 40],
        }
    )
    n = _accumulate_shipment_frame(
        df,
        "amazon",
        None,
        strip_pl=True,
        canonical_oms=False,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
    )
    assert n == 1
    assert ("OLD-SKU-M", "Oct-Dec 2024") not in quarter_sums
    assert quarter_sums[("1001YKBEIGE-M", "Oct-Dec 2024")] == 40


def test_accumulate_sales_skips_platform_days():
    """Unified sales must not double-count SKU-days already on platform side."""
    from backend.services.po_quarterly_fast import _accumulate_sales_df_shipments

    start_ts = pd.Timestamp("2024-06-01")
    end_ts = pd.Timestamp("2026-06-04")
    today = pd.Timestamp.today()
    from backend.services.po_engine import get_indian_fy_quarter, quarter_col_name

    fy, qn = get_indian_fy_quarter(pd.Timestamp("2026-06-02"))
    q_label_map = {(fy, qn): quarter_col_name(fy, qn)}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)
    platform_day_keys = {("amazon", "SKU-A", pd.Timestamp("2026-06-01").normalize())}

    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A", "SKU-A"],
            "TxnDate": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "Transaction Type": ["Shipment", "Shipment"],
            "Quantity": [50, 7],
        }
    )
    n = _accumulate_sales_df_shipments(
        sales,
        None,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
        platform_day_keys=platform_day_keys,
    )
    assert n == 1
    assert quarter_sums[("SKU-A", quarter_col_name(fy, qn))] == 7


def test_tier3_fills_same_day_different_platform():
    """Tier-3 must not be blocked on a day when only another platform is in Tier-1."""
    from backend.services.po_quarterly_fast import _accumulate_shipment_frame

    start_ts = pd.Timestamp("2025-07-01")
    end_ts = pd.Timestamp("2025-09-30")
    today = pd.Timestamp.today()
    q_label_map = {(2026, 2): "Jul-Sep 2025"}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)
    platform_day_keys: set = set()

    myntra = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-08-01"]),
            "OMS_SKU": ["SKU-A"],
            "TxnType": ["Shipment"],
            "Quantity": [5],
        }
    )
    amazon = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-08-01"]),
            "SKU": ["SKU-A"],
            "Transaction_Type": ["Shipment"],
            "Quantity": [7],
        }
    )
    _accumulate_shipment_frame(
        myntra, "myntra", None, strip_pl=False, canonical_oms=True, group_by_parent=False,
        start_ts=start_ts, end_ts=end_ts, cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30), q_label_map=q_label_map,
        quarter_sums=quarter_sums, units_90=units_90, units_30=units_30, days_30=days_30,
        platform_day_keys=platform_day_keys,
    )
    _accumulate_shipment_frame(
        amazon, "amazon", None, strip_pl=True, canonical_oms=False, group_by_parent=False,
        start_ts=start_ts, end_ts=end_ts, cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30), q_label_map=q_label_map,
        quarter_sums=quarter_sums, units_90=units_90, units_30=units_30, days_30=days_30,
        platform_day_keys=platform_day_keys, skip_days=platform_day_keys,
    )
    assert quarter_sums[("SKU-A", "Jul-Sep 2025")] == 12


def test_tier1_records_days_into_empty_keyset_so_tier3_dedups():
    """Regression: an initially-EMPTY platform_day_keys must still receive Tier-1
    (platform, SKU, day) keys so the Tier-3 gap-fill / sales_df supplement do not
    re-count the same shipments. The old code passed ``platform_day_keys or set()``
    to the recorder; because an empty set is falsy it handed over a THROWAWAY set,
    so Tier-1 days were never recorded and every SKU's quarterly units were inflated
    by the overlapping Tier-3 + sales_df rows."""
    from backend.services.po_quarterly_fast import _accumulate_shipment_frame

    start_ts = pd.Timestamp("2025-07-01")
    end_ts = pd.Timestamp("2025-09-30")
    today = pd.Timestamp.today()
    q_label_map = {(2026, 2): "Jul-Sep 2025"}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)
    platform_day_keys: set = set()  # starts EMPTY, exactly like the streaming first pass

    common = dict(
        strip_pl=True, canonical_oms=False, group_by_parent=False,
        start_ts=start_ts, end_ts=end_ts, cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30), q_label_map=q_label_map,
        quarter_sums=quarter_sums, units_90=units_90, units_30=units_30, days_30=days_30,
    )
    tier1 = pd.DataFrame({
        "Date": pd.to_datetime(["2025-08-01"]),
        "SKU": ["SKU-A"],
        "Transaction_Type": ["Shipment"],
        "Quantity": [5],
    })
    _accumulate_shipment_frame(tier1, "amazon", None, platform_day_keys=platform_day_keys, **common)
    # The Tier-1 day MUST have been recorded into the caller's set (the bug left it empty).
    assert ("amazon", "SKU-A", pd.Timestamp("2025-08-01")) in platform_day_keys

    # A Tier-3 re-upload of the same (platform, SKU, day) must be skipped, not doubled.
    tier3 = tier1.copy()
    _accumulate_shipment_frame(
        tier3, "amazon", None, platform_day_keys=platform_day_keys,
        skip_days=platform_day_keys, **common,
    )
    assert quarter_sums[("SKU-A", "Jul-Sep 2025")] == 5


# ── Flipkart Event Sub Type sign logic ───────────────────────────────────────

def _make_quarterly_kwargs(q_label_map: dict) -> dict:
    """Shared kwargs for accumulate tests (Net signing — matches historical assertions)."""
    today = pd.Timestamp.today()
    start_ts = pd.Timestamp("2024-04-01")
    end_ts = pd.Timestamp("2026-06-30")
    return dict(
        strip_pl=False,
        canonical_oms=True,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=defaultdict(int),
        units_90=defaultdict(int),
        units_30=defaultdict(int),
        days_30=defaultdict(set),
        demand_basis="Net",
    )


def test_flipkart_return_cancel_adds_to_quarterly():
    """Return Cancellation (TxnType='ReturnCancel') must add to net quarterly units."""
    q_label_map = {(2025, 4): "Jan-Mar 2025"}
    kwargs = _make_quarterly_kwargs(q_label_map)
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-15", "2025-01-16"]),
        "OMS_SKU": ["SKU-X", "SKU-X"],
        "TxnType": ["Shipment", "ReturnCancel"],
        "Quantity": [10, 2],
    })
    _accumulate_shipment_frame(df, "flipkart", None, **kwargs)
    assert kwargs["quarter_sums"][("SKU-X", "Jan-Mar 2025")] == 12


def test_flipkart_refund_subtracts_from_quarterly():
    """Return (TxnType='Refund') must subtract from net quarterly units."""
    q_label_map = {(2025, 4): "Jan-Mar 2025"}
    kwargs = _make_quarterly_kwargs(q_label_map)
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-15", "2025-01-15"]),
        "OMS_SKU": ["SKU-X", "SKU-X"],
        "TxnType": ["Shipment", "Refund"],
        "Quantity": [10, 3],
    })
    _accumulate_shipment_frame(df, "flipkart", None, **kwargs)
    assert kwargs["quarter_sums"][("SKU-X", "Jan-Mar 2025")] == 7


def test_amazon_mtr_net_excludes_cancel_from_quarterly():
    """Amazon PO quarterly: net = Shipment − Refund (Cancel does not add to demand)."""
    q_label_map = {(2025, 4): "Jan-Mar 2025"}
    kwargs = _make_quarterly_kwargs(q_label_map)
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-15"] * 3),
        "SKU": ["1197YKGREEN-M"] * 3,
        "Transaction_Type": ["Shipment", "Refund", "Cancel"],
        "Quantity": [16, 6, 3],
    })
    _accumulate_shipment_frame(df, "amazon", None, **kwargs)
    assert kwargs["quarter_sums"][("1197YKGREEN-M", "Jan-Mar 2025")] == 10


def test_amazon_sales_df_net_excludes_cancel_from_quarterly():
    """Amazon rows from unified sales_df must keep Cancel neutral (not negative)."""
    from backend.services.po_quarterly_fast import _accumulate_sales_df_shipments
    from backend.services.po_engine import get_indian_fy_quarter, quarter_col_name

    fy, qn = get_indian_fy_quarter(pd.Timestamp("2025-01-15"))
    q_label_map = {(fy, qn): quarter_col_name(fy, qn)}
    today = pd.Timestamp.today()
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)
    sales = pd.DataFrame({
        "Sku": ["AMZ-SKU-FAST"] * 3,
        "TxnDate": pd.to_datetime(["2025-01-10", "2025-01-11", "2025-01-12"]),
        "Transaction Type": ["Shipment", "Cancel", "Refund"],
        "Quantity": [16, 3, 6],
        "Source": ["Amazon", "Amazon", "Amazon"],
    })
    n = _accumulate_sales_df_shipments(
        sales,
        None,
        group_by_parent=False,
        start_ts=pd.Timestamp("2024-04-01"),
        end_ts=pd.Timestamp("2026-04-01"),
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
        platform_day_keys=set(),
        demand_basis="Net",
    )
    assert n > 0
    assert quarter_sums[("AMZ-SKU-FAST", quarter_col_name(fy, qn))] == 10


def test_flipkart_cancel_subtracts_from_quarterly():
    """Cancellation (TxnType='Cancel') must subtract from net quarterly units."""
    q_label_map = {(2025, 4): "Jan-Mar 2025"}
    kwargs = _make_quarterly_kwargs(q_label_map)
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2025-02-10", "2025-02-10"]),
        "OMS_SKU": ["SKU-Y", "SKU-Y"],
        "TxnType": ["Shipment", "Cancel"],
        "Quantity": [8, 2],
    })
    _accumulate_shipment_frame(df, "flipkart", None, **kwargs)
    assert kwargs["quarter_sums"][("SKU-Y", "Jan-Mar 2025")] == 6


def test_flipkart_net_all_four_event_types():
    """Net = Sale + ReturnCancel - Return - Cancel across all four Flipkart event types."""
    q_label_map = {(2025, 4): "Jan-Mar 2025"}
    kwargs = _make_quarterly_kwargs(q_label_map)
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-20"] * 4),
        "OMS_SKU": ["SKU-Z"] * 4,
        "TxnType": ["Shipment", "ReturnCancel", "Refund", "Cancel"],
        "Quantity": [20, 3, 4, 1],
    })
    _accumulate_shipment_frame(df, "flipkart", None, **kwargs)
    # Net = 20 + 3 - 4 - 1 = 18
    assert kwargs["quarter_sums"][("SKU-Z", "Jan-Mar 2025")] == 18


def test_flipkart_return_only_day_not_counted_as_selling_day():
    """A day with only returns (negative net) must NOT inflate Freq_30d."""
    from backend.services.po_quarterly_fast import _accumulate_shipment_frame
    today = pd.Timestamp.today()
    recent_day = today - timedelta(days=5)
    q_label_map = {}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)
    df = pd.DataFrame({
        "Date": [recent_day],
        "OMS_SKU": ["SKU-R"],
        "TxnType": ["Refund"],
        "Quantity": [5],
    })
    _accumulate_shipment_frame(
        df, "flipkart", None,
        strip_pl=False, canonical_oms=True, group_by_parent=False,
        start_ts=today - timedelta(days=180), end_ts=today,
        cutoff_90=today - timedelta(days=90), cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums, units_90=units_90, units_30=units_30, days_30=days_30,
    )
    assert len(days_30["SKU-R"]) == 0, "Return-only day should not count as selling day"


def test_flipkart_sales_df_returncancel_adds():
    """ReturnCancel in sales_df supplement must also add to quarterly via signed logic."""
    from backend.services.po_quarterly_fast import _accumulate_sales_df_shipments
    from backend.services.po_engine import get_indian_fy_quarter, quarter_col_name

    fy, qn = get_indian_fy_quarter(pd.Timestamp("2025-02-01"))
    q_label_map = {(fy, qn): quarter_col_name(fy, qn)}
    today = pd.Timestamp.today()
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)
    sales = pd.DataFrame({
        "Sku": ["FK-SKU", "FK-SKU", "FK-SKU", "FK-SKU"],
        "TxnDate": pd.to_datetime(["2025-02-01"] * 4),
        "Transaction Type": ["Shipment", "ReturnCancel", "Refund", "Cancel"],
        "Quantity": [15, 2, 3, 1],
    })
    n = _accumulate_sales_df_shipments(
        sales,
        None,
        group_by_parent=False,
        start_ts=pd.Timestamp("2024-04-01"),
        end_ts=pd.Timestamp("2026-04-01"),
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
        platform_day_keys=set(),
        demand_basis="Net",
    )
    assert n > 0
    # Net = 15 + 2 - 3 - 1 = 13
    assert quarter_sums[("FK-SKU", quarter_col_name(fy, qn))] == 13


def test_sold_gross_ignores_refund_and_cancel():
    """Sold (Gross) counts Shipment only — Refund/Cancel/ReturnCancel do not change qty."""
    q_label_map = {(2025, 4): "Jan-Mar 2025"}
    kwargs = _make_quarterly_kwargs(q_label_map)
    kwargs["demand_basis"] = "Sold"
    kwargs["quarter_sums"] = defaultdict(int)
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-20"] * 4),
        "OMS_SKU": ["SKU-GROSS"] * 4,
        "TxnType": ["Shipment", "ReturnCancel", "Refund", "Cancel"],
        "Quantity": [20, 3, 4, 1],
    })
    _accumulate_shipment_frame(df, "flipkart", None, **kwargs)
    assert kwargs["quarter_sums"][("SKU-GROSS", "Jan-Mar 2025")] == 20


def test_combo_listing_retained_in_quarterly_sold():
    """Combo explode keeps listing SKU row (qty=1×) plus component demand."""
    from backend.services.combo_sku_map import explode_sku_qty_dataframe

    df = pd.DataFrame({"SKU": ["COMBO-LISTING-M"], "Qty": [10.0]})
    combo = {"COMBO-LISTING-M": [("COMP-A-M", 1.0), ("COMP-B-M", 1.0)]}
    out = explode_sku_qty_dataframe(
        df,
        sku_col="SKU",
        qty_col="Qty",
        sku_mapping={},
        combo_map=combo,
        strip_pl=False,
        retain_combo_listings=True,
    )
    by_sku = dict(zip(out["SKU"], out["Qty"]))
    assert by_sku["COMBO-LISTING-M"] == 10.0
    assert by_sku["COMP-A-M"] == 10.0
    assert by_sku["COMP-B-M"] == 10.0
    assert "_Combo_Fan" in out.columns
    fan_by = dict(zip(out["SKU"], out["_Combo_Fan"]))
    assert fan_by["COMBO-LISTING-M"] is False or fan_by["COMBO-LISTING-M"] == False
    assert fan_by["COMP-A-M"] is True or fan_by["COMP-A-M"] == True
    assert fan_by["COMP-B-M"] is True or fan_by["COMP-B-M"] == True


def test_quarterly_sold_does_not_fan_combo_onto_components():
    """Quarterly Sold(Gross) attributes combo listing sales to the listing only."""
    q_label_map = {(2025, 4): "Jan-Mar 2025"}
    kwargs = _make_quarterly_kwargs(q_label_map)
    kwargs["demand_basis"] = "Sold"
    kwargs["quarter_sums"] = defaultdict(int)
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-15"]),
        "OMS_SKU": ["DPT34BLUE"],
        "TxnType": ["Shipment"],
        "Quantity": [5],
    })
    _accumulate_shipment_frame(df, "flipkart", None, **kwargs)
    assert kwargs["quarter_sums"][("DPT34BLUE", "Jan-Mar 2025")] == 5
    # Must NOT create phantom component keys from explode
    assert all(not str(k[0]).startswith("1592YKBLUE") for k in kwargs["quarter_sums"])


def test_meesho_quarterly_backfills_sku_from_suborder_sibling():
    """Refund rows without listing SKU must inherit OMS from shipment twin."""
    from collections import defaultdict

    start_ts = pd.Timestamp("2024-06-01")
    end_ts = pd.Timestamp("2026-06-04")
    today = pd.Timestamp.today()
    q_label_map = {(2026, 3): "Oct-Dec 2025"}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-11-15", "2025-11-20"]),
            "OMS_SKU": ["1055YKGREEN-XS", ""],
            "SKU": ["1055YKCGREEN-XS", ""],
            "TxnType": ["Shipment", "Refund"],
            "Quantity": [2, 1],
            "OrderId": ["258980610239862016_1", "258980610239862016_1"],
            "LineKey": ["258980610239862016_1", "258980610239862016_1"],
            "MeeshoSubOrder": ["258980610239862016_1", "258980610239862016_1"],
        }
    )
    n = _accumulate_shipment_frame(
        df,
        "meesho",
        None,
        strip_pl=False,
        canonical_oms=True,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
        demand_basis="Net",
    )
    assert n == 2
    assert quarter_sums[("1055YKGREEN-XS", "Oct-Dec 2025")] == 1
    assert ("NAN", "Oct-Dec 2025") not in quarter_sums


def test_meesho_quarterly_drops_nan_oms_tokens():
    from collections import defaultdict

    start_ts = pd.Timestamp("2024-06-01")
    end_ts = pd.Timestamp("2026-06-04")
    today = pd.Timestamp.today()
    q_label_map = {(2026, 3): "Oct-Dec 2025"}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-11-15"]),
            "OMS_SKU": [float("nan")],
            "SKU": [""],
            "TxnType": ["Shipment"],
            "Quantity": [5],
            "OrderId": ["lonely-order"],
            "LineKey": ["lonely-order"],
            "MeeshoSubOrder": ["lonely-order"],
        }
    )
    n = _accumulate_shipment_frame(
        df,
        "meesho",
        None,
        strip_pl=False,
        canonical_oms=True,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
    )
    assert n == 0
    assert not quarter_sums


def test_amazon_quarterly_excludes_free_replacement_zero_invoice():
    from collections import defaultdict

    start_ts = pd.Timestamp("2024-06-01")
    end_ts = pd.Timestamp("2026-06-04")
    today = pd.Timestamp.today()
    q_label_map = {(2026, 1): "Apr-Jun 2025"}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-05-15", "2025-05-16"]),
            "Reporting_Date": pd.to_datetime(["2025-05-15", "2025-05-16"]),
            "SKU": ["1379YKGREEN-XXL", "1379YKGREEN-XXL"],
            "Transaction_Type": ["Shipment", "Shipment"],
            "Quantity": [20, 3],
            "Invoice_Amount": [800.0, 0.0],
            "Order_Id": ["A1", "A2"],
            "Invoice_Number": ["I1", "I2"],
            "ASIN": ["B09XX", "B09XX"],
        }
    )
    n = _accumulate_shipment_frame(
        df,
        "amazon",
        None,
        strip_pl=True,
        canonical_oms=False,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
    )
    assert n == 1
    assert quarter_sums[("1379YKGREEN-XXL", "Apr-Jun 2025")] == 20


def test_amazon_quarterly_offset_pair_nets_zero_not_inflated():
    """Paired ship+refund must net to 0 in PO — not double-counted as gross demand."""
    kwargs = _make_quarterly_kwargs({(2025, 4): "Jan-Mar 2025"})
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-03-11", "2025-03-11"]),
            "Reporting_Date": pd.to_datetime(["2025-03-11", "2025-03-11"]),
            "SKU": ["1379YKGREEN-XXL", "1379YKGREEN-XXL"],
            "Transaction_Type": ["Shipment", "Refund"],
            "Quantity": [1.0, 1.0],
            "Invoice_Amount": [769.0, -769.0],
            "Order_Id": ["405-6046243-9940337", "405-6046243-9940337"],
            "Invoice_Number": ["CJB1-14914", "CJB1-14914"],
        }
    )
    _accumulate_shipment_frame(df, "amazon", None, **kwargs)
    assert kwargs["quarter_sums"][("1379YKGREEN-XXL", "Jan-Mar 2025")] == 0


def test_amazon_quarterly_1379_xxl_q1_2025_matches_deepdive_net():
    """PO quarterly Amazon net must match sales_df net for 1379YKGREEN-XXL Q1 2025."""
    from collections import defaultdict

    from backend.services.sales import _mtr_to_sales_df

    rows = []
    for i in range(31):
        rows.append(("2025-01-15", "Shipment", 769.0, f"J-S{i}", f"J-S{i}"))
    for i in range(6):
        rows.append(("2025-01-20", "Refund", -769.0, f"J-R{i}", f"J-R{i}"))
    for i in range(34):
        rows.append(("2025-02-10", "Shipment", 769.0, f"F-S{i}", f"F-S{i}"))
    for i in range(3):
        rows.append(("2025-02-11", "Shipment", 0.0, f"F-Z{i}", f"F-Z{i}"))
    for i in range(7):
        rows.append(("2025-02-12", "Refund", -769.0, f"F-R{i}", f"F-R{i}"))
    for i in range(28):
        rows.append(("2025-03-05", "Shipment", 769.0, f"M-S{i}", f"M-S{i}"))
    rows.append(("2025-03-19", "Shipment", 0.0, "M-Z0", "M-Z0"))
    for i in range(7):
        rows.append(("2025-03-20", "Refund", -769.0, f"M-R{i}", f"M-R{i}"))

    mtr_df = pd.DataFrame(
        {
            "Date": pd.to_datetime([r[0] for r in rows]),
            "Reporting_Date": pd.to_datetime([r[0] for r in rows]),
            "SKU": ["1379YKGREEN-XXL"] * len(rows),
            "Transaction_Type": [r[1] for r in rows],
            "Quantity": [1.0] * len(rows),
            "Invoice_Amount": [r[2] for r in rows],
            "Order_Id": [r[3] for r in rows],
            "Invoice_Number": [r[4] for r in rows],
        }
    )
    sales = _mtr_to_sales_df(mtr_df, {})
    sales["_month"] = sales["TxnDate"].dt.to_period("M").astype(str)
    sales_net = {
        m: int(sales.loc[sales["_month"] == m, "Units_Effective"].sum())
        for m in ["2025-01", "2025-02", "2025-03"]
    }

    start_ts = pd.Timestamp("2024-06-01")
    end_ts = pd.Timestamp("2026-06-04")
    today = pd.Timestamp.today()
    q_label_map = {(2025, 4): "Jan-Mar 2025"}
    quarter_sums: dict = defaultdict(int)
    n = _accumulate_shipment_frame(
        mtr_df,
        "amazon",
        None,
        strip_pl=True,
        canonical_oms=False,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=defaultdict(int),
        units_30=defaultdict(int),
        days_30=defaultdict(set),
        demand_basis="Net",
    )
    assert n > 0
    po_q1 = quarter_sums[("1379YKGREEN-XXL", "Jan-Mar 2025")]
    assert po_q1 == sum(sales_net.values()) == 25 + 27 + 21 == 73


def test_amazon_mtr_dedup_prevents_duplicate_quarterly_inflation():
    """Duplicate FBA shadow row must not inflate PO quarterly after dedup."""
    from backend.services.mtr import dedup_amazon_mtr_dataframe

    kwargs = _make_quarterly_kwargs({(2025, 4): "Jan-Mar 2025"})
    base = {
        "Date": pd.Timestamp("2025-01-15"),
        "Reporting_Date": pd.Timestamp("2025-01-15"),
        "SKU": "SKU-DUP",
        "Transaction_Type": "Shipment",
        "Quantity": 5.0,
        "Invoice_Amount": 500.0,
        "Order_Id": "O-DUP-1",
        "Invoice_Number": "INV-DUP",
    }
    keyed = pd.DataFrame([base])
    shadow = pd.DataFrame([{**base, "Order_Id": "", "Quantity": 5.0}])
    combined = pd.concat([keyed, shadow], ignore_index=True)
    deduped = dedup_amazon_mtr_dataframe(combined)
    assert len(deduped) == 1
    _accumulate_shipment_frame(deduped, "amazon", None, **kwargs)
    assert kwargs["quarter_sums"][("SKU-DUP", "Jan-Mar 2025")] == 5


def test_meesho_txndate_only_frame_accumulates():
    """Warm Meesho frames that only have TxnDate must still count (Deepdive parity)."""
    start_ts = pd.Timestamp("2024-06-01")
    end_ts = pd.Timestamp("2026-06-04")
    today = pd.Timestamp.today()
    q_label_map = {(2025, 4): "Jan-Mar 2025"}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)

    df = pd.DataFrame(
        {
            "TxnDate": pd.to_datetime(["2025-02-10"]),
            "OMS_SKU": ["1379YKGREEN-3XL"],
            "TxnType": ["Shipment"],
            "Quantity": [24],
        }
    )
    n = _accumulate_shipment_frame(
        df,
        "meesho",
        None,
        strip_pl=False,
        canonical_oms=True,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
    )
    assert n == 1
    assert quarter_sums[("1379YKGREEN-3XL", "Jan-Mar 2025")] == 24


def test_meesho_sales_not_suppressed_by_amazon_same_sku_day():
    """Meesho sales_df rows must fill when only Amazon claimed that SKU-day."""
    from backend.services.po_engine import get_indian_fy_quarter, quarter_col_name
    from backend.services.po_quarterly_fast import _accumulate_sales_df_shipments

    start_ts = pd.Timestamp("2024-06-01")
    end_ts = pd.Timestamp("2026-06-04")
    today = pd.Timestamp.today()
    day = pd.Timestamp("2025-02-10").normalize()
    fy, qn = get_indian_fy_quarter(day)
    q_label_map = {(fy, qn): quarter_col_name(fy, qn)}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)
    platform_day_keys = {("amazon", "1379YKGREEN-3XL", day)}

    sales = pd.DataFrame(
        {
            "Sku": ["1379YKGREEN-3XL"],
            "TxnDate": [day],
            "Transaction Type": ["Shipment"],
            "Quantity": [24],
            "Source": ["Meesho"],
        }
    )
    n = _accumulate_sales_df_shipments(
        sales,
        None,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
        platform_day_keys=platform_day_keys,
    )
    assert n == 1
    assert quarter_sums[("1379YKGREEN-3XL", quarter_col_name(fy, qn))] == 24


def test_resolve_meesho_frame_prefers_filled_disk(tmp_path, monkeypatch):
    from backend.services import shared_frames as sf

    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    blank = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-02-01"] * 4),
            "SKU": ["", "", "", ""],
            "OMS_SKU": ["", "", "", ""],
            "Quantity": [1, 1, 1, 1],
        }
    )
    filled = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-02-01"] * 4),
            "SKU": ["1379YKGREEN-3XL"] * 4,
            "OMS_SKU": ["1379YKGREEN-3XL"] * 4,
            "Quantity": [1, 1, 1, 1],
        }
    )
    filled.to_parquet(tmp_path / "meesho_df.parquet", index=False)
    out = sf.resolve_meesho_frame(blank)
    assert not out.empty
    assert (out["OMS_SKU"].astype(str) == "1379YKGREEN-3XL").all()


def test_meesho_blank_oms_backfilled_from_sku_column():
    """Tier-3-style Meesho rows with blank OMS + filled SKU must count."""
    start_ts = pd.Timestamp("2024-06-01")
    end_ts = pd.Timestamp("2026-06-04")
    today = pd.Timestamp.today()
    q_label_map = {(2025, 4): "Jan-Mar 2025"}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-02-10"]),
            "OMS_SKU": [""],
            "SKU": ["1379YKGREEN-3XL"],
            "TxnType": ["Shipment"],
            "Quantity": [31],
        }
    )
    n = _accumulate_shipment_frame(
        df,
        "meesho",
        None,
        strip_pl=False,
        canonical_oms=True,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
    )
    assert n == 1
    assert quarter_sums[("1379YKGREEN-3XL", "Jan-Mar 2025")] == 31


def test_sales_read_cols_include_combo_fan():
    """PO_SESSION_ONLY parquet load must keep _Combo_Fan or quarterly inflates."""
    from backend.services.po_quarterly_fast import _SALES_READ_COLS

    assert "_Combo_Fan" in _SALES_READ_COLS


def test_accumulate_sales_df_drops_combo_fan_component_copies():
    """Combo component copies must not land in File-matching quarterly totals."""
    from collections import defaultdict
    from datetime import timedelta

    import pandas as pd

    from backend.services.po_engine import quarter_col_name
    from backend.services.po_quarterly_fast import _accumulate_sales_df_shipments, _quarter_seq

    sales = pd.DataFrame(
        {
            "Sku": ["1037YKBLUE-3XL", "1037YKBLUE-3XL", "1037YKBLUE-3XL"],
            "TxnDate": pd.to_datetime(["2025-02-01", "2025-02-02", "2025-02-03"]),
            "Quantity": [10, 20, 5],
            "Transaction Type": ["Shipment", "Shipment", "Shipment"],
            "Source": ["Amazon", "Amazon", "Myntra"],
            "_Combo_Fan": [False, True, False],
        }
    )
    start_ts = pd.Timestamp("2024-10-01")
    end_ts = pd.Timestamp("2025-03-31") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    today = pd.Timestamp("2026-07-29")
    q_label_map = {(fy, qn): quarter_col_name(fy, qn) for fy, qn in _quarter_seq(8)}
    quarter_sums: dict = defaultdict(int)
    units_90: dict = defaultdict(int)
    units_30: dict = defaultdict(int)
    days_30: dict = defaultdict(set)
    _accumulate_sales_df_shipments(
        sales,
        None,
        group_by_parent=False,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_90=today - timedelta(days=90),
        cutoff_30=today - timedelta(days=30),
        q_label_map=q_label_map,
        quarter_sums=quarter_sums,
        units_90=units_90,
        units_30=units_30,
        days_30=days_30,
        platform_day_keys=set(),
        demand_basis="Sold",
    )
    # 10 + 5 real; the 20 combo-fan units must be excluded.
    assert quarter_sums[("1037YKBLUE-3XL", "Jan-Mar 2025")] == 15
