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
    """Shared kwargs for accumulate tests."""
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
    )
    assert n > 0
    # Net = 15 + 2 - 3 - 1 = 13
    assert quarter_sums[("FK-SKU", quarter_col_name(fy, qn))] == 13
