"""Load PO calculate inputs from PostgreSQL materializations (not session copies)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class PoInputs:
    sales_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    inventory_df_variant: pd.DataFrame = field(default_factory=pd.DataFrame)
    inventory_df_parent: pd.DataFrame = field(default_factory=pd.DataFrame)
    sku_mapping: dict = field(default_factory=dict)
    ads_source_label: str = "session"
    sales_source: str = "session"
    inventory_source: str = "session"


def po_inputs_from_pg_enabled() -> bool:
    try:
        from ..db.forecast_ops_tables import normalized_tables_enabled
        from ..db.forecast_sales_materializations import materializations_enabled

        return normalized_tables_enabled() and materializations_enabled()
    except Exception:
        return False


def load_po_inputs(sess, body: dict[str, Any]) -> PoInputs:
    """
    Prefer PostgreSQL materialized tables + normalized inventory.

    Falls back to shared warm-cache frames (single process copy), not session duplicates.
    """
    from .shared_frames import (
        session_inventory_parent,
        session_inventory_variant,
        session_sales_df,
        warm_frame,
    )

    out = PoInputs()
    period = int(body.get("period_days") or 30)
    planning_date = body.get("planning_date")
    use_seasonality = bool(body.get("use_seasonality", False))
    use_ly_fallback = bool(body.get("use_ly_fallback", True))

    if po_inputs_from_pg_enabled():
        try:
            from ..db.forecast_sales_materializations import load_po_sales_df

            mat = load_po_sales_df(
                sess,
                period_days=period,
                planning_date=planning_date,
                use_seasonality=use_seasonality,
                use_ly_fallback=use_ly_fallback,
            )
            if mat is not None and not mat.empty:
                out.sales_df = mat
                out.ads_source_label = "materialized-daily"
                out.sales_source = "postgres_materialized"
        except Exception:
            _log.exception("load_po_sales_df from PG failed")

        if out.sales_df.empty:
            try:
                from ..db.forecast_ops_tables import load_platform_sales_dataframe

                unified = load_platform_sales_dataframe("unified")
                if unified is not None and not unified.empty:
                    out.sales_df = unified
                    out.ads_source_label = "postgres_unified"
                    out.sales_source = "postgres_normalized"
            except Exception:
                _log.exception("load_platform_sales_dataframe unified failed")

        try:
            from ..db.forecast_ops_tables import load_inventory_dataframe, load_sku_mapping

            inv = load_inventory_dataframe()
            if inv is not None and not inv.empty:
                out.inventory_df_variant = inv
                out.inventory_source = "postgres_normalized"
            sm = load_sku_mapping()
            if sm:
                out.sku_mapping = sm
        except Exception:
            _log.exception("load PO inventory/sku_mapping from PG failed")

    if out.sales_df.empty:
        out.sales_df = session_sales_df(sess)
        if not out.sales_df.empty:
            out.sales_source = "warm_cache_shared"

    if out.inventory_df_variant.empty:
        out.inventory_df_variant = session_inventory_variant(sess)
        if not out.inventory_df_variant.empty and out.inventory_source == "session":
            out.inventory_source = "warm_cache_shared"

    out.inventory_df_parent = session_inventory_parent(sess)
    if out.inventory_df_parent.empty:
        parent = warm_frame("inventory_df_parent", sess)
        if parent is not None and not parent.empty:
            out.inventory_df_parent = parent

    if not out.sku_mapping:
        if isinstance(getattr(sess, "sku_mapping", None), dict) and sess.sku_mapping:
            out.sku_mapping = dict(sess.sku_mapping)
        else:
            try:
                import backend.main as _main

                sm = (_main._warm_cache or {}).get("sku_mapping")
                if isinstance(sm, dict):
                    out.sku_mapping = sm
            except Exception:
                pass

    return out
