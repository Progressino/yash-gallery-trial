"""PO calculation pipeline — validate, preprocess, snapshot, then vectorized engine.

Stages (no engine math until snapshot is locked):
  1. Gate — block if inputs still processing or stale
  2. Preprocess — hydrate sidecars, overlay inventory, resolve ADS sales source
  3. Materialize — persist intermediate parquet snapshots (version-keyed)
  4. Validate — completeness checks on the master input bundle
  5. Lock — assign snapshot_id consumed by shared-cache fingerprint
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

PO_PIPELINE_VERSION = 1

ProgressCb = Optional[Callable[[int, str], None]]

_SNAPSHOT_TABLES = (
    "ads_sales",
    "inventory",
    "existing_po",
    "raise_ledger",
    "returns",
    "inventory_history",
    "sku_status",
)


@dataclass
class DatasetVersions:
    """Per-dataset version tokens — any change invalidates dependent caches."""

    sales: str = ""
    inventory: str = ""
    existing_po: str = ""
    raises: str = ""
    returns: str = ""
    sku_status: str = ""
    history: str = ""
    sku_mapping: str = ""
    tier3: dict = field(default_factory=dict)
    warm_cache_generation: int = 0
    inventory_snapshot_date: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "sales": self.sales,
            "inventory": self.inventory,
            "existing_po": self.existing_po,
            "raises": self.raises,
            "returns": self.returns,
            "sku_status": self.sku_status,
            "history": self.history,
            "sku_mapping": self.sku_mapping,
            "tier3": self.tier3,
            "warm_cache_generation": self.warm_cache_generation,
            "inventory_snapshot_date": self.inventory_snapshot_date,
            "pipeline_version": PO_PIPELINE_VERSION,
        }

    def composite_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass
class PoInputSnapshot:
    """Immutable input bundle for one PO calculation run."""

    snapshot_id: str
    versions: DatasetVersions
    locked_at: str
    ready: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    sales_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    ads_label: str = "OMS"
    inv_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    inv_variant: pd.DataFrame = field(default_factory=pd.DataFrame)
    inv_parent: pd.DataFrame | None = None
    existing_po_df: pd.DataFrame | None = None
    inventory_history_df: pd.DataFrame | None = None
    po_raise_ledger_df: pd.DataFrame | None = None
    sku_status_df: pd.DataFrame | None = None
    po_return_overlay_df: pd.DataFrame | None = None
    sku_mapping: dict | None = None
    group_by_parent: bool = False
    lookback_days: int = 14
    missing_inventory_skus: int = 0
    missing_existing_po_skus: int = 0
    existing_po_generation: int = 0


def _pipeline_cache_dir() -> str:
    base = (os.environ.get("WARM_CACHE_DIR") or "/data/warm_cache").strip()
    root = os.path.join(base, "pipeline")
    os.makedirs(root, exist_ok=True)
    return root


def _version_meta_path() -> str:
    return os.path.join(_pipeline_cache_dir(), "versions.json")


def _table_parquet(name: str) -> str:
    return os.path.join(_pipeline_cache_dir(), f"{name}_snapshot.parquet")


def collect_dataset_versions(sess, body: dict) -> DatasetVersions:
    from .po_shared_cache import (
        _existing_po_fingerprint,
        _raise_ledger_fingerprint,
        _return_overlay_fingerprint,
        _sku_mapping_fingerprint,
        _sku_status_fingerprint,
        normalize_planning_date,
    )

    planning = normalize_planning_date(body)
    lookback = max(int(body.get("raise_ledger_lookback_days") or 14), 14)

    inv = getattr(sess, "inventory_df_variant", None)
    inv_rows = int(len(inv)) if inv is not None and hasattr(inv, "__len__") else 0
    inv_skus = 0
    if inv is not None and not getattr(inv, "empty", True) and "OMS_SKU" in inv.columns:
        inv_skus = int(inv["OMS_SKU"].astype(str).nunique())

    sales = getattr(sess, "sales_df", None)
    sales_rows = int(len(sales)) if sales is not None and hasattr(sales, "__len__") else 0

    hist = getattr(sess, "daily_inventory_history_df", None)
    hist_rows = int(len(hist)) if hist is not None and hasattr(hist, "__len__") else 0
    hist_max = str(getattr(sess, "daily_inventory_history_max_date", "") or "")[:10]
    hist_fn = str(getattr(sess, "daily_inventory_history_filename", "") or "")

    warm_gen = 0
    try:
        import backend.main as _main

        warm_gen = int(getattr(_main, "_warm_cache_generation", 0) or 0)
    except Exception:
        pass

    tier3_token: dict[str, str] = {}
    try:
        from .daily_store import get_tier3_sync_token

        tier3_token = get_tier3_sync_token() or {}
    except Exception:
        pass

    return DatasetVersions(
        sales=f"sales:{sales_rows}:{_sales_through_token(sess)}",
        inventory=f"inv:{inv_rows}:{inv_skus}:{str(getattr(sess, 'inventory_snapshot_date', '') or '')}",
        existing_po=_existing_po_fingerprint(sess),
        raises=_raise_ledger_fingerprint(planning, lookback),
        returns=_return_overlay_fingerprint(sess),
        sku_status=_sku_status_fingerprint(sess),
        history=f"hist:{hist_rows}:{hist_max}:{hist_fn}",
        sku_mapping=_sku_mapping_fingerprint(sess),
        tier3=tier3_token,
        warm_cache_generation=warm_gen,
        inventory_snapshot_date=str(getattr(sess, "inventory_snapshot_date", "") or "")[:10],
    )


def _sales_through_token(sess) -> str:
    try:
        from .tier3_session_merge import effective_sales_through

        return str(effective_sales_through(sess) or "")
    except Exception:
        return ""


def check_calculate_gate(sess, *, cov=None) -> dict[str, Any]:
    """Readiness gate — block calculate until inputs are stable and complete."""
    from .po_readiness import (
        PO_MIN_INVENTORY_ROWS,
        PO_MIN_SALES_ROWS,
        _effective_row_floors,
        background_job_names,
        critical_restore_running,
    )
    from .shared_frames import session_inventory_variant, session_sales_df

    blockers: list[str] = []
    warnings: list[str] = []

    if critical_restore_running(sess):
        blockers.append("Session or inventory restore is still running — wait and retry.")

    bg = background_job_names(sess)
    for job in ("inventory_upload", "daily_inventory_upload", "session_restore"):
        if job in bg:
            blockers.append(f"{job.replace('_', ' ')} is still processing.")

    if cov is not None:
        sales_rows, inv_rows = _effective_row_floors(cov, sess)
    else:
        from .shared_frames import frame_row_count

        sales_rows = int(frame_row_count("sales_df", sess) or len(session_sales_df(sess)))
        inv_rows = int(
            frame_row_count("inventory_df_variant", sess) or len(session_inventory_variant(sess))
        )
        from .po_readiness import _pg_row_floors, _warm_row_floors

        pg_sales, pg_inv = _pg_row_floors()
        warm_sales, warm_inv = _warm_row_floors()
        sales_rows = max(sales_rows, pg_sales, warm_sales)
        inv_rows = max(inv_rows, pg_inv, warm_inv)
    if sales_rows < PO_MIN_SALES_ROWS:
        blockers.append(
            f"Sales not ready ({sales_rows:,} rows; need {PO_MIN_SALES_ROWS:,})."
        )
    if inv_rows < PO_MIN_INVENTORY_ROWS:
        blockers.append(
            f"Inventory not ready ({inv_rows:,} rows; need {PO_MIN_INVENTORY_ROWS:,})."
        )

    ep_gen = int(getattr(sess, "existing_po_generation", 0) or 0)
    ep_df = getattr(sess, "existing_po_df", None)
    ep_loaded = ep_df is not None and not getattr(ep_df, "empty", True)
    if cov is not None and bool(getattr(cov, "existing_po", False)) and not ep_loaded:
        blockers.append("Existing PO sheet is flagged but not loaded — refresh or re-upload.")

    if ep_gen > 0 and not ep_loaded:
        blockers.append("Existing PO was uploaded but is not hydrated in this session.")

    try:
        from .inventory_staleness import build_inventory_staleness, daily_inventory_history_bounds

        hist_df = getattr(sess, "daily_inventory_history_df", None)
        _, hist_max = daily_inventory_history_bounds(hist_df)
        stale = build_inventory_staleness(
            inventory_loaded=inv_rows > 0,
            inventory_snapshot_date=str(getattr(sess, "inventory_snapshot_date", "") or "")[:10]
            or None,
            daily_inventory_history_loaded=bool(hist_max),
            daily_inventory_history_max_date=hist_max or None,
        )
        for w in stale.get("inventory_staleness_warnings") or []:
            warnings.append(str(w))
        strict_stale = (os.environ.get("PO_PIPELINE_STRICT_STALE") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if strict_stale and stale.get("inventory_snapshot_stale") and not hist_max:
            blockers.append(
                "Inventory snapshot is stale and daily inventory history is missing."
            )
    except Exception:
        logger.exception("inventory staleness gate check failed")

    strict = (os.environ.get("PO_PIPELINE_STRICT") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if strict and not ep_loaded:
        blockers.append("PO_PIPELINE_STRICT: Existing PO sheet is required.")

    return {
        "calculate_allowed": len(blockers) == 0,
        "blockers": blockers,
        "warnings": warnings,
        "dataset_versions": collect_dataset_versions(sess, {}).as_dict(),
        "pipeline_version": PO_PIPELINE_VERSION,
    }


def _validate_snapshot_completeness(
    snap: PoInputSnapshot,
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []

    if snap.sales_df is None or snap.sales_df.empty:
        blockers.append("ADS sales source is empty after preprocessing.")
    if snap.inv_df is None or snap.inv_df.empty:
        blockers.append("Inventory snapshot is empty after preprocessing.")

    inv_skus: set[str] = set()
    if snap.inv_df is not None and not snap.inv_df.empty and "OMS_SKU" in snap.inv_df.columns:
        inv_skus = set(snap.inv_df["OMS_SKU"].astype(str).str.strip().str.upper())
        inv_skus.discard("")

    ep_skus: set[str] = set()
    if (
        snap.existing_po_df is not None
        and not snap.existing_po_df.empty
        and "OMS_SKU" in snap.existing_po_df.columns
    ):
        ep_skus = set(snap.existing_po_df["OMS_SKU"].astype(str).str.strip().str.upper())
        ep_skus.discard("")

    if inv_skus and ep_skus:
        missing_ep = sorted(inv_skus - ep_skus)
        snap.missing_existing_po_skus = len(missing_ep)
        if missing_ep:
            msg = (
                f"{len(missing_ep):,} inventory SKUs have no Existing PO row "
                f"(e.g. {missing_ep[0]})."
            )
            warnings.append(msg)
            strict = (os.environ.get("PO_PIPELINE_STRICT") or "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if strict and len(missing_ep) > max(50, int(0.05 * len(inv_skus))):
                blockers.append(msg)

    ep_gen = snap.existing_po_generation
    if ep_gen > 0 and not ep_skus:
        blockers.append("Existing PO generation is set but pipeline snapshot has no EP rows.")

    return blockers, warnings


def _persist_table_if_changed(name: str, df: pd.DataFrame | None, version_token: str) -> None:
    if df is None or getattr(df, "empty", True):
        return
    meta_path = _version_meta_path()
    meta: dict[str, Any] = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    prev = (meta.get("tables") or {}).get(name)
    if prev == version_token:
        return
    path = _table_parquet(name)
    tmp = path + ".tmp"
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        tables = dict(meta.get("tables") or {})
        tables[name] = version_token
        meta["tables"] = tables
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(meta_path + ".tmp", "w") as f:
            json.dump(meta, f, default=str)
        os.replace(meta_path + ".tmp", meta_path)
    except Exception:
        logger.exception("pipeline snapshot persist failed for %s", name)


def materialize_intermediate_tables(snap: PoInputSnapshot) -> None:
    """Write parquet snapshots only when the source dataset version changes."""
    v = snap.versions
    _persist_table_if_changed("ads_sales", snap.sales_df, v.sales)
    _persist_table_if_changed("inventory", snap.inv_df, v.inventory)
    _persist_table_if_changed("existing_po", snap.existing_po_df, v.existing_po)
    _persist_table_if_changed("raise_ledger", snap.po_raise_ledger_df, v.raises)
    _persist_table_if_changed("returns", snap.po_return_overlay_df, v.returns)
    _persist_table_if_changed("inventory_history", snap.inventory_history_df, v.history)
    _persist_table_if_changed("sku_status", snap.sku_status_df, v.sku_status)


def _resolve_ads_sales_source(
    sess,
    body: dict,
    *,
    sales_df: pd.DataFrame,
    inputs,
    progress_cb: ProgressCb = None,
) -> tuple[pd.DataFrame, str]:
    from .po_calculate_run import _build_platform_sales_df, _trim_sales_for_po_memory
    from .po_inputs import po_inputs_from_pg_enabled
    from .tier3_session_merge import build_po_ads_platform_sales

    period = int(body.get("period_days", 30))
    if progress_cb:
        progress_cb(28, "Loading Tier-3 daily sales for ADS window…")

    platform_sales = build_po_ads_platform_sales(
        sess,
        planning_date=body.get("planning_date"),
        period_days=period,
        use_seasonality=bool(body.get("use_seasonality", False)),
        use_ly_fallback=bool(body.get("use_ly_fallback", True)),
    )
    mat_sales = None
    if po_inputs_from_pg_enabled() and not sales_df.empty and inputs.sales_source.startswith(
        "postgres"
    ):
        mat_sales = sales_df
    else:
        try:
            from ..db.forecast_sales_materializations import load_po_sales_df

            mat_sales = load_po_sales_df(
                sess,
                period_days=period,
                planning_date=body.get("planning_date"),
                use_seasonality=bool(body.get("use_seasonality", False)),
                use_ly_fallback=bool(body.get("use_ly_fallback", True)),
            )
        except Exception:
            logger.exception("load_po_sales_df failed — falling back to raw sales")

    if mat_sales is not None and not mat_sales.empty:
        ads_source = mat_sales
        ads_label = (
            inputs.ads_source_label
            if inputs.sales_source.startswith("postgres")
            else "materialized-daily"
        )
    elif not platform_sales.empty:
        ads_source = platform_sales
        ads_label = "platform"
    else:
        ads_source = _build_platform_sales_df(sess)
        ads_label = "platform-built"
        if ads_source.empty:
            ads_source = sales_df
            ads_label = "OMS"

    if ads_label != "materialized-daily":
        ads_source = _trim_sales_for_po_memory(
            ads_source,
            period_days=period,
            use_seasonality=bool(body.get("use_seasonality", False)),
            use_ly_fallback=bool(body.get("use_ly_fallback", True)),
        )
    return ads_source, ads_label


def _prepare_inventory_history(
    sess,
    body: dict,
    inv_variant: pd.DataFrame,
    inv_parent: pd.DataFrame | None,
    *,
    group_by_parent: bool,
    progress_cb: ProgressCb = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    import gc as _gc

    from .daily_inventory_history import (
        daily_inventory_meta_is_newer,
        overlay_inventory_variant_from_history,
        read_daily_inventory_history_disk_meta,
    )
    from .daily_inventory_upload_run import (
        _MAX_HISTORY_DAYS,
        _series_as_dates,
        _trim_history_to_recent,
    )
    from .po_session_hydrate import (
        ensure_inventory_history_authoritative_for_read,
        ensure_po_sidecars_hydrated,
    )

    period = int(body.get("period_days", 30))
    try:
        disk_meta = read_daily_inventory_history_disk_meta()
        if daily_inventory_meta_is_newer(disk_meta, sess) or getattr(
            sess, "daily_inventory_history_df", None
        ) is None or getattr(sess.daily_inventory_history_df, "empty", True):
            ensure_po_sidecars_hydrated(sess)
    except Exception:
        logger.exception("ensure daily inventory history before PO pipeline failed")

    try:
        raw_ih = ensure_inventory_history_authoritative_for_read(sess)
    except Exception:
        logger.exception("authoritative inventory history hydrate failed")
        raw_ih = getattr(sess, "daily_inventory_history_df", None)
    inv_history_for_calc: pd.DataFrame | None = None
    rows_dropped = 0

    if raw_ih is not None and not raw_ih.empty:
        ih_rows = len(raw_ih)
        if progress_cb:
            progress_cb(22, f"Preparing daily inventory history ({ih_rows:,} rows)…")
        if ih_rows > (_MAX_HISTORY_DAYS + 10) * 500:
            if progress_cb:
                progress_cb(24, f"Trimming oversized inventory history ({ih_rows:,} rows)…")
            trimmed_sess, _trim_note = _trim_history_to_recent(raw_ih, _MAX_HISTORY_DAYS)
            if len(trimmed_sess) < len(raw_ih):
                rows_dropped += len(raw_ih) - len(trimmed_sess)
                sess.daily_inventory_history_df = trimmed_sess
                raw_ih = trimmed_sess

        inv_dates = _series_as_dates(raw_ih["Date"])
        max_inv = inv_dates.max()
        if pd.notna(max_inv):
            depth_days = min(period + 14, _MAX_HISTORY_DAYS)
            pretrim_cutoff = pd.Timestamp(max_inv).normalize() - pd.Timedelta(days=depth_days)
            min_inv = inv_dates.min()
            if pd.notna(min_inv) and min_inv >= pretrim_cutoff:
                inv_history_for_calc = raw_ih
            else:
                mask = inv_dates >= pretrim_cutoff
                inv_history_for_calc = raw_ih.loc[mask].reset_index(drop=True)
                if len(inv_history_for_calc) < len(raw_ih):
                    rows_dropped += len(raw_ih) - len(inv_history_for_calc)
        else:
            inv_history_for_calc = raw_ih

        if rows_dropped >= 100_000:
            if progress_cb:
                progress_cb(27, "Releasing trimmed inventory memory…")
            _gc.collect()

    inv_df = (
        inv_parent
        if group_by_parent and inv_parent is not None and not inv_parent.empty
        else inv_variant
    )

    if inv_history_for_calc is not None and not inv_history_for_calc.empty:
        ref_date = str(body.get("planning_date") or "")[:10] or None
        inv_variant, ov_meta = overlay_inventory_variant_from_history(
            inv_variant,
            inv_history_for_calc,
            snapshot_date=str(getattr(sess, "inventory_snapshot_date", "") or "") or None,
            reference_date=ref_date,
        )
        if ov_meta.get("applied"):
            logger.info(
                "PO pipeline: inventory overlaid from history (%s SKUs, as of %s)",
                ov_meta.get("skus_updated", 0),
                ov_meta.get("history_as_of") or "?",
            )
            inv_df = (
                inv_parent
                if group_by_parent and inv_parent is not None and not inv_parent.empty
                else inv_variant
            )

    return inv_variant, inv_df, inv_history_for_calc


def prepare_po_snapshot(
    sess,
    body: dict,
    *,
    skip_hydrate: bool = False,
    progress_cb: ProgressCb = None,
    enforce_gate: bool = True,
    session_id: str | None = None,
    sync_sidecars: Callable[[], None] | None = None,
) -> PoInputSnapshot | dict[str, Any]:
    """Build a locked input snapshot — engine must only read tables from this bundle."""
    from .po_calculate_run import _po_return_overlay_for_calc
    from .po_inputs import load_po_inputs
    from .po_session_hydrate import (
        effective_sku_status_df_for_engine,
        hydrate_po_session_for_calculate,
    )
    from .shared_frames import session_inventory_variant, session_sales_df

    if enforce_gate:
        gate = check_calculate_gate(sess)
        if not gate.get("calculate_allowed"):
            return {
                "ok": False,
                "message": gate["blockers"][0] if gate.get("blockers") else "PO inputs not ready.",
                "pipeline_blockers": gate.get("blockers") or [],
                "pipeline_warnings": gate.get("warnings") or [],
            }

    if not skip_hydrate:
        hydrate_po_session_for_calculate(sess)

    inputs = load_po_inputs(sess, body)
    if progress_cb:
        progress_cb(5, "Validating sales and inventory…")

    sales_df = inputs.sales_df if not inputs.sales_df.empty else session_sales_df(sess)
    inv_variant = (
        inputs.inventory_df_variant
        if not inputs.inventory_df_variant.empty
        else session_inventory_variant(sess)
    )
    if sales_df.empty:
        return {"ok": False, "message": "Build Sales first (upload platforms, then POST /api/upload/build-sales)."}
    if inv_variant.empty:
        return {"ok": False, "message": "Upload Inventory first."}

    group_by_parent = bool(body.get("group_by_parent", False))
    inv_parent = (
        inputs.inventory_df_parent
        if not inputs.inventory_df_parent.empty
        else getattr(sess, "inventory_df_parent", None)
    )
    sku_mapping = inputs.sku_mapping or getattr(sess, "sku_mapping", None) or None
    lookback = max(int(body.get("raise_ledger_lookback_days") or 14), 14)

    if progress_cb:
        progress_cb(12, "Loading raise ledger and existing PO…")

    def _load_ledger() -> None:
        from .po_raise_import import hydrate_session_ledger_from_db

        hydrate_session_ledger_from_db(
            sess, body.get("planning_date"), lookback_days=lookback, authoritative=True
        )

    def _load_existing_po() -> None:
        from .existing_po import ensure_existing_po_hydrated

        ensure_existing_po_hydrated(sess)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_load_ledger), pool.submit(_load_existing_po)]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception:
                logger.exception("parallel sidecar load failed")

    if body.get("auto_import_yesterday_ledger", True) and session_id:
        from .po_raise_archive import try_auto_import_recent_ledgers

        ledger_auto_import = try_auto_import_recent_ledgers(
            sess,
            session_id,
            body.get("planning_date"),
            group_by_parent=group_by_parent,
            lookback_days=lookback,
        )
        if ledger_auto_import and ledger_auto_import.get("ok"):
            from .po_raise_import import hydrate_session_ledger_from_db

            hydrate_session_ledger_from_db(
                sess, body.get("planning_date"), lookback_days=lookback, authoritative=True
            )
            if sync_sidecars:
                try:
                    sync_sidecars()
                except Exception:
                    logger.exception("sync_sidecars after ledger auto-import failed")

    inv_variant, inv_df, inv_history = _prepare_inventory_history(
        sess,
        body,
        inv_variant,
        inv_parent,
        group_by_parent=group_by_parent,
        progress_cb=progress_cb,
    )

    ads_source, ads_label = _resolve_ads_sales_source(
        sess, body, sales_df=sales_df, inputs=inputs, progress_cb=progress_cb
    )
    logger.info("PO pipeline ADS source: %s (%d rows)", ads_label, len(ads_source))

    versions = collect_dataset_versions(sess, body)
    locked_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshot_id = f"PO_{locked_at.replace(':', '').replace('-', '')[:15]}_{versions.composite_hash()[:8]}"

    snap = PoInputSnapshot(
        snapshot_id=snapshot_id,
        versions=versions,
        locked_at=locked_at,
        ready=True,
        sales_df=ads_source,
        ads_label=ads_label,
        inv_df=inv_df,
        inv_variant=inv_variant,
        inv_parent=inv_parent,
        existing_po_df=(
            sess.existing_po_df
            if getattr(sess, "existing_po_df", None) is not None
            and not sess.existing_po_df.empty
            else None
        ),
        inventory_history_df=inv_history,
        po_raise_ledger_df=(
            sess.po_raise_ledger_df
            if getattr(sess, "po_raise_ledger_df", None) is not None
            and not sess.po_raise_ledger_df.empty
            else None
        ),
        sku_status_df=effective_sku_status_df_for_engine(sess),
        po_return_overlay_df=_po_return_overlay_for_calc(sess),
        sku_mapping=sku_mapping,
        group_by_parent=group_by_parent,
        lookback_days=lookback,
        existing_po_generation=int(getattr(sess, "existing_po_generation", 0) or 0),
    )
    val_blockers, val_warnings = _validate_snapshot_completeness(snap)
    gate_warnings = (check_calculate_gate(sess).get("warnings") or []) if enforce_gate else []
    snap.warnings = list(dict.fromkeys(gate_warnings + val_warnings))
    if val_blockers:
        snap.ready = False
        snap.blockers = val_blockers
        return {
            "ok": False,
            "message": val_blockers[0],
            "pipeline_blockers": val_blockers,
            "pipeline_warnings": snap.warnings,
            "snapshot_id": snapshot_id,
        }

    materialize_intermediate_tables(snap)
    sess.po_pipeline_snapshot_id = snapshot_id
    sess.po_pipeline_versions = versions.as_dict()
    return snap


def build_pipeline_readiness(sess, *, cov=None) -> dict[str, Any]:
    """Extended readiness payload for GET /api/po/readiness."""
    gate = check_calculate_gate(sess, cov=cov)
    versions = collect_dataset_versions(sess, {})
    return {
        **gate,
        "snapshot_id": str(getattr(sess, "po_pipeline_snapshot_id", "") or ""),
        "pipeline_version": PO_PIPELINE_VERSION,
        "dataset_versions": versions.as_dict(),
    }
