"""
Prebuilt Intelligence artifacts — hot path (summary) vs deep path (full bundle).

Request flow (no heavy SQLite on hot path when artifact exists):
  memory → disk JSON → stale artifact + async rebuild → Tier-3 fallback (last resort)

Artifacts live under ``{WARM_CACHE_DIR}/intelligence/daily/``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Optional

_log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

KIND_HOT = "hot"
KIND_DEEP = "deep"

_MEM: dict[tuple[str, str, str], dict[str, Any]] = {}
_MEM_LOCK = threading.Lock()

_BUILD_QUEUED: set[tuple[str, str, str]] = set()
_BUILD_LOCK = threading.Lock()

STANDARD_WINDOW_DAYS = (7, 30, 90)
PER_DAY_ARTIFACT_LOOKBACK = int(os.environ.get("INTELLIGENCE_DAY_ARTIFACT_DAYS", "90"))


def _artifact_root() -> str:
    base = os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")
    path = os.path.join(base, "intelligence", "daily")
    os.makedirs(path, exist_ok=True)
    return path


def intelligence_version_for_window(
    start_date: str,
    end_date: str,
    *,
    basis: str = "gross",
) -> str:
    """Shared cache version key, e.g. ``2026-06-24_2026-05-25-a1b2c3d4``."""
    try:
        from ..app_version import get_build_info
        from .daily_store import get_tier3_sync_token

        tok = tuple(sorted((get_tier3_sync_token() or {}).items()))
        build = str(get_build_info().get("git_sha") or "")
    except Exception:
        tok = ()
        build = ""
    raw = f"{start_date[:10]}:{end_date[:10]}:{basis}:{tok}:{build}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{end_date[:10]}_{start_date[:10]}-{digest}"


def standard_intelligence_windows() -> list[tuple[str, str]]:
    """Preset windows (7D / 30D / 90D) aligned with the Intelligence UI."""
    today = datetime.now(IST).date()
    end = today.isoformat()
    out: list[tuple[str, str]] = []
    for days in STANDARD_WINDOW_DAYS:
        start = (today - timedelta(days=int(days))).isoformat()
        out.append((start, end))
    return out


def _artifact_basename(start_date: str, end_date: str, kind: str) -> str:
    return f"intelligence_bundle_{start_date[:10]}_{end_date[:10]}_{kind}"


def _artifact_json_path(start_date: str, end_date: str, kind: str) -> str:
    return os.path.join(_artifact_root(), f"{_artifact_basename(start_date, end_date, kind)}.json")


def _read_disk_artifact(start_date: str, end_date: str, kind: str) -> dict[str, Any] | None:
    path = _artifact_json_path(start_date, end_date, kind)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("payload"), dict):
            return data
    except Exception:
        _log.exception("read intelligence artifact failed path=%s", path)
    return None


def _write_disk_artifact(
    start_date: str,
    end_date: str,
    kind: str,
    *,
    version: str,
    payload: dict[str, Any],
) -> None:
    path = _artifact_json_path(start_date, end_date, kind)
    tmp = f"{path}.tmp"
    entry = {
        "version": version,
        "built_at": datetime.now(IST).isoformat(),
        "kind": kind,
        "start_date": start_date[:10],
        "end_date": end_date[:10],
        "payload": payload,
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entry, f, default=str)
    os.replace(tmp, path)


def save_artifact(
    start_date: str,
    end_date: str,
    kind: Literal["hot", "deep"],
    payload: dict[str, Any],
    *,
    basis: str = "gross",
) -> str:
    from .intelligence_artifact_store import (
        deep_parquet_path,
        schedule_cdn_publish,
        write_day_parquet,
        write_deep_parquet,
    )

    version = intelligence_version_for_window(start_date, end_date, basis=basis)
    s, e = start_date[:10], end_date[:10]
    entry = {
        "version": version,
        "built_at": time.time(),
        "kind": kind,
        "start_date": s,
        "end_date": e,
        "payload": payload,
    }
    with _MEM_LOCK:
        _MEM[(s, e, kind)] = entry
    try:
        if kind == KIND_HOT:
            _write_disk_artifact(s, e, kind, version=version, payload=payload)
        else:
            # Deep analytics: columnar parquet + slim JSON pointer (no huge nested JSON).
            write_deep_parquet(s, e, payload)
            slim = {
                "version": version,
                "built_at": datetime.now(IST).isoformat(),
                "kind": kind,
                "start_date": s,
                "end_date": e,
                "storage": "parquet",
                "parquet": os.path.basename(deep_parquet_path(s, e)),
                "payload": {
                    "source": payload.get("source", "tier3_sqlite"),
                    "data_completeness": payload.get("data_completeness", "full"),
                    "sales_summary": payload.get("sales_summary") or {},
                },
            }
            path = _artifact_json_path(s, e, kind)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(slim, f, default=str)
            os.replace(tmp, path)
            schedule_cdn_publish(os.path.basename(deep_parquet_path(s, e)))
        if s == e:
            if write_day_parquet(s, payload):
                schedule_cdn_publish(f"by_date/intelligence_bundle_{s}.parquet")
    except Exception:
        _log.exception("write intelligence artifact failed")
    return version


def load_artifact(
    start_date: str,
    end_date: str,
    kind: Literal["hot", "deep"],
    *,
    basis: str = "gross",
    allow_stale: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """
    Return (payload, meta). meta includes version, stale, source.
  """
    s, e = start_date[:10], end_date[:10]
    current = intelligence_version_for_window(s, e, basis=basis)
    meta: dict[str, Any] = {
        "version": current,
        "stale": False,
        "source": "none",
        "current_version": current,
    }

    with _MEM_LOCK:
        mem = _MEM.get((s, e, kind))

    if mem and isinstance(mem.get("payload"), dict):
        if mem.get("version") == current:
            meta.update(source="memory", version=mem["version"])
            return mem["payload"], meta
        if allow_stale:
            meta.update(
                stale=True,
                source="memory_stale",
                version=str(mem.get("version") or ""),
                current_version=current,
            )
            return mem["payload"], meta

    disk = _read_disk_artifact(s, e, kind)
    payload_from_disk: dict[str, Any] | None = None
    if disk:
        if disk.get("storage") == "parquet" and kind == KIND_DEEP:
            from .intelligence_artifact_store import read_deep_parquet

            parquet_payload = read_deep_parquet(s, e)
            if parquet_payload:
                slim = disk.get("payload") if isinstance(disk.get("payload"), dict) else {}
                payload_from_disk = {**parquet_payload, **slim}
        elif isinstance(disk.get("payload"), dict):
            payload_from_disk = disk["payload"]

    if payload_from_disk:
        with _MEM_LOCK:
            _MEM[(s, e, kind)] = {
                "version": disk.get("version") if disk else current,
                "built_at": time.time(),
                "kind": kind,
                "start_date": s,
                "end_date": e,
                "payload": payload_from_disk,
            }
        disk_ver = str((disk or {}).get("version") or "")
        if disk_ver == current:
            meta.update(source="disk", version=current)
            return payload_from_disk, meta
        if allow_stale:
            meta.update(
                stale=True,
                source="disk_stale",
                version=disk_ver,
                current_version=current,
            )
            return payload_from_disk, meta

    if kind == KIND_DEEP:
        from .intelligence_artifact_store import read_deep_parquet

        parquet_only = read_deep_parquet(s, e)
        if parquet_only:
            meta.update(source="parquet", version=current)
            return parquet_only, meta

    return None, meta


def _build_hot_payload(sess, start_date: str, end_date: str, limit: int) -> dict[str, Any] | None:
    from ..routers.data import _build_intelligence_bundle_payload_from_tier3
    from .dashboard_summary import _compact_platforms

    tier3 = _build_intelligence_bundle_payload_from_tier3(
        sess,
        start_date[:10],
        end_date[:10],
        int(limit),
        "gross",
        include_extras=False,
        headline_only=True,
    )
    if not tier3 or not tier3.get("platform_summary"):
        return None
    return {
        "source": "tier3_sqlite",
        "platforms": _compact_platforms(tier3.get("platform_summary") or []),
        "platform_summary": tier3.get("platform_summary") or [],
        "top_skus": tier3.get("top_skus") or [],
        "sales_summary": tier3.get("sales_summary") or {},
        "data_completeness": "partial",
    }


def _build_deep_payload(
    sess,
    start_date: str,
    end_date: str,
    limit: int,
    *,
    include_extras: bool,
) -> dict[str, Any] | None:
    from ..routers.data import (
        _build_intelligence_bundle_payload_from_tier3,
        _build_intelligence_gapfill_bundle_payload,
        _bundle_payload_has_display_data,
    )

    payload = _build_intelligence_gapfill_bundle_payload(
        sess, start_date[:10], end_date[:10], int(limit), "gross", include_extras
    )
    if payload is None or not _bundle_payload_has_display_data(payload):
        payload = _build_intelligence_bundle_payload_from_tier3(
            sess,
            start_date[:10],
            end_date[:10],
            int(limit),
            "gross",
            include_extras=include_extras,
            headline_only=not include_extras,
        )
    if payload and _bundle_payload_has_display_data(payload):
        return payload
    return None


def build_and_store_artifact(
    sess,
    start_date: str,
    end_date: str,
    kind: Literal["hot", "deep"],
    *,
    limit: int = 10,
    include_extras: bool = False,
) -> str | None:
    """Build from Tier-3 (offline pipeline) and persist artifact. Returns version or None."""
    if kind == KIND_HOT:
        payload = _build_hot_payload(sess, start_date, end_date, limit)
    else:
        payload = _build_deep_payload(sess, start_date, end_date, limit, include_extras=include_extras)
    if not payload:
        return None
    return save_artifact(start_date, end_date, kind, payload)


def _artifact_build_worker(
    start_date: str,
    end_date: str,
    kind: str,
    *,
    limit: int,
    include_extras: bool,
) -> None:
    key = (start_date[:10], end_date[:10], kind)
    try:
        import backend.main as _main
        from ..session import AppSession

        if not _main._warm_cache_ready.wait(timeout=120.0):
            _log.warning("artifact build: warm cache not ready %s..%s %s", start_date, end_date, kind)
            return
        sess = AppSession()
        try:
            _main.try_attach_shared_frames_fast(sess)
        except Exception:
            pass
        ver = build_and_store_artifact(
            sess,
            start_date,
            end_date,
            kind,  # type: ignore[arg-type]
            limit=limit,
            include_extras=include_extras,
        )
        if ver:
            _log.info(
                "intelligence artifact built %s..%s kind=%s version=%s",
                start_date,
                end_date,
                kind,
                ver,
            )
    except Exception:
        _log.exception("artifact build failed %s..%s %s", start_date, end_date, kind)
    finally:
        with _BUILD_LOCK:
            _BUILD_QUEUED.discard(key)


def schedule_artifact_build(
    start_date: str,
    end_date: str,
    kind: Literal["hot", "deep"] = KIND_HOT,
    *,
    limit: int = 10,
    include_extras: bool = False,
) -> bool:
    """Queue proactive artifact build (upload / stale serve / startup)."""
    key = (start_date[:10], end_date[:10], kind)
    with _BUILD_LOCK:
        if key in _BUILD_QUEUED:
            return False
        _BUILD_QUEUED.add(key)
    try:
        from ..concurrency import HEAVY_EXECUTOR

        HEAVY_EXECUTOR.submit(
            _artifact_build_worker,
            start_date[:10],
            end_date[:10],
            kind,
            limit=int(limit),
            include_extras=bool(include_extras),
        )
        return True
    except Exception:
        with _BUILD_LOCK:
            _BUILD_QUEUED.discard(key)
        return False


def prebuild_day_artifacts(sess, *, lookback_days: int | None = None) -> None:
    """Prebuild per-day parquet drill-down artifacts for the last N calendar days."""
    n = int(lookback_days if lookback_days is not None else PER_DAY_ARTIFACT_LOOKBACK)
    today = datetime.now(IST).date()
    for offset in range(n):
        day = (today - timedelta(days=offset)).isoformat()
        from .intelligence_artifact_store import day_parquet_path, read_day_parquet

        if os.path.isfile(day_parquet_path(day)):
            continue
        if read_day_parquet(day):
            continue
        build_and_store_artifact(sess, day, day, KIND_HOT, include_extras=False)


def _migrate_legacy_deep_json_to_parquet() -> None:
    """One-time style migration: old full-json deep artifacts → parquet + slim JSON."""
    from .intelligence_artifact_store import deep_parquet_path, write_deep_parquet

    root = _artifact_root()
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        if not name.endswith("_deep.json"):
            continue
        path = os.path.join(root, name)
        try:
            with open(path, encoding="utf-8") as f:
                disk = json.load(f)
        except Exception:
            continue
        if disk.get("storage") == "parquet":
            continue
        payload = disk.get("payload")
        if not isinstance(payload, dict) or not payload.get("platform_summary"):
            continue
        s = str(disk.get("start_date") or "")[:10]
        e = str(disk.get("end_date") or "")[:10]
        if len(s) != 10 or len(e) != 10:
            continue
        if os.path.isfile(deep_parquet_path(s, e)):
            continue
        if not write_deep_parquet(s, e, payload):
            continue
        slim = {
            "version": disk.get("version"),
            "built_at": disk.get("built_at") or datetime.now(IST).isoformat(),
            "kind": KIND_DEEP,
            "start_date": s,
            "end_date": e,
            "storage": "parquet",
            "parquet": os.path.basename(deep_parquet_path(s, e)),
            "payload": {
                "source": payload.get("source", "tier3_sqlite"),
                "data_completeness": payload.get("data_completeness", "full"),
                "sales_summary": payload.get("sales_summary") or {},
            },
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(slim, f, default=str)
        os.replace(tmp, path)
        _log.info("migrated deep artifact to parquet %s..%s", s, e)


def prebuild_standard_artifacts(sess=None) -> None:
    """Build hot + deep artifacts for 7D / 30D / 90D windows (deploy + post-upload)."""
    if sess is None:
        from ..session import AppSession

        sess = AppSession()
        try:
            import backend.main as _main

            _main.try_attach_shared_frames_fast(sess)
        except Exception:
            pass
    try:
        _migrate_legacy_deep_json_to_parquet()
    except Exception:
        _log.exception("migrate legacy deep json failed")
    for start, end in standard_intelligence_windows():
        for kind, extras in ((KIND_HOT, False), (KIND_DEEP, True)):
            current = intelligence_version_for_window(start, end)
            existing, meta = load_artifact(start, end, kind, allow_stale=False)
            if existing and not meta.get("stale") and meta.get("version") == current:
                continue
            build_and_store_artifact(sess, start, end, kind, include_extras=extras)
    try:
        prebuild_day_artifacts(sess)
    except Exception:
        _log.exception("prebuild_day_artifacts failed")


def load_hot_summary_for_request(
    sess,
    start_date: str,
    end_date: str,
    limit: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """
    Hot path for ``/dashboard/summary`` — artifact first, stale-while-revalidate.
    """
    payload, meta = load_artifact(start_date, end_date, KIND_HOT, allow_stale=True)
    if payload:
        if meta.get("stale"):
            schedule_artifact_build(start_date, end_date, KIND_HOT, limit=limit)
        return payload, meta
    return None, meta


def load_day_drilldown(day: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Single-day drill-down from per-day parquet artifact."""
    from .intelligence_artifact_store import read_day_parquet

    iso = day[:10]
    payload = read_day_parquet(iso)
    meta = {"source": "day_parquet" if payload else "none", "date": iso}
    return payload, meta


def load_deep_bundle_for_request(
    sess,
    start_date: str,
    end_date: str,
    *,
    limit: int = 10,
    include_extras: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Deep path for ``/intelligence-bundle`` when extras/full analytics requested."""
    kind = KIND_DEEP if include_extras else KIND_HOT
    payload, meta = load_artifact(start_date, end_date, kind, allow_stale=True)
    if payload:
        if meta.get("stale"):
            schedule_artifact_build(
                start_date,
                end_date,
                kind,
                limit=limit,
                include_extras=include_extras,
            )
        if kind == KIND_HOT and "platform_summary" in payload:
            bundle = {
                "status": "ready",
                "data_completeness": payload.get("data_completeness", "partial"),
                "sales_summary": payload.get("sales_summary") or {},
                "platform_summary": payload.get("platform_summary") or [],
                "top_skus": payload.get("top_skus") or [],
                "anomalies": [],
                "dsr_brand_monthly": {"rows": [], "totals": {}, "note": ""},
                "tier3_auto_pull": True,
                "artifact_source": meta.get("source"),
            }
            return bundle, meta
        if kind == KIND_DEEP:
            return payload, meta
    return None, meta
