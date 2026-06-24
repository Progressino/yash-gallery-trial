"""
Parquet + CDN storage for Intelligence artifacts.

- Hot path: JSON on disk (fast headline serve).
- Deep path: columnar ``.parquet`` for platform daily series / top SKUs.
- Per-day: ``intelligence_bundle_{date}.parquet`` for single-day drill-downs.

CDN backends (first configured wins for publish; load tries local → CDN):
  1. GitHub Release tag ``intelligence-artifacts`` (default when GITHUB_TOKEN set)
  2. S3 bucket when ``INTELLIGENCE_ARTIFACT_S3_BUCKET`` is set (optional boto3)
"""
from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

import pandas as pd

from .helpers import _coerce_df_for_parquet

_log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

_GH_API = "https://api.github.com"
_ARTIFACT_RELEASE_TAG = os.environ.get("INTELLIGENCE_ARTIFACT_GH_TAG", "intelligence-artifacts")
_ARTIFACT_MANIFEST = "_intelligence_artifact_manifest.json"

_PUBLISH_QUEUED: set[str] = set()
_PUBLISH_LOCK = threading.Lock()

Section = Literal[
    "platform_headline",
    "platform_daily",
    "top_skus",
    "sales_summary",
    "day_platform",
]


def _artifact_root() -> str:
    base = os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")
    path = os.path.join(base, "intelligence", "daily")
    os.makedirs(path, exist_ok=True)
    return path


def _by_date_root() -> str:
    path = os.path.join(_artifact_root(), "by_date")
    os.makedirs(path, exist_ok=True)
    return path


def deep_parquet_path(start_date: str, end_date: str) -> str:
    s, e = start_date[:10], end_date[:10]
    return os.path.join(_artifact_root(), f"intelligence_bundle_{s}_{e}_deep.parquet")


def day_parquet_path(day: str) -> str:
    return os.path.join(_by_date_root(), f"intelligence_bundle_{day[:10]}.parquet")


def _cdn_object_key(filename: str) -> str:
    prefix = (os.environ.get("INTELLIGENCE_ARTIFACT_S3_PREFIX") or "intelligence/daily").strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def _flatten_deep_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Pack deep analytics tables into one long dataframe with a ``section`` column."""
    rows: list[dict[str, Any]] = []

    ss = payload.get("sales_summary") or {}
    if ss:
        rows.append({"section": "sales_summary", **{k: ss.get(k) for k in ss}})

    for p in payload.get("platform_summary") or []:
        plat = p.get("platform") or ""
        headline = {
            "section": "platform_headline",
            "platform": plat,
            "loaded": bool(p.get("loaded")),
            "total_units": int(p.get("total_units") or 0),
            "total_returns": int(p.get("total_returns") or 0),
            "net_units": int(p.get("net_units") or 0),
            "return_rate": float(p.get("return_rate") or 0),
            "top_sku": str(p.get("top_sku") or ""),
            "trend_direction": str(p.get("trend_direction") or ""),
            "trend_direction_net": str(p.get("trend_direction_net") or ""),
        }
        rows.append(headline)
        for d in p.get("daily") or []:
            rows.append(
                {
                    "section": "platform_daily",
                    "platform": plat,
                    "date": str(d.get("date") or "")[:10],
                    "units": int(d.get("units") or d.get("shipped") or 0),
                    "returns": int(d.get("returns") or 0),
                    "net_units": int(d.get("net_units") or 0),
                }
            )
        for m in p.get("monthly") or []:
            rows.append(
                {
                    "section": "platform_monthly",
                    "platform": plat,
                    "month": str(m.get("month") or ""),
                    "units": int(m.get("units") or 0),
                    "returns": int(m.get("returns") or 0),
                }
            )

    for t in payload.get("top_skus") or []:
        rows.append({"section": "top_skus", **{k: t.get(k) for k in t}})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _unflatten_deep_df(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty or "section" not in df.columns:
        return {}

    out: dict[str, Any] = {
        "platform_summary": [],
        "top_skus": [],
        "sales_summary": {},
        "source": "parquet_artifact",
    }
    headlines: dict[str, dict] = {}
    daily_by_plat: dict[str, list] = {}
    monthly_by_plat: dict[str, list] = {}

    for _, row in df.iterrows():
        sec = str(row.get("section") or "")
        if sec == "sales_summary":
            out["sales_summary"] = {
                k: row[k]
                for k in row.index
                if k != "section" and pd.notna(row[k])
            }
        elif sec == "platform_headline":
            plat = str(row.get("platform") or "")
            headlines[plat] = {
                "platform": plat,
                "loaded": bool(row.get("loaded")),
                "total_units": int(row.get("total_units") or 0),
                "total_returns": int(row.get("total_returns") or 0),
                "net_units": int(row.get("net_units") or 0),
                "return_rate": float(row.get("return_rate") or 0),
                "top_sku": str(row.get("top_sku") or ""),
                "trend_direction": str(row.get("trend_direction") or "flat"),
                "trend_direction_net": str(row.get("trend_direction_net") or "flat"),
                "monthly": [],
                "daily": [],
                "by_state": [],
            }
        elif sec == "platform_daily":
            plat = str(row.get("platform") or "")
            daily_by_plat.setdefault(plat, []).append(
                {
                    "date": str(row.get("date") or "")[:10],
                    "units": int(row.get("units") or 0),
                    "returns": int(row.get("returns") or 0),
                    "net_units": int(row.get("net_units") or 0),
                }
            )
        elif sec == "platform_monthly":
            plat = str(row.get("platform") or "")
            monthly_by_plat.setdefault(plat, []).append(
                {
                    "month": str(row.get("month") or ""),
                    "units": int(row.get("units") or 0),
                    "returns": int(row.get("returns") or 0),
                }
            )
        elif sec == "top_skus":
            out["top_skus"].append(
                {k: row[k] for k in row.index if k != "section" and pd.notna(row[k])}
            )

    for plat, h in headlines.items():
        h["daily"] = daily_by_plat.get(plat, [])
        h["monthly"] = monthly_by_plat.get(plat, [])
        out["platform_summary"].append(h)

    return out


def write_deep_parquet(start_date: str, end_date: str, payload: dict[str, Any]) -> bool:
    path = deep_parquet_path(start_date, end_date)
    df = _flatten_deep_payload(payload)
    if df.empty:
        return False
    tmp = f"{path}.tmp"
    try:
        _coerce_df_for_parquet(df).to_parquet(tmp, index=False, engine="pyarrow", compression="snappy")
        os.replace(tmp, path)
        return True
    except Exception:
        _log.exception("write deep parquet failed path=%s", path)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def read_deep_parquet(start_date: str, end_date: str) -> dict[str, Any] | None:
    path = deep_parquet_path(start_date, end_date)
    cdn_key = os.path.basename(path)
    if not os.path.isfile(path):
        if not _maybe_fetch_cdn_file(cdn_key, path):
            return None
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        payload = _unflatten_deep_df(df)
        return payload if payload.get("platform_summary") else None
    except Exception:
        _log.exception("read deep parquet failed path=%s", path)
        return None


def write_day_parquet(day: str, payload: dict[str, Any]) -> bool:
    """Persist single-day drill-down artifact."""
    iso = day[:10]
    path = day_parquet_path(iso)
    rows: list[dict[str, Any]] = []
    for p in payload.get("platform_summary") or []:
        plat = p.get("platform") or ""
        rows.append(
            {
                "section": "day_platform",
                "date": iso,
                "platform": plat,
                "loaded": bool(p.get("loaded")),
                "total_units": int(p.get("total_units") or 0),
                "total_returns": int(p.get("total_returns") or 0),
                "net_units": int(p.get("net_units") or 0),
                "return_rate": float(p.get("return_rate") or 0),
            }
        )
    ss = payload.get("sales_summary") or {}
    if ss:
        rows.append({"section": "sales_summary", "date": iso, **{k: ss.get(k) for k in ss}})
    if not rows:
        return False
    tmp = f"{path}.tmp"
    try:
        _coerce_df_for_parquet(pd.DataFrame(rows)).to_parquet(
            tmp, index=False, engine="pyarrow", compression="snappy"
        )
        os.replace(tmp, path)
        return True
    except Exception:
        _log.exception("write day parquet failed path=%s", path)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def read_day_parquet(day: str) -> dict[str, Any] | None:
    iso = day[:10]
    path = day_parquet_path(iso)
    cdn_key = f"by_date/intelligence_bundle_{iso}.parquet"
    if not os.path.isfile(path):
        if not _maybe_fetch_cdn_file(cdn_key, path):
            return None
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        if df.empty:
            return None
        platform_summary: list[dict] = []
        sales_summary: dict = {}
        for _, row in df.iterrows():
            sec = str(row.get("section") or "")
            if sec == "day_platform":
                platform_summary.append(
                    {
                        "platform": str(row.get("platform") or ""),
                        "loaded": bool(row.get("loaded")),
                        "total_units": int(row.get("total_units") or 0),
                        "total_returns": int(row.get("total_returns") or 0),
                        "net_units": int(row.get("net_units") or 0),
                        "return_rate": float(row.get("return_rate") or 0),
                        "daily": [],
                        "monthly": [],
                        "by_state": [],
                    }
                )
            elif sec == "sales_summary":
                sales_summary = {
                    k: row[k]
                    for k in row.index
                    if k not in ("section", "date") and pd.notna(row[k])
                }
        if not platform_summary:
            return None
        return {
            "source": "day_parquet",
            "date": iso,
            "platform_summary": platform_summary,
            "sales_summary": sales_summary,
        }
    except Exception:
        _log.exception("read day parquet failed path=%s", path)
        return None


# ── CDN (GitHub Release + optional S3) ────────────────────────────────────────


def _gh_repo() -> str | None:
    return (os.environ.get("GITHUB_REPO") or "").strip() or None


def _gh_headers() -> dict[str, str] | None:
    tok = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if not tok:
        return None
    return {
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_or_create_artifact_release() -> tuple[int | None, dict[str, tuple[int, str]], str | None]:
    repo = _gh_repo()
    headers = _gh_headers()
    if not repo or not headers:
        return None, {}, "GitHub not configured"
    try:
        import requests

        r = requests.get(
            f"{_GH_API}/repos/{repo}/releases/tags/{_ARTIFACT_RELEASE_TAG}",
            headers=headers,
            timeout=30,
        )
        if r.status_code == 404:
            cr = requests.post(
                f"{_GH_API}/repos/{repo}/releases",
                headers=headers,
                json={"tag_name": _ARTIFACT_RELEASE_TAG, "name": "Intelligence artifacts", "draft": False},
                timeout=30,
            )
            cr.raise_for_status()
            rel = cr.json()
        else:
            r.raise_for_status()
            rel = r.json()
        assets = {a["name"]: (a["id"], a["browser_download_url"]) for a in rel.get("assets", [])}
        return int(rel["id"]), assets, None
    except Exception as e:
        return None, {}, str(e)


def _gh_upload(filename: str, data: bytes) -> None:
    from .github_cache import _gh_delete_asset_by_name, _gh_upload_asset

    release_id, _, err = _get_or_create_artifact_release()
    if err or not release_id:
        raise RuntimeError(err or "no release")
    _gh_delete_asset_by_name(release_id, filename)
    time.sleep(1)
    _gh_upload_asset(release_id, filename, data)


def _gh_download(filename: str) -> bytes | None:
    import requests

    _, assets, err = _get_or_create_artifact_release()
    if err or filename not in assets:
        return None
    url = assets[filename][1]
    headers = _gh_headers() or {}
    headers["Accept"] = "application/octet-stream"
    r = requests.get(url, headers=headers, timeout=300, stream=True)
    if r.status_code != 200:
        return None
    buf = io.BytesIO()
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            buf.write(chunk)
    return buf.getvalue()


def _s3_configured() -> bool:
    return bool((os.environ.get("INTELLIGENCE_ARTIFACT_S3_BUCKET") or "").strip())


def _s3_upload(filename: str, data: bytes) -> None:
    import boto3

    bucket = os.environ["INTELLIGENCE_ARTIFACT_S3_BUCKET"].strip()
    key = _cdn_object_key(filename)
    client = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType="application/octet-stream",
        CacheControl="public, max-age=300",
    )


def _s3_download(filename: str) -> bytes | None:
    try:
        import boto3
        from botocore.exceptions import ClientError

        bucket = os.environ["INTELLIGENCE_ARTIFACT_S3_BUCKET"].strip()
        key = _cdn_object_key(filename)
        client = boto3.client(
            "s3", region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        )
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()
        except ClientError:
            return None
    except Exception:
        return None


def cdn_public_url(filename: str) -> str | None:
    """Return a browser-fetchable URL when CDN is configured."""
    bucket = (os.environ.get("INTELLIGENCE_ARTIFACT_S3_BUCKET") or "").strip()
    if bucket:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        base = (os.environ.get("INTELLIGENCE_ARTIFACT_S3_PUBLIC_BASE") or "").strip()
        if base:
            return f"{base.rstrip('/')}/{_cdn_object_key(filename)}"
        if region == "us-east-1":
            return f"https://{bucket}.s3.amazonaws.com/{_cdn_object_key(filename)}"
        return f"https://{bucket}.s3.{region}.amazonaws.com/{_cdn_object_key(filename)}"
    _, assets, err = _get_or_create_artifact_release()
    if err or filename not in assets:
        return None
    return assets[filename][1]


def _maybe_fetch_cdn_file(filename: str, dest_path: str) -> bool:
    """Download artifact bytes from S3 or GitHub into local warm cache."""
    raw = _s3_download(filename) if _s3_configured() else None
    if raw is None:
        raw = _gh_download(filename)
    if not raw:
        return False
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        tmp = f"{dest_path}.cdn_tmp"
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, dest_path)
        return True
    except Exception:
        _log.exception("cdn fetch write failed %s", dest_path)
        return False


def _read_manifest() -> dict[str, Any]:
    path = os.path.join(_artifact_root(), _ARTIFACT_MANIFEST)
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"files": {}, "updated_at": ""}


def _write_manifest(manifest: dict[str, Any]) -> None:
    path = os.path.join(_artifact_root(), _ARTIFACT_MANIFEST)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def _resolve_local_path(cdn_key: str) -> str | None:
    if cdn_key.startswith("by_date/"):
        path = os.path.join(_artifact_root(), cdn_key)
        return path if os.path.isfile(path) else None
    root_path = os.path.join(_artifact_root(), cdn_key)
    if os.path.isfile(root_path):
        return root_path
    day_path = os.path.join(_by_date_root(), cdn_key)
    return day_path if os.path.isfile(day_path) else None


def publish_file_to_cdn(cdn_key: str) -> bool:
    """Upload one local artifact file to configured CDN backend(s)."""
    local = _resolve_local_path(cdn_key)
    if not local:
        if not cdn_key.startswith("by_date/"):
            alt = _resolve_local_path(f"by_date/{cdn_key}")
            if alt:
                local = alt
                cdn_key = f"by_date/{cdn_key}"
    if not local:
        return False
    try:
        data = open(local, "rb").read()
        published = False
        if _gh_headers():
            try:
                _gh_upload(cdn_key, data)
                published = True
            except Exception:
                _log.exception("github cdn upload failed %s", cdn_key)
        if _s3_configured():
            try:
                _s3_upload(cdn_key, data)
                published = True
            except Exception:
                _log.exception("s3 cdn upload failed %s", cdn_key)
        if published:
            man = _read_manifest()
            man.setdefault("files", {})[cdn_key] = {
                "size": len(data),
                "updated_at": datetime.now(IST).isoformat(),
            }
            man["updated_at"] = datetime.now(IST).isoformat()
            _write_manifest(man)
        return published
    except Exception:
        _log.exception("publish_file_to_cdn failed %s", cdn_key)
        return False


def schedule_cdn_publish(*filenames: str) -> None:
    for fn in filenames:
        if not fn:
            continue
        with _PUBLISH_LOCK:
            if fn in _PUBLISH_QUEUED:
                continue
            _PUBLISH_QUEUED.add(fn)
        try:
            from ..concurrency import HEAVY_EXECUTOR

            HEAVY_EXECUTOR.submit(_cdn_publish_worker, fn)
        except Exception:
            with _PUBLISH_LOCK:
                _PUBLISH_QUEUED.discard(fn)


def _cdn_publish_worker(filename: str) -> None:
    try:
        publish_file_to_cdn(filename)
    finally:
        with _PUBLISH_LOCK:
            _PUBLISH_QUEUED.discard(filename)


def list_cdn_artifacts() -> dict[str, Any]:
    man = _read_manifest()
    files = []
    root = _artifact_root()
    for name in sorted(os.listdir(root)):
        if name.endswith((".parquet", ".json")) and not name.startswith("_"):
            files.append({"name": name, "local": True, "url": cdn_public_url(name)})
    by_date = _by_date_root()
    if os.path.isdir(by_date):
        for name in sorted(os.listdir(by_date)):
            if name.endswith(".parquet"):
                files.append(
                    {
                        "name": f"by_date/{name}",
                        "local": True,
                        "url": cdn_public_url(f"by_date/{name}"),
                    }
                )
    return {"manifest": man, "files": files, "cdn_configured": bool(_gh_headers() or _s3_configured())}
