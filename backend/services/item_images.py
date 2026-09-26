"""
Item / Style product images — disk store + JPEG optimize ≤100KB.

Files live under ITEM_MEDIA_DIR (default /data/item_media; local ./item_media_dev).
Canonical map is items.image_path; size variants inherit from parent when empty.
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

from ..db import item_db
from .helpers import get_parent_sku

MAX_IMAGE_BYTES = 100 * 1024  # 100KB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_SAFE_CODE_RE = re.compile(r"[^A-Za-z0-9._\-]+")


def media_dir() -> Path:
    raw = (os.environ.get("ITEM_MEDIA_DIR") or "").strip()
    if raw:
        p = Path(raw)
    else:
        # Prefer /data when writable (prod volume); else local fallback.
        data = Path("/data/item_media")
        p = data if data.parent.exists() and os.access(str(data.parent), os.W_OK) else Path("./item_media_dev")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _safe_filename_stem(item_code: str) -> str:
    stem = _SAFE_CODE_RE.sub("_", str(item_code or "").strip())
    return stem[:180] or "item"


def optimize_image_bytes(raw: bytes, *, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """Compress to RGB JPEG ≤ max_bytes (or smallest achievable)."""
    if not raw:
        raise ValueError("Empty image data")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise ValueError(f"Invalid image: {exc}") from exc

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in (img.info or {})):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")

    max_sides = (1200, 1000, 800, 640, 512, 400, 320, 240)
    qualities = (85, 75, 65, 55, 45, 35, 28, 22)
    best = b""

    for side in max_sides:
        work = img
        w, h = work.size
        longest = max(w, h)
        if longest > side:
            scale = side / float(longest)
            work = work.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        for q in qualities:
            buf = io.BytesIO()
            work.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
            data = buf.getvalue()
            if not best or len(data) < len(best):
                best = data
            if len(data) <= max_bytes:
                return data

    if not best:
        raise ValueError("Could not encode image")
    return best


def absolute_path(rel_or_name: str) -> Optional[Path]:
    name = str(rel_or_name or "").strip()
    if not name:
        return None
    p = Path(name)
    if p.is_absolute():
        return p if p.is_file() else None
    full = media_dir() / name
    return full if full.is_file() else None


def item_has_image_file(item: dict | None) -> bool:
    if not item:
        return False
    return absolute_path(str(item.get("image_path") or "")) is not None


def enrich_item_image_fields(item: dict | None) -> dict | None:
    if not item:
        return item
    code = str(item.get("item_code") or "").strip()
    has = item_has_image_file(item)
    if not has:
        pid = item.get("parent_id")
        if pid:
            try:
                from ..db.item_db import _connect

                conn = _connect()
                row = conn.execute(
                    "SELECT image_path FROM items WHERE id = ?", (int(pid),)
                ).fetchone()
                conn.close()
                if row:
                    has = absolute_path(str(row["image_path"] or "")) is not None
            except Exception:
                pass
    item["has_image"] = bool(has)
    item["image_url"] = f"/api/items/by-code/{code}/image" if code and has else ""
    return item


def enrich_item_image_fields_bulk(items: list[dict]) -> list[dict]:
    """``enrich_item_image_fields`` for list rows carrying ``_parent_image_path``.

    Size variants share the parent image, so each file is checked once and no
    per-row parent query is needed.
    """
    exists: dict[str, bool] = {}

    def _has(path) -> bool:
        key = str(path or "").strip()
        if not key:
            return False
        if key not in exists:
            exists[key] = absolute_path(key) is not None
        return exists[key]

    for item in items:
        parent_path = item.pop("_parent_image_path", None)
        code = str(item.get("item_code") or "").strip()
        has = _has(item.get("image_path"))
        if not has and item.get("parent_id"):
            has = _has(parent_path)
        item["has_image"] = bool(has)
        item["image_url"] = f"/api/items/by-code/{code}/image" if code and has else ""
    return items


def resolve_image_file_for_code(item_code: str) -> Optional[Path]:
    """Resolve on-disk image for a code (exact → parent_id → parent SKU)."""
    code = str(item_code or "").strip()
    if not code:
        return None

    seen: set[int] = set()

    def _from_row(row: dict | None) -> Optional[Path]:
        if not row:
            return None
        rid = int(row.get("id") or 0)
        if rid and rid in seen:
            return None
        if rid:
            seen.add(rid)
        path = absolute_path(str(row.get("image_path") or ""))
        if path:
            return path
        pid = row.get("parent_id")
        if pid:
            return _from_row(item_db.get_item(int(pid)))
        return None

    row = item_db.get_item_by_code(code) or item_db.get_item_by_code_ci(code)
    found = _from_row(row)
    if found:
        return found

    try:
        parent_code = str(get_parent_sku(code) or "").strip()
    except Exception:
        parent_code = ""
    if parent_code and parent_code.upper() != code.upper():
        prow = item_db.get_item_by_code(parent_code) or item_db.get_item_by_code_ci(parent_code)
        found = _from_row(prow)
        if found:
            return found
    return None


def save_item_image(item_id: int, raw: bytes) -> dict:
    item = item_db.get_item(int(item_id))
    if not item:
        raise ValueError("Item not found")
    optimized = optimize_image_bytes(raw)
    stem = _safe_filename_stem(item["item_code"])
    filename = f"{stem}.jpg"
    dest = media_dir() / filename
    # Remove previous file if different name
    old = absolute_path(str(item.get("image_path") or ""))
    dest.write_bytes(optimized)
    if old and old.resolve() != dest.resolve() and old.is_file():
        try:
            old.unlink()
        except OSError:
            pass
    now = _now_iso()
    item_db.set_item_image_meta(int(item_id), filename, now)
    return {
        "ok": True,
        "item_id": int(item_id),
        "item_code": item["item_code"],
        "image_path": filename,
        "bytes": len(optimized),
        "image_updated_at": now,
        "image_url": f"/api/items/by-code/{item['item_code']}/image",
    }


def delete_item_image(item_id: int) -> bool:
    item = item_db.get_item(int(item_id))
    if not item:
        return False
    path = absolute_path(str(item.get("image_path") or ""))
    item_db.clear_item_image_meta(int(item_id))
    if path and path.is_file():
        try:
            path.unlink()
        except OSError:
            pass
    return True


def _zip_image_entries(zf: zipfile.ZipFile) -> list[tuple[str, str, bytes]]:
    """Return list of (original_name, item_code_stem, bytes)."""
    out: list[tuple[str, str, bytes]] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename or ""
        base = os.path.basename(name)
        if not base or base.startswith("."):
            continue
        if "__MACOSX" in name.replace("\\", "/").split("/"):
            continue
        ext = Path(base).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        stem = Path(base).stem.strip()
        if not stem:
            continue
        try:
            data = zf.read(info)
        except Exception:
            continue
        if not data:
            continue
        out.append((base, stem, data))
    return out


def bulk_import_zip(zip_bytes: bytes) -> dict:
    if not zip_bytes:
        raise ValueError("Empty ZIP")
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid ZIP file") from exc

    matched: list[dict] = []
    unmatched: list[str] = []
    errors: list[str] = []

    with zf:
        entries = _zip_image_entries(zf)
        if not entries:
            raise ValueError("No image files found in ZIP (jpg/png/webp/gif/bmp)")

        for original, stem, data in entries:
            row = item_db.get_item_by_code(stem) or item_db.get_item_by_code_ci(stem)
            if not row:
                # Try parent SKU of filename (e.g. STYLE-M.jpg → STYLE)
                try:
                    parent = str(get_parent_sku(stem) or "").strip()
                except Exception:
                    parent = ""
                if parent and parent.upper() != stem.upper():
                    row = item_db.get_item_by_code(parent) or item_db.get_item_by_code_ci(parent)
            if not row:
                unmatched.append(original)
                continue
            try:
                result = save_item_image(int(row["id"]), data)
                matched.append(
                    {
                        "file": original,
                        "item_id": result["item_id"],
                        "item_code": result["item_code"],
                        "bytes": result["bytes"],
                    }
                )
            except Exception as exc:
                errors.append(f"{original}: {exc}")

    return {
        "ok": True,
        "matched": matched,
        "unmatched": unmatched,
        "errors": errors,
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "error_count": len(errors),
    }
