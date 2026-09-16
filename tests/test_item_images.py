"""Item / Style product images — optimize, upload, ZIP bulk, parent inheritance."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from backend.db import item_db
from backend.services import item_images


@pytest.fixture()
def items(tmp_path, monkeypatch):
    db_path = str(tmp_path / "items.db")
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setenv("ITEM_DB_PATH", db_path)
    monkeypatch.setenv("ITEM_MEDIA_DIR", str(media))
    monkeypatch.setattr(item_db, "DB_PATH", db_path)
    item_db.init_db()
    return item_db, media


def _png_bytes(w=400, h=400, color=(30, 120, 200)) -> bytes:
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _type_id(db):
    types = db.list_item_types()
    return int(types[0]["id"])


def test_optimize_image_under_100kb(items):
    raw = _png_bytes(1600, 1200)
    assert len(raw) > 100 * 1024 or True  # PNG may or may not be huge; force large
    big = _png_bytes(2000, 2000)
    out = item_images.optimize_image_bytes(big)
    assert len(out) <= item_images.MAX_IMAGE_BYTES
    assert out[:2] == b"\xff\xd8"  # JPEG SOI


def test_save_and_resolve_image(items):
    db, media = items
    tid = _type_id(db)
    iid = db.create_item("STYLE-IMG-1", "Style One", tid)
    result = item_images.save_item_image(iid, _png_bytes(800, 800))
    assert result["ok"] is True
    assert result["bytes"] <= item_images.MAX_IMAGE_BYTES
    path = item_images.resolve_image_file_for_code("STYLE-IMG-1")
    assert path is not None and path.is_file()
    item = db.get_item(iid)
    assert item["has_image"] is True
    assert (media / result["image_path"]).is_file()


def test_size_variant_inherits_parent_image(items):
    db, _ = items
    tid = _type_id(db)
    parent = db.create_item("PARENTIMG", "Parent", tid)
    item_images.save_item_image(parent, _png_bytes(600, 600))
    vids = db.create_size_variants(parent, ["M", "L"])
    assert vids
    variant = db.get_item(vids[0])
    assert variant["item_code"].endswith("-M") or "-L" in variant["item_code"]
    path = item_images.resolve_image_file_for_code(variant["item_code"])
    assert path is not None
    enriched = db.get_item(vids[0])
    assert enriched["has_image"] is True


def test_bulk_zip_match_and_unmatched(items):
    db, _ = items
    tid = _type_id(db)
    db.create_item("ZIPMATCH", "Zip Match", tid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ZIPMATCH.jpg", _png_bytes(500, 500))
        zf.writestr("UNKNOWNCODE.png", _png_bytes(200, 200))
        zf.writestr("__MACOSX/._ZIPMATCH.jpg", b"junk")
        zf.writestr("readme.txt", b"ignore me")
    result = item_images.bulk_import_zip(buf.getvalue())
    assert result["matched_count"] == 1
    assert result["matched"][0]["item_code"] == "ZIPMATCH"
    assert "UNKNOWNCODE.png" in result["unmatched"]
    assert item_images.resolve_image_file_for_code("ZIPMATCH") is not None


def test_delete_item_image(items):
    db, media = items
    tid = _type_id(db)
    iid = db.create_item("DELIMG", "Delete Me", tid)
    saved = item_images.save_item_image(iid, _png_bytes(300, 300))
    assert (media / saved["image_path"]).is_file()
    assert item_images.delete_item_image(iid) is True
    assert item_images.resolve_image_file_for_code("DELIMG") is None
    item = db.get_item(iid)
    assert not (item.get("image_path") or "").strip()
