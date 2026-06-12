"""Returns RAR import returns immediately; parse + net sales rebuild run in background."""

import time
from io import BytesIO
from unittest.mock import MagicMock

import pandas as pd


def test_returns_import_accepts_rar_and_queues_worker(client, auth_token, monkeypatch):
    submit_mock = MagicMock()
    monkeypatch.setattr("backend.concurrency.RETURNS_IMPORT_EXECUTOR.submit", submit_mock)

    r = client.post(
        "/api/po/returns/import-file",
        files=[("file", ("Return Data.rar", BytesIO(b"Rar!" + b"\x1a\x07\x00" + b"x"), "application/x-rar"))],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body.get("returns_import") == "running"
    assert body.get("sales_rebuild") == "pending"
    assert body.get("status") == "running"
    assert "background" in (body.get("message") or "").lower()
    assert submit_mock.call_count == 1
    assert submit_mock.call_args[0][0].__name__ == "_run_returns_import_worker"


def test_returns_import_worker_persists_overlay(monkeypatch, tmp_path):
    from backend.session import AppSession, store

    overlay = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"],
            "Return_Platform": ["amazon"],
            "Return_Date": ["2026-06-01"],
            "Return_Units": [2],
        }
    )
    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path / "warm"))
    monkeypatch.setattr(
        "backend.services.po_return_import.parse_return_upload_bytes",
        lambda *a, **k: (overlay, None),
    )
    def _apply(sess, df, **k):
        sess.po_return_overlay_df = df.copy()
        return {
            "ok": True,
            "message": "Return sheet: 1 SKU(s), 2 return units.",
            "skus": 1,
            "total_units": 2,
        }

    monkeypatch.setattr(
        "backend.services.po_return_import.apply_return_overlay_import",
        _apply,
    )
    import backend.main as main_mod

    monkeypatch.setattr(main_mod, "_DISK_CACHE_DIR", str(tmp_path / "warm"))
    followup: list[str] = []
    monkeypatch.setattr(
        "backend.routers.upload._run_returns_import_followup",
        lambda sid: followup.append(sid),
    )
    monkeypatch.setattr(
        "backend.db.forecast_session_pg.pg_session_persist_enabled",
        lambda: False,
    )

    sid, sess = store.get_or_create(None)
    sess.sku_mapping = {"SKU-A": "SKU-A"}
    store._sessions[sid] = sess

    from backend.routers.po import _run_returns_import_worker

    _run_returns_import_worker(
        sid,
        b"raw",
        "Return Data.rar",
        group_by_parent=False,
        replace=True,
        sku_mapping=sess.sku_mapping,
    )
    assert sess.returns_import_status == "done"
    assert not sess.po_return_overlay_df.empty
    assert (tmp_path / "warm" / "po_return_overlay_df.parquet").is_file()
    assert followup == [sid]


def test_light_coverage_restores_return_overlay_from_disk(client, monkeypatch, tmp_path):
    import pandas as pd

    from backend.session import store

    import backend.main as main_mod

    warm = tmp_path / "warm"
    warm.mkdir(parents=True)
    monkeypatch.setenv("WARM_CACHE_DIR", str(warm))
    monkeypatch.setattr(main_mod, "_DISK_CACHE_DIR", str(warm))
    main_mod._warm_cache = {}
    df = pd.DataFrame(
        {
            "OMS_SKU": ["1001YKBEIGE-M"],
            "Return_Platform": ["meesho"],
            "Return_Date": ["2026-06-01"],
            "Return_Units": [3],
        }
    )
    df.to_parquet(warm / "po_return_overlay_df.parquet", index=False)

    sid, sess = store.get_or_create(None)
    sess.po_return_overlay_df = pd.DataFrame()
    store._sessions[sid] = sess

    c2 = client.__class__(client.app)
    c2.cookies.set("auth_token", "test-token")
    c2.cookies.set("session_id", sid)

    r = c2.get("/api/data/coverage?light=1")
    assert r.status_code == 200
    body = r.json()
    assert body.get("return_sheet") is True
    assert body.get("return_sheet_skus") == 1
    assert body.get("return_sheet_units") == 3
    assert body.get("return_overlay_uploaded_at")


def test_apply_return_overlay_import_persists_upload_meta(monkeypatch, tmp_path):
    import pandas as pd

    from backend.services.po_return_import import (
        apply_return_overlay_import,
        load_return_overlay_meta_from_disk,
    )
    from backend.session import AppSession

    warm = tmp_path / "warm"
    warm.mkdir(parents=True)
    monkeypatch.setenv("WARM_CACHE_DIR", str(warm))
    import backend.main as main_mod

    monkeypatch.setattr(main_mod, "_DISK_CACHE_DIR", str(warm))
    main_mod._warm_cache = {}

    sess = AppSession()
    overlay = pd.DataFrame(
        {
            "OMS_SKU": ["SKU1", "SKU2"],
            "Return_Platform": ["flipkart", "flipkart"],
            "Return_Date": ["2026-06-01", "2026-06-02"],
            "Return_Units": [5, 7],
        }
    )
    out = apply_return_overlay_import(sess, overlay, replace=True, filename="Return-Data.rar")
    assert out["ok"] is True
    assert out["skus"] == 2
    assert out["total_units"] == 12
    assert sess.return_overlay_filename == "Return-Data.rar"
    assert sess.return_overlay_uploaded_at
    meta = load_return_overlay_meta_from_disk()
    assert meta.get("return_overlay_filename") == "Return-Data.rar"
    assert meta.get("return_overlay_skus") == 2
    assert meta.get("return_overlay_units") == 12


def test_skip_meesho_lost_member():
    from backend.services.po_return_import import _skip_return_archive_member

    assert _skip_return_archive_member(
        "Return Data/completed_lost_____Meesho Lost.csv"
    )
    assert not _skip_return_archive_member(
        "Return Data/completed_delivered_Meesho Return.csv"
    )
