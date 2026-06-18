import os

import pytest


def test_local_dev_mode_from_env(monkeypatch):
    monkeypatch.setenv("LOCAL_DEV", "1")
    monkeypatch.delenv("WARM_CACHE_DIR", raising=False)
    from backend.local_dev import local_dev_mode

    assert local_dev_mode() is True


def test_apply_local_dev_defaults_sets_po_session_only(monkeypatch):
    monkeypatch.setenv("LOCAL_DEV", "1")
    monkeypatch.delenv("WARM_CACHE_PO_SESSION_ONLY", raising=False)
    from backend import local_dev

    local_dev.apply_local_dev_defaults()
    assert os.environ.get("WARM_CACHE_PO_SESSION_ONLY") == "1"
