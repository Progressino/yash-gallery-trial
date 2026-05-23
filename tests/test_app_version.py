"""Build metadata endpoint."""

from backend.app_version import get_build_info


def test_get_build_info_has_label():
    info = get_build_info()
    assert info["version"]
    assert info["label"]
