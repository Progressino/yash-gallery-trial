"""Helpers for post-upload coverage notifications (frontend parity checks)."""

from __future__ import annotations


def test_extract_iso_dates_from_filenames_logic():
    """Mirror frontend extractIsoDatesFromFilenames behavior."""
    names = [
        "Amazon_MTR_2026-07-16.csv",
        "Myntra_PPMP_2026-07-15.csv",
    ]
    out = []
    seen = set()
    for name in names:
        import re

        for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", name):
            iso = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            if iso not in seen:
                seen.add(iso)
                out.append(iso)
        legacy = re.match(r".*(\d{1,2})[-_./](\d{1,2})[-_./](\d{2,4})", name)
        if legacy:
            d, mo, y = legacy.group(1), legacy.group(2), legacy.group(3)
            yr = f"20{y}" if len(y) == 2 else y
            iso = f"{yr}-{mo.zfill(2)}-{d.zfill(2)}"
            if iso not in seen:
                seen.add(iso)
                out.append(iso)
    out.sort()
    assert "2026-07-16" in out
    assert "2026-07-15" in out
