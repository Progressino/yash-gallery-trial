"""
PO SKU Replacement — allows correcting wrong/old SKU codes on uploaded PO sheets.

Unlike the main sku_mapping (which maps marketplace seller SKUs to OMS keys),
this replacement map is applied BEFORE the standard PO normalization so that
a fully incorrect PO SKU can be swapped for the correct ERP SKU.

Stored at: warm_cache/po_sku_replacement.json
Format:  {"WRONG-OLD-SKU": "CORRECT-ERP-SKU", ...}
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

_CACHE: Optional[Dict[str, str]] = None


def _replacement_disk_path() -> Path:
    base = os.environ.get("WARM_CACHE_DIR", "/data/warm_cache")
    return Path(base) / "po_sku_replacement.json"


def load_po_sku_replacement() -> Dict[str, str]:
    """Load PO SKU replacement map (disk → in-memory cache)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    p = _replacement_disk_path()
    if not p.is_file():
        _CACHE = {}
        return _CACHE
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        _CACHE = {str(k).strip().upper(): str(v).strip().upper()
                  for k, v in (raw if isinstance(raw, dict) else {}).items()
                  if k and v and str(k).strip() and str(v).strip()}
    except Exception:
        _CACHE = {}
    return _CACHE


def _clear_cache() -> None:
    global _CACHE
    _CACHE = None


def _persist(mapping: Dict[str, str]) -> None:
    """Write to disk and update in-memory cache."""
    global _CACHE
    p = _replacement_disk_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mapping, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _CACHE = dict(mapping)


def list_po_sku_replacements() -> List[Dict[str, str]]:
    """Return all replacements as a list of {old_sku, new_sku} dicts."""
    m = load_po_sku_replacement()
    return [
        {"old_sku": k, "new_sku": v}
        for k, v in sorted(m.items())
    ]


def add_or_update_replacement(old_sku: str, new_sku: str) -> Dict[str, str]:
    """Add or overwrite one replacement. Returns updated list."""
    old_k = str(old_sku or "").strip().upper()
    new_v = str(new_sku or "").strip().upper()
    if not old_k or not new_v:
        raise ValueError("Both old_sku and new_sku must be non-empty")
    if old_k == new_v:
        raise ValueError("old_sku and new_sku cannot be the same")
    m = dict(load_po_sku_replacement())
    m[old_k] = new_v
    _persist(m)
    return {"old_sku": old_k, "new_sku": new_v}


def remove_replacement(old_sku: str) -> bool:
    """Remove one replacement by its old_sku. Returns True if it existed."""
    k = str(old_sku or "").strip().upper()
    m = dict(load_po_sku_replacement())
    if k not in m:
        return False
    del m[k]
    _persist(m)
    return True


def apply_po_sku_replacement(raw_sku: str) -> str:
    """Return the corrected SKU if a replacement exists, else the original."""
    if not raw_sku:
        return raw_sku
    key = str(raw_sku).strip().upper()
    m = load_po_sku_replacement()
    return m.get(key, raw_sku)
