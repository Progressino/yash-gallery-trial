"""Component-level Ready-to-Cut from printed fabric reservations + Set BOM materials.

A style/component is Ready to Cut only when **all** fabrics required for that
component (from Set BOM material lines) are covered by active reservations
for the same SO + main SKU. Other components of the same style that lack fabric
stay off the list and do not block ready components.
"""
from __future__ import annotations

from typing import Any, Optional


def _norm(s: str) -> str:
    return str(s or "").strip().upper()


def _main_sku(sku: str) -> str:
    """Main FG/size SKU for grouping reservations — only strip real Set BOM component suffixes.

    Avoids treating incidental tails (e.g. STYLE-B) as components when no Set BOM defines them.
    """
    s = _norm(sku)
    try:
        from .set_components import parse_component_sku
        from .component_bom import effective_set_bom_for_cutting, set_component_lines

        main, comp = parse_component_sku(s)
        if not main or not comp:
            return s
        bom = effective_set_bom_for_cutting(main)
        codes = {
            str(ln.get("component_code") or "").strip().upper()
            for ln in (set_component_lines(bom) if bom else [])
        }
        if comp in codes:
            return _norm(main)
    except Exception:
        pass
    return s


def _component_sku(main: str, code: str) -> str:
    try:
        from .set_components import component_sku

        return component_sku(main, code)
    except Exception:
        return f"{main}-{code}"


def expand_ready_to_cut_rows(
    reservations: list[dict[str, Any]],
    *,
    jo_planned: Optional[dict[tuple[str, str], float]] = None,
    hide_if_jo: bool = False,
) -> list[dict[str, Any]]:
    """Expand Active reservations into ready rows (component-aware).

    ``jo_planned`` maps (so_number, sku) → planned cutting qty already on JOs.
    When ``hide_if_jo`` is True (Grey UI), drop rows with any JO planned for that sku.
    When False (Production Ready panel), expose ``available_qty`` = reserved − planned.
    """
    jo_planned = jo_planned or {}
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    by_fabric_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in reservations:
        so = str(r.get("so_number") or "").strip()
        sku = str(r.get("sku") or "").strip()
        fab = str(r.get("fabric_code") or "").strip()
        if not so or not sku or not fab:
            continue
        main = _main_sku(sku)
        key = (so, main)
        qty = float(r.get("reserved_qty") or r.get("qty") or 0)
        grouped.setdefault(key, {})
        grouped[key][fab] = grouped[key].get(fab, 0.0) + qty
        by_fabric_meta[(so, main, fab)] = r

    results: list[dict[str, Any]] = []

    def _remaining(so: str, sku: str, reserved: float) -> float:
        planned = float(jo_planned.get((so, sku), 0) or 0)
        return max(0.0, reserved - planned)

    for (so, main), fabric_map in grouped.items():
        try:
            from .component_bom import effective_set_bom_for_cutting, set_component_lines

            bom = effective_set_bom_for_cutting(main)
            comps = set_component_lines(bom) if bom else []
        except Exception:
            comps = []

        if not comps:
            for fab, res_qty in fabric_map.items():
                if res_qty <= 0:
                    continue
                if hide_if_jo and float(jo_planned.get((so, main), 0) or 0) > 0:
                    continue
                rem = _remaining(so, main, res_qty)
                if not hide_if_jo and rem <= 0:
                    continue
                meta = by_fabric_meta.get((so, main, fab)) or {}
                results.append(
                    {
                        "so_number": so,
                        "sku": main,
                        "main_sku": main,
                        "component_code": "",
                        "fabric_code": fab,
                        "fabric_name": (meta.get("fabric_name") or ""),
                        "reserved_qty": res_qty,
                        "available_qty": rem if not hide_if_jo else res_qty,
                        "cut_status": "Ready to Cut",
                        "readiness_scope": "sku",
                        "buyer": meta.get("buyer") or "",
                    }
                )
            continue

        fabric_keys_upper = {_norm(f): f for f in fabric_map.keys()}
        for ln in comps:
            code = str(ln.get("component_code") or "").strip().upper()
            if not code:
                continue
            csku = _component_sku(main, code)
            mats = [
                m
                for m in (ln.get("materials") or [])
                if float(m.get("quantity") or 0) > 0 and str(m.get("material_code") or "").strip()
            ]
            if not mats:
                continue
            missing = []
            covered_fabs: list[str] = []
            for m in mats:
                mc = str(m.get("material_code") or "").strip()
                mu = _norm(mc)
                orig = fabric_keys_upper.get(mu)
                if not orig or fabric_map.get(orig, 0) <= 0:
                    missing.append(mc)
                else:
                    covered_fabs.append(orig)
            if missing:
                continue
            primary = covered_fabs[0]
            meta = by_fabric_meta.get((so, main, primary)) or {}
            res_qty = fabric_map.get(primary, 0.0)
            if hide_if_jo and float(jo_planned.get((so, csku), 0) or 0) > 0:
                continue
            rem_comp = _remaining(so, csku, res_qty)
            rem_main = _remaining(so, main, res_qty)
            rem = rem_comp if rem_comp > 0 else rem_main
            if not hide_if_jo and rem <= 0:
                continue
            results.append(
                {
                    "so_number": so,
                    "sku": csku,
                    "main_sku": main,
                    "component_code": code,
                    "component_name": str(ln.get("component_name") or code),
                    "fabric_code": primary,
                    "fabric_name": meta.get("fabric_name") or "",
                    "reserved_qty": res_qty,
                    "available_qty": rem if not hide_if_jo else res_qty,
                    "cut_status": "Ready to Cut",
                    "readiness_scope": "component",
                    "required_fabrics": [str(m.get("material_code") or "") for m in mats],
                    "buyer": meta.get("buyer") or "",
                }
            )
    return results
