"""Operation-based routing with partial WIP (embroidery child ops + bundle gates).

Supports:
- Per-component routing paths (e.g. Cutting → Embroidery → Cutting → Stitching)
- Partial qty moves (only Front to Embroidery while Back/Sleeves stay in Cutting)
- Stitching blocked until all mandatory panels are back and the bundle is complete
"""
from __future__ import annotations

from typing import Any, Optional


_PROCESS_ALIASES = {
    "CUT": "Cutting",
    "CUTTING": "Cutting",
    "EMB": "Embroidery",
    "EMBROIDERY": "Embroidery",
    "PRINT": "Printing",
    "PRINTING": "Printing",
    "STITCH": "Stitching",
    "STITCHING": "Stitching",
    "FINISH": "Finishing",
    "FINISHING": "Finishing",
    "PACK": "Packing",
    "PACKING": "Packing",
}


def normalize_process_name(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    key = s.upper().replace(" ", "")
    return _PROCESS_ALIASES.get(key) or s.strip().title() if s.islower() else s


def parse_routing_path(raw: str | list | None) -> list[str]:
    """Parse ``Cutting>Embroidery>Cutting>Stitching`` or a list into process names."""
    if raw is None:
        return []
    if isinstance(raw, list):
        parts = [normalize_process_name(p) for p in raw]
        return [p for p in parts if p]
    text = str(raw).strip()
    if not text:
        return []
    for sep in (">", "→", "->", "|", ",", ";"):
        if sep in text:
            parts = [normalize_process_name(p) for p in text.split(sep)]
            return [p for p in parts if p]
    one = normalize_process_name(text)
    return [one] if one else []


def routing_path_to_string(path: list[str]) -> str:
    return ">".join(path)


def resolve_component_routing(
    *,
    routing: str | list | None = None,
    default_next_process: str | None = None,
    item_routing: list[str] | None = None,
) -> list[str]:
    """Prefer explicit component routing, else Cutting → default_next → …, else item routing."""
    path = parse_routing_path(routing)
    if path:
        return path
    nxt = normalize_process_name(default_next_process or "")
    base = [normalize_process_name(p) for p in (item_routing or []) if normalize_process_name(p)]
    if not base:
        base = ["Cutting", "Stitching", "Finishing"]
    if "Cutting" not in base:
        base = ["Cutting", *base]
    if nxt and nxt not in ("Cutting",):
        # Insert default_next right after Cutting when not already present.
        out = ["Cutting"]
        if nxt not in out:
            out.append(nxt)
        for p in base:
            if p == "Cutting":
                continue
            if p not in out:
                out.append(p)
        return out
    return base


def next_process_in_path(
    path: list[str],
    current_process: str,
    *,
    after_process: str | None = None,
) -> Optional[str]:
    """Next hop in a path. Supports revisiting Cutting after Embroidery.

    When ``after_process`` is set (e.g. ``Embroidery``), prefer the first ``current``
    hop that appears *after* that process so Cutting → Stitching works on return.
    """
    if not path:
        return None
    cur = normalize_process_name(current_process)
    after = normalize_process_name(after_process or "")
    if after:
        seen_after = False
        for i, step in enumerate(path):
            if not seen_after:
                if step == after:
                    seen_after = True
                continue
            if step == cur and i + 1 < len(path):
                return path[i + 1]
    for i, step in enumerate(path):
        if step == cur and i + 1 < len(path):
            return path[i + 1]
    if cur == "Cutting" and len(path) > 1:
        return path[1]
    return None


def embroidery_is_child_of_cutting(path: list[str]) -> bool:
    """True when Embroidery appears between Cutting hops (temporary child op)."""
    if "Embroidery" not in path or "Cutting" not in path:
        return False
    emb_idx = path.index("Embroidery")
    return any(i < emb_idx and p == "Cutting" for i, p in enumerate(path)) and any(
        i > emb_idx and p == "Cutting" for i, p in enumerate(path)
    )


def panel_wip_status(
    *,
    location: str,
    available_qty: int,
    path: list[str],
    bundle_ready: bool,
) -> str:
    loc = normalize_process_name(location)
    qty = max(int(available_qty or 0), 0)
    if qty <= 0:
        if "Embroidery" in path and embroidery_is_child_of_cutting(path):
            return "Awaiting Cut / Issue"
        return "Not Started"
    if loc == "Embroidery":
        return "In Process"
    if loc == "Stitching":
        return "In Stitching"
    if loc == "Cutting":
        if embroidery_is_child_of_cutting(path) and not bundle_ready:
            return "Waiting for Bundle"
        if bundle_ready:
            return "Bundle Ready"
        return "In Cutting WIP"
    if loc in ("Finishing", "Packing"):
        return loc
    return f"At {loc}" if loc else "Unknown"


def compute_bundle_readiness(
    components: list[dict[str, Any]],
    *,
    gate_process: str = "Cutting",
) -> dict[str, Any]:
    """
    Bundle is complete when every mandatory component has enough stock at ``gate_process``.

    Each component dict:
      component_code, qty_per_set, available_at_gate, location (optional),
      embroidery_outstanding (qty still at Embroidery), routing (optional)
    """
    gate = normalize_process_name(gate_process) or "Cutting"
    if not components:
        return {
            "bundle_complete": False,
            "complete_sets": 0,
            "gate_process": gate,
            "components": [],
            "blockers": ["No components defined"],
            "message": "No components defined",
        }

    floors: list[int] = []
    rows_out: list[dict[str, Any]] = []
    blockers: list[str] = []
    for row in components:
        code = str(row.get("component_code") or "").strip().upper()
        ratio = max(int(row.get("qty_per_set") or 1), 1)
        avail = max(int(row.get("available_at_gate") or 0), 0)
        emb_out = max(int(row.get("embroidery_outstanding") or 0), 0)
        floors.append(avail // ratio)
        ready = avail >= ratio and emb_out == 0
        status = panel_wip_status(
            location=str(row.get("location") or gate),
            available_qty=avail if emb_out == 0 else emb_out,
            path=parse_routing_path(row.get("routing")),
            bundle_ready=False,
        )
        if emb_out > 0:
            status = "In Process"
            blockers.append(f"{code}: {emb_out} still under Embroidery")
        elif avail < ratio:
            blockers.append(f"{code}: need {ratio} at {gate}, have {avail}")
        rows_out.append(
            {
                "component_code": code,
                "component_name": row.get("component_name") or code,
                "component_sku": row.get("component_sku"),
                "qty_per_set": ratio,
                "available_at_gate": avail,
                "embroidery_outstanding": emb_out,
                "location": row.get("location") or (gate if avail else "Embroidery" if emb_out else ""),
                "status": status,
                "ready": ready,
                "routing": routing_path_to_string(parse_routing_path(row.get("routing"))),
            }
        )

    complete = int(min(floors)) if floors else 0
    # Embroidery outstanding blocks even if Cutting has leftover panels.
    if any(int(r.get("embroidery_outstanding") or 0) > 0 for r in rows_out):
        complete = 0
    bundle_complete = complete > 0 and not any(
        int(r.get("embroidery_outstanding") or 0) > 0 for r in rows_out
    )
    for r in rows_out:
        r["status"] = panel_wip_status(
            location=str(r.get("location") or gate),
            available_qty=int(r.get("available_at_gate") or 0)
            if int(r.get("embroidery_outstanding") or 0) == 0
            else int(r.get("embroidery_outstanding") or 0),
            path=parse_routing_path(r.get("routing")),
            bundle_ready=bundle_complete,
        )
    return {
        "bundle_complete": bundle_complete,
        "complete_sets": complete,
        "gate_process": gate,
        "components": rows_out,
        "blockers": blockers if not bundle_complete else [],
        "message": (
            f"{complete} complete bundle(s) ready at {gate}"
            if bundle_complete
            else ("Bundle incomplete — " + "; ".join(blockers[:3]) if blockers else "Bundle incomplete")
        ),
    }


EMBROIDERY_PARTIAL_ROUTING = "Cutting>Embroidery>Cutting>Stitching"


def embroidery_timing_label(*, before_cutting: bool) -> str:
    """Human label for Set BOM / panel WIP."""
    return "Before cutting (fabric)" if before_cutting else "After cutting (panel)"


def embroidery_issue_label(
    *,
    issue_from: str,
    issue_to: str,
    before_cutting: bool,
) -> str:
    """Action label when issuing from Cutting to Embroidery."""
    to_p = normalize_process_name(issue_to)
    from_p = normalize_process_name(issue_from)
    if to_p != "Embroidery":
        return to_p or str(issue_to or "")
    if from_p == "Cutting":
        return "Embroidery (fabric)" if before_cutting else "Embroidery (panel)"
    return "Embroidery"


def normalize_embroidery_line_fields(line: dict[str, Any] | None) -> dict[str, Any]:
    """Ensure routing + flags stay aligned for embroidery scenarios."""
    ln = dict(line or {})
    routing = str(ln.get("routing") or "").strip()
    requires = bool(ln.get("requires_embroidery")) or "Embroidery" in routing
    before = bool(ln.get("embroidery_before_cutting"))
    if requires and not routing:
        routing = EMBROIDERY_PARTIAL_ROUTING
    if requires and "Embroidery" not in routing:
        routing = EMBROIDERY_PARTIAL_ROUTING
    if requires and not str(ln.get("default_next_process") or "").strip():
        ln["default_next_process"] = "Embroidery"
    ln["requires_embroidery"] = requires
    ln["embroidery_before_cutting"] = before if requires else False
    ln["routing"] = routing
    return ln
