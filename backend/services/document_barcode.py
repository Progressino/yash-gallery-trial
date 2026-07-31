"""Document barcode / QR payloads for gate scan and print embeds.

Payload format: ``TYPE:NUMBER`` (case-insensitive type), e.g. ``PO:PO-2607-001``,
``JO:JO-0042``, ``GIN:GIN-0007``, ``JWO:…``, ``MIN:…``, ``GP:…``, ``DC:…``, ``GRN:…``.
"""
from __future__ import annotations

import io
import re
from typing import Optional

VALID_TYPES = frozenset({"PO", "JWO", "JO", "GIN", "GRN", "MIN", "GP", "DC"})

_PAYLOAD_RE = re.compile(r"^([A-Za-z]+)\s*[:\-]\s*(.+)$")


def make_payload(doc_type: str, number: str) -> str:
    t = str(doc_type or "").strip().upper()
    n = str(number or "").strip()
    if t not in VALID_TYPES:
        raise ValueError(f"Unsupported document type for barcode: {doc_type}")
    if not n:
        raise ValueError("Document number required for barcode")
    return f"{t}:{n}"


def parse_payload(code: str) -> tuple[str, str]:
    """Return (TYPE, NUMBER). Accepts raw numbers with optional TYPE: prefix."""
    raw = str(code or "").strip()
    if not raw:
        raise ValueError("Empty scan code")
    upper = raw.upper()
    # Bare document numbers with known prefixes (before TYPE-NUMBER hyphen split)
    for prefix, dtype in (
        ("GIN-", "GIN"),
        ("GRN-", "GRN"),
        ("PO-", "PO"),
        ("JWO-", "JWO"),
        ("JO-", "JO"),
        ("PJO-", "JO"),
        ("MIN-", "MIN"),
        ("GP-", "GP"),
        ("DC-", "DC"),
    ):
        if upper.startswith(prefix):
            return dtype, raw.strip()
    m = _PAYLOAD_RE.match(raw)
    if m:
        t = m.group(1).upper()
        n = m.group(2).strip()
        if t in VALID_TYPES and n:
            # If number lost its prefix (JO:0042), restore when possible
            if t == "JO" and not n.upper().startswith(("JO-", "PJO-")):
                n = f"JO-{n}" if not n.upper().startswith("JO") else n
            return t, n
    raise ValueError(
        f"Unrecognized barcode '{raw}'. Expected TYPE:NUMBER "
        f"(PO / JWO / JO / GIN / GRN / MIN / GP / DC)."
    )


def qr_svg_data_url(payload: str, box_size: int = 4, border: int = 1) -> str:
    """Return a data:image/svg+xml URL for a QR code of ``payload``."""
    import base64

    import qrcode
    from qrcode.image.svg import SvgPathImage

    qr = qrcode.QRCode(version=None, box_size=box_size, border=border)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(image_factory=SvgPathImage)
    buf = io.BytesIO()
    img.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def code128_svg_data_url(payload: str) -> Optional[str]:
    """Return Code128 SVG data-URL, or None if library unavailable."""
    import base64

    try:
        from barcode import Code128
        from barcode.writer import SVGWriter
    except Exception:
        return None
    # Code128 has limited charset; use payload as-is (ASCII)
    buf = io.BytesIO()
    try:
        Code128(payload, writer=SVGWriter()).write(
            buf,
            options={"module_height": 10.0, "module_width": 0.3, "quiet_zone": 1.0, "write_text": False},
        )
    except Exception:
        return None
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def barcode_bundle(doc_type: str, number: str) -> dict:
    payload = make_payload(doc_type, number)
    return {
        "payload": payload,
        "qr_data_url": qr_svg_data_url(payload),
        "code128_data_url": code128_svg_data_url(payload),
    }
