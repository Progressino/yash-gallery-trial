"""Shared Yash Gallery brand HTML for server-rendered printable documents."""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

YASH_GALLERY = {
    "name": "Yash Gallery Pvt. Ltd.",
    "short_name": "Yash Gallery",
    "address": "Bhiwandi, Thane District, Maharashtra — 421302, India",
    "email": "purchase@yashgallery.com",
}

_LOGO_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "static" / "logo.png",
    Path(__file__).resolve().parents[2] / "frontend" / "public" / "logo.png",
    Path("/app/frontend/public/logo.png"),
    Path("/root/app/frontend/public/logo.png"),
    Path("/root/app/backend/static/logo.png"),
)


@lru_cache(maxsize=1)
def logo_data_url() -> str:
    for path in _LOGO_CANDIDATES:
        try:
            if path.is_file():
                raw = path.read_bytes()
                b64 = base64.b64encode(raw).decode("ascii")
                return f"data:image/png;base64,{b64}"
        except OSError:
            continue
    return "/logo.png"


def brand_header_html(
    *,
    doc_title: str,
    doc_number: str = "",
    department: str = "",
    barcode_html: str = "",
    extra_left: str = "",
) -> str:
    logo = logo_data_url()
    addr = YASH_GALLERY["address"]
    email = YASH_GALLERY.get("email") or ""
    dept = f'<div style="margin-top:4px;font-size:10px;font-weight:600;color:#002B5B">{department}</div>' if department else ""
    email_html = f'<div style="font-size:10px;color:#475569">{email}</div>' if email else ""
    num = f'<div style="font-size:20px;font-weight:800;color:#002B5B;text-align:right">{doc_number}</div>' if doc_number else ""
    bc = f'<div style="margin-top:8px;display:flex;justify-content:flex-end">{barcode_html}</div>' if barcode_html else ""
    return f"""
    <div class="header" style="display:flex;justify-content:space-between;align-items:flex-start;border-bottom:2px solid #002B5B;padding-bottom:12px;margin-bottom:16px">
      <div style="display:flex;gap:14px;align-items:flex-start">
        <img src="{logo}" alt="Yash Gallery" style="height:56px;width:auto;max-width:160px;object-fit:contain" />
        <div>
          <div style="font-size:18px;font-weight:700;color:#002B5B;line-height:1.2">{YASH_GALLERY["name"]}</div>
          <div style="font-size:10px;color:#475569;line-height:1.5;margin-top:2px">{addr}</div>
          {email_html}
          {dept}
          {extra_left}
        </div>
      </div>
      <div>
        <div style="font-size:16px;font-weight:600;color:#002B5B;text-align:right">{doc_title}</div>
        {num}
        {bc}
      </div>
    </div>"""
