"""Print-ready HTML for Stitching Costing report pack (browser Print → Save as PDF)."""
from __future__ import annotations

import html
from typing import Any


_PRINT_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11px; color: #1a1a1a; padding: 20px; }
.header { display: flex; justify-content: space-between; align-items: flex-start;
  border-bottom: 2px solid #002B5B; padding-bottom: 12px; margin-bottom: 16px; }
.company { font-size: 18px; font-weight: 700; color: #002B5B; }
.doc-title { font-size: 14px; font-weight: 600; color: #002B5B; text-align: right; }
.doc-sub { font-size: 10px; color: #64748b; text-align: right; margin-top: 4px; }
.kpi-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }
.kpi { flex: 1 1 140px; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; background: #f8fafc; }
.kpi-label { font-size: 9px; text-transform: uppercase; color: #64748b; font-weight: 600; }
.kpi-value { font-size: 15px; font-weight: 700; color: #002B5B; margin-top: 4px; }
h2 { font-size: 13px; color: #002B5B; margin: 20px 0 8px; page-break-after: avoid; }
h2.first { margin-top: 0; }
.note { font-size: 10px; color: #475569; margin-bottom: 10px; line-height: 1.4; }
table { width: 100%; border-collapse: collapse; margin-bottom: 16px; }
th { background: #002B5B; color: #fff; padding: 6px 8px; text-align: left; font-size: 10px; }
th.r, td.r { text-align: right; }
td { padding: 5px 8px; border-bottom: 1px solid #e2e8f0; font-size: 10px; vertical-align: top; }
tr:nth-child(even) td { background: #f8fafc; }
.loss td { color: #b91c1c; }
.profit td { color: #047857; }
.footer { margin-top: 24px; padding-top: 12px; border-top: 1px solid #e2e8f0;
  font-size: 9px; color: #94a3b8; text-align: center; }
@media print {
  body { padding: 10px; }
  h2 { page-break-before: auto; }
  table { page-break-inside: auto; }
  tr { page-break-inside: avoid; }
}
"""


def _esc(v: Any) -> str:
    if v is None:
        return ""
    return html.escape(str(v))


def _money(v: Any) -> str:
    try:
        n = float(v)
        return f"₹{n:,.2f}"
    except (TypeError, ValueError):
        return _esc(v)


def _table(title: str, rows: list[dict], cols: list[str], *, money_cols: set[str] | None = None) -> str:
    if not rows:
        return f'<h2>{_esc(title)}</h2><p class="note">No data for this period.</p>'
    money_cols = money_cols or set()
    head = "".join(
        f'<th class="{"r" if c in money_cols else ""}">{_esc(c.replace("_", " "))}</th>' for c in cols
    )
    body_rows = []
    for r in rows:
        cells = []
        for c in cols:
            val = r.get(c, "")
            if c in money_cols:
                text = _money(val)
            else:
                text = _esc(val)
            cells.append(f'<td class="{"r" if c in money_cols else ""}">{text}</td>')
        row_cls = ""
        if str(r.get("Profitable_On_Payroll", "")).lower() == "no" or str(r.get("Profitable_On_Benchmark", "")).lower() == "no":
            row_cls = ' class="loss"'
        elif str(r.get("Profitable_On_Payroll", "")).lower() == "yes":
            row_cls = ' class="profit"'
        body_rows.append(f"<tr{row_cls}>{''.join(cells)}</tr>")
    return f"""
<h2>{_esc(title)}</h2>
<table>
<thead><tr>{head}</tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table>
"""


def stitching_reports_print_html(hub: dict[str, Any]) -> str:
    """Full report pack HTML from ``stitching_reports_hub`` payload."""
    date_from = _esc(hub.get("date_from", ""))
    date_to = _esc(hub.get("date_to", ""))
    generated = _esc(hub.get("generated_at", ""))

    kp = hub.get("karigar_profitability", {}).get("summary", {}) or {}
    payroll = hub.get("payroll", {}) or {}
    ch_sum = hub.get("challan_labour", {}).get("summary", {}) or {}

    kpi_html = f"""
<div class="kpi-row">
  <div class="kpi"><div class="kpi-label">Total payroll paid</div>
    <div class="kpi-value">{_money(kp.get("total_payroll_paid", payroll.get("total_payroll", 0)))}</div></div>
  <div class="kpi"><div class="kpi-label">Profitable on payroll</div>
    <div class="kpi-value">{_esc(kp.get("profitable_on_payroll", "—"))} / {_esc(kp.get("karigar_count", "—"))}</div></div>
  <div class="kpi"><div class="kpi-label">Profitable on benchmark</div>
    <div class="kpi-value">{_esc(kp.get("profitable_on_benchmark", "—"))} / {_esc(kp.get("karigar_count", "—"))}</div></div>
  <div class="kpi"><div class="kpi-label">Net P&amp;L (benchmark)</div>
    <div class="kpi-value">{_money(kp.get("total_net_pl_benchmark", 0))}</div></div>
</div>
"""

    karigar_cols = [
        "Karigar_Name",
        "Total_Payroll_Paid",
        "Attendance_Pay",
        "Other_Work_Pay",
        "Piece_Value_Rs",
        "Pay_vs_Piece_Rs",
        "Profitable_On_Payroll",
        "Budgeted_Rs",
        "Net_PL_Benchmark",
        "Profitable_On_Benchmark",
        "Running_LTL",
        "Payroll_Only",
    ]
    money_k = {
        "Total_Payroll_Paid",
        "Attendance_Pay",
        "Other_Work_Pay",
        "Piece_Value_Rs",
        "Pay_vs_Piece_Rs",
        "Budgeted_Rs",
        "Net_PL_Benchmark",
    }

    challan_cols = [
        "Challan_No",
        "Style",
        "Karigar_Name",
        "Pieces",
        "Piece_Value_Rs",
        "Budgeted_Labour_Rs",
        "Total_Payroll_Paid",
        "Expense_On_Challan_Rs",
        "Attendance_Allocated_Rs",
        "Pay_vs_Budget",
        "Net_PL_Benchmark",
        "Profitable_On_Payroll",
        "Profitable_On_Benchmark",
    ]
    money_ch = {
        "Piece_Value_Rs",
        "Budgeted_Labour_Rs",
        "Total_Payroll_Paid",
        "Expense_On_Challan_Rs",
        "Attendance_Allocated_Rs",
        "Pay_vs_Budget",
        "Net_PL_Benchmark",
    }

    sections = [
        kpi_html,
        '<p class="note">Payroll = attendance (punches) + karigar expenses (part change, alter, etc.). '
        "Benchmark P&amp;L uses ₹480/day factory costing. LTL column shows applied tolerance targets.</p>",
        _table(
            "Karigar profitability — pay vs piece vs benchmark",
            hub.get("karigar_profitability", {}).get("rows", []) or [],
            karigar_cols,
            money_cols=money_k,
        ),
        _table(
            "Challan labour — salary paid and budget",
            hub.get("challan_labour", {}).get("rows", []) or [],
            challan_cols,
            money_cols=money_ch,
        ),
        _table(
            "Payroll register (all expenses)",
            payroll.get("rows", []) or [],
            ["Karigar_ID", "Name", "Days", "Attendance_Pay", "Other_Work_Pay", "Total"],
            money_cols={"Attendance_Pay", "Other_Work_Pay", "Total"},
        ),
    ]

    perf = hub.get("performance", {}) or {}
    if perf.get("ok") and perf.get("rows"):
        sections.append(
            _table(
                "Performance — piece value vs full payroll",
                perf.get("rows", []) or [],
                [
                    "Name",
                    "Total_Payroll_Paid",
                    "Attendance_Pay",
                    "Other_Work_Pay",
                    "Piece_Value",
                    "Surplus",
                    "Avg_Eff",
                    "Grade",
                ],
                money_cols={"Total_Payroll_Paid", "Attendance_Pay", "Other_Work_Pay", "Piece_Value", "Surplus"},
            )
        )

    comp = hub.get("comparison", {}) or {}
    if comp.get("karigar_comparison"):
        sections.append(
            _table(
                "P&amp;L compare — karigar (benchmark budget vs actual)",
                comp.get("karigar_comparison", []) or [],
                [
                    "Karigar_Name",
                    "Pieces",
                    "Budgeted_Rs",
                    "Actual_Rs",
                    "Net_PL_Rs",
                    "Running_LTL",
                    "Status",
                ],
                money_cols={"Budgeted_Rs", "Actual_Rs", "Net_PL_Rs"},
            )
        )

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Stitching Reports {date_from} – {date_to}</title>
<style>{_PRINT_CSS}</style>
</head>
<body>
<div class="header">
  <div>
    <div class="company">Garment ERP — Stitching Costing</div>
    <div style="font-size:11px;color:#64748b;margin-top:4px">Payroll &amp; profitability report pack</div>
  </div>
  <div>
    <div class="doc-title">STITCHING REPORTS</div>
    <div class="doc-sub">Period: {date_from} to {date_to}</div>
    <div class="doc-sub">Generated: {generated}</div>
  </div>
</div>
{body}
<div class="footer">Use Print → Save as PDF · Progressino Garment ERP</div>
<script>window.onload=function(){{window.focus();window.print();}}</script>
</body>
</html>"""
