/** Single-page HTML report for a confirmed PO raise batch. */

export type RaiseReportRow = {
  OMS_SKU: string
  Priority?: string
  Days_Left?: number
  ADS?: number
  Gross_PO_Qty?: number
  PO_Pipeline_Total?: number
  Final_PO_Qty: number
}

export function buildPoRaiseReportHtml(opts: {
  poNumber: string
  raisedDate: string
  rows: RaiseReportRow[]
  totalQty: number
}): string {
  const { poNumber, raisedDate, rows, totalQty } = opts
  const urgent = rows.filter(r => String(r.Priority ?? '').includes('URGENT')).length
  const top = [...rows].sort((a, b) => b.Final_PO_Qty - a.Final_PO_Qty).slice(0, 15)
  const esc = (s: string) =>
    s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

  const topRows = top
    .map(
      r => `<tr>
        <td>${esc(String(r.OMS_SKU))}</td>
        <td>${esc(String(r.Priority ?? ''))}</td>
        <td class="num">${Number(r.Days_Left ?? 0).toFixed(1)}</td>
        <td class="num">${Number(r.ADS ?? 0).toFixed(3)}</td>
        <td class="num"><strong>${r.Final_PO_Qty.toLocaleString()}</strong></td>
      </tr>`,
    )
    .join('')

  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>${esc(poNumber)} — PO Raise Report</title>
<style>
  body{font-family:system-ui,sans-serif;color:#1a1f2e;background:#f4f6fa;margin:0;font-size:11px;line-height:1.35}
  .page{width:210mm;min-height:297mm;margin:12px auto;padding:14mm 16mm;background:#fff;box-shadow:0 4px 24px rgba(0,0,0,.08)}
  h1{font-size:20px;color:#1e4d8c;margin:0}
  .sub{color:#5c6478;font-size:11px;margin-top:4px}
  .hero{display:grid;grid-template-columns:1.2fr 1fr 1fr;gap:8px;margin:12px 0}
  .kpi{border:1px solid #dde2ec;border-radius:6px;padding:8px 10px;background:#fafbfd}
  .kpi.primary{background:#e8f0fa;border-color:#b8cfe8}
  .kpi label{font-size:9px;text-transform:uppercase;letter-spacing:.06em;color:#5c6478;font-weight:600}
  .kpi .v{font-size:24px;font-weight:700;color:#1e4d8c;margin-top:2px}
  table{width:100%;border-collapse:collapse;font-size:9.5px;margin-top:8px}
  th,td{padding:3px 5px;border-bottom:1px solid #dde2ec;text-align:left}
  th{font-size:8.5px;text-transform:uppercase;color:#5c6478}
  td.num,th.num{text-align:right}
  footer{margin-top:12px;padding-top:8px;border-top:1px solid #dde2ec;color:#5c6478;font-size:8.5px}
  @media print{body{background:#fff}.page{margin:0;box-shadow:none}@page{size:A4;margin:10mm}}
</style></head><body><div class="page">
  <h1>${esc(poNumber)}</h1>
  <p class="sub">Confirmed purchase order raise · ${esc(raisedDate)}</p>
  <div class="hero">
    <div class="kpi primary"><label>Total units raised</label><div class="v">${totalQty.toLocaleString()}</div></div>
    <div class="kpi"><label>SKU lines</label><div class="v" style="font-size:20px">${rows.length.toLocaleString()}</div></div>
    <div class="kpi"><label>Urgent lines</label><div class="v" style="font-size:20px">${urgent.toLocaleString()}</div></div>
  </div>
  <p>This PO batch was confirmed via PO Engine. Quantities are recorded in the raise ledger and reduce the next PO recommendation for these SKUs.</p>
  <table><thead><tr><th>SKU</th><th>Priority</th><th class="num">Days left</th><th class="num">ADS</th><th class="num">PO Qty</th></tr></thead>
  <tbody>${topRows}</tbody></table>
  ${rows.length > 15 ? `<p class="sub">${rows.length - 15} additional SKU line(s) in the CSV export.</p>` : ''}
  <footer>PO Engine · Confidential · Generated ${esc(raisedDate)}</footer>
</div></body></html>`
}

export function downloadPoRaiseReport(html: string, poNumber: string) {
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${poNumber.replace(/\s+/g, '_')}_Report.html`
  a.click()
  URL.revokeObjectURL(url)
}
