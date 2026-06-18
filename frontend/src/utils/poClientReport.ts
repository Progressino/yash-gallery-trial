/** Downloadable HTML client summary for a PO Engine recommendation run. */

export type PoClientReportRow = Record<string, string | number | undefined>

export type PoClientReportParams = {
  period_days: number
  lead_time: number
  target_days: number
  demand_basis: string
  use_seasonality: boolean
  use_ly_fallback: boolean
}

export type PoClientReportMeta = {
  reportDate: string
  salesThrough?: string | null
  planningDate?: string | null
  poMergeVersion?: number
  totalRows: number
}

type Slice = { label: string; value: number; color: string }

const esc = (s: string) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

function n(v: unknown): number {
  const x = Number(v)
  return Number.isFinite(x) ? x : 0
}

function priorityBand(priority: unknown): 'urgent' | 'medium' | 'other' {
  const p = String(priority ?? '')
  if (p.includes('URGENT')) return 'urgent'
  if (p.includes('MEDIUM')) return 'medium'
  return 'other'
}

function parentStyle(sku: string): string {
  return sku.replace(
    /-(XS|S|M|L|XL|XXL|XXXL|2XL|3XL|4XL|5XL|6XL|7XL|8XL|\d+XL)(-(XS|S|M|L|XL|XXL|XXXL|2XL|3XL|4XL|5XL|6XL|7XL|8XL|\d+XL))?$/i,
    '',
  )
}

function svgPieChart(slices: Slice[], size = 220): string {
  const data = slices.filter(s => s.value > 0)
  const total = data.reduce((s, x) => s + x.value, 0)
  if (total <= 0) {
    return `<p class="muted">No PO units in this run.</p>`
  }
  const cx = size / 2
  const cy = size / 2
  const r = size / 2 - 12
  let angle = -Math.PI / 2
  const paths = data
    .map(s => {
      const frac = s.value / total
      const a2 = angle + frac * 2 * Math.PI
      const x1 = cx + r * Math.cos(angle)
      const y1 = cy + r * Math.sin(angle)
      const x2 = cx + r * Math.cos(a2)
      const y2 = cy + r * Math.sin(a2)
      const large = frac > 0.5 ? 1 : 0
      const d = `M ${cx} ${cy} L ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)} Z`
      angle = a2
      return `<path d="${d}" fill="${s.color}" stroke="#fff" stroke-width="1"/>`
    })
    .join('')
  const legend = data
    .map(
      s =>
        `<div class="legend-item"><span class="swatch" style="background:${s.color}"></span>` +
        `<span>${esc(s.label)}</span><strong>${s.value.toLocaleString()}</strong>` +
        `<span class="muted">(${((100 * s.value) / total).toFixed(1)}%)</span></div>`,
    )
    .join('')
  return `<div class="chart-row"><svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" role="img" aria-label="PO volume pie chart">${paths}</svg><div class="legend">${legend}</div></div>`
}

function barChartHtml(bars: { label: string; value: number }[]): string {
  const max = Math.max(...bars.map(b => b.value), 1)
  return bars
    .map(
      b =>
        `<div class="bar-row"><span class="bar-label">${esc(b.label)}</span>` +
        `<div class="bar-track"><div class="bar-fill" style="width:${((100 * b.value) / max).toFixed(1)}%"></div></div>` +
        `<span class="bar-val">${b.value.toLocaleString()}</span></div>`,
    )
    .join('')
}

function aggregate(rows: PoClientReportRow[]) {
  let totalPo = 0
  let grossPo = 0
  let skusWithPo = 0
  let pipelineUnits = 0
  let inventoryUnits = 0
  let sold30 = 0
  let returnUnits = 0
  let projSum = 0
  let projN = 0
  let postSum = 0
  let postN = 0
  let urgentPo = 0
  let mediumPo = 0
  let inPipelineRows = 0
  let okRows = 0

  const stylePo = new Map<string, number>()
  const buckets = { '1–50': 0, '51–100': 0, '101–200': 0, '201–500': 0, '501–1000': 0, '1000+': 0 }

  const poRows: Array<{
    sku: string
    priority: string
    sold: number
    ads: number
    inv: number
    pipe: number
    proj: number
    po: number
  }> = []

  for (const row of rows) {
    const po = n(row.PO_Qty)
    const pipe = n(row.PO_Pipeline_Effective ?? row.PO_Pipeline_Total)
    const inv = n(row.Total_Inventory)
    const sold = n(row.Sold_Units)
    const ads = n(row.ADS)
    const proj = n(row.Projected_Running_Days)
    const post = n(row.Post_PO_Cover_Days_Capped)
    const sku = String(row.OMS_SKU ?? '')

    totalPo += po
    grossPo += n(row.Gross_PO_Qty)
    pipelineUnits += pipe
    inventoryUnits += inv
    sold30 += sold
    returnUnits += n(row.Return_Units)

    const band = priorityBand(row.Priority)
    if (band === 'urgent') urgentPo += po
    else if (band === 'medium') mediumPo += po

    if (String(row.Priority ?? '').includes('Pipeline')) inPipelineRows += 1
    if (String(row.Priority ?? '').includes('OK')) okRows += 1

    if (po > 0) {
      skusWithPo += 1
      poRows.push({
        sku,
        priority: String(row.Priority ?? ''),
        sold,
        ads,
        inv,
        pipe,
        proj,
        po,
      })
      if (po <= 50) buckets['1–50'] += 1
      else if (po <= 100) buckets['51–100'] += 1
      else if (po <= 200) buckets['101–200'] += 1
      else if (po <= 500) buckets['201–500'] += 1
      else if (po <= 1000) buckets['501–1000'] += 1
      else buckets['1000+'] += 1

      const style = parentStyle(sku)
      stylePo.set(style, (stylePo.get(style) ?? 0) + po)
      projSum += proj
      projN += 1
      postSum += post
      postN += 1
    }
  }

  const topStyles = [...stylePo.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([style, po]) => ({ style, po, share: totalPo > 0 ? (100 * po) / totalPo : 0 }))

  const topSkus = [...poRows].sort((a, b) => b.po - a.po).slice(0, 12)

  return {
    totalPo,
    grossPo,
    skusWithPo,
    totalRows: rows.length,
    pipelineUnits,
    inventoryUnits,
    sold30,
    returnUnits,
    urgentPo,
    mediumPo,
    inPipelineRows,
    okRows,
    avgProj: projN > 0 ? projSum / projN : 0,
    avgPost: postN > 0 ? postSum / postN : 0,
    topStyles,
    topSkus,
    bucketBars: Object.entries(buckets).map(([label, value]) => ({ label, value })),
  }
}

export function buildPoClientReportHtml(opts: {
  rows: PoClientReportRow[]
  params: PoClientReportParams
  meta: PoClientReportMeta
}): string {
  const { rows, params, meta } = opts
  const a = aggregate(rows)
  const returnRate = a.sold30 > 0 ? (100 * a.returnUnits) / a.sold30 : 0

  const priorityPie = svgPieChart([
    { label: 'Urgent PO', value: a.urgentPo, color: '#dc2626' },
    { label: 'Medium PO', value: a.mediumPo, color: '#d97706' },
  ])

  const dispositionPie = svgPieChart([
    { label: 'SKUs with new PO', value: a.skusWithPo, color: '#1e4d8c' },
    { label: 'In pipeline (no PO)', value: a.inPipelineRows, color: '#64748b' },
    { label: 'OK (no PO)', value: a.okRows, color: '#94a3b8' },
  ])

  const styleRows = a.topStyles
    .map(
      s =>
        `<tr><td>${esc(s.style)}</td><td class="num">${s.po.toLocaleString()}</td>` +
        `<td class="num">${s.share.toFixed(1)}%</td></tr>`,
    )
    .join('')

  const skuRows = a.topSkus
    .map(
      s =>
        `<tr><td>${esc(s.sku)}</td><td>${esc(s.priority.replace(/[^\w\s]/g, '').trim() || '—')}</td>` +
        `<td class="num">${s.sold.toLocaleString()}</td><td class="num">${s.ads.toFixed(2)}</td>` +
        `<td class="num">${s.inv.toLocaleString()}</td><td class="num">${s.pipe.toLocaleString()}</td>` +
        `<td class="num">${s.proj.toFixed(0)}</td><td class="num"><strong>${s.po.toLocaleString()}</strong></td></tr>`,
    )
    .join('')

  const salesWindow =
    meta.salesThrough && meta.planningDate
      ? `${esc(meta.planningDate)} → ${esc(meta.salesThrough)}`
      : meta.salesThrough
        ? `through ${esc(meta.salesThrough)}`
        : esc(meta.reportDate)

  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PO Recommendation — ${esc(meta.reportDate)}</title>
<style>
  :root{--ink:#1a1f2e;--muted:#5c6478;--line:#dde2ec;--brand:#1e4d8c;--bg:#f4f6fa;--card:#fafbfd}
  *{box-sizing:border-box}
  body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--bg);margin:0;font-size:11px;line-height:1.45}
  .page{max-width:210mm;margin:16px auto;padding:14mm 16mm;background:#fff;box-shadow:0 4px 24px rgba(0,0,0,.08)}
  h1{font-size:22px;color:var(--brand);margin:0 0 4px}
  h2{font-size:14px;color:var(--brand);margin:24px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
  h3{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 8px}
  .sub{color:var(--muted);font-size:11px;margin:0}
  .hero{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:16px 0}
  .kpi{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--card)}
  .kpi.primary{background:#e8f0fa;border-color:#b8cfe8}
  .kpi label{display:block;font-size:9px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}
  .kpi .v{font-size:22px;font-weight:700;color:var(--brand);margin-top:4px}
  .callout{border-left:4px solid var(--brand);background:#f0f5fc;padding:10px 12px;margin:12px 0;border-radius:0 6px 6px 0}
  .callout.warn{border-left-color:#d97706;background:#fffbeb}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px}
  .card{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fff}
  .chart-row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
  .legend{flex:1;min-width:140px}
  .legend-item{display:grid;grid-template-columns:12px 1fr auto auto;gap:6px;align-items:center;margin-bottom:6px;font-size:10px}
  .swatch{width:12px;height:12px;border-radius:2px;display:inline-block}
  .muted{color:var(--muted)}
  table{width:100%;border-collapse:collapse;font-size:9.5px}
  th,td{padding:5px 6px;border-bottom:1px solid var(--line);text-align:left}
  th{font-size:8.5px;text-transform:uppercase;color:var(--muted);background:#f8fafc}
  tr:nth-child(even) td{background:#fafbfd}
  td.num,th.num{text-align:right}
  .bar-row{display:grid;grid-template-columns:52px 1fr 36px;gap:8px;align-items:center;margin-bottom:6px;font-size:10px}
  .bar-track{height:10px;background:#eef2f7;border-radius:4px;overflow:hidden}
  .bar-fill{height:100%;background:var(--brand);border-radius:4px}
  .bar-val{text-align:right;font-weight:600}
  .method{display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:10px}
  .method ul{margin:4px 0;padding-left:16px}
  .method li{margin-bottom:4px}
  footer{margin-top:20px;padding-top:10px;border-top:1px solid var(--line);color:var(--muted);font-size:8.5px}
  @media print{body{background:#fff}.page{margin:0;box-shadow:none;max-width:none}@page{size:A4;margin:10mm}}
</style></head><body><div class="page">
  <h1>PO Recommendation Report</h1>
  <p class="sub">Progressino PO Engine · ${esc(meta.reportDate)} · Sales window: ${salesWindow}</p>
  <p class="sub">${a.totalRows.toLocaleString()} SKU rows analysed · ${a.skusWithPo.toLocaleString()} SKUs recommended for new PO</p>

  <div class="hero">
    <div class="kpi primary"><label>New PO units</label><div class="v">${a.totalPo.toLocaleString()}</div></div>
    <div class="kpi"><label>SKUs with PO</label><div class="v" style="font-size:18px">${a.skusWithPo.toLocaleString()}</div></div>
    <div class="kpi"><label>Open pipeline</label><div class="v" style="font-size:18px">${Math.round(a.pipelineUnits / 1000)}k</div></div>
    <div class="kpi"><label>Warehouse stock</label><div class="v" style="font-size:18px">${Math.round(a.inventoryUnits / 1000)}k</div></div>
  </div>

  <div class="callout">
    Target cover after PO: <strong>${params.target_days} days</strong>. Average projected cover before PO (SKUs with PO): 
    <strong>${a.avgProj.toFixed(0)} days</strong>. Average post-PO cover: <strong>${a.avgPost.toFixed(0)} days</strong>.
  </div>

  <h2>Charts</h2>
  <div class="grid2">
    <div class="card">
      <h3>PO volume by priority</h3>
      ${priorityPie}
    </div>
    <div class="card">
      <h3>SKU disposition</h3>
      ${dispositionPie}
    </div>
  </div>
  <div class="card" style="margin-top:12px">
    <h3>SKU count by PO size band (units)</h3>
    ${barChartHtml(a.bucketBars)}
  </div>

  <h2>Concentration</h2>
  <div class="grid2">
    <div class="card">
      <h3>Top parent styles</h3>
      <table>
        <thead><tr><th>Style</th><th class="num">PO units</th><th class="num">% of total</th></tr></thead>
        <tbody>${styleRows || '<tr><td colspan="3" class="muted">No PO lines</td></tr>'}</tbody>
      </table>
    </div>
    <div class="card">
      <h3>Largest SKU recommendations</h3>
      <table>
        <thead><tr><th>SKU</th><th>Priority</th><th class="num">Sold</th><th class="num">ADS</th><th class="num">Inv</th><th class="num">Pipe</th><th class="num">Proj d</th><th class="num">PO</th></tr></thead>
        <tbody>${skuRows || '<tr><td colspan="8" class="muted">No PO lines</td></tr>'}</tbody>
      </table>
    </div>
  </div>

  <h2>Calculation methodology</h2>
  <div class="method">
    <div>
      <strong>Parameters</strong>
      <ul>
        <li>ADS window: last <strong>${params.period_days}</strong> calendar days</li>
        <li>Target cover after PO: <strong>${params.target_days}</strong> days</li>
        <li>Lead time (reference): <strong>${params.lead_time}</strong> days</li>
        <li>Demand basis: <strong>${esc(params.demand_basis)}</strong></li>
        <li>LY ADS fallback: <strong>${params.use_ly_fallback ? 'On' : 'Off'}</strong></li>
        <li>YoY seasonality: <strong>${params.use_seasonality ? 'On' : 'Off'}</strong></li>
        <li>30-day shipments: <strong>${a.sold30.toLocaleString()}</strong> units</li>
        <li>Returns in window: <strong>${a.returnUnits.toLocaleString()}</strong> (${returnRate.toFixed(1)}%)</li>
      </ul>
    </div>
    <div>
      <strong>Formulas</strong>
      <ul>
        <li>Recent ADS = Sold units ÷ effective selling days in the ADS window</li>
        <li>ADS = max(capped primary, seasonal floor, sheet FREQ floor)</li>
        <li>Burst cap: ADS cannot exceed sold÷${params.period_days} when sales are sparse</li>
        <li>Projected days = (Inventory + Pipeline) ÷ ADS</li>
        <li>Gross PO = ADS × (${params.target_days} − Projected days), pack-rounded</li>
        <li>Net PO = Gross PO after pipeline deductions and business gates</li>
      </ul>
    </div>
  </div>

  <div class="callout warn">
    <strong>Before placing orders:</strong> confirm the existing PO pipeline sheet is current (${a.pipelineUnits.toLocaleString()} units on file);
    review top styles and any SKU lines above 1,000 units; export the full CSV for line-level audit.
  </div>

  <footer>
    PO Engine client report · Generated ${esc(meta.reportDate)}
    ${meta.poMergeVersion != null ? ` · Engine v${meta.poMergeVersion}` : ''}
    · Confidential — for internal planning and supplier discussion
  </footer>
</div></body></html>`
}

export function downloadPoClientReport(html: string, reportDate: string) {
  const safe = reportDate.replace(/[^\d-]/g, '')
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `PO_Client_Report_${safe}.html`
  a.click()
  URL.revokeObjectURL(url)
}
