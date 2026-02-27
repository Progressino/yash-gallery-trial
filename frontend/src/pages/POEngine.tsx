import { useState } from 'react'
import { api } from '../api/client'

interface PORow {
  OMS_SKU: string
  Total_Inventory?: number
  Sold_Units?: number
  Return_Units?: number
  Net_Units?: number
  Recent_ADS?: number
  ADS?: number
  LY_ADS?: number
  Days_Left?: number
  Gross_PO_Qty?: number
  PO_Pipeline_Total?: number
  PO_Qty?: number
  Priority?: string
  Stockout_Flag?: string
  [key: string]: string | number | undefined
}

interface POResult {
  ok: boolean
  message?: string
  rows?: PORow[]
  columns?: string[]
}

interface QuarterlyRow {
  OMS_SKU: string
  Avg_Monthly?: number
  ADS?: number
  Units_90d?: number
  Units_30d?: number
  Freq_30d?: number
  Status?: string
  [key: string]: string | number | undefined
}

interface QuarterlyResult {
  loaded: boolean
  columns?: string[]
  rows?: QuarterlyRow[]
}

const PO_DISPLAY_COLS = [
  'Priority', 'OMS_SKU', 'Total_Inventory', 'Days_Left',
  'Sold_Units', 'Return_Units', 'ADS',
  'Gross_PO_Qty', 'PO_Pipeline_Total', 'PO_Qty',
]

const PRIORITY_ORDER: Record<string, number> = {
  '🔴 URGENT': 0, '🟡 HIGH': 1, '🟢 MEDIUM': 2, '🔄 In Pipeline': 3, '⚪ OK': 4,
}

const STATUS_COLORS: Record<string, string> = {
  'Fast Moving':  'text-green-700 bg-green-50',
  'Moderate':     'text-blue-700 bg-blue-50',
  'Slow Selling': 'text-yellow-700 bg-yellow-50',
  'Not Moving':   'text-red-600 bg-red-50',
}

type Tab = 'po' | 'quarterly'

export default function POEngine() {
  const [activeTab, setActiveTab]   = useState<Tab>('po')
  const [params, setParams]         = useState({
    period_days:     90,
    lead_time:       30,
    target_days:     60,
    demand_basis:    'Sold',
    use_seasonality: false,
    seasonal_weight: 0.5,
    group_by_parent: false,
    grace_days:      7,
    safety_pct:      20,
  })
  const [result, setResult]         = useState<POResult | null>(null)
  const [quarterly, setQuarterly]   = useState<QuarterlyResult | null>(null)
  const [loading, setLoading]       = useState(false)
  const [qLoading, setQLoading]     = useState(false)
  const [search, setSearch]         = useState('')
  const [qSearch, setQSearch]       = useState('')
  const [sortByPriority, setSortByPriority] = useState(true)

  const run = async () => {
    setLoading(true)
    try {
      const { data } = await api.post<POResult>('/po/calculate', params)
      setResult(data)
    } catch (e: unknown) {
      setResult({ ok: false, message: e instanceof Error ? e.message : 'Error' })
    } finally {
      setLoading(false)
    }
  }

  const loadQuarterly = async () => {
    setQLoading(true)
    try {
      const { data } = await api.get<QuarterlyResult>('/po/quarterly', {
        params: { group_by_parent: params.group_by_parent, n_quarters: 8 },
      })
      setQuarterly(data)
    } catch {
      setQuarterly({ loaded: false })
    } finally {
      setQLoading(false)
    }
  }

  // PO tab
  const allRows = result?.rows ?? []
  const filtered = allRows.filter(r =>
    !search || String(r['OMS_SKU'] ?? '').toLowerCase().includes(search.toLowerCase())
  )
  const rows = sortByPriority
    ? [...filtered].sort((a, b) =>
        (PRIORITY_ORDER[a['Priority'] as string] ?? 9) -
        (PRIORITY_ORDER[b['Priority'] as string] ?? 9)
      )
    : filtered

  const urgent       = allRows.filter(r => r['Priority'] === '🔴 URGENT').length
  const high         = allRows.filter(r => r['Priority'] === '🟡 HIGH').length
  const medium       = allRows.filter(r => r['Priority'] === '🟢 MEDIUM').length
  const pipeline     = allRows.filter(r => r['Priority'] === '🔄 In Pipeline').length
  const totalPOUnits = allRows.reduce((s, r) => s + (Number(r['PO_Qty']) || 0), 0)

  // Quarterly tab
  const qAllRows    = quarterly?.rows ?? []
  const qCols       = quarterly?.columns ?? []
  const quarterCols = qCols.filter(c =>
    !['OMS_SKU', 'Avg_Monthly', 'ADS', 'Units_90d', 'Units_30d', 'Freq_30d', 'Status'].includes(c)
  )
  const qFiltered = qAllRows.filter(r =>
    !qSearch || String(r['OMS_SKU'] ?? '').toLowerCase().includes(qSearch.toLowerCase())
  )

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-[#002B5B]">🎯 PO Engine</h2>
        <p className="text-gray-400 text-sm mt-1">Calculate purchase orders based on sales velocity and inventory.</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-200">
        {(['po', 'quarterly'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            className={`px-5 py-2.5 text-sm font-semibold rounded-t-lg border-b-2 transition-colors
              ${activeTab === t
                ? 'border-[#002B5B] text-[#002B5B] bg-white'
                : 'border-transparent text-gray-500 hover:text-gray-700'}`}
          >
            {t === 'po' ? '🎯 PO Recommendation' : '📊 Quarterly History'}
          </button>
        ))}
      </div>

      {/* ── PO Recommendation Tab ── */}
      {activeTab === 'po' && (
        <>
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="font-semibold text-[#002B5B] mb-4">Parameters</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Param label="Period (days)" type="number"
                value={params.period_days} onChange={v => setParams(p => ({ ...p, period_days: +v }))} />
              <Param label="Lead Time (days)" type="number"
                value={params.lead_time} onChange={v => setParams(p => ({ ...p, lead_time: +v }))} />
              <Param label="Target Cover (days)" type="number"
                value={params.target_days} onChange={v => setParams(p => ({ ...p, target_days: +v }))} />
              <div>
                <label className="text-xs font-semibold text-gray-500 uppercase block mb-1">Demand Basis</label>
                <select
                  value={params.demand_basis}
                  onChange={e => setParams(p => ({ ...p, demand_basis: e.target.value }))}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                >
                  <option value="Sold">Sold</option>
                  <option value="Net">Net</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
              <Param label="Grace Days (urgency buffer)" type="number"
                value={params.grace_days} onChange={v => setParams(p => ({ ...p, grace_days: +v }))} />
              <div>
                <label className="text-xs font-semibold text-gray-500 uppercase block mb-1">
                  Safety Stock % ({params.safety_pct}%)
                </label>
                <input
                  type="range" min={0} max={100} step={5}
                  value={params.safety_pct}
                  onChange={e => setParams(p => ({ ...p, safety_pct: +e.target.value }))}
                  className="w-full accent-[#002B5B]"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                  <span>0%</span><span>50%</span><span>100%</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-6 mt-4 flex-wrap">
              <Toggle
                label="YoY Seasonality"
                checked={params.use_seasonality}
                onChange={v => setParams(p => ({ ...p, use_seasonality: v }))}
              />
              {params.use_seasonality && (
                <Param label={`Seasonal Weight (${Math.round(params.seasonal_weight * 100)}%)`} type="range"
                  value={params.seasonal_weight} min={0} max={1} step={0.05}
                  onChange={v => setParams(p => ({ ...p, seasonal_weight: +v }))} />
              )}
              <Toggle
                label="Group by Parent SKU"
                checked={params.group_by_parent}
                onChange={v => setParams(p => ({ ...p, group_by_parent: v }))}
              />
            </div>

            <button
              onClick={run} disabled={loading}
              className="mt-5 px-6 py-2.5 rounded-lg text-sm font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-50"
            >
              {loading ? 'Calculating…' : '🎯 Calculate PO'}
            </button>

            {result && !result.ok && (
              <p className="mt-3 text-sm text-red-600 bg-red-50 rounded p-2">{result.message}</p>
            )}
          </div>

          {result?.ok && allRows.length > 0 && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <KpiCard label="🔴 URGENT"      value={urgent}       accent="border-l-red-500" />
                <KpiCard label="🟡 HIGH"        value={high}         accent="border-l-yellow-400" />
                <KpiCard label="🟢 MEDIUM"      value={medium}       accent="border-l-green-500" />
                <KpiCard label="🔄 In Pipeline" value={pipeline}     accent="border-l-blue-400" />
                <KpiCard label="Total PO Units" value={totalPOUnits} accent="border-l-[#002B5B]" />
              </div>

              <div className="flex items-center gap-3 flex-wrap">
                <input
                  value={search} onChange={e => setSearch(e.target.value)}
                  placeholder="Search SKU…"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-56 focus:outline-none focus:ring-2 focus:ring-blue-300"
                />
                <Toggle label="Sort by Priority" checked={sortByPriority} onChange={setSortByPriority} />
                <span className="text-xs text-gray-400">{rows.length} SKUs</span>
                <button
                  onClick={() => downloadCsv(result.rows ?? [], result.columns ?? [])}
                  className="ml-auto text-xs px-3 py-1.5 rounded border border-gray-300 hover:bg-gray-50"
                >
                  ⬇ Export CSV
                </button>
              </div>

              <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      {PO_DISPLAY_COLS.map(c => (
                        <th key={c} className="text-left px-4 py-3 font-semibold text-gray-600 whitespace-nowrap">
                          {c.replace(/_/g, ' ')}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.slice(0, 300).map((row, i) => {
                      const priority = String(row['Priority'] ?? '')
                      const rowBg =
                        priority === '🔴 URGENT'      ? 'bg-red-50' :
                        priority === '🟡 HIGH'        ? 'bg-yellow-50' :
                        priority === '🟢 MEDIUM'      ? 'bg-amber-50' :
                        priority === '🔄 In Pipeline' ? 'bg-blue-50' : ''
                      return (
                        <tr key={i} className={`border-b border-gray-100 hover:brightness-95 ${rowBg}`}>
                          {PO_DISPLAY_COLS.map(col => (
                            <td key={col} className="px-4 py-2 whitespace-nowrap text-gray-700">
                              {col === 'Priority'
                                ? <span className="font-semibold text-xs">{row[col] ?? '⚪ OK'}</span>
                                : col === 'OMS_SKU'
                                  ? <span className="font-medium text-gray-900">{row[col]}</span>
                                  : col === 'PO_Qty'
                                    ? <span className={`font-bold ${(row[col] ?? 0) > 0 ? 'text-orange-600' : 'text-gray-400'}`}>
                                        {row[col]}
                                      </span>
                                    : col === 'Days_Left'
                                      ? <DaysLeftBadge days={Number(row[col] ?? 999)} />
                                      : typeof row[col] === 'number'
                                        ? Number(row[col]).toLocaleString(undefined, { maximumFractionDigits: 3 })
                                        : row[col] ?? '—'
                              }
                            </td>
                          ))}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
                {rows.length > 300 && (
                  <p className="text-xs text-gray-400 text-center py-2">Showing 300 of {rows.length}</p>
                )}
              </div>
            </>
          )}
        </>
      )}

      {/* ── Quarterly History Tab ── */}
      {activeTab === 'quarterly' && (
        <>
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <div className="flex items-center gap-4 flex-wrap">
              <Toggle
                label="Group by Parent SKU"
                checked={params.group_by_parent}
                onChange={v => setParams(p => ({ ...p, group_by_parent: v }))}
              />
              <button
                onClick={loadQuarterly} disabled={qLoading}
                className="px-5 py-2.5 rounded-lg text-sm font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-50"
              >
                {qLoading ? 'Loading…' : '📊 Load Quarterly History'}
              </button>
              {quarterly && !quarterly.loaded && !qLoading && (
                <span className="text-sm text-red-500">No data — build Sales first.</span>
              )}
            </div>
          </div>

          {quarterly?.loaded && qAllRows.length > 0 && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <KpiCard label="Total SKUs" value={qAllRows.length} />
                <KpiCard label="Fast Moving"
                  value={qAllRows.filter(r => r['Status'] === 'Fast Moving').length}
                  accent="border-l-green-500" />
                <KpiCard label="Moderate"
                  value={qAllRows.filter(r => r['Status'] === 'Moderate').length}
                  accent="border-l-blue-400" />
                <KpiCard label="Not Moving"
                  value={qAllRows.filter(r => r['Status'] === 'Not Moving').length}
                  accent="border-l-red-400" />
              </div>

              <div className="flex items-center gap-3 flex-wrap">
                <input
                  value={qSearch} onChange={e => setQSearch(e.target.value)}
                  placeholder="Search SKU…"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-56 focus:outline-none focus:ring-2 focus:ring-blue-300"
                />
                <span className="text-xs text-gray-400">{qFiltered.length} SKUs</span>
                <button
                  onClick={() => downloadQCsv(quarterly.rows ?? [], quarterly.columns ?? [])}
                  className="ml-auto text-xs px-3 py-1.5 rounded border border-gray-300 hover:bg-gray-50"
                >
                  ⬇ Export CSV
                </button>
              </div>

              <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      <th className="text-left px-4 py-3 font-semibold text-gray-600 whitespace-nowrap sticky left-0 bg-gray-50 z-10 shadow-sm">
                        SKU
                      </th>
                      {quarterCols.map(c => (
                        <th key={c} className="text-right px-3 py-3 font-semibold text-gray-500 whitespace-nowrap text-xs">
                          {c}
                        </th>
                      ))}
                      <th className="text-right px-3 py-3 font-semibold text-gray-600 whitespace-nowrap">Avg/Month</th>
                      <th className="text-right px-3 py-3 font-semibold text-gray-600 whitespace-nowrap">Daily Avg</th>
                      <th className="text-right px-3 py-3 font-semibold text-gray-600 whitespace-nowrap">Last 30d</th>
                      <th className="text-left px-3 py-3 font-semibold text-gray-600 whitespace-nowrap">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {qFiltered.slice(0, 500).map((row, i) => {
                      const status = String(row['Status'] ?? '')
                      const statusClass = STATUS_COLORS[status] ?? ''
                      return (
                        <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                          <td className="px-4 py-2 font-medium text-gray-900 whitespace-nowrap sticky left-0 bg-white z-10 shadow-sm">
                            {row['OMS_SKU']}
                          </td>
                          {quarterCols.map(c => (
                            <td key={c} className="px-3 py-2 text-right whitespace-nowrap text-gray-700">
                              {Number(row[c] ?? 0) > 0
                                ? <span className="font-medium">{Number(row[c]).toLocaleString()}</span>
                                : <span className="text-gray-300">—</span>}
                            </td>
                          ))}
                          <td className="px-3 py-2 text-right whitespace-nowrap font-semibold text-gray-800">
                            {Number(row['Avg_Monthly'] ?? 0).toFixed(1)}
                          </td>
                          <td className="px-3 py-2 text-right whitespace-nowrap text-gray-700">
                            {Number(row['ADS'] ?? 0).toFixed(3)}
                          </td>
                          <td className="px-3 py-2 text-right whitespace-nowrap text-gray-700">
                            {Number(row['Units_30d'] ?? 0).toLocaleString()}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap">
                            <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${statusClass}`}>
                              {status || '—'}
                            </span>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
                {qFiltered.length > 500 && (
                  <p className="text-xs text-gray-400 text-center py-2">Showing 500 of {qFiltered.length}</p>
                )}
              </div>

              <p className="text-xs text-gray-400">
                Quarterly totals = forward shipments only. Avg/Month = last 4 quarters ÷ 3. Daily Avg = last 90d ÷ 90.
              </p>
            </>
          )}
        </>
      )}
    </div>
  )
}

// ── Helpers ───────────────────────────────────────────────────

function DaysLeftBadge({ days }: { days: number }) {
  if (days >= 999) return <span className="text-gray-400">∞</span>
  const color = days < 14 ? 'text-red-600 font-bold' : days < 30 ? 'text-yellow-600 font-semibold' : 'text-gray-700'
  return <span className={color}>{days}</span>
}

function KpiCard({ label, value, accent }: { label: string; value: number; accent?: string }) {
  return (
    <div className={`bg-white rounded-xl border border-gray-200 p-4 shadow-sm border-l-4 ${accent ?? 'border-l-[#002B5B]'}`}>
      <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-bold text-[#002B5B] mt-1">{value.toLocaleString()}</p>
    </div>
  )
}

function Param({
  label, type, value, min, max, step, onChange,
}: {
  label: string; type: string; value: number
  min?: number; max?: number; step?: number
  onChange: (v: string) => void
}) {
  return (
    <div>
      <label className="text-xs font-semibold text-gray-500 uppercase block mb-1">{label}</label>
      <input
        type={type} value={value} min={min} max={max} step={step}
        onChange={e => onChange(e.target.value)}
        className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
      />
    </div>
  )
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-700">
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} className="rounded" />
      {label}
    </label>
  )
}

function downloadCsv(rows: PORow[], columns: string[]) {
  const cols = columns.length ? columns : PO_DISPLAY_COLS
  const header = cols.join(',')
  const body = rows.map(r => cols.map(c => JSON.stringify(r[c] ?? '')).join(',')).join('\n')
  const blob = new Blob([header + '\n' + body], { type: 'text/csv' })
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
  a.download = 'po_recommendation.csv'; a.click()
}

function downloadQCsv(rows: QuarterlyRow[], columns: string[]) {
  const header = columns.join(',')
  const body = rows.map(r => columns.map(c => JSON.stringify(r[c] ?? '')).join(',')).join('\n')
  const blob = new Blob([header + '\n' + body], { type: 'text/csv' })
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
  a.download = 'quarterly_history.csv'; a.click()
}
