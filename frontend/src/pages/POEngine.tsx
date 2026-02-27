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

const DISPLAY_COLS = [
  'Priority', 'OMS_SKU', 'Total_Inventory', 'Days_Left',
  'Sold_Units', 'Return_Units', 'ADS',
  'Gross_PO_Qty', 'PO_Pipeline_Total', 'PO_Qty',
]

const PRIORITY_ORDER: Record<string, number> = {
  '🔴 URGENT': 0, '🟡 HIGH': 1, '🟢 MEDIUM': 2, '🔄 In Pipeline': 3, '⚪ OK': 4,
}

export default function POEngine() {
  const [params, setParams] = useState({
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
  const [result, setResult]   = useState<POResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [search, setSearch]   = useState('')
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

  const urgent     = allRows.filter(r => r['Priority'] === '🔴 URGENT').length
  const high       = allRows.filter(r => r['Priority'] === '🟡 HIGH').length
  const medium     = allRows.filter(r => r['Priority'] === '🟢 MEDIUM').length
  const pipeline   = allRows.filter(r => r['Priority'] === '🔄 In Pipeline').length
  const totalPOUnits = allRows.reduce((s, r) => s + (Number(r['PO_Qty']) || 0), 0)

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-[#002B5B]">🎯 PO Engine</h2>
        <p className="text-gray-400 text-sm mt-1">Calculate purchase orders based on sales velocity and inventory.</p>
      </div>

      {/* Parameters */}
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

      {/* Results */}
      {result?.ok && allRows.length > 0 && (
        <>
          {/* Priority KPIs */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <KpiCard label="🔴 URGENT"      value={urgent}      accent="border-l-red-500" />
            <KpiCard label="🟡 HIGH"        value={high}        accent="border-l-yellow-400" />
            <KpiCard label="🟢 MEDIUM"      value={medium}      accent="border-l-green-500" />
            <KpiCard label="🔄 In Pipeline" value={pipeline}    accent="border-l-blue-400" />
            <KpiCard label="Total PO Units" value={totalPOUnits} accent="border-l-[#002B5B]" />
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            <input
              value={search} onChange={e => setSearch(e.target.value)}
              placeholder="Search SKU…"
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-56 focus:outline-none focus:ring-2 focus:ring-blue-300"
            />
            <Toggle
              label="Sort by Priority"
              checked={sortByPriority}
              onChange={setSortByPriority}
            />
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
                  {DISPLAY_COLS.map(c => (
                    <th key={c} className="text-left px-4 py-3 font-semibold text-gray-600 whitespace-nowrap">{c.replace(/_/g,' ')}</th>
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
                      {DISPLAY_COLS.map(col => (
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
  const cols = columns.length ? columns : DISPLAY_COLS
  const header = cols.join(',')
  const body = rows.map(r => cols.map(c => JSON.stringify(r[c] ?? '')).join(',')).join('\n')
  const blob = new Blob([header + '\n' + body], { type: 'text/csv' })
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
  a.download = 'po_recommendation.csv'; a.click()
}
