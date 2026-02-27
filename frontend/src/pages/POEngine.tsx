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
  PO_Qty?: number
  Stockout_Flag?: string
  [key: string]: string | number | undefined
}

interface POResult {
  ok: boolean
  message?: string
  rows?: PORow[]
  columns?: string[]
}

const DISPLAY_COLS = ['OMS_SKU','Total_Inventory','Sold_Units','Return_Units','ADS','PO_Qty','Stockout_Flag']

export default function POEngine() {
  const [params, setParams] = useState({
    period_days: 90,
    lead_time: 30,
    target_days: 60,
    demand_basis: 'Sold',
    use_seasonality: false,
    seasonal_weight: 0.5,
    group_by_parent: false,
  })
  const [result, setResult]   = useState<POResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [search, setSearch]   = useState('')

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

  const rows = (result?.rows ?? []).filter(r =>
    !search || String(r['OMS_SKU'] ?? '').toLowerCase().includes(search.toLowerCase())
  )
  const poRows = rows.filter(r => (r['PO_Qty'] ?? 0) > 0)
  const oos    = rows.filter(r => r['Stockout_Flag'] === 'OOS')

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

        <div className="flex items-center gap-6 mt-4">
          <Toggle
            label="Use YoY Seasonality"
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
          className="mt-4 px-6 py-2.5 rounded-lg text-sm font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-50"
        >
          {loading ? 'Calculating…' : '🎯 Calculate PO'}
        </button>

        {result && !result.ok && (
          <p className="mt-3 text-sm text-red-600 bg-red-50 rounded p-2">{result.message}</p>
        )}
      </div>

      {/* Results */}
      {result?.ok && (result.rows ?? []).length > 0 && (
        <>
          <div className="grid grid-cols-3 gap-4">
            <KpiCard label="SKUs Needing PO" value={poRows.length} accent="border-l-orange-500" />
            <KpiCard label="Out of Stock"     value={oos.length}   accent="border-l-red-500" />
            <KpiCard label="Total SKUs"       value={rows.length} />
          </div>

          <div className="flex items-center gap-3">
            <input
              value={search} onChange={e => setSearch(e.target.value)}
              placeholder="Search SKU…"
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-56 focus:outline-none focus:ring-2 focus:ring-blue-300"
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
                    <th key={c} className="text-left px-4 py-3 font-semibold text-gray-600 whitespace-nowrap">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 300).map((row, i) => {
                  const isOos = row['Stockout_Flag'] === 'OOS'
                  const needsPo = (row['PO_Qty'] ?? 0) > 0
                  return (
                    <tr key={i} className={`border-b border-gray-100 hover:bg-blue-50
                      ${isOos ? 'bg-red-50' : needsPo ? 'bg-amber-50' : ''}`}>
                      {DISPLAY_COLS.map(col => (
                        <td key={col} className="px-4 py-2 whitespace-nowrap text-gray-700">
                          {col === 'Stockout_Flag'
                            ? (row[col] ? <span className="text-red-600 font-bold">⚠ OOS</span> : '—')
                            : col === 'OMS_SKU'
                              ? <span className="font-medium text-gray-900">{row[col]}</span>
                              : col === 'PO_Qty'
                                ? <span className={`font-bold ${(row[col] ?? 0) > 0 ? 'text-orange-600' : 'text-gray-400'}`}>
                                    {row[col]}
                                  </span>
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
