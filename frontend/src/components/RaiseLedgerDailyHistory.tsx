import { useEffect, useMemo, useState } from 'react'
import { calendarDateIST, yesterdayIST } from '../lib/dates'

export interface RaiseLedgerDayTotal {
  raised_date: string
  sku_count: number
  total_units: number
}

export interface RaiseLedgerSkuLine {
  oms_sku: string
  raised_qty: number
}

export interface RaiseLedgerSummary {
  ledger_loaded?: boolean
  daily_totals?: RaiseLedgerDayTotal[]
  by_day?: Record<string, RaiseLedgerSkuLine[]>
  total_skus?: number
  total_units?: number
  lookback_days?: number
  planning_date?: string | null
}

export function RaiseLedgerDailyHistory({ summary }: { summary: RaiseLedgerSummary | null | undefined }) {
  const daily = summary?.daily_totals ?? []
  const byDay = summary?.by_day ?? {}

  const defaultDay = useMemo(() => {
    if (daily.length === 0) return ''
    const yday = yesterdayIST()
    if (daily.some(d => d.raised_date === yday)) return yday
    return daily[0].raised_date
  }, [daily])

  const [selectedDay, setSelectedDay] = useState('')
  useEffect(() => {
    if (defaultDay) setSelectedDay(defaultDay)
  }, [defaultDay])

  const effectiveDay = selectedDay && byDay[selectedDay] !== undefined ? selectedDay : defaultDay
  const dayRows = effectiveDay ? (byDay[effectiveDay] ?? []) : []

  if (!summary?.ledger_loaded || daily.length === 0) {
    return (
      <section className="rounded-xl border border-dashed border-slate-300 bg-slate-50/80 p-4">
        <h3 className="text-sm font-bold text-[#002B5B]">PO raised by day</h3>
        <p className="text-xs text-slate-600 mt-2">
          No confirmed raises in the ledger yet. Use <strong>Export &amp; Confirm</strong> in Raise PO, or{' '}
          <strong>Import raises (CSV / Excel)</strong> on the PO Recommendation tab (e.g. yesterday&apos;s{' '}
          <code className="text-[11px]">po_recommendation.csv</code> or <code className="text-[11px]">.xlsx</code>).
        </p>
      </section>
    )
  }

  return (
    <section className="rounded-xl border border-violet-200 bg-violet-50/40 shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-violet-100 bg-white/70">
        <h3 className="text-sm font-bold text-[#002B5B]">PO raised by day</h3>
        <p className="text-xs text-gray-600 mt-0.5">
          {summary.total_skus?.toLocaleString()} SKU-day lines · {summary.total_units?.toLocaleString()} units in last{' '}
          {summary.lookback_days ?? 30} days · planning day {summary.planning_date ?? calendarDateIST()} (IST)
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="bg-white border-b border-violet-100">
            <tr>
              <th className="text-left px-4 py-2 font-semibold text-gray-600">Raise date</th>
              <th className="text-right px-4 py-2 font-semibold text-gray-600">SKUs</th>
              <th className="text-right px-4 py-2 font-semibold text-gray-600">Units raised</th>
              <th className="text-center px-4 py-2 font-semibold text-gray-600 w-24">Detail</th>
            </tr>
          </thead>
          <tbody>
            {daily.map(d => {
              const isSel = d.raised_date === effectiveDay
              const isYesterday = d.raised_date === yesterdayIST()
              return (
                <tr
                  key={d.raised_date}
                  className={`border-b border-violet-50/80 ${isSel ? 'bg-violet-100/80' : 'bg-white/50 hover:bg-violet-50/60'}`}
                >
                  <td className="px-4 py-2 font-medium text-gray-800">
                    {d.raised_date}
                    {isYesterday ? (
                      <span className="ml-2 text-[10px] font-semibold uppercase text-violet-700">yesterday</span>
                    ) : null}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums">{d.sku_count.toLocaleString()}</td>
                  <td className="px-4 py-2 text-right tabular-nums font-semibold text-[#002B5B]">
                    {d.total_units.toLocaleString()}
                  </td>
                  <td className="px-4 py-2 text-center">
                    <button
                      type="button"
                      onClick={() => setSelectedDay(d.raised_date)}
                      className={`text-[11px] px-2 py-0.5 rounded border ${
                        isSel
                          ? 'border-violet-600 bg-violet-600 text-white'
                          : 'border-violet-300 text-violet-800 hover:bg-violet-100'
                      }`}
                    >
                      View
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {effectiveDay ? (
        <div className="border-t border-violet-100 bg-white/80">
          <RaiseLedgerDayTable day={effectiveDay} rows={dayRows} />
        </div>
      ) : null}
    </section>
  )
}

function RaiseLedgerDayTable({ day, rows }: { day: string; rows: RaiseLedgerSkuLine[] }) {
  const [filter, setFilter] = useState('')

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase()
    if (!q) return rows
    return rows.filter(r => r.oms_sku.toLowerCase().includes(q))
  }, [rows, filter])

  if (rows.length === 0) {
    return <p className="text-xs text-gray-400 px-4 py-4 text-center">No lines for {day}.</p>
  }

  return (
    <>
      <div className="px-4 py-2 flex flex-wrap items-center justify-between gap-2 border-b border-gray-100">
        <span className="text-xs font-semibold text-gray-700">
          SKUs raised on <strong>{day}</strong> ({filtered.length.toLocaleString()} of {rows.length.toLocaleString()})
        </span>
        <input
          type="search"
          placeholder="Filter SKU…"
          value={filter}
          onChange={e => setFilter(e.target.value)}
          className="text-xs border border-gray-300 rounded px-2 py-1 w-44"
        />
      </div>
      <div className="overflow-x-auto max-h-[min(360px,45vh)]">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-gray-50 border-b border-gray-200 z-10">
            <tr>
              <th className="text-left px-4 py-2 font-semibold text-gray-600">OMS SKU</th>
              <th className="text-right px-4 py-2 font-semibold text-gray-600">Raised qty</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => (
              <tr key={`${r.oms_sku}-${i}`} className="border-b border-gray-50 hover:bg-gray-50/80">
                <td className="px-4 py-1.5 font-mono text-gray-800">{r.oms_sku}</td>
                <td className="px-4 py-1.5 text-right tabular-nums font-semibold text-violet-900">
                  {r.raised_qty.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {filter && filtered.length === 0 ? (
        <p className="text-xs text-gray-400 px-4 py-2">No SKUs match &quot;{filter}&quot;.</p>
      ) : null}
    </>
  )
}
