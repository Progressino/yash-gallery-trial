import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  api,
  getPoDailyInventoryHistoryMatrix,
  getPoDailyInventoryHistorySku,
  type InventoryHistoryChannel,
} from '../api/client'
import { useSession } from '../store/session'
import { InventoryStalenessBanner } from '../components/InventoryStalenessBanner'
import { todayIsoIST } from '../lib/reportingDates'

const PAGE_SIZE = 100
const HISTORY_WINDOW_DAYS = 30

type SkuDayRow = { date: string; qty: number; in_stock: boolean; source: string }

function downloadCsv(filename: string, headers: string[], rows: string[][]) {
  const lines = [headers.join(','), ...rows.map(r => r.map(c => `"${String(c).replace(/"/g, '""')}"`).join(','))]
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function formatDateCol(iso: string) {
  const p = iso.split('-')
  if (p.length === 3) return `${p[2]}/${p[1]}`
  return iso
}

export default function InventoryHistory() {
  const coverage = useSession()
  const [mode, setMode] = useState<'matrix' | 'sku'>('matrix')
  const [channel, setChannel] = useState<InventoryHistoryChannel>('oms')
  const [skuFilter, setSkuFilter] = useState('')
  const [skuQuery, setSkuQuery] = useState('')
  const [skuWindow, setSkuWindow] = useState(30)
  const [page, setPage] = useState(0)
  const [exporting, setExporting] = useState(false)

  const summaryQ = useQuery({
    queryKey: ['inv-history-summary', HISTORY_WINDOW_DAYS],
    queryFn: async () => {
      const { data } = await api.get('/po/daily-inventory-history', {
        params: { days: HISTORY_WINDOW_DAYS },
      })
      return data as {
        loaded?: boolean
        min_date?: string
        max_date?: string
        rows?: number
        skus?: number
        days?: number
        window_days?: number
        window_end?: string
        uploaded_at?: string
        filename?: string
      }
    },
  })

  const matrixQ = useQuery({
    queryKey: ['inv-history-matrix', skuFilter, page, HISTORY_WINDOW_DAYS, channel],
    retry: 1,
    queryFn: async () =>
      getPoDailyInventoryHistoryMatrix(skuFilter, PAGE_SIZE, page * PAGE_SIZE, {
        days: HISTORY_WINDOW_DAYS,
        channel,
      }),
  })

  const skuTimelineQ = useQuery({
    queryKey: ['inv-history-sku', skuQuery, skuWindow],
    enabled: mode === 'sku' && skuQuery.trim().length >= 3,
    queryFn: async () => getPoDailyInventoryHistorySku(skuQuery.trim(), skuWindow),
  })

  const dates = matrixQ.data?.dates ?? []
  const dateTotals = matrixQ.data?.date_totals ?? []
  const matrixRows = matrixQ.data?.rows ?? []
  const totalSkus = matrixQ.data?.total ?? 0
  const inStockMin = matrixQ.data?.in_stock_min_qty ?? 1
  const skuRows = (skuTimelineQ.data?.rows ?? []) as SkuDayRow[]

  const pageCount = Math.max(1, Math.ceil(totalSkus / PAGE_SIZE))

  const handleExportMatrix = async () => {
    setExporting(true)
    try {
      const data = await getPoDailyInventoryHistoryMatrix(skuFilter, 15000, 0, {
        days: HISTORY_WINDOW_DAYS,
        channel,
      })
      const hdr = ['SKU', ...data.dates]
      const body = data.rows.map(r => [r.sku, ...r.qtys.map(q => String(q))])
      const label = [
        channel !== 'combined' ? channel : null,
        skuFilter.trim() || null,
      ]
        .filter(Boolean)
        .join('-')
      downloadCsv(`${label ? `inventory-matrix-${label}` : 'inventory-matrix'}.csv`, hdr, body)
    } finally {
      setExporting(false)
    }
  }

  const handleExportSku = () => {
    downloadCsv(
      `inventory-history-${skuQuery.trim()}.csv`,
      ['date', 'sku', 'qty', 'in_stock', 'source'],
      skuRows.map(r => [r.date, skuQuery.trim(), String(r.qty), r.in_stock ? 'yes' : 'no', r.source]),
    )
  }

  const channelSplitAvailable = matrixQ.data?.channel_split_available ?? false

  const rangeLabel = useMemo(() => {
    if (!dates.length) return ''
    return `${dates[0]} → ${dates[dates.length - 1]}`
  }, [dates])

  return (
    <div className="max-w-[100vw] mx-auto p-4 md:p-6 space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">📅 Inventory History</h1>
          <p className="text-sm text-gray-600 mt-1 max-w-2xl">
            Last {HISTORY_WINDOW_DAYS} snapshot days of daily inventory (wide matrix from Upload → History &amp; setup,
            or snapshot columns from Upload → Daily uploads). Use this matrix to verify on-hand counts match PO{' '}
            <code className="font-mono text-xs">Eff_Days</code>.
          </p>
        </div>
        <Link to="/upload" className="text-sm font-medium text-indigo-700 hover:underline shrink-0">
          Upload snapshot →
        </Link>
      </div>

      <InventoryStalenessBanner />

      <div className="bg-white border border-gray-200 rounded-xl p-4 text-sm text-gray-700 flex flex-wrap gap-x-6 gap-y-1">
        <span>
          <strong>Loaded:</strong>{' '}
          {summaryQ.data?.loaded || coverage.daily_inventory_history ? 'Yes' : 'No'}
        </span>
        {summaryQ.data?.min_date && summaryQ.data?.max_date && (
          <span>
            <strong>Window:</strong> last {summaryQ.data.window_days ?? HISTORY_WINDOW_DAYS} days ·{' '}
            {summaryQ.data.min_date} → {summaryQ.data.max_date}
          </span>
        )}
        {summaryQ.data?.skus != null && (
          <span>
            <strong>SKUs:</strong> {summaryQ.data.skus.toLocaleString()} ·{' '}
            <strong>days:</strong> {summaryQ.data.days?.toLocaleString()}
          </span>
        )}
        <span className="text-gray-500">Today (IST): {todayIsoIST()}</span>
      </div>

      {!summaryQ.data?.loaded && !coverage.daily_inventory_history && (
        <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 p-6 text-sm text-gray-600">
          Upload daily snapshot inventory on{' '}
          <Link to="/upload" className="text-indigo-700 font-medium underline">
            Upload → Daily uploads → Snapshot inventory
          </Link>
          . Each day you upload builds one column in this matrix (last {HISTORY_WINDOW_DAYS} days).
        </div>
      )}

      {summaryQ.data?.loaded && !dates.length && !matrixQ.isLoading && !matrixQ.isFetching && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
          {summaryQ.data.max_date &&
          summaryQ.data.max_date < todayIsoIST().slice(0, 10) ? (
            <>
              Inventory history matrix ends <strong>{summaryQ.data.max_date}</strong> (today{' '}
              {todayIsoIST()}). Re-upload the wide Excel under Upload → History &amp; setup → Daily
              inventory history matrix (PO).
            </>
          ) : (
            <>
              Matrix summary loaded but the table is empty — try refreshing. If this persists after
              upload, use Upload → Server &amp; cache → Reload from server.
            </>
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-xs font-medium text-gray-500 mr-1">Channel:</span>
        {(
          [
            ['combined', 'Combined (max)'],
            ['oms', 'OMS warehouse'],
            ['amazon', 'Amazon FBA'],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => {
              setChannel(key)
              setPage(0)
            }}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium border ${
              channel === key ? 'bg-emerald-600 text-white border-emerald-600' : 'bg-white text-gray-700'
            }`}
          >
            {label}
          </button>
        ))}
        {!channelSplitAvailable && channel !== 'combined' && (
          <span className="text-xs text-amber-700 ml-2">
            OMS/Amazon split needs a re-upload of the wide matrix (OMS + Amazon sheets).
          </span>
        )}
      </div>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setMode('matrix')}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium border ${
            mode === 'matrix' ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-gray-700'
          }`}
        >
          Wide matrix (Excel)
        </button>
        <button
          type="button"
          onClick={() => setMode('sku')}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium border ${
            mode === 'sku' ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-gray-700'
          }`}
        >
          Single SKU timeline
        </button>
      </div>

      {mode === 'matrix' ? (
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
          <div className="p-4 border-b border-gray-100 flex flex-wrap gap-3 items-end">
            <label className="text-sm flex-1 min-w-[12rem] max-w-md">
              <span className="block text-gray-600 mb-1">Filter SKU</span>
              <input
                className="w-full border border-gray-300 rounded-lg px-2 py-1.5 font-mono text-sm"
                placeholder="e.g. 1317YKBLUE (leave empty for all)"
                value={skuFilter}
                onChange={e => {
                  setSkuFilter(e.target.value)
                  setPage(0)
                }}
              />
            </label>
            <button
              type="button"
              onClick={() => void handleExportMatrix()}
              disabled={exporting || !dates.length}
              className="px-3 py-1.5 text-sm font-medium rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-40"
            >
              {exporting ? 'Exporting…' : 'Download CSV (wide)'}
            </button>
          </div>

          {matrixQ.isLoading ? (
            <p className="p-4 text-sm text-gray-500">
              Loading matrix…
              {summaryQ.data?.max_date ? (
                <span className="block text-xs text-gray-400 mt-1">
                  Summary loaded ({summaryQ.data.skus?.toLocaleString() ?? '—'} SKUs through{' '}
                  {summaryQ.data.max_date}) — building table…
                </span>
              ) : null}
            </p>
          ) : matrixQ.isError ? (
            <p className="p-4 text-sm text-red-700">
              Matrix load failed — try refreshing. If this persists, ask admin to reload server cache.
            </p>
          ) : !dates.length ? (
            <p className="p-4 text-sm text-gray-500">No inventory history dates loaded.</p>
          ) : (
            <>
              <div className="px-4 py-2 text-xs text-gray-500 border-b border-gray-50 flex flex-wrap items-center justify-between gap-2">
                <span>
                  {totalSkus.toLocaleString()} SKUs · {dates.length} days
                  {rangeLabel ? ` · ${rangeLabel}` : ''}
                  {channel !== 'combined' ? ` · ${channel.toUpperCase()}` : ''}
                  {skuFilter.trim() ? ` · filter “${skuFilter.trim()}”` : ''}
                </span>
                <span className="flex items-center gap-2">
                  <button
                    type="button"
                    disabled={page <= 0}
                    onClick={() => setPage(p => Math.max(0, p - 1))}
                    className="px-2 py-0.5 rounded border border-gray-300 disabled:opacity-40"
                  >
                    ← Prev
                  </button>
                  <span>
                    Page {page + 1} / {pageCount}
                  </span>
                  <button
                    type="button"
                    disabled={page + 1 >= pageCount}
                    onClick={() => setPage(p => p + 1)}
                    className="px-2 py-0.5 rounded border border-gray-300 disabled:opacity-40"
                  >
                    Next →
                  </button>
                </span>
              </div>
              <div className="overflow-auto max-h-[calc(100vh-18rem)]">
                <table className="text-[11px] border-collapse min-w-full">
                  <thead>
                    <tr className="bg-slate-100">
                      <th className="sticky left-0 z-20 bg-slate-100 border border-gray-200 px-2 py-1.5 text-left font-semibold text-gray-700 min-w-[10rem]">
                        Item SkuCode
                      </th>
                      {dates.map(d => (
                        <th
                          key={d}
                          className="sticky top-0 z-10 bg-slate-100 border border-gray-200 px-1.5 py-1 text-center font-semibold text-gray-600 whitespace-nowrap min-w-[3rem]"
                          title={d}
                        >
                          {formatDateCol(d)}
                        </th>
                      ))}
                    </tr>
                    <tr className="bg-indigo-50">
                      <th className="sticky left-0 z-20 bg-indigo-50 border border-gray-200 px-2 py-1 text-left text-[10px] font-semibold text-indigo-900">
                        Total inv.
                      </th>
                      {dates.map((d, i) => {
                        const total = dateTotals[i] ?? 0
                        return (
                          <th
                            key={`${d}-total`}
                            className="border border-gray-200 px-1 py-1 text-center text-[10px] font-bold text-indigo-900 whitespace-nowrap tabular-nums"
                            title={`${d}: ${total.toLocaleString()} units`}
                          >
                            {total > 0
                              ? total.toLocaleString(undefined, { maximumFractionDigits: 0 })
                              : '—'}
                          </th>
                        )
                      })}
                    </tr>
                    <tr className="bg-slate-50">
                      <th className="sticky left-0 z-20 bg-slate-50 border border-gray-200 px-2 py-0.5 text-left text-[10px] text-gray-400 font-normal">
                        Date
                      </th>
                      {dates.map(d => (
                        <th
                          key={`${d}-sub`}
                          className="border border-gray-200 px-1 py-0.5 text-center text-[10px] text-gray-400 font-normal whitespace-nowrap"
                          title={d}
                        >
                          {d.slice(5)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {matrixRows.map(row => (
                      <tr key={row.sku} className="hover:bg-sky-50/40">
                        <td className="sticky left-0 z-10 bg-white border border-gray-200 px-2 py-1 font-mono text-gray-800 whitespace-nowrap">
                          {row.sku}
                        </td>
                        {row.qtys.map((qty, i) => {
                          const inStock = qty >= inStockMin
                          return (
                            <td
                              key={`${row.sku}-${dates[i]}`}
                              className={`border border-gray-200 px-1.5 py-1 text-right font-mono tabular-nums ${
                                inStock ? 'text-gray-800' : 'text-rose-400 bg-rose-50/60'
                              }`}
                              title={`${dates[i]}: ${qty}`}
                            >
                              {qty > 0 ? qty.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      ) : (
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
          <div className="p-4 border-b border-gray-100 flex flex-wrap gap-3 items-end">
            <label className="text-sm flex-1 min-w-[12rem]">
              <span className="block text-gray-600 mb-1">SKU</span>
              <input
                className="w-full border border-gray-300 rounded-lg px-2 py-1.5 font-mono"
                placeholder="e.g. 1001YKBEIGE-3XL"
                value={skuQuery}
                onChange={e => setSkuQuery(e.target.value)}
              />
            </label>
            <label className="text-sm">
              <span className="block text-gray-600 mb-1">Window (days)</span>
              <input
                type="number"
                min={7}
                max={90}
                className="border border-gray-300 rounded-lg px-2 py-1.5 w-20"
                value={skuWindow}
                onChange={e => setSkuWindow(Number(e.target.value) || 30)}
              />
            </label>
            <button
              type="button"
              onClick={handleExportSku}
              disabled={!skuRows.length}
              className="px-3 py-1.5 text-sm font-medium rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-40"
            >
              Download CSV
            </button>
          </div>
          {skuTimelineQ.isLoading ? (
            <p className="p-4 text-sm text-gray-500">Loading…</p>
          ) : skuQuery.trim().length < 3 ? (
            <p className="p-4 text-sm text-gray-500">Enter at least 3 characters of a SKU.</p>
          ) : skuRows.length === 0 ? (
            <p className="p-4 text-sm text-amber-700">No history rows for this SKU.</p>
          ) : (
            <div className="max-h-[32rem] overflow-auto">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="text-left px-4 py-2 font-semibold text-gray-600">Date</th>
                    <th className="text-right px-4 py-2 font-semibold text-gray-600">On-hand</th>
                    <th className="text-left px-4 py-2 font-semibold text-gray-600">Source</th>
                    <th className="text-left px-4 py-2 font-semibold text-gray-600">Counted?</th>
                  </tr>
                </thead>
                <tbody>
                  {skuRows.map(r => (
                    <tr key={r.date} className={r.in_stock ? '' : 'bg-rose-50/50'}>
                      <td className="px-4 py-1.5 font-mono">{r.date}</td>
                      <td className="px-4 py-1.5 text-right font-mono">{r.qty.toLocaleString()}</td>
                      <td className="px-4 py-1.5">{r.source === 'derived' ? 'auto · sales' : 'uploaded'}</td>
                      <td className="px-4 py-1.5">{r.in_stock ? '✓' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
