import { useState, useMemo, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api, getCoverage, resetStuckInventoryUpload } from '../api/client'
import { useSession } from '../store/session'
import * as XLSX from 'xlsx'

type MarketplaceRow = {
  key: string
  label: string
  included: boolean
  units: number
  skus: number
  parse_status?: string
}

interface InventoryData {
  loaded: boolean
  upload_in_progress?: boolean
  inventory_upload_status?: string
  inventory_upload_message?: string
  inventory_upload_progress?: number
  inventory_upload_warnings?: string[]
  rows: Array<Record<string, number | string>>
  columns: string[]
  totals?: Record<string, number>
  debug?: Record<string, unknown>
  marketplaces?: MarketplaceRow[]
  missing_marketplace_hints?: string[]
  snapshot_date?: string | null
  snapshot_date_label?: string | null
  snapshot_date_sources?: string[] | null
  snapshot_uploaded_at?: string | null
  total_rows?: number
  offset?: number
  limit?: number
}

function formatSnapshotUploadedAt(iso: string | null | undefined): string | null {
  if (!iso) return null
  try {
    const d = new Date(iso)
    if (Number.isNaN(d.getTime())) return null
    return d.toLocaleString(undefined, {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return null
  }
}

const PAGE_SIZE = 500

export default function Inventory() {
  const coverage = useSession()
  const setCoverage = useSession(s => s.setCoverage)
  const qc = useQueryClient()
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const [clearingStuck, setClearingStuck] = useState(false)

  const handleClearStuckInventory = async () => {
    setClearingStuck(true)
    try {
      await resetStuckInventoryUpload()
      const cov = await getCoverage({ light: true })
      setCoverage(cov)
      void qc.invalidateQueries({ queryKey: ['inventory'] })
    } finally {
      setClearingStuck(false)
    }
  }

  const invUploadRunning =
    coverage.inventory_upload_status === 'running'
  const snapshotToken =
    coverage.inventory_snapshot_uploaded_at ||
    coverage.inventory_snapshot_date ||
    ''

  useQuery({
    queryKey: ['coverage', 'inventory-upload-poll'],
    queryFn: async () => {
      const cov = await getCoverage({ light: true })
      setCoverage(cov)
      if (cov.inventory_upload_status === 'done') {
        void qc.invalidateQueries({ queryKey: ['inventory'] })
      }
      return cov
    },
    enabled: invUploadRunning,
    refetchInterval: invUploadRunning ? 2000 : false,
  })

  const { data, isLoading, isError, error, isFetching, refetch } = useQuery<InventoryData>({
    queryKey: ['inventory', snapshotToken, search, page],
    queryFn: () =>
      api
        .get('/data/inventory', {
          params: {
            search: search.trim() || undefined,
            offset: page * PAGE_SIZE,
            limit: PAGE_SIZE,
          },
          timeout: 120_000,
        })
        .then(r => r.data),
    retry: 1,
    staleTime: 0,
    refetchOnMount: 'always',
    enabled: !invUploadRunning,
  })

  useEffect(() => {
    if (invUploadRunning) return
    void getCoverage({ light: true }).then(cov => {
      setCoverage(cov)
      void qc.invalidateQueries({ queryKey: ['inventory'] })
    })
  }, [invUploadRunning, setCoverage, qc])

  const snapshotLabel =
    data?.snapshot_date_label ||
    data?.snapshot_date ||
    coverage.inventory_snapshot_date_label ||
    coverage.inventory_snapshot_date ||
    null
  const uploadedLabel = formatSnapshotUploadedAt(
    data?.snapshot_uploaded_at || coverage.inventory_snapshot_uploaded_at,
  )
  const snapshotSources =
    data?.snapshot_date_sources ||
    coverage.inventory_snapshot_date_sources ||
    []

  const rows = data?.rows ?? []
  const totalRows = data?.total_rows ?? rows.length
  const invCols = (data?.columns ?? []).filter(c => c !== 'OMS_SKU')
  const totalInventory = useMemo(() => {
    const fromTotals = Object.entries(data?.totals ?? {}).reduce(
      (s, [k, v]) => (k === 'Total_Inventory' ? s + Number(v) : s),
      0,
    )
    if (fromTotals > 0) return fromTotals
    return rows.reduce((s, r) => s + Number(r['Total_Inventory'] ?? 0), 0)
  }, [data?.totals, rows])

  const hasInventory = Boolean(data?.loaded && (data.total_rows ?? data.rows.length) > 0)
  const expectsInventory = Boolean(coverage.inventory)
  const uploadWarnings = [
    ...(data?.inventory_upload_warnings ?? []),
    ...(data?.missing_marketplace_hints ?? []),
    ...(coverage.inventory_upload_warnings ?? []),
  ]
  const uniqueWarnings = [...new Set(uploadWarnings.filter(Boolean))]

  const totalSkus = totalRows
  const zeroStock = rows.filter(r => Number(r['Total_Inventory'] ?? 0) <= 0).length
  const amzDisclaimer = (data?.debug?.amz_disclaimer as Record<string, unknown> | undefined) ?? undefined

  useEffect(() => {
    if (!invUploadRunning) return
    setPage(0)
  }, [invUploadRunning])

  const exportExcel = async () => {
    const { data: full } = await api.get<InventoryData>('/data/inventory', {
      params: { search: search.trim() || undefined, offset: 0, limit: 5000 },
      timeout: 180_000,
    })
    const exportRows = (full?.rows ?? []).map(r => {
      const out: Record<string, string | number> = { 'OMS SKU': String(r['OMS_SKU'] ?? '') }
      invCols.forEach(col => {
        out[col] = Number(r[col] ?? 0)
      })
      return out
    })
    const ws = XLSX.utils.json_to_sheet(exportRows)
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, ws, 'Inventory')
    const datePart = snapshotLabel ? String(snapshotLabel).replace(/\s+/g, '-') : new Date().toISOString().slice(0, 10)
    XLSX.writeFile(wb, `Inventory_${datePart}.xlsx`)
  }

  const pageCount = Math.max(1, Math.ceil(totalRows / PAGE_SIZE))

  if (invUploadRunning || data?.upload_in_progress) {
    const pct = coverage.inventory_upload_progress ?? data?.inventory_upload_progress ?? 0
    const msg =
      coverage.inventory_upload_message ||
      data?.inventory_upload_message ||
      'Parsing inventory snapshot on server…'
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-8 max-w-lg mx-auto">
        <p className="text-5xl">📦</p>
        <h2 className="text-xl font-bold text-[#002B5B]">Updating inventory</h2>
        <p className="text-sm text-gray-600">{msg}</p>
        <div className="w-full max-w-xs h-2 rounded-full bg-gray-200 overflow-hidden">
          <div
            className="h-full bg-[#002B5B] transition-all duration-500"
            style={{ width: `${Math.max(8, pct)}%` }}
          />
        </div>
        <p className="text-xs text-gray-400 tabular-nums">{pct}%</p>
        <p className="text-xs text-gray-500">
          Large RAR bundles can take 1–3 minutes. This page refreshes automatically when parsing finishes.
        </p>
        <button
          type="button"
          disabled={clearingStuck}
          onClick={() => void handleClearStuckInventory()}
          className="mt-2 px-4 py-2 rounded-lg border border-gray-300 bg-white text-sm font-medium text-gray-800 hover:bg-gray-50 disabled:opacity-50"
        >
          {clearingStuck ? 'Clearing…' : 'Clear stuck upload'}
        </button>
      </div>
    )
  }

  if (isLoading && !data) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-2 text-gray-500 p-8">
        <p className="text-sm">Loading inventory snapshot…</p>
        {expectsInventory && (
          <p className="text-xs text-gray-400">Reading snapshot from server cache…</p>
        )}
      </div>
    )
  }

  if (isError) {
    const msg =
      (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
      (error as Error)?.message ||
      'Could not load inventory'
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-8">
        <p className="text-5xl">📦</p>
        <h2 className="text-xl font-bold text-[#002B5B]">Inventory could not load</h2>
        <p className="text-sm text-red-700 max-w-md">{msg}</p>
        <button
          type="button"
          onClick={() => refetch()}
          className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium"
        >
          Retry
        </button>
      </div>
    )
  }

  if (!hasInventory && !expectsInventory) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-8">
        <p className="text-5xl">📦</p>
        <h2 className="text-2xl font-bold text-[#002B5B]">Inventory</h2>
        <p className="text-gray-500 max-w-sm">
          Upload OMS, Flipkart, Myntra, or Amazon inventory CSVs from the Dashboard.
        </p>
        <a href="/upload" className="text-sm text-blue-600 underline">
          Go to Upload Data →
        </a>
      </div>
    )
  }

  if (!hasInventory && expectsInventory) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-8">
        <p className="text-5xl">📦</p>
        <h2 className="text-xl font-bold text-[#002B5B]">Inventory still loading</h2>
        <p className="text-gray-500 max-w-sm text-sm">
          Server shows inventory is loaded, but the table is not ready yet. Try Load Cache on the sidebar, or retry.
        </p>
        <button
          type="button"
          onClick={() => refetch()}
          className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium"
        >
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-[#002B5B]">📦 Inventory</h2>
          <p className="text-gray-400 text-sm mt-1">
            {totalSkus.toLocaleString()} active SKUs
            {isFetching && <span className="ml-2 text-blue-600">· updating…</span>}
          </p>
        </div>
        {snapshotLabel ? (
          <div className="rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-950 min-w-[220px]">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-blue-700">Snapshot as of</p>
            <p className="text-lg font-bold text-[#002B5B] mt-0.5">{snapshotLabel}</p>
            {uploadedLabel && (
              <p className="text-xs text-blue-800/80 mt-1">Updated in app: {uploadedLabel}</p>
            )}
          </div>
        ) : (
          <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 min-w-[220px]">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-amber-800">Snapshot date</p>
            <p className="text-xs mt-1">Unknown — re-upload inventory with a dated filename (e.g. OMS 25-05-2026).</p>
          </div>
        )}
      </div>

      {uniqueWarnings.length > 0 && (
        <ul className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-950 space-y-1 list-disc list-inside">
          {uniqueWarnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}

      {(coverage.inventory_upload_file_results ?? []).some(r => r.status === 'skipped') ? (
        <div className="rounded-lg border border-amber-200 bg-amber-50/90 px-4 py-3 text-sm text-amber-950 space-y-1">
          <p className="font-semibold text-amber-900 text-xs uppercase tracking-wide">
            Skipped files from last inventory upload
          </p>
          <div className="max-h-40 overflow-y-auto space-y-1 text-xs">
            {(coverage.inventory_upload_file_results ?? [])
              .filter(r => r.status === 'skipped')
              .map((r, i) => (
                <p key={i}>
                  <span className="font-medium">{r.filename}</span>
                  {r.reason ? `: ${r.reason}` : ''}
                </p>
              ))}
          </div>
        </div>
      ) : null}

      {coverage.manual_intransit_sheet &&
      (coverage.manual_intransit_parse_report?.skip_details?.length ?? 0) > 0 ? (
        <div className="rounded-lg border border-blue-200 bg-blue-50/80 px-4 py-3 text-sm text-blue-950 space-y-1">
          <p className="font-semibold text-xs uppercase tracking-wide text-blue-800">
            Manual in-transit sheet — skip details
          </p>
          <div className="max-h-32 overflow-y-auto text-xs space-y-0.5">
            {(coverage.manual_intransit_parse_report?.skip_details ?? []).map((d, i) => (
              <p key={i}>
                <span className="font-medium">{d.sheet}</span>: {d.reason}
                {d.rows_affected ? ` (${d.rows_affected} rows)` : ''}
              </p>
            ))}
          </div>
        </div>
      ) : null}

      {snapshotLabel && snapshotSources.length > 0 && (
        <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-600">
          <span className="font-medium text-gray-700">Date from: </span>
          {snapshotSources.join(' · ')}
        </div>
      )}

      {data?.marketplaces && data.marketplaces.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
          <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide mb-3">
            Marketplaces in this snapshot
          </p>
          <p className="text-xs text-gray-500 mb-3">
            Green = stock loaded from your upload. Gray = not in the RAR/OMS bundle (sidebar “loaded” badges are for sales data, not inventory).
          </p>
          <div className="flex flex-wrap gap-2">
            {data.marketplaces
              .filter(m => !['Buffer_Stock', 'OMS_Inventory'].includes(m.key))
              .map(m => (
                <div
                  key={m.key}
                  className={`rounded-lg border px-3 py-2 text-xs min-w-[120px] ${
                    m.included
                      ? 'border-green-200 bg-green-50 text-green-900'
                      : 'border-gray-200 bg-gray-50 text-gray-500'
                  }`}
                >
                  <p className="font-semibold">{m.label}</p>
                  <p className="mt-0.5 tabular-nums">
                    {m.included
                      ? `${m.units.toLocaleString()} units · ${m.skus.toLocaleString()} SKUs`
                      : 'Not in upload'}
                  </p>
                </div>
              ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
          <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">Total Units</p>
          <p className="text-2xl font-bold text-[#002B5B] mt-1">
            {(data?.totals?.Total_Inventory != null
              ? Number(data.totals.Total_Inventory)
              : totalInventory
            ).toLocaleString()}
          </p>
        </div>
        <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
          <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">Active SKUs</p>
          <p className="text-2xl font-bold text-[#002B5B] mt-1">{totalSkus.toLocaleString()}</p>
        </div>
        <div className={`bg-white rounded-xl border p-4 shadow-sm ${zeroStock > 0 ? 'border-red-300' : 'border-gray-200'}`}>
          <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">Out of Stock (this page)</p>
          <p className={`text-2xl font-bold mt-1 ${zeroStock > 0 ? 'text-red-600' : 'text-[#002B5B]'}`}>{zeroStock}</p>
        </div>
      </div>

      {data?.totals && (
        <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
          <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide mb-3">Source Breakdown (all SKUs)</p>
          <div className="flex flex-wrap gap-4">
            {Object.entries(data.totals)
              .filter(([col]) => col !== 'Total_Inventory')
              .map(([col, val]) => (
                <div key={col} className="text-center">
                  <p className="text-xs text-gray-400">{col.replace(/_/g, ' ')}</p>
                  <p className="text-base font-bold text-[#002B5B]">{Number(val).toLocaleString()}</p>
                </div>
              ))}
          </div>
        </div>
      )}

      {amzDisclaimer && (
        <div className="bg-amber-50 rounded-xl border border-amber-200 p-4 shadow-sm text-sm text-amber-900">
          <p className="font-semibold">Amazon inventory disclaimer</p>
          <p className="mt-1 text-xs">
            Only latest 1 report day is used for Amazon inventory.
            {amzDisclaimer.latest_report_date ? ` Latest date: ${String(amzDisclaimer.latest_report_date)}.` : ''}
          </p>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <input
          value={search}
          onChange={e => {
            setSearch(e.target.value)
            setPage(0)
          }}
          placeholder="Search SKU…"
          className="w-full max-w-xs border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300"
        />
        <span className="text-xs text-gray-400">
          {totalRows.toLocaleString()} SKU{totalRows === 1 ? '' : 's'}
          {search.trim() ? ' matching search' : ''}
        </span>
        <button
          type="button"
          onClick={() => exportExcel()}
          className="ml-auto flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold text-white bg-green-600 hover:bg-green-700 shadow-sm"
        >
          ⬇ Export Excel
        </button>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="text-left px-4 py-3 font-semibold text-gray-600 sticky left-0 bg-gray-50 whitespace-nowrap">
                OMS SKU
              </th>
              {invCols.map(col => (
                <th key={col} className="text-right px-4 py-3 font-semibold text-gray-600 whitespace-nowrap">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const total = Number(row['Total_Inventory'] ?? 0)
              return (
                <tr key={`${row['OMS_SKU']}-${i}`} className={`border-b border-gray-100 hover:bg-blue-50 ${total <= 0 ? 'bg-red-50' : ''}`}>
                  <td className="px-4 py-2 font-medium text-gray-800 sticky left-0 bg-inherit whitespace-nowrap">
                    {row['OMS_SKU']}
                  </td>
                  {invCols.map(col => (
                    <td key={col} className="px-4 py-2 text-right text-gray-700">
                      {Number(row[col] ?? 0).toLocaleString()}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
        {totalRows > PAGE_SIZE && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-gray-100 text-xs text-gray-600">
            <span>
              Page {page + 1} of {pageCount} · showing {rows.length} of {totalRows.toLocaleString()} SKUs
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={page <= 0}
                onClick={() => setPage(p => Math.max(0, p - 1))}
                className="px-2 py-1 border rounded disabled:opacity-40"
              >
                ← Prev
              </button>
              <button
                type="button"
                disabled={page >= pageCount - 1}
                onClick={() => setPage(p => p + 1)}
                className="px-2 py-1 border rounded disabled:opacity-40"
              >
                Next →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
