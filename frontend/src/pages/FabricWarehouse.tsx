import { useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'
import { useTrackTab } from '../lib/usageTracker'

type Warehouse = 'all' | 'checked' | 'unchecked'
type Direction = 'all' | 'in' | 'out' | 'alloc'
type StockCategory = 'sfg' | 'grey' | 'all'

const CATEGORY_LABEL: Record<StockCategory, string> = {
  sfg: 'SFG (Printed / Dyed)',
  grey: 'Grey Fabric',
  all: 'All',
}
type View = 'stock' | 'ledger'

interface Dashboard {
  from_date: string
  to_date: string
  today: string
  opening: number
  in_qty: number
  out_qty: number
  closing: number
  today_in: number
  today_out: number
  so_allocated: number
  advance_reserved: number
  free: number
  returns_in: number
  adjustments_net: number
  returns_adjustments: number
  pending_unchecked: number
  pending_unchecked_codes: number
  physical_now: number
  ledger_closing_all_time: number
  recon_diff: number
  pcode_count: number
  txn_types: string[]
}

interface StockRow {
  pcode: string
  name: string
  physical: number
  checked: number
  unchecked: number
  so_allocated: number
  advance_reserved: number
  free: number
  opening_pool: number
  location: string
}

interface LedgerRow {
  seq: number
  date: string
  pcode: string
  name: string
  txn_type: string
  ref_type: string
  ref_no: string
  from_wh: string
  to_wh: string
  qty: number
  in_qty: number
  out_qty: number
  alloc_qty: number
  balance: number
  physical_move: boolean
  is_transfer: boolean
  so_ref: string
  jo_ref: string
  user: string
  remarks: string
  link?: { path?: string; query?: string; label?: string }
}

interface LedgerResp {
  rows: LedgerRow[]
  total: number
  opening: number
  closing: number
  total_in: number
  total_out: number
  txn_types: string[]
}

const PAGE = 100
const todayIso = () => new Date().toISOString().slice(0, 10)
const fmt = (n: number | undefined | null) =>
  (Number(n) || 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })

const TYPE_BADGE: Record<string, string> = {
  'GRN Receipt': 'bg-green-100 text-green-800',
  'QC Pass (Unchecked→Checked)': 'bg-blue-100 text-blue-800',
  'QC Reject': 'bg-red-100 text-red-800',
  'Opening Stock': 'bg-indigo-100 text-indigo-800',
  'Opening → Checked': 'bg-indigo-100 text-indigo-800',
  'Purchase MIN Issue': 'bg-orange-100 text-orange-800',
  'JO Fabric Issue': 'bg-orange-100 text-orange-800',
  'JO Fabric Return': 'bg-teal-100 text-teal-800',
  Adjustment: 'bg-gray-100 text-gray-700',
  'SO Allocation': 'bg-amber-100 text-amber-800',
  'Advance Reservation': 'bg-amber-100 text-amber-800',
  'Reservation Released': 'bg-amber-50 text-amber-700',
  'Reservation Consumed (Cutting)': 'bg-amber-50 text-amber-700',
}

interface Filters {
  from: string
  to: string
  pcode: string
  warehouse: Warehouse
  category: StockCategory
  direction: Direction
  txnType: string
  q: string
}

function Kpi({
  label, value, sub, tone, onClick, hint,
}: {
  label: string
  value: string
  sub?: string
  tone?: string
  onClick?: () => void
  hint?: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={hint}
      className={`text-left rounded-xl border bg-white px-3 py-2.5 shadow-sm active:scale-[0.98] transition
        hover:border-[#002B5B] focus:outline-none focus:ring-2 focus:ring-[#002B5B]/30 min-h-[72px] ${tone ?? ''}`}
    >
      <div className="text-[11px] uppercase tracking-wide text-gray-500 leading-tight">{label}</div>
      <div className="text-xl sm:text-2xl font-semibold text-[#002B5B] tabular-nums leading-tight mt-0.5">{value}</div>
      {sub && <div className="text-[10px] text-gray-400 mt-0.5 leading-tight">{sub}</div>}
    </button>
  )
}

export default function FabricWarehouse() {
  const initial = useMemo<Partial<Filters> & { view?: View }>(() => {
    const p = new URLSearchParams(window.location.search)
    const wh = p.get('warehouse')
    const cat = p.get('category')
    return {
      category: (cat === 'grey' || cat === 'all' ? cat : 'sfg') as StockCategory,
      pcode: p.get('pcode') || '',
      warehouse: wh === 'checked' || wh === 'unchecked' ? wh : 'all',
      view: p.get('tab') === 'ledger' ? 'ledger' : undefined,
    }
  }, [])

  const [view, setView] = useState<View>(initial.view ?? 'stock')
  useTrackTab(view)
  const [filters, setFilters] = useState<Filters>({
    from: todayIso(),
    to: todayIso(),
    pcode: initial.pcode ?? '',
    warehouse: (initial.warehouse as Warehouse) ?? 'all',
    category: initial.category ?? 'sfg',
    direction: 'all',
    txnType: '',
    q: '',
  })
  const [page, setPage] = useState(0)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const [exporting, setExporting] = useState<'' | 'csv' | 'xlsx'>('')
  const ledgerRef = useRef<HTMLDivElement>(null)

  const set = (patch: Partial<Filters>) => {
    setPage(0)
    setFilters(f => ({ ...f, ...patch }))
  }

  const params = useMemo(() => {
    const p: Record<string, string> = {
      from_date: filters.from,
      to_date: filters.to,
      warehouse: filters.warehouse,
      stock_category: filters.category,
    }
    if (filters.pcode.trim()) p.pcode = filters.pcode.trim()
    if (filters.txnType) p.txn_type = filters.txnType
    return p
  }, [filters])

  const dash = useQuery<Dashboard>({
    queryKey: ['fw-dashboard', params],
    queryFn: async () => (await api.get('/fabric-warehouse/dashboard', { params })).data,
    staleTime: 15_000,
  })

  const stock = useQuery<{ rows: StockRow[] }>({
    queryKey: ['fw-stock', filters.pcode, filters.category],
    queryFn: async () =>
      (await api.get('/fabric-warehouse/stock', { params: { q: filters.pcode.trim(), stock_category: filters.category } })).data,
    staleTime: 15_000,
  })

  const ledgerParams = useMemo(
    () => ({
      ...params,
      direction: filters.direction,
      q: filters.q.trim(),
      order: 'desc',
      limit: PAGE,
      offset: page * PAGE,
    }),
    [params, filters.direction, filters.q, page],
  )
  const ledger = useQuery<LedgerResp>({
    queryKey: ['fw-ledger', ledgerParams],
    queryFn: async () => (await api.get('/fabric-warehouse/ledger', { params: ledgerParams })).data,
    enabled: view === 'ledger',
    staleTime: 15_000,
    placeholderData: prev => prev,
  })

  const d = dash.data
  const today = todayIso()

  const openLedger = (patch: Partial<Filters>) => {
    setPage(0)
    setFilters(f => ({ ...f, direction: 'all', txnType: '', q: '', ...patch }))
    setView('ledger')
    setTimeout(() => ledgerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
  }

  const exportLedger = async (format: 'csv' | 'xlsx') => {
    setExporting(format)
    try {
      const res = await api.get('/fabric-warehouse/ledger/export', {
        params: { ...params, direction: filters.direction, q: filters.q.trim(), format },
        responseType: 'blob',
      })
      const url = URL.createObjectURL(res.data as Blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `fabric_ledger_${filters.category}_${filters.from}_${filters.to}.${format}`
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      window.alert('Export failed. Please try again.')
    } finally {
      setExporting('')
    }
  }

  const stockRows = useMemo(() => {
    const rows = stock.data?.rows ?? []
    const q = filters.pcode.trim().toLowerCase()
    return q ? rows.filter(r => r.pcode.toLowerCase().includes(q) || r.name.toLowerCase().includes(q)) : rows
  }, [stock.data, filters.pcode])

  const stockTotals = useMemo(
    () => stockRows.reduce(
      (a, r) => ({
        physical: a.physical + r.physical,
        checked: a.checked + r.checked,
        unchecked: a.unchecked + r.unchecked,
        so: a.so + r.so_allocated,
        adv: a.adv + r.advance_reserved,
        free: a.free + r.free,
      }),
      { physical: 0, checked: 0, unchecked: 0, so: 0, adv: 0, free: 0 },
    ),
    [stockRows],
  )

  const L = ledger.data
  const pages = L ? Math.max(1, Math.ceil(L.total / PAGE)) : 1
  const rangeLabel = filters.from === filters.to ? filters.from : `${filters.from} → ${filters.to}`

  const docLink = (r: LedgerRow) =>
    r.link?.path ? `${r.link.path}${r.link.query ? `?${r.link.query}` : ''}` : ''

  return (
    <div className="p-3 sm:p-5 space-y-4 max-w-[1600px] mx-auto print:p-0">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-lg sm:text-xl font-semibold text-[#002B5B]">Fabric Warehouse</h1>
          <p className="text-xs text-gray-500">
            {CATEGORY_LABEL[filters.category]} by P-Code — Checked / Unchecked, allocations and a date-wise ledger from existing documents.
          </p>
        </div>
        <div className="flex gap-2 print:hidden">
          <Link to="/grey?tab=printed-fabric" className="text-xs border rounded-lg px-3 py-1.5 bg-white hover:bg-gray-50">
            Grey › Printed QC / Reserve
          </Link>
          <button
            type="button"
            onClick={() => window.print()}
            className="text-xs border rounded-lg px-3 py-1.5 bg-white hover:bg-gray-50"
          >
            Print
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="rounded-xl border bg-white p-3 print:hidden">
        <div className="flex gap-2">
          <input
            type="search"
            list="fw-pcodes"
            placeholder="Search P-Code / fabric name"
            value={filters.pcode}
            onChange={e => set({ pcode: e.target.value })}
            className="flex-1 min-w-0 border rounded-lg px-3 py-2 text-sm"
          />
          <datalist id="fw-pcodes">
            {(stock.data?.rows ?? []).slice(0, 300).map(r => (
              <option key={r.pcode} value={r.pcode}>{r.name}</option>
            ))}
          </datalist>
          <button
            type="button"
            onClick={() => setFiltersOpen(o => !o)}
            className="sm:hidden border rounded-lg px-3 text-sm bg-gray-50"
          >
            Filters
          </button>
        </div>
        <div className={`${filtersOpen ? 'grid' : 'hidden'} sm:grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 mt-2`}>
          <label className="text-[11px] text-gray-500">From
            <input type="date" value={filters.from} max={filters.to}
              onChange={e => set({ from: e.target.value || today })}
              className="mt-0.5 w-full border rounded-lg px-2 py-1.5 text-sm text-gray-800" />
          </label>
          <label className="text-[11px] text-gray-500">To
            <input type="date" value={filters.to} min={filters.from}
              onChange={e => set({ to: e.target.value || today })}
              className="mt-0.5 w-full border rounded-lg px-2 py-1.5 text-sm text-gray-800" />
          </label>
          <label className="text-[11px] text-gray-500">Stock category
            <select value={filters.category} onChange={e => set({ category: e.target.value as StockCategory })}
              className="mt-0.5 w-full border rounded-lg px-2 py-1.5 text-sm text-gray-800">
              <option value="grey">Grey Fabric</option>
              <option value="sfg">SFG (Printed / Dyed)</option>
              <option value="all">All</option>
            </select>
          </label>
          <label className="text-[11px] text-gray-500">Warehouse
            <select value={filters.warehouse} onChange={e => set({ warehouse: e.target.value as Warehouse })}
              className="mt-0.5 w-full border rounded-lg px-2 py-1.5 text-sm text-gray-800">
              <option value="all">All</option>
              <option value="checked">Checked</option>
              <option value="unchecked">Unchecked</option>
            </select>
          </label>
          <label className="text-[11px] text-gray-500">Direction
            <select value={filters.direction} onChange={e => set({ direction: e.target.value as Direction })}
              className="mt-0.5 w-full border rounded-lg px-2 py-1.5 text-sm text-gray-800">
              <option value="all">All</option>
              <option value="in">IN</option>
              <option value="out">OUT</option>
              <option value="alloc">Allocation only</option>
            </select>
          </label>
          <label className="text-[11px] text-gray-500">Txn type
            <select value={filters.txnType} onChange={e => set({ txnType: e.target.value })}
              className="mt-0.5 w-full border rounded-lg px-2 py-1.5 text-sm text-gray-800">
              <option value="">All types</option>
              {(d?.txn_types ?? []).map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          <label className="text-[11px] text-gray-500">Ref / SO / JO / user
            <input type="search" value={filters.q} onChange={e => set({ q: e.target.value })}
              placeholder="GRN-, JO-, SO-…"
              className="mt-0.5 w-full border rounded-lg px-2 py-1.5 text-sm text-gray-800" />
          </label>
        </div>
      </div>

      {/* KPI strip */}
      <div>
        <div className="flex items-baseline justify-between mb-1.5">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
            {CATEGORY_LABEL[filters.category]} · Period {rangeLabel} · {filters.warehouse === 'all' ? 'All warehouses' : filters.warehouse}
          </h2>
          {dash.isFetching && <span className="text-[10px] text-gray-400">Refreshing…</span>}
        </div>
        {dash.isError ? (
          <div className="rounded-lg border border-red-200 bg-red-50 text-red-700 text-sm p-3">
            Could not load dashboard. <button className="underline" onClick={() => dash.refetch()}>Retry</button>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
            <Kpi label="Opening" value={fmt(d?.opening)} sub="Before period start"
              onClick={() => openLedger({})} />
            <Kpi label="Today IN" value={fmt(d?.today_in)} sub="Receipts, returns, openings"
              tone="border-green-200"
              onClick={() => openLedger({ from: today, to: today, direction: 'in' })} />
            <Kpi label="Today OUT" value={fmt(d?.today_out)} sub="MIN, JO issue, QC reject"
              tone="border-orange-200"
              onClick={() => openLedger({ from: today, to: today, direction: 'out' })} />
            <Kpi label="Closing" value={fmt(d?.closing)} sub={`Opening + IN ${fmt(d?.in_qty)} − OUT ${fmt(d?.out_qty)}`}
              onClick={() => openLedger({})} />
            <Kpi label="Pending Unchecked" value={fmt(d?.pending_unchecked)}
              sub={`${d?.pending_unchecked_codes ?? 0} P-Code(s) awaiting QC`}
              tone="border-blue-200"
              onClick={() => openLedger({ warehouse: 'unchecked', from: '2000-01-01', to: today })} />
            <Kpi label="SO Allocated" value={fmt(d?.so_allocated)} sub="Reserved — not OUT"
              tone="border-amber-200"
              hint="Allocations reduce Free only; they are not physical OUT."
              onClick={() => openLedger({ from: '2000-01-01', to: today, direction: 'alloc', txnType: 'SO Allocation' })} />
            <Kpi label="Advance Reserved" value={fmt(d?.advance_reserved)} sub="Style advance — not OUT"
              tone="border-amber-200"
              onClick={() => openLedger({ from: '2000-01-01', to: today, direction: 'alloc', txnType: 'Advance Reservation' })} />
            <Kpi label="Free" value={fmt(d?.free)} sub="Checked + opening − SO − Advance"
              tone="border-green-300"
              onClick={() => { setView('stock'); }} />
            <Kpi label="Returns / Adj" value={fmt(d?.returns_adjustments)}
              sub={`Returns ${fmt(d?.returns_in)} · Adj ${fmt(d?.adjustments_net)}`}
              onClick={() => openLedger({ txnType: 'JO Fabric Return|Adjustment' })} />
            <Kpi label="Physical now" value={fmt(d?.physical_now)}
              sub={d && Math.abs(d.recon_diff) > 0.01
                ? `Ledger differs by ${fmt(d.recon_diff)}`
                : 'Matches ledger'}
              tone={d && Math.abs(d.recon_diff) > 0.01 ? 'border-red-200' : ''}
              hint="Live stock tables vs sum of ledger movements (reconciliation report is Phase 3)."
              onClick={() => setView('stock')} />
          </div>
        )}
      </div>

      {/* View switch */}
      <div ref={ledgerRef} className="flex items-center justify-between gap-2 print:hidden">
        <div className="inline-flex rounded-lg border bg-white overflow-hidden text-sm">
          {(['stock', 'ledger'] as View[]).map(v => (
            <button key={v} type="button" onClick={() => setView(v)}
              className={`px-4 py-1.5 ${view === v ? 'bg-[#002B5B] text-white' : 'text-gray-700 hover:bg-gray-50'}`}>
              {v === 'stock' ? 'Stock by P-Code' : 'Ledger'}
            </button>
          ))}
        </div>
        {view === 'ledger' && (
          <div className="flex gap-2">
            <button type="button" disabled={!!exporting} onClick={() => exportLedger('xlsx')}
              className="text-xs border rounded-lg px-3 py-1.5 bg-white hover:bg-gray-50 disabled:opacity-50">
              {exporting === 'xlsx' ? 'Exporting…' : 'Excel'}
            </button>
            <button type="button" disabled={!!exporting} onClick={() => exportLedger('csv')}
              className="text-xs border rounded-lg px-3 py-1.5 bg-white hover:bg-gray-50 disabled:opacity-50">
              {exporting === 'csv' ? 'Exporting…' : 'CSV'}
            </button>
          </div>
        )}
      </div>

      {view === 'stock' && (
        <div className="rounded-xl border bg-white overflow-hidden">
          {stock.isLoading ? (
            <div className="p-6 text-sm text-gray-500">Loading stock…</div>
          ) : stockRows.length === 0 ? (
            <div className="p-6 text-sm text-gray-500">No {CATEGORY_LABEL[filters.category]} stock found.</div>
          ) : (
            <>
              <div className="hidden md:block overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-[11px] uppercase text-gray-500">
                    <tr>
                      <th className="text-left px-3 py-2">P-Code</th>
                      <th className="text-left px-3 py-2">Name</th>
                      <th className="text-right px-3 py-2">Physical</th>
                      <th className="text-right px-3 py-2">Checked</th>
                      <th className="text-right px-3 py-2">Unchecked</th>
                      <th className="text-right px-3 py-2">SO Alloc</th>
                      <th className="text-right px-3 py-2">Advance</th>
                      <th className="text-right px-3 py-2">Free</th>
                      <th className="text-left px-3 py-2">Location</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stockRows.map(r => (
                      <tr key={r.pcode} className="border-t hover:bg-blue-50/40 cursor-pointer"
                        onClick={() => openLedger({ pcode: r.pcode, from: '2000-01-01', to: today })}>
                        <td className="px-3 py-2 font-mono font-semibold text-[#002B5B]">{r.pcode}</td>
                        <td className="px-3 py-2 text-gray-600 max-w-[260px] truncate">{r.name}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{fmt(r.physical)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{fmt(r.checked)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{fmt(r.unchecked)}</td>
                        <td className="px-3 py-2 text-right tabular-nums text-amber-700">{fmt(r.so_allocated)}</td>
                        <td className="px-3 py-2 text-right tabular-nums text-amber-700">{fmt(r.advance_reserved)}</td>
                        <td className="px-3 py-2 text-right tabular-nums font-semibold text-green-700">{fmt(r.free)}</td>
                        <td className="px-3 py-2 text-xs text-gray-500">{r.location}</td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot className="bg-gray-50 font-semibold">
                    <tr>
                      <td className="px-3 py-2" colSpan={2}>Total ({stockRows.length})</td>
                      <td className="px-3 py-2 text-right tabular-nums">{fmt(stockTotals.physical)}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{fmt(stockTotals.checked)}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{fmt(stockTotals.unchecked)}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{fmt(stockTotals.so)}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{fmt(stockTotals.adv)}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{fmt(stockTotals.free)}</td>
                      <td />
                    </tr>
                  </tfoot>
                </table>
              </div>
              <ul className="md:hidden divide-y">
                {stockRows.map(r => (
                  <li key={r.pcode}>
                    <button type="button" className="w-full text-left px-3 py-2.5 active:bg-blue-50"
                      onClick={() => openLedger({ pcode: r.pcode, from: '2000-01-01', to: today })}>
                      <div className="flex justify-between gap-2">
                        <span className="font-mono font-semibold text-[#002B5B]">{r.pcode}</span>
                        <span className="text-green-700 font-semibold tabular-nums">Free {fmt(r.free)}</span>
                      </div>
                      <div className="text-xs text-gray-500 truncate">{r.name || '—'} · {r.location}</div>
                      <div className="grid grid-cols-4 gap-1 mt-1 text-[11px] text-gray-600">
                        <span>Phys <b>{fmt(r.physical)}</b></span>
                        <span>Chk <b>{fmt(r.checked)}</b></span>
                        <span>Unchk <b>{fmt(r.unchecked)}</b></span>
                        <span>Rsv <b>{fmt(r.so_allocated + r.advance_reserved)}</b></span>
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}

      {view === 'ledger' && (
        <div className="rounded-xl border bg-white overflow-hidden">
          <div className="px-3 py-2 border-b bg-gray-50 text-xs text-gray-600 flex flex-wrap gap-x-4 gap-y-1">
            <span>Period {rangeLabel}</span>
            <span>Opening <b className="tabular-nums">{fmt(L?.opening)}</b></span>
            <span className="text-green-700">IN <b className="tabular-nums">{fmt(L?.total_in)}</b></span>
            <span className="text-orange-700">OUT <b className="tabular-nums">{fmt(L?.total_out)}</b></span>
            <span>Closing <b className="tabular-nums">{fmt(L?.closing)}</b></span>
            <span className="text-gray-400">{L ? `${L.total} row(s)` : ''}</span>
          </div>
          {ledger.isLoading ? (
            <div className="p-6 text-sm text-gray-500">Loading ledger…</div>
          ) : ledger.isError ? (
            <div className="p-4 text-sm text-red-700">
              Could not load ledger. <button className="underline" onClick={() => ledger.refetch()}>Retry</button>
            </div>
          ) : !L || L.rows.length === 0 ? (
            <div className="p-6 text-sm text-gray-500">No movements for these filters.</div>
          ) : (
            <>
              <div className="hidden md:block overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="bg-gray-50 text-[11px] uppercase text-gray-500">
                    <tr>
                      <th className="text-left px-2 py-2">Date</th>
                      <th className="text-left px-2 py-2">P-Code</th>
                      <th className="text-left px-2 py-2">Type</th>
                      <th className="text-left px-2 py-2">Ref</th>
                      <th className="text-left px-2 py-2">From → To</th>
                      <th className="text-right px-2 py-2">IN</th>
                      <th className="text-right px-2 py-2">OUT</th>
                      <th className="text-right px-2 py-2" title="Reservation — affects Free only">Alloc</th>
                      <th className="text-right px-2 py-2">Bal</th>
                      <th className="text-left px-2 py-2">SO / JO</th>
                      <th className="text-left px-2 py-2">User</th>
                      <th className="text-left px-2 py-2">Remarks</th>
                    </tr>
                  </thead>
                  <tbody>
                    {L.rows.map(r => {
                      const href = docLink(r)
                      return (
                        <tr key={r.seq} className={`border-t ${r.physical_move ? '' : 'bg-amber-50/40'}`}>
                          <td className="px-2 py-1.5 whitespace-nowrap">{r.date}</td>
                          <td className="px-2 py-1.5 font-mono">
                            <button type="button" className="text-[#002B5B] hover:underline"
                              onClick={() => set({ pcode: r.pcode })}>{r.pcode}</button>
                          </td>
                          <td className="px-2 py-1.5">
                            <span className={`px-1.5 py-0.5 rounded text-[10px] ${TYPE_BADGE[r.txn_type] ?? 'bg-gray-100'}`}>
                              {r.txn_type}
                            </span>
                          </td>
                          <td className="px-2 py-1.5">
                            {href ? (
                              <Link to={href} title={r.link?.label} className="text-blue-700 hover:underline font-mono">
                                {r.ref_no || r.link?.label}
                              </Link>
                            ) : <span className="font-mono">{r.ref_no}</span>}
                          </td>
                          <td className="px-2 py-1.5 text-gray-600 whitespace-nowrap">
                            {r.from_wh || '—'} → {r.to_wh || '—'}
                            {r.is_transfer && <span className="ml-1 text-[10px] text-blue-600">(transfer {fmt(r.qty)})</span>}
                          </td>
                          <td className="px-2 py-1.5 text-right tabular-nums text-green-700">{r.in_qty ? fmt(r.in_qty) : ''}</td>
                          <td className="px-2 py-1.5 text-right tabular-nums text-orange-700">{r.out_qty ? fmt(r.out_qty) : ''}</td>
                          <td className="px-2 py-1.5 text-right tabular-nums text-amber-700">
                            {r.alloc_qty ? `${r.alloc_qty > 0 ? '+' : ''}${fmt(r.alloc_qty)}` : ''}
                          </td>
                          <td className="px-2 py-1.5 text-right tabular-nums font-semibold">{fmt(r.balance)}</td>
                          <td className="px-2 py-1.5 text-gray-600">{[r.so_ref, r.jo_ref].filter(Boolean).join(' / ')}</td>
                          <td className="px-2 py-1.5 text-gray-600">{r.user}</td>
                          <td className="px-2 py-1.5 text-gray-500 max-w-[200px] truncate" title={r.remarks}>{r.remarks}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
              <ul className="md:hidden divide-y">
                {L.rows.map(r => {
                  const href = docLink(r)
                  return (
                    <li key={r.seq} className={`px-3 py-2.5 ${r.physical_move ? '' : 'bg-amber-50/40'}`}>
                      <div className="flex justify-between gap-2 items-start">
                        <div>
                          <span className={`px-1.5 py-0.5 rounded text-[10px] ${TYPE_BADGE[r.txn_type] ?? 'bg-gray-100'}`}>
                            {r.txn_type}
                          </span>
                          <div className="font-mono font-semibold text-[#002B5B] text-sm mt-0.5">{r.pcode}</div>
                        </div>
                        <div className="text-right tabular-nums">
                          {r.in_qty > 0 && <div className="text-green-700 font-semibold">+{fmt(r.in_qty)}</div>}
                          {r.out_qty > 0 && <div className="text-orange-700 font-semibold">−{fmt(r.out_qty)}</div>}
                          {r.alloc_qty !== 0 && <div className="text-amber-700 text-xs">alloc {r.alloc_qty > 0 ? '+' : ''}{fmt(r.alloc_qty)}</div>}
                          {r.is_transfer && <div className="text-blue-600 text-xs">transfer {fmt(r.qty)}</div>}
                          <div className="text-[11px] text-gray-500">Bal {fmt(r.balance)}</div>
                        </div>
                      </div>
                      <div className="text-xs text-gray-500 mt-0.5">
                        {r.date} · {r.from_wh || '—'} → {r.to_wh || '—'}
                      </div>
                      <div className="text-xs mt-0.5 flex flex-wrap gap-x-3">
                        {href ? <Link to={href} className="text-blue-700 underline">{r.ref_no || r.link?.label}</Link>
                          : r.ref_no && <span className="font-mono">{r.ref_no}</span>}
                        {(r.so_ref || r.jo_ref) && <span className="text-gray-500">{[r.so_ref, r.jo_ref].filter(Boolean).join(' / ')}</span>}
                        {r.user && <span className="text-gray-400">{r.user}</span>}
                      </div>
                    </li>
                  )
                })}
              </ul>
              <div className="flex items-center justify-between px-3 py-2 border-t text-xs text-gray-600 print:hidden">
                <button type="button" disabled={page === 0} onClick={() => setPage(p => p - 1)}
                  className="border rounded px-3 py-1 disabled:opacity-40">Prev</button>
                <span>Page {page + 1} / {pages}</span>
                <button type="button" disabled={page + 1 >= pages} onClick={() => setPage(p => p + 1)}
                  className="border rounded px-3 py-1 disabled:opacity-40">Next</button>
              </div>
            </>
          )}
        </div>
      )}

      <p className="text-[11px] text-gray-400 print:hidden">
        Read-only view. Reservations (SO / Advance) reduce Free only and are never counted as OUT.
        Stock moves only through GRN, QC, MIN, JO issue / return and adjustments.
      </p>
    </div>
  )
}
