import { useState, useEffect, useCallback } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell,
} from 'recharts'
import api from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

interface Summary {
  shipped:     number
  returns:     number
  net_units:   number
  return_rate: number
  ads:         number
}

interface MonthRow  { month: string; shipped: number; returns: number; cancels: number; net: number }
interface PlatRow   { platform: string; shipped: number; returns: number; return_rate: number }
interface DailyRow  { date: string; units: number }

interface DeepDiveData {
  loaded:      boolean
  sku:         string
  start_date:  string
  end_date:    string
  summary:     Summary
  monthly:     MonthRow[]
  by_platform: PlatRow[]
  daily:       DailyRow[]
  first_sale:  string | null
  last_sale:   string | null
}

// ── Constants ─────────────────────────────────────────────────────────────────

const PLATFORM_COLORS: Record<string, string> = {
  Amazon:   '#f97316',
  Myntra:   '#ec4899',
  Meesho:   '#a855f7',
  Flipkart: '#eab308',
  Snapdeal: '#ef4444',
}

const PRESET_RANGES = [
  { label: '30D',  days: 30  },
  { label: '90D',  days: 90  },
  { label: '180D', days: 180 },
  { label: '1Y',   days: 365 },
  { label: 'All',  days: 0   },
]

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmt(n: number) { return n.toLocaleString('en-IN') }

function dateNDaysAgo(n: number): string {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().slice(0, 10)
}

function today(): string { return new Date().toISOString().slice(0, 10) }

function fmtMonth(m: string) {
  try {
    const [y, mo] = m.split('-')
    return new Date(Number(y), Number(mo) - 1).toLocaleDateString('en-IN', { month: 'short', year: '2-digit' })
  } catch { return m }
}

// ── KPI Card ─────────────────────────────────────────────────────────────────

function KPICard({ label, value, sub, color = 'text-gray-900' }: {
  label: string; value: string; sub?: string; color?: string
}) {
  return (
    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
      <p className="text-xs text-gray-400 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  )
}

// ── SKU Search ────────────────────────────────────────────────────────────────

function SKUSearch({ value, onChange }: { value: string; onChange: (s: string) => void }) {
  const [q, setQ]           = useState(value)
  const [open, setOpen]     = useState(false)
  const [focused, setFocused] = useState(false)

  const { data: suggestions = [] } = useQuery<string[]>({
    queryKey: ['sku-list', q],
    queryFn:  async () => {
      const { data } = await api.get(`/data/sku-list?q=${encodeURIComponent(q)}&limit=30`)
      return data
    },
    enabled: focused && q.length >= 1,
    staleTime: 60_000,
  })

  useEffect(() => { setQ(value) }, [value])

  return (
    <div className="relative w-full max-w-sm">
      <div className="flex items-center gap-2 border border-gray-200 rounded-xl px-3 py-2 bg-white focus-within:ring-2 focus-within:ring-indigo-400">
        <span className="text-gray-400">🔍</span>
        <input
          className="flex-1 text-sm outline-none bg-transparent"
          placeholder="Search SKU…"
          value={q}
          onChange={e => { setQ(e.target.value); setOpen(true) }}
          onFocus={() => { setFocused(true); setOpen(true) }}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={e => { if (e.key === 'Enter' && q.trim()) { onChange(q.trim()); setOpen(false) } }}
        />
        {q && (
          <button onClick={() => { setQ(''); onChange('') }} className="text-gray-300 hover:text-gray-500 text-xs">✕</button>
        )}
      </div>
      {open && suggestions.length > 0 && (
        <ul className="absolute z-20 w-full mt-1 bg-white border border-gray-200 rounded-xl shadow-lg max-h-56 overflow-y-auto">
          {suggestions.map(s => (
            <li
              key={s}
              onMouseDown={() => { onChange(s); setQ(s); setOpen(false) }}
              className={`px-4 py-2 text-sm cursor-pointer hover:bg-indigo-50 hover:text-indigo-700 ${s === value ? 'bg-indigo-50 font-medium text-indigo-700' : 'text-gray-700'}`}
            >
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// ── Custom Tooltip ────────────────────────────────────────────────────────────

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-lg px-4 py-3 text-xs">
      <p className="font-semibold text-gray-700 mb-2">{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: <span className="font-bold">{fmt(p.value)}</span>
        </p>
      ))}
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function SKUDeepDive() {
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate = useNavigate()

  const skuParam      = searchParams.get('sku') || ''
  const startParam    = searchParams.get('start') || dateNDaysAgo(90)
  const endParam      = searchParams.get('end')   || today()
  const [activeSku, setActiveSku] = useState(skuParam)
  const [start, setStart]         = useState(startParam)
  const [end, setEnd]             = useState(endParam)
  const [activePreset, setPreset] = useState<number | null>(90)

  // Sync URL params → state on initial load
  useEffect(() => {
    if (skuParam) setActiveSku(skuParam)
  }, [skuParam])

  const apply = useCallback((sku: string, s: string, e: string) => {
    const p: Record<string, string> = {}
    if (sku) p.sku   = sku
    if (s)   p.start = s
    if (e)   p.end   = e
    setSearchParams(p, { replace: true })
    setActiveSku(sku)
    setStart(s)
    setEnd(e)
  }, [setSearchParams])

  const handlePreset = (days: number) => {
    setPreset(days)
    const s = days > 0 ? dateNDaysAgo(days) : ''
    const e = days > 0 ? today() : ''
    apply(activeSku, s, e)
  }

  const { data, isLoading, isFetching } = useQuery<DeepDiveData>({
    queryKey: ['sku-deepdive', activeSku, start, end],
    queryFn: async () => {
      const params = new URLSearchParams({ sku: activeSku })
      if (start) params.set('start_date', start)
      if (end)   params.set('end_date', end)
      const { data } = await api.get(`/data/sku-deepdive?${params}`)
      return data
    },
    enabled: !!activeSku,
  })

  const loading = isLoading || isFetching
  const s = data?.summary

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      {/* ── Header ── */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">SKU Deep Dive</h1>
          <p className="text-sm text-gray-400 mt-0.5">Full sales breakdown for any product SKU</p>
        </div>
        <button
          onClick={() => navigate(-1)}
          className="text-sm text-gray-400 hover:text-gray-600 transition"
        >
          ← Back
        </button>
      </div>

      {/* ── Controls ── */}
      <div className="flex flex-wrap gap-3 items-center bg-white rounded-2xl border border-gray-100 shadow-sm p-4">
        <SKUSearch
          value={activeSku}
          onChange={sku => apply(sku, start, end)}
        />

        {/* Preset range pills */}
        <div className="flex gap-1.5 flex-wrap">
          {PRESET_RANGES.map(({ label, days }) => (
            <button
              key={label}
              onClick={() => handlePreset(days)}
              className={`px-3 py-1.5 text-xs rounded-lg font-medium transition ${
                activePreset === days
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Custom date range */}
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <input
            type="date"
            value={start}
            onChange={e => { setPreset(null); setStart(e.target.value) }}
            onBlur={() => apply(activeSku, start, end)}
            className="border border-gray-200 rounded-lg px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-400"
          />
          <span>→</span>
          <input
            type="date"
            value={end}
            onChange={e => { setPreset(null); setEnd(e.target.value) }}
            onBlur={() => apply(activeSku, start, end)}
            className="border border-gray-200 rounded-lg px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-400"
          />
        </div>
      </div>

      {/* ── Empty state ── */}
      {!activeSku && (
        <div className="text-center py-20 text-gray-400">
          <p className="text-4xl mb-4">🔍</p>
          <p className="font-medium text-gray-500">Search for a SKU above to see its full sales breakdown</p>
          <p className="text-sm mt-1">Type at least 1 character to see suggestions</p>
        </div>
      )}

      {/* ── Loading ── */}
      {activeSku && loading && (
        <div className="text-center py-16 text-gray-400 text-sm">Loading deepdive for <strong>{activeSku}</strong>…</div>
      )}

      {/* ── No data ── */}
      {activeSku && !loading && data && !data.loaded && (
        <div className="text-center py-16 text-gray-400">
          <p className="text-3xl mb-3">📭</p>
          <p className="font-medium text-gray-500">No sales data loaded yet</p>
          <p className="text-sm mt-1">Upload your sales files first from the Upload Data page</p>
        </div>
      )}

      {/* ── Zero results ── */}
      {activeSku && !loading && data?.loaded && data.summary.shipped === 0 && (
        <div className="text-center py-16 text-gray-400">
          <p className="text-3xl mb-3">📊</p>
          <p className="font-medium text-gray-500">No orders found for <strong>{activeSku}</strong> in this date range</p>
          <p className="text-sm mt-1">Try expanding the date range or check the SKU spelling</p>
        </div>
      )}

      {/* ── Results ── */}
      {activeSku && !loading && data?.loaded && s && s.shipped > 0 && (
        <>
          {/* SKU banner */}
          <div className="bg-indigo-600 text-white rounded-2xl px-6 py-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs text-indigo-200 mb-0.5">SKU</p>
              <p className="text-xl font-bold tracking-wide">{data.sku}</p>
            </div>
            <div className="flex gap-6 text-sm">
              {data.first_sale && (
                <div>
                  <p className="text-indigo-200 text-xs">First sale</p>
                  <p className="font-semibold">{data.first_sale}</p>
                </div>
              )}
              {data.last_sale && (
                <div>
                  <p className="text-indigo-200 text-xs">Last sale</p>
                  <p className="font-semibold">{data.last_sale}</p>
                </div>
              )}
              <div>
                <p className="text-indigo-200 text-xs">Period</p>
                <p className="font-semibold">{data.start_date} → {data.end_date}</p>
              </div>
            </div>
          </div>

          {/* KPI cards */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <KPICard label="Units Shipped"  value={fmt(s.shipped)}   sub="gross shipments" />
            <KPICard label="Net Units"      value={fmt(s.net_units)} sub="after returns" color={s.net_units >= 0 ? 'text-green-600' : 'text-red-600'} />
            <KPICard label="Returns"        value={fmt(s.returns)}   sub="units returned" />
            <KPICard
              label="Return Rate"
              value={`${s.return_rate}%`}
              sub="of shipped"
              color={s.return_rate > 20 ? 'text-red-600' : s.return_rate > 10 ? 'text-amber-600' : 'text-green-600'}
            />
            <KPICard label="Avg Daily Sales" value={s.ads.toFixed(1)} sub="units / day" />
          </div>

          {/* Monthly trend chart */}
          {data.monthly.length > 0 && (
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
              <h2 className="text-sm font-semibold text-gray-700 mb-4">Monthly Sales Trend</h2>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={data.monthly} barCategoryGap="30%">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                  <XAxis
                    dataKey="month"
                    tickFormatter={fmtMonth}
                    tick={{ fontSize: 11, fill: '#9ca3af' }}
                    axisLine={false} tickLine={false}
                  />
                  <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="shipped" name="Shipped"  fill="#6366f1" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="returns" name="Returns"  fill="#f87171" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Platform breakdown + Daily trend — side by side on wide screens */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

            {/* Platform breakdown */}
            {data.by_platform.length > 0 && (
              <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
                <h2 className="text-sm font-semibold text-gray-700 mb-4">Platform Breakdown</h2>
                <div className="space-y-3">
                  {data.by_platform.map(p => {
                    const total = data.summary.shipped || 1
                    const pct   = Math.round(p.shipped / total * 100)
                    const color = PLATFORM_COLORS[p.platform] ?? '#6366f1'
                    return (
                      <div key={p.platform}>
                        <div className="flex items-center justify-between mb-1 text-xs">
                          <span className="font-medium text-gray-700">{p.platform}</span>
                          <span className="text-gray-500">{fmt(p.shipped)} units &nbsp;·&nbsp; {p.return_rate}% return rate</span>
                        </div>
                        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{ width: `${pct}%`, backgroundColor: color }}
                          />
                        </div>
                        <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                          <span>{pct}% of total</span>
                          <span>{fmt(p.returns)} returns</span>
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Donut-style mini chart */}
                <div className="mt-4">
                  <ResponsiveContainer width="100%" height={140}>
                    <BarChart data={data.by_platform} layout="vertical" barCategoryGap="20%">
                      <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" horizontal={false} />
                      <XAxis type="number" tick={{ fontSize: 10, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
                      <YAxis dataKey="platform" type="category" tick={{ fontSize: 11, fill: '#6b7280' }} axisLine={false} tickLine={false} width={60} />
                      <Tooltip content={<ChartTooltip />} />
                      <Bar dataKey="shipped" name="Shipped" radius={[0, 4, 4, 0]}>
                        {data.by_platform.map(p => (
                          <Cell key={p.platform} fill={PLATFORM_COLORS[p.platform] ?? '#6366f1'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Daily trend */}
            {data.daily.length > 0 && (
              <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
                <h2 className="text-sm font-semibold text-gray-700 mb-4">Daily Shipments</h2>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={data.daily}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis
                      dataKey="date"
                      tickFormatter={d => { try { return new Date(d).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }) } catch { return d } }}
                      tick={{ fontSize: 10, fill: '#9ca3af' }}
                      axisLine={false} tickLine={false}
                      interval="preserveStartEnd"
                    />
                    <YAxis tick={{ fontSize: 11, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    <Line
                      type="monotone"
                      dataKey="units"
                      name="Shipped"
                      stroke="#6366f1"
                      strokeWidth={2}
                      dot={data.daily.length < 60}
                      activeDot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Monthly detail table */}
          {data.monthly.length > 0 && (
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
              <div className="px-5 py-4 border-b border-gray-100">
                <h2 className="text-sm font-semibold text-gray-700">Monthly Detail</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-xs text-gray-400 uppercase">
                    <tr>
                      {['Month', 'Shipped', 'Returns', 'Cancels', 'Net Units', 'Return %'].map(h => (
                        <th key={h} className="px-5 py-3 text-left font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {[...data.monthly].reverse().map(row => {
                      const rr = row.shipped > 0 ? (row.returns / row.shipped * 100).toFixed(1) : '0.0'
                      return (
                        <tr key={row.month} className="hover:bg-gray-50 transition">
                          <td className="px-5 py-3 font-medium text-gray-700">{fmtMonth(row.month)}</td>
                          <td className="px-5 py-3 text-gray-600">{fmt(row.shipped)}</td>
                          <td className="px-5 py-3 text-red-500">{fmt(row.returns)}</td>
                          <td className="px-5 py-3 text-gray-400">{fmt(row.cancels)}</td>
                          <td className={`px-5 py-3 font-semibold ${row.net >= 0 ? 'text-green-600' : 'text-red-600'}`}>{fmt(row.net)}</td>
                          <td className="px-5 py-3">
                            <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                              Number(rr) > 20 ? 'bg-red-100 text-red-700' :
                              Number(rr) > 10 ? 'bg-amber-100 text-amber-700' :
                              'bg-green-100 text-green-700'
                            }`}>{rr}%</span>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
