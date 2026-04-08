import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
} from 'recharts'
import api, { downloadIntelligenceSalesCsv, getCoverage } from '../api/client'
import { useSession } from '../store/session'

interface DailyBreakdownRow { date: string; platform: string; units: number; returns: number }

// ── Types ───────────────────────────────────────────────────────
interface PlatformSummaryItem {
  platform: string
  loaded: boolean
  total_units: number
  total_returns: number
  return_rate: number
  top_sku: string
  trend_direction: 'up' | 'down' | 'flat'
  monthly: { month: string; shipments: number; refunds: number }[]
  by_state: { state: string; units: number }[]
}
interface AnomalyItem {
  type: string
  severity: 'critical' | 'warning' | 'info'
  platform: string
  message: string
  sku?: string
}
interface SalesSummary {
  total_units: number
  total_returns: number
  net_units: number
  return_rate: number
  active_months: number
}
interface TopSku {
  sku: string
  units: number
}

// ── Constants ───────────────────────────────────────────────────
const PLATFORM_COLORS: Record<string, string> = {
  Amazon:   '#002B5B',
  Myntra:   '#E91E63',
  Meesho:   '#9C27B0',
  Flipkart: '#F7971D',
  Snapdeal: '#e53935',
}

const SKU_COLORS = [
  '#002B5B', '#1565C0', '#1976D2', '#1E88E5', '#2196F3',
  '#42A5F5', '#64B5F6', '#0D47A1', '#0277BD', '#01579B',
]

// Fully spelled-out Tailwind classes — no dynamic interpolation
const INTENSITY_CLASSES = [
  'bg-gray-100 text-gray-400',
  'bg-blue-100 text-blue-700',
  'bg-blue-200 text-blue-800',
  'bg-blue-400 text-white',
  'bg-blue-700 text-white',
]

const INDIAN_STATES = [
  'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
  'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
  'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
  'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
  'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
  'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Delhi', 'Jammu & Kashmir',
  'Ladakh', 'Puducherry', 'Chandigarh', 'Lakshadweep', 'Dadra & NH', 'Andaman & Nicobar',
]

// ── Helpers ─────────────────────────────────────────────────────
function fmtMonth(m: string) {
  try {
    const [y, mon] = m.split('-')
    return new Date(+y, +mon - 1).toLocaleString('default', { month: 'short', year: '2-digit' })
  } catch { return m }
}

function mergePlatformMonthly(platforms: PlatformSummaryItem[]) {
  const monthMap: Record<string, Record<string, number>> = {}
  for (const p of platforms) {
    if (!p.loaded) continue
    for (const row of p.monthly) {
      if (!monthMap[row.month]) monthMap[row.month] = {}
      monthMap[row.month][p.platform] = row.shipments
    }
  }
  return Object.entries(monthMap)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([month, vals]) => ({ month: fmtMonth(month), ...vals }))
}

function getIntensityClass(units: number, max: number) {
  if (max === 0 || units === 0) return INTENSITY_CLASSES[0]
  const ratio = units / max
  if (ratio < 0.1) return INTENSITY_CLASSES[1]
  if (ratio < 0.3) return INTENSITY_CLASSES[2]
  if (ratio < 0.6) return INTENSITY_CLASSES[3]
  return INTENSITY_CLASSES[4]
}

// ── Date helpers ─────────────────────────────────────────────────
function toIso(d: Date) { return d.toISOString().split('T')[0] }
function daysAgo(n: number) { const d = new Date(); d.setDate(d.getDate() - n); return toIso(d) }
function monthsAgo(n: number) { const d = new Date(); d.setMonth(d.getMonth() - n); return toIso(d) }
const TODAY = toIso(new Date())

const PRESETS = [
  { label: '30D',  start: () => daysAgo(30) },
  { label: '90D',  start: () => daysAgo(90) },
  { label: '6M',   start: () => monthsAgo(6) },
  { label: '1Y',   start: () => monthsAgo(12) },
  { label: 'All',  start: () => '' },
] as const

// ── Main Dashboard ──────────────────────────────────────────────
export default function Dashboard() {
  const navigate = useNavigate()
  const setCoverage = useSession((s) => s.setCoverage)
  const salesLoaded = useSession((s) => s.sales)
  const [dateStart, setDateStart] = useState(() => daysAgo(90))
  const [dateEnd,   setDateEnd]   = useState(TODAY)
  const [activePreset, setActivePreset] = useState<string>('90D')
  const [deepDiveTab, setDeepDiveTab] = useState<string>('Amazon')
  const [deepViewMode, setDeepViewMode] = useState<'monthly' | 'daily'>('monthly')
  const [heatmapPlatform, setHeatmapPlatform] = useState('Myntra')
  const [hiddenPlatforms, setHiddenPlatforms] = useState<Set<string>>(new Set())
  const [skuSearch, setSkuSearch] = useState('')
  const [topSkuLimit, setTopSkuLimit] = useState(10)
  const [exportingSales, setExportingSales] = useState(false)

  function applyPreset(label: string, startFn: () => string) {
    const s = startFn()
    setDateStart(s)
    setDateEnd(TODAY)
    setActivePreset(label)
  }

  function handleStartChange(val: string) { setDateStart(val); setActivePreset('') }
  function handleEndChange(val: string)   { setDateEnd(val);   setActivePreset('') }
  function togglePlatform(name: string) {
    setHiddenPlatforms(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  // Build query string params for date-filtered endpoints
  const dateParams = useMemo(() => {
    const p = new URLSearchParams({ limit: String(topSkuLimit) })
    if (dateStart) p.set('start_date', dateStart)
    if (dateEnd)   p.set('end_date',   dateEnd)
    return p.toString()
  }, [dateStart, dateEnd, topSkuLimit])

  const summaryParams = useMemo(() => {
    const p = new URLSearchParams({ months: '0' })
    if (dateStart) p.set('start_date', dateStart)
    if (dateEnd)   p.set('end_date',   dateEnd)
    return p.toString()
  }, [dateStart, dateEnd])

  const dailyParams = useMemo(() => {
    const p = new URLSearchParams()
    if (dateStart) p.set('start_date', dateStart)
    if (dateEnd)   p.set('end_date',   dateEnd)
    if (deepDiveTab) p.set('platform', deepDiveTab)
    return p.toString()
  }, [dateStart, dateEnd, deepDiveTab])

  useQuery({
    queryKey: ['coverage'],
    queryFn: async () => { const c = await getCoverage(); setCoverage(c); return c },
    refetchInterval: 30_000,
  })

  const { data: salesSummary, isLoading: loadingSales } = useQuery<SalesSummary>({
    queryKey: ['sales-summary', dateStart, dateEnd],
    queryFn: async () => { const { data } = await api.get(`/data/sales-summary?${summaryParams}`); return data },
    staleTime: 2 * 60 * 1000,
  })

  const { data: topSkusRaw, isLoading: loadingSkus } = useQuery<TopSku[]>({
    queryKey: ['top-skus', dateStart, dateEnd],
    queryFn: async () => { const { data } = await api.get(`/data/top-skus?${dateParams}`); return data },
    staleTime: 2 * 60 * 1000,
  })

  const { data: platformSummary, isLoading: loadingPlatforms } = useQuery<PlatformSummaryItem[]>({
    queryKey: ['platform-summary', dateStart, dateEnd],
    queryFn: async () => { const { data } = await api.get(`/data/platform-summary?${summaryParams}`); return data },
    staleTime: 5 * 60 * 1000,
  })

  const { data: anomalies, isLoading: loadingAnomalies } = useQuery<AnomalyItem[]>({
    queryKey: ['anomalies'],
    queryFn: async () => { const { data } = await api.get('/data/anomalies'); return data },
    staleTime: 2 * 60 * 1000,
  })

  const { data: dailyBreakdown = [] } = useQuery<DailyBreakdownRow[]>({
    queryKey: ['daily-breakdown', dailyParams],
    queryFn: async () => { const { data } = await api.get(`/data/daily-breakdown?${dailyParams}`); return data },
    enabled: deepViewMode === 'daily',
    staleTime: 60 * 1000,
  })

  const platforms = platformSummary ?? []
  const loadedPlatforms = platforms.filter(p => p.loaded)
  const activePlatformCount = loadedPlatforms.length
  const totalUnits = salesSummary?.total_units ?? 0
  const totalReturns = salesSummary?.total_returns ?? 0
  const netUnits = salesSummary?.net_units ?? 0
  const returnRate = salesSummary?.return_rate ?? 0

  // Filter monthly chart data client-side by date range + hidden platforms
  const filteredPlatforms = useMemo(() => {
    const startMonth = dateStart ? dateStart.slice(0, 7) : ''
    const endMonth   = dateEnd   ? dateEnd.slice(0, 7)   : ''
    return platforms.map(p => ({
      ...p,
      monthly: p.monthly.filter(r =>
        (!startMonth || r.month >= startMonth) &&
        (!endMonth   || r.month <= endMonth)
      ),
    }))
  }, [platforms, dateStart, dateEnd])

  const chartData = useMemo(() => mergePlatformMonthly(filteredPlatforms), [filteredPlatforms])

  const exportPlatforms = useMemo(() => {
    if (hiddenPlatforms.size === 0) return undefined
    const names = loadedPlatforms.filter(p => !hiddenPlatforms.has(p.platform)).map(p => p.platform)
    return names.length ? names : undefined
  }, [loadedPlatforms, hiddenPlatforms])

  const allPlatformsHidden =
    loadedPlatforms.length > 0 && loadedPlatforms.every(p => hiddenPlatforms.has(p.platform))

  async function handleDownloadSalesCsv() {
    try {
      setExportingSales(true)
      await downloadIntelligenceSalesCsv({
        startDate: dateStart || undefined,
        endDate: dateEnd || undefined,
        platforms: exportPlatforms,
      })
    } catch (e) {
      window.alert(e instanceof Error ? e.message : 'Download failed')
    } finally {
      setExportingSales(false)
    }
  }

  // Top SKUs with search filter
  const topSkusFiltered = useMemo(() => {
    if (!topSkusRaw) return []
    if (!skuSearch.trim()) return topSkusRaw
    const q = skuSearch.trim().toLowerCase()
    return topSkusRaw.filter(s => s.sku.toLowerCase().includes(q))
  }, [topSkusRaw, skuSearch])

  // Heatmap data
  const heatmapData = platforms.find(p => p.platform === heatmapPlatform)?.by_state ?? []
  const stateMap: Record<string, number> = {}
  for (const s of heatmapData) stateMap[s.state] = s.units
  const maxUnits = Math.max(...heatmapData.map(s => s.units), 0)

  // Deep dive — use filtered monthly
  const deepPlatform = filteredPlatforms.find(p => p.platform === deepDiveTab)
  const deepChartData = (deepPlatform?.monthly ?? []).map(r => ({
    month: fmtMonth(r.month),
    shipments: r.shipments,
    refunds: r.refunds,
  }))

  // Daily breakdown chart data
  const deepDailyData = useMemo(() => {
    const byDate: Record<string, { date: string; units: number; returns: number }> = {}
    for (const row of dailyBreakdown) {
      if (!byDate[row.date]) byDate[row.date] = { date: row.date, units: 0, returns: 0 }
      byDate[row.date].units   += row.units
      byDate[row.date].returns += row.returns
    }
    return Object.values(byDate).sort((a, b) => a.date.localeCompare(b.date))
  }, [dailyBreakdown])

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* ── Header ── */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-2xl font-bold text-[#002B5B]">📊 Intelligence Dashboard</h2>
          <p className="text-gray-500 text-sm mt-1">Real-time cross-platform analytics &amp; insights</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {(['Amazon', 'Myntra', 'Meesho', 'Flipkart'] as const).map(name => {
            const p = platforms.find(x => x.platform === name)
            const loaded = p?.loaded ?? false
            return (
              <span
                key={name}
                className={`text-xs font-medium px-3 py-1 rounded-full border ${
                  loaded
                    ? 'bg-green-50 border-green-300 text-green-700'
                    : 'bg-gray-50 border-gray-200 text-gray-400'
                }`}
              >
                {loaded ? '● ' : '○ '}{name}
              </span>
            )
          })}
        </div>
      </div>

      {/* ── Filter Bar ── */}
      <div className="bg-white rounded-xl border border-gray-200 px-4 py-3 shadow-sm space-y-2">
        {/* Date row */}
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Date</span>
          <div className="flex gap-1">
            {PRESETS.map(({ label, start }) => (
              <button
                key={label}
                onClick={() => applyPreset(label, start)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                  activePreset === label
                    ? 'bg-[#002B5B] text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2 ml-1">
            <input
              type="date" value={dateStart} max={dateEnd || TODAY}
              onChange={e => handleStartChange(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1 text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-300"
            />
            <span className="text-xs text-gray-400">→</span>
            <input
              type="date" value={dateEnd} min={dateStart} max={TODAY}
              onChange={e => handleEndChange(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1 text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-300"
            />
          </div>
          <div className="flex items-center gap-2 ml-auto">
            <button
              type="button"
              onClick={() => void handleDownloadSalesCsv()}
              disabled={!salesLoaded || exportingSales || allPlatformsHidden}
              className="text-xs font-medium px-3 py-1 rounded-md border border-[#002B5B] text-[#002B5B] bg-white hover:bg-blue-50 disabled:opacity-45 disabled:cursor-not-allowed"
              title={
                allPlatformsHidden
                  ? 'Show at least one platform to export.'
                  : 'Download unified sales (Sku + OMS_Sku from your mapping file, date, channel, shipment/refund). Amazon is included even before you upload a map (seller SKU is PL-normalised). Rebuild sales after changing mapping. Hidden platforms are excluded.'
              }
            >
              {exportingSales ? 'Preparing…' : 'Download CSV'}
            </button>
          </div>
        </div>
        {/* Platform toggle row */}
        {loadedPlatforms.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Platforms</span>
            {loadedPlatforms.map(p => {
              const hidden = hiddenPlatforms.has(p.platform)
              return (
                <button
                  key={p.platform}
                  onClick={() => togglePlatform(p.platform)}
                  className={`text-xs font-medium px-3 py-1 rounded-full border transition-all ${
                    hidden
                      ? 'bg-gray-50 border-gray-200 text-gray-400 line-through'
                      : 'border-transparent text-white'
                  }`}
                  style={hidden ? {} : { backgroundColor: PLATFORM_COLORS[p.platform] ?? '#6366F1' }}
                >
                  {p.platform}
                </button>
              )
            })}
            {hiddenPlatforms.size > 0 && (
              <button
                onClick={() => setHiddenPlatforms(new Set())}
                className="text-xs text-blue-600 hover:underline ml-1"
              >
                Show all
              </button>
            )}
          </div>
        )}
      </div>

      {/* ── KPI Strip ── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        <KpiCard label="Total Units Sold" value={loadingSales ? '…' : totalUnits.toLocaleString()} accent="#002B5B" />
        <KpiCard label="Total Returns"    value={loadingSales ? '…' : totalReturns.toLocaleString()} accent="#EF4444" />
        <KpiCard label="Return Rate"      value={loadingSales ? '…' : `${returnRate.toFixed(1)}%`} accent={returnRate > 30 ? '#EF4444' : returnRate > 15 ? '#F59E0B' : '#10B981'} />
        <KpiCard label="Net Units"        value={loadingSales ? '…' : netUnits.toLocaleString()} accent="#10B981" />
        <KpiCard label="Active Platforms" value={loadingPlatforms ? '…' : `${activePlatformCount} / ${platforms.length || 5}`} accent="#6366F1" />
      </div>

      {/* ── Platform Comparison Row ── */}
      <div>
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">Platform Overview</h3>
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
          {(platforms.length > 0 ? platforms.map(p => p.platform) : ['Amazon', 'Myntra', 'Meesho', 'Flipkart', 'Snapdeal']).map(name => {
            const p = platforms.find(x => x.platform === name)
            const color = PLATFORM_COLORS[name] ?? '#6366F1'
            if (loadingPlatforms && !p) {
              return (
                <div key={name} className="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm animate-pulse">
                  <div className="h-1.5 w-full" style={{ backgroundColor: color }} />
                  <div className="p-4 space-y-2">
                    <div className="h-4 bg-gray-200 rounded w-1/2" />
                    <div className="h-8 bg-gray-200 rounded w-3/4" />
                  </div>
                </div>
              )
            }
            const loaded = p?.loaded ?? false
            const rr = p?.return_rate ?? 0
            const rrColor = rr > 30 ? 'text-red-600' : rr > 15 ? 'text-amber-500' : 'text-green-600'
            const trendIcon = p?.trend_direction === 'up' ? '↑' : p?.trend_direction === 'down' ? '↓' : '→'
            const trendColor = p?.trend_direction === 'up' ? 'text-green-600' : p?.trend_direction === 'down' ? 'text-red-500' : 'text-gray-400'
            return (
              <div key={name} className={`bg-white rounded-xl border overflow-hidden shadow-sm relative ${loaded ? 'border-gray-200' : 'border-gray-100'}`}>
                <div className="h-1.5 w-full" style={{ backgroundColor: color }} />
                {!loaded && (
                  <div className="absolute inset-0 top-1.5 bg-white/80 flex items-center justify-center">
                    <span className="text-gray-400 text-xs font-medium">Not Loaded</span>
                  </div>
                )}
                <div className="p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold text-gray-500">{name}</span>
                    <span className={`text-lg font-bold ${trendColor}`}>{trendIcon}</span>
                  </div>
                  <p className="text-2xl font-bold text-gray-800">{(p?.total_units ?? 0).toLocaleString()}</p>
                  <p className="text-xs text-gray-400 mt-0.5">units shipped</p>
                  <div className="mt-3 pt-3 border-t border-gray-100 flex items-center justify-between">
                    <span className={`text-sm font-semibold ${rrColor}`}>{rr.toFixed(1)}% return</span>
                    {p?.top_sku && (
                      <span className="text-xs text-gray-400 truncate max-w-[100px]" title={p.top_sku}>
                        {p.top_sku}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Sales Trend + Anomalies ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Sales Trend Panel */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <h3 className="text-sm font-semibold text-gray-700">Sales Trend by Platform</h3>
            <span className="text-xs text-gray-400">
              {dateStart || 'All time'}{dateEnd ? ` → ${dateEnd}` : ''}
            </span>
          </div>
          {loadingPlatforms ? (
            <div className="h-64 flex items-center justify-center text-gray-400 text-sm animate-pulse">Loading…</div>
          ) : chartData.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-gray-400 text-sm">
              No data loaded — <a href="/upload" className="text-blue-600 ml-1 hover:underline">upload files</a>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
                <XAxis dataKey="month" tick={{ fontSize: 11, fill: '#9CA3AF' }} />
                <YAxis tick={{ fontSize: 11, fill: '#9CA3AF' }} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #E5E7EB' }}
                  formatter={(val: number | undefined) => [(val ?? 0).toLocaleString(), '']}
                />
                <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
                {platforms.filter(p => p.loaded && !hiddenPlatforms.has(p.platform)).map(p => (
                  <Line
                    key={p.platform}
                    type="monotone"
                    dataKey={p.platform}
                    stroke={PLATFORM_COLORS[p.platform]}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    activeDot={{ r: 5 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Anomaly Alerts */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm overflow-y-auto max-h-[380px]">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">⚡ Anomaly Alerts</h3>
          {loadingAnomalies ? (
            <div className="space-y-2">
              {[1, 2, 3].map(i => (
                <div key={i} className="h-14 bg-gray-100 rounded-lg animate-pulse" />
              ))}
            </div>
          ) : !anomalies || anomalies.length === 0 ? (
            <div className="flex items-center gap-3 bg-green-50 border border-green-200 rounded-lg p-3">
              <span className="text-2xl">✅</span>
              <div>
                <p className="text-sm font-medium text-green-800">All Clear</p>
                <p className="text-xs text-green-600">No anomalies detected</p>
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              {anomalies.map((a, i) => {
                const borderColor = a.severity === 'critical' ? 'border-red-400' : a.severity === 'warning' ? 'border-amber-400' : 'border-blue-400'
                const bgColor = a.severity === 'critical' ? 'bg-red-50' : a.severity === 'warning' ? 'bg-amber-50' : 'bg-blue-50'
                const badgeColor = a.severity === 'critical' ? 'bg-red-100 text-red-700' : a.severity === 'warning' ? 'bg-amber-100 text-amber-700' : 'bg-blue-100 text-blue-700'
                return (
                  <div key={i} className={`border-l-4 ${borderColor} ${bgColor} rounded-r-lg p-3`}>
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${badgeColor}`}>
                        {a.severity.toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-500">{a.platform}</span>
                    </div>
                    <p className="text-xs text-gray-700">{a.message}</p>
                    {a.sku && (
                      <code className="text-xs bg-white border border-gray-200 px-1.5 py-0.5 rounded mt-1 inline-block text-gray-600">
                        {a.sku}
                      </code>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── Top SKUs + State Heatmap ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Top SKUs */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
          <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
            <h3 className="text-sm font-semibold text-gray-700">🏆 Top SKUs</h3>
            <select
              value={topSkuLimit}
              onChange={e => setTopSkuLimit(Number(e.target.value))}
              className="text-xs border border-gray-200 rounded px-2 py-1 text-gray-600 focus:outline-none focus:ring-1 focus:ring-blue-300"
            >
              {[10, 20, 30, 50].map(n => (
                <option key={n} value={n}>Top {n}</option>
              ))}
            </select>
          </div>
          <input
            type="text"
            placeholder="Search SKU…"
            value={skuSearch}
            onChange={e => setSkuSearch(e.target.value)}
            className="w-full text-xs border border-gray-200 rounded px-3 py-1.5 mb-3 text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-300"
          />
          {loadingSkus ? (
            <div className="h-64 flex items-center justify-center text-gray-400 text-sm animate-pulse">Loading…</div>
          ) : topSkusFiltered.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-gray-400 text-sm">
              {skuSearch ? 'No SKUs match your search' : 'No SKU data'}
            </div>
          ) : (
            <div>
            <p className="text-[10px] text-blue-500 mb-2">Click any bar to open SKU Deepdive →</p>
            <ResponsiveContainer width="100%" height={Math.max(200, topSkusFiltered.length * 28)}>
              <BarChart
                data={topSkusFiltered}
                layout="vertical"
                margin={{ top: 0, right: 20, left: 0, bottom: 0 }}
                style={{ cursor: 'pointer' }}
                onClick={(e: unknown) => {
                  const ev = e as { activePayload?: { payload?: { sku?: string } }[] }
                  const sku = ev?.activePayload?.[0]?.payload?.sku
                  if (sku) navigate(`/sku-deepdive?sku=${encodeURIComponent(sku)}`)
                }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#F3F4F6" />
                <XAxis type="number" tick={{ fontSize: 11, fill: '#9CA3AF' }} />
                <YAxis
                  type="category"
                  dataKey="sku"
                  width={140}
                  tick={{ fontSize: 10, fill: '#4B5563', cursor: 'pointer' }}
                />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8 }}
                  formatter={(val: number | undefined) => [(val ?? 0).toLocaleString(), 'Units']}
                  cursor={{ fill: 'rgba(0,43,91,0.06)' }}
                />
                <Bar
                  dataKey="units"
                  radius={[0, 4, 4, 0]}
                  onClick={(data: unknown) => {
                    const d = data as { sku?: string }
                    if (d?.sku) navigate(`/sku-deepdive?sku=${encodeURIComponent(d.sku)}`)
                  }}
                >
                  {topSkusFiltered.map((_, index) => (
                    <Cell key={index} fill={SKU_COLORS[index % SKU_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* State Heatmap */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <h3 className="text-sm font-semibold text-gray-700">🗺️ Geographic Distribution</h3>
            <select
              value={heatmapPlatform}
              onChange={e => setHeatmapPlatform(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1 text-gray-600 focus:outline-none focus:ring-1 focus:ring-blue-300"
            >
              {platforms.filter(p => p.loaded && p.by_state?.length > 0).map(p => (
                <option key={p.platform} value={p.platform}>{p.platform}</option>
              ))}
            </select>
          </div>

          {loadingPlatforms ? (
            <div className="h-48 flex items-center justify-center text-gray-400 text-sm animate-pulse">Loading…</div>
          ) : (
            <>
              <div className="grid grid-cols-4 sm:grid-cols-6 gap-1 mb-3">
                {INDIAN_STATES.map(state => {
                  const units = stateMap[state] ?? 0
                  const cls = getIntensityClass(units, maxUnits)
                  return (
                    <div
                      key={state}
                      className={`rounded text-center py-1 px-0.5 text-[9px] font-medium leading-tight truncate ${cls}`}
                      title={`${state}: ${units.toLocaleString()} units`}
                    >
                      {state.length > 8 ? state.slice(0, 7) + '…' : state}
                    </div>
                  )
                })}
              </div>
              {/* Legend */}
              <div className="flex items-center gap-2 mt-2">
                <span className="text-xs text-gray-400">Intensity:</span>
                {['0', 'Low', 'Mid', 'High', 'Peak'].map((label, i) => (
                  <div key={label} className="flex items-center gap-1">
                    <div className={`w-3 h-3 rounded ${INTENSITY_CLASSES[i].split(' ')[0]}`} />
                    <span className="text-xs text-gray-500">{label}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      {/* ── Platform Deep Dive ── */}
      <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
          <h3 className="text-sm font-semibold text-gray-700">🔍 Platform Deep Dive</h3>
          <div className="flex items-center gap-3 flex-wrap">
            {/* Monthly / Daily toggle */}
            <div className="flex rounded-lg overflow-hidden border border-gray-200">
              {(['monthly', 'daily'] as const).map(mode => (
                <button
                  key={mode}
                  onClick={() => setDeepViewMode(mode)}
                  className={`px-3 py-1 text-xs font-medium transition-colors ${
                    deepViewMode === mode
                      ? 'bg-[#002B5B] text-white'
                      : 'bg-white text-gray-500 hover:bg-gray-50'
                  }`}
                >
                  {mode === 'monthly' ? 'Monthly' : 'Daily'}
                </button>
              ))}
            </div>
            {/* Platform tabs */}
            <div className="flex border-b border-gray-200">
              {(platforms.length > 0 ? platforms.map(p => p.platform) : ['Amazon', 'Myntra', 'Meesho', 'Flipkart', 'Snapdeal']).map(name => (
                <button
                  key={name}
                  onClick={() => setDeepDiveTab(name)}
                  className={`px-4 py-2 text-xs font-medium transition-colors ${
                    deepDiveTab === name
                      ? 'border-b-2 text-[#002B5B] -mb-px'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                  style={deepDiveTab === name ? { borderBottomColor: PLATFORM_COLORS[name] ?? '#002B5B' } : {}}
                >
                  {name}
                </button>
              ))}
            </div>
          </div>
        </div>

        {!deepPlatform?.loaded ? (
          <div className="h-48 flex flex-col items-center justify-center gap-2 text-gray-400">
            <span className="text-3xl">📭</span>
            <p className="text-sm">No {deepDiveTab} data loaded</p>
            <a href="/upload" className="text-xs text-blue-600 hover:underline">Go to Upload →</a>
          </div>
        ) : deepViewMode === 'daily' ? (
          deepDailyData.length === 0 ? (
            <div className="h-48 flex items-center justify-center text-gray-400 text-sm">
              No daily data in range — upload daily files to see day-by-day breakdown
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={deepDailyData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
                <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#9CA3AF' }} />
                <YAxis tick={{ fontSize: 11, fill: '#9CA3AF' }} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #E5E7EB' }}
                  formatter={(val: number | undefined) => [(val ?? 0).toLocaleString(), '']}
                />
                <Legend iconType="square" iconSize={10} wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="units"   name="Units"   fill={PLATFORM_COLORS[deepDiveTab] ?? '#002B5B'} radius={[3, 3, 0, 0]} />
                <Bar dataKey="returns" name="Returns" fill="#F87171" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )
        ) : deepChartData.length === 0 ? (
          <div className="h-48 flex items-center justify-center text-gray-400 text-sm">No monthly data available</div>
        ) : (
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={deepChartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
              <XAxis dataKey="month" tick={{ fontSize: 11, fill: '#9CA3AF' }} />
              <YAxis tick={{ fontSize: 11, fill: '#9CA3AF' }} />
              <Tooltip
                contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #E5E7EB' }}
                formatter={(val: number | undefined) => [(val ?? 0).toLocaleString(), '']}
              />
              <Legend iconType="square" iconSize={10} wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="shipments" name="Shipments" fill={PLATFORM_COLORS[deepDiveTab]} radius={[3, 3, 0, 0]} />
              <Bar dataKey="refunds"   name="Refunds"   fill="#F87171" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}

// ── Sub-components ──────────────────────────────────────────────
function KpiCard({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm" style={{ borderLeftColor: accent, borderLeftWidth: 4 }}>
      <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide leading-tight">{label}</p>
      <p className="text-xl font-bold text-gray-800 mt-1 truncate">{value}</p>
    </div>
  )
}
