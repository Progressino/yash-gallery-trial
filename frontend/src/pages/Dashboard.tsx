import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
} from 'recharts'
import api, { getCoverage } from '../api/client'
import { useSession } from '../store/session'

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
  Amazon: '#002B5B',
  Myntra: '#E91E63',
  Meesho: '#9C27B0',
  Flipkart: '#F7971D',
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

// ── Main Dashboard ──────────────────────────────────────────────
export default function Dashboard() {
  const setCoverage = useSession((s) => s.setCoverage)
  const [timeFilter, setTimeFilter] = useState<'1m' | '3m' | '6m'>('3m')
  const [deepDiveTab, setDeepDiveTab] = useState<'Amazon' | 'Myntra' | 'Meesho' | 'Flipkart'>('Amazon')
  const [heatmapPlatform, setHeatmapPlatform] = useState('Myntra')

  useQuery({
    queryKey: ['coverage'],
    queryFn: async () => { const c = await getCoverage(); setCoverage(c); return c },
    refetchInterval: 30_000,
  })

  const { data: salesSummary, isLoading: loadingSales } = useQuery<SalesSummary>({
    queryKey: ['sales-summary', 3],
    queryFn: async () => { const { data } = await api.get('/data/sales-summary?months=3'); return data },
    staleTime: 5 * 60 * 1000,
  })

  const { data: topSkusRaw, isLoading: loadingSkus } = useQuery<TopSku[]>({
    queryKey: ['top-skus', 10],
    queryFn: async () => { const { data } = await api.get('/data/top-skus?limit=10'); return data },
    staleTime: 5 * 60 * 1000,
  })

  const { data: platformSummary, isLoading: loadingPlatforms } = useQuery<PlatformSummaryItem[]>({
    queryKey: ['platform-summary'],
    queryFn: async () => { const { data } = await api.get('/data/platform-summary'); return data },
    staleTime: 5 * 60 * 1000,
  })

  const { data: anomalies, isLoading: loadingAnomalies } = useQuery<AnomalyItem[]>({
    queryKey: ['anomalies'],
    queryFn: async () => { const { data } = await api.get('/data/anomalies'); return data },
    staleTime: 2 * 60 * 1000,
  })

  const platforms = platformSummary ?? []
  const activePlatforms = platforms.filter(p => p.loaded).length
  const totalUnits = salesSummary?.total_units ?? 0
  const totalReturns = salesSummary?.total_returns ?? 0
  const netUnits = salesSummary?.net_units ?? 0
  const returnRate = salesSummary?.return_rate ?? 0

  // Sliced monthly data based on filter
  const monthsToShow = timeFilter === '1m' ? 1 : timeFilter === '3m' ? 3 : 6
  const slicedPlatforms = platforms.map(p => ({
    ...p,
    monthly: p.monthly.slice(-monthsToShow),
  }))
  const chartData = mergePlatformMonthly(slicedPlatforms)

  // Heatmap data
  const heatmapData = platforms.find(p => p.platform === heatmapPlatform)?.by_state ?? []
  const stateMap: Record<string, number> = {}
  for (const s of heatmapData) stateMap[s.state] = s.units
  const maxUnits = Math.max(...heatmapData.map(s => s.units), 0)

  // Deep dive
  const deepPlatform = platforms.find(p => p.platform === deepDiveTab)
  const deepChartData = (deepPlatform?.monthly ?? []).map(r => ({
    month: fmtMonth(r.month),
    shipments: r.shipments,
    refunds: r.refunds,
  }))

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

      {/* ── KPI Strip ── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        <KpiCard label="Total Units Sold" value={loadingSales ? '…' : totalUnits.toLocaleString()} accent="#002B5B" />
        <KpiCard label="Total Returns"    value={loadingSales ? '…' : totalReturns.toLocaleString()} accent="#EF4444" />
        <KpiCard label="Return Rate"      value={loadingSales ? '…' : `${returnRate.toFixed(1)}%`} accent={returnRate > 30 ? '#EF4444' : returnRate > 15 ? '#F59E0B' : '#10B981'} />
        <KpiCard label="Net Units"        value={loadingSales ? '…' : netUnits.toLocaleString()} accent="#10B981" />
        <KpiCard label="Active Platforms" value={loadingPlatforms ? '…' : `${activePlatforms} / 4`} accent="#6366F1" />
      </div>

      {/* ── Platform Comparison Row ── */}
      <div>
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">Platform Overview</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {(['Amazon', 'Myntra', 'Meesho', 'Flipkart'] as const).map(name => {
            const p = platforms.find(x => x.platform === name)
            const color = PLATFORM_COLORS[name]
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
            <div className="flex gap-1">
              {(['1m', '3m', '6m'] as const).map(f => (
                <button
                  key={f}
                  onClick={() => setTimeFilter(f)}
                  className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                    timeFilter === f
                      ? 'bg-[#002B5B] text-white'
                      : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                  }`}
                >
                  {f === '1m' ? '1 Mo' : f === '3m' ? '3 Mo' : '6 Mo'}
                </button>
              ))}
            </div>
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
                {platforms.filter(p => p.loaded).map(p => (
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
          <h3 className="text-sm font-semibold text-gray-700 mb-4">🏆 Top 10 SKUs</h3>
          {loadingSkus ? (
            <div className="h-64 flex items-center justify-center text-gray-400 text-sm animate-pulse">Loading…</div>
          ) : !topSkusRaw || topSkusRaw.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-gray-400 text-sm">No SKU data</div>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart
                data={topSkusRaw}
                layout="vertical"
                margin={{ top: 0, right: 20, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#F3F4F6" />
                <XAxis type="number" tick={{ fontSize: 11, fill: '#9CA3AF' }} />
                <YAxis
                  type="category"
                  dataKey="sku"
                  width={140}
                  tick={{ fontSize: 10, fill: '#4B5563' }}
                />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8 }}
                  formatter={(val: number | undefined) => [(val ?? 0).toLocaleString(), 'Units']}
                />
                <Bar dataKey="units" radius={[0, 4, 4, 0]}>
                  {topSkusRaw.map((_, index) => (
                    <Cell key={index} fill={SKU_COLORS[index % SKU_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
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
              {['Myntra', 'Meesho', 'Flipkart'].map(p => (
                <option key={p} value={p}>{p}</option>
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
          <div className="flex border-b border-gray-200">
            {(['Amazon', 'Myntra', 'Meesho', 'Flipkart'] as const).map(name => (
              <button
                key={name}
                onClick={() => setDeepDiveTab(name)}
                className={`px-4 py-2 text-xs font-medium transition-colors ${
                  deepDiveTab === name
                    ? 'border-b-2 border-[#002B5B] text-[#002B5B] -mb-px'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                {name}
              </button>
            ))}
          </div>
        </div>

        {!deepPlatform?.loaded ? (
          <div className="h-48 flex flex-col items-center justify-center gap-2 text-gray-400">
            <span className="text-3xl">📭</span>
            <p className="text-sm">No {deepDiveTab} data loaded</p>
            <a href="/upload" className="text-xs text-blue-600 hover:underline">Go to Upload →</a>
          </div>
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
