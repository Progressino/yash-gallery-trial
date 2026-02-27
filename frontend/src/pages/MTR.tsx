import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from 'recharts'

interface MTRData {
  loaded: boolean
  rows?: number
  date_range?: [string, string]
  shipped?: number
  returned?: number
  return_rate?: number
  monthly?: Array<{ Month: string; shipments?: number; refunds?: number }>
  top_skus?: Array<{ sku: string; units: number }>
}

function KpiCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">{label}</p>
      <p className="text-3xl font-bold text-[#002B5B]">{value}</p>
      {sub && <p className="text-sm text-gray-400 mt-1">{sub}</p>}
    </div>
  )
}

export default function MTR() {
  const { data, isLoading, error } = useQuery<MTRData>({
    queryKey: ['mtr-analytics'],
    queryFn: () => api.get('/data/mtr-analytics').then(r => r.data),
    refetchInterval: 10000,
  })

  if (isLoading) return (
    <div className="flex items-center justify-center h-full text-gray-400 text-lg">Loading MTR data…</div>
  )

  if (error || !data?.loaded) return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
      <p className="text-5xl">📑</p>
      <h2 className="text-2xl font-bold text-[#002B5B]">MTR Analytics</h2>
      <p className="text-gray-500">Upload Amazon MTR ZIP from the Dashboard to see analytics.</p>
    </div>
  )

  const monthly = (data.monthly ?? []).map(r => ({
    ...r,
    shipments: r.shipments ?? 0,
    refunds:   r.refunds   ?? 0,
  }))

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-[#002B5B]">📑 MTR Analytics</h2>
        <p className="text-gray-400 text-sm mt-1">
          {data.rows?.toLocaleString()} rows &nbsp;·&nbsp;
          {data.date_range?.[0]} → {data.date_range?.[1]}
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard label="Shipped Units"  value={(data.shipped ?? 0).toLocaleString()} />
        <KpiCard label="Returned Units" value={(data.returned ?? 0).toLocaleString()} />
        <KpiCard label="Return Rate"    value={`${data.return_rate ?? 0}%`} />
        <KpiCard label="Total Rows"     value={(data.rows ?? 0).toLocaleString()} />
      </div>

      {/* Monthly chart */}
      {monthly.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
          <h3 className="font-semibold text-[#002B5B] mb-4">Monthly Shipments vs Returns</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={monthly}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Month" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="shipments" fill="#002B5B" name="Shipments" />
              <Bar dataKey="refunds"   fill="#ef4444" name="Returns" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Top SKUs */}
      {(data.top_skus ?? []).length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
          <h3 className="font-semibold text-[#002B5B] mb-4">Top 20 SKUs by Shipments</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data.top_skus} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="sku" tick={{ fontSize: 10 }} width={140} />
              <Tooltip />
              <Bar dataKey="units" fill="#1e40af" name="Units" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
