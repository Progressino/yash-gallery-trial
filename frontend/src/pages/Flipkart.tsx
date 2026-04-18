import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell,
} from 'recharts'

interface FlipkartData {
  loaded: boolean
  rows?: number
  date_range?: [string, string]
  shipped?: number
  returned?: number
  return_rate?: number
  monthly?: Array<{ Month: string; shipments?: number; refunds?: number }>
  top_skus?: Array<{ sku: string; units: number }>
  by_state?: Array<{ state: string; units: number }>
}

const FK_COLORS = ['#F7971D','#f59e0b','#fbbf24','#fcd34d','#fde68a','#fef3c7']

function KpiCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">{label}</p>
      <p className="text-2xl font-bold text-[#002B5B]">{value}</p>
    </div>
  )
}

export default function Flipkart() {
  const { data, isLoading } = useQuery<FlipkartData>({
    queryKey: ['flipkart-analytics'],
    queryFn: () => api.get('/data/flipkart-analytics').then(r => r.data),
    refetchInterval: 15000,
  })

  if (isLoading) return <div className="flex items-center justify-center h-full text-gray-400">Loading…</div>

  if (!data?.loaded) return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-8">
      <p className="text-5xl">🟡</p>
      <h2 className="text-2xl font-bold text-[#002B5B]">Flipkart Analytics</h2>
      <p className="text-gray-500 max-w-sm">Upload Flipkart master ZIP from the Dashboard to see analytics.</p>
      <a href="/" className="text-sm text-blue-600 underline">Go to Dashboard →</a>
    </div>
  )

  const monthly = (data.monthly ?? []).map(r => ({
    ...r, shipments: r.shipments ?? 0, refunds: r.refunds ?? 0,
  }))

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-[#002B5B]">🟡 Flipkart Analytics</h2>
        <p className="text-gray-400 text-sm mt-1">
          {data.rows?.toLocaleString()} rows &nbsp;·&nbsp; {data.date_range?.[0]} → {data.date_range?.[1]}
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <KpiCard label="Net Units (Final Sale)" value={(data.net_units ?? (data.shipped ?? 0) - (data.returned ?? 0)).toLocaleString()} />
        <KpiCard label="Shipped"     value={(data.shipped  ?? 0).toLocaleString()} />
        <KpiCard label="Returns"     value={(data.returned ?? 0).toLocaleString()} />
        <KpiCard label="Return Rate" value={`${data.return_rate ?? 0}%`} />
        <KpiCard label="Total Rows"  value={(data.rows ?? 0).toLocaleString()} />
      </div>

      {monthly.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
          <h3 className="font-semibold text-[#002B5B] mb-4">Monthly Shipments vs Returns</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={monthly}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Month" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="shipments" fill="#F7971D" name="Shipments" />
              <Bar dataKey="refunds"   fill="#ef4444" name="Returns" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {(data.top_skus ?? []).length > 0 && (
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="font-semibold text-[#002B5B] mb-4">Top 20 SKUs</h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={data.top_skus} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="sku" tick={{ fontSize: 10 }} width={130} />
                <Tooltip />
                <Bar dataKey="units" name="Units">
                  {(data.top_skus ?? []).map((_, i) => <Cell key={i} fill={FK_COLORS[i % FK_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {(data.by_state ?? []).length > 0 && (
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="font-semibold text-[#002B5B] mb-4">Sales by State</h3>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={data.by_state} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="state" tick={{ fontSize: 10 }} width={110} />
                <Tooltip />
                <Bar dataKey="units" fill="#F7971D" name="Units" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}
