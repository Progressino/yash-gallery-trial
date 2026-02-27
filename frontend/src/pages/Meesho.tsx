import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'

interface MeeshoData {
  loaded: boolean
  rows?: number
  date_range?: [string, string]
  shipped?: number
  returned?: number
  return_rate?: number
  monthly?: Array<{ Month: string; shipments?: number; refunds?: number }>
  by_state?: Array<{ state: string; units: number }>
}

function KpiCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">{label}</p>
      <p className="text-2xl font-bold text-[#002B5B]">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  )
}

export default function Meesho() {
  const { data, isLoading } = useQuery<MeeshoData>({
    queryKey: ['meesho-analytics'],
    queryFn: () => api.get('/data/meesho-analytics').then(r => r.data),
    refetchInterval: 15000,
  })

  if (isLoading) return <div className="flex items-center justify-center h-full text-gray-400">Loading…</div>

  if (!data?.loaded) return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-8">
      <p className="text-5xl">🛒</p>
      <h2 className="text-2xl font-bold text-[#002B5B]">Meesho Analytics</h2>
      <p className="text-gray-500 max-w-sm">Upload Meesho master ZIP from the Dashboard to see analytics.</p>
      <a href="/" className="text-sm text-blue-600 underline">Go to Dashboard →</a>
    </div>
  )

  const monthly = (data.monthly ?? []).map(r => ({
    ...r, shipments: r.shipments ?? 0, refunds: r.refunds ?? 0,
  }))

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-[#002B5B]">🛒 Meesho Analytics</h2>
        <p className="text-gray-400 text-sm mt-1">
          {data.rows?.toLocaleString()} rows &nbsp;·&nbsp; {data.date_range?.[0]} → {data.date_range?.[1]}
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
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
              <Bar dataKey="shipments" fill="#e91e8c" name="Shipments" />
              <Bar dataKey="refunds"   fill="#ef4444"  name="Returns" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {(data.by_state ?? []).length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
          <h3 className="font-semibold text-[#002B5B] mb-4">Top States by Shipments</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.by_state} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="state" tick={{ fontSize: 10 }} width={110} />
              <Tooltip />
              <Bar dataKey="units" fill="#e91e8c" name="Units" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
