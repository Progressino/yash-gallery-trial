import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'

interface SnapdealData {
  loaded: boolean
  rows?: number
  companies?: string[]
  date_range?: [string, string]
  shipped?: number
  returned?: number
  return_rate?: number
  monthly?: Array<{ Month: string; shipments?: number; refunds?: number }>
  top_skus?: Array<{ sku: string; units: number }>
  by_state?: Array<{ state: string; units: number }>
}

function KpiCard({ label, value, sub, accent }: {
  label: string
  value: string | number
  sub?: string
  accent?: string
}) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">{label}</p>
      <p className={`text-2xl font-bold ${accent ?? 'text-[#002B5B]'}`}>{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  )
}

const SNAPDEAL_RED   = '#e53935'
const SNAPDEAL_CORAL = '#ff7043'

export default function Snapdeal() {
  const [company, setCompany] = useState<string>('')

  const { data, isLoading } = useQuery<SnapdealData>({
    queryKey: ['snapdeal-analytics', company],
    queryFn: () => api.get('/data/snapdeal-analytics', { params: company ? { company } : {} }).then(r => r.data),
    refetchInterval: 15000,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400">
        Loading…
      </div>
    )
  }

  if (!data?.loaded) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-8">
        <p className="text-5xl">🔴</p>
        <h2 className="text-2xl font-bold text-[#002B5B]">Snapdeal Analytics</h2>
        <p className="text-gray-500 max-w-sm">
          Upload your Snapdeal seller report ZIP from the{' '}
          <a href="/upload" className="text-blue-600 underline">Upload page</a>{' '}
          to see analytics here.
        </p>
        <div className="mt-2 bg-gray-50 rounded-lg p-4 text-xs text-gray-500 text-left max-w-md">
          <p className="font-semibold mb-1">Accepted formats:</p>
          <ul className="space-y-0.5">
            <li>• <strong>OMS order reports</strong> (CSV/ZIP) — has <code>Product Sku Code</code>, <code>Channel Name</code> columns</li>
            <li>• Old AG/PE/YG financial ZIP reports (no SKU data — not recommended)</li>
            <li>• Multiple companies (Yash Gallery, Pushpa Enterprises, Aashirwad) combined — filter by company after upload</li>
          </ul>
        </div>
      </div>
    )
  }

  const monthly = (data.monthly ?? []).map(r => ({
    ...r,
    shipments: r.shipments ?? 0,
    refunds:   r.refunds   ?? 0,
  }))

  const topSkus    = data.top_skus   ?? []
  const byState    = data.by_state   ?? []
  const returnRate = data.return_rate ?? 0

  const companies = data.companies ?? []

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-2xl font-bold text-[#002B5B]">🔴 Snapdeal Analytics</h2>
          <p className="text-gray-400 text-sm mt-1">
            {(data.rows ?? 0).toLocaleString()} rows&nbsp;·&nbsp;
            {data.date_range?.[0]} → {data.date_range?.[1]}
          </p>
        </div>
        <span className={`px-3 py-1 rounded-full text-xs font-bold text-white self-start ${
          returnRate > 25 ? 'bg-red-500' : returnRate > 15 ? 'bg-amber-500' : 'bg-green-500'
        }`}>
          {returnRate}% return rate
        </span>
      </div>

      {/* Company filter */}
      {companies.length > 1 && (
        <div className="flex flex-wrap gap-2 items-center">
          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Company:</span>
          <button
            onClick={() => setCompany('')}
            className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
              company === ''
                ? 'bg-[#002B5B] text-white border-[#002B5B]'
                : 'bg-white text-gray-600 border-gray-300 hover:border-gray-400'
            }`}
          >
            All
          </button>
          {companies.map(c => (
            <button
              key={c}
              onClick={() => setCompany(c === company ? '' : c)}
              className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                company === c
                  ? 'bg-[#e53935] text-white border-[#e53935]'
                  : 'bg-white text-gray-600 border-gray-300 hover:border-gray-400'
              }`}
            >
              {c}
            </button>
          ))}
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard
          label="Shipped Units"
          value={(data.shipped ?? 0).toLocaleString()}
          accent="text-[#e53935]"
        />
        <KpiCard
          label="Returns / RTO"
          value={(data.returned ?? 0).toLocaleString()}
          accent="text-red-700"
        />
        <KpiCard
          label="Return Rate"
          value={`${returnRate}%`}
          accent={returnRate > 25 ? 'text-red-600' : returnRate > 15 ? 'text-amber-600' : 'text-green-600'}
        />
        <KpiCard
          label="Total Records"
          value={(data.rows ?? 0).toLocaleString()}
        />
      </div>

      {/* Monthly Chart */}
      {monthly.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
          <h3 className="font-semibold text-[#002B5B] mb-4">Monthly Shipments vs Returns</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={monthly}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Month" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip formatter={(v: number | undefined) => (v ?? 0).toLocaleString()} />
              <Legend />
              <Bar dataKey="shipments" fill={SNAPDEAL_RED}   name="Shipments" />
              <Bar dataKey="refunds"   fill={SNAPDEAL_CORAL} name="Returns / RTO" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top SKUs */}
        {topSkus.length > 0 && (
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="font-semibold text-[#002B5B] mb-4">Top SKUs by Shipments</h3>
            <div className="space-y-2">
              {topSkus.slice(0, 10).map((s, i) => (
                <div key={s.sku} className="flex items-center gap-3">
                  <span className="text-xs font-bold text-gray-400 w-5 text-right shrink-0">
                    {i + 1}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="text-xs font-medium text-gray-700 truncate">{s.sku}</span>
                      <span className="text-xs font-bold text-gray-900 ml-2 shrink-0">
                        {s.units.toLocaleString()}
                      </span>
                    </div>
                    <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.min(100, (s.units / (topSkus[0]?.units || 1)) * 100)}%`,
                          backgroundColor: SNAPDEAL_RED,
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Top States */}
        {byState.length > 0 && (
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="font-semibold text-[#002B5B] mb-4">Top States by Shipments</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={byState.slice(0, 10)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis
                  type="category"
                  dataKey="state"
                  tick={{ fontSize: 10 }}
                  width={100}
                />
                <Tooltip formatter={(v: number | undefined) => (v ?? 0).toLocaleString()} />
                <Bar dataKey="units" fill={SNAPDEAL_RED} name="Units" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}
