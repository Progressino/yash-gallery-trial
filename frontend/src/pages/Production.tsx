import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

type MainTab = 'dashboard' | 'run-mrp' | 'requirements' | 'reservations' | 'reports' | 'orders'

interface ProdStats {
  total_jos: number; open_jos: number; in_progress: number
  completed_today: number; soft_reservations: number
}

interface JobOrder {
  id: number; jo_number: string; jo_date: string; so_number: string
  sku: string; sku_name: string; process: string; exec_type: string
  vendor_name: string; so_qty: number; planned_qty: number; output_qty: number
  status: string; expected_completion: string; issued_to: string
}

interface OpenSO {
  so_number: string; so_date: string; buyer: string; delivery_date: string
  status: string; total_qty: number; pending_qty: number; line_count: number; skus: string[]
}

interface MRPBreakdown {
  so_no: string; sku: string; qty_req: number; source: string
}

interface MRPMaterial {
  name: string; type: string; unit: string
  total_req: number; stock: number; reserved: number; available: number
  soft_reserved: number; net_available: number; net_req: number; net_req_with_soft: number
  breakdown: MRPBreakdown[]; level: number
}

interface MRPResult {
  run_time: string | null; so_numbers: string[]
  result: Record<string, MRPMaterial>
}

interface SoftReservationV2 {
  id: number; material_code: string; material_name: string; unit: string
  so_no: string; sku: string; qty: number; status: string; created_at: string
}

// ── Constants ─────────────────────────────────────────────────────────────────

const PROCESSES = ['Cutting', 'Stitching', 'Finishing', 'Embroidery', 'Dyeing', 'Kaja Button', 'Packing', 'Quality Check', 'Other']
const JO_STATUSES = ['Created', 'Material Issued', 'In Progress', 'Partially Completed', 'Completed', 'Closed', 'Cancelled']

const statusColor = (s: string) => {
  if (['Completed', 'Closed'].includes(s)) return 'bg-green-100 text-green-700'
  if (s === 'In Progress') return 'bg-blue-100 text-blue-700'
  if (s === 'Material Issued') return 'bg-purple-100 text-purple-700'
  if (s === 'Cancelled') return 'bg-red-100 text-red-700'
  return 'bg-yellow-50 text-yellow-700'
}

const matStatusBadge = (mat: MRPMaterial) => {
  if (mat.net_req_with_soft <= 0) return <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700">OK</span>
  if (mat.net_req > 0) return <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700">Short</span>
  return <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-700">Partial</span>
}

function exportCSV(data: Array<Record<string, unknown>>, filename: string) {
  if (!data.length) return
  const headers = Object.keys(data[0])
  const rows = data.map(r => headers.map(h => JSON.stringify(r[h] ?? '')).join(','))
  const csv = [headers.join(','), ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function Production() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<MainTab>('dashboard')

  // Job Orders state
  const [showJOForm, setShowJOForm] = useState(false)
  const [filterStatus, setFilterStatus] = useState('')
  const [joForm, setJOForm] = useState({
    so_number: '', sku: '', sku_name: '', process: 'Cutting', exec_type: 'Inhouse',
    vendor_name: '', so_qty: 0, planned_qty: 0, expected_completion: '', issued_to: '', remarks: ''
  })

  // Run MRP state
  const [selectedSOs, setSelectedSOs] = useState<string[]>([])
  const [filterBuyer, setFilterBuyer] = useState('')

  // Requirements state
  const [reqTypeFilter, setReqTypeFilter] = useState('')
  const [reqSearch, setReqSearch] = useState('')
  const [shortageOnly, setShortageOnly] = useState(false)
  const [selectedMaterial, setSelectedMaterial] = useState<string | null>(null)

  // Reservations state
  const [resSubTab, setResSubTab] = useState<'material' | 'so' | 'release'>('material')

  // Reports state
  const [reportType, setReportType] = useState<'all' | 'fabric' | 'accessories' | 'packaging' | 'buyer'>('all')

  // ── Queries ──────────────────────────────────────────────────────────────────

  const { data: stats } = useQuery<ProdStats>({
    queryKey: ['prod-stats'],
    queryFn: () => api.get('/production/stats').then(r => r.data)
  })

  const { data: jos = [] } = useQuery<JobOrder[]>({
    queryKey: ['jos', filterStatus],
    queryFn: () => api.get('/production/orders' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data),
    enabled: tab === 'orders' || tab === 'dashboard'
  })

  const { data: openSOs = [] } = useQuery<OpenSO[]>({
    queryKey: ['mrp-open-sos'],
    queryFn: () => api.get('/production/mrp/open-sos').then(r => r.data),
    enabled: tab === 'run-mrp' || tab === 'dashboard'
  })

  const { data: lastMRP, isLoading: lastMRPLoading } = useQuery<MRPResult>({
    queryKey: ['mrp-last'],
    queryFn: () => api.get('/production/mrp/last').then(r => r.data),
    enabled: tab === 'requirements' || tab === 'reservations' || tab === 'reports'
  })

  const { data: softReservations = [] } = useQuery<SoftReservationV2[]>({
    queryKey: ['mrp-soft-res'],
    queryFn: () => api.get('/production/mrp/soft-reservations').then(r => r.data),
    enabled: tab === 'reservations'
  })

  // ── Mutations ─────────────────────────────────────────────────────────────────

  const createJOMut = useMutation({
    mutationFn: (b: object) => api.post('/production/orders', b),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jos'] })
      qc.invalidateQueries({ queryKey: ['prod-stats'] })
      setShowJOForm(false)
    }
  })

  const updateJOMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/production/orders/${id}`, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['jos'] })
  })

  const runMRPMut = useMutation({
    mutationFn: (body: { so_numbers: string[] }) => api.post('/production/mrp/run', body).then(r => r.data),
    onSuccess: (data: MRPResult) => {
      qc.setQueryData(['mrp-last'], data)
      setTab('requirements')
    }
  })

  const softReserveAllMut = useMutation({
    mutationFn: () => api.post('/production/mrp/soft-reserve-all').then(r => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['mrp-soft-res'] })
      qc.invalidateQueries({ queryKey: ['mrp-last'] })
    }
  })

  const releaseSOResMut = useMutation({
    mutationFn: (so_no: string) => api.delete(`/production/mrp/soft-reservations/${so_no}`).then(r => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['mrp-soft-res'] })
      qc.invalidateQueries({ queryKey: ['mrp-last'] })
    }
  })

  // ── Derived Data ─────────────────────────────────────────────────────────────

  const mrpResult = lastMRP?.result ?? {}
  const mrpMaterials = Object.entries(mrpResult) as [string, MRPMaterial][]

  const filteredMaterials = useMemo(() => {
    return mrpMaterials.filter(([code, mat]) => {
      if (reqTypeFilter && mat.type !== reqTypeFilter) return false
      if (shortageOnly && mat.net_req_with_soft <= 0) return false
      if (reqSearch) {
        const s = reqSearch.toLowerCase()
        if (!code.toLowerCase().includes(s) && !mat.name.toLowerCase().includes(s)) return false
      }
      return true
    })
  }, [mrpMaterials, reqTypeFilter, shortageOnly, reqSearch])

  const buyers = useMemo(() => [...new Set(openSOs.map(s => s.buyer).filter(Boolean))], [openSOs])
  const filteredSOs = filterBuyer ? openSOs.filter(s => s.buyer === filterBuyer) : openSOs

  const matTypes = useMemo(() => [...new Set(mrpMaterials.map(([, m]) => m.type).filter(Boolean))], [mrpMaterials])

  // Group soft reservations by material
  const resByMaterial = useMemo(() => {
    const grouped: Record<string, { total: number; unit: string; sos: string[] }> = {}
    for (const r of softReservations) {
      if (!grouped[r.material_code]) {
        grouped[r.material_code] = { total: 0, unit: r.unit, sos: [] }
      }
      grouped[r.material_code].total += r.qty
      if (!grouped[r.material_code].sos.includes(r.so_no)) {
        grouped[r.material_code].sos.push(r.so_no)
      }
    }
    return grouped
  }, [softReservations])

  // Group soft reservations by SO
  const resBySO = useMemo(() => {
    const grouped: Record<string, { mats: number; total_qty: number }> = {}
    for (const r of softReservations) {
      if (!grouped[r.so_no]) grouped[r.so_no] = { mats: 0, total_qty: 0 }
      grouped[r.so_no].mats += 1
      grouped[r.so_no].total_qty += r.qty
    }
    return grouped
  }, [softReservations])

  // Report filtering
  const reportMaterials = useMemo(() => {
    const TYPE_MAP: Record<string, string[]> = {
      fabric: ['GF', 'RM'],
      accessories: ['ACC'],
      packaging: ['PKG'],
    }
    return mrpMaterials.filter(([, mat]) => {
      if (reportType === 'all') return true
      if (reportType === 'buyer') return true
      const types = TYPE_MAP[reportType] ?? []
      return types.includes(mat.type)
    })
  }, [mrpMaterials, reportType])

  const TABS: [MainTab, string][] = [
    ['dashboard', 'Dashboard'],
    ['run-mrp', 'Run MRP'],
    ['requirements', 'Material Requirements'],
    ['reservations', 'Reservations'],
    ['reports', 'MRP Reports'],
    ['orders', 'Job Orders'],
  ]

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-gray-800">Production &amp; MRP</h1>
        <p className="text-sm text-gray-500">Material Requirement Planning, Job Orders, Reservations</p>
      </div>

      <div className="flex gap-1 bg-gray-100 p-1 rounded-lg flex-wrap">
        {TABS.map(([key, label]) => (
          <button key={key} onClick={() => setTab(key)}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${tab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* ── DASHBOARD ── */}
      {tab === 'dashboard' && (
        <div className="space-y-4">
          {stats && (
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {[
                { label: 'TOTAL JOs', value: stats.total_jos, color: 'text-gray-700' },
                { label: 'OPEN JOs', value: stats.open_jos, color: 'text-yellow-600' },
                { label: 'IN PROGRESS', value: stats.in_progress, color: 'text-blue-600' },
                { label: 'COMPLETED TODAY', value: stats.completed_today, color: 'text-green-600' },
                { label: 'OPEN SOs', value: openSOs.length, color: 'text-purple-600' },
              ].map(({ label, value, color }) => (
                <div key={label} className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
                  <p className={`text-2xl font-bold ${color}`}>{value}</p>
                  <p className="text-xs text-gray-500 mt-1 font-semibold tracking-wide">{label}</p>
                </div>
              ))}
            </div>
          )}

          {/* Open SOs with BOM status */}
          <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
            <div className="flex justify-between items-center mb-3">
              <h3 className="font-semibold text-gray-700 text-sm">Open Sales Orders</h3>
              <button onClick={() => setTab('run-mrp')}
                className="text-xs px-3 py-1.5 bg-[#002B5B] text-white rounded-lg hover:bg-blue-800">
                Run MRP
              </button>
            </div>
            <table className="w-full text-sm">
              <thead className="text-gray-400 text-xs uppercase">
                <tr>{['SO #', 'Buyer', 'Delivery Date', 'Status', 'Pending Qty', 'SKUs'].map(h =>
                  <th key={h} className="text-left px-3 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {openSOs.slice(0, 10).map(so => (
                  <tr key={so.so_number} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-3 py-2 font-medium text-gray-700">{so.so_number}</td>
                    <td className="px-3 py-2 text-gray-600">{so.buyer || '—'}</td>
                    <td className="px-3 py-2 text-gray-500">{so.delivery_date || '—'}</td>
                    <td className="px-3 py-2">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(so.status)}`}>{so.status}</span>
                    </td>
                    <td className="px-3 py-2 font-semibold text-blue-600">{so.pending_qty.toLocaleString()}</td>
                    <td className="px-3 py-2 text-xs text-gray-400">{so.skus.slice(0, 3).join(', ')}{so.skus.length > 3 ? ` +${so.skus.length - 3}` : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {openSOs.length === 0 && <p className="text-center text-gray-400 py-4 text-sm">No open sales orders</p>}
          </div>

          {/* Recent JOs */}
          <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
            <h3 className="font-semibold text-gray-700 mb-3 text-sm">Recent Job Orders</h3>
            {jos.slice(0, 8).map(jo => (
              <div key={jo.id} className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
                <div>
                  <p className="text-sm font-medium text-gray-700">{jo.jo_number} · {jo.process}</p>
                  <p className="text-xs text-gray-400">{jo.sku_name || jo.sku} · SO: {jo.so_number || '—'}</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">{jo.output_qty}/{jo.planned_qty}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(jo.status)}`}>{jo.status}</span>
                </div>
              </div>
            ))}
            {jos.length === 0 && <p className="text-xs text-gray-400">No job orders yet</p>}
          </div>
        </div>
      )}

      {/* ── RUN MRP ── */}
      {tab === 'run-mrp' && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border p-4 space-y-4">
            <div>
              <h3 className="font-semibold text-gray-700">Run MRP</h3>
              <p className="text-xs text-gray-500 mt-1">Select Sales Orders to explode through BOMs and calculate material requirements.</p>
            </div>

            <div className="flex gap-3 items-end flex-wrap">
              <div>
                <label className="text-xs text-gray-500">Filter by Buyer</label>
                <select value={filterBuyer} onChange={e => setFilterBuyer(e.target.value)}
                  className="block border border-gray-200 rounded px-2 py-1.5 text-sm mt-1 min-w-[140px]">
                  <option value="">All Buyers</option>
                  {buyers.map(b => <option key={b}>{b}</option>)}
                </select>
              </div>
              <div className="flex gap-2">
                <button onClick={() => setSelectedSOs(filteredSOs.map(s => s.so_number))}
                  className="px-3 py-1.5 border border-gray-200 rounded text-sm text-gray-600 hover:bg-gray-50">
                  Select All
                </button>
                <button onClick={() => setSelectedSOs([])}
                  className="px-3 py-1.5 border border-gray-200 rounded text-sm text-gray-600 hover:bg-gray-50">
                  Clear
                </button>
              </div>
            </div>

            {/* SO selection table */}
            <div className="border border-gray-100 rounded-lg overflow-hidden max-h-80 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 text-gray-400 text-xs uppercase sticky top-0">
                  <tr>
                    <th className="px-3 py-2 w-8"></th>
                    {['SO #', 'Buyer', 'Delivery', 'Status', 'Pending Qty', 'Lines'].map(h =>
                      <th key={h} className="text-left px-3 py-2">{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {filteredSOs.map(so => (
                    <tr key={so.so_number} className={`border-t border-gray-50 hover:bg-blue-50 cursor-pointer ${selectedSOs.includes(so.so_number) ? 'bg-blue-50' : ''}`}
                      onClick={() => setSelectedSOs(prev =>
                        prev.includes(so.so_number) ? prev.filter(x => x !== so.so_number) : [...prev, so.so_number]
                      )}>
                      <td className="px-3 py-2">
                        <input type="checkbox" checked={selectedSOs.includes(so.so_number)} readOnly
                          className="rounded" />
                      </td>
                      <td className="px-3 py-2 font-medium text-gray-700">{so.so_number}</td>
                      <td className="px-3 py-2 text-gray-600">{so.buyer || '—'}</td>
                      <td className="px-3 py-2 text-gray-500">{so.delivery_date || '—'}</td>
                      <td className="px-3 py-2">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(so.status)}`}>{so.status}</span>
                      </td>
                      <td className="px-3 py-2 font-semibold text-blue-600">{so.pending_qty.toLocaleString()}</td>
                      <td className="px-3 py-2 text-gray-500">{so.line_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {filteredSOs.length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No open sales orders</p>}
            </div>

            {/* Selected summary */}
            {selectedSOs.length > 0 && (
              <div className="bg-blue-50 rounded-lg p-3">
                <p className="text-sm font-medium text-blue-700">{selectedSOs.length} SO(s) selected</p>
                <p className="text-xs text-blue-500 mt-1">{selectedSOs.join(', ')}</p>
              </div>
            )}

            <button
              onClick={() => runMRPMut.mutate({ so_numbers: selectedSOs })}
              disabled={selectedSOs.length === 0 || runMRPMut.isPending}
              className="px-6 py-2.5 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800 disabled:opacity-50 disabled:cursor-not-allowed">
              {runMRPMut.isPending ? 'Running MRP...' : 'Run MRP Now'}
            </button>
          </div>

          {/* Last run summary */}
          {lastMRP && lastMRP.run_time && (
            <div className="bg-white rounded-xl border p-4">
              <p className="text-xs text-gray-500">Last run: <span className="font-medium text-gray-700">{lastMRP.run_time}</span> · {lastMRP.so_numbers.length} SOs · {Object.keys(lastMRP.result).length} materials</p>
              <button onClick={() => setTab('requirements')} className="mt-2 text-xs text-[#002B5B] hover:underline">View Requirements →</button>
            </div>
          )}
        </div>
      )}

      {/* ── MATERIAL REQUIREMENTS ── */}
      {tab === 'requirements' && (
        <div className="space-y-4">
          {lastMRPLoading && <p className="text-sm text-gray-400">Loading MRP results...</p>}

          {!lastMRPLoading && (!lastMRP || !lastMRP.run_time) && (
            <div className="bg-white rounded-xl border p-8 text-center">
              <p className="text-gray-400 text-sm">No MRP results yet.</p>
              <button onClick={() => setTab('run-mrp')} className="mt-3 px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
                Run MRP First
              </button>
            </div>
          )}

          {lastMRP && lastMRP.run_time && (
            <>
              {/* Info bar */}
              <div className="bg-blue-50 rounded-lg p-3 flex items-center justify-between flex-wrap gap-2">
                <div>
                  <p className="text-sm font-medium text-blue-700">MRP Result — {Object.keys(mrpResult).length} Materials</p>
                  <p className="text-xs text-blue-500">Run: {lastMRP.run_time} · SOs: {lastMRP.so_numbers.join(', ')}</p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => exportCSV(
                      filteredMaterials.map(([code, m]) => ({
                        material_code: code, name: m.name, type: m.type, unit: m.unit,
                        total_req: m.total_req, stock: m.stock, available: m.available,
                        soft_reserved: m.soft_reserved, net_req_with_soft: m.net_req_with_soft
                      })),
                      'mrp_requirements.csv'
                    )}
                    className="text-xs px-3 py-1.5 border border-blue-200 text-blue-700 rounded-lg hover:bg-blue-100">
                    Export CSV
                  </button>
                </div>
              </div>

              {/* Filters */}
              <div className="flex gap-3 flex-wrap items-end">
                <div>
                  <label className="text-xs text-gray-500">Type</label>
                  <select value={reqTypeFilter} onChange={e => setReqTypeFilter(e.target.value)}
                    className="block border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">All Types</option>
                    {matTypes.map(t => <option key={t}>{t}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Search</label>
                  <input value={reqSearch} onChange={e => setReqSearch(e.target.value)}
                    placeholder="Code or name..." className="block border border-gray-200 rounded px-2 py-1.5 text-sm mt-1 w-48" />
                </div>
                <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer mt-4">
                  <input type="checkbox" checked={shortageOnly} onChange={e => setShortageOnly(e.target.checked)} className="rounded" />
                  Shortage Only
                </label>
                <p className="text-xs text-gray-400 self-end pb-2">{filteredMaterials.length} of {mrpMaterials.length} materials</p>
              </div>

              {/* Materials table */}
              <div className="bg-white rounded-xl border overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                    <tr>
                      {['Code', 'Name', 'Type', 'Unit', 'Total Req', 'In Stock', 'Available', 'Soft Res', 'Net Req (w/ Soft)', 'Status'].map(h =>
                        <th key={h} className="text-left px-3 py-2">{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {filteredMaterials.map(([code, mat]) => (
                      <>
                        <tr key={code}
                          onClick={() => setSelectedMaterial(selectedMaterial === code ? null : code)}
                          className={`border-t border-gray-50 hover:bg-gray-50 cursor-pointer ${selectedMaterial === code ? 'bg-blue-50' : ''}`}>
                          <td className="px-3 py-2 font-medium text-gray-700">{code}</td>
                          <td className="px-3 py-2 text-gray-600">{mat.name}</td>
                          <td className="px-3 py-2"><span className="text-xs px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded">{mat.type}</span></td>
                          <td className="px-3 py-2 text-gray-500">{mat.unit}</td>
                          <td className="px-3 py-2 font-medium text-gray-700">{mat.total_req.toFixed(2)}</td>
                          <td className="px-3 py-2 text-green-600">{mat.stock.toFixed(2)}</td>
                          <td className="px-3 py-2 text-blue-600">{mat.available.toFixed(2)}</td>
                          <td className="px-3 py-2 text-purple-600">{mat.soft_reserved.toFixed(2)}</td>
                          <td className={`px-3 py-2 font-bold ${mat.net_req_with_soft > 0 ? 'text-red-600' : 'text-green-600'}`}>
                            {mat.net_req_with_soft.toFixed(2)}
                          </td>
                          <td className="px-3 py-2">{matStatusBadge(mat)}</td>
                        </tr>
                        {selectedMaterial === code && (
                          <tr key={`${code}-breakdown`} className="bg-blue-50">
                            <td colSpan={10} className="px-6 py-3">
                              <p className="text-xs font-semibold text-blue-700 mb-2">SO/SKU Breakdown for {code}</p>
                              <table className="w-full text-xs">
                                <thead>
                                  <tr className="text-gray-500">
                                    {['SO Number', 'SKU', 'Qty Required', 'Source'].map(h =>
                                      <th key={h} className="text-left pr-4 pb-1">{h}</th>)}
                                  </tr>
                                </thead>
                                <tbody>
                                  {mat.breakdown.map((bd, i) => (
                                    <tr key={i} className="border-t border-blue-100">
                                      <td className="pr-4 py-1 font-medium text-gray-700">{bd.so_no}</td>
                                      <td className="pr-4 py-1 text-gray-600">{bd.sku}</td>
                                      <td className="pr-4 py-1 text-gray-700">{bd.qty_req.toFixed(3)}</td>
                                      <td className="pr-4 py-1 text-gray-400">{bd.source}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </td>
                          </tr>
                        )}
                      </>
                    ))}
                  </tbody>
                </table>
                {filteredMaterials.length === 0 && (
                  <p className="text-center text-gray-400 py-8 text-sm">No materials match the current filters.</p>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {/* ── RESERVATIONS ── */}
      {tab === 'reservations' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex gap-1 bg-gray-100 p-1 rounded-lg">
              {([['material', 'Material-wise'], ['so', 'SO-wise'], ['release', 'Release']] as [typeof resSubTab, string][]).map(([key, label]) => (
                <button key={key} onClick={() => setResSubTab(key)}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${resSubTab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
                  {label}
                </button>
              ))}
            </div>
            <button
              onClick={() => softReserveAllMut.mutate()}
              disabled={softReserveAllMut.isPending || !lastMRP?.run_time}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 disabled:opacity-50">
              {softReserveAllMut.isPending ? 'Reserving...' : 'Soft Reserve All (from last MRP)'}
            </button>
          </div>

          {resSubTab === 'material' && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-3 border-b bg-gray-50">
                <p className="text-sm font-semibold text-gray-700">{Object.keys(resByMaterial).length} materials with active reservations</p>
              </div>
              <table className="w-full text-sm">
                <thead className="text-gray-400 text-xs uppercase">
                  <tr>{['Material Code', 'Total Reserved', 'Unit', 'SOs'].map(h =>
                    <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
                </thead>
                <tbody>
                  {Object.entries(resByMaterial).map(([code, info]) => (
                    <tr key={code} className="border-t border-gray-50 hover:bg-gray-50">
                      <td className="px-4 py-2 font-medium text-gray-700">{code}</td>
                      <td className="px-4 py-2 font-semibold text-purple-700">{info.total.toFixed(3)}</td>
                      <td className="px-4 py-2 text-gray-500">{info.unit}</td>
                      <td className="px-4 py-2 text-xs text-gray-400">{info.sos.join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {Object.keys(resByMaterial).length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No active reservations</p>}
            </div>
          )}

          {resSubTab === 'so' && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-3 border-b bg-gray-50">
                <p className="text-sm font-semibold text-gray-700">{Object.keys(resBySO).length} SOs with active reservations</p>
              </div>
              <table className="w-full text-sm">
                <thead className="text-gray-400 text-xs uppercase">
                  <tr>{['SO Number', 'Materials Reserved', 'Total Qty'].map(h =>
                    <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
                </thead>
                <tbody>
                  {Object.entries(resBySO).map(([so_no, info]) => (
                    <tr key={so_no} className="border-t border-gray-50 hover:bg-gray-50">
                      <td className="px-4 py-2 font-medium text-gray-700">{so_no}</td>
                      <td className="px-4 py-2 text-gray-600">{info.mats}</td>
                      <td className="px-4 py-2 font-semibold text-purple-700">{info.total_qty.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {Object.keys(resBySO).length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No active reservations</p>}
            </div>
          )}

          {resSubTab === 'release' && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-3 border-b bg-gray-50">
                <p className="text-sm font-semibold text-gray-700">Release Reservations by SO</p>
                <p className="text-xs text-gray-400 mt-1">This will mark all MRP soft reservations for an SO as Released.</p>
              </div>
              <table className="w-full text-sm">
                <thead className="text-gray-400 text-xs uppercase">
                  <tr>{['SO Number', 'Materials', 'Total Qty', 'Action'].map(h =>
                    <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
                </thead>
                <tbody>
                  {Object.entries(resBySO).map(([so_no, info]) => (
                    <tr key={so_no} className="border-t border-gray-50 hover:bg-gray-50">
                      <td className="px-4 py-2 font-medium text-gray-700">{so_no}</td>
                      <td className="px-4 py-2 text-gray-600">{info.mats}</td>
                      <td className="px-4 py-2 text-purple-700">{info.total_qty.toFixed(3)}</td>
                      <td className="px-4 py-2">
                        <button
                          onClick={() => releaseSOResMut.mutate(so_no)}
                          disabled={releaseSOResMut.isPending}
                          className="text-xs px-3 py-1 bg-red-50 text-red-600 rounded hover:bg-red-100">
                          Release
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {Object.keys(resBySO).length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No active reservations to release</p>}
            </div>
          )}
        </div>
      )}

      {/* ── MRP REPORTS ── */}
      {tab === 'reports' && (
        <div className="space-y-4">
          <div className="flex gap-1 bg-gray-100 p-1 rounded-lg flex-wrap">
            {([
              ['all', 'All Materials'],
              ['fabric', 'Fabric'],
              ['accessories', 'Accessories'],
              ['packaging', 'Packaging'],
              ['buyer', 'Buyer-wise'],
            ] as [typeof reportType, string][]).map(([key, label]) => (
              <button key={key} onClick={() => setReportType(key)}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${reportType === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
                {label}
              </button>
            ))}
          </div>

          {(!lastMRP || !lastMRP.run_time) && (
            <div className="bg-white rounded-xl border p-8 text-center">
              <p className="text-gray-400 text-sm">No MRP results yet.</p>
              <button onClick={() => setTab('run-mrp')} className="mt-3 px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
                Run MRP First
              </button>
            </div>
          )}

          {lastMRP && lastMRP.run_time && (
            <>
              <div className="flex justify-between items-center">
                <p className="text-sm text-gray-500">{reportMaterials.length} materials</p>
                <button
                  onClick={() => exportCSV(
                    reportMaterials.map(([code, m]) => ({
                      material_code: code, name: m.name, type: m.type, unit: m.unit,
                      total_req: m.total_req, stock: m.stock, available: m.available,
                      soft_reserved: m.soft_reserved, net_req: m.net_req, net_req_with_soft: m.net_req_with_soft,
                    })),
                    `mrp_report_${reportType}.csv`
                  )}
                  className="text-xs px-3 py-1.5 border border-gray-200 text-gray-600 rounded-lg hover:bg-gray-50">
                  Export CSV
                </button>
              </div>

              {reportType === 'buyer' ? (
                // Buyer-wise: group breakdown by buyer via SO
                (() => {
                  // Build buyer -> material map from breakdowns
                  const soToBuyer: Record<string, string> = {}
                  openSOs.forEach(s => { soToBuyer[s.so_number] = s.buyer || 'Unknown' })

                  const buyerData: Record<string, Record<string, { total_req: number; unit: string }>> = {}
                  mrpMaterials.forEach(([code, mat]) => {
                    mat.breakdown.forEach(bd => {
                      const buyer = soToBuyer[bd.so_no] || 'Unknown'
                      if (!buyerData[buyer]) buyerData[buyer] = {}
                      if (!buyerData[buyer][code]) buyerData[buyer][code] = { total_req: 0, unit: mat.unit }
                      buyerData[buyer][code].total_req += bd.qty_req
                    })
                  })

                  return (
                    <div className="space-y-4">
                      {Object.entries(buyerData).map(([buyer, mats]) => (
                        <div key={buyer} className="bg-white rounded-xl border overflow-hidden">
                          <div className="px-4 py-3 border-b bg-gray-50">
                            <p className="font-semibold text-gray-700 text-sm">{buyer}</p>
                            <p className="text-xs text-gray-400">{Object.keys(mats).length} materials</p>
                          </div>
                          <table className="w-full text-sm">
                            <thead className="text-gray-400 text-xs uppercase">
                              <tr>{['Material', 'Qty Required', 'Unit'].map(h =>
                                <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
                            </thead>
                            <tbody>
                              {Object.entries(mats).map(([code, info]) => (
                                <tr key={code} className="border-t border-gray-50">
                                  <td className="px-4 py-2 text-gray-700">{code}</td>
                                  <td className="px-4 py-2 font-medium text-gray-700">{info.total_req.toFixed(3)}</td>
                                  <td className="px-4 py-2 text-gray-500">{info.unit}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ))}
                    </div>
                  )
                })()
              ) : (
                <div className="bg-white rounded-xl border overflow-hidden">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                      <tr>
                        {['Code', 'Name', 'Type', 'Unit', 'Total Req', 'Stock', 'Available', 'Net Req (w/ Soft)', 'Status'].map(h =>
                          <th key={h} className="text-left px-3 py-2">{h}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {reportMaterials.map(([code, mat]) => (
                        <tr key={code} className="border-t border-gray-50 hover:bg-gray-50">
                          <td className="px-3 py-2 font-medium text-gray-700">{code}</td>
                          <td className="px-3 py-2 text-gray-600">{mat.name}</td>
                          <td className="px-3 py-2"><span className="text-xs px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded">{mat.type}</span></td>
                          <td className="px-3 py-2 text-gray-500">{mat.unit}</td>
                          <td className="px-3 py-2 font-medium text-gray-700">{mat.total_req.toFixed(2)}</td>
                          <td className="px-3 py-2 text-green-600">{mat.stock.toFixed(2)}</td>
                          <td className="px-3 py-2 text-blue-600">{mat.available.toFixed(2)}</td>
                          <td className={`px-3 py-2 font-bold ${mat.net_req_with_soft > 0 ? 'text-red-600' : 'text-green-600'}`}>
                            {mat.net_req_with_soft.toFixed(2)}
                          </td>
                          <td className="px-3 py-2">{matStatusBadge(mat)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {reportMaterials.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No materials for this report type.</p>}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ── JOB ORDERS ── */}
      {tab === 'orders' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All Statuses</option>
              {JO_STATUSES.map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={() => setShowJOForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ New JO</button>
          </div>

          {showJOForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Job Order</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {([['so_number', 'SO Number'], ['sku', 'SKU Code'], ['sku_name', 'SKU Name'], ['issued_to', 'Issued To'], ['remarks', 'Remarks']] as [string, string][]).map(([k, l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(joForm as Record<string, string | number>)[k] as string}
                      onChange={e => setJOForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                <div><label className="text-xs text-gray-500">Process</label>
                  <select value={joForm.process} onChange={e => setJOForm(f => ({ ...f, process: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {PROCESSES.map(p => <option key={p}>{p}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">Exec Type</label>
                  <select value={joForm.exec_type} onChange={e => setJOForm(f => ({ ...f, exec_type: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    <option>Inhouse</option><option>Outsource</option>
                  </select>
                </div>
                {joForm.exec_type === 'Outsource' && (
                  <div><label className="text-xs text-gray-500">Vendor Name</label>
                    <input value={joForm.vendor_name} onChange={e => setJOForm(f => ({ ...f, vendor_name: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                )}
                <div><label className="text-xs text-gray-500">SO Qty</label>
                  <input type="number" value={joForm.so_qty} onChange={e => setJOForm(f => ({ ...f, so_qty: +e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">Planned Qty</label>
                  <input type="number" value={joForm.planned_qty} onChange={e => setJOForm(f => ({ ...f, planned_qty: +e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">Expected Completion</label>
                  <input type="date" value={joForm.expected_completion} onChange={e => setJOForm(f => ({ ...f, expected_completion: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createJOMut.mutate(joForm)} disabled={createJOMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createJOMut.isPending ? 'Saving…' : 'Create JO'}
                </button>
                <button onClick={() => setShowJOForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}

          <div className="bg-white rounded-xl border border-gray-100 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                <tr>{['JO #', 'Process', 'SKU', 'SO #', 'Planned', 'Output', 'Exec', 'Status', 'Action'].map(h =>
                  <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {jos.map(jo => (
                  <tr key={jo.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">{jo.jo_number}</td>
                    <td className="px-4 py-2 text-gray-700">{jo.process}</td>
                    <td className="px-4 py-2 text-gray-600">{jo.sku_name || jo.sku}</td>
                    <td className="px-4 py-2 text-gray-500">{jo.so_number || '—'}</td>
                    <td className="px-4 py-2 text-gray-700">{jo.planned_qty.toLocaleString()}</td>
                    <td className="px-4 py-2 text-blue-600 font-medium">{jo.output_qty.toLocaleString()}</td>
                    <td className="px-4 py-2 text-gray-500">{jo.exec_type}</td>
                    <td className="px-4 py-2">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(jo.status)}`}>{jo.status}</span>
                    </td>
                    <td className="px-4 py-2">
                      <select value={jo.status} onChange={e => updateJOMut.mutate({ id: jo.id, data: { status: e.target.value } })}
                        className="border border-gray-200 rounded px-2 py-1 text-xs">
                        {JO_STATUSES.map(s => <option key={s}>{s}</option>)}
                      </select>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {jos.length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No job orders found.</p>}
          </div>
        </div>
      )}
    </div>
  )
}
