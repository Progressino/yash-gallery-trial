import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type Tab = 'dashboard' | 'orders' | 'mrp' | 'reservations'

interface ProdStats { total_jos: number; open_jos: number; in_progress: number; completed_today: number; soft_reservations: number }
interface JobOrder {
  id: number; jo_number: string; jo_date: string; so_number: string
  sku: string; sku_name: string; process: string; exec_type: string
  vendor_name: string; so_qty: number; planned_qty: number; output_qty: number
  status: string; expected_completion: string; issued_to: string
}
interface MRPItem { material_code: string; material_name: string; unit: string; required_qty: number; reserved_qty: number; net_requirement: number; so_refs: string[] }
interface SoftReservation { id: number; material_code: string; material_name: string; reserved_qty: number; unit: string; against_so: string; status: string; reservation_date: string }

const PROCESSES = ['Cutting', 'Stitching', 'Finishing', 'Embroidery', 'Dyeing', 'Kaja Button', 'Packing', 'Quality Check', 'Other']
const JO_STATUSES = ['Created', 'Material Issued', 'In Progress', 'Partially Completed', 'Completed', 'Closed', 'Cancelled']

const statusColor = (s: string) => {
  if (['Completed', 'Closed'].includes(s)) return 'bg-green-100 text-green-700'
  if (s === 'In Progress') return 'bg-blue-100 text-blue-700'
  if (s === 'Material Issued') return 'bg-purple-100 text-purple-700'
  if (s === 'Cancelled') return 'bg-red-100 text-red-700'
  return 'bg-yellow-50 text-yellow-700'
}

export default function Production() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('dashboard')
  const [showJOForm, setShowJOForm] = useState(false)
  const [filterStatus, setFilterStatus] = useState('')
  const [mrpSO, setMrpSO] = useState('')
  const [showResForm, setShowResForm] = useState(false)
  const [resForm, setResForm] = useState({ material_code: '', material_name: '', reserved_qty: 0, unit: 'PCS', against_so: '' })

  const [joForm, setJOForm] = useState({
    so_number: '', sku: '', sku_name: '', process: 'Cutting', exec_type: 'Inhouse',
    vendor_name: '', so_qty: 0, planned_qty: 0, expected_completion: '', issued_to: '', remarks: ''
  })

  const { data: stats } = useQuery<ProdStats>({ queryKey: ['prod-stats'], queryFn: () => api.get('/production/stats').then(r => r.data) })
  const { data: jos = [] } = useQuery<JobOrder[]>({
    queryKey: ['jos', filterStatus],
    queryFn: () => api.get('/production/orders' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data),
    enabled: tab === 'orders' || tab === 'dashboard'
  })
  const { data: mrpData = [] } = useQuery<MRPItem[]>({
    queryKey: ['mrp', mrpSO],
    queryFn: () => api.get('/production/mrp' + (mrpSO ? `?so_number=${mrpSO}` : '')).then(r => r.data),
    enabled: tab === 'mrp'
  })
  const { data: reservations = [] } = useQuery<SoftReservation[]>({
    queryKey: ['soft-res'],
    queryFn: () => api.get('/production/reservations').then(r => r.data),
    enabled: tab === 'reservations' || tab === 'mrp'
  })

  const createJOMut = useMutation({
    mutationFn: (b: object) => api.post('/production/orders', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['jos'] }); qc.invalidateQueries({ queryKey: ['prod-stats'] }); setShowJOForm(false) }
  })
  const updateJOMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/production/orders/${id}`, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['jos'] })
  })
  const createResMut = useMutation({
    mutationFn: (b: object) => api.post('/production/reservations', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['soft-res'] }); qc.invalidateQueries({ queryKey: ['prod-stats'] }); setShowResForm(false); setResForm({ material_code: '', material_name: '', reserved_qty: 0, unit: 'PCS', against_so: '' }) }
  })
  const releaseResMut = useMutation({
    mutationFn: (id: number) => api.delete(`/production/reservations/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['soft-res'] })
  })

  const TABS: [Tab, string][] = [['dashboard', '📊 Dashboard'], ['orders', '🔧 Job Orders'], ['mrp', '📊 MRP'], ['reservations', '🔒 Reservations']]

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-gray-800">Production</h1>
        <p className="text-sm text-gray-500">Job orders, MRP calculations, material reservations</p>
      </div>

      <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit">
        {TABS.map(([key, label]) => (
          <button key={key} onClick={() => setTab(key)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${tab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* Dashboard */}
      {tab === 'dashboard' && stats && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {[
              { label: 'TOTAL JOs', value: stats.total_jos, color: 'text-gray-700' },
              { label: 'OPEN', value: stats.open_jos, color: 'text-yellow-600' },
              { label: 'IN PROGRESS', value: stats.in_progress, color: 'text-blue-600' },
              { label: 'COMPLETED TODAY', value: stats.completed_today, color: 'text-green-600' },
              { label: 'SOFT RESERVATIONS', value: stats.soft_reservations, color: 'text-purple-600' },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
                <p className={`text-2xl font-bold ${color}`}>{value}</p>
                <p className="text-xs text-gray-500 mt-1 font-semibold tracking-wide">{label}</p>
              </div>
            ))}
          </div>
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
          <button onClick={() => { setTab('orders'); setShowJOForm(true) }}
            className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
            + New Job Order
          </button>
        </div>
      )}

      {/* Job Orders */}
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
                {[['so_number','SO Number'],['sku','SKU Code'],['sku_name','SKU Name'],['issued_to','Issued To'],['remarks','Remarks']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(joForm as Record<string,string|number>)[k] as string}
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
                <tr>{['JO #','Process','SKU','SO #','Planned','Output','Exec','Status','Action'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
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

      {/* MRP */}
      {tab === 'mrp' && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border p-4 space-y-3">
            <h3 className="font-semibold text-gray-700">Run MRP</h3>
            <p className="text-xs text-gray-500">Calculates material requirements by exploding open Sales Order lines through BOMs.</p>
            <div className="flex gap-3 items-end">
              <div className="flex-1">
                <label className="text-xs text-gray-500">Filter by SO Number (optional)</label>
                <input value={mrpSO} onChange={e => setMrpSO(e.target.value)} placeholder="e.g. SO-0001"
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
              </div>
              <button onClick={() => qc.invalidateQueries({ queryKey: ['mrp'] })}
                className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
                Refresh MRP
              </button>
            </div>
          </div>

          {mrpData.length > 0 ? (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-3 border-b bg-gray-50">
                <p className="text-sm font-semibold text-gray-700">{mrpData.length} materials required</p>
              </div>
              <table className="w-full text-sm">
                <thead className="text-gray-400 text-xs uppercase">
                  <tr>{['Material Code','Name','Required','Reserved','Net Req','Unit','SO Refs','Action'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
                </thead>
                <tbody>
                  {mrpData.map(m => (
                    <tr key={m.material_code} className={`border-t border-gray-50 hover:bg-gray-50 ${m.net_requirement > 0 ? '' : 'opacity-60'}`}>
                      <td className="px-4 py-2 font-medium text-gray-700">{m.material_code}</td>
                      <td className="px-4 py-2 text-gray-600">{m.material_name}</td>
                      <td className="px-4 py-2 text-gray-700">{m.required_qty.toFixed(2)}</td>
                      <td className="px-4 py-2 text-purple-600">{m.reserved_qty.toFixed(2)}</td>
                      <td className={`px-4 py-2 font-bold ${m.net_requirement > 0 ? 'text-red-600' : 'text-green-600'}`}>{m.net_requirement.toFixed(2)}</td>
                      <td className="px-4 py-2 text-gray-500">{m.unit}</td>
                      <td className="px-4 py-2 text-xs text-gray-400">{m.so_refs.join(', ')}</td>
                      <td className="px-4 py-2">
                        {m.net_requirement > 0 && (
                          <button onClick={() => { setResForm({ material_code: m.material_code, material_name: m.material_name, reserved_qty: m.net_requirement, unit: m.unit, against_so: m.so_refs[0] || '' }); setTab('reservations'); setShowResForm(true) }}
                            className="text-xs px-2 py-0.5 bg-purple-50 text-purple-700 rounded hover:bg-purple-100">Reserve</button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-center text-gray-400 py-8 text-sm">No open SO lines with BOM data found. Upload sales orders first.</p>
          )}
        </div>
      )}

      {/* Soft Reservations */}
      {tab === 'reservations' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <p className="text-sm text-gray-500">{reservations.length} active reservations</p>
            <button onClick={() => setShowResForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ New Reservation</button>
          </div>

          {showResForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">Soft Reserve Material</h3>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {[['material_code','Code'],['material_name','Name'],['against_so','Against SO']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(resForm as Record<string,string|number>)[k] as string}
                      onChange={e => setResForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                <div><label className="text-xs text-gray-500">Qty</label>
                  <input type="number" value={resForm.reserved_qty} onChange={e => setResForm(f => ({ ...f, reserved_qty: +e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">Unit</label>
                  <input value={resForm.unit} onChange={e => setResForm(f => ({ ...f, unit: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createResMut.mutate(resForm)} disabled={createResMut.isPending || !resForm.material_code}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createResMut.isPending ? 'Saving…' : 'Reserve'}
                </button>
                <button onClick={() => setShowResForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}

          <div className="bg-white rounded-xl border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                <tr>{['Code','Name','Reserved Qty','Unit','Against SO','Date','Action'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {reservations.map(r => (
                  <tr key={r.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">{r.material_code}</td>
                    <td className="px-4 py-2 text-gray-600">{r.material_name}</td>
                    <td className="px-4 py-2 font-semibold text-purple-700">{r.reserved_qty}</td>
                    <td className="px-4 py-2 text-gray-500">{r.unit}</td>
                    <td className="px-4 py-2 text-gray-500">{r.against_so || '—'}</td>
                    <td className="px-4 py-2 text-gray-400 text-xs">{r.reservation_date?.split('T')[0]}</td>
                    <td className="px-4 py-2">
                      <button onClick={() => releaseResMut.mutate(r.id)} className="text-xs text-red-500 hover:text-red-700">Release</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {reservations.length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No soft reservations</p>}
          </div>
        </div>
      )}
    </div>
  )
}
