import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type Tab = 'dashboard' | 'demands' | 'orders'

interface Demand {
  id: number; demand_number: string; demand_date: string
  demand_source: string; buyer: string; priority: string; status: string
  notes: string; lines: DemandLine[]
}
interface DemandLine { id: number; sku: string; sku_name: string; demand_qty: number; delivered_qty: number }
interface SalesOrder {
  id: number; so_number: string; so_date: string; buyer: string
  warehouse: string; delivery_date: string; status: string; notes: string
  lines: SOLine[]
}
interface SOLine { id: number; sku: string; sku_name: string; qty: number; produced_qty: number; dispatch_qty: number; unit: string }

const SOURCES = ['Sales Team', 'Forecasting System', 'Buyer Indent', 'Marketing Team']
const PRIORITIES = ['Normal', 'High', 'Urgent']
const DEMAND_STATUSES = ['Draft', 'Submitted', 'Approved', 'In Production', 'Closed', 'Cancelled']
const SO_STATUSES = ['Draft', 'Submitted', 'Approved', 'In Production', 'Partially Dispatched', 'Dispatched', 'Closed', 'Cancelled']
const SOURCE_TYPES = ['Sales Team Demand', 'Buyer PO', 'Offline Order']

const statusColor = (s: string) => {
  if (['Approved', 'Dispatched', 'Closed'].includes(s)) return 'bg-green-100 text-green-700'
  if (['Draft', 'Submitted'].includes(s)) return 'bg-yellow-100 text-yellow-700'
  if (s === 'In Production') return 'bg-blue-100 text-blue-700'
  if (['Cancelled'].includes(s)) return 'bg-red-100 text-red-700'
  return 'bg-gray-100 text-gray-600'
}

const priorityColor = (p: string) => {
  if (p === 'Urgent') return 'bg-red-100 text-red-700'
  if (p === 'High') return 'bg-orange-100 text-orange-700'
  return 'bg-gray-100 text-gray-600'
}

export default function SalesOrders() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('dashboard')
  const [showNewDemand, setShowNewDemand] = useState(false)
  const [showNewSO, setShowNewSO] = useState(false)
  const [expandedDemand, setExpandedDemand] = useState<number | null>(null)
  const [expandedSO, setExpandedSO] = useState<number | null>(null)
  const [filterStatus, setFilterStatus] = useState('')

  // Demand form
  const [dForm, setDForm] = useState({ demand_source: 'Sales Team', buyer: '', priority: 'Normal', notes: '' })
  const [dLines, setDLines] = useState<{ sku: string; sku_name: string; demand_qty: number }[]>([])

  // SO form
  const [soForm, setSOForm] = useState({ buyer: '', warehouse: '', sales_team: '', source_type: 'Sales Team Demand', ref_demand: '', delivery_date: '', payment_terms: '', notes: '' })
  const [soLines, setSOLines] = useState<{ sku: string; sku_name: string; qty: number; unit: string }[]>([])

  const { data: demands = [] } = useQuery<Demand[]>({ queryKey: ['demands', filterStatus], queryFn: () => api.get('/sales/demands' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data) })
  const { data: orders = [] } = useQuery<SalesOrder[]>({ queryKey: ['sales-orders', filterStatus], queryFn: () => api.get('/sales/orders' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data) })

  const createDemandMut = useMutation({
    mutationFn: (body: object) => api.post('/sales/demands', body),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['demands'] }); setShowNewDemand(false); setDLines([]); setDForm({ demand_source: 'Sales Team', buyer: '', priority: 'Normal', notes: '' }) }
  })
  const createSOMut = useMutation({
    mutationFn: (body: object) => api.post('/sales/orders', body),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['sales-orders'] }); setShowNewSO(false); setSOLines([]); setSOForm({ buyer: '', warehouse: '', sales_team: '', source_type: 'Sales Team Demand', ref_demand: '', delivery_date: '', payment_terms: '', notes: '' }) }
  })
  const updateDemandMut = useMutation({
    mutationFn: ({ id, status }: { id: number; status: string }) => api.patch(`/sales/demands/${id}/status`, { status }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['demands'] })
  })
  const updateSOMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/sales/orders/${id}`, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['sales-orders'] })
  })

  const addDLine = () => setDLines(l => [...l, { sku: '', sku_name: '', demand_qty: 0 }])
  const addSOLine = () => setSOLines(l => [...l, { sku: '', sku_name: '', qty: 0, unit: 'PCS' }])

  const openDemands = demands.filter(d => !['Closed', 'Cancelled'].includes(d.status)).length
  const openOrders = orders.filter(o => !['Closed', 'Cancelled'].includes(o.status)).length
  const urgentDemands = demands.filter(d => d.priority === 'Urgent' && !['Closed', 'Cancelled'].includes(d.status)).length

  const TABS: [Tab, string][] = [['dashboard', '📊 Dashboard'], ['demands', '📋 Demands'], ['orders', '🧾 Sales Orders']]

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-800">Sales Orders & Demand</h1>
          <p className="text-sm text-gray-500">Manage demands, sales orders and tracking</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit">
        {TABS.map(([key, label]) => (
          <button key={key} onClick={() => setTab(key)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${tab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* Dashboard */}
      {tab === 'dashboard' && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'OPEN DEMANDS', value: openDemands, color: 'text-blue-600' },
              { label: 'OPEN ORDERS', value: openOrders, color: 'text-green-600' },
              { label: 'URGENT', value: urgentDemands, color: 'text-red-600' },
              { label: 'TOTAL DEMANDS', value: demands.length, color: 'text-gray-700' },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
                <p className={`text-2xl font-bold ${color}`}>{value}</p>
                <p className="text-xs text-gray-500 mt-1 font-semibold tracking-wide">{label}</p>
              </div>
            ))}
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
              <h3 className="font-semibold text-gray-700 mb-3 text-sm">Recent Demands</h3>
              {demands.slice(0, 5).map(d => (
                <div key={d.id} className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
                  <div>
                    <p className="text-sm font-medium text-gray-700">{d.demand_number}</p>
                    <p className="text-xs text-gray-400">{d.buyer} · {d.demand_date}</p>
                  </div>
                  <div className="flex gap-1">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${priorityColor(d.priority)}`}>{d.priority}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(d.status)}`}>{d.status}</span>
                  </div>
                </div>
              ))}
              {demands.length === 0 && <p className="text-xs text-gray-400">No demands yet</p>}
            </div>
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
              <h3 className="font-semibold text-gray-700 mb-3 text-sm">Recent Sales Orders</h3>
              {orders.slice(0, 5).map(o => (
                <div key={o.id} className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
                  <div>
                    <p className="text-sm font-medium text-gray-700">{o.so_number}</p>
                    <p className="text-xs text-gray-400">{o.buyer} · {o.delivery_date}</p>
                  </div>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(o.status)}`}>{o.status}</span>
                </div>
              ))}
              {orders.length === 0 && <p className="text-xs text-gray-400">No orders yet</p>}
            </div>
          </div>
          <div className="flex gap-3">
            <button onClick={() => { setTab('demands'); setShowNewDemand(true) }}
              className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
              + New Demand
            </button>
            <button onClick={() => { setTab('orders'); setShowNewSO(true) }}
              className="px-4 py-2 border border-[#002B5B] text-[#002B5B] rounded-lg text-sm font-medium hover:bg-gray-50">
              + New Sales Order
            </button>
          </div>
        </div>
      )}

      {/* Demands */}
      {tab === 'demands' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex gap-2">
              <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)}
                className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
                <option value="">All Statuses</option>
                {DEMAND_STATUSES.map(s => <option key={s}>{s}</option>)}
              </select>
            </div>
            <button onClick={() => setShowNewDemand(true)}
              className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
              + New Demand
            </button>
          </div>

          {showNewDemand && (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4 space-y-4">
              <h3 className="font-semibold text-gray-700">New Demand</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div>
                  <label className="text-xs text-gray-500">Source</label>
                  <select value={dForm.demand_source} onChange={e => setDForm(f => ({ ...f, demand_source: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {SOURCES.map(s => <option key={s}>{s}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Buyer</label>
                  <input value={dForm.buyer} onChange={e => setDForm(f => ({ ...f, buyer: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div>
                  <label className="text-xs text-gray-500">Priority</label>
                  <select value={dForm.priority} onChange={e => setDForm(f => ({ ...f, priority: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {PRIORITIES.map(p => <option key={p}>{p}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Notes</label>
                  <input value={dForm.notes} onChange={e => setDForm(f => ({ ...f, notes: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-gray-600">SKU Lines</p>
                  <button onClick={addDLine} className="text-xs text-blue-600 hover:underline">+ Add Line</button>
                </div>
                {dLines.map((ln, i) => (
                  <div key={i} className="flex gap-2 mb-2">
                    <input placeholder="SKU Code" value={ln.sku} onChange={e => setDLines(l => l.map((x, j) => j === i ? { ...x, sku: e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm flex-1" />
                    <input placeholder="SKU Name" value={ln.sku_name} onChange={e => setDLines(l => l.map((x, j) => j === i ? { ...x, sku_name: e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm flex-1" />
                    <input type="number" placeholder="Qty" value={ln.demand_qty} onChange={e => setDLines(l => l.map((x, j) => j === i ? { ...x, demand_qty: +e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm w-24" />
                    <button onClick={() => setDLines(l => l.filter((_, j) => j !== i))} className="text-red-400 hover:text-red-600 text-sm px-1">✕</button>
                  </div>
                ))}
                {dLines.length === 0 && <p className="text-xs text-gray-400">No lines added. Click "+ Add Line" to add SKUs.</p>}
              </div>
              <div className="flex gap-2">
                <button onClick={() => createDemandMut.mutate({ ...dForm, lines: dLines })}
                  disabled={createDemandMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800 disabled:opacity-50">
                  {createDemandMut.isPending ? 'Saving…' : 'Create Demand'}
                </button>
                <button onClick={() => setShowNewDemand(false)} className="px-4 py-2 border border-gray-200 rounded-lg text-sm text-gray-600 hover:bg-gray-50">Cancel</button>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {demands.map(d => (
              <div key={d.id} className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
                <div className="flex items-center justify-between p-4 cursor-pointer" onClick={() => setExpandedDemand(expandedDemand === d.id ? null : d.id)}>
                  <div className="flex items-center gap-3">
                    <div>
                      <p className="font-semibold text-gray-800 text-sm">{d.demand_number}</p>
                      <p className="text-xs text-gray-500">{d.buyer} · {d.demand_source} · {d.demand_date}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${priorityColor(d.priority)}`}>{d.priority}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(d.status)}`}>{d.status}</span>
                    <select value={d.status} onChange={e => updateDemandMut.mutate({ id: d.id, status: e.target.value })}
                      onClick={e => e.stopPropagation()}
                      className="border border-gray-200 rounded px-2 py-1 text-xs">
                      {DEMAND_STATUSES.map(s => <option key={s}>{s}</option>)}
                    </select>
                    <span className="text-gray-400 text-xs">{expandedDemand === d.id ? '▲' : '▼'}</span>
                  </div>
                </div>
                {expandedDemand === d.id && (
                  <div className="border-t border-gray-50 px-4 pb-4">
                    <table className="w-full text-xs mt-3">
                      <thead><tr className="text-gray-400 uppercase"><th className="text-left py-1">SKU</th><th className="text-left py-1">Name</th><th className="text-right py-1">Demanded</th><th className="text-right py-1">Delivered</th></tr></thead>
                      <tbody>
                        {d.lines.map(l => (
                          <tr key={l.id} className="border-t border-gray-50">
                            <td className="py-1.5 font-medium text-gray-700">{l.sku}</td>
                            <td className="py-1.5 text-gray-600">{l.sku_name}</td>
                            <td className="py-1.5 text-right text-gray-700">{l.demand_qty.toLocaleString()}</td>
                            <td className="py-1.5 text-right text-green-600">{l.delivered_qty.toLocaleString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {d.notes && <p className="text-xs text-gray-500 mt-2">📝 {d.notes}</p>}
                  </div>
                )}
              </div>
            ))}
            {demands.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No demands found. Create your first demand.</p>}
          </div>
        </div>
      )}

      {/* Sales Orders */}
      {tab === 'orders' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)}
              className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All Statuses</option>
              {SO_STATUSES.map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={() => setShowNewSO(true)}
              className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
              + New Sales Order
            </button>
          </div>

          {showNewSO && (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4 space-y-4">
              <h3 className="font-semibold text-gray-700">New Sales Order</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  { label: 'Buyer', key: 'buyer' },
                  { label: 'Warehouse', key: 'warehouse' },
                  { label: 'Sales Team', key: 'sales_team' },
                  { label: 'Payment Terms', key: 'payment_terms' },
                  { label: 'Ref Demand #', key: 'ref_demand' },
                  { label: 'Notes', key: 'notes' },
                ].map(({ label, key }) => (
                  <div key={key}>
                    <label className="text-xs text-gray-500">{label}</label>
                    <input value={(soForm as Record<string, string>)[key]} onChange={e => setSOForm(f => ({ ...f, [key]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                <div>
                  <label className="text-xs text-gray-500">Delivery Date</label>
                  <input type="date" value={soForm.delivery_date} onChange={e => setSOForm(f => ({ ...f, delivery_date: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div>
                  <label className="text-xs text-gray-500">Source Type</label>
                  <select value={soForm.source_type} onChange={e => setSOForm(f => ({ ...f, source_type: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {SOURCE_TYPES.map(s => <option key={s}>{s}</option>)}
                  </select>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-2">
                  <p className="text-sm font-medium text-gray-600">SKU Lines</p>
                  <button onClick={addSOLine} className="text-xs text-blue-600 hover:underline">+ Add Line</button>
                </div>
                {soLines.map((ln, i) => (
                  <div key={i} className="flex gap-2 mb-2">
                    <input placeholder="SKU Code" value={ln.sku} onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, sku: e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm flex-1" />
                    <input placeholder="SKU Name" value={ln.sku_name} onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, sku_name: e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm flex-1" />
                    <input type="number" placeholder="Qty" value={ln.qty} onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, qty: +e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm w-24" />
                    <select value={ln.unit} onChange={e => setSOLines(l => l.map((x, j) => j === i ? { ...x, unit: e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm w-20">
                      {['PCS', 'SET', 'PAIR', 'BOX'].map(u => <option key={u}>{u}</option>)}
                    </select>
                    <button onClick={() => setSOLines(l => l.filter((_, j) => j !== i))} className="text-red-400 hover:text-red-600 text-sm px-1">✕</button>
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <button onClick={() => createSOMut.mutate({ ...soForm, lines: soLines })}
                  disabled={createSOMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800 disabled:opacity-50">
                  {createSOMut.isPending ? 'Saving…' : 'Create Sales Order'}
                </button>
                <button onClick={() => setShowNewSO(false)} className="px-4 py-2 border border-gray-200 rounded-lg text-sm text-gray-600 hover:bg-gray-50">Cancel</button>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {orders.map(o => (
              <div key={o.id} className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
                <div className="flex items-center justify-between p-4 cursor-pointer" onClick={() => setExpandedSO(expandedSO === o.id ? null : o.id)}>
                  <div>
                    <p className="font-semibold text-gray-800 text-sm">{o.so_number}</p>
                    <p className="text-xs text-gray-500">{o.buyer} · Delivery: {o.delivery_date || '—'}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(o.status)}`}>{o.status}</span>
                    <select value={o.status} onChange={e => updateSOMut.mutate({ id: o.id, data: { status: e.target.value } })}
                      onClick={e => e.stopPropagation()}
                      className="border border-gray-200 rounded px-2 py-1 text-xs">
                      {SO_STATUSES.map(s => <option key={s}>{s}</option>)}
                    </select>
                    <span className="text-gray-400 text-xs">{expandedSO === o.id ? '▲' : '▼'}</span>
                  </div>
                </div>
                {expandedSO === o.id && (
                  <div className="border-t border-gray-50 px-4 pb-4">
                    <table className="w-full text-xs mt-3">
                      <thead><tr className="text-gray-400 uppercase">
                        <th className="text-left py-1">SKU</th>
                        <th className="text-left py-1">Name</th>
                        <th className="text-right py-1">Ordered</th>
                        <th className="text-right py-1">Produced</th>
                        <th className="text-right py-1">Dispatched</th>
                        <th className="text-right py-1">Pending</th>
                      </tr></thead>
                      <tbody>
                        {o.lines.map(l => (
                          <tr key={l.id} className="border-t border-gray-50">
                            <td className="py-1.5 font-medium text-gray-700">{l.sku}</td>
                            <td className="py-1.5 text-gray-600">{l.sku_name}</td>
                            <td className="py-1.5 text-right text-gray-700">{l.qty.toLocaleString()}</td>
                            <td className="py-1.5 text-right text-blue-600">{l.produced_qty.toLocaleString()}</td>
                            <td className="py-1.5 text-right text-green-600">{l.dispatch_qty.toLocaleString()}</td>
                            <td className="py-1.5 text-right text-orange-600">{Math.max(0, l.qty - l.dispatch_qty).toLocaleString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ))}
            {orders.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No sales orders found.</p>}
          </div>
        </div>
      )}
    </div>
  )
}
