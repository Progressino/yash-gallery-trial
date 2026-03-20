import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type Tab = 'dashboard' | 'tracker' | 'ledger' | 'reservations'

interface GreyStats { total_trackers: number; in_transit: number; at_factory: number; at_printer: number; pending_qc: number; hard_reserved: number; transit_meters: number }
interface GreyEntry {
  id: number; tracker_key: string; po_number: string; material_code: string; material_name: string
  supplier: string; so_reference: string; ordered_qty: number; dispatched_qty: number
  received_qty: number; factory_qty: number; printer_qty: number; checked_qty: number
  rejected_qty: number; bilty_no: string; vehicle_no: string; transporter: string
  dispatch_date: string; expected_arrival: string
  status: string; qc_status: string; qc_checked_by: string; qc_remarks: string
}
interface LedgerEntry { id: number; entry_date: string; material_code: string; material_name: string; transaction_type: string; qty: number; from_location: string; to_location: string; reference_no: string; remarks: string }
interface HardReservation { id: number; fabric_code: string; fabric_name: string; so_number: string; sku: string; qty: number; unit: string; status: string; reserved_date: string }

const STATUSES = ['PO Created', 'Vendor Dispatch Pending', 'In Transit', 'At Transport Location', 'Sent to Factory', 'At Factory', 'Sent to Printer', 'At Printer', 'Printed Fabric Received', 'QC Pass', 'QC Reject', 'Rework', 'Closed']

const statusColor = (s: string) => {
  if (['QC Pass', 'Closed'].includes(s)) return 'bg-green-100 text-green-700'
  if (['In Transit', 'At Transport Location'].includes(s)) return 'bg-blue-100 text-blue-700'
  if (['At Factory', 'At Printer', 'Printed Fabric Received'].includes(s)) return 'bg-purple-100 text-purple-700'
  if (['QC Reject', 'Rework'].includes(s)) return 'bg-red-100 text-red-700'
  if (['Sent to Factory', 'Sent to Printer'].includes(s)) return 'bg-orange-100 text-orange-700'
  return 'bg-gray-100 text-gray-600'
}

export default function GreyFabric() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('dashboard')
  const [filterStatus, setFilterStatus] = useState('')
  const [editEntry, setEditEntry] = useState<GreyEntry | null>(null)
  const [editData, setEditData] = useState<Record<string, string | number>>({})
  const [showNewForm, setShowNewForm] = useState(false)
  const [newForm, setNewForm] = useState({ po_number: '', material_code: '', material_name: '', supplier: '', so_reference: '', ordered_qty: 0 })
  const [showResForm, setShowResForm] = useState(false)
  const [resForm, setResForm] = useState({ fabric_code: '', fabric_name: '', so_number: '', sku: '', qty: 0 })

  const { data: stats } = useQuery<GreyStats>({ queryKey: ['grey-stats'], queryFn: () => api.get('/grey/stats').then(r => r.data) })
  const { data: entries = [] } = useQuery<GreyEntry[]>({
    queryKey: ['grey', filterStatus],
    queryFn: () => api.get('/grey' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data),
    enabled: tab === 'tracker' || tab === 'dashboard'
  })
  const { data: ledger = [] } = useQuery<LedgerEntry[]>({ queryKey: ['grey-ledger'], queryFn: () => api.get('/grey/ledger').then(r => r.data), enabled: tab === 'ledger' })
  const { data: hardRes = [] } = useQuery<HardReservation[]>({ queryKey: ['hard-res'], queryFn: () => api.get('/grey/reservations').then(r => r.data), enabled: tab === 'reservations' })

  const createMut = useMutation({
    mutationFn: (b: object) => api.post('/grey', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey'] }); qc.invalidateQueries({ queryKey: ['grey-stats'] }); setShowNewForm(false) }
  })
  const updateMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/grey/${id}`, data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey'] }); qc.invalidateQueries({ queryKey: ['grey-stats'] }); qc.invalidateQueries({ queryKey: ['grey-ledger'] }); setEditEntry(null) }
  })
  const createResMut = useMutation({
    mutationFn: (b: object) => api.post('/grey/reservations', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['hard-res'] }); qc.invalidateQueries({ queryKey: ['grey-stats'] }); setShowResForm(false) }
  })
  const releaseResMut = useMutation({
    mutationFn: (id: number) => api.delete(`/grey/reservations/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['hard-res'] })
  })

  const openEdit = (entry: GreyEntry) => {
    setEditEntry(entry)
    setEditData({ status: entry.status, dispatched_qty: entry.dispatched_qty, bilty_no: entry.bilty_no, vehicle_no: entry.vehicle_no, transporter: entry.transporter, dispatch_date: entry.dispatch_date, expected_arrival: entry.expected_arrival, qc_status: entry.qc_status, qc_checked_by: entry.qc_checked_by, qc_remarks: entry.qc_remarks })
  }

  const TABS: [Tab, string][] = [['dashboard', '📊 Dashboard'], ['tracker', '🚛 Tracker'], ['ledger', '📜 Ledger'], ['reservations', '🔒 Hard Reservations']]

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-gray-800">Grey Fabric</h1>
        <p className="text-sm text-gray-500">Track grey fabric from vendor dispatch to factory/printer</p>
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
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
            {[
              { label: 'TOTAL', value: stats.total_trackers, color: 'text-gray-700' },
              { label: 'IN TRANSIT', value: stats.in_transit, color: 'text-blue-600' },
              { label: 'AT FACTORY', value: stats.at_factory, color: 'text-purple-600' },
              { label: 'AT PRINTER', value: stats.at_printer, color: 'text-orange-600' },
              { label: 'PENDING QC', value: stats.pending_qc, color: 'text-yellow-600' },
              { label: 'HARD RESERVED', value: stats.hard_reserved, color: 'text-green-600' },
              { label: 'TRANSIT MTR', value: stats.transit_meters?.toFixed(0) ?? '0', color: 'text-indigo-600' },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
                <p className={`text-2xl font-bold ${color}`}>{value}</p>
                <p className="text-xs text-gray-500 mt-1 font-semibold tracking-wide">{label}</p>
              </div>
            ))}
          </div>
          <div className="bg-white rounded-xl border p-4">
            <h3 className="font-semibold text-gray-700 mb-3 text-sm">Active Entries</h3>
            {entries.filter(e => !['Closed', 'QC Pass'].includes(e.status)).slice(0, 8).map(e => (
              <div key={e.id} className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
                <div>
                  <p className="text-sm font-medium text-gray-700">{e.tracker_key} · {e.material_name || e.material_code}</p>
                  <p className="text-xs text-gray-400">{e.supplier} · PO: {e.po_number}</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">{e.dispatched_qty} MTR</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(e.status)}`}>{e.status}</span>
                </div>
              </div>
            ))}
            {entries.length === 0 && <p className="text-xs text-gray-400">No entries yet</p>}
          </div>
          <button onClick={() => { setTab('tracker'); setShowNewForm(true) }}
            className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
            + New Grey Entry
          </button>
        </div>
      )}

      {/* Tracker */}
      {tab === 'tracker' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All Statuses</option>
              {STATUSES.map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={() => setShowNewForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ New Entry</button>
          </div>

          {showNewForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Grey Fabric Entry</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {[['po_number','PO Number'],['material_code','Material Code'],['material_name','Material Name'],['supplier','Supplier'],['so_reference','SO Reference']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(newForm as Record<string,string|number>)[k] as string}
                      onChange={e => setNewForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                <div><label className="text-xs text-gray-500">Ordered Qty (MTR)</label>
                  <input type="number" value={newForm.ordered_qty} onChange={e => setNewForm(f => ({ ...f, ordered_qty: +e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createMut.mutate(newForm)} disabled={createMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createMut.isPending ? 'Saving…' : 'Create Entry'}
                </button>
                <button onClick={() => setShowNewForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}

          {/* Edit Panel */}
          {editEntry && (
            <div className="bg-blue-50 rounded-xl border border-blue-200 p-4 space-y-3">
              <div className="flex justify-between">
                <h3 className="font-semibold text-gray-700">Update: {editEntry.tracker_key}</h3>
                <button onClick={() => setEditEntry(null)} className="text-gray-400 hover:text-gray-600">✕</button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div><label className="text-xs text-gray-500">Status</label>
                  <select value={editData.status as string || editEntry.status}
                    onChange={e => setEditData(d => ({ ...d, status: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {STATUSES.map(s => <option key={s}>{s}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">Dispatched Qty (MTR)</label>
                  <input type="number" value={editData.dispatched_qty as number ?? editEntry.dispatched_qty}
                    onChange={e => setEditData(d => ({ ...d, dispatched_qty: +e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">Bilty / LR No</label>
                  <input value={editData.bilty_no as string ?? editEntry.bilty_no}
                    onChange={e => setEditData(d => ({ ...d, bilty_no: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">Vehicle No</label>
                  <input value={editData.vehicle_no as string ?? editEntry.vehicle_no}
                    onChange={e => setEditData(d => ({ ...d, vehicle_no: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">Transporter</label>
                  <input value={editData.transporter as string ?? editEntry.transporter}
                    onChange={e => setEditData(d => ({ ...d, transporter: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">Dispatch Date</label>
                  <input type="date" value={editData.dispatch_date as string ?? editEntry.dispatch_date}
                    onChange={e => setEditData(d => ({ ...d, dispatch_date: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">Expected Arrival</label>
                  <input type="date" value={editData.expected_arrival as string ?? editEntry.expected_arrival}
                    onChange={e => setEditData(d => ({ ...d, expected_arrival: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">QC Status</label>
                  <select value={editData.qc_status as string ?? editEntry.qc_status}
                    onChange={e => setEditData(d => ({ ...d, qc_status: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    <option>Pending</option><option>Pass</option><option>Fail</option>
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">QC By</label>
                  <input value={editData.qc_checked_by as string ?? editEntry.qc_checked_by}
                    onChange={e => setEditData(d => ({ ...d, qc_checked_by: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => updateMut.mutate({ id: editEntry.id, data: editData })} disabled={updateMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {updateMut.isPending ? 'Saving…' : 'Save Changes'}
                </button>
                <button onClick={() => setEditEntry(null)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {entries.map(e => (
              <div key={e.id} className="bg-white rounded-xl border shadow-sm p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <p className="font-semibold text-gray-800 text-sm">{e.tracker_key}</p>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(e.status)}`}>{e.status}</span>
                      {e.qc_status !== 'Pending' && (
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${e.qc_status === 'Pass' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>QC: {e.qc_status}</span>
                      )}
                    </div>
                    <p className="text-sm text-gray-600">{e.material_name || e.material_code} · {e.supplier}</p>
                    <p className="text-xs text-gray-400">PO: {e.po_number} {e.so_reference ? `· SO: ${e.so_reference}` : ''}</p>
                    <div className="flex gap-4 mt-2 text-xs">
                      <span className="text-gray-500">Ordered: <b>{e.ordered_qty}</b></span>
                      <span className="text-blue-600">Dispatched: <b>{e.dispatched_qty}</b></span>
                      <span className="text-green-600">Factory: <b>{e.factory_qty}</b></span>
                      {e.bilty_no && <span className="text-gray-400">LR: {e.bilty_no}</span>}
                    </div>
                  </div>
                  <button onClick={() => openEdit(e)} className="text-xs text-blue-500 hover:text-blue-700 ml-4">Update</button>
                </div>
              </div>
            ))}
            {entries.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No grey fabric entries. Add PO-linked grey fabric tracking.</p>}
          </div>
        </div>
      )}

      {/* Ledger */}
      {tab === 'ledger' && (
        <div className="bg-white rounded-xl border overflow-hidden">
          <div className="px-4 py-3 border-b bg-gray-50">
            <p className="text-sm font-semibold text-gray-700">Stock Ledger — Last 200 transactions</p>
          </div>
          <table className="w-full text-sm">
            <thead className="text-gray-400 text-xs uppercase">
              <tr>{['Date','Material','Type','Qty','From','To','Reference','Remarks'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
            </thead>
            <tbody>
              {ledger.map(l => (
                <tr key={l.id} className="border-t border-gray-50 hover:bg-gray-50">
                  <td className="px-4 py-2 text-gray-500">{l.entry_date}</td>
                  <td className="px-4 py-2 font-medium text-gray-700">{l.material_code}</td>
                  <td className="px-4 py-2 text-gray-600">{l.transaction_type}</td>
                  <td className="px-4 py-2 font-semibold text-gray-700">{l.qty}</td>
                  <td className="px-4 py-2 text-gray-400">{l.from_location}</td>
                  <td className="px-4 py-2 text-gray-400">{l.to_location}</td>
                  <td className="px-4 py-2 text-gray-400">{l.reference_no}</td>
                  <td className="px-4 py-2 text-gray-400">{l.remarks}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {ledger.length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No ledger entries yet</p>}
        </div>
      )}

      {/* Hard Reservations */}
      {tab === 'reservations' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <p className="text-sm text-gray-500">{hardRes.length} active hard reservations</p>
            <button onClick={() => setShowResForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ Reserve Fabric</button>
          </div>
          {showResForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">Hard Reserve Fabric</h3>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {[['fabric_code','Fabric Code'],['fabric_name','Fabric Name'],['so_number','SO Number'],['sku','SKU']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(resForm as Record<string,string|number>)[k] as string}
                      onChange={e => setResForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                <div><label className="text-xs text-gray-500">Qty (MTR)</label>
                  <input type="number" value={resForm.qty} onChange={e => setResForm(f => ({ ...f, qty: +e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createResMut.mutate(resForm)} disabled={createResMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">Reserve</button>
                <button onClick={() => setShowResForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}
          <div className="bg-white rounded-xl border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                <tr>{['Fabric Code','Name','SO #','SKU','Qty (MTR)','Date','Action'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {hardRes.map(r => (
                  <tr key={r.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">{r.fabric_code}</td>
                    <td className="px-4 py-2 text-gray-600">{r.fabric_name}</td>
                    <td className="px-4 py-2 text-gray-500">{r.so_number || '—'}</td>
                    <td className="px-4 py-2 text-gray-500">{r.sku || '—'}</td>
                    <td className="px-4 py-2 font-semibold text-green-700">{r.qty}</td>
                    <td className="px-4 py-2 text-gray-400 text-xs">{r.reserved_date?.split('T')[0]}</td>
                    <td className="px-4 py-2">
                      <button onClick={() => releaseResMut.mutate(r.id)} className="text-xs text-red-500 hover:text-red-700">Release</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {hardRes.length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No hard reservations</p>}
          </div>
        </div>
      )}
    </div>
  )
}
