import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type Tab =
  | 'dashboard'
  | 'locations'
  | 'tracker'
  | 'mrp'
  | 'jobwork'
  | 'qc'
  | 'ledger'
  | 'reservations'
  | 'printed-fabric'
  | 'reports'

interface GreyStats {
  total_trackers: number
  in_transit: number
  at_transport?: number
  at_factory: number
  at_printer: number
  pending_qc: number
  hard_reserved: number
  transit_meters: number
  location_totals?: {
    in_transit_mtr: number
    transport_mtr: number
    factory_mtr: number
    printer_mtr: number
    rejected_recorded_mtr: number
    return_vendor_mtr: number
    rework_mtr: number
    printed_fabric_mtr: number
  }
}

interface GreyEntry {
  id: number
  tracker_key: string
  po_number: string
  material_code: string
  material_name: string
  supplier: string
  so_reference: string
  ordered_qty: number
  rate?: number
  delivery_location?: string
  dispatched_qty: number
  received_qty: number
  transport_qty: number
  factory_qty: number
  printer_qty: number
  in_transit_qty?: number
  checked_qty: number
  passed_qty?: number
  rejected_qty: number
  bilty_no: string
  vehicle_no: string
  transporter: string
  dispatch_date: string
  expected_arrival: string
  status: string
  qc_status: string
  qc_checked_by: string
  qc_remarks: string
  rework_qty?: number
}

interface LedgerEntry {
  id: number
  entry_date: string
  material_code: string
  material_name: string
  transaction_type: string
  qty: number
  from_location: string
  to_location: string
  reference_no: string
  remarks: string
}

interface HardReservation {
  id: number
  fabric_code: string
  fabric_name: string
  so_number: string
  sku: string
  qty: number
  unit: string
  status: string
  reserved_date: string
}

const STATUSES = [
  'PO Created', 'Vendor Dispatch', 'In Transit', 'At Transport Location',
  'Sent to Factory', 'At Factory', 'Sent to Printer', 'At Printer',
  'Printed Fabric Received', 'QC Pending', 'QC Done', 'Rejected',
  'Return to Vendor', 'Rework', 'Closed',
]

const QC_OUTCOMES = ['Pass', 'Partial Pass', 'Reject', 'Rework', 'QC Done']

const statusColor = (s: string) => {
  if (['QC Done', 'Closed'].includes(s)) return 'bg-green-100 text-green-700'
  if (['In Transit', 'At Transport Location', 'Vendor Dispatch'].includes(s)) return 'bg-blue-100 text-blue-700'
  if (['At Factory', 'At Printer', 'Printed Fabric Received', 'Sent to Factory', 'Sent to Printer'].includes(s)) return 'bg-purple-100 text-purple-700'
  if (['Rejected', 'Rework', 'Return to Vendor'].includes(s)) return 'bg-red-100 text-red-700'
  if (s === 'QC Pending') return 'bg-amber-100 text-amber-800'
  return 'bg-gray-100 text-gray-600'
}

export default function GreyFabric() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('dashboard')
  const [filterStatus, setFilterStatus] = useState('')
  const [editEntry, setEditEntry] = useState<GreyEntry | null>(null)
  const [editData, setEditData] = useState<Record<string, string | number>>({})
  const [showNewForm, setShowNewForm] = useState(false)
  const [newForm, setNewForm] = useState({
    po_number: '', material_code: '', material_name: '',
    supplier: '', so_reference: '', ordered_qty: 0, rate: 0, delivery_location: '',
  })
  const [showResForm, setShowResForm] = useState(false)
  const [resForm, setResForm] = useState({ fabric_code: '', fabric_name: '', so_number: '', sku: '', qty: 0 })
  const [dispatchModal, setDispatchModal] = useState<GreyEntry | null>(null)
  const [dispatchForm, setDispatchForm] = useState({ bilty_no: '', transporter: '', dispatch_date: '', expected_arrival: '', dispatched_qty: 0, vehicle_no: '' })
  const [transferModal, setTransferModal] = useState<GreyEntry | null>(null)
  const [transferForm, setTransferForm] = useState({ to_location: 'factory' as 'factory' | 'printer', qty: 0 })
  const [qcModal, setQcModal] = useState<GreyEntry | null>(null)
  const [qcForm, setQcForm] = useState({ received_qty: 0, checked_qty: 0, passed_qty: 0, rejected_qty: 0, rework_qty: 0, outcome: 'Partial Pass', qc_remarks: '', qc_by: '', qc_date: '' })
  const [printerModal, setPrinterModal] = useState<GreyEntry | null>(null)
  const [printerForm, setPrinterForm] = useState({ job_order_no: '', issue_qty: 0, from_location: 'Transport Location', to_vendor: '', issue_date: '', challan_no: '', gate_pass: '', remarks: '' })
  const [receiveModal, setReceiveModal] = useState<{ issueId: number; trackerId: number } | null>(null)
  const [receiveForm, setReceiveForm] = useState({ received_back_qty: 0, grey_input_mtr: 0, printed_item_code: '', printed_output_mtr: 0, wastage_mtr: 0, conversion_date: '', remarks: '' })
  const [returnModal, setReturnModal] = useState<GreyEntry | null>(null)
  const [returnForm, setReturnForm] = useState({ return_qty: 0, debit_note_no: '', return_challan: '', return_date: '', remarks: '' })
  const [mrpForm, setMrpForm] = useState({ run_label: '', material_code: '', material_name: '', so_number: '', sku: '', qty_required: 0, notes: '' })
  const [drillMaterial, setDrillMaterial] = useState('')
  const [reportKey, setReportKey] = useState<string | null>(null)
  const [reportRows, setReportRows] = useState<unknown[]>([])

  // Printed Fabric states
  const [pfSubTab, setPfSubTab] = useState<'unchecked' | 'checked' | 'ready-to-cut'>('unchecked')
  const [pfQCForm, setPFQCForm] = useState({ fabric_code: '', fabric_name: '', jwo_ref: '', passed_qty: 0, failed_qty: 0, qc_by: '', qc_date: '' })
  const [showPFQCForm, setShowPFQCForm] = useState(false)
  const [pfQCTarget, setPFQCTarget] = useState<any>(null)
  const [showPFReserveForm, setShowPFReserveForm] = useState(false)
  const [pfReserveForm, setPFReserveForm] = useState({ fabric_code: '', fabric_name: '', so_number: '', sku: '', qty: 0, remarks: '' })

  // ── Queries ──────────────────────────────────────────────────────────────────
  const { data: stats } = useQuery<GreyStats>({
    queryKey: ['grey-stats'],
    queryFn: () => api.get('/grey/stats').then(r => r.data),
  })
  const { data: locData } = useQuery({
    queryKey: ['grey-locations'],
    queryFn: () => api.get('/grey/locations').then(r => r.data),
    enabled: tab === 'locations' || tab === 'dashboard',
  })
  const { data: entries = [] } = useQuery<GreyEntry[]>({
    queryKey: ['grey', filterStatus],
    queryFn: () => api.get('/grey' + (filterStatus ? `?status=${encodeURIComponent(filterStatus)}` : '')).then(r => r.data),
    enabled: tab === 'tracker' || tab === 'dashboard' || tab === 'jobwork',
  })
  const { data: ledger = [] } = useQuery<LedgerEntry[]>({
    queryKey: ['grey-ledger'],
    queryFn: () => api.get('/grey/ledger').then(r => r.data),
    enabled: tab === 'ledger',
  })
  const { data: hardRes = [] } = useQuery<HardReservation[]>({
    queryKey: ['hard-res'],
    queryFn: () => api.get('/grey/reservations').then(r => r.data),
    enabled: tab === 'reservations',
  })
  const { data: mrpReqs = [] } = useQuery({
    queryKey: ['grey-mrp'],
    queryFn: () => api.get('/grey/mrp/requirements').then(r => r.data),
    enabled: tab === 'mrp',
  })
  const { data: mrpTotals = [] } = useQuery({
    queryKey: ['grey-mrp-totals'],
    queryFn: () => api.get('/grey/mrp/totals').then(r => r.data),
    enabled: tab === 'mrp',
  })
  const { data: drilldown } = useQuery({
    queryKey: ['grey-mrp-drill', drillMaterial],
    queryFn: () => api.get(`/grey/mrp/by-material/${encodeURIComponent(drillMaterial)}`).then(r => r.data),
    enabled: !!drillMaterial && tab === 'mrp',
  })
  const { data: printerIssues = [] } = useQuery({
    queryKey: ['grey-printer-issues'],
    queryFn: () => api.get('/grey/printer-issue/list').then(r => r.data),
    enabled: tab === 'jobwork',
  })
  const { data: qcEvents = [] } = useQuery({
    queryKey: ['grey-qc-events'],
    queryFn: () => api.get('/grey/qc-events').then(r => r.data),
    enabled: tab === 'qc',
  })
  // Printed Fabric queries
  const { data: printedFabricUnchecked = [] } = useQuery({
    queryKey: ['printed-fabric-unchecked'],
    queryFn: () => api.get('/grey/printed-fabric/unchecked').then(r => r.data),
    enabled: tab === 'printed-fabric',
  })
  const { data: printedFabricChecked = [] } = useQuery({
    queryKey: ['printed-fabric-checked'],
    queryFn: () => api.get('/grey/printed-fabric/checked').then(r => r.data),
    enabled: tab === 'printed-fabric' && pfSubTab === 'checked',
  })
  const { data: printedReadyToCut = [] } = useQuery({
    queryKey: ['printed-ready-to-cut'],
    queryFn: () => api.get('/grey/printed-fabric/ready-to-cut').then(r => r.data),
    enabled: tab === 'printed-fabric' && pfSubTab === 'ready-to-cut',
  })

  // ── Mutations ─────────────────────────────────────────────────────────────────
  const createMut = useMutation({
    mutationFn: (b: object) => api.post('/grey', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey'] }); qc.invalidateQueries({ queryKey: ['grey-stats'] }); setShowNewForm(false) },
  })
  const updateMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/grey/${id}`, data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey'] }); qc.invalidateQueries({ queryKey: ['grey-stats'] }); setEditEntry(null) },
  })
  const dispatchMut = useMutation({
    mutationFn: ({ id, body }: { id: number; body: object }) => api.post(`/grey/${id}/vendor-dispatch`, body),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey'] }); qc.invalidateQueries({ queryKey: ['grey-stats'] }); setDispatchModal(null) },
  })
  const arriveMut = useMutation({
    mutationFn: ({ id, qty }: { id: number; qty?: number }) => api.post(`/grey/${id}/arrive-transport`, { qty }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey'] }); qc.invalidateQueries({ queryKey: ['grey-stats'] }) },
  })
  const transferMut = useMutation({
    mutationFn: ({ id, body }: { id: number; body: object }) => api.post(`/grey/${id}/transfer`, body),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey'] }); qc.invalidateQueries({ queryKey: ['grey-stats'] }); setTransferModal(null) },
  })
  const qcMut = useMutation({
    mutationFn: ({ id, body }: { id: number; body: object }) => api.post(`/grey/${id}/qc`, body),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey'] }); qc.invalidateQueries({ queryKey: ['grey-qc-events'] }); setQcModal(null) },
  })
  const printerMut = useMutation({
    mutationFn: (body: object) => api.post('/grey/printer-issue', body),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey-printer-issues'] }); setPrinterModal(null) },
  })
  const receivePrintedMut = useMutation({
    mutationFn: ({ issueId, body }: { issueId: number; body: object }) => api.post(`/grey/printer-issue/${issueId}/receive-printed`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['grey-printer-issues'] })
      qc.invalidateQueries({ queryKey: ['printed-fabric-unchecked'] })
      setReceiveModal(null)
    },
  })
  const returnVendorMut = useMutation({
    mutationFn: ({ id, body }: { id: number; body: object }) => api.post(`/grey/${id}/return-vendor`, body),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey'] }); setReturnModal(null) },
  })
  const mrpCreateMut = useMutation({
    mutationFn: (b: object) => api.post('/grey/mrp/requirements', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey-mrp'] }); qc.invalidateQueries({ queryKey: ['grey-mrp-totals'] }) },
  })
  const createResMut = useMutation({
    mutationFn: (b: object) => api.post('/grey/reservations', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['hard-res'] }); setShowResForm(false) },
  })
  const releaseResMut = useMutation({
    mutationFn: (id: number) => api.delete(`/grey/reservations/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['hard-res'] }),
  })
  const printedFabricQCMut = useMutation({
    mutationFn: (b: object) => api.post('/grey/printed-fabric/qc', b),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['printed-fabric-unchecked'] })
      qc.invalidateQueries({ queryKey: ['printed-fabric-checked'] })
      qc.invalidateQueries({ queryKey: ['printed-ready-to-cut'] })
      setShowPFQCForm(false)
      setPFQCTarget(null)
    }
  })
  const pfReserveMut = useMutation({
    mutationFn: (b: object) => api.post('/grey/printed-fabric/reserve', b),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['printed-fabric-checked'] })
      qc.invalidateQueries({ queryKey: ['printed-ready-to-cut'] })
      setShowPFReserveForm(false)
    }
  })

  const openEdit = (entry: GreyEntry) => {
    setEditEntry(entry)
    setEditData({
      status: entry.status, dispatched_qty: entry.dispatched_qty,
      transport_qty: entry.transport_qty ?? 0, factory_qty: entry.factory_qty ?? 0,
      printer_qty: entry.printer_qty ?? 0, in_transit_qty: entry.in_transit_qty ?? 0,
      bilty_no: entry.bilty_no, vehicle_no: entry.vehicle_no, transporter: entry.transporter,
      dispatch_date: entry.dispatch_date, expected_arrival: entry.expected_arrival,
      rate: entry.rate ?? 0, delivery_location: entry.delivery_location ?? '',
      qc_status: entry.qc_status, qc_checked_by: entry.qc_checked_by, qc_remarks: entry.qc_remarks,
    })
  }

  const loadReport = async (key: string, url: string) => {
    setReportKey(key)
    const { data } = await api.get(url)
    if (Array.isArray(data)) setReportRows(data)
    else if (data?.trackers) setReportRows([data])
    else setReportRows([data])
  }

  const TABS: [Tab, string][] = [
    ['dashboard', '📊 Dashboard'],
    ['locations', '📍 Locations'],
    ['tracker', '🚛 Tracker'],
    ['mrp', '📐 MRP / SO'],
    ['jobwork', '🖨 Job work'],
    ['qc', '✅ QC'],
    ['ledger', '📜 Ledger'],
    ['reservations', '🔒 Reserved'],
    ['printed-fabric', '🖨️ Printed Fabric'],
    ['reports', '📑 Reports'],
  ]

  const lt = stats?.location_totals

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-gray-800">Grey Fabric</h1>
        <p className="text-sm text-gray-500 max-w-3xl">
          Material lifecycle: PO → transit → transport → factory/printer → QC → Printed Fabric → Ready to Cut
        </p>
      </div>

      <div className="flex flex-wrap gap-1 bg-gray-100 p-1 rounded-lg">
        {TABS.map(([key, label]) => (
          <button key={key} onClick={() => setTab(key)}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${tab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* DASHBOARD */}
      {tab === 'dashboard' && stats && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
            {[
              { label: 'TRACKERS', value: stats.total_trackers, color: 'text-gray-700' },
              { label: 'IN TRANSIT', value: stats.in_transit, color: 'text-blue-600' },
              { label: 'AT TRANSPORT', value: stats.at_transport ?? '—', color: 'text-cyan-600' },
              { label: 'FACTORY', value: stats.at_factory, color: 'text-purple-600' },
              { label: 'PRINTER / JW', value: stats.at_printer, color: 'text-orange-600' },
              { label: 'PENDING QC', value: stats.pending_qc, color: 'text-yellow-600' },
              { label: 'RESERVED', value: stats.hard_reserved, color: 'text-green-600' },
              { label: 'TRANSIT MTR', value: Number(stats.transit_meters ?? 0).toFixed(0), color: 'text-indigo-600' },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-white rounded-xl p-3 border border-gray-100 shadow-sm">
                <p className={`text-xl font-bold ${color}`}>{value}</p>
                <p className="text-[10px] text-gray-500 mt-1 font-semibold tracking-wide">{label}</p>
              </div>
            ))}
          </div>
          {lt && (
            <div className="bg-white rounded-xl border p-4">
              <h3 className="font-semibold text-gray-700 mb-2 text-sm">Quantity by location (MTR)</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                {Object.entries(lt).map(([k, v]) => (
                  <div key={k} className="flex justify-between border-b border-gray-50 py-1">
                    <span className="text-gray-500 capitalize">{k.replace(/_/g, ' ')}</span>
                    <span className="font-semibold text-gray-800">{Number(v).toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="bg-white rounded-xl border p-4">
            <h3 className="font-semibold text-gray-700 mb-3 text-sm">Active pipeline</h3>
            {entries.filter(e => !['Closed', 'QC Done'].includes(e.status)).slice(0, 10).map(e => (
              <div key={e.id} className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
                <div>
                  <p className="text-sm font-medium text-gray-700">{e.tracker_key} · {e.material_name || e.material_code}</p>
                  <p className="text-xs text-gray-400">{e.supplier} · PO {e.po_number}</p>
                </div>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(e.status)}`}>{e.status}</span>
              </div>
            ))}
            {entries.length === 0 && <p className="text-xs text-gray-400">No entries yet</p>}
          </div>
        </div>
      )}

      {/* LOCATIONS */}
      {tab === 'locations' && locData && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border p-4">
            <h3 className="font-semibold text-[#002B5B] mb-3 text-sm">Totals</h3>
            <pre className="text-xs bg-slate-50 p-3 rounded-lg overflow-x-auto">{JSON.stringify((locData as any).totals ?? locData, null, 2)}</pre>
          </div>
          <div className="bg-white rounded-xl border p-4">
            <h3 className="font-semibold text-[#002B5B] mb-3 text-sm">By material code</h3>
            <div className="overflow-x-auto max-h-96 overflow-y-auto text-sm">
              <table className="w-full border-collapse">
                <thead className="sticky top-0 bg-gray-50">
                  <tr>{['Material','Transit','Transport','Factory','Printer','Rejected','Return'].map(h => (
                    <th key={h} className="text-left px-2 py-2 text-xs font-semibold text-gray-500">{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {Object.entries((locData as any).by_material ?? {}).map(([code, buckets]: [string, any]) => (
                    <tr key={code} className="border-t border-gray-100">
                      <td className="px-2 py-1.5 font-mono text-xs">{code}</td>
                      <td className="px-2 py-1.5">{buckets.in_transit_qty}</td>
                      <td className="px-2 py-1.5">{buckets.transport_qty}</td>
                      <td className="px-2 py-1.5">{buckets.factory_qty}</td>
                      <td className="px-2 py-1.5">{buckets.printer_qty}</td>
                      <td className="px-2 py-1.5">{buckets.rejected_qty}</td>
                      <td className="px-2 py-1.5">{buckets.return_to_vendor_qty}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TRACKER */}
      {tab === 'tracker' && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All statuses</option>
              {STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <button onClick={() => setShowNewForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ New entry</button>
          </div>

          {showNewForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New grey fabric (PO-linked)</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {([['po_number','PO number'],['material_code','Material code'],['material_name','Material name'],['supplier','Supplier'],['so_reference','SO reference'],['delivery_location','Delivery location']] as const).map(([k, l]) => (
                  <div key={k}>
                    <label className="text-xs text-gray-500">{l}</label>
                    <input value={(newForm as any)[k]} onChange={e => setNewForm(f => ({ ...f, [k]: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                <div><label className="text-xs text-gray-500">Qty (MTR)</label>
                  <input type="number" value={newForm.ordered_qty} onChange={e => setNewForm(f => ({ ...f, ordered_qty: +e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div><label className="text-xs text-gray-500">Rate</label>
                  <input type="number" value={newForm.rate} onChange={e => setNewForm(f => ({ ...f, rate: +e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createMut.mutate(newForm)} disabled={createMut.isPending} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">{createMut.isPending ? 'Saving…' : 'Create'}</button>
                <button onClick={() => setShowNewForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}

          {editEntry && (
            <div className="bg-blue-50 rounded-xl border border-blue-200 p-4 space-y-3">
              <div className="flex justify-between">
                <h3 className="font-semibold text-gray-700">Update {editEntry.tracker_key}</h3>
                <button onClick={() => setEditEntry(null)} className="text-gray-400 hover:text-gray-600">✕</button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div>
                  <label className="text-xs text-gray-500">Status</label>
                  <select value={(editData.status as string) || editEntry.status} onChange={e => setEditData(d => ({ ...d, status: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {STATUSES.map(s => <option key={s}>{s}</option>)}
                  </select>
                </div>
                {[['in_transit_qty','In transit Qty'],['transport_qty','Transport Qty'],['factory_qty','Factory Qty'],['printer_qty','Printer Qty']].map(([k,l]) => (
                  <div key={k}>
                    <label className="text-xs text-gray-500">{l}</label>
                    <input type="number" value={(editData[k] as number) ?? (editEntry as any)[k] ?? 0} onChange={e => setEditData(d => ({ ...d, [k]: +e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                {[['bilty_no','Bilty / LR'],['transporter','Transporter'],['delivery_location','Delivery location']].map(([k,l]) => (
                  <div key={k}>
                    <label className="text-xs text-gray-500">{l}</label>
                    <input value={(editData[k] as string) ?? (editEntry as any)[k] ?? ''} onChange={e => setEditData(d => ({ ...d, [k]: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                <div><label className="text-xs text-gray-500">Rate</label>
                  <input type="number" value={(editData.rate as number) ?? editEntry.rate ?? 0} onChange={e => setEditData(d => ({ ...d, rate: +e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
              </div>
              <div className="flex flex-wrap gap-2">
                <button onClick={() => updateMut.mutate({ id: editEntry.id, data: editData })} disabled={updateMut.isPending} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">Save</button>
                <button onClick={() => { setDispatchModal(editEntry); setDispatchForm({ bilty_no: editEntry.bilty_no || '', transporter: editEntry.transporter || '', dispatch_date: editEntry.dispatch_date || '', expected_arrival: editEntry.expected_arrival || '', dispatched_qty: editEntry.dispatched_qty || editEntry.ordered_qty || 0, vehicle_no: editEntry.vehicle_no || '' }) }} className="px-3 py-2 bg-blue-600 text-white rounded-lg text-xs font-medium">Vendor dispatch → In transit</button>
                <button onClick={() => arriveMut.mutate({ id: editEntry.id })} className="px-3 py-2 bg-cyan-600 text-white rounded-lg text-xs font-medium">Arrive transport hub</button>
                <button onClick={() => { setTransferModal(editEntry); setTransferForm({ to_location: 'factory', qty: editEntry.transport_qty || 0 }) }} className="px-3 py-2 bg-purple-600 text-white rounded-lg text-xs font-medium">Transfer from transport…</button>
                <button onClick={() => { setQcModal(editEntry); setQcForm(f => ({ ...f, received_qty: editEntry.received_qty || editEntry.ordered_qty || 0, checked_qty: editEntry.checked_qty || 0, passed_qty: editEntry.passed_qty ?? 0, rejected_qty: editEntry.rejected_qty || 0, rework_qty: editEntry.rework_qty || 0 })) }} className="px-3 py-2 bg-amber-600 text-white rounded-lg text-xs font-medium">Record QC</button>
                <button onClick={() => { setPrinterModal(editEntry); setPrinterForm(f => ({ ...f, issue_qty: editEntry.transport_qty || editEntry.printer_qty || 0 })) }} className="px-3 py-2 bg-orange-600 text-white rounded-lg text-xs font-medium">Issue to printer</button>
                <button onClick={() => { setReturnModal(editEntry); setReturnForm(r => ({ ...r, return_qty: editEntry.rejected_qty || 0 })) }} className="px-3 py-2 bg-red-700 text-white rounded-lg text-xs font-medium">Return to vendor (DN)</button>
                <button onClick={() => setEditEntry(null)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Close</button>
              </div>
            </div>
          )}

          {/* Dispatch Modal */}
          {dispatchModal && (
            <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
              <div className="bg-white rounded-xl max-w-md w-full p-5 space-y-3 shadow-xl">
                <h3 className="font-semibold text-gray-800">Vendor dispatch (bilty)</h3>
                {(['bilty_no','transporter','dispatch_date','expected_arrival','vehicle_no'] as const).map(f => (
                  <div key={f}><label className="text-xs text-gray-500">{f.replace(/_/g, ' ')}</label>
                    <input className="w-full border rounded px-2 py-1.5 text-sm mt-0.5" value={(dispatchForm as any)[f]} onChange={e => setDispatchForm(x => ({ ...x, [f]: e.target.value }))} /></div>
                ))}
                <div><label className="text-xs text-gray-500">Dispatched Qty (MTR)</label>
                  <input type="number" className="w-full border rounded px-2 py-1.5 text-sm mt-0.5" value={dispatchForm.dispatched_qty} onChange={e => setDispatchForm(x => ({ ...x, dispatched_qty: +e.target.value }))} /></div>
                <div className="flex gap-2 pt-2">
                  <button onClick={() => dispatchMut.mutate({ id: dispatchModal.id, body: dispatchForm })} disabled={dispatchMut.isPending || !dispatchForm.bilty_no} className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">Confirm dispatch</button>
                  <button onClick={() => setDispatchModal(null)} className="px-4 py-2 border rounded-lg text-sm">Cancel</button>
                </div>
              </div>
            </div>
          )}

          {/* Transfer Modal */}
          {transferModal && (
            <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
              <div className="bg-white rounded-xl max-w-sm w-full p-5 space-y-3 shadow-xl">
                <h3 className="font-semibold">Move from transport hub</h3>
                <select value={transferForm.to_location} onChange={e => setTransferForm(f => ({ ...f, to_location: e.target.value as 'factory' | 'printer' }))} className="w-full border rounded px-2 py-2 text-sm">
                  <option value="factory">→ Factory / inhouse</option>
                  <option value="printer">→ Direct printer (job work)</option>
                </select>
                <input type="number" placeholder="Qty MTR" value={transferForm.qty || ''} onChange={e => setTransferForm(f => ({ ...f, qty: +e.target.value }))} className="w-full border rounded px-2 py-2 text-sm" />
                <div className="flex gap-2">
                  <button onClick={() => transferMut.mutate({ id: transferModal.id, body: transferForm })} className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm">Transfer</button>
                  <button onClick={() => setTransferModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
                </div>
              </div>
            </div>
          )}

          {/* QC Modal */}
          {qcModal && (
            <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4 overflow-y-auto">
              <div className="bg-white rounded-xl max-w-lg w-full p-5 space-y-3 shadow-xl my-8">
                <h3 className="font-semibold">Grey Fabric QC</h3>
                {(['received_qty','checked_qty','passed_qty','rejected_qty','rework_qty'] as const).map(f => (
                  <div key={f}><label className="text-xs text-gray-500">{f.replace(/_/g, ' ')}</label>
                    <input type="number" className="w-full border rounded px-2 py-1.5 text-sm mt-0.5" value={qcForm[f]} onChange={e => setQcForm(x => ({ ...x, [f]: +e.target.value }))} /></div>
                ))}
                <div><label className="text-xs text-gray-500">Outcome</label>
                  <select value={qcForm.outcome} onChange={e => setQcForm(x => ({ ...x, outcome: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-0.5">
                    {QC_OUTCOMES.map(o => <option key={o}>{o}</option>)}
                  </select></div>
                {(['qc_remarks','qc_by','qc_date'] as const).map(f => (
                  <div key={f}><label className="text-xs text-gray-500">{f.replace(/_/g, ' ')}</label>
                    <input className="w-full border rounded px-2 py-1.5 text-sm mt-0.5" value={qcForm[f]} onChange={e => setQcForm(x => ({ ...x, [f]: e.target.value }))} /></div>
                ))}
                <div className="flex gap-2">
                  <button onClick={() => qcMut.mutate({ id: qcModal.id, body: qcForm })} className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm">Submit QC</button>
                  <button onClick={() => setQcModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
                </div>
              </div>
            </div>
          )}

          {/* Return Modal */}
          {returnModal && (
            <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
              <div className="bg-white rounded-xl max-w-md w-full p-5 space-y-3 shadow-xl">
                <h3 className="font-semibold text-red-800">Return to vendor</h3>
                <input type="number" placeholder="Return qty (MTR)" className="w-full border rounded px-2 py-1.5 text-sm" value={returnForm.return_qty || ''} onChange={e => setReturnForm(f => ({ ...f, return_qty: +e.target.value }))} />
                {(['debit_note_no','return_challan','return_date','remarks'] as const).map(k => (
                  <input key={k} placeholder={k.replace(/_/g, ' ')} className="w-full border rounded px-2 py-1.5 text-sm" value={returnForm[k]} onChange={e => setReturnForm(f => ({ ...f, [k]: e.target.value }))} />
                ))}
                <div className="flex gap-2">
                  <button onClick={() => returnVendorMut.mutate({ id: returnModal.id, body: returnForm })} className="flex-1 py-2 bg-red-700 text-white rounded-lg text-sm">Post return</button>
                  <button onClick={() => setReturnModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
                </div>
              </div>
            </div>
          )}

          {/* Printer Issue Modal */}
          {printerModal && (
            <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
              <div className="bg-white rounded-xl max-w-md w-full p-5 space-y-3 shadow-xl">
                <h3 className="font-semibold">Grey issue to printer</h3>
                {([['job_order_no','Job order no'],['from_location','From location'],['to_vendor','Printer vendor'],['issue_date','Issue date'],['challan_no','Challan no'],['gate_pass','Gate pass']] as const).map(([k, lab]) => (
                  <div key={k}><label className="text-xs text-gray-500">{lab}</label>
                    <input className="w-full border rounded px-2 py-1.5 text-sm mt-0.5" value={(printerForm as any)[k]} onChange={e => setPrinterForm(x => ({ ...x, [k]: e.target.value }))} /></div>
                ))}
                <div><label className="text-xs text-gray-500">Issue Qty (MTR)</label>
                  <input type="number" className="w-full border rounded px-2 py-1.5 text-sm mt-0.5" value={printerForm.issue_qty} onChange={e => setPrinterForm(x => ({ ...x, issue_qty: +e.target.value }))} /></div>
                <div className="flex gap-2">
                  <button onClick={() => printerMut.mutate({ tracker_id: printerModal.id, material_code: printerModal.material_code, ...printerForm })} className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm">Create issue</button>
                  <button onClick={() => setPrinterModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
                </div>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {entries.map(e => (
              <div key={e.id} className="bg-white rounded-xl border shadow-sm p-4">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap items-center gap-2 mb-1">
                      <p className="font-semibold text-gray-800 text-sm">{e.tracker_key}</p>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(e.status)}`}>{e.status}</span>
                    </div>
                    <p className="text-sm text-gray-600">{e.material_name || e.material_code} · {e.supplier}</p>
                    <p className="text-xs text-gray-400">PO {e.po_number} {e.so_reference ? `· SO ${e.so_reference}` : ''}</p>
                    <div className="flex flex-wrap gap-3 mt-2 text-xs text-gray-600">
                      <span>Ord <b>{e.ordered_qty}</b></span>
                      <span className="text-blue-700">IT <b>{e.in_transit_qty ?? 0}</b></span>
                      <span className="text-cyan-700">T <b>{e.transport_qty ?? 0}</b></span>
                      <span className="text-purple-700">F <b>{e.factory_qty ?? 0}</b></span>
                      <span className="text-orange-700">P <b>{e.printer_qty ?? 0}</b></span>
                    </div>
                  </div>
                  <button onClick={() => openEdit(e)} className="text-xs px-2 py-1 border border-gray-200 rounded text-gray-500 hover:bg-gray-50 shrink-0">✏️ Actions</button>
                </div>
              </div>
            ))}
            {entries.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No grey fabric trackers.</p>}
          </div>
        </div>
      )}

      {/* MRP */}
      {tab === 'mrp' && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border p-4 space-y-3">
            <h3 className="font-semibold text-[#002B5B] text-sm">Add MRP requirement line</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {(['run_label','material_code','material_name','so_number','sku','notes'] as const).map(k => (
                <div key={k}><label className="text-xs text-gray-500">{k}</label>
                  <input className="w-full border rounded px-2 py-1.5 text-sm" value={(mrpForm as any)[k]} onChange={e => setMrpForm(f => ({ ...f, [k]: e.target.value }))} /></div>
              ))}
              <div><label className="text-xs text-gray-500">Qty required (MTR)</label>
                <input type="number" className="w-full border rounded px-2 py-1.5 text-sm" value={mrpForm.qty_required} onChange={e => setMrpForm(f => ({ ...f, qty_required: +e.target.value }))} /></div>
            </div>
            <button onClick={() => mrpCreateMut.mutate(mrpForm)} disabled={!mrpForm.material_code} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">Add line</button>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white rounded-xl border p-4">
              <h3 className="font-semibold text-sm mb-2">Totals by material</h3>
              <div className="max-h-64 overflow-y-auto text-sm space-y-1">
                {(mrpTotals as any[]).map(r => (
                  <button key={r.material_code} onClick={() => setDrillMaterial(r.material_code)} className="w-full flex justify-between py-1 border-b border-gray-50 hover:bg-gray-50 px-1 rounded text-left">
                    <span className="font-mono text-xs">{r.material_code}</span>
                    <span className="font-semibold">{r.total_required}</span>
                  </button>
                ))}
              </div>
            </div>
            <div className="bg-white rounded-xl border p-4">
              <h3 className="font-semibold text-sm mb-2">SO / SKU breakup</h3>
              {drilldown && (
                <div className="text-sm">
                  <p className="text-gray-600 mb-2">Total: <b>{(drilldown as any).total_qty_required}</b></p>
                  <ul className="space-y-1 max-h-56 overflow-y-auto">
                    {((drilldown as any).lines || []).map((ln: any, i: number) => (
                      <li key={i} className="text-xs border-b border-gray-50 py-1">SO {ln.so_number || '—'} · SKU {ln.sku || '—'} → <b>{ln.qty_required}</b> MTR</li>
                    ))}
                  </ul>
                </div>
              )}
              {!drillMaterial && <p className="text-xs text-gray-400">Click a material in the left list.</p>}
            </div>
          </div>
          <div className="bg-white rounded-xl border overflow-hidden">
            <div className="px-3 py-2 bg-gray-50 text-xs font-semibold text-gray-600">All requirement lines</div>
            <table className="w-full text-xs">
              <thead><tr className="text-left text-gray-400 border-b">{['Mat.','SO','SKU','Qty','Run'].map(h => <th key={h} className="px-2 py-2">{h}</th>)}</tr></thead>
              <tbody>
                {(mrpReqs as any[]).map(r => (
                  <tr key={r.id} className="border-b border-gray-50">
                    <td className="px-2 py-1.5 font-mono">{r.material_code}</td>
                    <td className="px-2 py-1.5">{r.so_number}</td>
                    <td className="px-2 py-1.5">{r.sku}</td>
                    <td className="px-2 py-1.5 font-semibold">{r.qty_required}</td>
                    <td className="px-2 py-1.5 text-gray-500">{r.run_label}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* JOB WORK */}
      {tab === 'jobwork' && (
        <div className="space-y-4">
          <p className="text-sm text-gray-600">Grey fabric issues to printer. Balance = issue − received-back.</p>
          <div className="bg-white rounded-xl border overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-xs text-gray-500">
                <tr>{['Job #','Tracker','Qty','Bal.','Vendor','Challan','Date',''].map(h => <th key={h} className="text-left px-3 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {(printerIssues as any[]).map(p => (
                  <tr key={p.id} className="border-t border-gray-100">
                    <td className="px-3 py-2">{p.job_order_no}</td>
                    <td className="px-3 py-2 font-mono text-xs">#{p.tracker_id}</td>
                    <td className="px-3 py-2">{p.issue_qty}</td>
                    <td className="px-3 py-2 font-semibold text-amber-700">{p.balance_qty}</td>
                    <td className="px-3 py-2">{p.to_vendor}</td>
                    <td className="px-3 py-2 text-xs">{p.challan_no}</td>
                    <td className="px-3 py-2 text-xs">{p.issue_date}</td>
                    <td className="px-3 py-2">
                      <button className="text-blue-600 text-xs" onClick={() => { setReceiveModal({ issueId: p.id, trackerId: p.tracker_id }); setReceiveForm({ received_back_qty: p.balance_qty, grey_input_mtr: 0, printed_item_code: '', printed_output_mtr: 0, wastage_mtr: 0, conversion_date: '', remarks: '' }) }}>Receive printed</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {receiveModal && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl max-w-md w-full p-5 space-y-3 shadow-xl">
            <h3 className="font-semibold">Receive printed fabric</h3>
            <input type="number" placeholder="Received back (MTR)" className="w-full border rounded px-2 py-1.5 text-sm" value={receiveForm.received_back_qty} onChange={e => setReceiveForm(f => ({ ...f, received_back_qty: +e.target.value }))} />
            <input placeholder="Printed item code" className="w-full border rounded px-2 py-1.5 text-sm" value={receiveForm.printed_item_code} onChange={e => setReceiveForm(f => ({ ...f, printed_item_code: e.target.value }))} />
            <div className="grid grid-cols-3 gap-2">
              <input type="number" placeholder="Grey in" className="border rounded px-2 py-1 text-xs" value={receiveForm.grey_input_mtr || ''} onChange={e => setReceiveForm(f => ({ ...f, grey_input_mtr: +e.target.value }))} />
              <input type="number" placeholder="Printed out" className="border rounded px-2 py-1 text-xs" value={receiveForm.printed_output_mtr || ''} onChange={e => setReceiveForm(f => ({ ...f, printed_output_mtr: +e.target.value }))} />
              <input type="number" placeholder="Wastage" className="border rounded px-2 py-1 text-xs" value={receiveForm.wastage_mtr || ''} onChange={e => setReceiveForm(f => ({ ...f, wastage_mtr: +e.target.value }))} />
            </div>
            <div className="flex gap-2">
              <button onClick={() => receivePrintedMut.mutate({ issueId: receiveModal.issueId, body: receiveForm })} className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm">Post receipt</button>
              <button onClick={() => setReceiveModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* QC TAB */}
      {tab === 'qc' && (
        <div className="space-y-4">
          <p className="text-sm text-gray-600">Grey fabric QC events.</p>
          <div className="bg-white rounded-xl border overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-xs">
                <tr>{['Tracker','Recv','Chk','Pass','Rej','Rework','Outcome','By','Date'].map(h => <th key={h} className="text-left px-2 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {(qcEvents as any[]).map(q => (
                  <tr key={q.id} className="border-t border-gray-100 text-xs">
                    <td className="px-2 py-1.5">#{q.tracker_id}</td>
                    <td className="px-2 py-1.5">{q.received_qty}</td>
                    <td className="px-2 py-1.5">{q.checked_qty}</td>
                    <td className="px-2 py-1.5">{q.passed_qty}</td>
                    <td className="px-2 py-1.5">{q.rejected_qty}</td>
                    <td className="px-2 py-1.5">{q.rework_qty}</td>
                    <td className="px-2 py-1.5 font-medium">{q.outcome}</td>
                    <td className="px-2 py-1.5">{q.qc_by}</td>
                    <td className="px-2 py-1.5">{q.qc_date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* LEDGER */}
      {tab === 'ledger' && (
        <div className="bg-white rounded-xl border overflow-hidden">
          <div className="px-4 py-3 border-b bg-gray-50"><p className="text-sm font-semibold text-gray-700">Ledger (last 500)</p></div>
          <table className="w-full text-sm">
            <thead className="text-gray-400 text-xs uppercase">
              <tr>{['Date','Material','Type','Qty','From','To','Ref','Remarks'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
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
        </div>
      )}

      {/* RESERVATIONS */}
      {tab === 'reservations' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <button onClick={() => setShowResForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ Reserve fabric</button>
          </div>
          {showResForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {(['fabric_code','fabric_name','so_number','sku'] as const).map(k => (
                  <div key={k}><label className="text-xs text-gray-500">{k}</label>
                    <input value={(resForm as any)[k]} onChange={e => setResForm(f => ({ ...f, [k]: e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                ))}
                <div><label className="text-xs text-gray-500">Qty MTR</label>
                  <input type="number" value={resForm.qty} onChange={e => setResForm(f => ({ ...f, qty: +e.target.value }))} className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createResMut.mutate(resForm)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm">Reserve</button>
                <button onClick={() => setShowResForm(false)} className="px-4 py-2 border rounded-lg text-sm">Cancel</button>
              </div>
            </div>
          )}
          <div className="bg-white rounded-xl border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                <tr>{['Fabric','SO','SKU','Qty','Date',''].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {hardRes.map(r => (
                  <tr key={r.id} className="border-t border-gray-50">
                    <td className="px-4 py-2 font-mono text-xs">{r.fabric_code}</td>
                    <td className="px-4 py-2">{r.so_number || '—'}</td>
                    <td className="px-4 py-2">{r.sku || '—'}</td>
                    <td className="px-4 py-2 font-semibold text-green-700">{r.qty}</td>
                    <td className="px-4 py-2 text-xs text-gray-400">{r.reserved_date?.split('T')[0]}</td>
                    <td className="px-4 py-2"><button onClick={() => releaseResMut.mutate(r.id)} className="text-xs text-red-600">Release</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* PRINTED FABRIC */}
      {tab === 'printed-fabric' && (
        <div className="space-y-4">
          <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit">
            {([['unchecked','⏳ Unchecked'],['checked','📦 Checked Stock'],['ready-to-cut','✂️ Ready to Cut']] as const).map(([key, label]) => (
              <button key={key} onClick={() => setPfSubTab(key)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${pfSubTab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
                {label}
              </button>
            ))}
          </div>

          {/* Unchecked */}
          {pfSubTab === 'unchecked' && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-3 border-b bg-amber-50">
                <p className="text-sm font-semibold text-gray-700">🖨️ Printed Fabric — Unchecked Warehouse</p>
                <p className="text-xs text-gray-500">JWO GRN ke baad aaya fabric — QC pending</p>
              </div>
              <table className="w-full text-sm">
                <thead className="text-gray-400 text-xs uppercase">
                  <tr>{['Fabric Code','Name','Printer','Qty (MTR)','JWO Ref','GRN Ref','Received Date','Action'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
                </thead>
                <tbody>
                  {(printedFabricUnchecked as any[]).map((u: any, i: number) => (
                    <tr key={i} className="border-t border-gray-50 hover:bg-gray-50">
                      <td className="px-4 py-2 font-mono text-xs text-[#002B5B] font-semibold">{u.fabric_code}</td>
                      <td className="px-4 py-2 text-gray-700">{u.fabric_name || '—'}</td>
                      <td className="px-4 py-2 text-gray-500">{u.printer || '—'}</td>
                      <td className="px-4 py-2 font-semibold text-amber-600">{u.qty} m</td>
                      <td className="px-4 py-2 text-xs text-blue-600">{u.jwo_ref || '—'}</td>
                      <td className="px-4 py-2 text-xs text-gray-400">{u.grn_ref || '—'}</td>
                      <td className="px-4 py-2 text-xs text-gray-400">{u.receive_date || '—'}</td>
                      <td className="px-4 py-2">
                        <button onClick={() => {
                          setPFQCTarget(u)
                          setPFQCForm({ fabric_code: u.fabric_code, fabric_name: u.fabric_name || '', jwo_ref: u.jwo_ref || '', passed_qty: u.qty, failed_qty: 0, qc_by: '', qc_date: '' })
                          setShowPFQCForm(true)
                        }} className="text-xs px-3 py-1 bg-blue-600 text-white rounded-lg hover:bg-blue-700">✅ QC Check</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {printedFabricUnchecked.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No printed fabric pending QC.</p>}
            </div>
          )}

          {/* Checked Stock */}
          {pfSubTab === 'checked' && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-3 border-b bg-green-50 flex justify-between items-center">
                <div>
                  <p className="text-sm font-semibold text-gray-700">📦 Checked Printed Fabric Stock</p>
                  <p className="text-xs text-gray-500">QC pass hua fabric — Ready to Cut ke liye available</p>
                </div>
                <p className="text-xs text-gray-500 font-semibold">
                  Total: {(printedFabricChecked as any[]).reduce((a: number, c: any) => a + (c.available_qty || 0), 0).toFixed(1)}m available
                </p>
              </div>
              <table className="w-full text-sm">
                <thead className="text-gray-400 text-xs uppercase">
                  <tr>{['Fabric Code','Name','Checked','Passed','Reserved','Available'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
                </thead>
                <tbody>
                  {(printedFabricChecked as any[]).map((c: any, i: number) => (
                    <tr key={i} className="border-t border-gray-50 hover:bg-gray-50">
                      <td className="px-4 py-2 font-mono text-xs text-[#002B5B] font-semibold">{c.fabric_code}</td>
                      <td className="px-4 py-2 text-gray-700">{c.fabric_name}</td>
                      <td className="px-4 py-2 text-gray-600">{c.checked_qty} m</td>
                      <td className="px-4 py-2 text-green-600 font-semibold">{c.passed_qty} m</td>
                      <td className="px-4 py-2 text-purple-600">{c.reserved_qty || 0} m</td>
                      <td className="px-4 py-2 font-bold text-[#002B5B]">{c.available_qty || 0} m</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {printedFabricChecked.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No checked fabric yet. QC submit karo pehle.</p>}
            </div>
          )}

          {/* Ready to Cut */}
          {pfSubTab === 'ready-to-cut' && (
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <div>
                  <p className="text-sm font-semibold text-gray-700">✂️ Ready to Cut</p>
                  <p className="text-xs text-gray-500">Checked printed fabric — SO ke against reserve karke cutting ke liye bhejo</p>
                </div>
                <button onClick={() => setShowPFReserveForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ Reserve for SO</button>
              </div>
              {showPFReserveForm && (
                <div className="bg-white rounded-xl border p-4 space-y-3">
                  <h3 className="font-semibold text-gray-700">Reserve Printed Fabric against SO</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {([['fabric_code','Fabric Code *'],['fabric_name','Fabric Name'],['so_number','SO Number *'],['sku','SKU'],['remarks','Remarks']] as const).map(([k,l]) => (
                      <div key={k}><label className="text-xs text-gray-500">{l}</label>
                        <input value={(pfReserveForm as any)[k]} onChange={e => setPFReserveForm(f => ({ ...f, [k]: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                    ))}
                    <div><label className="text-xs text-gray-500">Qty (MTR)</label>
                      <input type="number" value={pfReserveForm.qty} onChange={e => setPFReserveForm(f => ({ ...f, qty: +e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                  </div>
                  <div className="flex gap-2">
                    <button onClick={() => pfReserveMut.mutate(pfReserveForm)} disabled={pfReserveMut.isPending || !pfReserveForm.fabric_code || !pfReserveForm.so_number} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">{pfReserveMut.isPending ? 'Saving…' : '🔒 Reserve'}</button>
                    <button onClick={() => setShowPFReserveForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
                  </div>
                </div>
              )}
              <div className="bg-white rounded-xl border overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="text-gray-400 text-xs uppercase bg-gray-50">
                    <tr>{['SO Number','SKU','Fabric Code','Reserved Qty','Available','Status','Action'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
                  </thead>
                  <tbody>
                    {(printedReadyToCut as any[]).map((r: any, i: number) => (
                      <tr key={i} className="border-t border-gray-50 hover:bg-gray-50">
                        <td className="px-4 py-2 font-semibold text-[#002B5B]">{r.so_number || '—'}</td>
                        <td className="px-4 py-2 text-gray-600">{r.sku || '—'}</td>
                        <td className="px-4 py-2 font-mono text-xs font-semibold">{r.fabric_code}</td>
                        <td className="px-4 py-2 font-semibold text-purple-600">{r.reserved_qty} m</td>
                        <td className="px-4 py-2 text-blue-600">{r.available_qty || 0} m</td>
                        <td className="px-4 py-2">
                          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${r.cut_status === 'Ready to Cut' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-600'}`}>{r.cut_status}</span>
                        </td>
                        <td className="px-4 py-2">
                          <button onClick={() => {
                            const p = new URLSearchParams({ fabric: r.fabric_code, qty: String(r.reserved_qty), so: r.so_number || '', sku: r.sku || '' })
                            window.location.href = `/production?${p.toString()}`
                          }} className="text-xs px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700">✂️ Create JO</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {printedReadyToCut.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No reservations yet. Checked Stock mein fabric reserve karo.</p>}
              </div>
            </div>
          )}

          {/* Printed Fabric QC Modal */}
          {showPFQCForm && pfQCTarget && (
            <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
              <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="font-semibold text-gray-700">✅ Printed Fabric QC</h3>
                    <p className="text-xs text-gray-400">{pfQCTarget.fabric_code} · JWO: {pfQCTarget.jwo_ref}</p>
                  </div>
                  <button onClick={() => setShowPFQCForm(false)} className="text-gray-400 hover:text-gray-600 text-xl">✕</button>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div><label className="text-xs text-gray-500">Fabric Code</label>
                    <input value={pfQCForm.fabric_code} readOnly className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1 bg-gray-50 font-mono" /></div>
                  <div><label className="text-xs text-gray-500">Total Received</label>
                    <input value={pfQCTarget.qty + ' m'} readOnly className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1 bg-gray-50" /></div>
                  <div><label className="text-xs text-gray-500">Passed Qty (MTR) ✅</label>
                    <input type="number" value={pfQCForm.passed_qty} onChange={e => setPFQCForm(f => ({ ...f, passed_qty: +e.target.value }))} className="w-full border border-green-300 rounded px-2 py-1.5 text-sm mt-1 bg-green-50 font-semibold" /></div>
                  <div><label className="text-xs text-gray-500">Failed Qty (MTR) ❌</label>
                    <input type="number" value={pfQCForm.failed_qty} onChange={e => setPFQCForm(f => ({ ...f, failed_qty: +e.target.value }))} className="w-full border border-red-300 rounded px-2 py-1.5 text-sm mt-1 bg-red-50" /></div>
                  <div><label className="text-xs text-gray-500">QC By</label>
                    <input value={pfQCForm.qc_by} onChange={e => setPFQCForm(f => ({ ...f, qc_by: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                  <div><label className="text-xs text-gray-500">QC Date</label>
                    <input type="date" value={pfQCForm.qc_date} onChange={e => setPFQCForm(f => ({ ...f, qc_date: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                </div>
                {pfQCForm.passed_qty + pfQCForm.failed_qty > 0 && (
                  <div className={`text-xs px-3 py-2 rounded-lg ${Math.abs((pfQCForm.passed_qty + pfQCForm.failed_qty) - pfQCTarget.qty) < 0.01 ? 'bg-green-50 text-green-700' : 'bg-amber-50 text-amber-700'}`}>
                    Pass + Fail = {pfQCForm.passed_qty + pfQCForm.failed_qty}m
                    {Math.abs((pfQCForm.passed_qty + pfQCForm.failed_qty) - pfQCTarget.qty) < 0.01 ? ' ✅ Match!' : ` ⚠️ Total should be ${pfQCTarget.qty}m`}
                  </div>
                )}
                <div className="bg-blue-50 rounded-lg p-3 text-xs text-blue-700">
                  ✅ Passed qty → <b>Checked Stock</b> → Reserve for SO → ✂️ Create JO
                </div>
                <div className="flex gap-2">
                  <button onClick={() => printedFabricQCMut.mutate(pfQCForm)} disabled={printedFabricQCMut.isPending} className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                    {printedFabricQCMut.isPending ? 'Saving…' : '✅ Submit QC'}
                  </button>
                  <button onClick={() => setShowPFQCForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* REPORTS */}
      {tab === 'reports' && (
        <div className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {[['transit','/grey/reports/transit','Grey transit'],['stock','/grey/reports/stock-locations','Stock by location'],['qc','/grey/reports/qc','QC events'],['rej','/grey/reports/rejects-returns','Rejected / returns'],['prn','/grey/reports/printer-issues','Printer issues'],['cons','/grey/reports/consumption','Consumption']].map(([k, url, lab]) => (
              <button key={k} onClick={() => loadReport(k, url)} className="px-3 py-2 rounded-lg border text-sm bg-white hover:bg-gray-50 border-gray-200">{lab}</button>
            ))}
          </div>
          {reportKey && (
            <div className="bg-white rounded-xl border p-3">
              <p className="text-xs text-gray-500 mb-2">Report: {reportKey}</p>
              <pre className="text-xs bg-slate-50 p-3 rounded-lg overflow-x-auto max-h-96">{JSON.stringify(reportRows, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}