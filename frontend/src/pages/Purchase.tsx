import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type Tab = 'dashboard' | 'suppliers' | 'processors' | 'pr' | 'po' | 'jwo' | 'grn'

interface Stats { open_prs: number; open_pos: number; open_jwos: number; pending_grns: number; total_suppliers: number; total_processors: number }
interface Supplier { id: number; supplier_code: string; supplier_name: string; supplier_type: string; contact_person: string; email: string; phone: string; payment_terms: string; active: number }
interface Processor { id: number; processor_code: string; processor_name: string; processor_type: string; contact_person: string; phone: string }
interface PR { id: number; pr_number: string; pr_date: string; requested_by: string; department: string; priority: string; status: string; so_reference: string; lines: PRLine[] }
interface PRLine { id: number; material_code: string; material_name: string; material_type: string; required_qty: number; unit: string; required_by_date: string }
interface PO { id: number; po_number: string; po_date: string; supplier_name: string; status: string; total: number; delivery_date: string; lines: POLine[] }
interface POLine { id: number; material_code: string; material_name: string; po_qty: number; unit: string; rate: number; gst_pct: number; amount: number }
interface JWO { id: number; jwo_number: string; jwo_date: string; processor_name: string; status: string; total: number; expected_return_date: string; lines: JWOLine[] }
interface JWOLine { id: number; input_material: string; output_material: string; output_qty: number; process_type: string; rate: number; amount: number }
interface GRN { id: number; grn_number: string; grn_date: string; grn_type: string; party_name: string; status: string; total_value: number; lines: GRNLine[] }
interface GRNLine { id: number; material_code: string; received_qty: number; accepted_qty: number; rejected_qty: number; qc_status: string; rate: number; amount: number }

const SUP_TYPES = ['Fabric Supplier', 'Accessories Supplier', 'Job Work', 'Others']
const PROC_TYPES = ['Printing Unit', 'Dyeing Unit', 'Embroidery', 'Others']
const DEPTS = ['Production', 'Stores', 'Others']
const PRIORITIES = ['Low', 'Normal', 'High', 'Urgent']
const MAT_TYPES = ['RM', 'ACC', 'PKG', 'SFG', 'FG']
const PROC_TYPES2 = ['Printing', 'Dyeing', 'Embroidery', 'Other']
const GRN_TYPES = ['PO Receipt', 'JWO Receipt']

const statusColor = (s: string) => {
  if (['Approved', 'Received', 'Verified', 'Posted', 'Closed'].includes(s)) return 'bg-green-100 text-green-700'
  if (['Draft', 'Pending Approval'].includes(s)) return 'bg-yellow-100 text-yellow-700'
  if (['Sent to Supplier', 'In Process', 'Confirmed', 'Issued to Processor'].includes(s)) return 'bg-blue-100 text-blue-700'
  if (['Rejected', 'Cancelled'].includes(s)) return 'bg-red-100 text-red-700'
  return 'bg-gray-100 text-gray-600'
}

const fmt = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN')

export default function Purchase() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('dashboard')
  const [expanded, setExpanded] = useState<number | null>(null)
  const [filterStatus, setFilterStatus] = useState('')

  // Supplier form
  const [showSupForm, setShowSupForm] = useState(false)
  const [supForm, setSupForm] = useState({ supplier_name: '', supplier_type: 'Others', contact_person: '', email: '', phone: '', address: '', gst_number: '', payment_terms: 'Net 30' })

  // Processor form
  const [showProcForm, setShowProcForm] = useState(false)
  const [procForm, setProcForm] = useState({ processor_name: '', processor_type: 'Others', contact_person: '', email: '', phone: '', address: '' })

  // PR form
  const [showPRForm, setShowPRForm] = useState(false)
  const [prForm, setPRForm] = useState({ requested_by: '', department: 'Production', priority: 'Normal', so_reference: '', notes: '' })
  const [prLines, setPRLines] = useState<{ material_code: string; material_name: string; material_type: string; required_qty: number; unit: string; required_by_date: string }[]>([])

  // PO form
  const [showPOForm, setShowPOForm] = useState(false)
  const [poForm, setPOForm] = useState({ supplier_id: undefined as number | undefined, supplier_name: '', delivery_date: '', payment_terms: '', delivery_location: '', pr_reference: '', so_reference: '', remarks: '' })
  const [poLines, setPOLines] = useState<{ material_code: string; material_name: string; material_type: string; po_qty: number; unit: string; rate: number; gst_pct: number }[]>([])

  // JWO form
  const [showJWOForm, setShowJWOForm] = useState(false)
  const [jwoForm, setJWOForm] = useState({ processor_id: undefined as number | undefined, processor_name: '', expected_return_date: '', pr_reference: '', so_reference: '', issued_by: '', remarks: '' })
  const [jwoLines, setJWOLines] = useState<{ input_material: string; input_qty: number; output_material: string; output_qty: number; process_type: string; rate: number }[]>([])

  // GRN form
  const [showGRNForm, setShowGRNForm] = useState(false)
  const [grnForm, setGRNForm] = useState({ grn_type: 'PO Receipt', reference_number: '', party_name: '', challan_no: '', invoice_no: '', vehicle_no: '', transporter: '', warehouse: '', remarks: '' })
  const [grnLines, setGRNLines] = useState<{ material_code: string; material_name: string; received_qty: number; accepted_qty: number; rejected_qty: number; unit: string; rate: number; qc_status: string }[]>([])

  const { data: stats } = useQuery<Stats>({ queryKey: ['purchase-stats'], queryFn: () => api.get('/purchase/stats').then(r => r.data) })
  const { data: suppliers = [] } = useQuery<Supplier[]>({ queryKey: ['suppliers'], queryFn: () => api.get('/purchase/suppliers').then(r => r.data), enabled: tab === 'suppliers' || tab === 'dashboard' })
  const { data: processors = [] } = useQuery<Processor[]>({ queryKey: ['processors'], queryFn: () => api.get('/purchase/processors').then(r => r.data), enabled: tab === 'processors' || tab === 'jwo' })
  const { data: prs = [] } = useQuery<PR[]>({ queryKey: ['prs', filterStatus], queryFn: () => api.get('/purchase/pr' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data), enabled: tab === 'pr' })
  const { data: pos = [] } = useQuery<PO[]>({ queryKey: ['pos', filterStatus], queryFn: () => api.get('/purchase/po' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data), enabled: tab === 'po' })
  const { data: jwos = [] } = useQuery<JWO[]>({ queryKey: ['jwos', filterStatus], queryFn: () => api.get('/purchase/jwo' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data), enabled: tab === 'jwo' })
  const { data: grns = [] } = useQuery<GRN[]>({ queryKey: ['grns', filterStatus], queryFn: () => api.get('/purchase/grn' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data), enabled: tab === 'grn' })

  const invalidate = () => qc.invalidateQueries({ queryKey: ['purchase-stats'] })
  const createSupMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/suppliers', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['suppliers'] }); invalidate(); setShowSupForm(false) } })
  const createProcMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/processors', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['processors'] }); setShowProcForm(false) } })
  const createPRMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/pr', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['prs'] }); invalidate(); setShowPRForm(false); setPRLines([]) } })
  const approvePRMut = useMutation({ mutationFn: ({ id, approver }: { id: number; approver: string }) => api.post(`/purchase/pr/${id}/approve`, { approver }), onSuccess: () => qc.invalidateQueries({ queryKey: ['prs'] }) })
  const createPOMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/po', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['pos'] }); invalidate(); setShowPOForm(false); setPOLines([]) } })
  const updatePOStatusMut = useMutation({ mutationFn: ({ id, status }: { id: number; status: string }) => api.patch(`/purchase/po/${id}/status`, { status }), onSuccess: () => qc.invalidateQueries({ queryKey: ['pos'] }) })
  const createJWOMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/jwo', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['jwos'] }); invalidate(); setShowJWOForm(false); setJWOLines([]) } })
  const updateJWOStatusMut = useMutation({ mutationFn: ({ id, status }: { id: number; status: string }) => api.patch(`/purchase/jwo/${id}/status`, { status }), onSuccess: () => qc.invalidateQueries({ queryKey: ['jwos'] }) })
  const createGRNMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/grn', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['grns'] }); invalidate(); setShowGRNForm(false); setGRNLines([]) } })
  const verifyGRNMut = useMutation({ mutationFn: ({ id, status }: { id: number; status: string }) => api.patch(`/purchase/grn/${id}/verify`, { status }), onSuccess: () => qc.invalidateQueries({ queryKey: ['grns'] }) })

  const TABS: [Tab, string][] = [
    ['dashboard', '📊 Dashboard'],
    ['suppliers', '🏢 Suppliers'],
    ['processors', '🏭 Processors'],
    ['pr', '📝 Requisitions'],
    ['po', '🛒 Purchase Orders'],
    ['jwo', '⚙️ Job Work Orders'],
    ['grn', '📦 GRN'],
  ]

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-gray-800">Purchase Module</h1>
        <p className="text-sm text-gray-500">Suppliers, PR, PO, JWO, GRN management</p>
      </div>

      <div className="flex flex-wrap gap-1 bg-gray-100 p-1 rounded-lg w-fit">
        {TABS.map(([key, label]) => (
          <button key={key} onClick={() => { setTab(key); setFilterStatus(''); setExpanded(null) }}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${tab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* Dashboard */}
      {tab === 'dashboard' && stats && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {[
              { label: 'Open PRs', value: stats.open_prs, color: 'text-yellow-600' },
              { label: 'Open POs', value: stats.open_pos, color: 'text-blue-600' },
              { label: 'Open JWOs', value: stats.open_jwos, color: 'text-purple-600' },
              { label: 'Pending GRNs', value: stats.pending_grns, color: 'text-orange-600' },
              { label: 'Suppliers', value: stats.total_suppliers, color: 'text-green-600' },
              { label: 'Processors', value: stats.total_processors, color: 'text-gray-700' },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
                <p className={`text-2xl font-bold ${color}`}>{value}</p>
                <p className="text-xs text-gray-500 mt-1 font-semibold">{label}</p>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-3">
            {([['pr', '+ New PR'], ['po', '+ New PO'], ['jwo', '+ JWO'], ['grn', '+ GRN']] as [Tab, string][]).map(([t, label]) => (
              <button key={t} onClick={() => setTab(t)}
                className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Suppliers */}
      {tab === 'suppliers' && (
        <div className="space-y-4">
          <div className="flex justify-between">
            <p className="text-sm text-gray-500">{suppliers.length} active suppliers</p>
            <button onClick={() => setShowSupForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ Add Supplier</button>
          </div>
          {showSupForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Supplier</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[['supplier_name','Name *'],['contact_person','Contact'],['email','Email'],['phone','Phone'],['gst_number','GST'],['address','Address'],['payment_terms','Payment Terms']].map(([k,l]) => (
                  <div key={k}>
                    <label className="text-xs text-gray-500">{l}</label>
                    <input value={(supForm as Record<string,string>)[k]} onChange={e => setSupForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                <div>
                  <label className="text-xs text-gray-500">Type</label>
                  <select value={supForm.supplier_type} onChange={e => setSupForm(f => ({ ...f, supplier_type: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {SUP_TYPES.map(t => <option key={t}>{t}</option>)}
                  </select>
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createSupMut.mutate(supForm)} disabled={createSupMut.isPending || !supForm.supplier_name}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createSupMut.isPending ? 'Saving…' : 'Save'}
                </button>
                <button onClick={() => setShowSupForm(false)} className="px-4 py-2 border border-gray-200 rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}
          <div className="bg-white rounded-xl border border-gray-100 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                <tr>{['Code','Name','Type','Contact','Phone','Payment Terms'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {suppliers.map(s => (
                  <tr key={s.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">{s.supplier_code}</td>
                    <td className="px-4 py-2 text-gray-700">{s.supplier_name}</td>
                    <td className="px-4 py-2 text-gray-500">{s.supplier_type}</td>
                    <td className="px-4 py-2 text-gray-500">{s.contact_person}</td>
                    <td className="px-4 py-2 text-gray-500">{s.phone}</td>
                    <td className="px-4 py-2 text-gray-500">{s.payment_terms}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {suppliers.length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No suppliers yet</p>}
          </div>
        </div>
      )}

      {/* Processors */}
      {tab === 'processors' && (
        <div className="space-y-4">
          <div className="flex justify-between">
            <p className="text-sm text-gray-500">{processors.length} processors</p>
            <button onClick={() => setShowProcForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ Add Processor</button>
          </div>
          {showProcForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Processor</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[['processor_name','Name *'],['contact_person','Contact'],['phone','Phone'],['address','Address']].map(([k,l]) => (
                  <div key={k}>
                    <label className="text-xs text-gray-500">{l}</label>
                    <input value={(procForm as Record<string,string>)[k]} onChange={e => setProcForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                <div>
                  <label className="text-xs text-gray-500">Type</label>
                  <select value={procForm.processor_type} onChange={e => setProcForm(f => ({ ...f, processor_type: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {PROC_TYPES.map(t => <option key={t}>{t}</option>)}
                  </select>
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createProcMut.mutate(procForm)} disabled={createProcMut.isPending || !procForm.processor_name}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">Save</button>
                <button onClick={() => setShowProcForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}
          <div className="bg-white rounded-xl border border-gray-100 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                <tr>{['Code','Name','Type','Contact','Phone'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {processors.map(p => (
                  <tr key={p.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">{p.processor_code}</td>
                    <td className="px-4 py-2 text-gray-700">{p.processor_name}</td>
                    <td className="px-4 py-2 text-gray-500">{p.processor_type}</td>
                    <td className="px-4 py-2 text-gray-500">{p.contact_person}</td>
                    <td className="px-4 py-2 text-gray-500">{p.phone}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {processors.length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No processors yet</p>}
          </div>
        </div>
      )}

      {/* Purchase Requisitions */}
      {tab === 'pr' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All</option>
              {['Draft','Pending Approval','Approved','Rejected','PO Created','Closed'].map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={() => setShowPRForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ New PR</button>
          </div>
          {showPRForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Purchase Requisition</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[['requested_by','Requested By'],['so_reference','SO Reference'],['notes','Notes']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(prForm as Record<string,string>)[k]} onChange={e => setPRForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                <div><label className="text-xs text-gray-500">Department</label>
                  <select value={prForm.department} onChange={e => setPRForm(f => ({ ...f, department: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {DEPTS.map(d => <option key={d}>{d}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">Priority</label>
                  <select value={prForm.priority} onChange={e => setPRForm(f => ({ ...f, priority: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {PRIORITIES.map(p => <option key={p}>{p}</option>)}
                  </select>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-2"><p className="text-sm font-medium text-gray-600">Material Lines</p>
                  <button onClick={() => setPRLines(l => [...l, { material_code: '', material_name: '', material_type: 'RM', required_qty: 0, unit: 'PCS', required_by_date: '' }])}
                    className="text-xs text-blue-600 hover:underline">+ Add Line</button>
                </div>
                {prLines.map((ln, i) => (
                  <div key={i} className="grid grid-cols-6 gap-2 mb-2">
                    <input placeholder="Material Code" value={ln.material_code} onChange={e => setPRLines(l => l.map((x, j) => j === i ? { ...x, material_code: e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm col-span-1" />
                    <input placeholder="Material Name" value={ln.material_name} onChange={e => setPRLines(l => l.map((x, j) => j === i ? { ...x, material_name: e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm col-span-2" />
                    <select value={ln.material_type} onChange={e => setPRLines(l => l.map((x, j) => j === i ? { ...x, material_type: e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm">
                      {MAT_TYPES.map(t => <option key={t}>{t}</option>)}
                    </select>
                    <input type="number" placeholder="Qty" value={ln.required_qty} onChange={e => setPRLines(l => l.map((x, j) => j === i ? { ...x, required_qty: +e.target.value } : x))}
                      className="border border-gray-200 rounded px-2 py-1.5 text-sm" />
                    <button onClick={() => setPRLines(l => l.filter((_, j) => j !== i))} className="text-red-400 text-sm">✕</button>
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <button onClick={() => createPRMut.mutate({ ...prForm, lines: prLines })} disabled={createPRMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createPRMut.isPending ? 'Saving…' : 'Create PR'}
                </button>
                <button onClick={() => setShowPRForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}
          <div className="space-y-2">
            {prs.map(pr => (
              <div key={pr.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                <div className="flex items-center justify-between p-4 cursor-pointer" onClick={() => setExpanded(expanded === pr.id ? null : pr.id)}>
                  <div>
                    <p className="font-semibold text-sm text-gray-800">{pr.pr_number}</p>
                    <p className="text-xs text-gray-500">{pr.requested_by} · {pr.department} · {pr.pr_date}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(pr.status)}`}>{pr.status}</span>
                    {pr.status === 'Draft' && (
                      <button onClick={e => { e.stopPropagation(); approvePRMut.mutate({ id: pr.id, approver: 'Admin' }) }}
                        className="text-xs px-2 py-0.5 bg-green-50 text-green-700 rounded hover:bg-green-100">Approve</button>
                    )}
                    <span className="text-gray-400 text-xs">{expanded === pr.id ? '▲' : '▼'}</span>
                  </div>
                </div>
                {expanded === pr.id && (
                  <div className="border-t px-4 pb-3">
                    <table className="w-full text-xs mt-3">
                      <thead><tr className="text-gray-400 uppercase"><th className="text-left py-1">Code</th><th className="text-left py-1">Name</th><th className="text-left py-1">Type</th><th className="text-right py-1">Qty</th><th className="text-right py-1">Unit</th></tr></thead>
                      <tbody>
                        {pr.lines.map(l => (
                          <tr key={l.id} className="border-t border-gray-50">
                            <td className="py-1.5 font-medium">{l.material_code}</td>
                            <td className="py-1.5">{l.material_name}</td>
                            <td className="py-1.5">{l.material_type}</td>
                            <td className="py-1.5 text-right">{l.required_qty}</td>
                            <td className="py-1.5 text-right">{l.unit}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ))}
            {prs.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No purchase requisitions found.</p>}
          </div>
        </div>
      )}

      {/* Purchase Orders */}
      {tab === 'po' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All</option>
              {['Draft','Sent to Supplier','Confirmed','Partial Received','Received','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={() => setShowPOForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ New PO</button>
          </div>
          {showPOForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Purchase Order</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[['supplier_name','Supplier Name'],['delivery_date','Delivery Date'],['payment_terms','Payment Terms'],['delivery_location','Delivery Location'],['pr_reference','PR Reference'],['so_reference','SO Reference'],['remarks','Remarks']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input type={k === 'delivery_date' ? 'date' : 'text'} value={(poForm as Record<string,unknown>)[k] as string}
                      onChange={e => setPOForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
              </div>
              <div>
                <div className="flex justify-between mb-2"><p className="text-sm font-medium text-gray-600">PO Lines</p>
                  <button onClick={() => setPOLines(l => [...l, { material_code: '', material_name: '', material_type: 'RM', po_qty: 0, unit: 'PCS', rate: 0, gst_pct: 0 }])}
                    className="text-xs text-blue-600 hover:underline">+ Add Line</button>
                </div>
                {poLines.map((ln, i) => (
                  <div key={i} className="grid grid-cols-7 gap-2 mb-2 text-sm">
                    <input placeholder="Code" value={ln.material_code} onChange={e => setPOLines(l => l.map((x, j) => j === i ? { ...x, material_code: e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input placeholder="Name" value={ln.material_name} onChange={e => setPOLines(l => l.map((x, j) => j === i ? { ...x, material_name: e.target.value } : x))} className="border rounded px-2 py-1.5 col-span-2" />
                    <input type="number" placeholder="Qty" value={ln.po_qty} onChange={e => setPOLines(l => l.map((x, j) => j === i ? { ...x, po_qty: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Rate ₹" value={ln.rate} onChange={e => setPOLines(l => l.map((x, j) => j === i ? { ...x, rate: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <span className="py-1.5 text-right text-gray-600 font-medium">{fmt(ln.po_qty * ln.rate)}</span>
                    <button onClick={() => setPOLines(l => l.filter((_, j) => j !== i))} className="text-red-400 text-sm">✕</button>
                  </div>
                ))}
                {poLines.length > 0 && <p className="text-sm font-semibold text-right text-gray-700 mt-2">Total: {fmt(poLines.reduce((s, l) => s + l.po_qty * l.rate, 0))}</p>}
              </div>
              <div className="flex gap-2">
                <button onClick={() => createPOMut.mutate({ ...poForm, lines: poLines.map(l => ({ ...l, amount: l.po_qty * l.rate })) })}
                  disabled={createPOMut.isPending} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createPOMut.isPending ? 'Saving…' : 'Create PO'}
                </button>
                <button onClick={() => setShowPOForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}
          <div className="space-y-2">
            {pos.map(po => (
              <div key={po.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                <div className="flex items-center justify-between p-4 cursor-pointer" onClick={() => setExpanded(expanded === po.id ? null : po.id)}>
                  <div>
                    <p className="font-semibold text-sm text-gray-800">{po.po_number}</p>
                    <p className="text-xs text-gray-500">{po.supplier_name} · Delivery: {po.delivery_date || '—'}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-700">{fmt(po.total)}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(po.status)}`}>{po.status}</span>
                    <select value={po.status} onClick={e => e.stopPropagation()} onChange={e => updatePOStatusMut.mutate({ id: po.id, status: e.target.value })}
                      className="border border-gray-200 rounded px-2 py-1 text-xs">
                      {['Draft','Sent to Supplier','Confirmed','Partial Received','Received','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
                    </select>
                    <span className="text-gray-400 text-xs">{expanded === po.id ? '▲' : '▼'}</span>
                  </div>
                </div>
                {expanded === po.id && (
                  <div className="border-t px-4 pb-3">
                    <table className="w-full text-xs mt-3">
                      <thead><tr className="text-gray-400 uppercase"><th className="text-left">Code</th><th className="text-left">Name</th><th className="text-right">Qty</th><th className="text-right">Rate</th><th className="text-right">Amount</th></tr></thead>
                      <tbody>{po.lines.map(l => (
                        <tr key={l.id} className="border-t border-gray-50">
                          <td className="py-1.5 font-medium">{l.material_code}</td>
                          <td className="py-1.5">{l.material_name}</td>
                          <td className="py-1.5 text-right">{l.po_qty}</td>
                          <td className="py-1.5 text-right">{fmt(l.rate)}</td>
                          <td className="py-1.5 text-right font-medium">{fmt(l.amount)}</td>
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                )}
              </div>
            ))}
            {pos.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No purchase orders found.</p>}
          </div>
        </div>
      )}

      {/* Job Work Orders */}
      {tab === 'jwo' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All</option>
              {['Draft','Issued to Processor','In Process','Partial Received','Received','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={() => setShowJWOForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ New JWO</button>
          </div>
          {showJWOForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Job Work Order</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div><label className="text-xs text-gray-500">Processor</label>
                  <select value={jwoForm.processor_id ?? ''} onChange={e => {
                    const p = processors.find(x => x.id === +e.target.value)
                    setJWOForm(f => ({ ...f, processor_id: +e.target.value, processor_name: p?.processor_name || '' }))
                  }} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select processor</option>
                    {processors.map(p => <option key={p.id} value={p.id}>{p.processor_name}</option>)}
                  </select>
                </div>
                {[['expected_return_date','Return Date','date'],['so_reference','SO Reference','text'],['issued_by','Issued By','text']].map(([k,l,t]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input type={t} value={(jwoForm as Record<string,unknown>)[k] as string}
                      onChange={e => setJWOForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
              </div>
              <div>
                <div className="flex justify-between mb-2"><p className="text-sm font-medium text-gray-600">Process Lines</p>
                  <button onClick={() => setJWOLines(l => [...l, { input_material: '', input_qty: 0, output_material: '', output_qty: 0, process_type: 'Printing', rate: 0 }])}
                    className="text-xs text-blue-600 hover:underline">+ Add Line</button>
                </div>
                {jwoLines.map((ln, i) => (
                  <div key={i} className="grid grid-cols-6 gap-2 mb-2 text-sm">
                    <input placeholder="Input Material" value={ln.input_material} onChange={e => setJWOLines(l => l.map((x, j) => j === i ? { ...x, input_material: e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Input Qty" value={ln.input_qty} onChange={e => setJWOLines(l => l.map((x, j) => j === i ? { ...x, input_qty: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input placeholder="Output Material" value={ln.output_material} onChange={e => setJWOLines(l => l.map((x, j) => j === i ? { ...x, output_material: e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Output Qty" value={ln.output_qty} onChange={e => setJWOLines(l => l.map((x, j) => j === i ? { ...x, output_qty: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <select value={ln.process_type} onChange={e => setJWOLines(l => l.map((x, j) => j === i ? { ...x, process_type: e.target.value } : x))} className="border rounded px-2 py-1.5">
                      {PROC_TYPES2.map(t => <option key={t}>{t}</option>)}
                    </select>
                    <button onClick={() => setJWOLines(l => l.filter((_, j) => j !== i))} className="text-red-400">✕</button>
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <button onClick={() => createJWOMut.mutate({ ...jwoForm, lines: jwoLines })} disabled={createJWOMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createJWOMut.isPending ? 'Saving…' : 'Create JWO'}
                </button>
                <button onClick={() => setShowJWOForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}
          <div className="space-y-2">
            {jwos.map(jwo => (
              <div key={jwo.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                <div className="flex items-center justify-between p-4 cursor-pointer" onClick={() => setExpanded(expanded === jwo.id ? null : jwo.id)}>
                  <div>
                    <p className="font-semibold text-sm text-gray-800">{jwo.jwo_number}</p>
                    <p className="text-xs text-gray-500">{jwo.processor_name} · Return: {jwo.expected_return_date || '—'}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-700">{fmt(jwo.total)}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(jwo.status)}`}>{jwo.status}</span>
                    <select value={jwo.status} onClick={e => e.stopPropagation()} onChange={e => updateJWOStatusMut.mutate({ id: jwo.id, status: e.target.value })}
                      className="border border-gray-200 rounded px-2 py-1 text-xs">
                      {['Draft','Issued to Processor','In Process','Partial Received','Received','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
                    </select>
                    <span className="text-gray-400 text-xs">{expanded === jwo.id ? '▲' : '▼'}</span>
                  </div>
                </div>
              </div>
            ))}
            {jwos.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No job work orders found.</p>}
          </div>
        </div>
      )}

      {/* GRN */}
      {tab === 'grn' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All</option>
              {['Draft','Verified','Posted'].map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={() => setShowGRNForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ New GRN</button>
          </div>
          {showGRNForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New GRN</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div><label className="text-xs text-gray-500">GRN Type</label>
                  <select value={grnForm.grn_type} onChange={e => setGRNForm(f => ({ ...f, grn_type: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {GRN_TYPES.map(t => <option key={t}>{t}</option>)}
                  </select>
                </div>
                {[['reference_number','PO/JWO #'],['party_name','Party Name'],['challan_no','Challan No'],['invoice_no','Invoice No'],['vehicle_no','Vehicle No'],['transporter','Transporter'],['warehouse','Warehouse']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(grnForm as Record<string,string>)[k]} onChange={e => setGRNForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
              </div>
              <div>
                <div className="flex justify-between mb-2"><p className="text-sm font-medium text-gray-600">Material Lines</p>
                  <button onClick={() => setGRNLines(l => [...l, { material_code: '', material_name: '', received_qty: 0, accepted_qty: 0, rejected_qty: 0, unit: 'PCS', rate: 0, qc_status: 'Pending' }])}
                    className="text-xs text-blue-600 hover:underline">+ Add Line</button>
                </div>
                {grnLines.map((ln, i) => (
                  <div key={i} className="grid grid-cols-7 gap-2 mb-2 text-sm">
                    <input placeholder="Code" value={ln.material_code} onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, material_code: e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input placeholder="Name" value={ln.material_name} onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, material_name: e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Received" value={ln.received_qty} onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, received_qty: +e.target.value, accepted_qty: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Accepted" value={ln.accepted_qty} onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, accepted_qty: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Rate ₹" value={ln.rate} onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, rate: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <select value={ln.qc_status} onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, qc_status: e.target.value } : x))} className="border rounded px-2 py-1.5">
                      {['Pending','Pass','Fail'].map(s => <option key={s}>{s}</option>)}
                    </select>
                    <button onClick={() => setGRNLines(l => l.filter((_, j) => j !== i))} className="text-red-400">✕</button>
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <button onClick={() => createGRNMut.mutate({ ...grnForm, lines: grnLines.map(l => ({ ...l, amount: l.accepted_qty * l.rate })) })}
                  disabled={createGRNMut.isPending} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createGRNMut.isPending ? 'Saving…' : 'Create GRN'}
                </button>
                <button onClick={() => setShowGRNForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}
          <div className="space-y-2">
            {grns.map(grn => (
              <div key={grn.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                <div className="flex items-center justify-between p-4 cursor-pointer" onClick={() => setExpanded(expanded === grn.id ? null : grn.id)}>
                  <div>
                    <p className="font-semibold text-sm text-gray-800">{grn.grn_number} <span className="text-xs font-normal text-gray-400">({grn.grn_type})</span></p>
                    <p className="text-xs text-gray-500">{grn.party_name} · {grn.grn_date}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-700">{fmt(grn.total_value)}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(grn.status)}`}>{grn.status}</span>
                    {grn.status === 'Draft' && (
                      <button onClick={e => { e.stopPropagation(); verifyGRNMut.mutate({ id: grn.id, status: 'Verified' }) }}
                        className="text-xs px-2 py-0.5 bg-green-50 text-green-700 rounded hover:bg-green-100">Verify</button>
                    )}
                    <span className="text-gray-400 text-xs">{expanded === grn.id ? '▲' : '▼'}</span>
                  </div>
                </div>
                {expanded === grn.id && (
                  <div className="border-t px-4 pb-3">
                    <table className="w-full text-xs mt-3">
                      <thead><tr className="text-gray-400 uppercase"><th className="text-left">Code</th><th className="text-left">Name</th><th className="text-right">Received</th><th className="text-right">Accepted</th><th className="text-right">Rejected</th><th className="text-right">QC</th></tr></thead>
                      <tbody>{grn.lines.map(l => (
                        <tr key={l.id} className="border-t border-gray-50">
                          <td className="py-1.5 font-medium">{l.material_code}</td>
                          <td className="py-1.5"></td>
                          <td className="py-1.5 text-right">{l.received_qty}</td>
                          <td className="py-1.5 text-right text-green-600">{l.accepted_qty}</td>
                          <td className="py-1.5 text-right text-red-500">{l.rejected_qty}</td>
                          <td className="py-1.5 text-right"><span className={`px-1.5 py-0.5 rounded text-xs ${l.qc_status === 'Pass' ? 'bg-green-100 text-green-700' : l.qc_status === 'Fail' ? 'bg-red-100 text-red-600' : 'bg-yellow-100 text-yellow-600'}`}>{l.qc_status}</span></td>
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                )}
              </div>
            ))}
            {grns.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No GRNs found.</p>}
          </div>
        </div>
      )}
    </div>
  )
}
