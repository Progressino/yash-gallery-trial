import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type Tab = 'dashboard' | 'suppliers' | 'processors' | 'pr' | 'po' | 'jwo' | 'grn'
type PRSubTab = 'list' | 'new' | 'from-mrp'

interface Stats { open_prs: number; open_pos: number; open_jwos: number; pending_grns: number; total_suppliers: number; total_processors: number }
interface Supplier { id: number; supplier_code: string; supplier_name: string; supplier_type: string; contact_person: string; email: string; phone: string; payment_terms: string; active: number }
interface Processor { id: number; processor_code: string; processor_name: string; processor_type: string; contact_person: string; phone: string }
interface PRLine { id: number; material_code: string; material_name: string; material_type: string; required_qty: number; po_qty: number; unit: string; required_by_date: string }
interface PR { id: number; pr_number: string; pr_date: string; requested_by: string; department: string; priority: string; status: string; so_reference: string; pr_type: string; source: string; required_by_date: string; lines: PRLine[] }
interface PO { id: number; po_number: string; po_date: string; supplier_name: string; status: string; total: number; delivery_date: string; pr_reference: string; so_reference: string; lines: POLine[] }
interface POLine { id: number; material_code: string; material_name: string; po_qty: number; unit: string; rate: number; gst_pct: number; amount: number }
interface JWO { id: number; jwo_number: string; jwo_date: string; processor_name: string; status: string; total: number; expected_return_date: string; lines: JWOLine[] }
interface JWOLine { id: number; input_material: string; output_material: string; output_qty: number; process_type: string; rate: number; amount: number }
interface GRN { id: number; grn_number: string; grn_date: string; grn_type: string; party_name: string; status: string; total_value: number; lines: GRNLine[] }
interface GRNLine { id: number; material_code: string; received_qty: number; accepted_qty: number; rejected_qty: number; qc_status: string; rate: number; amount: number }
interface MRPLineItem { material_code: string; material_name: string; material_type: string; required_qty: number; net_req: number; unit: string; inputs?: { material_code: string; material_name: string; quantity: number; unit: string }[] }
interface MRPLinesResult { so_number: string; purchase_items: MRPLineItem[]; sfg_items: MRPLineItem[]; error?: string; warning?: string }
interface POFromPRLine { supplier_id?: number; supplier_name: string; qty: number; rate: number; gst_pct: number }

const SUP_TYPES = ['Fabric Supplier', 'Accessories Supplier', 'Job Work', 'Others']
const PROC_TYPES = ['Printing Unit', 'Dyeing Unit', 'Embroidery', 'Others']
const DEPTS = ['Production', 'Stores', 'Others']
const PRIORITIES = ['Low', 'Normal', 'High', 'Urgent']
const MAT_TYPES = ['RM', 'ACC', 'PKG', 'SFG', 'FG', 'GF']
const PROC_TYPES2 = ['Printing', 'Dyeing', 'Embroidery', 'Other']
const GRN_TYPES = ['PO Receipt', 'JWO Receipt']
const PAYMENT_TERMS = ['Immediate', 'Net 15', 'Net 30', 'Net 45', 'Net 60']
const GST_RATES = [0, 5, 12, 18, 28]

const statusColor = (s: string) => {
  if (['Approved', 'Received', 'Verified', 'Posted', 'Closed'].includes(s)) return 'bg-green-100 text-green-700'
  if (['Draft', 'Pending Approval'].includes(s)) return 'bg-yellow-100 text-yellow-700'
  if (['PO Created', 'Sent to Supplier', 'In Process', 'Confirmed', 'Issued to Processor'].includes(s)) return 'bg-blue-100 text-blue-700'
  if (['Rejected', 'Cancelled'].includes(s)) return 'bg-red-100 text-red-700'
  return 'bg-gray-100 text-gray-600'
}

const fmt = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN')

export default function Purchase() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('dashboard')
  const [expanded, setExpanded] = useState<number | null>(null)
  const [filterStatus, setFilterStatus] = useState('')
  const [prSubTab, setPRSubTab] = useState<PRSubTab>('list')

  // Supplier form
  const [showSupForm, setShowSupForm] = useState(false)
  const [supForm, setSupForm] = useState({ supplier_name: '', supplier_type: 'Others', contact_person: '', email: '', phone: '', address: '', gst_number: '', payment_terms: 'Net 30' })

  // Processor form
  const [showProcForm, setShowProcForm] = useState(false)
  const [procForm, setProcForm] = useState({ processor_name: '', processor_type: 'Others', contact_person: '', email: '', phone: '', address: '' })

  // PR manual form
  const [prForm, setPRForm] = useState({ requested_by: '', department: 'Production', priority: 'Normal', so_reference: '', required_by_date: '', notes: '' })
  const [prLines, setPRLines] = useState<{ material_code: string; material_name: string; material_type: string; required_qty: number; unit: string }[]>([])

  // From MRP
  const [mrpSO, setMrpSO] = useState('')
  const [mrpLinesData, setMrpLinesData] = useState<MRPLinesResult | null>(null)
  const [mrpLoadingLines, setMrpLoadingLines] = useState(false)
  const [sfgActions, setSfgActions] = useState<Record<string, 'job_work' | 'direct_purchase'>>({})
  const [selectedPurchaseItems, setSelectedPurchaseItems] = useState<Set<string>>(new Set())
  const [mrpReqDate, setMrpReqDate] = useState('')
  const [generatingPRs, setGeneratingPRs] = useState(false)
  const [generatedPRs, setGeneratedPRs] = useState<string[]>([])

  // Create PO from PR
  const [showCreatePO, setShowCreatePO] = useState<number | null>(null)
  const [poFromPRLines, setPoFromPRLines] = useState<Record<string, POFromPRLine>>({})
  const [poFromPRMeta, setPoFromPRMeta] = useState({ delivery_date: '', payment_terms: 'Immediate' })
  const [createdPOs, setCreatedPOs] = useState<string[]>([])

  // PO / JWO / GRN forms
  const [showPOForm, setShowPOForm] = useState(false)
  const [poForm, setPOForm] = useState({ supplier_id: undefined as number | undefined, supplier_name: '', delivery_date: '', payment_terms: '', delivery_location: '', pr_reference: '', so_reference: '', remarks: '' })
  const [poLines, setPOLines] = useState<{ material_code: string; material_name: string; material_type: string; po_qty: number; unit: string; rate: number; gst_pct: number }[]>([])
  const [showJWOForm, setShowJWOForm] = useState(false)
  const [jwoForm, setJWOForm] = useState({ processor_id: undefined as number | undefined, processor_name: '', expected_return_date: '', pr_reference: '', so_reference: '', issued_by: '', remarks: '' })
  const [jwoLines, setJWOLines] = useState<{ input_material: string; input_qty: number; output_material: string; output_qty: number; process_type: string; rate: number }[]>([])
  const [showGRNForm, setShowGRNForm] = useState(false)
  const [grnForm, setGRNForm] = useState({ grn_type: 'PO Receipt', reference_number: '', party_name: '', challan_no: '', invoice_no: '', vehicle_no: '', transporter: '', warehouse: '', remarks: '' })
  const [grnLines, setGRNLines] = useState<{ material_code: string; material_name: string; received_qty: number; accepted_qty: number; rejected_qty: number; unit: string; rate: number; qc_status: string }[]>([])

  const { data: stats } = useQuery<Stats>({ queryKey: ['purchase-stats'], queryFn: () => api.get('/purchase/stats').then(r => r.data) })
  const { data: suppliers = [] } = useQuery<Supplier[]>({ queryKey: ['suppliers'], queryFn: () => api.get('/purchase/suppliers').then(r => r.data), enabled: tab === 'suppliers' || tab === 'po' || tab === 'dashboard' || tab === 'pr' })
  const { data: processors = [] } = useQuery<Processor[]>({ queryKey: ['processors'], queryFn: () => api.get('/purchase/processors').then(r => r.data), enabled: tab === 'processors' || tab === 'jwo' })
  const { data: prs = [] } = useQuery<PR[]>({ queryKey: ['prs', filterStatus], queryFn: () => api.get('/purchase/pr' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data), enabled: tab === 'pr' })
  const { data: pos = [] } = useQuery<PO[]>({ queryKey: ['pos', filterStatus], queryFn: () => api.get('/purchase/po' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data), enabled: tab === 'po' })
  const { data: jwos = [] } = useQuery<JWO[]>({ queryKey: ['jwos', filterStatus], queryFn: () => api.get('/purchase/jwo' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data), enabled: tab === 'jwo' })
  const { data: grns = [] } = useQuery<GRN[]>({ queryKey: ['grns', filterStatus], queryFn: () => api.get('/purchase/grn' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data), enabled: tab === 'grn' })

  const invalidate = () => qc.invalidateQueries({ queryKey: ['purchase-stats'] })
  const createSupMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/suppliers', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['suppliers'] }); invalidate(); setShowSupForm(false) } })
  const createProcMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/processors', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['processors'] }); setShowProcForm(false) } })
  const createPRMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/pr', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['prs'] }); invalidate(); setPRSubTab('list'); setPRLines([]) } })
  const approvePRMut = useMutation({ mutationFn: ({ id, approver }: { id: number; approver: string }) => api.post(`/purchase/pr/${id}/approve`, { approver }), onSuccess: () => qc.invalidateQueries({ queryKey: ['prs'] }) })
  const rejectPRMut = useMutation({ mutationFn: (id: number) => api.post(`/purchase/pr/${id}/reject`, { remarks: '' }), onSuccess: () => qc.invalidateQueries({ queryKey: ['prs'] }) })
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

  // ── From MRP helpers ──────────────────────────────────────────────────────
  const loadMRPLines = async () => {
    if (!mrpSO.trim()) return
    setMrpLoadingLines(true)
    setMrpLinesData(null)
    setGeneratedPRs([])
    try {
      const res = await api.get(`/production/mrp/lines-for-so?so_number=${encodeURIComponent(mrpSO.trim())}`)
      const d: MRPLinesResult = res.data
      setMrpLinesData(d)
      // Default: all purchase items selected, all SFGs → job_work
      const sel = new Set(d.purchase_items.map(i => i.material_code))
      setSelectedPurchaseItems(sel)
      const actions: Record<string, 'job_work' | 'direct_purchase'> = {}
      d.sfg_items.forEach(i => { actions[i.material_code] = 'job_work' })
      setSfgActions(actions)
    } catch {
      setMrpLinesData({ so_number: mrpSO, purchase_items: [], sfg_items: [], error: 'Failed to load MRP lines.' })
    }
    setMrpLoadingLines(false)
  }

  const generatePRsFromMRP = async () => {
    if (!mrpLinesData) return
    setGeneratingPRs(true)
    setGeneratedPRs([])
    const created: string[] = []
    try {
      // Purchase lines: selected RM/ACC/etc + SFGs marked as direct_purchase
      const purchaseLines = [
        ...mrpLinesData.purchase_items
          .filter(i => selectedPurchaseItems.has(i.material_code))
          .map(i => ({ material_code: i.material_code, material_name: i.material_name, material_type: i.material_type, required_qty: i.net_req || i.required_qty, unit: i.unit })),
        ...mrpLinesData.sfg_items
          .filter(i => sfgActions[i.material_code] === 'direct_purchase')
          .map(i => ({ material_code: i.material_code, material_name: i.material_name, material_type: i.material_type, required_qty: i.net_req || i.required_qty, unit: i.unit })),
      ]
      const jwLines = mrpLinesData.sfg_items
        .filter(i => sfgActions[i.material_code] === 'job_work' || !sfgActions[i.material_code])
        .map(i => ({ material_code: i.material_code, material_name: i.material_name, material_type: i.material_type, required_qty: i.net_req || i.required_qty, unit: i.unit }))

      if (purchaseLines.length > 0) {
        const r = await api.post('/purchase/pr', { pr_type: 'Purchase', source: 'MRP', so_reference: mrpSO, required_by_date: mrpReqDate, department: 'Production', priority: 'Normal', lines: purchaseLines })
        created.push(r.data.pr_number)
      }
      if (jwLines.length > 0) {
        const r = await api.post('/purchase/pr', { pr_type: 'Job Work', source: 'MRP', so_reference: mrpSO, required_by_date: mrpReqDate, department: 'Production', priority: 'Normal', lines: jwLines })
        created.push(r.data.pr_number)
      }
      setGeneratedPRs(created)
      qc.invalidateQueries({ queryKey: ['prs'] })
      invalidate()
    } catch { /* ignore */ }
    setGeneratingPRs(false)
  }

  // ── Create PO from PR helpers ─────────────────────────────────────────────
  const openCreatePO = (pr: PR) => {
    const init: Record<string, POFromPRLine> = {}
    pr.lines.forEach(l => {
      const pending = l.required_qty - (l.po_qty || 0)
      if (pending > 0) init[`${pr.id}-${l.material_code}`] = { supplier_name: '', qty: pending, rate: 0, gst_pct: 12 }
    })
    setPoFromPRLines(init)
    setPoFromPRMeta({ delivery_date: '', payment_terms: 'Immediate' })
    setCreatedPOs([])
    setShowCreatePO(pr.id)
  }

  const updatePOLine = (prId: number, matCode: string, field: keyof POFromPRLine, value: string | number) => {
    const key = `${prId}-${matCode}`
    setPoFromPRLines(prev => ({ ...prev, [key]: { ...prev[key], [field]: value } }))
  }

  const submitPOFromPR = async (pr: PR) => {
    const lines = pr.lines
      .filter(l => (l.required_qty - (l.po_qty || 0)) > 0)
      .map(l => {
        const key = `${pr.id}-${l.material_code}`
        const ld = poFromPRLines[key] || { supplier_name: '', qty: 0, rate: 0, gst_pct: 12 }
        const sup = suppliers.find(s => s.id === ld.supplier_id)
        return {
          material_code: l.material_code,
          material_name: l.material_name,
          material_type: l.material_type,
          unit: l.unit,
          qty: ld.qty,
          rate: ld.rate,
          gst_pct: ld.gst_pct,
          supplier_id: ld.supplier_id,
          supplier_name: ld.supplier_name || sup?.supplier_name || '',
        }
      })
      .filter(l => l.supplier_name || l.supplier_id)

    if (lines.length === 0) { alert('Please assign at least one supplier.'); return }
    try {
      const r = await api.post('/purchase/po/from-pr', { pr_id: pr.id, ...poFromPRMeta, lines })
      setCreatedPOs(r.data.po_numbers || [])
      qc.invalidateQueries({ queryKey: ['prs'] })
      qc.invalidateQueries({ queryKey: ['pos'] })
      invalidate()
    } catch { alert('Failed to create POs.') }
  }

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

      {/* ── Purchase Requisitions ─────────────────────────────────────────── */}
      {tab === 'pr' && (
        <div className="space-y-4">
          {/* Sub-tabs */}
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex gap-1 bg-gray-100 p-1 rounded-lg">
              {([['list', 'PR List'], ['new', '+ Create PR'], ['from-mrp', 'From MRP']] as [PRSubTab, string][]).map(([st, lbl]) => (
                <button key={st} onClick={() => { setPRSubTab(st); setGeneratedPRs([]) }}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${prSubTab === st ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
                  {lbl}
                </button>
              ))}
            </div>
            {prSubTab === 'list' && (
              <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
                <option value="">All Statuses</option>
                {['Pending Approval','Approved','Rejected','PO Created','Closed'].map(s => <option key={s}>{s}</option>)}
              </select>
            )}
          </div>

          {/* ── PR List ── */}
          {prSubTab === 'list' && (
            <div className="space-y-2">
              {prs.map(pr => {
                const totalPending = pr.lines.reduce((s, l) => s + Math.max(0, l.required_qty - (l.po_qty || 0)), 0)
                const isExpanded = expanded === pr.id
                return (
                  <div key={pr.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                    {/* Header */}
                    <div className="flex items-start justify-between p-4 cursor-pointer" onClick={() => { setExpanded(isExpanded ? null : pr.id); if (isExpanded) setShowCreatePO(null) }}>
                      <div>
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className="font-semibold text-sm text-gray-800">{pr.pr_number}</p>
                          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${pr.pr_type === 'Job Work' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'}`}>
                            {pr.pr_type === 'Job Work' ? '✏️ Job Work' : '🛒 Purchase'}
                          </span>
                          {pr.source === 'MRP' && <span className="text-xs px-2 py-0.5 rounded-full bg-orange-100 text-orange-700 font-medium">From MRP</span>}
                        </div>
                        <p className="text-xs text-gray-500 mt-0.5">
                          SO: {pr.so_reference || '—'} · {pr.lines.length} items · Pending: {Math.round(totalPending)}
                        </p>
                      </div>
                      <div className="flex items-center gap-2 ml-2">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(pr.status)}`}>{pr.status}</span>
                        {['Pending Approval', 'Draft'].includes(pr.status) && (
                          <>
                            <button onClick={e => { e.stopPropagation(); approvePRMut.mutate({ id: pr.id, approver: 'Admin' }) }}
                              className="text-xs px-2 py-1 bg-green-600 text-white rounded font-medium hover:bg-green-700 flex items-center gap-1">
                              ✓ Approve
                            </button>
                            <button onClick={e => { e.stopPropagation(); rejectPRMut.mutate(pr.id) }}
                              className="text-xs px-2 py-1 bg-red-500 text-white rounded font-medium hover:bg-red-600 flex items-center gap-1">
                              ✗ Reject
                            </button>
                          </>
                        )}
                        <span className="text-gray-400 text-xs">{isExpanded ? '▲' : '▼'}</span>
                      </div>
                    </div>

                    {/* Expanded */}
                    {isExpanded && (
                      <div className="border-t px-4 pb-4 space-y-4">
                        {/* Info grid */}
                        <div className="grid grid-cols-3 gap-4 mt-3">
                          <div>
                            <p className="text-xs text-gray-400 uppercase font-semibold mb-1">PR Info</p>
                            <p className="text-xs text-gray-700">Date: {pr.pr_date}</p>
                            <p className="text-xs text-gray-700">Req By: {pr.required_by_date || pr.lines[0]?.required_by_date || '—'}</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-400 uppercase font-semibold mb-1">Source</p>
                            <p className="text-xs text-gray-700">SO: {pr.so_reference || '—'}</p>
                            <p className="text-xs text-gray-700">From: {pr.source || 'Manual'}</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-400 uppercase font-semibold mb-1">Status</p>
                            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(pr.status)}`}>{pr.status}</span>
                          </div>
                        </div>

                        {/* Lines table */}
                        <div>
                          <p className="text-sm font-semibold text-gray-600 mb-2">PR Lines — Pending Qty:</p>
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="text-gray-400 uppercase text-left border-b">
                                <th className="pb-1">Material</th>
                                <th className="pb-1">Type</th>
                                <th className="pb-1 text-right">Required Qty</th>
                                <th className="pb-1 text-right">PO Created</th>
                                <th className="pb-1 text-right">Pending</th>
                                <th className="pb-1 text-right">Unit</th>
                              </tr>
                            </thead>
                            <tbody>
                              {pr.lines.map(l => {
                                const pending = Math.max(0, l.required_qty - (l.po_qty || 0))
                                return (
                                  <tr key={l.id} className="border-t border-gray-50">
                                    <td className="py-1.5 font-medium text-gray-800">{l.material_name || l.material_code}</td>
                                    <td className="py-1.5 text-gray-500">{l.material_type}</td>
                                    <td className="py-1.5 text-right">{l.required_qty}</td>
                                    <td className="py-1.5 text-right text-blue-600">{l.po_qty || 0}</td>
                                    <td className={`py-1.5 text-right font-semibold ${pending > 0 ? 'text-red-500' : 'text-green-600'}`}>{pending}</td>
                                    <td className="py-1.5 text-right text-gray-500">{l.unit}</td>
                                  </tr>
                                )
                              })}
                            </tbody>
                          </table>
                        </div>

                        {/* Create PO from PR (approved Purchase PRs only) */}
                        {pr.status === 'Approved' && pr.pr_type !== 'Job Work' && totalPending > 0 && (
                          <div className="border-t pt-4">
                            {showCreatePO !== pr.id ? (
                              <button onClick={() => openCreatePO(pr)}
                                className="flex items-center gap-2 px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
                                🛒 Create Purchase Order from this PR
                              </button>
                            ) : (
                              <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                  <h3 className="font-semibold text-gray-800">🛒 Create Purchase Order from this PR</h3>
                                  <button onClick={() => setShowCreatePO(null)} className="text-gray-400 hover:text-gray-600 text-sm">✕ Close</button>
                                </div>
                                <p className="text-xs text-blue-700 bg-blue-50 px-3 py-2 rounded-lg">
                                  Har item ke liye alag supplier select karo — same supplier wali lines ek hi PO mein merge ho jaayengi automatically.
                                </p>

                                <div className="grid grid-cols-2 gap-3">
                                  <div>
                                    <label className="text-xs text-gray-500">Default Delivery Date</label>
                                    <input type="date" value={poFromPRMeta.delivery_date} onChange={e => setPoFromPRMeta(m => ({ ...m, delivery_date: e.target.value }))}
                                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                                  </div>
                                  <div>
                                    <label className="text-xs text-gray-500">Payment Terms</label>
                                    <select value={poFromPRMeta.payment_terms} onChange={e => setPoFromPRMeta(m => ({ ...m, payment_terms: e.target.value }))}
                                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                                      {PAYMENT_TERMS.map(t => <option key={t}>{t}</option>)}
                                    </select>
                                  </div>
                                </div>

                                <p className="text-xs text-gray-500 font-medium">Har item ke liye supplier + qty + rate bharein:</p>
                                {pr.lines.filter(l => (l.required_qty - (l.po_qty || 0)) > 0).map(l => {
                                  const key = `${pr.id}-${l.material_code}`
                                  const ld = poFromPRLines[key] || { supplier_name: '', qty: l.required_qty - (l.po_qty || 0), rate: 0, gst_pct: 12 }
                                  return (
                                    <div key={l.id} className="border border-gray-200 rounded-lg p-3 space-y-2">
                                      <div>
                                        <p className="font-medium text-sm text-gray-800">{l.material_name || l.material_code}</p>
                                        <p className="text-xs text-gray-400">{l.material_type} | Pending: {l.required_qty - (l.po_qty || 0)} {l.unit}</p>
                                      </div>
                                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                        <div>
                                          <label className="text-xs text-gray-400">Supplier</label>
                                          <select value={ld.supplier_id || ''}
                                            onChange={e => {
                                              const s = suppliers.find(x => x.id === +e.target.value)
                                              updatePOLine(pr.id, l.material_code, 'supplier_id', +e.target.value)
                                              updatePOLine(pr.id, l.material_code, 'supplier_name', s?.supplier_name || '')
                                            }}
                                            className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-0.5">
                                            <option value="">Select supplier</option>
                                            {suppliers.map(s => <option key={s.id} value={s.id}>{s.supplier_name}</option>)}
                                          </select>
                                        </div>
                                        <div>
                                          <label className="text-xs text-gray-400">Qty</label>
                                          <div className="flex items-center gap-1 mt-0.5">
                                            <button onClick={() => updatePOLine(pr.id, l.material_code, 'qty', Math.max(0, ld.qty - 1))} className="w-7 h-7 border rounded text-sm hover:bg-gray-100">−</button>
                                            <input type="number" value={ld.qty} onChange={e => updatePOLine(pr.id, l.material_code, 'qty', +e.target.value)}
                                              className="flex-1 border border-gray-200 rounded px-2 py-1 text-sm text-center min-w-0" />
                                            <button onClick={() => updatePOLine(pr.id, l.material_code, 'qty', ld.qty + 1)} className="w-7 h-7 border rounded text-sm hover:bg-gray-100">+</button>
                                          </div>
                                        </div>
                                        <div>
                                          <label className="text-xs text-gray-400">Rate (₹)</label>
                                          <div className="flex items-center gap-1 mt-0.5">
                                            <button onClick={() => updatePOLine(pr.id, l.material_code, 'rate', Math.max(0, ld.rate - 1))} className="w-7 h-7 border rounded text-sm hover:bg-gray-100">−</button>
                                            <input type="number" value={ld.rate} onChange={e => updatePOLine(pr.id, l.material_code, 'rate', +e.target.value)}
                                              className="flex-1 border border-gray-200 rounded px-2 py-1 text-sm text-center min-w-0" />
                                            <button onClick={() => updatePOLine(pr.id, l.material_code, 'rate', ld.rate + 1)} className="w-7 h-7 border rounded text-sm hover:bg-gray-100">+</button>
                                          </div>
                                        </div>
                                        <div>
                                          <label className="text-xs text-gray-400">GST%</label>
                                          <select value={ld.gst_pct} onChange={e => updatePOLine(pr.id, l.material_code, 'gst_pct', +e.target.value)}
                                            className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-0.5">
                                            {GST_RATES.map(g => <option key={g} value={g}>{g}%</option>)}
                                          </select>
                                        </div>
                                      </div>
                                    </div>
                                  )
                                })}

                                {createdPOs.length > 0 ? (
                                  <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                                    <p className="text-sm font-semibold text-green-700">POs Created Successfully!</p>
                                    <p className="text-xs text-green-600 mt-1">{createdPOs.join(', ')}</p>
                                  </div>
                                ) : (
                                  <button onClick={() => submitPOFromPR(pr)}
                                    className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
                                    Create PO from PR
                                  </button>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
              {prs.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No purchase requisitions found.</p>}
            </div>
          )}

          {/* ── Create PR manually ── */}
          {prSubTab === 'new' && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Purchase Requisition</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[['requested_by','Requested By'],['so_reference','SO Reference'],['notes','Notes']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(prForm as Record<string,string>)[k]} onChange={e => setPRForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                <div><label className="text-xs text-gray-500">Required By Date</label>
                  <input type="date" value={prForm.required_by_date} onChange={e => setPRForm(f => ({ ...f, required_by_date: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">Department</label>
                  <select value={prForm.department} onChange={e => setPRForm(f => ({ ...f, department: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {DEPTS.map(d => <option key={d}>{d}</option>)}
                  </select>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-2"><p className="text-sm font-medium text-gray-600">Material Lines</p>
                  <button onClick={() => setPRLines(l => [...l, { material_code: '', material_name: '', material_type: 'RM', required_qty: 0, unit: 'PCS' }])}
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
                <button onClick={() => createPRMut.mutate({ ...prForm, source: 'Manual', pr_type: 'Purchase', lines: prLines })} disabled={createPRMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createPRMut.isPending ? 'Saving…' : 'Create PR'}
                </button>
                <button onClick={() => setPRSubTab('list')} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}

          {/* ── From MRP ── */}
          {prSubTab === 'from-mrp' && (
            <div className="space-y-4">
              <div className="bg-white rounded-xl border p-4 space-y-4">
                <h3 className="font-semibold text-gray-700">Generate PR from MRP</h3>
                <div className="grid grid-cols-3 gap-3 items-end">
                  <div>
                    <label className="text-xs text-gray-500">SO Reference</label>
                    <input value={mrpSO} onChange={e => setMrpSO(e.target.value)} placeholder="SO-0001"
                      onKeyDown={e => e.key === 'Enter' && loadMRPLines()}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Required By Date</label>
                    <input type="date" value={mrpReqDate} onChange={e => setMrpReqDate(e.target.value)}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                  <button onClick={loadMRPLines} disabled={!mrpSO.trim() || mrpLoadingLines}
                    className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800 disabled:opacity-50">
                    {mrpLoadingLines ? 'Loading…' : 'Load MRP'}
                  </button>
                </div>

                {mrpLinesData?.error && (
                  <p className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">{mrpLinesData.error}</p>
                )}
                {mrpLinesData?.warning && (
                  <p className="text-sm text-yellow-700 bg-yellow-50 px-3 py-2 rounded-lg">{mrpLinesData.warning}</p>
                )}

                {mrpLinesData && !mrpLinesData.error && (
                  <div className="space-y-5">
                    {/* Purchase Items */}
                    {mrpLinesData.purchase_items.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-gray-700 mb-2 flex items-center gap-2">
                          🛒 Purchase PR — Raw Materials &amp; Accessories
                          <span className="text-xs text-gray-400 font-normal">{mrpLinesData.purchase_items.filter(i => selectedPurchaseItems.has(i.material_code)).length} selected</span>
                        </h4>
                        <div className="bg-white border border-gray-100 rounded-lg overflow-hidden">
                          <table className="w-full text-sm">
                            <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                              <tr>
                                <th className="px-3 py-2 text-left w-8">
                                  <input type="checkbox"
                                    checked={mrpLinesData.purchase_items.every(i => selectedPurchaseItems.has(i.material_code))}
                                    onChange={e => setSelectedPurchaseItems(e.target.checked ? new Set(mrpLinesData.purchase_items.map(i => i.material_code)) : new Set())} />
                                </th>
                                <th className="px-3 py-2 text-left">Code</th>
                                <th className="px-3 py-2 text-left">Name</th>
                                <th className="px-3 py-2 text-left">Type</th>
                                <th className="px-3 py-2 text-right">Net Required</th>
                                <th className="px-3 py-2 text-right">Unit</th>
                              </tr>
                            </thead>
                            <tbody>
                              {mrpLinesData.purchase_items.map(item => (
                                <tr key={item.material_code} className="border-t border-gray-50 hover:bg-gray-50">
                                  <td className="px-3 py-2">
                                    <input type="checkbox" checked={selectedPurchaseItems.has(item.material_code)}
                                      onChange={e => setSelectedPurchaseItems(prev => {
                                        const n = new Set(prev)
                                        e.target.checked ? n.add(item.material_code) : n.delete(item.material_code)
                                        return n
                                      })} />
                                  </td>
                                  <td className="px-3 py-2 font-medium text-gray-800">{item.material_code}</td>
                                  <td className="px-3 py-2 text-gray-600">{item.material_name}</td>
                                  <td className="px-3 py-2"><span className="text-xs bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded">{item.material_type}</span></td>
                                  <td className="px-3 py-2 text-right font-semibold text-gray-800">{item.net_req || item.required_qty}</td>
                                  <td className="px-3 py-2 text-right text-gray-500">{item.unit}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* SFG Items */}
                    {mrpLinesData.sfg_items.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-gray-700 mb-2">✏️ SFG Materials — Choose Purchase or Job Work</h4>
                        <div className="space-y-2">
                          {mrpLinesData.sfg_items.map(item => (
                            <div key={item.material_code} className="border border-gray-200 rounded-lg p-3 space-y-2">
                              <div>
                                <p className="font-medium text-sm text-gray-800">{item.material_code} — {item.material_name}</p>
                                <p className="text-xs text-gray-500">Required: {item.net_req || item.required_qty} {item.unit}</p>
                                {(item.inputs || []).length > 0 && (
                                  <p className="text-xs text-blue-600 mt-1">
                                    Input: {item.inputs!.map(inp => `${inp.material_code} (${inp.quantity} ${inp.unit})`).join(', ')}
                                  </p>
                                )}
                              </div>
                              <div className="flex gap-4 text-sm">
                                <label className="flex items-center gap-1.5 cursor-pointer">
                                  <input type="radio" name={`sfg-${item.material_code}`}
                                    checked={!sfgActions[item.material_code] || sfgActions[item.material_code] === 'job_work'}
                                    onChange={() => setSfgActions(a => ({ ...a, [item.material_code]: 'job_work' }))} />
                                  <span className="text-gray-700">Job Work</span>
                                </label>
                                <label className="flex items-center gap-1.5 cursor-pointer">
                                  <input type="radio" name={`sfg-${item.material_code}`}
                                    checked={sfgActions[item.material_code] === 'direct_purchase'}
                                    onChange={() => setSfgActions(a => ({ ...a, [item.material_code]: 'direct_purchase' }))} />
                                  <span className="text-gray-700">Direct Purchase</span>
                                </label>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {generatedPRs.length > 0 ? (
                      <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                        <p className="text-sm font-semibold text-green-700">PRs Generated Successfully!</p>
                        <p className="text-xs text-green-600 mt-1">{generatedPRs.join(', ')}</p>
                        <button onClick={() => { setPRSubTab('list'); setMrpLinesData(null) }}
                          className="mt-2 text-xs text-green-700 underline hover:no-underline">View PR List →</button>
                      </div>
                    ) : (
                      <button onClick={generatePRsFromMRP} disabled={generatingPRs || (mrpLinesData.purchase_items.length === 0 && mrpLinesData.sfg_items.length === 0)}
                        className="px-5 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800 disabled:opacity-50">
                        {generatingPRs ? 'Generating PRs…' : 'Generate PRs'}
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
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
                    <p className="text-xs text-gray-500">{po.supplier_name} · SO: {po.so_reference || '—'} · Delivery: {po.delivery_date || '—'}</p>
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
