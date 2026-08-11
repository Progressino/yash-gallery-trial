import { useState, useEffect, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type Tab =
  | 'dashboard'
  | 'locations'
  | 'tracker'
  | 'mrp'
  | 'planning'
  | 'jobwork'
  | 'qc'
  | 'ledger'
  | 'reservations'
  | 'printed-fabric'
  | 'reports'

const planColor = (c: string) => {
  switch (c) {
    case 'green': return 'bg-green-100 text-green-800 border-green-200'
    case 'blue': return 'bg-blue-100 text-blue-800 border-blue-200'
    case 'orange': return 'bg-orange-100 text-orange-800 border-orange-200'
    case 'red': return 'bg-red-100 text-red-800 border-red-200'
    default: return 'bg-gray-100 text-gray-600 border-gray-200'
  }
}

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
  const [tab, setTab] = useState<Tab>(() => {
    try {
      const p = new URLSearchParams(window.location.search).get('tab')
      const allowed: Tab[] = ['dashboard', 'locations', 'tracker', 'mrp', 'planning', 'jobwork', 'qc', 'ledger', 'reservations', 'printed-fabric', 'reports']
      return (allowed.includes(p as Tab) ? p : 'dashboard') as Tab
    } catch {
      return 'dashboard'
    }
  })
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
  const [pfSkuSearch, setPFSkuSearch] = useState('')
  const [pfSelectedSkus, setPFSelectedSkus] = useState<string[]>([])
  const [pfSkuQtyMap, setPFSkuQtyMap] = useState<Record<string, number>>({})

  // Planning / Allocation
  const [planView, setPlanView] = useState<'tree' | 'allocate' | 'reallocate' | 'print-jo' | 'history' | 'report'>(() => {
    try {
      const p = new URLSearchParams(window.location.search).get('plan')
      const allowed = ['tree', 'allocate', 'reallocate', 'print-jo', 'history', 'report'] as const
      return (allowed.includes(p as typeof allowed[number]) ? p : 'tree') as typeof allowed[number]
    } catch {
      return 'tree'
    }
  })
  const [expandedGrey, setExpandedGrey] = useState<Record<string, boolean>>({})
  const [expandedPrinted, setExpandedPrinted] = useState<Record<string, boolean>>({})
  const [greyAllocForm, setGreyAllocForm] = useState({
    grey_code: '', printed_code: '', qty: 0, so_number: '', fg_sku: '', reason: '',
  })
  const [pfAllocForm, setPfAllocForm] = useState({
    printed_code: '', so_number: '', fg_sku: '', qty: 0, reason: '',
  })
  const [reallocForm, setReallocForm] = useState({
    reservation_id: '', from_so: '', from_sku: '', to_so: '', to_sku: '', printed_code: '', qty: '', reason: '',
  })
  const [planMsg, setPlanMsg] = useState('')

  // ── Queries ──────────────────────────────────────────────────────────────────
  const { data: stats } = useQuery<GreyStats>({
    queryKey: ['grey-stats'],
    queryFn: () => api.get('/grey/stats').then(r => r.data),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })
  const { data: locData } = useQuery({
    queryKey: ['grey-locations'],
    queryFn: () => api.get('/grey/locations').then(r => r.data),
    enabled: tab === 'locations' || tab === 'dashboard',
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })
  const { data: entries = [] } = useQuery<GreyEntry[]>({
    queryKey: ['grey', filterStatus],
    queryFn: () => api.get('/grey' + (filterStatus ? `?status=${encodeURIComponent(filterStatus)}` : '')).then(r => r.data),
    enabled: tab === 'tracker' || tab === 'dashboard' || tab === 'jobwork',
    staleTime: 10_000,
    refetchOnMount: 'always',
    refetchOnWindowFocus: false,
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
    staleTime: 10_000,
    refetchOnMount: 'always',
    refetchOnWindowFocus: false,
  })
  const { data: qcEvents = [] } = useQuery({
    queryKey: ['grey-qc-events'],
    queryFn: () => api.get('/grey/qc-events').then(r => r.data),
    enabled: tab === 'qc',
  })
  // Printed Fabric queries
  const { data: printedFabricUnchecked = [] } = useQuery({
    queryKey: ['printed-fabric-unchecked', tab, pfSubTab],
    queryFn: () => api.get('/grey/printed-fabric/unchecked').then(r => r.data),
    enabled: tab === 'printed-fabric' && pfSubTab === 'unchecked',
    staleTime: 5_000,
    refetchOnMount: 'always',
    refetchOnWindowFocus: false,
  })
  const { data: printedFabricChecked = [] } = useQuery({
    queryKey: ['printed-fabric-checked', tab, pfSubTab],
    queryFn: () => api.get('/grey/printed-fabric/checked').then(r => r.data),
    enabled: tab === 'printed-fabric' && pfSubTab === 'checked',
    staleTime: 10_000,
    refetchOnMount: 'always',
    refetchOnWindowFocus: false,
  })
  const { data: pfReserveOptions } = useQuery({
    queryKey: ['printed-fabric-reserve-options', pfReserveForm.fabric_code || ''],
    queryFn: () =>
      api
        .get('/grey/printed-fabric/reserve-options', {
          params: pfReserveForm.fabric_code
            ? { fabric_code: pfReserveForm.fabric_code }
            : undefined,
        })
        .then(r => r.data),
    enabled: tab === 'printed-fabric' && (pfSubTab === 'ready-to-cut' || showPFReserveForm),
  })
  const { data: printedReadyToCut = [] } = useQuery({
    queryKey: ['printed-ready-to-cut', tab, pfSubTab],
    queryFn: () => api.get('/grey/printed-fabric/ready-to-cut').then(r => r.data),
    enabled: tab === 'printed-fabric' && pfSubTab === 'ready-to-cut',
    staleTime: 10_000,
    refetchOnMount: 'always',
    refetchOnWindowFocus: false,
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
    onError: (err: unknown) => { console.error('vendor-dispatch failed', err) },
  })
  const arriveMut = useMutation({
    mutationFn: ({ id, qty }: { id: number; qty?: number }) => api.post(`/grey/${id}/arrive-transport`, { qty }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey'] }); qc.invalidateQueries({ queryKey: ['grey-stats'] }); setEditEntry(null) },
    onError: (err: unknown) => { console.error('arrive-transport failed', err) },
  })
  const reverseArriveMut = useMutation({
    mutationFn: (id: number) => api.post(`/grey/${id}/reverse-arrive-transport`, {}),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey'] }); qc.invalidateQueries({ queryKey: ['grey-stats'] }) },
    onError: (err: unknown) => {
      const e = err as { response?: { data?: { detail?: string } } }
      alert(e?.response?.data?.detail || 'Reverse failed')
    },
  })
  const reverseDispatchMut = useMutation({
    mutationFn: (id: number) => api.post(`/grey/${id}/reverse-vendor-dispatch`, {}),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['grey'] }); qc.invalidateQueries({ queryKey: ['grey-stats'] }) },
    onError: (err: unknown) => {
      const e = err as { response?: { data?: { detail?: string } } }
      alert(e?.response?.data?.detail || 'Reverse failed')
    },
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
  const { data: planTree, isFetching: planTreeLoading } = useQuery({
    queryKey: ['grey-planning-tree'],
    queryFn: () => api.get('/grey/planning/tree').then(r => r.data),
    enabled: tab === 'planning' && planView === 'tree',
  })
  const { data: planGreyStock = [] } = useQuery({
    queryKey: ['grey-planning-stock'],
    queryFn: () => api.get('/grey/planning/grey-stock').then(r => r.data),
    enabled: tab === 'planning' && (planView === 'allocate' || planView === 'tree'),
  })
  const { data: planHistory = [] } = useQuery({
    queryKey: ['grey-planning-history'],
    queryFn: () => api.get('/grey/planning/history', { params: { limit: 100 } }).then(r => r.data),
    enabled: tab === 'planning' && planView === 'history',
  })
  const { data: planPrintJo = [] } = useQuery({
    queryKey: ['grey-planning-print-jo'],
    queryFn: () => api.get('/grey/planning/printing-jo-status').then(r => r.data),
    enabled: tab === 'planning' && planView === 'print-jo',
  })
  const { data: planFgReport = [] } = useQuery({
    queryKey: ['grey-planning-fg-report'],
    queryFn: () => api.get('/grey/planning/fg-status-report').then(r => r.data),
    enabled: tab === 'planning' && planView === 'report',
  })
  const { data: planPfAlloc = [] } = useQuery({
    queryKey: ['grey-planning-pf-alloc'],
    queryFn: () => api.get('/grey/planning/printed-allocations', { params: { status: 'Active' } }).then(r => r.data),
    enabled: tab === 'planning' && (planView === 'reallocate' || planView === 'allocate'),
  })

  const invalidatePlanning = () => {
    qc.invalidateQueries({ queryKey: ['grey-planning-tree'] })
    qc.invalidateQueries({ queryKey: ['grey-planning-stock'] })
    qc.invalidateQueries({ queryKey: ['grey-planning-history'] })
    qc.invalidateQueries({ queryKey: ['grey-planning-print-jo'] })
    qc.invalidateQueries({ queryKey: ['grey-planning-fg-report'] })
    qc.invalidateQueries({ queryKey: ['grey-planning-pf-alloc'] })
    qc.invalidateQueries({ queryKey: ['printed-fabric-checked'] })
    qc.invalidateQueries({ queryKey: ['printed-ready-to-cut'] })
  }

  const greyAllocMut = useMutation({
    mutationFn: (b: object) => api.post('/grey/planning/allocate-grey', b),
    onSuccess: () => { setPlanMsg('Grey allocated'); invalidatePlanning() },
    onError: (e: any) => setPlanMsg(e?.response?.data?.detail || e?.message || 'Allocate failed'),
  })
  const pfAllocMut = useMutation({
    mutationFn: (b: object) => api.post('/grey/planning/allocate-printed', b),
    onSuccess: () => { setPlanMsg('Printed fabric allocated'); invalidatePlanning() },
    onError: (e: any) => setPlanMsg(e?.response?.data?.detail || e?.message || 'Allocate failed'),
  })
  const reallocMut = useMutation({
    mutationFn: (b: object) => api.post('/grey/planning/reallocate-printed', b),
    onSuccess: () => { setPlanMsg('Reallocated (print history unchanged)'); invalidatePlanning() },
    onError: (e: any) => setPlanMsg(e?.response?.data?.detail || e?.message || 'Reallocate failed'),
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
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['printed-fabric-unchecked'] })
      qc.invalidateQueries({ queryKey: ['printed-fabric-checked'] })
      qc.invalidateQueries({ queryKey: ['printed-ready-to-cut'] })
      setShowPFQCForm(false)
      setPFQCTarget(null)
      const pending = Number(res?.data?.pending_qty ?? 0)
      if (pending > 0.01) {
        alert(`QC saved. ${pending.toFixed(1)} m still pending in Unchecked for this JWO.`)
      }
    },
    onError: (e: unknown) => {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      alert(msg || 'QC could not be saved')
    },
  })
  const pfCheckedAvailable = (pfReserveOptions?.fabrics ?? []) as {
    fabric_code: string
    fabric_name?: string
    available_qty?: number
    passed_qty?: number
  }[]
  const pfSalesOrders = (pfReserveOptions?.sales_orders ?? []) as {
    so_number: string
    buyer?: string
    status?: string
    lines: { sku: string; sku_name?: string; qty?: number; unit?: string }[]
  }[]
  const pfSelectedFabric = pfCheckedAvailable.find(f => f.fabric_code === pfReserveForm.fabric_code)
  const pfSelectedSO = pfSalesOrders.find(o => o.so_number === pfReserveForm.so_number)
  const pfSkuLines = pfSelectedSO?.lines ?? []
  const pfSkuLinesFiltered = useMemo(() => {
    const q = pfSkuSearch.trim().toLowerCase()
    if (!q) return pfSkuLines
    return pfSkuLines.filter(ln =>
      (ln.sku || '').toLowerCase().includes(q)
      || (ln.sku_name || '').toLowerCase().includes(q),
    )
  }, [pfSkuLines, pfSkuSearch])

  useEffect(() => {
    const fabricCode = pfReserveForm.fabric_code
    if (!fabricCode || !pfReserveForm.so_number || pfSelectedSkus.length === 0) return
    let cancelled = false
    const maxAvail = pfSelectedFabric?.available_qty ?? 0
    Promise.all(
      pfSelectedSkus.map(async (sku) => {
        const line = pfSkuLines.find(l => l.sku === sku)
        const orderQty = Number(line?.qty) || 0
        if (orderQty <= 0) return [sku, 0] as const
        try {
          const res = await api.get(`/production/bom-inputs/${encodeURIComponent(sku)}`, { params: { qty: orderQty } })
          const inputs = (res.data?.inputs ?? []) as { material_code?: string; adj_qty?: number }[]
          const match = inputs.find(i => (i.material_code || '').toUpperCase() === fabricCode.toUpperCase())
          const bomMtr = Number(match?.adj_qty) || 0
          return [sku, Math.round(Math.min(bomMtr, maxAvail || bomMtr) * 1000) / 1000] as const
        } catch {
          return [sku, pfSkuQtyMap[sku] || 0] as const
        }
      }),
    ).then((pairs) => {
      if (cancelled) return
      setPFSkuQtyMap((prev) => {
        const next = { ...prev }
        for (const [sku, qty] of pairs) next[sku] = qty
        return next
      })
    })
    return () => { cancelled = true }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- refresh when selection / fabric / SO changes
  }, [pfSelectedSkus.join('|'), pfReserveForm.fabric_code, pfReserveForm.so_number, pfSkuLines, pfSelectedFabric?.available_qty])

  const pfReserveMut = useMutation({
    mutationFn: async () => {
      const fabric_code = pfReserveForm.fabric_code
      const so_number = pfReserveForm.so_number
      if (!fabric_code || !so_number || pfSelectedSkus.length === 0) {
        throw new Error('Select fabric, SO, and at least one SKU')
      }
      const errors: string[] = []
      let ok = 0
      for (const sku of pfSelectedSkus) {
        const qty = Number(pfSkuQtyMap[sku] || 0)
        if (qty <= 0) {
          errors.push(`${sku}: qty required`)
          continue
        }
        try {
          await api.post('/grey/printed-fabric/reserve', {
            fabric_code,
            fabric_name: pfReserveForm.fabric_name,
            so_number,
            sku,
            qty,
            remarks: pfReserveForm.remarks,
          })
          ok += 1
        } catch (e: unknown) {
          const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
          errors.push(`${sku}: ${detail || 'failed'}`)
        }
      }
      if (ok === 0) throw new Error(errors.join('\n') || 'Reserve failed')
      return { ok, errors }
    },
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['printed-fabric-checked'] })
      qc.invalidateQueries({ queryKey: ['printed-ready-to-cut'] })
      qc.invalidateQueries({ queryKey: ['printed-fabric-reserve-options'] })
      setShowPFReserveForm(false)
      setPFReserveForm({ fabric_code: '', fabric_name: '', so_number: '', sku: '', qty: 0, remarks: '' })
      setPFSelectedSkus([])
      setPFSkuQtyMap({})
      setPFSkuSearch('')
      if (res.errors.length) alert(`Reserved ${res.ok} SKU(s).\nSome failed:\n${res.errors.join('\n')}`)
    },
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
    ['mrp', '📐 Material Req. Planning / SO'],
    ['planning', '🧩 Planning & Allocation'],
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
                <button onClick={() => {
                  const todayIso = new Date().toISOString().slice(0, 10)
                  setDispatchModal(editEntry)
                  setDispatchForm({
                    bilty_no: editEntry.bilty_no || '',
                    transporter: editEntry.transporter || '',
                    dispatch_date: editEntry.dispatch_date || todayIso,
                    expected_arrival: editEntry.expected_arrival || '',
                    dispatched_qty: editEntry.dispatched_qty || editEntry.ordered_qty || 0,
                    vehicle_no: editEntry.vehicle_no || '',
                  })
                }} className="px-3 py-2 bg-blue-600 text-white rounded-lg text-xs font-medium">Vendor dispatch → In transit</button>
                <button
                  onClick={() => arriveMut.mutate({ id: editEntry.id })}
                  disabled={arriveMut.isPending || editEntry.status !== 'In Transit' || (editEntry.in_transit_qty ?? 0) <= 0}
                  title={
                    editEntry.status !== 'In Transit' || (editEntry.in_transit_qty ?? 0) <= 0
                      ? 'Already received at transport or nothing in transit'
                      : 'Receive at transport (one-time stock post)'
                  }
                  className="px-3 py-2 bg-cyan-600 text-white rounded-lg text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {arriveMut.isPending ? 'Receiving…' : editEntry.status === 'At Transport Location' ? 'At transport ✓' : 'Arrive transport hub'}
                </button>
                <button onClick={() => { setTransferModal(editEntry); setTransferForm({ to_location: 'factory', qty: editEntry.transport_qty || 0 }) }} className="px-3 py-2 bg-purple-600 text-white rounded-lg text-xs font-medium">Transfer from transport…</button>
                <button onClick={() => { setQcModal(editEntry); setQcForm(f => ({ ...f, received_qty: editEntry.received_qty || editEntry.ordered_qty || 0, checked_qty: editEntry.checked_qty || 0, passed_qty: editEntry.passed_qty ?? 0, rejected_qty: editEntry.rejected_qty || 0, rework_qty: editEntry.rework_qty || 0 })) }} className="px-3 py-2 bg-amber-600 text-white rounded-lg text-xs font-medium">Record QC</button>
                <button onClick={() => { setPrinterModal(editEntry); setPrinterForm(f => ({ ...f, issue_qty: editEntry.transport_qty || editEntry.printer_qty || 0 })) }} className="px-3 py-2 bg-orange-600 text-white rounded-lg text-xs font-medium">Issue to printer</button>
                <button onClick={() => { setReturnModal(editEntry); setReturnForm(r => ({ ...r, return_qty: editEntry.rejected_qty || 0 })) }} className="px-3 py-2 bg-red-700 text-white rounded-lg text-xs font-medium">Return to vendor (DN)</button>
                {(editEntry.transport_qty ?? 0) > 0 && (editEntry.factory_qty ?? 0) <= 0 && (editEntry.printer_qty ?? 0) <= 0 && (
                  <button
                    onClick={() => { if (confirm('Reverse transport receive? Stock moves back to in-transit.')) reverseArriveMut.mutate(editEntry.id) }}
                    disabled={reverseArriveMut.isPending}
                    className="px-3 py-2 bg-rose-100 text-rose-800 border border-rose-200 rounded-lg text-xs font-medium disabled:opacity-50"
                  >
                    Cancel transport receive
                  </button>
                )}
                {editEntry.status === 'In Transit' && (editEntry.in_transit_qty ?? 0) > 0 && (editEntry.transport_qty ?? 0) <= 0 && (
                  <button
                    onClick={() => { if (confirm('Cancel vendor dispatch?')) reverseDispatchMut.mutate(editEntry.id) }}
                    disabled={reverseDispatchMut.isPending}
                    className="px-3 py-2 bg-rose-100 text-rose-800 border border-rose-200 rounded-lg text-xs font-medium disabled:opacity-50"
                  >
                    Cancel dispatch
                  </button>
                )}
                <button onClick={() => setEditEntry(null)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Close</button>
              </div>
            </div>
          )}

          {/* Dispatch Modal */}
          {dispatchModal && (() => {
            const bilty = dispatchForm.bilty_no.trim()
            const qty = Number(dispatchForm.dispatched_qty) || 0
            const disabledReason =
              !bilty ? 'Bilty / LR number is required'
              : qty <= 0 ? 'Dispatched quantity must be greater than 0'
              : null
            const errMsg = (() => {
              const e = dispatchMut.error as { response?: { data?: { detail?: string } }; message?: string } | null
              if (!e) return ''
              return e?.response?.data?.detail || e?.message || 'Dispatch failed — try again'
            })()
            return (
            <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
              <div className="bg-white rounded-xl max-w-md w-full p-5 space-y-3 shadow-xl">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-800">Vendor dispatch → In transit</h3>
                    <p className="text-xs text-gray-500 mt-0.5">{dispatchModal.po_number} · {dispatchModal.material_code}</p>
                  </div>
                  <button onClick={() => setDispatchModal(null)} className="text-gray-400 hover:text-gray-700 text-xl leading-none">×</button>
                </div>

                <div>
                  <label className="text-xs text-gray-500">Bilty / LR no <span className="text-rose-600">*</span></label>
                  <input
                    className="w-full border rounded px-2 py-1.5 text-sm mt-0.5"
                    value={dispatchForm.bilty_no}
                    onChange={e => setDispatchForm(x => ({ ...x, bilty_no: e.target.value }))}
                    placeholder="e.g. 9837"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500">Transporter</label>
                  <input
                    className="w-full border rounded px-2 py-1.5 text-sm mt-0.5"
                    value={dispatchForm.transporter}
                    onChange={e => setDispatchForm(x => ({ ...x, transporter: e.target.value }))}
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-xs text-gray-500">Dispatch date</label>
                    <input
                      type="date"
                      className="w-full border rounded px-2 py-1.5 text-sm mt-0.5"
                      value={dispatchForm.dispatch_date}
                      onChange={e => setDispatchForm(x => ({ ...x, dispatch_date: e.target.value }))}
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Expected arrival</label>
                    <input
                      type="date"
                      className="w-full border rounded px-2 py-1.5 text-sm mt-0.5"
                      value={dispatchForm.expected_arrival}
                      onChange={e => setDispatchForm(x => ({ ...x, expected_arrival: e.target.value }))}
                    />
                  </div>
                </div>
                <div>
                  <label className="text-xs text-gray-500">Vehicle no</label>
                  <input
                    className="w-full border rounded px-2 py-1.5 text-sm mt-0.5"
                    value={dispatchForm.vehicle_no}
                    onChange={e => setDispatchForm(x => ({ ...x, vehicle_no: e.target.value }))}
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500">Dispatched Qty (MTR) <span className="text-rose-600">*</span></label>
                  <input
                    type="number"
                    min="0"
                    step="any"
                    className="w-full border rounded px-2 py-1.5 text-sm mt-0.5"
                    value={dispatchForm.dispatched_qty}
                    onChange={e => setDispatchForm(x => ({ ...x, dispatched_qty: +e.target.value }))}
                  />
                </div>

                {errMsg && (
                  <div className="px-3 py-2 bg-rose-50 border border-rose-200 rounded text-xs text-rose-700">
                    {errMsg}
                  </div>
                )}
                {disabledReason && (
                  <div className="text-xs text-amber-700">⚠ {disabledReason}</div>
                )}

                <div className="flex gap-2 pt-2">
                  <button
                    onClick={() => dispatchMut.mutate({ id: dispatchModal.id, body: dispatchForm })}
                    disabled={dispatchMut.isPending || !!disabledReason}
                    className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-semibold hover:bg-[#003a78] disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {dispatchMut.isPending ? 'Confirming…' : 'Confirm dispatch'}
                  </button>
                  <button onClick={() => setDispatchModal(null)} className="px-4 py-2 border rounded-lg text-sm">Cancel</button>
                </div>
              </div>
            </div>
            )
          })()}

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

      {/* Material requirement planning */}
      {tab === 'planning' && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border p-4">
            <p className="text-sm text-gray-600 max-w-3xl">
              Grey is planned against <b>Printed Fabric (P-Code)</b>, not FG directly.
              Hierarchy: Grey → Printed → FG SKU → Sales Order.
              Reallocate received printed fabric until Cutting Issue; then allocation is locked.
            </p>
            <div className="flex flex-wrap gap-1 mt-3">
              {([
                ['tree', 'MRP Tree'],
                ['allocate', 'Allocate'],
                ['reallocate', 'Reallocate'],
                ['print-jo', 'Printing JO'],
                ['report', 'FG Status'],
                ['history', 'Audit Trail'],
              ] as const).map(([k, lab]) => (
                <button key={k} type="button" onClick={() => { setPlanView(k); setPlanMsg('') }}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium ${planView === k ? 'bg-[#002B5B] text-white' : 'bg-gray-100 text-gray-600'}`}>
                  {lab}
                </button>
              ))}
            </div>
            {planMsg && <p className="mt-2 text-xs text-amber-800 bg-amber-50 border border-amber-100 rounded px-2 py-1">{planMsg}</p>}
            <div className="flex flex-wrap gap-2 mt-3 text-[10px]">
              {[
                ['green', 'Allocated'],
                ['blue', 'Printed / Locked'],
                ['orange', 'Partial / Reserved'],
                ['grey', 'Pending'],
                ['red', 'Shortage'],
              ].map(([c, lab]) => (
                <span key={c} className={`px-2 py-0.5 rounded border ${planColor(c)}`}>{lab}</span>
              ))}
            </div>
          </div>

          {planView === 'tree' && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <div className="px-4 py-2 border-b flex justify-between items-center">
                <span className="text-sm font-semibold text-gray-700">Planning tree</span>
                <button type="button" onClick={() => qc.invalidateQueries({ queryKey: ['grey-planning-tree'] })}
                  className="text-xs text-blue-600">Refresh</button>
              </div>
              {planTreeLoading && <p className="p-4 text-sm text-gray-400">Loading…</p>}
              {!planTreeLoading && (!(planTree as any)?.nodes?.length) && (
                <p className="p-4 text-sm text-gray-400">No open SO lines or BOM links. Ensure SOs and FG→SFG→GF BOM exist.</p>
              )}
              <ul className="divide-y">
                {((planTree as any)?.nodes || []).map((g: any) => (
                  <li key={g.code} className="px-3 py-2">
                    <button type="button" className="w-full flex items-center gap-2 text-left"
                      onClick={() => setExpandedGrey(e => ({ ...e, [g.code]: !e[g.code] }))}>
                      <span className="text-gray-400">{expandedGrey[g.code] ? '▼' : '▶'}</span>
                      <span className={`text-xs px-2 py-0.5 rounded border ${planColor(g.color)}`}>Grey</span>
                      <span className="font-mono font-semibold text-sm">{g.code}</span>
                      <span className="text-xs text-gray-500">{g.name}</span>
                      <span className="ml-auto text-xs text-gray-500">
                        free {Number(g.free_qty || 0).toFixed(0)} · alloc {Number(g.allocated_qty || 0).toFixed(0)} m
                      </span>
                    </button>
                    {expandedGrey[g.code] && (g.printed || []).map((p: any) => {
                      const pk = `${g.code}::${p.code}`
                      return (
                        <div key={p.code} className="ml-6 mt-2 border-l-2 border-blue-100 pl-3">
                          <button type="button" className="w-full flex items-center gap-2 text-left"
                            onClick={() => setExpandedPrinted(e => ({ ...e, [pk]: !e[pk] }))}>
                            <span className="text-gray-400 text-xs">{expandedPrinted[pk] ? '▼' : '▶'}</span>
                            <span className={`text-xs px-2 py-0.5 rounded border ${planColor(p.color)}`}>P-Code</span>
                            <span className="font-mono text-sm font-semibold text-blue-800">{p.code}</span>
                            <span className="text-xs text-gray-500">avail {Number(p.available_qty || 0).toFixed(0)} m</span>
                          </button>
                          {expandedPrinted[pk] && (
                            <ul className="mt-1 space-y-1">
                              {(p.fg_lines || []).map((fl: any, i: number) => (
                                <li key={i} className="ml-4 flex flex-wrap items-center gap-2 text-xs py-1">
                                  <span className="font-semibold">{fl.sku}</span>
                                  <span className="text-gray-400">→</span>
                                  <span className="font-mono">{fl.so_number}</span>
                                  <span className={`px-2 py-0.5 rounded border ${planColor(fl.color)}`}>{fl.status}</span>
                                  <span className="text-gray-500">
                                    PF {Number(fl.allocated_printed || 0).toFixed(0)}/{Number(fl.req_printed || 0).toFixed(0)} ·
                                    Grey intent {Number(fl.allocated_grey || 0).toFixed(0)} m
                                  </span>
                                </li>
                              ))}
                            </ul>
                          )}
                        </div>
                      )
                    })}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {planView === 'allocate' && (
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white rounded-xl border p-4 space-y-3">
                <h3 className="text-sm font-semibold">Stage 1 — Allocate Grey → P-Code</h3>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <label className="block">Grey code
                    <input className="mt-1 w-full border rounded px-2 py-1.5" list="plan-grey-list"
                      value={greyAllocForm.grey_code}
                      onChange={e => setGreyAllocForm(f => ({ ...f, grey_code: e.target.value }))} />
                  </label>
                  <label className="block">Printed (P-code)
                    <input className="mt-1 w-full border rounded px-2 py-1.5" value={greyAllocForm.printed_code}
                      onChange={e => setGreyAllocForm(f => ({ ...f, printed_code: e.target.value }))} />
                  </label>
                  <label className="block">Qty (m)
                    <input type="number" className="mt-1 w-full border rounded px-2 py-1.5" value={greyAllocForm.qty}
                      onChange={e => setGreyAllocForm(f => ({ ...f, qty: +e.target.value }))} />
                  </label>
                  <label className="block">Reason
                    <input className="mt-1 w-full border rounded px-2 py-1.5" value={greyAllocForm.reason}
                      onChange={e => setGreyAllocForm(f => ({ ...f, reason: e.target.value }))} />
                  </label>
                  <label className="block">SO (optional intent)
                    <input className="mt-1 w-full border rounded px-2 py-1.5" value={greyAllocForm.so_number}
                      onChange={e => setGreyAllocForm(f => ({ ...f, so_number: e.target.value }))} />
                  </label>
                  <label className="block">FG SKU (optional intent)
                    <input className="mt-1 w-full border rounded px-2 py-1.5" value={greyAllocForm.fg_sku}
                      onChange={e => setGreyAllocForm(f => ({ ...f, fg_sku: e.target.value }))} />
                  </label>
                </div>
                <datalist id="plan-grey-list">
                  {(planGreyStock as any[]).map((s: any) => (
                    <option key={s.fabric_code} value={s.fabric_code}>{s.grey_free_qty ?? s.available_qty} free</option>
                  ))}
                </datalist>
                <button type="button" disabled={!greyAllocForm.grey_code || !greyAllocForm.printed_code || greyAllocForm.qty <= 0 || greyAllocMut.isPending}
                  onClick={() => greyAllocMut.mutate(greyAllocForm)}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">
                  Allocate Grey
                </button>
              </div>
              <div className="bg-white rounded-xl border p-4 space-y-3">
                <h3 className="text-sm font-semibold">Stage 4 — Allocate Printed → FG + SO</h3>
                <p className="text-[11px] text-amber-900 bg-amber-50 border border-amber-100 rounded px-2 py-1.5">
                  Same source of truth as <b>Printed Fabric → Reserve for SO</b> (`printed_fabric_reservations`).
                  Use this form for planning / reallocation audit history; day-to-day QC reserve stays on the Printed Fabric tab.
                </p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <label className="block">Printed code
                    <input className="mt-1 w-full border rounded px-2 py-1.5" value={pfAllocForm.printed_code}
                      onChange={e => setPfAllocForm(f => ({ ...f, printed_code: e.target.value }))} />
                  </label>
                  <label className="block">Qty (m)
                    <input type="number" className="mt-1 w-full border rounded px-2 py-1.5" value={pfAllocForm.qty}
                      onChange={e => setPfAllocForm(f => ({ ...f, qty: +e.target.value }))} />
                  </label>
                  <label className="block">SO number
                    <input className="mt-1 w-full border rounded px-2 py-1.5" value={pfAllocForm.so_number}
                      onChange={e => setPfAllocForm(f => ({ ...f, so_number: e.target.value }))} />
                  </label>
                  <label className="block">FG SKU
                    <input className="mt-1 w-full border rounded px-2 py-1.5" value={pfAllocForm.fg_sku}
                      onChange={e => setPfAllocForm(f => ({ ...f, fg_sku: e.target.value }))} />
                  </label>
                </div>
                <button type="button" disabled={!pfAllocForm.printed_code || !pfAllocForm.so_number || !pfAllocForm.fg_sku || pfAllocForm.qty <= 0 || pfAllocMut.isPending}
                  onClick={() => pfAllocMut.mutate({
                    printed_code: pfAllocForm.printed_code,
                    so_number: pfAllocForm.so_number,
                    fg_sku: pfAllocForm.fg_sku,
                    qty: pfAllocForm.qty,
                    reason: pfAllocForm.reason,
                  })}
                  className="px-4 py-2 bg-blue-700 text-white rounded-lg text-sm disabled:opacity-50">
                  Allocate Printed
                </button>
                <div className="text-[11px] text-gray-500 max-h-32 overflow-auto">
                  Active PF allocations: {(planPfAlloc as any[]).length}
                  {(planPfAlloc as any[]).slice(0, 8).map((r: any) => (
                    <div key={r.id} className="font-mono">#{r.id} {r.fabric_code} → {r.sku}/{r.so_number} {r.qty}m</div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {planView === 'reallocate' && (
            <div className="bg-white rounded-xl border p-4 space-y-3 max-w-2xl">
              <h3 className="text-sm font-semibold">Reallocate Printed Fabric between FG SKUs</h3>
              <p className="text-xs text-gray-500">Does not change grey purchase, printing JO, or consumption. Blocked after Cutting Issue.</p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <label className="block col-span-2">Source reservation
                  <select className="mt-1 w-full border rounded px-2 py-1.5"
                    value={reallocForm.reservation_id}
                    onChange={e => {
                      const id = e.target.value
                      const r = (planPfAlloc as any[]).find((x: any) => String(x.id) === id)
                      setReallocForm(f => ({
                        ...f,
                        reservation_id: id,
                        from_so: r?.so_number || '',
                        from_sku: r?.sku || '',
                        printed_code: r?.fabric_code || '',
                        qty: r ? String(r.qty) : f.qty,
                      }))
                    }}>
                    <option value="">Select Active reservation…</option>
                    {(planPfAlloc as any[]).map((r: any) => (
                      <option key={r.id} value={r.id}>#{r.id} {r.fabric_code} · {r.sku} · {r.so_number} · {r.qty}m</option>
                    ))}
                  </select>
                </label>
                <label className="block">To SO
                  <input className="mt-1 w-full border rounded px-2 py-1.5" value={reallocForm.to_so}
                    onChange={e => setReallocForm(f => ({ ...f, to_so: e.target.value }))} />
                </label>
                <label className="block">To FG SKU
                  <input className="mt-1 w-full border rounded px-2 py-1.5" value={reallocForm.to_sku}
                    onChange={e => setReallocForm(f => ({ ...f, to_sku: e.target.value }))} />
                </label>
                <label className="block">Qty (blank = full)
                  <input className="mt-1 w-full border rounded px-2 py-1.5" value={reallocForm.qty}
                    onChange={e => setReallocForm(f => ({ ...f, qty: e.target.value }))} />
                </label>
                <label className="block">Reason (required)
                  <input className="mt-1 w-full border rounded px-2 py-1.5" value={reallocForm.reason}
                    onChange={e => setReallocForm(f => ({ ...f, reason: e.target.value }))} />
                </label>
              </div>
              <button type="button"
                disabled={!reallocForm.reservation_id || !reallocForm.to_so || !reallocForm.to_sku || !reallocForm.reason || reallocMut.isPending}
                onClick={() => reallocMut.mutate({
                  reservation_id: Number(reallocForm.reservation_id),
                  to_so: reallocForm.to_so,
                  to_sku: reallocForm.to_sku,
                  printed_code: reallocForm.printed_code,
                  qty: reallocForm.qty === '' ? null : Number(reallocForm.qty),
                  reason: reallocForm.reason,
                })}
                className="px-4 py-2 bg-orange-600 text-white rounded-lg text-sm disabled:opacity-50">
                Reallocate
              </button>
            </div>
          )}

          {planView === 'print-jo' && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 text-xs text-gray-500">
                  <tr>
                    <th className="text-left px-3 py-2">P-Code</th>
                    <th className="text-left px-3 py-2">Grey</th>
                    <th className="text-right px-3 py-2">Grey Allocated</th>
                    <th className="text-left px-3 py-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(planPrintJo as any[]).map((r: any, i: number) => (
                    <tr key={i} className="border-t">
                      <td className="px-3 py-2 font-mono font-semibold text-blue-800">{r.printed_code}</td>
                      <td className="px-3 py-2 font-mono">{r.grey_code} <span className="text-gray-400">{r.grey_name}</span></td>
                      <td className="px-3 py-2 text-right font-semibold">{Number(r.grey_allocated || 0).toFixed(0)} m</td>
                      <td className="px-3 py-2">
                        <span className={`text-xs px-2 py-0.5 rounded border ${r.ready_to_issue ? planColor('green') : planColor('grey')}`}>
                          {r.status_label}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!(planPrintJo as any[])?.length && <p className="p-4 text-sm text-gray-400">No active grey allocations yet.</p>}
            </div>
          )}

          {planView === 'report' && (
            <div className="bg-white rounded-xl border overflow-hidden">
              <p className="px-3 py-2 text-xs text-gray-500 border-b">Status from <b>current</b> printed fabric allocation (not original grey intent)</p>
              <table className="w-full text-sm">
                <thead className="bg-gray-50 text-xs text-gray-500">
                  <tr>
                    <th className="text-left px-3 py-2">P-Code</th>
                    <th className="text-left px-3 py-2">FG SKU</th>
                    <th className="text-left px-3 py-2">SO</th>
                    <th className="text-right px-3 py-2">Qty</th>
                    <th className="text-left px-3 py-2">Report status</th>
                  </tr>
                </thead>
                <tbody>
                  {(planFgReport as any[]).map((r: any, i: number) => (
                    <tr key={i} className="border-t">
                      <td className="px-3 py-2 font-mono">{r.printed_code}</td>
                      <td className="px-3 py-2 font-semibold">{r.fg_sku}</td>
                      <td className="px-3 py-2 font-mono text-xs">{r.so_number}</td>
                      <td className="px-3 py-2 text-right">{Number(r.qty || 0).toFixed(0)}</td>
                      <td className="px-3 py-2">
                        <span className={`text-xs px-2 py-0.5 rounded border ${planColor(r.color)}`}>{r.report_status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!(planFgReport as any[])?.length && <p className="p-4 text-sm text-gray-400">No printed allocations to report.</p>}
            </div>
          )}

          {planView === 'history' && (
            <div className="bg-white rounded-xl border overflow-auto max-h-[28rem]">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 text-gray-500 sticky top-0">
                  <tr>
                    <th className="text-left px-2 py-2">When</th>
                    <th className="text-left px-2 py-2">Event</th>
                    <th className="text-left px-2 py-2">From → To</th>
                    <th className="text-right px-2 py-2">Qty</th>
                    <th className="text-left px-2 py-2">User / Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {(planHistory as any[]).map((h: any) => (
                    <tr key={h.id} className="border-t">
                      <td className="px-2 py-1.5 whitespace-nowrap">{h.created_at}</td>
                      <td className="px-2 py-1.5 font-mono">{h.event_type}</td>
                      <td className="px-2 py-1.5">
                        {(h.printed_code || h.grey_code || '')}{' '}
                        {h.from_sku || h.from_so ? `${h.from_sku}/${h.from_so}` : '—'}
                        {' → '}
                        {h.to_sku || h.to_so ? `${h.to_sku}/${h.to_so}` : '—'}
                      </td>
                      <td className="px-2 py-1.5 text-right">{Number(h.qty || 0).toFixed(0)}</td>
                      <td className="px-2 py-1.5 text-gray-600">{h.user_name} {h.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!(planHistory as any[])?.length && <p className="p-4 text-sm text-gray-400">No allocation events yet.</p>}
            </div>
          )}
        </div>
      )}

      {tab === 'mrp' && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border p-4 space-y-3">
            <h3 className="font-semibold text-[#002B5B] text-sm">Add material requirement line</h3>
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
                  <p className="text-xs text-gray-500">QC-passed fabric — available for Ready to Cut</p>
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
              {printedFabricChecked.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No checked fabric yet. Submit QC first.</p>}
            </div>
          )}

          {/* Ready to Cut */}
          {pfSubTab === 'ready-to-cut' && (
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <div>
                  <p className="text-sm font-semibold text-gray-700">✂️ Ready to Cut</p>
                  <p className="text-xs text-gray-500">Checked printed fabric — reserve against SO and send for cutting</p>
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      const a = document.createElement('a')
                      a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(
                        'fabric_code,so_number,sku,qty,remarks\nP500Border Cotton,SO-0001,1046YKBLUE-L,6,sample\n',
                      )
                      a.download = 'printed_fabric_reserve_import_template.csv'
                      a.click()
                    }}
                    className="px-3 py-2 border border-gray-300 rounded-lg text-sm text-gray-600"
                  >
                    📥 Template
                  </button>
                  <button
                    onClick={() => {
                      setPFReserveForm({ fabric_code: '', fabric_name: '', so_number: '', sku: '', qty: 0, remarks: '' })
                      setPFSelectedSkus([])
                      setPFSkuQtyMap({})
                      setPFSkuSearch('')
                      setShowPFReserveForm(true)
                    }}
                    disabled={pfCheckedAvailable.length === 0}
                    className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800 disabled:opacity-50"
                  >
                    + Reserve for SO
                  </button>
                </div>
              </div>
              {pfCheckedAvailable.length === 0 && !showPFReserveForm && (
                <p className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                  No checked printed fabric with available metres. Complete QC under <b>Checked Stock</b> first.
                </p>
              )}
              {showPFReserveForm && (
                <div className="bg-white rounded-xl border p-4 space-y-3">
                  <h3 className="font-semibold text-gray-700">Reserve Printed Fabric against SO</h3>
                  <p className="text-xs text-gray-500">
                    Primary operational path (same reservations as Planning → Allocate Printed).
                    After you pick a fabric, only SKUs/SOs that use that fabric (BOM / Set BOM) are listed.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-gray-500">Checked fabric *</label>
                      <select
                        value={pfReserveForm.fabric_code}
                        onChange={e => {
                          const f = pfCheckedAvailable.find(x => x.fabric_code === e.target.value)
                          setPFReserveForm(prev => ({
                            ...prev,
                            fabric_code: e.target.value,
                            fabric_name: f?.fabric_name || '',
                            so_number: '',
                            qty: 0,
                          }))
                          setPFSelectedSkus([])
                          setPFSkuQtyMap({})
                          setPFSkuSearch('')
                        }}
                        className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1"
                      >
                        <option value="">Select checked fabric…</option>
                        {pfCheckedAvailable.map(f => (
                          <option key={f.fabric_code} value={f.fabric_code}>
                            {f.fabric_code} — {f.fabric_name || '—'} ({(f.available_qty ?? 0).toFixed(1)} m available)
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Fabric name</label>
                      <input
                        readOnly
                        value={pfReserveForm.fabric_name || pfSelectedFabric?.fabric_name || ''}
                        className="w-full border border-gray-100 bg-gray-50 rounded px-2 py-1.5 text-sm mt-1 text-gray-600"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500">Sales order *</label>
                      <select
                        value={pfReserveForm.so_number}
                        onChange={e => {
                          setPFReserveForm(prev => ({ ...prev, so_number: e.target.value, sku: '' }))
                          setPFSelectedSkus([])
                          setPFSkuQtyMap({})
                          setPFSkuSearch('')
                        }}
                        className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1"
                      >
                        <option value="">Select SO…</option>
                        {pfSalesOrders.map(o => (
                          <option key={o.so_number} value={o.so_number}>
                            {o.so_number}{o.buyer ? ` — ${o.buyer}` : ''} ({o.status || 'Open'})
                          </option>
                        ))}
                      </select>
                      {pfSalesOrders.length === 0 && (
                        <p className="text-[10px] text-amber-600 mt-1">
                          {pfReserveForm.fabric_code
                            ? 'No open SO lines for this fabric (BOM filter). Check Item/Set BOM mapping or pick another fabric.'
                            : 'No open sales orders. Create one under Sales Orders first.'}
                        </p>
                      )}
                      {pfReserveOptions?.fabric_filter_active && (
                        <p className="text-[10px] text-green-700 mt-1">BOM filter on — only SKUs that use this fabric are shown.</p>
                      )}
                    </div>
                    <div className="md:col-span-2">
                      <label className="text-xs text-gray-500">Style / SKU * (search & multi-select)</label>
                      <input
                        type="search"
                        value={pfSkuSearch}
                        onChange={e => setPFSkuSearch(e.target.value)}
                        disabled={!pfReserveForm.so_number}
                        placeholder="Search SKU or style…"
                        className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1 disabled:bg-gray-100"
                      />
                      <div className="mt-2 max-h-48 overflow-y-auto border border-gray-200 rounded-lg divide-y bg-white">
                        {!pfReserveForm.so_number ? (
                          <p className="text-xs text-gray-400 px-3 py-2">Select a sales order first.</p>
                        ) : pfSkuLinesFiltered.length === 0 ? (
                          <p className="text-xs text-amber-600 px-3 py-2">
                            {pfSkuLines.length === 0
                              ? 'No SKUs left to reserve on this order (already reserved or a cutting JO exists).'
                              : 'No SKUs match your search.'}
                          </p>
                        ) : (
                          pfSkuLinesFiltered.map(ln => {
                            const checked = pfSelectedSkus.includes(ln.sku)
                            return (
                              <label key={ln.sku} className="flex items-start gap-2 px-3 py-2 text-sm hover:bg-gray-50 cursor-pointer">
                                <input
                                  type="checkbox"
                                  className="mt-1"
                                  checked={checked}
                                  onChange={() => {
                                    setPFSelectedSkus(prev =>
                                      checked ? prev.filter(s => s !== ln.sku) : [...prev, ln.sku],
                                    )
                                  }}
                                />
                                <span className="min-w-0 flex-1">
                                  <span className="font-mono font-semibold text-[#002B5B]">{ln.sku}</span>
                                  {ln.sku_name ? <span className="text-gray-500"> — {ln.sku_name}</span> : null}
                                  <span className="text-gray-400 text-xs block">order {ln.qty} {ln.unit || 'PCS'}</span>
                                </span>
                                {checked && (
                                  <input
                                    type="number"
                                    min={0}
                                    step={0.1}
                                    value={pfSkuQtyMap[ln.sku] ?? ''}
                                    onClick={e => e.stopPropagation()}
                                    onChange={e => setPFSkuQtyMap(prev => ({ ...prev, [ln.sku]: +e.target.value }))}
                                    className="w-24 border rounded px-1.5 py-0.5 text-xs"
                                    placeholder="MTR"
                                    title="Qty to reserve (MTR)"
                                  />
                                )}
                              </label>
                            )
                          })
                        )}
                      </div>
                      {pfSelectedSkus.length > 0 && (
                        <p className="text-[10px] text-gray-500 mt-1">
                          {pfSelectedSkus.length} SKU(s) selected
                          {pfSelectedFabric ? ` · max ${(pfSelectedFabric.available_qty ?? 0).toFixed(1)} m fabric available` : ''}
                          {' '}· qty auto-fills from BOM when possible
                        </p>
                      )}
                    </div>
                    <div className="md:col-span-2">
                      <label className="text-xs text-gray-500">Remarks</label>
                      <input
                        value={pfReserveForm.remarks}
                        onChange={e => setPFReserveForm(prev => ({ ...prev, remarks: e.target.value }))}
                        className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1"
                      />
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => pfReserveMut.mutate()}
                      disabled={
                        pfReserveMut.isPending
                        || !pfReserveForm.fabric_code
                        || !pfReserveForm.so_number
                        || pfSelectedSkus.length === 0
                        || pfSelectedSkus.every(s => !(pfSkuQtyMap[s] > 0))
                      }
                      className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50"
                    >
                      {pfReserveMut.isPending ? 'Saving…' : `🔒 Reserve ${pfSelectedSkus.length || ''} SKU(s)`.trim()}
                    </button>
                    <button onClick={() => setShowPFReserveForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
                  </div>
                  {pfReserveMut.isError && (
                    <p className="text-xs text-red-600 whitespace-pre-wrap">
                      {String(
                        (pfReserveMut.error as { message?: string; response?: { data?: { detail?: unknown } } })?.message
                        || (pfReserveMut.error as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail
                        || 'Reserve failed',
                      )}
                    </p>
                  )}
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
                {printedReadyToCut.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No reservations yet. Reserve fabric from Checked Stock.</p>}
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
                  <div className={`text-xs px-3 py-2 rounded-lg ${
                    pfQCForm.passed_qty + pfQCForm.failed_qty > pfQCTarget.qty + 0.01
                      ? 'bg-red-50 text-red-700'
                      : 'bg-amber-50 text-amber-700'
                  }`}>
                    Pending in Unchecked: <b>{pfQCTarget.qty} m</b> · This QC: <b>{pfQCForm.passed_qty + pfQCForm.failed_qty} m</b>
                    {pfQCForm.passed_qty + pfQCForm.failed_qty <= pfQCTarget.qty + 0.01 && (
                      <> · After submit: <b>{Math.max(0, pfQCTarget.qty - pfQCForm.passed_qty - pfQCForm.failed_qty).toFixed(1)} m</b> stays in Unchecked</>
                    )}
                    {pfQCForm.passed_qty + pfQCForm.failed_qty > pfQCTarget.qty + 0.01 && (
                      <> · Cannot exceed pending qty</>
                    )}
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