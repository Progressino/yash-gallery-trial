import { useState, useMemo, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────
interface ItemType    { id: number; name: string; code: string }
interface SizeGroup   { id: number; name: string; sizes: string[] }
interface RoutingStep { id: number; name: string; description: string; sort_order: number }
interface Merchant    { id: number; merchant_code: string; merchant_name: string }
interface ItemStats {
  total_items: number; total_boms: number; certified_boms: number
  by_type: { name: string; code: string; cnt: number }[]
}
interface AllBOM {
  id: number; item_id: number; item_code: string; item_name: string
  bom_name: string; applies_to: string; line_count: number
  is_certified: number; lines_total: number; grand_total: number
  cmt_cost: number; other_cost: number; created_at: string
}
interface Buyer       { id: number; buyer_code: string; buyer_name: string }
interface PackagingLine {
  id: number; item_id: number; buyer_id: number; packaging_item_id: number
  pkg_item_code: string; pkg_item_name: string; pkg_item_type: string
  quantity: number; unit: string; remarks: string
}

interface Item {
  id: number; item_code: string; item_name: string
  item_type_id: number; item_type_name: string; item_type_code: string
  hsn_code: string; season: string; merchant_code: string
  selling_price: number; purchase_price: number
  parent_id: number | null; size_label: string; launch_date: string
  uom: string; variant_count: number; created_at: string
}
interface ItemDetail extends Item {
  variants: { id: number; item_code: string; size_label: string }[]
  routing:  { id: number; name: string; sort_order: number }[]
}
interface BOMHeader {
  id: number; item_id: number; bom_name: string; applies_to: string
  is_default: number; line_count: number; created_at: string
  is_certified: number; certified_by: string; certified_at: string
}
interface BOMLine {
  id: number; bom_id: number; component_item_id: number | null
  component_name: string; component_type: string
  quantity: number; unit: string; rate: number
  process_id: number | null; process_name: string | null
  shrinkage_pct: number; wastage_pct: number; remarks: string
}
interface BOMDetail extends BOMHeader { lines: BOMLine[]; cmt_cost: number; other_cost: number }

// ── Constants ─────────────────────────────────────────────────────────────────
const COMPONENT_TYPES = ['FG', 'SFG', 'RM', 'ACC', 'PKG', 'FUEL', 'SVC', 'CMT']
const UNITS = ['PCS', 'MTR', 'KG', 'LTR', 'SET', 'PAIR', 'BOX', 'ROLL']
const fmt  = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN')
const fmtN = (n: number, d = 2) => +n.toFixed(d)

// ── BOM cost helpers ──────────────────────────────────────────────────────────
function netQty(l: BOMLine) { return l.quantity * (1 + l.shrinkage_pct / 100) * (1 + l.wastage_pct / 100) }
function lineAmt(l: BOMLine) { return netQty(l) * l.rate }

// ── Blank form states ─────────────────────────────────────────────────────────
const UOM_OPTIONS = ['PCS', 'MTR', 'KG', 'LTR', 'SET', 'PAIR', 'BOX', 'ROLL']

const blankItem = () => ({
  item_code: '', item_name: '', item_type_id: 1,
  hsn_code: '', season: '', merchant_code: '',
  selling_price: '', purchase_price: '', launch_date: '',
  uom: 'PCS',
  sizes: [] as string[], custom_size: '',
  routing_step_ids: [] as number[], size_group_id: '',
})
const blankBOMLine = () => ({
  component_name: '', component_type: 'RM', quantity: '1', unit: 'PCS',
  rate: '0', component_item_id: null as number | null,
  process_id: '', shrinkage_pct: '0', wastage_pct: '0', remarks: '',
})

// ── Main Component ────────────────────────────────────────────────────────────
export default function ItemMaster() {
  const qc = useQueryClient()
  const [activeTab, setActiveTab] = useState<'dashboard' | 'items' | 'bom' | 'routing' | 'import' | 'merchants' | 'packaging'>('items')

  // ── Meta query ────────────────────────────────────────────────────────────
  const { data: meta } = useQuery({
    queryKey: ['item-meta'],
    queryFn: async () => {
      const { data } = await api.get('/items/meta')
      return data as { item_types: ItemType[]; size_groups: SizeGroup[]; routing_steps: RoutingStep[]; merchants: Merchant[]; buyers: Buyer[] }
    },
    staleTime: 5 * 60 * 1000,
  })
  const itemTypes    = meta?.item_types    ?? []
  const sizeGroups   = meta?.size_groups   ?? []
  const routingSteps = meta?.routing_steps ?? []
  const merchants    = meta?.merchants     ?? []
  const buyers       = meta?.buyers        ?? []

  const { data: stats } = useQuery<ItemStats>({
    queryKey: ['item-stats'],
    queryFn: async () => { const { data } = await api.get('/items/stats'); return data },
    staleTime: 30 * 1000,
  })

  const { data: allBoms = [] } = useQuery<AllBOM[]>({
    queryKey: ['item-all-boms'],
    queryFn: async () => { const { data } = await api.get('/items/boms/all'); return data },
    staleTime: 30 * 1000,
    enabled: activeTab === 'bom',
  })

  // ══════════════════════════════════════════════════════════════════════════════
  // TAB 1 — ITEMS
  // ══════════════════════════════════════════════════════════════════════════════
  const [typeFilter,   setTypeFilter]   = useState<string>('')
  const [seasonFilter, setSeasonFilter] = useState<string>('')
  const [searchQ,      setSearchQ]      = useState('')
  const [parentOnly,   setParentOnly]   = useState(true)
  const [expandedId,   setExpandedId]   = useState<number | null>(null)
  const [showNewItem,  setShowNewItem]  = useState(false)
  const [codeMode,     setCodeMode]     = useState<'manual' | 'auto'>('manual')
  const [newItem,      setNewItem]      = useState(blankItem)
  const [newItemErr,   setNewItemErr]   = useState('')
  const [sizePreview,  setSizePreview]  = useState<string[]>([])
  // Edit item
  const [showEditItem, setShowEditItem] = useState(false)
  const [editItem,     setEditItem]     = useState<null | {
    id: number; item_code: string; item_name: string; item_type_id: number
    hsn_code: string; season: string; merchant_code: string
    selling_price: string; purchase_price: string; launch_date: string; uom: string
  }>(null)
  const [editItemErr,  setEditItemErr]  = useState('')

  const { data: items = [], isLoading: loadItems } = useQuery<Item[]>({
    queryKey: ['items', typeFilter, seasonFilter, searchQ, parentOnly],
    queryFn:  async () => {
      const p = new URLSearchParams()
      if (typeFilter)   p.set('type_id', typeFilter)
      if (searchQ)      p.set('search', searchQ)
      if (parentOnly)   p.set('parent_only', 'true')
      const { data } = await api.get(`/items?${p}`)
      const result = seasonFilter ? data.filter((i: Item) => i.season === seasonFilter) : data
      return result
    },
    staleTime: 60 * 1000,
  })

  const { data: expandedDetail } = useQuery<ItemDetail>({
    queryKey: ['item-detail', expandedId],
    queryFn:  async () => { const { data } = await api.get(`/items/${expandedId}`); return data },
    enabled:  expandedId !== null,
    staleTime: 30 * 1000,
  })

  // BOMs for expanded item (Items tab)
  const { data: expandedBoms = [] } = useQuery<BOMHeader[]>({
    queryKey: ['item-boms-expand', expandedId],
    queryFn:  async () => { const { data } = await api.get(`/items/${expandedId}/boms`); return data },
    enabled:  expandedId !== null,
    staleTime: 30 * 1000,
  })

  const createItemMut = useMutation({
    mutationFn: (body: object) => api.post('/items', body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['items'] })
      setShowNewItem(false); setNewItem(blankItem()); setSizePreview([]); setNewItemErr('')
    },
    onError: (e: any) => setNewItemErr(e?.response?.data?.detail ?? 'Failed to create item.'),
  })

  const deleteItemMut = useMutation({
    mutationFn: (id: number) => api.delete(`/items/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['items'] }),
  })

  const updateItemMut = useMutation({
    mutationFn: ({ id, body }: { id: number; body: object }) => api.put(`/items/${id}`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['items'] })
      qc.invalidateQueries({ queryKey: ['item-detail'] })
      setShowEditItem(false); setEditItem(null); setEditItemErr('')
    },
    onError: (e: any) => setEditItemErr(e?.response?.data?.detail ?? 'Update failed.'),
  })

  function applyPreviewSizes(groupId: string) {
    const grp = sizeGroups.find(g => g.id === +groupId)
    const sizes = grp ? [...grp.sizes] : []
    setNewItem(p => ({ ...p, sizes, size_group_id: groupId }))
    setSizePreview(sizes)
  }

  function addCustomSize() {
    const s = newItem.custom_size.trim().toUpperCase()
    if (s && !newItem.sizes.includes(s)) {
      const next = [...newItem.sizes, s]
      setNewItem(p => ({ ...p, sizes: next, custom_size: '' }))
      setSizePreview(next)
    }
  }

  function toggleSize(sz: string) {
    const next = newItem.sizes.includes(sz)
      ? newItem.sizes.filter(s => s !== sz)
      : [...newItem.sizes, sz]
    setNewItem(p => ({ ...p, sizes: next }))
    setSizePreview(next)
  }

  // Ordered routing helpers
  function toggleRoutingStep(stepId: number) {
    setNewItem(p => ({
      ...p,
      routing_step_ids: p.routing_step_ids.includes(stepId)
        ? p.routing_step_ids.filter(x => x !== stepId)
        : [...p.routing_step_ids, stepId],
    }))
  }
  function moveRoutingStep(idx: number, dir: -1 | 1) {
    setNewItem(p => {
      const arr = [...p.routing_step_ids]
      const target = idx + dir
      if (target < 0 || target >= arr.length) return p
      ;[arr[idx], arr[target]] = [arr[target], arr[idx]]
      return { ...p, routing_step_ids: arr }
    })
  }

  function handleCreateItem() {
    if (!newItem.item_code.trim() || !newItem.item_name.trim()) {
      setNewItemErr('Item Code and Item Name are required.'); return
    }
    if (!newItem.uom) {
      setNewItemErr('UOM is required.'); return
    }
    createItemMut.mutate({
      item_code:      newItem.item_code.trim(),
      item_name:      newItem.item_name.trim(),
      item_type_id:   newItem.item_type_id,
      hsn_code:       newItem.hsn_code,
      season:         newItem.season,
      merchant_code:  newItem.merchant_code,
      selling_price:  parseFloat(String(newItem.selling_price)) || 0,
      purchase_price: parseFloat(String(newItem.purchase_price)) || 0,
      launch_date:    newItem.launch_date,
      uom:            newItem.uom,
      sizes:          newItem.sizes,
      routing_step_ids: newItem.routing_step_ids,
    })
  }

  // ══════════════════════════════════════════════════════════════════════════════
  // TAB 2 — BOM BUILDER
  // ══════════════════════════════════════════════════════════════════════════════
  const [bomSubTab,     setBomSubTab]     = useState<'builder' | 'list'>('builder')
  const [bomItemSearch, setBomItemSearch] = useState('')
  const [bomItemId,     setBomItemId]     = useState<number | null>(null)
  const [bomItemName,   setBomItemName]   = useState('')
  const [selectedBomId, setSelectedBomId] = useState<number | null>(null)
  const [showNewBOM,    setShowNewBOM]    = useState(false)
  const [newBOMName,    setNewBOMName]    = useState('')
  const [newBOMApply,   setNewBOMApply]   = useState('all')
  const [showCopyBOM,   setShowCopyBOM]   = useState(false)
  const [copyTargetSearch, setCopyTargetSearch] = useState('')
  const [copyTargetId,  setCopyTargetId]  = useState<number | null>(null)
  // Process costs + CMT
  const [bomCmtCost,    setBomCmtCost]    = useState<string>('0')
  const [bomOtherCost,  setBomOtherCost]  = useState<string>('0')
  const [showAddSvc,    setShowAddSvc]    = useState(false)
  const [newSvcLine,    setNewSvcLine]    = useState({ process_id: '', name: '', quantity: '1', unit: 'PCS', rate: '0', remarks: '' })
  const [copyTargetName,setCopyTargetName]= useState('')
  const [copyName,      setCopyName]      = useState('')
  const [newLine,       setNewLine]       = useState(blankBOMLine)
  const [showAddLine,   setShowAddLine]   = useState(false)
  // Component item search within BOM line form
  const [compSearch,    setCompSearch]    = useState('')
  const [certifyErr,    setCertifyErr]    = useState('')

  const { data: bomSearchResults = [] } = useQuery<Item[]>({
    queryKey: ['item-search', bomItemSearch],
    queryFn:  async () => {
      const { data } = await api.get(`/items/search?q=${encodeURIComponent(bomItemSearch)}`)
      return data
    },
    enabled:  bomItemSearch.length >= 2,
    staleTime: 30 * 1000,
  })

  const { data: compSearchResults = [] } = useQuery<Item[]>({
    queryKey: ['item-search', compSearch],
    queryFn:  async () => {
      const { data } = await api.get(`/items/search?q=${encodeURIComponent(compSearch)}`)
      return data
    },
    enabled:  compSearch.length >= 2,
    staleTime: 30 * 1000,
  })

  const { data: copyTargetResults = [] } = useQuery<Item[]>({
    queryKey: ['item-search', copyTargetSearch],
    queryFn:  async () => {
      const { data } = await api.get(`/items/search?q=${encodeURIComponent(copyTargetSearch)}`)
      return data
    },
    enabled:  copyTargetSearch.length >= 2,
    staleTime: 30 * 1000,
  })

  const { data: boms = [] } = useQuery<BOMHeader[]>({
    queryKey: ['boms', bomItemId],
    queryFn:  async () => { const { data } = await api.get(`/items/${bomItemId}/boms`); return data },
    enabled:  bomItemId !== null,
    staleTime: 30 * 1000,
  })

  const { data: bomItemDetail } = useQuery<ItemDetail>({
    queryKey: ['item-detail', bomItemId],
    queryFn:  async () => { const { data } = await api.get(`/items/${bomItemId}`); return data },
    enabled:  bomItemId !== null,
    staleTime: 60 * 1000,
  })

  const { data: bomDetail } = useQuery<BOMDetail>({
    queryKey: ['bom-detail', selectedBomId],
    queryFn:  async () => {
      const { data } = await api.get(`/items/${bomItemId}/boms/${selectedBomId}`)
      return data
    },
    enabled:  selectedBomId !== null && bomItemId !== null,
    staleTime: 30 * 1000,
  })

  const createBOMMut = useMutation({
    mutationFn: (body: object) => api.post(`/items/${bomItemId}/boms`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['boms', bomItemId] })
      qc.invalidateQueries({ queryKey: ['item-boms-expand', bomItemId] })
      setShowNewBOM(false); setNewBOMName('')
    },
  })

  const deleteBOMMut = useMutation({
    mutationFn: (bomId: number) => api.delete(`/items/${bomItemId}/boms/${bomId}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['boms', bomItemId] })
      qc.invalidateQueries({ queryKey: ['item-boms-expand', bomItemId] })
      setSelectedBomId(null)
    },
    onError: (e: any) => setCertifyErr(e?.response?.data?.detail ?? 'Delete failed.'),
  })

  const certifyBOMMut = useMutation({
    mutationFn: (bomId: number) => api.post(`/items/${bomItemId}/boms/${bomId}/certify`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['bom-detail', selectedBomId] })
      qc.invalidateQueries({ queryKey: ['boms', bomItemId] })
      setCertifyErr('')
    },
    onError: (e: any) => setCertifyErr(e?.response?.data?.detail ?? 'Certify failed.'),
  })

  const uncertifyBOMMut = useMutation({
    mutationFn: (bomId: number) => api.delete(`/items/${bomItemId}/boms/${bomId}/certify`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['bom-detail', selectedBomId] })
      qc.invalidateQueries({ queryKey: ['boms', bomItemId] })
      setCertifyErr('')
    },
    onError: (e: any) => setCertifyErr(e?.response?.data?.detail ?? 'Uncertify failed.'),
  })

  const addLineMut = useMutation({
    mutationFn: (body: object) => api.post(`/items/${bomItemId}/boms/${selectedBomId}/lines`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['bom-detail', selectedBomId] })
      qc.invalidateQueries({ queryKey: ['boms', bomItemId] })
      qc.invalidateQueries({ queryKey: ['item-boms-expand', bomItemId] })
      setNewLine(blankBOMLine()); setShowAddLine(false); setCompSearch('')
    },
    onError: (e: any) => setCertifyErr(e?.response?.data?.detail ?? 'Add failed.'),
  })

  const deleteLineMut = useMutation({
    mutationFn: ({ bomId, lineId }: { bomId: number; lineId: number }) =>
      api.delete(`/items/${bomItemId}/boms/${bomId}/lines/${lineId}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['bom-detail', selectedBomId] })
      qc.invalidateQueries({ queryKey: ['boms', bomItemId] })
    },
    onError: (e: any) => setCertifyErr(e?.response?.data?.detail ?? 'Delete failed.'),
  })

  const copyBOMMut = useMutation({
    mutationFn: (body: object) => api.post(`/items/${bomItemId}/boms/${selectedBomId}/copy`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['boms'] })
      qc.invalidateQueries({ queryKey: ['item-boms-expand'] })
      setShowCopyBOM(false); setCopyTargetSearch(''); setCopyTargetId(null); setCopyTargetName('')
    },
  })

  // Sync CMT/Other costs when BOM detail loads
  const prevBomDetailId = useRef<number | null>(null)
  if (bomDetail && bomDetail.id !== prevBomDetailId.current) {
    prevBomDetailId.current = bomDetail.id
    setBomCmtCost(String(bomDetail.cmt_cost ?? 0))
    setBomOtherCost(String(bomDetail.other_cost ?? 0))
  }

  const updateBomCostsMut = useMutation({
    mutationFn: (body: { cmt_cost: number; other_cost: number }) =>
      api.put(`/items/${bomItemId}/boms/${selectedBomId}`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['bom', bomItemId, selectedBomId] }),
  })

  const matLines = useMemo(() => (bomDetail?.lines ?? []).filter(l => l.component_type !== 'SVC'), [bomDetail])
  const svcLines = useMemo(() => (bomDetail?.lines ?? []).filter(l => l.component_type === 'SVC'), [bomDetail])

  const totalCost = useMemo(() =>
    (bomDetail?.lines ?? []).reduce((s, l) => s + lineAmt(l), 0), [bomDetail])

  const grandTotal = useMemo(() =>
    totalCost + (parseFloat(bomCmtCost) || 0) + (parseFloat(bomOtherCost) || 0), [totalCost, bomCmtCost, bomOtherCost])

  const costByType = useMemo(() => {
    const acc: Record<string, number> = {}
    for (const l of bomDetail?.lines ?? []) {
      acc[l.component_type] = (acc[l.component_type] ?? 0) + lineAmt(l)
    }
    return acc
  }, [bomDetail])

  function handleAddLine() {
    if (!newLine.component_item_id) {
      setCertifyErr('Select a component from Item Master search.')
      return
    }
    setCertifyErr('')
    addLineMut.mutate({
      component_name:    newLine.component_name,
      component_type:    newLine.component_type,
      quantity:          parseFloat(newLine.quantity) || 1,
      unit:              newLine.unit,
      rate:              parseFloat(newLine.rate) || 0,
      component_item_id: newLine.component_item_id,
      process_id:        newLine.process_id ? +newLine.process_id : null,
      shrinkage_pct:     parseFloat(newLine.shrinkage_pct) || 0,
      wastage_pct:       parseFloat(newLine.wastage_pct) || 0,
      remarks:           newLine.remarks,
    })
  }

  // ══════════════════════════════════════════════════════════════════════════════
  // TAB 3 — ROUTING MASTER
  // ══════════════════════════════════════════════════════════════════════════════
  const [newRoute, setNewRoute] = useState({ name: '', description: '', sort_order: '' })

  const createRouteMut = useMutation({
    mutationFn: (body: object) => api.post('/items/routing', body),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['item-meta'] }); setNewRoute({ name: '', description: '', sort_order: '' }) },
  })

  const deleteRouteMut = useMutation({
    mutationFn: (id: number) => api.delete(`/items/routing/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['item-meta'] }),
  })

  // ══════════════════════════════════════════════════════════════════════════════
  // TAB 4 — IMPORT
  // ══════════════════════════════════════════════════════════════════════════════
  const importRef = useRef<HTMLInputElement>(null)
  const [importFile,    setImportFile]    = useState<File | null>(null)
  const [importPreview, setImportPreview] = useState<any[] | null>(null)
  const [importTotal,   setImportTotal]   = useState(0)
  const [importMsg,     setImportMsg]     = useState('')
  const [importing,     setImporting]     = useState(false)

  async function handleImportPreview(file: File) {
    setImportPreview(null); setImportMsg('')
    const fd = new FormData(); fd.append('file', file)
    try {
      const { data } = await api.post('/items/import/preview', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      setImportPreview(data.rows); setImportTotal(data.total)
    } catch (e: any) {
      setImportMsg('✗ ' + (e?.response?.data?.detail ?? 'Parse failed.'))
    }
  }

  async function handleImportConfirm() {
    if (!importFile) return
    setImporting(true); setImportMsg('')
    const fd = new FormData(); fd.append('file', importFile)
    try {
      const { data } = await api.post('/items/import/confirm', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      setImportMsg(`✓ Created ${data.created} items. Skipped ${data.skipped} duplicates.${data.errors.length ? ' Errors: ' + data.errors.slice(0, 3).join('; ') : ''}`)
      qc.invalidateQueries({ queryKey: ['items'] })
      setImportPreview(null); setImportFile(null)
    } catch (e: any) {
      setImportMsg('✗ ' + (e?.response?.data?.detail ?? 'Import failed.'))
    } finally { setImporting(false) }
  }

  // ══════════════════════════════════════════════════════════════════════════════
  // TAB 5 — MERCHANTS
  // ══════════════════════════════════════════════════════════════════════════════
  const [newMerchant, setNewMerchant] = useState({ merchant_code: '', merchant_name: '' })
  const [merchantErr, setMerchantErr] = useState('')

  const createMerchantMut = useMutation({
    mutationFn: (body: object) => api.post('/items/merchants', body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['item-meta'] })
      setNewMerchant({ merchant_code: '', merchant_name: '' }); setMerchantErr('')
    },
    onError: (e: any) => setMerchantErr(e?.response?.data?.detail ?? 'Failed to create merchant.'),
  })

  const deleteMerchantMut = useMutation({
    mutationFn: (id: number) => api.delete(`/items/merchants/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['item-meta'] }),
  })

  // ══════════════════════════════════════════════════════════════════════════════
  // TAB 6 — BUYER PACKAGING
  // ══════════════════════════════════════════════════════════════════════════════
  const [newBuyer,       setNewBuyer]       = useState({ buyer_code: '', buyer_name: '' })
  const [buyerErr,       setBuyerErr]       = useState('')
  // Packaging configurator state
  const [pkgItemSearch,  setPkgItemSearch]  = useState('')
  const [pkgItemId,      setPkgItemId]      = useState<number | null>(null)
  const [pkgItemName,    setPkgItemName]    = useState('')
  const [pkgBuyerId,     setPkgBuyerId]     = useState<number | null>(null)
  const [pkgCompSearch,  setPkgCompSearch]  = useState('')
  const [showPkgForm,    setShowPkgForm]    = useState(false)
  const [newPkgLine,     setNewPkgLine]     = useState({ packaging_item_id: null as number | null, pkg_label: '', quantity: '1', unit: 'PCS', remarks: '' })
  const [pkgErr,         setPkgErr]         = useState('')

  const { data: pkgItemSearchResults = [] } = useQuery<Item[]>({
    queryKey: ['item-search', pkgItemSearch],
    queryFn:  async () => { const { data } = await api.get(`/items/search?q=${encodeURIComponent(pkgItemSearch)}`); return data },
    enabled:  pkgItemSearch.length >= 2,
    staleTime: 30 * 1000,
  })

  const { data: pkgCompSearchResults = [] } = useQuery<Item[]>({
    queryKey: ['item-search', pkgCompSearch],
    queryFn:  async () => { const { data } = await api.get(`/items/search?q=${encodeURIComponent(pkgCompSearch)}`); return data },
    enabled:  pkgCompSearch.length >= 2,
    staleTime: 30 * 1000,
  })

  const { data: packagingLines = [] } = useQuery<PackagingLine[]>({
    queryKey: ['item-packaging', pkgItemId, pkgBuyerId],
    queryFn:  async () => {
      if (pkgBuyerId) {
        const { data } = await api.get(`/items/${pkgItemId}/packaging/${pkgBuyerId}`)
        return data
      }
      const { data } = await api.get(`/items/${pkgItemId}/packaging`)
      return data
    },
    enabled:  pkgItemId !== null,
    staleTime: 30 * 1000,
  })

  const createBuyerMut = useMutation({
    mutationFn: (body: object) => api.post('/items/buyers', body),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['item-meta'] }); setNewBuyer({ buyer_code: '', buyer_name: '' }); setBuyerErr('') },
    onError: (e: any) => setBuyerErr(e?.response?.data?.detail ?? 'Failed.'),
  })

  const deleteBuyerMut = useMutation({
    mutationFn: (id: number) => api.delete(`/items/buyers/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['item-meta'] }),
  })

  const addPkgLineMut = useMutation({
    mutationFn: (body: object) => api.post(`/items/${pkgItemId}/packaging/${pkgBuyerId}/lines`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['item-packaging', pkgItemId, pkgBuyerId] })
      setNewPkgLine({ packaging_item_id: null, pkg_label: '', quantity: '1', unit: 'PCS', remarks: '' })
      setPkgCompSearch(''); setShowPkgForm(false); setPkgErr('')
    },
    onError: (e: any) => setPkgErr(e?.response?.data?.detail ?? 'Add failed.'),
  })

  const deletePkgLineMut = useMutation({
    mutationFn: ({ buyerId, lineId }: { buyerId: number; lineId: number }) =>
      api.delete(`/items/${pkgItemId}/packaging/${buyerId}/lines/${lineId}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['item-packaging', pkgItemId, pkgBuyerId] }),
  })

  // ── Render ────────────────────────────────────────────────────────────────────
  const TABS = [
    ['dashboard', '📊 Dashboard'],
    ['items',     '📦 Items'],
    ['bom',       '🧩 BOM Builder'],
    ['routing',   '⚙️ Routing'],
    ['import',    '📥 Import'],
    ['merchants', '🏪 Merchants'],
    ['packaging', '🛍️ Buyer Packaging'],
  ] as const

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Item Master & BOM</h1>
          <p className="text-sm text-gray-500 mt-0.5">Manage finished goods, raw materials, and bill of materials</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="flex border-b border-gray-200 gap-1 px-2 pt-2 overflow-x-auto">
          {TABS.map(([id, label]) => (
            <button key={id} onClick={() => setActiveTab(id)}
              className={`px-4 py-2.5 text-sm font-medium transition-colors rounded-t-lg whitespace-nowrap ${
                activeTab === id
                  ? 'border-b-2 border-[#002B5B] text-[#002B5B] bg-blue-50'
                  : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
              }`}>
              {label}
            </button>
          ))}
        </div>

        <div className="p-5">

          {/* ================================================================
              TAB 0 — DASHBOARD
              ================================================================ */}
          {activeTab === 'dashboard' && (
            <div className="space-y-6">
              {/* KPI cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[
                  { label: 'Total Items',    value: stats?.total_items    ?? '—', color: 'bg-[#002B5B]' },
                  { label: 'Finished Goods', value: stats?.by_type?.find(t => t.code === 'FG')?.cnt ?? '—', color: 'bg-emerald-700' },
                  { label: 'BOMs Created',   value: stats?.total_boms     ?? '—', color: 'bg-amber-700' },
                  { label: 'Certified BOMs', value: stats?.certified_boms ?? '—', color: 'bg-blue-700' },
                ].map(kpi => (
                  <div key={kpi.label} className={`${kpi.color} rounded-2xl p-5 text-white`}>
                    <div className="text-3xl font-extrabold">{kpi.value}</div>
                    <div className="text-xs uppercase tracking-widest opacity-70 mt-1">{kpi.label}</div>
                  </div>
                ))}
              </div>

              {/* Items by type */}
              {stats && stats.by_type.length > 0 && (
                <div className="bg-gray-50 border border-gray-200 rounded-2xl p-5">
                  <h3 className="text-sm font-semibold text-gray-700 mb-4 uppercase tracking-wide">Items by Category</h3>
                  <div className="space-y-2">
                    {stats.by_type.filter(t => t.cnt > 0).map(t => (
                      <div key={t.code} className="flex items-center gap-3">
                        <span className="w-36 text-xs font-medium text-gray-600 truncate">{t.name}</span>
                        <div className="flex-1 bg-gray-200 rounded-full h-2 overflow-hidden">
                          <div
                            className="h-2 rounded-full bg-[#002B5B] transition-all"
                            style={{ width: `${Math.round((t.cnt / Math.max(stats.total_items, 1)) * 100)}%` }}
                          />
                        </div>
                        <span className="text-xs font-semibold text-gray-700 w-6 text-right">{t.cnt}</span>
                      </div>
                    ))}
                    {stats.by_type.filter(t => t.cnt > 0).length === 0 && (
                      <p className="text-sm text-gray-400 text-center py-4">No items yet. Create your first item in the Items tab.</p>
                    )}
                  </div>
                </div>
              )}

              {/* Quick action buttons */}
              <div className="flex flex-wrap gap-3">
                <button onClick={() => setActiveTab('items')}
                  className="bg-[#002B5B] text-white text-sm px-5 py-2.5 rounded-xl hover:bg-[#003d80] transition-colors font-medium">
                  + New Item
                </button>
                <button onClick={() => setActiveTab('bom')}
                  className="bg-white border border-gray-200 text-gray-700 text-sm px-5 py-2.5 rounded-xl hover:bg-gray-50 transition-colors font-medium">
                  🧩 BOM Builder
                </button>
                <button onClick={() => setActiveTab('import')}
                  className="bg-white border border-gray-200 text-gray-700 text-sm px-5 py-2.5 rounded-xl hover:bg-gray-50 transition-colors font-medium">
                  📥 Bulk Import
                </button>
              </div>
            </div>
          )}

          {/* ================================================================
              TAB 1 — ITEMS
              ================================================================ */}
          {activeTab === 'items' && (
            <div className="space-y-4">
              {/* Controls row */}
              <div className="flex flex-wrap gap-3 items-center justify-between">
                <div className="flex gap-2 flex-wrap">
                  <input
                    type="text" placeholder="Search items…" value={searchQ}
                    onChange={e => setSearchQ(e.target.value)}
                    className="border border-gray-200 rounded-lg px-3 py-2 text-sm w-48 focus:outline-none focus:ring-2 focus:ring-[#002B5B]"
                  />
                  <select value={typeFilter} onChange={e => setTypeFilter(e.target.value)}
                    className="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                    <option value="">All Types</option>
                    {itemTypes.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
                  </select>
                  <input type="text" placeholder="Season…" value={seasonFilter}
                    onChange={e => setSeasonFilter(e.target.value)}
                    className="border border-gray-200 rounded-lg px-3 py-2 text-sm w-28 focus:outline-none focus:ring-2 focus:ring-[#002B5B]"
                  />
                  <label className="flex items-center gap-1.5 text-sm text-gray-600 cursor-pointer">
                    <input type="checkbox" checked={parentOnly} onChange={e => setParentOnly(e.target.checked)} className="rounded" />
                    Parents only
                  </label>
                </div>
                <button onClick={() => { setShowNewItem(true); setCodeMode('manual'); setNewItem(blankItem()) }}
                  className="bg-[#002B5B] text-white text-sm px-4 py-2 rounded-lg hover:bg-[#003d80] transition-colors font-medium">
                  + New Item
                </button>
              </div>

              {/* Items table */}
              {loadItems ? (
                <div className="text-center py-10 text-gray-400">Loading…</div>
              ) : items.length === 0 ? (
                <div className="text-center py-14 text-gray-400">
                  <div className="text-4xl mb-2">📦</div>
                  <p>No items yet. Create your first item or import from Excel.</p>
                </div>
              ) : (
                <div className="overflow-x-auto rounded-xl border border-gray-200">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="bg-gray-50 border-b border-gray-200">
                        <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">Item Code</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">Name</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">Type</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">HSN</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">Season</th>
                        <th className="px-4 py-3 text-right text-xs font-semibold text-gray-500 uppercase tracking-wide">Sizes</th>
                        <th className="px-4 py-3 text-right text-xs font-semibold text-gray-500 uppercase tracking-wide">Selling ₹</th>
                        <th className="px-4 py-3 text-right text-xs font-semibold text-gray-500 uppercase tracking-wide">Cost ₹</th>
                        <th className="px-4 py-3 text-center text-xs font-semibold text-gray-500 uppercase tracking-wide">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {items.map(item => (
                        <>
                          <tr key={item.id}
                            className="hover:bg-gray-50 cursor-pointer transition-colors"
                            onClick={() => setExpandedId(expandedId === item.id ? null : item.id)}>
                            <td className="px-4 py-3 font-mono font-medium text-[#002B5B]">{item.item_code}</td>
                            <td className="px-4 py-3 font-medium text-gray-900">{item.item_name}</td>
                            <td className="px-4 py-3">
                              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-50 text-blue-700">
                                {item.item_type_code}
                              </span>
                            </td>
                            <td className="px-4 py-3 text-gray-500">{item.hsn_code || '—'}</td>
                            <td className="px-4 py-3 text-gray-500">{item.season || '—'}</td>
                            <td className="px-4 py-3 text-right">
                              {item.variant_count > 0
                                ? <span className="text-blue-600 text-xs font-medium">{item.variant_count} sizes</span>
                                : <span className="text-gray-400 text-xs">—</span>}
                            </td>
                            <td className="px-4 py-3 text-right text-gray-700">{item.selling_price > 0 ? fmt(item.selling_price) : '—'}</td>
                            <td className="px-4 py-3 text-right text-gray-700">{item.purchase_price > 0 ? fmt(item.purchase_price) : '—'}</td>
                            <td className="px-4 py-3 text-center">
                              <div className="flex items-center justify-center gap-1">
                                <button
                                  onClick={e => { e.stopPropagation(); setEditItem({ id: item.id, item_code: item.item_code, item_name: item.item_name, item_type_id: item.item_type_id, hsn_code: item.hsn_code, season: item.season, merchant_code: item.merchant_code, selling_price: String(item.selling_price), purchase_price: String(item.purchase_price), launch_date: item.launch_date, uom: item.uom || 'PCS' }); setShowEditItem(true); setEditItemErr('') }}
                                  className="text-blue-500 hover:text-blue-700 text-xs px-2 py-1 rounded hover:bg-blue-50 transition-colors">
                                  Edit
                                </button>
                                <button
                                  onClick={e => { e.stopPropagation(); if (confirm(`Delete ${item.item_code}?`)) deleteItemMut.mutate(item.id) }}
                                  className="text-red-400 hover:text-red-600 text-xs px-2 py-1 rounded hover:bg-red-50 transition-colors">
                                  Delete
                                </button>
                              </div>
                            </td>
                          </tr>
                          {expandedId === item.id && expandedDetail && expandedDetail.id === item.id && (
                            <tr key={`exp-${item.id}`} className="bg-blue-50/40">
                              <td colSpan={9} className="px-6 py-3">
                                <div className="flex flex-wrap gap-5 text-xs text-gray-600">
                                  {expandedDetail.variants.length > 0 && (
                                    <div>
                                      <span className="font-semibold text-gray-700 block mb-1">Size Variants</span>
                                      <div className="flex flex-wrap gap-1">
                                        {expandedDetail.variants.map(v => (
                                          <span key={v.id} className="bg-white border border-gray-200 rounded px-2 py-0.5 font-mono">{v.size_label}</span>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  {expandedDetail.routing.length > 0 && (
                                    <div>
                                      <span className="font-semibold text-gray-700 block mb-1">Routing</span>
                                      <div className="flex flex-wrap gap-1">
                                        {expandedDetail.routing.map((r, i) => (
                                          <span key={r.id} className="bg-white border border-gray-200 rounded px-2 py-0.5">
                                            {i + 1}. {r.name}
                                          </span>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  {expandedDetail.merchant_code && (
                                    <div>
                                      <span className="font-semibold text-gray-700 block mb-1">Merchant</span>
                                      <span className="bg-white border border-gray-200 rounded px-2 py-0.5">{expandedDetail.merchant_code}</span>
                                    </div>
                                  )}
                                  {expandedBoms.length > 0 && (
                                    <div>
                                      <span className="font-semibold text-gray-700 block mb-1">BOMs</span>
                                      <div className="flex flex-wrap gap-1">
                                        {expandedBoms.map(b => (
                                          <button
                                            key={b.id}
                                            onClick={() => { setActiveTab('bom'); setBomItemId(item.id); setBomItemName(item.item_code + ' — ' + item.item_name); setSelectedBomId(b.id) }}
                                            className="bg-white border border-blue-200 rounded px-2 py-0.5 text-blue-700 hover:bg-blue-50 transition-colors flex items-center gap-1">
                                            {b.is_certified ? '🔒' : '📋'} {b.bom_name}
                                            <span className="text-gray-400">({b.line_count} lines)</span>
                                            {b.applies_to !== 'all' && <span className="text-gray-400 text-[10px]">· {b.applies_to}</span>}
                                          </button>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </td>
                            </tr>
                          )}
                        </>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* New Item Slide-over */}
              {showNewItem && (
                <div className="fixed inset-0 bg-black/40 z-50 flex justify-end" onClick={() => setShowNewItem(false)}>
                  <div className="bg-white w-full max-w-lg h-full overflow-y-auto shadow-xl p-6 space-y-5" onClick={e => e.stopPropagation()}>
                    <div className="flex items-center justify-between">
                      <h2 className="text-lg font-bold text-gray-900">New Item</h2>
                      <button onClick={() => setShowNewItem(false)} className="text-gray-400 hover:text-gray-600 text-xl">✕</button>
                    </div>

                    {/* Auto-generate code toggle */}
                    <div className="flex items-center gap-3 bg-gray-50 border border-gray-200 rounded-xl px-4 py-2.5">
                      <span className="text-xs font-medium text-gray-600 uppercase tracking-wide">Item Code</span>
                      <div className="flex gap-1 ml-auto">
                        {(['manual', 'auto'] as const).map(m => (
                          <button key={m} type="button"
                            onClick={() => {
                              setCodeMode(m)
                              if (m === 'auto') setNewItem(p => ({ ...p, item_code: 'YG-' + Math.random().toString(36).slice(2, 8).toUpperCase() }))
                            }}
                            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${codeMode === m ? 'bg-[#002B5B] text-white' : 'bg-white text-gray-600 border border-gray-200 hover:border-[#002B5B]'}`}>
                            {m === 'manual' ? 'Manual' : 'Auto Generate'}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Basic fields */}
                    <div className="grid grid-cols-2 gap-3">
                      {[
                        ['item_code',      'Item Code *',    'text'],
                        ['item_name',      'Item Name *',    'text'],
                        ['hsn_code',       'HSN Code',       'text'],
                        ['season',         'Season',         'text'],
                        ['launch_date',    'Launch Date',    'date'],
                        ['selling_price',  'Selling Price ₹','number'],
                        ['purchase_price', 'Cost Price ₹',  'number'],
                      ].map(([key, label, type]) => (
                        <div key={key} className="space-y-1">
                          <label className="text-xs font-medium text-gray-600 uppercase tracking-wide">{label}</label>
                          <input type={type} value={(newItem as any)[key]}
                            readOnly={key === 'item_code' && codeMode === 'auto'}
                            onChange={e => setNewItem(p => ({ ...p, [key]: e.target.value }))}
                            className={`w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B] ${key === 'item_code' && codeMode === 'auto' ? 'bg-gray-50 text-gray-500 font-mono' : ''}`} />
                        </div>
                      ))}
                      {/* Item Type */}
                      <div className="space-y-1">
                        <label className="text-xs font-medium text-gray-600 uppercase tracking-wide">Item Type *</label>
                        <select value={newItem.item_type_id}
                          onChange={e => setNewItem(p => ({ ...p, item_type_id: +e.target.value }))}
                          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                          {itemTypes.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
                        </select>
                      </div>
                      {/* Merchant dropdown */}
                      <div className="space-y-1">
                        <label className="text-xs font-medium text-gray-600 uppercase tracking-wide">Merchant</label>
                        <select value={newItem.merchant_code}
                          onChange={e => setNewItem(p => ({ ...p, merchant_code: e.target.value }))}
                          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                          <option value="">— None —</option>
                          {merchants.map(m => <option key={m.id} value={m.merchant_code}>{m.merchant_code} — {m.merchant_name}</option>)}
                        </select>
                        {merchants.length === 0 && (
                          <p className="text-[11px] text-gray-400">Add merchants in the Merchants tab first.</p>
                        )}
                      </div>
                      {/* UOM — mandatory */}
                      <div className="space-y-1">
                        <label className="text-xs font-medium text-gray-600 uppercase tracking-wide">UOM <span className="text-red-500">*</span></label>
                        <select value={newItem.uom}
                          onChange={e => setNewItem(p => ({ ...p, uom: e.target.value }))}
                          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                          {UOM_OPTIONS.map(u => <option key={u}>{u}</option>)}
                        </select>
                      </div>
                    </div>

                    {/* Size selection */}
                    <div className="space-y-2">
                      <label className="text-xs font-medium text-gray-600 uppercase tracking-wide">Size Variants</label>
                      <select value={newItem.size_group_id}
                        onChange={e => applyPreviewSizes(e.target.value)}
                        className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                        <option value="">Pick a size group…</option>
                        {sizeGroups.map(g => <option key={g.id} value={g.id}>{g.name} ({g.sizes.join(', ')})</option>)}
                      </select>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {(sizeGroups.find(g => g.id === +newItem.size_group_id)?.sizes ?? []).map(sz => (
                          <button key={sz} type="button"
                            onClick={() => toggleSize(sz)}
                            className={`px-2 py-0.5 rounded border text-xs font-medium transition-colors ${
                              newItem.sizes.includes(sz)
                                ? 'bg-[#002B5B] text-white border-[#002B5B]'
                                : 'bg-white text-gray-600 border-gray-200 hover:border-[#002B5B]'
                            }`}>{sz}</button>
                        ))}
                      </div>
                      <div className="flex gap-2">
                        <input type="text" placeholder="Custom size (e.g. 3XL)" value={newItem.custom_size}
                          onChange={e => setNewItem(p => ({ ...p, custom_size: e.target.value }))}
                          onKeyDown={e => e.key === 'Enter' && addCustomSize()}
                          className="flex-1 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                        <button type="button" onClick={addCustomSize}
                          className="px-3 py-2 bg-gray-100 rounded-lg text-sm hover:bg-gray-200 transition-colors">Add</button>
                      </div>
                      {sizePreview.length > 0 && (
                        <div className="bg-gray-50 rounded-lg p-2 text-xs text-gray-600">
                          <span className="font-medium">Will create: </span>
                          {sizePreview.map(s => `${newItem.item_code}-${s}`).join(', ')}
                        </div>
                      )}
                    </div>

                    {/* Ordered Routing */}
                    <div className="space-y-2">
                      <label className="text-xs font-medium text-gray-600 uppercase tracking-wide">Production Routing (in order)</label>
                      {/* Available processes */}
                      <div className="flex flex-wrap gap-1">
                        {routingSteps
                          .filter(r => !newItem.routing_step_ids.includes(r.id))
                          .map(r => (
                            <button key={r.id} type="button" onClick={() => toggleRoutingStep(r.id)}
                              className="px-2 py-0.5 rounded border text-xs font-medium bg-white text-gray-600 border-gray-200 hover:border-[#002B5B] hover:text-[#002B5B] transition-colors">
                              + {r.name}
                            </button>
                          ))}
                      </div>
                      {/* Selected in order */}
                      {newItem.routing_step_ids.length > 0 && (
                        <div className="space-y-1 mt-1">
                          {newItem.routing_step_ids.map((stepId, idx) => {
                            const step = routingSteps.find(r => r.id === stepId)
                            return step ? (
                              <div key={stepId} className="flex items-center gap-2 bg-[#002B5B]/5 border border-[#002B5B]/20 rounded-lg px-3 py-1.5">
                                <span className="text-xs font-bold text-[#002B5B] w-4">{idx + 1}.</span>
                                <span className="text-xs font-medium text-gray-800 flex-1">{step.name}</span>
                                <div className="flex gap-1">
                                  <button type="button" onClick={() => moveRoutingStep(idx, -1)} disabled={idx === 0}
                                    className="text-gray-400 hover:text-gray-600 disabled:opacity-30 text-xs px-1">↑</button>
                                  <button type="button" onClick={() => moveRoutingStep(idx, 1)} disabled={idx === newItem.routing_step_ids.length - 1}
                                    className="text-gray-400 hover:text-gray-600 disabled:opacity-30 text-xs px-1">↓</button>
                                  <button type="button" onClick={() => toggleRoutingStep(stepId)}
                                    className="text-red-400 hover:text-red-600 text-xs px-1 ml-1">✕</button>
                                </div>
                              </div>
                            ) : null
                          })}
                        </div>
                      )}
                    </div>

                    {newItemErr && <p className="text-red-500 text-sm">{newItemErr}</p>}
                    <button onClick={handleCreateItem}
                      disabled={createItemMut.isPending}
                      className="w-full bg-[#002B5B] text-white py-3 rounded-xl font-medium hover:bg-[#003d80] disabled:opacity-50 transition-colors">
                      {createItemMut.isPending ? 'Creating…' : 'Create Item'}
                    </button>
                  </div>
                </div>
              )}

              {/* Edit Item Slide-over */}
              {showEditItem && editItem && (
                <div className="fixed inset-0 bg-black/40 z-50 flex justify-end" onClick={() => setShowEditItem(false)}>
                  <div className="bg-white w-full max-w-lg h-full overflow-y-auto shadow-xl p-6 space-y-5" onClick={e => e.stopPropagation()}>
                    <div className="flex items-center justify-between">
                      <div>
                        <h2 className="text-lg font-bold text-gray-900">Edit Item</h2>
                        <p className="text-xs font-mono text-[#002B5B]">{editItem.item_code}</p>
                      </div>
                      <button onClick={() => setShowEditItem(false)} className="text-gray-400 hover:text-gray-600 text-xl">✕</button>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      {([
                        ['item_name',      'Item Name *',    'text'],
                        ['hsn_code',       'HSN Code',       'text'],
                        ['season',         'Season',         'text'],
                        ['launch_date',    'Launch Date',    'date'],
                        ['selling_price',  'Selling Price ₹','number'],
                        ['purchase_price', 'Cost Price ₹',  'number'],
                      ] as [string, string, string][]).map(([key, label, type]) => (
                        <div key={key} className="space-y-1">
                          <label className="text-xs font-medium text-gray-600 uppercase tracking-wide">{label}</label>
                          <input type={type} value={(editItem as any)[key]}
                            onChange={e => setEditItem(p => p ? { ...p, [key]: e.target.value } : p)}
                            className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                        </div>
                      ))}
                      <div className="space-y-1">
                        <label className="text-xs font-medium text-gray-600 uppercase tracking-wide">Item Type *</label>
                        <select value={editItem.item_type_id}
                          onChange={e => setEditItem(p => p ? { ...p, item_type_id: +e.target.value } : p)}
                          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                          {itemTypes.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
                        </select>
                      </div>
                      <div className="space-y-1">
                        <label className="text-xs font-medium text-gray-600 uppercase tracking-wide">Merchant</label>
                        <select value={editItem.merchant_code}
                          onChange={e => setEditItem(p => p ? { ...p, merchant_code: e.target.value } : p)}
                          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                          <option value="">— None —</option>
                          {merchants.map(m => <option key={m.id} value={m.merchant_code}>{m.merchant_code} — {m.merchant_name}</option>)}
                        </select>
                      </div>
                      <div className="space-y-1">
                        <label className="text-xs font-medium text-gray-600 uppercase tracking-wide">UOM <span className="text-red-500">*</span></label>
                        <select value={editItem.uom}
                          onChange={e => setEditItem(p => p ? { ...p, uom: e.target.value } : p)}
                          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                          {UOM_OPTIONS.map(u => <option key={u}>{u}</option>)}
                        </select>
                      </div>
                    </div>
                    {editItemErr && <p className="text-red-500 text-sm">{editItemErr}</p>}
                    <button
                      onClick={() => updateItemMut.mutate({
                        id: editItem.id,
                        body: {
                          item_name:      editItem.item_name,
                          item_type_id:   editItem.item_type_id,
                          hsn_code:       editItem.hsn_code,
                          season:         editItem.season,
                          merchant_code:  editItem.merchant_code,
                          selling_price:  parseFloat(editItem.selling_price) || 0,
                          purchase_price: parseFloat(editItem.purchase_price) || 0,
                          launch_date:    editItem.launch_date,
                          uom:            editItem.uom,
                        }
                      })}
                      disabled={updateItemMut.isPending}
                      className="w-full bg-[#002B5B] text-white py-3 rounded-xl font-medium hover:bg-[#003d80] disabled:opacity-50 transition-colors">
                      {updateItemMut.isPending ? 'Saving…' : 'Save Changes'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ================================================================
              TAB 2 — BOM BUILDER
              ================================================================ */}
          {activeTab === 'bom' && (
            <div className="space-y-4">
              {/* BOM sub-tabs */}
              <div className="flex gap-1 border-b border-gray-200 pb-0">
                {([['builder', '🧩 BOM Builder'], ['list', '📋 All BOMs']] as const).map(([id, label]) => (
                  <button key={id} onClick={() => setBomSubTab(id)}
                    className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${bomSubTab === id ? 'border-b-2 border-[#002B5B] text-[#002B5B] bg-blue-50' : 'text-gray-500 hover:text-gray-700'}`}>
                    {label}
                  </button>
                ))}
              </div>

            {bomSubTab === 'list' ? (
              /* ── ALL BOMs LIST ─────────────────────────────────────── */
              <div className="overflow-x-auto rounded-xl border border-gray-200">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      {['Item Code', 'Item Name', 'BOM Name', 'Applies To', 'Lines', 'Mat Cost', 'CMT', 'Total', 'Status', ''].map(h => (
                        <th key={h} className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {allBoms.length === 0 ? (
                      <tr><td colSpan={10} className="px-4 py-8 text-center text-gray-400 text-sm">No BOMs created yet.</td></tr>
                    ) : allBoms.map(b => (
                      <tr key={b.id} className="hover:bg-gray-50 transition-colors">
                        <td className="px-3 py-2 font-mono font-medium text-[#002B5B] text-xs">{b.item_code}</td>
                        <td className="px-3 py-2 text-gray-700 text-xs max-w-[160px] truncate">{b.item_name}</td>
                        <td className="px-3 py-2 font-medium text-gray-800 text-xs">{b.bom_name}</td>
                        <td className="px-3 py-2 text-xs text-gray-500">{b.applies_to === 'all' ? 'All sizes' : b.applies_to}</td>
                        <td className="px-3 py-2 text-xs text-gray-600">{b.line_count}</td>
                        <td className="px-3 py-2 text-xs tabular-nums">{fmt(b.lines_total)}</td>
                        <td className="px-3 py-2 text-xs tabular-nums text-amber-700">{(b.cmt_cost || b.other_cost) ? fmt((b.cmt_cost ?? 0) + (b.other_cost ?? 0)) : '—'}</td>
                        <td className="px-3 py-2 text-xs tabular-nums font-semibold text-[#002B5B]">{fmt(b.grand_total)}</td>
                        <td className="px-3 py-2">
                          {b.is_certified
                            ? <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-medium">🔒 Certified</span>
                            : <span className="text-xs bg-yellow-50 text-yellow-700 px-2 py-0.5 rounded-full">Draft</span>}
                        </td>
                        <td className="px-3 py-2">
                          <button
                            onClick={() => { setBomSubTab('builder'); setBomItemId(b.item_id); setBomItemName(b.item_code + ' — ' + b.item_name); setSelectedBomId(b.id) }}
                            className="text-[#002B5B] hover:text-[#003d80] text-xs px-2 py-1 rounded hover:bg-blue-50 transition-colors">
                            Open
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
            <div className="flex gap-5 min-h-[500px]">
              {/* Left: Item selector */}
              <div className="w-64 flex-shrink-0 space-y-3">
                <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Select Item</div>
                <input type="text" placeholder="Search item code / name…"
                  value={bomItemSearch} onChange={e => setBomItemSearch(e.target.value)}
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                {bomItemSearch.length >= 2 && bomSearchResults.length > 0 && (
                  <div className="border border-gray-200 rounded-lg divide-y divide-gray-100 max-h-60 overflow-y-auto bg-white shadow-sm">
                    {bomSearchResults.map(it => (
                      <button key={it.id} onClick={() => {
                        setBomItemId(it.id); setBomItemName(it.item_code + ' — ' + it.item_name)
                        setBomItemSearch(''); setSelectedBomId(null); setCertifyErr('')
                      }}
                        className="w-full text-left px-3 py-2 hover:bg-blue-50 transition-colors">
                        <div className="text-xs font-mono font-medium text-[#002B5B]">{it.item_code}</div>
                        <div className="text-xs text-gray-500">{it.item_name}</div>
                      </button>
                    ))}
                  </div>
                )}
                {bomItemId && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-2 text-xs font-medium text-blue-800">
                    {bomItemName}
                  </div>
                )}
              </div>

              {/* Right: BOM panel */}
              <div className="flex-1 space-y-4">
                {!bomItemId ? (
                  <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
                    Search and select an item to view/edit its BOM
                  </div>
                ) : (
                  <>
                    {/* BOM list */}
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide mr-1">BOMs:</span>
                      {boms.map(b => (
                        <button key={b.id} onClick={() => { setSelectedBomId(b.id); setCertifyErr('') }}
                          className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors flex items-center gap-1 ${
                            selectedBomId === b.id
                              ? 'bg-[#002B5B] text-white border-[#002B5B]'
                              : 'bg-white text-gray-700 border-gray-200 hover:border-[#002B5B]'
                          }`}>
                          {b.is_certified ? '🔒' : ''}{b.bom_name}
                          {b.applies_to !== 'all' && <span className="ml-1 opacity-70">({b.applies_to})</span>}
                          <span className="ml-1 opacity-60">[{b.line_count}]</span>
                        </button>
                      ))}
                      <button onClick={() => setShowNewBOM(true)}
                        className="px-3 py-1 rounded-full text-xs font-medium border border-dashed border-gray-300 text-gray-500 hover:border-[#002B5B] hover:text-[#002B5B] transition-colors">
                        + Add BOM
                      </button>
                    </div>

                    {/* New BOM form */}
                    {showNewBOM && (
                      <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-3">
                        <div className="flex gap-3 flex-wrap">
                          <div className="flex-1 space-y-1">
                            <label className="text-xs font-medium text-gray-600">BOM Name</label>
                            <input type="text" placeholder='e.g. BOM-1 (56" Fabric)' value={newBOMName}
                              onChange={e => setNewBOMName(e.target.value)}
                              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                          </div>
                          <div className="w-48 space-y-1">
                            <label className="text-xs font-medium text-gray-600">Applies To</label>
                            <input type="text" placeholder="all   or   XL,XXL,3XL" value={newBOMApply}
                              onChange={e => setNewBOMApply(e.target.value)}
                              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <button onClick={() => createBOMMut.mutate({ bom_name: newBOMName || 'Default', applies_to: newBOMApply || 'all' })}
                            className="bg-[#002B5B] text-white text-xs px-4 py-2 rounded-lg hover:bg-[#003d80] transition-colors">
                            Create BOM
                          </button>
                          <button onClick={() => setShowNewBOM(false)}
                            className="text-gray-500 text-xs px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors">
                            Cancel
                          </button>
                        </div>
                      </div>
                    )}

                    {/* BOM Detail */}
                    {selectedBomId && bomDetail && (
                      <div className="space-y-4">
                        {/* BOM header bar */}
                        <div className="flex items-center justify-between flex-wrap gap-2">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-sm font-semibold text-gray-800">{bomDetail.bom_name}</span>
                            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
                              {bomDetail.applies_to === 'all' ? 'All sizes' : `Sizes: ${bomDetail.applies_to}`}
                            </span>
                            {bomDetail.is_certified ? (
                              <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-medium flex items-center gap-1">
                                🔒 Certified {bomDetail.certified_by && `by ${bomDetail.certified_by}`}
                              </span>
                            ) : (
                              <span className="text-xs bg-yellow-50 text-yellow-700 px-2 py-0.5 rounded-full">Draft</span>
                            )}
                          </div>
                          <div className="flex gap-2 items-center">
                            {bomDetail.is_certified ? (
                              <button
                                onClick={() => { if (confirm('Remove certification? This will allow editing.')) uncertifyBOMMut.mutate(selectedBomId) }}
                                disabled={uncertifyBOMMut.isPending}
                                className="text-xs text-orange-600 hover:text-orange-800 px-2 py-1 rounded hover:bg-orange-50 transition-colors border border-orange-200">
                                🔓 Uncertify
                              </button>
                            ) : (
                              <button
                                onClick={() => { if (confirm('Certify this BOM? Regular users will not be able to edit it.')) certifyBOMMut.mutate(selectedBomId) }}
                                disabled={certifyBOMMut.isPending}
                                className="text-xs text-green-600 hover:text-green-800 px-2 py-1 rounded hover:bg-green-50 transition-colors border border-green-200">
                                ✓ Certify BOM
                              </button>
                            )}
                            <button onClick={() => setShowCopyBOM(true)}
                              className="text-xs text-blue-600 hover:text-blue-800 px-2 py-1 rounded hover:bg-blue-50 transition-colors">
                              Copy BOM
                            </button>
                            {!bomDetail.is_certified && (
                              <button onClick={() => { if (confirm('Delete this BOM?')) deleteBOMMut.mutate(selectedBomId) }}
                                className="text-xs text-red-500 hover:text-red-700 px-2 py-1 rounded hover:bg-red-50 transition-colors">
                                Delete BOM
                              </button>
                            )}
                          </div>
                        </div>

                        {certifyErr && (
                          <p className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">{certifyErr}</p>
                        )}

                        {/* Copy BOM modal */}
                        {showCopyBOM && (
                          <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-3">
                            <p className="text-sm font-medium text-gray-700">Copy BOM to another item</p>
                            <div className="space-y-2">
                              <input type="text" placeholder="Search target item…" value={copyTargetSearch}
                                onChange={e => { setCopyTargetSearch(e.target.value); setCopyTargetId(null) }}
                                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                              {copyTargetSearch.length >= 2 && copyTargetResults.length > 0 && !copyTargetId && (
                                <div className="border border-gray-200 rounded-lg divide-y divide-gray-100 max-h-40 overflow-y-auto bg-white shadow-sm">
                                  {copyTargetResults.map(it => (
                                    <button key={it.id} onClick={() => { setCopyTargetId(it.id); setCopyTargetName(it.item_code + ' — ' + it.item_name); setCopyTargetSearch('') }}
                                      className="w-full text-left px-3 py-2 hover:bg-blue-50 text-xs transition-colors">
                                      <span className="font-mono font-medium text-[#002B5B]">{it.item_code}</span> — {it.item_name}
                                    </button>
                                  ))}
                                </div>
                              )}
                              {copyTargetId && <div className="text-xs text-blue-800 bg-blue-50 border border-blue-200 rounded px-2 py-1">{copyTargetName}</div>}
                              <input type="text" placeholder="New BOM name" value={copyName}
                                onChange={e => setCopyName(e.target.value)}
                                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                            </div>
                            <div className="flex gap-2">
                              <button
                                onClick={() => copyTargetId && copyBOMMut.mutate({ target_item_id: copyTargetId, new_name: copyName || 'Copied BOM' })}
                                disabled={!copyTargetId}
                                className="bg-[#002B5B] text-white text-xs px-4 py-2 rounded-lg hover:bg-[#003d80] disabled:opacity-50 transition-colors">Copy</button>
                              <button onClick={() => setShowCopyBOM(false)}
                                className="text-gray-500 text-xs px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors">Cancel</button>
                            </div>
                          </div>
                        )}

                        {/* Material / Component Lines table */}
                        <div>
                          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">🧵 Material / Component Lines</p>
                        </div>
                        <div className="overflow-x-auto rounded-xl border border-gray-200">
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="bg-gray-50 border-b border-gray-200">
                                {['Component', 'Type', 'Qty', 'Unit', 'Rate ₹', 'Process', 'Shrink%', 'Waste%', 'Net Qty', 'Amount ₹', 'Remarks', ''].map(h => (
                                  <th key={h} className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide whitespace-nowrap">{h}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-100">
                              {matLines.map(l => (
                                <tr key={l.id} className="hover:bg-gray-50 transition-colors">
                                  <td className="px-3 py-2 font-medium text-gray-800">{l.component_name}</td>
                                  <td className="px-3 py-2">
                                    <span className="bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded text-xs">{l.component_type}</span>
                                  </td>
                                  <td className="px-3 py-2 tabular-nums">{l.quantity}</td>
                                  <td className="px-3 py-2 text-gray-500">{l.unit}</td>
                                  <td className="px-3 py-2 tabular-nums">{l.rate.toLocaleString('en-IN')}</td>
                                  <td className="px-3 py-2 text-gray-500">{l.process_name || '—'}</td>
                                  <td className="px-3 py-2 tabular-nums text-orange-600">{l.shrinkage_pct > 0 ? l.shrinkage_pct + '%' : '—'}</td>
                                  <td className="px-3 py-2 tabular-nums text-orange-600">{l.wastage_pct > 0 ? l.wastage_pct + '%' : '—'}</td>
                                  <td className="px-3 py-2 tabular-nums text-gray-700">{fmtN(netQty(l))}</td>
                                  <td className="px-3 py-2 tabular-nums font-medium text-gray-900">{lineAmt(l).toLocaleString('en-IN', { maximumFractionDigits: 0 })}</td>
                                  <td className="px-3 py-2 text-gray-400 max-w-[120px] truncate">{l.remarks || '—'}</td>
                                  <td className="px-3 py-2">
                                    {!bomDetail.is_certified && (
                                      <button onClick={() => deleteLineMut.mutate({ bomId: selectedBomId, lineId: l.id })}
                                        className="text-red-400 hover:text-red-600 transition-colors">✕</button>
                                    )}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {matLines.length === 0 && (
                            <div className="text-center py-8 text-gray-400 text-sm">No material lines yet. Add components below.</div>
                          )}
                        </div>

                        {/* Add Line form — hidden when certified */}
                        {!bomDetail.is_certified && (
                          !showAddLine ? (
                            <button onClick={() => { setShowAddLine(true); setCertifyErr('') }}
                              className="text-sm text-[#002B5B] hover:text-[#003d80] font-medium border border-dashed border-[#002B5B]/30 rounded-lg px-4 py-2 hover:bg-blue-50 transition-colors w-full">
                              + Add Component
                            </button>
                          ) : (
                            <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-3">
                              <p className="text-xs font-semibold text-gray-700 uppercase tracking-wide">Add Component (from Item Master)</p>

                              {/* Component item search */}
                              <div className="space-y-1">
                                <label className="text-xs text-gray-500">Component Item *</label>
                                {newLine.component_item_id ? (
                                  <div className="flex items-center gap-2 bg-blue-50 border border-blue-200 rounded-lg px-3 py-2">
                                    <span className="text-xs font-medium text-blue-800 flex-1">{newLine.component_name}</span>
                                    <button onClick={() => setNewLine(p => ({ ...p, component_item_id: null, component_name: '' }))}
                                      className="text-gray-400 hover:text-red-500 text-xs">✕</button>
                                  </div>
                                ) : (
                                  <div className="relative">
                                    <input type="text" placeholder="Type item code or name to search…" value={compSearch}
                                      onChange={e => setCompSearch(e.target.value)}
                                      className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                                    {compSearch.length >= 2 && compSearchResults.length > 0 && (
                                      <div className="absolute z-10 top-full mt-1 w-full border border-gray-200 rounded-lg divide-y divide-gray-100 max-h-48 overflow-y-auto bg-white shadow-lg">
                                        {compSearchResults.map(it => (
                                          <button key={it.id} onClick={() => {
                                            setNewLine(p => ({
                                              ...p,
                                              component_item_id: it.id,
                                              component_name: it.item_code + ' — ' + it.item_name,
                                              component_type: it.item_type_code,
                                            }))
                                            setCompSearch('')
                                          }}
                                            className="w-full text-left px-3 py-2 hover:bg-blue-50 transition-colors">
                                            <div className="text-xs font-mono font-medium text-[#002B5B]">{it.item_code}</div>
                                            <div className="text-xs text-gray-500">{it.item_name} · {it.item_type_code}</div>
                                          </button>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>

                              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                                <div className="space-y-1">
                                  <label className="text-xs text-gray-500">Type</label>
                                  <select value={newLine.component_type}
                                    onChange={e => setNewLine(p => ({ ...p, component_type: e.target.value }))}
                                    className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                                    {COMPONENT_TYPES.map(t => <option key={t}>{t}</option>)}
                                  </select>
                                </div>
                                <div className="space-y-1">
                                  <label className="text-xs text-gray-500">Unit</label>
                                  <select value={newLine.unit}
                                    onChange={e => setNewLine(p => ({ ...p, unit: e.target.value }))}
                                    className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                                    {UNITS.map(u => <option key={u}>{u}</option>)}
                                  </select>
                                </div>
                                {[
                                  ['quantity',      'Quantity',  'number'],
                                  ['rate',          'Rate ₹',    'number'],
                                  ['shrinkage_pct', 'Shrink %',  'number'],
                                  ['wastage_pct',   'Wastage %', 'number'],
                                ].map(([key, label, type]) => (
                                  <div key={key} className="space-y-1">
                                    <label className="text-xs text-gray-500">{label}</label>
                                    <input type={type} value={(newLine as any)[key]}
                                      onChange={e => setNewLine(p => ({ ...p, [key]: e.target.value }))}
                                      className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                                  </div>
                                ))}
                                <div className="space-y-1">
                                  <label className="text-xs text-gray-500">Process</label>
                                  <select value={newLine.process_id}
                                    onChange={e => setNewLine(p => ({ ...p, process_id: e.target.value }))}
                                    className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                                    <option value="">— None —</option>
                                    {routingSteps.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
                                  </select>
                                </div>
                                <div className="col-span-2 space-y-1">
                                  <label className="text-xs text-gray-500">Remarks</label>
                                  <input type="text" value={newLine.remarks}
                                    onChange={e => setNewLine(p => ({ ...p, remarks: e.target.value }))}
                                    className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                                </div>
                              </div>
                              <div className="flex gap-2">
                                <button onClick={handleAddLine} disabled={addLineMut.isPending}
                                  className="bg-[#002B5B] text-white text-xs px-4 py-2 rounded-lg hover:bg-[#003d80] disabled:opacity-50 transition-colors">
                                  {addLineMut.isPending ? 'Adding…' : 'Add Component'}
                                </button>
                                <button onClick={() => { setShowAddLine(false); setNewLine(blankBOMLine()); setCompSearch(''); setCertifyErr('') }}
                                  className="text-gray-500 text-xs px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors">Cancel</button>
                              </div>
                            </div>
                          )
                        )}

                        {/* ── Process Cost Lines ── */}
                        {!bomDetail.is_certified && (
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <p className="text-xs font-semibold text-gray-700 uppercase tracking-wide">⚙️ Process Cost Lines</p>
                              <div className="flex gap-2">
                                {bomItemDetail?.routing && bomItemDetail.routing.length > 0 && svcLines.length === 0 && (
                                  <button
                                    onClick={() => {
                                      bomItemDetail.routing.forEach(r => {
                                        addLineMut.mutate({ component_name: r.name, component_type: 'SVC', quantity: 1, unit: 'PCS', rate: 0, process_id: r.id, shrinkage_pct: 0, wastage_pct: 0, remarks: '' })
                                      })
                                    }}
                                    className="text-xs text-blue-600 hover:text-blue-800 px-2 py-1 rounded hover:bg-blue-50 border border-blue-200 transition-colors">
                                    Auto-load from Routing
                                  </button>
                                )}
                                <button onClick={() => setShowAddSvc(v => !v)}
                                  className="text-xs text-[#002B5B] hover:text-[#003d80] px-2 py-1 rounded hover:bg-blue-50 border border-dashed border-[#002B5B]/30 transition-colors">
                                  {showAddSvc ? 'Cancel' : '+ Add Process'}
                                </button>
                              </div>
                            </div>

                            {svcLines.length > 0 && (
                              <div className="overflow-x-auto rounded-xl border border-gray-200">
                                <table className="w-full text-xs">
                                  <thead>
                                    <tr className="bg-gray-50 border-b border-gray-200">
                                      {['Process', 'Qty', 'Unit', 'Rate ₹', 'Amount ₹', 'Remarks', ''].map(h => (
                                        <th key={h} className="px-3 py-2 text-left font-semibold text-gray-500 uppercase tracking-wide">{h}</th>
                                      ))}
                                    </tr>
                                  </thead>
                                  <tbody className="divide-y divide-gray-100">
                                    {svcLines.map(l => (
                                      <tr key={l.id} className="hover:bg-gray-50">
                                        <td className="px-3 py-2 font-medium text-gray-800">{l.component_name}</td>
                                        <td className="px-3 py-2 tabular-nums">{l.quantity}</td>
                                        <td className="px-3 py-2 text-gray-500">{l.unit}</td>
                                        <td className="px-3 py-2 tabular-nums">{l.rate}</td>
                                        <td className="px-3 py-2 tabular-nums font-medium">{fmt(lineAmt(l))}</td>
                                        <td className="px-3 py-2 text-gray-400">{l.remarks || '—'}</td>
                                        <td className="px-3 py-2">
                                          <button onClick={() => deleteLineMut.mutate({ bomId: selectedBomId!, lineId: l.id })}
                                            className="text-red-400 hover:text-red-600">✕</button>
                                        </td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            )}
                            {svcLines.length === 0 && (
                              <div className="text-xs text-gray-400 italic py-2 px-3 bg-gray-50 rounded-lg border border-dashed border-gray-200">
                                No process cost lines. Auto-load from routing or add manually.
                              </div>
                            )}

                            {showAddSvc && (
                              <div className="bg-gray-50 border border-gray-200 rounded-xl p-3 space-y-2">
                                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                                  <div className="space-y-1">
                                    <label className="text-xs text-gray-500">Process *</label>
                                    <select value={newSvcLine.process_id}
                                      onChange={e => {
                                        const r = routingSteps.find(s => s.id === +e.target.value)
                                        setNewSvcLine(p => ({ ...p, process_id: e.target.value, name: r?.name ?? '' }))
                                      }}
                                      className="w-full border border-gray-200 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                                      <option value="">— Select —</option>
                                      {routingSteps.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
                                    </select>
                                  </div>
                                  <div className="space-y-1">
                                    <label className="text-xs text-gray-500">Qty</label>
                                    <input type="number" value={newSvcLine.quantity} onChange={e => setNewSvcLine(p => ({ ...p, quantity: e.target.value }))}
                                      className="w-full border border-gray-200 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                                  </div>
                                  <div className="space-y-1">
                                    <label className="text-xs text-gray-500">Rate ₹</label>
                                    <input type="number" value={newSvcLine.rate} onChange={e => setNewSvcLine(p => ({ ...p, rate: e.target.value }))}
                                      className="w-full border border-gray-200 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                                  </div>
                                  <div className="col-span-2 space-y-1">
                                    <label className="text-xs text-gray-500">Remarks</label>
                                    <input type="text" value={newSvcLine.remarks} onChange={e => setNewSvcLine(p => ({ ...p, remarks: e.target.value }))}
                                      className="w-full border border-gray-200 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                                  </div>
                                </div>
                                <button
                                  onClick={() => {
                                    if (!newSvcLine.process_id) return
                                    addLineMut.mutate({ component_name: newSvcLine.name || 'Process', component_type: 'SVC', quantity: parseFloat(newSvcLine.quantity) || 1, unit: 'PCS', rate: parseFloat(newSvcLine.rate) || 0, process_id: +newSvcLine.process_id, shrinkage_pct: 0, wastage_pct: 0, remarks: newSvcLine.remarks })
                                    setNewSvcLine({ process_id: '', name: '', quantity: '1', unit: 'PCS', rate: '0', remarks: '' })
                                    setShowAddSvc(false)
                                  }}
                                  className="bg-[#002B5B] text-white text-xs px-3 py-1.5 rounded-lg hover:bg-[#003d80] transition-colors">
                                  Add Process Line
                                </button>
                              </div>
                            )}
                          </div>
                        )}

                        {/* ── CMT + Other + Grand Total ── */}
                        <div className="bg-gradient-to-r from-[#002B5B]/5 to-blue-50 border border-[#002B5B]/10 rounded-xl p-4 space-y-3">
                          <p className="text-xs font-semibold text-gray-700 uppercase tracking-wide">Cost Summary</p>
                          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 text-xs mb-1">
                            {Object.entries(costByType).filter(([t]) => t !== 'SVC').map(([type, amt]) => (
                              <div key={type} className="bg-white rounded-lg p-2 border border-gray-100">
                                <div className="text-gray-500">{type}</div>
                                <div className="font-semibold text-gray-900 mt-0.5">{fmt(amt)}</div>
                              </div>
                            ))}
                            {svcLines.length > 0 && (
                              <div className="bg-white rounded-lg p-2 border border-gray-100">
                                <div className="text-gray-500">Process (SVC)</div>
                                <div className="font-semibold text-gray-900 mt-0.5">{fmt(svcLines.reduce((s, l) => s + lineAmt(l), 0))}</div>
                              </div>
                            )}
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <div className="space-y-1">
                              <label className="text-xs text-gray-500">CMT Cost ₹</label>
                              <input type="number" value={bomCmtCost} onChange={e => setBomCmtCost(e.target.value)}
                                onBlur={() => updateBomCostsMut.mutate({ cmt_cost: parseFloat(bomCmtCost) || 0, other_cost: parseFloat(bomOtherCost) || 0 })}
                                disabled={!!bomDetail.is_certified}
                                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B] disabled:bg-gray-50" />
                            </div>
                            <div className="space-y-1">
                              <label className="text-xs text-gray-500">Other Cost ₹</label>
                              <input type="number" value={bomOtherCost} onChange={e => setBomOtherCost(e.target.value)}
                                onBlur={() => updateBomCostsMut.mutate({ cmt_cost: parseFloat(bomCmtCost) || 0, other_cost: parseFloat(bomOtherCost) || 0 })}
                                disabled={!!bomDetail.is_certified}
                                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B] disabled:bg-gray-50" />
                            </div>
                          </div>
                          <div className="flex items-center justify-between border-t border-[#002B5B]/10 pt-3">
                            <div className="text-xs text-gray-500 space-y-0.5">
                              <div>Material + Process: <span className="font-medium text-gray-700">{fmt(totalCost)}</span></div>
                              <div>CMT + Other: <span className="font-medium text-amber-700">{fmt((parseFloat(bomCmtCost) || 0) + (parseFloat(bomOtherCost) || 0))}</span></div>
                            </div>
                            <div className="text-right">
                              <div className="text-xs text-gray-500 uppercase tracking-wide">Grand Total</div>
                              <div className="text-xl font-extrabold text-[#002B5B]">{fmt(grandTotal)}</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
            )}
            </div>
          )}

          {/* ================================================================
              TAB 3 — ROUTING MASTER
              ================================================================ */}
          {activeTab === 'routing' && (
            <div className="space-y-4 max-w-2xl">
              <div className="overflow-x-auto rounded-xl border border-gray-200">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      {['#', 'Process Name', 'Description', 'Order', 'Action'].map(h => (
                        <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {routingSteps.map((r, i) => (
                      <tr key={r.id} className="hover:bg-gray-50 transition-colors">
                        <td className="px-4 py-3 text-gray-400 text-xs">{i + 1}</td>
                        <td className="px-4 py-3 font-medium text-gray-800">{r.name}</td>
                        <td className="px-4 py-3 text-gray-500">{r.description || '—'}</td>
                        <td className="px-4 py-3 text-gray-500">{r.sort_order}</td>
                        <td className="px-4 py-3">
                          <button onClick={() => { if (confirm(`Delete "${r.name}"?`)) deleteRouteMut.mutate(r.id) }}
                            className="text-red-400 hover:text-red-600 text-xs px-2 py-1 rounded hover:bg-red-50 transition-colors">
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Add routing step */}
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-3">
                <p className="text-xs font-semibold text-gray-700 uppercase tracking-wide">Add Process</p>
                <div className="flex gap-3 flex-wrap">
                  <input type="text" placeholder="Process name *" value={newRoute.name}
                    onChange={e => setNewRoute(p => ({ ...p, name: e.target.value }))}
                    className="flex-1 min-w-[150px] border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                  <input type="text" placeholder="Description" value={newRoute.description}
                    onChange={e => setNewRoute(p => ({ ...p, description: e.target.value }))}
                    className="flex-1 min-w-[200px] border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                  <input type="number" placeholder="Sort #" value={newRoute.sort_order}
                    onChange={e => setNewRoute(p => ({ ...p, sort_order: e.target.value }))}
                    className="w-20 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                  <button
                    onClick={() => createRouteMut.mutate({ name: newRoute.name, description: newRoute.description, sort_order: +newRoute.sort_order || 0 })}
                    disabled={!newRoute.name.trim()}
                    className="bg-[#002B5B] text-white text-sm px-4 py-2 rounded-lg hover:bg-[#003d80] disabled:opacity-50 transition-colors">
                    Add
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ================================================================
              TAB 4 — IMPORT
              ================================================================ */}
          {activeTab === 'import' && (
            <div className="space-y-5 max-w-3xl">
              <div className="bg-blue-50 border border-blue-100 rounded-xl p-4">
                <p className="text-xs font-semibold text-blue-800 mb-2">Expected columns (Excel / CSV)</p>
                <div className="flex flex-wrap gap-1.5">
                  {['item_code *', 'item_name *', 'item_type', 'hsn_code', 'season', 'merchant_code', 'selling_price', 'purchase_price', 'sizes', 'launch_date'].map(c => (
                    <span key={c} className={`px-2 py-0.5 rounded text-xs font-mono ${c.includes('*') ? 'bg-blue-600 text-white' : 'bg-white border border-blue-200 text-blue-700'}`}>{c}</span>
                  ))}
                </div>
                <p className="text-xs text-blue-600 mt-2">
                  <strong>item_type</strong> values: FG, SFG, RM, ACC, PKG, FUEL, SVC &nbsp;·&nbsp;
                  <strong>sizes</strong>: comma-separated e.g. "S,M,L,XL"
                </p>
              </div>

              <div
                className="border-2 border-dashed border-gray-300 rounded-xl p-10 text-center hover:border-[#002B5B] transition-colors cursor-pointer"
                onClick={() => importRef.current?.click()}
                onDragOver={e => e.preventDefault()}
                onDrop={e => {
                  e.preventDefault()
                  const f = e.dataTransfer.files[0]
                  if (f) { setImportFile(f); handleImportPreview(f) }
                }}>
                <div className="text-3xl mb-2">📥</div>
                <p className="text-sm text-gray-600 font-medium">Click or drag & drop your Excel / CSV file</p>
                <p className="text-xs text-gray-400 mt-1">.xlsx, .xls, .csv supported</p>
                <input ref={importRef} type="file" accept=".xlsx,.xls,.csv" className="hidden"
                  onChange={e => {
                    const f = e.target.files?.[0]
                    if (f) { setImportFile(f); handleImportPreview(f) }
                  }} />
              </div>

              {importMsg && (
                <p className={`text-sm font-medium ${importMsg.startsWith('✓') ? 'text-green-600' : 'text-red-500'}`}>{importMsg}</p>
              )}

              {importPreview && importPreview.length > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-gray-700">
                      Preview — showing {importPreview.length} of {importTotal} rows
                    </p>
                    <button onClick={handleImportConfirm} disabled={importing}
                      className="bg-green-600 text-white text-sm px-5 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50 font-medium transition-colors">
                      {importing ? 'Importing…' : `Confirm Import (${importTotal} rows)`}
                    </button>
                  </div>
                  <div className="overflow-x-auto rounded-xl border border-gray-200">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="bg-gray-50 border-b border-gray-200">
                          {['item_code', 'item_name', 'item_type', 'hsn_code', 'season', 'sizes', 'selling_price', 'purchase_price'].map(h => (
                            <th key={h} className="px-3 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide whitespace-nowrap">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {importPreview.map((row, i) => (
                          <tr key={i} className="hover:bg-gray-50">
                            <td className="px-3 py-2 font-mono font-medium text-[#002B5B]">{row.item_code}</td>
                            <td className="px-3 py-2 text-gray-700">{row.item_name}</td>
                            <td className="px-3 py-2 text-gray-500">{row.item_type}</td>
                            <td className="px-3 py-2 text-gray-500">{row.hsn_code || '—'}</td>
                            <td className="px-3 py-2 text-gray-500">{row.season || '—'}</td>
                            <td className="px-3 py-2 text-blue-600">{(row.sizes ?? []).join(', ') || '—'}</td>
                            <td className="px-3 py-2 tabular-nums">{row.selling_price || '—'}</td>
                            <td className="px-3 py-2 tabular-nums">{row.purchase_price || '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ================================================================
              TAB 6 — BUYER PACKAGING
              ================================================================ */}
          {activeTab === 'packaging' && (
            <div className="space-y-6">

              {/* ── Buyer Master ─────────────────────────────────────────── */}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-gray-700">Buyer Master</h3>
                <div className="overflow-x-auto rounded-xl border border-gray-200">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="bg-gray-50 border-b border-gray-200">
                        {['#', 'Buyer Code', 'Buyer Name', 'Action'].map(h => (
                          <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {buyers.length === 0 ? (
                        <tr><td colSpan={4} className="px-4 py-6 text-center text-gray-400 text-sm">No buyers yet. Add one below.</td></tr>
                      ) : buyers.map((b, i) => (
                        <tr key={b.id} className="hover:bg-gray-50 transition-colors">
                          <td className="px-4 py-2.5 text-gray-400 text-xs">{i + 1}</td>
                          <td className="px-4 py-2.5 font-mono font-medium text-[#002B5B]">{b.buyer_code}</td>
                          <td className="px-4 py-2.5 text-gray-800">{b.buyer_name}</td>
                          <td className="px-4 py-2.5">
                            <button onClick={() => { if (confirm(`Delete buyer "${b.buyer_code}"?`)) deleteBuyerMut.mutate(b.id) }}
                              className="text-red-400 hover:text-red-600 text-xs px-2 py-1 rounded hover:bg-red-50 transition-colors">
                              Delete
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-3">
                  <p className="text-xs font-semibold text-gray-700 uppercase tracking-wide">Add Buyer</p>
                  <div className="flex gap-3 flex-wrap">
                    <input type="text" placeholder="Buyer Code *" value={newBuyer.buyer_code}
                      onChange={e => setNewBuyer(p => ({ ...p, buyer_code: e.target.value }))}
                      className="flex-1 min-w-[130px] border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                    <input type="text" placeholder="Buyer Name *" value={newBuyer.buyer_name}
                      onChange={e => setNewBuyer(p => ({ ...p, buyer_name: e.target.value }))}
                      className="flex-1 min-w-[200px] border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                    <button
                      onClick={() => {
                        if (!newBuyer.buyer_code.trim() || !newBuyer.buyer_name.trim()) { setBuyerErr('Both fields required.'); return }
                        createBuyerMut.mutate(newBuyer)
                      }}
                      disabled={createBuyerMut.isPending}
                      className="bg-[#002B5B] text-white text-sm px-4 py-2 rounded-lg hover:bg-[#003d80] disabled:opacity-50 transition-colors">
                      Add
                    </button>
                  </div>
                  {buyerErr && <p className="text-xs text-red-500">{buyerErr}</p>}
                </div>
              </div>

              <hr className="border-gray-200" />

              {/* ── Packaging Configurator ───────────────────────────────── */}
              <div>
                <h3 className="text-sm font-semibold text-gray-700 mb-3">Define Packaging per Item per Buyer</h3>
                <div className="flex gap-5 min-h-[400px]">

                  {/* Left: item search + buyer selector */}
                  <div className="w-64 flex-shrink-0 space-y-4">
                    <div className="space-y-2">
                      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">1. Select Item</div>
                      <input type="text" placeholder="Search item code / name…"
                        value={pkgItemSearch} onChange={e => setPkgItemSearch(e.target.value)}
                        className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                      {pkgItemSearch.length >= 2 && pkgItemSearchResults.length > 0 && (
                        <div className="border border-gray-200 rounded-lg divide-y divide-gray-100 max-h-52 overflow-y-auto bg-white shadow-sm">
                          {pkgItemSearchResults.map(it => (
                            <button key={it.id} onClick={() => {
                              setPkgItemId(it.id); setPkgItemName(it.item_code + ' — ' + it.item_name)
                              setPkgItemSearch(''); setPkgBuyerId(null); setPkgErr('')
                            }}
                              className="w-full text-left px-3 py-2 hover:bg-blue-50 transition-colors">
                              <div className="text-xs font-mono font-medium text-[#002B5B]">{it.item_code}</div>
                              <div className="text-xs text-gray-500">{it.item_name}</div>
                            </button>
                          ))}
                        </div>
                      )}
                      {pkgItemId && (
                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-2 text-xs font-medium text-blue-800">{pkgItemName}</div>
                      )}
                    </div>

                    {pkgItemId && (
                      <div className="space-y-2">
                        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">2. Select Buyer</div>
                        {buyers.length === 0 ? (
                          <p className="text-xs text-gray-400">Add buyers above first.</p>
                        ) : (
                          <div className="space-y-1">
                            <button
                              onClick={() => setPkgBuyerId(null)}
                              className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition-colors border ${
                                pkgBuyerId === null ? 'bg-[#002B5B] text-white border-[#002B5B]' : 'bg-white text-gray-700 border-gray-200 hover:border-[#002B5B]'
                              }`}>
                              All Buyers
                            </button>
                            {buyers.map(b => (
                              <button key={b.id}
                                onClick={() => setPkgBuyerId(b.id)}
                                className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition-colors border ${
                                  pkgBuyerId === b.id ? 'bg-[#002B5B] text-white border-[#002B5B]' : 'bg-white text-gray-700 border-gray-200 hover:border-[#002B5B]'
                                }`}>
                                <div className="font-mono">{b.buyer_code}</div>
                                <div className="opacity-70 font-normal">{b.buyer_name}</div>
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Right: packaging lines */}
                  <div className="flex-1 space-y-4">
                    {!pkgItemId ? (
                      <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
                        Search and select an item to view/define its packaging
                      </div>
                    ) : (
                      <>
                        {/* Heading */}
                        <div className="flex items-center justify-between">
                          <div>
                            <span className="text-sm font-semibold text-gray-800">
                              {pkgBuyerId ? buyers.find(b => b.id === pkgBuyerId)?.buyer_name ?? '' : 'All Buyers'}
                            </span>
                            <span className="ml-2 text-xs text-gray-500">packaging for {pkgItemName.split('—')[0].trim()}</span>
                          </div>
                          {pkgBuyerId && (
                            <button onClick={() => { setShowPkgForm(true); setPkgErr('') }}
                              className="text-sm text-[#002B5B] hover:text-[#003d80] font-medium border border-dashed border-[#002B5B]/30 rounded-lg px-4 py-1.5 hover:bg-blue-50 transition-colors">
                              + Add Packaging
                            </button>
                          )}
                        </div>

                        {/* Packaging table */}
                        {packagingLines.length === 0 ? (
                          <div className="text-center py-10 text-gray-400 text-sm border border-dashed border-gray-200 rounded-xl">
                            {pkgBuyerId
                              ? 'No packaging defined for this buyer. Click "+ Add Packaging" to start.'
                              : 'Select a specific buyer to add packaging, or view all buyers below.'}
                          </div>
                        ) : (
                          <div className="overflow-x-auto rounded-xl border border-gray-200">
                            <table className="w-full text-xs">
                              <thead>
                                <tr className="bg-gray-50 border-b border-gray-200">
                                  {!pkgBuyerId && <th className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">Buyer</th>}
                                  <th className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">Packaging Item</th>
                                  <th className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">Type</th>
                                  <th className="px-3 py-2.5 text-right text-xs font-semibold text-gray-500 uppercase tracking-wide">Qty</th>
                                  <th className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">Unit</th>
                                  <th className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">Remarks</th>
                                  <th className="px-3 py-2.5"></th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-gray-100">
                                {packagingLines.map((l: any) => (
                                  <tr key={l.id} className="hover:bg-gray-50 transition-colors">
                                    {!pkgBuyerId && (
                                      <td className="px-3 py-2">
                                        <div className="font-mono font-medium text-[#002B5B]">{l.buyer_code}</div>
                                        <div className="text-gray-400">{l.buyer_name}</div>
                                      </td>
                                    )}
                                    <td className="px-3 py-2">
                                      <div className="font-medium text-gray-800">{l.pkg_item_code}</div>
                                      <div className="text-gray-400">{l.pkg_item_name}</div>
                                    </td>
                                    <td className="px-3 py-2">
                                      <span className="bg-amber-50 text-amber-700 px-1.5 py-0.5 rounded text-xs">{l.pkg_item_type}</span>
                                    </td>
                                    <td className="px-3 py-2 text-right tabular-nums font-medium">{l.quantity}</td>
                                    <td className="px-3 py-2 text-gray-500">{l.unit}</td>
                                    <td className="px-3 py-2 text-gray-400 max-w-[140px] truncate">{l.remarks || '—'}</td>
                                    <td className="px-3 py-2">
                                      <button onClick={() => deletePkgLineMut.mutate({ buyerId: l.buyer_id, lineId: l.id })}
                                        className="text-red-400 hover:text-red-600 transition-colors">✕</button>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}

                        {/* Add packaging line form */}
                        {showPkgForm && pkgBuyerId && (
                          <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-3">
                            <p className="text-xs font-semibold text-gray-700 uppercase tracking-wide">
                              Add packaging for {buyers.find(b => b.id === pkgBuyerId)?.buyer_name}
                            </p>
                            {/* Packaging item search */}
                            <div className="space-y-1">
                              <label className="text-xs text-gray-500">Packaging Item * (from Item Master)</label>
                              {newPkgLine.packaging_item_id ? (
                                <div className="flex items-center gap-2 bg-blue-50 border border-blue-200 rounded-lg px-3 py-2">
                                  <span className="text-xs font-medium text-blue-800 flex-1">{newPkgLine.pkg_label}</span>
                                  <button onClick={() => setNewPkgLine(p => ({ ...p, packaging_item_id: null, pkg_label: '' }))}
                                    className="text-gray-400 hover:text-red-500 text-xs">✕</button>
                                </div>
                              ) : (
                                <div className="relative">
                                  <input type="text" placeholder="Type item code or name (e.g. Polybag, Tag)…" value={pkgCompSearch}
                                    onChange={e => setPkgCompSearch(e.target.value)}
                                    className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                                  {pkgCompSearch.length >= 2 && pkgCompSearchResults.length > 0 && (
                                    <div className="absolute z-10 top-full mt-1 w-full border border-gray-200 rounded-lg divide-y divide-gray-100 max-h-48 overflow-y-auto bg-white shadow-lg">
                                      {pkgCompSearchResults.map(it => (
                                        <button key={it.id} onClick={() => {
                                          setNewPkgLine(p => ({ ...p, packaging_item_id: it.id, pkg_label: it.item_code + ' — ' + it.item_name }))
                                          setPkgCompSearch('')
                                        }}
                                          className="w-full text-left px-3 py-2 hover:bg-blue-50 transition-colors">
                                          <div className="text-xs font-mono font-medium text-[#002B5B]">{it.item_code}</div>
                                          <div className="text-xs text-gray-500">{it.item_name} · {it.item_type_code}</div>
                                        </button>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                            <div className="flex gap-3 flex-wrap">
                              <div className="space-y-1 w-24">
                                <label className="text-xs text-gray-500">Quantity</label>
                                <input type="number" value={newPkgLine.quantity}
                                  onChange={e => setNewPkgLine(p => ({ ...p, quantity: e.target.value }))}
                                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                              </div>
                              <div className="space-y-1 w-28">
                                <label className="text-xs text-gray-500">Unit</label>
                                <select value={newPkgLine.unit}
                                  onChange={e => setNewPkgLine(p => ({ ...p, unit: e.target.value }))}
                                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]">
                                  {UNITS.map(u => <option key={u}>{u}</option>)}
                                </select>
                              </div>
                              <div className="space-y-1 flex-1 min-w-[160px]">
                                <label className="text-xs text-gray-500">Remarks</label>
                                <input type="text" placeholder="e.g. Per 12 pcs carton" value={newPkgLine.remarks}
                                  onChange={e => setNewPkgLine(p => ({ ...p, remarks: e.target.value }))}
                                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                              </div>
                            </div>
                            {pkgErr && <p className="text-xs text-red-500">{pkgErr}</p>}
                            <div className="flex gap-2">
                              <button
                                onClick={() => {
                                  if (!newPkgLine.packaging_item_id) { setPkgErr('Select a packaging item first.'); return }
                                  addPkgLineMut.mutate({
                                    packaging_item_id: newPkgLine.packaging_item_id,
                                    quantity: parseFloat(newPkgLine.quantity) || 1,
                                    unit: newPkgLine.unit,
                                    remarks: newPkgLine.remarks,
                                  })
                                }}
                                disabled={addPkgLineMut.isPending}
                                className="bg-[#002B5B] text-white text-xs px-4 py-2 rounded-lg hover:bg-[#003d80] disabled:opacity-50 transition-colors">
                                {addPkgLineMut.isPending ? 'Adding…' : 'Add'}
                              </button>
                              <button onClick={() => { setShowPkgForm(false); setPkgCompSearch(''); setPkgErr('') }}
                                className="text-gray-500 text-xs px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors">Cancel</button>
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ================================================================
              TAB 5 — MERCHANTS
              ================================================================ */}
          {activeTab === 'merchants' && (
            <div className="space-y-4 max-w-2xl">
              <p className="text-sm text-gray-500">
                Create merchant records here. The Merchant dropdown in the Item creation form will use this list.
              </p>

              {/* Merchants table */}
              <div className="overflow-x-auto rounded-xl border border-gray-200">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      {['#', 'Merchant Code', 'Merchant Name', 'Action'].map(h => (
                        <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {merchants.length === 0 ? (
                      <tr>
                        <td colSpan={4} className="px-4 py-8 text-center text-gray-400">
                          No merchants yet. Add one below.
                        </td>
                      </tr>
                    ) : merchants.map((m, i) => (
                      <tr key={m.id} className="hover:bg-gray-50 transition-colors">
                        <td className="px-4 py-3 text-gray-400 text-xs">{i + 1}</td>
                        <td className="px-4 py-3 font-mono font-medium text-[#002B5B]">{m.merchant_code}</td>
                        <td className="px-4 py-3 text-gray-800">{m.merchant_name}</td>
                        <td className="px-4 py-3">
                          <button onClick={() => { if (confirm(`Delete merchant "${m.merchant_code}"?`)) deleteMerchantMut.mutate(m.id) }}
                            className="text-red-400 hover:text-red-600 text-xs px-2 py-1 rounded hover:bg-red-50 transition-colors">
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Add merchant form */}
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-3">
                <p className="text-xs font-semibold text-gray-700 uppercase tracking-wide">Add Merchant</p>
                <div className="flex gap-3 flex-wrap">
                  <input type="text" placeholder="Merchant Code * (e.g. YG001)" value={newMerchant.merchant_code}
                    onChange={e => setNewMerchant(p => ({ ...p, merchant_code: e.target.value }))}
                    className="flex-1 min-w-[140px] border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                  <input type="text" placeholder="Merchant Name *" value={newMerchant.merchant_name}
                    onChange={e => setNewMerchant(p => ({ ...p, merchant_name: e.target.value }))}
                    className="flex-1 min-w-[200px] border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#002B5B]" />
                  <button
                    onClick={() => {
                      if (!newMerchant.merchant_code.trim() || !newMerchant.merchant_name.trim()) {
                        setMerchantErr('Both fields are required.'); return
                      }
                      createMerchantMut.mutate(newMerchant)
                    }}
                    disabled={createMerchantMut.isPending}
                    className="bg-[#002B5B] text-white text-sm px-4 py-2 rounded-lg hover:bg-[#003d80] disabled:opacity-50 transition-colors">
                    Add
                  </button>
                </div>
                {merchantErr && <p className="text-xs text-red-500">{merchantErr}</p>}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}
