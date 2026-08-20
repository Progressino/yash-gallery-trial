import { useState, useEffect, useMemo, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'
import api from '../api/client'
import { barcodePrintBlock, fetchDocBarcode } from '../lib/docBarcode'
import {
  appendManualJoLine,
  expandJoExportRows,
  formatJoHeaderSku,
  joLineSkus,
  joMatchesSkuFilter,
  JO_EXPORT_HEADERS,
} from './joLineHelpers'
import SetBomPanel from '../components/SetBomPanel'
import CuttingReportsPanel from './CuttingReportsPanel'
import { downloadCsv } from '../lib/exportCsv'

type MasterSkuOption = {
  sku: string
  sku_name: string
  source?: string
}

/** Searchable SKU dropdown for Manual SO Cutting JO (item master + sales/ready SKUs). */
function ManualJoSkuDropdown({
  value,
  catalog,
  onPick,
}: {
  value: string
  catalog: MasterSkuOption[]
  onPick: (sku: string, skuName: string) => void
}) {
  const [filter, setFilter] = useState('')
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const { data: itemHits = [] } = useQuery<
    { id?: number; item_code?: string; item_name?: string; size_label?: string }[]
  >({
    queryKey: ['manual-jo-sku-search', filter],
    queryFn: () =>
      api
        .get(`/items/search?q=${encodeURIComponent(filter.trim())}`)
        .then(r => r.data || []),
    enabled: open && filter.trim().length >= 2,
    staleTime: 30_000,
  })

  const options = useMemo(() => {
    const map = new Map<string, MasterSkuOption>()
    for (const row of catalog) {
      const sku = String(row.sku || '').trim()
      if (!sku) continue
      if (!map.has(sku.toUpperCase())) {
        map.set(sku.toUpperCase(), {
          sku,
          sku_name: String(row.sku_name || '').trim(),
          source: row.source,
        })
      }
    }
    for (const it of itemHits) {
      const sku = String(it.item_code || '').trim()
      if (!sku) continue
      const key = sku.toUpperCase()
      if (!map.has(key)) {
        map.set(key, {
          sku,
          sku_name: [it.item_name, it.size_label].filter(Boolean).join(' · '),
          source: 'item_master',
        })
      }
    }
    let list = [...map.values()]
    const q = filter.trim().toLowerCase()
    if (q) {
      list = list.filter(
        r =>
          r.sku.toLowerCase().includes(q)
          || r.sku_name.toLowerCase().includes(q),
      )
    }
    list.sort((a, b) => a.sku.localeCompare(b.sku))
    return list.slice(0, 400)
  }, [catalog, itemHits, filter])

  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [])

  const selectedLabel = useMemo(() => {
    const hit = catalog.find(c => c.sku.toUpperCase() === value.toUpperCase())
      || options.find(o => o.sku.toUpperCase() === value.toUpperCase())
    if (!value) return ''
    return hit?.sku_name ? `${value} — ${hit.sku_name}` : value
  }, [value, catalog, options])

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className={`w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1 text-left font-mono bg-white
          ${value ? 'text-gray-900' : 'text-gray-400'}`}
      >
        {value ? selectedLabel : 'Select SKU…'}
      </button>
      {open && (
        <div className="absolute z-50 mt-1 w-full min-w-[16rem] max-w-md bg-white border border-gray-200 rounded-lg shadow-lg">
          <div className="p-2 border-b">
            <input
              autoFocus
              type="search"
              value={filter}
              onChange={e => setFilter(e.target.value)}
              placeholder="Search item master / SO SKUs…"
              className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm font-mono"
            />
          </div>
          <div className="max-h-52 overflow-y-auto">
            {options.length === 0 ? (
              <p className="px-3 py-2 text-xs text-gray-500">
                {filter.trim().length < 2
                  ? 'Type 2+ characters to search Item Master, or pick from SO/JO list.'
                  : 'No matching SKUs.'}
              </p>
            ) : (
              options.map(opt => (
                <button
                  key={opt.sku}
                  type="button"
                  className={`w-full text-left px-3 py-2 text-sm border-b border-gray-50 last:border-0 hover:bg-blue-50
                    ${value === opt.sku ? 'bg-blue-50' : ''}`}
                  onClick={() => {
                    onPick(opt.sku, opt.sku_name)
                    setOpen(false)
                    setFilter('')
                  }}
                >
                  <span className="font-mono font-semibold text-[#002B5B]">{opt.sku}</span>
                  {opt.sku_name && (
                    <span className="ml-2 text-xs text-gray-500 truncate">{opt.sku_name}</span>
                  )}
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function apiErrorMessage(err: unknown, fallback: string): string {
  if (!axios.isAxiosError(err)) return fallback
  const detail = err.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) {
    return detail.map((d: { msg?: string }) => d?.msg || String(d)).join('\n') || fallback
  }
  if (err.message) return err.message
  return fallback
}

/** Garment pcs for fabric BOM — embroidery JOs bill in measurement units, not pieces. */
function garmentPcsForFabricIssue(jo: JO): number {
  if (jo.process === 'Embroidery') {
    const lines = (jo.lines || []).filter(l => l.sku)
    const fromLines = lines.reduce((s, l) => s + (Number((l as JOLine & { garment_qty?: number }).garment_qty) || 0), 0)
    const header = Number(jo.garment_qty) || 0
    if (fromLines > 0) return fromLines
    if (header > 0) return header
  }
  const lines = (jo.lines || []).filter(l => (l.planned_qty || 0) > 0 && l.sku)
  const fromLines = lines.reduce((s, l) => s + (Number(l.planned_qty) || 0), 0)
  return fromLines || Number(jo.planned_qty) || 1
}

type ModalType = 'issue-fabric' | 'return-fabric' | 'receive' | 'issue-pieces' | 'add-cost' | 'new-jo' | null

interface JOLine {
  id: number
  so_number: string
  sku: string
  sku_name: string
  style: string
  planned_qty: number
  issued_qty: number
  received_qty: number
  rejected_qty: number
  balance_qty: number
  vendor_rate: number
  process_cost: number
  remarks: string
}

interface JO {
  id: number
  jo_number: string
  jo_date: string
  so_number: string
  /** system = from sales SO; manual = free-text SO reference only */
  so_source?: string
  sku: string
  sku_name: string
  process: string
  exec_type: string
  vendor_name: string
  vendor_rate: number
  so_qty: number
  planned_qty: number
  issued_qty: number
  received_qty: number
  rejected_qty: number
  balance_qty: number
  output_qty: number
  status: string
  expected_completion: string
  fabric_code: string
  fabric_qty: number
  fabric_unit: string
  fabric_issued_qty: number
  fabric_received_qty: number
  fabric_consumption: number
  process_cost: number
  total_cost: number
  parent_jo_id: number | null
  next_stage_jo_id: number | null
  remarks: string
  lines: JOLine[]
  fabric_issues: any[]
  fabric_returns: any[]
  cost_entries: any[]
  routing: string[]
  next_process: string | null
  process_stocks: Record<string, { available: number; in: number; out: number }>
  issue_note?: IssueNote | null
  garment_qty?: number
  measurement_qty?: number
  embroidery_type?: string
  embroidery_unit?: string
}

interface IssueNoteLine {
  id: number
  line_no: number
  finished_item_code: string
  finished_item_name: string
  finished_planned_qty: number
  material_code: string
  material_name: string
  material_type: string
  bom_qty_per_unit: number
  required_qty: number
  unit: string
  issued_qty: number
  remarks: string
}

interface IssueNote {
  id: number
  in_number: string
  in_date: string
  jo_id: number
  jo_number: string
  jo_date: string
  so_number: string
  process: string
  finished_item_code: string
  finished_item_name: string
  planned_qty: number
  status: string
  remarks: string
  lines: IssueNoteLine[]
  line_count?: number
}

const STATUS_COLORS: Record<string, string> = {
  Created: 'bg-gray-100 text-gray-600',
  'In Progress': 'bg-amber-100 text-amber-700',
  Completed: 'bg-green-100 text-green-700',
  Closed: 'bg-gray-200 text-gray-500',
  Cancelled: 'bg-red-100 text-red-600',
}

function JOIssueNotePanel({ joId, joNumber }: { joId: number; joNumber: string }) {
  const qc = useQueryClient()
  const { data: note, isLoading, isError } = useQuery<IssueNote>({
    queryKey: ['jo-issue-note', joId],
    queryFn: () => api.get(`/production/orders/${joId}/issue-note`).then(r => r.data),
  })
  const regenMut = useMutation({
    mutationFn: () => api.post(`/production/orders/${joId}/regenerate-issue-note`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jo-issue-note', joId] })
      qc.invalidateQueries({ queryKey: ['prod-issue-notes'] })
    },
    onError: (e: any) => alert(e.response?.data?.detail || 'Could not regenerate issue note'),
  })

  if (isLoading) return <p className="text-xs text-gray-400 py-2">Loading issue note…</p>
  if (isError || !note) {
    return (
      <div className="bg-white rounded-lg border border-dashed border-gray-200 p-3 flex items-center justify-between gap-2">
        <p className="text-xs text-gray-500">No material issue note for {joNumber}.</p>
        <button onClick={() => regenMut.mutate()} disabled={regenMut.isPending}
          className="text-xs px-2 py-1 bg-[#002B5B] text-white rounded hover:bg-blue-800 disabled:opacity-50">
          Generate from BOM
        </button>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg border border-indigo-100 overflow-hidden">
      <div className="px-3 py-2 bg-indigo-50 flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-xs font-semibold text-indigo-800">
            📋 Material Issue Note — <span className="font-mono">{note.in_number}</span>
          </p>
          <p className="text-xs text-indigo-600 mt-0.5">
            JO <b>{note.jo_number}</b> · {note.in_date} · For: <b>{note.finished_item_code}</b>
            {note.finished_item_name ? ` — ${note.finished_item_name}` : ''} · Qty <b>{note.planned_qty}</b>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs px-2 py-0.5 rounded-full bg-white text-indigo-700 border border-indigo-200">{note.status}</span>
          <button onClick={() => regenMut.mutate()} disabled={regenMut.isPending}
            className="text-xs px-2 py-1 border border-indigo-200 rounded text-indigo-700 hover:bg-white disabled:opacity-50">
            ↻ Refresh BOM
          </button>
        </div>
      </div>
      {note.lines.length === 0 ? (
        <p className="text-xs text-gray-400 p-3">{note.remarks || 'No BOM lines found.'}</p>
      ) : (
        <table className="w-full text-xs">
          <thead className="text-gray-400 uppercase bg-gray-50 border-b">
            <tr>
              <th className="text-left px-3 py-2">For (finished item)</th>
              <th className="text-left px-3 py-2">Material</th>
              <th className="text-right px-3 py-2">BOM / unit</th>
              <th className="text-right px-3 py-2">Required</th>
              <th className="text-left px-3 py-2">Unit</th>
            </tr>
          </thead>
          <tbody>
            {note.lines.map(ln => (
              <tr key={ln.id} className="border-t border-gray-50 hover:bg-indigo-50/30">
                <td className="px-3 py-2">
                  <span className="font-mono font-semibold text-[#002B5B]">{ln.finished_item_code}</span>
                  {ln.finished_item_name && <span className="text-gray-500 ml-1">({ln.finished_item_name})</span>}
                  <span className="text-gray-400 ml-1">× {ln.finished_planned_qty}</span>
                </td>
                <td className="px-3 py-2">
                  <span className="font-mono font-semibold">{ln.material_code}</span>
                  {ln.material_name && ln.material_name !== ln.material_code && (
                    <span className="text-gray-500 ml-1">— {ln.material_name}</span>
                  )}
                </td>
                <td className="px-3 py-2 text-right text-gray-600">{ln.bom_qty_per_unit}</td>
                <td className="px-3 py-2 text-right font-bold text-indigo-700">{ln.required_qty}</td>
                <td className="px-3 py-2 text-gray-500">{ln.unit}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

const PROCESS_COLORS: Record<string, string> = {
  Cutting: 'bg-blue-100 text-blue-700',
  Printing: 'bg-pink-100 text-pink-700',
  Embroidery: 'bg-rose-100 text-rose-700',
  Stitching: 'bg-purple-100 text-purple-700',
  Finishing: 'bg-green-100 text-green-700',
  Packing: 'bg-teal-100 text-teal-700',
  'Kajh Button': 'bg-orange-100 text-orange-700',
  'Kaj Button': 'bg-orange-100 text-orange-700',
  Handwork: 'bg-rose-100 text-rose-700',
}

const PROCESS_ICONS: Record<string, string> = {
  Cutting: '✂️', Printing: '🖨️', Embroidery: '🧶',
  Stitching: '🧵', Finishing: '✨', Packing: '📦',
  'Kajh Button': '🔘',
  'Kaj Button': '🔘',
  Handwork: '🪡',
}

const fmt = (n: number) => Math.round(n || 0).toLocaleString('en-IN')
const fmtR = (n: number) => '₹' + Math.round(n || 0).toLocaleString('en-IN')

const EXEC_TYPE_OPTIONS = [
  { value: 'Inhouse', label: 'In-house' },
  { value: 'Outsource', label: 'Outsource' },
] as const

const PRODUCTION_MODE_OPTIONS = [
  { value: 'inhouse', label: 'In-house' },
  { value: 'cut_to_pack', label: 'Cut-to-Pack (vendor)' },
  { value: 'stitch_to_pack', label: 'Stitch-to-Pack (vendor)' },
] as const

const PRODUCTION_MODE_LABEL: Record<string, string> = Object.fromEntries(
  PRODUCTION_MODE_OPTIONS.map(m => [m.value, m.label]),
)

function isOutsourceExec(execType: string) {
  return String(execType || '').trim().toLowerCase() === 'outsource'
}

function suggestedExecType(mode: string | undefined, process: string) {
  const m = String(mode || 'inhouse').toLowerCase().replace(/[-\s]/g, '_')
  if ((m === 'cut_to_pack' || m === 'cutpack' || m === 'cut_pack' || m === 'c2p') && process === 'Cutting') return 'Outsource'
  if ((m === 'stitch_to_pack' || m === 'stich_to_pack' || m === 'stitchpack' || m === 'stichpack' || m === 's2p') && process === 'Stitching') return 'Outsource'
  return 'Inhouse'
}

function execTypeLabel(execType: string) {
  return isOutsourceExec(execType) ? 'Outsource' : 'In-house'
}

function VendorExecutionEditor({
  jo,
  vendorSuggestions,
  saving,
  onSave,
}: {
  jo: JO
  vendorSuggestions: string[]
  saving: boolean
  onSave: (data: { exec_type: string; vendor_name: string }) => void
}) {
  const [execType, setExecType] = useState(jo.exec_type || 'Inhouse')
  const [vendorName, setVendorName] = useState(jo.vendor_name || '')

  useEffect(() => {
    setExecType(jo.exec_type || 'Inhouse')
    setVendorName(jo.vendor_name || '')
  }, [jo.id, jo.exec_type, jo.vendor_name])

  return (
    <div className="bg-white rounded-lg border p-3 space-y-3">
      <p className="text-xs font-semibold text-gray-500 uppercase">Execution / Vendor</p>
      <div className="grid sm:grid-cols-3 gap-3">
        <div>
          <label className="text-xs text-gray-500">Execution type</label>
          <select
            value={execType}
            onChange={e => {
              const v = e.target.value
              setExecType(v)
              if (!isOutsourceExec(v)) setVendorName('')
            }}
            className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1"
          >
            {EXEC_TYPE_OPTIONS.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
        {isOutsourceExec(execType) && (
          <div className="sm:col-span-2">
            <label className="text-xs text-gray-500">Vendor name *</label>
            <input
              list="jo-vendor-suggestions"
              value={vendorName}
              onChange={e => setVendorName(e.target.value)}
              placeholder="Outsource vendor / party name"
              className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1"
            />
            <datalist id="jo-vendor-suggestions">
              {vendorSuggestions.map(v => (
                <option key={v} value={v} />
              ))}
            </datalist>
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={saving || (isOutsourceExec(execType) && !vendorName.trim())}
          onClick={() => onSave({
            exec_type: execType,
            vendor_name: isOutsourceExec(execType) ? vendorName.trim() : '',
          })}
          className="px-3 py-1.5 text-xs bg-[#002B5B] text-white rounded-lg font-medium disabled:opacity-50"
        >
          {saving ? 'Saving…' : 'Save vendor'}
        </button>
        <span className="text-xs text-gray-500">
          Current: <b>{execTypeLabel(jo.exec_type)}</b>
          {isOutsourceExec(jo.exec_type) && jo.vendor_name ? ` · ${jo.vendor_name}` : ''}
        </span>
      </div>
    </div>
  )
}

// ── Print JO ──────────────────────────────────────────────────────────────────
const printJO = async (jo: JO) => {
  const totalCost = jo.lines.reduce((s, l) => s + (l.planned_qty * l.vendor_rate), 0)
  let barcodeHtml = ''
  try {
    const bundle = await fetchDocBarcode('JO', jo.jo_number)
    barcodeHtml = barcodePrintBlock(bundle)
  } catch { /* optional */ }
  const win = window.open('', '_blank', 'width=900,height=700')
  if (!win) { alert('Allow popups to print'); return }
  win.document.write(`<!DOCTYPE html><html><head><title>JO - ${jo.jo_number}</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:'Segoe UI',sans-serif;font-size:12px;color:#1a1a1a;padding:24px}
    .header{display:flex;justify-content:space-between;border-bottom:2px solid #002B5B;padding-bottom:12px;margin-bottom:16px}
    .company{font-size:20px;font-weight:700;color:#002B5B}
    .doc-title{font-size:16px;font-weight:600;color:#002B5B;text-align:right}
    .doc-num{font-size:22px;font-weight:800;color:#002B5B;text-align:right}
    .info-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px}
    .info-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px}
    .info-label{font-size:10px;text-transform:uppercase;color:#64748b;font-weight:600;margin-bottom:4px}
    .info-value{font-size:13px;font-weight:600;color:#1e293b}
    table{width:100%;border-collapse:collapse;margin-bottom:16px}
    th{background:#002B5B;color:white;padding:7px 10px;text-align:left;font-size:11px}
    th.r,td.r{text-align:right}
    td{padding:6px 10px;border-bottom:1px solid #e2e8f0;font-size:12px}
    tr:nth-child(even) td{background:#f8fafc}
    .total-row{display:flex;justify-content:flex-end}
    .totals{width:260px;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden}
    .tr{display:flex;justify-content:space-between;padding:7px 12px;border-bottom:1px solid #e2e8f0;font-size:12px}
    .tr.grand{background:#002B5B;color:white;font-weight:700;font-size:14px}
    .routing-bar{display:flex;gap:4px;margin-bottom:16px;align-items:center}
    .step{padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;background:#e2e8f0;color:#475569}
    .step.active{background:#002B5B;color:white}
    .arrow{color:#94a3b8;font-size:10px}
    .footer{margin-top:32px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px;border-top:1px solid #e2e8f0;padding-top:16px}
    .sign-box{text-align:center}
    .sign-line{border-top:1px solid #64748b;margin-top:32px;padding-top:6px;font-size:10px;color:#64748b}
    @media print{body{padding:12px}}
  </style></head><body>
  <div class="header">
    <div><div class="company">🧵 Garment ERP</div><div style="font-size:11px;color:#64748b">Production Department</div></div>
    <div><div class="doc-title">JOB ORDER</div><div class="doc-num">${jo.jo_number}</div>${barcodeHtml ? `<div style="margin-top:8px;display:flex;justify-content:flex-end">${barcodeHtml}</div>` : ''}</div>
  </div>
  <div class="routing-bar">
    ${(jo.routing || []).map(p => `<span class="step ${p === jo.process ? 'active' : ''}">${PROCESS_ICONS[p] || ''} ${p}</span>${p !== jo.routing[jo.routing.length-1] ? '<span class="arrow">→</span>' : ''}`).join('')}
  </div>
  <div class="info-grid">
    <div class="info-box"><div class="info-label">Process</div><div class="info-value">${jo.process}</div>
      <div class="info-label" style="margin-top:8px">Execution</div><div class="info-value">${execTypeLabel(jo.exec_type)}</div></div>
    <div class="info-box"><div class="info-label">Vendor / Party</div><div class="info-value">${isOutsourceExec(jo.exec_type) ? (jo.vendor_name || '—') : 'In-house'}</div>
      <div class="info-label" style="margin-top:8px">SO Number</div><div class="info-value">${jo.so_number || '—'}</div></div>
    <div class="info-box"><div class="info-label">JO Date</div><div class="info-value">${jo.jo_date}</div>
      <div class="info-label" style="margin-top:8px">Expected Completion</div><div class="info-value">${jo.expected_completion || '—'}</div></div>
    ${jo.process === 'Cutting' && jo.fabric_code ? `
    <div class="info-box"><div class="info-label">Fabric Code</div><div class="info-value">${jo.fabric_code}</div>
      <div class="info-label" style="margin-top:8px">Fabric Qty</div><div class="info-value">${jo.fabric_qty} ${jo.fabric_unit}</div></div>` : ''}
  </div>
  <table>
    <thead><tr>
      <th>#</th><th>SKU</th><th>Style / Description</th>
      <th class="r">Planned Qty</th><th class="r">Rate (₹)</th><th class="r">Amount (₹)</th><th>Remarks</th>
    </tr></thead>
    <tbody>
      ${jo.lines.map((l, i) => `<tr>
        <td>${i+1}</td>
        <td><strong>${l.sku}</strong></td>
        <td>${l.sku_name}${l.style ? ' — ' + l.style : ''}</td>
        <td class="r">${fmt(l.planned_qty)}</td>
        <td class="r">${fmtR(l.vendor_rate)}</td>
        <td class="r"><strong>${fmtR(l.planned_qty * l.vendor_rate)}</strong></td>
        <td>${l.remarks || '—'}</td>
      </tr>`).join('')}
    </tbody>
  </table>
  <div class="total-row"><div class="totals">
    <div class="tr"><span>Total Pieces</span><span>${fmt(jo.planned_qty)}</span></div>
    <div class="tr grand"><span>Total Amount</span><span>${fmtR(totalCost)}</span></div>
  </div></div>
  ${jo.remarks ? `<div style="background:#fef9c3;border:1px solid #fde047;border-radius:6px;padding:10px;margin-top:16px;font-size:11px"><strong>Remarks:</strong> ${jo.remarks}</div>` : ''}
  <div class="footer">
    <div class="sign-box"><div class="sign-line">Prepared By</div></div>
    <div class="sign-box"><div class="sign-line">Authorized By</div></div>
    <div class="sign-box"><div class="sign-line">${jo.exec_type === 'Outsource' ? 'Vendor Acknowledgement' : 'Received By'}</div></div>
  </div>
  <script>window.onload=()=>window.print()<\/script>
  </body></html>`)
  win.document.close()
}

function isPrintedMaterial(mat: { type?: string }, code: string): boolean {
  const t = String(mat?.type || '').toUpperCase()
  const c = String(code || '').trim().toUpperCase()
  if (['SFG', 'PRINTED', 'PRINTED FABRIC', 'PF'].includes(t)) return true
  return Boolean(c) && c.startsWith('P') && /\d/.test(c.slice(0, 6))
}

function isGreyOrFabricMaterial(mat: { type?: string; unit?: string }): boolean {
  const t = String(mat?.type || '').toUpperCase()
  const u = String(mat?.unit || '').toUpperCase()
  if (['GF', 'GREY', 'GREY FABRIC', 'RM', 'FABRIC'].includes(t)) return true
  if (['MTR', 'M', 'METER', 'METRE'].includes(u)) return true
  return false
}

function canInlineGreyAlloc(mat: any, code: string): boolean {
  if (isPrintedMaterial(mat, code)) return false
  if (isGreyOrFabricMaterial(mat)) return true
  const bd = Array.isArray(mat?.breakdown) ? mat.breakdown : []
  return bd.some((b: any) => b?.p_code || b?.printed_code || b?.allocated_grey != null)
}

function GreyInlineAllocPanel({
  materialCode,
  mat,
  greyFreeQty,
  onSaved,
}: {
  materialCode: string
  mat: any
  greyFreeQty: number
  onSaved: () => void
}) {
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')
  const unit = String(mat.unit || 'MTR')
  const breakdown = Array.isArray(mat.breakdown) ? mat.breakdown : []
  const alreadyTotal = breakdown.reduce((s: number, b: any) => s + (Number(b.allocated_qty ?? b.allocated_grey) || 0), 0)
  const draftTotal = breakdown.reduce((s: number, b: any, i: number) => {
    const key = `${b.so_no || b.so_number || ''}|${b.sku || b.fg_sku || ''}|${i}`
    return s + (Number(drafts[key]) || 0)
  }, 0)
  const remaining = Math.round((greyFreeQty - draftTotal) * 1000) / 1000
  const over = draftTotal > greyFreeQty + 0.001

  const save = async () => {
    const rows = breakdown
      .map((b: any, i: number) => {
        const key = `${b.so_no || b.so_number || ''}|${b.sku || b.fg_sku || ''}|${i}`
        return {
          qty: Number(drafts[key]) || 0,
          so_number: String(b.so_no || b.so_number || ''),
          fg_sku: String(b.sku || b.fg_sku || ''),
          printed_code: String(b.p_code || b.printed_code || materialCode),
        }
      })
      .filter((r: { qty: number }) => r.qty > 0)
    if (!rows.length) {
      setMsg('Enter allocation qty on at least one SKU')
      return
    }
    if (over) {
      setMsg(`Total allocation ${draftTotal} exceeds available ${greyFreeQty} ${unit}`)
      return
    }
    setSaving(true)
    setMsg('')
    try {
      for (const row of rows) {
        await api.post('/grey/planning/allocate-grey', {
          grey_code: materialCode,
          printed_code: row.printed_code || materialCode,
          qty: row.qty,
          so_number: row.so_number,
          fg_sku: row.fg_sku,
          unit,
          reason: 'MRP inline allocation',
        })
      }
      setDrafts({})
      setMsg(`Allocated ${draftTotal} ${unit} across ${rows.length} SKU${rows.length === 1 ? '' : 's'}`)
      onSaved()
    } catch (e: unknown) {
      setMsg(apiErrorMessage(e, 'Allocate failed'))
    }
    setSaving(false)
  }

  return (
    <div className="py-2 space-y-2">
      <p className="text-xs font-semibold text-gray-500 uppercase mb-1">
        Breakdown — Grey → P-Code → FG hierarchy:
      </p>
      <div className="flex flex-wrap gap-3 text-xs bg-white border border-blue-100 rounded-lg px-3 py-2">
        <span>Total Grey Available: <b className="font-mono text-[#002B5B]">{greyFreeQty} {unit}</b></span>
        <span>Already allocated: <b>{alreadyTotal} {unit}</b></span>
        <span>This session: <b className={over ? 'text-red-600' : 'text-green-700'}>{draftTotal} {unit}</b></span>
        <span>Remaining: <b className={remaining < 0 ? 'text-red-600' : 'text-gray-800'}>{remaining} {unit}</b></span>
      </div>
      {over && (
        <p className="text-[11px] text-red-700">Total allocation exceeds available grey — reduce quantities before saving.</p>
      )}
      <table className="w-full text-xs">
        <thead>
          <tr className="text-gray-400">
            <th className="text-left py-1 pr-3">SO Number</th>
            <th className="text-left py-1 pr-3">FG SKU</th>
            <th className="text-left py-1 pr-3">P-Code / Printed Fabric</th>
            <th className="text-right py-1 pr-3">Required Qty</th>
            <th className="text-right py-1 pr-3">Already Allocated</th>
            <th className="text-right py-1 pr-3">Allocation Qty</th>
            <th className="text-left py-1">Status</th>
          </tr>
        </thead>
        <tbody>
          {breakdown.map((b: any, i: number) => {
            const status = String(b.status || 'Pending')
            const statusColor =
              status === 'Allocated' || status === 'Printed Available' || status === 'Grey Allocated'
                ? 'text-green-700 bg-green-50'
                : status === 'Partial' || status === 'Partial Printed'
                  ? 'text-amber-700 bg-amber-50'
                  : status === 'Locked-Cut'
                    ? 'text-blue-700 bg-blue-100'
                    : status === '—'
                      ? 'text-gray-500 bg-gray-50'
                      : 'text-red-700 bg-red-50'
            const alloc = Number(b.allocated_qty ?? 0)
            const key = `${b.so_no || b.so_number || ''}|${b.sku || b.fg_sku || ''}|${i}`
            return (
              <tr key={i} className="border-t border-blue-100">
                <td className="py-1 pr-3 font-semibold text-[#002B5B]">{b.so_no}</td>
                <td className="py-1 pr-3 font-mono text-gray-700">{b.sku || b.fg_sku || '—'}</td>
                <td className="py-1 pr-3 font-mono text-[#002B5B]">
                  {b.p_code || b.printed_code || '—'}
                </td>
                <td className="py-1 pr-3 text-right font-semibold">
                  {b.qty_req} {unit}
                </td>
                <td className="py-1 pr-3 text-right font-semibold text-gray-700">
                  {alloc} {unit}
                </td>
                <td className="py-1 pr-3 text-right" onClick={e => e.stopPropagation()}>
                  <input
                    type="number"
                    min={0}
                    step="0.001"
                    value={drafts[key] ?? ''}
                    placeholder="0"
                    onChange={e => setDrafts(d => ({ ...d, [key]: e.target.value }))}
                    className="w-24 border border-gray-200 rounded px-1.5 py-0.5 text-right font-mono"
                  />
                </td>
                <td className="py-1">
                  <span className={`inline-block px-1.5 py-0.5 rounded text-[11px] font-semibold ${statusColor}`}>
                    {status}
                  </span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div className="flex flex-wrap items-center gap-2" onClick={e => e.stopPropagation()}>
        <button
          type="button"
          disabled={saving || over || draftTotal <= 0}
          onClick={save}
          className="text-xs px-3 py-1.5 rounded-lg bg-[#002B5B] text-white font-medium hover:bg-blue-900 disabled:opacity-40"
        >
          {saving ? 'Saving…' : 'Save allocations'}
        </button>
        <a
          href="/grey?tab=planning&plan=allocate"
          className="inline-flex items-center gap-1 text-xs px-3 py-1.5 rounded-lg border border-gray-300 text-gray-700 font-medium hover:bg-gray-50"
        >
          Advanced Grey / Printed →
        </a>
        {msg && <span className={`text-xs ${msg.toLowerCase().includes('fail') || msg.toLowerCase().includes('exceed') ? 'text-red-600' : 'text-green-700'}`}>{msg}</span>}
      </div>
    </div>
  )
}

type MRPTabProps = {
  onCreateJO?: (p: { so_number: string; fabric_code: string; fabric_name: string; fabric_qty: number }) => void
}

function MRPTab({ onCreateJO }: MRPTabProps) {
  const qc = useQueryClient()
  const [selectedSOs, setSelectedSOs] = useState<string[]>([])
  const [mrpResult, setMrpResult] = useState<any>(null)
  const [running, setRunning] = useState(false)
  const [expandedMat, setExpandedMat] = useState<string | null>(null)
  const [auditSO, setAuditSO] = useState('')
  const [auditData, setAuditData] = useState<any>(null)
  const [auditLoading, setAuditLoading] = useState(false)

  const { data: openSOs = [] } = useQuery({
    queryKey: ['mrp-open-sos'],
    queryFn: () => api.get('/production/mrp/open-sos').then(r => r.data),
  })
  const { data: lastMRP } = useQuery({
    queryKey: ['mrp-last'],
    queryFn: () => api.get('/production/mrp/last').then(r => r.data),
  })
  const { data: greyStock = [] } = useQuery<any[]>({
    queryKey: ['grey-planning-stock'],
    queryFn: () => api.get('/grey/planning/grey-stock').then(r => r.data),
    staleTime: 15_000,
  })

  const activeSONumbers: string[] = mrpResult?.so_numbers || lastMRP?.so_numbers || []
  const embroideryPreviewSos = selectedSOs.length > 0 ? selectedSOs : activeSONumbers

  const { data: embStockData } = useQuery({
    queryKey: ['mrp-embroidery-stock', embroideryPreviewSos.join(',')],
    queryFn: () =>
      api
        .get(`/production/mrp/embroidery-stock?so_numbers=${encodeURIComponent(embroideryPreviewSos.join(','))}`)
        .then(r => r.data),
    enabled: embroideryPreviewSos.length > 0,
  })

  const embroideryStock: any[] =
    (mrpResult?.embroidery_stock as any[]) ||
    (embStockData?.items as any[]) ||
    (lastMRP?.embroidery_stock as any[]) ||
    []

  const loadAudit = async () => {
    if (!auditSO.trim()) return
    setAuditLoading(true)
    try {
      const res = await api.get(`/production/mrp/audit-chain?so_number=${encodeURIComponent(auditSO.trim())}`)
      setAuditData(res.data)
    } catch {
      setAuditData(null)
      alert('Could not load document chain')
    }
    setAuditLoading(false)
  }

  const runMRP = async () => {
    if (!selectedSOs.length) { alert('Select at least one SO'); return }
    setRunning(true)
    try {
      const res = await api.post('/production/mrp/run', { so_numbers: selectedSOs })
      setMrpResult(res.data)
      qc.invalidateQueries({ queryKey: ['mrp-last'] })
      qc.invalidateQueries({ queryKey: ['mrp-embroidery-stock'] })
    } catch (e) { alert('Material requirement planning run failed') }
    setRunning(false)
  }

  const toggleSO = (so: string) => setSelectedSOs(s => s.includes(so) ? s.filter(x => x !== so) : [...s, so])

  const result = mrpResult?.result || lastMRP?.result || {}
  const materials = Object.entries(result) as [string, any][]
  const materialsToProcure = materials.filter(([, mat]) => (mat?.net_req ?? 0) > 0)
  const warnings: string[] = (mrpResult?.warnings || lastMRP?.warnings || []) as string[]
  const matchedSOs: string[] = (mrpResult?.matched_sos || lastMRP?.matched_sos || []) as string[]
  const showWarnings = warnings.length > 0 && (mrpResult || lastMRP?.run_time)

  return (
    <div className="space-y-4">
      {/* SO Selection */}
      <div className="bg-white rounded-xl border p-4">
        <div className="flex justify-between items-center mb-3">
          <h3 className="font-semibold text-gray-700">📐 Material Requirement Planning</h3>
          <button onClick={runMRP} disabled={running || !selectedSOs.length}
            className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
            {running ? '⏳ Running…' : '▶️ Run planning'}
          </button>
        </div>
        <p className="text-xs text-gray-500 mb-3">Select SOs for material requirement calculation:</p>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
          {(openSOs as any[]).map((so: any) => (
            <button key={so.so_number} onClick={() => toggleSO(so.so_number)}
              className={`text-left border rounded-lg px-3 py-2 text-xs transition-colors ${selectedSOs.includes(so.so_number) ? 'bg-[#002B5B] text-white border-[#002B5B]' : 'bg-white hover:bg-gray-50'}`}>
              <p className="font-semibold">{so.so_number}</p>
              <p className={selectedSOs.includes(so.so_number) ? 'text-blue-200' : 'text-gray-400'}>
                {so.buyer} · {so.pending_qty} pcs pending
              </p>
            </button>
          ))}
          {(openSOs as any[]).length === 0 && <p className="text-gray-400 text-sm col-span-3">No open SOs</p>}
        </div>
        {lastMRP?.run_time && !mrpResult && (
          <p className="text-xs text-gray-400 mt-2">Last run: {lastMRP.run_time} · SOs: {lastMRP.so_numbers?.join(', ')}</p>
        )}
      </div>

      {/* Embroidery leftovers — show before placing material PO / JO for these styles */}
      {embroideryPreviewSos.length > 0 && (
        <div className="bg-white rounded-xl border overflow-hidden">
          <div className="px-4 py-3 bg-teal-800 text-white flex justify-between items-center">
            <span className="font-semibold text-sm">
              Embroidery leftovers in stock
              {embroideryStock.length > 0
                ? ` — ${embroideryStock.length} item${embroideryStock.length === 1 ? '' : 's'}`
                : ''}
            </span>
            <span className="text-teal-100 text-xs">Use before placing next PO / embroidery JO</span>
          </div>
          {embroideryStock.length === 0 ? (
            <p className="px-4 py-3 text-xs text-gray-500">
              No leftover Border / Yog / Boota stock for the selected SO styles.
            </p>
          ) : (
            <table className="w-full text-sm">
              <thead className="text-gray-400 text-xs uppercase bg-teal-50">
                <tr>
                  <th className="text-left px-4 py-2">Style</th>
                  <th className="text-left px-4 py-2">Part</th>
                  <th className="text-left px-4 py-2">Type</th>
                  <th className="text-right px-4 py-2">Available</th>
                  <th className="text-left px-4 py-2">Unit</th>
                  <th className="text-left px-4 py-2">SKUs</th>
                  <th className="text-left px-4 py-2">Notes</th>
                </tr>
              </thead>
              <tbody>
                {embroideryStock.map((item: any, idx: number) => (
                  <tr key={`${item.style_key}-${item.component_code}-${item.embroidery_type}-${idx}`} className="border-t hover:bg-teal-50/40">
                    <td className="px-4 py-2 font-mono font-semibold text-xs text-[#002B5B]">{item.style_key}</td>
                    <td className="px-4 py-2 text-xs">{item.component_code || '—'}</td>
                    <td className="px-4 py-2 text-xs font-semibold">{item.embroidery_type || '—'}</td>
                    <td className="px-4 py-2 text-right font-bold text-teal-800">
                      {Number(item.available_qty || 0)}
                    </td>
                    <td className="px-4 py-2 text-xs text-gray-500">{item.unit || 'PCS'}</td>
                    <td className="px-4 py-2 text-xs font-mono text-gray-600">
                      {(item.sample_skus || []).join(', ') || '—'}
                    </td>
                    <td className="px-4 py-2 text-xs text-gray-500 max-w-[220px] truncate" title={item.remarks || ''}>
                      {item.remarks || (item.so_number ? `SO ${item.so_number}` : '—')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {embroideryStock.length > 0 && (
            <p className="px-4 py-2 text-xs text-teal-900 bg-teal-50 border-t border-teal-100">
              These leftovers will be applied automatically when the next Embroidery job order is created for the same style/part.
              Reduce any new Border / Yog purchase accordingly.
            </p>
          )}
        </div>
      )}

      {/* Warnings — show when SOs/SKUs couldn't be exploded so the user knows what to fix. */}
      {showWarnings && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
          <p className="text-sm font-semibold text-amber-900 mb-2">⚠️ Planning could not generate rows for some lines</p>
          {matchedSOs.length > 0 && (
            <p className="text-xs text-amber-800 mb-2">
              Materials below cover SOs: <span className="font-mono">{matchedSOs.join(', ')}</span>.
            </p>
          )}
          <ul className="list-disc list-inside text-xs text-amber-900 space-y-0.5">
            {warnings.slice(0, 12).map((w, i) => <li key={i}>{w}</li>)}
            {warnings.length > 12 && <li className="text-amber-700">…and {warnings.length - 12} more</li>}
          </ul>
          <p className="text-xs text-amber-800 mt-2">
            Add the missing SKU/parent style in <strong>Item Master</strong> with a default <strong>BOM</strong>, then re-run planning.
          </p>
        </div>
      )}

      {materials.length > 0 && (mrpResult || lastMRP?.run_time) && (
        <div className="bg-white rounded-xl border overflow-hidden">
          <div className="px-4 py-3 bg-[#002B5B] text-white flex justify-between items-center">
            <span className="font-semibold">
              Material requirements — {materials.length} material{materials.length === 1 ? '' : 's'}
              {materialsToProcure.length > 0
                ? ` (${materialsToProcure.length} to procure)`
                : ' (all covered by stock)'}
            </span>
            <span className="text-blue-200 text-xs">{mrpResult?.run_time || lastMRP?.run_time}</span>
          </div>
          {materialsToProcure.length === 0 && (
            <p className="px-4 py-2 text-xs text-green-700 bg-green-50 border-b border-green-100">
              All material requirements are covered by available stock — nothing to procure for this planning run.
            </p>
          )}
          <table className="w-full text-sm">
            <thead className="text-gray-400 text-xs uppercase bg-gray-50">
              <tr>
                <th className="text-left px-4 py-2">Material</th>
                <th className="text-right px-4 py-2">Total Req</th>
                <th className="text-right px-4 py-2">Stock</th>
                <th className="text-right px-4 py-2">Available</th>
                <th className="text-right px-4 py-2">Net Req</th>
                <th className="text-left px-4 py-2">Unit</th>
                <th className="text-right px-4 py-2">Action</th>
              </tr>
            </thead>
            <tbody>
              {materials.sort((a,b) => (b[1].net_req||0) - (a[1].net_req||0)).map(([code, mat]) => {
                const firstSo = (mat.breakdown?.[0]?.so_no as string) || activeSONumbers[0] || ''
                const netReq = mat.net_req ?? 0
                const isFabric = (mat.unit || '').toUpperCase() === 'MTR' || ['GF', 'RM', 'SFG', 'Fabric'].some(t => (mat.type || '').toUpperCase().includes(t))
                const canJO = isFabric && netReq > 0 && !!onCreateJO && !!firstSo
                return (
                <>
                  <tr key={code} className="border-t hover:bg-gray-50 cursor-pointer" onClick={() => setExpandedMat(expandedMat === code ? null : code)}>
                    <td className="px-4 py-2">
                      <div className="flex items-center gap-2">
                        <span className="text-gray-400 text-xs">{expandedMat === code ? '▼' : '▶'}</span>
                        <div>
                          <p className="font-mono font-semibold text-xs text-[#002B5B]">{code}</p>
                          <p className="text-xs text-gray-500">{mat.name}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-2 text-right font-semibold">{mat.total_req}</td>
                    <td className="px-4 py-2 text-right">{mat.stock || 0}</td>
                    <td className="px-4 py-2 text-right text-green-600">{mat.available || 0}</td>
                    <td className={`px-4 py-2 text-right font-bold ${netReq > 0 ? 'text-red-600' : 'text-green-600'}`}>
                      {netReq || 0}
                    </td>
                    <td className="px-4 py-2 text-gray-500 text-xs">{mat.unit}</td>
                    <td className="px-4 py-2 text-right" onClick={e => e.stopPropagation()}>
                      {isFabric && onCreateJO && (
                        <button
                          type="button"
                          disabled={!canJO}
                          title={netReq <= 0 ? 'No net requirement for this fabric' : `Create Cutting JO for ${firstSo}`}
                          onClick={() => onCreateJO({
                            so_number: firstSo,
                            fabric_code: code,
                            fabric_name: mat.name || code,
                            fabric_qty: Math.max(0, netReq),
                          })}
                          className="text-xs px-2 py-1 bg-[#002B5B] text-white rounded disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                          Create JO
                        </button>
                      )}
                    </td>
                  </tr>
                  {expandedMat === code && mat.breakdown && (
                    <tr key={`${code}-breakdown`}>
                      <td colSpan={7} className="px-4 py-0 bg-blue-50">
                        {canInlineGreyAlloc(mat, code) ? (
                          <GreyInlineAllocPanel
                            materialCode={code}
                            mat={mat}
                            greyFreeQty={(() => {
                              const snap = (greyStock as any[]).find(
                                (s: any) => String(s.fabric_code || '').trim().toUpperCase() === String(code).trim().toUpperCase(),
                              )
                              const free = Number(snap?.grey_free_qty ?? snap?.available_qty)
                              if (Number.isFinite(free) && free >= 0) return free
                              return Number(mat.available || 0)
                            })()}
                            onSaved={async () => {
                              qc.invalidateQueries({ queryKey: ['mrp-last'] })
                              qc.invalidateQueries({ queryKey: ['grey-planning-stock'] })
                              try {
                                const res = await api.get('/production/mrp/last')
                                setMrpResult(res.data)
                              } catch { /* keep current MRP rows */ }
                            }}
                          />
                        ) : (
                        <div className="py-2 space-y-1">
                          <p className="text-xs font-semibold text-gray-500 uppercase mb-1">
                            Breakdown — Grey → P-Code → FG hierarchy:
                          </p>
                          <p className="text-[11px] text-gray-500 mb-1">
                            Allocation is planned at P-Code; FG stays visible for full SKU status traceability.
                          </p>
                          <div className="flex flex-wrap gap-2 mb-2">
                            <a
                              href="/grey?tab=planning&plan=tree"
                              className="inline-flex items-center gap-1 text-xs px-3 py-1.5 rounded-lg border border-gray-300 text-gray-700 font-medium hover:bg-gray-50"
                            >
                              Planning tree
                            </a>
                          </div>
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="text-gray-400">
                                <th className="text-left py-1 pr-3">SO Number</th>
                                <th className="text-left py-1 pr-3">FG SKU</th>
                                <th className="text-left py-1 pr-3">P-Code / Printed Fabric</th>
                                <th className="text-right py-1 pr-3">Required Qty</th>
                                <th className="text-right py-1 pr-3">Allocated Qty</th>
                                <th className="text-left py-1">Status</th>
                              </tr>
                            </thead>
                            <tbody>
                              {mat.breakdown.map((b: any, i: number) => {
                                const status = String(b.status || 'Pending')
                                const statusColor =
                                  status === 'Allocated' || status === 'Printed Available' || status === 'Grey Allocated'
                                    ? 'text-green-700 bg-green-50'
                                    : status === 'Partial' || status === 'Partial Printed'
                                      ? 'text-amber-700 bg-amber-50'
                                      : status === 'Locked-Cut'
                                        ? 'text-blue-700 bg-blue-100'
                                        : status === '—'
                                          ? 'text-gray-500 bg-gray-50'
                                          : 'text-red-700 bg-red-50'
                                const alloc = Number(b.allocated_qty ?? 0)
                                return (
                                <tr key={i} className="border-t border-blue-100">
                                  <td className="py-1 pr-3 font-semibold text-[#002B5B]">{b.so_no}</td>
                                  <td className="py-1 pr-3 font-mono text-gray-700">{b.sku || b.fg_sku || '—'}</td>
                                  <td className="py-1 pr-3 font-mono text-[#002B5B]">
                                    {b.p_code || b.printed_code || '—'}
                                  </td>
                                  <td className="py-1 pr-3 text-right font-semibold">
                                    {b.qty_req} {mat.unit}
                                  </td>
                                  <td className="py-1 pr-3 text-right font-semibold text-gray-700">
                                    {alloc} {mat.unit}
                                  </td>
                                  <td className="py-1">
                                    <span className={`inline-block px-1.5 py-0.5 rounded text-[11px] font-semibold ${statusColor}`}>
                                      {status}
                                    </span>
                                  </td>
                                </tr>
                              )})}
                            </tbody>
                          </table>
                        </div>
                        )}
                      </td>
                    </tr>
                  )}
                </>
              )})}
            </tbody>
          </table>
        </div>
      )}

      {activeSONumbers.length > 0 && (
        <div className="bg-white rounded-xl border p-4 space-y-3">
          <h3 className="font-semibold text-gray-700">Document chain audit (Material planning → PO → GRN → Grey ledger)</h3>
          <div className="flex flex-wrap gap-2 items-end">
            <div>
              <label className="text-xs text-gray-500">SO number</label>
              <select className="block border rounded px-2 py-1.5 text-sm mt-0.5" value={auditSO} onChange={e => setAuditSO(e.target.value)}>
                <option value="">Select SO…</option>
                {activeSONumbers.map(so => <option key={so} value={so}>{so}</option>)}
              </select>
            </div>
            <button onClick={loadAudit} disabled={!auditSO || auditLoading}
              className="px-3 py-2 bg-slate-700 text-white rounded-lg text-sm disabled:opacity-50">
              {auditLoading ? 'Loading…' : 'Load chain'}
            </button>
          </div>
          {auditData?.materials?.length > 0 && (
            <div className="space-y-3 max-h-[420px] overflow-y-auto">
              {auditData.materials.map((m: any) => (
                <div key={m.material_code} className="border rounded-lg p-3 text-xs">
                  <p className="font-mono font-bold text-[#002B5B]">{m.material_code}</p>
                  <p className="text-gray-500 mb-2">
                    Planned {m.mrp_qty} · PO committed {m.po_committed_qty} · JO fabric {m.jo_committed_qty} · Remaining {m.remaining_qty}
                  </p>
                  <div className="grid md:grid-cols-2 gap-2 text-gray-600">
                    <div><span className="font-semibold">POs:</span> {(m.pos || []).map((p: any) => p.po_number).join(', ') || '—'}</div>
                    <div><span className="font-semibold">GRNs:</span> {(m.grns || []).map((g: any) => `${g.grn_number}(${g.accepted_qty})`).join(', ') || '—'}</div>
                    <div><span className="font-semibold">JOs:</span> {(m.job_orders || []).map((j: any) => j.jo_number).join(', ') || '—'}</div>
                    <div><span className="font-semibold">Grey:</span> {(m.grey_trackers || []).map((t: any) => `${t.tracker_key}:${t.status}`).join(', ') || '—'}</div>
                  </div>
                  {(m.grey_ledger || []).length > 0 && (
                    <table className="w-full mt-2 border-t pt-2">
                      <thead><tr className="text-gray-400"><th className="text-left">Date</th><th className="text-left">Type</th><th className="text-right">Qty</th><th className="text-left">From→To</th></tr></thead>
                      <tbody>
                        {m.grey_ledger.slice(0, 8).map((l: any, i: number) => (
                          <tr key={i} className="border-t border-gray-100">
                            <td>{l.entry_date}</td><td>{l.transaction_type}</td><td className="text-right">{l.qty}</td>
                            <td>{l.from_location} → {l.to_location}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function Production() {
  const qc = useQueryClient()
  const [activeProcess, setActiveProcess] = useState('Cutting')
  const [tab, setTab] = useState<'process' | 'tracker' | 'issue-notes' | 'reports' | 'mrp' | 'sets'>('process')
  const [expandedIssueNote, setExpandedIssueNote] = useState<number | null>(null)
  const [expanded, setExpanded] = useState<number | null>(null)
  const [filterStatus, setFilterStatus] = useState('')
  const [listSearch, setListSearch] = useState('')
  const [filterSO, setFilterSO] = useState('')
  const [filterSku, setFilterSku] = useState('')
  const [filterVendor, setFilterVendor] = useState('')
  const [filterJO, setFilterJO] = useState('')
  const [filterMinQty, setFilterMinQty] = useState('')
  const [filterDateFrom, setFilterDateFrom] = useState('')
  const [filterDateTo, setFilterDateTo] = useState('')
  const [sortBy, setSortBy] = useState<'so_number' | 'sku' | 'vendor_name' | 'available_qty' | 'jo_date' | 'status'>('so_number')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')
  const [modal, setModal] = useState<ModalType>(null)
  const [activeJO, setActiveJO] = useState<JO | null>(null)
  const [activeLineId, setActiveLineId] = useState<number | null>(null)

  // New JO form — so_source: system (open SO) | manual (free-text SO, no master SO created)
  const [newForm, setNewForm] = useState({
    so_number: '', so_source: 'system' as 'system' | 'manual', sku: '', sku_name: '', process: 'Cutting',
    exec_type: 'Inhouse', vendor_name: '', vendor_rate: 0,
    planned_qty: 0, so_qty: 0, fabric_code: '', fabric_qty: 0,
    fabric_unit: 'MTR', expected_completion: '', remarks: '',
    production_mode: 'inhouse',
    from_ready_to: false,
  })
  const [editPlannedQty, setEditPlannedQty] = useState<Record<number, string>>({})
  const [editLineQty, setEditLineQty] = useState<Record<number, string>>({})
  const [reportsView, setReportsView] = useState<'cutting' | 'process'>('cutting')
  const [newLines, setNewLines] = useState<{ so_number: string; sku: string; sku_name: string; style: string; planned_qty: number; vendor_rate: number; remarks: string; so_qty?: number }[]>([])
  const [soLineSearch, setSOLineSearch] = useState('')
  const joImportRef = useRef<HTMLInputElement>(null)
  const wipImportRef = useRef<HTMLInputElement>(null)

  // Modal forms
  const [fabricIssueForm, setFabricIssueForm] = useState({ fabric_code: '', fabric_name: '', issued_qty: 0, unit: 'MTR', issued_by: '', remarks: '' })
  const [fabricReturnForm, setFabricReturnForm] = useState({ fabric_code: '', returned_qty: 0, unit: 'MTR', returned_by: '', remarks: '' })
  const [receiveForm, setReceiveForm] = useState({ received_qty: 0, rejected_qty: 0, received_by: '', remarks: '', split_components: true, leftover_measurement: 0 })
  const [issuePiecesForm, setIssuePiecesForm] = useState({ issued_qty: 0, to_process: '', issued_by: '', remarks: '' })
  const [issuePiecesSku, setIssuePiecesSku] = useState('')
  const [issueFromProcess, setIssueFromProcess] = useState('')
  const [costForm, setCostForm] = useState({ cost_type: 'Labour', amount: 0, description: '' })

  // URL params auto-fill
  useEffect(() => {
    const p = new URLSearchParams(window.location.search)
    const fabric = p.get('fabric'), qty = p.get('qty'), so = p.get('so'), sku = p.get('sku')
    if (fabric) {
      setNewForm(f => ({ ...f, fabric_code: fabric, fabric_qty: parseFloat(qty||'0'), so_number: so||'', sku: sku||'', process: 'Cutting' }))
      setActiveProcess('Cutting')
      setTab('process')
      setModal('new-jo')
    }
  }, [])

  // ── Queries ──────────────────────────────────────────────────────────────────
  const { data: processes = [] } = useQuery<string[]>({
    queryKey: ['processes'],
    queryFn: () => api.get('/production/processes').then(r => r.data),
  })
  const { data: stats } = useQuery({
    queryKey: ['prod-stats'],
    queryFn: () => api.get('/production/stats').then(r => r.data),
  })
  const { data: processJOs = [], isLoading: josLoading, isFetching: josFetching, isError: josError, error: josErr } = useQuery<JO[]>({
    queryKey: ['jos-process', activeProcess, filterStatus],
    queryFn: () => api.get(`/production/orders?process=${encodeURIComponent(activeProcess)}${filterStatus ? `&status=${filterStatus}` : ''}&light=1`).then(r => r.data),
    enabled: tab === 'process',
    staleTime: 30_000,
  })
  const { data: allJOs = [], isLoading: allJosLoading } = useQuery<JO[]>({
    queryKey: ['jos-all', filterStatus],
    queryFn: () => api.get(`/production/orders${filterStatus ? `?status=${filterStatus}` : ''}&light=1`).then(r => r.data),
    enabled: tab === 'tracker',
    staleTime: 30_000,
  })
  const { data: readyLines = [] } = useQuery({
    queryKey: ['ready-to-process', activeProcess, filterJO, filterSku, filterVendor, filterMinQty, filterDateFrom, filterDateTo, listSearch],
    queryFn: () => {
      const params = new URLSearchParams()
      if (listSearch.trim()) params.set('q', listSearch.trim())
      if (filterJO) params.set('jo', filterJO)
      if (filterSku) params.set('sku', filterSku)
      if (filterVendor) params.set('vendor', filterVendor)
      if (filterMinQty) params.set('min_qty', filterMinQty)
      if (filterDateFrom) params.set('date_from', filterDateFrom)
      if (filterDateTo) params.set('date_to', filterDateTo)
      const qs = params.toString()
      return api.get(`/production/ready-to-process/${encodeURIComponent(activeProcess)}${qs ? `?${qs}` : ''}`).then(r => r.data)
    },
    enabled: tab === 'process',
  })
  const { data: processReport = [] } = useQuery({
    queryKey: ['process-report'],
    queryFn: () => api.get('/production/process-report').then(r => r.data),
    enabled: tab === 'reports',
  })
  const { data: issueNotes = [] } = useQuery<IssueNote[]>({
    queryKey: ['prod-issue-notes'],
    queryFn: () => api.get('/production/issue-notes').then(r => r.data),
    enabled: tab === 'issue-notes',
  })
  const receiveSku = activeJO
    ? (activeLineId ? activeJO.lines.find(l => l.id === activeLineId)?.sku : activeJO.sku) || activeJO.sku
    : ''
  const { data: receiveSetBomInfo } = useQuery({
    queryKey: ['set-bom-for-sku', receiveSku],
    queryFn: () => api.get(`/production/set-bom-for-sku/${encodeURIComponent(receiveSku)}`).then(r => r.data),
    enabled: modal === 'receive' && !!receiveSku && activeJO?.process === 'Cutting',
  })
  const { data: expandedPanelWip } = useQuery({
    queryKey: ['jo-panel-wip', expanded],
    queryFn: () => api.get(`/production/orders/${expanded}/panel-wip`).then(r => r.data),
    enabled: expanded != null,
  })
  const { data: expandedJoDetail } = useQuery<JO>({
    queryKey: ['jo-detail', expanded],
    queryFn: () => api.get(`/production/orders/${expanded}`).then(r => r.data),
    enabled: expanded != null,
    staleTime: 15_000,
  })
  const { data: soList = [] } = useQuery({
    queryKey: ['so-list'],
    queryFn: () => api.get('/sales/orders').then(r => r.data || []),
  })
  const { data: soLines = [] } = useQuery({
    queryKey: ['so-lines', newForm.so_number],
    queryFn: () => api.get('/sales/orders').then(r => {
      const so = (r.data || []).find((s: any) => s.so_number === newForm.so_number)
      return so?.lines || []
    }),
    enabled: !!newForm.so_number,
    staleTime: 0,
  })
  const readyLinesForJo = useMemo(() => {
    const so = String(newForm.so_number || '')
    return (readyLines as any[])
      .filter(r => String(r.so_number || '') === so)
      .map(r => ({
        sku: r.sku,
        sku_name: r.sku_name || r.item_name || '',
        qty: Number(r.available_qty || r.reserved_qty || 0),
        available_qty: Number(r.available_qty || r.reserved_qty || 0),
      }))
  }, [readyLines, newForm.so_number])
  const joLinePickerRows = useMemo(() => {
    const rows = newForm.from_ready_to ? readyLinesForJo : (soLines as any[])
    const q = soLineSearch.trim().toLowerCase()
    if (!q) return rows
    return rows.filter((l: any) =>
      String(l.sku || '').toLowerCase().includes(q)
      || String(l.sku_name || l.item_name || '').toLowerCase().includes(q),
    )
  }, [newForm.from_ready_to, readyLinesForJo, soLines, soLineSearch])
  const { data: itemRouting } = useQuery({
    queryKey: ['item-routing', newForm.sku, newForm.so_number, newForm.production_mode, newForm.process],
    queryFn: () => api.get(`/production/item-routing/${encodeURIComponent(newForm.sku)}`, {
      params: {
        ...(newForm.so_number ? { so_number: newForm.so_number } : {}),
        ...(newForm.production_mode ? { production_mode: newForm.production_mode } : {}),
        ...(newForm.process ? { process: newForm.process } : {}),
      },
    }).then(r => r.data),
    enabled: !!newForm.sku,
  })
  const { data: pathCommitment } = useQuery({
    queryKey: ['path-commitment', newForm.so_number, newForm.sku, newForm.process],
    queryFn: () => api.get('/production/path-commitment', {
      params: { so_number: newForm.so_number, sku: newForm.sku, process: newForm.process },
    }).then(r => r.data),
    enabled: modal === 'new-jo' && !!newForm.so_number && !!newForm.sku && newForm.so_source === 'system',
  })
  useEffect(() => {
    if (!newForm.production_mode || !newForm.process) return
    const exec = suggestedExecType(newForm.production_mode, newForm.process)
    setNewForm(f => (f.exec_type === exec ? f : { ...f, exec_type: exec }))
  }, [newForm.production_mode, newForm.process])
  const { data: processors = [] } = useQuery<{ processor_name?: string }[]>({
    queryKey: ['purchase-processors'],
    queryFn: () => api.get('/purchase/processors').then(r => r.data),
  })
  const vendorSuggestions = [...new Set(
    (processors || [])
      .map(p => String(p.processor_name || '').trim())
      .filter(Boolean),
  )].sort((a, b) => a.localeCompare(b))

  const soBuyerMap = useMemo(() => {
    const m = new Map<string, string>()
    for (const so of soList as { so_number?: string; buyer?: string; customer?: string }[]) {
      const key = String(so.so_number || '').trim()
      if (!key) continue
      m.set(key, String(so.buyer || so.customer || '').trim())
    }
    return m
  }, [soList])

  const soOptions = useMemo(() => {
    const fromJO = processJOs.map(j => j.so_number).concat(allJOs.map(j => j.so_number))
    const fromReady = (readyLines as { so_number?: string }[]).map(r => String(r.so_number || ''))
    const fromSO = (soList as { so_number?: string }[]).map(s => String(s.so_number || ''))
    return [...new Set([...fromJO, ...fromReady, ...fromSO].map(s => s.trim()).filter(Boolean))]
      .sort((a, b) => a.localeCompare(b))
  }, [processJOs, allJOs, readyLines, soList])

  const skuOptions = useMemo(() => {
    const fromJO = processJOs.flatMap(j => joLineSkus(j)).concat(allJOs.flatMap(j => joLineSkus(j)))
    const fromReady = (readyLines as { sku?: string }[]).map(r => String(r.sku || ''))
    return [...new Set([...fromJO, ...fromReady].map(s => s.trim()).filter(Boolean))]
      .sort((a, b) => a.localeCompare(b))
  }, [processJOs, allJOs, readyLines])

  /** SKU catalog for Manual SO JO dropdown (SOs + JOs + ready lines). */
  const manualJoSkuCatalog = useMemo(() => {
    const map = new Map<string, MasterSkuOption>()
    const add = (sku: string, name: string, source: string) => {
      const s = String(sku || '').trim()
      if (!s) return
      const k = s.toUpperCase()
      if (map.has(k)) return
      map.set(k, { sku: s, sku_name: String(name || '').trim(), source })
    }
    for (const so of soList as { lines?: { sku?: string; sku_name?: string; item_name?: string }[] }[]) {
      for (const ln of so.lines || []) {
        add(String(ln.sku || ''), String(ln.sku_name || ln.item_name || ''), 'sales')
      }
    }
    for (const j of [...processJOs, ...allJOs] as JO[]) {
      add(j.sku, j.sku_name || '', 'jo')
    }
    for (const r of readyLines as { sku?: string; sku_name?: string }[]) {
      add(String(r.sku || ''), String(r.sku_name || ''), 'ready')
    }
    return [...map.values()].sort((a, b) => a.sku.localeCompare(b.sku))
  }, [soList, processJOs, allJOs, readyLines])

  const vendorOptions = useMemo(() => {
    const fromJO = processJOs.map(j => j.vendor_name).concat(allJOs.map(j => j.vendor_name))
    const fromSO = [...soBuyerMap.values()]
    return [...new Set([...fromJO, ...fromSO, ...vendorSuggestions].map(s => String(s || '').trim()).filter(Boolean))]
      .sort((a, b) => a.localeCompare(b))
  }, [processJOs, allJOs, soBuyerMap, vendorSuggestions])

  const matchesListQuery = (parts: Array<string | number | null | undefined>) => {
    const q = listSearch.trim().toLowerCase()
    if (!q) return true
    return parts.some(p => String(p ?? '').toLowerCase().includes(q))
  }

  const filteredReadyLines = useMemo(() => {
    let rows = (readyLines as any[]).map(r => ({
      ...r,
      vendor_name: r.vendor_name || soBuyerMap.get(String(r.so_number || '').trim()) || '',
    }))
    if (filterSO) rows = rows.filter(r => String(r.so_number || '') === filterSO)
    if (filterSku) rows = rows.filter(r => String(r.sku || '') === filterSku)
    if (filterJO) {
      const j = filterJO.toUpperCase()
      rows = rows.filter(r =>
        String(r.jo_number || '').toUpperCase().includes(j)
        || (Array.isArray(r.jo_numbers) && r.jo_numbers.some((x: string) => String(x).toUpperCase().includes(j)))
      )
    }
    if (filterVendor) {
      rows = rows.filter(r => String(r.vendor_name || '').toLowerCase() === filterVendor.toLowerCase())
    }
    if (filterMinQty) {
      const mq = Number(filterMinQty)
      if (!Number.isNaN(mq)) rows = rows.filter(r => Number(r.available_qty || r.reserved_qty || 0) >= mq)
    }
    if (filterDateFrom) rows = rows.filter(r => String(r.updated_at || '').slice(0, 10) >= filterDateFrom)
    if (filterDateTo) rows = rows.filter(r => !r.updated_at || String(r.updated_at).slice(0, 10) <= filterDateTo)
    rows = rows.filter(r =>
      matchesListQuery([r.so_number, r.sku, r.vendor_name, r.fabric_code, r.fabric_name, r.jo_number, r.from_process, r.to_process, r.batch]),
    )
    const dir = sortDir === 'asc' ? 1 : -1
    rows = [...rows].sort((a, b) => {
      if (sortBy === 'available_qty') {
        return (Number(a.available_qty || a.reserved_qty || 0) - Number(b.available_qty || b.reserved_qty || 0)) * dir
      }
      const ak = String(a[sortBy === 'vendor_name' ? 'vendor_name' : sortBy === 'sku' ? 'sku' : 'so_number'] || '')
      const bk = String(b[sortBy === 'vendor_name' ? 'vendor_name' : sortBy === 'sku' ? 'sku' : 'so_number'] || '')
      return ak.localeCompare(bk, undefined, { numeric: true }) * dir
    })
    return rows
  }, [readyLines, soBuyerMap, filterSO, filterSku, filterVendor, filterJO, filterMinQty, filterDateFrom, filterDateTo, listSearch, sortBy, sortDir])

  const filteredProcessJOs = useMemo(() => {
    let rows = [...processJOs]
    if (filterSO) rows = rows.filter(j => j.so_number === filterSO)
    if (filterSku) rows = rows.filter(j => joMatchesSkuFilter(j, filterSku))
    if (filterVendor) {
      rows = rows.filter(j =>
        String(j.vendor_name || '').toLowerCase() === filterVendor.toLowerCase()
        || String(soBuyerMap.get(j.so_number) || '').toLowerCase() === filterVendor.toLowerCase(),
      )
    }
    rows = rows.filter(j =>
      matchesListQuery([
        j.jo_number, j.so_number, j.sku, j.sku_name, j.vendor_name, soBuyerMap.get(j.so_number),
        ...joLineSkus(j),
      ]),
    )
    const dir = sortDir === 'asc' ? 1 : -1
    rows.sort((a, b) => {
      if (sortBy === 'jo_date') return String(a.jo_date || '').localeCompare(String(b.jo_date || '')) * dir
      if (sortBy === 'status') return String(a.status || '').localeCompare(String(b.status || '')) * dir
      if (sortBy === 'available_qty') return ((a.balance_qty || 0) - (b.balance_qty || 0)) * dir
      const key = sortBy === 'vendor_name' ? 'vendor_name' : sortBy === 'sku' ? 'sku' : 'so_number'
      return String(a[key] || '').localeCompare(String(b[key] || ''), undefined, { numeric: true }) * dir
    })
    return rows
  }, [processJOs, filterSO, filterSku, filterVendor, listSearch, sortBy, sortDir, soBuyerMap])

  const filteredAllJOs = useMemo(() => {
    let rows = [...allJOs]
    if (filterSO) rows = rows.filter(j => j.so_number === filterSO)
    if (filterSku) rows = rows.filter(j => joMatchesSkuFilter(j, filterSku))
    if (filterVendor) {
      rows = rows.filter(j =>
        String(j.vendor_name || '').toLowerCase() === filterVendor.toLowerCase()
        || String(soBuyerMap.get(j.so_number) || '').toLowerCase() === filterVendor.toLowerCase(),
      )
    }
    rows = rows.filter(j =>
      matchesListQuery([
        j.jo_number, j.so_number, j.sku, j.sku_name, j.vendor_name, j.process, soBuyerMap.get(j.so_number),
        ...joLineSkus(j),
      ]),
    )
    const dir = sortDir === 'asc' ? 1 : -1
    rows.sort((a, b) => {
      if (sortBy === 'jo_date') return String(a.jo_date || '').localeCompare(String(b.jo_date || '')) * dir
      if (sortBy === 'status') return String(a.status || '').localeCompare(String(b.status || '')) * dir
      if (sortBy === 'available_qty') return ((a.balance_qty || 0) - (b.balance_qty || 0)) * dir
      const key = sortBy === 'vendor_name' ? 'vendor_name' : sortBy === 'sku' ? 'sku' : 'so_number'
      return String(a[key] || '').localeCompare(String(b[key] || ''), undefined, { numeric: true }) * dir
    })
    return rows
  }, [allJOs, filterSO, filterSku, filterVendor, listSearch, sortBy, sortDir, soBuyerMap])

  const clearListFilters = () => {
    setListSearch('')
    setFilterSO('')
    setFilterSku('')
    setFilterVendor('')
    setFilterJO('')
    setFilterMinQty('')
    setFilterDateFrom('')
    setFilterDateTo('')
    setSortBy('so_number')
    setSortDir('asc')
  }

  const { data: joValidation } = useQuery({
    queryKey: ['jo-validate', newForm.process, newForm.so_number, newForm.sku, newForm.planned_qty],
    queryFn: () => api.get(`/production/orders/validate?process=${newForm.process}&so_number=${newForm.so_number}&sku=${newForm.sku}&planned_qty=${newForm.planned_qty}`).then(r => r.data),
    enabled: !!newForm.so_number && !!newForm.sku && newForm.process !== 'Cutting',
  })

  const cuttingMainSku = (newForm.sku || newLines[0]?.sku || '').trim()
  const { data: cuttingSetBomInfo } = useQuery({
    queryKey: ['set-bom-for-sku', cuttingMainSku],
    queryFn: () => api.get(`/production/set-bom-for-sku/${encodeURIComponent(cuttingMainSku)}`).then(r => r.data),
    enabled: modal === 'new-jo' && newForm.process === 'Cutting' && cuttingMainSku.length > 0,
  })

  const invalidateAll = () => {
    qc.invalidateQueries({ queryKey: ['prod-stats'] })
    qc.invalidateQueries({ queryKey: ['jos-process'] })
    qc.invalidateQueries({ queryKey: ['jos-all'] })
    qc.invalidateQueries({ queryKey: ['ready-to-process'] })
    qc.invalidateQueries({ queryKey: ['process-report'] })
    qc.invalidateQueries({ queryKey: ['cutting-report'] })
    qc.invalidateQueries({ queryKey: ['prod-issue-notes'] })
    qc.invalidateQueries({ queryKey: ['jo-issue-note'] })
    qc.invalidateQueries({ queryKey: ['jo-panel-wip'] })
  }

  // ── Mutations ─────────────────────────────────────────────────────────────────
  const createJOMut = useMutation({
    mutationFn: (b: object) => api.post('/production/orders', b),
    onSuccess: (res) => {
      invalidateAll()
      qc.invalidateQueries({ queryKey: ['mrp-commit-map'] })
      setModal(null)
      setNewLines([])
      const data = res?.data
      if (data?.component_jos && Array.isArray(data.orders)) {
        const nums = data.orders.map((o: any) => o.jo_number).filter(Boolean).join(', ')
        setTab('issue-notes')
        alert(`Created ${data.orders.length} component Cutting JO(s): ${nums}`)
        return
      }
      const inNum = data?.issue_note?.in_number
      if (inNum) {
        setTab('issue-notes')
        alert(`Job order created. Material issue note ${inNum} generated from BOM.`)
      }
    },
    onError: (e: unknown) => alert(apiErrorMessage(e, 'Error creating JO')),
  })
  const updateJOMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/production/orders/${id}`, data),
    onSuccess: () => {
      invalidateAll()
      setActiveJO(null)
    },
    onError: (e: unknown) => alert(apiErrorMessage(e, 'Could not update job order')),
  })
  const updateJoQtyMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/production/orders/${id}`, data),
    onSuccess: async (_res, vars) => {
      invalidateAll()
      try {
        const fresh = await api.get(`/production/orders/${vars.id}`).then(r => r.data)
        setActiveJO(fresh)
      } catch {
        /* list refresh is enough */
      }
    },
    onError: (e: unknown) => alert(apiErrorMessage(e, 'Could not update planned quantity')),
  })
  const issueFabricMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.post(`/production/orders/${id}/issue-fabric`, data),
    onSuccess: () => { invalidateAll(); setModal(null) },
    onError: (e: unknown) => alert(apiErrorMessage(e, 'Error')),
  })
  const returnFabricMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.post(`/production/orders/${id}/return-fabric`, data),
    onSuccess: () => { invalidateAll(); setModal(null) },
  })
  const receiveMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.post(`/production/orders/${id}/receive-pieces`, data),
    onSuccess: (res) => {
      invalidateAll()
      setModal(null)
      const split = res?.data?.split
      if (split?.components?.length) {
        const names = split.components.map((c: any) => `${c.component_sku}=${c.qty}`).join(', ')
        alert(`Split into components:\n${names}`)
      }
    },
    onError: (e: unknown) => alert(apiErrorMessage(e, 'Error')),
  })
  const issuePiecesMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.post(`/production/orders/${id}/issue-pieces`, data),
    onSuccess: (res) => {
      invalidateAll()
      setModal(null)
      const child = res?.data?.child_jo
      if (child?.jo_number) {
        const action = child.created ? 'Created' : 'Updated'
        const qtyPart = child.measurement_qty && child.embroidery_unit
          ? `${child.measurement_qty} ${child.embroidery_unit} ${child.embroidery_type || ''}`.trim()
          : `${child.planned_qty} pcs`
        const stockNote = child.stock_used > 0
          ? `\n(${child.stock_used} ${child.embroidery_unit || ''} applied from leftover stock)`
          : ''
        alert(
          `${action} ${child.process} JO ${child.jo_number} for ${child.sku} `
          + `(${qtyPart})${stockNote}.\nOpen the ${child.process} tab to receive / process / return.`,
        )
      }
    },
    onError: (e: unknown) => alert(apiErrorMessage(e, 'Error')),
  })
  const addCostMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.post(`/production/orders/${id}/add-cost`, data),
    onSuccess: () => { invalidateAll(); setModal(null) },
  })
  const nextProcessMut = useMutation({
    mutationFn: (id: number) => api.post(`/production/orders/${id}/next-process`, {}),
    onSuccess: (res) => { invalidateAll(); alert(`✅ Next process JO created: ${res.data.jo_number} — ${res.data.process}`) },
    onError: (e: unknown) => alert(apiErrorMessage(e, 'Error')),
  })

  const openModal = async (
    type: ModalType,
    jo: JO,
    lineId?: number,
    opts?: { sku?: string; toProcess?: string; fromProcess?: string; issuedQty?: number },
  ) => {
    setActiveJO(jo)
    setActiveLineId(lineId || null)
    setModal(type)
    if (type === 'issue-fabric') {
      const garmentPcs = garmentPcsForFabricIssue(jo)
      setFabricIssueForm(f => ({ ...f, fabric_code: jo.fabric_code || '', issued_qty: jo.fabric_qty || 0, fabric_name: '' }))
      // Sum BOM fabric across JO lines using garment pieces (not embroidery measurement qty).
      const lines = (jo.lines || []).filter(l => (l.planned_qty || 0) > 0 && l.sku)
      const anchorSku = jo.sku || lines[0]?.sku
      if (anchorSku && garmentPcs > 0) {
        try {
          const res = await api.get(`/production/bom-inputs/${encodeURIComponent(anchorSku)}`, { params: { qty: garmentPcs } })
          const inputs = (res.data?.inputs ?? []) as { material_code?: string; material_name?: string; adj_qty?: number; unit?: string }[]
          const fabric = inputs.find(i => String(i.unit || '').toUpperCase().includes('MTR')) || inputs[0]
          if (fabric?.material_code) {
            setFabricIssueForm(f => ({
              ...f,
              fabric_code: fabric.material_code || f.fabric_code,
              fabric_name: fabric.material_name || '',
              issued_qty: Number(fabric.adj_qty) || f.issued_qty,
              unit: fabric.unit || f.unit || 'MTR',
            }))
          } else if (lines.length > 1) {
            // Fallback: sum per-line BOM if anchor call returned nothing
            let sumQty = 0
            let code = jo.fabric_code || ''
            let name = ''
            let unit = 'MTR'
            for (const line of lines) {
              const linePcs = jo.process === 'Embroidery'
                ? (Number((line as JOLine & { garment_qty?: number }).garment_qty) || Number(jo.garment_qty) || Number(line.planned_qty) || 1)
                : (Number(line.planned_qty) || 1)
              const r2 = await api.get(`/production/bom-inputs/${encodeURIComponent(line.sku)}`, {
                params: { qty: linePcs },
              })
              const ins = (r2.data?.inputs ?? []) as { material_code?: string; material_name?: string; adj_qty?: number; unit?: string }[]
              const fab = ins.find(i => String(i.unit || '').toUpperCase().includes('MTR')) || ins[0]
              if (fab?.material_code) {
                code = fab.material_code
                name = fab.material_name || name
                unit = fab.unit || unit
                sumQty += Number(fab.adj_qty) || 0
              }
            }
            if (sumQty > 0) {
              setFabricIssueForm(f => ({
                ...f,
                fabric_code: code || f.fabric_code,
                fabric_name: name,
                issued_qty: sumQty,
                unit,
              }))
            }
          }
        } catch { /* keep JO fabric defaults */ }
      }
    }
    if (type === 'return-fabric') setFabricReturnForm(f => ({ ...f, fabric_code: jo.fabric_code || '' }))
    if (type === 'receive') {
      const line = jo.lines.find(l => l.id === lineId)
      setReceiveForm(f => ({
        ...f,
        received_qty: line ? line.planned_qty - line.received_qty : jo.planned_qty - jo.received_qty,
        split_components: true,
      }))
    }
    if (type === 'issue-pieces') {
      const line = jo.lines.find(l => l.id === lineId)
      const sku = opts?.sku || line?.sku || jo.sku
      const fromProc = opts?.fromProcess || jo.process
      let toProc = opts?.toProcess || ''
      if (!toProc) {
        try {
          const route = await api.get(`/production/item-routing/${encodeURIComponent(sku)}`, {
            params: {
              ...(jo.so_number ? { so_number: jo.so_number } : {}),
              process: fromProc,
            },
          })
          toProc = route.data?.next_process || ''
          if (!toProc && fromProc === 'Embroidery') {
            const hops = (route.data?.routing || []) as string[]
            toProc = hops.includes('Cutting') ? 'Cutting' : (route.data?.next_after_cutting || '')
          }
        } catch {
          toProc = jo.next_process || ''
        }
      }
      if (!toProc) toProc = jo.next_process || ''
      const defaultQty = opts?.issuedQty ?? (
        opts?.sku
          ? (expandedPanelWip?.panels?.find((p: { component_sku: string; issueable_qty?: number }) => p.component_sku === opts.sku)?.issueable_qty || 0)
          : (line ? line.received_qty : jo.received_qty)
      )
      setIssuePiecesSku(sku)
      setIssueFromProcess(fromProc)
      setIssuePiecesForm(f => ({ ...f, to_process: toProc, issued_qty: defaultQty }))
    }
    if (type === 'add-cost') setCostForm({ cost_type: 'Labour', amount: 0, description: '' })
  }

  // ── Add SO lines to new JO ─────────────────────────────────────────────────
  const addSOLineToJO = async (line: any) => {
    if (newLines.find(l => l.sku === line.sku)) return
    const soQty = Number(line.available_qty ?? line.qty) || 0
    // Cutting planned qty defaults to SO qty but is independent (may cut fewer pcs).
    setNewLines(ls => [...ls, {
      so_number: newForm.so_number,
      sku: line.sku || '',
      sku_name: line.sku_name || line.item_name || '',
      style: '',
      planned_qty: soQty,
      so_qty: soQty,
      vendor_rate: newForm.vendor_rate || 0,
      remarks: '',
    }])
    if (line.sku) {
      setNewForm(f => ({ ...f, sku: f.sku || line.sku }))
    }
    // Cutting: auto-fill fabric from BOM for the cutting (planned) qty
    if (newForm.process === 'Cutting' && line.sku) {
      try {
        const orderQty = soQty || 1
        const res = await api.get(`/production/bom-inputs/${encodeURIComponent(line.sku)}`, { params: { qty: orderQty } })
        const inputs = (res.data?.inputs ?? []) as { material_code?: string; material_name?: string; adj_qty?: number; unit?: string }[]
        const fabric = inputs.find(i => {
          const u = String(i.unit || '').toUpperCase()
          const code = String(i.material_code || '').toUpperCase()
          return u.includes('MTR') || u.includes('M') || code.includes('FABRIC') || code.includes('RAYON') || code.includes('COTTON')
        }) || inputs[0]
        if (fabric?.material_code) {
          const adj = Number(fabric.adj_qty) || 0
          setNewForm(f => ({
            ...f,
            fabric_code: f.fabric_code || fabric.material_code || '',
            fabric_qty: (Number(f.fabric_qty) || 0) + adj,
            fabric_unit: fabric.unit || f.fabric_unit || 'MTR',
          }))
          setFabricIssueForm(ff => ({
            ...ff,
            fabric_code: fabric.material_code || ff.fabric_code,
            fabric_name: fabric.material_name || ff.fabric_name,
            issued_qty: (Number(ff.issued_qty) || 0) + adj,
            unit: fabric.unit || ff.unit || 'MTR',
          }))
        }
      } catch { /* BOM optional */ }
    }
  }

  const updateCuttingLineQty = async (index: number, cuttingQty: number) => {
    const ln = newLines[index]
    if (!ln) return
    const prev = Number(ln.planned_qty) || 0
    setNewLines(ls => ls.map((x, j) => j === index ? { ...x, planned_qty: cuttingQty } : x))
    if (newForm.process !== 'Cutting' || !ln.sku) return
    try {
      const pickFab = (data: any) => {
        const inputs = (data?.inputs ?? []) as { material_code?: string; material_name?: string; adj_qty?: number; unit?: string }[]
        return inputs.find(i => {
          const u = String(i.unit || '').toUpperCase()
          const code = String(i.material_code || '').toUpperCase()
          return u.includes('MTR') || u.includes('M') || code.includes('FABRIC')
        }) || inputs[0]
      }
      const [oldBom, newBom] = await Promise.all([
        prev > 0
          ? api.get(`/production/bom-inputs/${encodeURIComponent(ln.sku)}`, { params: { qty: prev } })
          : Promise.resolve({ data: { inputs: [] } }),
        api.get(`/production/bom-inputs/${encodeURIComponent(ln.sku)}`, { params: { qty: Math.max(0, cuttingQty) } }),
      ])
      const oldAdj = Number(pickFab(oldBom.data)?.adj_qty) || 0
      const fab = pickFab(newBom.data)
      const newAdj = Number(fab?.adj_qty) || 0
      const delta = newAdj - oldAdj
      setNewForm(f => ({
        ...f,
        fabric_code: f.fabric_code || fab?.material_code || '',
        fabric_qty: Math.max(0, Math.round(((Number(f.fabric_qty) || 0) + delta) * 1000) / 1000),
        fabric_unit: fab?.unit || f.fabric_unit || 'MTR',
      }))
      setFabricIssueForm(ff => ({
        ...ff,
        fabric_code: fab?.material_code || ff.fabric_code,
        fabric_name: fab?.material_name || ff.fabric_name,
        issued_qty: Math.max(0, Math.round(((Number(ff.issued_qty) || 0) + delta) * 1000) / 1000),
        unit: fab?.unit || ff.unit || 'MTR',
      }))
    } catch { /* keep prior fabric_qty */ }
  }

  const allProcesses = processes.length > 0 ? processes : ['Cutting', 'Printing', 'Embroidery', 'Stitching', 'Finishing', 'Packing']

  const renderJOCard = (baseJo: JO) => {
    const isExpanded = expanded === baseJo.id
    const detail = isExpanded && expandedJoDetail?.id === baseJo.id ? expandedJoDetail : null
    const jo: JO = detail
      ? { ...baseJo, ...detail, lines: detail.lines?.length ? detail.lines : baseJo.lines }
      : baseJo
    const panelCtx = isExpanded && expandedPanelWip?.has_panels && expandedPanelWip?.jo_id === jo.id
      ? expandedPanelWip
      : null
    const totalPlanned = jo.lines.reduce((s, l) => s + l.planned_qty, 0) || jo.planned_qty
    const totalReceived = jo.lines.reduce((s, l) => s + l.received_qty, 0) || jo.received_qty
    const totalBalance = totalPlanned - totalReceived
    const pct = totalPlanned > 0 ? Math.min(100, (totalReceived / totalPlanned) * 100) : 0

    return (
      <div key={jo.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
        {/* Header */}
        <div className="flex items-start justify-between p-4 cursor-pointer" onClick={() => setExpanded(isExpanded ? null : jo.id)}>
          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className="font-bold text-[#002B5B] text-sm">{jo.jo_number}</span>
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${PROCESS_COLORS[jo.process] || 'bg-gray-100 text-gray-600'}`}>
                {PROCESS_ICONS[jo.process] || ''} {jo.process}
              </span>
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_COLORS[jo.status] || ''}`}>{jo.status}</span>
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                isOutsourceExec(jo.exec_type)
                  ? 'bg-amber-100 text-amber-700'
                  : 'bg-slate-100 text-slate-700'
              }`}>
                {isOutsourceExec(jo.exec_type)
                  ? `🏭 Outsource · ${jo.vendor_name || '—'}`
                  : '🏠 In-house'}
              </span>
            </div>
            <p className="text-sm text-gray-600">
              SO: <b>{jo.so_number || '—'}</b>
              {String(jo.so_source || '').toLowerCase() === 'manual' && (
                <span className="ml-1 text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 font-medium">manual SO</span>
              )}
              {' · '}SKU: <b>{formatJoHeaderSku(jo)}</b>
            </p>
            {/* Routing bar */}
            {jo.routing && jo.routing.length > 0 && (
              <div className="flex items-center gap-1 mt-1 flex-wrap">
                {jo.routing.map((p, i) => (
                  <span key={p}>
                    <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${p === jo.process ? 'bg-[#002B5B] text-white' : 'bg-gray-100 text-gray-500'}`}>
                      {PROCESS_ICONS[p] || ''} {p}
                    </span>
                    {i < jo.routing.length - 1 && <span className="text-gray-300 text-xs mx-0.5">→</span>}
                  </span>
                ))}
              </div>
            )}
            {/* Progress */}
            <div className="flex items-center gap-2 mt-2">
              <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div className="h-full bg-green-400 rounded-full" style={{ width: `${pct}%` }} />
              </div>
              <span className="text-xs text-gray-500 shrink-0">{fmt(totalReceived)}/{fmt(totalPlanned)} pcs</span>
            </div>
          </div>
          <div className="flex items-center gap-2 ml-2 shrink-0">
            <button onClick={e => { e.stopPropagation(); printJO(jo) }} className="text-xs px-2 py-1 border border-gray-200 rounded text-gray-500 hover:bg-gray-50">🖨️</button>
            <span className="text-gray-400 text-xs">{isExpanded ? '▲' : '▼'}</span>
          </div>
        </div>

        {/* Expanded */}
        {isExpanded && (
          <div className="border-t bg-gray-50 px-4 pb-4 space-y-4">
            {/* Stats row */}
            <div className="grid grid-cols-3 md:grid-cols-6 gap-2 pt-3">
              {[
                ['Planned', fmt(totalPlanned), 'text-gray-700'],
                ['Issued', fmt(jo.issued_qty || 0), 'text-blue-600'],
                ['Received', fmt(totalReceived), 'text-green-600'],
                ['Rejected', fmt(jo.rejected_qty || 0), 'text-red-500'],
                ['Balance', fmt(totalBalance), 'text-amber-600'],
                ['Cost', fmtR(jo.process_cost || 0), 'text-purple-600'],
              ].map(([l, v, c]) => (
                <div key={l} className="bg-white rounded-lg p-2 border text-center">
                  <p className={`font-bold text-sm ${c}`}>{v}</p>
                  <p className="text-xs text-gray-400">{l}</p>
                </div>
              ))}
            </div>

            {/* Planned qty: size-wise when the JO has multiple SKU/size lines; otherwise header total. */}
            {(jo.status === 'Created' || jo.status === 'In Progress') && jo.lines.length <= 1 && (
              <div className="bg-white border rounded-lg p-3 flex flex-wrap items-end gap-2">
                <div>
                  <label className="text-xs text-gray-500">Edit planned qty</label>
                  <input
                    type="number"
                    min={Math.max(jo.issued_qty || 0, jo.received_qty || 0, jo.output_qty || 0)}
                    value={editPlannedQty[jo.id] ?? String(jo.planned_qty)}
                    onChange={e => setEditPlannedQty(m => ({ ...m, [jo.id]: e.target.value }))}
                    className="block w-28 border border-amber-200 bg-amber-50 rounded px-2 py-1 text-sm font-semibold mt-0.5"
                  />
                </div>
                <button
                  type="button"
                  disabled={updateJoQtyMut.isPending}
                  onClick={() => {
                    const v = parseInt(editPlannedQty[jo.id] ?? String(jo.planned_qty), 10)
                    if (!Number.isFinite(v) || v < 0) {
                      alert('Enter a valid planned quantity')
                      return
                    }
                    updateJoQtyMut.mutate({
                      id: jo.id,
                      data: {
                        planned_qty: v,
                        qty_change_remarks: 'UI planned_qty edit',
                      },
                    })
                  }}
                  className="px-3 py-1.5 text-xs bg-amber-600 text-white rounded-lg font-medium disabled:opacity-50"
                >
                  Save qty
                </button>
                <p className="text-[10px] text-gray-500 max-w-xs">
                  Cannot go below issued/received/output. JO total equals this size. Written to jo_qty_history.
                </p>
              </div>
            )}
            {(jo.status === 'Created' || jo.status === 'In Progress') && jo.lines.length > 1 && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-[11px] text-amber-900">
                Edit planned quantity per size in the lines table. JO total is the sum of sizes
                ({fmt(jo.lines.reduce((s, l) => s + (Number(l.planned_qty) || 0), 0))} pcs).
              </div>
            )}

            <VendorExecutionEditor
              jo={jo}
              vendorSuggestions={vendorSuggestions}
              saving={updateJOMut.isPending}
              onSave={data => updateJOMut.mutate({ id: jo.id, data })}
            />

            {/* Process stock visibility */}
            {jo.process_stocks && Object.keys(jo.process_stocks).length > 0 && (
              <div className="bg-white rounded-lg border p-3">
                <p className="text-xs font-semibold text-gray-500 uppercase mb-2">Process Stock — {jo.sku}</p>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(jo.process_stocks).map(([proc, stock]) => (
                    <div key={proc} className={`text-xs px-3 py-1.5 rounded-lg font-medium border ${proc === jo.process ? 'bg-[#002B5B] text-white border-[#002B5B]' : 'bg-gray-50 text-gray-700 border-gray-200'}`}>
                      {PROCESS_ICONS[proc] || ''} {proc}: <b>{stock.available}</b> pcs
                      <span className="opacity-60 ml-1">(in:{stock.in} out:{stock.out})</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Panel WIP (FRONT/BACK under parent Cutting JO) */}
            {panelCtx && (
              <div className="bg-white rounded-lg border border-indigo-200 overflow-hidden">
                <div className="px-3 py-2 bg-indigo-50 text-xs font-semibold text-indigo-900 flex flex-wrap justify-between gap-2">
                  <span>Panel WIP — managed inside this Cutting JO</span>
                  <span className={panelCtx.bundle_complete ? 'text-emerald-700' : 'text-amber-700'}>
                    {panelCtx.bundle_message || (panelCtx.bundle_complete ? 'Bundle complete' : 'Bundle incomplete')}
                  </span>
                </div>
                <p className="px-3 py-2 text-[11px] text-indigo-800/80 border-b border-indigo-100">
                  {panelCtx.hint}
                </p>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-gray-400 border-b uppercase">
                      <th className="text-left px-3 py-2">Panel</th>
                      <th className="text-left px-3 py-2">SKU</th>
                      <th className="text-left px-3 py-2">Routing</th>
                      <th className="text-left px-3 py-2">Embroidery</th>
                      <th className="text-left px-3 py-2">Location</th>
                      <th className="text-right px-3 py-2">Cutting</th>
                      <th className="text-right px-3 py-2">Embroidery</th>
                      <th className="text-left px-3 py-2">Status</th>
                      <th className="text-left px-3 py-2">Embroidery JO</th>
                      <th className="text-center px-3 py-2">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(panelCtx.panels || []).map((panel: {
                      component_code: string
                      component_name: string
                      component_sku: string
                      routing: string
                      current_location: string
                      available_qty: number
                      embroidery_outstanding: number
                      status: string
                      issue_from_process: string
                      issue_to_process: string
                      issue_to_label?: string
                      embroidery_timing?: string
                      embroidery_before_cutting?: boolean
                      embroidery_type?: string
                      embroidery_qty_per_piece?: number
                      embroidery_unit?: string
                      embroidery_stock_available?: number
                      issueable_qty: number
                      embroidery_jo?: {
                        id: number
                        jo_number: string
                        status: string
                        planned_qty: number
                        received_qty: number
                        measurement_qty?: number
                        garment_qty?: number
                        embroidery_type?: string
                        embroidery_unit?: string
                      } | null
                    }) => (
                      <tr key={panel.component_sku} className="border-t border-gray-50 hover:bg-indigo-50/40">
                        <td className="px-3 py-2 font-semibold text-indigo-900">{panel.component_name || panel.component_code}</td>
                        <td className="px-3 py-2 font-mono text-[11px]">{panel.component_sku}</td>
                        <td className="px-3 py-2 text-gray-600">{panel.routing || '—'}</td>
                        <td className="px-3 py-2 text-[11px] text-purple-800">
                          {panel.embroidery_timing || '—'}
                          {panel.embroidery_type ? (
                            <span className="block text-gray-600">
                              {panel.embroidery_type}
                              {panel.embroidery_qty_per_piece ? ` · ${panel.embroidery_qty_per_piece}/${panel.embroidery_unit || 'unit'}/pc` : ''}
                            </span>
                          ) : null}
                          {(panel.embroidery_stock_available || 0) > 0 && (
                            <span className="block text-amber-700 font-semibold">
                              Stock: {panel.embroidery_stock_available} {panel.embroidery_unit || ''}
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-2">{panel.current_location || '—'}</td>
                        <td className="px-3 py-2 text-right">{fmt(panel.available_qty || 0)}</td>
                        <td className="px-3 py-2 text-right text-purple-700">{fmt(panel.embroidery_outstanding || 0)}</td>
                        <td className="px-3 py-2">{panel.status || '—'}</td>
                        <td className="px-3 py-2 font-mono text-[11px]">
                          {panel.embroidery_jo ? (
                            <span className="text-rose-700" title={panel.embroidery_jo.status}>
                              {panel.embroidery_jo.jo_number}
                              <span className="block text-gray-500 font-sans">
                                {panel.embroidery_jo.measurement_qty && panel.embroidery_jo.embroidery_unit
                                  ? `${panel.embroidery_jo.received_qty}/${panel.embroidery_jo.planned_qty} ${panel.embroidery_jo.embroidery_unit}`
                                  : `${panel.embroidery_jo.received_qty}/${panel.embroidery_jo.planned_qty} pcs`}
                                {' · '}{panel.embroidery_jo.status}
                              </span>
                            </span>
                          ) : (
                            <span className="text-gray-400">—</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-center">
                          {panel.issueable_qty > 0 && panel.issue_to_process ? (
                            <button
                              onClick={() => openModal('issue-pieces', jo, undefined, {
                                sku: panel.component_sku,
                                fromProcess: panel.issue_from_process,
                                toProcess: panel.issue_to_process,
                                issuedQty: panel.issueable_qty,
                              })}
                              className="px-2 py-0.5 text-xs bg-purple-600 text-white rounded hover:bg-purple-700"
                            >
                              → {panel.issue_to_label || panel.issue_to_process}
                            </button>
                          ) : (
                            <span className="text-gray-400">Receive parent first</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Lines table */}
            {jo.lines.length > 0 && (
              <div className="bg-white rounded-lg border overflow-hidden">
                <div className="px-3 py-2 bg-gray-50 text-xs font-semibold text-gray-600 flex justify-between items-center gap-2">
                  <span>Lines — Issue / Receive per SKU</span>
                  <span className="flex items-center gap-2">
                    {(jo.status === 'Created' || jo.status === 'In Progress') && (
                      <button
                        type="button"
                        disabled={updateJoQtyMut.isPending}
                        onClick={() => {
                          const lines = jo.lines.map(line => {
                            const raw = editLineQty[line.id]
                            const v = raw == null ? line.planned_qty : parseInt(raw, 10)
                            return { id: line.id, planned_qty: v }
                          })
                          if (lines.some(l => !Number.isFinite(l.planned_qty) || l.planned_qty < 0)) {
                            alert('Each size planned quantity must be a number ≥ 0')
                            return
                          }
                          updateJoQtyMut.mutate({
                            id: jo.id,
                            data: { lines, qty_change_remarks: 'UI size-wise planned_qty edit' },
                          })
                        }}
                        className="px-2 py-1 bg-amber-600 text-white rounded font-medium disabled:opacity-50"
                      >
                        Save size qtys
                      </button>
                    )}
                    <span className="text-gray-400">{jo.lines.length} lines</span>
                  </span>
                </div>
                <table className="w-full text-xs">
                  <thead><tr className="text-gray-400 border-b uppercase">
                    <th className="text-left px-3 py-2">SKU</th>
                    <th className="text-left px-3 py-2">Style</th>
                    <th className="text-right px-3 py-2">Planned</th>
                    <th className="text-right px-3 py-2">Received</th>
                    <th className="text-right px-3 py-2">Rejected</th>
                    <th className="text-right px-3 py-2">Balance</th>
                    <th className="text-right px-3 py-2">Rate</th>
                    <th className="text-right px-3 py-2">Amount</th>
                    <th className="text-center px-3 py-2">Actions</th>
                  </tr></thead>
                  <tbody>
                    {jo.lines.map(line => (
                      <tr key={line.id} className="border-t border-gray-50 hover:bg-gray-50">
                        <td className="px-3 py-2 font-mono font-semibold text-[#002B5B]">{line.sku}</td>
                        <td className="px-3 py-2 text-gray-500">{line.style || '—'}</td>
                        <td className="px-3 py-2 text-right">
                          {(jo.status === 'Created' || jo.status === 'In Progress') ? (
                            <input
                              type="number"
                              min={Math.max(line.issued_qty || 0, line.received_qty || 0, 0)}
                              value={editLineQty[line.id] ?? String(line.planned_qty)}
                              onChange={e => setEditLineQty(m => ({ ...m, [line.id]: e.target.value }))}
                              className="w-20 border border-amber-200 bg-amber-50 rounded px-1 py-0.5 text-right font-semibold"
                            />
                          ) : (
                            fmt(line.planned_qty)
                          )}
                        </td>
                        <td className="px-3 py-2 text-right text-green-600 font-semibold">{fmt(line.received_qty)}</td>
                        <td className="px-3 py-2 text-right text-red-500">{fmt(line.rejected_qty)}</td>
                        <td className={`px-3 py-2 text-right font-semibold ${line.balance_qty > 0 ? 'text-amber-600' : 'text-green-600'}`}>{fmt(line.balance_qty)}</td>
                        <td className="px-3 py-2 text-right">{fmtR(line.vendor_rate)}</td>
                        <td className="px-3 py-2 text-right font-semibold">{fmtR(line.planned_qty * line.vendor_rate)}</td>
                        <td className="px-3 py-2 text-center">
                          <div className="flex gap-1 justify-center">
                            <button onClick={() => openModal('receive', jo, line.id)}
                              className="px-2 py-0.5 text-xs bg-green-600 text-white rounded hover:bg-green-700">✅ Rec</button>
                            {jo.next_process && !panelCtx && (
                              <button onClick={() => openModal('issue-pieces', jo, line.id)}
                                className="px-2 py-0.5 text-xs bg-purple-600 text-white rounded hover:bg-purple-700">
                                → {jo.next_process}
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <JOIssueNotePanel joId={jo.id} joNumber={jo.jo_number} />

            {/* Fabric issues (Cutting only) */}
            {jo.process === 'Cutting' && (
              <div className="bg-white rounded-lg border p-3">
                <p className="text-xs font-semibold text-gray-500 uppercase mb-2">Fabric</p>
                <div className="flex flex-wrap gap-3 text-xs">
                  <span>Code: <b className="font-mono">{jo.fabric_code || '—'}</b></span>
                  <span>Planned: <b>{jo.fabric_qty} {jo.fabric_unit}</b></span>
                  <span className="text-blue-600">Issued: <b>{jo.fabric_issued_qty || 0}</b></span>
                  <span className="text-amber-600">Returned: <b>{jo.fabric_received_qty || 0}</b></span>
                  <span className="text-red-600">Consumed: <b>{(jo.fabric_issued_qty || 0) - (jo.fabric_received_qty || 0)}</b></span>
                </div>
                {jo.fabric_issues && jo.fabric_issues.length > 0 && (
                  <table className="w-full text-xs mt-2">
                    <thead><tr className="text-gray-400 border-b"><th className="text-left pb-1">Date</th><th className="text-left pb-1">Code</th><th className="text-right pb-1">Issued</th><th className="text-left pb-1">By</th></tr></thead>
                    <tbody>{jo.fabric_issues.map((f: any) => (
                      <tr key={f.id} className="border-t border-gray-50">
                        <td className="py-1">{f.issue_date}</td><td className="py-1 font-mono">{f.fabric_code}</td>
                        <td className="py-1 text-right text-blue-600 font-semibold">{f.issued_qty} {f.unit}</td>
                        <td className="py-1 text-gray-500">{f.issued_by || '—'}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                )}
              </div>
            )}

            {/* Action buttons */}
            <div className="flex flex-wrap gap-2">
              {jo.process === 'Cutting' && (
                <>
                  <button onClick={() => openModal('issue-fabric', jo)} className="px-3 py-1.5 text-xs bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700">📦 Issue Fabric</button>
                  <button onClick={() => openModal('return-fabric', jo)} className="px-3 py-1.5 text-xs bg-blue-100 text-blue-700 rounded-lg font-medium">↩️ Return Fabric</button>
                </>
              )}
              <button onClick={() => openModal('receive', jo)} className="px-3 py-1.5 text-xs bg-green-600 text-white rounded-lg font-medium hover:bg-green-700">✅ Receive (JO level)</button>
              {jo.next_process && !panelCtx && (
                <button onClick={() => openModal('issue-pieces', jo)} className="px-3 py-1.5 text-xs bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700">
                  ➡️ Issue to {jo.next_process}
                </button>
              )}
              {!jo.next_process && !panelCtx && (
                <span className="px-3 py-1.5 text-xs bg-gray-50 text-gray-600 rounded-lg border border-gray-200">
                  No next process configured for this routing
                </span>
              )}
              <button onClick={() => openModal('add-cost', jo)} className="px-3 py-1.5 text-xs bg-amber-600 text-white rounded-lg font-medium">💰 Add Cost</button>
              <select value={jo.status} onChange={e => updateJOMut.mutate({ id: jo.id, data: { status: e.target.value } })}
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-xs">
                {['Created','In Progress','Completed','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
              </select>
              {jo.status === 'Completed' && !jo.next_stage_jo_id && jo.next_process && (
                <button onClick={() => nextProcessMut.mutate(jo.id)} disabled={nextProcessMut.isPending}
                  className="px-3 py-1.5 text-xs bg-[#002B5B] text-white rounded-lg font-medium hover:bg-blue-800 disabled:opacity-50">
                  🔄 Create {jo.next_process} JO →
                </button>
              )}
              {jo.next_stage_jo_id && (
                <span className="px-3 py-1.5 text-xs bg-green-50 text-green-700 rounded-lg border border-green-200">✅ {jo.next_process} JO linked</span>
              )}
              <button onClick={() => printJO(jo)} className="px-3 py-1.5 text-xs border border-gray-200 rounded-lg text-gray-600 hover:bg-gray-50">🖨️ Print JO</button>
            </div>

            {/* Cost log */}
            {jo.cost_entries && jo.cost_entries.length > 0 && (
              <div className="bg-white rounded-lg border p-3">
                <p className="text-xs font-semibold text-gray-500 uppercase mb-2">Cost Log — Total: {fmtR(jo.total_cost || 0)}</p>
                <table className="w-full text-xs">
                  <thead><tr className="text-gray-400 border-b"><th className="text-left pb-1">Date</th><th className="text-left pb-1">Type</th><th className="text-right pb-1">Amount</th><th className="text-left pb-1">Desc</th></tr></thead>
                  <tbody>{jo.cost_entries.map((c: any) => (
                    <tr key={c.id} className="border-t border-gray-50">
                      <td className="py-1">{c.cost_date}</td><td className="py-1">{c.cost_type}</td>
                      <td className="py-1 text-right font-semibold text-amber-700">{fmtR(c.amount)}</td>
                      <td className="py-1 text-gray-500">{c.description || '—'}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-gray-800">Production</h1>
        <p className="text-sm text-gray-500">Dynamic routing — {allProcesses.join(' → ')}</p>
      </div>

      {/* Main tabs */}
      <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit">
        {([['process','⚙️ Process'], ['sets','👔 Set Components'], ['tracker','📋 All JOs'], ['issue-notes','📋 Issue Notes'], ['reports','📊 Reports'], ['mrp','📐 Material Req. Planning']] as const).map(([key, label]) => (
          <button key={key} onClick={() => { setTab(key); setExpanded(null) }}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${tab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* Stats cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            ['Total JOs', stats.total_jos, 'text-gray-700'],
            ['Open', stats.open_jos, 'text-amber-600'],
            ['In Progress', stats.in_progress, 'text-blue-600'],
            ['Done Today', stats.completed_today, 'text-green-600'],
          ].map(([l, v, c]) => (
            <div key={l as string} className="bg-white rounded-xl p-3 border shadow-sm">
              <p className={`text-xl font-bold ${c}`}>{v}</p>
              <p className="text-xs text-gray-500 mt-1 font-semibold">{l}</p>
            </div>
          ))}
          {stats.process_counts && Object.entries(stats.process_counts as Record<string, number>).map(([p, cnt]) => (
            <div key={p} className="bg-white rounded-xl p-3 border shadow-sm">
              <p className="text-xl font-bold text-gray-700">{cnt}</p>
              <p className="text-xs text-gray-500 mt-1 font-semibold">{PROCESS_ICONS[p] || ''} {p}</p>
            </div>
          ))}
        </div>
      )}

      {/* PROCESS TAB */}
      {tab === 'process' && (
        <div className="space-y-4">
          {/* Process selector */}
          <div className="flex flex-wrap gap-1 bg-gray-100 p-1 rounded-lg w-fit">
            {allProcesses.map(p => (
              <button key={p} onClick={() => { setActiveProcess(p); setExpanded(null); clearListFilters() }}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${activeProcess === p ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
                {PROCESS_ICONS[p] || ''} {p}
              </button>
            ))}
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-3 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="search"
                value={listSearch}
                onChange={e => setListSearch(e.target.value)}
                placeholder="Search SO / SKU / vendor…"
                className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm min-w-[14rem] flex-1"
              />
              <select value={filterSO} onChange={e => setFilterSO(e.target.value)}
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm max-w-[11rem]">
                <option value="">All SOs</option>
                {soOptions.map(so => <option key={so} value={so}>{so}</option>)}
              </select>
              <select value={filterSku} onChange={e => setFilterSku(e.target.value)}
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm max-w-[12rem]">
                <option value="">All SKUs</option>
                {skuOptions.map(sku => <option key={sku} value={sku}>{sku}</option>)}
              </select>
              <select value={filterVendor} onChange={e => setFilterVendor(e.target.value)}
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm max-w-[12rem]">
                <option value="">All vendors / buyers</option>
                {vendorOptions.map(v => <option key={v} value={v}>{v}</option>)}
              </select>
              <input
                type="search"
                value={filterJO}
                onChange={e => setFilterJO(e.target.value)}
                placeholder="Job Order…"
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm w-36"
              />
              <input
                type="number"
                min={0}
                value={filterMinQty}
                onChange={e => setFilterMinQty(e.target.value)}
                placeholder="Min qty"
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm w-24"
              />
              <input type="date" value={filterDateFrom} onChange={e => setFilterDateFrom(e.target.value)}
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm" title="From date" />
              <input type="date" value={filterDateTo} onChange={e => setFilterDateTo(e.target.value)}
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm" title="To date" />
              <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
                <option value="">All Statuses</option>
                {['Created','In Progress','Completed','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-gray-500">Sort</span>
              <select value={sortBy} onChange={e => setSortBy(e.target.value as typeof sortBy)}
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm">
                <option value="so_number">SO</option>
                <option value="sku">SKU</option>
                <option value="vendor_name">Vendor / Buyer</option>
                <option value="available_qty">Qty</option>
                <option value="jo_date">JO date</option>
                <option value="status">Status</option>
              </select>
              <button type="button" onClick={() => setSortDir(d => d === 'asc' ? 'desc' : 'asc')}
                className="px-2 py-1.5 border border-gray-200 rounded-lg text-sm text-gray-700 hover:bg-gray-50"
                title="Toggle sort direction">
                {sortDir === 'asc' ? '↑ Asc' : '↓ Desc'}
              </button>
              {(listSearch || filterSO || filterSku || filterVendor) && (
                <button type="button" onClick={clearListFilters}
                  className="text-xs text-indigo-700 hover:underline ml-1">
                  Clear filters
                </button>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between flex-wrap gap-2">
            <p className="text-xs text-gray-500">
              Showing {filteredReadyLines.length}/{readyLines.length} ready · {filteredProcessJOs.length}/{processJOs.length} JOs
            </p>
            <div className="flex items-center gap-2">
              <input
                ref={joImportRef}
                type="file"
                accept=".csv,.xlsx,.xls"
                className="hidden"
                onChange={async (e) => {
                  const file = e.target.files?.[0]
                  e.target.value = ''
                  if (!file) return
                  const fd = new FormData()
                  fd.append('file', file)
                  fd.append('process', activeProcess)
                  try {
                    const res = await api.post('/production/orders/import', fd, {
                      headers: { 'Content-Type': 'multipart/form-data' },
                    })
                    qc.invalidateQueries({ queryKey: ['jos-process'] })
                    qc.invalidateQueries({ queryKey: ['ready-to-process'] })
                    const errs = (res.data?.errors || []).slice(0, 6).join('\n')
                    alert(
                      (res.data?.message || `Imported ${res.data?.created ?? 0} job order(s).`)
                      + (errs ? `\n\n${errs}` : '')
                      + (res.data?.hint ? `\n\n${res.data.hint}` : ''),
                    )
                  } catch (err) {
                    alert(apiErrorMessage(err, 'Import failed'))
                  }
                }}
              />
              <button
                type="button"
                onClick={async () => {
                  try {
                    const res = await api.get('/production/orders/import-template', {
                      responseType: 'blob',
                    })
                    const url = URL.createObjectURL(new Blob([res.data], { type: 'text/csv' }))
                    const a = document.createElement('a')
                    a.href = url
                    a.download = 'production_jo_import_template.csv'
                    a.click()
                    URL.revokeObjectURL(url)
                  } catch (err) {
                    // Fallback local template if API unavailable
                    const csv = [
                      'so_number,sku,component_code,planned_qty,process,create_component_jos,exec_type,vendor_name,vendor_rate,expected_completion,fabric_code,fabric_qty,fabric_unit,sku_name,remarks',
                      'SO-0001,TEST SKU-M,,10,Cutting,yes,Inhouse,,0,2026-08-15,,,,Test Style M,"One row per size → TOP/PANT/DUPATTA JOs. Do NOT add FRONT/BACK rows — panels appear inside TOP JO after Receive."',
                      'SO-0001,TEST SKU-M,TOP,10,Cutting,no,Inhouse,,0,2026-08-15,FABRIC-TOP,20,MTR,Top only,"Optional: only TOP Cutting JO; FRONT/BACK stay as panels on this JO"',
                      'SO-0001,TEST SKU-M,PANT,10,Cutting,no,Inhouse,,0,2026-08-15,FABRIC-PANT,15,MTR,Pant only,"Optional: PANT or DUPATTA alone"',
                    ].join('\n')
                    const a = document.createElement('a')
                    a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv)
                    a.download = 'production_jo_import_template.csv'
                    a.click()
                    alert(apiErrorMessage(err, 'Used offline template (API template failed)'))
                  }
                }}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm text-gray-600 hover:bg-gray-50"
                title="Download CSV template — main size SKU (or TOP/PANT/DUPATTA). FRONT/BACK are panels, not JO rows."
              >
                📥 Template
              </button>
              <button
                type="button"
                onClick={() => joImportRef.current?.click()}
                className="px-4 py-2 border border-[#002B5B] text-[#002B5B] rounded-lg text-sm font-medium hover:bg-blue-50"
              >
                📥 Import
              </button>
              <button
                type="button"
                onClick={() => {
                  const rows = filteredProcessJOs.flatMap(j => expandJoExportRows(j))
                  downloadCsv(
                    `${activeProcess.toLowerCase()}_job_orders_${new Date().toISOString().slice(0, 10)}.csv`,
                    [...JO_EXPORT_HEADERS],
                    rows,
                  )
                }}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm text-gray-700 hover:bg-gray-50"
                title="Export currently listed job orders (respects filters/search)"
              >
                ↓ Export Excel
              </button>
              <button onClick={() => {
                setNewForm(f => ({
                  ...f,
                  process: activeProcess,
                  so_source: 'system',
                  so_number: '',
                  sku: '',
                  sku_name: '',
                  from_ready_to: false,
                }))
                setNewLines([])
                setSOLineSearch('')
                setModal('new-jo')
              }}
                className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
                + New {activeProcess} JO
              </button>
            </div>
          </div>
          <p className="text-xs text-gray-500 -mt-1 mb-2">
            JO import — required <code className="bg-gray-100 px-1 rounded">so_number</code> + <code className="bg-gray-100 px-1 rounded">sku</code> (main size, e.g. STYLE-M).
            Cutting with Set BOM: one row creates <strong>TOP / PANT / DUPATTA</strong> JOs.
            Optional <code className="bg-gray-100 px-1 rounded">component_code</code>=TOP|PANT|DUPATTA for one component only.
            <strong> Do not import FRONT/BACK</strong> — they are panels inside the TOP JO after Receive.
            Optional <code className="bg-gray-100 px-1 rounded">create_component_jos</code>: yes (default) | no.
            {' '}Optional <code className="bg-gray-100 px-1 rounded">production_mode</code>: <em>inhouse</em> | <em>cut_to_pack</em> | <em>stitch_to_pack</em> (blank = SO default). Split same SKU across paths with separate rows.
          </p>

          {/* Ready to process — always visible for stage context */}
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <p className="text-sm font-semibold text-amber-800">
                ⚡ {activeProcess === 'Cutting' ? 'Ready to Cut' : activeProcess === 'Stitching' ? 'Ready to Stitch' : activeProcess === 'Embroidery' ? 'Ready for Embroidery' : `Ready for ${activeProcess}`}
                {readyLines.length > 0 ? ` — ${filteredReadyLines.length} of ${readyLines.length} line(s)` : ''}
              </p>
              <div className="flex items-center gap-2 ml-auto">
                <a
                  className="text-xs text-indigo-700 hover:underline"
                  href={`/api/production/ready-to-wip/import-template?stage=${encodeURIComponent(activeProcess)}`}
                >⬇ WIP template</a>
                <input
                  ref={wipImportRef}
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  className="hidden"
                  onChange={async e => {
                    const f = e.target.files?.[0]
                    e.target.value = ''
                    if (!f) return
                    const fd = new FormData()
                    fd.append('file', f)
                    fd.append('stage', activeProcess)
                    try {
                      const { data } = await api.post('/production/ready-to-wip/import', fd, {
                        headers: { 'Content-Type': 'multipart/form-data' },
                      })
                      const failed = data.failed ?? (data.errors?.length ?? 0)
                      const errs = (data.errors || []).slice(0, 8).join('\n')
                      alert(
                        (data.message || `Imported ${data.imported ?? 0} Ready-To WIP row(s)`)
                        + (failed && !data.message?.includes('failed') ? `; ${failed} failed` : '')
                        + (errs ? `\n\n${errs}` : '')
                        + (data.import_batch ? `\nBatch: ${data.import_batch}` : ''),
                      )
                      invalidateAll()
                    } catch (err: any) {
                      alert(err?.response?.data?.detail || err?.message || 'WIP import failed')
                    }
                  }}
                />
                <button
                  type="button"
                  onClick={() => wipImportRef.current?.click()}
                  className="text-xs px-2 py-1 rounded border border-indigo-200 text-indigo-800 hover:bg-indigo-50"
                >Import Ready-To WIP</button>
              </div>
            </div>
            {readyLines.length === 0 ? (
              <p className="text-xs text-amber-700">
                {activeProcess === 'Cutting'
                  ? 'No printed-fabric reservations yet. Reserve fabric under Grey Fabric → Ready to Cut.'
                    : `No pieces waiting at ${activeProcess} (or the previous process).`}
              </p>
            ) : filteredReadyLines.length === 0 ? (
              <p className="text-xs text-amber-700">No ready lines match the current search / filters.</p>
            ) : (
              <div className="space-y-2 max-h-56 overflow-y-auto">
                {filteredReadyLines.map((r: any, i: number) => (
                  <div key={i} className="bg-white rounded-lg border border-amber-200 px-3 py-2 flex items-center justify-between gap-2">
                    <div className="text-xs min-w-0">
                      <span className="font-semibold text-[#002B5B]">SO: {r.so_number}</span>
                      <span className="mx-2 text-gray-400">·</span>
                      <span className="font-mono break-all">{r.sku}</span>
                      {r.component_code && (
                        <span className="ml-1.5 text-[10px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-800 font-medium">
                          {r.component_code}
                          {r.readiness_scope === 'component' ? ' ready' : ''}
                        </span>
                      )}
                      {r.vendor_name && (
                        <>
                          <span className="mx-2 text-gray-400">·</span>
                          <span className="text-gray-700">Buyer: <b>{r.vendor_name}</b></span>
                        </>
                      )}
                      {r.fabric_code && <span className="mx-2 text-gray-400">· Fabric: <b>{r.fabric_code}</b></span>}
                      <span className="mx-2 text-gray-400">·</span>
                      <span className="text-green-700 font-semibold">
                        {r.available_qty || r.reserved_qty} {activeProcess === 'Cutting' ? 'm/pcs' : 'pcs'} available
                      </span>
                      {r.already_planned > 0 && (
                        <span className="ml-2 text-gray-400 italic">(Total: {r.reserved_qty}, JO mein: {r.already_planned})</span>
                      )}
                    </div>
                    <button onClick={() => {
                      const avail = Number(r.available_qty || r.reserved_qty || 0)
                      const so = (soList as any[]).find((s: any) => s.so_number === r.so_number)
                      const defaultMode = so?.production_mode || 'inhouse'
                      const exec = suggestedExecType(defaultMode, activeProcess)
                      setNewForm(f => ({
                        ...f,
                        so_number: r.so_number,
                        so_source: 'system',
                        sku: r.sku || '',
                        sku_name: r.sku_name || r.item_name || '',
                        fabric_code: r.fabric_code || '',
                        fabric_qty: r.reserved_qty || 0,
                        process: activeProcess,
                        planned_qty: avail,
                        so_qty: avail,
                        production_mode: defaultMode,
                        exec_type: exec,
                        from_ready_to: true,
                      }))
                      setNewLines([{
                        so_number: r.so_number,
                        sku: r.sku || '',
                        sku_name: r.sku_name || r.item_name || '',
                        style: '',
                        planned_qty: avail,
                        so_qty: avail,
                        vendor_rate: 0,
                        remarks: '',
                      }])
                      setSOLineSearch('')
                      setModal('new-jo')
                    }} className="text-xs px-2 py-1 bg-[#002B5B] text-white rounded hover:bg-blue-800 shrink-0">
                      Create JO →
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* JO list */}
          <div className="space-y-3">
            {filteredProcessJOs.map(renderJOCard)}
            {(josLoading || josFetching) && processJOs.length === 0 && (
              <p className="text-center text-amber-700 py-8 text-sm">Loading {activeProcess} job orders…</p>
            )}
            {josError && (
              <p className="text-center text-red-600 py-4 text-sm">
                Failed to load job orders: {(josErr as any)?.message || 'request error'}
              </p>
            )}
            {!josLoading && !josFetching && processJOs.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No {activeProcess} job orders.</p>}
            {processJOs.length > 0 && filteredProcessJOs.length === 0 && (
              <p className="text-center text-gray-400 py-8 text-sm">No job orders match the current search / filters.</p>
            )}
          </div>
        </div>
      )}

      {/* TRACKER TAB */}
      {tab === 'tracker' && (
        <div className="space-y-3">
          <div className="bg-white border border-gray-200 rounded-xl p-3 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="search"
                value={listSearch}
                onChange={e => setListSearch(e.target.value)}
                placeholder="Search SO / SKU / vendor / JO…"
                className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm min-w-[14rem] flex-1"
              />
              <select value={filterSO} onChange={e => setFilterSO(e.target.value)}
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm max-w-[11rem]">
                <option value="">All SOs</option>
                {soOptions.map(so => <option key={so} value={so}>{so}</option>)}
              </select>
              <select value={filterSku} onChange={e => setFilterSku(e.target.value)}
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm max-w-[12rem]">
                <option value="">All SKUs</option>
                {skuOptions.map(sku => <option key={sku} value={sku}>{sku}</option>)}
              </select>
              <select value={filterVendor} onChange={e => setFilterVendor(e.target.value)}
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm max-w-[12rem]">
                <option value="">All vendors / buyers</option>
                {vendorOptions.map(v => <option key={v} value={v}>{v}</option>)}
              </select>
              <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
                <option value="">All Statuses</option>
                {['Created','In Progress','Completed','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-gray-500">Sort</span>
              <select value={sortBy} onChange={e => setSortBy(e.target.value as typeof sortBy)}
                className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm">
                <option value="so_number">SO</option>
                <option value="sku">SKU</option>
                <option value="vendor_name">Vendor / Buyer</option>
                <option value="jo_date">JO date</option>
                <option value="status">Status</option>
                <option value="available_qty">Balance qty</option>
              </select>
              <button type="button" onClick={() => setSortDir(d => d === 'asc' ? 'desc' : 'asc')}
                className="px-2 py-1.5 border border-gray-200 rounded-lg text-sm text-gray-700 hover:bg-gray-50">
                {sortDir === 'asc' ? '↑ Asc' : '↓ Desc'}
              </button>
              {(listSearch || filterSO || filterSku || filterVendor) && (
                <button type="button" onClick={clearListFilters} className="text-xs text-indigo-700 hover:underline">
                  Clear filters
                </button>
              )}
              <p className="text-sm text-gray-500 ml-auto">{filteredAllJOs.length} of {allJOs.length} job orders</p>
            </div>
          </div>
          {filteredAllJOs.map(renderJOCard)}
          {allJosLoading && allJOs.length === 0 && <p className="text-center text-amber-700 py-8 text-sm">Loading all job orders…</p>}
          {!allJosLoading && allJOs.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No job orders found.</p>}
          {allJOs.length > 0 && filteredAllJOs.length === 0 && (
            <p className="text-center text-gray-400 py-8 text-sm">No job orders match the current search / filters.</p>
          )}
        </div>
      )}

      {/* ISSUE NOTES TAB */}
      {tab === 'issue-notes' && (
        <div className="space-y-3">
          <p className="text-sm text-gray-500">
            Material issue notes auto-created from Job Orders and Item Master BOM. Quantities scale with JO planned qty.
          </p>
          {issueNotes.length === 0 && (
            <p className="text-center text-gray-400 py-12 text-sm">No issue notes yet. Create a Job Order to generate one.</p>
          )}
          {issueNotes.map(note => {
            const open = expandedIssueNote === note.id
            return (
              <div key={note.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                <button type="button" onClick={() => setExpandedIssueNote(open ? null : note.id)}
                  className="w-full text-left px-4 py-3 flex flex-wrap items-center justify-between gap-2 hover:bg-gray-50">
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-bold text-indigo-800 font-mono">{note.in_number}</span>
                      <span className="text-xs text-gray-400">·</span>
                      <span className="text-sm font-semibold text-[#002B5B]">{note.jo_number}</span>
                      <span className={`text-xs px-2 py-0.5 rounded-full ${PROCESS_COLORS[note.process] || 'bg-gray-100 text-gray-600'}`}>
                        {note.process}
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">{note.status}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      Date {note.in_date} · SO {note.so_number || '—'} · For <b className="text-[#002B5B]">{note.finished_item_code}</b>
                      {note.finished_item_name ? ` — ${note.finished_item_name}` : ''} · JO qty <b>{note.planned_qty}</b>
                      · {note.line_count ?? note.lines?.length ?? 0} material line(s)
                    </p>
                  </div>
                  <span className="text-gray-400 text-xs">{open ? '▲' : '▼'}</span>
                </button>
                {open && note.lines && note.lines.length > 0 && (
                  <table className="w-full text-xs border-t">
                    <thead className="text-gray-400 uppercase bg-gray-50">
                      <tr>
                        <th className="text-left px-4 py-2">For (finished item)</th>
                        <th className="text-left px-4 py-2">Material to issue</th>
                        <th className="text-right px-4 py-2">BOM / unit</th>
                        <th className="text-right px-4 py-2">Required qty</th>
                        <th className="text-left px-4 py-2">Unit</th>
                      </tr>
                    </thead>
                    <tbody>
                      {note.lines.map(ln => (
                        <tr key={ln.id} className="border-t border-gray-50">
                          <td className="px-4 py-2">
                            <span className="font-mono font-semibold text-[#002B5B]">{ln.finished_item_code}</span>
                            <span className="text-gray-400 ml-1">× {ln.finished_planned_qty}</span>
                          </td>
                          <td className="px-4 py-2">
                            <span className="font-mono font-semibold">{ln.material_code}</span>
                            {ln.material_name && ln.material_name !== ln.material_code && (
                              <span className="text-gray-500 ml-1">— {ln.material_name}</span>
                            )}
                          </td>
                          <td className="px-4 py-2 text-right text-gray-600">{ln.bom_qty_per_unit}</td>
                          <td className="px-4 py-2 text-right font-bold text-indigo-700">{ln.required_qty}</td>
                          <td className="px-4 py-2 text-gray-500">{ln.unit}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {open && (!note.lines || note.lines.length === 0) && (
                  <p className="text-xs text-gray-400 px-4 py-3 border-t">{note.remarks || 'No BOM lines.'}</p>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* REPORTS TAB */}
      {tab === 'reports' && (
        <div className="space-y-4">
          <div className="flex gap-2">
            {([['cutting', 'Cutting'], ['process', 'All processes']] as const).map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setReportsView(key)}
                className={`px-3 py-1.5 text-xs rounded-lg font-medium ${reportsView === key ? 'bg-[#002B5B] text-white' : 'bg-white border text-gray-600'}`}
              >
                {label}
              </button>
            ))}
          </div>
          {reportsView === 'cutting' ? (
            <CuttingReportsPanel />
          ) : (
        <div className="space-y-4">
          <h3 className="font-semibold text-gray-700">Process-wise Issue / Receive / Balance Report</h3>
          {allProcesses.map(proc => {
            const procRows = (processReport as any[]).filter(r => r.process === proc)
            if (procRows.length === 0) return null
            return (
              <div key={proc} className="bg-white rounded-xl border overflow-hidden">
                <div className={`px-4 py-2 text-sm font-semibold ${PROCESS_COLORS[proc] || 'bg-gray-100 text-gray-700'}`}>
                  {PROCESS_ICONS[proc] || ''} {proc}
                </div>
                <table className="w-full text-sm">
                  <thead className="text-gray-400 text-xs uppercase bg-gray-50">
                    <tr>
                      <th className="text-left px-4 py-2">SO</th>
                      <th className="text-left px-4 py-2">SKU</th>
                      <th className="text-right px-4 py-2">Planned</th>
                      <th className="text-right px-4 py-2">Issued</th>
                      <th className="text-right px-4 py-2">Received</th>
                      <th className="text-right px-4 py-2">Rejected</th>
                      <th className="text-right px-4 py-2">Balance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {procRows.map((r: any, i: number) => (
                      <tr key={i} className="border-t border-gray-50">
                        <td className="px-4 py-2 font-semibold text-[#002B5B]">{r.so_number}</td>
                        <td className="px-4 py-2 font-mono text-xs">{r.sku}</td>
                        <td className="px-4 py-2 text-right">{fmt(r.planned)}</td>
                        <td className="px-4 py-2 text-right text-blue-600">{fmt(r.issued)}</td>
                        <td className="px-4 py-2 text-right text-green-600 font-semibold">{fmt(r.received)}</td>
                        <td className="px-4 py-2 text-right text-red-500">{fmt(r.rejected)}</td>
                        <td className={`px-4 py-2 text-right font-bold ${(r.balance || 0) > 0 ? 'text-amber-600' : 'text-green-600'}`}>{fmt(r.balance)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          })}
          {processReport.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No data yet.</p>}
        </div>
          )}
        </div>
      )}

      {/* SETS TAB — Set BOM + Set Match */}
      {tab === 'sets' && (
        <SetBomPanel processes={allProcesses} showSetMatch />
      )}

      {/* MRP TAB */}
      {tab === 'mrp' && (
        <MRPTab
          onCreateJO={(p) => {
            setNewForm(f => ({
              ...f,
              so_number: p.so_number,
              fabric_code: p.fabric_code,
              fabric_qty: p.fabric_qty,
              process: 'Cutting',
              sku_name: p.fabric_name,
            }))
            setActiveProcess('Cutting')
            setTab('process')
            setModal('new-jo')
          }}
        />
      )}

      {/* ── NEW JO MODAL ─────────────────────────────────────────────────────── */}
      {modal === 'new-jo' && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4 overflow-y-auto">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl p-6 space-y-4 my-4">
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-gray-700 text-lg">
                {PROCESS_ICONS[newForm.process] || ''} New {newForm.process} Job Order
              </h3>
              <button onClick={() => setModal(null)} className="text-gray-400 hover:text-gray-600 text-xl">✕</button>
            </div>

            {/* Routing preview */}
            {itemRouting?.routing && (
              <div className="flex items-center gap-1 flex-wrap">
                {(itemRouting.routing as string[]).map((p, i) => (
                  <span key={p}>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${p === newForm.process ? 'bg-[#002B5B] text-white' : 'bg-gray-100 text-gray-500'}`}>
                      {PROCESS_ICONS[p] || ''} {p}
                    </span>
                    {i < itemRouting.routing.length - 1 && <span className="text-gray-300 text-xs mx-1">→</span>}
                  </span>
                ))}
              </div>
            )}

            {newForm.from_ready_to && (
              <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                Opened from Ready to {newForm.process}. Choose production path and qty for this JO — the same SO/SKU can be split across In-house, Cut-to-Pack, and Stitch-to-Pack.
              </p>
            )}
            {pathCommitment && Number(pathCommitment.total_planned || 0) > 0 && (
              <p className="text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2">
                Already on open JOs for this SO/SKU: <b>{pathCommitment.total_planned}</b> pcs
                {Object.entries(pathCommitment.by_mode || {}).map(([mode, qty]) => (
                  <span key={mode} className="ml-2">{PRODUCTION_MODE_LABEL[mode] || mode}: {qty as number}</span>
                ))}
              </p>
            )}

            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {/* SO source + number — manual does not create a Sales SO record */}
              <div className="md:col-span-3 flex flex-wrap gap-3 items-end">
                <div>
                  <label className="text-xs text-gray-500 block mb-1">SO reference *</label>
                  <div className="flex rounded-lg border border-gray-200 overflow-hidden text-xs">
                    <button
                      type="button"
                      disabled={newForm.from_ready_to}
                      onClick={() => {
                        setNewForm(f => ({ ...f, so_source: 'system', so_number: '', sku: '', sku_name: '', from_ready_to: false }))
                        setNewLines([])
                      }}
                      className={`px-3 py-1.5 ${newForm.so_source === 'system' ? 'bg-[#002B5B] text-white' : 'bg-white text-gray-600'} disabled:opacity-50`}
                    >
                      Existing SO
                    </button>
                    <button
                      type="button"
                      disabled={newForm.from_ready_to}
                      onClick={() => {
                        setNewForm(f => ({ ...f, so_source: 'manual', sku: '', sku_name: '', from_ready_to: false }))
                        setNewLines([])
                      }}
                      className={`px-3 py-1.5 border-l ${newForm.so_source === 'manual' ? 'bg-[#002B5B] text-white' : 'bg-white text-gray-600'} disabled:opacity-50`}
                    >
                      Manual SO #
                    </button>
                  </div>
                </div>
                <div className="flex-1 min-w-[12rem]">
                  <label className="text-xs text-gray-500">SO Number *</label>
                  {newForm.so_source === 'manual' ? (
                    <input
                      value={newForm.so_number}
                      onChange={e => setNewForm(f => ({ ...f, so_number: e.target.value.trim() }))}
                      placeholder="e.g. SO-OTS-001 (not created in Sales)"
                      className="w-full border border-amber-200 bg-amber-50 rounded px-2 py-1.5 text-sm mt-1"
                    />
                  ) : newForm.from_ready_to ? (
                    <input
                      value={newForm.so_number}
                      readOnly
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1 bg-gray-50"
                    />
                  ) : (
                    <select value={newForm.so_number} onChange={e => {
                      setNewForm(f => ({ ...f, so_number: e.target.value, sku: '', sku_name: '', from_ready_to: false }))
                      setNewLines([])
                      setSOLineSearch('')
                    }}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                      <option value="">Select SO</option>
                      {(soList as any[]).map((s: any) => <option key={s.so_number} value={s.so_number}>{s.so_number} — {s.buyer || ''}</option>)}
                    </select>
                  )}
                  {newForm.so_source === 'manual' && (
                    <p className="text-[10px] text-amber-800 mt-1">Stored on the JO only — no Sales Order is created.</p>
                  )}
                </div>
              </div>
              {/* Process */}
              <div><label className="text-xs text-gray-500">Process *</label>
                <select value={newForm.process} onChange={e => setNewForm(f => ({ ...f, process: e.target.value }))}
                  disabled={newForm.from_ready_to}
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1 disabled:bg-gray-50">
                  {allProcesses.map(p => <option key={p}>{p}</option>)}
                </select>
              </div>
              {/* Production path — decided at JO creation (Ready-to-Cut), not SO level */}
              <div className="md:col-span-2">
                <label className="text-xs text-gray-500">Production path *</label>
                <select
                  value={newForm.production_mode}
                  onChange={e => setNewForm(f => ({ ...f, production_mode: e.target.value }))}
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1"
                >
                  {PRODUCTION_MODE_OPTIONS.map(m => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))}
                </select>
                <p className="text-[10px] text-gray-500 mt-1">
                  Path is chosen here when creating the JO. Enter planned qty below to split the same SKU across paths (e.g. 500 in-house + 500 Cut-to-Pack).
                  {itemRouting?.routing?.length ? ` Routing: ${itemRouting.routing.join(' → ')}` : ''}
                </p>
              </div>
              {/* Execution / vendor */}
              <div><label className="text-xs text-gray-500">Execution type</label>
                <select
                  value={newForm.exec_type}
                  onChange={e => {
                    const v = e.target.value
                    setNewForm(f => ({
                      ...f,
                      exec_type: v,
                      vendor_name: isOutsourceExec(v) ? f.vendor_name : '',
                    }))
                  }}
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1"
                >
                  {EXEC_TYPE_OPTIONS.map(o => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </div>
              {isOutsourceExec(newForm.exec_type) && (
                <div className="md:col-span-2">
                  <label className="text-xs text-gray-500">Vendor name *</label>
                  <input
                    list="new-jo-vendor-suggestions"
                    value={newForm.vendor_name}
                    onChange={e => setNewForm(f => ({ ...f, vendor_name: e.target.value }))}
                    placeholder="Outsource vendor / party name"
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1"
                  />
                  <datalist id="new-jo-vendor-suggestions">
                    {vendorSuggestions.map(v => (
                      <option key={v} value={v} />
                    ))}
                  </datalist>
                </div>
              )}
              <div>
                <label className="text-xs text-gray-500">Delivery Date</label>
                <input
                  type="date"
                  value={newForm.expected_completion}
                  onChange={e => setNewForm(f => ({ ...f, expected_completion: e.target.value }))}
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">Rate (₹)</label>
                <input
                  type="number"
                  value={newForm.vendor_rate || ''}
                  onChange={e => setNewForm(f => ({ ...f, vendor_rate: +e.target.value }))}
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">Remarks</label>
                <input
                  value={newForm.remarks}
                  onChange={e => setNewForm(f => ({ ...f, remarks: e.target.value }))}
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1"
                />
              </div>
              {newForm.process === 'Cutting' && (
                <>
                  <div><label className="text-xs text-gray-500">Fabric Code (from BOM)</label>
                    <input value={newForm.fabric_code} onChange={e => setNewForm(f => ({ ...f, fabric_code: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1 font-mono" /></div>
                  <div><label className="text-xs text-gray-500">Fabric Qty (MTR)</label>
                    <input type="number" value={newForm.fabric_qty} onChange={e => setNewForm(f => ({ ...f, fabric_qty: +e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                </>
              )}
            </div>

            {newForm.process === 'Cutting' && cuttingSetBomInfo?.has_set_bom && cuttingMainSku && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                <p className="font-semibold">Multi-component set — creates separate Cutting JOs</p>
                <p className="mt-1 font-mono">
                  {(cuttingSetBomInfo.cutting_components || cuttingSetBomInfo.bom?.lines || [])
                    .filter((l: { component_role?: string }) => String(l.component_role || 'SET_COMPONENT').toUpperCase() !== 'PANEL')
                    .map((l: { component_code: string }) => `${cuttingMainSku}-${l.component_code}`)
                    .join(' · ')}
                </p>
                {(cuttingSetBomInfo.panels || []).length > 0 && (
                  <p className="mt-1 text-amber-800/80">
                    Panels (no JO): {(cuttingSetBomInfo.panels || []).map((l: { component_code: string; parent_component_code?: string }) =>
                      `${l.component_code}${l.parent_component_code ? `→${l.parent_component_code}` : ''}`
                    ).join(', ')}
                  </p>
                )}
              </div>
            )}

            {/* Validation message */}
            {newForm.process !== 'Cutting' && newForm.so_number && newForm.sku && joValidation && (
              <div className={`rounded-lg px-3 py-2 text-xs ${joValidation.ok ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
                {joValidation.ok ? `✅ ${joValidation.available} pieces available` : `❌ ${joValidation.message}`}
              </div>
            )}

            {/* Manual SO — SKU dropdown / qty (no SO master lines required) */}
            {newForm.so_source === 'manual' && (
              <div className="border border-amber-200 bg-amber-50/50 rounded-xl p-3 grid grid-cols-2 md:grid-cols-4 gap-2">
                <div className="md:col-span-2">
                  <label className="text-xs text-gray-500">SKU *</label>
                  <ManualJoSkuDropdown
                    value={newForm.sku}
                    catalog={manualJoSkuCatalog}
                    onPick={(sku, skuName) =>
                      setNewForm(f => ({
                        ...f,
                        sku,
                        sku_name: skuName || f.sku_name,
                      }))
                    }
                  />
                  <p className="text-[10px] text-gray-500 mt-1">
                    Pick from open SO lines / ready-to-cut / prior JOs, or search Item Master.
                  </p>
                </div>
                <div>
                  <label className="text-xs text-gray-500">SKU name</label>
                  <input
                    value={newForm.sku_name}
                    onChange={e => setNewForm(f => ({ ...f, sku_name: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500">Planned qty *</label>
                  <input
                    type="number"
                    value={newForm.planned_qty || ''}
                    onChange={e => setNewForm(f => ({ ...f, planned_qty: +e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1"
                  />
                </div>
                <div className="flex items-end md:col-span-4 gap-2">
                  <button
                    type="button"
                    className="px-4 py-1.5 text-xs bg-[#002B5B] text-white rounded-lg disabled:opacity-50"
                    disabled={!newForm.sku.trim() || !(Number(newForm.planned_qty) > 0)}
                    onClick={() => {
                      const sku = (newForm.sku || '').trim()
                      const qty = Number(newForm.planned_qty) || 0
                      if (!sku || qty <= 0) return
                      const nextLine = {
                        so_number: newForm.so_number,
                        sku,
                        sku_name: newForm.sku_name || '',
                        style: '',
                        planned_qty: qty,
                        so_qty: qty,
                        vendor_rate: newForm.vendor_rate || 0,
                        remarks: '',
                      }
                      setNewLines(ls => appendManualJoLine(ls, nextLine))
                      // Clear pickers so the next size can be entered for the same JO
                      setNewForm(f => ({
                        ...f,
                        sku: '',
                        sku_name: '',
                        planned_qty: 0,
                      }))
                    }}
                  >
                    Add as JO line
                  </button>
                  {newLines.length > 0 && (
                    <span className="text-[11px] text-gray-600 self-center">
                      {newLines.length} line{newLines.length === 1 ? '' : 's'} on this JO — add another size/SKU if needed
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* SO / Ready-To lines — add to JO */}
            {newForm.so_source === 'system' && newForm.so_number && (newForm.from_ready_to ? readyLinesForJo.length > 0 : soLines.length > 0) && (
              <div className="border rounded-xl overflow-hidden">
                <div className="px-3 py-2 bg-blue-50 text-xs font-semibold text-blue-700 flex flex-wrap justify-between gap-2 items-center">
                  <span>
                    {newForm.from_ready_to
                      ? `Ready to ${newForm.process} on this SO — only sizes currently at this stage`
                      : 'SO Lines — all styles (live list). Search & scroll to find SKUs.'}
                  </span>
                  <div className="flex items-center gap-2">
                    <input
                      type="search"
                      value={soLineSearch}
                      onChange={e => setSOLineSearch(e.target.value)}
                      placeholder="Search SKU…"
                      className="border rounded px-2 py-1 text-xs font-normal w-40"
                    />
                    <button onClick={() => joLinePickerRows.forEach(addSOLineToJO)} className="text-blue-600 hover:underline">Add all shown</button>
                  </div>
                </div>
                <div className="max-h-56 overflow-y-auto">
                  <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-gray-50 z-10"><tr className="text-gray-400 border-b">
                      <th className="text-left px-3 py-1.5">SKU</th><th className="text-left px-3 py-1.5">Name</th>
                      <th className="text-right px-3 py-1.5">{newForm.from_ready_to ? 'Ready qty' : 'SO Qty'}</th><th className="px-3 py-1.5"></th>
                    </tr></thead>
                    <tbody>
                      {joLinePickerRows.map((l: any) => {
                        const added = newLines.some(nl => nl.sku === l.sku)
                        return (
                          <tr key={l.sku} className="border-t hover:bg-gray-50">
                            <td className="px-3 py-1.5 font-mono font-semibold break-all">{l.sku}</td>
                            <td className="px-3 py-1.5 text-gray-600 break-words">{l.sku_name || l.item_name || '—'}</td>
                            <td className="px-3 py-1.5 text-right">{l.available_qty ?? l.qty}</td>
                            <td className="px-3 py-1.5">
                              <button onClick={() => added ? setNewLines(ls => ls.filter(nl => nl.sku !== l.sku)) : addSOLineToJO(l)}
                                className={`px-2 py-0.5 rounded text-xs font-medium ${added ? 'bg-green-100 text-green-700' : 'bg-[#002B5B] text-white hover:bg-blue-800'}`}>
                                {added ? '✓ Added' : '+ Add'}
                              </button>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
                {joLinePickerRows.length === 0 && (
                  <p className="text-xs text-amber-600 px-3 py-2">No SKUs match &quot;{soLineSearch}&quot;.</p>
                )}
              </div>
            )}

            {/* JO Lines */}
            {newLines.length > 0 && (
              <div className="border rounded-xl overflow-hidden">
                <div className="px-3 py-2 bg-gray-50 text-xs font-semibold text-gray-600 flex flex-wrap justify-between gap-2">
                  <span>JO Lines ({newLines.length})</span>
                  {newForm.process === 'Cutting' && (
                    <span className="font-normal text-amber-800">
                      Edit Cutting Qty freely — it may be lower than SO (demand cut / fabric reject). Fabric MTR re-scales from BOM.
                    </span>
                  )}
                </div>
                <div className="max-h-52 overflow-y-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-white"><tr className="text-gray-400 border-b">
                    <th className="text-left px-3 py-1.5">SKU</th><th className="text-left px-3 py-1.5">Style</th>
                    {newForm.process === 'Cutting' && <th className="text-right px-3 py-1.5">SO Qty</th>}
                    <th className="text-right px-3 py-1.5">{newForm.process === 'Cutting' ? 'Cutting Qty' : 'Planned Qty'}</th>
                    <th className="text-right px-3 py-1.5">Rate (₹)</th>
                    <th className="text-left px-3 py-1.5">Remarks</th>
                    <th className="text-right px-3 py-1.5">Amount</th>
                    <th className="px-3 py-1.5"></th>
                  </tr></thead>
                  <tbody>
                    {newLines.map((ln, i) => (
                      <tr key={i} className="border-t">
                        <td className="px-3 py-1 font-mono font-semibold text-[#002B5B] break-all">{ln.sku}</td>
                        <td className="px-3 py-1"><input value={ln.style} onChange={e => setNewLines(ls => ls.map((x,j) => j===i ? {...x, style: e.target.value} : x))}
                          placeholder="Style/desc" className="border rounded px-1.5 py-0.5 text-xs w-full" /></td>
                        {newForm.process === 'Cutting' && (
                          <td className="px-3 py-1 text-right text-gray-500">{(ln as { so_qty?: number }).so_qty ?? '—'}</td>
                        )}
                        <td className="px-3 py-1">
                          <input
                            type="number"
                            value={ln.planned_qty}
                            onChange={e => {
                              const v = +e.target.value
                              if (newForm.process === 'Cutting') void updateCuttingLineQty(i, v)
                              else setNewLines(ls => ls.map((x, j) => j === i ? { ...x, planned_qty: v } : x))
                            }}
                            className="border border-amber-200 bg-amber-50 rounded px-1.5 py-0.5 text-xs w-20 text-right font-semibold"
                          />
                        </td>
                        <td className="px-3 py-1"><input type="number" value={ln.vendor_rate} onChange={e => setNewLines(ls => ls.map((x,j) => j===i ? {...x, vendor_rate: +e.target.value} : x))}
                          className="border rounded px-1.5 py-0.5 text-xs w-20 text-right" /></td>
                        <td className="px-3 py-1"><input value={ln.remarks} onChange={e => setNewLines(ls => ls.map((x,j) => j===i ? {...x, remarks: e.target.value} : x))}
                          placeholder="Remarks" className="border rounded px-1.5 py-0.5 text-xs w-full" /></td>
                        <td className="px-3 py-1 text-right font-semibold">{fmtR(ln.planned_qty * ln.vendor_rate)}</td>
                        <td className="px-3 py-1"><button onClick={() => setNewLines(ls => ls.filter((_,j) => j!==i))} className="text-red-400 hover:text-red-600">✕</button></td>
                      </tr>
                    ))}
                    <tr className="border-t bg-gray-50 font-semibold">
                      <td colSpan={newForm.process === 'Cutting' ? 6 : 5} className="px-3 py-1.5 text-right text-xs text-gray-600">Total:</td>
                      <td className="px-3 py-1.5 text-right text-xs">{fmtR(newLines.reduce((s,l) => s + l.planned_qty * l.vendor_rate, 0))}</td>
                      <td></td>
                    </tr>
                  </tbody>
                </table>
                </div>
              </div>
            )}

            <div className="flex gap-2 pt-2">
              <button onClick={() => {
                const lines = newLines.length
                  ? newLines
                  : (newForm.so_source === 'manual' && newForm.sku.trim()
                    ? [{
                        so_number: newForm.so_number,
                        sku: newForm.sku.trim(),
                        sku_name: newForm.sku_name,
                        style: '',
                        planned_qty: newForm.planned_qty || 1,
                        vendor_rate: newForm.vendor_rate || 0,
                        remarks: newForm.remarks || '',
                      }]
                    : [])
                const totalPlanned = lines.reduce((s, l) => s + (Number(l.planned_qty) || 0), 0)
                  || newForm.planned_qty
                  || 0
                const headerSku = (newForm.sku || '').trim() || (lines[0]?.sku || '')
                const headerName = (newForm.sku_name || '').trim() || (lines[0]?.sku_name || '')
                const { from_ready_to: _fromReady, ...formPayload } = newForm
                createJOMut.mutate({
                  ...formPayload,
                  so_source: newForm.so_source || 'system',
                  sku: headerSku,
                  sku_name: headerName,
                  planned_qty: totalPlanned,
                  lines,
                })
              }}
                disabled={
                  createJOMut.isPending
                  || !newForm.so_number.trim()
                  || (newForm.so_source === 'manual' && newForm.process === 'Cutting'
                    && !newForm.sku.trim() && newLines.length === 0)
                  || (isOutsourceExec(newForm.exec_type) && !newForm.vendor_name.trim())
                  || (newForm.process !== 'Cutting' && joValidation && !joValidation?.ok)
                }
                className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                {createJOMut.isPending ? 'Creating…' : `Create ${newForm.process} JO`}
              </button>
              <button onClick={() => { setModal(null); setNewLines([]) }} className="px-4 border rounded-lg text-sm text-gray-600">Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* ── ISSUE FABRIC MODAL ───────────────────────────────────────────────── */}
      {modal === 'issue-fabric' && activeJO && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-gray-700">📦 Issue Fabric — {activeJO.jo_number}</h3>
              <button onClick={() => setModal(null)} className="text-gray-400 text-xl">✕</button>
            </div>
            <div className="bg-blue-50 rounded-lg p-3 text-xs text-blue-700">
              Fabric: <b>{activeJO.fabric_code}</b> · Planned: <b>{activeJO.fabric_qty} {activeJO.fabric_unit}</b> · Already issued: <b>{activeJO.fabric_issued_qty || 0}</b>
              {activeJO.process === 'Embroidery' && (activeJO.garment_qty || 0) > 0 && (
                <span className="block mt-1 text-green-800">
                  Fabric based on <b>{activeJO.garment_qty} garment pcs</b>
                  {activeJO.measurement_qty
                    ? ` (embroidery JO qty ${activeJO.measurement_qty} ${activeJO.embroidery_unit || ''} is separate)`
                    : ''}
                </span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-3">
              {([['fabric_code','Fabric Code'],['fabric_name','Fabric Name'],['issued_by','Issued By'],['remarks','Remarks']] as const).map(([k,l]) => (
                <div key={k}><label className="text-xs text-gray-500">{l}</label>
                  <input value={(fabricIssueForm as any)[k]} onChange={e => setFabricIssueForm(f => ({ ...f, [k]: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
              ))}
              <div><label className="text-xs text-gray-500">Issue Qty (MTR) *</label>
                <input type="number" value={fabricIssueForm.issued_qty} onChange={e => setFabricIssueForm(f => ({ ...f, issued_qty: +e.target.value }))}
                  className="w-full border border-blue-200 rounded px-2 py-1.5 text-sm mt-1 bg-blue-50 font-semibold" /></div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => issueFabricMut.mutate({ id: activeJO.id, data: fabricIssueForm })}
                disabled={issueFabricMut.isPending || !fabricIssueForm.issued_qty}
                className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">
                {issueFabricMut.isPending ? 'Saving…' : '📦 Issue Fabric'}
              </button>
              <button onClick={() => setModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* ── RETURN FABRIC MODAL ──────────────────────────────────────────────── */}
      {modal === 'return-fabric' && activeJO && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-gray-700">↩️ Return Fabric — {activeJO.jo_number}</h3>
              <button onClick={() => setModal(null)} className="text-gray-400 text-xl">✕</button>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {([['fabric_code','Fabric Code'],['returned_by','Returned By'],['remarks','Remarks']] as const).map(([k,l]) => (
                <div key={k}><label className="text-xs text-gray-500">{l}</label>
                  <input value={(fabricReturnForm as any)[k]} onChange={e => setFabricReturnForm(f => ({ ...f, [k]: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
              ))}
              <div><label className="text-xs text-gray-500">Returned Qty (MTR) *</label>
                <input type="number" value={fabricReturnForm.returned_qty} onChange={e => setFabricReturnForm(f => ({ ...f, returned_qty: +e.target.value }))}
                  className="w-full border border-amber-200 rounded px-2 py-1.5 text-sm mt-1 bg-amber-50 font-semibold" /></div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => returnFabricMut.mutate({ id: activeJO.id, data: fabricReturnForm })}
                disabled={returnFabricMut.isPending} className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">
                {returnFabricMut.isPending ? 'Saving…' : '↩️ Return'}
              </button>
              <button onClick={() => setModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* ── RECEIVE MODAL ────────────────────────────────────────────────────── */}
      {modal === 'receive' && activeJO && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-gray-700">✅ Receive — {activeJO.jo_number}</h3>
              <button onClick={() => setModal(null)} className="text-gray-400 text-xl">✕</button>
            </div>
            <div className="bg-green-50 rounded-lg p-3 text-xs text-green-700">
              Process: <b>{activeJO.process}</b>
              {' · '}
              Planned: <b>
                {activeJO.process === 'Embroidery' && (activeJO as any).measurement_qty
                  ? `${(activeJO as any).measurement_qty} ${(activeJO as any).embroidery_unit || ''} ${(activeJO as any).embroidery_type || ''}`.trim()
                  : `${activeJO.planned_qty} pcs`}
              </b>
              {' · '}
              Received so far: <b>{activeJO.received_qty}{activeJO.process === 'Embroidery' && (activeJO as any).embroidery_unit ? ` ${(activeJO as any).embroidery_unit}` : ' pcs'}</b>
              {(activeJO as any).garment_qty > 0 && activeJO.process === 'Embroidery' && (
                <span className="block mt-1 text-green-800">Garment pieces covered: {(activeJO as any).garment_qty}</span>
              )}
              {activeJO.process === 'Cutting' && (
                <span className="block mt-1 text-green-800">
                  Extra pieces can be received with no percentage cap (temporary). Under-receive is always allowed.
                </span>
              )}
            </div>
            {activeJO.process === 'Cutting' && receiveSetBomInfo?.has_set_bom && (() => {
              const receiveParentCode = receiveSku.includes('-') ? receiveSku.split('-').pop()?.toUpperCase() : ''
              const childPanels = (receiveSetBomInfo.panels || []).filter((p: { parent_component_code?: string }) =>
                !receiveParentCode || !p.parent_component_code || String(p.parent_component_code).toUpperCase() === receiveParentCode,
              )
              const setComponents = (receiveSetBomInfo.cutting_components || receiveSetBomInfo.bom?.lines || [])
                .filter((l: { component_role?: string }) => String(l.component_role || 'SET_COMPONENT').toUpperCase() !== 'PANEL')
              return (
                <label className="flex items-start gap-2 text-xs text-gray-700 bg-indigo-50 border border-indigo-100 rounded-lg p-3">
                  <input type="checkbox" className="mt-0.5" checked={!!receiveForm.split_components}
                    onChange={e => setReceiveForm(f => ({ ...f, split_components: e.target.checked }))} />
                  <span>
                    {childPanels.length > 0 && receiveParentCode
                      ? `Create panel stock after receive (${childPanels.map((p: { component_code: string }) => p.component_code).join(' + ')})`
                      : 'Split into components after receive'}
                    {childPanels.length > 0 && (
                      <span className="block text-indigo-700 mt-0.5 font-mono">
                        Panels: {childPanels.map((p: { component_code: string; routing?: string }) =>
                          `${p.component_code}${p.routing ? ` (${p.routing})` : ''}`,
                        ).join(' · ')}
                      </span>
                    )}
                    {!receiveParentCode && setComponents.length > 0 && (
                      <span className="block text-indigo-700 mt-0.5 font-mono">
                        Components: {setComponents.map((l: { component_code: string }) => l.component_code).join(' + ')}
                        {childPanels.length > 0 && ` · Panels: ${childPanels.map((p: { component_code: string }) => p.component_code).join(' + ')}`}
                      </span>
                    )}
                  </span>
                </label>
              )
            })()}
            <div className="grid grid-cols-2 gap-3">
              <div><label className="text-xs text-gray-500">Received Qty{activeJO.process === 'Embroidery' ? ` (${(activeJO as any).embroidery_unit || 'units'})` : ' (pcs)'} *</label>
                <input type="number" value={receiveForm.received_qty} onChange={e => setReceiveForm(f => ({ ...f, received_qty: +e.target.value }))}
                  className="w-full border border-green-200 rounded px-2 py-1.5 text-sm mt-1 bg-green-50 font-semibold" /></div>
              <div><label className="text-xs text-gray-500">Rejected Qty</label>
                <input type="number" value={receiveForm.rejected_qty} onChange={e => setReceiveForm(f => ({ ...f, rejected_qty: +e.target.value }))}
                  className="w-full border border-red-200 rounded px-2 py-1.5 text-sm mt-1 bg-red-50" /></div>
              {activeJO.process === 'Embroidery' && (
                <div className="col-span-2">
                  <label className="text-xs text-gray-500">Leftover to stock ({(activeJO as any).embroidery_unit || 'units'}) — unused border/Yog returned</label>
                  <input type="number" step="0.01" min={0} value={receiveForm.leftover_measurement || ''}
                    onChange={e => setReceiveForm(f => ({ ...f, leftover_measurement: +e.target.value }))}
                    className="w-full border border-amber-200 rounded px-2 py-1.5 text-sm mt-1 bg-amber-50" />
                </div>
              )}
              {([['received_by','Received By'],['remarks','Remarks']] as const).map(([k,l]) => (
                <div key={k}><label className="text-xs text-gray-500">{l}</label>
                  <input value={(receiveForm as any)[k]} onChange={e => setReceiveForm(f => ({ ...f, [k]: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
              ))}
            </div>
            <div className="flex gap-2">
              <button onClick={() => {
                const line = activeLineId ? activeJO.lines.find(l => l.id === activeLineId) : null
                receiveMut.mutate({
                  id: activeJO.id,
                  data: {
                    ...receiveForm,
                    process: activeJO.process,
                    sku: line?.sku || activeJO.sku,
                    jo_line_id: (activeLineId ?? activeJO.lines?.[0]?.id) || undefined,
                  },
                })
              }}
                disabled={receiveMut.isPending || !receiveForm.received_qty}
                className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">
                {receiveMut.isPending ? 'Saving…' : '✅ Confirm Receipt'}
              </button>
              <button onClick={() => setModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* ── ISSUE PIECES MODAL ───────────────────────────────────────────────── */}
      {modal === 'issue-pieces' && activeJO && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-gray-700">➡️ Issue to Next Process — {activeJO.jo_number}</h3>
              <button onClick={() => setModal(null)} className="text-gray-400 text-xl">✕</button>
            </div>
            <div className="bg-purple-50 rounded-lg p-3 text-xs text-purple-700">
              SKU: <b className="font-mono">{issuePiecesSku || activeJO.sku}</b>
              <br />
              From: <b>{issueFromProcess || activeJO.process}</b> → To: <b>{issuePiecesForm.to_process || activeJO.next_process || '—'}</b>
              {!(issuePiecesForm.to_process || activeJO.next_process) && (
                <span className="block mt-1 text-amber-800">No next process configured for this routing.</span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div><label className="text-xs text-gray-500">Panel / SKU</label>
                <input value={issuePiecesSku || activeJO.sku} readOnly
                  className="w-full border rounded px-2 py-1.5 text-sm mt-1 font-mono bg-gray-50" /></div>
              <div><label className="text-xs text-gray-500">From process</label>
                <input value={issueFromProcess || activeJO.process} readOnly
                  className="w-full border rounded px-2 py-1.5 text-sm mt-1 bg-gray-50" /></div>
              <div><label className="text-xs text-gray-500">To Process (from routing)</label>
                <input
                  value={issuePiecesForm.to_process}
                  readOnly
                  className="w-full border rounded px-2 py-1.5 text-sm mt-1 bg-gray-50 font-semibold"
                />
              </div>
              <div><label className="text-xs text-gray-500">Issue Qty (pcs) *</label>
                <input type="number" value={issuePiecesForm.issued_qty} onChange={e => setIssuePiecesForm(f => ({ ...f, issued_qty: +e.target.value }))}
                  className="w-full border border-purple-200 rounded px-2 py-1.5 text-sm mt-1 bg-purple-50 font-semibold" /></div>
              {([['issued_by','Issued By'],['remarks','Remarks']] as const).map(([k,l]) => (
                <div key={k}><label className="text-xs text-gray-500">{l}</label>
                  <input value={(issuePiecesForm as any)[k]} onChange={e => setIssuePiecesForm(f => ({ ...f, [k]: e.target.value }))}
                    className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
              ))}
            </div>
            <div className="flex gap-2">
              <button onClick={() => issuePiecesMut.mutate({
                id: activeJO.id,
                data: {
                  ...issuePiecesForm,
                  from_process: issueFromProcess || activeJO.process,
                  sku: issuePiecesSku || activeJO.sku,
                  jo_line_id: activeLineId ?? undefined,
                },
              })}
                disabled={issuePiecesMut.isPending || !issuePiecesForm.issued_qty || !issuePiecesForm.to_process}
                className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">
                {issuePiecesMut.isPending ? 'Saving…' : '➡️ Issue Pieces'}
              </button>
              <button onClick={() => setModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* ── ADD COST MODAL ───────────────────────────────────────────────────── */}
      {modal === 'add-cost' && activeJO && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-gray-700">💰 Add Cost — {activeJO.jo_number}</h3>
              <button onClick={() => setModal(null)} className="text-gray-400 text-xl">✕</button>
            </div>
            <div className="bg-amber-50 rounded-lg p-3 text-xs text-amber-700">Current total: <b>₹{fmt(activeJO.total_cost || 0)}</b></div>
            <div className="grid grid-cols-2 gap-3">
              <div><label className="text-xs text-gray-500">Cost Type</label>
                <select value={costForm.cost_type} onChange={e => setCostForm(f => ({ ...f, cost_type: e.target.value }))}
                  className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                  {['Labour','Machine','Material','Overhead','Other'].map(t => <option key={t}>{t}</option>)}
                </select>
              </div>
              <div><label className="text-xs text-gray-500">Amount (₹) *</label>
                <input type="number" value={costForm.amount} onChange={e => setCostForm(f => ({ ...f, amount: +e.target.value }))}
                  className="w-full border border-amber-200 rounded px-2 py-1.5 text-sm mt-1 bg-amber-50 font-semibold" /></div>
              <div className="col-span-2"><label className="text-xs text-gray-500">Description</label>
                <input value={costForm.description} onChange={e => setCostForm(f => ({ ...f, description: e.target.value }))}
                  className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => addCostMut.mutate({ id: activeJO.id, data: { ...costForm, process: activeJO.process } })}
                disabled={addCostMut.isPending || !costForm.amount}
                className="flex-1 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50">
                {addCostMut.isPending ? 'Saving…' : '💰 Add Cost'}
              </button>
              <button onClick={() => setModal(null)} className="px-4 border rounded-lg text-sm">Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
