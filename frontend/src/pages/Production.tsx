import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'
import api from '../api/client'

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
}

const PROCESS_ICONS: Record<string, string> = {
  Cutting: '✂️', Printing: '🖨️', Embroidery: '🧶',
  Stitching: '🧵', Finishing: '✨', Packing: '📦',
  'Kajh Button': '🔘',
}

const fmt = (n: number) => Math.round(n || 0).toLocaleString('en-IN')
const fmtR = (n: number) => '₹' + Math.round(n || 0).toLocaleString('en-IN')

const EXEC_TYPE_OPTIONS = [
  { value: 'Inhouse', label: 'In-house' },
  { value: 'Outsource', label: 'Outsource' },
] as const

function isOutsourceExec(execType: string) {
  return String(execType || '').trim().toLowerCase() === 'outsource'
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
const printJO = (jo: JO) => {
  const totalCost = jo.lines.reduce((s, l) => s + (l.planned_qty * l.vendor_rate), 0)
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
    <div><div class="doc-title">JOB ORDER</div><div class="doc-num">${jo.jo_number}</div></div>
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

  const activeSONumbers: string[] = mrpResult?.so_numbers || lastMRP?.so_numbers || []

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
    } catch (e) { alert('Material requirement planning run failed') }
    setRunning(false)
  }

  const toggleSO = (so: string) => setSelectedSOs(s => s.includes(so) ? s.filter(x => x !== so) : [...s, so])

  const result = mrpResult?.result || lastMRP?.result || {}
  const materials = Object.entries(result) as [string, any][]
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

      {/* MRP Results */}
      {materials.length > 0 && (
        <div className="bg-white rounded-xl border overflow-hidden">
          <div className="px-4 py-3 bg-[#002B5B] text-white flex justify-between items-center">
            <span className="font-semibold">Material requirements — {materials.length} materials</span>
            <span className="text-blue-200 text-xs">{mrpResult?.run_time || lastMRP?.run_time}</span>
          </div>
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
                        <div className="py-2 space-y-1">
                          <p className="text-xs font-semibold text-gray-500 uppercase mb-1">Breakdown — quantity allocation:</p>
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="text-gray-400">
                                <th className="text-left py-1 pr-4">SO Number</th>
                                <th className="text-left py-1 pr-4">SKU / FG</th>
                                <th className="text-right py-1">Qty Required</th>
                              </tr>
                            </thead>
                            <tbody>
                              {mat.breakdown.map((b: any, i: number) => (
                                <tr key={i} className="border-t border-blue-100">
                                  <td className="py-1 pr-4 font-semibold text-[#002B5B]">{b.so_no}</td>
                                  <td className="py-1 pr-4 font-mono text-gray-600">{b.sku}</td>
                                  <td className="py-1 text-right font-semibold">{b.qty_req} {mat.unit}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
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
  const [tab, setTab] = useState<'process' | 'tracker' | 'issue-notes' | 'reports' | 'mrp'>('process')
  const [expandedIssueNote, setExpandedIssueNote] = useState<number | null>(null)
  const [expanded, setExpanded] = useState<number | null>(null)
  const [filterStatus, setFilterStatus] = useState('')
  const [modal, setModal] = useState<ModalType>(null)
  const [activeJO, setActiveJO] = useState<JO | null>(null)
  const [activeLineId, setActiveLineId] = useState<number | null>(null)

  // New JO form
  const [newForm, setNewForm] = useState({
    so_number: '', sku: '', sku_name: '', process: 'Cutting',
    exec_type: 'Inhouse', vendor_name: '', vendor_rate: 0,
    planned_qty: 0, so_qty: 0, fabric_code: '', fabric_qty: 0,
    fabric_unit: 'MTR', expected_completion: '', remarks: '',
  })
  const [newLines, setNewLines] = useState<{ so_number: string; sku: string; sku_name: string; style: string; planned_qty: number; vendor_rate: number; remarks: string }[]>([])

  // Modal forms
  const [fabricIssueForm, setFabricIssueForm] = useState({ fabric_code: '', fabric_name: '', issued_qty: 0, unit: 'MTR', issued_by: '', remarks: '' })
  const [fabricReturnForm, setFabricReturnForm] = useState({ fabric_code: '', returned_qty: 0, unit: 'MTR', returned_by: '', remarks: '' })
  const [receiveForm, setReceiveForm] = useState({ received_qty: 0, rejected_qty: 0, received_by: '', remarks: '' })
  const [issuePiecesForm, setIssuePiecesForm] = useState({ issued_qty: 0, to_process: '', issued_by: '', remarks: '' })
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
  const { data: processJOs = [] } = useQuery<JO[]>({
    queryKey: ['jos-process', activeProcess, filterStatus],
    queryFn: () => api.get(`/production/orders?process=${encodeURIComponent(activeProcess)}${filterStatus ? `&status=${filterStatus}` : ''}`).then(r => r.data),
    enabled: tab === 'process',
  })
  const { data: allJOs = [] } = useQuery<JO[]>({
    queryKey: ['jos-all', filterStatus],
    queryFn: () => api.get(`/production/orders${filterStatus ? `?status=${filterStatus}` : ''}`).then(r => r.data),
    enabled: tab === 'tracker',
  })
  const { data: readyLines = [] } = useQuery({
    queryKey: ['ready-to-process', activeProcess],
    queryFn: () => api.get(`/production/ready-to-process/${encodeURIComponent(activeProcess)}`).then(r => r.data),
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
  })
  const { data: itemRouting } = useQuery({
    queryKey: ['item-routing', newForm.sku],
    queryFn: () => api.get(`/production/item-routing/${encodeURIComponent(newForm.sku)}`).then(r => r.data),
    enabled: !!newForm.sku,
  })
  const { data: processors = [] } = useQuery<{ processor_name?: string }[]>({
    queryKey: ['purchase-processors'],
    queryFn: () => api.get('/purchase/processors').then(r => r.data),
  })
  const vendorSuggestions = [...new Set(
    (processors || [])
      .map(p => String(p.processor_name || '').trim())
      .filter(Boolean),
  )].sort((a, b) => a.localeCompare(b))

  const { data: joValidation } = useQuery({
    queryKey: ['jo-validate', newForm.process, newForm.so_number, newForm.sku, newForm.planned_qty],
    queryFn: () => api.get(`/production/orders/validate?process=${newForm.process}&so_number=${newForm.so_number}&sku=${newForm.sku}&planned_qty=${newForm.planned_qty}`).then(r => r.data),
    enabled: !!newForm.so_number && !!newForm.sku && newForm.process !== 'Cutting',
  })

  const invalidateAll = () => {
    qc.invalidateQueries({ queryKey: ['prod-stats'] })
    qc.invalidateQueries({ queryKey: ['jos-process'] })
    qc.invalidateQueries({ queryKey: ['jos-all'] })
    qc.invalidateQueries({ queryKey: ['ready-to-process'] })
    qc.invalidateQueries({ queryKey: ['process-report'] })
    qc.invalidateQueries({ queryKey: ['prod-issue-notes'] })
    qc.invalidateQueries({ queryKey: ['jo-issue-note'] })
  }

  // ── Mutations ─────────────────────────────────────────────────────────────────
  const createJOMut = useMutation({
    mutationFn: (b: object) => api.post('/production/orders', b),
    onSuccess: (res) => {
      invalidateAll()
      qc.invalidateQueries({ queryKey: ['mrp-commit-map'] })
      setModal(null)
      setNewLines([])
      const inNum = res?.data?.issue_note?.in_number
      if (inNum) {
        setTab('issue-notes')
        alert(`Job order created. Material issue note ${inNum} generated from BOM.`)
      }
    },
    onError: (e: unknown) => alert(apiErrorMessage(e, 'Error creating JO')),
  })
  const updateJOMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/production/orders/${id}`, data),
    onSuccess: () => invalidateAll(),
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
    onSuccess: () => { invalidateAll(); setModal(null) },
    onError: (e: unknown) => alert(apiErrorMessage(e, 'Error')),
  })
  const issuePiecesMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.post(`/production/orders/${id}/issue-pieces`, data),
    onSuccess: () => { invalidateAll(); setModal(null) },
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

  const openModal = (type: ModalType, jo: JO, lineId?: number) => {
    setActiveJO(jo)
    setActiveLineId(lineId || null)
    setModal(type)
    if (type === 'issue-fabric') setFabricIssueForm(f => ({ ...f, fabric_code: jo.fabric_code || '', issued_qty: jo.fabric_qty || 0 }))
    if (type === 'return-fabric') setFabricReturnForm(f => ({ ...f, fabric_code: jo.fabric_code || '' }))
    if (type === 'receive') {
      const line = jo.lines.find(l => l.id === lineId)
      setReceiveForm(f => ({ ...f, received_qty: line ? line.planned_qty - line.received_qty : jo.planned_qty - jo.received_qty }))
    }
    if (type === 'issue-pieces') {
      const line = jo.lines.find(l => l.id === lineId)
      setIssuePiecesForm(f => ({ ...f, to_process: jo.next_process || '', issued_qty: line ? line.received_qty : jo.received_qty }))
    }
    if (type === 'add-cost') setCostForm({ cost_type: 'Labour', amount: 0, description: '' })
  }

  // ── Add SO lines to new JO ─────────────────────────────────────────────────
  const addSOLineToJO = (line: any) => {
    if (newLines.find(l => l.sku === line.sku)) return
    setNewLines(ls => [...ls, {
      so_number: newForm.so_number,
      sku: line.sku || '',
      sku_name: line.sku_name || line.item_name || '',
      style: '',
      planned_qty: line.qty || 0,
      vendor_rate: 0,
      remarks: '',
    }])
  }

  const allProcesses = processes.length > 0 ? processes : ['Cutting', 'Printing', 'Embroidery', 'Stitching', 'Finishing', 'Packing']

  const renderJOCard = (jo: JO) => {
    const isExpanded = expanded === jo.id
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
            <p className="text-sm text-gray-600">SO: <b>{jo.so_number || '—'}</b> · SKU: <b>{jo.sku}</b> {jo.sku_name ? `— ${jo.sku_name}` : ''}</p>
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

            {/* Lines table */}
            {jo.lines.length > 0 && (
              <div className="bg-white rounded-lg border overflow-hidden">
                <div className="px-3 py-2 bg-gray-50 text-xs font-semibold text-gray-600 flex justify-between">
                  <span>Lines — Issue / Receive per SKU</span>
                  <span className="text-gray-400">{jo.lines.length} lines</span>
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
                        <td className="px-3 py-2 text-right">{fmt(line.planned_qty)}</td>
                        <td className="px-3 py-2 text-right text-green-600 font-semibold">{fmt(line.received_qty)}</td>
                        <td className="px-3 py-2 text-right text-red-500">{fmt(line.rejected_qty)}</td>
                        <td className={`px-3 py-2 text-right font-semibold ${line.balance_qty > 0 ? 'text-amber-600' : 'text-green-600'}`}>{fmt(line.balance_qty)}</td>
                        <td className="px-3 py-2 text-right">{fmtR(line.vendor_rate)}</td>
                        <td className="px-3 py-2 text-right font-semibold">{fmtR(line.planned_qty * line.vendor_rate)}</td>
                        <td className="px-3 py-2 text-center">
                          <div className="flex gap-1 justify-center">
                            <button onClick={() => openModal('receive', jo, line.id)}
                              className="px-2 py-0.5 text-xs bg-green-600 text-white rounded hover:bg-green-700">✅ Rec</button>
                            {jo.next_process && (
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
              {jo.next_process && (
                <button onClick={() => openModal('issue-pieces', jo)} className="px-3 py-1.5 text-xs bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700">
                  ➡️ Issue to {jo.next_process}
                </button>
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
        {([['process','⚙️ Process'], ['tracker','📋 All JOs'], ['issue-notes','📋 Issue Notes'], ['reports','📊 Reports'], ['mrp','📐 Material Req. Planning']] as const).map(([key, label]) => (
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
              <button key={p} onClick={() => { setActiveProcess(p); setExpanded(null) }}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${activeProcess === p ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
                {PROCESS_ICONS[p] || ''} {p}
              </button>
            ))}
          </div>

          <div className="flex items-center justify-between flex-wrap gap-2">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All Statuses</option>
              {['Created','In Progress','Completed','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={() => { setNewForm(f => ({ ...f, process: activeProcess })); setModal('new-jo') }}
              className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
              + New {activeProcess} JO
            </button>
          </div>

          {/* Ready to process */}
          {readyLines.length > 0 && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
              <p className="text-sm font-semibold text-amber-800 mb-2">
                ⚡ Ready for {activeProcess} — {readyLines.length} line(s)
              </p>
              <div className="space-y-2">
                {(readyLines as any[]).map((r: any, i: number) => (
                  <div key={i} className="bg-white rounded-lg border border-amber-200 px-3 py-2 flex items-center justify-between">
                    <div className="text-xs">
                      <span className="font-semibold text-[#002B5B]">SO: {r.so_number}</span>
                      <span className="mx-2 text-gray-400">·</span>
                      <span className="font-mono">{r.sku}</span>
                      {r.fabric_code && <span className="mx-2 text-gray-400">· Fabric: <b>{r.fabric_code}</b></span>}
                      <span className="mx-2 text-gray-400">·</span>
                      <span className="text-green-700 font-semibold">
                        {r.available_qty || r.reserved_qty} pcs available
                      </span>
                      {r.already_planned > 0 && (
                        <span className="ml-2 text-gray-400 italic">(Total: {r.reserved_qty}, JO mein: {r.already_planned})</span>
                      )}
                    </div>
                    <button onClick={() => {
                      setNewForm(f => ({
                        ...f, so_number: r.so_number, sku: r.sku || '', fabric_code: r.fabric_code || '',
                        fabric_qty: r.reserved_qty || 0, process: activeProcess,
                        planned_qty: r.available_qty || r.reserved_qty || 0,
                      }))
                      setModal('new-jo')
                    }} className="text-xs px-2 py-1 bg-[#002B5B] text-white rounded hover:bg-blue-800 shrink-0">
                      Create JO →
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* JO list */}
          <div className="space-y-3">
            {processJOs.map(renderJOCard)}
            {processJOs.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No {activeProcess} job orders.</p>}
          </div>
        </div>
      )}

      {/* TRACKER TAB */}
      {tab === 'tracker' && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All Statuses</option>
              {['Created','In Progress','Completed','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
            </select>
            <p className="text-sm text-gray-500">{allJOs.length} job orders</p>
          </div>
          {allJOs.map(renderJOCard)}
          {allJOs.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No job orders found.</p>}
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

            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {/* SO */}
              <div><label className="text-xs text-gray-500">SO Number *</label>
                <select value={newForm.so_number} onChange={e => setNewForm(f => ({ ...f, so_number: e.target.value, sku: '', sku_name: '' }))}
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                  <option value="">Select SO</option>
                  {(soList as any[]).map((s: any) => <option key={s.so_number} value={s.so_number}>{s.so_number} — {s.buyer || ''}</option>)}
                </select>
              </div>
              {/* Process */}
              <div><label className="text-xs text-gray-500">Process *</label>
                <select value={newForm.process} onChange={e => setNewForm(f => ({ ...f, process: e.target.value }))}
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                  {allProcesses.map(p => <option key={p}>{p}</option>)}
                </select>
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
              {[['expected_completion','Expected Date'],['remarks','Remarks']].map(([k,l]) => (
                <div key={k}><label className="text-xs text-gray-500">{l}</label>
                  <input value={(newForm as any)[k]} onChange={e => setNewForm(f => ({ ...f, [k]: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
              ))}
              {newForm.process === 'Cutting' && (
                <>
                  <div><label className="text-xs text-gray-500">Fabric Code</label>
                    <input value={newForm.fabric_code} onChange={e => setNewForm(f => ({ ...f, fabric_code: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1 font-mono" /></div>
                  <div><label className="text-xs text-gray-500">Fabric Qty (MTR)</label>
                    <input type="number" value={newForm.fabric_qty} onChange={e => setNewForm(f => ({ ...f, fabric_qty: +e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                </>
              )}
            </div>

            {/* Validation message */}
            {newForm.process !== 'Cutting' && newForm.so_number && newForm.sku && joValidation && (
              <div className={`rounded-lg px-3 py-2 text-xs ${joValidation.ok ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
                {joValidation.ok ? `✅ ${joValidation.available} pieces available` : `❌ ${joValidation.message}`}
              </div>
            )}

            {/* SO Lines — add to JO */}
            {newForm.so_number && soLines.length > 0 && (
              <div className="border rounded-xl overflow-hidden">
                <div className="px-3 py-2 bg-blue-50 text-xs font-semibold text-blue-700 flex justify-between">
                  <span>SO Lines — Click to add to JO</span>
                  <button onClick={() => (soLines as any[]).forEach(addSOLineToJO)} className="text-blue-600 hover:underline">Add all</button>
                </div>
                <table className="w-full text-xs">
                  <thead><tr className="text-gray-400 border-b bg-gray-50">
                    <th className="text-left px-3 py-1.5">SKU</th><th className="text-left px-3 py-1.5">Name</th>
                    <th className="text-right px-3 py-1.5">SO Qty</th><th className="px-3 py-1.5"></th>
                  </tr></thead>
                  <tbody>
                    {(soLines as any[]).map((l: any) => {
                      const added = newLines.some(nl => nl.sku === l.sku)
                      return (
                        <tr key={l.sku} className="border-t hover:bg-gray-50">
                          <td className="px-3 py-1.5 font-mono font-semibold">{l.sku}</td>
                          <td className="px-3 py-1.5 text-gray-600">{l.sku_name || l.item_name || '—'}</td>
                          <td className="px-3 py-1.5 text-right">{l.qty}</td>
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
            )}

            {/* JO Lines */}
            {newLines.length > 0 && (
              <div className="border rounded-xl overflow-hidden">
                <div className="px-3 py-2 bg-gray-50 text-xs font-semibold text-gray-600">JO Lines ({newLines.length})</div>
                <table className="w-full text-xs">
                  <thead><tr className="text-gray-400 border-b">
                    <th className="text-left px-3 py-1.5">SKU</th><th className="text-left px-3 py-1.5">Style</th>
                    <th className="text-right px-3 py-1.5">Planned Qty</th>
                    <th className="text-right px-3 py-1.5">Rate (₹)</th>
                    <th className="text-right px-3 py-1.5">Amount</th>
                    <th className="px-3 py-1.5"></th>
                  </tr></thead>
                  <tbody>
                    {newLines.map((ln, i) => (
                      <tr key={i} className="border-t">
                        <td className="px-3 py-1 font-mono font-semibold text-[#002B5B]">{ln.sku}</td>
                        <td className="px-3 py-1"><input value={ln.style} onChange={e => setNewLines(ls => ls.map((x,j) => j===i ? {...x, style: e.target.value} : x))}
                          placeholder="Style/desc" className="border rounded px-1.5 py-0.5 text-xs w-full" /></td>
                        <td className="px-3 py-1"><input type="number" value={ln.planned_qty} onChange={e => setNewLines(ls => ls.map((x,j) => j===i ? {...x, planned_qty: +e.target.value} : x))}
                          className="border rounded px-1.5 py-0.5 text-xs w-20 text-right" /></td>
                        <td className="px-3 py-1"><input type="number" value={ln.vendor_rate} onChange={e => setNewLines(ls => ls.map((x,j) => j===i ? {...x, vendor_rate: +e.target.value} : x))}
                          className="border rounded px-1.5 py-0.5 text-xs w-20 text-right" /></td>
                        <td className="px-3 py-1 text-right font-semibold">{fmtR(ln.planned_qty * ln.vendor_rate)}</td>
                        <td className="px-3 py-1"><button onClick={() => setNewLines(ls => ls.filter((_,j) => j!==i))} className="text-red-400 hover:text-red-600">✕</button></td>
                      </tr>
                    ))}
                    <tr className="border-t bg-gray-50 font-semibold">
                      <td colSpan={4} className="px-3 py-1.5 text-right text-xs text-gray-600">Total:</td>
                      <td className="px-3 py-1.5 text-right text-xs">{fmtR(newLines.reduce((s,l) => s + l.planned_qty * l.vendor_rate, 0))}</td>
                      <td></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}

            <div className="flex gap-2 pt-2">
              <button onClick={() => createJOMut.mutate({
                ...newForm,
                planned_qty: newLines.reduce((s,l) => s+l.planned_qty, 0) || newForm.planned_qty,
                lines: newLines,
              })}
                disabled={
                  createJOMut.isPending
                  || !newForm.so_number
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
              Process: <b>{activeJO.process}</b> · Planned: <b>{activeJO.planned_qty} pcs</b> · Received so far: <b>{activeJO.received_qty} pcs</b>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div><label className="text-xs text-gray-500">Received Qty (pcs) *</label>
                <input type="number" value={receiveForm.received_qty} onChange={e => setReceiveForm(f => ({ ...f, received_qty: +e.target.value }))}
                  className="w-full border border-green-200 rounded px-2 py-1.5 text-sm mt-1 bg-green-50 font-semibold" /></div>
              <div><label className="text-xs text-gray-500">Rejected Qty</label>
                <input type="number" value={receiveForm.rejected_qty} onChange={e => setReceiveForm(f => ({ ...f, rejected_qty: +e.target.value }))}
                  className="w-full border border-red-200 rounded px-2 py-1.5 text-sm mt-1 bg-red-50" /></div>
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
                    jo_line_id: activeLineId ?? undefined,
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
              From: <b>{activeJO.process}</b> → To: <b>{activeJO.next_process || 'Next'}</b>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div><label className="text-xs text-gray-500">To Process</label>
                <select value={issuePiecesForm.to_process} onChange={e => setIssuePiecesForm(f => ({ ...f, to_process: e.target.value }))}
                  className="w-full border rounded px-2 py-1.5 text-sm mt-1">
                  <option value="">Select</option>
                  {allProcesses.filter(p => p !== activeJO.process).map(p => <option key={p}>{p}</option>)}
                </select>
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
              <button onClick={() => issuePiecesMut.mutate({ id: activeJO.id, data: { ...issuePiecesForm, from_process: activeJO.process, sku: activeJO.sku, jo_line_id: activeLineId } })}
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
