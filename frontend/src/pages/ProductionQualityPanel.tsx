import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'
import { brandLogoDataUrl, brandPrintHeaderHtml, BRAND_PRINT_CSS, YASH_GALLERY } from '../lib/printBrand'

const PROCESSES = [
  'Cutting', 'Printing', 'Embroidery', 'Stitching', 'Kajh Button', 'Handwork', 'Finishing', 'Packing',
]
const fmt = (n: number | null | undefined) => Number(n || 0).toLocaleString()
const fmtMoney = (n: number | null | undefined) =>
  Number(n || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })

type DefectForm = {
  defect_source_process: string
  action: string
  qty: number
  reason: string
  responsible_vendor: string
  rework_by_type: string
  rework_by_vendor: string
  debit_required: boolean
  with_fabric: boolean
}

const emptyDefect = (): DefectForm => ({
  defect_source_process: 'Stitching',
  action: 'Rework',
  qty: 0,
  reason: '',
  responsible_vendor: '',
  rework_by_type: 'SameVendor',
  rework_by_vendor: '',
  debit_required: false,
  with_fabric: false,
})

function printHtml(title: string, body: string) {
  const win = window.open('', '_blank', 'width=900,height=700')
  if (!win) {
    alert('Please allow popups to print.')
    return
  }
  win.document.write(`<!DOCTYPE html><html><head><title>${title}</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:'Segoe UI',sans-serif;font-size:12px;color:#1a1a1a;padding:24px}
    ${BRAND_PRINT_CSS}
    .info-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
    .info-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px}
    .info-label{font-size:10px;text-transform:uppercase;color:#64748b;font-weight:600;margin-bottom:4px}
    .info-value{font-size:13px;font-weight:600;color:#1e293b}
    table{width:100%;border-collapse:collapse;margin-bottom:16px}
    th{background:#002B5B;color:white;padding:7px 10px;text-align:left;font-size:11px}
    th.r,td.r{text-align:right}
    td{padding:6px 10px;border-bottom:1px solid #e2e8f0}
    .banner{background:#7f1d1d;color:#fff;text-align:center;font-weight:800;letter-spacing:.08em;padding:8px;margin-bottom:14px;border-radius:4px}
    .footer{margin-top:32px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px;border-top:1px solid #e2e8f0;padding-top:16px}
    .sign-line{border-top:1px solid #64748b;margin-top:32px;padding-top:6px;font-size:10px;color:#64748b;text-align:center}
    @media print{body{padding:12px}}
  </style></head><body>${body}
  <script>window.onload=()=>window.print()<\/script>
  </body></html>`)
  win.document.close()
}

async function printReworkJO(r: any) {
  const logoSrc = await brandLogoDataUrl()
  const joRef = r.original_jo_number || (r.original_jo_id ? `JO#${r.original_jo_id}` : '—')
  const html = `
    <div class="banner">REWORK JOB ORDER</div>
    ${brandPrintHeaderHtml({
      logoSrc,
      department: 'Production / QC',
      docTitle: 'REWORK JOB ORDER',
      docNumber: r.rework_no,
    })}
    <div class="info-grid">
      <div class="info-box">
        <div class="info-label">Original JO</div><div class="info-value">${joRef}</div>
        <div class="info-label" style="margin-top:8px">SO</div><div class="info-value">${r.so_number || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Style / SKU</div><div class="info-value">${r.sku || '—'}</div>
        <div class="info-label" style="margin-top:8px">Component</div><div class="info-value">${r.component_code || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Process</div><div class="info-value">${r.process || '—'}</div>
        <div class="info-label" style="margin-top:8px">Found At</div><div class="info-value">${r.found_at_process || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Rework Vendor / Dept</div>
        <div class="info-value">${r.rework_by_type || '—'}${r.rework_by_vendor ? ` · ${r.rework_by_vendor}` : ''}</div>
        <div class="info-label" style="margin-top:8px">Responsible Vendor</div>
        <div class="info-value">${r.responsible_vendor || '—'}</div>
      </div>
    </div>
    <table>
      <thead><tr>
        <th>Rework Qty</th><th class="r">Received</th><th class="r">Pass</th><th class="r">Reject</th><th class="r">Balance</th><th>Status</th>
      </tr></thead>
      <tbody><tr>
        <td><strong>${fmt(r.planned_qty)}</strong></td>
        <td class="r">${fmt(r.received_qty)}</td>
        <td class="r">${fmt(r.pass_qty)}</td>
        <td class="r">${fmt(r.reject_qty)}</td>
        <td class="r">${fmt(r.balance_qty)}</td>
        <td>${r.status || '—'}</td>
      </tr></tbody>
    </table>
    <div class="info-box" style="margin-bottom:16px">
      <div class="info-label">Defect / Rework Reason</div>
      <div class="info-value">${r.defect_reason || r.notes || '—'}</div>
      <div class="info-label" style="margin-top:8px">Defect Source Process</div>
      <div class="info-value">${r.defect_source_process || '—'}</div>
    </div>
    <div class="footer">
      <div><div class="sign-line">Prepared By (QC)</div></div>
      <div><div class="sign-line">Authorized By</div></div>
      <div><div class="sign-line">Vendor / Dept Acknowledgement</div></div>
    </div>
    <p style="margin-top:16px;font-size:10px;color:#64748b;text-align:center">${YASH_GALLERY.name} — Rework material/pieces issue document</p>`
  printHtml(`Rework JO - ${r.rework_no}`, html)
}

async function printDebitNote(d: any) {
  const logoSrc = await brandLogoDataUrl()
  const joRef = d.original_jo_number || (d.original_jo_id ? `JO#${d.original_jo_id}` : '—')
  let fabricLines: any[] = d.fabric_lines || []
  if (!fabricLines.length && d.fabric_lines_json) {
    try { fabricLines = JSON.parse(d.fabric_lines_json) } catch { fabricLines = [] }
  }
  const fabricRows = fabricLines.length
    ? fabricLines.map((l: any, i: number) => `<tr>
        <td>${i + 1}</td><td>${l.material_code || '—'}</td><td>${l.material_name || '—'}</td>
        <td class="r">${fmt(l.qty)}</td><td>${l.unit || ''}</td>
        <td class="r">${fmtMoney(l.rate)}</td><td class="r">${fmtMoney(l.amount)}</td>
      </tr>`).join('')
    : '<tr><td colspan="7" style="text-align:center;color:#94a3b8">No fabric recovery lines</td></tr>'
  const html = `
    ${brandPrintHeaderHtml({
      logoSrc,
      department: 'Accounts / QC',
      docTitle: 'DEBIT NOTE',
      docNumber: d.debit_no,
    })}
    <div class="info-grid">
      <div class="info-box">
        <div class="info-label">Vendor / Party</div><div class="info-value">${d.responsible_vendor || '—'}</div>
        <div class="info-label" style="margin-top:8px">Debit Date</div><div class="info-value">${d.debit_date || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Original JO</div><div class="info-value">${joRef}</div>
        <div class="info-label" style="margin-top:8px">SO / SKU</div><div class="info-value">${d.so_number || '—'} · ${d.sku || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Process</div><div class="info-value">${d.process || '—'}</div>
        <div class="info-label" style="margin-top:8px">Component</div><div class="info-value">${d.component_code || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Qty</div><div class="info-value">${fmt(d.qty)}</div>
        <div class="info-label" style="margin-top:8px">With Fabric</div><div class="info-value">${Number(d.with_fabric) ? 'Yes' : 'No'}</div>
      </div>
    </div>
    <div class="info-box" style="margin-bottom:16px">
      <div class="info-label">Defect / Reason</div>
      <div class="info-value">${d.defect_reason || d.notes || '—'}</div>
    </div>
    <table>
      <thead><tr><th>Description</th><th class="r">Qty</th><th class="r">Rate (₹)</th><th class="r">Amount (₹)</th></tr></thead>
      <tbody>
        <tr>
          <td>Workmanship debit</td>
          <td class="r">${fmt(d.qty)}</td>
          <td class="r">${fmtMoney(d.workmanship_rate)}</td>
          <td class="r"><strong>${fmtMoney(d.workmanship_amount)}</strong></td>
        </tr>
        <tr>
          <td>Fabric recovery</td>
          <td class="r">—</td><td class="r">—</td>
          <td class="r"><strong>${fmtMoney(d.fabric_amount)}</strong></td>
        </tr>
        <tr>
          <td colspan="3" style="text-align:right;font-weight:700">Total Debit</td>
          <td class="r" style="font-weight:800;background:#002B5B;color:#fff">${fmtMoney(d.total_amount)}</td>
        </tr>
      </tbody>
    </table>
    <h4 style="margin:12px 0 8px;color:#002B5B;font-size:12px">Fabric / Material Details</h4>
    <table>
      <thead><tr><th>#</th><th>Code</th><th>Name</th><th class="r">Qty</th><th>Unit</th><th class="r">Rate</th><th class="r">Amount</th></tr></thead>
      <tbody>${fabricRows}</tbody>
    </table>
    <div class="footer">
      <div><div class="sign-line">Prepared By</div></div>
      <div><div class="sign-line">Accounts</div></div>
      <div><div class="sign-line">Authorized By</div></div>
    </div>`
  printHtml(`Debit Note - ${d.debit_no}`, html)
}

/**
 * Production QC + Rework WIP + Debit Notes + Billing eligibility.
 */
export default function ProductionQualityPanel() {
  const qc = useQueryClient()
  const [sub, setSub] = useState<'qc' | 'rework' | 'debit' | 'billing'>('qc')
  const [joLookup, setJoLookup] = useState('')
  const [billingJoId, setBillingJoId] = useState<number | ''>('')
  const [loadedJo, setLoadedJo] = useState<{
    id: number
    jo_number?: string
    process?: string
    planned_qty?: number
    received_qty?: number
    vendor_name?: string
    exec_type?: string
  } | null>(null)

  const [qcForm, setQcForm] = useState({
    found_at_process: 'Finishing',
    original_jo_id: '' as number | '',
    so_number: '',
    sku: '',
    component_code: '',
    checked_qty: 0,
    pass_qty: 0,
    notes: '',
    qc_date: new Date().toISOString().slice(0, 10),
  })
  const [defects, setDefects] = useState<DefectForm[]>([emptyDefect()])

  const [reworkReceive, setReworkReceive] = useState({
    rework_id: '' as number | '',
    received_qty: 0,
    pass_qty: 0,
    reject_qty: 0,
    remarks: '',
  })

  const { data: qcReports = [], isFetching: qcLoading } = useQuery({
    queryKey: ['prod-qc-reports'],
    queryFn: () => api.get('/production/qc-reports', { params: { limit: 100 } }).then(r => r.data),
  })
  const { data: reworks = [], isFetching: rwLoading } = useQuery({
    queryKey: ['prod-rework-orders'],
    queryFn: () => api.get('/production/rework-orders', { params: { limit: 100 } }).then(r => r.data),
  })
  const { data: debits = [] } = useQuery({
    queryKey: ['prod-debit-notes'],
    queryFn: () => api.get('/production/debit-notes', { params: { limit: 100 } }).then(r => r.data),
  })
  const { data: wip = [] } = useQuery({
    queryKey: ['prod-rework-wip'],
    queryFn: () => api.get('/production/rework-wip').then(r => r.data),
  })
  const { data: billing, isFetching: billLoading } = useQuery({
    queryKey: ['prod-billing-elig', billingJoId],
    queryFn: () => api.get(`/production/orders/${billingJoId}/billing-eligibility`).then(r => r.data),
    enabled: !!billingJoId,
  })

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['prod-qc-reports'] })
    qc.invalidateQueries({ queryKey: ['prod-rework-orders'] })
    qc.invalidateQueries({ queryKey: ['prod-debit-notes'] })
    qc.invalidateQueries({ queryKey: ['prod-rework-wip'] })
    qc.invalidateQueries({ queryKey: ['prod-billing-elig'] })
  }

  const createQcMut = useMutation({
    mutationFn: (body: object) => api.post('/production/qc-reports', body),
    onSuccess: (res) => {
      invalidate()
      const d = res.data
      alert(
        `QC ${d.report_no} saved.`
        + (d.created_reworks?.length ? `\nRework orders: ${d.created_reworks.map((r: any) => r.rework_no).join(', ')}` : '')
        + (d.created_debits?.length ? `\nDebit notes: ${d.created_debits.map((r: any) => r.debit_no).join(', ')}` : ''),
      )
      setDefects([emptyDefect()])
    },
    onError: (err: any) => alert(err?.response?.data?.detail || 'QC save failed'),
  })

  const receiveRwMut = useMutation({
    mutationFn: ({ id, body }: { id: number; body: object }) =>
      api.post(`/production/rework-orders/${id}/receive`, body),
    onSuccess: () => {
      invalidate()
      alert('Rework receive saved')
    },
    onError: (err: any) => alert(err?.response?.data?.detail || 'Rework receive failed'),
  })

  const loadJo = async () => {
    const q = joLookup.trim()
    if (!q) return
    try {
      const list = await api.get('/production/orders', { params: { q, limit: 5, light: 1 } }).then(r => r.data)
      const rows = Array.isArray(list) ? list : (list?.items || list?.rows || [])
      const jo = rows[0]
      if (!jo) {
        alert('JO not found')
        return
      }
      setLoadedJo({
        id: jo.id,
        jo_number: jo.jo_number,
        process: jo.process,
        planned_qty: Number(jo.planned_qty) || 0,
        received_qty: Number(jo.received_qty) || 0,
        vendor_name: jo.vendor_name,
        exec_type: jo.exec_type,
      })
      setQcForm(f => ({
        ...f,
        original_jo_id: jo.id,
        so_number: jo.so_number || '',
        sku: jo.sku || '',
        component_code: jo.component_code || '',
        found_at_process: jo.process || f.found_at_process,
      }))
      setBillingJoId(jo.id)
      if (jo.vendor_name) {
        setDefects(ds => ds.map((d, i) => (i === 0 ? { ...d, responsible_vendor: jo.vendor_name, rework_by_vendor: jo.vendor_name } : d)))
      }
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Could not load JO')
    }
  }

  const defectQtySum = useMemo(() => defects.reduce((s, d) => s + (Number(d.qty) || 0), 0), [defects])

  const submitQc = () => {
    const checked = Number(qcForm.checked_qty) || 0
    if (!qcForm.found_at_process || checked <= 0) {
      alert('Found At process and Checked qty are required')
      return
    }
    const cleanDefects = defects.filter(d => Number(d.qty) > 0 && d.defect_source_process)
    const rework_qty = cleanDefects
      .filter(d => d.action === 'Rework' || d.action === 'Alteration')
      .reduce((s, d) => s + Number(d.qty), 0)
    const reject_qty = cleanDefects
      .filter(d => d.action === 'FinalReject')
      .reduce((s, d) => s + Number(d.qty), 0)
    const pass_qty = Math.max(0, checked - rework_qty - reject_qty)
    createQcMut.mutate({
      found_at_process: qcForm.found_at_process,
      original_jo_id: qcForm.original_jo_id || null,
      so_number: qcForm.so_number,
      sku: qcForm.sku,
      component_code: qcForm.component_code,
      checked_qty: checked,
      pass_qty,
      rework_qty,
      reject_qty,
      qc_date: qcForm.qc_date,
      notes: qcForm.notes,
      defects: cleanDefects.map(d => ({
        defect_source_process: d.defect_source_process,
        action: d.action,
        qty: Number(d.qty),
        reason: d.reason,
        responsible_vendor: d.responsible_vendor,
        rework_by_type: d.rework_by_type,
        rework_by_vendor: d.rework_by_vendor,
        debit_required: d.debit_required || d.rework_by_type === 'Inhouse' || d.action === 'FinalReject' ? 1 : 0,
        with_fabric: d.with_fabric ? 1 : 0,
      })),
    })
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="font-semibold text-gray-800">QC / Rework / Debit Notes</h3>
        <p className="text-[11px] text-gray-500">
          Found At ≠ Defect Source ≠ Responsible ≠ Rework By. Receive qty stays separate from billable qty.
        </p>
      </div>

      <div className="flex gap-2 flex-wrap">
        {([
          ['qc', 'A · QC + Defects'],
          ['rework', 'B · Rework WIP'],
          ['debit', 'C · Debit Notes'],
          ['billing', 'Billing eligibility'],
        ] as const).map(([k, label]) => (
          <button
            key={k}
            type="button"
            onClick={() => setSub(k)}
            className={`px-3 py-1.5 text-xs rounded-lg font-medium ${sub === k ? 'bg-[#002B5B] text-white' : 'bg-white border text-gray-600'}`}
          >
            {label}
          </button>
        ))}
      </div>

      {sub === 'qc' && (
        <div className="space-y-4">
          <div className="bg-white border rounded-xl p-4 space-y-3">
            <div className="flex flex-wrap gap-2 items-end">
              <label className="text-xs">
                <span className="text-gray-500">Lookup JO (number / SO / SKU)</span>
                <input value={joLookup} onChange={e => setJoLookup(e.target.value)} className="mt-0.5 block border rounded px-2 py-1 text-sm min-w-[14rem]" />
              </label>
              <button type="button" onClick={() => void loadJo()} className="px-3 py-1.5 text-xs bg-slate-700 text-white rounded-lg">Load JO</button>
              {qcForm.original_jo_id ? <span className="text-xs text-emerald-700">JO id #{qcForm.original_jo_id}</span> : null}
            </div>
            {loadedJo && (
              <div className="rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs text-sky-900 flex flex-wrap gap-x-4 gap-y-1">
                <span><b>{loadedJo.jo_number || `JO#${loadedJo.id}`}</b></span>
                <span>Process: <b>{loadedJo.process || '—'}</b></span>
                <span>Planned Qty = <b>{fmt(loadedJo.planned_qty)}</b></span>
                <span>Received = <b>{fmt(loadedJo.received_qty)}</b></span>
                {loadedJo.vendor_name ? <span>Vendor: <b>{loadedJo.vendor_name}</b></span> : null}
              </div>
            )}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <label className="block">
                <span className="text-gray-500">Found At *</span>
                <select value={qcForm.found_at_process} onChange={e => setQcForm(f => ({ ...f, found_at_process: e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1">
                  {PROCESSES.map(p => <option key={p}>{p}</option>)}
                </select>
              </label>
              <label className="block">
                <span className="text-gray-500">QC date</span>
                <input type="date" value={qcForm.qc_date} onChange={e => setQcForm(f => ({ ...f, qc_date: e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1" />
              </label>
              <label className="block">
                <span className="text-gray-500">SO</span>
                <input value={qcForm.so_number} onChange={e => setQcForm(f => ({ ...f, so_number: e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1" />
              </label>
              <label className="block">
                <span className="text-gray-500">SKU</span>
                <input value={qcForm.sku} onChange={e => setQcForm(f => ({ ...f, sku: e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1" />
              </label>
              <label className="block">
                <span className="text-gray-500">
                  Checked qty *{loadedJo ? ` · Planned ${fmt(loadedJo.planned_qty)}` : ''}
                </span>
                <input type="number" value={qcForm.checked_qty || ''} onChange={e => setQcForm(f => ({ ...f, checked_qty: +e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1" placeholder={loadedJo ? `of ${loadedJo.planned_qty}` : ''} />
              </label>
              <label className="block col-span-2">
                <span className="text-gray-500">Notes</span>
                <input value={qcForm.notes} onChange={e => setQcForm(f => ({ ...f, notes: e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1" />
              </label>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold text-gray-700">Defects (qty sum {fmt(defectQtySum)})</p>
                <button type="button" onClick={() => setDefects(d => [...d, emptyDefect()])} className="text-xs text-blue-600">+ Add defect</button>
              </div>
              {defects.map((d, idx) => (
                <div key={idx} className="grid grid-cols-2 md:grid-cols-4 gap-2 border rounded-lg p-2 bg-slate-50 text-[11px]">
                  <label>Source process
                    <select value={d.defect_source_process} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, defect_source_process: e.target.value } : x))} className="mt-0.5 w-full border rounded px-1 py-1">
                      {PROCESSES.map(p => <option key={p}>{p}</option>)}
                    </select>
                  </label>
                  <label>Action
                    <select value={d.action} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, action: e.target.value, debit_required: e.target.value === 'FinalReject' || x.rework_by_type === 'Inhouse' } : x))} className="mt-0.5 w-full border rounded px-1 py-1">
                      <option>Rework</option><option>Alteration</option><option>FinalReject</option>
                    </select>
                  </label>
                  <label>Qty
                    <input type="number" value={d.qty || ''} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, qty: +e.target.value } : x))} className="mt-0.5 w-full border rounded px-1 py-1" />
                  </label>
                  <label>Reason
                    <input value={d.reason} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, reason: e.target.value } : x))} className="mt-0.5 w-full border rounded px-1 py-1" />
                  </label>
                  <label>Responsible vendor
                    <input value={d.responsible_vendor} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, responsible_vendor: e.target.value } : x))} className="mt-0.5 w-full border rounded px-1 py-1" />
                  </label>
                  <label>Rework by
                    <select value={d.rework_by_type} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, rework_by_type: e.target.value, debit_required: e.target.value === 'Inhouse' || x.action === 'FinalReject' } : x))} className="mt-0.5 w-full border rounded px-1 py-1">
                      <option value="SameVendor">SameVendor</option>
                      <option value="OtherVendor">OtherVendor</option>
                      <option value="Inhouse">Inhouse</option>
                    </select>
                  </label>
                  <label>Rework vendor
                    <input value={d.rework_by_vendor} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, rework_by_vendor: e.target.value } : x))} className="mt-0.5 w-full border rounded px-1 py-1" />
                  </label>
                  <div className="flex flex-col gap-1 justify-end pb-1">
                    <label className="flex items-center gap-1"><input type="checkbox" checked={d.debit_required} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, debit_required: e.target.checked } : x))} /> Debit note</label>
                    <label className="flex items-center gap-1"><input type="checkbox" checked={d.with_fabric} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, with_fabric: e.target.checked } : x))} /> With fabric</label>
                    {defects.length > 1 && (
                      <button type="button" className="text-rose-600 text-left" onClick={() => setDefects(arr => arr.filter((_, i) => i !== idx))}>Remove</button>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <button
              type="button"
              disabled={createQcMut.isPending}
              onClick={submitQc}
              className="px-4 py-2 text-sm bg-[#002B5B] text-white rounded-lg font-medium disabled:opacity-50"
            >
              {createQcMut.isPending ? 'Saving…' : 'Save QC report'}
            </button>
          </div>

          <div className="bg-white border rounded-xl overflow-auto">
            <div className="px-3 py-2 text-xs text-gray-500">{qcLoading ? 'Loading…' : `${qcReports.length} recent QC reports`}</div>
            <table className="w-full text-[11px]">
              <thead className="bg-gray-50 text-gray-500 uppercase">
                <tr>
                  {['Report', 'Date', 'Found At', 'SO', 'SKU', 'Checked', 'Pass', 'Rework', 'Reject'].map(h => (
                    <th key={h} className="text-left px-2 py-2">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {qcReports.map((r: any) => (
                  <tr key={r.id} className="border-t">
                    <td className="px-2 py-1.5 font-mono">{r.report_no}</td>
                    <td className="px-2 py-1.5">{r.qc_date}</td>
                    <td className="px-2 py-1.5">{r.found_at_process}</td>
                    <td className="px-2 py-1.5">{r.so_number}</td>
                    <td className="px-2 py-1.5 font-mono">{r.sku}</td>
                    <td className="px-2 py-1.5 text-right">{fmt(r.checked_qty)}</td>
                    <td className="px-2 py-1.5 text-right text-emerald-700">{fmt(r.pass_qty)}</td>
                    <td className="px-2 py-1.5 text-right text-amber-700">{fmt(r.rework_qty)}</td>
                    <td className="px-2 py-1.5 text-right text-rose-700">{fmt(r.reject_qty)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {sub === 'rework' && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {(wip as any[]).map((w: any, i: number) => (
              <div key={i} className="bg-white border rounded-xl p-3 text-xs">
                <p className="text-gray-500">{w.process} · {w.sku || '—'}</p>
                <p className="font-mono text-[10px] text-gray-400">{w.so_number}</p>
                <p className="text-amber-700 font-bold mt-1">Pending {fmt(w.rework_pending)}</p>
              </div>
            ))}
            {(wip as any[]).length === 0 && <p className="text-xs text-gray-400 col-span-4">No open rework WIP.</p>}
          </div>

          <div className="bg-white border rounded-xl p-4 grid grid-cols-2 md:grid-cols-5 gap-2 text-xs items-end">
            <label>Rework id
              <input type="number" value={reworkReceive.rework_id || ''} onChange={e => setReworkReceive(f => ({ ...f, rework_id: e.target.value ? +e.target.value : '' }))} className="mt-0.5 w-full border rounded px-2 py-1" />
            </label>
            <label>Received
              <input type="number" value={reworkReceive.received_qty || ''} onChange={e => setReworkReceive(f => ({ ...f, received_qty: +e.target.value, pass_qty: +e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1" />
            </label>
            <label>Pass
              <input type="number" value={reworkReceive.pass_qty || ''} onChange={e => setReworkReceive(f => ({ ...f, pass_qty: +e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1" />
            </label>
            <label>Reject
              <input type="number" value={reworkReceive.reject_qty || ''} onChange={e => setReworkReceive(f => ({ ...f, reject_qty: +e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1" />
            </label>
            <button
              type="button"
              disabled={!reworkReceive.rework_id || receiveRwMut.isPending}
              onClick={() => receiveRwMut.mutate({
                id: Number(reworkReceive.rework_id),
                body: {
                  received_qty: reworkReceive.received_qty,
                  pass_qty: reworkReceive.pass_qty,
                  reject_qty: reworkReceive.reject_qty,
                  remarks: reworkReceive.remarks,
                },
              })}
              className="px-3 py-2 bg-emerald-700 text-white rounded-lg disabled:opacity-50"
            >
              Receive rework
            </button>
          </div>

          <div className="bg-white border rounded-xl overflow-auto">
            <div className="px-3 py-2 text-xs text-gray-500">{rwLoading ? 'Loading…' : `${reworks.length} rework orders`}</div>
            <table className="w-full text-[11px]">
              <thead className="bg-gray-50 text-gray-500 uppercase">
                <tr>
                  {['Rework', 'Orig JO', 'Process', 'Plan', 'Rec', 'Pass', 'Reject', 'Bal', 'By', 'Charge?', 'Status', ''].map(h => (
                    <th key={h || 'print'} className="text-left px-2 py-2">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {reworks.map((r: any) => (
                  <tr key={r.id} className="border-t hover:bg-slate-50">
                    <td className="px-2 py-1.5 font-mono cursor-pointer" onClick={() => setReworkReceive(f => ({ ...f, rework_id: r.id }))}>{r.rework_no}</td>
                    <td className="px-2 py-1.5">{r.original_jo_number || `#${r.original_jo_id}`}</td>
                    <td className="px-2 py-1.5">{r.process}</td>
                    <td className="px-2 py-1.5 text-right">{fmt(r.planned_qty)}</td>
                    <td className="px-2 py-1.5 text-right">{fmt(r.received_qty)}</td>
                    <td className="px-2 py-1.5 text-right text-emerald-700">{fmt(r.pass_qty)}</td>
                    <td className="px-2 py-1.5 text-right text-rose-700">{fmt(r.reject_qty)}</td>
                    <td className="px-2 py-1.5 text-right font-semibold text-amber-700">{fmt(r.balance_qty)}</td>
                    <td className="px-2 py-1.5">{r.rework_by_type}{r.rework_by_vendor ? ` · ${r.rework_by_vendor}` : ''}</td>
                    <td className="px-2 py-1.5">{Number(r.chargeable) ? 'Yes' : 'No'}</td>
                    <td className="px-2 py-1.5">{r.status}</td>
                    <td className="px-2 py-1.5">
                      <button type="button" className="text-xs px-2 py-0.5 border rounded hover:bg-gray-50" onClick={() => void printReworkJO(r)}>🖨 Print</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {sub === 'debit' && (
        <div className="bg-white border rounded-xl overflow-auto">
          <div className="px-3 py-2 text-xs text-gray-500">{debits.length} debit notes (auto-created from QC when Debit / In-house / Final Reject)</div>
          <table className="w-full text-[11px]">
            <thead className="bg-gray-50 text-gray-500 uppercase">
              <tr>
                {['Debit No', 'Date', 'Vendor', 'Process', 'SO', 'SKU', 'Qty', 'Workmanship', 'Fabric', 'Total', 'With Fabric', 'Status', ''].map(h => (
                  <th key={h || 'print'} className="text-left px-2 py-2">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {debits.map((d: any) => (
                <tr key={d.id} className="border-t">
                  <td className="px-2 py-1.5 font-mono">{d.debit_no}</td>
                  <td className="px-2 py-1.5">{d.debit_date}</td>
                  <td className="px-2 py-1.5">{d.responsible_vendor}</td>
                  <td className="px-2 py-1.5">{d.process}</td>
                  <td className="px-2 py-1.5">{d.so_number}</td>
                  <td className="px-2 py-1.5 font-mono">{d.sku}</td>
                  <td className="px-2 py-1.5 text-right">{fmt(d.qty)}</td>
                  <td className="px-2 py-1.5 text-right">{fmt(d.workmanship_amount)}</td>
                  <td className="px-2 py-1.5 text-right">{fmt(d.fabric_amount)}</td>
                  <td className="px-2 py-1.5 text-right font-semibold">{fmt(d.total_amount)}</td>
                  <td className="px-2 py-1.5">{Number(d.with_fabric) ? 'Yes' : 'No'}</td>
                  <td className="px-2 py-1.5">{d.status}</td>
                  <td className="px-2 py-1.5">
                    <button type="button" className="text-xs px-2 py-0.5 border rounded hover:bg-gray-50" onClick={() => void printDebitNote(d)}>🖨 Print</button>
                  </td>
                </tr>
              ))}
              {debits.length === 0 && <tr><td colSpan={13} className="text-center text-gray-400 py-8">No debit notes yet.</td></tr>}
            </tbody>
          </table>
        </div>
      )}

      {sub === 'billing' && (
        <div className="space-y-4">
          <div className="bg-white border rounded-xl p-4 flex flex-wrap gap-2 items-end text-xs">
            <label className="block">
              <span className="text-gray-500">Vendor JO id</span>
              <input type="number" value={billingJoId || ''} onChange={e => setBillingJoId(e.target.value ? +e.target.value : '')} className="mt-0.5 block border rounded px-2 py-1 min-w-[10rem]" />
            </label>
            <p className="text-gray-500">Or load a JO from the QC tab first.</p>
          </div>
          {billLoading && <p className="text-xs text-gray-400">Loading…</p>}
          {billing && (
            <div className="bg-white border rounded-xl p-4 space-y-3 text-xs">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {([
                  ['Planned', billing.planned_qty],
                  ['Received', billing.received_qty],
                  ['QC Pass', billing.qc_pass_qty],
                  ['Billable', billing.billable_qty],
                  ['Rework Pending', billing.rework_pending_qty],
                  ['Rejected', billing.reject_qty],
                  ['Debit Qty', billing.debit_qty],
                  ['Debit Amt', billing.debit_amount],
                ] as const).map(([lab, val]) => (
                  <div key={lab} className="border rounded-lg p-2 bg-slate-50">
                    <div className="text-gray-500">{lab}</div>
                    <div className="font-semibold text-sm mt-0.5">{fmt(val as number)}</div>
                  </div>
                ))}
              </div>
              <p className="text-amber-800 bg-amber-50 border border-amber-200 rounded px-2 py-1">
                Received ≠ billable. Eligible qty follows QC pass; open rework holds full clearance.
                Same-vendor rework is non-chargeable by default.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
