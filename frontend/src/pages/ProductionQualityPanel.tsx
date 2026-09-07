import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

const PROCESSES = [
  'Cutting', 'Printing', 'Embroidery', 'Stitching', 'Kajh Button', 'Handwork', 'Finishing', 'Packing',
]
const fmt = (n: number | null | undefined) => Number(n || 0).toLocaleString()

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

/**
 * Production QC + Rework WIP + Debit Notes + Billing eligibility.
 */
export default function ProductionQualityPanel() {
  const qc = useQueryClient()
  const [sub, setSub] = useState<'qc' | 'rework' | 'debit' | 'billing'>('qc')
  const [joLookup, setJoLookup] = useState('')
  const [billingJoId, setBillingJoId] = useState<number | ''>('')

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
      setQcForm(f => ({
        ...f,
        original_jo_id: jo.id,
        so_number: jo.so_number || '',
        sku: jo.sku || '',
        component_code: jo.component_code || '',
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
                <span className="text-gray-500">Checked qty *</span>
                <input type="number" value={qcForm.checked_qty || ''} onChange={e => setQcForm(f => ({ ...f, checked_qty: +e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1" />
              </label>
              <label className="block col-span-2">
                <span className="text-gray-500">Notes</span>
                <input value={qcForm.notes} onChange={e => setQcForm(f => ({ ...f, notes: e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1" />
              </label>
            </div>

            <div className="border-t pt-3 space-y-2">
              <div className="flex justify-between items-center">
                <p className="text-xs font-semibold text-gray-700">Defects (Found At stays above; Source / Responsible / Rework By per line)</p>
                <button type="button" className="text-xs text-blue-700" onClick={() => setDefects(d => [...d, emptyDefect()])}>+ Add defect</button>
              </div>
              {defects.map((d, idx) => (
                <div key={idx} className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2 text-xs bg-slate-50 p-2 rounded-lg">
                  <label className="block">
                    <span className="text-gray-500">Defect Source *</span>
                    <select value={d.defect_source_process} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, defect_source_process: e.target.value } : x))} className="mt-0.5 w-full border rounded px-1 py-1">
                      {PROCESSES.map(p => <option key={p}>{p}</option>)}
                    </select>
                  </label>
                  <label className="block">
                    <span className="text-gray-500">Action</span>
                    <select value={d.action} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, action: e.target.value } : x))} className="mt-0.5 w-full border rounded px-1 py-1">
                      <option value="Rework">Rework</option>
                      <option value="Alteration">Alteration</option>
                      <option value="FinalReject">Final Reject</option>
                    </select>
                  </label>
                  <label className="block">
                    <span className="text-gray-500">Qty</span>
                    <input type="number" value={d.qty || ''} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, qty: +e.target.value } : x))} className="mt-0.5 w-full border rounded px-1 py-1" />
                  </label>
                  <label className="block">
                    <span className="text-gray-500">Responsible Vendor</span>
                    <input value={d.responsible_vendor} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, responsible_vendor: e.target.value } : x))} className="mt-0.5 w-full border rounded px-1 py-1" />
                  </label>
                  <label className="block">
                    <span className="text-gray-500">Rework By</span>
                    <select value={d.rework_by_type} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, rework_by_type: e.target.value, debit_required: e.target.value === 'Inhouse' || x.action === 'FinalReject' } : x))} className="mt-0.5 w-full border rounded px-1 py-1">
                      <option value="SameVendor">Same Vendor</option>
                      <option value="OtherVendor">Other Vendor</option>
                      <option value="Inhouse">In-house</option>
                    </select>
                  </label>
                  <label className="block">
                    <span className="text-gray-500">Rework Vendor / Dept</span>
                    <input value={d.rework_by_vendor} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, rework_by_vendor: e.target.value } : x))} className="mt-0.5 w-full border rounded px-1 py-1" />
                  </label>
                  <label className="block">
                    <span className="text-gray-500">Reason</span>
                    <input value={d.reason} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, reason: e.target.value } : x))} className="mt-0.5 w-full border rounded px-1 py-1" />
                  </label>
                  <div className="flex flex-col gap-1 justify-end pb-1">
                    <label className="flex items-center gap-1"><input type="checkbox" checked={d.debit_required} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, debit_required: e.target.checked } : x))} /> Debit note</label>
                    <label className="flex items-center gap-1"><input type="checkbox" checked={d.with_fabric} onChange={e => setDefects(arr => arr.map((x, i) => i === idx ? { ...x, with_fabric: e.target.checked } : x))} /> With Fabric</label>
                    {defects.length > 1 && (
                      <button type="button" className="text-rose-600 text-left" onClick={() => setDefects(arr => arr.filter((_, i) => i !== idx))}>Remove</button>
                    )}
                  </div>
                </div>
              ))}
              <p className="text-[11px] text-gray-500">
                Defect qty total: {defectQtySum}. Pass will be Checked − Rework − Reject.
              </p>
            </div>

            <button
              type="button"
              disabled={createQcMut.isPending}
              onClick={submitQc}
              className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50"
            >
              {createQcMut.isPending ? 'Saving…' : 'Save QC Report'}
            </button>
          </div>

          <div className="bg-white border rounded-xl overflow-auto">
            <div className="px-3 py-2 text-xs text-gray-500">{qcLoading ? 'Loading…' : `${qcReports.length} recent QC reports`}</div>
            <table className="w-full text-[11px]">
              <thead className="bg-gray-50 text-gray-500 uppercase">
                <tr>
                  {['Report', 'Date', 'Found At', 'SO', 'SKU', 'Checked', 'Pass', 'Rework', 'Reject', 'Defects'].map(h => (
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
                    <td className="px-2 py-1.5">{(r.defects || []).length}</td>
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
            {(wip as any[]).slice(0, 8).map((w: any, i: number) => (
              <div key={i} className="bg-white border rounded-lg p-2 text-xs">
                <p className="font-semibold text-[#002B5B]">{w.process} Rework</p>
                <p className="text-gray-500">{w.so_number} · {w.sku}</p>
                <p className="text-amber-700 font-bold mt-1">Pending {fmt(w.rework_pending)}</p>
              </div>
            ))}
            {(wip as any[]).length === 0 && <p className="text-xs text-gray-400 col-span-4">No open rework WIP.</p>}
          </div>

          <div className="bg-white border rounded-xl p-4 grid grid-cols-2 md:grid-cols-5 gap-2 text-xs items-end">
            <label className="block">
              <span className="text-gray-500">Rework id</span>
              <input type="number" value={reworkReceive.rework_id || ''} onChange={e => setReworkReceive(f => ({ ...f, rework_id: e.target.value ? +e.target.value : '' }))} className="mt-0.5 w-full border rounded px-2 py-1" />
            </label>
            <label className="block">
              <span className="text-gray-500">Received</span>
              <input type="number" value={reworkReceive.received_qty || ''} onChange={e => setReworkReceive(f => ({ ...f, received_qty: +e.target.value, pass_qty: +e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1" />
            </label>
            <label className="block">
              <span className="text-gray-500">Pass</span>
              <input type="number" value={reworkReceive.pass_qty || ''} onChange={e => setReworkReceive(f => ({ ...f, pass_qty: +e.target.value }))} className="mt-0.5 w-full border rounded px-2 py-1" />
            </label>
            <label className="block">
              <span className="text-gray-500">Final Reject</span>
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
              className="px-3 py-1.5 bg-emerald-700 text-white rounded-lg disabled:opacity-50"
            >
              Receive rework
            </button>
          </div>

          <div className="bg-white border rounded-xl overflow-auto">
            <div className="px-3 py-2 text-xs text-gray-500">{rwLoading ? 'Loading…' : `${reworks.length} rework orders`}</div>
            <table className="w-full text-[11px]">
              <thead className="bg-gray-50 text-gray-500 uppercase">
                <tr>
                  {['Rework', 'Orig JO', 'Process', 'Plan', 'Rec', 'Pass', 'Reject', 'Bal', 'By', 'Charge?', 'Status'].map(h => (
                    <th key={h} className="text-left px-2 py-2">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {reworks.map((r: any) => (
                  <tr key={r.id} className="border-t hover:bg-slate-50 cursor-pointer" onClick={() => setReworkReceive(f => ({ ...f, rework_id: r.id }))}>
                    <td className="px-2 py-1.5 font-mono">{r.rework_no}</td>
                    <td className="px-2 py-1.5">#{r.original_jo_id}</td>
                    <td className="px-2 py-1.5">{r.process}</td>
                    <td className="px-2 py-1.5 text-right">{fmt(r.planned_qty)}</td>
                    <td className="px-2 py-1.5 text-right">{fmt(r.received_qty)}</td>
                    <td className="px-2 py-1.5 text-right text-emerald-700">{fmt(r.pass_qty)}</td>
                    <td className="px-2 py-1.5 text-right text-rose-700">{fmt(r.reject_qty)}</td>
                    <td className="px-2 py-1.5 text-right font-semibold text-amber-700">{fmt(r.balance_qty)}</td>
                    <td className="px-2 py-1.5">{r.rework_by_type}{r.rework_by_vendor ? ` · ${r.rework_by_vendor}` : ''}</td>
                    <td className="px-2 py-1.5">{Number(r.chargeable) ? 'Yes' : 'No'}</td>
                    <td className="px-2 py-1.5">{r.status}</td>
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
                {['Debit No', 'Date', 'Vendor', 'Process', 'SO', 'SKU', 'Qty', 'Workmanship', 'Fabric', 'Total', 'With Fabric', 'Status'].map(h => (
                  <th key={h} className="text-left px-2 py-2">{h}</th>
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
                </tr>
              ))}
              {debits.length === 0 && <tr><td colSpan={12} className="text-center text-gray-400 py-8">No debit notes yet.</td></tr>}
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
            <div className="bg-white border rounded-xl p-4 space-y-3 text-sm">
              <div className="flex flex-wrap gap-4">
                <div><span className="text-gray-500 text-xs">JO</span><p className="font-semibold">{billing.jo_number} · {billing.process}</p></div>
                <div><span className="text-gray-500 text-xs">Vendor</span><p className="font-semibold">{billing.vendor_name || '—'}</p></div>
                <div><span className="text-gray-500 text-xs">Billing QC process</span><p className="font-semibold">{billing.billing_qc_process}</p></div>
                <div><span className="text-gray-500 text-xs">Status</span><p className="font-bold text-amber-800">{billing.billing_status}</p></div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2 text-xs">
                {[
                  ['Planned', billing.planned_qty],
                  ['Received', billing.received_qty],
                  ['QC Checked', billing.qc_checked_qty],
                  ['QC Pass', billing.qc_pass_qty],
                  ['Rework Pending', billing.rework_pending_qty],
                  ['Eligible to bill', billing.eligible_billing_qty],
                  ['Final Reject', billing.qc_reject_qty],
                  ['Debit Qty', billing.debit_qty],
                  ['Debit Amt', billing.debit_amount],
                ].map(([l, v]) => (
                  <div key={String(l)} className="border rounded-lg p-2 text-center">
                    <p className="font-bold text-[#002B5B]">{fmt(v as number)}</p>
                    <p className="text-gray-500">{l}</p>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-gray-500">
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
