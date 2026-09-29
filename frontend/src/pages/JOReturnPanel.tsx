import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'
import api from '../api/client'

export interface ReturnableLine {
  id: number
  sku: string
  style?: string
  planned_qty: number
  received_qty: number
  unprocessed_return_qty?: number
}

export interface ReturnableJO {
  id: number
  jo_number: string
  process: string
  vendor_name: string
  sku: string
  planned_qty: number
  received_qty: number
  unprocessed_return_qty?: number
  lines: ReturnableLine[]
}

interface ReconRow {
  material_kind: 'PIECES' | 'FABRIC' | 'ACCESSORY'
  material_code: string
  material_name: string
  unit: string
  issued_qty: number
  consumed_qty: number
  returned_qty: number
  wastage_qty: number
  balance_qty: number
  expected_consumption: number | null
  settled: boolean
  editable: string[]
  remarks: string
  confirmed: boolean
}

interface Reconciliation {
  jo_id: number
  jo_number: string
  process: string
  vendor_name: string
  status: string
  reconciliation_status: string
  reconciled_at: string
  reconciled_by: string
  rows: ReconRow[]
  pending_materials: string[]
  can_complete: boolean
}

interface ReturnHistory {
  summary: {
    original_qty: number
    processed_qty: number
    unprocessed_returned_qty: number
    reallocated_qty: number
    pending_in_ready_qty: number
  }
  returns: {
    id: number
    return_number: string
    return_date: string
    vendor_name: string
    processed_qty: number
    unprocessed_qty: number
    rejected_qty: number
    reason: string
    returned_by: string
  }[]
  outgoing: {
    id: number
    return_number: string
    sku: string
    process: string
    source_vendor: string
    unprocessed_qty: number
    ready_at: string
    pending_in_ready_qty: number
    allocations: { id: number; new_jo_number: string; new_vendor: string; qty: number; released: number; created_at: string }[]
  }[]
  incoming: {
    id: number
    qty: number
    released: number
    source_jo_number: string
    source_vendor: string
    return_number: string
    sku: string
    ready_at: string
  }[]
}

const KIND_LABEL: Record<string, string> = { PIECES: 'Pieces', FABRIC: 'Fabric', ACCESSORY: 'Accessory' }

function errMsg(err: unknown, fallback: string): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string' && detail.trim()) return detail
  }
  return fallback
}

const n = (v: number | null | undefined) => {
  const x = Number(v || 0)
  return Number.isInteger(x) ? x.toLocaleString('en-IN') : x.toLocaleString('en-IN', { maximumFractionDigits: 3 })
}

export function ReconciliationBadge({ status }: { status?: string }) {
  if (status === 'Pending') {
    return (
      <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-rose-100 text-rose-700">
        Reconciliation Pending
      </span>
    )
  }
  if (status === 'Completed') {
    return (
      <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-emerald-50 text-emerald-700">
        Reconciled
      </span>
    )
  }
  return null
}

function useInvalidateJoReturn() {
  const qc = useQueryClient()
  return (joId: number) => {
    for (const key of ['jos-process', 'jos-all', 'ready-to-process', 'prod-stats', 'process-report', 'cutting-report']) {
      qc.invalidateQueries({ queryKey: [key] })
    }
    qc.invalidateQueries({ queryKey: ['jo-reconciliation', joId] })
    qc.invalidateQueries({ queryKey: ['jo-return-history', joId] })
  }
}

export function JOReturnModal({
  jo,
  onClose,
  onReturned,
}: {
  jo: ReturnableJO
  onClose: () => void
  onReturned: () => void
}) {
  const invalidate = useInvalidateJoReturn()
  const srcLines: ReturnableLine[] = jo.lines.length
    ? jo.lines
    : [{ id: 0, sku: jo.sku, planned_qty: jo.planned_qty, received_qty: jo.received_qty }]
  const [qty, setQty] = useState<Record<number, { processed: string; unprocessed: string; rejected: string }>>({})
  const [form, setForm] = useState({
    return_date: new Date().toISOString().slice(0, 10),
    reason: '',
    remarks: '',
    returned_by: '',
  })

  const pendingOf = (l: ReturnableLine) => Math.max(0, (l.planned_qty || 0) - (l.received_qty || 0))
  const val = (id: number, k: 'processed' | 'unprocessed' | 'rejected') => parseInt(qty[id]?.[k] || '0', 10) || 0
  const setVal = (id: number, k: 'processed' | 'unprocessed' | 'rejected', v: string) =>
    setQty(m => ({ ...m, [id]: { ...(m[id] || { processed: '', unprocessed: '', rejected: '' }), [k]: v } }))

  const totals = srcLines.reduce(
    (t, l) => ({ p: t.p + val(l.id, 'processed'), u: t.u + val(l.id, 'unprocessed') }),
    { p: 0, u: 0 },
  )
  const overLine = srcLines.find(l => val(l.id, 'processed') + val(l.id, 'unprocessed') > pendingOf(l))

  const mut = useMutation({
    mutationFn: () =>
      api.post(`/production/orders/${jo.id}/return`, {
        ...form,
        lines: srcLines.map(l => ({
          jo_line_id: l.id || null,
          processed_qty: val(l.id, 'processed'),
          unprocessed_qty: val(l.id, 'unprocessed'),
          rejected_qty: val(l.id, 'rejected'),
        })),
      }),
    onSuccess: res => {
      invalidate(jo.id)
      const d = res.data
      alert(
        `Return ${d.return_number} posted.\n`
        + `Processed: ${d.processed_qty} pcs (normal receive)\n`
        + `Unprocessed: ${d.unprocessed_qty} pcs → back to Ready To ${jo.process}\n\n`
        + 'Complete the material reconciliation next.',
      )
      onReturned()
    },
    onError: e => alert(errMsg(e, 'Could not post return')),
  })

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl p-6 space-y-4 my-4">
        <div className="flex justify-between items-center">
          <h3 className="font-semibold text-gray-700">↩ Vendor Return — {jo.jo_number}</h3>
          <button onClick={onClose} className="text-gray-400 text-xl">✕</button>
        </div>
        <p className="text-xs text-gray-500">
          <b>{jo.vendor_name || 'Vendor'}</b> · {jo.process}. <b>Processed</b> pieces are received normally and move
          to the next process. <b>Unprocessed</b> pieces go back to <b>Ready To {jo.process}</b> for a new JO.
        </p>
        <table className="w-full text-xs border rounded-lg overflow-hidden">
          <thead className="bg-gray-50 text-gray-500 uppercase">
            <tr>
              <th className="text-left px-3 py-2">SKU</th>
              <th className="text-right px-3 py-2">With vendor</th>
              <th className="text-right px-3 py-2">Processed</th>
              <th className="text-right px-3 py-2">Rejected (of processed)</th>
              <th className="text-right px-3 py-2">Unprocessed</th>
              <th className="text-right px-3 py-2">Still with vendor</th>
            </tr>
          </thead>
          <tbody>
            {srcLines.map(l => {
              const pending = pendingOf(l)
              const left = pending - val(l.id, 'processed') - val(l.id, 'unprocessed')
              return (
                <tr key={l.id} className="border-t">
                  <td className="px-3 py-2 font-mono font-semibold text-[#002B5B]">{l.sku}</td>
                  <td className="px-3 py-2 text-right">{n(pending)}</td>
                  {(['processed', 'rejected', 'unprocessed'] as const).map(k => (
                    <td key={k} className="px-3 py-2 text-right">
                      <input
                        type="number"
                        min={0}
                        max={k === 'rejected' ? undefined : pending}
                        disabled={pending <= 0}
                        value={qty[l.id]?.[k] ?? ''}
                        onChange={e => setVal(l.id, k, e.target.value)}
                        className="w-20 border rounded px-1 py-0.5 text-right"
                      />
                    </td>
                  ))}
                  <td className={`px-3 py-2 text-right font-semibold ${left < 0 ? 'text-red-600' : 'text-amber-600'}`}>{n(left)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <label className="space-y-1">
            <span className="text-gray-500">Return date</span>
            <input type="date" value={form.return_date} onChange={e => setForm(f => ({ ...f, return_date: e.target.value }))}
              className="w-full border rounded px-2 py-1" />
          </label>
          <label className="space-y-1">
            <span className="text-gray-500">Received by</span>
            <input value={form.returned_by} onChange={e => setForm(f => ({ ...f, returned_by: e.target.value }))}
              className="w-full border rounded px-2 py-1" />
          </label>
          <label className="space-y-1 col-span-2">
            <span className="text-gray-500">Reason for unprocessed return</span>
            <input value={form.reason} onChange={e => setForm(f => ({ ...f, reason: e.target.value }))}
              placeholder="e.g. vendor capacity, quality concern, job cancelled"
              className="w-full border rounded px-2 py-1" />
          </label>
          <label className="space-y-1 col-span-2">
            <span className="text-gray-500">Remarks</span>
            <input value={form.remarks} onChange={e => setForm(f => ({ ...f, remarks: e.target.value }))}
              className="w-full border rounded px-2 py-1" />
          </label>
        </div>
        {overLine && (
          <p className="text-xs text-red-600">{overLine.sku}: processed + unprocessed exceeds pieces with vendor.</p>
        )}
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="px-3 py-1.5 text-xs border rounded-lg">Cancel</button>
          <button
            onClick={() => mut.mutate()}
            disabled={mut.isPending || Boolean(overLine) || totals.p + totals.u <= 0}
            className="px-3 py-1.5 text-xs bg-rose-600 text-white rounded-lg font-medium disabled:opacity-50"
          >
            {mut.isPending ? 'Posting…' : `Post return (${totals.p} processed · ${totals.u} unprocessed)`}
          </button>
        </div>
      </div>
    </div>
  )
}

type Draft = Record<string, Partial<Record<'issued_qty' | 'consumed_qty' | 'returned_qty' | 'wastage_qty' | 'remarks', string>>>

export function JOReconciliationModal({ joId, onClose }: { joId: number; onClose: () => void }) {
  const invalidate = useInvalidateJoReturn()
  const { data, isLoading } = useQuery<Reconciliation>({
    queryKey: ['jo-reconciliation', joId],
    queryFn: () => api.get(`/production/orders/${joId}/reconciliation`).then(r => r.data),
  })
  const [draft, setDraft] = useState<Draft>({})
  const [by, setBy] = useState('')

  const key = (r: ReconRow) => `${r.material_kind}|${r.material_code}`
  const num = (r: ReconRow, f: 'issued_qty' | 'consumed_qty' | 'returned_qty' | 'wastage_qty') => {
    const raw = draft[key(r)]?.[f]
    return raw == null || raw === '' ? Number(r[f] || 0) : Number(raw)
  }
  const balanceOf = (r: ReconRow) =>
    Math.round((num(r, 'issued_qty') - num(r, 'consumed_qty') - num(r, 'returned_qty') - num(r, 'wastage_qty')) * 1000) / 1000

  const save = useMutation({
    mutationFn: (complete: boolean) =>
      api.post(`/production/orders/${joId}/reconciliation`, {
        complete,
        reconciled_by: by,
        rows: (data?.rows || []).map(r => {
          const d = draft[key(r)] || {}
          const row: Record<string, unknown> = {
            material_kind: r.material_kind,
            material_code: r.material_code,
            remarks: d.remarks ?? r.remarks,
          }
          for (const f of r.editable) row[f] = num(r, f as 'issued_qty')
          return row
        }),
      }),
    onSuccess: (res, complete) => {
      invalidate(joId)
      setDraft({})
      if (complete) {
        alert(`Reconciliation completed for ${res.data.jo_number}.`)
        onClose()
      }
    },
    onError: e => alert(errMsg(e, 'Could not save reconciliation')),
  })

  const cell = (r: ReconRow, f: 'issued_qty' | 'consumed_qty' | 'returned_qty' | 'wastage_qty') =>
    r.editable.includes(f) && data?.reconciliation_status !== 'Completed' ? (
      <input
        type="number"
        min={0}
        step="any"
        value={draft[key(r)]?.[f] ?? String(r[f] ?? 0)}
        onChange={e => setDraft(m => ({ ...m, [key(r)]: { ...m[key(r)], [f]: e.target.value } }))}
        className="w-20 border border-amber-200 bg-amber-50 rounded px-1 py-0.5 text-right"
      />
    ) : (
      n(r[f])
    )

  const unsettled = (data?.rows || []).filter(r => {
    const b = balanceOf(r)
    return r.material_kind === 'PIECES' ? b > 0.001 : Math.abs(b) > 0.001
  })

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl p-6 space-y-4 my-4">
        <div className="flex justify-between items-center">
          <h3 className="font-semibold text-gray-700 flex items-center gap-2">
            🧾 Material Reconciliation — {data?.jo_number || '…'}
            <ReconciliationBadge status={data?.reconciliation_status} />
          </h3>
          <button onClick={onClose} className="text-gray-400 text-xl">✕</button>
        </div>
        {isLoading || !data ? (
          <p className="text-xs text-gray-400">Loading…</p>
        ) : (
          <>
            <p className="text-xs text-gray-500">
              {data.vendor_name || 'Vendor'} · {data.process}. Every material issued against this JO must balance:
              <b> Issued = Consumed + Returned + Wastage/Loss + Balance with vendor</b>. Billing stays on hold until the
              balance is zero for every row.
            </p>
            {data.rows.length === 0 ? (
              <p className="text-xs text-gray-500">No materials were issued against this JO.</p>
            ) : (
              <table className="w-full text-xs border rounded-lg overflow-hidden">
                <thead className="bg-gray-50 text-gray-500 uppercase">
                  <tr>
                    <th className="text-left px-3 py-2">Type</th>
                    <th className="text-left px-3 py-2">Material</th>
                    <th className="text-right px-3 py-2">Issued</th>
                    <th className="text-right px-3 py-2">Consumed</th>
                    <th className="text-right px-3 py-2">Returned</th>
                    <th className="text-right px-3 py-2">Wastage / Loss</th>
                    <th className="text-right px-3 py-2">Balance with vendor</th>
                    <th className="text-left px-3 py-2">Remarks</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map(r => {
                    const bal = balanceOf(r)
                    const ok = r.material_kind === 'PIECES' ? bal <= 0.001 : Math.abs(bal) <= 0.001
                    return (
                      <tr key={key(r)} className="border-t">
                        <td className="px-3 py-2 text-gray-500">{KIND_LABEL[r.material_kind]}</td>
                        <td className="px-3 py-2">
                          <span className="font-mono font-semibold text-[#002B5B]">{r.material_code}</span>
                          {r.material_name && r.material_name !== r.material_code && (
                            <span className="block text-gray-500">{r.material_name}</span>
                          )}
                          {r.expected_consumption != null && (
                            <span className="block text-[10px] text-gray-400">BOM expected: {n(r.expected_consumption)} {r.unit}</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right">{cell(r, 'issued_qty')}</td>
                        <td className="px-3 py-2 text-right">{cell(r, 'consumed_qty')}</td>
                        <td className="px-3 py-2 text-right">{cell(r, 'returned_qty')}</td>
                        <td className="px-3 py-2 text-right">{cell(r, 'wastage_qty')}</td>
                        <td className={`px-3 py-2 text-right font-semibold ${ok ? 'text-emerald-600' : 'text-rose-600'}`}>
                          {n(bal)} {r.unit}
                        </td>
                        <td className="px-3 py-2">
                          <input
                            value={draft[key(r)]?.remarks ?? r.remarks}
                            disabled={data.reconciliation_status === 'Completed'}
                            onChange={e => setDraft(m => ({ ...m, [key(r)]: { ...m[key(r)], remarks: e.target.value } }))}
                            className="w-full border rounded px-1 py-0.5"
                          />
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}
            <p className="text-[11px] text-gray-400">
              Pieces are derived from receipts and returns (only wastage is editable). Fabric returns are posted with
              “Return Fabric”. Accessory issued qty is pre-filled from the BOM issue note — correct it if the actual
              issue differed.
            </p>
            {data.reconciliation_status === 'Completed' ? (
              <p className="text-xs text-emerald-700">
                Reconciled {data.reconciled_at ? `on ${data.reconciled_at}` : ''} {data.reconciled_by ? `by ${data.reconciled_by}` : ''}
              </p>
            ) : (
              <div className="flex flex-wrap items-center justify-end gap-2">
                {unsettled.length > 0 && (
                  <span className="text-xs text-rose-600 mr-auto">
                    Balance remains on: {unsettled.map(r => r.material_code).join(', ')}
                  </span>
                )}
                <input value={by} onChange={e => setBy(e.target.value)} placeholder="Reconciled by"
                  className="border rounded px-2 py-1 text-xs" />
                <button onClick={() => save.mutate(false)} disabled={save.isPending}
                  className="px-3 py-1.5 text-xs border rounded-lg disabled:opacity-50">
                  Save draft
                </button>
                <button onClick={() => save.mutate(true)} disabled={save.isPending || unsettled.length > 0}
                  className="px-3 py-1.5 text-xs bg-emerald-600 text-white rounded-lg font-medium disabled:opacity-50">
                  ✅ Complete reconciliation
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export function JOReturnHistoryPanel({ joId }: { joId: number }) {
  const { data } = useQuery<ReturnHistory>({
    queryKey: ['jo-return-history', joId],
    queryFn: () => api.get(`/production/orders/${joId}/return-history`).then(r => r.data),
  })
  if (!data || (!data.returns.length && !data.incoming.length)) return null
  const s = data.summary
  return (
    <div className="bg-white rounded-lg border border-rose-100 overflow-hidden">
      <div className="px-3 py-2 bg-rose-50 text-xs font-semibold text-rose-900 flex flex-wrap gap-x-4 gap-y-1">
        <span>↩ Return history &amp; traceability</span>
        {data.returns.length > 0 && (
          <>
            <span className="font-normal">Original qty <b>{n(s.original_qty)}</b></span>
            <span className="font-normal">Processed <b>{n(s.processed_qty)}</b></span>
            <span className="font-normal">Unprocessed return <b>{n(s.unprocessed_returned_qty)}</b></span>
            <span className="font-normal">Re-issued to new JOs <b>{n(s.reallocated_qty)}</b></span>
            <span className="font-normal">Still in Ready-To <b>{n(s.pending_in_ready_qty)}</b></span>
          </>
        )}
      </div>
      {data.incoming.length > 0 && (
        <div className="px-3 py-2 text-xs border-b border-rose-50">
          <p className="font-semibold text-gray-600 mb-1">This JO was issued from returned qty:</p>
          {data.incoming.map(a => (
            <p key={a.id} className={a.released ? 'line-through text-gray-400' : 'text-gray-700'}>
              {n(a.qty)} pcs {a.sku} — from <b>{a.source_jo_number}</b> ({a.source_vendor || '—'}) via{' '}
              <span className="font-mono">{a.return_number}</span>, back in Ready-To {a.ready_at}
            </p>
          ))}
        </div>
      )}
      {data.returns.length > 0 && (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-400 border-b uppercase">
              <th className="text-left px-3 py-2">Return</th>
              <th className="text-left px-3 py-2">Date</th>
              <th className="text-right px-3 py-2">Processed</th>
              <th className="text-right px-3 py-2">Unprocessed</th>
              <th className="text-left px-3 py-2">Back in Ready-To → New JO / vendor</th>
              <th className="text-left px-3 py-2">Reason</th>
            </tr>
          </thead>
          <tbody>
            {data.returns.map(r => {
              const links = data.outgoing.filter(l => l.return_number === r.return_number)
              return (
                <tr key={r.id} className="border-t border-gray-50 align-top">
                  <td className="px-3 py-2 font-mono font-semibold text-rose-800">{r.return_number}</td>
                  <td className="px-3 py-2">{r.return_date}</td>
                  <td className="px-3 py-2 text-right text-green-700">{n(r.processed_qty)}</td>
                  <td className="px-3 py-2 text-right text-rose-700">{n(r.unprocessed_qty)}</td>
                  <td className="px-3 py-2">
                    {links.length === 0 && <span className="text-gray-400">—</span>}
                    {links.map(l => (
                      <div key={l.id} className="mb-1">
                        <span className="text-gray-600">{n(l.unprocessed_qty)} {l.sku} → Ready To {l.process} at {l.ready_at}</span>
                        {l.allocations.map(a => (
                          <span key={a.id} className={`block pl-3 ${a.released ? 'line-through text-gray-400' : 'text-indigo-700'}`}>
                            → {n(a.qty)} on <b>{a.new_jo_number}</b> ({a.new_vendor || 'in-house'}) · {a.created_at}
                          </span>
                        ))}
                        {l.pending_in_ready_qty > 0 && (
                          <span className="block pl-3 text-amber-700">{n(l.pending_in_ready_qty)} awaiting new JO</span>
                        )}
                      </div>
                    ))}
                  </td>
                  <td className="px-3 py-2 text-gray-500">{r.reason || '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}
