import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'
import { useAuth } from '../store/auth'

type Dash = {
  party_name?: string
  party_code?: string
  party_kind?: string
  portal_role?: string
  purchase_orders?: number
  job_work_orders?: number
  production_job_orders?: number
  note?: string
}

type PoRow = {
  id: number
  po_number?: string
  po_date?: string
  status?: string
  delivery_date?: string
  total?: number
  vendor_response?: { action?: string; reason?: string } | null
}

type JwoRow = {
  id: number
  jwo_number?: string
  jwo_date?: string
  status?: string
  expected_return_date?: string
  process_type?: string
  total?: number
  vendor_response?: { action?: string; reason?: string } | null
}

type MinLine = { id: number; material_code?: string; material_name?: string; issue_qty?: number; latest_ack?: { received_qty?: number; variance_qty?: number } | null }
type MinNote = { id: number; min_number?: string; min_date?: string; lines?: MinLine[] }

type JoRow = {
  id: number
  jo_number?: string
  jo_date?: string
  sku?: string
  process?: string
  planned_qty?: number
  issued_qty?: number
  received_qty?: number
  balance_qty?: number
  status?: string
}

function Card({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="bg-white border rounded-xl px-4 py-3">
      <p className="text-[11px] uppercase tracking-wide text-gray-400">{label}</p>
      <p className="text-2xl font-semibold text-[#002B5B] mt-1">{value}</p>
    </div>
  )
}

export default function VendorPortal() {
  const user = useAuth(s => s.user)
  const clear = useAuth(s => s.clear)
  const qc = useQueryClient()
  const [msg, setMsg] = useState('')
  const [challan, setChallan] = useState('')
  const [qty, setQty] = useState('')
  const dash = useQuery({
    queryKey: ['vendor-dashboard'],
    queryFn: () => api.get<Dash>('/vendor/dashboard').then(r => r.data),
  })
  const pos = useQuery({
    queryKey: ['vendor-pos'],
    queryFn: () => api.get<PoRow[]>('/vendor/purchase-orders').then(r => r.data),
    enabled: dash.data?.party_kind === 'supplier',
  })
  const jwos = useQuery({
    queryKey: ['vendor-jwos'],
    queryFn: () => api.get<JwoRow[]>('/vendor/job-work-orders').then(r => r.data),
    enabled: dash.data?.party_kind === 'processor',
  })
  const jos = useQuery({
    queryKey: ['vendor-jos'],
    queryFn: () => api.get<JoRow[]>('/vendor/job-orders').then(r => r.data),
  })
  const issues = useQuery({
    queryKey: ['vendor-mins'],
    queryFn: () => api.get<MinNote[]>('/vendor/material-issues').then(r => r.data),
    enabled: !!dash.data,
  })

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ['vendor-pos'] })
    qc.invalidateQueries({ queryKey: ['vendor-jwos'] })
    qc.invalidateQueries({ queryKey: ['vendor-jos'] })
    qc.invalidateQueries({ queryKey: ['vendor-mins'] })
  }

  const respond = async (path: string, action: string) => {
    let reason = ''
    if (action !== 'accept') {
      reason = window.prompt('Reason') || ''
      if (!reason.trim()) return
    }
    try {
      await api.post(path, { action, reason })
      setMsg(action === 'accept' ? 'Response saved. Factory stock is unchanged.' : 'Response saved.')
      refresh()
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setMsg(detail || 'Could not save the response')
    }
  }

  const ackLine = async (minId: number, line: MinLine) => {
    const raw = window.prompt(`Quantity physically received for ${line.material_code || 'this line'} (issued ${line.issue_qty ?? 0})`, String(line.issue_qty ?? 0))
    if (raw == null) return
    const received = Number(raw)
    let reason = ''
    if (received !== Number(line.issue_qty ?? 0)) {
      reason = window.prompt('Received quantity differs from issued. Reason') || ''
      if (!reason.trim()) return
    }
    try {
      await api.post(`/vendor/material-issues/${minId}/acknowledge`, { lines: [{ line_id: line.id, received_qty: received, reason }] })
      setMsg('Receipt recorded. Issued quantity in the ERP was not changed.')
      refresh()
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setMsg(detail || 'Could not save acknowledgement')
    }
  }

  const sendDelivery = async (docType: string, docId: number) => {
    const n = Number(qty)
    if (!n || n <= 0) { setMsg('Enter a delivery quantity first.'); return }
    try {
      const res = await api.post('/vendor/delivery-intimations', { doc_type: docType, doc_id: docId, qty: n, challan_no: challan })
      setMsg(`Delivery intimation ${res.data.intimation_no} submitted. This is not a GRN.`)
      setQty('')
      refresh()
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setMsg(detail || 'Could not submit delivery')
    }
  }

  const signOut = async () => {
    try { await api.post('/auth/logout') } catch { /* cookie clear is best-effort */ }
    clear()
    window.location.href = '/login'
  }

  const d = dash.data
  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-[#002B5B] text-white px-4 py-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs text-blue-200">Yash Gallery ERP · Vendor Portal</p>
          <h1 className="text-lg font-semibold">{d?.party_name || user?.full_name || user?.username}</h1>
          <p className="text-xs text-blue-100">
            {d?.party_code ? `${d.party_code} · ` : ''}
            {d?.party_kind === 'processor' ? 'Job work / printing' : d?.party_kind === 'supplier' ? 'Purchase supplier' : 'Linked account'}
            {d?.portal_role ? ` · ${d.portal_role.replace('vendor_', '')}` : ''}
          </p>
        </div>
        <button type="button" onClick={signOut} className="text-sm px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20">
          Sign out
        </button>
      </header>

      <main className="max-w-6xl mx-auto p-4 space-y-4">
        {dash.isError && (
          <p className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
            This login is not linked to an active supplier or processor. Ask ERP admin to activate vendor access.
          </p>
        )}
        {d && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <Card label="Purchase orders" value={d.purchase_orders ?? 0} />
              <Card label="Job work orders" value={d.job_work_orders ?? 0} />
              <Card label="Production job orders" value={d.production_job_orders ?? 0} />
            </div>
            <p className="text-xs text-gray-500">{d.note}</p>
            {msg && <p className="text-sm text-[#002B5B] bg-blue-50 border border-blue-100 rounded-lg px-3 py-2">{msg}</p>}
            <div className="flex flex-wrap gap-2 items-end">
              <label className="text-xs text-gray-500">Delivery qty
                <input value={qty} onChange={e => setQty(e.target.value)} className="block border rounded px-2 py-1 text-sm mt-1 w-28" />
              </label>
              <label className="text-xs text-gray-500">Challan no.
                <input value={challan} onChange={e => setChallan(e.target.value)} className="block border rounded px-2 py-1 text-sm mt-1 w-40" />
              </label>
            </div>
          </>
        )}

        {d?.party_kind === 'supplier' && (
          <section className="bg-white border rounded-xl overflow-hidden">
            <h2 className="px-4 py-2.5 font-semibold text-sm text-[#002B5B] border-b">Purchase orders</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 text-gray-500 text-xs">
                  <tr>
                    <th className="text-left px-4 py-2">PO</th>
                    <th className="text-left px-4 py-2">Date</th>
                    <th className="text-left px-4 py-2">Delivery</th>
                    <th className="text-left px-4 py-2">Status</th>
                    <th className="text-left px-4 py-2">Your response</th>
                    <th className="text-right px-4 py-2">Total</th>
                    <th className="px-4 py-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {(pos.data || []).map(row => (
                    <tr key={row.id} className="border-t">
                      <td className="px-4 py-2 font-medium">{row.po_number}</td>
                      <td className="px-4 py-2">{row.po_date}</td>
                      <td className="px-4 py-2">{row.delivery_date || '—'}</td>
                      <td className="px-4 py-2">{row.status}</td>
                      <td className="px-4 py-2">{row.vendor_response?.action || '—'}</td>
                      <td className="px-4 py-2 text-right">{row.total == null ? '—' : row.total}</td>
                      <td className="px-4 py-2 text-right whitespace-nowrap">
                        <button type="button" className="text-xs text-teal-700 mr-2" onClick={() => respond(`/vendor/purchase-orders/${row.id}/response`, 'accept')}>Accept</button>
                        <button type="button" className="text-xs text-red-700 mr-2" onClick={() => respond(`/vendor/purchase-orders/${row.id}/response`, 'reject')}>Reject</button>
                        <button type="button" className="text-xs text-blue-700" onClick={() => sendDelivery('po', row.id)}>Send qty</button>
                      </td>
                    </tr>
                  ))}
                  {!pos.isLoading && (pos.data || []).length === 0 && (
                    <tr><td colSpan={7} className="px-4 py-6 text-center text-gray-400">No purchase orders for this supplier.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {d?.party_kind === 'processor' && (
          <section className="bg-white border rounded-xl overflow-hidden">
            <h2 className="px-4 py-2.5 font-semibold text-sm text-[#002B5B] border-b">Job work orders</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 text-gray-500 text-xs">
                  <tr>
                    <th className="text-left px-4 py-2">JWO</th>
                    <th className="text-left px-4 py-2">Date</th>
                    <th className="text-left px-4 py-2">Expected return</th>
                    <th className="text-left px-4 py-2">Status</th>
                    <th className="text-left px-4 py-2">Your response</th>
                    <th className="px-4 py-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {(jwos.data || []).map(row => (
                    <tr key={row.id} className="border-t">
                      <td className="px-4 py-2 font-medium">{row.jwo_number}</td>
                      <td className="px-4 py-2">{row.jwo_date}</td>
                      <td className="px-4 py-2">{row.expected_return_date || '—'}</td>
                      <td className="px-4 py-2">{row.status}</td>
                      <td className="px-4 py-2">{row.vendor_response?.action || '—'}</td>
                      <td className="px-4 py-2 text-right whitespace-nowrap">
                        <button type="button" className="text-xs text-teal-700 mr-2" onClick={() => respond(`/vendor/job-work-orders/${row.id}/response`, 'accept')}>Accept</button>
                        <button type="button" className="text-xs text-red-700 mr-2" onClick={() => respond(`/vendor/job-work-orders/${row.id}/response`, 'reject')}>Reject</button>
                        <button type="button" className="text-xs text-blue-700" onClick={() => sendDelivery('jwo', row.id)}>Send qty</button>
                      </td>
                    </tr>
                  ))}
                  {!jwos.isLoading && (jwos.data || []).length === 0 && (
                    <tr><td colSpan={6} className="px-4 py-6 text-center text-gray-400">No job work orders for this processor.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}

        <section className="bg-white border rounded-xl overflow-hidden">
          <h2 className="px-4 py-2.5 font-semibold text-sm text-[#002B5B] border-b">Material issued to you</h2>
          <div className="divide-y">
            {(issues.data || []).length === 0 && <p className="px-4 py-6 text-sm text-gray-400 text-center">No material issue notes match this vendor name.</p>}
            {(issues.data || []).map(note => (
              <div key={note.id} className="px-4 py-3">
                <p className="text-sm font-medium">{note.min_number} · {note.min_date}</p>
                <ul className="mt-1 space-y-1">
                  {(note.lines || []).map(line => (
                    <li key={line.id} className="flex flex-wrap items-center gap-2 text-xs text-gray-600">
                      <span>{line.material_code} · issued {line.issue_qty ?? 0}</span>
                      <span>acknowledged {line.latest_ack?.received_qty ?? '—'}</span>
                      <button type="button" className="text-teal-700" onClick={() => ackLine(note.id, line)}>Confirm received</button>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        <section className="bg-white border rounded-xl overflow-hidden">
          <h2 className="px-4 py-2.5 font-semibold text-sm text-[#002B5B] border-b">Production job orders matched by vendor name</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-500 text-xs">
                <tr>
                  <th className="text-left px-4 py-2">JO</th>
                  <th className="text-left px-4 py-2">SKU</th>
                  <th className="text-left px-4 py-2">Process</th>
                  <th className="text-right px-4 py-2">Planned</th>
                  <th className="text-right px-4 py-2">Issued</th>
                  <th className="text-right px-4 py-2">Received</th>
                  <th className="text-right px-4 py-2">Balance</th>
                  <th className="text-left px-4 py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {(jos.data || []).map(row => (
                  <tr key={row.id} className="border-t">
                    <td className="px-4 py-2 font-medium">{row.jo_number}</td>
                    <td className="px-4 py-2">{row.sku || '—'}</td>
                    <td className="px-4 py-2">{row.process || '—'}</td>
                    <td className="px-4 py-2 text-right">{row.planned_qty ?? 0}</td>
                    <td className="px-4 py-2 text-right">{row.issued_qty ?? 0}</td>
                    <td className="px-4 py-2 text-right">{row.received_qty ?? 0}</td>
                    <td className="px-4 py-2 text-right">{row.balance_qty ?? 0}</td>
                    <td className="px-4 py-2">{row.status}</td>
                  </tr>
                ))}
                {!jos.isLoading && (jos.data || []).length === 0 && (
                  <tr><td colSpan={8} className="px-4 py-6 text-center text-gray-400">No production job orders use this vendor name.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  )
}
