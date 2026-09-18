import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'
import { mayAccessErpAdmin, useAuth } from '../store/auth'

/** Verify/Unverify is Accounts/Admin/Audit — not granted by Production or Purchase module access. */
function canDocumentVerify(role: string, user: Parameters<typeof mayAccessErpAdmin>[0]): boolean {
  if (mayAccessErpAdmin(user) && /^(Super Admin|Admin|Sir)$/i.test(String(user?.role || role))) {
    return true
  }
  return /accounts|account|finance|audit|auditor/i.test(role)
}

const DOC_TYPES = ['PO', 'JWO', 'GRN', 'MIN', 'JO', 'GIN', 'JO_ISSUE', 'JO_RECEIVE'] as const

export default function DocumentAudit() {
  const { user } = useAuth()
  const role = String(user?.role || '')
  const canVerify = canDocumentVerify(role, user)
  const canForce = mayAccessErpAdmin(user) && /^(Super Admin|Admin|Sir)$/i.test(role)
  const qc = useQueryClient()
  const [filters, setFilters] = useState({
    audit_status: 'Pending',
    doc_type: '',
    date_from: '',
    date_to: '',
    search: '',
  })
  const [unverifyModal, setUnverifyModal] = useState<null | { doc_type: string; doc_id: number; doc_number: string }>(null)
  const [reason, setReason] = useState('')
  const [force, setForce] = useState(false)
  const [detail, setDetail] = useState<any>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  const params = useMemo(() => {
    const p: Record<string, string | number> = { limit: 300 }
    if (filters.audit_status) p.audit_status = filters.audit_status
    if (filters.doc_type) p.doc_type = filters.doc_type
    if (filters.date_from) p.date_from = filters.date_from
    if (filters.date_to) p.date_to = filters.date_to
    if (filters.search.trim()) p.search = filters.search.trim()
    return p
  }, [filters])

  const { data, isFetching, isLoading } = useQuery({
    queryKey: ['document-audit', params],
    queryFn: () => api.get('/document-audit', { params }).then(r => r.data),
    staleTime: 15_000,
  })

  const verifyMut = useMutation({
    mutationFn: ({ doc_type, doc_id }: { doc_type: string; doc_id: number }) =>
      api.post(`/document-audit/${doc_type}/${doc_id}/verify`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['document-audit'] })
      setDetail(null)
    },
    onError: (e: any) => alert(e?.response?.data?.detail || 'Verify failed'),
  })

  const unverifyMut = useMutation({
    mutationFn: ({ doc_type, doc_id, reason, force }: { doc_type: string; doc_id: number; reason: string; force: boolean }) =>
      api.post(`/document-audit/${doc_type}/${doc_id}/unverify`, { reason, force }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['document-audit'] })
      setUnverifyModal(null)
      setReason('')
      setForce(false)
      setDetail(null)
    },
    onError: (e: any) => alert(e?.response?.data?.detail || 'Unverify failed'),
  })

  const backfillMut = useMutation({
    mutationFn: () => api.post('/document-audit/backfill', null, { params: { limit_per_type: 8000 } }).then(r => r.data),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ['document-audit'] })
      alert(`Backfill complete. New enrollments: ${res?.total_new ?? 0}\n${JSON.stringify(res, null, 2)}`)
    },
    onError: (e: any) => alert(e?.response?.data?.detail || 'Backfill failed'),
  })

  const rows = data?.rows || []
  const set = (k: string, v: string) => setFilters(f => ({ ...f, [k]: v }))

  const openDetail = async (row: any) => {
    setDetailLoading(true)
    setDetail({ ...row, events: [], dependency_blockers: [], _loading: true })
    try {
      // Fast trail: events only (no cross-DB blockers)
      const { data: d } = await api.get(`/document-audit/${row.doc_type}/${row.doc_id}`, {
        params: { include_blockers: 0 },
      })
      setDetail(d)
      // Lazy blockers (optional, for Unverify guidance)
      void api.get(`/document-audit/${row.doc_type}/${row.doc_id}/blockers`).then(r => {
        setDetail((prev: any) => prev && prev.doc_id === row.doc_id && prev.doc_type === row.doc_type
          ? { ...prev, dependency_blockers: r.data?.dependency_blockers || [] }
          : prev)
      }).catch(() => undefined)
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Could not load trail')
      setDetail(null)
    } finally {
      setDetailLoading(false)
    }
  }

  return (
    <div className="p-4 md:p-6 space-y-4 max-w-7xl mx-auto">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-[#002B5B]">Document Verification (Accounts)</h1>
          <p className="text-xs text-gray-500 max-w-3xl mt-1">
            Daily audit of ERP documents. Flow: Create → Edit while Pending → Verify (locks edit) →
            Unverify with reason → Edit → Re-verify. Downstream Receive/Issue/WIP chains require Admin force-unverify.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs items-center">
          <span className="px-2 py-1 rounded bg-amber-100 text-amber-900 font-semibold">Pending {data?.pending ?? '—'}</span>
          <span className="px-2 py-1 rounded bg-emerald-100 text-emerald-900 font-semibold">Verified {data?.verified ?? '—'}</span>
          {(canVerify || canForce) && (
            <button
              type="button"
              disabled={backfillMut.isPending}
              onClick={() => {
                if (!window.confirm('Enroll all existing PO / JWO / GRN / MIN / JO / GIN / Issue / Receive into Doc Verify?')) return
                backfillMut.mutate()
              }}
              className="px-2 py-1 rounded border border-sky-300 bg-sky-50 text-sky-900 font-medium disabled:opacity-50"
            >
              {backfillMut.isPending ? 'Backfilling…' : 'Sync / Backfill docs'}
            </button>
          )}
        </div>
      </div>

      <div className="bg-white border rounded-xl p-3 grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
        <label className="block">
          <span className="text-gray-500">Status</span>
          <select value={filters.audit_status} onChange={e => set('audit_status', e.target.value)} className="mt-0.5 w-full border rounded px-2 py-1">
            <option value="">All</option>
            <option value="Pending">Pending</option>
            <option value="Verified">Verified</option>
          </select>
        </label>
        <label className="block">
          <span className="text-gray-500">Doc type</span>
          <select value={filters.doc_type} onChange={e => set('doc_type', e.target.value)} className="mt-0.5 w-full border rounded px-2 py-1">
            <option value="">All</option>
            {DOC_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label className="block">
          <span className="text-gray-500">From (optional)</span>
          <input type="date" value={filters.date_from} onChange={e => set('date_from', e.target.value)} className="mt-0.5 w-full border rounded px-2 py-1" />
        </label>
        <label className="block">
          <span className="text-gray-500">To (optional)</span>
          <input type="date" value={filters.date_to} onChange={e => set('date_to', e.target.value)} className="mt-0.5 w-full border rounded px-2 py-1" />
        </label>
        <label className="block">
          <span className="text-gray-500">Search</span>
          <input value={filters.search} onChange={e => set('search', e.target.value)} placeholder="Doc / SO / party" className="mt-0.5 w-full border rounded px-2 py-1" />
        </label>
      </div>

      <div className="bg-white border rounded-xl overflow-auto">
        <div className="px-3 py-2 text-xs text-gray-500">
          {isLoading || isFetching ? 'Loading…' : `${Number(data?.total || 0).toLocaleString()} documents`}
          {!filters.date_from && !filters.date_to ? ' · showing all dates (Pending filter)' : ''}
        </div>
        <table className="w-full text-xs">
          <thead className="bg-gray-50 text-gray-500 uppercase">
            <tr>
              {['Type', 'Doc No', 'Date', 'SO Ref', 'Party', 'Process', 'Status', 'Verified', 'Actions'].map(h => (
                <th key={h} className="text-left px-3 py-2 whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r: any) => (
              <tr key={`${r.doc_type}-${r.doc_id}`} className="border-t hover:bg-slate-50">
                <td className="px-3 py-1.5 font-semibold">{r.doc_type}</td>
                <td className="px-3 py-1.5 font-mono text-[#002B5B]">{r.doc_number || r.doc_id}</td>
                <td className="px-3 py-1.5">{r.doc_date || (r.created_at || '').slice(0, 10)}</td>
                <td className="px-3 py-1.5">{r.so_reference || '—'}</td>
                <td className="px-3 py-1.5">{r.party_name || '—'}</td>
                <td className="px-3 py-1.5">{r.process_name || '—'}</td>
                <td className="px-3 py-1.5">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${r.audit_status === 'Verified' ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-900'}`}>
                    {r.audit_status}
                  </span>
                </td>
                <td className="px-3 py-1.5 text-[10px] text-gray-500">
                  {r.verified_by ? `${r.verified_by} · ${r.verified_at}` : '—'}
                </td>
                <td className="px-3 py-1.5">
                  <div className="flex flex-wrap gap-1">
                    <button type="button" className="text-[10px] px-2 py-0.5 border rounded" onClick={() => void openDetail(r)}>Trail</button>
                    {canVerify && r.audit_status !== 'Verified' && (
                      <button type="button" className="text-[10px] px-2 py-0.5 bg-emerald-600 text-white rounded disabled:opacity-50"
                        disabled={verifyMut.isPending}
                        onClick={() => verifyMut.mutate({ doc_type: r.doc_type, doc_id: r.doc_id })}>Verify</button>
                    )}
                    {canVerify && r.audit_status === 'Verified' && (
                      <button type="button" className="text-[10px] px-2 py-0.5 bg-amber-600 text-white rounded"
                        onClick={() => { setUnverifyModal({ doc_type: r.doc_type, doc_id: r.doc_id, doc_number: r.doc_number }); setReason(''); setForce(false) }}>
                        Unverify
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
            {rows.length === 0 && !isLoading && (
              <tr>
                <td colSpan={9} className="text-center text-gray-400 py-10">
                  No documents in this filter. Click <b>Sync / Backfill docs</b> to enroll existing PO/JWO/GRN/MIN/JO/GIN/Issue/Receive,
                  or clear the date filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {detail && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4" onClick={() => setDetail(null)}>
          <div className="bg-white rounded-xl max-w-lg w-full p-4 space-y-3" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-start">
              <div>
                <h3 className="font-semibold text-gray-900">{detail.doc_type} {detail.doc_number}</h3>
                <p className="text-xs text-gray-500">Status: {detail.audit_status}{detailLoading ? ' · loading…' : ''}</p>
              </div>
              <button type="button" onClick={() => setDetail(null)}>✕</button>
            </div>
            {(detail.dependency_blockers || []).length > 0 && (
              <div className="text-xs bg-rose-50 text-rose-800 border border-rose-100 rounded p-2">
                <b>Downstream:</b> {(detail.dependency_blockers as string[]).join(' · ')}
              </div>
            )}
            <div className="max-h-64 overflow-y-auto text-xs space-y-1">
              {(detail.events || []).length === 0 && <p className="text-gray-400">No events yet.</p>}
              {(detail.events || []).map((ev: any) => (
                <div key={ev.id} className="border-b border-gray-100 py-1">
                  <b>{ev.event_type}</b> · {ev.actor || '—'} · {ev.event_at}
                  {ev.reason ? <div className="text-amber-800">Reason: {ev.reason}</div> : null}
                  {ev.detail ? <div className="text-gray-500">{ev.detail}</div> : null}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {unverifyModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4" onClick={() => setUnverifyModal(null)}>
          <div className="bg-white rounded-xl max-w-md w-full p-4 space-y-3" onClick={e => e.stopPropagation()}>
            <h3 className="font-semibold">Unverify {unverifyModal.doc_type} {unverifyModal.doc_number}</h3>
            <p className="text-xs text-gray-500">Mandatory reason. Document returns to Pending and becomes editable (unless downstream force is required).</p>
            <textarea value={reason} onChange={e => setReason(e.target.value)} rows={3} className="w-full border rounded-lg px-2 py-1.5 text-sm" placeholder="Reason for unverify…" />
            {canForce && (
              <label className="flex items-center gap-2 text-xs text-rose-800">
                <input type="checkbox" checked={force} onChange={e => setForce(e.target.checked)} />
                Admin force (downstream dependencies present)
              </label>
            )}
            <div className="flex justify-end gap-2">
              <button type="button" className="px-3 py-1.5 text-xs border rounded" onClick={() => setUnverifyModal(null)}>Cancel</button>
              <button type="button" disabled={reason.trim().length < 3 || unverifyMut.isPending}
                className="px-3 py-1.5 text-xs bg-amber-600 text-white rounded disabled:opacity-40"
                onClick={() => unverifyMut.mutate({ ...unverifyModal, reason: reason.trim(), force })}>
                Unverify
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
