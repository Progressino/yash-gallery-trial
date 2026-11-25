import { useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'
import { PRIORITIES, priorityStyle } from './hrmConstants'

/** Cross-department assignment workflow UI: tracker, acceptance gate, HOD review, helpers. */

export type AssignExtra = { assign_remarks: string; attachment_links_text: string; voice_note_url: string }
export const EMPTY_ASSIGN_EXTRA: AssignExtra = { assign_remarks: '', attachment_links_text: '', voice_note_url: '' }

export function assignExtraPayload(x: AssignExtra) {
  const links = x.attachment_links_text.split(/[\s,]+/).map(s => s.trim()).filter(Boolean)
  return {
    assign_remarks: x.assign_remarks.trim(),
    attachment_links: links,
    voice_note_url: x.voice_note_url.trim(),
  }
}

export const WF_LABEL: Record<string, string> = {
  pending_hod_review: 'Pending HOD Review',
  clarification_required: 'Explanation Required',
  pending_employee_acceptance: 'Pending Acceptance',
  accepted: 'Accepted',
  on_hold: 'On Hold',
  cancelled: 'Cancelled',
  rejected_by_hod: 'Rejected by HOD',
}

const WF_STYLE: Record<string, string> = {
  pending_hod_review: 'bg-amber-100 text-amber-800',
  clarification_required: 'bg-purple-100 text-purple-800',
  pending_employee_acceptance: 'bg-blue-100 text-blue-800',
  accepted: 'bg-green-100 text-green-800',
  on_hold: 'bg-orange-100 text-orange-800',
  cancelled: 'bg-gray-200 text-gray-600',
  rejected_by_hod: 'bg-red-100 text-red-800',
}

export function WorkflowBadge({ status }: { status?: string }) {
  if (!status) return null
  return (
    <span className={`text-[11px] px-2 py-0.5 rounded-full font-medium whitespace-nowrap ${WF_STYLE[status] || 'bg-gray-100 text-gray-600'}`}>
      {WF_LABEL[status] || status}
    </span>
  )
}

export function AttachmentLinks({ links, voice }: { links?: string[]; voice?: string }) {
  const list = links || []
  if (!list.length && !voice) return <span className="text-gray-400">—</span>
  return (
    <span className="flex flex-col gap-0.5">
      {list.map((u, i) => (
        <a key={u} href={u} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-700 underline break-all">
          📎 Attachment {list.length > 1 ? i + 1 : ''}
        </a>
      ))}
      {voice && (
        <a href={voice} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-700 underline">🎙 Voice note</a>
      )}
    </span>
  )
}

/** Remarks / attachment links / voice note link on the Assign forms. */
export function AssignExtraFields({ value, onChange }: { value: AssignExtra; onChange: (v: AssignExtra) => void }) {
  return (
    <div className="col-span-2 md:col-span-3 grid grid-cols-1 md:grid-cols-3 gap-3 border-t pt-3">
      <div>
        <label className="text-xs text-gray-500">Remarks for the assignee / HOD</label>
        <textarea rows={2} value={value.assign_remarks} onChange={e => onChange({ ...value, assign_remarks: e.target.value })}
          className="w-full border rounded px-2 py-1.5 text-sm mt-1" placeholder="Context, expected output…" />
      </div>
      <div>
        <label className="text-xs text-gray-500">Attachment links (one per line)</label>
        <textarea rows={2} value={value.attachment_links_text} onChange={e => onChange({ ...value, attachment_links_text: e.target.value })}
          className="w-full border rounded px-2 py-1.5 text-sm mt-1" placeholder="https://…" />
      </div>
      <div>
        <label className="text-xs text-gray-500">Voice note link (optional)</label>
        <input value={value.voice_note_url} onChange={e => onChange({ ...value, voice_note_url: e.target.value })}
          className="w-full border rounded px-2 py-1.5 text-sm mt-1" placeholder="https://…" />
      </div>
    </div>
  )
}

export type AssignView = 'to_me' | 'by_me'

export function AssignViewToggle({ value, onChange }: { value: AssignView; onChange: (v: AssignView) => void }) {
  const btn = (v: AssignView, label: string) => (
    <button type="button" onClick={() => onChange(v)}
      className={`px-3 py-1.5 text-xs font-medium rounded-md ${value === v ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
      {label}
    </button>
  )
  return <div className="inline-flex gap-1 bg-gray-100 p-1 rounded-lg">{btn('to_me', 'Assigned To Me')}{btn('by_me', 'Assigned By Me')}</div>
}

const errText = (e: any, fallback: string) => e?.response?.data?.detail || fallback

function useRefreshAssignments() {
  const qc = useQueryClient()
  return () => {
    for (const key of ['hrm-assign', 'hrm-pending-approvals', 'hrm-resps', 'hrm-one-time-tasks', 'hrm-my-tasks', 'hrm-employee-check', 'hrm-assigned-by-me']) {
      qc.invalidateQueries({ queryKey: [key] })
    }
  }
}

type Row = any

const ACTION_CFG: Record<string, { label: string; endpoint: string; required: boolean; prompt: string; tone: string }> = {
  forward: { label: 'Forward to employee', endpoint: 'hod-forward', required: false, prompt: 'Note for the employee (optional)', tone: 'bg-green-600' },
  reject: { label: 'Reject', endpoint: 'hod-reject', required: true, prompt: 'Reason for rejection *', tone: 'bg-red-600' },
  hodClarify: { label: 'Ask assigner to explain', endpoint: 'clarify', required: true, prompt: 'What do you need explained? *', tone: 'bg-purple-600' },
  clarify: { label: 'Explanation required', endpoint: 'clarify', required: true, prompt: 'What do you need explained? *', tone: 'bg-purple-600' },
  reply: { label: 'Send explanation', endpoint: 'clarify-reply', required: true, prompt: 'Your explanation *', tone: 'bg-indigo-600' },
  remind: { label: 'Send reminder', endpoint: 'remind', required: false, prompt: 'Message (optional)', tone: 'bg-amber-600' },
  cancel: { label: 'Cancel assignment', endpoint: 'cancel', required: true, prompt: 'Reason for cancelling *', tone: 'bg-gray-700' },
  accept: { label: 'Accept', endpoint: 'accept', required: false, prompt: 'Remarks (optional)', tone: 'bg-green-600' },
}

/** Buttons + remarks dialog for one assignment row. */
export function AssignmentActions({ row, onChanged, compact }: { row: Row; onChanged?: () => void; compact?: boolean }) {
  const refresh = useRefreshAssignments()
  const [dlg, setDlg] = useState<string | null>(null)
  const [remarks, setRemarks] = useState('')
  const [busy, setBusy] = useState(false)
  const [history, setHistory] = useState<any[] | null>(null)
  const a = row.actions || {}

  const submit = async () => {
    if (!dlg) return
    const cfg = ACTION_CFG[dlg]
    if (cfg.required && !remarks.trim()) { alert('Please enter the details first'); return }
    setBusy(true)
    try {
      const r = await api.post(`/hrm/assignments/${cfg.endpoint}`, { entity_type: row.entity_type, entity_id: row.id, remarks })
      if (dlg === 'remind') alert('Reminder sent')
      setDlg(null); setRemarks('')
      refresh(); onChanged?.()
      return r
    } catch (e: any) {
      alert(errText(e, 'Action failed'))
    } finally { setBusy(false) }
  }

  const loadHistory = async () => {
    try {
      const r = await api.get('/hrm/assignments/history', { params: { entity_type: row.entity_type, entity_id: row.id } })
      setHistory(r.data.events || [])
    } catch (e: any) { alert(errText(e, 'Could not load history')) }
  }

  const btn = (key: string, text: string, tone: string) => (
    <button key={key} type="button" onClick={() => { setRemarks(''); setDlg(key) }}
      className={`text-[11px] px-2 py-1 rounded text-white whitespace-nowrap ${tone}`}>{text}</button>
  )

  return (
    <div className={`flex ${compact ? 'flex-wrap' : 'flex-wrap'} gap-1 items-center`}>
      {a.can_accept && btn('accept', 'Accept', 'bg-green-600')}
      {a.can_clarify && btn('clarify', 'Explanation Required', 'bg-purple-600')}
      {a.can_forward && btn('forward', 'Forward', 'bg-green-600')}
      {a.can_reject && btn('reject', 'Reject', 'bg-red-600')}
      {a.can_hod_clarify && btn('hodClarify', 'Clarify', 'bg-purple-600')}
      {a.can_reply && btn('reply', 'Reply', 'bg-indigo-600')}
      {a.can_remind && btn('remind', 'Remind', 'bg-amber-600')}
      {a.can_cancel && btn('cancel', 'Cancel', 'bg-gray-600')}
      <button type="button" onClick={loadHistory} className="text-[11px] px-2 py-1 rounded border text-gray-600 whitespace-nowrap">History</button>

      {dlg && (
        <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-md p-4 space-y-3">
            <h4 className="font-semibold text-gray-800">{ACTION_CFG[dlg].label}</h4>
            <p className="text-xs text-gray-500">{row.title} · {row.employee_name}</p>
            <textarea rows={3} autoFocus value={remarks} onChange={e => setRemarks(e.target.value)}
              placeholder={ACTION_CFG[dlg].prompt} className="w-full border rounded px-2 py-1.5 text-sm" />
            <div className="flex justify-end gap-2">
              <button type="button" onClick={() => setDlg(null)} className="px-3 py-1.5 border rounded text-sm">Back</button>
              <button type="button" disabled={busy} onClick={submit}
                className={`px-3 py-1.5 rounded text-sm text-white disabled:opacity-50 ${ACTION_CFG[dlg].tone}`}>
                {busy ? 'Saving…' : ACTION_CFG[dlg].label}
              </button>
            </div>
          </div>
        </div>
      )}

      {history && (
        <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-lg p-4 space-y-3 max-h-[80vh] overflow-y-auto">
            <h4 className="font-semibold text-gray-800">History — {row.title}</h4>
            <ol className="space-y-2">
              {history.map(ev => (
                <li key={ev.id} className="text-xs border-l-2 border-blue-300 pl-2">
                  <b>{String(ev.event_type).replace(/_/g, ' ')}</b> · {ev.actor_name || '—'} · <span className="text-gray-400">{ev.created_at}</span>
                  {ev.remarks && <p className="text-gray-700 mt-0.5">{ev.remarks}</p>}
                </li>
              ))}
              {history.length === 0 && <li className="text-xs text-gray-400">No history yet</li>}
            </ol>
            <div className="text-right"><button type="button" onClick={() => setHistory(null)} className="px-3 py-1.5 border rounded text-sm">Close</button></div>
          </div>
        </div>
      )}
    </div>
  )
}

/** Latest clarification question from assignment history (for mandatory popups). */
function latestClarificationQuestion(events: any[]): string {
  for (const ev of [...(events || [])].reverse()) {
    const t = String(ev.event_type || '')
    if ((t.includes('clarif') || t.includes('explanation') || t === 'clarify') && ev.remarks) {
      return String(ev.remarks)
    }
  }
  return ''
}

/** Blocking popup for HOD review / assigner clarification / employee acceptance. */
export function AssignmentMandatoryActionGate({ employeeId }: { employeeId?: number | null }) {
  const refresh = useRefreshAssignments()
  const [busy, setBusy] = useState(false)
  const [remarks, setRemarks] = useState('')
  const [history, setHistory] = useState<any[] | null>(null)
  const [mode, setMode] = useState<'act' | 'clarify' | 'reply' | 'reject'>('act')

  const { data: toMe } = useQuery({
    queryKey: ['hrm-assign', 'gate-to-me', employeeId],
    queryFn: () => api.get('/hrm/assignments/tracker', { params: { view: 'assigned_to_me' } }).then(r => r.data),
    enabled: !!employeeId,
    refetchInterval: 45_000,
  })
  const { data: byMe } = useQuery({
    queryKey: ['hrm-assign', 'gate-by-me', employeeId],
    queryFn: () => api.get('/hrm/assignments/tracker', { params: { view: 'assigned_by_me', status: 'clarification_required' } }).then(r => r.data),
    enabled: !!employeeId,
    refetchInterval: 45_000,
  })
  const { data: hod } = useQuery({
    queryKey: ['hrm-assign', 'gate-hod', employeeId],
    queryFn: () => api.get('/hrm/assignments/tracker', { params: { view: 'hod_inbox' } }).then(r => r.data),
    enabled: !!employeeId,
    refetchInterval: 45_000,
  })

  const candidates: Row[] = []
  for (const r of (toMe?.rows || []) as Row[]) {
    if (r.workflow_status === 'pending_employee_acceptance' || r.actions?.can_accept) candidates.push({ ...r, _gate: 'accept' })
  }
  for (const r of (byMe?.rows || []) as Row[]) {
    if (r.actions?.can_reply || r.workflow_status === 'clarification_required') candidates.push({ ...r, _gate: 'reply' })
  }
  for (const r of (hod?.rows || []) as Row[]) {
    if (r.actions?.can_forward || r.workflow_status === 'pending_hod_review') candidates.push({ ...r, _gate: 'hod' })
  }
  const row = candidates[0]
  const { data: histData } = useQuery({
    queryKey: ['hrm-assign', 'gate-hist', row?.entity_type, row?.id],
    queryFn: () => api.get('/hrm/assignments/history', { params: { entity_type: row.entity_type, entity_id: row.id } }).then(r => r.data),
    enabled: !!row,
  })
  const question = latestClarificationQuestion(histData?.events || [])

  if (!row) return null

  const run = async (endpoint: string, required = false) => {
    if (required && !remarks.trim()) { alert('Please enter the details first'); return }
    setBusy(true)
    try {
      await api.post(`/hrm/assignments/${endpoint}`, { entity_type: row.entity_type, entity_id: row.id, remarks })
      setRemarks(''); setMode('act'); setHistory(null)
      refresh()
    } catch (e: any) {
      alert(errText(e, 'Action failed'))
    } finally { setBusy(false) }
  }

  const title =
    row._gate === 'hod' ? 'Assignment needs your review'
      : row._gate === 'reply' ? 'Clarification requested — please reply'
        : 'New work assigned to you'

  return (
    <div className="fixed inset-0 z-[70] bg-black/60 flex items-center justify-center p-4" role="dialog" aria-modal="true">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg p-5 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-base font-bold text-[#002B5B]">{title}</h3>
          {candidates.length > 1 && <span className="text-[11px] text-gray-500">1 of {candidates.length}</span>}
        </div>
        <div className="bg-blue-50 border border-blue-100 rounded-lg p-3 space-y-1 text-sm">
          <p className="font-semibold text-gray-800">{row.title}</p>
          {row.description && <p className="text-xs text-gray-600">{row.description}</p>}
          <p className="text-xs text-gray-600">
            {row.type} · For <b>{row.employee_name}</b> · Assigned by <b>{row.assigned_by_name || '—'}</b>
            {row.hod_name ? <> · HOD <b>{row.hod_name}</b></> : null}
          </p>
          <p className="text-xs text-gray-600">Deadline: <b>{row.deadline || '—'}</b> · Priority: {row.priority}</p>
          {row.assign_remarks && <p className="text-xs text-gray-700">“{row.assign_remarks}”</p>}
          {question && (
            <div className="mt-2 bg-purple-50 border border-purple-200 rounded p-2 text-xs text-purple-900">
              <b>Question:</b> {question}
            </div>
          )}
          <AttachmentLinks links={row.attachment_links} voice={row.voice_note_url} />
        </div>

        {mode === 'clarify' || mode === 'reply' || mode === 'reject' ? (
          <div className="space-y-2">
            <textarea rows={3} autoFocus value={remarks} onChange={e => setRemarks(e.target.value)}
              placeholder={mode === 'reject' ? 'Reason for rejection *' : mode === 'reply' ? 'Your explanation *' : 'What needs to be explained? *'}
              className="w-full border rounded px-2 py-1.5 text-sm" />
            <div className="flex justify-end gap-2">
              <button type="button" onClick={() => setMode('act')} className="px-3 py-1.5 border rounded text-sm">Back</button>
              <button type="button" disabled={busy} onClick={() => run(
                mode === 'reject' ? 'hod-reject' : mode === 'reply' ? 'clarify-reply' : 'clarify', true,
              )} className={`px-3 py-1.5 rounded text-sm text-white disabled:opacity-50 ${mode === 'reject' ? 'bg-red-600' : 'bg-purple-600'}`}>
                {busy ? 'Saving…' : mode === 'reject' ? 'Reject' : mode === 'reply' ? 'Send explanation' : 'Send question'}
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-wrap justify-end gap-2">
            <button type="button" onClick={async () => {
              try {
                const r = await api.get('/hrm/assignments/history', { params: { entity_type: row.entity_type, entity_id: row.id } })
                setHistory(r.data.events || [])
              } catch (e: any) { alert(errText(e, 'Could not load history')) }
            }} className="px-3 py-2 rounded-lg text-sm border text-gray-700">View History</button>
            {row._gate === 'accept' && (
              <>
                <button type="button" onClick={() => setMode('clarify')} className="px-4 py-2 rounded-lg text-sm border border-purple-600 text-purple-700">Explanation Required</button>
                <button type="button" disabled={busy} onClick={() => run('accept')} className="px-4 py-2 rounded-lg text-sm text-white bg-green-600 disabled:opacity-50">{busy ? 'Saving…' : 'Accept'}</button>
              </>
            )}
            {row._gate === 'reply' && (
              <button type="button" onClick={() => setMode('reply')} className="px-4 py-2 rounded-lg text-sm text-white bg-indigo-600">Reply with explanation</button>
            )}
            {row._gate === 'hod' && (
              <>
                <button type="button" onClick={() => setMode('clarify')} className="px-3 py-2 rounded-lg text-sm border border-purple-600 text-purple-700">Ask assigner</button>
                <button type="button" onClick={() => setMode('reject')} className="px-3 py-2 rounded-lg text-sm text-white bg-red-600">Reject</button>
                <button type="button" disabled={busy} onClick={() => run('hod-forward')} className="px-4 py-2 rounded-lg text-sm text-white bg-green-600 disabled:opacity-50">{busy ? 'Saving…' : 'Forward to employee'}</button>
              </>
            )}
          </div>
        )}
        <p className="text-[11px] text-gray-400">This popup stays until you take an action.</p>

        {history && (
          <div className="fixed inset-0 z-[80] bg-black/40 flex items-center justify-center p-4">
            <div className="bg-white rounded-xl shadow-xl w-full max-w-lg p-4 space-y-3 max-h-[80vh] overflow-y-auto">
              <h4 className="font-semibold text-gray-800">History — {row.title}</h4>
              <ol className="space-y-2">
                {history.map(ev => (
                  <li key={ev.id} className="text-xs border-l-2 border-blue-300 pl-2">
                    <b>{String(ev.event_type).replace(/_/g, ' ')}</b> · {ev.actor_name || '—'} · <span className="text-gray-400">{ev.created_at}</span>
                    {ev.remarks && <p className="text-gray-700 mt-0.5">{ev.remarks}</p>}
                  </li>
                ))}
              </ol>
              <div className="text-right"><button type="button" onClick={() => setHistory(null)} className="px-3 py-1.5 border rounded text-sm">Close</button></div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

/** Blocking popup: the assignee must Accept or ask for an explanation before continuing. */
export function AssignmentAcceptanceGate({ employeeId }: { employeeId?: number | null }) {
  const refresh = useRefreshAssignments()
  const [asking, setAsking] = useState(false)
  const [remarks, setRemarks] = useState('')
  const [busy, setBusy] = useState(false)
  const { data } = useQuery({
    queryKey: ['hrm-assign', 'acceptance-gate', employeeId],
    queryFn: () => api.get('/hrm/assignments/tracker', { params: { view: 'assigned_to_me', status: 'pending_employee_acceptance' } }).then(r => r.data),
    enabled: !!employeeId,
    refetchInterval: 60_000,
  })
  const rows: Row[] = data?.rows || []
  const row = rows[0]
  if (!row) return null

  const run = async (endpoint: 'accept' | 'clarify') => {
    if (endpoint === 'clarify' && !remarks.trim()) { alert('Please write what needs to be explained'); return }
    setBusy(true)
    try {
      await api.post(`/hrm/assignments/${endpoint}`, { entity_type: row.entity_type, entity_id: row.id, remarks })
      setAsking(false); setRemarks('')
      refresh()
    } catch (e: any) {
      alert(errText(e, 'Action failed'))
    } finally { setBusy(false) }
  }

  return (
    <div className="fixed inset-0 z-[70] bg-black/60 flex items-center justify-center p-4" role="dialog" aria-modal="true">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-bold text-[#002B5B]">New work assigned to you</h3>
          {rows.length > 1 && <span className="text-[11px] text-gray-500">1 of {rows.length}</span>}
        </div>
        <div className="bg-blue-50 border border-blue-100 rounded-lg p-3 space-y-1 text-sm">
          <p className="font-semibold text-gray-800">{row.title}</p>
          {row.description && <p className="text-xs text-gray-600">{row.description}</p>}
          <p className="text-xs text-gray-600">
            {row.type}{row.frequency ? ` · ${row.frequency}` : ''} · Assigned by <b>{row.assigned_by_name || '—'}</b>
            {row.hod_name ? <> · HOD <b>{row.hod_name}</b></> : null}
          </p>
          <p className="text-xs text-gray-600">
            Deadline: <b>{row.deadline || '—'}</b> · Priority: <span className={`px-1.5 rounded ${priorityStyle(row.priority)}`}>{row.priority}</span>
          </p>
          {row.assign_remarks && <p className="text-xs text-gray-700">“{row.assign_remarks}”</p>}
          <AttachmentLinks links={row.attachment_links} voice={row.voice_note_url} />
        </div>
        {asking ? (
          <div className="space-y-2">
            <textarea rows={3} autoFocus value={remarks} onChange={e => setRemarks(e.target.value)}
              placeholder={`What should ${row.assigned_by_name || 'the assigner'} explain? *`}
              className="w-full border rounded px-2 py-1.5 text-sm" />
            <div className="flex justify-end gap-2">
              <button type="button" onClick={() => setAsking(false)} className="px-3 py-1.5 border rounded text-sm">Back</button>
              <button type="button" disabled={busy} onClick={() => run('clarify')} className="px-3 py-1.5 rounded text-sm text-white bg-purple-600 disabled:opacity-50">
                {busy ? 'Sending…' : 'Send question'}
              </button>
            </div>
          </div>
        ) : (
          <div className="flex justify-end gap-2">
            <button type="button" onClick={() => setAsking(true)} className="px-4 py-2 rounded-lg text-sm border border-purple-600 text-purple-700">
              Explanation Required
            </button>
            <button type="button" disabled={busy} onClick={() => run('accept')} className="px-4 py-2 rounded-lg text-sm text-white bg-green-600 disabled:opacity-50">
              {busy ? 'Saving…' : 'Accept'}
            </button>
          </div>
        )}
        <p className="text-[11px] text-gray-400">You need to respond to continue. Work starts counting once you accept.</p>
      </div>
    </div>
  )
}

/** HOD inbox shown inside Approvals: cross-department work waiting for review. */
export function AssignmentReviewList({ rows }: { rows: Row[] }) {
  if (!rows?.length) return null
  return (
    <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 space-y-2">
      <h4 className="text-sm font-semibold text-amber-900">Assignments waiting for your review ({rows.length})</h4>
      <p className="text-[11px] text-amber-800">Another department assigned work to your team. Forward it to the employee, reject it, or ask the assigner to explain.</p>
      <ul className="space-y-2">
        {rows.map(r => (
          <li key={`${r.entity_type}-${r.id}`} className="bg-white border rounded-lg p-3 flex flex-wrap items-start justify-between gap-2">
            <div className="text-sm min-w-0">
              <p className="font-medium text-gray-800">{r.title} <span className="text-[11px] text-gray-400">({r.type})</span></p>
              <p className="text-xs text-gray-500">
                For <b>{r.employee_name}</b> · from <b>{r.assigned_by_name || '—'}</b> · Deadline {r.deadline || '—'} · {r.priority}
              </p>
              {r.assign_remarks && <p className="text-xs text-gray-600 mt-0.5">“{r.assign_remarks}”</p>}
              <AttachmentLinks links={r.attachment_links} voice={r.voice_note_url} />
            </div>
            <AssignmentActions row={r} />
          </li>
        ))}
      </ul>
    </div>
  )
}

function ColFilter({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <th className="text-left px-2 py-1.5 align-bottom sticky top-0 z-10 bg-gray-50">
      <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="mt-0.5">{children}</div>
    </th>
  )
}

/** Rows the current user assigned (any department) — same data as the list endpoints with assigned_by_me=1. */
export function AssignedByMeTable({ kind, onOpenTracker }: { kind: 'responsibility' | 'task'; onOpenTracker?: () => void }) {
  const { data: rows = [], isLoading } = useQuery({
    queryKey: ['hrm-assigned-by-me', kind],
    queryFn: () => api.get(kind === 'responsibility' ? '/hrm/responsibilities' : '/hrm/one-time-tasks', { params: { assigned_by_me: 1 } }).then(r => r.data),
  })
  const [fTitle, setFTitle] = useState('')
  const [fTo, setFTo] = useState('')
  const [fDept, setFDept] = useState('')
  const [fDeadline, setFDeadline] = useState('')
  const [fPriority, setFPriority] = useState('')
  const [fWorkflow, setFWorkflow] = useState('')
  const [fStatus, setFStatus] = useState('')
  const [fAssignFrom, setFAssignFrom] = useState('')
  const [fAssignTo, setFAssignTo] = useState('')

  const assignedDateOf = (r: Row) =>
    String(r.accepted_at || r.updated_at || r.created_at || '').slice(0, 10)

  const filtered = useMemo(() => {
    return (rows as Row[]).filter(r => {
      if (fTitle && !String(r.title || '').toLowerCase().includes(fTitle.toLowerCase())) return false
      if (fTo && !String(r.employee_name || '').toLowerCase().includes(fTo.toLowerCase())) return false
      if (fDept && !String(r.department_name || '').toLowerCase().includes(fDept.toLowerCase())) return false
      const deadline = kind === 'task' ? String(r.due_date || '') : String(r.frequency || '')
      if (fDeadline && !deadline.toLowerCase().includes(fDeadline.toLowerCase())) return false
      if (fPriority && String(r.priority || 'Medium') !== fPriority) return false
      if (fWorkflow && String(r.workflow_status || '') !== fWorkflow) return false
      const st = kind === 'task' ? String(r.status || '') : String(r.linked_to_employee_name || 'You (assigner)')
      if (fStatus && !st.toLowerCase().includes(fStatus.toLowerCase())) return false
      const ad = assignedDateOf(r)
      if (fAssignFrom && ad && ad < fAssignFrom) return false
      if (fAssignTo && ad && ad > fAssignTo) return false
      return true
    })
  }, [rows, fTitle, fTo, fDept, fDeadline, fPriority, fWorkflow, fStatus, fAssignFrom, fAssignTo, kind])

  const workflowOpts = useMemo(() => {
    const s = new Set<string>()
    ;(rows as Row[]).forEach(r => { if (r.workflow_status) s.add(String(r.workflow_status)) })
    return [...s].sort()
  }, [rows])

  const inp = 'w-full border rounded px-1.5 py-1 text-xs text-gray-700 bg-white'
  return (
    <div className="bg-white rounded-xl border overflow-hidden">
      <div className="px-4 py-2 bg-gray-50 border-b flex items-center justify-between flex-wrap gap-2">
        <p className="text-xs text-gray-600">
          {kind === 'responsibility' ? 'Responsibilities' : 'Tasks'} you assigned to people in any department. Reminders and replies are in the Work Tracker.
        </p>
        <div className="flex items-center gap-2">
          <button type="button" className="text-[11px] text-gray-500 underline"
            onClick={() => { setFTitle(''); setFTo(''); setFDept(''); setFDeadline(''); setFPriority(''); setFWorkflow(''); setFStatus(''); setFAssignFrom(''); setFAssignTo('') }}>
            Clear filters
          </button>
          {onOpenTracker && <button type="button" onClick={onOpenTracker} className="text-xs text-[#002B5B] underline">Open Work Tracker →</button>}
        </div>
      </div>
      <div className="hrm-sticky-scroll max-h-[70vh] overflow-auto">
        <table className="w-full text-sm min-w-[1100px]">
          <thead className="text-gray-400 text-xs uppercase">
            <tr>
              <ColFilter label="Title"><input className={inp} value={fTitle} onChange={e => setFTitle(e.target.value)} placeholder="Search…" /></ColFilter>
              <ColFilter label="Assigned To"><input className={inp} value={fTo} onChange={e => setFTo(e.target.value)} placeholder="Name…" /></ColFilter>
              <ColFilter label="Department"><input className={inp} value={fDept} onChange={e => setFDept(e.target.value)} placeholder="Dept…" /></ColFilter>
              <ColFilter label="Assigned Date">
                <div className="flex gap-0.5">
                  <input type="date" className={inp} value={fAssignFrom} onChange={e => setFAssignFrom(e.target.value)} title="From" />
                  <input type="date" className={inp} value={fAssignTo} onChange={e => setFAssignTo(e.target.value)} title="To" />
                </div>
              </ColFilter>
              <ColFilter label={kind === 'task' ? 'Deadline' : 'Frequency'}>
                <input className={inp} value={fDeadline} onChange={e => setFDeadline(e.target.value)} placeholder={kind === 'task' ? 'Date…' : 'Freq…'} />
              </ColFilter>
              <ColFilter label="Priority">
                <select className={inp} value={fPriority} onChange={e => setFPriority(e.target.value)}>
                  <option value="">All</option>
                  {['Critical', 'High', 'Medium', 'Low'].map(p => <option key={p} value={p}>{p}</option>)}
                </select>
              </ColFilter>
              <ColFilter label="Workflow">
                <select className={inp} value={fWorkflow} onChange={e => setFWorkflow(e.target.value)}>
                  <option value="">All</option>
                  {workflowOpts.map(w => <option key={w} value={w}>{w.replace(/_/g, ' ')}</option>)}
                </select>
              </ColFilter>
              <ColFilter label={kind === 'task' ? 'Task Status' : 'Linked Person'}>
                <input className={inp} value={fStatus} onChange={e => setFStatus(e.target.value)} placeholder="Filter…" />
              </ColFilter>
              <th className="text-left px-2 py-1.5 sticky top-0 z-10 bg-gray-50 text-[10px] font-semibold text-gray-500 uppercase">Attachments</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(r => (
              <tr key={r.id} className="border-t align-top">
                <td className="px-4 py-2"><p className="font-medium text-gray-800">{r.title}</p>{r.description && <p className="text-xs text-gray-400">{r.description}</p>}</td>
                <td className="px-4 py-2">{r.employee_name || '—'}</td>
                <td className="px-4 py-2 text-xs text-gray-600">{r.department_name || '—'}</td>
                <td className="px-4 py-2 text-xs text-gray-600 whitespace-nowrap">{assignedDateOf(r) || '—'}</td>
                <td className="px-4 py-2 text-gray-600">{kind === 'task' ? (r.due_date || '—') : (r.frequency || '—')}</td>
                <td className="px-4 py-2"><span className={`text-xs px-1.5 py-0.5 rounded ${priorityStyle(r.priority || 'Medium')}`}>{r.priority || 'Medium'}</span></td>
                <td className="px-4 py-2"><WorkflowBadge status={r.workflow_status} /></td>
                <td className="px-4 py-2 text-xs text-gray-600">{kind === 'task' ? (r.status || '—') : (r.linked_to_employee_name || 'You (assigner)')}</td>
                <td className="px-4 py-2"><AttachmentLinks links={r.attachment_links_list} voice={r.voice_note_url} /></td>
              </tr>
            ))}
            {!isLoading && filtered.length === 0 && (
              <tr><td colSpan={9} className="px-4 py-6 text-center text-gray-400 text-sm">
                {(rows as Row[]).length === 0
                  ? `You have not assigned any ${kind === 'task' ? 'tasks' : 'responsibilities'} yet`
                  : 'No rows match the current filters'}
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

type TrackerView = 'assigned_to_me' | 'assigned_by_me' | 'hod_inbox'

const CHIPS: { key: string; label: string; tone: string }[] = [
  { key: 'all', label: 'All', tone: 'bg-gray-100 text-gray-700' },
  { key: 'pending_hod_review', label: 'Pending HOD', tone: 'bg-amber-100 text-amber-800' },
  { key: 'clarification_required', label: 'Explanation Required', tone: 'bg-purple-100 text-purple-800' },
  { key: 'pending_employee_acceptance', label: 'Pending Acceptance', tone: 'bg-blue-100 text-blue-800' },
  { key: 'accepted', label: 'Accepted', tone: 'bg-green-100 text-green-800' },
  { key: 'rejected_by_hod', label: 'Rejected', tone: 'bg-red-100 text-red-800' },
  { key: 'cancelled', label: 'Cancelled', tone: 'bg-gray-200 text-gray-600' },
  { key: 'overdue', label: 'Overdue', tone: 'bg-red-100 text-red-800' },
]

export function AssignmentNotifications({ employeeId }: { employeeId?: number | null }) {
  const qc = useQueryClient()
  const [open, setOpen] = useState(false)
  const { data } = useQuery({
    queryKey: ['hrm-assign', 'notifications', employeeId],
    queryFn: () => api.get('/hrm/assignments/notifications', { params: { limit: 30 } }).then(r => r.data),
    enabled: !!employeeId,
    refetchInterval: 60_000,
  })
  const notes: Row[] = data?.notifications || []
  const unread = data?.unread || 0
  return (
    <div className="bg-white border rounded-xl">
      <button type="button" onClick={() => setOpen(o => !o)} className="w-full px-4 py-2 flex items-center justify-between text-sm">
        <span className="font-medium text-gray-700">🔔 Assignment notifications {unread > 0 && <span className="ml-1 px-1.5 rounded-full bg-indigo-600 text-white text-[10px]">{unread}</span>}</span>
        <span className="text-xs text-gray-400">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open && (
        <div className="border-t px-4 py-2 space-y-1 max-h-64 overflow-y-auto">
          {unread > 0 && (
            <button type="button" className="text-xs text-indigo-700 underline"
              onClick={() => api.post('/hrm/assignments/notifications/read').then(() => qc.invalidateQueries({ queryKey: ['hrm-assign'] }))}>
              Mark all read
            </button>
          )}
          {notes.map(n => (
            <div key={n.id} className={`text-xs border-l-2 pl-2 py-0.5 ${Number(n.is_read) ? 'border-gray-200 text-gray-500' : 'border-indigo-500 text-gray-800'}`}>
              <b>{n.title}</b> — {n.message} <span className="text-gray-400">· {n.created_at}</span>
            </div>
          ))}
          {notes.length === 0 && <p className="text-xs text-gray-400">No notifications</p>}
        </div>
      )}
    </div>
  )
}

export function WorkTrackerTab({ employeeId, showHodInbox, depts }: { employeeId?: number | null; showHodInbox: boolean; depts: any[] }) {
  const [view, setView] = useState<TrackerView>('assigned_to_me')
  const [status, setStatus] = useState('all')
  const [entityType, setEntityType] = useState('')
  const [priority, setPriority] = useState('')
  const [deptId, setDeptId] = useState('')
  const [q, setQ] = useState('')

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['hrm-assign', 'tracker', view, status, entityType, priority, deptId, q],
    queryFn: () => api.get('/hrm/assignments/tracker', {
      params: {
        view,
        ...(status !== 'all' ? { status } : {}),
        ...(entityType ? { entity_type: entityType } : {}),
        ...(priority ? { priority } : {}),
        ...(deptId ? { department_id: deptId } : {}),
        ...(q.trim() ? { q: q.trim() } : {}),
      },
    }).then(r => r.data),
    refetchInterval: 60_000,
  })
  const rows: Row[] = data?.rows || []
  const summary = data?.summary || {}
  const count = (k: string) => (k === 'all' ? summary.total : summary[k]) || 0

  const tabBtn = (v: TrackerView, label: string) => (
    <button type="button" onClick={() => { setView(v); setStatus('all') }}
      className={`px-3 py-1.5 text-xs font-medium rounded-md ${view === v ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>{label}</button>
  )

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="font-semibold text-gray-800">Work Tracker</h3>
          <p className="text-xs text-gray-500">Track work assigned across departments — who has it, what it is waiting on and what is overdue.</p>
          <p className="text-[11px] text-gray-400 mt-1">
            Status legend: Pending HOD Review → Explanation Required / Pending Acceptance → Accepted (then work status: Pending / In Progress / Done / Approved…).
            Pending With shows who must act next (Employee HOD, Employee, Assigner, Approver).
          </p>
        </div>
        <div className="inline-flex gap-1 bg-gray-100 p-1 rounded-lg">
          {tabBtn('assigned_to_me', 'Assigned To Me')}
          {tabBtn('assigned_by_me', 'Assigned By Me')}
          {showHodInbox && tabBtn('hod_inbox', 'HOD Inbox')}
        </div>
      </div>

      <AssignmentNotifications employeeId={employeeId} />

      <div className="flex flex-wrap gap-2">
        {CHIPS.map(c => (
          <button key={c.key} type="button" onClick={() => setStatus(c.key)}
            className={`text-xs px-3 py-1 rounded-full font-medium border ${c.tone} ${status === c.key ? 'ring-2 ring-[#002B5B]' : 'border-transparent'}`}>
            {c.label} <b>{count(c.key)}</b>
          </button>
        ))}
      </div>

      <div className="bg-white rounded-xl border p-3 flex flex-wrap gap-2 text-xs">
        <input placeholder="Search title / employee / assigner" value={q} onChange={e => setQ(e.target.value)} className="border rounded px-2 py-1 w-64" />
        <select value={entityType} onChange={e => setEntityType(e.target.value)} className="border rounded px-2 py-1">
          <option value="">Type: all</option>
          <option value="responsibility">Responsibility</option>
          <option value="task">Task</option>
        </select>
        <select value={priority} onChange={e => setPriority(e.target.value)} className="border rounded px-2 py-1">
          <option value="">Priority: all</option>
          {PRIORITIES.map(p => <option key={p}>{p}</option>)}
        </select>
        <select value={deptId} onChange={e => setDeptId(e.target.value)} className="border rounded px-2 py-1">
          <option value="">Department: all</option>
          {depts.map((d: any) => <option key={d.id} value={d.id}>{d.name}</option>)}
        </select>
        <button type="button" onClick={() => refetch()} className="px-2 py-1 border rounded text-gray-600">Refresh</button>
      </div>

      <div className="bg-white rounded-xl border overflow-x-auto">
        <table className="w-full text-sm min-w-[1250px]">
          <thead className="text-gray-400 text-xs uppercase bg-gray-50">
            <tr>
              {['Employee', 'Dept', 'Title', 'Type', 'Assigned By', 'Assign Date', 'Deadline', 'Priority', 'Status', 'Pending With', 'Working Time', 'Due / Overdue', 'Actions'].map(h => (
                <th key={h} className="text-left px-3 py-2 whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={`${r.entity_type}-${r.id}`} className="border-t align-top hover:bg-gray-50">
                <td className="px-3 py-2 whitespace-nowrap font-medium">{r.employee_name}</td>
                <td className="px-3 py-2 text-xs text-gray-600 whitespace-nowrap">{r.department_name || '—'}</td>
                <td className="px-3 py-2 min-w-[200px]">
                  <p className="font-medium text-gray-800">{r.title}</p>
                  {r.assign_remarks && <p className="text-[11px] text-gray-500">“{r.assign_remarks}”</p>}
                  {r.notes && <p className="text-[11px] text-gray-600 mt-0.5">Notes: {r.notes}</p>}
                  <AttachmentLinks links={r.attachment_links} voice={r.voice_note_url} />
                </td>
                <td className="px-3 py-2 text-xs whitespace-nowrap">{r.type}{r.frequency ? ` · ${r.frequency}` : ''}</td>
                <td className="px-3 py-2 text-xs whitespace-nowrap">{r.assigned_by_name || '—'}</td>
                <td className="px-3 py-2 text-xs whitespace-nowrap">{r.assign_date || '—'}</td>
                <td className="px-3 py-2 text-xs whitespace-nowrap">{r.deadline || '—'}</td>
                <td className="px-3 py-2"><span className={`text-xs px-1.5 py-0.5 rounded ${priorityStyle(r.priority || 'Medium')}`}>{r.priority}</span></td>
                <td className="px-3 py-2">
                  <WorkflowBadge status={r.workflow_status} />
                  {r.workflow_status === 'accepted' && r.task_status && <p className="text-[10px] text-gray-500 mt-0.5">{r.task_status}</p>}
                  {r.approval_status && <p className="text-[10px] text-indigo-700 mt-0.5">Approval: {r.approval_status}</p>}
                  {r.acceptance_mode === 'accepted_with_clarification' && <p className="text-[10px] text-purple-700 mt-0.5">after explanation</p>}
                  {r.acceptance_mode === 'accepted_without_clarification' && <p className="text-[10px] text-green-700 mt-0.5">Accepted without clarification</p>}
                </td>
                <td className="px-3 py-2 text-xs whitespace-nowrap">{r.pending_with ? <>{r.pending_with}<span className="text-gray-400"> ({r.pending_with_role})</span></> : '—'}</td>
                <td className="px-3 py-2 text-xs whitespace-nowrap">
                  {r.total_working_time_label || (r.time_slot_count ? `${r.time_slot_count} slots` : '—')}
                </td>
                <td className="px-3 py-2 text-xs whitespace-nowrap">
                  {r.due_state ? <span className={r.due_state === 'Overdue' ? 'text-red-700 font-semibold' : 'text-amber-700'}>{r.due_state}</span> : '—'}
                </td>
                <td className="px-3 py-2 min-w-[220px]"><AssignmentActions row={r} /></td>
              </tr>
            ))}
            {!isLoading && rows.length === 0 && (
              <tr><td colSpan={13} className="px-4 py-8 text-center text-gray-400 text-sm">Nothing here</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
