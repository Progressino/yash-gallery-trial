import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'
import { fmtHM, todayIst, tomorrowIst } from './hrmConstants'


const hhmm = (ts?: string) => (ts ? String(ts).slice(11, 16) : '')

const clock12 = (ts?: string) => {
  const v = hhmm(ts)
  if (!v) return '—'
  const [h, m] = v.split(':').map(Number)
  return `${((h + 11) % 12) + 1}:${String(m).padStart(2, '0')} ${h < 12 ? 'AM' : 'PM'}`
}

const errMsg = (e: any, fallback: string) => e?.response?.data?.detail || fallback

// ── Lunch / Tea break confirmation ───────────────────────────────────────────

export type BreakInfo = {
  name: string
  window: string
  date?: string
  break_minutes: number
  overlap_minutes: number
}

export function BreakConfirmModal({
  title, breaks, busy, onDecide, onCancel,
}: {
  title: string
  breaks: BreakInfo[]
  busy?: boolean
  onDecide: (decision: 'count' | 'deduct') => void
  onCancel: () => void
}) {
  const overlap = breaks.reduce((a, b) => a + Number(b.overlap_minutes || 0), 0)
  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
        <h3 className="font-semibold text-[#002B5B]">Break time check</h3>
        <p className="text-sm text-gray-600">{title}</p>
        <div className="space-y-2">
          {breaks.map((b, i) => (
            <div key={i} className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
              Your work time overlaps <b>{b.name}</b> ({b.window}, {b.break_minutes} min)
              {b.overlap_minutes !== b.break_minutes ? ` — ${b.overlap_minutes} min overlap` : ''}.
            </div>
          ))}
        </div>
        <p className="text-sm font-medium text-gray-800">Did you work during the break?</p>
        <div className="flex gap-2">
          <button type="button" disabled={busy} onClick={() => onDecide('count')}
            className="flex-1 py-2 bg-green-600 text-white rounded-lg text-sm disabled:opacity-50">
            Yes — count as work
          </button>
          <button type="button" disabled={busy} onClick={() => onDecide('deduct')}
            className="flex-1 py-2 bg-amber-600 text-white rounded-lg text-sm disabled:opacity-50">
            No — deduct {overlap} min
          </button>
        </div>
        <button type="button" onClick={onCancel} className="w-full text-xs text-gray-500">Cancel</button>
      </div>
    </div>
  )
}

// ── Time slots (each pause/resume work period) ───────────────────────────────

export type SlotTarget =
  | { entity_type: 'responsibility'; responsibility_id: number; log_date: string }
  | { entity_type: 'one_time'; task_id: number }

export function SlotList({
  slots, canEdit, target, onChanged,
}: {
  slots: any[]
  canEdit: boolean
  target?: SlotTarget
  onChanged: () => void
}) {
  const [editId, setEditId] = useState<number | null>(null)
  const [noteId, setNoteId] = useState<number | null>(null)
  const [noteText, setNoteText] = useState('')
  const [form, setForm] = useState({ start: '', end: '', notes: '' })
  const [adding, setAdding] = useState(false)
  const [open, setOpen] = useState(false)

  const saveMut = useMutation({
    mutationFn: ({ id, body }: { id: number; body: object }) => api.patch(`/hrm/time-slots/${id}`, body),
    onSuccess: () => { setEditId(null); setNoteId(null); onChanged() },
    onError: (e: any) => alert(errMsg(e, 'Could not save time slot')),
  })
  const addMut = useMutation({
    mutationFn: (body: object) => api.post('/hrm/time-slots', body),
    onSuccess: () => { setAdding(false); onChanged() },
    onError: (e: any) => alert(errMsg(e, 'Could not add time slot')),
  })

  const list = Array.isArray(slots) ? slots : []
  const total = list.reduce((a, s) => a + Number(s.net_seconds || 0), 0)
  const hasManual = list.some(s => Number(s.manual_edited))
  if (!list.length && !canEdit) return null

  const beginEdit = (s: any) => {
    setAdding(false)
    setNoteId(null)
    setEditId(s.id)
    setForm({ start: hhmm(s.started_at), end: hhmm(s.ended_at), notes: '' })
  }
  const beginNote = (s: any) => {
    setAdding(false)
    setEditId(null)
    setNoteId(s.id)
    setNoteText(s.notes || '')
  }
  const submitEdit = (s: any) => {
    const body: Record<string, string> = {}
    if (form.start && form.start !== hhmm(s.started_at)) body.started_at = form.start
    if (!s.is_open && form.end && form.end !== hhmm(s.ended_at)) body.ended_at = form.end
    if (!Object.keys(body).length) { setEditId(null); return }
    saveMut.mutate({ id: s.id, body })
  }
  const submitAdd = () => {
    if (!target || !form.start || !form.end) return
    addMut.mutate({ ...target, started_at: form.start, ended_at: form.end, notes: form.notes })
  }

  return (
    <div className="mt-1.5 text-[11px]">
      <button type="button" onClick={() => setOpen(v => !v)} className="flex items-center gap-1.5 text-[#002B5B] font-semibold">
        <span>{open ? '▾' : '▸'}</span>
        <span>Time slots ({list.length}) · Total {fmtHM(total)}</span>
        {hasManual && <span className="px-1.5 py-0.5 rounded bg-yellow-100 text-yellow-800 font-medium">✎ Manually edited</span>}
      </button>
      {open && (
        <div className="mt-1 border rounded-lg divide-y bg-gray-50/60">
          {list.map((s, idx) => (
            <div key={s.id} className={`px-2 py-1.5 ${Number(s.manual_edited) ? 'bg-yellow-50' : ''}`}>
              {editId === s.id ? (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-gray-500">#{idx + 1}</span>
                  <input type="time" value={form.start} onChange={e => setForm(f => ({ ...f, start: e.target.value }))} className="border rounded px-1 py-0.5" />
                  <span>–</span>
                  <input type="time" value={form.end} disabled={s.is_open} onChange={e => setForm(f => ({ ...f, end: e.target.value }))} className="border rounded px-1 py-0.5 disabled:bg-gray-100" />
                  <button type="button" disabled={saveMut.isPending} onClick={() => submitEdit(s)} className="px-2 py-0.5 bg-green-700 text-white rounded">Save time</button>
                  <button type="button" onClick={() => setEditId(null)} className="text-gray-500">Cancel</button>
                </div>
              ) : noteId === s.id ? (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-gray-500">#{idx + 1}</span>
                  <span className="font-medium text-gray-800">{clock12(s.started_at)} – {s.is_open ? 'running' : clock12(s.ended_at)}</span>
                  <input autoFocus value={noteText} onChange={e => setNoteText(e.target.value)} placeholder="Note for this time slot"
                    className="border rounded px-1 py-0.5 flex-1 min-w-[10rem]" />
                  <button type="button" disabled={saveMut.isPending} onClick={() => saveMut.mutate({ id: s.id, body: { notes: noteText } })} className="px-2 py-0.5 bg-[#002B5B] text-white rounded">Save note</button>
                  <button type="button" onClick={() => setNoteId(null)} className="text-gray-500">Cancel</button>
                </div>
              ) : (
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-gray-500">#{idx + 1}</span>
                  <span className="font-medium text-gray-800">
                    {String(s.started_at).slice(5, 10)} {clock12(s.started_at)} – {s.is_open ? <span className="text-blue-700">running</span> : clock12(s.ended_at)}
                    {!s.is_open && String(s.ended_at).slice(0, 10) !== String(s.started_at).slice(0, 10) && (
                      <span className="text-gray-500"> ({String(s.ended_at).slice(5, 10)})</span>
                    )}
                  </span>
                  <span className="text-[#002B5B] font-semibold">{s.duration_label || fmtHM(s.net_seconds)}</span>
                  {Number(s.break_deduct_seconds) > 0 && <span className="text-amber-700">−{fmtHM(s.break_deduct_seconds)} break</span>}
                  {Number(s.manual_edited) > 0 && (
                    <span className="text-yellow-800" title={s.original_started_at ? `Original: ${clock12(s.original_started_at)} – ${clock12(s.original_ended_at)}` : 'Added manually'}>
                      ✎ edited{s.edited_by ? ` by ${s.edited_by}` : ''}
                    </span>
                  )}
                  {s.end_reason === 'office_close' && <span className="text-gray-500">(auto-paused at Office Close)</span>}
                  {s.notes && <span className="text-gray-600 italic">“{s.notes}”</span>}
                  {canEdit && (
                    <span className="ml-auto flex items-center gap-2">
                      <button type="button" onClick={() => beginNote(s)} className="text-indigo-600 underline" title="Add or edit the note for this slot">
                        📝 {s.notes ? 'Note' : 'Add note'}
                      </button>
                      <button type="button" onClick={() => beginEdit(s)} className="text-blue-600 underline" title="Edit start / end time">Edit time</button>
                    </span>
                  )}
                </div>
              )}
            </div>
          ))}
          {!list.length && <p className="px-2 py-1.5 text-gray-400">No time slots yet.</p>}
          {canEdit && target && (
            <div className="px-2 py-1.5">
              {adding ? (
                <div className="flex flex-wrap items-center gap-1.5">
                  <input type="time" value={form.start} onChange={e => setForm(f => ({ ...f, start: e.target.value }))} className="border rounded px-1 py-0.5" />
                  <span>–</span>
                  <input type="time" value={form.end} onChange={e => setForm(f => ({ ...f, end: e.target.value }))} className="border rounded px-1 py-0.5" />
                  <input value={form.notes} onChange={e => setForm(f => ({ ...f, notes: e.target.value }))} placeholder="Notes" className="border rounded px-1 py-0.5 flex-1 min-w-[8rem]" />
                  <button type="button" disabled={addMut.isPending || !form.start || !form.end} onClick={submitAdd} className="px-2 py-0.5 bg-green-700 text-white rounded disabled:opacity-50">Add</button>
                  <button type="button" onClick={() => setAdding(false)} className="text-gray-500">Cancel</button>
                </div>
              ) : (
                <button type="button" onClick={() => { setEditId(null); setNoteId(null); setForm({ start: '', end: '', notes: '' }); setAdding(true) }} className="text-blue-600 underline">
                  + Add time slot
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Office time header (Employee Check) ──────────────────────────────────────

export function OfficeTimeBar({
  employeeId, summary, isToday, canAct, onChanged,
}: {
  employeeId: number
  summary: any
  isToday: boolean
  canAct: boolean
  onChanged: () => void
}) {
  const [pending, setPending] = useState<{ message: string; items: any[] } | null>(null)
  const [showGaps, setShowGaps] = useState(false)
  const [leaveOpen, setLeaveOpen] = useState(false)
  const s = summary || {}

  const startMut = useMutation({
    mutationFn: () => api.post('/hrm/office/start', { employee_id: employeeId }),
    onSuccess: () => { setPending(null); onChanged() },
    onError: (e: any) => alert(errMsg(e, 'Could not start office time')),
  })
  const closeMut = useMutation({
    mutationFn: () => api.post('/hrm/office/close', { employee_id: employeeId }),
    onSuccess: (res) => {
      setPending(null)
      const d = res.data || {}
      const n = Number(d.auto_paused_tasks || 0) + Number(d.auto_paused_responsibilities || 0)
      if (n > 0) alert(`Office closed. ${n} running item(s) were auto-paused at ${clock12(d.office_close)}.`)
      onChanged()
    },
    onError: (e: any) => {
      if (e?.response?.status === 409) {
        setPending({ message: e.response.data?.detail || 'Pending responsibilities', items: e.response.data?.pending || [] })
        return
      }
      alert(errMsg(e, 'Could not close office time'))
    },
  })

  return (
    <div className="bg-white border rounded-xl p-3 space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        {canAct && isToday && !s.office_start && !s.on_leave && (
          <button type="button" disabled={startMut.isPending} onClick={() => startMut.mutate()}
            className="px-3 py-1.5 bg-teal-700 text-white rounded-lg text-sm font-medium disabled:opacity-50">
            ▶ Office Time Start
          </button>
        )}
        {canAct && isToday && s.office_open && (
          <button type="button" disabled={closeMut.isPending} onClick={() => closeMut.mutate()}
            className="px-3 py-1.5 bg-rose-700 text-white rounded-lg text-sm font-medium disabled:opacity-50">
            ■ Office Time Close
          </button>
        )}
        {s.on_leave && <span className="px-2 py-1 rounded-lg bg-slate-600 text-white text-xs font-semibold">On Leave</span>}
        <div className="flex flex-wrap gap-2 text-xs">
          <Stat label="Office" value={s.office_start ? `${clock12(s.office_start)} – ${s.office_close ? clock12(s.office_close) : (s.office_open ? 'open' : '—')}` : 'Not started'} />
          <Stat label="Total Office Time" value={fmtHM(s.office_seconds)} strong />
          <Stat label="Actual Working" value={fmtHM(s.working_seconds)} strong tone="text-[#002B5B]" />
          <Stat label="Free Time" value={fmtHM(s.free_seconds)} tone="text-amber-700" />
        </div>
        {(s.free_gaps || []).length > 0 && (
          <button type="button" onClick={() => setShowGaps(v => !v)} className="text-xs text-blue-600 underline">
            {showGaps ? 'Hide' : 'Show'} free gaps ({s.free_gaps.length})
          </button>
        )}
        {canAct && (
          <button type="button" onClick={() => setLeaveOpen(true)} className="ml-auto px-3 py-1.5 border border-slate-500 text-slate-700 rounded-lg text-sm">
            🏖 Leave
          </button>
        )}
      </div>
      {showGaps && (
        <div className="text-xs text-gray-600 flex flex-wrap gap-1.5">
          {(s.free_gaps || []).map((g: any, i: number) => (
            <span key={i} className="px-2 py-0.5 rounded bg-amber-50 border border-amber-200">
              {clock12(g.start)} – {clock12(g.end)} · {g.label}
            </span>
          ))}
        </div>
      )}
      {pending && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
          <p className="font-semibold">Office Close blocked</p>
          <p className="text-xs mt-0.5">Update the status of these responsibilities first:</p>
          <ul className="list-disc ml-5 mt-1 text-xs">
            {pending.items.map((p: any, i: number) => <li key={i}>{p.title}{p.frequency ? ` (${p.frequency})` : ''}</li>)}
          </ul>
        </div>
      )}
      {leaveOpen && <LeaveModal employeeId={employeeId} onClose={() => setLeaveOpen(false)} onSaved={onChanged} />}
    </div>
  )
}

function Stat({ label, value, strong, tone }: { label: string; value: string; strong?: boolean; tone?: string }) {
  return (
    <div className="px-2.5 py-1 rounded-lg bg-gray-50 border">
      <p className="text-[10px] uppercase text-gray-400">{label}</p>
      <p className={`${strong ? 'font-bold' : 'font-medium'} ${tone || 'text-gray-800'}`}>{value}</p>
    </div>
  )
}

// ── Leave (backup cover for the leave period) ────────────────────────────────

export function LeaveModal({ employeeId, onClose, onSaved }: { employeeId: number; onClose: () => void; onSaved: () => void }) {
  const qc = useQueryClient()
  const [from, setFrom] = useState(todayIst())
  const [to, setTo] = useState(todayIst())
  const [reason, setReason] = useState('')
  const [result, setResult] = useState<any>(null)
  const { data: leaves = [] } = useQuery({
    queryKey: ['hrm-leaves', employeeId],
    queryFn: () => api.get(`/hrm/leaves?employee_id=${employeeId}`).then(r => r.data),
  })
  const refresh = () => { qc.invalidateQueries({ queryKey: ['hrm-leaves'] }); onSaved() }
  const saveMut = useMutation({
    mutationFn: () => api.post('/hrm/leaves', { employee_id: employeeId, from_date: from, to_date: to, reason }),
    onSuccess: (res) => { setResult(res.data); refresh() },
    onError: (e: any) => alert(errMsg(e, 'Could not save leave')),
  })
  const cancelMut = useMutation({
    mutationFn: (id: number) => api.post(`/hrm/leaves/${id}/cancel`),
    onSuccess: refresh,
    onError: (e: any) => alert(errMsg(e, 'Could not cancel leave')),
  })
  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg p-6 space-y-4">
        <h3 className="font-semibold text-slate-700">🏖 Leave</h3>
        {result ? (
          <div className="space-y-2 text-sm">
            <p className="text-green-700 font-medium">Leave saved: {result.from_date} → {result.to_date} ({result.days} day{result.days === 1 ? '' : 's'}{result.sundays_included ? `, incl. ${result.sundays_included} Sunday` : ''}).</p>
            <p className="text-gray-600">{result.backup_assignments} mandatory responsibility day(s) assigned to Backup Person. Ownership returns automatically after the leave.</p>
            {(result.mandatory_without_backup || []).length > 0 && (
              <p className="text-amber-700 text-xs">No backup set for: {result.mandatory_without_backup.join(', ')}</p>
            )}
            <button type="button" onClick={onClose} className="w-full py-2 bg-[#002B5B] text-white rounded-lg text-sm">Done</button>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-3">
              <label className="text-xs text-gray-500">Leave From Date
                <input type="date" value={from} onChange={e => { setFrom(e.target.value); if (to < e.target.value) setTo(e.target.value) }} className="w-full border rounded px-2 py-1.5 text-sm mt-1" />
              </label>
              <label className="text-xs text-gray-500">Leave To Date
                <input type="date" value={to} min={from} onChange={e => setTo(e.target.value)} className="w-full border rounded px-2 py-1.5 text-sm mt-1" />
              </label>
              <label className="text-xs text-gray-500 col-span-2">Reason (optional)
                <input value={reason} onChange={e => setReason(e.target.value)} className="w-full border rounded px-2 py-1.5 text-sm mt-1" />
              </label>
            </div>
            <p className="text-[11px] text-gray-500">Sundays inside the range are counted automatically. Mandatory responsibilities go to their Backup Person for these dates only.</p>
            <div className="flex gap-2">
              <button type="button" disabled={saveMut.isPending || !from || !to} onClick={() => saveMut.mutate()}
                className="flex-1 py-2 bg-slate-700 text-white rounded-lg text-sm disabled:opacity-50">
                {saveMut.isPending ? 'Saving…' : 'Save leave'}
              </button>
              <button type="button" onClick={onClose} className="px-4 border rounded-lg text-sm">Close</button>
            </div>
          </>
        )}
        {(leaves as any[]).length > 0 && (
          <div className="border-t pt-3">
            <p className="text-xs font-semibold text-gray-500 mb-1">Leaves</p>
            <ul className="text-xs divide-y">
              {(leaves as any[]).slice(0, 8).map((l: any) => (
                <li key={l.id} className="py-1 flex items-center gap-2">
                  <span>{l.from_date} → {l.to_date} · {l.days}d</span>
                  {l.reason && <span className="text-gray-400 truncate">{l.reason}</span>}
                  {l.to_date >= todayIst() && (
                    <button type="button" disabled={cancelMut.isPending} onClick={() => { if (window.confirm('Cancel this leave?')) cancelMut.mutate(l.id) }}
                      className="ml-auto text-red-600 underline">Cancel</button>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Task hold ────────────────────────────────────────────────────────────────

export function HoldModal({ task, onClose, onSaved }: { task: { id: number; title: string }; onClose: () => void; onSaved: () => void }) {
  const [resume, setResume] = useState(tomorrowIst())
  const [reason, setReason] = useState('')
  const mut = useMutation({
    mutationFn: () => api.post(`/hrm/one-time-tasks/${task.id}/hold`, { resume_date: resume, reason }),
    onSuccess: () => { onSaved(); onClose() },
    onError: (e: any) => alert(errMsg(e, 'Could not put task on hold')),
  })
  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
        <h3 className="font-semibold text-orange-700">⏸ Hold task</h3>
        <p className="text-sm text-gray-600">{task.title}</p>
        <label className="block text-xs text-gray-500">Resume Date
          <input type="date" value={resume} min={tomorrowIst()} onChange={e => setResume(e.target.value)} className="w-full border rounded px-2 py-1.5 text-sm mt-1" />
        </label>
        <label className="block text-xs text-gray-500">Reason (optional)
          <input value={reason} onChange={e => setReason(e.target.value)} className="w-full border rounded px-2 py-1.5 text-sm mt-1" />
        </label>
        <p className="text-[11px] text-gray-500">The task is hidden from the Task tab and Employee Check, accumulates no time, and resumes automatically on this date.</p>
        <div className="flex gap-2">
          <button type="button" disabled={mut.isPending || !resume} onClick={() => mut.mutate()} className="flex-1 py-2 bg-orange-600 text-white rounded-lg text-sm disabled:opacity-50">
            {mut.isPending ? 'Saving…' : 'Hold until resume date'}
          </button>
          <button type="button" onClick={onClose} className="px-4 border rounded-lg text-sm">Cancel</button>
        </div>
      </div>
    </div>
  )
}

// ── Holidays (Admin) ─────────────────────────────────────────────────────────

export function HolidayModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient()
  const [d, setD] = useState('')
  const [name, setName] = useState('')
  const { data: holidays = [] } = useQuery({ queryKey: ['hrm-holidays'], queryFn: () => api.get('/hrm/holidays').then(r => r.data) })
  const addMut = useMutation({
    mutationFn: () => api.post('/hrm/holidays', { holiday_date: d, name }),
    onSuccess: () => { setD(''); setName(''); qc.invalidateQueries({ queryKey: ['hrm-holidays'] }) },
    onError: (e: any) => alert(errMsg(e, 'Could not save holiday')),
  })
  const delMut = useMutation({
    mutationFn: (hd: string) => api.delete(`/hrm/holidays/${hd}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['hrm-holidays'] }),
  })
  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-3">
        <h3 className="font-semibold text-[#002B5B]">Company holidays</h3>
        <p className="text-[11px] text-gray-500">Scheduled responsibilities due on a Sunday, holiday or leave day move to the next working day.</p>
        <div className="flex gap-2">
          <input type="date" value={d} onChange={e => setD(e.target.value)} className="border rounded px-2 py-1.5 text-sm" />
          <input value={name} onChange={e => setName(e.target.value)} placeholder="Name" className="border rounded px-2 py-1.5 text-sm flex-1" />
          <button type="button" disabled={!d || addMut.isPending} onClick={() => addMut.mutate()} className="px-3 bg-[#002B5B] text-white rounded text-sm disabled:opacity-50">Add</button>
        </div>
        <ul className="max-h-64 overflow-y-auto divide-y text-sm">
          {(holidays as any[]).map((h: any) => (
            <li key={h.holiday_date} className="py-1.5 flex items-center gap-2">
              <span className="font-medium">{h.holiday_date}</span>
              <span className="text-gray-500">{h.name}</span>
              <button type="button" onClick={() => delMut.mutate(h.holiday_date)} className="ml-auto text-xs text-red-600">Remove</button>
            </li>
          ))}
          {!(holidays as any[]).length && <li className="py-2 text-gray-400 text-xs">No holidays configured.</li>}
        </ul>
        <button type="button" onClick={onClose} className="w-full py-2 border rounded-lg text-sm">Close</button>
      </div>
    </div>
  )
}

// ── Schedule rule input (Monthly / Fortnightly / Yearly) ─────────────────────

const RULE_EXAMPLES = [
  '1st Monday', '2nd Monday', '1st Saturday', '2nd Saturday', '2nd & 4th Saturday',
  'Last Friday', 'Last Working Day', 'First Working Day',
]

export function ScheduleRuleInput({ value, onChange, compact }: { value: string; onChange: (v: string) => void; compact?: boolean }) {
  return (
    <div>
      <label className={compact ? 'text-[10px] text-gray-400' : 'text-xs text-gray-500'}>Dynamic Schedule Rule (optional)</label>
      <input list="hrm-schedule-rules" value={value || ''} onChange={e => onChange(e.target.value)} placeholder="e.g. 1st Monday, Last Working Day"
        className={`w-full border rounded px-2 ${compact ? 'py-1' : 'py-1.5 mt-1'} text-sm`} />
      <datalist id="hrm-schedule-rules">
        {RULE_EXAMPLES.map(r => <option key={r} value={r} />)}
      </datalist>
      <p className="text-[10px] text-gray-400 mt-0.5">Overrides Day of Month / Weekday when set. Sunday, holiday or leave moves it to the next working day.</p>
    </div>
  )
}

const WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

/** "Twice a Week" schedule — two different weekdays stored as "Monday,Thursday". */
export function WeekdayPairInput({ value, onChange, compact }: { value: string; onChange: (v: string) => void; compact?: boolean }) {
  const parts = String(value || '').split(/[,/&]/).map(s => s.trim())
  const [d1, d2] = [parts[0] || '', parts[1] || '']
  const set = (a: string, b: string) => onChange([a, b].filter(Boolean).join(','))
  const cls = `w-full border rounded px-2 ${compact ? 'py-1' : 'py-1.5 mt-1'} text-sm`
  return (
    <div>
      <label className={compact ? 'text-[10px] text-gray-400' : 'text-xs text-gray-500'}>Weekdays (two) *</label>
      <div className="flex gap-1">
        <select value={d1} onChange={e => set(e.target.value, d2)} className={cls}>
          <option value="">First</option>
          {WEEK.map(d => <option key={d} value={d} disabled={d === d2}>{d}</option>)}
        </select>
        <select value={d2} onChange={e => set(d1, e.target.value)} className={cls}>
          <option value="">Second</option>
          {WEEK.map(d => <option key={d} value={d} disabled={d === d1}>{d}</option>)}
        </select>
      </div>
    </div>
  )
}

export const weekdayPairValid = (v: string) => new Set(String(v || '').split(/[,/&]/).map(s => s.trim()).filter(Boolean)).size === 2

// ── Reports: Daily Working Report (date / range) ─────────────────────────────

const typeBadge: Record<string, string> = {
  responsibility: 'bg-blue-50 text-blue-700',
  one_time: 'bg-indigo-50 text-indigo-700',
  backup_cover: 'bg-violet-50 text-violet-700',
}
const typeLabel: Record<string, string> = { responsibility: 'Responsibility', one_time: 'One-time task', backup_cover: 'Backup cover' }

const isoDay = (d: Date) => d.toISOString().slice(0, 10)
const istNow = () => new Date(Date.now() + 330 * 60_000)

/** Quick ranges in IST: Today, Yesterday, Previous Week (Mon–Sun), Previous Month. */
export function quickRange(kind: 'today' | 'yesterday' | 'prev_week' | 'prev_month'): [string, string] {
  const n = istNow()
  const day = new Date(Date.UTC(n.getUTCFullYear(), n.getUTCMonth(), n.getUTCDate()))
  const add = (d: Date, k: number) => new Date(d.getTime() + k * 86_400_000)
  if (kind === 'today') return [isoDay(day), isoDay(day)]
  if (kind === 'yesterday') return [isoDay(add(day, -1)), isoDay(add(day, -1))]
  if (kind === 'prev_week') {
    const mon = add(day, -((day.getUTCDay() + 6) % 7) - 7)
    return [isoDay(mon), isoDay(add(mon, 6))]
  }
  const first = new Date(Date.UTC(day.getUTCFullYear(), day.getUTCMonth() - 1, 1))
  const last = new Date(Date.UTC(day.getUTCFullYear(), day.getUTCMonth(), 0))
  return [isoDay(first), isoDay(last)]
}

const itemKey = (r: any) => `${r.row_type}:${r.responsibility_id || r.task_id || r.title}:${r.employee_id}`
const monthLabel = (ym: string) => {
  const [y, m] = ym.split('-').map(Number)
  return new Date(Date.UTC(y, m - 1, 1)).toLocaleDateString('en-IN', { month: 'short', year: 'numeric', timeZone: 'UTC' })
}

type DwrFilters = { date: string; employee: string; item: string; type: string; status: string; linked: string; minMinutes: string }
const EMPTY_FILTERS: DwrFilters = { date: '', employee: '', item: '', type: '', status: '', linked: '', minMinutes: '' }

export function DwrReport({ employeeId, departmentId }: { employeeId?: number | ''; departmentId?: number | '' }) {
  const [from, setFrom] = useState(todayIst())
  const [to, setTo] = useState(todayIst())
  const [sortDesc, setSortDesc] = useState(true)
  const [showSlots, setShowSlots] = useState(true)
  const [f, setF] = useState<DwrFilters>(EMPTY_FILTERS)
  const [focus, setFocus] = useState<string | null>(null)
  const [showAllActivities, setShowAllActivities] = useState(false)
  const { data, isFetching, error } = useQuery({
    queryKey: ['hrm-dwr', employeeId || '', departmentId || '', from, to],
    queryFn: () => {
      const p = new URLSearchParams({ from_date: from, to_date: to })
      if (employeeId) p.set('employee_id', String(employeeId))
      if (departmentId && !employeeId) p.set('department_id', String(departmentId))
      return api.get(`/hrm/dwr?${p}`).then(r => r.data)
    },
    enabled: !!from && !!to && to >= from,
  })
  const allRows = (data?.rows as any[]) || []
  const has = (v: any, q: string) => String(v || '').toLowerCase().includes(q.trim().toLowerCase())
  const rows = useMemo(() => {
    const minSec = Number(f.minMinutes || 0) * 60
    const r = allRows.filter(row =>
      (!f.date || row.check_date === f.date)
      && (!f.employee || has(row.employee_name, f.employee))
      && (!f.item || has(row.title, f.item) || has(row.remarks, f.item))
      && (!f.type || row.row_type === f.type)
      && (!f.status || row.status === f.status)
      && (!f.linked || has(row.linked_person, f.linked))
      && (!minSec || Number(row.duration_seconds || 0) >= minSec))
    r.sort((a, b) => (sortDesc ? 1 : -1) * (Number(b.duration_seconds || 0) - Number(a.duration_seconds || 0)))
    return r
  }, [allRows, f, sortDesc])
  const leaveRows = ((data?.leave_rows as any[]) || []).filter(r =>
    (!f.date || r.check_date === f.date) && (!f.employee || has(r.employee_name, f.employee)) && !f.item && !f.type && !f.status)
  const multiDay = from !== to
  const filtered = Object.values(f).some(Boolean)
  const filteredTotal = rows.reduce((a, r) => a + Number(r.duration_seconds || 0), 0)
  const opt = (k: string) => Array.from(new Set(allRows.map(r => String(r[k] || '')).filter(Boolean))).sort()

  // Activity totals inside the selected range (and filters), split by calendar month
  const activities = useMemo(() => {
    type Activity = { key: string; title: string; employee: string; type: string; frequency: string; total: number; days: number; months: Record<string, number> }
    const m = new Map<string, Activity>()
    for (const r of rows) {
      const k = itemKey(r)
      const a: Activity = m.get(k) || { key: k, title: r.title, employee: r.employee_name, type: r.row_type, frequency: r.frequency || '', total: 0, days: 0, months: {} }
      const secs = Number(r.duration_seconds || 0)
      a.total += secs
      a.days += secs > 0 ? 1 : 0
      const ym = String(r.check_date || '').slice(0, 7)
      a.months[ym] = (a.months[ym] || 0) + secs
      m.set(k, a)
    }
    return Array.from(m.values()).sort((x, y) => y.total - x.total)
  }, [rows])
  const months = useMemo(() => Array.from(new Set(rows.map(r => String(r.check_date || '').slice(0, 7)))).sort(), [rows])
  const topKey = activities.length && activities[0].total > 0 ? activities[0].key : null
  const focused = focus ? activities.find(a => a.key === focus) : null
  const setQuick = (k: Parameters<typeof quickRange>[0]) => { const [a, b] = quickRange(k); setFrom(a); setTo(b) }
  const filterInput = 'w-full border rounded px-1.5 py-0.5 text-[11px] font-normal normal-case text-gray-700 bg-white'

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-2">
        <label className="text-[10px] text-gray-400">From
          <input type="date" value={from} onChange={e => { setFrom(e.target.value); if (to < e.target.value) setTo(e.target.value) }} className="block border rounded-lg px-3 py-1.5 text-sm" />
        </label>
        <label className="text-[10px] text-gray-400">To
          <input type="date" value={to} min={from} onChange={e => setTo(e.target.value)} className="block border rounded-lg px-3 py-1.5 text-sm" />
        </label>
        <div className="flex flex-wrap gap-1">
          {([['today', 'Today'], ['yesterday', 'Yesterday'], ['prev_week', 'Previous Week'], ['prev_month', 'Previous Month']] as const).map(([k, label]) => (
            <button key={k} type="button" onClick={() => setQuick(k)} className="px-3 py-1.5 border rounded-lg text-xs hover:bg-blue-50">{label}</button>
          ))}
        </div>
        <button type="button" onClick={() => setShowSlots(v => !v)} className="px-3 py-1.5 border rounded-lg text-xs font-medium text-[#002B5B] hover:bg-blue-50">
          {showSlots ? 'Hide time slots & notes' : 'Show time slots & notes'}
        </button>
        {filtered && <button type="button" onClick={() => setF(EMPTY_FILTERS)} className="px-3 py-1.5 border border-red-200 text-red-700 rounded-lg text-xs">Clear filters</button>}
        {isFetching && <span className="text-xs text-gray-400">Loading…</span>}
      </div>
      <p className="text-[10px] text-gray-400">Only time worked inside each date counts on that date. Up to 31 days for a team, 93 days for one employee.</p>
      {error && <p className="text-sm text-red-600">{errMsg(error, 'Could not load report')}</p>}

      {focused && (
        <div className="bg-white rounded-xl border border-teal-200 p-3 space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm font-semibold text-[#002B5B]">
              {focused.title} <span className="text-xs font-normal text-gray-500">· {focused.employee} · {typeLabel[focused.type] || focused.type}{focused.frequency && focused.type !== 'one_time' ? ` · ${focused.frequency}` : ''}</span>
            </p>
            <button type="button" onClick={() => setFocus(null)} className="text-xs text-gray-500">Close ✕</button>
          </div>
          <p className="text-sm">Total in {from === to ? from : `${from} → ${to}`}: <b className="text-teal-800">{fmtHM(focused.total)}</b> across {focused.days} day{focused.days === 1 ? '' : 's'}</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(focused.months).sort().map(([ym, secs]) => (
              <span key={ym} className="px-2 py-1 rounded-lg bg-teal-50 text-teal-900 text-xs">{monthLabel(ym)}: <b>{fmtHM(secs)}</b></span>
            ))}
          </div>
        </div>
      )}

      {activities.length > 1 && (
        <div className="bg-white rounded-xl border overflow-hidden">
          <div className="px-4 py-2 bg-gray-50 text-xs font-semibold text-gray-600 flex justify-between gap-2 flex-wrap">
            <span>Time by activity (month-wise){filtered ? ' · filtered' : ''}</span>
            <span className="font-normal text-gray-400">Click a row to see its total</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="text-gray-400 uppercase">
                <tr>
                  <th className="text-left px-3 py-1.5">Activity</th>
                  {months.map(ym => <th key={ym} className="text-right px-3 py-1.5 whitespace-nowrap">{monthLabel(ym)}</th>)}
                  <th className="text-right px-3 py-1.5">Total</th>
                </tr>
              </thead>
              <tbody>
                {(showAllActivities ? activities : activities.slice(0, 8)).map(a => (
                  <tr key={a.key} onClick={() => setFocus(a.key)}
                    className={`border-t cursor-pointer hover:bg-teal-50 ${a.key === topKey ? 'bg-amber-50 font-semibold' : ''} ${a.key === focus ? 'ring-1 ring-inset ring-teal-400' : ''}`}>
                    <td className="px-3 py-1.5">
                      {a.key === topKey && <span className="mr-1 px-1 rounded bg-amber-200 text-amber-900 text-[10px]">▲ Most time</span>}
                      {a.title} <span className="text-gray-400 font-normal">· {a.employee}</span>
                    </td>
                    {months.map(ym => <td key={ym} className="text-right px-3 py-1.5 whitespace-nowrap">{a.months[ym] ? fmtHM(a.months[ym]) : '—'}</td>)}
                    <td className="text-right px-3 py-1.5 whitespace-nowrap">{fmtHM(a.total)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {activities.length > 8 && (
            <button type="button" onClick={() => setShowAllActivities(v => !v)} className="w-full py-1.5 text-xs text-blue-600 border-t">
              {showAllActivities ? 'Show top 8' : `Show all ${activities.length}`}
            </button>
          )}
        </div>
      )}

      <div className="bg-white rounded-xl border overflow-hidden">
        <div className="px-4 py-3 bg-teal-800 text-white font-semibold flex justify-between gap-2 flex-wrap">
          <span>Daily Working Report — {multiDay ? `${from} → ${to}` : from}</span>
          <span className="text-teal-100 text-xs font-normal">
            Total {fmtHM(data?.total_seconds)}{filtered ? ` · Filtered ${fmtHM(filteredTotal)} (${rows.length} rows)` : ''} ·{' '}
            <span className="px-1 rounded bg-yellow-200 text-yellow-900">manual edit</span>{' '}
            <span className="px-1 rounded bg-purple-200 text-purple-900">auto-approved</span>{' '}
            <span className="px-1 rounded bg-blue-200 text-blue-900">HOD edited</span>
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-gray-400 text-xs uppercase bg-gray-50">
              <tr>
                {multiDay && <th className="text-left px-3 py-2">Date</th>}
                <th className="text-left px-3 py-2">Employee</th>
                <th className="text-left px-3 py-2">Item</th>
                <th className="text-left px-3 py-2">Status</th>
                <th className="text-left px-3 py-2 cursor-pointer select-none" onClick={() => setSortDesc(v => !v)}>
                  DWR Duration {sortDesc ? '▼' : '▲'}
                </th>
                <th className="text-left px-3 py-2">Linked Person</th>
              </tr>
              <tr className="bg-gray-50">
                {multiDay && (
                  <th className="px-2 pb-2">
                    <select value={f.date} onChange={e => setF(x => ({ ...x, date: e.target.value }))} className={filterInput}>
                      <option value="">All dates</option>
                      {opt('check_date').map(d => <option key={d} value={d}>{d}</option>)}
                    </select>
                  </th>
                )}
                <th className="px-2 pb-2"><input value={f.employee} onChange={e => setF(x => ({ ...x, employee: e.target.value }))} placeholder="Filter…" className={filterInput} /></th>
                <th className="px-2 pb-2">
                  <div className="flex gap-1">
                    <input value={f.item} onChange={e => setF(x => ({ ...x, item: e.target.value }))} placeholder="Item / remarks…" className={filterInput} />
                    <select value={f.type} onChange={e => setF(x => ({ ...x, type: e.target.value }))} className={filterInput}>
                      <option value="">All types</option>
                      {Object.entries(typeLabel).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                    </select>
                  </div>
                </th>
                <th className="px-2 pb-2">
                  <select value={f.status} onChange={e => setF(x => ({ ...x, status: e.target.value }))} className={filterInput}>
                    <option value="">All</option>
                    {opt('status').map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </th>
                <th className="px-2 pb-2"><input type="number" min={0} value={f.minMinutes} onChange={e => setF(x => ({ ...x, minMinutes: e.target.value }))} placeholder="≥ min" className={filterInput} /></th>
                <th className="px-2 pb-2"><input value={f.linked} onChange={e => setF(x => ({ ...x, linked: e.target.value }))} placeholder="Filter…" className={filterInput} /></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row: any, idx: number) => {
                const key = itemKey(row)
                const tone = row.hod_edited ? 'bg-blue-100' : row.has_manual_slots ? 'bg-yellow-50' : row.auto_approved ? 'bg-purple-50' : ''
                const hodTip = row.hod_edited
                  ? `Status corrected by ${row.hod_edited_by || 'HOD'}${row.hod_edited_role ? ` (${row.hod_edited_role})` : ''} at ${row.hod_edited_at} · original: ${row.hod_original_status || 'Pending'}${row.hod_edit_count > 1 ? ` · ${row.hod_edit_count} edits` : ''}`
                  : undefined
                return (
                  <tr key={`${row.row_type}-${row.check_date}-${row.employee_id}-${row.responsibility_id || row.task_id}-${idx}`}
                    className={`border-t align-top ${tone}`} title={hodTip}>
                    {multiDay && <td className="px-3 py-2 text-xs text-gray-600 whitespace-nowrap">{row.check_date}</td>}
                    <td className="px-3 py-2">{row.employee_name}</td>
                    <td className="px-3 py-2">
                      <button type="button" onClick={() => setFocus(focus === key ? null : key)} className="font-medium text-left hover:underline" title="Show total time for this item in the selected range">
                        {row.title}
                      </button>
                      <p className="text-[10px] mt-0.5 flex gap-1 flex-wrap">
                        <span className={`px-1 rounded ${typeBadge[row.row_type] || 'bg-gray-50 text-gray-600'}`}>{typeLabel[row.row_type] || row.row_type}</span>
                        {row.frequency && row.row_type !== 'one_time' && <span className="text-gray-400">{row.frequency}</span>}
                        {row.carried_from_date && <span className="px-1 rounded bg-slate-100 text-slate-700">work for {row.carried_from_date}</span>}
                        {key === topKey && <span className="px-1 rounded bg-amber-200 text-amber-900">▲ Most time</span>}
                        {row.has_manual_slots && <span className="px-1 rounded bg-yellow-200 text-yellow-900">✎ manual time</span>}
                        {row.hod_edited && <span className="px-1 rounded bg-blue-200 text-blue-900">HOD edited · was {row.hod_original_status || 'Pending'}</span>}
                      </p>
                      {row.remarks && <p className="text-[11px] text-gray-500 mt-0.5">{row.remarks}</p>}
                      {showSlots && (row.slots || []).length > 0 && (
                        <ul className="mt-1 space-y-0.5 text-[11px] text-gray-600">
                          {(row.slots as any[]).map((s: any, i: number) => (
                            <li key={s.id || i} className={Number(s.manual_edited) ? 'bg-yellow-100 rounded px-1' : ''}>
                              {clock12(s.started_at)} – {s.is_open ? 'running' : clock12(s.ended_at)} · {s.duration_label || fmtHM(s.net_seconds)}
                              {s.crosses_day ? <span className="text-teal-700"> ({s.day_duration_label} on this date)</span> : null}
                              {Number(s.manual_edited) ? ' · ✎ edited' : ''}
                              {s.notes ? <span className="italic"> — {s.notes}</span> : null}
                            </li>
                          ))}
                        </ul>
                      )}
                    </td>
                    <td className="px-3 py-2 text-xs">
                      <p>{row.status}</p>
                      {row.approval_status && (
                        <p className={`mt-0.5 inline-block px-1 rounded ${row.auto_approved ? 'bg-purple-200 text-purple-900 font-semibold' : 'text-gray-500'}`}>
                          {row.approval_status}
                        </p>
                      )}
                    </td>
                    <td className="px-3 py-2 text-xs font-semibold whitespace-nowrap">{row.duration_label || fmtHM(row.duration_seconds)}</td>
                    <td className="px-3 py-2 text-xs text-indigo-800">{row.linked_person || 'Self-complete'}</td>
                  </tr>
                )
              })}
              {leaveRows.map((row: any) => (
                <tr key={`leave-${row.employee_id}-${row.check_date}`} className="border-t bg-slate-50 text-slate-600">
                  {multiDay && <td className="px-3 py-2 text-xs">{row.check_date}</td>}
                  <td className="px-3 py-2">{row.employee_name}</td>
                  <td className="px-3 py-2 font-medium" colSpan={4}>🏖 Leave <span className="text-xs font-normal">({row.leave_from} → {row.leave_to})</span></td>
                </tr>
              ))}
            </tbody>
          </table>
          {!rows.length && !leaveRows.length && !isFetching && (
            <p className="text-center text-gray-400 py-8 text-sm">{filtered ? 'No rows match these filters.' : 'No status updates for this date range.'}</p>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Reports: Actual Working Hours & Free Time ────────────────────────────────

export function WorkingHoursReport({ employeeId, departmentId }: { employeeId?: number | ''; departmentId?: number | '' }) {
  const [from, setFrom] = useState(todayIst())
  const [to, setTo] = useState(todayIst())
  const [openRow, setOpenRow] = useState<string | null>(null)
  const { data, isFetching, error } = useQuery({
    queryKey: ['hrm-working-hours', employeeId || '', departmentId || '', from, to],
    queryFn: () => {
      const p = new URLSearchParams({ from_date: from, to_date: to })
      if (employeeId) p.set('employee_id', String(employeeId))
      if (departmentId && !employeeId) p.set('department_id', String(departmentId))
      return api.get(`/hrm/reports/working-hours?${p}`).then(r => r.data)
    },
    enabled: !!from && !!to && to >= from,
  })
  const rows = (data?.rows as any[]) || []
  const emps = (data?.employees as any[]) || []
  const totals = data?.totals || {}
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-2">
        <label className="text-[10px] text-gray-400">From
          <input type="date" value={from} onChange={e => { setFrom(e.target.value); if (to < e.target.value) setTo(e.target.value) }} className="block border rounded-lg px-3 py-1.5 text-sm" />
        </label>
        <label className="text-[10px] text-gray-400">To
          <input type="date" value={to} min={from} onChange={e => setTo(e.target.value)} className="block border rounded-lg px-3 py-1.5 text-sm" />
        </label>
        {isFetching && <span className="text-xs text-gray-400">Loading…</span>}
      </div>
      {error && <p className="text-sm text-red-600">{errMsg(error, 'Could not load report')}</p>}
      <div className="grid grid-cols-3 gap-3">
        <Stat label="Total Office Time" value={fmtHM(totals.office_seconds)} strong />
        <Stat label="Actual Working Hours" value={fmtHM(totals.working_seconds)} strong tone="text-[#002B5B]" />
        <Stat label="Total Free Time" value={fmtHM(totals.free_seconds)} strong tone="text-amber-700" />
      </div>
      {emps.length > 1 && (
        <div className="bg-white rounded-xl border overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-gray-400 text-xs uppercase bg-gray-50">
              <tr>
                <th className="text-left px-3 py-2">Employee</th>
                <th className="text-left px-3 py-2">Days</th>
                <th className="text-left px-3 py-2">Office Time</th>
                <th className="text-left px-3 py-2">Working Hours</th>
                <th className="text-left px-3 py-2">Free Time</th>
              </tr>
            </thead>
            <tbody>
              {emps.map((e: any) => (
                <tr key={e.employee_id} className="border-t">
                  <td className="px-3 py-2">{e.employee_name}<span className="text-xs text-gray-400"> · {e.department_name || '—'}</span></td>
                  <td className="px-3 py-2 text-xs">{e.days}{e.leave_days ? ` (${e.leave_days} leave)` : ''}</td>
                  <td className="px-3 py-2 text-xs">{e.office_label}</td>
                  <td className="px-3 py-2 text-xs font-semibold text-[#002B5B]">{e.working_label}</td>
                  <td className="px-3 py-2 text-xs text-amber-700">{e.free_label}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="bg-white rounded-xl border overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-gray-400 text-xs uppercase bg-gray-50">
            <tr>
              <th className="text-left px-3 py-2">Date</th>
              <th className="text-left px-3 py-2">Employee</th>
              <th className="text-left px-3 py-2">Office</th>
              <th className="text-left px-3 py-2">Office Time</th>
              <th className="text-left px-3 py-2">Working Hours</th>
              <th className="text-left px-3 py-2">Free Time</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r: any) => {
              const key = `${r.employee_id}-${r.work_date}`
              return (
                <tr key={key} className={`border-t align-top ${r.on_leave ? 'bg-slate-50 text-slate-500' : ''}`}>
                  <td className="px-3 py-2 text-xs whitespace-nowrap">{r.work_date}</td>
                  <td className="px-3 py-2">{r.employee_name}</td>
                  <td className="px-3 py-2 text-xs">
                    {r.on_leave ? 'Leave' : r.office_start ? `${clock12(r.office_start)} – ${r.office_close ? clock12(r.office_close) : 'open'}` : '—'}
                  </td>
                  <td className="px-3 py-2 text-xs">{r.office_label}</td>
                  <td className="px-3 py-2 text-xs font-semibold text-[#002B5B]">{r.working_label}</td>
                  <td className="px-3 py-2 text-xs text-amber-700">
                    {r.free_label}
                    {(r.free_gaps || []).length > 0 && (
                      <button type="button" onClick={() => setOpenRow(openRow === key ? null : key)} className="ml-1 text-blue-600 underline">
                        {r.free_gaps.length} gap{r.free_gaps.length === 1 ? '' : 's'}
                      </button>
                    )}
                    {openRow === key && (
                      <ul className="mt-1 space-y-0.5 text-gray-600">
                        {(r.free_gaps as any[]).map((g: any, i: number) => <li key={i}>{clock12(g.start)} – {clock12(g.end)} · {g.label}</li>)}
                      </ul>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {!rows.length && !isFetching && <p className="text-center text-gray-400 py-8 text-sm">No office time or time slots in this range.</p>}
      </div>
    </div>
  )
}
