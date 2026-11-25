import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'
import { todayIst } from './hrmConstants'

type MonitorRow = {
  employee_id: number
  employee_name: string
  department_name: string
  attendance_status: string
  biometric_in: string
  biometric_out: string
  hrm_started: boolean
  hrm_start_at: string
  work_started: boolean
  work_start_at: string
  current_work: string
  current_work_state: string
  working_label: string
  last_reminder_at: string
  can_remind_hrm: boolean
  can_remind_work: boolean
}

const clock = (ts: string) => {
  if (!ts) return '—'
  const m = String(ts).match(/(\d{2}):(\d{2})/)
  if (!m) return ts
  let h = +m[1]
  const ap = h >= 12 ? 'PM' : 'AM'
  h = h % 12 || 12
  return `${h}:${m[2]} ${ap}`
}

export function AttendanceMonitorReport({
  depts,
  emps,
  onOpenDwr,
}: {
  depts: { id: number; name: string }[]
  emps: { id: number; name: string; department_id?: number }[]
  onOpenDwr: (employeeId: number, day: string) => void
}) {
  const qc = useQueryClient()
  const [day, setDay] = useState(todayIst())
  const [dept, setDept] = useState('')
  const [emp, setEmp] = useState('')
  const [hod, setHod] = useState('')
  const [att, setAtt] = useState('')
  const [hrm, setHrm] = useState('')
  const [work, setWork] = useState('')
  const [cardFilter, setCardFilter] = useState('')

  const params = useMemo(() => ({
    work_date: day,
    ...(dept ? { department_id: dept } : {}),
    ...(emp ? { employee_id: emp } : {}),
    ...(hod ? { reporting_hod_id: hod } : {}),
    ...(att ? { attendance_status: att } : {}),
    ...(hrm ? { hrm_started: hrm } : {}),
    ...(work ? { work_started: work } : {}),
  }), [day, dept, emp, hod, att, hrm, work])

  const { data, isLoading, isFetching, error } = useQuery({
    queryKey: ['hrm-att-monitor', params],
    queryFn: () => api.get('/hrm/reports/attendance-monitor', { params }).then(r => r.data),
    staleTime: 20_000,
  })

  const remindMut = useMutation({
    mutationFn: (body: { employee_id: number; kind: string }) =>
      api.post('/hrm/reports/attendance-monitor/remind', { ...body, work_date: day }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['hrm-att-monitor'] }),
    onError: (e: any) => alert(e?.response?.data?.detail || 'Reminder failed'),
  })

  const rows: MonitorRow[] = data?.rows || []
  const summary = data?.summary || {}
  const sync = data?.sync || {}

  const visible = useMemo(() => {
    if (cardFilter === 'present') return rows.filter(r => r.attendance_status === 'Present')
    if (cardFilter === 'absent') return rows.filter(r => r.attendance_status === 'Absent')
    if (cardFilter === 'hrm_gap') return rows.filter(r => r.attendance_status === 'Present' && !r.hrm_started)
    if (cardFilter === 'work_gap') return rows.filter(r => r.hrm_started && !r.work_started)
    return rows
  }, [rows, cardFilter])

  const clear = () => {
    setDept(''); setEmp(''); setHod(''); setAtt(''); setHrm(''); setWork(''); setCardFilter('')
  }

  const Card = ({ label, value, tone, onClick }: { label: string; value: number; tone?: string; onClick?: () => void }) => (
    <button type="button" onClick={onClick}
      className={`text-left rounded-xl border bg-white px-3 py-2 shadow-sm hover:border-[#002B5B] ${tone || ''}`}>
      <div className="text-[10px] uppercase text-gray-500">{label}</div>
      <div className="text-xl font-semibold text-[#002B5B] tabular-nums">{value ?? 0}</div>
    </button>
  )

  return (
    <div className="space-y-3">
      <div>
        <h3 className="font-semibold text-gray-800">Attendance & Work Monitor</h3>
        <p className="text-xs text-gray-500">
          Biometric attendance, HRM Office Start and Work Start on one screen. Default date is today.
          Without biometric sync, employees are shown as <b>Not synced</b> (never forced Absent).
        </p>
        <p className="text-[11px] text-gray-400 mt-1">
          Sync: {sync.last_ok_at || sync.last_sync_at || '—'}
          {sync.last_error ? ` · Error: ${sync.last_error}` : ''}
          {sync.note ? ` · ${sync.note}` : ''}
        </p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
        <Card label="Total Active" value={summary.total_active} onClick={() => setCardFilter('')} />
        <Card label="Present" value={summary.present} tone="border-green-200" onClick={() => setCardFilter('present')} />
        <Card label="Absent" value={summary.absent} tone="border-red-200" onClick={() => setCardFilter('absent')} />
        <Card label="Present · HRM not started" value={summary.present_hrm_not_started} tone="border-amber-200" onClick={() => setCardFilter('hrm_gap')} />
        <Card label="HRM · Work not started" value={summary.hrm_started_work_not_started} tone="border-orange-200" onClick={() => setCardFilter('work_gap')} />
      </div>

      <div className="flex flex-wrap gap-2 bg-white border rounded-xl p-3 items-end">
        <label className="text-[10px] text-gray-500">Date
          <input type="date" value={day} onChange={e => setDay(e.target.value || todayIst())} className="block mt-0.5 border rounded px-2 py-1.5 text-sm" />
        </label>
        <label className="text-[10px] text-gray-500">Department
          <select value={dept} onChange={e => setDept(e.target.value)} className="block mt-0.5 border rounded px-2 py-1.5 text-sm">
            <option value="">All</option>
            {depts.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
        </label>
        <label className="text-[10px] text-gray-500">Employee
          <select value={emp} onChange={e => setEmp(e.target.value)} className="block mt-0.5 border rounded px-2 py-1.5 text-sm min-w-[10rem]">
            <option value="">All</option>
            {emps.map(e => <option key={e.id} value={e.id}>{e.name}</option>)}
          </select>
        </label>
        <label className="text-[10px] text-gray-500">Reporting HOD
          <select value={hod} onChange={e => setHod(e.target.value)} className="block mt-0.5 border rounded px-2 py-1.5 text-sm min-w-[10rem]">
            <option value="">All</option>
            {emps.map(e => <option key={e.id} value={e.id}>{e.name}</option>)}
          </select>
        </label>
        <label className="text-[10px] text-gray-500">Attendance
          <select value={att} onChange={e => setAtt(e.target.value)} className="block mt-0.5 border rounded px-2 py-1.5 text-sm">
            <option value="">All</option>
            {['Present', 'Absent', 'Leave', 'Weekly Off', 'Holiday', 'Not synced', 'Pending'].map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="text-[10px] text-gray-500">HRM Started
          <select value={hrm} onChange={e => setHrm(e.target.value)} className="block mt-0.5 border rounded px-2 py-1.5 text-sm">
            <option value="">All</option>
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </label>
        <label className="text-[10px] text-gray-500">Work Started
          <select value={work} onChange={e => setWork(e.target.value)} className="block mt-0.5 border rounded px-2 py-1.5 text-sm">
            <option value="">All</option>
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </label>
        <button type="button" onClick={clear} className="text-xs border rounded px-3 py-1.5">Clear Filters</button>
        {isFetching && <span className="text-[10px] text-gray-400">Refreshing…</span>}
      </div>

      {error && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
          {(error as any)?.response?.data?.detail || 'Could not load attendance monitor.'}
        </div>
      )}

      <div className="bg-white border rounded-xl overflow-hidden">
        <div className="hrm-sticky-scroll max-h-[65vh]">
          <table className="w-full text-sm min-w-[1200px]">
            <thead>
              <tr className="text-[10px] uppercase text-gray-500 bg-gray-50">
                {['Employee', 'Department', 'Attendance', 'Bio In', 'Bio Out', 'HRM Started', 'Work Started', 'Current Work', 'Actual Working', 'Action'].map(h => (
                  <th key={h} className="text-left px-3 py-2 sticky top-0 z-10 bg-gray-50">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visible.map(r => (
                <tr key={r.employee_id} className="border-t align-top">
                  <td className="px-3 py-2">
                    <button type="button" className="text-[#002B5B] font-medium underline text-left"
                      onClick={() => onOpenDwr(r.employee_id, day)}>
                      {r.employee_name}
                    </button>
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-600">{r.department_name || '—'}</td>
                  <td className="px-3 py-2 text-xs">{r.attendance_status}</td>
                  <td className="px-3 py-2 text-xs whitespace-nowrap">{clock(r.biometric_in)}</td>
                  <td className="px-3 py-2 text-xs whitespace-nowrap">{clock(r.biometric_out)}</td>
                  <td className="px-3 py-2 text-xs">{r.hrm_started ? `Yes · ${clock(r.hrm_start_at)}` : 'No'}</td>
                  <td className="px-3 py-2 text-xs">{r.work_started ? `Yes · ${clock(r.work_start_at)}` : 'No'}</td>
                  <td className="px-3 py-2 text-xs">
                    <span className={r.current_work_state === 'Running' ? 'text-green-700' : 'text-gray-600'}>
                      {r.current_work}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-xs tabular-nums">{r.working_label}</td>
                  <td className="px-3 py-2 text-xs space-y-1">
                    {r.can_remind_hrm && (
                      <button type="button" disabled={remindMut.isPending}
                        className="block px-2 py-1 rounded bg-amber-600 text-white"
                        onClick={() => remindMut.mutate({ employee_id: r.employee_id, kind: 'start_hrm' })}>
                        Remind to Start HRM
                      </button>
                    )}
                    {r.can_remind_work && (
                      <button type="button" disabled={remindMut.isPending}
                        className="block px-2 py-1 rounded bg-orange-600 text-white"
                        onClick={() => remindMut.mutate({ employee_id: r.employee_id, kind: 'start_work' })}>
                        Remind to Start Work
                      </button>
                    )}
                    {r.last_reminder_at && <div className="text-[10px] text-gray-400">Last: {r.last_reminder_at}</div>}
                    {!r.can_remind_hrm && !r.can_remind_work && <span className="text-gray-400">—</span>}
                  </td>
                </tr>
              ))}
              {!isLoading && visible.length === 0 && (
                <tr><td colSpan={10} className="px-4 py-8 text-center text-gray-400">No employees match the filters</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
