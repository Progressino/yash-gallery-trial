import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

interface TNAItem {
  id: number; tna_number: string; so_number: string; style_name: string
  buyer: string; merchandiser: string; order_qty: number; delivery_date: string
  priority: string; status: string; template_used: string
  total_activities: number; completed_count: number; delayed_count: number
  lines: TNALine[]
}
interface TNALine {
  id: number; sr: number; activity: string; activity_group: string
  planned_start: string; planned_end: string
  actual_start: string; actual_end: string
  status: string; responsible: string; delay_days: number; delay_reason: string; remarks: string
}
interface TNAStats { total: number; active: number; delayed_activities: number; completed: number }

const PRIORITIES = ['Normal', 'High', 'Urgent']
const LINE_STATUSES = ['Not Started', 'In Progress', 'Completed', 'Delayed', 'Cancelled']
const GROUPS = ['Merchandising', 'Fabric', 'Sampling', 'CAD', 'Purchase', 'Printing', 'Dyeing', 'Cutting', 'Stitching', 'Finishing', 'Packing', 'Quality', 'Dispatch', 'Logistics']

const statusColor = (s: string) => {
  if (s === 'Completed') return 'bg-green-100 text-green-700'
  if (s === 'In Progress') return 'bg-blue-100 text-blue-700'
  if (s === 'Delayed') return 'bg-red-100 text-red-700'
  if (s === 'Cancelled') return 'bg-gray-100 text-gray-500'
  return 'bg-yellow-50 text-yellow-700'
}

const today = new Date().toISOString().split('T')[0]

export default function TNA() {
  const qc = useQueryClient()
  const [expanded, setExpanded] = useState<number | null>(null)
  const [editLine, setEditLine] = useState<number | null>(null)
  const [editData, setEditData] = useState<Partial<TNALine>>({})
  const [showNewForm, setShowNewForm] = useState(false)
  const [filterStatus, setFilterStatus] = useState('')
  const [form, setForm] = useState({
    so_number: '', style_name: '', buyer: '', po_number: '', merchandiser: '',
    season: '', order_qty: 0, delivery_date: '', priority: 'Normal',
    template_used: 'Domestic Order TNA'
  })

  const { data: stats } = useQuery<TNAStats>({ queryKey: ['tna-stats'], queryFn: () => api.get('/tna/stats').then(r => r.data) })
  const { data: templates = [] } = useQuery<string[]>({ queryKey: ['tna-templates'], queryFn: () => api.get('/tna/templates').then(r => r.data) })
  const { data: tnas = [] } = useQuery<TNAItem[]>({ queryKey: ['tnas', filterStatus], queryFn: () => api.get('/tna' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data) })

  const createMut = useMutation({
    mutationFn: (b: object) => api.post('/tna', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['tnas'] }); qc.invalidateQueries({ queryKey: ['tna-stats'] }); setShowNewForm(false) }
  })
  const updateLineMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/tna/lines/${id}`, data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['tnas'] }); setEditLine(null) }
  })
  const updateStatusMut = useMutation({
    mutationFn: ({ id, status }: { id: number; status: string }) => api.patch(`/tna/${id}/status`, { status }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['tnas'] })
  })

  const openEdit = (line: TNALine) => {
    setEditLine(line.id)
    setEditData({ actual_start: line.actual_start, actual_end: line.actual_end, status: line.status, responsible: line.responsible, delay_reason: line.delay_reason, remarks: line.remarks })
  }

  const progressPct = (t: TNAItem) => t.total_activities > 0 ? Math.round((t.completed_count / t.total_activities) * 100) : 0

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-800">TNA — Time & Action Calendar</h1>
          <p className="text-sm text-gray-500">Track activities from BOM to dispatch for each style</p>
        </div>
        <button onClick={() => setShowNewForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
          + New TNA
        </button>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'TOTAL TNAs', value: stats.total, color: 'text-gray-700' },
            { label: 'ACTIVE', value: stats.active, color: 'text-blue-600' },
            { label: 'DELAYED ACTIVITIES', value: stats.delayed_activities, color: 'text-red-600' },
            { label: 'COMPLETED', value: stats.completed, color: 'text-green-600' },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
              <p className={`text-2xl font-bold ${color}`}>{value}</p>
              <p className="text-xs text-gray-500 mt-1 font-semibold tracking-wide">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* New TNA Form */}
      {showNewForm && (
        <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4 space-y-4">
          <h3 className="font-semibold text-gray-700">Create New TNA</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[['so_number','SO Number'],['style_name','Style Name'],['buyer','Buyer'],['merchandiser','Merchandiser'],['season','Season'],['po_number','PO Number']].map(([k,l]) => (
              <div key={k}>
                <label className="text-xs text-gray-500">{l}</label>
                <input value={(form as Record<string,string|number>)[k] as string}
                  onChange={e => setForm(f => ({ ...f, [k]: e.target.value }))}
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
              </div>
            ))}
            <div>
              <label className="text-xs text-gray-500">Delivery Date *</label>
              <input type="date" value={form.delivery_date} onChange={e => setForm(f => ({ ...f, delivery_date: e.target.value }))}
                className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
            </div>
            <div>
              <label className="text-xs text-gray-500">Order Qty</label>
              <input type="number" value={form.order_qty} onChange={e => setForm(f => ({ ...f, order_qty: +e.target.value }))}
                className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
            </div>
            <div>
              <label className="text-xs text-gray-500">Priority</label>
              <select value={form.priority} onChange={e => setForm(f => ({ ...f, priority: e.target.value }))}
                className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                {PRIORITIES.map(p => <option key={p}>{p}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500">Template</label>
              <select value={form.template_used} onChange={e => setForm(f => ({ ...f, template_used: e.target.value }))}
                className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                {templates.map(t => <option key={t}>{t}</option>)}
              </select>
            </div>
          </div>
          <div className="flex gap-2">
            <button onClick={() => createMut.mutate(form)} disabled={createMut.isPending || !form.delivery_date}
              className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800 disabled:opacity-50">
              {createMut.isPending ? 'Creating…' : 'Create TNA'}
            </button>
            <button onClick={() => setShowNewForm(false)} className="px-4 py-2 border border-gray-200 rounded-lg text-sm text-gray-600">Cancel</button>
          </div>
          {!form.delivery_date && <p className="text-xs text-orange-500">Delivery date required to auto-calculate activity dates.</p>}
        </div>
      )}

      {/* Filter */}
      <div className="flex gap-2">
        {['', 'Active', 'Completed', 'On Hold'].map(s => (
          <button key={s} onClick={() => setFilterStatus(s)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium ${filterStatus === s ? 'bg-[#002B5B] text-white' : 'bg-white border border-gray-200 text-gray-600 hover:bg-gray-50'}`}>
            {s || 'All'}
          </button>
        ))}
      </div>

      {/* TNA List */}
      <div className="space-y-3">
        {tnas.map(t => (
          <div key={t.id} className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
            {/* Header */}
            <div className="p-4 cursor-pointer" onClick={() => setExpanded(expanded === t.id ? null : t.id)}>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <p className="font-semibold text-gray-800">{t.tna_number}</p>
                    {t.so_number && <span className="text-xs text-gray-400">· {t.so_number}</span>}
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${t.priority === 'Urgent' ? 'bg-red-100 text-red-700' : t.priority === 'High' ? 'bg-orange-100 text-orange-700' : 'bg-gray-100 text-gray-500'}`}>{t.priority}</span>
                  </div>
                  <p className="text-sm text-gray-600">{t.style_name || '—'} {t.buyer ? `· ${t.buyer}` : ''}</p>
                  <p className="text-xs text-gray-400 mt-0.5">Delivery: {t.delivery_date || '—'} · Template: {t.template_used}</p>
                </div>
                <div className="flex items-center gap-3 ml-4">
                  {t.delayed_count > 0 && (
                    <span className="text-xs bg-red-50 text-red-600 px-2 py-0.5 rounded-full font-medium">{t.delayed_count} delayed</span>
                  )}
                  <div className="text-right">
                    <p className="text-xs text-gray-500">{t.completed_count}/{t.total_activities} done</p>
                    <div className="w-24 h-1.5 bg-gray-100 rounded-full mt-1">
                      <div className="h-1.5 bg-green-500 rounded-full" style={{ width: `${progressPct(t)}%` }} />
                    </div>
                  </div>
                  <select value={t.status} onClick={e => e.stopPropagation()} onChange={e => updateStatusMut.mutate({ id: t.id, status: e.target.value })}
                    className="border border-gray-200 rounded px-2 py-1 text-xs">
                    {['Active', 'On Hold', 'Completed'].map(s => <option key={s}>{s}</option>)}
                  </select>
                  <span className="text-gray-400 text-xs">{expanded === t.id ? '▲' : '▼'}</span>
                </div>
              </div>
            </div>

            {/* Activity Lines */}
            {expanded === t.id && (
              <div className="border-t border-gray-50">
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead className="bg-gray-50">
                      <tr className="text-gray-400 uppercase">
                        <th className="text-left px-3 py-2 w-6">#</th>
                        <th className="text-left px-3 py-2">Activity</th>
                        <th className="text-left px-3 py-2">Group</th>
                        <th className="text-left px-3 py-2">Planned End</th>
                        <th className="text-left px-3 py-2">Actual End</th>
                        <th className="text-left px-3 py-2">Status</th>
                        <th className="text-left px-3 py-2">Responsible</th>
                        <th className="text-right px-3 py-2">Delay</th>
                        <th className="px-3 py-2"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {t.lines.map(line => {
                        const isLate = line.status !== 'Completed' && line.planned_end && line.planned_end < today
                        return (
                          <tr key={line.id} className={`border-t border-gray-50 ${isLate ? 'bg-red-50/30' : ''}`}>
                            {editLine === line.id ? (
                              <>
                                <td className="px-3 py-2 text-gray-400">{line.sr}</td>
                                <td className="px-3 py-2 font-medium text-gray-700">{line.activity}</td>
                                <td className="px-3 py-2 text-gray-500">{line.activity_group}</td>
                                <td className="px-3 py-2 text-gray-500">{line.planned_end}</td>
                                <td className="px-3 py-2">
                                  <input type="date" value={editData.actual_end || ''} onChange={e => setEditData(d => ({ ...d, actual_end: e.target.value }))}
                                    className="border border-gray-200 rounded px-1 py-0.5 text-xs w-32" />
                                </td>
                                <td className="px-3 py-2">
                                  <select value={editData.status || line.status} onChange={e => setEditData(d => ({ ...d, status: e.target.value }))}
                                    className="border border-gray-200 rounded px-1 py-0.5 text-xs">
                                    {LINE_STATUSES.map(s => <option key={s}>{s}</option>)}
                                  </select>
                                </td>
                                <td className="px-3 py-2">
                                  <input value={editData.responsible || ''} onChange={e => setEditData(d => ({ ...d, responsible: e.target.value }))}
                                    placeholder="Person" className="border border-gray-200 rounded px-1 py-0.5 text-xs w-24" />
                                </td>
                                <td className="px-3 py-2 text-right text-gray-400">—</td>
                                <td className="px-3 py-2">
                                  <div className="flex gap-1">
                                    <button onClick={() => updateLineMut.mutate({ id: line.id, data: editData })}
                                      className="text-xs text-green-600 hover:text-green-700 font-medium">Save</button>
                                    <button onClick={() => setEditLine(null)} className="text-xs text-gray-400">✕</button>
                                  </div>
                                </td>
                              </>
                            ) : (
                              <>
                                <td className="px-3 py-2 text-gray-400">{line.sr}</td>
                                <td className="px-3 py-2 font-medium text-gray-700">{line.activity}</td>
                                <td className="px-3 py-2 text-gray-500">{line.activity_group}</td>
                                <td className={`px-3 py-2 ${isLate ? 'text-red-600 font-medium' : 'text-gray-500'}`}>{line.planned_end}</td>
                                <td className="px-3 py-2 text-gray-600">{line.actual_end || '—'}</td>
                                <td className="px-3 py-2">
                                  <span className={`px-2 py-0.5 rounded-full font-medium text-xs ${statusColor(line.status)}`}>{line.status}</span>
                                </td>
                                <td className="px-3 py-2 text-gray-600">{line.responsible || '—'}</td>
                                <td className={`px-3 py-2 text-right font-medium ${line.delay_days > 0 ? 'text-red-600' : 'text-gray-400'}`}>
                                  {line.delay_days > 0 ? `+${line.delay_days}d` : '—'}
                                </td>
                                <td className="px-3 py-2">
                                  <button onClick={() => openEdit(line)} className="text-xs text-blue-500 hover:text-blue-700">Edit</button>
                                </td>
                              </>
                            )}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        ))}
        {tnas.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No TNAs found. Create your first TNA to start tracking activities.</p>}
      </div>
    </div>
  )
}
