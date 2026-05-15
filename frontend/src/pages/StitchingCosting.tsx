import { useCallback, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type TabId =
  | 'dashboard'
  | 'production'
  | 'challan'
  | 'style'
  | 'efficiency'
  | 'payroll'
  | 'attendance'
  | 'operating'
  | 'master'

const TABS: { id: TabId; label: string }[] = [
  { id: 'dashboard', label: '🏠 Dashboard' },
  { id: 'production', label: '📋 Production Entry' },
  { id: 'challan', label: '🧾 Challans' },
  { id: 'style', label: '💎 Style Costing' },
  { id: 'efficiency', label: '📊 Efficiency' },
  { id: 'payroll', label: '💰 Payroll' },
  { id: 'attendance', label: '🕐 Karigar Attendance' },
  { id: 'operating', label: '🏢 Operating Staff' },
  { id: 'master', label: '⚙️ Master Data' },
]

interface HourDef {
  col: string
  label: string
}

interface DashboardData {
  date: string
  metrics: {
    active_karigar: number
    total_karigar: number
    pieces_today: number
    avg_efficiency: number
    piece_value_today: number
    total_challans: number
    pending_challans: number
  }
  karigar_status: { Karigar_ID: string; Name: string; Skill: string; Status: string }[]
  challan_register: { Challan_No: string; Style: string; Party: string; Pending: number; Status: string }[]
  today_production: Record<string, unknown>[]
}

function todayStr() {
  return new Date().toISOString().slice(0, 10)
}

export default function StitchingCosting() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<TabId>('dashboard')
  const [msg, setMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)

  const { data: status } = useQuery({
    queryKey: ['stitching-status'],
    queryFn: () => api.get('/stitching/status').then(r => r.data),
  })

  const flash = (type: 'ok' | 'err', text: string) => {
    setMsg({ type, text })
    setTimeout(() => setMsg(null), 5000)
  }

  const syncFrom = useMutation({
    mutationFn: () => api.post('/stitching/sync/from-gsheet'),
    onSuccess: r => {
      qc.invalidateQueries({ queryKey: ['stitching'] })
      flash(r.data.ok ? 'ok' : 'err', r.data.ok ? 'Loaded from Google Sheets' : r.data.message || 'Sync failed')
    },
    onError: () => flash('err', 'Google Sheets sync failed'),
  })

  const syncTo = useMutation({
    mutationFn: () => api.post('/stitching/sync/to-gsheet'),
    onSuccess: r => flash(r.data.ok ? 'ok' : 'err', r.data.ok ? 'Saved to Google Sheets' : r.data.message || 'Sync failed'),
  })

  return (
    <div className="space-y-4 max-w-[1600px]">
      <div className="rounded-xl bg-gradient-to-br from-[#1a3a5c] via-[#2c5aa0] to-[#1e7ed4] text-white p-5 shadow-md">
        <h1 className="text-xl font-bold">🧵 Stitching Costing — Yash Gallery</h1>
        <p className="text-sm opacity-90 mt-1">
          Karigar tracking · Challan management · Style costing · Payroll
          {status?.gsheet?.available ? ' · Google Sheets connected' : ' · Local database'}
        </p>
      </div>

      {msg && (
        <div
          className={`text-sm px-4 py-2 rounded-lg border ${
            msg.type === 'ok' ? 'bg-green-50 text-green-800 border-green-200' : 'bg-rose-50 text-rose-800 border-rose-200'
          }`}
        >
          {msg.text}
        </div>
      )}

      <div className="flex flex-wrap gap-2 items-center">
        <button
          type="button"
          onClick={() => syncFrom.mutate()}
          disabled={syncFrom.isPending || !status?.gsheet?.available}
          className="text-xs px-3 py-1.5 rounded-lg border border-[#2c5aa0] text-[#2c5aa0] hover:bg-blue-50 disabled:opacity-40"
        >
          {syncFrom.isPending ? 'Loading…' : '↻ Pull from Google Sheets'}
        </button>
        <button
          type="button"
          onClick={() => syncTo.mutate()}
          disabled={syncTo.isPending || !status?.gsheet?.available}
          className="text-xs px-3 py-1.5 rounded-lg border border-[#2c5aa0] text-[#2c5aa0] hover:bg-blue-50 disabled:opacity-40"
        >
          {syncTo.isPending ? 'Saving…' : '↑ Push to Google Sheets'}
        </button>
        <a
          href="/api/stitching/export-zip"
          className="text-xs px-3 py-1.5 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50"
        >
          📦 Export ZIP
        </a>
      </div>

      <div className="flex flex-wrap gap-1 border-b border-gray-200 pb-1">
        {TABS.map(t => (
          <button
            key={t.id}
            type="button"
            onClick={() => setTab(t.id)}
            className={`text-xs sm:text-sm px-3 py-2 rounded-t-lg font-medium transition-colors ${
              tab === t.id ? 'bg-[#002B5B] text-white' : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'dashboard' && <DashboardTab />}
      {tab === 'production' && <ProductionTab hours={status?.hours ?? []} onSaved={() => qc.invalidateQueries({ queryKey: ['stitching-dashboard'] })} />}
      {tab === 'challan' && <ChallanTab />}
      {tab === 'style' && <StyleCostingTab />}
      {tab === 'efficiency' && <EfficiencyTab />}
      {tab === 'payroll' && <PayrollTab />}
      {tab === 'attendance' && <AttendanceTab type="karigar" />}
      {tab === 'operating' && <AttendanceTab type="operating" />}
      {tab === 'master' && <MasterTab />}
    </div>
  )
}

function DashboardTab() {
  const { data, isLoading } = useQuery<DashboardData>({
    queryKey: ['stitching-dashboard'],
    queryFn: () => api.get('/stitching/dashboard').then(r => r.data),
  })
  if (isLoading) return <p className="text-sm text-gray-500">Loading dashboard…</p>
  if (!data) return null
  const m = data.metrics
  const cards = [
    ['Active Karigar', `${m.active_karigar} / ${m.total_karigar}`],
    ['Pieces today', m.pieces_today.toLocaleString()],
    ['Avg efficiency', `${m.avg_efficiency}%`],
    ['Piece value', `₹${m.piece_value_today.toLocaleString()}`],
    ['Challans', String(m.total_challans)],
    ['Pending', String(m.pending_challans)],
  ]
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {cards.map(([l, v]) => (
          <div key={l} className="bg-white rounded-lg border border-blue-100 p-3 shadow-sm">
            <p className="text-[10px] uppercase text-gray-500 font-semibold">{l}</p>
            <p className="text-xl font-bold text-[#2c5aa0] mt-1">{v}</p>
          </div>
        ))}
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        <Section title="Karigar status">
          <DataTable
            rows={data.karigar_status}
            cols={['Karigar_ID', 'Name', 'Skill', 'Status']}
          />
        </Section>
        <Section title="Challan register">
          <DataTable rows={data.challan_register} cols={['Challan_No', 'Style', 'Party', 'Pending', 'Status']} />
        </Section>
      </div>
      {data.today_production.length > 0 && (
        <Section title="Today's production">
          <DataTable
            rows={data.today_production}
            cols={['Karigar_Name', 'Challan_No', 'Style', 'Operation', 'Total_Pieces', 'Efficiency_%', 'Piece_Value_Rs']}
          />
        </Section>
      )}
    </div>
  )
}

function ProductionTab({ hours, onSaved }: { hours: HourDef[]; onSaved: () => void }) {
  const [entryDate, setEntryDate] = useState(todayStr())
  const [karigarId, setKarigarId] = useState('')
  const [style, setStyle] = useState('')
  const [challanNo, setChallanNo] = useState('')
  const [hourState, setHourState] = useState<Record<string, { operation: string; pieces: number }>>({})

  const { data: karigarSheet } = useQuery({
    queryKey: ['stitching-sheet', 'karigar_master'],
    queryFn: () => api.get('/stitching/sheets/karigar_master').then(r => r.data),
  })
  const { data: styleSheet } = useQuery({
    queryKey: ['stitching-sheet', 'style_master'],
    queryFn: () => api.get('/stitching/sheets/style_master').then(r => r.data),
  })
  const { data: challanSheet } = useQuery({
    queryKey: ['stitching-sheet', 'challan_master'],
    queryFn: () => api.get('/stitching/sheets/challan_master').then(r => r.data),
  })

  const karigars = (karigarSheet?.rows ?? []) as { Karigar_ID: string; Name: string }[]
  const styles = useMemo(() => {
    const s = new Set<string>()
    for (const r of (styleSheet?.rows ?? []) as { Style: string }[]) {
      if (r.Style) s.add(String(r.Style))
    }
    return [...s]
  }, [styleSheet])
  const challans = useMemo(() => {
    return ((challanSheet?.rows ?? []) as { Challan_No: string; Style: string; Party?: string }[]).filter(
      c => !style || c.Style === style,
    )
  }, [challanSheet, style])

  const operations = useMemo(() => {
    return ((styleSheet?.rows ?? []) as { Style: string; Operation: string; Target: number; Rate_Rs: number }[])
      .filter(r => r.Style === style)
      .map(r => ({ op: r.Operation, target: r.Target, rate: r.Rate_Rs }))
  }, [styleSheet, style])

  const loadEntry = useCallback(async () => {
    if (!karigarId || !challanNo || !style) return
    const { data } = await api.get('/stitching/production-entry/load', {
      params: { date: entryDate, karigar_id: karigarId, challan_no: challanNo, style },
    })
    const next: Record<string, { operation: string; pieces: number }> = {}
    for (const h of hours) {
      const e = data.hours?.[h.col] ?? { operation: '', pieces: 0 }
      next[h.col] = { operation: e.operation || '', pieces: Number(e.pieces) || 0 }
    }
    setHourState(next)
  }, [entryDate, karigarId, challanNo, style, hours])

  useEffect(() => {
    void loadEntry()
  }, [loadEntry])

  const saveMut = useMutation({
    mutationFn: () => {
      const k = karigars.find(x => String(x.Karigar_ID) === karigarId)
      const hour_entries = hours
        .filter(h => h.col !== 'H_13_14')
        .map(h => ({
          hour_col: h.col,
          operation: hourState[h.col]?.operation || '',
          pieces: hourState[h.col]?.pieces || 0,
        }))
      return api.post('/stitching/production-entry', {
        date: entryDate,
        karigar_id: karigarId,
        karigar_name: k?.Name ?? karigarId,
        challan_no: challanNo,
        style,
        hour_entries,
      })
    },
    onSuccess: r => {
      if (r.data.ok) onSaved()
      alert(r.data.message || (r.data.ok ? 'Saved' : 'Failed'))
    },
  })

  const setHour = (col: string, patch: Partial<{ operation: string; pieces: number }>) => {
    setHourState(prev => ({ ...prev, [col]: { ...prev[col], operation: prev[col]?.operation ?? '', pieces: prev[col]?.pieces ?? 0, ...patch } }))
  }

  const opOptions = ['', ...operations.map(o => o.op)]

  return (
    <div className="space-y-4">
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 bg-white p-4 rounded-xl border">
        <label className="text-xs">
          <span className="font-semibold text-gray-700">Date</span>
          <input type="date" className="mt-1 w-full border rounded px-2 py-1.5 text-sm" value={entryDate} onChange={e => setEntryDate(e.target.value)} />
        </label>
        <label className="text-xs">
          <span className="font-semibold text-gray-700">Karigar</span>
          <select className="mt-1 w-full border rounded px-2 py-1.5 text-sm" value={karigarId} onChange={e => setKarigarId(e.target.value)}>
            <option value="">Select…</option>
            {karigars.map(k => (
              <option key={k.Karigar_ID} value={String(k.Karigar_ID)}>
                {k.Karigar_ID} — {k.Name}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs">
          <span className="font-semibold text-gray-700">Style</span>
          <select className="mt-1 w-full border rounded px-2 py-1.5 text-sm" value={style} onChange={e => { setStyle(e.target.value); setChallanNo('') }}>
            <option value="">Select…</option>
            {styles.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
        <label className="text-xs">
          <span className="font-semibold text-gray-700">Challan</span>
          <select className="mt-1 w-full border rounded px-2 py-1.5 text-sm" value={challanNo} onChange={e => setChallanNo(e.target.value)}>
            <option value="">Select…</option>
            {challans.map(c => (
              <option key={c.Challan_No} value={String(c.Challan_No)}>
                {c.Challan_No} {c.Party ? `| ${c.Party}` : ''}
              </option>
            ))}
          </select>
        </label>
      </div>

      <Section title="Hour-wise entry">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-[#1a3a5c] text-white">
                <th className="px-2 py-2">Time</th>
                <th className="px-2 py-2">Operation</th>
                <th className="px-2 py-2">Target/hr</th>
                <th className="px-2 py-2">Pieces</th>
                <th className="px-2 py-2">Eff%</th>
              </tr>
            </thead>
            <tbody>
              {hours.map(h => {
                if (h.col === 'H_13_14') {
                  return (
                    <tr key={h.col} className="bg-gray-50 text-gray-400 italic">
                      <td className="px-2 py-2 text-center">{h.label}</td>
                      <td colSpan={4} className="px-2 py-2">Lunch break</td>
                    </tr>
                  )
                }
                const st = hourState[h.col] ?? { operation: '', pieces: 0 }
                const opMeta = operations.find(o => o.op === st.operation)
                const eff = opMeta && st.pieces > 0 ? Math.round((st.pieces / Math.max(opMeta.target, 1)) * 100) : null
                return (
                  <tr key={h.col} className="border-b border-gray-100">
                    <td className="px-2 py-2 font-semibold text-center text-[#2c5aa0]">{h.label}</td>
                    <td className="px-2 py-1">
                      <select
                        className="w-full border rounded px-1 py-1"
                        value={st.operation}
                        onChange={e => setHour(h.col, { operation: e.target.value })}
                      >
                        {opOptions.map(o => (
                          <option key={o || '_'} value={o}>{o || '—'}</option>
                        ))}
                      </select>
                    </td>
                    <td className="px-2 py-2 text-center">{opMeta?.target ?? '—'}</td>
                    <td className="px-2 py-1">
                      <input
                        type="number"
                        min={0}
                        className="w-20 border rounded px-1 py-1"
                        value={st.pieces}
                        onChange={e => setHour(h.col, { pieces: +e.target.value || 0 })}
                      />
                    </td>
                    <td className="px-2 py-2 text-center">{eff != null ? `${eff}%` : '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <button
          type="button"
          onClick={() => saveMut.mutate()}
          disabled={saveMut.isPending || !karigarId || !challanNo || !style}
          className="mt-4 w-full py-2.5 rounded-lg bg-[#002B5B] text-white font-semibold text-sm disabled:opacity-50"
        >
          {saveMut.isPending ? 'Saving…' : '💾 Save production entry'}
        </button>
      </Section>
    </div>
  )
}

function ChallanTab() {
  const qc = useQueryClient()
  const { data, refetch } = useQuery({
    queryKey: ['stitching-sheet', 'challan_master'],
    queryFn: () => api.get('/stitching/sheets/challan_master').then(r => r.data),
  })
  const { data: styles } = useQuery({
    queryKey: ['stitching-sheet', 'style_master'],
    queryFn: () => api.get('/stitching/sheets/style_master').then(r => r.data),
  })
  const styleList = useMemo(() => {
    const s = new Set<string>()
    for (const r of (styles?.rows ?? []) as { Style: string }[]) if (r.Style) s.add(String(r.Style))
    return [...s]
  }, [styles])

  const [form, setForm] = useState({
    Challan_No: '',
    Style: '',
    Party: '',
    Total_Qty: 100,
    Received_Qty: 0,
    Deposit_Rs: 0,
    Rate_Per_Pc: 35,
    Date: todayStr(),
    Delivery_By: '',
  })

  const addMut = useMutation({
    mutationFn: () => api.post('/stitching/challans', form),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['stitching-sheet', 'challan_master'] })
      refetch()
    },
  })

  const rows = (data?.rows ?? []) as Record<string, unknown>[]

  return (
    <div className="space-y-4">
      <Section title="Add challan">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-2 text-xs">
          {(
            [
              ['Challan_No', 'text'],
              ['Style', 'select'],
              ['Party', 'text'],
              ['Total_Qty', 'number'],
              ['Received_Qty', 'number'],
              ['Deposit_Rs', 'number'],
              ['Rate_Per_Pc', 'number'],
              ['Date', 'date'],
              ['Delivery_By', 'text'],
            ] as const
          ).map(([k, kind]) => (
            <label key={k}>
              <span className="font-semibold text-gray-600">{k.replace(/_/g, ' ')}</span>
              {kind === 'select' ? (
                <select
                  className="mt-1 w-full border rounded px-2 py-1"
                  value={form.Style}
                  onChange={e => setForm(f => ({ ...f, Style: e.target.value }))}
                >
                  <option value="">—</option>
                  {styleList.map(s => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              ) : (
                <input
                  type={kind}
                  className="mt-1 w-full border rounded px-2 py-1"
                  value={String(form[k as keyof typeof form] ?? '')}
                  onChange={e =>
                    setForm(f => ({
                      ...f,
                      [k]: kind === 'number' ? +e.target.value : e.target.value,
                    }))
                  }
                />
              )}
            </label>
          ))}
        </div>
        <button
          type="button"
          onClick={() => addMut.mutate()}
          className="mt-3 px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-semibold"
        >
          Add challan
        </button>
      </Section>
      <Section title={`Challan register (${rows.length})`}>
        <DataTable
          rows={rows}
          cols={['Challan_No', 'Style', 'Party', 'Total_Qty', 'Received_Qty', 'Rate_Per_Pc', 'Date']}
        />
      </Section>
    </div>
  )
}

function StyleCostingTab() {
  const [month, setMonth] = useState('All')
  const [style, setStyle] = useState('All')
  const { data } = useQuery({
    queryKey: ['stitching-style-costing', month, style],
    queryFn: () => api.get('/stitching/style-costing', { params: { month, style } }).then(r => r.data),
  })
  const s = data?.summary
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <input className="border rounded px-2 py-1 text-sm" placeholder="Month YYYY-MM or All" value={month} onChange={e => setMonth(e.target.value)} />
        <input className="border rounded px-2 py-1 text-sm" placeholder="Style or All" value={style} onChange={e => setStyle(e.target.value)} />
      </div>
      {s && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
          {[
            ['Challans', s.challans],
            ['Pending', s.pending],
            ['Party value', `₹${Number(s.party_value).toLocaleString()}`],
            ['Expense', `₹${Number(s.actual_expense).toLocaleString()}`],
            ['Net P&L', `₹${Number(s.net_pl).toLocaleString()}`],
          ].map(([l, v]) => (
            <div key={String(l)} className="bg-white border rounded-lg p-3 text-sm">
              <p className="text-gray-500 text-xs">{l}</p>
              <p className="font-bold text-[#2c5aa0]">{v}</p>
            </div>
          ))}
        </div>
      )}
      <Section title="Detail">
        <DataTable
          rows={data?.rows ?? []}
          cols={['Challan_No', 'Style', 'Party', 'Total_Qty', 'Actual_Labour_Rs', 'Party_Value_Rs', 'PL_Rs', 'Margin_%']}
        />
      </Section>
      <Section title="Style roll-up">
        <DataTable rows={data?.style_rollup ?? []} cols={['Style', 'Challans', 'Qty', 'PL', 'Margin_%', 'Result']} />
      </Section>
    </div>
  )
}

function EfficiencyTab() {
  const [from, setFrom] = useState(() => {
    const d = new Date()
    d.setDate(d.getDate() - 7)
    return d.toISOString().slice(0, 10)
  })
  const [to, setTo] = useState(todayStr())
  const { data } = useQuery({
    queryKey: ['stitching-efficiency', from, to],
    queryFn: () => api.get('/stitching/efficiency', { params: { date_from: from, date_to: to } }).then(r => r.data),
  })
  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <input type="date" className="border rounded px-2 py-1 text-sm" value={from} onChange={e => setFrom(e.target.value)} />
        <input type="date" className="border rounded px-2 py-1 text-sm" value={to} onChange={e => setTo(e.target.value)} />
      </div>
      {data?.metrics && (
        <div className="grid grid-cols-3 gap-2 text-sm">
          <div className="bg-white border rounded p-3">Avg eff: <b>{data.metrics.avg_efficiency}%</b></div>
          <div className="bg-white border rounded p-3">Pieces: <b>{data.metrics.total_pieces?.toLocaleString()}</b></div>
          <div className="bg-white border rounded p-3">Value: <b>₹{data.metrics.total_piece_value?.toLocaleString()}</b></div>
        </div>
      )}
      <Section title="Karigar-wise">
        <DataTable rows={data?.karigar_wise ?? []} cols={['Karigar_Name', 'Avg_Eff', 'Pieces', 'Value', 'Grade']} />
      </Section>
      <Section title="Operation-wise">
        <DataTable rows={data?.operation_wise ?? []} cols={['Operation', 'Avg_Eff', 'Pieces', 'Value']} />
      </Section>
    </div>
  )
}

function PayrollTab() {
  const [from, setFrom] = useState(() => {
    const d = new Date()
    d.setDate(d.getDate() - 6)
    return d.toISOString().slice(0, 10)
  })
  const [to, setTo] = useState(todayStr())
  const { data, refetch, isFetching } = useQuery({
    queryKey: ['stitching-payroll', from, to],
    queryFn: () => api.get('/stitching/payroll', { params: { date_from: from, date_to: to } }).then(r => r.data),
    enabled: false,
  })
  return (
    <div className="space-y-4">
      <div className="flex gap-2 items-end">
        <label className="text-xs">
          From <input type="date" className="block border rounded px-2 py-1 mt-1" value={from} onChange={e => setFrom(e.target.value)} />
        </label>
        <label className="text-xs">
          To <input type="date" className="block border rounded px-2 py-1 mt-1" value={to} onChange={e => setTo(e.target.value)} />
        </label>
        <button type="button" onClick={() => refetch()} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm">
          Calculate
        </button>
      </div>
      {isFetching && <p className="text-sm text-gray-500">Calculating…</p>}
      {data && (
        <>
          <p className="text-lg font-bold text-[#2c5aa0]">Total payroll: ₹{Number(data.total_payroll).toLocaleString()}</p>
          <DataTable rows={data.rows ?? []} cols={['E_Code', 'Name', 'Days', 'Hrs', 'Normal', 'OT_Pay', 'Total']} />
        </>
      )}
    </div>
  )
}

function AttendanceTab({ type }: { type: 'karigar' | 'operating' }) {
  const qc = useQueryClient()
  const sheet = type === 'karigar' ? 'karigar_attendance' : 'operating_attendance'
  const { data, refetch } = useQuery({
    queryKey: ['stitching-sheet', sheet],
    queryFn: () => api.get(`/stitching/sheets/${sheet}`).then(r => r.data),
  })
  const { data: em } = useQuery({
    queryKey: ['stitching-sheet', 'employee_master'],
    queryFn: () => api.get('/stitching/sheets/employee_master').then(r => r.data),
  })
  const employees = ((em?.rows ?? []) as { E_Code: string; Name: string; Type: string }[]).filter(
    e => type === 'karigar' ? e.Type === 'Karigar' : e.Type === 'Operating',
  )
  const [form, setForm] = useState({ Date: todayStr(), E_Code: '', In_Punch: '09:00', Out_Punch: '18:00' })
  const saveMut = useMutation({
    mutationFn: () =>
      api.post(type === 'karigar' ? '/stitching/attendance/karigar' : '/stitching/attendance/operating', form),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['stitching-sheet', sheet] })
      refetch()
    },
  })
  return (
    <div className="space-y-4">
      <Section title="Manual entry">
        <div className="grid sm:grid-cols-4 gap-2 text-xs">
          <label>
            Date
            <input type="date" className="w-full border rounded mt-1 px-2 py-1" value={form.Date} onChange={e => setForm(f => ({ ...f, Date: e.target.value }))} />
          </label>
          <label>
            Employee
            <select className="w-full border rounded mt-1 px-2 py-1" value={form.E_Code} onChange={e => setForm(f => ({ ...f, E_Code: e.target.value }))}>
              <option value="">—</option>
              {employees.map(e => (
                <option key={e.E_Code} value={e.E_Code}>{e.E_Code} — {e.Name}</option>
              ))}
            </select>
          </label>
          <label>
            In
            <input className="w-full border rounded mt-1 px-2 py-1" value={form.In_Punch} onChange={e => setForm(f => ({ ...f, In_Punch: e.target.value }))} />
          </label>
          <label>
            Out
            <input className="w-full border rounded mt-1 px-2 py-1" value={form.Out_Punch} onChange={e => setForm(f => ({ ...f, Out_Punch: e.target.value }))} />
          </label>
        </div>
        <button type="button" onClick={() => saveMut.mutate()} className="mt-2 px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm">
          Save attendance
        </button>
      </Section>
      <Section title="Records">
        <DataTable rows={(data?.rows ?? []).slice(-50)} cols={type === 'karigar' ? ['Date', 'E_Code', 'Name', 'Payable_Hrs', 'Total_Pay'] : ['Date', 'E_Code', 'Name', 'Total_Hours', 'Total_Pay']} />
      </Section>
    </div>
  )
}

function MasterTab() {
  const qc = useQueryClient()
  const keys = ['style_master', 'karigar_master', 'employee_master'] as const
  const [active, setActive] = useState<(typeof keys)[number]>('style_master')
  const { data, refetch } = useQuery({
    queryKey: ['stitching-sheet', active],
    queryFn: () => api.get(`/stitching/sheets/${active}`).then(r => r.data),
  })

  const [styleForm, setStyleForm] = useState({ Style: '', Operation: '', Target: 80, Rate_Rs: 3 })
  const [karForm, setKarForm] = useState({ Karigar_ID: '', Name: '', Skill: 'Stitching', Daily_Rate_Rs: 420 })

  const addStyle = useMutation({
    mutationFn: () => api.post('/stitching/master/style-operation', styleForm),
    onSuccess: () => refetch(),
  })
  const addKar = useMutation({
    mutationFn: () => api.post('/stitching/master/karigar', karForm),
    onSuccess: () => {
      refetch()
      qc.invalidateQueries({ queryKey: ['stitching-sheet', 'karigar_master'] })
      qc.invalidateQueries({ queryKey: ['stitching-sheet', 'employee_master'] })
    },
  })

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        {keys.map(k => (
          <button
            key={k}
            type="button"
            onClick={() => setActive(k)}
            className={`text-xs px-3 py-1.5 rounded-lg ${active === k ? 'bg-[#002B5B] text-white' : 'border'}`}
          >
            {k.replace(/_/g, ' ')}
          </button>
        ))}
      </div>
      {active === 'style_master' && (
        <div className="grid sm:grid-cols-4 gap-2 text-xs mb-3">
          {(['Style', 'Operation'] as const).map(k => (
            <label key={k}>
              {k}
              <input className="w-full border rounded mt-1 px-2 py-1" value={styleForm[k]} onChange={e => setStyleForm(f => ({ ...f, [k]: e.target.value }))} />
            </label>
          ))}
          <label>
            Target
            <input type="number" className="w-full border rounded mt-1 px-2 py-1" value={styleForm.Target} onChange={e => setStyleForm(f => ({ ...f, Target: +e.target.value }))} />
          </label>
          <label>
            Rate
            <input type="number" step={0.25} className="w-full border rounded mt-1 px-2 py-1" value={styleForm.Rate_Rs} onChange={e => setStyleForm(f => ({ ...f, Rate_Rs: +e.target.value }))} />
          </label>
          <button type="button" onClick={() => addStyle.mutate()} className="self-end px-3 py-2 bg-[#002B5B] text-white rounded text-xs">
            Add operation
          </button>
        </div>
      )}
      {active === 'karigar_master' && (
        <div className="grid sm:grid-cols-4 gap-2 text-xs mb-3">
          {(['Karigar_ID', 'Name', 'Skill'] as const).map(k => (
            <label key={k}>
              {k}
              <input className="w-full border rounded mt-1 px-2 py-1" value={karForm[k]} onChange={e => setKarForm(f => ({ ...f, [k]: e.target.value }))} />
            </label>
          ))}
          <label>
            Daily rate
            <input type="number" className="w-full border rounded mt-1 px-2 py-1" value={karForm.Daily_Rate_Rs} onChange={e => setKarForm(f => ({ ...f, Daily_Rate_Rs: +e.target.value }))} />
          </label>
          <button type="button" onClick={() => addKar.mutate()} className="self-end px-3 py-2 bg-[#002B5B] text-white rounded text-xs">
            Add karigar
          </button>
        </div>
      )}
      <DataTable rows={data?.rows ?? []} cols={(data?.columns as string[]) ?? []} />
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
      <div className="px-4 py-2 bg-[#2c5aa0] text-white text-sm font-semibold">{title}</div>
      <div className="p-4">{children}</div>
    </section>
  )
}

function DataTable({ rows, cols }: { rows: Record<string, unknown>[]; cols: string[] }) {
  if (!rows.length) return <p className="text-sm text-gray-400 text-center py-4">No rows</p>
  const useCols = cols.length ? cols.filter(c => c in rows[0] || rows[0][c] !== undefined) : Object.keys(rows[0] ?? {})
  return (
    <div className="overflow-x-auto max-h-[min(420px,50vh)]">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-gray-50 border-b">
          <tr>
            {useCols.map(c => (
              <th key={c} className="text-left px-2 py-2 font-semibold text-gray-600 whitespace-nowrap">
                {c.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-gray-50 hover:bg-gray-50/80">
              {useCols.map(c => (
                <td key={c} className="px-2 py-1.5 whitespace-nowrap">
                  {String(r[c] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
