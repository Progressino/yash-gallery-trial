import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'
import { useStitchingAdmin } from '../lib/stitchingAdmin'
import { useAuth } from '../store/auth'

type TabId =
  | 'dashboard'
  | 'production'
  | 'challan'
  | 'style'
  | 'efficiency'
  | 'payroll'
  | 'attendance'
  | 'operating'
  | 'performance'
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
  { id: 'performance', label: '🌟 Performance' },
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

export default function StitchingCosting({ karigarOnly = false }: { karigarOnly?: boolean }) {
  const qc = useQueryClient()
  const admin = useStitchingAdmin()
  const authUser = useAuth(s => s.user)
  const lockedKarigarId = karigarOnly ? authUser?.karigar_id || '' : ''
  const [tab, setTab] = useState<TabId>(karigarOnly ? 'production' : 'dashboard')
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

  const visibleTabs = karigarOnly ? TABS.filter(t => t.id === 'production') : TABS

  return (
    <div className={`space-y-4 ${karigarOnly ? '' : 'max-w-[1600px]'}`}>
      {!karigarOnly && (
      <div className="rounded-xl bg-gradient-to-br from-[#1a3a5c] via-[#2c5aa0] to-[#1e7ed4] text-white p-5 shadow-md">
        <h1 className="text-xl font-bold">🧵 Stitching Costing — Yash Gallery</h1>
        <p className="text-sm opacity-90 mt-1">
          Karigar tracking · Challan management · Style costing · Payroll
          {status?.gsheet?.available ? ' · Google Sheets connected' : ' · Local database'}
        </p>
        </div>
      )}

      {msg && (
        <div
          className={`text-sm px-4 py-2 rounded-lg border ${
            msg.type === 'ok' ? 'bg-green-50 text-green-800 border-green-200' : 'bg-rose-50 text-rose-800 border-rose-200'
          }`}
        >
          {msg.text}
        </div>
      )}

      {!karigarOnly && (
        <>
          <BackupRestoreBar
            gsheetAvailable={!!status?.gsheet?.available}
            syncFromPending={syncFrom.isPending}
            syncToPending={syncTo.isPending}
            onPull={() => syncFrom.mutate()}
            onPush={() => syncTo.mutate()}
            onFlash={flash}
            onRestored={() => qc.invalidateQueries({ queryKey: ['stitching'] })}
          />
          <AdminUnlockPanel admin={admin} onFlash={flash} />
        </>
      )}

      {!karigarOnly && visibleTabs.length > 1 && (
      <div className="flex flex-wrap gap-1 border-b border-gray-200 pb-1">
        {visibleTabs.map(t => (
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
      )}

      {tab === 'dashboard' && !karigarOnly && <DashboardTab />}
      {tab === 'production' && (
        <ProductionTab
          hours={status?.hours ?? []}
          admin={admin}
          karigarOnly={karigarOnly}
          lockedKarigarId={lockedKarigarId}
          onSaved={() => qc.invalidateQueries({ queryKey: ['stitching-dashboard'] })}
        />
      )}
      {!karigarOnly && tab === 'challan' && <ChallanTab />}
      {!karigarOnly && tab === 'style' && <StyleCostingTab />}
      {!karigarOnly && tab === 'efficiency' && <EfficiencyTab />}
      {!karigarOnly && tab === 'payroll' && <PayrollTab />}
      {!karigarOnly && tab === 'attendance' && <AttendanceTab type="karigar" />}
      {!karigarOnly && tab === 'operating' && <AttendanceTab type="operating" />}
      {!karigarOnly && tab === 'performance' && <PerformanceTab />}
      {!karigarOnly && tab === 'master' && <MasterTab admin={admin} />}
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

type AdminApi = ReturnType<typeof useStitchingAdmin>

type KarigarRow = { Karigar_ID: string; Name: string; Skill?: string; Daily_Rate_Rs?: number }

type SearchSelectOption = {
  value: string
  primary: string
  secondary?: string
  haystack: string
}

function SearchSelect({
  label,
  placeholder,
  value,
  onChange,
  options,
  disabled,
  emptyMessage = 'No matches',
  hint,
}: {
  label: string
  placeholder: string
  value: string
  onChange: (v: string) => void
  options: SearchSelectOption[]
  disabled?: boolean
  emptyMessage?: string
  hint?: string
}) {
  const [search, setSearch] = useState('')
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)

  const selected = options.find(o => o.value === value)

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return options.slice(0, 80)
    return options.filter(o => o.haystack.includes(q)).slice(0, 80)
  }, [options, search])

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [])

  const displayValue = open
    ? search
    : selected
      ? selected.secondary
        ? `${selected.primary} — ${selected.secondary}`
        : selected.primary
      : search

  return (
    <div ref={wrapRef} className={`relative ${disabled ? 'opacity-60' : ''}`}>
      <span className="font-semibold text-gray-700 text-xs">{label}</span>
      {hint && !disabled && <p className="text-[10px] text-gray-400 mt-0.5">{hint}</p>}
      <input
        type="search"
        disabled={disabled}
        placeholder={placeholder}
        className="mt-1 w-full border rounded-lg px-3 py-2.5 text-sm touch-manipulation disabled:bg-gray-50"
        value={displayValue}
        onChange={e => {
          setSearch(e.target.value)
          setOpen(true)
          if (!e.target.value) onChange('')
        }}
        onFocus={() => {
          if (disabled) return
          setOpen(true)
          setSearch('')
        }}
      />
      {open && !disabled && (
        <ul className="absolute z-30 left-0 right-0 mt-1 max-h-56 overflow-y-auto bg-white border rounded-lg shadow-lg">
          {filtered.length === 0 ? (
            <li className="px-3 py-3 text-sm text-gray-400">{emptyMessage}</li>
          ) : (
            filtered.map(o => (
              <li key={o.value}>
                <button
                  type="button"
                  className="w-full text-left px-3 py-3 text-sm hover:bg-blue-50 active:bg-blue-100 border-b border-gray-50 last:border-0 touch-manipulation min-h-[44px]"
                  onClick={() => {
                    onChange(o.value)
                    setSearch('')
                    setOpen(false)
                  }}
                >
                  <span className="font-mono font-semibold text-[#002B5B]">{o.primary}</span>
                  {o.secondary && <span className="text-gray-500 text-xs block mt-0.5 sm:inline sm:ml-2 sm:mt-0">{o.secondary}</span>}
                </button>
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  )
}

function KarigarSearchSelect({
  karigars,
  value,
  onChange,
}: {
  karigars: KarigarRow[]
  value: string
  onChange: (id: string) => void
}) {
  const [search, setSearch] = useState('')
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return karigars
    return karigars.filter(
      k =>
        String(k.Karigar_ID).toLowerCase().includes(q) ||
        String(k.Name).toLowerCase().includes(q),
    )
  }, [karigars, search])

  const selected = karigars.find(k => String(k.Karigar_ID) === value)

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [])

  return (
    <div ref={wrapRef} className="relative">
      <span className="font-semibold text-gray-700 text-xs">Karigar</span>
      <input
        type="search"
        placeholder="Search name or ID…"
        className="mt-1 w-full border rounded-lg px-3 py-2.5 text-sm touch-manipulation"
        value={open ? search : selected ? `${selected.Karigar_ID} — ${selected.Name}` : search}
        onChange={e => {
          setSearch(e.target.value)
          setOpen(true)
          if (!e.target.value) onChange('')
        }}
        onFocus={() => {
          setOpen(true)
          setSearch('')
        }}
      />
      {open && (
        <ul className="absolute z-30 left-0 right-0 mt-1 max-h-56 overflow-y-auto bg-white border rounded-lg shadow-lg">
          {filtered.length === 0 ? (
            <li className="px-3 py-3 text-sm text-gray-400">No karigar found</li>
          ) : (
            filtered.map(k => (
              <li key={k.Karigar_ID}>
                <button
                  type="button"
                  className="w-full text-left px-3 py-3 text-sm hover:bg-blue-50 active:bg-blue-100 border-b border-gray-50 last:border-0 touch-manipulation min-h-[44px]"
                  onClick={() => {
                    onChange(String(k.Karigar_ID))
                    setSearch('')
                    setOpen(false)
                  }}
                >
                  <span className="font-mono font-semibold text-[#002B5B]">{k.Karigar_ID}</span>
                  <span className="text-gray-600 ml-2">{k.Name}</span>
                  {k.Skill && <span className="text-gray-400 text-xs ml-1">· {k.Skill}</span>}
                </button>
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  )
}

function computeEntrySummary(
  hourState: Record<string, { operation: string; pieces: number }>,
  hours: HourDef[],
  operations: { op: string; target: number; rate: number }[],
) {
  const opTotals: Record<string, { pieces: number; value: number }> = {}
  let totalPcs = 0
  for (const h of hours) {
    if (h.col === 'H_13_14') continue
    const st = hourState[h.col]
    if (!st?.operation || !st.pieces) continue
    const meta = operations.find(o => o.op === st.operation)
    if (!meta) continue
    totalPcs += st.pieces
    if (!opTotals[st.operation]) opTotals[st.operation] = { pieces: 0, value: 0 }
    opTotals[st.operation].pieces += st.pieces
    opTotals[st.operation].value += st.pieces * meta.rate
  }
  const totalValue = Object.values(opTotals).reduce((s, d) => s + d.value, 0)
  const totalBudget = Object.entries(opTotals).reduce((s, [op]) => {
    const meta = operations.find(o => o.op === op)
    return s + (meta ? meta.rate * meta.target : 0)
  }, 0)
  return { totalPcs, totalValue, totalBudget, pl: totalValue - totalBudget, opTotals }
}

function ProductionTab({
  hours,
  admin,
  onSaved,
  karigarOnly = false,
  lockedKarigarId = '',
}: {
  hours: HourDef[]
  admin: AdminApi
  onSaved: () => void
  karigarOnly?: boolean
  lockedKarigarId?: string
}) {
  const [entryDate, setEntryDate] = useState(todayStr())
  const [karigarId, setKarigarId] = useState(lockedKarigarId || '')
  const [style, setStyle] = useState('')
  const [challanNo, setChallanNo] = useState('')
  const [hourState, setHourState] = useState<Record<string, { operation: string; pieces: number }>>({})
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [saveMessage, setSaveMessage] = useState('')
  const [autoSaveEnabled, setAutoSaveEnabled] = useState(true)
  const skipAutoSaveRef = useRef(false)
  const qc = useQueryClient()

  useEffect(() => {
    if (lockedKarigarId) setKarigarId(lockedKarigarId)
  }, [lockedKarigarId])

  const { data: karigarSheet } = useQuery({
    queryKey: ['stitching-sheet', 'karigar_master'],
    queryFn: () => api.get('/stitching/sheets/karigar_master').then(r => r.data),
  })
  const { data: styleSheet, refetch: refetchStyles } = useQuery({
    queryKey: ['stitching-sheet', 'style_master'],
    queryFn: () => api.get('/stitching/sheets/style_master').then(r => r.data),
  })
  const [rateEdits, setRateEdits] = useState<Record<string, { Target: number; Rate_Rs: number }>>({})
  const { data: challanSheet } = useQuery({
    queryKey: ['stitching-sheet', 'challan_master'],
    queryFn: () => api.get('/stitching/sheets/challan_master').then(r => r.data),
  })

  const karigars = (karigarSheet?.rows ?? []) as KarigarRow[]
  const styles = useMemo(() => {
    const s = new Set<string>()
    for (const r of (styleSheet?.rows ?? []) as { Style: string }[]) {
      if (r.Style) s.add(String(r.Style))
    }
    return [...s]
  }, [styleSheet])
  type ChallanRow = {
    Challan_No: string
    Style: string
    Party?: string
    Total_Qty?: number
    Received_Qty?: number
  }

  const allChallans = useMemo(
    () => (challanSheet?.rows ?? []) as ChallanRow[],
    [challanSheet],
  )

  const challansForStyle = useMemo(
    () => allChallans.filter(c => style && String(c.Style) === style),
    [allChallans, style],
  )

  const styleOptions = useMemo<SearchSelectOption[]>(
    () =>
      styles
        .slice()
        .sort((a, b) => a.localeCompare(b))
        .map(s => ({
          value: s,
          primary: s,
          haystack: s.toLowerCase(),
        })),
    [styles],
  )

  const challanOptions = useMemo<SearchSelectOption[]>(
    () =>
      challansForStyle.map(c => {
        const qty = Number(c.Total_Qty) || 0
        const recv = Number(c.Received_Qty) || 0
        const no = String(c.Challan_No)
        const party = String(c.Party || '')
        return {
          value: no,
          primary: no,
          secondary: `${party || '—'} · Qty ${qty} · Recv ${recv}`,
          haystack: `${no} ${party} ${c.Style}`.toLowerCase(),
        }
      }),
    [challansForStyle],
  )

  const operations = useMemo(() => {
    return ((styleSheet?.rows ?? []) as { Style: string; Operation: string; Target: number; Rate_Rs: number }[])
      .filter(r => r.Style === style)
      .map(r => {
        const key = `${r.Style}::${r.Operation}`
        const ed = rateEdits[key]
        return {
          op: r.Operation,
          target: ed?.Target ?? Number(r.Target),
          rate: ed?.Rate_Rs ?? Number(r.Rate_Rs),
        }
      })
  }, [styleSheet, style, rateEdits])

  const saveRatesMut = useMutation({
    mutationFn: async () => {
      const pw = admin.adminPassword()
      if (!pw) throw new Error('Unlock admin first')
      for (const op of operations) {
        const key = `${style}::${op.op}`
        const ed = rateEdits[key]
        if (!ed) continue
        await api.patch('/stitching/master/style-operation', {
          Style: style,
          Operation: op.op,
          Target: ed.Target,
          Rate_Rs: ed.Rate_Rs,
          admin_password: pw,
        })
      }
    },
    onSuccess: () => {
      setRateEdits({})
      refetchStyles()
      alert('Targets and rates updated')
    },
    onError: (e: Error) => alert(e.message || 'Update failed'),
  })

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
    skipAutoSaveRef.current = true
    setHourState(next)
  }, [entryDate, karigarId, challanNo, style, hours])

  useEffect(() => {
    void loadEntry()
  }, [loadEntry])

  const { data: entryReports, refetch: refetchReports } = useQuery({
    queryKey: ['stitching-pe-reports', entryDate, karigarId],
    queryFn: () =>
      api
        .get('/stitching/production-entry/reports', {
          params: { date: entryDate, karigar_id: karigarId || undefined },
        })
        .then(r => r.data),
    enabled: !!entryDate,
  })

  const liveSummary = useMemo(
    () => computeEntrySummary(hourState, hours, operations),
    [hourState, hours, operations],
  )

  const buildHourEntries = useCallback(
    () =>
      hours
        .filter(h => h.col !== 'H_13_14')
        .map(h => ({
          hour_col: h.col,
          operation: hourState[h.col]?.operation || '',
          pieces: hourState[h.col]?.pieces || 0,
        })),
    [hours, hourState],
  )

  const doSave = useCallback(
    async (silent: boolean) => {
      if (!karigarId || !challanNo || !style) return
      if (liveSummary.totalPcs <= 0) return
      const k = karigars.find(x => String(x.Karigar_ID) === karigarId)
      setSaveStatus('saving')
      try {
        const r = await api.post('/stitching/production-entry', {
          date: entryDate,
          karigar_id: karigarId,
          karigar_name: k?.Name ?? karigarId,
          challan_no: challanNo,
          style,
          hour_entries: buildHourEntries(),
        })
        if (r.data.ok) {
          setSaveStatus('saved')
          setSaveMessage(r.data.message || 'Saved')
          onSaved()
          refetchReports()
          qc.invalidateQueries({ queryKey: ['stitching-dashboard'] })
          if (!silent) alert(r.data.message || 'Saved')
        } else {
          setSaveStatus('error')
          if (!silent) alert(r.data.message || 'Save failed')
        }
      } catch {
        setSaveStatus('error')
        if (!silent) alert('Save failed')
      }
    },
    [
      karigarId,
      challanNo,
      style,
      liveSummary.totalPcs,
      karigars,
      entryDate,
      buildHourEntries,
      onSaved,
      refetchReports,
      qc,
    ],
  )

  useEffect(() => {
    if (!autoSaveEnabled || skipAutoSaveRef.current) {
      skipAutoSaveRef.current = false
      return
    }
    if (!karigarId || !challanNo || !style || liveSummary.totalPcs <= 0) return
    const t = setTimeout(() => void doSave(true), 2000)
    return () => clearTimeout(t)
  }, [hourState, entryDate, karigarId, challanNo, style, autoSaveEnabled, liveSummary.totalPcs, doSave])

  const saveMut = useMutation({
    mutationFn: () => doSave(false),
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
          <input type="date" className="mt-1 w-full border rounded-lg px-3 py-2.5 text-sm touch-manipulation" value={entryDate} onChange={e => setEntryDate(e.target.value)} />
        </label>
        {lockedKarigarId ? (
          <label className="text-xs block">
            <span className="font-semibold text-gray-700">Karigar</span>
            <div className="mt-1 w-full border rounded-lg px-3 py-2.5 text-sm bg-gray-50 font-mono text-[#002B5B]">
              {karigars.find(k => String(k.Karigar_ID) === lockedKarigarId)?.Name || lockedKarigarId}
              <span className="text-gray-500 font-normal ml-1">({lockedKarigarId})</span>
            </div>
          </label>
        ) : (
          <KarigarSearchSelect karigars={karigars} value={karigarId} onChange={setKarigarId} />
        )}
        <SearchSelect
          label="Style / SKU"
          placeholder="Type to search SKU, e.g. 1065YKBLUE…"
          value={style}
          onChange={v => {
            setStyle(v)
            setChallanNo('')
          }}
          options={styleOptions}
          emptyMessage="No style found — check Master Data"
        />
        <SearchSelect
          label="Challan"
          placeholder={style ? 'Search challan no. or party…' : 'Select style first'}
          value={challanNo}
          onChange={setChallanNo}
          options={challanOptions}
          disabled={!style}
          emptyMessage={style ? 'No challan for this style — add in Challans tab' : 'Select a style first'}
          hint={style ? `${challansForStyle.length} challan(s) for ${style}` : undefined}
        />
      </div>

      {karigarId && karigars.find(k => String(k.Karigar_ID) === karigarId) && (
        <div className="text-xs bg-blue-50 border border-blue-100 rounded-lg px-3 py-2 text-blue-900">
          👤 <b>{karigars.find(k => String(k.Karigar_ID) === karigarId)?.Name}</b>
          {karigars.find(k => String(k.Karigar_ID) === karigarId)?.Skill
            ? ` · ${karigars.find(k => String(k.Karigar_ID) === karigarId)?.Skill}`
            : ''}
        </div>
      )}

      {style && operations.length > 0 && !karigarOnly && (
        <Section title="Style operations — targets & rates">
          {!admin.unlocked ? (
            <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
              🔐 <strong>Locked</strong> — Target and rate are read-only. Use <strong>Admin unlock</strong> above to edit.
            </p>
          ) : (
            <p className="text-xs text-green-800 bg-green-50 border border-green-200 rounded-lg px-3 py-2">
              ✅ <strong>Unlocked</strong> — You can edit targets and rates for this style.
            </p>
          )}
          <div className="overflow-x-auto mt-3">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 border-b">
                  <th className="px-2 py-2 text-left">Operation</th>
                  <th className="px-2 py-2 text-right">Target</th>
                  <th className="px-2 py-2 text-right">Rate ₹/pc</th>
                </tr>
              </thead>
              <tbody>
                {operations.map(op => {
                  const key = `${style}::${op.op}`
                  const ed = rateEdits[key] ?? { Target: op.target, Rate_Rs: op.rate }
                  return (
                    <tr key={op.op} className="border-b border-gray-50">
                      <td className="px-2 py-1.5">{op.op}</td>
                      <td className="px-2 py-1.5 text-right">
                        {admin.unlocked ? (
                          <input
                            type="number"
                            className="w-20 border rounded px-1 py-0.5 text-right"
                            value={ed.Target}
                            onChange={e =>
                              setRateEdits(prev => ({
                                ...prev,
                                [key]: { ...ed, Target: +e.target.value || 0 },
                              }))
                            }
                          />
                        ) : (
                          op.target
                        )}
                      </td>
                      <td className="px-2 py-1.5 text-right">
                        {admin.unlocked ? (
                          <input
                            type="number"
                            step={0.25}
                            className="w-20 border rounded px-1 py-0.5 text-right"
                            value={ed.Rate_Rs}
                            onChange={e =>
                              setRateEdits(prev => ({
                                ...prev,
                                [key]: { ...ed, Rate_Rs: +e.target.value || 0 },
                              }))
                            }
                          />
                        ) : (
                          op.rate
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {admin.unlocked && Object.keys(rateEdits).length > 0 && (
            <button
              type="button"
              onClick={() => saveRatesMut.mutate()}
              disabled={saveRatesMut.isPending}
              className="mt-3 text-xs px-4 py-2 bg-violet-700 text-white rounded-lg font-semibold"
            >
              {saveRatesMut.isPending ? 'Saving…' : '💾 Save target & rate changes'}
            </button>
          )}
        </Section>
      )}

      <Section title="Hour-wise entry">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <label className="flex items-center gap-2 text-xs text-gray-600 touch-manipulation">
            <input type="checkbox" checked={autoSaveEnabled} onChange={e => setAutoSaveEnabled(e.target.checked)} />
            Auto-save (2 sec)
          </label>
          {saveStatus === 'saving' && <span className="text-xs text-amber-600 font-medium">Saving…</span>}
          {saveStatus === 'saved' && <span className="text-xs text-green-600 font-medium">✓ {saveMessage}</span>}
          {saveStatus === 'error' && <span className="text-xs text-red-600">Save failed</span>}
        </div>

        <div className="space-y-2 md:hidden mb-3">
          {hours.map(h => {
            if (h.col === 'H_13_14') {
              return (
                <div key={h.col} className="rounded-lg bg-gray-50 border border-dashed px-3 py-2 text-xs text-gray-400 italic text-center">
                  {h.label} — Lunch
                </div>
              )
            }
            const st = hourState[h.col] ?? { operation: '', pieces: 0 }
            const opMeta = operations.find(o => o.op === st.operation)
            const eff = opMeta && st.pieces > 0 ? Math.round((st.pieces / Math.max(opMeta.target, 1)) * 100) : null
            return (
              <div key={h.col} className="rounded-xl border bg-white p-3 shadow-sm">
                <div className="flex justify-between mb-2">
                  <span className="font-bold text-[#2c5aa0]">{h.label}</span>
                  {eff != null && (
                    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${eff >= 100 ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
                      {eff}%
                    </span>
                  )}
                </div>
                <select
                  className="w-full border rounded-lg px-3 py-2.5 text-sm mb-2 touch-manipulation"
                  value={st.operation}
                  onChange={e => setHour(h.col, { operation: e.target.value })}
                >
                  {opOptions.map(o => (
                    <option key={o || '_'} value={o}>{o || '— Operation —'}</option>
                  ))}
                </select>
                <div className="flex gap-2 items-center">
                  <input
                    type="number"
                    inputMode="numeric"
                    min={0}
                    className="flex-1 border rounded-lg px-3 py-3 text-lg font-semibold touch-manipulation"
                    value={st.pieces || ''}
                    onChange={e => setHour(h.col, { pieces: +e.target.value || 0 })}
                  />
                  <span className="text-xs text-gray-400 shrink-0">tgt {opMeta?.target ?? '—'}</span>
                </div>
              </div>
            )
          })}
        </div>

        <div className="overflow-x-auto hidden md:block">
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
        {liveSummary.totalPcs > 0 && (
          <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-2">
            <div className="bg-white border rounded-lg p-3 text-center">
              <p className="text-[10px] uppercase text-gray-500">Pieces</p>
              <p className="text-lg font-bold text-[#2c5aa0]">{liveSummary.totalPcs}</p>
            </div>
            <div className="bg-white border rounded-lg p-3 text-center">
              <p className="text-[10px] uppercase text-gray-500">Actual ₹</p>
              <p className="text-lg font-bold text-green-700">₹{liveSummary.totalValue.toFixed(2)}</p>
            </div>
            <div className="bg-white border rounded-lg p-3 text-center col-span-2 sm:col-span-1">
              <p className="text-[10px] uppercase text-gray-500">P&amp;L</p>
              <p className={`text-lg font-bold ${liveSummary.pl >= 0 ? 'text-amber-700' : 'text-green-700'}`}>
                ₹{liveSummary.pl.toFixed(2)}
              </p>
            </div>
          </div>
        )}

        <button
          type="button"
          onClick={() => saveMut.mutate()}
          disabled={saveMut.isPending || !karigarId || !challanNo || !style || liveSummary.totalPcs <= 0}
          className="mt-4 w-full py-3.5 rounded-xl bg-[#002B5B] text-white font-semibold text-sm disabled:opacity-50 touch-manipulation shadow-md"
        >
          {saveMut.isPending ? 'Saving…' : '💾 Save now'}
        </button>
      </Section>

      {(entryReports?.recent_saves?.length > 0 || entryReports?.history?.length > 0) && (
        <Section title={karigarId ? "Today's saves — this karigar" : "Today's saves"}>
          <div className="space-y-2">
            {(entryReports.recent_saves?.length ? entryReports.recent_saves : entryReports.history)
              .slice(0, 8)
              .map((row: Record<string, unknown>, i: number) => (
                <div key={i} className="text-xs bg-blue-50 border-l-4 border-[#2c5aa0] rounded-r-lg px-3 py-2.5">
                  {String(row.Save_Time || '—')} · <b>{String(row.Karigar_Name)}</b> · {String(row.Challan_No)} /{' '}
                  {String(row.Operation)} · <b>{String(row.Total_Pieces)} pcs</b>
                </div>
              ))}
          </div>
        </Section>
      )}

      {entryReports?.report1?.length > 0 && (
        <Section title="Report 1 — Production summary">
          <DataTable
            rows={entryReports.report1}
            cols={['Karigar_Name', 'Challan_No', 'Operation', 'Working_Hours', 'Total_Pieces', 'Efficiency_%', 'PL_Rs']}
          />
        </Section>
      )}

      {entryReports?.report2_summary?.length > 0 && (
        <Section title="Report 2 — Salary vs piece value">
          <DataTable
            rows={entryReports.report2_summary}
            cols={['Karigar', 'Operation', 'Hours_Worked', 'Total_Pieces', 'Total_Net_PL', 'Result']}
          />
          {entryReports.grand_total && (
            <p className="mt-2 text-xs rounded-lg px-3 py-2 bg-gray-50 border text-gray-700">
              Grand net P&amp;L: ₹{entryReports.grand_total.total_net_pl}
            </p>
          )}
        </Section>
      )}

      {entryReports?.report2_hourly?.length > 0 && (
        <Section title="Report 2 — Hour-wise detail">
          <details className="text-xs">
            <summary className="cursor-pointer py-3 font-medium text-[#2c5aa0] touch-manipulation min-h-[44px]">
              Show hour-wise rows ({entryReports.report2_hourly.length})
            </summary>
            <div className="mt-2 max-h-64 overflow-auto">
              <DataTable
                rows={entryReports.report2_hourly}
                cols={['Karigar', 'Hour', 'Operation', 'Pieces_Done', 'Actual_Piece_Val_Rs', 'Net_PL_Rs', 'Status']}
              />
            </div>
          </details>
        </Section>
      )}
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

function MasterTab({ admin: _admin }: { admin: AdminApi }) {
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

function BackupRestoreBar({
  gsheetAvailable,
  syncFromPending,
  syncToPending,
  onPull,
  onPush,
  onFlash,
  onRestored,
}: {
  gsheetAvailable: boolean
  syncFromPending: boolean
  syncToPending: boolean
  onPull: () => void
  onPush: () => void
  onFlash: (type: 'ok' | 'err', text: string) => void
  onRestored: () => void
}) {
  const zipRef = useRef<HTMLInputElement>(null)
  const [restorePending, setRestorePending] = useState(false)

  const handleExport = async () => {
    try {
      const res = await api.get('/stitching/export-zip', { responseType: 'blob' })
      const url = URL.createObjectURL(res.data)
      const a = document.createElement('a')
      a.href = url
      a.download = `stitching_${todayStr()}.zip`
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      onFlash('err', 'Export failed')
    }
  }

  const handleRestore = async (file: File) => {
    if (!window.confirm('Replace all stitching data with this ZIP? This cannot be undone.')) return
    setRestorePending(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const { data } = await api.post('/stitching/import-zip', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      onFlash('ok', data.message || 'Restored')
      onRestored()
    } catch {
      onFlash('err', 'Restore failed')
    } finally {
      setRestorePending(false)
      if (zipRef.current) zipRef.current.value = ''
    }
  }

  return (
    <div className="rounded-xl border border-blue-100 bg-blue-50/50 p-4 space-y-3">
      <p className="text-xs font-semibold text-[#1a3a5c]">💾 Backup & restore</p>
      <p className="text-xs text-gray-600">Data is stored on the server. Export a ZIP backup or restore from a previous export.</p>
      <div className="flex flex-wrap gap-2 items-center">
        <button
          type="button"
          onClick={onPull}
          disabled={syncFromPending || !gsheetAvailable}
          className="text-xs px-3 py-1.5 rounded-lg border border-[#2c5aa0] text-[#2c5aa0] hover:bg-white disabled:opacity-40"
        >
          {syncFromPending ? 'Loading…' : '↻ Pull from Google Sheets'}
        </button>
        <button
          type="button"
          onClick={onPush}
          disabled={syncToPending || !gsheetAvailable}
          className="text-xs px-3 py-1.5 rounded-lg border border-[#2c5aa0] text-[#2c5aa0] hover:bg-white disabled:opacity-40"
        >
          {syncToPending ? 'Saving…' : '↑ Push to Google Sheets'}
        </button>
        <button
          type="button"
          onClick={() => void handleExport()}
          className="text-xs px-3 py-1.5 rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
        >
          📦 Export ZIP
        </button>
        <input
          ref={zipRef}
          type="file"
          accept=".zip"
          className="hidden"
          onChange={e => {
            const f = e.target.files?.[0]
            if (f) void handleRestore(f)
          }}
        />
        <button
          type="button"
          onClick={() => zipRef.current?.click()}
          disabled={restorePending}
          className="text-xs px-3 py-1.5 rounded-lg border border-amber-400 bg-amber-50 text-amber-900 hover:bg-amber-100 disabled:opacity-50"
        >
          {restorePending ? 'Restoring…' : '📂 Restore from ZIP'}
        </button>
      </div>
    </div>
  )
}

function AdminUnlockPanel({
  admin,
  onFlash,
}: {
  admin: AdminApi
  onFlash: (type: 'ok' | 'err', text: string) => void
}) {
  const [pw, setPw] = useState('')
  const [showChange, setShowChange] = useState(false)
  const [cur, setCur] = useState('')
  const [n1, setN1] = useState('')
  const [n2, setN2] = useState('')

  const tryUnlock = async () => {
    try {
      const data = await admin.unlock(pw)
      if (data.ok) {
        onFlash('ok', 'Admin unlocked — you can edit targets and rates')
        setPw('')
      } else {
        onFlash('err', data.message || 'Wrong password')
      }
    } catch {
      onFlash('err', 'Unlock failed')
    }
  }

  const changePw = async () => {
    try {
      const { data } = await api.post('/stitching/admin/change-password', {
        current: cur,
        new_password: n1,
        confirm: n2,
      })
      onFlash('ok', data.message || 'Password changed')
      setCur('')
      setN1('')
      setN2('')
      setShowChange(false)
    } catch (e: unknown) {
      const msg =
        e && typeof e === 'object' && 'response' in e
          ? String((e as { response?: { data?: { detail?: string } } }).response?.data?.detail ?? 'Failed')
          : 'Failed'
      onFlash('err', msg)
    }
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 space-y-3">
      <p className="text-xs font-semibold text-gray-700">🔐 Admin — unlock targets & rates</p>
      {admin.unlocked ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-green-700 bg-green-50 border border-green-200 px-3 py-1.5 rounded-lg">
            ✅ Unlocked — Target & rate edits enabled on Production Entry
          </span>
          <button type="button" onClick={admin.lock} className="text-xs px-3 py-1.5 rounded-lg border border-gray-300">
            🔒 Lock
          </button>
          <button type="button" onClick={() => setShowChange(s => !s)} className="text-xs px-3 py-1.5 rounded-lg border border-gray-300">
            Change password
          </button>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2 items-center">
          <span className="text-xs text-amber-800 bg-amber-50 border border-amber-200 px-3 py-1.5 rounded-lg">
            Locked — default password: admin123
          </span>
          <input
            type="password"
            placeholder="Admin password"
            value={pw}
            onChange={e => setPw(e.target.value)}
            className="text-xs border rounded px-2 py-1.5 w-40"
          />
          <button type="button" onClick={() => void tryUnlock()} className="text-xs px-3 py-1.5 rounded-lg bg-[#002B5B] text-white font-semibold">
            🔓 Unlock
          </button>
        </div>
      )}
      {showChange && (
        <div className="grid sm:grid-cols-3 gap-2 text-xs pt-2 border-t">
          <label>
            Current
            <input type="password" className="w-full border rounded mt-1 px-2 py-1" value={cur} onChange={e => setCur(e.target.value)} />
          </label>
          <label>
            New
            <input type="password" className="w-full border rounded mt-1 px-2 py-1" value={n1} onChange={e => setN1(e.target.value)} />
          </label>
          <label>
            Confirm
            <input type="password" className="w-full border rounded mt-1 px-2 py-1" value={n2} onChange={e => setN2(e.target.value)} />
          </label>
          <button type="button" onClick={() => void changePw()} className="self-end px-3 py-2 bg-violet-700 text-white rounded text-xs sm:col-span-3 max-w-xs">
            Save new password
          </button>
        </div>
      )}
    </div>
  )
}

function PerformanceTab() {
  const [from, setFrom] = useState(() => {
    const d = new Date()
    d.setDate(d.getDate() - 29)
    return d.toISOString().slice(0, 10)
  })
  const [to, setTo] = useState(todayStr())
  const { data, refetch, isFetching } = useQuery({
    queryKey: ['stitching-performance', from, to],
    queryFn: () => api.get('/stitching/performance', { params: { date_from: from, date_to: to } }).then(r => r.data),
    enabled: false,
  })

  return (
    <div className="space-y-4">
      <Section title="Employee performance — piece value vs salary">
        <p className="text-xs text-gray-600 mb-3">
          Compares production piece-value to karigar attendance pay over the selected period. Requires both production log and attendance with salary.
        </p>
        <div className="flex flex-wrap gap-2 items-end mb-4">
          <label className="text-xs">
            From
            <input type="date" className="block border rounded mt-1 px-2 py-1" value={from} onChange={e => setFrom(e.target.value)} />
          </label>
          <label className="text-xs">
            To
            <input type="date" className="block border rounded mt-1 px-2 py-1" value={to} onChange={e => setTo(e.target.value)} />
          </label>
          <button type="button" onClick={() => refetch()} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm">
            Run report
          </button>
        </div>
        {isFetching && <p className="text-sm text-gray-500">Loading…</p>}
        {data && !data.ok && <p className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">{data.message}</p>}
        {data?.ok && data.summary && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
            <div className="bg-white border rounded-lg p-3">
              <p className="text-xs text-gray-500">Total piece value</p>
              <p className="text-lg font-bold text-[#2c5aa0]">₹{Number(data.summary.total_piece_value).toLocaleString()}</p>
            </div>
            <div className="bg-white border rounded-lg p-3">
              <p className="text-xs text-gray-500">Total salary paid</p>
              <p className="text-lg font-bold text-gray-800">₹{Number(data.summary.total_salary).toLocaleString()}</p>
            </div>
            <div className="bg-white border rounded-lg p-3">
              <p className="text-xs text-gray-500">Net surplus</p>
              <p className={`text-lg font-bold ${Number(data.summary.net_surplus) >= 0 ? 'text-green-700' : 'text-red-600'}`}>
                ₹{Number(data.summary.net_surplus).toLocaleString()}
              </p>
            </div>
          </div>
        )}
        {data?.ok && (data.rows?.length ?? 0) > 0 && (
          <DataTable
            rows={data.rows}
            cols={['E_Code', 'Name', 'Days', 'Hrs', 'Salary', 'Total_Pieces', 'Piece_Value', 'Surplus', 'ROI_%', 'Avg_Eff', 'Grade']}
          />
        )}
      </Section>
    </div>
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
