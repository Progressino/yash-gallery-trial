import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import axios from 'axios'
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

  const visibleTabs = karigarOnly ? TABS.filter(t => t.id === 'production') : TABS

  return (
    <div className={`space-y-4 min-w-0 ${karigarOnly ? 'pb-8' : 'max-w-[1600px]'}`}>
      {!karigarOnly && (
      <div className="rounded-xl bg-gradient-to-br from-[#1a3a5c] via-[#2c5aa0] to-[#1e7ed4] text-white p-5 shadow-md">
        <h1 className="text-xl font-bold">🧵 Stitching Costing — Yash Gallery</h1>
        <p className="text-sm opacity-90 mt-1">
          Karigar tracking · Challan management · Style costing · Payroll
          · Local database
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
      {!karigarOnly && tab === 'challan' && <ChallanTab onFlash={flash} />}
      {!karigarOnly && tab === 'style' && <StyleCostingTab />}
      {!karigarOnly && tab === 'efficiency' && <EfficiencyTab />}
      {!karigarOnly && tab === 'payroll' && <PayrollTab />}
      {!karigarOnly && tab === 'attendance' && <AttendanceTab type="karigar" />}
      {!karigarOnly && tab === 'operating' && <AttendanceTab type="operating" />}
      {!karigarOnly && tab === 'performance' && <PerformanceTab />}
      {!karigarOnly && tab === 'master' && <MasterTab admin={admin} onFlash={flash} />}
    </div>
  )
}

function DashboardTab() {
  const { data, isLoading, isError, error } = useQuery<DashboardData>({
    queryKey: ['stitching-dashboard'],
    queryFn: () => api.get('/stitching/dashboard', { timeout: 120_000 }).then(r => r.data),
    retry: 1,
  })
  if (isLoading) return <p className="text-sm text-gray-500">Loading dashboard…</p>
  if (isError) {
    const msg =
      error instanceof Error ? error.message : 'Could not load stitching dashboard.'
    return (
      <p className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
        {msg}
      </p>
    )
  }
  if (!data?.metrics) {
    return (
      <p className="text-sm text-gray-500">
        No dashboard data yet. Add karigars and production entries on the Production Entry tab.
      </p>
    )
  }
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

type HourEntryState = {
  operation: string
  pieces: number
  sticker_in: number
  sticker_out: number
  manual_pieces: boolean
}

function emptyHourEntry(): HourEntryState {
  return { operation: '', pieces: 0, sticker_in: 0, sticker_out: 0, manual_pieces: false }
}

/** Sticker in − out when stickers used; otherwise manual pieces. */
function resolveHourPieces(st: HourEntryState | undefined): number {
  if (!st) return 0
  if ((st.sticker_in > 0 || st.sticker_out > 0) && !st.manual_pieces) {
    return Math.max(0, st.sticker_in - st.sticker_out)
  }
  return st.pieces || 0
}

function computeEntrySummary(
  hourState: Record<string, HourEntryState>,
  hours: HourDef[],
  operations: { op: string; target: number; rate: number }[],
) {
  const opTotals: Record<string, { pieces: number; value: number }> = {}
  let totalPcs = 0
  for (const h of hours) {
    if (h.col === 'H_13_14') continue
    const st = hourState[h.col]
    const pcs = resolveHourPieces(st)
    if (!st?.operation || !pcs) continue
    const meta = operations.find(o => o.op === st.operation)
    if (!meta) continue
    totalPcs += pcs
    if (!opTotals[st.operation]) opTotals[st.operation] = { pieces: 0, value: 0 }
    opTotals[st.operation].pieces += pcs
    opTotals[st.operation].value += pcs * meta.rate
  }
  const totalValue = Object.values(opTotals).reduce((s, d) => s + d.value, 0)
  const totalBudget = Object.entries(opTotals).reduce((s, [op]) => {
    const meta = operations.find(o => o.op === op)
    return s + (meta ? meta.rate * meta.target : 0)
  }, 0)
  return { totalPcs, totalValue, totalBudget, pl: totalBudget - totalValue, opTotals }
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
  const [hourState, setHourState] = useState<Record<string, HourEntryState>>({})
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
    const next: Record<string, HourEntryState> = {}
    for (const h of hours) {
      const e = data.hours?.[h.col] ?? emptyHourEntry()
      next[h.col] = {
        operation: e.operation || '',
        pieces: Number(e.pieces) || 0,
        sticker_in: Number(e.sticker_in) || 0,
        sticker_out: Number(e.sticker_out) || 0,
        manual_pieces: Boolean(e.manual_pieces),
      }
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

  const [showSalaryCols, setShowSalaryCols] = useState(!!karigarId)
  useEffect(() => {
    if (karigarId) setShowSalaryCols(true)
  }, [karigarId])
  const showSalary = !!karigarId || showSalaryCols

  const report1Cols = useMemo(() => {
    const base = [
      'Karigar_Name',
      'Challan_No',
      'Style',
      'Operation',
      'Working_Hours',
      'Total_Pieces',
      'Efficiency_%',
    ]
    if (showSalary) {
      base.push('Daily_Salary_Rs', 'Hourly_Salary_Rs', 'Actual_Expense_Rs', 'PL_Rs')
    } else {
      base.push('PL_Rs')
    }
    return base
  }, [showSalary])

  const report2SummaryCols = useMemo(() => {
    const base = ['Karigar', 'Style', 'Challan_No', 'Operation', 'Hours_Worked', 'Total_Pieces', 'Total_Net_PL', 'Result']
    if (showSalary) base.splice(4, 0, 'Daily_Salary_Rs', 'Total_Salary_Cost')
    return base
  }, [showSalary])

  const report2HourlyCols = useMemo(() => {
    const base = ['Karigar', 'Style', 'Challan_No', 'Hour', 'Operation', 'Pieces_Done', 'Actual_Piece_Val_Rs', 'Net_PL_Rs', 'Status']
    if (showSalary) base.splice(5, 0, 'Daily_Salary_Rs', 'Hourly_Salary_Rs')
    return base
  }, [showSalary])

  const historyCols = useMemo(
    () => ['Save_Time', 'Karigar_Name', 'Challan_No', 'Style', 'Operation', 'Total_Pieces', 'PL_Rs'],
    [],
  )

  const liveSummary = useMemo(
    () => computeEntrySummary(hourState, hours, operations),
    [hourState, hours, operations],
  )

  const buildHourEntries = useCallback(
    () =>
      hours
        .filter(h => h.col !== 'H_13_14')
        .map(h => {
          const st = hourState[h.col] ?? emptyHourEntry()
          return {
            hour_col: h.col,
            operation: st.operation,
            pieces: resolveHourPieces(st),
            sticker_in: st.sticker_in,
            sticker_out: st.sticker_out,
            manual_pieces: st.manual_pieces,
          }
        }),
    [hours, hourState],
  )

  const normalizeHourEntriesForSave = useCallback(() => {
    const entries = buildHourEntries()
    if (operations.length === 1) {
      const onlyOp = operations[0].op
      for (const e of entries) {
        if ((e.pieces || 0) > 0 && !e.operation) e.operation = onlyOp
      }
    }
    return entries
  }, [buildHourEntries, operations])

  const piecesWithoutOp = useMemo(() => {
    let n = 0
    for (const h of hours) {
      if (h.col === 'H_13_14') continue
      const st = hourState[h.col]
      const pcs = resolveHourPieces(st)
      if (pcs > 0 && !st?.operation) n += pcs
    }
    return n
  }, [hours, hourState])

  const saveBlockReason = useMemo(() => {
    if (!karigarId) return 'Select karigar'
    if (!style) return 'Select style / SKU'
    if (!challanNo) return 'Select challan'
    if (!operations.length) return 'No operations in master for this style'
    if (piecesWithoutOp > 0 && operations.length > 1) {
      return 'Pick an operation for each hour with pieces'
    }
    if (liveSummary.totalPcs <= 0) return 'Enter pieces in at least one hour'
    return ''
  }, [karigarId, style, challanNo, operations.length, piecesWithoutOp, liveSummary.totalPcs])

  const doSave = useCallback(
    async (silent: boolean) => {
      if (!karigarId || !challanNo || !style) return
      const entries = normalizeHourEntriesForSave()
      const savablePcs = entries.reduce(
        (s, e) => s + (e.operation && operations.some(o => o.op === e.operation) ? e.pieces || 0 : 0),
        0,
      )
      if (savablePcs <= 0) {
        const msg = saveBlockReason || 'Nothing to save'
        setSaveStatus('error')
        setSaveMessage(msg)
        if (!silent) alert(msg)
        return
      }
      const k = karigars.find(x => String(x.Karigar_ID) === karigarId)
      setSaveStatus('saving')
      setSaveMessage('')
      try {
        const r = await api.post('/stitching/production-entry', {
          date: entryDate,
          karigar_id: karigarId,
          karigar_name: k?.Name ?? karigarId,
          challan_no: challanNo,
          style,
          hour_entries: entries,
        })
        if (r.data.ok) {
          setSaveStatus('saved')
          setSaveMessage(r.data.message || 'Saved')
          onSaved()
          refetchReports()
          qc.invalidateQueries({ queryKey: ['stitching-dashboard'] })
          if (!silent) alert(r.data.message || 'Saved')
        } else {
          const msg = r.data.message || 'Save failed'
          setSaveStatus('error')
          setSaveMessage(msg)
          if (!silent) alert(msg)
        }
      } catch (err: unknown) {
        const msg =
          axios.isAxiosError(err) && err.response?.data?.detail
            ? String(err.response.data.detail)
            : 'Save failed — check connection and try again'
        setSaveStatus('error')
        setSaveMessage(msg)
        if (!silent) alert(msg)
      }
    },
    [
      karigarId,
      challanNo,
      style,
      karigars,
      entryDate,
      normalizeHourEntriesForSave,
      operations,
      saveBlockReason,
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
    if (!karigarId || !challanNo || !style || saveBlockReason) return
    const t = setTimeout(() => void doSave(true), 2000)
    return () => clearTimeout(t)
  }, [hourState, entryDate, karigarId, challanNo, style, autoSaveEnabled, saveBlockReason, doSave])

  const saveMut = useMutation({
    mutationFn: () => doSave(false),
  })

  const setHour = (col: string, patch: Partial<HourEntryState>) => {
    setHourState(prev => {
      const base = { ...emptyHourEntry(), ...prev[col], ...patch }
      if ('sticker_in' in patch || 'sticker_out' in patch) {
        if (!base.manual_pieces) {
          base.pieces = Math.max(0, base.sticker_in - base.sticker_out)
        }
      }
      if ('pieces' in patch && patch.pieces !== undefined) {
        base.manual_pieces = true
      }
      if (operations.length === 1 && resolveHourPieces(base) > 0 && !base.operation) {
        base.operation = operations[0].op
      }
      return { ...prev, [col]: base }
    })
  }

  const opOptions = ['', ...operations.map(o => o.op)]

  return (
    <div className={`space-y-4 min-w-0 ${karigarOnly ? 'pb-36' : ''}`}>
      <div className="grid gap-3 bg-white p-4 rounded-xl border sm:grid-cols-2 lg:grid-cols-4">
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
          {saveStatus === 'error' && (
            <span className="text-xs text-red-600 font-medium">{saveMessage || 'Save failed'}</span>
          )}
        </div>
        {saveBlockReason && (
          <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mb-3">
            {saveBlockReason}
          </p>
        )}
        <p className="text-xs text-gray-500 mb-3">
          <strong>Sticker in / out</strong> — pieces = in − out (e.g. in 10, out 5 → 5 pcs). Leave stickers empty to type pieces manually.
        </p>

        <div className="space-y-2 mb-3 md:hidden">
          {hours.map(h => {
            if (h.col === 'H_13_14') {
              return (
                <div key={h.col} className="rounded-lg bg-gray-50 border border-dashed px-3 py-2 text-xs text-gray-400 italic text-center">
                  {h.label} — Lunch
                </div>
              )
            }
            const st = hourState[h.col] ?? emptyHourEntry()
            const pcs = resolveHourPieces(st)
            const fromSticker = (st.sticker_in > 0 || st.sticker_out > 0) && !st.manual_pieces
            const opMeta = operations.find(o => o.op === st.operation)
            const eff = opMeta && pcs > 0 ? Math.round((pcs / Math.max(opMeta.target, 1)) * 100) : null
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
                <div className="grid grid-cols-2 gap-2 mb-2">
                  <label className="text-[10px] text-gray-600">
                    Sticker in
                    <input
                      type="number"
                      inputMode="numeric"
                      min={0}
                      className="w-full border rounded-lg px-2 py-2 text-sm mt-0.5 touch-manipulation"
                      value={st.sticker_in || ''}
                      onChange={e => setHour(h.col, { sticker_in: +e.target.value || 0, manual_pieces: false })}
                    />
                  </label>
                  <label className="text-[10px] text-gray-600">
                    Sticker out
                    <input
                      type="number"
                      inputMode="numeric"
                      min={0}
                      className="w-full border rounded-lg px-2 py-2 text-sm mt-0.5 touch-manipulation"
                      value={st.sticker_out || ''}
                      onChange={e => setHour(h.col, { sticker_out: +e.target.value || 0, manual_pieces: false })}
                    />
                  </label>
                </div>
                <div className="flex gap-2 items-center">
                  <label className="flex-1 text-[10px] text-gray-600">
                    Pieces {fromSticker ? '(auto)' : '(manual)'}
                    <input
                      type="number"
                      inputMode="numeric"
                      min={0}
                      readOnly={fromSticker}
                      className={`w-full border rounded-lg px-3 py-3 text-lg font-semibold mt-0.5 touch-manipulation ${fromSticker ? 'bg-sky-50 text-[#2c5aa0]' : ''}`}
                      value={fromSticker ? pcs || '' : st.pieces || ''}
                      onChange={e => setHour(h.col, { pieces: +e.target.value || 0, manual_pieces: true })}
                    />
                  </label>
                  <span className="text-xs text-gray-400 shrink-0 pt-4">tgt {opMeta?.target ?? '—'}</span>
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
                <th className="px-2 py-2">Sticker in</th>
                <th className="px-2 py-2">Sticker out</th>
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
                      <td colSpan={6} className="px-2 py-2">Lunch break</td>
                    </tr>
                  )
                }
                const st = hourState[h.col] ?? emptyHourEntry()
                const pcs = resolveHourPieces(st)
                const fromSticker = (st.sticker_in > 0 || st.sticker_out > 0) && !st.manual_pieces
                const opMeta = operations.find(o => o.op === st.operation)
                const eff = opMeta && pcs > 0 ? Math.round((pcs / Math.max(opMeta.target, 1)) * 100) : null
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
                        className="w-16 border rounded px-1 py-1"
                        value={st.sticker_in || ''}
                        onChange={e => setHour(h.col, { sticker_in: +e.target.value || 0, manual_pieces: false })}
                      />
                    </td>
                    <td className="px-2 py-1">
                      <input
                        type="number"
                        min={0}
                        className="w-16 border rounded px-1 py-1"
                        value={st.sticker_out || ''}
                        onChange={e => setHour(h.col, { sticker_out: +e.target.value || 0, manual_pieces: false })}
                      />
                    </td>
                    <td className="px-2 py-1">
                      <input
                        type="number"
                        min={0}
                        readOnly={fromSticker}
                        title={fromSticker ? 'Auto from stickers' : 'Manual pieces'}
                        className={`w-16 border rounded px-1 py-1 ${fromSticker ? 'bg-sky-50' : ''}`}
                        value={fromSticker ? pcs : st.pieces || ''}
                        onChange={e => setHour(h.col, { pieces: +e.target.value || 0, manual_pieces: true })}
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
              <p className="text-[10px] uppercase text-gray-500">P&amp;L (budget − actual)</p>
              <p className={`text-lg font-bold ${liveSummary.pl >= 0 ? 'text-green-700' : 'text-red-600'}`}>
                ₹{liveSummary.pl.toFixed(2)}
              </p>
            </div>
          </div>
        )}

        {!karigarOnly && (
          <button
            type="button"
            onClick={() => saveMut.mutate()}
            disabled={saveMut.isPending || !!saveBlockReason}
            className="mt-4 w-full py-3.5 rounded-xl bg-[#002B5B] text-white font-semibold text-sm disabled:opacity-50 touch-manipulation shadow-md"
          >
            {saveMut.isPending ? 'Saving…' : '💾 Save now'}
          </button>
        )}
      </Section>

      {karigarOnly && (
        <div className="fixed bottom-0 left-0 right-0 z-30 p-3 bg-gradient-to-t from-gray-100 via-gray-100 to-transparent pb-[max(0.75rem,env(safe-area-inset-bottom))] pointer-events-none">
          <div className="max-w-lg md:max-w-3xl lg:max-w-5xl mx-auto pointer-events-auto space-y-1">
            {saveBlockReason && (
              <p className="text-[11px] text-center text-amber-900 bg-amber-50 border border-amber-200 rounded-lg px-2 py-1">
                {saveBlockReason}
              </p>
            )}
            <button
              type="button"
              onClick={() => saveMut.mutate()}
              disabled={saveMut.isPending || !!saveBlockReason}
              className="w-full py-4 rounded-xl bg-[#002B5B] text-white font-bold text-base disabled:opacity-50 touch-manipulation shadow-lg"
            >
              {saveMut.isPending ? 'Saving…' : saveStatus === 'saved' ? '✓ Saved' : '💾 Save production'}
            </button>
          </div>
        </div>
      )}

      {(entryReports?.recent_saves?.length > 0 || entryReports?.history?.length > 0) && (
        <ReportTableSection
          title={karigarId ? "Today's saves — this karigar" : "Today's saves"}
          rows={(entryReports.recent_saves?.length ? entryReports.recent_saves : entryReports.history) as Record<string, unknown>[]}
          cols={historyCols}
          downloadName={`production-saves-${entryDate}.csv`}
        >
          <div className="space-y-2">
            {(entryReports.recent_saves?.length ? entryReports.recent_saves : entryReports.history)
              .slice(0, 8)
              .map((row: Record<string, unknown>, i: number) => (
                <div key={i} className="text-xs bg-blue-50 border-l-4 border-[#2c5aa0] rounded-r-lg px-3 py-2.5">
                  {String(row.Save_Time || '—')} · <b>{String(row.Karigar_Name)}</b> · {String(row.Challan_No)} ·{' '}
                  <b>{String(row.Style || '—')}</b> · {String(row.Operation)} · <b>{String(row.Total_Pieces)} pcs</b>
                </div>
              ))}
          </div>
        </ReportTableSection>
      )}

      {!karigarOnly && !karigarId && (
        <label className="flex items-center gap-2 text-xs text-gray-600 bg-white border rounded-lg px-3 py-2 w-fit">
          <input
            type="checkbox"
            checked={showSalaryCols}
            onChange={e => setShowSalaryCols(e.target.checked)}
            className="rounded"
          />
          Show karigar salary columns in reports
        </label>
      )}

      {entryReports?.report1?.length > 0 && (
        <ReportTableSection
          title="Report 1 — Production summary"
          rows={entryReports.report1 as Record<string, unknown>[]}
          cols={report1Cols}
          downloadName={`production-summary-${entryDate}${karigarId ? `-${karigarId}` : ''}.csv`}
        />
      )}

      {entryReports?.report2_summary?.length > 0 && (
        <ReportTableSection
          title="Report 2 — Salary vs piece value"
          rows={entryReports.report2_summary as Record<string, unknown>[]}
          cols={report2SummaryCols}
          downloadName={`production-salary-summary-${entryDate}${karigarId ? `-${karigarId}` : ''}.csv`}
        >
          {entryReports.grand_total && (
            <p className="mt-2 text-xs rounded-lg px-3 py-2 bg-gray-50 border text-gray-700">
              Grand net P&amp;L: ₹{entryReports.grand_total.total_net_pl}
              {showSalary && entryReports.grand_total.total_salary_cost != null && (
                <span className="ml-2">· Total salary cost: ₹{entryReports.grand_total.total_salary_cost}</span>
              )}
            </p>
          )}
        </ReportTableSection>
      )}

      {entryReports?.report2_hourly?.length > 0 && (
        <ReportTableSection
          title="Report 2 — Hour-wise detail"
          rows={entryReports.report2_hourly as Record<string, unknown>[]}
          cols={report2HourlyCols}
          downloadName={`production-hourly-${entryDate}${karigarId ? `-${karigarId}` : ''}.csv`}
        >
          <details className="text-xs">
            <summary className="cursor-pointer py-3 font-medium text-[#2c5aa0] touch-manipulation min-h-[44px]">
              Show hour-wise rows ({entryReports.report2_hourly.length})
            </summary>
            <div className="mt-2 max-h-64 overflow-auto">
              <DataTable rows={entryReports.report2_hourly} cols={report2HourlyCols} />
            </div>
          </details>
        </ReportTableSection>
      )}
    </div>
  )
}

function SheetUploadBar({
  sheetKey,
  label,
  onFlash,
  onSaved,
}: {
  sheetKey: string
  label?: string
  onFlash: (type: 'ok' | 'err', text: string) => void
  onSaved: () => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [mode, setMode] = useState<'merge' | 'replace' | 'append'>('merge')

  const upload = async (file: File) => {
    setBusy(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const { data } = await api.post<{ ok?: boolean; rows?: number }>(
        `/stitching/sheets/${sheetKey}/import-file`,
        fd,
        {
          params: { mode },
          headers: { 'Content-Type': 'multipart/form-data' },
        },
      )
      onFlash(
        'ok',
        data?.rows != null
          ? `Saved ${data.rows.toLocaleString()} row(s) to ${label || sheetKey.replace(/_/g, ' ')}.`
          : 'Upload saved.',
      )
      onSaved()
    } catch {
      onFlash('err', 'Upload failed — use .csv or .xlsx with column headers matching the sheet.')
    } finally {
      setBusy(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  return (
    <div className="rounded-lg border border-dashed border-[#2c5aa0]/40 bg-blue-50/40 p-3 flex flex-wrap items-center gap-2 text-xs">
      <span className="font-semibold text-[#1a3a5c]">📤 {label || 'Import sheet'}</span>
      <span className="text-gray-500">Upload saves automatically to the server database.</span>
      <select
        className="border rounded px-2 py-1 bg-white"
        value={mode}
        onChange={e => setMode(e.target.value as 'merge' | 'replace' | 'append')}
        disabled={busy}
      >
        <option value="merge">Merge (keep existing + add new)</option>
        <option value="append">Append all file rows</option>
        <option value="replace">Replace all rows</option>
      </select>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx,.xls"
        className="hidden"
        onChange={e => {
          const f = e.target.files?.[0]
          if (f) void upload(f)
        }}
      />
      <button
        type="button"
        disabled={busy}
        onClick={() => inputRef.current?.click()}
        className="px-3 py-1.5 rounded-lg bg-[#002B5B] text-white font-semibold disabled:opacity-50"
      >
        {busy ? 'Saving…' : 'Choose file'}
      </button>
    </div>
  )
}

function ChallanTab({ onFlash }: { onFlash: (type: 'ok' | 'err', text: string) => void }) {
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

  const invalidateChallan = () => {
    qc.invalidateQueries({ queryKey: ['stitching-sheet', 'challan_master'] })
    qc.invalidateQueries({ queryKey: ['stitching-style-costing'] })
    qc.invalidateQueries({ queryKey: ['stitching-dashboard'] })
    void refetch()
  }

  const addMut = useMutation({
    mutationFn: () => api.post('/stitching/challans', form),
    onSuccess: () => {
      invalidateChallan()
      onFlash('ok', `Challan ${form.Challan_No} saved.`)
    },
    onError: () => onFlash('err', 'Could not save challan.'),
  })

  const rows = (data?.rows ?? []) as Record<string, unknown>[]

  return (
    <div className="space-y-4">
      <SheetUploadBar
        sheetKey="challan_master"
        label="Import challans"
        onFlash={onFlash}
        onSaved={invalidateChallan}
      />
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
      <ChallanDepositSection rows={rows} onFlash={onFlash} onSaved={invalidateChallan} />
      <Section title={`Challan register (${rows.length})`}>
        <p className="text-xs text-gray-500 mb-2">
          Edit <strong>Received</strong> — saves automatically. Style costing uses received qty when set (e.g. 90 received of 100 ordered).
        </p>
        <ChallanRegisterTable rows={rows} onFlash={onFlash} onSaved={invalidateChallan} />
      </Section>
    </div>
  )
}

function ChallanDepositSection({
  rows,
  onFlash,
  onSaved,
}: {
  rows: Record<string, unknown>[]
  onFlash: (type: 'ok' | 'err', text: string) => void
  onSaved: () => void
}) {
  const [challanNo, setChallanNo] = useState('')
  const [depositQty, setDepositQty] = useState(0)
  const [addDepositRs, setAddDepositRs] = useState(0)
  const [saving, setSaving] = useState(false)

  const row = useMemo(
    () => rows.find(r => String(r.Challan_No ?? '') === challanNo),
    [rows, challanNo],
  )

  const total = Number(row?.Total_Qty ?? 0)
  const received = Number(row?.Received_Qty ?? 0)
  const pending = Math.max(0, total - received)
  const currentDepositRs = Number(row?.Deposit_Rs ?? 0)

  const challanOptions = useMemo<SearchSelectOption[]>(
    () =>
      rows
        .map(r => {
          const no = String(r.Challan_No ?? '')
          const t = Number(r.Total_Qty ?? 0)
          const rec = Number(r.Received_Qty ?? 0)
          const pend = Math.max(0, t - rec)
          return {
            value: no,
            primary: no,
            secondary: `${String(r.Style ?? '')} · Pending ${pend} / ${t}`,
            haystack: `${no} ${r.Style} ${r.Party}`.toLowerCase(),
          }
        })
        .filter(o => o.value)
        .sort((a, b) => a.primary.localeCompare(b.primary)),
    [rows],
  )

  const applyDeposit = async () => {
    if (!challanNo || !row) {
      onFlash('err', 'Select a challan first.')
      return
    }
    const add = Math.max(0, Math.floor(depositQty))
    if (add <= 0) {
      onFlash('err', 'Enter quantity received in this deposit.')
      return
    }
    if (pending <= 0) {
      onFlash('err', 'This challan is already fully received.')
      return
    }
    const newReceived = Math.min(total, received + add)
    const newDepositRs = currentDepositRs + Math.max(0, addDepositRs)
    setSaving(true)
    try {
      await api.patch(`/stitching/challans/${encodeURIComponent(challanNo)}`, {
        Received_Qty: newReceived,
        Deposit_Rs: newDepositRs,
      })
      onFlash(
        'ok',
        `Challan ${challanNo}: received ${newReceived} of ${total} (+${add} this deposit).`,
      )
      setDepositQty(0)
      setAddDepositRs(0)
      onSaved()
    } catch {
      onFlash('err', `Could not update challan ${challanNo}.`)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Section title="Record material deposit (update received qty)">
      <p className="text-xs text-gray-600 mb-3">
        When party sends the <strong>remaining quantity</strong>, select the challan, enter how many pieces arrived in this
        deposit, and apply. <strong>Received</strong> increases (capped at ordered qty). Optional: add to{' '}
        <strong>Deposit ₹</strong> for this receipt.
      </p>
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
        <div className="sm:col-span-2">
          <SearchSelect
            label="Challan"
            placeholder="Search challan no. or style…"
            value={challanNo}
            onChange={setChallanNo}
            options={challanOptions}
            emptyMessage="No challans — add one above"
          />
        </div>
        {row && (
          <div className="sm:col-span-2 rounded-lg bg-sky-50 border border-sky-100 px-3 py-2 text-sky-900">
            <span className="font-semibold">{String(row.Style)}</span>
            {row.Party ? ` · ${String(row.Party)}` : ''}
            <br />
            Ordered <b>{total}</b> · Received <b>{received}</b> ·{' '}
            <span className={pending > 0 ? 'text-amber-700' : 'text-green-700'}>
              Pending <b>{pending}</b>
            </span>
            {currentDepositRs > 0 && (
              <>
                <br />
                Deposit recorded: ₹{currentDepositRs.toFixed(2)}
              </>
            )}
          </div>
        )}
        <label>
          <span className="font-semibold text-gray-700">Qty this deposit</span>
          <input
            type="number"
            min={0}
            max={pending > 0 ? pending : undefined}
            className="mt-1 w-full border rounded-lg px-3 py-2"
            value={depositQty || ''}
            onChange={e => setDepositQty(+e.target.value || 0)}
            disabled={!challanNo || pending <= 0}
          />
          {pending > 0 && (
            <span className="text-[10px] text-gray-500 mt-0.5 block">Max {pending} remaining</span>
          )}
        </label>
        <label>
          <span className="font-semibold text-gray-700">Add deposit ₹ (optional)</span>
          <input
            type="number"
            min={0}
            step={0.01}
            className="mt-1 w-full border rounded-lg px-3 py-2"
            value={addDepositRs || ''}
            onChange={e => setAddDepositRs(+e.target.value || 0)}
            disabled={!challanNo}
          />
        </label>
      </div>
      <button
        type="button"
        onClick={() => void applyDeposit()}
        disabled={saving || !challanNo || pending <= 0}
        className="mt-3 px-4 py-2.5 bg-emerald-700 text-white rounded-lg text-sm font-semibold disabled:opacity-50"
      >
        {saving ? 'Updating…' : 'Apply deposit & update challan'}
      </button>
    </Section>
  )
}

function ChallanRegisterTable({
  rows,
  onFlash,
  onSaved,
}: {
  rows: Record<string, unknown>[]
  onFlash: (type: 'ok' | 'err', text: string) => void
  onSaved: () => void
}) {
  const [draft, setDraft] = useState<Record<string, { received?: number; total?: number }>>({})
  const saveTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})

  useEffect(() => {
    return () => {
      Object.values(saveTimers.current).forEach(clearTimeout)
    }
  }, [])

  const scheduleSave = (challanNo: string, patch: { Received_Qty?: number; Total_Qty?: number }) => {
    if (saveTimers.current[challanNo]) clearTimeout(saveTimers.current[challanNo])
    saveTimers.current[challanNo] = setTimeout(() => {
      void api
        .patch(`/stitching/challans/${encodeURIComponent(challanNo)}`, patch)
        .then(() => {
          onFlash('ok', `Challan ${challanNo} saved.`)
          onSaved()
        })
        .catch(() => onFlash('err', `Could not save ${challanNo}.`))
    }, 600)
  }

  if (!rows.length) return <p className="text-sm text-gray-400 text-center py-4">No challans</p>

  return (
    <div className="overflow-x-auto max-h-[min(420px,50vh)]">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-gray-50 border-b">
          <tr>
            {['Challan_No', 'Style', 'Party', 'Total_Qty', 'Received_Qty', 'Pending', 'Rate_Per_Pc', 'Date'].map(c => (
              <th key={c} className="text-left px-2 py-2 font-semibold text-gray-600 whitespace-nowrap">
                {c.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const no = String(r.Challan_No ?? '')
            const total = draft[no]?.total ?? Number(r.Total_Qty ?? 0)
            const received = draft[no]?.received ?? Number(r.Received_Qty ?? 0)
            const pending = Math.max(0, total - received)
            return (
              <tr key={no || i} className="border-b border-gray-50 hover:bg-gray-50/80">
                <td className="px-2 py-1.5 font-mono">{no}</td>
                <td className="px-2 py-1.5">{String(r.Style ?? '')}</td>
                <td className="px-2 py-1.5">{String(r.Party ?? '')}</td>
                <td className="px-2 py-1.5">
                  <input
                    type="number"
                    min={0}
                    className="w-20 border rounded px-1 py-0.5 text-right"
                    value={total}
                    onChange={e => {
                      const v = +e.target.value || 0
                      setDraft(d => ({ ...d, [no]: { ...d[no], total: v, received } }))
                      scheduleSave(no, { Total_Qty: v, Received_Qty: received })
                    }}
                  />
                </td>
                <td className="px-2 py-1.5">
                  <input
                    type="number"
                    min={0}
                    className="w-20 border rounded px-1 py-0.5 text-right bg-sky-50"
                    value={received}
                    onChange={e => {
                      const v = +e.target.value || 0
                      setDraft(d => ({ ...d, [no]: { ...d[no], received: v, total } }))
                      scheduleSave(no, { Received_Qty: v, Total_Qty: total })
                    }}
                  />
                </td>
                <td className={`px-2 py-1.5 font-semibold ${pending > 0 ? 'text-amber-700' : 'text-green-700'}`}>
                  {pending}
                </td>
                <td className="px-2 py-1.5">{String(r.Rate_Per_Pc ?? '')}</td>
                <td className="px-2 py-1.5 whitespace-nowrap">{String(r.Date ?? '')}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
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
      <p className="text-xs text-gray-500">
        P&amp;L uses <strong>received</strong> qty when set (e.g. 90 of 100 ordered). Party value (ordered) is shown for reference.
      </p>
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
          cols={[
            'Challan_No',
            'Style',
            'Party',
            'Total_Qty',
            'Received_Qty',
            'Pending',
            'Party_Value_Ordered_Rs',
            'Party_Value_Received_Rs',
            'Party_Value_Rs',
            'Actual_Labour_Rs',
            'Target_Labour_Rs',
            'PL_Rs',
            'Margin_%',
          ]}
        />
      </Section>
      <Section title="Style roll-up">
        <DataTable
          rows={data?.style_rollup ?? []}
          cols={['Style', 'Challans', 'Qty_Ordered', 'Qty_Received', 'Qty', 'PL', 'Margin_%', 'Result']}
        />
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

const MASTER_SHEETS = ['style_master', 'karigar_master', 'employee_master'] as const
type MasterSheetKey = (typeof MASTER_SHEETS)[number]

function masterRowKey(sheet: MasterSheetKey, row: Record<string, unknown>): string {
  if (sheet === 'style_master') return `${row.Style}||${row.Operation}`
  if (sheet === 'karigar_master') return String(row.Karigar_ID ?? '')
  return String(row.E_Code ?? '')
}

function masterRowPayload(sheet: MasterSheetKey, row: Record<string, unknown>): Record<string, string> {
  if (sheet === 'style_master') {
    return { Style: String(row.Style ?? ''), Operation: String(row.Operation ?? '') }
  }
  if (sheet === 'karigar_master') return { Karigar_ID: String(row.Karigar_ID ?? '') }
  return { E_Code: String(row.E_Code ?? '') }
}

function MasterTab({ admin: _admin, onFlash }: { admin: AdminApi; onFlash: (type: 'ok' | 'err', text: string) => void }) {
  const qc = useQueryClient()
  const [active, setActive] = useState<MasterSheetKey>('style_master')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [editKarigar, setEditKarigar] = useState<Record<string, unknown> | null>(null)
  const [karEdit, setKarEdit] = useState({
    Name: '',
    Skill: '',
    Daily_Rate_Rs: 420,
    Effective_From: todayStr(),
  })
  const { data, refetch } = useQuery({
    queryKey: ['stitching-sheet', active],
    queryFn: () => api.get(`/stitching/sheets/${active}`).then(r => r.data),
  })

  const rows = (data?.rows ?? []) as Record<string, unknown>[]
  const cols = (data?.columns as string[]) ?? []

  const invalidateSheet = () => {
    setSelected(new Set())
    void refetch()
    qc.invalidateQueries({ queryKey: ['stitching-sheet'] })
    qc.invalidateQueries({ queryKey: ['stitching-dashboard'] })
    qc.invalidateQueries({ queryKey: ['stitching-style-costing'] })
    qc.invalidateQueries({ queryKey: ['stitching-pe-reports'] })
  }

  const [styleForm, setStyleForm] = useState({ Style: '', Operation: '', Target: 80, Rate_Rs: 3 })
  const [karForm, setKarForm] = useState({
    Karigar_ID: '',
    Name: '',
    Skill: 'Stitching',
    Daily_Rate_Rs: 420,
    Effective_From: todayStr(),
  })

  const addStyle = useMutation({
    mutationFn: () => api.post('/stitching/master/style-operation', styleForm),
    onSuccess: () => {
      onFlash('ok', 'Style operation added.')
      setStyleForm({ Style: '', Operation: '', Target: 80, Rate_Rs: 3 })
      invalidateSheet()
    },
    onError: (e: { response?: { data?: { detail?: string } } }) =>
      onFlash('err', String(e.response?.data?.detail || 'Duplicate or invalid')),
  })
  const addKar = useMutation({
    mutationFn: () =>
      api.post('/stitching/master/karigar', { ...karForm, Effective_From: karForm.Effective_From || todayStr() }),
    onSuccess: () => {
      onFlash('ok', 'Karigar added.')
      setKarForm({ Karigar_ID: '', Name: '', Skill: 'Stitching', Daily_Rate_Rs: 420, Effective_From: todayStr() })
      invalidateSheet()
    },
    onError: (e: { response?: { data?: { detail?: string } } }) =>
      onFlash('err', String(e.response?.data?.detail || 'Karigar ID may already exist')),
  })

  const deleteMut = useMutation({
    mutationFn: (payload: { sheet: MasterSheetKey; rows: Record<string, string>[] }) =>
      api.post('/stitching/master/delete-rows', payload),
    onSuccess: r => {
      onFlash('ok', r.data.message || 'Deleted')
      setEditKarigar(null)
      invalidateSheet()
    },
    onError: () => onFlash('err', 'Delete failed'),
  })

  const updateKarMut = useMutation({
    mutationFn: () =>
      api.patch(`/stitching/master/karigar/${encodeURIComponent(String(editKarigar?.Karigar_ID ?? ''))}`, {
        Name: karEdit.Name,
        Skill: karEdit.Skill,
        Daily_Rate_Rs: karEdit.Daily_Rate_Rs,
        Effective_From: karEdit.Effective_From || todayStr(),
      }),
    onSuccess: r => {
      onFlash('ok', r.data.message || 'Updated')
      invalidateSheet()
    },
    onError: (e: { response?: { data?: { detail?: string } } }) =>
      onFlash('err', String(e.response?.data?.detail || 'Update failed')),
  })

  const toggleRow = (key: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const toggleAll = () => {
    if (selected.size === rows.length) setSelected(new Set())
    else setSelected(new Set(rows.map(r => masterRowKey(active, r))))
  }

  const deleteSelected = () => {
    const toDelete = rows.filter(r => selected.has(masterRowKey(active, r))).map(r => masterRowPayload(active, r))
    if (!toDelete.length) return
    if (!window.confirm(`Delete ${toDelete.length} selected row(s)?`)) return
    deleteMut.mutate({ sheet: active, rows: toDelete })
  }

  const deleteOne = (row: Record<string, unknown>) => {
    if (!window.confirm(`Delete "${masterRowKey(active, row)}"?`)) return
    deleteMut.mutate({ sheet: active, rows: [masterRowPayload(active, row)] })
  }

  const openKarigarEdit = (row: Record<string, unknown>) => {
    setEditKarigar(row)
    setKarEdit({
      Name: String(row.Name ?? ''),
      Skill: String(row.Skill ?? ''),
      Daily_Rate_Rs: Number(row.Daily_Rate_Rs) || 420,
      Effective_From: todayStr(),
    })
  }

  return (
    <div className="space-y-4">
      <SheetUploadBar
        sheetKey={active}
        label={`Import ${active.replace(/_/g, ' ')}`}
        onFlash={onFlash}
        onSaved={invalidateSheet}
      />
      <p className="text-xs text-gray-600 bg-gray-50 border rounded-lg px-3 py-2">
        Select rows to delete one or many. Duplicates are blocked on add; imports are de-duplicated. Karigar rate changes need an{' '}
        <strong>effective date</strong> so that day&apos;s payroll and production reports use the correct daily rate.
      </p>
      <div className="flex flex-wrap gap-2">
        {MASTER_SHEETS.map(k => (
          <button
            key={k}
            type="button"
            onClick={() => {
              setActive(k)
              setSelected(new Set())
              setEditKarigar(null)
            }}
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
        <>
          <div className="grid sm:grid-cols-2 lg:grid-cols-6 gap-2 text-xs mb-3">
            {(['Karigar_ID', 'Name', 'Skill'] as const).map(k => (
              <label key={k}>
                {k}
                <input className="w-full border rounded mt-1 px-2 py-1" value={karForm[k]} onChange={e => setKarForm(f => ({ ...f, [k]: e.target.value }))} />
              </label>
            ))}
            <label>
              Daily rate ₹
              <input type="number" className="w-full border rounded mt-1 px-2 py-1" value={karForm.Daily_Rate_Rs} onChange={e => setKarForm(f => ({ ...f, Daily_Rate_Rs: +e.target.value }))} />
            </label>
            <label>
              Effective from
              <input type="date" className="w-full border rounded mt-1 px-2 py-1" value={karForm.Effective_From} onChange={e => setKarForm(f => ({ ...f, Effective_From: e.target.value }))} />
            </label>
            <button type="button" onClick={() => addKar.mutate()} disabled={addKar.isPending} className="self-end px-3 py-2 bg-[#002B5B] text-white rounded text-xs disabled:opacity-50">
              Add karigar
            </button>
          </div>
          {editKarigar && (
            <div className="text-xs border border-violet-200 bg-violet-50 rounded-xl p-4 space-y-2 mb-3">
              <p className="font-semibold text-violet-900">
                Update — {String(editKarigar.Karigar_ID)} ({String(editKarigar.Name)})
              </p>
              <p className="text-violet-800">
                Daily rate applies from the effective date across attendance, production P&amp;L, and reports for that date.
              </p>
              <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-2">
                <label>
                  Name
                  <input className="w-full border rounded mt-1 px-2 py-1" value={karEdit.Name} onChange={e => setKarEdit(f => ({ ...f, Name: e.target.value }))} />
                </label>
                <label>
                  Skill
                  <input className="w-full border rounded mt-1 px-2 py-1" value={karEdit.Skill} onChange={e => setKarEdit(f => ({ ...f, Skill: e.target.value }))} />
                </label>
                <label>
                  Daily rate ₹
                  <input type="number" className="w-full border rounded mt-1 px-2 py-1" value={karEdit.Daily_Rate_Rs} onChange={e => setKarEdit(f => ({ ...f, Daily_Rate_Rs: +e.target.value }))} />
                </label>
                <label>
                  Effective from
                  <input type="date" className="w-full border rounded mt-1 px-2 py-1" value={karEdit.Effective_From} onChange={e => setKarEdit(f => ({ ...f, Effective_From: e.target.value }))} />
                </label>
                <div className="flex gap-2 self-end">
                  <button type="button" onClick={() => updateKarMut.mutate()} disabled={updateKarMut.isPending} className="px-3 py-2 bg-violet-700 text-white rounded-lg font-semibold disabled:opacity-50">
                    Save
                  </button>
                  <button type="button" onClick={() => setEditKarigar(null)} className="px-3 py-2 border rounded-lg">
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}
      <MasterDataTable
        rows={rows}
        cols={cols}
        sheet={active}
        selected={selected}
        onToggleRow={toggleRow}
        onToggleAll={toggleAll}
        onDeleteOne={deleteOne}
        onEditKarigar={active === 'karigar_master' ? openKarigarEdit : undefined}
        toolbar={
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <button
              type="button"
              onClick={deleteSelected}
              disabled={!selected.size || deleteMut.isPending}
              className="text-xs px-3 py-1.5 rounded-lg bg-red-600 text-white font-semibold disabled:opacity-40"
            >
              {deleteMut.isPending ? 'Deleting…' : `Delete selected (${selected.size})`}
            </button>
            <span className="text-xs text-gray-500">{rows.length} row(s)</span>
          </div>
        }
      />
    </div>
  )
}

function MasterDataTable({
  rows,
  cols,
  sheet,
  selected,
  onToggleRow,
  onToggleAll,
  onDeleteOne,
  onEditKarigar,
  toolbar,
}: {
  rows: Record<string, unknown>[]
  cols: string[]
  sheet: MasterSheetKey
  selected: Set<string>
  onToggleRow: (key: string) => void
  onToggleAll: () => void
  onDeleteOne: (row: Record<string, unknown>) => void
  onEditKarigar?: (row: Record<string, unknown>) => void
  toolbar?: React.ReactNode
}) {
  if (!rows.length) return <p className="text-sm text-gray-400 text-center py-4">No rows</p>
  const useCols = cols.length ? cols.filter(c => c in rows[0] || rows[0][c] !== undefined) : Object.keys(rows[0] ?? {})
  const allSelected = rows.length > 0 && selected.size === rows.length

  return (
    <div>
      {toolbar}
      <div className="overflow-x-auto max-h-[min(480px,55vh)] border rounded-lg">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-gray-50 border-b z-10">
            <tr>
              <th className="px-2 py-2 w-8">
                <input type="checkbox" checked={allSelected} onChange={onToggleAll} title="Select all" />
              </th>
              {useCols.map(c => (
                <th key={c} className="text-left px-2 py-2 font-semibold text-gray-600 whitespace-nowrap">
                  {c.replace(/_/g, ' ')}
                </th>
              ))}
              <th className="px-2 py-2 w-28 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const rk = masterRowKey(sheet, r)
              const isSel = selected.has(rk)
              return (
                <tr
                  key={rk || i}
                  className={`border-b border-gray-50 ${isSel ? 'bg-sky-50' : 'hover:bg-gray-50/80'}`}
                >
                  <td className="px-2 py-1.5">
                    <input type="checkbox" checked={isSel} onChange={() => onToggleRow(rk)} />
                  </td>
                  {useCols.map(c => (
                    <td key={c} className="px-2 py-1.5 whitespace-nowrap">
                      {String(r[c] ?? '')}
                    </td>
                  ))}
                  <td className="px-2 py-1.5 text-right whitespace-nowrap">
                    {onEditKarigar && (
                      <button type="button" onClick={() => onEditKarigar(r)} className="text-violet-700 hover:underline mr-2">
                        Rate
                      </button>
                    )}
                    <button type="button" onClick={() => onDeleteOne(r)} className="text-red-600 hover:underline">
                      Delete
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
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
  onFlash,
  onRestored,
}: {
  onFlash: (type: 'ok' | 'err', text: string) => void
  onRestored: () => void
}) {
  const zipRef = useRef<HTMLInputElement>(null)
  const [restorePending, setRestorePending] = useState(false)
  const [gsheetPending, setGsheetPending] = useState(false)

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

  const handlePullGsheet = async () => {
    setGsheetPending(true)
    try {
      const { data } = await api.post<{
        ok?: boolean
        message?: string
        added_rows?: Record<string, number>
      }>('/stitching/sync/from-gsheet/merge')
      const added = data.added_rows
        ? Object.entries(data.added_rows)
            .filter(([, n]) => n > 0)
            .map(([k, n]) => `${k.replace(/_/g, ' ')}: +${n}`)
            .join(', ')
        : ''
      onFlash('ok', added ? `${data.message || 'Merged.'} ${added}` : data.message || 'Merged from Google Sheet.')
      onRestored()
    } catch {
      onFlash('err', 'Google Sheet sync failed — check server credentials (STITCHING_GCP_*).')
    } finally {
      setGsheetPending(false)
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
      <p className="text-xs text-gray-600">
        Data is stored on the server database (persists across deploys). Export a ZIP backup, restore from ZIP, or pull
        missing master rows from the linked Google Sheet without deleting today&apos;s entries.
      </p>
      <div className="flex flex-wrap gap-2 items-center">
        <button
          type="button"
          onClick={() => void handlePullGsheet()}
          disabled={gsheetPending}
          className="text-xs px-3 py-1.5 rounded-lg border border-green-500 bg-green-50 text-green-900 hover:bg-green-100 disabled:opacity-50"
        >
          {gsheetPending ? 'Pulling…' : '📥 Pull from Google Sheet'}
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

function csvEscapeCell(v: unknown): string {
  const s = String(v ?? '')
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`
  return s
}

function downloadReportCsv(filename: string, rows: Record<string, unknown>[], cols: string[]) {
  if (!rows.length) return
  const useCols = cols.filter(c => rows.some(r => r[c] !== undefined && r[c] !== ''))
  if (!useCols.length) return
  const header = useCols.join(',')
  const body = rows.map(r => useCols.map(c => csvEscapeCell(r[c])).join(',')).join('\n')
  const blob = new Blob([`${header}\n${body}`], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename.endsWith('.csv') ? filename : `${filename}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

function ReportTableSection({
  title,
  rows,
  cols,
  downloadName,
  children,
}: {
  title: string
  rows: Record<string, unknown>[]
  cols: string[]
  downloadName: string
  children?: ReactNode
}) {
  const canDownload = rows.length > 0
  return (
    <Section title={title}>
      <div className="flex flex-wrap items-center justify-end gap-2 mb-2 -mt-1">
        <button
          type="button"
          disabled={!canDownload}
          onClick={() => downloadReportCsv(downloadName, rows, cols)}
          className="px-3 py-1.5 text-xs font-medium rounded-lg border border-[#002B5B] text-[#002B5B] bg-white hover:bg-blue-50 disabled:opacity-40 touch-manipulation"
        >
          ⬇ Download CSV
        </button>
      </div>
      {children ?? <DataTable rows={rows} cols={cols} />}
    </Section>
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
