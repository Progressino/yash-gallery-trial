import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import axios from 'axios'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'
import { useStitchingAdmin } from '../lib/stitchingAdmin'
import {
  applyHourEntryPatch,
  emptyHourEntry,
  isStickerMode,
  normalizeLoadedHourEntry,
  piecesInputValue,
  resolveHourPieces,
  resolveSessionHourPieces,
  type HourEntryState,
} from '../lib/stitchingHourEntry'
import { computeDayFinancialSummary, formatProfitLoss } from '../lib/stitchingFinancial'
import { printStitchingReportsPack } from '../lib/stitchingReportPrint'
import {
  formatProductionEntrySaveError,
  saveProductionEntry,
} from '../lib/productionEntrySave'
import { useAuth } from '../store/auth'

type TabId =
  | 'dashboard'
  | 'production'
  | 'target_control'
  | 'ltl_setup'
  | 'expenses'
  | 'comparison'
  | 'challan'
  | 'style'
  | 'efficiency'
  | 'payroll'
  | 'attendance'
  | 'operating'
  | 'performance'
  | 'reports'
  | 'master'

function normStyleKey(s: string) {
  return s.trim().toLowerCase()
}

const TABS: { id: TabId; label: string }[] = [
  { id: 'dashboard', label: '🏠 Dashboard' },
  { id: 'production', label: '📋 Production Entry' },
  { id: 'target_control', label: '🎯 Target Control' },
  { id: 'ltl_setup', label: '📐 LTL Setup' },
  { id: 'expenses', label: '💸 Karigar Expenses' },
  { id: 'comparison', label: '📈 P&L Compare' },
  { id: 'challan', label: '🧾 Challans' },
  { id: 'style', label: '💎 Style Costing' },
  { id: 'efficiency', label: '📊 Efficiency' },
  { id: 'payroll', label: '💰 Payroll' },
  { id: 'attendance', label: '🕐 Karigar Attendance' },
  { id: 'operating', label: '🏢 Operating Staff' },
  { id: 'performance', label: '🌟 Performance' },
  { id: 'reports', label: '📑 Reports' },
  { id: 'master', label: '⚙️ Master Data' },
]

interface HourDef {
  col: string
  label: string
}

/** Sunday shift is 09:00–16:00 — hide 16–17 and later production hours. */
const SUNDAY_CLOSED_HOUR_COLS = new Set([
  'H_16_17',
  'H_17_18',
  'H_18_19',
  'H_19_20',
  'H_20_21',
])

function hoursForEntryDate(allHours: HourDef[], dateStr: string): HourDef[] {
  const d = new Date(`${dateStr}T12:00:00`)
  if (Number.isNaN(d.getTime()) || d.getDay() !== 0) return allHours
  return allHours.filter(h => !SUNDAY_CLOSED_HOUR_COLS.has(h.col))
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
      {!karigarOnly && tab === 'target_control' && (
        <TargetControlTab admin={admin} onFlash={flash} />
      )}
      {!karigarOnly && tab === 'ltl_setup' && <LtlSetupTab admin={admin} onFlash={flash} />}
      {!karigarOnly && tab === 'expenses' && <KarigarExpensesTab admin={admin} onFlash={flash} />}
      {!karigarOnly && tab === 'comparison' && <ComparisonDashboardTab />}
      {!karigarOnly && tab === 'challan' && <ChallanTab onFlash={flash} />}
      {!karigarOnly && tab === 'style' && <StyleCostingTab />}
      {!karigarOnly && tab === 'efficiency' && <EfficiencyTab />}
      {!karigarOnly && tab === 'payroll' && <PayrollTab />}
      {!karigarOnly && tab === 'attendance' && <AttendanceTab type="karigar" />}
      {!karigarOnly && tab === 'operating' && <AttendanceTab type="operating" />}
      {!karigarOnly && tab === 'performance' && <PerformanceTab />}
      {!karigarOnly && tab === 'reports' && <StitchingReportsTab />}
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

function computeEntrySummary(
  hourState: Record<string, HourEntryState>,
  hours: HourDef[],
  operations: { op: string; target: number; rate: number }[],
  dailyRate: number,
) {
  const sessionPcs = resolveSessionHourPieces(hours, hourState)
  const opTotals: Record<string, { pieces: number; value: number; target: number }> = {}
  for (const h of hours) {
    if (h.col === 'H_13_14') continue
    const st = hourState[h.col]
    const pcs = sessionPcs[h.col] ?? 0
    if (!st?.operation || !pcs) continue
    const meta = operations.find(o => o.op === st.operation)
    if (!meta) continue
    if (!opTotals[st.operation]) opTotals[st.operation] = { pieces: 0, value: 0, target: meta.target }
    opTotals[st.operation].pieces += pcs
    opTotals[st.operation].value += pcs * meta.rate
  }
  const totalValue = Object.values(opTotals).reduce((s, d) => s + d.value, 0)
  const fin = computeDayFinancialSummary(opTotals, dailyRate)
  return {
    totalPcs: fin.totalPcs,
    totalValue,
    totalBudgeted: fin.totalBudgeted,
    totalActual: fin.totalActual,
    pl: fin.pl,
    opTotals,
    opFin: fin.opFin,
    sessionPcs,
  }
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
  const activeHours = useMemo(() => hoursForEntryDate(hours, entryDate), [hours, entryDate])
  const [karigarId, setKarigarId] = useState(lockedKarigarId || '')
  const [style, setStyle] = useState('')
  const [challanNo, setChallanNo] = useState('')
  const [hourState, setHourState] = useState<Record<string, HourEntryState>>({})
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [saveMessage, setSaveMessage] = useState('')
  const [autoSaveEnabled, setAutoSaveEnabled] = useState(true)
  const [hourDeleteBusyKey, setHourDeleteBusyKey] = useState('')
  const skipAutoSaveRef = useRef(true)
  const loadInProgressRef = useRef(false)
  const saveInFlightRef = useRef(false)
  const reportsRefreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const qc = useQueryClient()

  useEffect(() => {
    if (lockedKarigarId) setKarigarId(lockedKarigarId)
  }, [lockedKarigarId])

  useEffect(() => {
    skipAutoSaveRef.current = true
    loadInProgressRef.current = true
  }, [entryDate, karigarId, challanNo, style])

  const { data: karigarSheet } = useQuery({
    queryKey: ['stitching-sheet', 'karigar_master'],
    queryFn: () => api.get('/stitching/sheets/karigar_master').then(r => r.data),
  })
  const { data: styleSheet, refetch: refetchStyles } = useQuery({
    queryKey: ['stitching-sheet', 'style_master'],
    queryFn: () => api.get('/stitching/sheets/style_master').then(r => r.data),
  })
  const [rateEdits, setRateEdits] = useState<Record<string, { Target: number; Rate_Rs: number }>>({})
  const { data: challanSheet, refetch: refetchChallans } = useQuery({
    queryKey: ['stitching-sheet', 'challan_master'],
    queryFn: () => api.get('/stitching/sheets/challan_master').then(r => r.data),
    staleTime: 0,
  })

  const karigars = (karigarSheet?.rows ?? []) as KarigarRow[]
  const allChallansEarly = (challanSheet?.rows ?? []) as { Style?: string }[]
  const styles = useMemo(() => {
    const byKey = new Map<string, string>()
    for (const r of (styleSheet?.rows ?? []) as { Style: string }[]) {
      const label = String(r.Style ?? '').trim()
      if (label) byKey.set(normStyleKey(label), label)
    }
    for (const c of allChallansEarly) {
      const label = String(c.Style ?? '').trim()
      if (label && !byKey.has(normStyleKey(label))) byKey.set(normStyleKey(label), label)
    }
    return [...byKey.values()].sort((a, b) => a.localeCompare(b))
  }, [styleSheet, allChallansEarly])
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
    () => (style ? allChallans.filter(c => normStyleKey(String(c.Style ?? '')) === normStyleKey(style)) : []),
    [allChallans, style],
  )

  useEffect(() => {
    void refetchChallans()
  }, [refetchChallans])

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
      .filter(r => style && normStyleKey(String(r.Style)) === normStyleKey(style))
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
    loadInProgressRef.current = true
    skipAutoSaveRef.current = true
    try {
      const { data } = await api.get('/stitching/production-entry/load', {
        params: { date: entryDate, karigar_id: karigarId, challan_no: challanNo, style },
        timeout: 30_000,
      })
      const next: Record<string, HourEntryState> = {}
      for (const h of activeHours) {
        next[h.col] = normalizeLoadedHourEntry(data.hours?.[h.col] ?? {})
      }
      setHourState(next)
    } catch {
      // Keep in-progress edits if load fails (e.g. server busy during ERP upload).
    } finally {
      loadInProgressRef.current = false
      skipAutoSaveRef.current = true
    }
  }, [entryDate, karigarId, challanNo, style, activeHours])

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

  useEffect(() => {
    return () => {
      if (reportsRefreshTimerRef.current) clearTimeout(reportsRefreshTimerRef.current)
    }
  }, [])

  const scheduleReportsRefresh = useCallback(
    (delayMs: number) => {
      if (reportsRefreshTimerRef.current) clearTimeout(reportsRefreshTimerRef.current)
      reportsRefreshTimerRef.current = setTimeout(() => {
        reportsRefreshTimerRef.current = null
        void refetchReports()
      }, delayMs)
    },
    [refetchReports],
  )

  const [showSalaryCols, setShowSalaryCols] = useState(true)
  useEffect(() => {
    if (karigarId) setShowSalaryCols(true)
  }, [karigarId])
  const showSalary = !!karigarId || showSalaryCols

  const report1Cols = useMemo(() => {
    const base = [
      'Date',
      'Save_Time',
      'Karigar_Name',
      'Challan_No',
      'Challan_Description',
      'Style',
      'Operation',
      'Base_Target',
      'Applied_LTL',
      'Daily_Salary_Rs',
      'Total_Pieces',
      'Budget_Rate_Per_Piece',
      'Budgeted_Expense_Rs',
      'Actual_Rate_Per_Piece',
      'Actual_Expense_Rs',
      'Profit_Loss',
      'Efficiency_%',
      'Working_Hours',
    ]
    return base
  }, [])

  const report2SummaryCols = useMemo(() => {
    const base = ['Date', 'Save_Time', 'Karigar', 'Style', 'Challan_No', 'Challan_Description', 'Operation', 'Hours_Worked', 'Total_Pieces', 'Total_Net_PL', 'Result']
    base.splice(4, 0, 'Daily_Salary_Rs', 'Total_Salary_Cost')
    return base
  }, [])

  const report2HourlyCols = useMemo(() => {
    const base = ['Date', 'Save_Time', 'Karigar', 'Style', 'Challan_No', 'Challan_Description', 'Daily_Salary_Rs', 'Hourly_Salary_Rs', 'Hour', 'Operation', 'Pieces_Done', 'Actual_Piece_Val_Rs', 'Net_PL_Rs', 'Status']
    return base
  }, [])

  const deleteHourRow = async (row: Report2HourlyRow) => {
    const pw = admin.adminPassword()
    if (!pw) {
      alert('Unlock admin first (Admin unlock panel at top of Stitching Costing page).')
      return
    }
    let resolvedKarigarId = String(row.Karigar_ID || '').trim()
    if (!resolvedKarigarId && row.Karigar) {
      const hit = karigars.find(
        k =>
          String(k.Name || '').trim().toLowerCase() === String(row.Karigar).trim().toLowerCase(),
      )
      resolvedKarigarId = hit ? String(hit.Karigar_ID) : ''
    }
    if (!resolvedKarigarId) {
      alert('Could not resolve karigar ID for this row. Try deleting from Admin — edit or delete production.')
      return
    }
    if (
      !window.confirm(
        `Delete ${row.Hour} entry for ${row.Karigar} / ${row.Operation}? Related summary rows will auto-update.`,
      )
    ) {
      return
    }
    const busyKey = `${row.Date}::${resolvedKarigarId}::${row.Challan_No}::${row.Style}::${row.Operation}::${row.Hour}`
    setHourDeleteBusyKey(busyKey)
    try {
      await api.post('/stitching/production-entry/admin/delete-hour', {
        date: row.Date,
        karigar_id: resolvedKarigarId,
        challan_no: row.Challan_No,
        style: row.Style,
        operation: row.Operation,
        hour: row.Hour,
        admin_password: pw,
      })
      await refetchReports()
      if (
        resolvedKarigarId === karigarId &&
        entryDate === row.Date &&
        challanNo === row.Challan_No &&
        normStyleKey(style) === normStyleKey(row.Style)
      ) {
        await loadEntry()
      }
    } catch (e: unknown) {
      alert(axios.isAxiosError(e) ? String(e.response?.data?.detail || 'Delete failed') : 'Delete failed')
    } finally {
      setHourDeleteBusyKey('')
    }
  }

  const historyCols = useMemo(
    () => [
      'Date',
      'Save_Time',
      'Karigar_Name',
      'Challan_No',
      'Style',
      'Operation',
      'Total_Pieces',
      'Budgeted_Expense_Rs',
      'Actual_Expense_Rs',
      'Profit_Loss',
    ],
    [],
  )

  const karigarDailyRate = useMemo(() => {
    if (!karigarId) return 0
    const k = karigars.find(x => String(x.Karigar_ID) === karigarId)
    return Number(k?.Daily_Rate_Rs) || 0
  }, [karigars, karigarId])

  const liveSummary = useMemo(
    () => computeEntrySummary(hourState, activeHours, operations, karigarDailyRate),
    [hourState, activeHours, operations, karigarDailyRate],
  )

  const buildHourEntries = useCallback(() => {
    const sessionPcs = resolveSessionHourPieces(activeHours, hourState)
    return activeHours
      .filter(h => h.col !== 'H_13_14')
      .map(h => {
        const st = hourState[h.col] ?? emptyHourEntry()
        return {
          hour_col: h.col,
          operation: st.operation,
          pieces: sessionPcs[h.col] ?? 0,
          sticker_in: st.sticker_in,
          sticker_out: st.sticker_out,
          manual_pieces: st.manual_pieces,
        }
      })
  }, [activeHours, hourState])

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
    for (const h of activeHours) {
      if (h.col === 'H_13_14') continue
      const st = hourState[h.col]
      const pcs = resolveHourPieces(st)
      if (pcs > 0 && !st?.operation) n += pcs
    }
    return n
  }, [activeHours, hourState])

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
      if (saveInFlightRef.current) return
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
      saveInFlightRef.current = true
      setSaveStatus('saving')
      setSaveMessage('')
      try {
        const data = await saveProductionEntry({
          date: entryDate,
          karigar_id: karigarId,
          karigar_name: k?.Name ?? karigarId,
          challan_no: challanNo,
          style,
          hour_entries: entries,
        })
        if (data.ok) {
          setSaveStatus('saved')
          setSaveMessage(data.message || 'Saved')
          if (silent) {
            scheduleReportsRefresh(8000)
          } else {
            onSaved()
            void refetchReports()
            qc.invalidateQueries({ queryKey: ['stitching-dashboard'] })
            alert(data.message || 'Saved')
          }
        } else {
          const msg = data.message || 'Save failed'
          setSaveStatus('error')
          setSaveMessage(msg)
          if (!silent) alert(msg)
        }
      } catch (err: unknown) {
        const msg = formatProductionEntrySaveError(err)
        setSaveStatus('error')
        setSaveMessage(msg)
        if (!silent) alert(msg)
      } finally {
        saveInFlightRef.current = false
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
      scheduleReportsRefresh,
      qc,
    ],
  )

  useEffect(() => {
    if (!autoSaveEnabled || skipAutoSaveRef.current || loadInProgressRef.current || saveInFlightRef.current) return
    if (!karigarId || !challanNo || !style || saveBlockReason) return
    const t = setTimeout(() => void doSave(true), 2500)
    return () => clearTimeout(t)
  }, [hourState, entryDate, karigarId, challanNo, style, autoSaveEnabled, saveBlockReason, doSave])

  const saveMut = useMutation({
    mutationFn: () => doSave(false),
  })

  const setHour = (col: string, patch: Partial<HourEntryState>) => {
    skipAutoSaveRef.current = false
    setHourState(prev => {
      const merged = { ...prev, [col]: applyHourEntryPatch(prev[col], patch) }
      const sessionPcs = resolveSessionHourPieces(activeHours, merged)
      for (const h of activeHours) {
        if (h.col === 'H_13_14') continue
        const st = merged[h.col]
        if (!st || st.manual_pieces) continue
        if (st.sticker_in !== 0 || st.sticker_out !== 0) {
          merged[h.col] = { ...st, pieces: sessionPcs[h.col] ?? 0, manual_pieces: false }
        }
      }
      const base = merged[col]
      if (operations.length === 1 && (sessionPcs[col] ?? 0) > 0 && !base.operation) {
        base.operation = operations[0].op
      }
      return merged
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
          emptyMessage={
            style
              ? challansForStyle.length === 0 && allChallans.length > 0
                ? 'No challan for this style — check Style spelling matches Challans tab'
                : 'No challan for this style — add in Challans tab'
              : 'Select a style first'
          }
          hint={style ? `${challansForStyle.length} challan(s) for ${style} · ${allChallans.length} total in register` : undefined}
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
          <strong>Sticker in / out</strong> — pieces = |in − out|; sticker out only uses hourly count (rising out = cumulative delta).
          Leave stickers empty to type pieces manually. Same challan re-save replaces the previous entry.
        </p>

        <div className="space-y-2 mb-3 md:hidden">
          {activeHours.map(h => {
            if (h.col === 'H_13_14') {
              return (
                <div key={h.col} className="rounded-lg bg-gray-50 border border-dashed px-3 py-2 text-xs text-gray-400 italic text-center">
                  {h.label} — Lunch
                </div>
              )
            }
            const st = hourState[h.col] ?? emptyHourEntry()
            const pcs = liveSummary.sessionPcs?.[h.col] ?? resolveHourPieces(st)
            const fromSticker = isStickerMode(st)
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
                      value={piecesInputValue(st, liveSummary.sessionPcs, h.col)}
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
              {activeHours.map(h => {
                if (h.col === 'H_13_14') {
                  return (
                    <tr key={h.col} className="bg-gray-50 text-gray-400 italic">
                      <td className="px-2 py-2 text-center">{h.label}</td>
                      <td colSpan={6} className="px-2 py-2">Lunch break</td>
                    </tr>
                  )
                }
                const st = hourState[h.col] ?? emptyHourEntry()
                const pcs = liveSummary.sessionPcs?.[h.col] ?? resolveHourPieces(st)
                const fromSticker = isStickerMode(st)
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
                        value={piecesInputValue(st, liveSummary.sessionPcs, h.col)}
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
          <div className="mt-4 space-y-3">
            <p className="text-[10px] text-gray-500">
              Stickers: in − out per hour, or <strong>sticker out only</strong> (rising counts use hourly delta).
              Financial audit: budget rate = ₹480 ÷ base target · budgeted capped at ₹480/day total · actual = daily wage ·
              P&amp;L = budgeted − actual. Re-saving the same date + challan + style updates the row (no duplicate).
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <div className="bg-white border rounded-lg p-3 text-center">
                <p className="text-[10px] uppercase text-gray-500">Pieces</p>
                <p className="text-lg font-bold text-[#2c5aa0]">{liveSummary.totalPcs}</p>
              </div>
              <div className="bg-white border rounded-lg p-3 text-center">
                <p className="text-[10px] uppercase text-gray-500">Budgeted</p>
                <p className="text-lg font-bold text-[#2c5aa0]">₹{liveSummary.totalBudgeted.toFixed(2)}</p>
              </div>
              <div className="bg-white border rounded-lg p-3 text-center">
                <p className="text-[10px] uppercase text-gray-500">Actual (daily)</p>
                <p className="text-lg font-bold text-amber-800">₹{liveSummary.totalActual.toFixed(2)}</p>
              </div>
              <div className="bg-white border rounded-lg p-3 text-center">
                <p className="text-[10px] uppercase text-gray-500">Profit / Loss</p>
                <p className={`text-lg font-bold ${liveSummary.pl >= 0 ? 'text-green-700' : 'text-red-600'}`}>
                  {formatProfitLoss(liveSummary.pl)}
                </p>
              </div>
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

      {admin.unlocked && !karigarOnly && (
        <AdminProductionPanel
          entryDate={entryDate}
          karigarId={karigarId}
          admin={admin}
          onEdit={row => {
            setEntryDate(String(row.Date || entryDate))
            setKarigarId(String(row.Karigar_ID || ''))
            setChallanNo(String(row.Challan_No || ''))
            setStyle(String(row.Style || ''))
          }}
          onDeleted={() => {
            void refetchReports()
            qc.invalidateQueries({ queryKey: ['stitching-dashboard'] })
          }}
        />
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
          {!admin.unlocked && (
            <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mb-2">
              Unlock admin (panel at top of page) to use <strong>Delete hour</strong>.
            </p>
          )}
          <details className="text-xs">
            <summary className="cursor-pointer py-3 font-medium text-[#2c5aa0] touch-manipulation min-h-[44px]">
              Show hour-wise rows ({entryReports.report2_hourly.length})
            </summary>
            <div className="mt-2 max-h-64 overflow-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-gray-50 border-b">
                    {report2HourlyCols.map(c => (
                      <th key={c} className="text-left px-2 py-1.5 whitespace-nowrap">
                        {c.replace(/_/g, ' ')}
                      </th>
                    ))}
                    <th className="text-left px-2 py-1.5 whitespace-nowrap">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {(entryReports.report2_hourly as Report2HourlyRow[]).map((r, i) => {
                    const busyKey = `${r.Date}::${r.Karigar_ID}::${r.Challan_No}::${r.Style}::${r.Operation}::${r.Hour}`
                    return (
                      <tr key={`${busyKey}-${i}`} className="border-b">
                        {report2HourlyCols.map(c => (
                          <td key={c} className="px-2 py-1.5 whitespace-nowrap">
                            {String((r as Record<string, unknown>)[c] ?? '')}
                          </td>
                        ))}
                        <td className="px-2 py-1.5 whitespace-nowrap">
                          <button
                            type="button"
                            disabled={hourDeleteBusyKey === busyKey}
                            className="px-2 py-1 rounded border border-red-300 text-red-700 disabled:opacity-50"
                            onClick={() => void deleteHourRow(r)}
                          >
                            Delete hour
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
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
        <ChallanRegisterTable rows={rows} onFlash={onFlash} onSaved={invalidateChallan} onDelete={invalidateChallan} />
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
  onDelete,
}: {
  rows: Record<string, unknown>[]
  onFlash: (type: 'ok' | 'err', text: string) => void
  onSaved: () => void
  onDelete?: () => void
}) {
  const [draft, setDraft] = useState<Record<string, { received?: number; total?: number; rate?: number }>>({})
  const saveTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})

  useEffect(() => {
    return () => {
      Object.values(saveTimers.current).forEach(clearTimeout)
    }
  }, [])

  const scheduleSave = (challanNo: string, patch: { Received_Qty?: number; Total_Qty?: number; Rate_Per_Pc?: number }) => {
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
            <th className="px-2 py-2 w-20 text-right font-semibold text-gray-600">Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const no = String(r.Challan_No ?? '')
            const total = draft[no]?.total ?? Number(r.Total_Qty ?? 0)
            const received = draft[no]?.received ?? Number(r.Received_Qty ?? 0)
            const rate = draft[no]?.rate ?? Number(r.Rate_Per_Pc ?? 0)
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
                <td className="px-2 py-1.5">
                  <input
                    type="number"
                    min={0}
                    step={0.5}
                    className="w-20 border rounded px-1 py-0.5 text-right bg-amber-50"
                    value={rate}
                    onChange={e => {
                      const v = +e.target.value || 0
                      setDraft(d => ({ ...d, [no]: { ...d[no], rate: v, total, received } }))
                      scheduleSave(no, { Rate_Per_Pc: v, Total_Qty: total, Received_Qty: received })
                    }}
                  />
                </td>
                <td className="px-2 py-1.5 whitespace-nowrap">{String(r.Date ?? '')}</td>
                <td className="px-2 py-1.5 text-right">
                  <button
                    type="button"
                    className="text-red-600 hover:underline font-semibold"
                    onClick={async () => {
                      if (!no || !window.confirm(`Delete challan ${no}? This cannot be undone.`)) return
                      try {
                        await api.delete(`/stitching/challans/${encodeURIComponent(no)}`)
                        onFlash('ok', `Challan ${no} deleted.`)
                        onSaved()
                        onDelete?.()
                      } catch {
                        onFlash('err', `Could not delete ${no}.`)
                      }
                    }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function ComparisonDashboardTab() {
  const [from, setFrom] = useState(() => {
    const d = new Date()
    d.setDate(d.getDate() - 6)
    return d.toISOString().slice(0, 10)
  })
  const [to, setTo] = useState(todayStr())
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['stitching-comparison', from, to],
    queryFn: () =>
      api.get('/stitching/comparison-dashboard', { params: { date_from: from, date_to: to } }).then(r => r.data),
  })
  const s = data?.summary
  const plClass = (v: number) => (v < 0 ? 'text-rose-700 font-semibold' : 'text-emerald-700 font-semibold')

  return (
    <div className="space-y-4">
      <p className="text-xs text-gray-600">
        Compare benchmark (₹480/day) budget vs actual karigar cost, running LTL, and P&amp;L by karigar, SKU, and challan.
      </p>
      <div className="flex flex-wrap gap-2 items-end text-xs">
        <label>
          From
          <input type="date" className="block border rounded mt-1 px-2 py-1" value={from} onChange={e => setFrom(e.target.value)} />
        </label>
        <label>
          To
          <input type="date" className="block border rounded mt-1 px-2 py-1" value={to} onChange={e => setTo(e.target.value)} />
        </label>
        <button type="button" onClick={() => void refetch()} className="px-3 py-1.5 bg-[#002B5B] text-white rounded-lg">
          {isLoading ? 'Loading…' : 'Refresh'}
        </button>
      </div>
      {s && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
          <div className="bg-white border rounded-lg p-3">
            <p className="text-gray-500 text-xs">Karigars at loss</p>
            <p className={`font-bold ${plClass(-1)}`}>{s.karigars_at_loss}</p>
          </div>
          <div className="bg-white border rounded-lg p-3">
            <p className="text-gray-500 text-xs">Karigars at profit</p>
            <p className="font-bold text-emerald-700">{s.karigars_at_profit}</p>
          </div>
          <div className="bg-white border rounded-lg p-3">
            <p className="text-gray-500 text-xs">SKUs at loss</p>
            <p className={`font-bold ${plClass(-1)}`}>{s.skus_at_loss}</p>
          </div>
          <div className="bg-white border rounded-lg p-3">
            <p className="text-gray-500 text-xs">Challans over budget</p>
            <p className="font-bold text-amber-700">{s.challans_over_budget}</p>
          </div>
          <div className="bg-white border rounded-lg p-3 col-span-2">
            <p className="text-gray-500 text-xs">Net P&amp;L (budget − actual)</p>
            <p className={`font-bold text-lg ${plClass(s.total_net_pl_rs)}`}>₹{Number(s.total_net_pl_rs).toLocaleString()}</p>
          </div>
        </div>
      )}
      <ReportTableSection
        title="Karigar — actual vs budget (who is losing money)"
        rows={(data?.karigar_comparison ?? []) as Record<string, unknown>[]}
        cols={['Karigar_Name', 'Pieces', 'Budgeted_Rs', 'Actual_Rs', 'Net_PL_Rs', 'Running_LTL', 'Variance_%', 'Status']}
        downloadName={`pl_compare_karigar_${from}_${to}`}
      />
      <ReportTableSection
        title="SKU — profit / loss"
        rows={(data?.sku_comparison ?? []) as Record<string, unknown>[]}
        cols={['Style', 'Pieces', 'Piece_Value_Rs', 'Budgeted_Rs', 'Actual_Rs', 'Net_PL_Rs', 'Status']}
        downloadName={`pl_compare_sku_${from}_${to}`}
      />
      <ReportTableSection
        title="Challan — over / under budget"
        rows={(data?.challan_comparison ?? []) as Record<string, unknown>[]}
        cols={['Challan_No', 'Style', 'Budgeted_Rs', 'Actual_Rs', 'Net_PL_Rs', 'Status']}
        downloadName={`pl_compare_challan_${from}_${to}`}
      />
      <ReportTableSection
        title="Karigar × SKU detail"
        rows={(data?.karigar_sku_detail ?? []) as Record<string, unknown>[]}
        cols={['Karigar_Name', 'Style', 'Pieces', 'Budgeted_Rs', 'Actual_Rs', 'Net_PL_Rs', 'Running_LTL', 'Status']}
        downloadName={`pl_compare_karigar_sku_${from}_${to}`}
      />
    </div>
  )
}

function StyleCostingTab() {
  const [month, setMonth] = useState('All')
  const [style, setStyle] = useState('All')
  const { data, isLoading, isFetching, isError, refetch } = useQuery({
    queryKey: ['stitching-style-costing', month, style],
    queryFn: () => api.get('/stitching/style-costing', { params: { month, style } }).then(r => r.data),
  })
  const s = data?.summary
  return (
    <div className="space-y-4">
      {(isLoading || isFetching) && (
        <div className="flex items-center gap-2 text-sm text-[#2c5aa0] bg-blue-50 border border-blue-100 rounded-lg px-4 py-3">
          <span className="inline-block w-4 h-4 border-2 border-[#2c5aa0] border-t-transparent rounded-full animate-spin" />
          Loading style costing…
        </div>
      )}
      {isError && (
        <p className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
          Failed to load style costing.{' '}
          <button type="button" className="underline font-semibold" onClick={() => void refetch()}>
            Retry
          </button>
        </p>
      )}
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
  const rows = ((data?.rows ?? []) as Record<string, unknown>[]).filter(r => Number(r.Total ?? 0) > 0)
  const payrollCols = [
    'Karigar_ID',
    'Name',
    'Days',
    'Hrs',
    'Normal',
    'OT_Pay',
    'Attendance_Pay',
    'Other_Hours',
    'Other_Work_Pay',
    'Total',
  ]
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
          <div className="flex flex-wrap gap-4 text-sm">
            <p className="text-lg font-bold text-[#2c5aa0]">
              Total payroll: ₹{Number(data.total_payroll).toLocaleString()}
            </p>
            {data.total_attendance != null && (
              <p className="text-gray-600">Attendance: ₹{Number(data.total_attendance).toLocaleString()}</p>
            )}
            {data.total_other_work != null && (
              <p className="text-gray-600">Other work: ₹{Number(data.total_other_work).toLocaleString()}</p>
            )}
          </div>
          <p className="text-xs text-gray-500">
            Includes attendance pay plus karigar expenses (part change, alter, trainee, etc.) from the Karigar Expenses tab.
          </p>
          <ReportTableSection
            title="Karigar payroll (all expenses)"
            rows={rows}
            cols={payrollCols}
            downloadName={`payroll_${from}_${to}`}
          />
        </>
      )}
    </div>
  )
}

function StitchingReportsTab() {
  const [from, setFrom] = useState(() => {
    const d = new Date()
    d.setDate(d.getDate() - 6)
    return d.toISOString().slice(0, 10)
  })
  const [to, setTo] = useState(todayStr())
  const [printing, setPrinting] = useState(false)
  const { data, refetch, isFetching } = useQuery({
    queryKey: ['stitching-reports-hub', from, to],
    queryFn: () =>
      api.get('/stitching/reports/hub', { params: { date_from: from, date_to: to } }).then(r => r.data),
    enabled: false,
  })

  const kp = data?.karigar_profitability?.summary
  const ch = data?.challan_labour?.summary

  return (
    <div className="space-y-4">
      <Section title="How payroll is calculated">
        <div className="text-xs text-gray-600 space-y-2 max-w-3xl">
          <p>
            <strong>Attendance pay</strong> comes from biometric punches (weekdays 09:00–18:00; Sunday 09:00–16:00
            with 13:00–14:00 lunch; hourly = daily ÷ 8 weekdays, daily ÷ 6 Sunday). Stored in{' '}
            <em>Karigar Attendance</em> as Normal + OT = Total_Pay.
          </p>
          <p>
            <strong>Other work pay</strong> is every row in <em>Karigar Expenses</em> (part change, alter, trainee, etc.)
            with amount on the challan.
          </p>
          <p>
            <strong>Total payroll</strong> = attendance + other work. Production costing still uses ₹480/day benchmark
            and daily-rate allocation for P&amp;L; this report pack compares <em>actual payroll paid</em> to piece value
            and benchmark.
          </p>
        </div>
      </Section>
      <div className="flex flex-wrap gap-2 items-end text-xs">
        <label>
          From
          <input type="date" className="block border rounded mt-1 px-2 py-1" value={from} onChange={e => setFrom(e.target.value)} />
        </label>
        <label>
          To
          <input type="date" className="block border rounded mt-1 px-2 py-1" value={to} onChange={e => setTo(e.target.value)} />
        </label>
        <button type="button" onClick={() => void refetch()} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm">
          Run all reports
        </button>
        <button
          type="button"
          disabled={printing}
          onClick={async () => {
            setPrinting(true)
            try {
              await printStitchingReportsPack(from, to)
            } catch (e: unknown) {
              alert(e instanceof Error ? e.message : 'Could not open print view')
            } finally {
              setPrinting(false)
            }
          }}
          className="px-4 py-2 border border-[#002B5B] text-[#002B5B] bg-white rounded-lg text-sm font-medium hover:bg-blue-50 disabled:opacity-50"
        >
          {printing ? 'Opening…' : '🖨 Print / Save PDF'}
        </button>
        {data?.generated_at && <span className="text-gray-500">Generated {data.generated_at}</span>}
      </div>
      <p className="text-xs text-gray-500 -mt-2">
        Print opens a new tab with all sections; choose <strong>Save as PDF</strong> in the print dialog.
      </p>
      {isFetching && <p className="text-sm text-gray-500">Building report pack…</p>}
      {data && (
        <>
          {kp && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
              <div className="bg-white border rounded-lg p-3">
                <p className="text-xs text-gray-500">Total payroll paid</p>
                <p className="font-bold text-[#2c5aa0]">₹{Number(kp.total_payroll_paid).toLocaleString()}</p>
              </div>
              <div className="bg-white border rounded-lg p-3">
                <p className="text-xs text-gray-500">Profitable on payroll</p>
                <p className="font-bold text-emerald-700">{kp.profitable_on_payroll} / {kp.karigar_count}</p>
              </div>
              <div className="bg-white border rounded-lg p-3">
                <p className="text-xs text-gray-500">Profitable on benchmark</p>
                <p className="font-bold text-emerald-700">{kp.profitable_on_benchmark} / {kp.karigar_count}</p>
              </div>
              <div className="bg-white border rounded-lg p-3">
                <p className="text-xs text-gray-500">Net P&amp;L (benchmark)</p>
                <p className="font-bold">₹{Number(kp.total_net_pl_benchmark).toLocaleString()}</p>
              </div>
            </div>
          )}
          <ReportTableSection
            title="Karigar profitability — payroll paid vs piece value vs benchmark (LTL column)"
            rows={(data.karigar_profitability?.rows ?? []) as Record<string, unknown>[]}
            cols={[
              'Karigar_Name',
              'Total_Payroll_Paid',
              'Attendance_Pay',
              'Other_Work_Pay',
              'Piece_Value_Rs',
              'Pay_vs_Piece_Rs',
              'Profitable_On_Payroll',
              'Budgeted_Rs',
              'Net_PL_Benchmark',
              'Profitable_On_Benchmark',
              'Running_LTL',
              'LTL_Note',
              'Payroll_Only',
            ]}
            downloadName={`karigar_profitability_${from}_${to}`}
          />
          <ReportTableSection
            title="Challan labour — budget, payroll paid (expense + allocated attendance), P&amp;L"
            rows={(data.challan_labour?.rows ?? []) as Record<string, unknown>[]}
            cols={[
              'Challan_No',
              'Style',
              'Karigar_Name',
              'Pieces',
              'Piece_Value_Rs',
              'Budgeted_Labour_Rs',
              'Total_Payroll_Paid',
              'Expense_On_Challan_Rs',
              'Attendance_Allocated_Rs',
              'Pay_vs_Budget',
              'Net_PL_Benchmark',
              'Profitable_On_Payroll',
              'Profitable_On_Benchmark',
            ]}
            downloadName={`challan_labour_${from}_${to}`}
          />
          {ch && (
            <p className="text-xs text-gray-500">
              Challan lines: {ch.challan_lines} · Total payroll on challans: ₹
              {Number(ch.total_payroll_paid).toLocaleString()}
            </p>
          )}
          <ReportTableSection
            title="Payroll register (all karigar expenses)"
            rows={(data.payroll?.rows ?? []) as Record<string, unknown>[]}
            cols={[
              'Karigar_ID',
              'Name',
              'Days',
              'Attendance_Pay',
              'Other_Work_Pay',
              'Total',
            ]}
            downloadName={`payroll_register_${from}_${to}`}
          />
          {data.performance?.ok && (
            <ReportTableSection
              title="Performance — piece value vs full payroll"
              rows={(data.performance.rows ?? []) as Record<string, unknown>[]}
              cols={[
                'Name',
                'Total_Payroll_Paid',
                'Attendance_Pay',
                'Other_Work_Pay',
                'Piece_Value',
                'Surplus',
                'Profitable_On_Payroll',
                'Avg_Eff',
                'Grade',
              ]}
              downloadName={`performance_${from}_${to}`}
            />
          )}
        </>
      )}
    </div>
  )
}

type ProductionSessionRow = {
  Date: string
  Karigar_ID: string
  Karigar_Name: string
  Challan_No: string
  Style: string
  Operation: string
  Total_Pieces: number
}

type Report2HourlyRow = {
  Date: string
  Karigar_ID: string
  Karigar: string
  Challan_No: string
  Style: string
  Operation: string
  Hour: string
}

function AdminProductionPanel({
  entryDate,
  karigarId,
  admin,
  onEdit,
  onDeleted,
}: {
  entryDate: string
  karigarId: string
  admin: AdminApi
  onEdit: (row: ProductionSessionRow) => void
  onDeleted: () => void
}) {
  const [busy, setBusy] = useState('')
  const { data, refetch } = useQuery({
    queryKey: ['stitching-admin-sessions', entryDate, karigarId],
    queryFn: () =>
      api
        .get('/stitching/production-entry/admin/sessions', {
          params: { date: entryDate, karigar_id: karigarId || undefined },
        })
        .then(r => r.data),
    enabled: admin.unlocked,
  })
  const sessions = (data?.sessions ?? []) as ProductionSessionRow[]

  const doDelete = async (row: ProductionSessionRow, scope: 'operation' | 'session') => {
    const pw = admin.adminPassword()
    if (!pw) {
      alert('Unlock admin first')
      return
    }
    const label =
      scope === 'operation'
        ? `${row.Operation} on ${row.Challan_No}`
        : `all operations on challan ${row.Challan_No} / ${row.Style}`
    if (!window.confirm(`Delete ${label} for ${row.Karigar_Name}? This removes it from all four production reports.`)) {
      return
    }
    const key = `${row.Karigar_ID}-${row.Challan_No}-${scope}`
    setBusy(key)
    try {
      await api.post('/stitching/production-entry/admin/delete', {
        date: row.Date || entryDate,
        karigar_id: row.Karigar_ID,
        challan_no: row.Challan_No,
        style: row.Style,
        operation: scope === 'operation' ? row.Operation : '',
        admin_password: pw,
      })
      await refetch()
      onDeleted()
    } catch (e: unknown) {
      alert(axios.isAxiosError(e) ? String(e.response?.data?.detail || 'Delete failed') : 'Delete failed')
    } finally {
      setBusy('')
    }
  }

  return (
    <Section title="Admin — edit or delete production (all reports)">
      <p className="text-xs text-gray-600 mb-2">
        Deleting here removes rows from production log — history, Report 1, and both Report 2 views update together.
      </p>
      {!sessions.length ? (
        <p className="text-sm text-gray-400">No production rows for this date{karigarId ? ' / karigar' : ''}.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="text-left px-2 py-2">Karigar</th>
                <th className="text-left px-2 py-2">Challan</th>
                <th className="text-left px-2 py-2">Style</th>
                <th className="text-left px-2 py-2">Operation</th>
                <th className="text-right px-2 py-2">Pcs</th>
                <th className="text-right px-2 py-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((row, i) => {
                const key = `${row.Karigar_ID}-${row.Challan_No}-${row.Operation}-${i}`
                return (
                  <tr key={key} className="border-b border-gray-50">
                    <td className="px-2 py-1.5">{row.Karigar_Name || row.Karigar_ID}</td>
                    <td className="px-2 py-1.5">{row.Challan_No}</td>
                    <td className="px-2 py-1.5">{row.Style}</td>
                    <td className="px-2 py-1.5">{row.Operation}</td>
                    <td className="px-2 py-1.5 text-right">{row.Total_Pieces}</td>
                    <td className="px-2 py-1.5 text-right whitespace-nowrap space-x-1">
                      <button
                        type="button"
                        className="px-2 py-0.5 rounded border text-[#002B5B]"
                        onClick={() => onEdit(row)}
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        disabled={busy === key}
                        className="px-2 py-0.5 rounded border text-amber-800"
                        onClick={() => void doDelete(row, 'operation')}
                      >
                        Del op
                      </button>
                      <button
                        type="button"
                        disabled={busy === key}
                        className="px-2 py-0.5 rounded border text-red-700"
                        onClick={() => void doDelete(row, 'session')}
                      >
                        Del session
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  )
}

function LtlSetupTab({
  admin,
  onFlash,
}: {
  admin: AdminApi
  onFlash: (type: 'ok' | 'err', text: string) => void
}) {
  const qc = useQueryClient()
  const [style, setStyle] = useState('')
  const [operation, setOperation] = useState('')
  const [karigarId, setKarigarId] = useState('')
  const [manualLtl, setManualLtl] = useState('')
  const [notes, setNotes] = useState('')
  const [applyAllStyles, setApplyAllStyles] = useState(false)
  const [bandRows, setBandRows] = useState<{ Min_Rs: string; Max_Rs: string; Tolerance_Pct: string }[]>([])

  const { data: setup, refetch } = useQuery({
    queryKey: ['stitching-ltl-setup'],
    queryFn: () => api.get('/stitching/ltl-setup').then(r => r.data),
  })

  const { data: styleSheet } = useQuery({
    queryKey: ['stitching-sheet', 'style_master'],
    queryFn: () => api.get('/stitching/sheets/style_master').then(r => r.data),
  })
  const { data: karSheet } = useQuery({
    queryKey: ['stitching-sheet', 'karigar_master'],
    queryFn: () => api.get('/stitching/sheets/karigar_master').then(r => r.data),
  })

  const styles = useMemo(() => {
    const s = new Set<string>()
    for (const r of (styleSheet?.rows ?? []) as { Style: string }[]) {
      if (r.Style) s.add(String(r.Style))
    }
    return [...s].sort()
  }, [styleSheet])

  const operations = useMemo(() => {
    const s = new Set<string>()
    for (const r of (styleSheet?.rows ?? []) as { Style: string; Operation: string }[]) {
      if (!style || r.Style === style) if (r.Operation) s.add(String(r.Operation))
    }
    return [...s].sort()
  }, [styleSheet, style])

  useEffect(() => {
    const raw = (setup?.tolerance_band_rows ?? []) as { Min_Rs: number; Max_Rs: number; Tolerance_Pct: number }[]
    if (raw.length) {
      setBandRows(
        raw.map(b => ({
          Min_Rs: String(b.Min_Rs ?? ''),
          Max_Rs: String(b.Max_Rs ?? ''),
          Tolerance_Pct: String(b.Tolerance_Pct ?? ''),
        })),
      )
    }
  }, [setup?.tolerance_band_rows])

  const saveBands = async () => {
    const pw = admin.adminPassword()
    if (!pw) {
      onFlash('err', 'Unlock admin to edit tolerance bands')
      return
    }
    try {
      const { data } = await api.put('/stitching/ltl-setup/tolerance-bands', {
        bands: bandRows.map(b => ({
          Min_Rs: parseFloat(b.Min_Rs) || 0,
          Max_Rs: parseFloat(b.Max_Rs) || 0,
          Tolerance_Pct: parseFloat(b.Tolerance_Pct) || 0,
        })),
        admin_password: pw,
        recalculate_production: true,
      })
      onFlash('ok', data.message || 'Bands saved')
      void refetch()
      qc.invalidateQueries({ queryKey: ['stitching-pe-reports'] })
    } catch (e: unknown) {
      onFlash('err', axios.isAxiosError(e) ? String(e.response?.data?.detail || 'Save failed') : 'Save failed')
    }
  }

  const saveOverride = async () => {
    const pw = admin.adminPassword()
    if (!pw) {
      onFlash('err', 'Unlock admin to set manual LTL')
      return
    }
    if ((!applyAllStyles && !style) || !operation || !karigarId) {
      onFlash('err', applyAllStyles ? 'Operation and karigar are required' : 'Style, operation, and karigar are required')
      return
    }
    const manual = manualLtl === '' ? null : Math.max(0, parseInt(manualLtl, 10) || 0)
    try {
      const { data } = await api.put('/stitching/target-control/override', {
        Style: style,
        Operation: operation,
        Karigar_ID: karigarId,
        Manual_LTL: manual,
        Notes: notes,
        apply_all_styles: applyAllStyles,
        admin_password: pw,
      })
      onFlash('ok', data.message || 'LTL override saved')
      setManualLtl('')
      setNotes('')
      void refetch()
      qc.invalidateQueries({ queryKey: ['stitching-pe-reports'] })
      qc.invalidateQueries({ queryKey: ['stitching-target-control'] })
    } catch (e: unknown) {
      onFlash('err', axios.isAxiosError(e) ? String(e.response?.data?.detail || 'Save failed') : 'Save failed')
    }
  }

  const overrides = (setup?.overrides ?? []) as Record<string, unknown>[]

  return (
    <div className="space-y-4">
      <Section title="LTL Setup — manual overrides">
        <p className="text-xs text-gray-600 mb-3">
          Set manual LTL per style / operation / karigar. Production entry and costing use the applied LTL automatically.
          Clear manual LTL (leave blank) to revert to formula.
        </p>
        <label className="flex items-center gap-2 text-xs mb-2">
          <input type="checkbox" checked={applyAllStyles} onChange={e => setApplyAllStyles(e.target.checked)} />
          Apply manual LTL to <strong>all styles</strong> for this operation + karigar
        </label>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
          <label className={applyAllStyles ? 'opacity-50' : ''}>
            Style
            <select
              className="block w-full border rounded mt-1 px-2 py-1.5"
              value={style}
              disabled={applyAllStyles}
              onChange={e => setStyle(e.target.value)}
            >
              <option value="">—</option>
              {styles.map(s => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label>
            Operation
            <select
              className="block w-full border rounded mt-1 px-2 py-1.5"
              value={operation}
              onChange={e => setOperation(e.target.value)}
            >
              <option value="">—</option>
              {operations.map(o => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </label>
          <label>
            Karigar
            <select
              className="block w-full border rounded mt-1 px-2 py-1.5"
              value={karigarId}
              onChange={e => setKarigarId(e.target.value)}
            >
              <option value="">—</option>
              {((karSheet?.rows ?? []) as { Karigar_ID: string; Name: string }[]).map(k => (
                <option key={k.Karigar_ID} value={k.Karigar_ID}>
                  {k.Name || k.Karigar_ID}
                </option>
              ))}
            </select>
          </label>
          <label>
            Manual LTL (pcs/hr)
            <input
              type="number"
              className="block w-full border rounded mt-1 px-2 py-1.5"
              value={manualLtl}
              onChange={e => setManualLtl(e.target.value)}
              placeholder="Blank = formula only"
            />
          </label>
          <label className="sm:col-span-2">
            Notes
            <input
              className="block w-full border rounded mt-1 px-2 py-1.5"
              value={notes}
              onChange={e => setNotes(e.target.value)}
            />
          </label>
        </div>
        <button
          type="button"
          onClick={() => void saveOverride()}
          className="mt-3 px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm"
        >
          Save LTL override
        </button>
      </Section>

      <Section title="LTL tolerance by salary range">
        <p className="text-xs text-gray-600 mb-3">
          Karigars in each daily-pay band get this tolerance % on formula LTL (e.g. ₹200–300 at 20% → target × 0.80 × rate/480).
        </p>
        <div className="space-y-2 text-xs max-w-xl">
          {bandRows.map((b, i) => (
            <div key={i} className="grid grid-cols-4 gap-2 items-end">
              <label>
                Min ₹
                <input
                  type="number"
                  className="w-full border rounded mt-1 px-2 py-1"
                  value={b.Min_Rs}
                  onChange={e => {
                    const next = [...bandRows]
                    next[i] = { ...next[i], Min_Rs: e.target.value }
                    setBandRows(next)
                  }}
                />
              </label>
              <label>
                Max ₹
                <input
                  type="number"
                  className="w-full border rounded mt-1 px-2 py-1"
                  value={b.Max_Rs}
                  onChange={e => {
                    const next = [...bandRows]
                    next[i] = { ...next[i], Max_Rs: e.target.value }
                    setBandRows(next)
                  }}
                />
              </label>
              <label>
                Tolerance %
                <input
                  type="number"
                  className="w-full border rounded mt-1 px-2 py-1"
                  value={b.Tolerance_Pct}
                  onChange={e => {
                    const next = [...bandRows]
                    next[i] = { ...next[i], Tolerance_Pct: e.target.value }
                    setBandRows(next)
                  }}
                />
              </label>
              <button
                type="button"
                className="text-red-600 text-xs py-1"
                onClick={() => setBandRows(bandRows.filter((_, j) => j !== i))}
              >
                Remove
              </button>
            </div>
          ))}
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              className="px-3 py-1.5 border rounded-lg"
              onClick={() => setBandRows([...bandRows, { Min_Rs: '', Max_Rs: '', Tolerance_Pct: '' }])}
            >
              + Add band
            </button>
            <button type="button" className="px-3 py-1.5 bg-violet-700 text-white rounded-lg" onClick={() => void saveBands()}>
              Save bands &amp; recalc production
            </button>
          </div>
        </div>
      </Section>

      <Section title="Saved manual LTL overrides">
        <DataTable
          rows={overrides}
          cols={[
            'Style',
            'Operation',
            'Karigar_ID',
            'Manual_LTL',
            'Formula_LTL',
            'Final_Applied_LTL',
            'LTL_Source',
            'Notes',
          ]}
        />
      </Section>
    </div>
  )
}

function KarigarExpensesTab({
  admin,
  onFlash,
}: {
  admin: AdminApi
  onFlash: (type: 'ok' | 'err', text: string) => void
}) {
  const qc = useQueryClient()
  const [from, setFrom] = useState(() => {
    const d = new Date()
    d.setDate(d.getDate() - 6)
    return d.toISOString().slice(0, 10)
  })
  const [to, setTo] = useState(todayStr())
  const [selectedChallans, setSelectedChallans] = useState<string[]>([])
  const [form, setForm] = useState({
    Date: todayStr(),
    Karigar_ID: '',
    Work_Type: 'Part Change',
    Challan_No: '',
    Style: '',
    Hours: '',
    Amount_Rs: '',
    Notes: '',
    Operation: '',
    Output: '',
    Expense_ID: '',
  })

  const { data: karSheet } = useQuery({
    queryKey: ['stitching-sheet', 'karigar_master'],
    queryFn: () => api.get('/stitching/sheets/karigar_master').then(r => r.data),
  })
  const { data: challanSheet } = useQuery({
    queryKey: ['stitching-sheet', 'challan_master'],
    queryFn: () => api.get('/stitching/sheets/challan_master').then(r => r.data),
  })

  const { data: expenseData, refetch } = useQuery({
    queryKey: ['stitching-expenses', from, to, form.Karigar_ID],
    queryFn: () =>
      api
        .get('/stitching/expenses', {
          params: {
            date_from: from,
            date_to: to,
            ...(form.Karigar_ID ? { karigar_id: form.Karigar_ID } : {}),
          },
        })
        .then(r => r.data),
  })

  const workTypes = (expenseData?.work_types ?? []) as string[]
  const rows = (expenseData?.rows ?? []) as Record<string, unknown>[]
  const karigarOptions = useMemo(() => {
    const fromApi = (expenseData?.karigars ?? []) as { Karigar_ID: string; Name: string }[]
    if (fromApi.length) {
      return fromApi.map(k => ({
        id: String(k.Karigar_ID),
        name: String(k.Name || k.Karigar_ID),
      }))
    }
    return ((karSheet?.rows ?? []) as { Karigar_ID: string; Name: string }[]).map(k => ({
      id: String(k.Karigar_ID),
      name: String(k.Name || k.Karigar_ID),
    }))
  }, [expenseData, karSheet])

  const karigarRows = useMemo(
    (): KarigarRow[] =>
      karigarOptions.map(k => ({ Karigar_ID: k.id, Name: k.name })),
    [karigarOptions],
  )

  const challans = useMemo(() => {
    const fromProd = (expenseData?.challans_for_karigar ?? []) as {
      Challan_No: string
      Style: string
      Last_Date?: string
    }[]
    if (fromProd.length) {
      return fromProd.map(c => ({
        no: String(c.Challan_No),
        style: String(c.Style || ''),
        lastDate: String(c.Last_Date || ''),
      }))
    }
    return ((challanSheet?.rows ?? []) as { Challan_No: string; Style: string }[]).map(c => ({
      no: String(c.Challan_No),
      style: String(c.Style || ''),
      lastDate: '',
    }))
  }, [challanSheet, expenseData])

  const saveExpense = async () => {
    const pw = admin.adminPassword()
    if (!pw) {
      onFlash('err', 'Unlock admin to add expenses')
      return
    }
    if (!form.Karigar_ID) {
      onFlash('err', 'Select a karigar')
      return
    }
    try {
      const { data } = await api.post('/stitching/expenses', {
        ...form,
        Challan_Nos: selectedChallans.length ? selectedChallans : form.Challan_No ? [form.Challan_No] : [],
        Hours: parseFloat(form.Hours) || 0,
        Amount_Rs: parseFloat(form.Amount_Rs) || 0,
        admin_password: pw,
      })
      onFlash('ok', data.message || 'Saved')
      setSelectedChallans([])
      setForm(f => ({
        ...f,
        Hours: '',
        Amount_Rs: '',
        Notes: '',
        Operation: '',
        Output: '',
        Expense_ID: '',
        Challan_No: '',
        Style: '',
      }))
      void refetch()
      qc.invalidateQueries({ queryKey: ['stitching-payroll'] })
    } catch (e: unknown) {
      onFlash('err', axios.isAxiosError(e) ? String(e.response?.data?.detail || 'Save failed') : 'Save failed')
    }
  }

  const editRow = (r: Record<string, unknown>) => {
    const cn = String(r.Challan_No || '')
    setSelectedChallans(cn.includes(',') ? cn.split(',').map(s => s.trim()) : cn ? [cn] : [])
    setForm({
      Date: String(r.Date || todayStr()),
      Karigar_ID: String(r.Karigar_ID || ''),
      Work_Type: String(r.Work_Type || 'Other'),
      Challan_No: cn,
      Style: String(r.Style || ''),
      Hours: String(r.Hours ?? ''),
      Amount_Rs: String(r.Amount_Rs ?? ''),
      Notes: String(r.Notes || ''),
      Operation: String(r.Operation || ''),
      Output: String(r.Output || ''),
      Expense_ID: String(r.Expense_ID || ''),
    })
  }

  const toggleChallanPick = (no: string) => {
    setSelectedChallans(prev => (prev.includes(no) ? prev.filter(x => x !== no) : [...prev, no]))
  }

  const removeExpense = async (expenseId: string) => {
    const pw = admin.adminPassword()
    if (!pw) return
    if (!window.confirm('Delete this expense?')) return
    try {
      await api.delete(`/stitching/expenses/${expenseId}`, { params: { admin_password: pw } })
      onFlash('ok', 'Expense deleted')
      void refetch()
      qc.invalidateQueries({ queryKey: ['stitching-payroll'] })
    } catch {
      onFlash('err', 'Delete failed')
    }
  }

  return (
    <div className="space-y-4">
      <Section title="Karigar expenses — other work on challans">
        <p className="text-xs text-gray-600 mb-3">
          Record part change, alter, trainee, or other tasks tied to a challan. Amounts roll into Payroll with attendance pay.
        </p>
        {!admin.unlocked && (
          <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mb-3">
            Unlock admin at the top of this page to add or delete karigar expenses.
          </p>
        )}
        {!karigarOptions.length && (
          <p className="text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2 mb-3">
            No karigars found — add them in Master Data → karigar master, or ensure employee master has Type = Karigar.
          </p>
        )}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
          <label>
            Date
            <input
              type="date"
              className="block w-full border rounded mt-1 px-2 py-1.5"
              value={form.Date}
              onChange={e => setForm(f => ({ ...f, Date: e.target.value }))}
            />
          </label>
          <div className="sm:col-span-2">
            <KarigarSearchSelect
              karigars={karigarRows}
              value={form.Karigar_ID}
              onChange={id => {
                setForm(f => ({ ...f, Karigar_ID: id }))
                setSelectedChallans([])
              }}
            />
          </div>
          <label>
            Work type
            <select
              className="block w-full border rounded mt-1 px-2 py-1.5"
              value={form.Work_Type}
              onChange={e => setForm(f => ({ ...f, Work_Type: e.target.value }))}
            >
              {(workTypes.length ? workTypes : ['Part Change', 'Alter', 'Trainee', 'Other Task']).map(w => (
                <option key={w} value={w}>
                  {w}
                </option>
              ))}
            </select>
          </label>
          <div className="sm:col-span-2 lg:col-span-3">
            <span className="font-semibold text-gray-700 text-xs">Challans (multi-select)</span>
            <p className="text-[10px] text-gray-400 mt-0.5">
              {form.Karigar_ID
                ? 'Shows challans this karigar worked on recently. Select one or more.'
                : 'Select a karigar first to see their challans.'}
            </p>
            <div className="mt-1 max-h-36 overflow-y-auto border rounded-lg p-2 space-y-1 bg-gray-50">
              {!challans.length && <p className="text-xs text-gray-400 px-1">No challans</p>}
              {challans.map(c => (
                <label key={c.no} className="flex items-center gap-2 text-xs cursor-pointer py-1">
                  <input
                    type="checkbox"
                    checked={selectedChallans.includes(c.no)}
                    onChange={() => toggleChallanPick(c.no)}
                  />
                  <span>
                    <span className="font-mono font-semibold">{c.no}</span>
                    <span className="text-gray-500"> · {c.style}</span>
                    {c.lastDate && <span className="text-gray-400"> · {c.lastDate}</span>}
                  </span>
                </label>
              ))}
            </div>
          </div>
          <label>
            Operation <span className="text-gray-400">(optional)</span>
            <input
              className="block w-full border rounded mt-1 px-2 py-1.5"
              value={form.Operation}
              onChange={e => setForm(f => ({ ...f, Operation: e.target.value }))}
              placeholder="e.g. Alter sleeve"
            />
          </label>
          <label>
            Style
            <input
              className="block w-full border rounded mt-1 px-2 py-1.5"
              value={form.Style}
              onChange={e => setForm(f => ({ ...f, Style: e.target.value }))}
            />
          </label>
          <label>
            Hours
            <input
              type="number"
              step="0.5"
              className="block w-full border rounded mt-1 px-2 py-1.5"
              value={form.Hours}
              onChange={e => setForm(f => ({ ...f, Hours: e.target.value }))}
            />
          </label>
          <label>
            Amount ₹
            <input
              type="number"
              className="block w-full border rounded mt-1 px-2 py-1.5"
              value={form.Amount_Rs}
              onChange={e => setForm(f => ({ ...f, Amount_Rs: e.target.value }))}
            />
          </label>
          <label className="sm:col-span-2">
            Notes
            <input
              className="block w-full border rounded mt-1 px-2 py-1.5"
              value={form.Notes}
              onChange={e => setForm(f => ({ ...f, Notes: e.target.value }))}
            />
          </label>
          <label className="sm:col-span-2 lg:col-span-3">
            Output / remarks about karigar
            <textarea
              className="block w-full border rounded mt-1 px-2 py-1.5 min-h-[72px]"
              value={form.Output}
              onChange={e => setForm(f => ({ ...f, Output: e.target.value }))}
              placeholder="What was done, quality notes, etc."
            />
          </label>
        </div>
        <button
          type="button"
          onClick={() => void saveExpense()}
          className="mt-3 px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm"
        >
          {form.Expense_ID ? 'Update expense' : 'Add expense'}
        </button>
        {form.Expense_ID && (
          <button
            type="button"
            className="mt-3 ml-2 px-4 py-2 border rounded-lg text-sm"
            onClick={() => {
              setSelectedChallans([])
              setForm({
                Date: todayStr(),
                Karigar_ID: '',
                Work_Type: 'Part Change',
                Challan_No: '',
                Style: '',
                Hours: '',
                Amount_Rs: '',
                Notes: '',
                Operation: '',
                Output: '',
                Expense_ID: '',
              })
            }}
          >
            Cancel edit
          </button>
        )}
      </Section>

      <Section title="Expense list">
        <div className="flex gap-2 items-end mb-3 text-xs">
          <label>
            From
            <input type="date" className="block border rounded mt-1 px-2 py-1" value={from} onChange={e => setFrom(e.target.value)} />
          </label>
          <label>
            To
            <input type="date" className="block border rounded mt-1 px-2 py-1" value={to} onChange={e => setTo(e.target.value)} />
          </label>
          <button type="button" onClick={() => void refetch()} className="px-3 py-1.5 bg-[#002B5B] text-white rounded-lg">
            Refresh
          </button>
        </div>
        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-gray-50 border-b">
              <tr>
                {['Date', 'Karigar_Name', 'Work_Type', 'Challan_No', 'Style', 'Hours', 'Amount_Rs', 'Notes', ''].map(
                  h => (
                    <th key={h || 'act'} className="text-left px-2 py-2 whitespace-nowrap">
                      {h || 'Actions'}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={String(r.Expense_ID || i)} className="border-b">
                  <td className="px-2 py-1">{String(r.Date)}</td>
                  <td className="px-2 py-1">{String(r.Karigar_Name || r.Karigar_ID)}</td>
                  <td className="px-2 py-1">{String(r.Work_Type)}</td>
                  <td className="px-2 py-1">{String(r.Challan_No)}</td>
                  <td className="px-2 py-1">{String(r.Style)}</td>
                  <td className="px-2 py-1">{String(r.Hours)}</td>
                  <td className="px-2 py-1">₹{String(r.Amount_Rs)}</td>
                  <td className="px-2 py-1 max-w-[120px] truncate">{String(r.Notes)}</td>
                  <td className="px-2 py-1 whitespace-nowrap">
                    <button type="button" className="text-[#002B5B] mr-2" onClick={() => editRow(r)}>
                      Edit
                    </button>
                    <button
                      type="button"
                      className="text-red-700"
                      onClick={() => void removeExpense(String(r.Expense_ID))}
                    >
                      Del
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!rows.length && <p className="text-sm text-gray-400 py-4 text-center">No expenses in range</p>}
        </div>
      </Section>
    </div>
  )
}

type PunchPairDraft = { in_time: string; out_time: string }

type MissPunchDraft = {
  Date: string
  E_Code: string
  Name: string
  In_Punch: string
  Out_Punch: string
  punch_pairs: PunchPairDraft[]
  Waive_Lunch_Break: boolean
  Waive_Tea_Break: boolean
  Lunch_Break_Minutes: string
  Tea_Break_Minutes: string
}

function parsePunchPairsFromRow(r: Record<string, unknown>): PunchPairDraft[] {
  const raw = r.Punch_Pairs
  if (typeof raw === 'string' && raw.trim().startsWith('[')) {
    try {
      const arr = JSON.parse(raw) as [string, string][]
      if (Array.isArray(arr) && arr.length) {
        return arr.map(([inn, out]) => ({ in_time: inn || '', out_time: out || '' }))
      }
    } catch {
      /* fall through */
    }
  }
  const inn = String(r.In_Punch ?? '09:00')
  const out = String(r.Out_Punch ?? '18:00')
  return [{ in_time: inn, out_time: out }]
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
  const [uploadMsg, setUploadMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)
  const [uploadBusy, setUploadBusy] = useState(false)
  /** Blank = use date from sheet header / filename (not today). */
  const [uploadReportDate, setUploadReportDate] = useState('')
  const [missDrafts, setMissDrafts] = useState<Record<string, MissPunchDraft>>({})
  const fileRef = useRef<HTMLInputElement>(null)
  const allRows = (data?.rows ?? []) as Record<string, unknown>[]
  const needsMissPunch = type === 'karigar'
    ? allRows.filter(r => r.Needs_Miss_Punch === true || r.Needs_Miss_Punch === 'true')
    : []
  const saveMut = useMutation({
    mutationFn: () =>
      api.post(type === 'karigar' ? '/stitching/attendance/karigar' : '/stitching/attendance/operating', form),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['stitching-sheet', sheet] })
      refetch()
    },
  })
  const patchMissMut = useMutation({
    mutationFn: (body: MissPunchDraft) => {
      const pairs = body.punch_pairs.filter(p => p.in_time.trim())
      const first = pairs[0]
      const last = pairs[pairs.length - 1]
      return api.patch('/stitching/attendance/karigar', {
        Date: body.Date,
        E_Code: body.E_Code,
        In_Punch: first?.in_time ?? body.In_Punch,
        Out_Punch: last?.out_time ?? body.Out_Punch,
        punch_pairs: pairs,
        Waive_Lunch_Break: body.Waive_Lunch_Break,
        Waive_Tea_Break: body.Waive_Tea_Break,
        Lunch_Break_Minutes: body.Lunch_Break_Minutes.trim() === '' ? null : Number(body.Lunch_Break_Minutes),
        Tea_Break_Minutes: body.Tea_Break_Minutes.trim() === '' ? null : Number(body.Tea_Break_Minutes),
      })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['stitching-sheet', sheet] })
      qc.invalidateQueries({ queryKey: ['stitching-payroll'] })
      refetch()
    },
  })
  const uploadAttendance = async (file: File) => {
    if (type !== 'karigar') return
    setUploadBusy(true)
    setUploadMsg(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      if (uploadReportDate.trim()) fd.append('report_date', uploadReportDate.trim())
      const { data: res } = await api.post<{ ok?: boolean; message?: string; date?: string; warnings?: string[] }>(
        '/stitching/attendance/karigar/upload',
        fd,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 120_000 },
      )
      const warn = res.warnings?.length ? ` ${res.warnings.join(' ')}` : ''
      setUploadMsg({ type: 'ok', text: `${res.message || 'Imported.'}${warn}` })
      qc.invalidateQueries({ queryKey: ['stitching-sheet', sheet] })
      qc.invalidateQueries({ queryKey: ['stitching-payroll'] })
      refetch()
    } catch (e: unknown) {
      const msg =
        axios.isAxiosError(e) && e.response?.data?.detail
          ? String(e.response.data.detail)
          : 'Upload failed'
      setUploadMsg({ type: 'err', text: msg })
    } finally {
      setUploadBusy(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }
  const missPunchKey = (r: Record<string, unknown>) => `${r.Date}::${r.E_Code}`
  const openMissPunch = (r: Record<string, unknown>) => {
    const key = missPunchKey(r)
    const pairs = parsePunchPairsFromRow(r)
    setMissDrafts(d => ({
      ...d,
      [key]: d[key] ?? {
        Date: String(r.Date ?? ''),
        E_Code: String(r.E_Code ?? ''),
        Name: String(r.Name ?? ''),
        In_Punch: pairs[0]?.in_time ?? '09:00',
        Out_Punch: pairs[pairs.length - 1]?.out_time ?? '18:00',
        punch_pairs: pairs,
        Waive_Lunch_Break: false,
        Waive_Tea_Break: false,
        Lunch_Break_Minutes: '',
        Tea_Break_Minutes: '',
      },
    }))
  }
  const karigarCols = [
    'Date',
    'E_Code',
    'Name',
    'Status',
    'Needs_Miss_Punch',
    'Punch_Count',
    'In_Punch',
    'Out_Punch',
    'Payable_Hrs',
    'Late_Deduction_Hrs',
    'Late_Deduction_Rs',
    'OT_Hours',
    'Hourly_Rate_Rs',
    'Normal_Pay',
    'OT_Pay',
    'Total_Pay',
  ]
  return (
    <div className="space-y-4">
      {type === 'karigar' && (
        <div className="rounded-xl border border-blue-100 bg-blue-50/60 px-4 py-3 text-xs text-blue-950 space-y-1">
          <p className="font-semibold text-[#002B5B]">Workflow</p>
          <ol className="list-decimal list-inside space-y-0.5">
            <li>Upload the biometric IN/OUT sheet (matched by <strong>E. Code</strong>).</li>
            <li>Fix any <strong>miss punch</strong> rows (single IN only) and set break time if needed.</li>
            <li>
              Payroll uses 8h regular pay minus <strong>late minutes after 17 min grace</strong> (from 09:00 in), break deductions, plus OT after 18:00.
            </li>
            <li>Use <strong>Performance</strong> tab to find karigars in payroll but not in production costing.</li>
          </ol>
        </div>
      )}
      {type === 'karigar' && (
        <Section title="Upload biometric attendance (IN/OUT punch report)">
          <p className="text-xs text-gray-500 mb-2">
            Upload the daily <strong>Daily Attendance IN/OUT Punch Report</strong> (.xls / .xlsx). The attendance date is taken from the{' '}
            <strong>Date</strong> row on the sheet (or the filename like <code>07-06-2026.xls</code>) — not from when you upload.
            You can upload Saturday&apos;s sheet on Monday; leave the date picker blank to use the sheet date, or pick the day manually if the header is unclear.
          </p>
          <label className="text-xs text-gray-600 block mb-2 max-w-xs">
            Attendance date (optional override)
            <input
              type="date"
              className="w-full border rounded mt-1 px-2 py-1.5"
              value={uploadReportDate}
              onChange={e => setUploadReportDate(e.target.value)}
            />
          </label>
          <input
            ref={fileRef}
            type="file"
            accept=".xls,.xlsx,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            className="hidden"
            onChange={e => {
              const f = e.target.files?.[0]
              if (f) void uploadAttendance(f)
            }}
          />
          <button
            type="button"
            disabled={uploadBusy}
            onClick={() => fileRef.current?.click()}
            className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm disabled:opacity-50"
          >
            {uploadBusy ? 'Importing…' : '📤 Upload attendance sheet'}
          </button>
          {uploadMsg && (
            <p
              className={`mt-2 text-xs rounded-lg px-3 py-2 border ${
                uploadMsg.type === 'ok'
                  ? 'bg-emerald-50 text-emerald-900 border-emerald-200'
                  : 'bg-red-50 text-red-800 border-red-200'
              }`}
            >
              {uploadMsg.text}
            </p>
          )}
        </Section>
      )}
      {type === 'karigar' && needsMissPunch.length > 0 && (
        <Section title={`Miss punch / break fixes (${needsMissPunch.length})`}>
          <p className="text-xs text-gray-500 mb-3">
            These rows need a valid <strong>Out</strong> for every <strong>In</strong> (e.g. employee 921 with multiple segments).
            Standard breaks: <strong>30 min lunch</strong> + <strong>2×15 min tea</strong> (deducted if not present during break windows).
            Check &quot;Took lunch/tea&quot; if breaks were taken; or enter custom minutes.
          </p>
          <div className="space-y-3">
            {needsMissPunch.slice(0, 40).map(r => {
              const key = missPunchKey(r)
              const draft = missDrafts[key]
              return (
                <div key={key} className="border rounded-lg p-3 bg-amber-50/50 text-xs space-y-2">
                  <div className="flex flex-wrap justify-between gap-2">
                    <span className="font-semibold text-[#002B5B]">
                      {String(r.E_Code)} — {String(r.Name)} · {String(r.Date)}
                    </span>
                    {!draft && (
                      <button type="button" onClick={() => openMissPunch(r)} className="text-blue-700 underline font-medium">
                        Fix punch
                      </button>
                    )}
                  </div>
                  {draft && (
                    <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-2">
                      <div className="sm:col-span-2 lg:col-span-4 space-y-2">
                        <p className="font-medium text-gray-700">IN / OUT pairs (from biometric)</p>
                        {draft.punch_pairs.map((pair, idx) => (
                          <div key={idx} className="flex flex-wrap gap-2 items-end">
                            <label className="flex-1 min-w-[100px]">
                              In {idx + 1}
                              <input
                                className="w-full border rounded mt-1 px-2 py-1"
                                value={pair.in_time}
                                onChange={e => {
                                  const next = [...draft.punch_pairs]
                                  next[idx] = { ...pair, in_time: e.target.value }
                                  setMissDrafts(d => ({ ...d, [key]: { ...draft, punch_pairs: next } }))
                                }}
                              />
                            </label>
                            <label className="flex-1 min-w-[100px]">
                              Out {idx + 1}
                              <input
                                className="w-full border rounded mt-1 px-2 py-1"
                                value={pair.out_time}
                                onChange={e => {
                                  const next = [...draft.punch_pairs]
                                  next[idx] = { ...pair, out_time: e.target.value }
                                  setMissDrafts(d => ({ ...d, [key]: { ...draft, punch_pairs: next } }))
                                }}
                              />
                            </label>
                            {draft.punch_pairs.length > 1 && (
                              <button
                                type="button"
                                className="text-red-700 text-xs pb-1"
                                onClick={() => {
                                  const next = draft.punch_pairs.filter((_, i) => i !== idx)
                                  setMissDrafts(d => ({ ...d, [key]: { ...draft, punch_pairs: next.length ? next : [{ in_time: '', out_time: '' }] } }))
                                }}
                              >
                                Remove
                              </button>
                            )}
                          </div>
                        ))}
                        <button
                          type="button"
                          className="text-xs text-[#002B5B] underline"
                          onClick={() =>
                            setMissDrafts(d => ({
                              ...d,
                              [key]: { ...draft, punch_pairs: [...draft.punch_pairs, { in_time: '', out_time: '' }] },
                            }))
                          }
                        >
                          + Add punch pair
                        </button>
                      </div>
                      <label className="flex items-end gap-2 pb-1">
                        <input type="checkbox" checked={draft.Waive_Lunch_Break}
                          onChange={e => setMissDrafts(d => ({ ...d, [key]: { ...draft, Waive_Lunch_Break: e.target.checked } }))} />
                        Took lunch (no 30m deduct)
                      </label>
                      <label className="flex items-end gap-2 pb-1">
                        <input type="checkbox" checked={draft.Waive_Tea_Break}
                          onChange={e => setMissDrafts(d => ({ ...d, [key]: { ...draft, Waive_Tea_Break: e.target.checked } }))} />
                        Took both teas (no 2×15m deduct)
                      </label>
                      <label>
                        Custom lunch break (min)
                        <input className="w-full border rounded mt-1 px-2 py-1" placeholder="30 default" value={draft.Lunch_Break_Minutes}
                          onChange={e => setMissDrafts(d => ({ ...d, [key]: { ...draft, Lunch_Break_Minutes: e.target.value } }))} />
                      </label>
                      <label>
                        Custom tea total (min)
                        <input className="w-full border rounded mt-1 px-2 py-1" placeholder="30 = 2×15 default" value={draft.Tea_Break_Minutes}
                          onChange={e => setMissDrafts(d => ({ ...d, [key]: { ...draft, Tea_Break_Minutes: e.target.value } }))} />
                      </label>
                      <div className="sm:col-span-2 flex gap-2 items-end">
                        <button type="button" disabled={patchMissMut.isPending}
                          onClick={() => patchMissMut.mutate(draft)}
                          className="px-3 py-2 bg-[#002B5B] text-white rounded-lg text-xs font-medium disabled:opacity-50">
                          {patchMissMut.isPending ? 'Saving…' : 'Save & recalc payroll'}
                        </button>
                        <button type="button" onClick={() => setMissDrafts(d => { const n = { ...d }; delete n[key]; return n })}
                          className="px-3 py-2 border rounded-lg text-xs">Cancel</button>
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </Section>
      )}
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
        <DataTable
          rows={(() => {
            const all = (data?.rows ?? []) as Record<string, unknown>[]
            const toNum = (v: unknown) => {
              const n = Number(String(v ?? '').trim())
              return Number.isFinite(n) ? n : Number.POSITIVE_INFINITY
            }
            // Show the most recent attendance day first; within the day sort by E_Code ascending.
            // Cap to 250 rows to avoid huge renders.
            return [...all]
              .sort((a, b) => {
                const da = String(a.Date ?? '')
                const db = String(b.Date ?? '')
                if (da !== db) return db.localeCompare(da) // newest date first
                return toNum(a.E_Code) - toNum(b.E_Code)
              })
              .slice(0, 250)
          })()}
          cols={type === 'karigar' ? karigarCols : ['Date', 'E_Code', 'Name', 'Total_Hours', 'Total_Pay']}
        />
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

type TargetControlRow = {
  Style: string
  Operation: string
  Operation_Type?: string
  Karigar_ID: string
  Karigar_Name?: string
  Daily_Rate_Rs: number
  Base_Target: number
  'Tolerance_%'?: number
  Formula_LTL: number
  Manual_Override: number | string
  Final_Applied_LTL: number
  Final_Applied_LTL_Period?: number
  Target_For_Period?: number
  Period?: string
  Target_Type: string
  LTL_Source: string
}

function TargetControlTab({
  admin,
  onFlash,
}: {
  admin: AdminApi
  onFlash: (type: 'ok' | 'err', text: string) => void
}) {
  const qc = useQueryClient()
  const [planDate, setPlanDate] = useState(todayStr())
  const [styleFilter, setStyleFilter] = useState('')
  const [karFilter, setKarFilter] = useState('')
  const [opFilter, setOpFilter] = useState('')
  const [period, setPeriod] = useState<'daily' | 'weekly' | 'monthly'>('daily')
  const [overrideEdits, setOverrideEdits] = useState<Record<string, string>>({})

  const { data: styleSheet } = useQuery({
    queryKey: ['stitching-sheet', 'style_master'],
    queryFn: () => api.get('/stitching/sheets/style_master').then(r => r.data),
  })
  const { data: karSheet } = useQuery({
    queryKey: ['stitching-sheet', 'karigar_master'],
    queryFn: () => api.get('/stitching/sheets/karigar_master').then(r => r.data),
  })

  const styles = useMemo(() => {
    const s = new Set<string>()
    for (const r of (styleSheet?.rows ?? []) as { Style: string }[]) {
      if (r.Style) s.add(String(r.Style))
    }
    return [...s].sort()
  }, [styleSheet])

  const karigars = useMemo(() => {
    return ((karSheet?.rows ?? []) as { Karigar_ID: string; Name: string }[])
      .map(k => ({ id: String(k.Karigar_ID), name: String(k.Name || k.Karigar_ID) }))
      .sort((a, b) => a.name.localeCompare(b.name))
  }, [karSheet])

  const ops = useMemo(() => {
    const s = new Set<string>()
    for (const r of (styleSheet?.rows ?? []) as { Style: string; Operation: string }[]) {
      if (!styleFilter || r.Style === styleFilter) {
        if (r.Operation) s.add(String(r.Operation))
      }
    }
    return [...s].sort()
  }, [styleSheet, styleFilter])

  const { data: preview, isFetching, refetch } = useQuery({
    queryKey: ['stitching-target-control', planDate, styleFilter, karFilter, opFilter, period],
    queryFn: () =>
      api
        .get('/stitching/target-control/preview', {
          params: {
            date: planDate,
            style: styleFilter || undefined,
            karigar_id: karFilter || undefined,
            operation: opFilter || undefined,
            period,
          },
        })
        .then(r => r.data),
  })

  const rows = (preview?.rows ?? []) as TargetControlRow[]
  const rowKey = (r: TargetControlRow) => `${r.Style}::${r.Operation}::${r.Karigar_ID}`

  const saveOverride = async (r: TargetControlRow) => {
    const pw = admin.adminPassword()
    if (!pw) {
      onFlash('err', 'Unlock admin to set manual overrides')
      return
    }
    const raw = overrideEdits[rowKey(r)]
    const manual = raw === '' || raw === undefined ? null : Math.max(0, parseInt(raw, 10) || 0)
    try {
      const { data } = await api.put('/stitching/target-control/override', {
        Style: r.Style,
        Operation: r.Operation,
        Karigar_ID: r.Karigar_ID,
        Manual_LTL: manual,
        admin_password: pw,
      })
      onFlash('ok', data.message || 'Override saved')
      setOverrideEdits(prev => {
        const next = { ...prev }
        delete next[rowKey(r)]
        return next
      })
      void refetch()
      qc.invalidateQueries({ queryKey: ['stitching-pe-reports'] })
    } catch (e: unknown) {
      onFlash('err', axios.isAxiosError(e) ? String(e.response?.data?.detail || 'Save failed') : 'Save failed')
    }
  }

  const cols = [
    'Style',
    'Operation',
    'Operation_Type',
    'Karigar_ID',
    'Karigar_Name',
    'Daily_Rate_Rs',
    'Base_Target',
    'Tolerance_%',
    'Formula_LTL',
    'Manual_Override',
    'Final_Applied_LTL',
    'Final_Applied_LTL_Period',
    'Target_For_Period',
    'Target_Type',
  ]

  return (
    <div className="space-y-4">
      <Section title="Target Control — Multi-Style LTL">
        <p className="text-xs text-gray-600 mb-3">
          Benchmark daily rate <strong>₹{preview?.benchmark_daily_rate_rs ?? 480}</strong>. Tolerance % by karigar daily
          rate (applied as <strong>Base Target × (1 − tolerance %)</strong> × rate/480):{' '}
          {(
            (preview?.tolerance_bands as { from_rs: number; to_rs: number; tolerance_pct: number }[] | undefined) ?? [
              { from_rs: 200, to_rs: 300, tolerance_pct: 35 },
              { from_rs: 300, to_rs: 400, tolerance_pct: 12 },
            ]
          ).map(b => (
            <span key={`${b.from_rs}-${b.to_rs}`} className="inline-block mr-2">
              ₹{b.from_rs}–{b.to_rs}: <strong>{b.tolerance_pct}%</strong>
            </span>
          ))}
          . Production uses <strong>Final Applied LTL</strong> (manual override if set, else formula).
        </p>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5 mb-4">
          <label className="text-xs block">
            <span className="font-semibold text-gray-700">Planning date</span>
            <input
              type="date"
              className="mt-1 w-full border rounded-lg px-3 py-2 text-sm"
              value={planDate}
              onChange={e => setPlanDate(e.target.value)}
            />
          </label>
          <label className="text-xs block">
            <span className="font-semibold text-gray-700">Style</span>
            <select
              className="mt-1 w-full border rounded-lg px-3 py-2 text-sm"
              value={styleFilter}
              onChange={e => setStyleFilter(e.target.value)}
            >
              <option value="">All styles</option>
              {styles.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </label>
          <label className="text-xs block">
            <span className="font-semibold text-gray-700">Karigar</span>
            <select
              className="mt-1 w-full border rounded-lg px-3 py-2 text-sm"
              value={karFilter}
              onChange={e => setKarFilter(e.target.value)}
            >
              <option value="">All karigars</option>
              {karigars.map(k => (
                <option key={k.id} value={k.id}>{k.id} — {k.name}</option>
              ))}
            </select>
          </label>
          <label className="text-xs block">
            <span className="font-semibold text-gray-700">Rate mode</span>
            <select
              className="mt-1 w-full border rounded-lg px-3 py-2 text-sm"
              value={period}
              onChange={e => setPeriod(e.target.value as 'daily' | 'weekly' | 'monthly')}
            >
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
            </select>
          </label>
          <label className="text-xs block">
            <span className="font-semibold text-gray-700">Operation</span>
            <select
              className="mt-1 w-full border rounded-lg px-3 py-2 text-sm"
              value={opFilter}
              onChange={e => setOpFilter(e.target.value)}
            >
              <option value="">All operations</option>
              {ops.map(o => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          </label>
        </div>
        {isFetching && <p className="text-xs text-gray-500 mb-2">Refreshing ledger…</p>}
        <div className="overflow-x-auto border rounded-lg">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-[#1a3a5c] text-white">
                {cols.map(c => (
                  <th key={c} className="px-2 py-2 text-left whitespace-nowrap">{c.replace(/_/g, ' ')}</th>
                ))}
                <th className="px-2 py-2">Override action</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr>
                  <td colSpan={cols.length + 1} className="px-3 py-6 text-center text-gray-400">
                    No rows — add style operations and karigars in Master Data first.
                  </td>
                </tr>
              )}
              {rows.map(r => {
                const key = rowKey(r)
                const editVal = overrideEdits[key]
                const displayOverride =
                  editVal !== undefined ? editVal : r.Manual_Override === '' ? '' : String(r.Manual_Override)
                const isOverride = r.Target_Type === 'Manual Override'
                return (
                  <tr key={key} className={`border-b ${isOverride ? 'bg-amber-50' : ''}`}>
                    <td className="px-2 py-1 font-mono">{r.Style}</td>
                    <td className="px-2 py-1">{r.Operation}</td>
                    <td className="px-2 py-1">{r.Karigar_ID}</td>
                    <td className="px-2 py-1">{r.Karigar_Name}</td>
                    <td className="px-2 py-1 text-right">₹{r.Daily_Rate_Rs}</td>
                    <td className="px-2 py-1 text-center">{r.Base_Target}</td>
                    <td className="px-2 py-1 text-center font-semibold text-[#2c5aa0]">{r.Formula_LTL}</td>
                    <td className="px-2 py-1">
                      <input
                        type="number"
                        min={0}
                        placeholder="Blank = formula"
                        className="w-20 border rounded px-1 py-1"
                        value={displayOverride}
                        onChange={e => setOverrideEdits(prev => ({ ...prev, [key]: e.target.value }))}
                      />
                    </td>
                    <td className="px-2 py-1 text-center font-bold">{r.Final_Applied_LTL}</td>
                    <td className="px-2 py-1 text-[10px]">{r.Target_Type}</td>
                    <td className="px-2 py-1">
                      <button
                        type="button"
                        className="text-[10px] px-2 py-1 rounded bg-[#002B5B] text-white"
                        onClick={() => void saveOverride(r)}
                      >
                        Apply
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <ReportTableSection
          title="Export audit ledger"
          rows={rows as unknown as Record<string, unknown>[]}
          cols={cols}
          downloadName={`target-control-${planDate}.csv`}
        />
      </Section>
    </div>
  )
}

type StyleMasterReportView = 'summary' | 'master' | 'production' | 'costing' | 'full'

function StyleMasterReportPanel({
  styleOptions,
  focusStyle,
  onFocusStyle,
}: {
  styleOptions: SearchSelectOption[]
  focusStyle: string
  onFocusStyle: (s: string) => void
}) {
  const [reportStyle, setReportStyle] = useState('')
  const [dateFrom, setDateFrom] = useState(() => {
    const d = new Date()
    d.setDate(d.getDate() - 29)
    return d.toISOString().slice(0, 10)
  })
  const [dateTo, setDateTo] = useState(todayStr())
  const [view, setView] = useState<StyleMasterReportView>('summary')
  const [showKarigar, setShowKarigar] = useState(true)
  const [showByDate, setShowByDate] = useState(true)
  const [showDetail, setShowDetail] = useState(false)
  const [showCostingTable, setShowCostingTable] = useState(true)

  useEffect(() => {
    if (focusStyle) setReportStyle(focusStyle)
  }, [focusStyle])

  const activeStyle = reportStyle.trim()
  const { data, isFetching, isError } = useQuery({
    queryKey: ['stitching-style-master-report', activeStyle, dateFrom, dateTo, view, showDetail],
    queryFn: () =>
      api
        .get('/stitching/master/style-report', {
          params: {
            style: activeStyle,
            date_from: dateFrom,
            date_to: dateTo,
            view,
            include_production_detail: showDetail,
          },
        })
        .then(r => r.data),
    enabled: !!activeStyle,
  })

  const totals = data?.totals as Record<string, number> | undefined
  const master = data?.master as { operations?: Record<string, unknown>[]; totals?: Record<string, unknown> } | undefined
  const production = data?.production as {
    summary?: Record<string, number>
    by_operation?: Record<string, unknown>[]
    by_karigar?: Record<string, unknown>[]
    by_date?: Record<string, unknown>[]
    detail?: Record<string, unknown>[]
  } | undefined
  const costing = data?.costing as { summary?: Record<string, number>; challans?: Record<string, unknown>[] } | undefined
  const columnSets = (data?.column_sets ?? {}) as Record<string, string[]>

  const viewTabs: { id: StyleMasterReportView; label: string }[] = [
    { id: 'summary', label: 'Totals' },
    { id: 'master', label: 'Operations' },
    { id: 'production', label: 'Production' },
    { id: 'costing', label: 'Costing' },
    { id: 'full', label: 'Full report' },
  ]

  return (
    <div className="border border-sky-200 bg-sky-50/60 rounded-xl p-4 space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="text-sm font-semibold text-[#002B5B]">Style report</p>
          <p className="text-xs text-gray-600 mt-0.5">
            Pick a style for totals across master operations, floor production, and challan costing. Customize sections below.
          </p>
        </div>
      </div>
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-2 text-xs">
        <div className="sm:col-span-2">
          <SearchSelect
            label="Style"
            placeholder="Search style code…"
            value={reportStyle}
            onChange={v => {
              setReportStyle(v)
              onFocusStyle(v)
            }}
            options={styleOptions}
            emptyMessage="No styles in master"
            hint="Click a style in the table below to load its report"
          />
        </div>
        <label>
          From
          <input type="date" className="w-full border rounded mt-1 px-2 py-1.5" value={dateFrom} onChange={e => setDateFrom(e.target.value)} />
        </label>
        <label>
          To
          <input type="date" className="w-full border rounded mt-1 px-2 py-1.5" value={dateTo} onChange={e => setDateTo(e.target.value)} />
        </label>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {viewTabs.map(t => (
          <button
            key={t.id}
            type="button"
            onClick={() => setView(t.id)}
            className={`text-xs px-3 py-1.5 rounded-lg ${view === t.id ? 'bg-[#002B5B] text-white' : 'bg-white border'}`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap gap-3 text-xs text-gray-700">
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showKarigar} onChange={e => setShowKarigar(e.target.checked)} />
          By karigar
        </label>
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showByDate} onChange={e => setShowByDate(e.target.checked)} />
          By date
        </label>
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showDetail} onChange={e => setShowDetail(e.target.checked)} />
          Production detail rows
        </label>
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={showCostingTable} onChange={e => setShowCostingTable(e.target.checked)} />
          Challan table
        </label>
      </div>
      {!activeStyle && <p className="text-xs text-gray-500">Select a style to load the report.</p>}
      {activeStyle && isFetching && <p className="text-xs text-gray-500">Loading report…</p>}
      {activeStyle && isError && <p className="text-xs text-red-600">Could not load report.</p>}
      {activeStyle && totals && (view === 'summary' || view === 'full') && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2">
          {[
            ['Operations', totals.master_operations],
            ['Labour ₹/pc', `₹${Number(totals.labour_rate_per_piece_rs ?? 0).toLocaleString()}`],
            ['Prod pieces', totals.production_pieces],
            ['Prod value', `₹${Number(totals.production_piece_value_rs ?? 0).toLocaleString()}`],
            ['Prod P&L', `₹${Number(totals.production_pl_rs ?? 0).toLocaleString()}`],
            ['Party value', `₹${Number(totals.costing_party_value_rs ?? 0).toLocaleString()}`],
            ['Costing P&L', `₹${Number(totals.costing_net_pl_rs ?? 0).toLocaleString()}`],
          ].map(([l, v]) => (
            <div key={String(l)} className="bg-white border rounded-lg p-2.5">
              <p className="text-[10px] text-gray-500">{l}</p>
              <p className="font-bold text-[#2c5aa0] text-sm">{v}</p>
            </div>
          ))}
        </div>
      )}
      {activeStyle && master && (view === 'master' || view === 'full') && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-gray-700">
            Master operations ({String(master.totals?.operation_count ?? 0)}) — labour benchmark ₹
            {Number(master.totals?.benchmark_labour_per_piece_rs ?? 0).toLocaleString()}/pc (₹480/day SOP)
          </p>
          <DataTable rows={master.operations ?? []} cols={columnSets.master_operations ?? ['Operation', 'Target', 'Rate_Rs']} />
        </div>
      )}
      {activeStyle && production && (view === 'production' || view === 'full') && (
        <div className="space-y-3">
          {production.summary && (
            <p className="text-xs text-gray-600">
              Production {dateFrom} → {dateTo}: {production.summary.pieces} pcs, {production.summary.karigars} karigar(s), avg eff{' '}
              {production.summary.avg_efficiency_pct}%
            </p>
          )}
          <DataTable
            rows={production.by_operation ?? []}
            cols={columnSets.production_by_operation ?? ['Operation', 'Pieces', 'Piece_Value_Rs', 'PL_Rs']}
          />
          {showKarigar && (
            <DataTable
              rows={production.by_karigar ?? []}
              cols={columnSets.production_by_karigar ?? ['Karigar_ID', 'Karigar_Name', 'Pieces', 'PL_Rs']}
            />
          )}
          {showByDate && (
            <DataTable rows={production.by_date ?? []} cols={columnSets.production_by_date ?? ['Date', 'Pieces', 'PL_Rs']} />
          )}
          {showDetail && (
            <DataTable
              rows={production.detail ?? []}
              cols={columnSets.production_detail ?? ['Date', 'Karigar_ID', 'Operation', 'Total_Pieces', 'PL_Rs']}
            />
          )}
        </div>
      )}
      {activeStyle && costing && (view === 'costing' || view === 'full') && (
        <div className="space-y-2">
          {costing.summary && (
            <p className="text-xs text-gray-600">
              {costing.summary.challans} challan(s), party ₹{Number(costing.summary.party_value ?? 0).toLocaleString()}, net P&L ₹
              {Number(costing.summary.net_pl ?? 0).toLocaleString()}
            </p>
          )}
          {showCostingTable && (
            <DataTable rows={costing.challans ?? []} cols={columnSets.costing_challans ?? ['Challan_No', 'Party_Value_Rs', 'PL_Rs']} />
          )}
        </div>
      )}
    </div>
  )
}

function masterRowPayload(sheet: MasterSheetKey, row: Record<string, unknown>): Record<string, string> {
  if (sheet === 'style_master') {
    return { Style: String(row.Style ?? ''), Operation: String(row.Operation ?? '') }
  }
  if (sheet === 'karigar_master') return { Karigar_ID: String(row.Karigar_ID ?? '') }
  return { E_Code: String(row.E_Code ?? '') }
}

function MasterTab({ admin, onFlash }: { admin: AdminApi; onFlash: (type: 'ok' | 'err', text: string) => void }) {
  const qc = useQueryClient()
  const [active, setActive] = useState<MasterSheetKey>('style_master')
  const [reportFocusStyle, setReportFocusStyle] = useState('')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [editKarigar, setEditKarigar] = useState<Record<string, unknown> | null>(null)
  const [editEmployee, setEditEmployee] = useState<Record<string, unknown> | null>(null)
  const [editStyleRow, setEditStyleRow] = useState<Record<string, unknown> | null>(null)
  const [karEdit, setKarEdit] = useState({
    Name: '',
    Skill: '',
    Daily_Rate_Rs: 420,
    Effective_From: todayStr(),
    New_Karigar_ID: '',
  })
  const [empEdit, setEmpEdit] = useState({
    Name: '',
    Type: 'Karigar',
    Daily_Rate_Rs: 420,
    New_E_Code: '',
  })
  const [styleEdit, setStyleEdit] = useState({
    Target: 80,
    Rate_Rs: 3,
    Operation_Type: 'Medium',
  })
  const { data, refetch } = useQuery({
    queryKey: ['stitching-sheet', active],
    queryFn: () => api.get(`/stitching/sheets/${active}`).then(r => r.data),
  })

  const rows = (data?.rows ?? []) as Record<string, unknown>[]
  const cols = (data?.columns as string[]) ?? []

  const styleReportOptions = useMemo((): SearchSelectOption[] => {
    if (active !== 'style_master') return []
    const seen = new Set<string>()
    const opts: SearchSelectOption[] = []
    for (const r of rows) {
      const s = String(r.Style ?? '').trim()
      if (!s || seen.has(s)) continue
      seen.add(s)
      const opCount = rows.filter(x => String(x.Style ?? '').trim() === s).length
      opts.push({
        value: s,
        primary: s,
        secondary: `${opCount} operation(s)`,
        haystack: s.toLowerCase(),
      })
    }
    return opts.sort((a, b) => a.primary.localeCompare(b.primary))
  }, [active, rows])

  const invalidateSheet = () => {
    setSelected(new Set())
    void refetch()
    qc.invalidateQueries({ queryKey: ['stitching-sheet'] })
    qc.invalidateQueries({ queryKey: ['stitching-dashboard'] })
    qc.invalidateQueries({ queryKey: ['stitching-style-costing'] })
    qc.invalidateQueries({ queryKey: ['stitching-pe-reports'] })
  }

  const [styleForm, setStyleForm] = useState({
    Style: '',
    Operation: '',
    Operation_Type: 'Medium',
    Target: 80,
    Rate_Rs: 3,
  })
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
      setStyleForm({ Style: '', Operation: '', Operation_Type: 'Medium', Target: 80, Rate_Rs: 3 })
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
        New_Karigar_ID: karEdit.New_Karigar_ID.trim() || undefined,
      }),
    onSuccess: r => {
      onFlash('ok', r.data.message || 'Updated')
      setEditKarigar(null)
      invalidateSheet()
    },
    onError: (e: { response?: { data?: { detail?: string } } }) =>
      onFlash('err', String(e.response?.data?.detail || 'Update failed')),
  })

  const setKarigarActiveMut = useMutation({
    mutationFn: (payload: { karigar_id: string; active: boolean }) => {
      const pw = admin.adminPassword()
      if (!pw) throw new Error('Unlock admin first')
      return api.post('/stitching/master/karigar/active', { ...payload, admin_password: pw })
    },
    onSuccess: r => {
      onFlash('ok', r.data.message || 'Updated')
      invalidateSheet()
    },
    onError: (e: { response?: { data?: { detail?: string } } }) =>
      onFlash('err', String(e.response?.data?.detail || 'Update failed')),
  })

  const updateEmpMut = useMutation({
    mutationFn: () => {
      const pw = admin.adminPassword()
      if (!pw) throw new Error('Unlock admin first')
      return api.patch(`/stitching/master/employee/${encodeURIComponent(String(editEmployee?.E_Code ?? ''))}`, {
        Name: empEdit.Name,
        Type: empEdit.Type,
        Daily_Rate_Rs: empEdit.Daily_Rate_Rs,
        New_E_Code: empEdit.New_E_Code.trim() || undefined,
        admin_password: pw,
      })
    },
    onSuccess: r => {
      onFlash('ok', r.data.message || 'Employee updated')
      setEditEmployee(null)
      invalidateSheet()
    },
    onError: (e: { response?: { data?: { detail?: string } } }) =>
      onFlash('err', String(e.response?.data?.detail || 'Update failed')),
  })

  const updateStyleMut = useMutation({
    mutationFn: () => {
      const pw = admin.adminPassword()
      if (!pw) throw new Error('Unlock admin first')
      return api.patch('/stitching/master/style-operation', {
        Style: String(editStyleRow?.Style ?? ''),
        Operation: String(editStyleRow?.Operation ?? ''),
        Target: styleEdit.Target,
        Rate_Rs: styleEdit.Rate_Rs,
        Operation_Type: styleEdit.Operation_Type,
        admin_password: pw,
      })
    },
    onSuccess: r => {
      onFlash('ok', r.data.message || 'Style operation updated')
      setEditStyleRow(null)
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
    setEditEmployee(null)
    setEditStyleRow(null)
    setEditKarigar(row)
    setKarEdit({
      Name: String(row.Name ?? ''),
      Skill: String(row.Skill ?? ''),
      Daily_Rate_Rs: Number(row.Daily_Rate_Rs) || 420,
      Effective_From: todayStr(),
      New_Karigar_ID: '',
    })
  }

  const openEmployeeEdit = (row: Record<string, unknown>) => {
    setEditKarigar(null)
    setEditStyleRow(null)
    setEditEmployee(row)
    setEmpEdit({
      Name: String(row.Name ?? ''),
      Type: String(row.Type ?? 'Karigar'),
      Daily_Rate_Rs: Number(row.Daily_Rate_Rs) || 420,
      New_E_Code: '',
    })
  }

  const openStyleEdit = (row: Record<string, unknown>) => {
    setEditKarigar(null)
    setEditEmployee(null)
    setEditStyleRow(row)
    setStyleEdit({
      Target: Number(row.Target) || 80,
      Rate_Rs: Number(row.Rate_Rs) || 3,
      Operation_Type: String(row.Operation_Type || 'Medium'),
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
              setEditEmployee(null)
              setEditStyleRow(null)
            }}
            className={`text-xs px-3 py-1.5 rounded-lg ${active === k ? 'bg-[#002B5B] text-white' : 'border'}`}
          >
            {k.replace(/_/g, ' ')}
          </button>
        ))}
      </div>
      {active === 'style_master' && (
        <StyleMasterReportPanel
          styleOptions={styleReportOptions}
          focusStyle={reportFocusStyle}
          onFocusStyle={setReportFocusStyle}
        />
      )}
      {active === 'style_master' && (
        <div className="grid sm:grid-cols-5 gap-2 text-xs mb-3">
          {(['Style', 'Operation'] as const).map(k => (
            <label key={k}>
              {k}
              <input className="w-full border rounded mt-1 px-2 py-1" value={styleForm[k]} onChange={e => setStyleForm(f => ({ ...f, [k]: e.target.value }))} />
            </label>
          ))}
          <label>
            Type
            <select
              className="w-full border rounded mt-1 px-2 py-1"
              value={styleForm.Operation_Type}
              onChange={e => setStyleForm(f => ({ ...f, Operation_Type: e.target.value }))}
            >
              <option value="Easy">Easy</option>
              <option value="Medium">Medium</option>
              <option value="Hard">Hard</option>
            </select>
          </label>
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
                <label>
                  New E. Code (rename)
                  <input
                    className="w-full border rounded mt-1 px-2 py-1 font-mono"
                    placeholder={String(editKarigar.Karigar_ID ?? '')}
                    value={karEdit.New_Karigar_ID}
                    onChange={e => setKarEdit(f => ({ ...f, New_Karigar_ID: e.target.value }))}
                  />
                </label>
                <div className="flex gap-2 self-end">
                  <button type="button" onClick={() => updateKarMut.mutate()} disabled={updateKarMut.isPending} className="px-3 py-2 bg-violet-700 text-white rounded-lg font-semibold disabled:opacity-50">
                    Save
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setKarigarActiveMut.mutate({
                        karigar_id: String(editKarigar?.Karigar_ID ?? ''),
                        active: String((editKarigar as any)?.Active ?? 'true').toLowerCase() === 'true' ? false : true,
                      })
                    }
                    disabled={setKarigarActiveMut.isPending}
                    className="px-3 py-2 border rounded-lg"
                  >
                    {setKarigarActiveMut.isPending ? 'Updating…' : 'Toggle Active'}
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
      {active === 'employee_master' && editEmployee && (
        <div className="text-xs border border-teal-200 bg-teal-50 rounded-xl p-4 space-y-2 mb-3">
          <p className="font-semibold text-teal-900">Update employee — {String(editEmployee.E_Code)}</p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-2">
            <label>
              Name
              <input className="w-full border rounded mt-1 px-2 py-1" value={empEdit.Name} onChange={e => setEmpEdit(f => ({ ...f, Name: e.target.value }))} />
            </label>
            <label>
              Type
              <select className="w-full border rounded mt-1 px-2 py-1" value={empEdit.Type} onChange={e => setEmpEdit(f => ({ ...f, Type: e.target.value }))}>
                <option value="Karigar">Karigar</option>
                <option value="Stitching">Stitching</option>
                <option value="Operating">Operating</option>
                <option value="Admin">Admin</option>
              </select>
            </label>
            <label>
              Daily rate ₹
              <input type="number" className="w-full border rounded mt-1 px-2 py-1" value={empEdit.Daily_Rate_Rs} onChange={e => setEmpEdit(f => ({ ...f, Daily_Rate_Rs: +e.target.value }))} />
            </label>
            <label>
              New E. Code (rename)
              <input
                className="w-full border rounded mt-1 px-2 py-1 font-mono"
                placeholder={String(editEmployee.E_Code ?? '')}
                value={empEdit.New_E_Code}
                onChange={e => setEmpEdit(f => ({ ...f, New_E_Code: e.target.value }))}
              />
            </label>
            <div className="flex gap-2 self-end">
              <button type="button" onClick={() => updateEmpMut.mutate()} disabled={updateEmpMut.isPending} className="px-3 py-2 bg-teal-700 text-white rounded-lg font-semibold disabled:opacity-50">
                Save
              </button>
              <button type="button" onClick={() => setEditEmployee(null)} className="px-3 py-2 border rounded-lg">
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      {active === 'style_master' && editStyleRow && (
        <div className="text-xs border border-sky-200 bg-sky-50 rounded-xl p-4 space-y-2 mb-3">
          <p className="font-semibold text-sky-900">
            Edit — {String(editStyleRow.Style)} / {String(editStyleRow.Operation)}
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-2">
            <label>
              Target
              <input type="number" className="w-full border rounded mt-1 px-2 py-1" value={styleEdit.Target} onChange={e => setStyleEdit(f => ({ ...f, Target: +e.target.value }))} />
            </label>
            <label>
              Rate ₹
              <input type="number" step={0.25} className="w-full border rounded mt-1 px-2 py-1" value={styleEdit.Rate_Rs} onChange={e => setStyleEdit(f => ({ ...f, Rate_Rs: +e.target.value }))} />
            </label>
            <label>
              Type
              <select className="w-full border rounded mt-1 px-2 py-1" value={styleEdit.Operation_Type} onChange={e => setStyleEdit(f => ({ ...f, Operation_Type: e.target.value }))}>
                <option value="Easy">Easy</option>
                <option value="Medium">Medium</option>
                <option value="Hard">Hard</option>
              </select>
            </label>
            <div className="flex gap-2 self-end">
              <button type="button" onClick={() => updateStyleMut.mutate()} disabled={updateStyleMut.isPending} className="px-3 py-2 bg-sky-700 text-white rounded-lg font-semibold disabled:opacity-50">
                Save
              </button>
              <button type="button" onClick={() => setEditStyleRow(null)} className="px-3 py-2 border rounded-lg">
                Cancel
              </button>
            </div>
          </div>
        </div>
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
        onEditEmployee={active === 'employee_master' ? openEmployeeEdit : undefined}
        onEditStyle={active === 'style_master' ? openStyleEdit : undefined}
        onStyleReport={active === 'style_master' ? setReportFocusStyle : undefined}
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
  onEditEmployee,
  onEditStyle,
  onStyleReport,
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
  onEditEmployee?: (row: Record<string, unknown>) => void
  onEditStyle?: (row: Record<string, unknown>) => void
  onStyleReport?: (style: string) => void
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
                      {sheet === 'style_master' && c === 'Style' && onStyleReport ? (
                        <button
                          type="button"
                          className="text-[#2c5aa0] font-medium hover:underline text-left"
                          onClick={() => onStyleReport(String(r[c] ?? '').trim())}
                        >
                          {String(r[c] ?? '')}
                        </button>
                      ) : (
                        String(r[c] ?? '')
                      )}
                    </td>
                  ))}
                  <td className="px-2 py-1.5 text-right whitespace-nowrap">
                    {sheet === 'style_master' && onStyleReport && (
                      <button
                        type="button"
                        onClick={() => onStyleReport(String(r.Style ?? '').trim())}
                        className="text-[#2c5aa0] hover:underline mr-2"
                      >
                        Report
                      </button>
                    )}
                    {onEditKarigar && (
                      <button type="button" onClick={() => onEditKarigar(r)} className="text-violet-700 hover:underline mr-2">
                        Edit
                      </button>
                    )}
                    {onEditEmployee && (
                      <button type="button" onClick={() => onEditEmployee(r)} className="text-teal-700 hover:underline mr-2">
                        Edit
                      </button>
                    )}
                    {onEditStyle && (
                      <button type="button" onClick={() => onEditStyle(r)} className="text-sky-700 hover:underline mr-2">
                        Edit
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
  const opTypeRows = ((data?.operation_type_breakup ?? []) as Record<string, unknown>[])
    .map(r => ({
      Operation_Type: String(r.Operation_Type ?? 'Medium'),
      Count: Number(r.Count ?? 0),
      Pieces: Number(r.Pieces ?? 0),
      Piece_Value: Number(r.Piece_Value ?? 0),
    }))
    .sort((a, b) => {
      const order = { Easy: 1, Medium: 2, Hard: 3 } as Record<string, number>
      return (order[a.Operation_Type] ?? 99) - (order[b.Operation_Type] ?? 99)
    })

  return (
    <div className="space-y-4">
      <Section title="Employee performance — piece value vs salary">
        <p className="text-xs text-gray-600 mb-3">
          Compares production piece-value to <strong>full payroll</strong> (attendance + karigar expenses). Rows flagged{' '}
          <strong>Payroll only</strong> are paid but have no production in the period.
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
              <p className="text-xs text-gray-500">Total payroll paid</p>
              <p className="text-lg font-bold text-gray-800">₹{Number(data.summary.total_salary).toLocaleString()}</p>
              {data.summary.total_other_work_pay != null && (
                <p className="text-[10px] text-gray-500 mt-1">
                  Attendance ₹{Number(data.summary.total_attendance_pay ?? 0).toLocaleString()} + other ₹
                  {Number(data.summary.total_other_work_pay).toLocaleString()}
                </p>
              )}
            </div>
            <div className="bg-white border rounded-lg p-3">
              <p className="text-xs text-gray-500">Net surplus</p>
              <p className={`text-lg font-bold ${Number(data.summary.net_surplus) >= 0 ? 'text-green-700' : 'text-red-600'}`}>
                ₹{Number(data.summary.net_surplus).toLocaleString()}
              </p>
            </div>
          </div>
        )}
        {data?.ok && opTypeRows.length > 0 && (
          <div className="mb-4 border rounded-lg bg-slate-50/70 p-3">
            <p className="text-xs font-semibold text-gray-700 mb-2">Operation Type summary</p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              {opTypeRows.map(r => {
                const tone =
                  r.Operation_Type === 'Easy'
                    ? 'bg-green-50 border-green-200'
                    : r.Operation_Type === 'Hard'
                      ? 'bg-rose-50 border-rose-200'
                      : 'bg-amber-50 border-amber-200'
                const titleTone =
                  r.Operation_Type === 'Easy'
                    ? 'text-green-700'
                    : r.Operation_Type === 'Hard'
                      ? 'text-rose-700'
                      : 'text-amber-700'
                return (
                <div key={r.Operation_Type} className={`border rounded-lg px-3 py-2 ${tone}`}>
                  <p className={`text-xs font-semibold ${titleTone}`}>{r.Operation_Type}</p>
                  <p className="text-[11px] text-gray-600">Count: {r.Count.toLocaleString()}</p>
                  <p className="text-[11px] text-gray-600">Pieces: {r.Pieces.toLocaleString()}</p>
                  <p className="text-[11px] text-gray-600">Value: ₹{r.Piece_Value.toLocaleString()}</p>
                </div>
                )
              })}
            </div>
          </div>
        )}
        {data?.ok && (data.rows?.length ?? 0) > 0 && (
          <>
            {(data.rows as Record<string, unknown>[]).some(r => r.Payroll_Only_Expense) && (
              <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mb-3">
                Highlighted rows: paid in payroll but missing from production costing — mark as expense.
              </p>
            )}
            <ReportTableSection
              title="Karigar detail"
              rows={(data.rows as Record<string, unknown>[]).map(r => ({
                ...r,
                Payroll_Only_Expense: r.Payroll_Only_Expense ? 'Yes — expense' : '',
                Profitable_On_Payroll: Number(r.Surplus) >= 0 ? 'Yes' : 'No',
              }))}
              cols={[
                'E_Code',
                'Name',
                'Days',
                'Hrs',
                'Total_Payroll_Paid',
                'Attendance_Pay',
                'Other_Work_Pay',
                'Total_Pieces',
                'Piece_Value',
                'Surplus',
                'Profitable_On_Payroll',
                'Payroll_Only_Expense',
                'ROI_%',
                'Avg_Eff',
                'Grade',
              ]}
              downloadName={`performance_${from}_${to}`}
            />
          </>
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
  const useCols = cols.length
    ? cols
    : Object.keys(rows[0] ?? {})
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
                  {r[c] === 0 || r[c] === '0' ? '0' : String(r[c] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
