import { useState, useMemo, useRef, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Cell,
} from 'recharts'
import api from '../api/client'

// ── Types ────────────────────────────────────────────────────────
interface PLStatement {
  gross_revenue:    number
  returns_value:    number
  net_revenue:      number
  cogs:             number
  cogs_available:   boolean
  gross_profit:     number
  gross_margin_pct: number
  total_expenses:   number
  net_profit:       number
  platform_breakdown: PlatformRev[]
}
interface PlatformRev {
  platform:        string
  loaded:          boolean
  gross_revenue:   number
  returns_value:   number
  net_revenue:     number
  return_rate_pct: number
}
interface GSTSummary {
  loaded:  boolean
  months:  { month: string; cgst: number; sgst: number; igst: number; total_tax: number }[]
  totals:  { cgst: number; sgst: number; igst: number; total_tax: number }
}
interface Expense {
  id:          number
  date:        string
  category:    string
  description: string
  amount:      number
  gst_amount:  number
  created_at:  string
}

// ── Constants ────────────────────────────────────────────────────
const PLATFORM_COLORS: Record<string, string> = {
  Amazon: '#002B5B', Myntra: '#E91E63', Meesho: '#9C27B0', Flipkart: '#F7971D',
}
const EXPENSE_CATEGORIES = ['Logistics', 'Platform Fees', 'Marketing', 'Rent', 'Salaries', 'Utilities', 'Other']

// ── Formatting ───────────────────────────────────────────────────
const fmt    = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN')
const fmtPct = (n: number) => n.toFixed(1) + '%'

// ── Date helpers (same as Dashboard) ────────────────────────────
function toIso(d: Date) { return d.toISOString().split('T')[0] }
function daysAgo(n: number)   { const d = new Date(); d.setDate(d.getDate() - n); return toIso(d) }
function monthsAgo(n: number) { const d = new Date(); d.setMonth(d.getMonth() - n); return toIso(d) }
const TODAY = toIso(new Date())

const PRESETS = [
  { label: '30D', start: () => daysAgo(30)   },
  { label: '90D', start: () => daysAgo(90)   },
  { label: '6M',  start: () => monthsAgo(6)  },
  { label: '1Y',  start: () => monthsAgo(12) },
  { label: 'All', start: () => ''            },
] as const

// ── Main Component ───────────────────────────────────────────────
export default function Finance() {
  const qc = useQueryClient()

  // ── PIN lock state ───────────────────────────────────────────
  const [pinUnlocked, setPinUnlocked] = useState(false)
  const [pinInput,    setPinInput]    = useState('')
  const [pinError,    setPinError]    = useState('')
  const [pinChecking, setPinChecking] = useState(false)
  const [showPin,     setShowPin]     = useState(false)

  // Auto-unlock if no PIN is configured on the server
  useEffect(() => {
    api.get('/finance/pin-required').then(({ data }) => {
      if (!data.required) setPinUnlocked(true)
    }).catch(() => { /* ignore — show lock screen on error */ })
  }, [])

  const [activeTab,    setActiveTab]    = useState<'pl'|'gst'|'expenses'|'revenue'>('pl')
  const [dateStart,    setDateStart]    = useState(() => daysAgo(90))
  const [dateEnd,      setDateEnd]      = useState(TODAY)
  const [activePreset, setActivePreset] = useState<string>('90D')

  // Expense form state
  const [expDate,  setExpDate]  = useState(TODAY)
  const [expCat,   setExpCat]   = useState('Logistics')
  const [expDesc,  setExpDesc]  = useState('')
  const [expAmt,   setExpAmt]   = useState('')
  const [expGst,   setExpGst]   = useState('')
  const [addError, setAddError] = useState('')

  // COGS upload
  const cogsRef = useRef<HTMLInputElement>(null)
  const [cogsMsg, setCogsMsg] = useState('')

  function applyPreset(label: string, startFn: () => string) {
    setDateStart(startFn()); setDateEnd(TODAY); setActivePreset(label)
  }
  function handleStartChange(v: string) { setDateStart(v); setActivePreset('') }
  function handleEndChange(v: string)   { setDateEnd(v);   setActivePreset('') }

  const dateQ = useMemo(() => {
    const p = new URLSearchParams()
    if (dateStart) p.set('start_date', dateStart)
    if (dateEnd)   p.set('end_date',   dateEnd)
    return p.toString()
  }, [dateStart, dateEnd])

  // ── Queries ──────────────────────────────────────────────────
  const { data: pl, isLoading: loadPL } = useQuery<PLStatement>({
    queryKey: ['finance-pl', dateStart, dateEnd],
    queryFn:  async () => { const { data } = await api.get(`/finance/pl?${dateQ}`); return data },
    staleTime: 2 * 60 * 1000,
  })

  const { data: gst, isLoading: loadGST } = useQuery<GSTSummary>({
    queryKey: ['finance-gst', dateStart, dateEnd],
    queryFn:  async () => { const { data } = await api.get(`/finance/gst?${dateQ}`); return data },
    staleTime: 2 * 60 * 1000,
    enabled:   activeTab === 'gst',
  })

  const { data: platformRev, isLoading: loadRev } = useQuery<PlatformRev[]>({
    queryKey: ['finance-platform-rev', dateStart, dateEnd],
    queryFn:  async () => { const { data } = await api.get(`/finance/platform-revenue?${dateQ}`); return data },
    staleTime: 2 * 60 * 1000,
    enabled:   activeTab === 'revenue',
  })

  const { data: expenses, isLoading: loadExp } = useQuery<Expense[]>({
    queryKey: ['finance-expenses', dateStart, dateEnd],
    queryFn:  async () => { const { data } = await api.get(`/finance/expenses?${dateQ}`); return data },
    staleTime: 60 * 1000,
    enabled:   activeTab === 'expenses',
  })

  // ── Mutations ────────────────────────────────────────────────
  const addMut = useMutation({
    mutationFn: (body: object) => api.post('/finance/expenses', body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['finance-expenses'] })
      qc.invalidateQueries({ queryKey: ['finance-pl'] })
      setExpDesc(''); setExpAmt(''); setExpGst(''); setAddError('')
    },
    onError: () => setAddError('Failed to add expense.'),
  })

  const delMut = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/expenses/${id}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['finance-expenses'] })
      qc.invalidateQueries({ queryKey: ['finance-pl'] })
    },
  })

  function handleAddExpense() {
    const amount = parseFloat(expAmt)
    if (!expDate || !expCat || isNaN(amount) || amount <= 0) {
      setAddError('Date, category and a valid amount are required.')
      return
    }
    addMut.mutate({ date: expDate, category: expCat, description: expDesc, amount, gst_amount: parseFloat(expGst) || 0 })
  }

  async function handleCogsUpload(file: File) {
    setCogsMsg('')
    const fd = new FormData(); fd.append('file', file)
    try {
      const { data } = await api.post('/upload/cogs', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      setCogsMsg(data.ok ? `✓ ${data.message}` : `✗ ${data.message}`)
      if (data.ok) qc.invalidateQueries({ queryKey: ['finance-pl'] })
    } catch { setCogsMsg('✗ Upload failed.') }
  }

  const expenseTotal = useMemo(() =>
    (expenses ?? []).reduce((s, e) => s + e.amount + e.gst_amount, 0), [expenses])

  const revChartData = useMemo(() =>
    (platformRev ?? []).filter(p => p.loaded).map(p => ({
      platform: p.platform, Gross: Math.round(p.gross_revenue), Net: Math.round(p.net_revenue),
    })), [platformRev])

  async function handlePinSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!pinInput.trim()) { setPinError('Enter the Finance PIN.'); return }
    setPinChecking(true); setPinError('')
    try {
      const { data } = await api.post('/finance/verify-pin', { pin: pinInput })
      if (data.ok) {
        setPinUnlocked(true); setPinInput('')
      } else {
        setPinError('Incorrect PIN. Please try again.')
        setPinInput('')
      }
    } catch { setPinError('Verification failed. Please try again.') }
    finally { setPinChecking(false) }
  }

  // ── PIN lock screen ──────────────────────────────────────────
  if (!pinUnlocked) {
    return (
      <div className="max-w-7xl mx-auto flex items-center justify-center min-h-[70vh]">
        <div className="bg-white rounded-2xl border border-gray-200 shadow-lg p-10 w-full max-w-sm text-center">
          <div className="w-16 h-16 rounded-full bg-[#002B5B] flex items-center justify-center mx-auto mb-5 shadow-md">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
            </svg>
          </div>
          <h2 className="text-xl font-bold text-gray-800 mb-1">Finance Module</h2>
          <p className="text-sm text-gray-500 mb-6">Enter your Finance PIN to access financial data</p>

          <form onSubmit={handlePinSubmit} className="space-y-4">
            <div className="relative">
              <input
                type={showPin ? 'text' : 'password'}
                value={pinInput}
                onChange={e => setPinInput(e.target.value)}
                placeholder="Enter PIN"
                autoFocus
                className="w-full border border-gray-200 rounded-xl px-4 py-3 pr-11 text-sm text-center tracking-[0.3em] bg-gray-50 focus:outline-none focus:ring-2 focus:ring-[#002B5B] focus:border-transparent transition"
              />
              <button
                type="button"
                onClick={() => setShowPin(v => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                tabIndex={-1}
              >
                {showPin ? (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
                  </svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                )}
              </button>
            </div>

            {pinError && (
              <p className="text-xs text-red-600 font-medium">{pinError}</p>
            )}

            <button
              type="submit"
              disabled={pinChecking}
              className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-[#002B5B] hover:bg-[#003875] disabled:opacity-50 shadow-md transition-all"
            >
              {pinChecking ? 'Verifying…' : 'Unlock Finance'}
            </button>
          </form>
          <p className="text-xs text-gray-400 mt-4">PIN is set by your administrator</p>
        </div>
      </div>
    )
  }

  // ── Render ───────────────────────────────────────────────────
  return (
    <div className="max-w-7xl mx-auto space-y-5">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-[#002B5B]">💰 Finance</h2>
        <p className="text-gray-500 text-sm mt-1">P&amp;L · GST · Expenses · Platform Revenue</p>
      </div>

      {/* Date Filter Bar */}
      <div className="bg-white rounded-xl border border-gray-200 px-4 py-3 shadow-sm flex flex-wrap items-center gap-3">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Date Range</span>
        <div className="flex gap-1">
          {PRESETS.map(({ label, start }) => (
            <button key={label} onClick={() => applyPreset(label, start)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${activePreset === label ? 'bg-[#002B5B] text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}>
              {label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 ml-2">
          <span className="text-xs text-gray-400">From</span>
          <input type="date" value={dateStart} max={dateEnd || TODAY} onChange={e => handleStartChange(e.target.value)}
            className="text-xs border border-gray-200 rounded px-2 py-1 text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          <span className="text-xs text-gray-400">To</span>
          <input type="date" value={dateEnd} min={dateStart} max={TODAY} onChange={e => handleEndChange(e.target.value)}
            className="text-xs border border-gray-200 rounded px-2 py-1 text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-300" />
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 gap-1">
        {([['pl','P&L Statement'],['gst','GST Summary'],['expenses','Expenses'],['revenue','Platform Revenue']] as const).map(([id, label]) => (
          <button key={id} onClick={() => setActiveTab(id)}
            className={`px-4 py-2.5 text-sm font-medium transition-colors rounded-t ${activeTab === id ? 'border-b-2 border-[#002B5B] text-[#002B5B] bg-blue-50' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* ── Tab 1: P&L ── */}
      {activeTab === 'pl' && (
        <div className="space-y-4">
          {/* COGS Upload Card */}
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-center gap-4">
            <div className="flex-1">
              <p className="text-sm font-semibold text-amber-800">Cost of Goods Sold (COGS)</p>
              <p className="text-xs text-amber-600 mt-0.5">
                {pl?.cogs_available
                  ? `COGS data loaded — ₹${Math.round(pl.cogs).toLocaleString('en-IN')} deducted`
                  : 'Upload a COGS sheet (Excel/CSV with SKU + Cost Price) to enable gross profit calculation.'}
              </p>
              {cogsMsg && <p className={`text-xs mt-1 font-medium ${cogsMsg.startsWith('✓') ? 'text-green-700' : 'text-red-600'}`}>{cogsMsg}</p>}
            </div>
            <div>
              <input ref={cogsRef} type="file" accept=".xlsx,.csv" className="hidden"
                onChange={e => { if (e.target.files?.[0]) handleCogsUpload(e.target.files[0]); e.target.value = '' }} />
              <button onClick={() => cogsRef.current?.click()}
                className="px-3 py-1.5 text-xs font-medium bg-amber-600 text-white rounded hover:bg-amber-700 transition-colors">
                {pl?.cogs_available ? 'Re-upload COGS' : 'Upload COGS Sheet'}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* P&L Table */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-5 py-3 border-b border-gray-100">
                <h3 className="text-sm font-semibold text-gray-700">Profit &amp; Loss Statement</h3>
                <p className="text-xs text-gray-400 mt-0.5">{dateStart || 'All time'} → {dateEnd || 'today'}</p>
              </div>
              {loadPL ? (
                <div className="p-5 space-y-3">{[1,2,3,4,5,6,7,8].map(i => <div key={i} className="h-6 bg-gray-100 rounded animate-pulse" />)}</div>
              ) : (
                <table className="w-full text-sm">
                  <tbody>
                    <PLRow label="Gross Revenue"       value={pl?.gross_revenue    ?? 0} />
                    <PLRow label="(–) Returns"          value={-(pl?.returns_value  ?? 0)} indent />
                    <PLRow label="Net Revenue"          value={pl?.net_revenue      ?? 0} bold />
                    <PLRow label="(–) COGS"             value={-(pl?.cogs           ?? 0)} indent muted={!pl?.cogs_available} />
                    <PLRow label="Gross Profit"         value={pl?.gross_profit     ?? 0} bold colored />
                    <PLRow label="Gross Margin"         value={pl?.gross_margin_pct ?? 0} isPct muted={!pl?.cogs_available} />
                    <PLRow label="(–) Operating Expenses" value={-(pl?.total_expenses ?? 0)} indent />
                    <PLRow label="Net Profit"           value={pl?.net_profit       ?? 0} bold colored large />
                  </tbody>
                </table>
              )}
            </div>

            {/* Bar chart */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
              <h3 className="text-sm font-semibold text-gray-700 mb-4">Financial Overview</h3>
              {loadPL ? (
                <div className="h-48 flex items-center justify-center text-gray-400 text-sm animate-pulse">Loading…</div>
              ) : (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart
                    data={[
                      { name: 'Revenue',   value: Math.round(pl?.net_revenue   ?? 0) },
                      { name: 'COGS',      value: Math.round(pl?.cogs          ?? 0) },
                      { name: 'Expenses',  value: Math.round(pl?.total_expenses ?? 0) },
                      { name: 'Net Profit',value: Math.round(pl?.net_profit     ?? 0) },
                    ]}
                    margin={{ top: 5, right: 10, left: 10, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#6B7280' }} />
                    <YAxis tick={{ fontSize: 10, fill: '#9CA3AF' }} tickFormatter={v => `₹${(v/1000).toFixed(0)}k`} />
                    <Tooltip
                      contentStyle={{ fontSize: 12, borderRadius: 8 }}
                      formatter={(v: number | undefined) => [fmt(v ?? 0), '']}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {['#002B5B', '#EF4444', '#F59E0B', '#10B981'].map((color, i) => (
                        <Cell key={i} fill={color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 2: GST ── */}
      {activeTab === 'gst' && (
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-3 text-xs text-blue-700 flex gap-2 items-start">
            <span className="mt-0.5">ℹ️</span>
            <span>GST component breakdown (CGST / SGST / IGST) is available for <strong>Amazon MTR only</strong>. For other platforms, obtain GST data from their respective portals or GSTR-2B.</span>
          </div>

          {loadGST ? (
            <div className="h-48 flex items-center justify-center text-gray-400 text-sm animate-pulse">Loading…</div>
          ) : !gst?.loaded ? (
            <div className="bg-white rounded-xl border border-gray-200 p-10 text-center">
              <p className="text-3xl mb-2">📭</p>
              <p className="text-sm text-gray-500">No Amazon MTR data loaded.</p>
              <a href="/upload" className="text-xs text-blue-600 hover:underline mt-1 inline-block">Go to Upload →</a>
            </div>
          ) : (
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-5 py-3 border-b border-gray-100">
                <h3 className="text-sm font-semibold text-gray-700">Amazon GST Summary</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
                      <th className="px-4 py-2.5 text-left">Month</th>
                      <th className="px-4 py-2.5 text-right">IGST</th>
                      <th className="px-4 py-2.5 text-right">CGST</th>
                      <th className="px-4 py-2.5 text-right">SGST</th>
                      <th className="px-4 py-2.5 text-right font-semibold">Total Tax</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(gst.months ?? []).map(row => (
                      <tr key={row.month} className="border-t border-gray-50 hover:bg-gray-50">
                        <td className="px-4 py-2 font-medium text-gray-700">{row.month}</td>
                        <td className="px-4 py-2 text-right text-gray-600">{fmt(row.igst)}</td>
                        <td className="px-4 py-2 text-right text-gray-600">{fmt(row.cgst)}</td>
                        <td className="px-4 py-2 text-right text-gray-600">{fmt(row.sgst)}</td>
                        <td className="px-4 py-2 text-right font-semibold text-gray-800">{fmt(row.total_tax)}</td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr className="bg-[#002B5B] text-white text-xs font-semibold">
                      <td className="px-4 py-2.5">Total</td>
                      <td className="px-4 py-2.5 text-right">{fmt(gst.totals.igst)}</td>
                      <td className="px-4 py-2.5 text-right">{fmt(gst.totals.cgst)}</td>
                      <td className="px-4 py-2.5 text-right">{fmt(gst.totals.sgst)}</td>
                      <td className="px-4 py-2.5 text-right">{fmt(gst.totals.total_tax)}</td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Tab 3: Expenses ── */}
      {activeTab === 'expenses' && (
        <div className="space-y-4">
          {/* Add Form */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Add Expense</h3>
            <div className="flex flex-wrap gap-2 items-end">
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500">Date</label>
                <input type="date" value={expDate} max={TODAY} onChange={e => setExpDate(e.target.value)}
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500">Category</label>
                <select value={expCat} onChange={e => setExpCat(e.target.value)}
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
                  {EXPENSE_CATEGORIES.map(c => <option key={c}>{c}</option>)}
                </select>
              </div>
              <div className="flex flex-col gap-1 flex-1 min-w-[140px]">
                <label className="text-xs text-gray-500">Description</label>
                <input type="text" value={expDesc} onChange={e => setExpDesc(e.target.value)} placeholder="e.g. Amazon FBA fees Jan"
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
              <div className="flex flex-col gap-1 w-28">
                <label className="text-xs text-gray-500">Amount (₹)</label>
                <input type="number" value={expAmt} onChange={e => setExpAmt(e.target.value)} placeholder="0"
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
              <div className="flex flex-col gap-1 w-28">
                <label className="text-xs text-gray-500">GST Amount (₹)</label>
                <input type="number" value={expGst} onChange={e => setExpGst(e.target.value)} placeholder="0"
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
              <button onClick={handleAddExpense} disabled={addMut.isPending}
                className="px-4 py-1.5 text-xs font-semibold bg-[#002B5B] text-white rounded hover:bg-blue-900 transition-colors disabled:opacity-60">
                {addMut.isPending ? 'Adding…' : '+ Add'}
              </button>
            </div>
            {addError && <p className="text-xs text-red-600 mt-2">{addError}</p>}
          </div>

          {/* Expenses Table */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            {loadExp ? (
              <div className="p-8 flex items-center justify-center text-gray-400 text-sm animate-pulse">Loading…</div>
            ) : !expenses || expenses.length === 0 ? (
              <div className="p-8 text-center text-gray-400 text-sm">No expenses recorded yet.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
                      <th className="px-4 py-2.5 text-left">Date</th>
                      <th className="px-4 py-2.5 text-left">Category</th>
                      <th className="px-4 py-2.5 text-left">Description</th>
                      <th className="px-4 py-2.5 text-right">Amount</th>
                      <th className="px-4 py-2.5 text-right">GST</th>
                      <th className="px-4 py-2.5 text-right">Total</th>
                      <th className="px-4 py-2.5"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {expenses.map(e => (
                      <tr key={e.id} className="border-t border-gray-50 hover:bg-gray-50">
                        <td className="px-4 py-2 text-gray-600">{e.date}</td>
                        <td className="px-4 py-2">
                          <span className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">{e.category}</span>
                        </td>
                        <td className="px-4 py-2 text-gray-500 max-w-[200px] truncate" title={e.description}>{e.description || '—'}</td>
                        <td className="px-4 py-2 text-right text-gray-700">{fmt(e.amount)}</td>
                        <td className="px-4 py-2 text-right text-gray-500">{e.gst_amount > 0 ? fmt(e.gst_amount) : '—'}</td>
                        <td className="px-4 py-2 text-right font-medium text-gray-800">{fmt(e.amount + e.gst_amount)}</td>
                        <td className="px-4 py-2 text-right">
                          <button
                            onClick={() => { if (window.confirm('Delete this expense?')) delMut.mutate(e.id) }}
                            className="text-red-400 hover:text-red-600 text-xs font-medium transition-colors">
                            ✕
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr className="bg-gray-50 text-sm font-semibold border-t border-gray-200">
                      <td className="px-4 py-2.5" colSpan={5}>Total Expenses</td>
                      <td className="px-4 py-2.5 text-right text-gray-800">{fmt(expenseTotal)}</td>
                      <td />
                    </tr>
                  </tfoot>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Tab 4: Platform Revenue ── */}
      {activeTab === 'revenue' && (
        <div className="space-y-4">
          {/* Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {loadRev ? (
              [1,2,3,4].map(i => <div key={i} className="bg-white rounded-xl border border-gray-200 h-28 animate-pulse" />)
            ) : (
              (platformRev ?? []).map(p => {
                const color = PLATFORM_COLORS[p.platform] ?? '#6B7280'
                const rrColor = p.return_rate_pct > 30 ? 'text-red-600' : p.return_rate_pct > 15 ? 'text-amber-500' : 'text-green-600'
                return (
                  <div key={p.platform} className={`bg-white rounded-xl border overflow-hidden shadow-sm relative ${p.loaded ? 'border-gray-200' : 'border-gray-100'}`}>
                    <div className="h-1 w-full" style={{ backgroundColor: color }} />
                    {!p.loaded && (
                      <div className="absolute inset-0 top-1 bg-white/80 flex items-center justify-center">
                        <span className="text-gray-400 text-xs">Not Loaded</span>
                      </div>
                    )}
                    <div className="p-4">
                      <p className="text-xs font-semibold text-gray-500 mb-1">{p.platform}</p>
                      <p className="text-lg font-bold text-gray-800">{fmt(p.net_revenue)}</p>
                      <p className="text-xs text-gray-400">net revenue</p>
                      <div className="mt-2 pt-2 border-t border-gray-100 flex justify-between text-xs">
                        <span className="text-gray-400">Gross: {fmt(p.gross_revenue)}</span>
                        <span className={rrColor}>{fmtPct(p.return_rate_pct)} ret.</span>
                      </div>
                    </div>
                  </div>
                )
              })
            )}
          </div>

          {/* Chart */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
            <h3 className="text-sm font-semibold text-gray-700 mb-4">Gross vs Net Revenue by Platform</h3>
            {loadRev ? (
              <div className="h-48 flex items-center justify-center text-gray-400 text-sm animate-pulse">Loading…</div>
            ) : revChartData.length === 0 ? (
              <div className="h-48 flex items-center justify-center text-gray-400 text-sm">No platform data loaded</div>
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={revChartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
                  <XAxis dataKey="platform" tick={{ fontSize: 12, fill: '#6B7280' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#9CA3AF' }} tickFormatter={v => `₹${(v/1000).toFixed(0)}k`} />
                  <Tooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8 }}
                    formatter={(v: number | undefined) => [fmt(v ?? 0), '']}
                  />
                  <Legend iconType="square" iconSize={10} wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Gross" name="Gross Revenue" fill="#93C5FD" radius={[3,3,0,0]} />
                  <Bar dataKey="Net"   name="Net Revenue"   fill="#002B5B" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            )}

            {/* Summary Row */}
            {!loadRev && (platformRev ?? []).some(p => p.loaded) && (
              <div className="mt-4 pt-4 border-t border-gray-100 grid grid-cols-3 gap-4 text-center">
                <div>
                  <p className="text-xs text-gray-400">Total Gross</p>
                  <p className="font-bold text-gray-800">{fmt((platformRev ?? []).reduce((s,p) => s + p.gross_revenue, 0))}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Total Returns</p>
                  <p className="font-bold text-red-600">{fmt((platformRev ?? []).reduce((s,p) => s + p.returns_value, 0))}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Total Net</p>
                  <p className="font-bold text-green-700">{fmt((platformRev ?? []).reduce((s,p) => s + p.net_revenue, 0))}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── P&L Row helper ───────────────────────────────────────────────
function PLRow({
  label, value, indent = false, bold = false, colored = false,
  isPct = false, muted = false, large = false,
}: {
  label: string; value: number; indent?: boolean; bold?: boolean
  colored?: boolean; isPct?: boolean; muted?: boolean; large?: boolean
}) {
  const isNeg = value < 0
  const display = isPct ? fmtPct(value) : fmt(Math.abs(value))
  const signedDisplay = isPct ? display : (isNeg ? `(${display})` : display)

  let valColor = 'text-gray-700'
  if (colored) valColor = value >= 0 ? 'text-green-700' : 'text-red-600'
  else if (isNeg) valColor = 'text-red-500'
  if (muted) valColor = 'text-gray-400'

  return (
    <tr className={`border-t border-gray-50 ${large ? 'bg-gray-50' : ''}`}>
      <td className={`px-5 py-2.5 ${indent ? 'pl-8' : ''} ${bold ? 'font-semibold' : ''} ${large ? 'text-base' : 'text-sm'} text-gray-700`}>
        {label}
      </td>
      <td className={`px-5 py-2.5 text-right ${bold ? 'font-bold' : 'font-medium'} ${large ? 'text-base' : 'text-sm'} ${valColor}`}>
        {signedDisplay}
      </td>
    </tr>
  )
}
