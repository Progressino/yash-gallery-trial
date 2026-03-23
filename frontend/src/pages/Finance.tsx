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
interface LedgerGroup {
  id:           number
  name:         string
  parent_group: string
  nature:       string
}
interface Ledger {
  id:             number
  name:           string
  group_id:       number | null
  group_name:     string
  gstin:          string
  pan:            string
  state:          string
  state_code:     string
  address:        string
  tds_applicable: number
  tds_section:    string
  is_active:      number
}
interface GSTClassification {
  id:       number
  name:     string
  hsn_sac:  string
  gst_rate: number
  type:     string
}
interface TDSSection {
  id:              number
  section:         string
  description:     string
  rate_individual: number
  rate_company:    number
  threshold:       number
}
interface VoucherLine {
  id:           number
  voucher_id:   number
  expense_head: string
  description:  string
  amount:       number
  cost_centre:  string
}
interface Voucher {
  id:             number
  voucher_no:     string
  voucher_date:   string
  voucher_type:   string
  party_name:     string
  party_gstin:    string
  party_state:    string
  bill_no:        string
  bill_date:      string
  supply_type:    string
  narration:      string
  taxable_amount: number
  cgst_amount:    number
  sgst_amount:    number
  igst_amount:    number
  tds_section:    string
  tds_rate:       number
  tds_amount:     number
  total_amount:   number
  net_payable:    number
  lines:          VoucherLine[]
}
interface SalesUpload {
  id:            number
  platform:      string
  period:        string
  filename:      string
  total_revenue: number
  total_orders:  number
  total_returns: number
  net_revenue:   number
  is_locked:     number
  uploaded_by:   string
  upload_notes:  string
  created_at:    string
}

// ── Constants ────────────────────────────────────────────────────
const PLATFORM_COLORS: Record<string, string> = {
  Amazon: '#002B5B', Myntra: '#E91E63', Meesho: '#9C27B0', Flipkart: '#F7971D',
}
const EXPENSE_CATEGORIES = ['Logistics', 'Platform Fees', 'Marketing', 'Rent', 'Salaries', 'Utilities', 'Other']
const SALES_PLATFORMS = ['Amazon', 'Myntra', 'Meesho', 'Flipkart', 'Snapdeal', 'All Platforms']

// ── Formatting ───────────────────────────────────────────────────
const fmt    = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN')
const fmtPct = (n: number) => n.toFixed(1) + '%'

// ── Date helpers ─────────────────────────────────────────────────
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

type FinanceTab = 'pl' | 'gst' | 'expenses' | 'revenue' | 'vouchers' | 'masters' | 'sales-uploads'
type MastersSubTab = 'ledger-groups' | 'ledgers' | 'gst-classifications' | 'tds-sections'

// ── Main Component ───────────────────────────────────────────────
export default function Finance() {
  const qc = useQueryClient()

  // ── PIN lock state ───────────────────────────────────────────
  const [pinUnlocked, setPinUnlocked] = useState(false)
  const [pinInput,    setPinInput]    = useState('')
  const [pinError,    setPinError]    = useState('')
  const [pinChecking, setPinChecking] = useState(false)
  const [showPin,     setShowPin]     = useState(false)

  useEffect(() => {
    api.get('/finance/pin-required').then(({ data }) => {
      if (!data.required) setPinUnlocked(true)
    }).catch(() => {})
  }, [])

  const [activeTab,    setActiveTab]    = useState<FinanceTab>('pl')
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
              <button type="button" onClick={() => setShowPin(v => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors" tabIndex={-1}>
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
            {pinError && <p className="text-xs text-red-600 font-medium">{pinError}</p>}
            <button type="submit" disabled={pinChecking}
              className="w-full py-3 rounded-xl text-sm font-semibold text-white bg-[#002B5B] hover:bg-[#003875] disabled:opacity-50 shadow-md transition-all">
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
        <h2 className="text-2xl font-bold text-[#002B5B]">Finance</h2>
        <p className="text-gray-500 text-sm mt-1">P&amp;L · GST · Expenses · Platform Revenue · Vouchers · Masters</p>
      </div>

      {/* Date Filter Bar — only for analytics tabs */}
      {(activeTab === 'pl' || activeTab === 'gst' || activeTab === 'expenses' || activeTab === 'revenue') && (
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
      )}

      {/* Tabs */}
      <div className="flex border-b border-gray-200 gap-1 flex-wrap">
        {([
          ['pl','P&L Statement'],
          ['gst','GST Summary'],
          ['expenses','Expenses'],
          ['revenue','Platform Revenue'],
          ['vouchers','Expense Vouchers'],
          ['masters','Masters'],
          ['sales-uploads','Sales Uploads'],
        ] as [FinanceTab, string][]).map(([id, label]) => (
          <button key={id} onClick={() => setActiveTab(id)}
            className={`px-4 py-2.5 text-sm font-medium transition-colors rounded-t ${activeTab === id ? 'border-b-2 border-[#002B5B] text-[#002B5B] bg-blue-50' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* ── Tab: P&L ── */}
      {activeTab === 'pl' && (
        <div className="space-y-4">
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
                    <PLRow label="Gross Revenue"         value={pl?.gross_revenue    ?? 0} />
                    <PLRow label="(–) Returns"            value={-(pl?.returns_value  ?? 0)} indent />
                    <PLRow label="Net Revenue"            value={pl?.net_revenue      ?? 0} bold />
                    <PLRow label="(–) COGS"               value={-(pl?.cogs           ?? 0)} indent muted={!pl?.cogs_available} />
                    <PLRow label="Gross Profit"           value={pl?.gross_profit     ?? 0} bold colored />
                    <PLRow label="Gross Margin"           value={pl?.gross_margin_pct ?? 0} isPct muted={!pl?.cogs_available} />
                    <PLRow label="(–) Operating Expenses" value={-(pl?.total_expenses ?? 0)} indent />
                    <PLRow label="Net Profit"             value={pl?.net_profit       ?? 0} bold colored large />
                  </tbody>
                </table>
              )}
            </div>

            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
              <h3 className="text-sm font-semibold text-gray-700 mb-4">Financial Overview</h3>
              {loadPL ? (
                <div className="h-48 flex items-center justify-center text-gray-400 text-sm animate-pulse">Loading…</div>
              ) : (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart
                    data={[
                      { name: 'Revenue',    value: Math.round(pl?.net_revenue   ?? 0) },
                      { name: 'COGS',       value: Math.round(pl?.cogs          ?? 0) },
                      { name: 'Expenses',   value: Math.round(pl?.total_expenses ?? 0) },
                      { name: 'Net Profit', value: Math.round(pl?.net_profit    ?? 0) },
                    ]}
                    margin={{ top: 5, right: 10, left: 10, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#F3F4F6" />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#6B7280' }} />
                    <YAxis tick={{ fontSize: 10, fill: '#9CA3AF' }} tickFormatter={v => `₹${(v/1000).toFixed(0)}k`} />
                    <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v: number | undefined) => [fmt(v ?? 0), '']} />
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

      {/* ── Tab: GST ── */}
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

      {/* ── Tab: Expenses ── */}
      {activeTab === 'expenses' && (
        <div className="space-y-4">
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
                        <td className="px-4 py-2"><span className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">{e.category}</span></td>
                        <td className="px-4 py-2 text-gray-500 max-w-[200px] truncate" title={e.description}>{e.description || '—'}</td>
                        <td className="px-4 py-2 text-right text-gray-700">{fmt(e.amount)}</td>
                        <td className="px-4 py-2 text-right text-gray-500">{e.gst_amount > 0 ? fmt(e.gst_amount) : '—'}</td>
                        <td className="px-4 py-2 text-right font-medium text-gray-800">{fmt(e.amount + e.gst_amount)}</td>
                        <td className="px-4 py-2 text-right">
                          <button onClick={() => { if (window.confirm('Delete this expense?')) delMut.mutate(e.id) }}
                            className="text-red-400 hover:text-red-600 text-xs font-medium transition-colors">✕</button>
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

      {/* ── Tab: Platform Revenue ── */}
      {activeTab === 'revenue' && (
        <div className="space-y-4">
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
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v: number | undefined) => [fmt(v ?? 0), '']} />
                  <Legend iconType="square" iconSize={10} wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Gross" name="Gross Revenue" fill="#93C5FD" radius={[3,3,0,0]} />
                  <Bar dataKey="Net"   name="Net Revenue"   fill="#002B5B" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
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

      {/* ── Tab: Vouchers ── */}
      {activeTab === 'vouchers' && <VouchersTab />}

      {/* ── Tab: Masters ── */}
      {activeTab === 'masters' && <MastersTab />}

      {/* ── Tab: Sales Uploads ── */}
      {activeTab === 'sales-uploads' && <SalesUploadsTab />}
    </div>
  )
}

// ── Vouchers Tab ─────────────────────────────────────────────────
function VouchersTab() {
  const qc = useQueryClient()

  // Queries for dropdowns
  const { data: ledgers = [] } = useQuery<Ledger[]>({
    queryKey: ['finance-ledgers'],
    queryFn: async () => { const { data } = await api.get('/finance/masters/ledgers'); return data },
    staleTime: 5 * 60 * 1000,
  })
  const { data: tdsSections = [] } = useQuery<TDSSection[]>({
    queryKey: ['finance-tds-sections'],
    queryFn: async () => { const { data } = await api.get('/finance/masters/tds-sections'); return data },
    staleTime: 5 * 60 * 1000,
  })
  const { data: vouchers = [], isLoading } = useQuery<Voucher[]>({
    queryKey: ['finance-vouchers'],
    queryFn: async () => { const { data } = await api.get('/finance/vouchers'); return data },
    staleTime: 60 * 1000,
  })

  // Form state
  const [vType,      setVType]      = useState('Expense')
  const [vDate,      setVDate]      = useState(toIso(new Date()))
  const [billNo,     setBillNo]     = useState('')
  const [billDate,   setBillDate]   = useState('')
  const [partyName,  setPartyName]  = useState('')
  const [partyGstin, setPartyGstin] = useState('')
  const [partyState, setPartyState] = useState('')
  const [supplyType, setSupplyType] = useState<'Intra' | 'Inter'>('Intra')
  const [cgstRate,   setCgstRate]   = useState('9')
  const [igstRate,   setIgstRate]   = useState('18')
  const [applyTds,   setApplyTds]   = useState(false)
  const [tdsSection, setTdsSection] = useState('')
  const [tdsRate,    setTdsRate]    = useState('')
  const [narration,  setNarration]  = useState('')
  const [lines, setLines] = useState([{ expense_head: '', description: '', amount: '', cost_centre: '' }])
  const [saveErr, setSaveErr] = useState('')

  const ledgerNames = useMemo(() => ledgers.map(l => l.name), [ledgers])

  const taxableAmount = useMemo(() =>
    lines.reduce((s, l) => s + (parseFloat(l.amount) || 0), 0), [lines])

  const cgstAmt = useMemo(() =>
    supplyType === 'Intra' ? taxableAmount * (parseFloat(cgstRate) || 0) / 100 : 0, [supplyType, taxableAmount, cgstRate])
  const sgstAmt = cgstAmt
  const igstAmt = useMemo(() =>
    supplyType === 'Inter' ? taxableAmount * (parseFloat(igstRate) || 0) / 100 : 0, [supplyType, taxableAmount, igstRate])
  const totalGst = supplyType === 'Intra' ? cgstAmt + sgstAmt : igstAmt
  const totalAmt = taxableAmount + totalGst

  const tdsAmtComputed = useMemo(() => {
    if (!applyTds && vType !== 'JWO Payment') return 0
    return taxableAmount * (parseFloat(tdsRate) || 0) / 100
  }, [applyTds, vType, taxableAmount, tdsRate])

  const netPayable = totalAmt - tdsAmtComputed

  function handleTdsSectionChange(sec: string) {
    setTdsSection(sec)
    const found = tdsSections.find(t => t.section === sec)
    if (found) setTdsRate(String(found.rate_individual))
  }

  function addLine() {
    setLines(prev => [...prev, { expense_head: '', description: '', amount: '', cost_centre: '' }])
  }
  function removeLine(idx: number) {
    setLines(prev => prev.filter((_, i) => i !== idx))
  }
  function updateLine(idx: number, field: string, value: string) {
    setLines(prev => prev.map((l, i) => i === idx ? { ...l, [field]: value } : l))
  }

  const saveMut = useMutation({
    mutationFn: (body: object) => api.post('/finance/vouchers', body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['finance-vouchers'] })
      setLines([{ expense_head: '', description: '', amount: '', cost_centre: '' }])
      setBillNo(''); setBillDate(''); setPartyName(''); setPartyGstin(''); setPartyState('')
      setNarration(''); setTdsSection(''); setTdsRate(''); setApplyTds(false); setSaveErr('')
    },
    onError: () => setSaveErr('Failed to save voucher.'),
  })

  const delVoucher = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/vouchers/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['finance-vouchers'] }),
  })

  function handleSave() {
    if (!vDate) { setSaveErr('Voucher date is required.'); return }
    if (lines.every(l => !l.expense_head)) { setSaveErr('At least one expense line is required.'); return }
    const showTds = applyTds || vType === 'JWO Payment'
    saveMut.mutate({
      voucher_type: vType,
      voucher_date: vDate,
      bill_no: billNo, bill_date: billDate,
      party_name: partyName, party_gstin: partyGstin, party_state: partyState,
      supply_type: supplyType,
      narration,
      taxable_amount: taxableAmount,
      cgst_amount: cgstAmt,
      sgst_amount: sgstAmt,
      igst_amount: igstAmt,
      tds_section: showTds ? tdsSection : '',
      tds_rate: showTds ? parseFloat(tdsRate) || 0 : 0,
      tds_amount: showTds ? tdsAmtComputed : 0,
      total_amount: totalAmt,
      net_payable: netPayable,
      lines: lines.filter(l => l.expense_head).map(l => ({
        expense_head: l.expense_head,
        description: l.description,
        amount: parseFloat(l.amount) || 0,
        cost_centre: l.cost_centre,
      })),
    })
  }

  const showTdsSection = applyTds || vType === 'JWO Payment'

  return (
    <div className="space-y-5">
      {/* Form */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-4">
        <h3 className="text-sm font-semibold text-gray-700">New Expense Voucher</h3>

        {/* Row 1: type, date, bill no, bill date */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Voucher Type</label>
            <select value={vType} onChange={e => setVType(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
              <option>Expense</option>
              <option>JWO Payment</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Voucher Date *</label>
            <input type="date" value={vDate} onChange={e => setVDate(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Bill No</label>
            <input type="text" value={billNo} onChange={e => setBillNo(e.target.value)} placeholder="Vendor bill no"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Bill Date</label>
            <input type="date" value={billDate} onChange={e => setBillDate(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
        </div>

        {/* Row 2: party */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Party Name</label>
            <input type="text" value={partyName} onChange={e => setPartyName(e.target.value)}
              list="ledger-names-list" placeholder="Vendor / party name"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
            <datalist id="ledger-names-list">
              {ledgerNames.map(n => <option key={n} value={n} />)}
            </datalist>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Party GSTIN</label>
            <input type="text" value={partyGstin} onChange={e => setPartyGstin(e.target.value)} placeholder="15-digit GSTIN"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Party State</label>
            <input type="text" value={partyState} onChange={e => setPartyState(e.target.value)} placeholder="e.g. Maharashtra"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
        </div>

        {/* Supply Type */}
        <div className="flex items-center gap-6">
          <span className="text-xs text-gray-500 font-medium">Supply Type:</span>
          <label className="flex items-center gap-1.5 text-xs cursor-pointer">
            <input type="radio" name="supply_type" value="Intra" checked={supplyType === 'Intra'} onChange={() => setSupplyType('Intra')} />
            Intra-state (CGST + SGST)
          </label>
          <label className="flex items-center gap-1.5 text-xs cursor-pointer">
            <input type="radio" name="supply_type" value="Inter" checked={supplyType === 'Inter'} onChange={() => setSupplyType('Inter')} />
            Inter-state (IGST)
          </label>
        </div>

        {/* Lines */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-medium text-gray-600">Expense Lines</label>
            <button onClick={addLine} className="text-xs text-blue-600 hover:text-blue-800 font-medium">+ Add Line</button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 text-gray-500 uppercase tracking-wide">
                  <th className="px-3 py-2 text-left w-1/3">Expense Head</th>
                  <th className="px-3 py-2 text-left">Description</th>
                  <th className="px-3 py-2 text-right w-28">Amount (₹)</th>
                  <th className="px-3 py-2 text-left w-28">Cost Centre</th>
                  <th className="px-3 py-2 w-8"></th>
                </tr>
              </thead>
              <tbody>
                {lines.map((line, idx) => (
                  <tr key={idx} className="border-t border-gray-100">
                    <td className="px-3 py-1.5">
                      <input type="text" value={line.expense_head} onChange={e => updateLine(idx, 'expense_head', e.target.value)}
                        list="expense-heads-list" placeholder="Expense head"
                        className="w-full border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-300" />
                    </td>
                    <td className="px-3 py-1.5">
                      <input type="text" value={line.description} onChange={e => updateLine(idx, 'description', e.target.value)}
                        placeholder="Details"
                        className="w-full border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-300" />
                    </td>
                    <td className="px-3 py-1.5">
                      <input type="number" value={line.amount} onChange={e => updateLine(idx, 'amount', e.target.value)}
                        placeholder="0" className="w-full border border-gray-200 rounded px-2 py-1 text-right focus:outline-none focus:ring-1 focus:ring-blue-300" />
                    </td>
                    <td className="px-3 py-1.5">
                      <input type="text" value={line.cost_centre} onChange={e => updateLine(idx, 'cost_centre', e.target.value)}
                        placeholder="Optional"
                        className="w-full border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-300" />
                    </td>
                    <td className="px-3 py-1.5 text-center">
                      {lines.length > 1 && (
                        <button onClick={() => removeLine(idx)} className="text-red-400 hover:text-red-600">✕</button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <datalist id="expense-heads-list">
            {ledgerNames.map(n => <option key={n} value={n} />)}
          </datalist>
        </div>

        {/* GST */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 bg-gray-50 rounded-lg p-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Taxable Amount</label>
            <div className="text-sm font-semibold text-gray-800">{fmt(taxableAmount)}</div>
          </div>
          {supplyType === 'Intra' ? (
            <>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500">CGST Rate %</label>
                <input type="number" value={cgstRate} onChange={e => setCgstRate(e.target.value)} placeholder="9"
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 w-20 focus:outline-none focus:ring-1 focus:ring-blue-300" />
                <span className="text-xs text-gray-400">{fmt(cgstAmt)}</span>
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500">SGST Rate %</label>
                <div className="text-xs text-gray-600 py-1.5">{cgstRate}% (same as CGST)</div>
                <span className="text-xs text-gray-400">{fmt(sgstAmt)}</span>
              </div>
            </>
          ) : (
            <div className="flex flex-col gap-1 col-span-2">
              <label className="text-xs text-gray-500">IGST Rate %</label>
              <input type="number" value={igstRate} onChange={e => setIgstRate(e.target.value)} placeholder="18"
                className="text-xs border border-gray-200 rounded px-2 py-1.5 w-20 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              <span className="text-xs text-gray-400">{fmt(igstAmt)}</span>
            </div>
          )}
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Total (with GST)</label>
            <div className="text-sm font-semibold text-gray-800">{fmt(totalAmt)}</div>
          </div>
        </div>

        {/* TDS */}
        {vType !== 'JWO Payment' && (
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input type="checkbox" checked={applyTds} onChange={e => setApplyTds(e.target.checked)} />
            Apply TDS
          </label>
        )}
        {showTdsSection && (
          <div className="grid grid-cols-3 gap-3 bg-amber-50 rounded-lg p-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">TDS Section</label>
              <select value={tdsSection} onChange={e => handleTdsSectionChange(e.target.value)}
                className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
                <option value="">— Select —</option>
                {tdsSections.map(t => (
                  <option key={t.id} value={t.section}>{t.section} – {t.description}</option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">TDS Rate %</label>
              <input type="number" value={tdsRate} onChange={e => setTdsRate(e.target.value)} placeholder="0"
                className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">TDS Amount</label>
              <div className="text-sm font-semibold text-amber-800">{fmt(tdsAmtComputed)}</div>
            </div>
          </div>
        )}

        {/* Narration */}
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500">Narration</label>
          <textarea value={narration} onChange={e => setNarration(e.target.value)} rows={2} placeholder="Narration / notes"
            className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300 resize-none" />
        </div>

        {/* Totals summary */}
        <div className="flex items-center justify-between bg-[#002B5B] text-white rounded-lg px-4 py-3">
          <div className="text-xs space-y-0.5">
            <div>Taxable: <span className="font-semibold">{fmt(taxableAmount)}</span></div>
            <div>GST: <span className="font-semibold">{fmt(totalGst)}</span></div>
            {showTdsSection && <div>TDS: <span className="font-semibold">({fmt(tdsAmtComputed)})</span></div>}
          </div>
          <div className="text-right">
            <div className="text-xs text-blue-200">Net Payable</div>
            <div className="text-xl font-bold">{fmt(netPayable)}</div>
          </div>
        </div>

        {saveErr && <p className="text-xs text-red-600">{saveErr}</p>}
        <button onClick={handleSave} disabled={saveMut.isPending}
          className="px-5 py-2 text-sm font-semibold bg-[#002B5B] text-white rounded-lg hover:bg-[#003875] disabled:opacity-60 transition-colors">
          {saveMut.isPending ? 'Saving…' : 'Save Voucher'}
        </button>
      </div>

      {/* Vouchers Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-100">
          <h3 className="text-sm font-semibold text-gray-700">Saved Vouchers</h3>
        </div>
        {isLoading ? (
          <div className="p-8 text-center text-gray-400 text-sm animate-pulse">Loading…</div>
        ) : vouchers.length === 0 ? (
          <div className="p-8 text-center text-gray-400 text-sm">No vouchers saved yet.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 text-gray-500 uppercase tracking-wide">
                  <th className="px-3 py-2.5 text-left">Date</th>
                  <th className="px-3 py-2.5 text-left">Voucher No</th>
                  <th className="px-3 py-2.5 text-left">Type</th>
                  <th className="px-3 py-2.5 text-left">Party</th>
                  <th className="px-3 py-2.5 text-left">Bill No</th>
                  <th className="px-3 py-2.5 text-right">Taxable</th>
                  <th className="px-3 py-2.5 text-right">CGST</th>
                  <th className="px-3 py-2.5 text-right">SGST</th>
                  <th className="px-3 py-2.5 text-right">IGST</th>
                  <th className="px-3 py-2.5 text-right">TDS</th>
                  <th className="px-3 py-2.5 text-right">Net Payable</th>
                  <th className="px-3 py-2.5"></th>
                </tr>
              </thead>
              <tbody>
                {vouchers.map(v => (
                  <tr key={v.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-3 py-2 text-gray-600">{v.voucher_date}</td>
                    <td className="px-3 py-2 font-mono font-semibold text-[#002B5B]">{v.voucher_no}</td>
                    <td className="px-3 py-2">
                      <span className="bg-blue-50 text-blue-700 px-1.5 py-0.5 rounded">{v.voucher_type}</span>
                    </td>
                    <td className="px-3 py-2 text-gray-600 max-w-[100px] truncate" title={v.party_name}>{v.party_name || '—'}</td>
                    <td className="px-3 py-2 text-gray-500">{v.bill_no || '—'}</td>
                    <td className="px-3 py-2 text-right">{fmt(v.taxable_amount)}</td>
                    <td className="px-3 py-2 text-right text-gray-500">{v.cgst_amount > 0 ? fmt(v.cgst_amount) : '—'}</td>
                    <td className="px-3 py-2 text-right text-gray-500">{v.sgst_amount > 0 ? fmt(v.sgst_amount) : '—'}</td>
                    <td className="px-3 py-2 text-right text-gray-500">{v.igst_amount > 0 ? fmt(v.igst_amount) : '—'}</td>
                    <td className="px-3 py-2 text-right text-amber-700">{v.tds_amount > 0 ? fmt(v.tds_amount) : '—'}</td>
                    <td className="px-3 py-2 text-right font-semibold text-gray-800">{fmt(v.net_payable)}</td>
                    <td className="px-3 py-2 text-right">
                      <button onClick={() => { if (window.confirm('Delete voucher ' + v.voucher_no + '?')) delVoucher.mutate(v.id) }}
                        className="text-red-400 hover:text-red-600">✕</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Masters Tab ──────────────────────────────────────────────────
function MastersTab() {
  const [sub, setSub] = useState<MastersSubTab>('ledger-groups')

  return (
    <div className="space-y-4">
      {/* Sub-tab bar */}
      <div className="flex gap-1 border-b border-gray-200">
        {([
          ['ledger-groups', 'Ledger Groups'],
          ['ledgers', 'Ledgers'],
          ['gst-classifications', 'GST Classifications'],
          ['tds-sections', 'TDS Sections'],
        ] as [MastersSubTab, string][]).map(([id, label]) => (
          <button key={id} onClick={() => setSub(id)}
            className={`px-3 py-2 text-xs font-medium transition-colors rounded-t ${sub === id ? 'border-b-2 border-[#002B5B] text-[#002B5B] bg-blue-50' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {sub === 'ledger-groups'       && <LedgerGroupsSubTab />}
      {sub === 'ledgers'             && <LedgersSubTab />}
      {sub === 'gst-classifications' && <GSTClassificationsSubTab />}
      {sub === 'tds-sections'        && <TDSSectionsSubTab />}
    </div>
  )
}

function LedgerGroupsSubTab() {
  const qc = useQueryClient()
  const { data: groups = [], isLoading } = useQuery<LedgerGroup[]>({
    queryKey: ['finance-ledger-groups'],
    queryFn: async () => { const { data } = await api.get('/finance/masters/ledger-groups'); return data },
  })
  const [name, setName] = useState('')
  const [parent, setParent] = useState('')
  const [nature, setNature] = useState('expense')
  const [err, setErr] = useState('')

  const addMut = useMutation({
    mutationFn: (b: object) => api.post('/finance/masters/ledger-groups', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['finance-ledger-groups'] }); setName(''); setParent(''); setErr('') },
    onError: () => setErr('Failed to save.'),
  })
  const delMut = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/masters/ledger-groups/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['finance-ledger-groups'] }),
  })

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
        <h4 className="text-xs font-semibold text-gray-600 mb-3">Add Ledger Group</h4>
        <div className="flex flex-wrap gap-2 items-end">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Name *</label>
            <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="Group name"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Parent Group</label>
            <input type="text" value={parent} onChange={e => setParent(e.target.value)} placeholder="Optional"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Nature</label>
            <select value={nature} onChange={e => setNature(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
              <option value="income">Income</option>
              <option value="expense">Expense</option>
              <option value="asset">Asset</option>
              <option value="liability">Liability</option>
            </select>
          </div>
          <button onClick={() => { if (!name.trim()) { setErr('Name required'); return }; addMut.mutate({ name, parent_group: parent, nature }) }}
            disabled={addMut.isPending}
            className="px-4 py-1.5 text-xs font-semibold bg-[#002B5B] text-white rounded hover:bg-blue-900 disabled:opacity-60">
            + Add
          </button>
        </div>
        {err && <p className="text-xs text-red-600 mt-1">{err}</p>}
      </div>
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {isLoading ? <div className="p-6 text-center text-gray-400 text-sm animate-pulse">Loading…</div> : (
          <table className="w-full text-sm">
            <thead><tr className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
              <th className="px-4 py-2.5 text-left">Name</th>
              <th className="px-4 py-2.5 text-left">Parent</th>
              <th className="px-4 py-2.5 text-left">Nature</th>
              <th className="px-4 py-2.5"></th>
            </tr></thead>
            <tbody>
              {groups.map(g => (
                <tr key={g.id} className="border-t border-gray-50 hover:bg-gray-50">
                  <td className="px-4 py-2 font-medium text-gray-700">{g.name}</td>
                  <td className="px-4 py-2 text-gray-500">{g.parent_group || '—'}</td>
                  <td className="px-4 py-2">
                    <NatureBadge nature={g.nature} />
                  </td>
                  <td className="px-4 py-2 text-right">
                    <button onClick={() => { if (window.confirm('Delete group?')) delMut.mutate(g.id) }}
                      className="text-red-400 hover:text-red-600 text-xs">✕</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function LedgersSubTab() {
  const qc = useQueryClient()
  const { data: groups = [] } = useQuery<LedgerGroup[]>({
    queryKey: ['finance-ledger-groups'],
    queryFn: async () => { const { data } = await api.get('/finance/masters/ledger-groups'); return data },
  })
  const [filterGroup, setFilterGroup] = useState<string>('')
  const { data: ledgers = [], isLoading } = useQuery<Ledger[]>({
    queryKey: ['finance-ledgers', filterGroup],
    queryFn: async () => {
      const params = filterGroup ? `?group_id=${filterGroup}` : ''
      const { data } = await api.get(`/finance/masters/ledgers${params}`)
      return data
    },
  })

  const [name, setName] = useState('')
  const [groupId, setGroupId] = useState('')
  const [gstin, setGstin] = useState('')
  const [pan, setPan] = useState('')
  const [state, setState] = useState('')
  const [stateCode, setStateCode] = useState('')
  const [address, setAddress] = useState('')
  const [tdsApp, setTdsApp] = useState(false)
  const [tdsSection, setTdsSection] = useState('')
  const [err, setErr] = useState('')

  const addMut = useMutation({
    mutationFn: (b: object) => api.post('/finance/masters/ledgers', b),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['finance-ledgers'] })
      setName(''); setGstin(''); setPan(''); setState(''); setStateCode(''); setAddress(''); setTdsApp(false); setTdsSection(''); setErr('')
    },
    onError: () => setErr('Failed to save.'),
  })
  const delMut = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/masters/ledgers/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['finance-ledgers'] }),
  })

  const selectedGroup = groups.find(g => String(g.id) === groupId)

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
        <h4 className="text-xs font-semibold text-gray-600 mb-3">Add Ledger</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Name *</label>
            <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="Ledger name"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Group</label>
            <select value={groupId} onChange={e => setGroupId(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
              <option value="">— Select —</option>
              {groups.map(g => <option key={g.id} value={String(g.id)}>{g.name}</option>)}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">GSTIN</label>
            <input type="text" value={gstin} onChange={e => setGstin(e.target.value)} placeholder="15-digit GSTIN"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">PAN</label>
            <input type="text" value={pan} onChange={e => setPan(e.target.value)} placeholder="PAN"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">State</label>
            <input type="text" value={state} onChange={e => setState(e.target.value)} placeholder="State"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">State Code</label>
            <input type="text" value={stateCode} onChange={e => setStateCode(e.target.value)} placeholder="e.g. 27"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1 col-span-2">
            <label className="text-xs text-gray-500">Address</label>
            <input type="text" value={address} onChange={e => setAddress(e.target.value)} placeholder="Address"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
        </div>
        <div className="flex items-center gap-4 mb-3">
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input type="checkbox" checked={tdsApp} onChange={e => setTdsApp(e.target.checked)} />
            TDS Applicable
          </label>
          {tdsApp && (
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">TDS Section</label>
              <input type="text" value={tdsSection} onChange={e => setTdsSection(e.target.value)} placeholder="e.g. 194C"
                className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
            </div>
          )}
        </div>
        <button onClick={() => {
          if (!name.trim()) { setErr('Name required'); return }
          addMut.mutate({ name, group_id: groupId ? parseInt(groupId) : null, group_name: selectedGroup?.name || '', gstin, pan, state, state_code: stateCode, address, tds_applicable: tdsApp ? 1 : 0, tds_section: tdsSection })
        }} disabled={addMut.isPending}
          className="px-4 py-1.5 text-xs font-semibold bg-[#002B5B] text-white rounded hover:bg-blue-900 disabled:opacity-60">
          + Add Ledger
        </button>
        {err && <p className="text-xs text-red-600 mt-1">{err}</p>}
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-100 flex items-center gap-3">
          <span className="text-xs text-gray-500">Filter by group:</span>
          <select value={filterGroup} onChange={e => setFilterGroup(e.target.value)}
            className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none">
            <option value="">All Groups</option>
            {groups.map(g => <option key={g.id} value={String(g.id)}>{g.name}</option>)}
          </select>
        </div>
        {isLoading ? <div className="p-6 text-center text-gray-400 text-sm animate-pulse">Loading…</div> : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead><tr className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
                <th className="px-4 py-2.5 text-left">Name</th>
                <th className="px-4 py-2.5 text-left">Group</th>
                <th className="px-4 py-2.5 text-left">GSTIN</th>
                <th className="px-4 py-2.5 text-left">PAN</th>
                <th className="px-4 py-2.5 text-left">State</th>
                <th className="px-4 py-2.5 text-center">TDS?</th>
                <th className="px-4 py-2.5"></th>
              </tr></thead>
              <tbody>
                {ledgers.map(l => (
                  <tr key={l.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">{l.name}</td>
                    <td className="px-4 py-2 text-gray-500">{l.group_name || '—'}</td>
                    <td className="px-4 py-2 text-gray-500 font-mono text-xs">{l.gstin || '—'}</td>
                    <td className="px-4 py-2 text-gray-500 font-mono text-xs">{l.pan || '—'}</td>
                    <td className="px-4 py-2 text-gray-500">{l.state || '—'}</td>
                    <td className="px-4 py-2 text-center">
                      {l.tds_applicable ? <span className="text-xs bg-amber-50 text-amber-700 px-1.5 py-0.5 rounded">Yes ({l.tds_section})</span> : '—'}
                    </td>
                    <td className="px-4 py-2 text-right">
                      <button onClick={() => { if (window.confirm('Delete ledger?')) delMut.mutate(l.id) }}
                        className="text-red-400 hover:text-red-600 text-xs">✕</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function GSTClassificationsSubTab() {
  const qc = useQueryClient()
  const { data: items = [], isLoading } = useQuery<GSTClassification[]>({
    queryKey: ['finance-gst-classifications'],
    queryFn: async () => { const { data } = await api.get('/finance/masters/gst-classifications'); return data },
  })
  const [name, setName] = useState('')
  const [hsn, setHsn] = useState('')
  const [rate, setRate] = useState('')
  const [type, setType] = useState('Goods')
  const [err, setErr] = useState('')

  const addMut = useMutation({
    mutationFn: (b: object) => api.post('/finance/masters/gst-classifications', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['finance-gst-classifications'] }); setName(''); setHsn(''); setRate(''); setErr('') },
    onError: () => setErr('Failed to save.'),
  })
  const delMut = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/masters/gst-classifications/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['finance-gst-classifications'] }),
  })

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
        <h4 className="text-xs font-semibold text-gray-600 mb-3">Add GST Classification</h4>
        <div className="flex flex-wrap gap-2 items-end">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Name *</label>
            <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="e.g. 18% Goods"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">HSN/SAC</label>
            <input type="text" value={hsn} onChange={e => setHsn(e.target.value)} placeholder="HSN or SAC code"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">GST Rate %</label>
            <input type="number" value={rate} onChange={e => setRate(e.target.value)} placeholder="18"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 w-20 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Type</label>
            <select value={type} onChange={e => setType(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
              <option>Goods</option>
              <option>Services</option>
            </select>
          </div>
          <button onClick={() => { if (!name.trim()) { setErr('Name required'); return }; addMut.mutate({ name, hsn_sac: hsn, gst_rate: parseFloat(rate) || 0, type }) }}
            disabled={addMut.isPending}
            className="px-4 py-1.5 text-xs font-semibold bg-[#002B5B] text-white rounded hover:bg-blue-900 disabled:opacity-60">
            + Add
          </button>
        </div>
        {err && <p className="text-xs text-red-600 mt-1">{err}</p>}
      </div>
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {isLoading ? <div className="p-6 text-center text-gray-400 text-sm animate-pulse">Loading…</div> : (
          <table className="w-full text-sm">
            <thead><tr className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
              <th className="px-4 py-2.5 text-left">Name</th>
              <th className="px-4 py-2.5 text-left">HSN/SAC</th>
              <th className="px-4 py-2.5 text-right">Rate %</th>
              <th className="px-4 py-2.5 text-left">Type</th>
              <th className="px-4 py-2.5"></th>
            </tr></thead>
            <tbody>
              {items.map(g => (
                <tr key={g.id} className="border-t border-gray-50 hover:bg-gray-50">
                  <td className="px-4 py-2 font-medium text-gray-700">{g.name}</td>
                  <td className="px-4 py-2 text-gray-500 font-mono text-xs">{g.hsn_sac || '—'}</td>
                  <td className="px-4 py-2 text-right font-semibold text-gray-700">{g.gst_rate}%</td>
                  <td className="px-4 py-2 text-gray-500">{g.type}</td>
                  <td className="px-4 py-2 text-right">
                    <button onClick={() => { if (window.confirm('Delete?')) delMut.mutate(g.id) }}
                      className="text-red-400 hover:text-red-600 text-xs">✕</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function TDSSectionsSubTab() {
  const qc = useQueryClient()
  const { data: sections = [], isLoading } = useQuery<TDSSection[]>({
    queryKey: ['finance-tds-sections'],
    queryFn: async () => { const { data } = await api.get('/finance/masters/tds-sections'); return data },
  })
  const [section, setSection] = useState('')
  const [desc, setDesc] = useState('')
  const [rateInd, setRateInd] = useState('')
  const [rateCo, setRateCo] = useState('')
  const [threshold, setThreshold] = useState('')
  const [err, setErr] = useState('')

  const addMut = useMutation({
    mutationFn: (b: object) => api.post('/finance/masters/tds-sections', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['finance-tds-sections'] }); setSection(''); setDesc(''); setRateInd(''); setRateCo(''); setThreshold(''); setErr('') },
    onError: () => setErr('Failed to save.'),
  })
  const delMut = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/masters/tds-sections/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['finance-tds-sections'] }),
  })

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
        <h4 className="text-xs font-semibold text-gray-600 mb-3">Add TDS Section</h4>
        <div className="flex flex-wrap gap-2 items-end">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Section *</label>
            <input type="text" value={section} onChange={e => setSection(e.target.value)} placeholder="e.g. 194C"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 w-24 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1 flex-1 min-w-[140px]">
            <label className="text-xs text-gray-500">Description</label>
            <input type="text" value={desc} onChange={e => setDesc(e.target.value)} placeholder="e.g. Contractor"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Indiv Rate %</label>
            <input type="number" value={rateInd} onChange={e => setRateInd(e.target.value)} placeholder="1"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 w-20 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Company Rate %</label>
            <input type="number" value={rateCo} onChange={e => setRateCo(e.target.value)} placeholder="2"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 w-20 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Threshold (₹)</label>
            <input type="number" value={threshold} onChange={e => setThreshold(e.target.value)} placeholder="30000"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 w-24 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <button onClick={() => { if (!section.trim()) { setErr('Section required'); return }; addMut.mutate({ section, description: desc, rate_individual: parseFloat(rateInd) || 0, rate_company: parseFloat(rateCo) || 0, threshold: parseFloat(threshold) || 0 }) }}
            disabled={addMut.isPending}
            className="px-4 py-1.5 text-xs font-semibold bg-[#002B5B] text-white rounded hover:bg-blue-900 disabled:opacity-60">
            + Add
          </button>
        </div>
        {err && <p className="text-xs text-red-600 mt-1">{err}</p>}
      </div>
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {isLoading ? <div className="p-6 text-center text-gray-400 text-sm animate-pulse">Loading…</div> : (
          <table className="w-full text-sm">
            <thead><tr className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wide">
              <th className="px-4 py-2.5 text-left">Section</th>
              <th className="px-4 py-2.5 text-left">Description</th>
              <th className="px-4 py-2.5 text-right">Indiv %</th>
              <th className="px-4 py-2.5 text-right">Company %</th>
              <th className="px-4 py-2.5 text-right">Threshold</th>
              <th className="px-4 py-2.5"></th>
            </tr></thead>
            <tbody>
              {sections.map(s => (
                <tr key={s.id} className="border-t border-gray-50 hover:bg-gray-50">
                  <td className="px-4 py-2 font-semibold text-[#002B5B] font-mono">{s.section}</td>
                  <td className="px-4 py-2 text-gray-600">{s.description || '—'}</td>
                  <td className="px-4 py-2 text-right text-gray-700">{s.rate_individual}%</td>
                  <td className="px-4 py-2 text-right text-gray-700">{s.rate_company}%</td>
                  <td className="px-4 py-2 text-right text-gray-500">{fmt(s.threshold)}</td>
                  <td className="px-4 py-2 text-right">
                    <button onClick={() => { if (window.confirm('Delete section?')) delMut.mutate(s.id) }}
                      className="text-red-400 hover:text-red-600 text-xs">✕</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

// ── Sales Uploads Tab ─────────────────────────────────────────────
function SalesUploadsTab() {
  const qc = useQueryClient()
  const [filterPlatform, setFilterPlatform] = useState('')
  const [filterPeriod, setFilterPeriod] = useState('')

  const { data: uploads = [], isLoading } = useQuery<SalesUpload[]>({
    queryKey: ['finance-sales-uploads', filterPlatform, filterPeriod],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filterPlatform) params.set('platform', filterPlatform)
      if (filterPeriod) params.set('period', filterPeriod)
      const { data } = await api.get(`/finance/sales-uploads?${params}`)
      return data
    },
  })

  const [platform, setPlatform]   = useState('Amazon')
  const [period, setPeriod]       = useState('')
  const [filename, setFilename]   = useState('')
  const [revenue, setRevenue]     = useState('')
  const [orders, setOrders]       = useState('')
  const [returns, setReturns]     = useState('')
  const [netRev, setNetRev]       = useState('')
  const [uploadedBy, setUploadedBy] = useState('')
  const [notes, setNotes]         = useState('')
  const [err, setErr]             = useState('')

  const addMut = useMutation({
    mutationFn: (b: object) => api.post('/finance/sales-uploads', b),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['finance-sales-uploads'] })
      setFilename(''); setRevenue(''); setOrders(''); setReturns(''); setNetRev(''); setUploadedBy(''); setNotes(''); setErr('')
    },
    onError: () => setErr('Failed to save.'),
  })
  const delMut = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/sales-uploads/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['finance-sales-uploads'] }),
  })

  function handleSave() {
    if (!period) { setErr('Period is required.'); return }
    addMut.mutate({
      platform, period, filename,
      total_revenue: parseFloat(revenue) || 0,
      total_orders: parseInt(orders) || 0,
      total_returns: parseFloat(returns) || 0,
      net_revenue: parseFloat(netRev) || 0,
      uploaded_by: uploadedBy,
      upload_notes: notes,
    })
  }

  return (
    <div className="space-y-5">
      {/* Info Banner */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 flex gap-2 items-start">
        <span className="text-amber-500 mt-0.5">🔒</span>
        <p className="text-xs text-amber-700">Data uploaded here is locked. Only the Finance team can remove entries.</p>
      </div>

      {/* Upload Form */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4 space-y-3">
        <h3 className="text-sm font-semibold text-gray-700">Add Sales Upload Record</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Platform *</label>
            <select value={platform} onChange={e => setPlatform(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
              {SALES_PLATFORMS.map(p => <option key={p}>{p}</option>)}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Period (Month) *</label>
            <input type="month" value={period} onChange={e => setPeriod(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Filename</label>
            <input type="text" value={filename} onChange={e => setFilename(e.target.value)} placeholder="e.g. amazon_jan25.xlsx"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Uploaded By</label>
            <input type="text" value={uploadedBy} onChange={e => setUploadedBy(e.target.value)} placeholder="Name"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Total Revenue (₹)</label>
            <input type="number" value={revenue} onChange={e => setRevenue(e.target.value)} placeholder="0"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Total Orders</label>
            <input type="number" value={orders} onChange={e => setOrders(e.target.value)} placeholder="0"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Total Returns (₹)</label>
            <input type="number" value={returns} onChange={e => setReturns(e.target.value)} placeholder="0"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Net Revenue (₹)</label>
            <input type="number" value={netRev} onChange={e => setNetRev(e.target.value)} placeholder="0"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500">Notes</label>
          <input type="text" value={notes} onChange={e => setNotes(e.target.value)} placeholder="Optional notes"
            className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
        </div>
        {err && <p className="text-xs text-red-600">{err}</p>}
        <button onClick={handleSave} disabled={addMut.isPending}
          className="px-5 py-2 text-xs font-semibold bg-[#002B5B] text-white rounded-lg hover:bg-[#003875] disabled:opacity-60 transition-colors">
          {addMut.isPending ? 'Saving…' : 'Save Record'}
        </button>
      </div>

      {/* Filter */}
      <div className="bg-white rounded-xl border border-gray-200 px-4 py-3 shadow-sm flex flex-wrap items-center gap-3">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Filter</span>
        <select value={filterPlatform} onChange={e => setFilterPlatform(e.target.value)}
          className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none">
          <option value="">All Platforms</option>
          {SALES_PLATFORMS.map(p => <option key={p}>{p}</option>)}
        </select>
        <input type="month" value={filterPeriod} onChange={e => setFilterPeriod(e.target.value)}
          className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none" />
        {(filterPlatform || filterPeriod) && (
          <button onClick={() => { setFilterPlatform(''); setFilterPeriod('') }}
            className="text-xs text-gray-400 hover:text-gray-600">Clear</button>
        )}
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {isLoading ? (
          <div className="p-8 text-center text-gray-400 text-sm animate-pulse">Loading…</div>
        ) : uploads.length === 0 ? (
          <div className="p-8 text-center text-gray-400 text-sm">No sales uploads recorded yet.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 text-gray-500 uppercase tracking-wide">
                  <th className="px-3 py-2.5 text-left">Platform</th>
                  <th className="px-3 py-2.5 text-left">Period</th>
                  <th className="px-3 py-2.5 text-left">Filename</th>
                  <th className="px-3 py-2.5 text-right">Revenue</th>
                  <th className="px-3 py-2.5 text-right">Orders</th>
                  <th className="px-3 py-2.5 text-right">Returns</th>
                  <th className="px-3 py-2.5 text-right">Net Revenue</th>
                  <th className="px-3 py-2.5 text-left">Uploaded By</th>
                  <th className="px-3 py-2.5 text-left">Date</th>
                  <th className="px-3 py-2.5"></th>
                </tr>
              </thead>
              <tbody>
                {uploads.map(u => (
                  <tr key={u.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-3 py-2">
                      <span className="bg-blue-50 text-blue-700 px-1.5 py-0.5 rounded font-medium">{u.platform}</span>
                    </td>
                    <td className="px-3 py-2 font-medium text-gray-700">{u.period}</td>
                    <td className="px-3 py-2 text-gray-500 max-w-[100px] truncate" title={u.filename}>{u.filename || '—'}</td>
                    <td className="px-3 py-2 text-right text-gray-700">{fmt(u.total_revenue)}</td>
                    <td className="px-3 py-2 text-right text-gray-600">{u.total_orders.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right text-red-600">{u.total_returns > 0 ? fmt(u.total_returns) : '—'}</td>
                    <td className="px-3 py-2 text-right font-semibold text-green-700">{fmt(u.net_revenue)}</td>
                    <td className="px-3 py-2 text-gray-500">{u.uploaded_by || '—'}</td>
                    <td className="px-3 py-2 text-gray-400">{u.created_at?.slice(0, 10)}</td>
                    <td className="px-3 py-2 text-right">
                      <button onClick={() => { if (window.confirm('Delete this sales upload record?')) delMut.mutate(u.id) }}
                        className="text-red-400 hover:text-red-600">✕</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Helpers ───────────────────────────────────────────────────────
function NatureBadge({ nature }: { nature: string }) {
  const cls = {
    income:    'bg-green-50 text-green-700',
    expense:   'bg-red-50 text-red-700',
    asset:     'bg-blue-50 text-blue-700',
    liability: 'bg-purple-50 text-purple-700',
  }[nature] ?? 'bg-gray-100 text-gray-600'
  return <span className={`text-xs px-1.5 py-0.5 rounded capitalize ${cls}`}>{nature}</span>
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
