import React, { useState, useMemo, useRef, useEffect } from 'react'
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
  id:                    number
  name:                  string
  group_id:              number | null
  group_name:            string
  gstin:                 string
  pan:                   string
  state:                 string
  state_code:            string
  address:               string
  tds_applicable:        number
  tds_section:           string
  is_active:             number
  alias:                 string
  credit_period:         number
  maintain_bill_by_bill: number
  is_tcs_applicable:     number
  country:               string
  pincode:               string
  registration_type:     string
  bank_name:             string
  bank_account:          string
  bank_ifsc:             string
  opening_balance:       number
}
interface VoucherType {
  id:               number
  name:             string
  voucher_category: string
  abbreviation:     string
  is_active:        number
  allow_narration:  number
  numbering_method: string
  created_at:       string
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
const SALES_PLATFORMS = ['Amazon', 'Myntra', 'Meesho', 'Flipkart', 'Snapdeal', 'All Platforms'] as const
void SALES_PLATFORMS

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

type FinanceTab = 'dashboard' | 'daybook' | 'vouchers' | 'gstr' | 'pl' | 'gst' | 'expenses' | 'revenue' | 'masters' | 'sales-uploads' | 'coa' | 'trial-balance'

// ── Chart of Accounts types ───────────────────────────────────────
interface CoALedger {
  id: number; name: string; group_id: number | null; group_name: string; opening_balance: number
}
interface CoAGroup {
  id: number; name: string; parent_group: string; nature: string
  children: CoAGroup[]
  ledgers: CoALedger[]
}

// ── Trial Balance types ───────────────────────────────────────────
interface TBRow {
  ledger: string; group: string; nature: string
  opening_balance: number
  period_dr: number; period_cr: number
  debit: number; credit: number; closing: number
}
type MastersSubTab = 'ledger-groups' | 'ledgers' | 'gst-classifications' | 'tds-sections' | 'voucher-types'

const VOUCHER_COLORS: Record<string, string> = {
  'Expense':          'bg-red-50 text-red-700',
  'JWO Payment':      'bg-orange-50 text-orange-700',
  'Payment':          'bg-red-100 text-red-800',
  'Receipt':          'bg-green-100 text-green-800',
  'Journal':          'bg-purple-50 text-purple-700',
  'Contra':           'bg-blue-50 text-blue-700',
  'Purchase Invoice': 'bg-amber-50 text-amber-700',
  'Sales Invoice':    'bg-emerald-50 text-emerald-700',
}

interface DaybookVoucher {
  id: number; voucher_no: string; voucher_date: string; voucher_type: string
  party_name: string; narration: string; taxable_amount: number
  cgst_amount: number; sgst_amount: number; igst_amount: number
  tds_amount: number; total_amount: number; net_payable: number
  payment_mode: string; bank_ledger: string; lines: VoucherLine[]
}

interface GSTR3BData {
  outward:    { taxable: number; cgst: number; sgst: number; igst: number; total: number }
  inward_itc: { taxable: number; cgst: number; sgst: number; igst: number; total: number }
  net_cgst: number; net_sgst: number; net_igst: number; net_total: number
  breakdown: { voucher_no: string; voucher_date: string; voucher_type: string; party_name: string; taxable_amount: number; cgst_amount: number; sgst_amount: number; igst_amount: number; total_amount: number }[]
}

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

  const [activeTab,    setActiveTab]    = useState<FinanceTab>('dashboard')
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
          ['dashboard','Dashboard'],
          ['daybook','Day Book'],
          ['vouchers','Vouchers'],
          ['gstr','GSTR3B'],
          ['pl','P&L'],
          ['gst','GST Summary'],
          ['expenses','Expenses'],
          ['masters','Masters'],
          ['coa','Chart of Accounts'],
          ['trial-balance','Trial Balance'],
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

      {/* ── Tab: Dashboard ── */}
      {activeTab === 'dashboard' && <DashboardTab onNavigate={setActiveTab} />}

      {/* ── Tab: Day Book ── */}
      {activeTab === 'daybook' && <DaybookTab />}

      {/* ── Tab: GSTR3B ── */}
      {activeTab === 'gstr' && <GSTR3BTab />}

      {/* ── Tab: Vouchers ── */}
      {activeTab === 'vouchers' && <VouchersTab />}

      {/* ── Tab: Masters ── */}
      {activeTab === 'masters' && <MastersTab />}

      {/* ── Tab: Sales Uploads ── */}
      {activeTab === 'sales-uploads' && <SalesUploadsTab />}

      {/* ── Tab: Chart of Accounts ── */}
      {activeTab === 'coa' && <ChartOfAccountsTab />}

      {/* ── Tab: Trial Balance ── */}
      {activeTab === 'trial-balance' && <TrialBalanceTab />}
    </div>
  )
}

// ── Dashboard Tab ────────────────────────────────────────────────
function DashboardTab({ onNavigate }: { onNavigate: (tab: FinanceTab) => void }) {
  const todayStr = toIso(new Date())

  const { data: todayVouchers = [], isLoading: loadToday } = useQuery<DaybookVoucher[]>({
    queryKey: ['finance-daybook-today'],
    queryFn: async () => { const { data } = await api.get(`/finance/daybook?date=${todayStr}`); return data },
    staleTime: 30 * 1000,
  })

  const { data: monthExpenses } = useQuery<Expense[]>({
    queryKey: ['finance-expenses-month'],
    queryFn: async () => {
      const now = new Date()
      const start = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-01`
      const { data } = await api.get(`/finance/expenses?start_date=${start}&end_date=${todayStr}`)
      return data
    },
    staleTime: 60 * 1000,
  })

  const { data: gstr3b } = useQuery<GSTR3BData>({
    queryKey: ['finance-gstr3b-current'],
    queryFn: async () => {
      const now = new Date()
      const y = now.getFullYear()
      const m = String(now.getMonth() + 1).padStart(2, '0')
      const { data } = await api.get(`/finance/gstr3b?start_date=${y}-${m}-01&end_date=${y}-${m}-31`)
      return data
    },
    staleTime: 60 * 1000,
  })

  const { data: ledgerBalances } = useQuery<{ debtors: { name: string; balance: number }[]; creditors: { name: string; balance: number }[] }>({
    queryKey: ['finance-ledger-balances'],
    queryFn: async () => { const { data } = await api.get('/finance/ledger-balances'); return data },
    staleTime: 2 * 60 * 1000,
  })

  const todayCount = todayVouchers.length
  const todayTotal = todayVouchers.reduce((s, v) => s + v.net_payable, 0)
  const monthExpTotal = (monthExpenses ?? []).reduce((s, e) => s + e.amount + e.gst_amount, 0)
  const gstPayable = gstr3b?.net_total ?? 0

  const QUICK_ACTIONS: { label: string; vtype?: string; icon: string }[] = [
    { label: 'New Expense', icon: '📋' },
    { label: 'New Payment', icon: '💳' },
    { label: 'New Receipt', icon: '🧾' },
    { label: 'New Journal', icon: '📒' },
  ]

  return (
    <div className="space-y-5">
      {/* KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
          <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">Today's Vouchers</p>
          {loadToday ? (
            <div className="h-7 bg-gray-100 rounded animate-pulse mt-2" />
          ) : (
            <p className="text-2xl font-bold text-[#002B5B] mt-1">{todayCount}</p>
          )}
          <p className="text-xs text-gray-400 mt-0.5">{todayStr}</p>
        </div>
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
          <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">Today's Total</p>
          {loadToday ? (
            <div className="h-7 bg-gray-100 rounded animate-pulse mt-2" />
          ) : (
            <p className="text-2xl font-bold text-gray-800 mt-1">{fmt(todayTotal)}</p>
          )}
          <p className="text-xs text-gray-400 mt-0.5">net payable</p>
        </div>
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
          <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">This Month Expenses</p>
          <p className="text-2xl font-bold text-red-600 mt-1">{fmt(monthExpTotal)}</p>
          <p className="text-xs text-gray-400 mt-0.5">incl. GST</p>
        </div>
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
          <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">GST Payable</p>
          <p className={`text-2xl font-bold mt-1 ${gstPayable > 0 ? 'text-orange-600' : 'text-green-600'}`}>{fmt(gstPayable)}</p>
          <p className="text-xs text-gray-400 mt-0.5">current month net</p>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Quick Actions</h3>
        <div className="flex flex-wrap gap-2">
          {QUICK_ACTIONS.map(({ label, icon }) => (
            <button key={label} onClick={() => onNavigate('vouchers')}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#002B5B] text-white rounded-lg hover:bg-[#003875] transition-colors">
              <span>{icon}</span>
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Today's Day Book */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Today's Day Book</h3>
            <button onClick={() => onNavigate('daybook')} className="text-xs text-blue-600 hover:underline">View Full →</button>
          </div>
          {loadToday ? (
            <div className="p-6 text-center text-gray-400 text-sm animate-pulse">Loading…</div>
          ) : todayVouchers.length === 0 ? (
            <div className="p-6 text-center text-gray-400 text-sm">No vouchers today.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-gray-50 text-gray-500 uppercase tracking-wide">
                    <th className="px-3 py-2 text-left">Voucher No</th>
                    <th className="px-3 py-2 text-left">Type</th>
                    <th className="px-3 py-2 text-left">Party</th>
                    <th className="px-3 py-2 text-right">Net</th>
                  </tr>
                </thead>
                <tbody>
                  {todayVouchers.slice(0, 8).map(v => (
                    <tr key={v.id} className="border-t border-gray-50 hover:bg-gray-50">
                      <td className="px-3 py-1.5 font-mono font-semibold text-[#002B5B]">{v.voucher_no}</td>
                      <td className="px-3 py-1.5">
                        <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${VOUCHER_COLORS[v.voucher_type] ?? 'bg-gray-100 text-gray-600'}`}>
                          {v.voucher_type}
                        </span>
                      </td>
                      <td className="px-3 py-1.5 text-gray-600 max-w-[80px] truncate">{v.party_name || '—'}</td>
                      <td className="px-3 py-1.5 text-right font-semibold text-gray-800">{fmt(v.net_payable)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Outstanding Summary */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-100">
            <h3 className="text-sm font-semibold text-gray-700">Outstanding Summary</h3>
          </div>
          {!ledgerBalances ? (
            <div className="p-6 text-center text-gray-400 text-sm animate-pulse">Loading…</div>
          ) : (
            <div className="grid grid-cols-2 divide-x divide-gray-100">
              <div className="p-3">
                <p className="text-[10px] font-bold text-green-700 uppercase tracking-widest mb-2">Top Debtors</p>
                {(ledgerBalances.debtors ?? []).slice(0, 5).length === 0 ? (
                  <p className="text-xs text-gray-400">None</p>
                ) : (
                  (ledgerBalances.debtors ?? []).slice(0, 5).map((d, i) => (
                    <div key={i} className="flex justify-between items-center py-0.5">
                      <span className="text-xs text-gray-600 truncate max-w-[80px]" title={d.name}>{d.name}</span>
                      <span className="text-xs font-semibold text-green-700">{fmt(d.balance)}</span>
                    </div>
                  ))
                )}
              </div>
              <div className="p-3">
                <p className="text-[10px] font-bold text-red-700 uppercase tracking-widest mb-2">Top Creditors</p>
                {(ledgerBalances.creditors ?? []).slice(0, 5).length === 0 ? (
                  <p className="text-xs text-gray-400">None</p>
                ) : (
                  (ledgerBalances.creditors ?? []).slice(0, 5).map((c, i) => (
                    <div key={i} className="flex justify-between items-center py-0.5">
                      <span className="text-xs text-gray-600 truncate max-w-[80px]" title={c.name}>{c.name}</span>
                      <span className="text-xs font-semibold text-red-600">{fmt(c.balance)}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Day Book Tab ──────────────────────────────────────────────────
function DaybookTab() {
  const qc = useQueryClient()
  const [dateStr, setDateStr] = useState(toIso(new Date()))

  function prevDay() {
    const d = new Date(dateStr)
    d.setDate(d.getDate() - 1)
    setDateStr(toIso(d))
  }
  function nextDay() {
    const d = new Date(dateStr)
    d.setDate(d.getDate() + 1)
    setDateStr(toIso(d))
  }

  const { data: vouchers = [], isLoading } = useQuery<DaybookVoucher[]>({
    queryKey: ['finance-daybook', dateStr],
    queryFn: async () => { const { data } = await api.get(`/finance/daybook?date=${dateStr}`); return data },
    staleTime: 30 * 1000,
  })

  const delMut = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/vouchers/${id}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['finance-daybook', dateStr] })
      qc.invalidateQueries({ queryKey: ['finance-daybook-today'] })
    },
  })

  const totalTaxable = vouchers.reduce((s, v) => s + v.taxable_amount, 0)
  const totalGst = vouchers.reduce((s, v) => s + v.cgst_amount + v.sgst_amount + v.igst_amount, 0)
  const totalTds = vouchers.reduce((s, v) => s + v.tds_amount, 0)
  const totalNet = vouchers.reduce((s, v) => s + v.net_payable, 0)

  return (
    <div className="space-y-4">
      {/* Date navigator */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm px-4 py-3 flex items-center gap-3">
        <button onClick={prevDay} className="px-3 py-1.5 text-xs font-medium bg-gray-100 text-gray-600 rounded hover:bg-gray-200 transition-colors">← Prev</button>
        <input type="date" value={dateStr} max={toIso(new Date())} onChange={e => setDateStr(e.target.value)}
          className="text-sm font-semibold border border-gray-200 rounded px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
        <button onClick={nextDay} className="px-3 py-1.5 text-xs font-medium bg-gray-100 text-gray-600 rounded hover:bg-gray-200 transition-colors">Next →</button>
        <span className="text-xs text-gray-400 ml-2">{vouchers.length} voucher{vouchers.length !== 1 ? 's' : ''}</span>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-100">
          <h3 className="text-sm font-semibold text-gray-700">Day Book — {dateStr}</h3>
        </div>
        {isLoading ? (
          <div className="p-8 text-center text-gray-400 text-sm animate-pulse">Loading…</div>
        ) : vouchers.length === 0 ? (
          <div className="p-8 text-center text-gray-400 text-sm">No vouchers for this date.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 text-gray-500 uppercase tracking-wide">
                  <th className="px-3 py-2.5 text-left">Date</th>
                  <th className="px-3 py-2.5 text-left">Voucher No</th>
                  <th className="px-3 py-2.5 text-left">Type</th>
                  <th className="px-3 py-2.5 text-left">Party</th>
                  <th className="px-3 py-2.5 text-right">Taxable</th>
                  <th className="px-3 py-2.5 text-right">CGST</th>
                  <th className="px-3 py-2.5 text-right">SGST</th>
                  <th className="px-3 py-2.5 text-right">IGST</th>
                  <th className="px-3 py-2.5 text-right">TDS</th>
                  <th className="px-3 py-2.5 text-right">Total</th>
                  <th className="px-3 py-2.5 text-right">Net Payable</th>
                  <th className="px-3 py-2.5"></th>
                </tr>
              </thead>
              <tbody>
                {vouchers.map(v => (
                  <tr key={v.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-3 py-2 text-gray-500">{v.voucher_date}</td>
                    <td className="px-3 py-2 font-mono font-semibold text-[#002B5B]">{v.voucher_no}</td>
                    <td className="px-3 py-2">
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${VOUCHER_COLORS[v.voucher_type] ?? 'bg-gray-100 text-gray-600'}`}>
                        {v.voucher_type}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-gray-600 max-w-[100px] truncate" title={v.party_name}>{v.party_name || '—'}</td>
                    <td className="px-3 py-2 text-right">{fmt(v.taxable_amount)}</td>
                    <td className="px-3 py-2 text-right text-gray-500">{v.cgst_amount > 0 ? fmt(v.cgst_amount) : '—'}</td>
                    <td className="px-3 py-2 text-right text-gray-500">{v.sgst_amount > 0 ? fmt(v.sgst_amount) : '—'}</td>
                    <td className="px-3 py-2 text-right text-gray-500">{v.igst_amount > 0 ? fmt(v.igst_amount) : '—'}</td>
                    <td className="px-3 py-2 text-right text-amber-700">{v.tds_amount > 0 ? fmt(v.tds_amount) : '—'}</td>
                    <td className="px-3 py-2 text-right text-gray-700">{fmt(v.total_amount)}</td>
                    <td className="px-3 py-2 text-right font-semibold text-gray-800">{fmt(v.net_payable)}</td>
                    <td className="px-3 py-2 text-right">
                      <button onClick={() => { if (window.confirm('Delete voucher ' + v.voucher_no + '?')) delMut.mutate(v.id) }}
                        className="text-red-400 hover:text-red-600">✕</button>
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="bg-[#002B5B] text-white text-xs font-semibold">
                  <td className="px-3 py-2.5" colSpan={4}>Total ({vouchers.length} vouchers)</td>
                  <td className="px-3 py-2.5 text-right">{fmt(totalTaxable)}</td>
                  <td className="px-3 py-2.5 text-right" colSpan={3}>{fmt(totalGst)} GST</td>
                  <td className="px-3 py-2.5 text-right">{totalTds > 0 ? fmt(totalTds) : '—'}</td>
                  <td className="px-3 py-2.5 text-right">{fmt(totalTaxable + totalGst)}</td>
                  <td className="px-3 py-2.5 text-right">{fmt(totalNet)}</td>
                  <td />
                </tr>
              </tfoot>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

// ── GSTR3B Tab ───────────────────────────────────────────────────
function GSTR3BTab() {
  const now = new Date()
  const defaultMonth = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`
  const [month, setMonth] = useState(defaultMonth)

  const [y, m] = month.split('-')
  const startDate = `${y}-${m}-01`
  const endDate = `${y}-${m}-31`

  const { data: gstr, isLoading } = useQuery<GSTR3BData>({
    queryKey: ['finance-gstr3b', month],
    queryFn: async () => { const { data } = await api.get(`/finance/gstr3b?start_date=${startDate}&end_date=${endDate}`); return data },
    staleTime: 60 * 1000,
  })

  return (
    <div className="space-y-4">
      {/* Month picker */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm px-4 py-3 flex items-center gap-3">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Return Period</span>
        <input type="month" value={month} onChange={e => setMonth(e.target.value)}
          className="text-sm font-semibold border border-gray-200 rounded px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
        <span className="text-xs text-gray-400">{startDate} to {endDate}</span>
      </div>

      {isLoading ? (
        <div className="p-8 text-center text-gray-400 text-sm animate-pulse">Loading GSTR3B data…</div>
      ) : !gstr ? (
        <div className="p-8 text-center text-gray-400 text-sm">No data found for this period.</div>
      ) : (
        <div className="space-y-4">
          {/* GSTR3B Form */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-5 py-3 border-b border-gray-100 bg-[#002B5B] text-white">
              <h3 className="text-sm font-semibold">GSTR-3B — {month}</h3>
              <p className="text-xs text-blue-200 mt-0.5">Monthly Summary Return</p>
            </div>

            {/* Section 3.1 */}
            <div className="px-5 py-4 border-b border-gray-100">
              <h4 className="text-xs font-bold text-blue-800 uppercase tracking-widest mb-3">3.1 — Outward Supplies (Output Tax)</h4>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="bg-blue-50 rounded-lg p-3 text-center">
                  <p className="text-[10px] text-blue-500 font-semibold uppercase">Taxable</p>
                  <p className="text-base font-bold text-blue-800 mt-1">{fmt(gstr.outward.taxable)}</p>
                </div>
                <div className="bg-blue-50 rounded-lg p-3 text-center">
                  <p className="text-[10px] text-blue-500 font-semibold uppercase">CGST</p>
                  <p className="text-base font-bold text-blue-800 mt-1">{fmt(gstr.outward.cgst)}</p>
                </div>
                <div className="bg-blue-50 rounded-lg p-3 text-center">
                  <p className="text-[10px] text-blue-500 font-semibold uppercase">SGST</p>
                  <p className="text-base font-bold text-blue-800 mt-1">{fmt(gstr.outward.sgst)}</p>
                </div>
                <div className="bg-blue-50 rounded-lg p-3 text-center">
                  <p className="text-[10px] text-blue-500 font-semibold uppercase">IGST</p>
                  <p className="text-base font-bold text-blue-800 mt-1">{fmt(gstr.outward.igst)}</p>
                </div>
                <div className="bg-blue-100 rounded-lg p-3 text-center border border-blue-200">
                  <p className="text-[10px] text-blue-700 font-bold uppercase">Total Output</p>
                  <p className="text-lg font-bold text-blue-900 mt-1">{fmt(gstr.outward.total)}</p>
                </div>
              </div>
            </div>

            {/* Section 4 */}
            <div className="px-5 py-4 border-b border-gray-100">
              <h4 className="text-xs font-bold text-green-800 uppercase tracking-widest mb-3">4 — Eligible Input Tax Credit (ITC)</h4>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="bg-green-50 rounded-lg p-3 text-center">
                  <p className="text-[10px] text-green-500 font-semibold uppercase">Taxable</p>
                  <p className="text-base font-bold text-green-800 mt-1">{fmt(gstr.inward_itc.taxable)}</p>
                </div>
                <div className="bg-green-50 rounded-lg p-3 text-center">
                  <p className="text-[10px] text-green-500 font-semibold uppercase">CGST ITC</p>
                  <p className="text-base font-bold text-green-800 mt-1">{fmt(gstr.inward_itc.cgst)}</p>
                </div>
                <div className="bg-green-50 rounded-lg p-3 text-center">
                  <p className="text-[10px] text-green-500 font-semibold uppercase">SGST ITC</p>
                  <p className="text-base font-bold text-green-800 mt-1">{fmt(gstr.inward_itc.sgst)}</p>
                </div>
                <div className="bg-green-50 rounded-lg p-3 text-center">
                  <p className="text-[10px] text-green-500 font-semibold uppercase">IGST ITC</p>
                  <p className="text-base font-bold text-green-800 mt-1">{fmt(gstr.inward_itc.igst)}</p>
                </div>
                <div className="bg-green-100 rounded-lg p-3 text-center border border-green-200">
                  <p className="text-[10px] text-green-700 font-bold uppercase">Total ITC</p>
                  <p className="text-lg font-bold text-green-900 mt-1">{fmt(gstr.inward_itc.total)}</p>
                </div>
              </div>
            </div>

            {/* Net Tax Payable */}
            <div className="px-5 py-4 bg-gray-50">
              <h4 className="text-xs font-bold text-gray-600 uppercase tracking-widest mb-3">Net Tax Payable (Output − ITC)</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className={`rounded-lg p-3 text-center ${gstr.net_cgst > 0 ? 'bg-orange-50 border border-orange-200' : 'bg-gray-50 border border-gray-200'}`}>
                  <p className="text-[10px] font-semibold uppercase text-gray-500">CGST Payable</p>
                  <p className={`text-base font-bold mt-1 ${gstr.net_cgst > 0 ? 'text-orange-700' : 'text-gray-600'}`}>{fmt(gstr.net_cgst)}</p>
                </div>
                <div className={`rounded-lg p-3 text-center ${gstr.net_sgst > 0 ? 'bg-orange-50 border border-orange-200' : 'bg-gray-50 border border-gray-200'}`}>
                  <p className="text-[10px] font-semibold uppercase text-gray-500">SGST Payable</p>
                  <p className={`text-base font-bold mt-1 ${gstr.net_sgst > 0 ? 'text-orange-700' : 'text-gray-600'}`}>{fmt(gstr.net_sgst)}</p>
                </div>
                <div className={`rounded-lg p-3 text-center ${gstr.net_igst > 0 ? 'bg-orange-50 border border-orange-200' : 'bg-gray-50 border border-gray-200'}`}>
                  <p className="text-[10px] font-semibold uppercase text-gray-500">IGST Payable</p>
                  <p className={`text-base font-bold mt-1 ${gstr.net_igst > 0 ? 'text-orange-700' : 'text-gray-600'}`}>{fmt(gstr.net_igst)}</p>
                </div>
                <div className={`rounded-lg p-4 text-center ${gstr.net_total > 0 ? 'bg-red-50 border-2 border-red-300' : 'bg-green-50 border-2 border-green-300'}`}>
                  <p className="text-[10px] font-bold uppercase text-gray-600">Total Payable</p>
                  <p className={`text-xl font-bold mt-1 ${gstr.net_total > 0 ? 'text-red-700' : 'text-green-700'}`}>{fmt(gstr.net_total)}</p>
                  {gstr.net_total <= 0 && <p className="text-[10px] text-green-600 mt-0.5">No tax due</p>}
                </div>
              </div>
            </div>
          </div>

          {/* Voucher Breakdown */}
          {(gstr.breakdown ?? []).length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-5 py-3 border-b border-gray-100">
                <h3 className="text-sm font-semibold text-gray-700">Voucher Breakdown</h3>
                <p className="text-xs text-gray-400 mt-0.5">{gstr.breakdown.length} vouchers included in this return</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-gray-50 text-gray-500 uppercase tracking-wide">
                      <th className="px-3 py-2.5 text-left">Voucher No</th>
                      <th className="px-3 py-2.5 text-left">Date</th>
                      <th className="px-3 py-2.5 text-left">Type</th>
                      <th className="px-3 py-2.5 text-left">Party</th>
                      <th className="px-3 py-2.5 text-right">Taxable</th>
                      <th className="px-3 py-2.5 text-right">CGST</th>
                      <th className="px-3 py-2.5 text-right">SGST</th>
                      <th className="px-3 py-2.5 text-right">IGST</th>
                      <th className="px-3 py-2.5 text-right">Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gstr.breakdown.map((row, i) => (
                      <tr key={i} className="border-t border-gray-50 hover:bg-gray-50">
                        <td className="px-3 py-1.5 font-mono text-[#002B5B]">{row.voucher_no}</td>
                        <td className="px-3 py-1.5 text-gray-500">{row.voucher_date}</td>
                        <td className="px-3 py-1.5">
                          <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${VOUCHER_COLORS[row.voucher_type] ?? 'bg-gray-100 text-gray-600'}`}>
                            {row.voucher_type}
                          </span>
                        </td>
                        <td className="px-3 py-1.5 text-gray-600 max-w-[100px] truncate">{row.party_name || '—'}</td>
                        <td className="px-3 py-1.5 text-right">{fmt(row.taxable_amount)}</td>
                        <td className="px-3 py-1.5 text-right text-blue-600">{row.cgst_amount > 0 ? fmt(row.cgst_amount) : '—'}</td>
                        <td className="px-3 py-1.5 text-right text-blue-600">{row.sgst_amount > 0 ? fmt(row.sgst_amount) : '—'}</td>
                        <td className="px-3 py-1.5 text-right text-blue-600">{row.igst_amount > 0 ? fmt(row.igst_amount) : '—'}</td>
                        <td className="px-3 py-1.5 text-right font-semibold">{fmt(row.total_amount)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
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
  const [vType,       setVType]       = useState('Expense')
  const [vDate,       setVDate]       = useState(toIso(new Date()))
  const [billNo,      setBillNo]      = useState('')
  const [billDate,    setBillDate]    = useState('')
  const [partyName,   setPartyName]   = useState('')
  const [partyGstin,  setPartyGstin]  = useState('')
  const [partyState,  setPartyState]  = useState('')
  const [supplyType,  setSupplyType]  = useState<'Intra' | 'Inter'>('Intra')
  const [cgstRate,    setCgstRate]    = useState('9')
  const [igstRate,    setIgstRate]    = useState('18')
  const [applyTds,    setApplyTds]    = useState(false)
  const [tdsSection,  setTdsSection]  = useState('')
  const [tdsRate,     setTdsRate]     = useState('')
  const [narration,   setNarration]   = useState('')
  const [lines, setLines] = useState([{ expense_head: '', description: '', amount: '', cost_centre: '', is_debit: 1 }])
  const [saveErr, setSaveErr] = useState('')

  // Additional fields for Payment / Receipt / Contra types
  const [paymentMode, setPaymentMode] = useState('Cash')
  const [bankLedger,  setBankLedger]  = useState('')
  const [chequeNo,    setChequeNo]    = useState('')
  const [refNumber,   setRefNumber]   = useState('')
  const [singleAmount, setSingleAmount] = useState('')
  const [fromAccount, setFromAccount] = useState('')
  const [toAccount,   setToAccount]   = useState('')



  const isPaymentReceipt = useMemo(() => ['Payment', 'Receipt'].includes(vType), [vType])
  const isJournal        = useMemo(() => vType === 'Journal', [vType])
  const isContra         = useMemo(() => vType === 'Contra', [vType])
  const isPurchaseInvoice = useMemo(() => vType === 'Purchase Invoice', [vType])
  const isExpenseType    = useMemo(() => ['Expense', 'JWO Payment', 'Purchase Invoice'].includes(vType), [vType])
  const showGstSection   = useMemo(() => isExpenseType, [isExpenseType])

  const taxableAmount = useMemo(() => {
    if (isPaymentReceipt || isContra) return parseFloat(singleAmount) || 0
    return lines.reduce((s, l) => s + (parseFloat(l.amount) || 0), 0)
  }, [lines, isPaymentReceipt, isContra, singleAmount])

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

  const journalDrTotal = useMemo(() =>
    lines.filter(l => l.is_debit === 1).reduce((s, l) => s + (parseFloat(l.amount) || 0), 0), [lines])
  const journalCrTotal = useMemo(() =>
    lines.filter(l => l.is_debit === 0).reduce((s, l) => s + (parseFloat(l.amount) || 0), 0), [lines])
  const journalBalanced = useMemo(() => Math.abs(journalDrTotal - journalCrTotal) < 0.01, [journalDrTotal, journalCrTotal])

  const netPayable = totalAmt - tdsAmtComputed

  function handleTdsSectionChange(sec: string) {
    setTdsSection(sec)
    const found = tdsSections.find(t => t.section === sec)
    if (found) setTdsRate(String(found.rate_individual))
  }

  function addLine() {
    setLines(prev => [...prev, { expense_head: '', description: '', amount: '', cost_centre: '', is_debit: 1 }])
  }
  function removeLine(idx: number) {
    setLines(prev => prev.filter((_, i) => i !== idx))
  }
  function updateLine(idx: number, field: string, value: string) {
    setLines(prev => prev.map((l, i) => i === idx ? { ...l, [field]: field === 'is_debit' ? parseInt(value) : value } : l))
  }

  const saveMut = useMutation({
    mutationFn: (body: object) => api.post('/finance/vouchers', body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['finance-vouchers'] })
      qc.invalidateQueries({ queryKey: ['finance-daybook-today'] })
      setLines([{ expense_head: '', description: '', amount: '', cost_centre: '', is_debit: 1 }])
      setBillNo(''); setBillDate(''); setPartyName(''); setPartyGstin(''); setPartyState('')
      setNarration(''); setTdsSection(''); setTdsRate(''); setApplyTds(false); setSaveErr('')
      setSingleAmount(''); setChequeNo(''); setRefNumber(''); setFromAccount(''); setToAccount('')
    },
    onError: () => setSaveErr('Failed to save voucher.'),
  })

  const delVoucher = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/vouchers/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['finance-vouchers'] }),
  })

  function handleSave() {
    if (!vDate) { setSaveErr('Voucher date is required.'); return }
    if (isPaymentReceipt || isContra) {
      if (!singleAmount || parseFloat(singleAmount) <= 0) { setSaveErr('Amount is required.'); return }
    } else if (isJournal) {
      if (lines.every(l => !l.expense_head)) { setSaveErr('At least one journal line is required.'); return }
      if (!journalBalanced) { setSaveErr('Journal entries must balance (Dr = Cr).'); return }
    } else {
      if (lines.every(l => !l.expense_head)) { setSaveErr('At least one expense line is required.'); return }
    }
    const showTds = applyTds || vType === 'JWO Payment'
    const body: Record<string, unknown> = {
      voucher_type: vType,
      voucher_date: vDate,
      bill_no: billNo, bill_date: billDate,
      party_name: partyName, party_gstin: partyGstin, party_state: partyState,
      supply_type: supplyType,
      narration,
      taxable_amount: taxableAmount,
      cgst_amount: showGstSection ? cgstAmt : 0,
      sgst_amount: showGstSection ? sgstAmt : 0,
      igst_amount: showGstSection ? igstAmt : 0,
      tds_section: showTds ? tdsSection : '',
      tds_rate: showTds ? parseFloat(tdsRate) || 0 : 0,
      tds_amount: showTds ? tdsAmtComputed : 0,
      total_amount: totalAmt,
      net_payable: netPayable,
      payment_mode: (isPaymentReceipt || isContra) ? paymentMode : '',
      bank_ledger: (isPaymentReceipt || isContra) ? bankLedger : '',
      cheque_no: chequeNo,
      ref_number: refNumber,
      lines: (isPaymentReceipt || isContra) ? [] : lines.filter(l => l.expense_head).map(l => ({
        expense_head: l.expense_head,
        description: l.description,
        amount: parseFloat(l.amount) || 0,
        cost_centre: l.cost_centre,
        is_debit: l.is_debit,
      })),
    }
    if (isContra) {
      body.from_account = fromAccount
      body.to_account = toAccount
    }
    saveMut.mutate(body)
  }

  const showTdsSection = applyTds || vType === 'JWO Payment' || (isPaymentReceipt && applyTds)

  return (
    <div className="space-y-5">
      {/* Form */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-4">
        <h3 className="text-sm font-semibold text-gray-700">
          {isJournal ? 'New Journal Entry' : isContra ? 'New Contra Voucher' : isPaymentReceipt ? `New ${vType} Voucher` : 'New Expense Voucher'}
        </h3>

        {/* Row 1: type, date, bill no, bill date */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Voucher Type</label>
            <select value={vType} onChange={e => setVType(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
              <option>Expense</option>
              <option>JWO Payment</option>
              <option>Payment</option>
              <option>Receipt</option>
              <option>Journal</option>
              <option>Contra</option>
              <option>Purchase Invoice</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Voucher Date *</label>
            <input type="date" value={vDate} onChange={e => setVDate(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          {(isExpenseType || isPurchaseInvoice) && (
            <>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500">Bill No{isPurchaseInvoice ? ' *' : ''}</label>
                <input type="text" value={billNo} onChange={e => setBillNo(e.target.value)} placeholder="Vendor bill no"
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500">Bill Date</label>
                <input type="date" value={billDate} onChange={e => setBillDate(e.target.value)}
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
            </>
          )}
          {isPaymentReceipt && (
            <>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500">Reference No</label>
                <input type="text" value={refNumber} onChange={e => setRefNumber(e.target.value)} placeholder="Ref / UTR"
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
            </>
          )}
        </div>

        {/* Party row — not for Contra or Journal */}
        {!isContra && !isJournal && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">{vType === 'Receipt' ? 'Received From' : 'Pay To / Party Name'}</label>
              <LedgerCombobox
                value={partyName}
                allLedgers={ledgers.filter(l => ['Sundry Creditors','Sundry Debtors','Cash-in-Hand','Bank Accounts'].includes(l.group_name))}
                placeholder="Vendor / party name"
                onChange={(name, ledger) => {
                  setPartyName(name)
                  if (ledger?.gstin) setPartyGstin(ledger.gstin)
                  if (ledger?.state) setPartyState(ledger.state)
                }}
              />
            </div>
            {isExpenseType && (
              <>
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
              </>
            )}
          </div>
        )}

        {/* Payment / Receipt — single amount + payment mode */}
        {isPaymentReceipt && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 bg-gray-50 rounded-lg p-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">Amount (₹) *</label>
              <input type="number" value={singleAmount} onChange={e => setSingleAmount(e.target.value)} placeholder="0"
                className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">Payment Mode</label>
              <select value={paymentMode} onChange={e => setPaymentMode(e.target.value)}
                className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
                <option>Cash</option>
                <option>Cheque</option>
                <option>NEFT</option>
                <option>RTGS</option>
                <option>IMPS</option>
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">Bank Account</label>
              <LedgerCombobox
                value={bankLedger}
                allLedgers={ledgers.filter(l => ['Bank Accounts','Cash-in-Hand'].includes(l.group_name))}
                placeholder="Bank / Cash ledger"
                onChange={name => setBankLedger(name)}
              />
            </div>
            {paymentMode === 'Cheque' && (
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500">Cheque No</label>
                <input type="text" value={chequeNo} onChange={e => setChequeNo(e.target.value)} placeholder="Cheque number"
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
            )}
            {['NEFT', 'RTGS', 'IMPS'].includes(paymentMode) && (
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500">UTR / Ref No</label>
                <input type="text" value={refNumber} onChange={e => setRefNumber(e.target.value)} placeholder="UTR number"
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
            )}
          </div>
        )}

        {/* Contra — from/to accounts */}
        {isContra && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 bg-blue-50 rounded-lg p-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">From Account *</label>
              <LedgerCombobox
                value={fromAccount}
                allLedgers={ledgers.filter(l => ['Bank Accounts','Cash-in-Hand'].includes(l.group_name))}
                placeholder="Cash / Bank ledger"
                onChange={name => setFromAccount(name)}
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">To Account *</label>
              <LedgerCombobox
                value={toAccount}
                allLedgers={ledgers.filter(l => ['Bank Accounts','Cash-in-Hand'].includes(l.group_name))}
                placeholder="Cash / Bank ledger"
                onChange={name => setToAccount(name)}
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">Amount (₹) *</label>
              <input type="number" value={singleAmount} onChange={e => setSingleAmount(e.target.value)} placeholder="0"
                className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-gray-500">Transaction Type</label>
              <select value={paymentMode} onChange={e => setPaymentMode(e.target.value)}
                className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
                <option>Cash</option>
                <option>Cheque</option>
                <option>NEFT</option>
                <option>RTGS</option>
                <option>IMPS</option>
              </select>
            </div>
            {paymentMode === 'Cheque' && (
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500">Cheque No</label>
                <input type="text" value={chequeNo} onChange={e => setChequeNo(e.target.value)} placeholder="Cheque number"
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
            )}
            {['NEFT', 'RTGS', 'IMPS'].includes(paymentMode) && (
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray-500">UTR No</label>
                <input type="text" value={refNumber} onChange={e => setRefNumber(e.target.value)} placeholder="UTR number"
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
            )}
          </div>
        )}

        {/* Supply Type — only for Expense / Purchase Invoice */}
        {isExpenseType && (
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
        )}

        {/* Lines — for Expense / Journal / Purchase Invoice */}
        {!isPaymentReceipt && !isContra && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-medium text-gray-600">
                {isJournal ? 'Journal Lines' : 'Expense Lines'}
              </label>
              <div className="flex items-center gap-3">
                {isJournal && (
                  <span className={`text-xs font-medium ${journalBalanced ? 'text-green-600' : 'text-red-500'}`}>
                    Dr: {fmt(journalDrTotal)} | Cr: {fmt(journalCrTotal)}
                    {!journalBalanced && ' (Unbalanced)'}
                  </span>
                )}
                <button onClick={addLine} className="text-xs text-blue-600 hover:text-blue-800 font-medium">+ Add Line</button>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-gray-50 text-gray-500 uppercase tracking-wide">
                    <th className="px-3 py-2 text-left w-1/3">{isJournal ? 'Ledger / Account' : 'Expense Head'}</th>
                    <th className="px-3 py-2 text-left">Description</th>
                    <th className="px-3 py-2 text-right w-28">Amount (₹)</th>
                    {isJournal && <th className="px-3 py-2 text-center w-20">Dr / Cr</th>}
                    {!isJournal && <th className="px-3 py-2 text-left w-28">Cost Centre</th>}
                    <th className="px-3 py-2 w-8"></th>
                  </tr>
                </thead>
                <tbody>
                  {lines.map((line, idx) => (
                    <tr key={idx} className="border-t border-gray-100">
                      <td className="px-3 py-1.5">
                        <LedgerCombobox
                          value={line.expense_head}
                          allLedgers={ledgers}
                          placeholder={isJournal ? 'Ledger / Account' : 'Expense head'}
                          onChange={(name, ledger) => {
                            updateLine(idx, 'expense_head', name)
                            // Auto-fill GSTIN, TDS when a known ledger is selected
                            if (ledger) {
                              if (ledger.gstin) setPartyGstin(ledger.gstin)
                              if (ledger.state) setPartyState(ledger.state)
                              if (ledger.tds_applicable && ledger.tds_section) {
                                setApplyTds(true)
                                setTdsSection(ledger.tds_section)
                                const sec = tdsSections.find(s => s.section === ledger.tds_section)
                                if (sec) setTdsRate(String(sec.rate_individual))
                              }
                            }
                          }}
                        />
                      </td>
                      <td className="px-3 py-1.5">
                        <input type="text" value={line.description} onChange={e => updateLine(idx, 'description', e.target.value)}
                          list={`desc-list-${idx}`}
                          placeholder="Details / narration"
                          className="w-full border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-300" />
                        <datalist id={`desc-list-${idx}`}>
                          {/* Suggest previous descriptions used for this expense head */}
                          {vouchers.flatMap(v => v.lines ?? [])
                            .filter(l => l.expense_head === line.expense_head && l.description)
                            .map(l => l.description)
                            .filter((d, i, arr) => arr.indexOf(d) === i)
                            .slice(0, 8)
                            .map(d => <option key={d} value={d} />)}
                        </datalist>
                      </td>
                      <td className="px-3 py-1.5">
                        <input type="number" value={line.amount} onChange={e => updateLine(idx, 'amount', e.target.value)}
                          placeholder="0" className="w-full border border-gray-200 rounded px-2 py-1 text-right focus:outline-none focus:ring-1 focus:ring-blue-300" />
                      </td>
                      {isJournal && (
                        <td className="px-3 py-1.5 text-center">
                          <select value={String(line.is_debit)} onChange={e => updateLine(idx, 'is_debit', e.target.value)}
                            className={`text-xs border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-300 font-semibold ${line.is_debit === 1 ? 'border-blue-300 bg-blue-50 text-blue-700' : 'border-green-300 bg-green-50 text-green-700'}`}>
                            <option value="1">Dr</option>
                            <option value="0">Cr</option>
                          </select>
                        </td>
                      )}
                      {!isJournal && (
                        <td className="px-3 py-1.5">
                          <input type="text" value={line.cost_centre} onChange={e => updateLine(idx, 'cost_centre', e.target.value)}
                            placeholder="Optional"
                            className="w-full border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-300" />
                        </td>
                      )}
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
          </div>
        )}

        {/* GST — only for Expense / Purchase Invoice */}
        {showGstSection && (
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
        )}

        {/* TDS — for Expense / JWO Payment / Payment (contractor) */}
        {!isJournal && !isContra && vType !== 'JWO Payment' && (
          <label className="flex items-center gap-2 text-xs cursor-pointer">
            <input type="checkbox" checked={applyTds} onChange={e => setApplyTds(e.target.checked)} />
            Apply TDS {isPaymentReceipt ? '(Contractor Payment)' : ''}
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
            {!isJournal && (
              <>
                <div>Taxable: <span className="font-semibold">{fmt(taxableAmount)}</span></div>
                {showGstSection && <div>GST: <span className="font-semibold">{fmt(totalGst)}</span></div>}
                {showTdsSection && <div>TDS: <span className="font-semibold">({fmt(tdsAmtComputed)})</span></div>}
              </>
            )}
            {isJournal && (
              <>
                <div>Dr Total: <span className="font-semibold">{fmt(journalDrTotal)}</span></div>
                <div>Cr Total: <span className="font-semibold">{fmt(journalCrTotal)}</span></div>
                {!journalBalanced && <div className="text-yellow-300">Entries unbalanced</div>}
              </>
            )}
          </div>
          <div className="text-right">
            <div className="text-xs text-blue-200">{isJournal ? 'Dr / Cr' : 'Net Payable'}</div>
            <div className="text-xl font-bold">{isJournal ? fmt(journalDrTotal) : fmt(netPayable)}</div>
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
          ['voucher-types', 'Voucher Types'],
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
      {sub === 'voucher-types'       && <VoucherTypesSubTab />}
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

const blankLedgerForm = () => ({
  name: '', alias: '', group_id: '', maintain_bill_by_bill: false, credit_period: '',
  tds_applicable: false, tds_section: '', is_tcs_applicable: false,
  address: '', state: '', country: 'India', pincode: '', registration_type: '', gstin: '', pan: '',
  bank_name: '', bank_account: '', bank_ifsc: '',
})

function LedgersSubTab() {
  const qc = useQueryClient()
  const { data: groups = [] } = useQuery<LedgerGroup[]>({
    queryKey: ['finance-ledger-groups'],
    queryFn: async () => { const { data } = await api.get('/finance/masters/ledger-groups'); return data },
  })
  const { data: tdsSections = [] } = useQuery<TDSSection[]>({
    queryKey: ['finance-tds-sections'],
    queryFn: async () => { const { data } = await api.get('/finance/masters/tds-sections'); return data },
    staleTime: 5 * 60 * 1000,
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

  const [form, setForm] = useState(blankLedgerForm)
  const [err, setErr] = useState('')
  const F = (k: string, v: string | boolean) => setForm(p => ({ ...p, [k]: v }))

  const addMut = useMutation({
    mutationFn: (b: object) => api.post('/finance/masters/ledgers', b),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['finance-ledgers'] })
      setForm(blankLedgerForm()); setErr('')
    },
    onError: () => setErr('Failed to save.'),
  })
  const delMut = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/masters/ledgers/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['finance-ledgers'] }),
  })

  const selectedGroup = groups.find(g => String(g.id) === form.group_id)

  function handleSave() {
    if (!form.name.trim()) { setErr('Name is required'); return }
    addMut.mutate({
      name: form.name.trim(),
      alias: form.alias,
      group_id: form.group_id ? parseInt(form.group_id) : null,
      group_name: selectedGroup?.name || '',
      maintain_bill_by_bill: form.maintain_bill_by_bill ? 1 : 0,
      credit_period: parseInt(form.credit_period) || 0,
      tds_applicable: form.tds_applicable ? 1 : 0,
      tds_section: form.tds_section,
      is_tcs_applicable: form.is_tcs_applicable ? 1 : 0,
      address: form.address,
      state: form.state,
      country: form.country,
      pincode: form.pincode,
      registration_type: form.registration_type,
      gstin: form.gstin,
      pan: form.pan,
      bank_name: form.bank_name,
      bank_account: form.bank_account,
      bank_ifsc: form.bank_ifsc,
    })
  }

  const inp = "text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300 w-full"
  const lbl = "text-xs text-gray-500 mb-0.5"

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
        <h4 className="text-sm font-semibold text-[#002B5B] mb-4">Add Ledger</h4>

        {/* Section: General */}
        <div className="mb-4">
          <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">General</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="flex flex-col">
              <label className={lbl}>Name *</label>
              <input type="text" value={form.name} onChange={e => F('name', e.target.value)} placeholder="Ledger name" className={inp} />
            </div>
            <div className="flex flex-col">
              <label className={lbl}>Alias</label>
              <input type="text" value={form.alias} onChange={e => F('alias', e.target.value)} placeholder="Alternate name" className={inp} />
            </div>
            <div className="flex flex-col">
              <label className={lbl}>Under (Group)</label>
              <select value={form.group_id} onChange={e => F('group_id', e.target.value)} className={inp}>
                <option value="">— Select —</option>
                {groups.map(g => <option key={g.id} value={String(g.id)}>{g.name}</option>)}
              </select>
            </div>
            <div className="flex flex-col">
              <label className={lbl}>Default Credit Period (days)</label>
              <input type="number" value={form.credit_period} onChange={e => F('credit_period', e.target.value)} placeholder="0" className={inp} />
            </div>
            <div className="flex flex-col justify-end">
              <label className="text-xs text-gray-500 mb-1">Maintain Bill-by-Bill</label>
              <div className="flex gap-3">
                {(['Yes', 'No'] as const).map(v => (
                  <label key={v} className="flex items-center gap-1 text-xs cursor-pointer">
                    <input type="radio" checked={form.maintain_bill_by_bill === (v === 'Yes')} onChange={() => F('maintain_bill_by_bill', v === 'Yes')} />
                    {v}
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Section: Statutory */}
        <div className="mb-4">
          <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">Statutory Details</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="flex flex-col justify-end">
              <label className="text-xs text-gray-500 mb-1">Is TDS Deductable</label>
              <div className="flex gap-3">
                {(['Yes', 'No'] as const).map(v => (
                  <label key={v} className="flex items-center gap-1 text-xs cursor-pointer">
                    <input type="radio" checked={form.tds_applicable === (v === 'Yes')} onChange={() => F('tds_applicable', v === 'Yes')} />
                    {v}
                  </label>
                ))}
              </div>
            </div>
            {form.tds_applicable && (
              <div className="flex flex-col">
                <label className={lbl}>TDS Section</label>
                <select value={form.tds_section} onChange={e => F('tds_section', e.target.value)} className={inp}>
                  <option value="">— Select —</option>
                  {tdsSections.map(s => <option key={s.id} value={s.section}>{s.section} — {s.description}</option>)}
                </select>
              </div>
            )}
            <div className="flex flex-col justify-end">
              <label className="text-xs text-gray-500 mb-1">Is TCS Applicable</label>
              <div className="flex gap-3">
                {(['Yes', 'No'] as const).map(v => (
                  <label key={v} className="flex items-center gap-1 text-xs cursor-pointer">
                    <input type="radio" checked={form.is_tcs_applicable === (v === 'Yes')} onChange={() => F('is_tcs_applicable', v === 'Yes')} />
                    {v}
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Section: Mailing Details */}
        <div className="mb-4">
          <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">Mailing Details</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="flex flex-col col-span-2">
              <label className={lbl}>Address</label>
              <textarea value={form.address} onChange={e => F('address', e.target.value)} placeholder="Full address" rows={2}
                className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300 w-full resize-none" />
            </div>
            <div className="flex flex-col">
              <label className={lbl}>State</label>
              <input type="text" value={form.state} onChange={e => F('state', e.target.value)} placeholder="e.g. Maharashtra" className={inp} />
            </div>
            <div className="flex flex-col">
              <label className={lbl}>Country</label>
              <input type="text" value={form.country} onChange={e => F('country', e.target.value)} placeholder="India" className={inp} />
            </div>
            <div className="flex flex-col">
              <label className={lbl}>Pincode</label>
              <input type="text" value={form.pincode} onChange={e => F('pincode', e.target.value)} placeholder="400001" className={inp} />
            </div>
            <div className="flex flex-col">
              <label className={lbl}>Registration Type</label>
              <select value={form.registration_type} onChange={e => F('registration_type', e.target.value)} className={inp}>
                <option value="">— None —</option>
                <option>Regular</option>
                <option>Composition</option>
                <option>e-Commerce Operator</option>
                <option>Unregistered</option>
                <option>SEZ</option>
              </select>
            </div>
            <div className="flex flex-col">
              <label className={lbl}>GSTIN / UIN</label>
              <input type="text" value={form.gstin} onChange={e => F('gstin', e.target.value)} placeholder="15-digit GSTIN" className={inp} />
            </div>
            <div className="flex flex-col">
              <label className={lbl}>PAN / IT No</label>
              <input type="text" value={form.pan} onChange={e => F('pan', e.target.value)} placeholder="PAN" className={inp} />
            </div>
          </div>
        </div>

        {/* Section: Banking Details */}
        <div className="mb-5">
          <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">Banking Details</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="flex flex-col">
              <label className={lbl}>Bank Name</label>
              <input type="text" value={form.bank_name} onChange={e => F('bank_name', e.target.value)} placeholder="e.g. HDFC Bank" className={inp} />
            </div>
            <div className="flex flex-col">
              <label className={lbl}>Account No</label>
              <input type="text" value={form.bank_account} onChange={e => F('bank_account', e.target.value)} placeholder="Account number" className={inp} />
            </div>
            <div className="flex flex-col">
              <label className={lbl}>IFSC Code</label>
              <input type="text" value={form.bank_ifsc} onChange={e => F('bank_ifsc', e.target.value)} placeholder="HDFC0001234" className={inp} />
            </div>
          </div>
        </div>

        <button onClick={handleSave} disabled={addMut.isPending}
          className="px-5 py-2 text-xs font-semibold bg-[#002B5B] text-white rounded hover:bg-blue-900 disabled:opacity-60">
          {addMut.isPending ? 'Saving…' : '+ Save Ledger'}
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
                <th className="px-4 py-2.5 text-center">TCS?</th>
                <th className="px-4 py-2.5"></th>
              </tr></thead>
              <tbody>
                {ledgers.map(l => (
                  <tr key={l.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">
                      {l.name}
                      {l.alias ? <span className="ml-1 text-[10px] text-gray-400">({l.alias})</span> : null}
                    </td>
                    <td className="px-4 py-2 text-gray-500">{l.group_name || '—'}</td>
                    <td className="px-4 py-2 text-gray-500 font-mono text-xs">{l.gstin || '—'}</td>
                    <td className="px-4 py-2 text-gray-500 font-mono text-xs">{l.pan || '—'}</td>
                    <td className="px-4 py-2 text-gray-500">{l.state || '—'}</td>
                    <td className="px-4 py-2 text-center">
                      {l.tds_applicable ? <span className="text-xs bg-amber-50 text-amber-700 px-1.5 py-0.5 rounded">Yes ({l.tds_section})</span> : '—'}
                    </td>
                    <td className="px-4 py-2 text-center">
                      {l.is_tcs_applicable ? <span className="text-xs bg-orange-50 text-orange-700 px-1.5 py-0.5 rounded">Yes</span> : '—'}
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

// ── Voucher Types Sub-tab ─────────────────────────────────────────
function VoucherTypesSubTab() {
  const qc = useQueryClient()
  const { data: vtypes = [], isLoading } = useQuery<VoucherType[]>({
    queryKey: ['finance-voucher-types'],
    queryFn: async () => { const { data } = await api.get('/finance/masters/voucher-types'); return data },
  })

  const [name, setName] = useState('')
  const [category, setCategory] = useState('Sales')
  const [abbr, setAbbr] = useState('')
  const [allowNarration, setAllowNarration] = useState(true)
  const [numberingMethod, setNumberingMethod] = useState('Auto')
  const [err, setErr] = useState('')
  const [editId, setEditId] = useState<number | null>(null)
  const [editForm, setEditForm] = useState<Partial<VoucherType>>({})

  const CATEGORIES = ['Sales', 'Purchase', 'Payment', 'Receipt', 'Journal', 'Contra']

  const addMut = useMutation({
    mutationFn: (b: object) => api.post('/finance/masters/voucher-types', b),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['finance-voucher-types'] })
      setName(''); setAbbr(''); setErr('')
    },
    onError: () => setErr('Failed to save.'),
  })

  const updateMut = useMutation({
    mutationFn: ({ id, body }: { id: number; body: object }) => api.put(`/finance/masters/voucher-types/${id}`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['finance-voucher-types'] })
      setEditId(null); setEditForm({})
    },
    onError: () => setErr('Update failed.'),
  })

  const delMut = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/masters/voucher-types/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['finance-voucher-types'] }),
  })

  const CATEGORY_COLORS: Record<string, string> = {
    Sales: 'bg-green-50 text-green-700',
    Purchase: 'bg-blue-50 text-blue-700',
    Payment: 'bg-red-50 text-red-700',
    Receipt: 'bg-teal-50 text-teal-700',
    Journal: 'bg-purple-50 text-purple-700',
    Contra: 'bg-gray-100 text-gray-700',
  }

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
        <h4 className="text-xs font-semibold text-gray-600 mb-3">Add Voucher Type</h4>
        <div className="flex flex-wrap gap-2 items-end">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Name *</label>
            <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Amazon Sales"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 w-40 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Category</label>
            <select value={category} onChange={e => setCategory(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
              {CATEGORIES.map(c => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Abbreviation</label>
            <input type="text" value={abbr} onChange={e => setAbbr(e.target.value)} placeholder="e.g. Amz"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 w-20 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Numbering</label>
            <select value={numberingMethod} onChange={e => setNumberingMethod(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
              <option>Auto</option>
              <option>Manual</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Allow Narration</label>
            <label className="flex items-center gap-1.5 h-[28px] cursor-pointer">
              <input type="checkbox" checked={allowNarration} onChange={e => setAllowNarration(e.target.checked)} />
              <span className="text-xs">{allowNarration ? 'Yes' : 'No'}</span>
            </label>
          </div>
          <button onClick={() => {
            if (!name.trim()) { setErr('Name required'); return }
            addMut.mutate({ name, voucher_category: category, abbreviation: abbr, allow_narration: allowNarration ? 1 : 0, numbering_method: numberingMethod })
          }} disabled={addMut.isPending}
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
              <th className="px-4 py-2.5 text-left">Category</th>
              <th className="px-4 py-2.5 text-left">Abbreviation</th>
              <th className="px-4 py-2.5 text-center">Narration</th>
              <th className="px-4 py-2.5 text-center">Numbering</th>
              <th className="px-4 py-2.5 text-center">Status</th>
              <th className="px-4 py-2.5 text-center">Edit</th>
              <th className="px-4 py-2.5 text-center">Delete</th>
            </tr></thead>
            <tbody>
              {vtypes.map(vt => (
                editId === vt.id ? (
                  <tr key={vt.id} className="border-t border-blue-100 bg-blue-50">
                    <td className="px-2 py-1.5">
                      <input type="text" value={editForm.name ?? vt.name} onChange={e => setEditForm(p => ({ ...p, name: e.target.value }))}
                        className="text-xs border border-gray-200 rounded px-2 py-1 w-full focus:outline-none focus:ring-1 focus:ring-blue-300" />
                    </td>
                    <td className="px-2 py-1.5">
                      <select value={editForm.voucher_category ?? vt.voucher_category} onChange={e => setEditForm(p => ({ ...p, voucher_category: e.target.value }))}
                        className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-300">
                        {CATEGORIES.map(c => <option key={c}>{c}</option>)}
                      </select>
                    </td>
                    <td className="px-2 py-1.5">
                      <input type="text" value={editForm.abbreviation ?? vt.abbreviation} onChange={e => setEditForm(p => ({ ...p, abbreviation: e.target.value }))}
                        className="text-xs border border-gray-200 rounded px-2 py-1 w-20 focus:outline-none focus:ring-1 focus:ring-blue-300" />
                    </td>
                    <td className="px-2 py-1.5 text-center">
                      <input type="checkbox" checked={(editForm.allow_narration ?? vt.allow_narration) === 1}
                        onChange={e => setEditForm(p => ({ ...p, allow_narration: e.target.checked ? 1 : 0 }))} />
                    </td>
                    <td className="px-2 py-1.5">
                      <select value={editForm.numbering_method ?? vt.numbering_method} onChange={e => setEditForm(p => ({ ...p, numbering_method: e.target.value }))}
                        className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-300">
                        <option>Auto</option><option>Manual</option>
                      </select>
                    </td>
                    <td className="px-2 py-1.5 text-center">
                      <select value={editForm.is_active ?? vt.is_active} onChange={e => setEditForm(p => ({ ...p, is_active: +e.target.value }))}
                        className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-300">
                        <option value={1}>Active</option><option value={0}>Inactive</option>
                      </select>
                    </td>
                    <td className="px-2 py-1.5 text-center" colSpan={2}>
                      <button onClick={() => updateMut.mutate({ id: vt.id, body: editForm })}
                        className="text-xs px-2 py-1 bg-[#002B5B] text-white rounded hover:bg-blue-900 mr-1">Save</button>
                      <button onClick={() => { setEditId(null); setEditForm({}) }}
                        className="text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded hover:bg-gray-200">Cancel</button>
                    </td>
                  </tr>
                ) : (
                  <tr key={vt.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">{vt.name}</td>
                    <td className="px-4 py-2">
                      <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${CATEGORY_COLORS[vt.voucher_category] ?? 'bg-gray-100 text-gray-600'}`}>
                        {vt.voucher_category}
                      </span>
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-gray-600">{vt.abbreviation || '—'}</td>
                    <td className="px-4 py-2 text-center text-xs text-gray-500">{vt.allow_narration ? 'Yes' : 'No'}</td>
                    <td className="px-4 py-2 text-center text-xs text-gray-500">{vt.numbering_method}</td>
                    <td className="px-4 py-2 text-center">
                      {vt.is_active
                        ? <span className="text-xs bg-green-50 text-green-700 px-1.5 py-0.5 rounded font-medium">Active</span>
                        : <span className="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded font-medium">Inactive</span>}
                    </td>
                    <td className="px-4 py-2 text-center">
                      <button onClick={() => { setEditId(vt.id); setEditForm({}) }}
                        className="text-xs text-blue-500 hover:text-blue-700 font-medium">Edit</button>
                    </td>
                    <td className="px-4 py-2 text-center">
                      <button onClick={() => { if (window.confirm('Delete voucher type?')) delMut.mutate(vt.id) }}
                        className="text-red-400 hover:text-red-600 text-xs">✕</button>
                    </td>
                  </tr>
                )
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

// ── Sales Uploads Tab ─────────────────────────────────────────────
const PLATFORM_FILE_HINTS: Record<string, string> = {
  Amazon:           'ZIP containing Amazon MTR CSV reports (B2C/B2B)',
  Myntra:           'ZIP containing Myntra settlement CSVs',
  Meesho:           'ZIP containing Meesho orders or gst_* finance ZIP',
  Flipkart:         'ZIP containing Flipkart order/settlement reports',
  Snapdeal:         'Snapdeal settlement XLSX (monthly finance report)',
  'Monthly Package':'Full monthly sales ZIP — auto-detects all platforms',
}

interface UploadResult {
  id?: number; platform: string; period: string; filename: string
  total_revenue: number; total_orders: number; total_returns: number
  net_revenue: number; rows_parsed?: number
  // Monthly package fields
  saved?: { platform: string; id: number; orders: number; revenue: number; returns: number; net_revenue: number; note?: string }[]
  skipped?: string[]
}

interface PreviewResult {
  preview: true; period: string
  platforms: { platform: string; orders: number; revenue: number; returns: number; net_revenue: number; note?: string }[]
  skipped: string[]
  total_revenue: number; total_returns: number; net_revenue: number
}

function SalesUploadsTab() {
  const qc = useQueryClient()
  const fileRef = useRef<HTMLInputElement>(null)

  const [platform,    setPlatform]    = useState('Amazon')
  const [period,      setPeriod]      = useState(() => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}` })
  const [uploadedBy,  setUploadedBy]  = useState('')
  const [notes,       setNotes]       = useState('')
  const [dragging,    setDragging]    = useState(false)
  const [uploading,   setUploading]   = useState(false)
  const [result,      setResult]      = useState<UploadResult | null>(null)
  const [preview,     setPreview]     = useState<PreviewResult | null>(null)
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [confirming,  setConfirming]  = useState(false)
  const [err,         setErr]         = useState('')
  const [filterPlatform, setFilterPlatform] = useState('')
  const [filterPeriod,   setFilterPeriod]   = useState('')

  const { data: uploads = [], isLoading } = useQuery<SalesUpload[]>({
    queryKey: ['finance-sales-uploads', filterPlatform, filterPeriod],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filterPlatform) params.set('platform', filterPlatform)
      if (filterPeriod)   params.set('period',   filterPeriod)
      const { data } = await api.get(`/finance/sales-uploads?${params}`)
      return data
    },
  })

  const delMut = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/sales-uploads/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['finance-sales-uploads'] }),
  })

  const isPackage = platform === 'Monthly Package'

  async function handleFile(file: File) {
    if (!period) { setErr('Select a period (month) before uploading.'); return }
    setErr(''); setResult(null); setPreview(null); setPendingFile(null); setUploading(true)
    try {
      const fd = new FormData()
      fd.append('file',   file)
      fd.append('period', period)
      if (isPackage) {
        // Preview first — don't save yet
        const { data } = await api.post('/finance/sales-uploads/preview-monthly-package', fd, {
          headers: { 'Content-Type': 'multipart/form-data' },
        })
        setPreview(data as PreviewResult)
        setPendingFile(file)
      } else {
        fd.append('uploaded_by', uploadedBy)
        fd.append('notes',       notes)
        fd.append('platform',    platform)
        const { data } = await api.post('/finance/sales-uploads/upload-file', fd, {
          headers: { 'Content-Type': 'multipart/form-data' },
        })
        setResult(data)
        qc.invalidateQueries({ queryKey: ['finance-sales-uploads'] })
      }
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setErr(msg || 'Upload failed. Check the file format.')
    } finally { setUploading(false) }
  }

  async function confirmSave() {
    if (!pendingFile || !period) return
    setErr(''); setConfirming(true)
    try {
      const fd = new FormData()
      fd.append('file',        pendingFile)
      fd.append('period',      period)
      fd.append('uploaded_by', uploadedBy)
      fd.append('notes',       notes)
      const { data } = await api.post('/finance/sales-uploads/upload-monthly-package', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setPreview(null); setPendingFile(null)
      setResult(data)
      qc.invalidateQueries({ queryKey: ['finance-sales-uploads'] })
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setErr(msg || 'Save failed.')
    } finally { setConfirming(false) }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  return (
    <div className="space-y-5">
      {/* Lock banner */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 flex gap-2 items-start">
        <span className="text-amber-500 mt-0.5">🔒</span>
        <div>
          <p className="text-xs font-semibold text-amber-800">Finance-locked uploads</p>
          <p className="text-xs text-amber-700 mt-0.5">Records saved here are immutable. Only Finance team members can delete them. Totals are parsed automatically from the file.</p>
        </div>
      </div>

      {/* Upload card */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-4">
        <h3 className="text-sm font-semibold text-[#002B5B]">Upload Sales File</h3>

        {/* Platform + Period + Uploaded By */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Platform *</label>
            <select value={platform} onChange={e => { setPlatform(e.target.value); setResult(null); setPreview(null); setPendingFile(null) }}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300">
              {['Monthly Package','Amazon','Myntra','Meesho','Flipkart','Snapdeal'].map(p => <option key={p}>{p}</option>)}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Period (Month) *</label>
            <input type="month" value={period} onChange={e => setPeriod(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Uploaded By</label>
            <input type="text" value={uploadedBy} onChange={e => setUploadedBy(e.target.value)} placeholder="Your name"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Notes</label>
            <input type="text" value={notes} onChange={e => setNotes(e.target.value)} placeholder="Optional notes"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
        </div>

        {/* Dropzone */}
        <div
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => fileRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors
            ${dragging ? 'border-[#002B5B] bg-blue-50' : 'border-gray-200 hover:border-gray-400 bg-gray-50'}`}
        >
          <input ref={fileRef} type="file" accept=".zip,.xlsx,.csv,.rar" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = '' }} />
          {uploading ? (
            <div className="space-y-2">
              <div className="w-8 h-8 border-2 border-[#002B5B] border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-sm text-gray-500">{isPackage ? 'Processing all platforms…' : `Parsing ${platform} file…`}</p>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="text-3xl">{isPackage ? '📦' : '📂'}</div>
              <p className="text-sm font-medium text-gray-700">
                {isPackage ? 'Drop monthly sales ZIP here' : `Drop ${platform} sales file here`}
              </p>
              <p className="text-xs text-gray-400">{PLATFORM_FILE_HINTS[platform]}</p>
              {isPackage && (
                <p className="text-xs text-amber-600">Parses Amazon, Myntra, Meesho, Snapdeal automatically. Flipkart PDFs counted as invoice records.</p>
              )}
              <p className="text-xs text-gray-400">or <span className="text-[#002B5B] font-medium underline">click to browse</span></p>
            </div>
          )}
        </div>

        {err && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-xs text-red-700">{err}</div>
        )}

        {/* Preview panel — shown before saving Monthly Package */}
        {preview && !result && (
          <div className="bg-amber-50 border border-amber-300 rounded-xl p-4 space-y-3">
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="text-amber-500 text-lg">🔍</span>
                <div>
                  <p className="text-sm font-semibold text-amber-800">Preview — not saved yet</p>
                  <p className="text-xs text-amber-600">{preview.period} · {preview.platforms.length} platform{preview.platforms.length !== 1 ? 's' : ''} detected · verify numbers before saving</p>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {[
                { label: 'Total Revenue', value: fmt(preview.total_revenue), color: 'text-gray-800' },
                { label: 'Total Returns', value: fmt(preview.total_returns), color: 'text-red-600'  },
                { label: 'Net Revenue',   value: fmt(preview.net_revenue),   color: 'text-amber-700'},
              ].map(({ label, value, color }) => (
                <div key={label} className="bg-white rounded-lg p-3 border border-amber-200 text-center">
                  <p className="text-xs text-gray-500 mb-1">{label}</p>
                  <p className={`text-sm font-bold ${color}`}>{value}</p>
                </div>
              ))}
            </div>
            <div className="space-y-1.5">
              {preview.platforms.map((r, i) => (
                <div key={i} className="flex items-center justify-between bg-white rounded-lg px-3 py-2 border border-amber-200 text-xs">
                  <span className="font-medium text-gray-700">{r.platform}</span>
                  {r.note
                    ? <span className="text-amber-600">{r.note}</span>
                    : <span className="text-gray-500">Rev {fmt(r.revenue)} · Ret {fmt(r.returns)} · {r.orders.toLocaleString()} orders</span>
                  }
                </div>
              ))}
              {preview.skipped.length > 0 && (
                <div className="text-xs text-amber-600 mt-1">Skipped: {preview.skipped.join(' | ')}</div>
              )}
            </div>
            <div className="flex gap-2 pt-1">
              <button
                onClick={confirmSave}
                disabled={confirming}
                className="flex-1 bg-[#002B5B] text-white text-xs font-semibold rounded-lg px-4 py-2 hover:bg-[#003f80] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {confirming ? 'Saving…' : 'Looks correct — Save to Finance'}
              </button>
              <button
                onClick={() => { setPreview(null); setPendingFile(null) }}
                disabled={confirming}
                className="px-4 py-2 text-xs font-semibold text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Parse result */}
        {result && (
          <div className="bg-green-50 border border-green-200 rounded-xl p-4 space-y-3">
            {result.saved ? (
              /* Monthly package result */
              <>
                <div className="flex items-center gap-2">
                  <span className="text-green-600 text-lg">✅</span>
                  <div>
                    <p className="text-sm font-semibold text-green-800">Monthly package processed — {result.saved.length} platform{result.saved.length !== 1 ? 's' : ''} saved</p>
                    <p className="text-xs text-green-600">{result.period}</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {[
                    { label: 'Total Revenue', value: fmt(result.total_revenue), color: 'text-gray-800' },
                    { label: 'Total Returns', value: fmt(result.total_returns), color: 'text-red-600'  },
                    { label: 'Net Revenue',   value: fmt(result.net_revenue),   color: 'text-green-700'},
                  ].map(({ label, value, color }) => (
                    <div key={label} className="bg-white rounded-lg p-3 border border-green-100 text-center">
                      <p className="text-xs text-gray-500 mb-1">{label}</p>
                      <p className={`text-sm font-bold ${color}`}>{value}</p>
                    </div>
                  ))}
                </div>
                <div className="space-y-1.5">
                  {result.saved.map((r, i) => (
                    <div key={i} className="flex items-center justify-between bg-white rounded-lg px-3 py-2 border border-green-100 text-xs">
                      <span className="font-medium text-gray-700">{r.platform}</span>
                      {r.note
                        ? <span className="text-amber-600">{r.note}</span>
                        : <span className="text-gray-500">Rev {fmt(r.revenue)} · Ret {fmt(r.returns)} · {r.orders.toLocaleString()} orders</span>
                      }
                    </div>
                  ))}
                  {result.skipped && result.skipped.length > 0 && (
                    <div className="text-xs text-amber-600 mt-1">
                      Skipped: {result.skipped.join(' | ')}
                    </div>
                  )}
                </div>
              </>
            ) : (
              /* Single platform result */
              <>
                <div className="flex items-center gap-2">
                  <span className="text-green-600 text-lg">✅</span>
                  <div>
                    <p className="text-sm font-semibold text-green-800">Upload saved — {(result.rows_parsed ?? result.total_orders).toLocaleString()} rows parsed</p>
                    <p className="text-xs text-green-600">{result.filename} · {result.platform} · {result.period}</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {[
                    { label: 'Gross Revenue', value: fmt(result.total_revenue), color: 'text-gray-800' },
                    { label: 'Returns',       value: fmt(result.total_returns), color: 'text-red-600'  },
                    { label: 'Net Revenue',   value: fmt(result.net_revenue),   color: 'text-green-700'},
                    { label: 'Orders',        value: result.total_orders.toLocaleString(), color: 'text-gray-700' },
                  ].map(({ label, value, color }) => (
                    <div key={label} className="bg-white rounded-lg p-3 border border-green-100 text-center">
                      <p className="text-xs text-gray-500 mb-1">{label}</p>
                      <p className={`text-sm font-bold ${color}`}>{value}</p>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Filter bar */}
      <div className="bg-white rounded-xl border border-gray-200 px-4 py-3 shadow-sm flex flex-wrap items-center gap-3">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Filter</span>
        <select value={filterPlatform} onChange={e => setFilterPlatform(e.target.value)}
          className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none">
          <option value="">All Platforms</option>
          {['Amazon','Myntra','Meesho','Flipkart','Snapdeal'].map(p => <option key={p}>{p}</option>)}
        </select>
        <input type="month" value={filterPeriod} onChange={e => setFilterPeriod(e.target.value)}
          className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none" />
        {(filterPlatform || filterPeriod) && (
          <button onClick={() => { setFilterPlatform(''); setFilterPeriod('') }}
            className="text-xs text-gray-400 hover:text-gray-600">Clear</button>
        )}
      </div>

      {/* Records table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {isLoading ? (
          <div className="p-8 text-center text-gray-400 text-sm animate-pulse">Loading…</div>
        ) : uploads.length === 0 ? (
          <div className="p-10 text-center text-gray-400 text-sm">
            <div className="text-3xl mb-2">📭</div>
            No sales uploads recorded yet. Upload a platform file above to get started.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 text-gray-500 uppercase tracking-wide">
                  <th className="px-3 py-2.5 text-left">Platform</th>
                  <th className="px-3 py-2.5 text-left">Period</th>
                  <th className="px-3 py-2.5 text-left">File</th>
                  <th className="px-3 py-2.5 text-right">Gross Rev</th>
                  <th className="px-3 py-2.5 text-right">Returns</th>
                  <th className="px-3 py-2.5 text-right">Net Revenue</th>
                  <th className="px-3 py-2.5 text-right">Orders</th>
                  <th className="px-3 py-2.5 text-left">Uploaded By</th>
                  <th className="px-3 py-2.5 text-left">Date</th>
                  <th className="px-3 py-2.5 text-center">🔒</th>
                  <th className="px-3 py-2.5"></th>
                </tr>
              </thead>
              <tbody>
                {uploads.map(u => (
                  <tr key={u.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-3 py-2">
                      <span style={{ backgroundColor: PLATFORM_COLORS[u.platform] ?? '#6B7280' }}
                        className="text-white px-1.5 py-0.5 rounded text-[10px] font-semibold">{u.platform}</span>
                    </td>
                    <td className="px-3 py-2 font-medium text-gray-700">{u.period}</td>
                    <td className="px-3 py-2 text-gray-500 max-w-[120px] truncate" title={u.filename}>{u.filename || '—'}</td>
                    <td className="px-3 py-2 text-right text-gray-700">{fmt(u.total_revenue)}</td>
                    <td className="px-3 py-2 text-right text-red-600">{u.total_returns > 0 ? fmt(u.total_returns) : '—'}</td>
                    <td className="px-3 py-2 text-right font-semibold text-green-700">{fmt(u.net_revenue)}</td>
                    <td className="px-3 py-2 text-right text-gray-600">{u.total_orders.toLocaleString()}</td>
                    <td className="px-3 py-2 text-gray-500">{u.uploaded_by || '—'}</td>
                    <td className="px-3 py-2 text-gray-400">{u.created_at?.slice(0, 10)}</td>
                    <td className="px-3 py-2 text-center">
                      {u.is_locked ? <span className="text-amber-500" title="Locked">🔒</span> : <span className="text-gray-300">—</span>}
                    </td>
                    <td className="px-3 py-2 text-right">
                      <button onClick={() => { if (window.confirm('Delete this locked sales upload? This cannot be undone.')) delMut.mutate(u.id) }}
                        className="text-red-400 hover:text-red-600">✕</button>
                    </td>
                  </tr>
                ))}
              </tbody>
              {uploads.length > 0 && (
                <tfoot>
                  <tr className="bg-[#002B5B] text-white text-xs font-semibold">
                    <td className="px-3 py-2.5" colSpan={3}>Total ({uploads.length} records)</td>
                    <td className="px-3 py-2.5 text-right">{fmt(uploads.reduce((s,u)=>s+u.total_revenue,0))}</td>
                    <td className="px-3 py-2.5 text-right">{fmt(uploads.reduce((s,u)=>s+u.total_returns,0))}</td>
                    <td className="px-3 py-2.5 text-right">{fmt(uploads.reduce((s,u)=>s+u.net_revenue,0))}</td>
                    <td className="px-3 py-2.5 text-right">{uploads.reduce((s,u)=>s+u.total_orders,0).toLocaleString()}</td>
                    <td colSpan={4} />
                  </tr>
                </tfoot>
              )}
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

// ── LedgerCombobox ─────────────────────────────────────────────────
// Live-search dropdown that queries /finance/masters/ledgers and returns full ledger object on select
interface LedgerComboboxProps {
  value: string
  onChange: (name: string, ledger?: Ledger) => void
  placeholder?: string
  className?: string
  allLedgers?: Ledger[]   // optional pre-loaded list (avoids extra fetches)
}
function LedgerCombobox({ value, onChange, placeholder = 'Search ledger…', className = '', allLedgers }: LedgerComboboxProps) {
  const [query,   setQuery]   = useState(value)
  const [open,    setOpen]    = useState(false)
  const [focused, setFocused] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Sync external value changes (e.g. form reset)
  useEffect(() => { setQuery(value) }, [value])

  // Filter from pre-loaded list or fetch
  const { data: searchResults = [] } = useQuery<Ledger[]>({
    queryKey: ['ledger-search', query],
    queryFn: async () => {
      if (!query || query.length < 1) return allLedgers ?? []
      const { data } = await api.get(`/finance/masters/ledgers?search=${encodeURIComponent(query)}`)
      return data
    },
    enabled: focused,
    staleTime: 30_000,
  })

  const options = focused
    ? (query.length === 0
        ? (allLedgers ?? searchResults)
        : searchResults.filter(l => l.name.toLowerCase().includes(query.toLowerCase()))
      )
    : []

  function select(l: Ledger) {
    setQuery(l.name)
    setOpen(false)
    onChange(l.name, l)
  }

  // Close on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false); setFocused(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      <input
        type="text"
        value={query}
        placeholder={placeholder}
        autoComplete="off"
        onFocus={() => { setFocused(true); setOpen(true) }}
        onChange={e => {
          setQuery(e.target.value)
          onChange(e.target.value)
          setOpen(true)
        }}
        onBlur={() => setTimeout(() => { setOpen(false) }, 180)}
        className="w-full border border-gray-200 rounded px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-blue-300"
      />
      {open && options.length > 0 && (
        <ul className="absolute z-50 left-0 right-0 mt-0.5 bg-white border border-gray-200 rounded-lg shadow-lg max-h-52 overflow-y-auto text-xs">
          {options.map(l => (
            <li key={l.id}
              onMouseDown={e => { e.preventDefault(); select(l) }}
              className="px-3 py-2 cursor-pointer hover:bg-blue-50 flex items-center justify-between gap-2">
              <span className="font-medium text-gray-800">{l.name}</span>
              <span className="text-gray-400 text-[10px] shrink-0">{l.group_name}</span>
            </li>
          ))}
        </ul>
      )}
      {open && focused && query.length > 0 && options.length === 0 && (
        <div className="absolute z-50 left-0 right-0 mt-0.5 bg-white border border-gray-200 rounded-lg shadow-lg px-3 py-2 text-xs text-gray-400">
          No ledger found — you can type a custom name
        </div>
      )}
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

// ── Chart of Accounts Tab ─────────────────────────────────────────
function ChartOfAccountsTab() {
  const { data, isLoading } = useQuery<{ groups: CoAGroup[] }>({
    queryKey: ['finance-coa'],
    queryFn: async () => { const { data } = await api.get('/finance/chart-of-accounts'); return data },
    staleTime: 60 * 1000,
  })
  const [collapsed, setCollapsed] = useState<Set<number>>(new Set())
  const toggle = (id: number) => setCollapsed(s => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n })

  const NATURE_SECTIONS: { nature: string; label: string; color: string }[] = [
    { nature: 'asset',     label: 'Assets',      color: 'bg-blue-50 border-blue-200' },
    { nature: 'liability', label: 'Liabilities',  color: 'bg-amber-50 border-amber-200' },
    { nature: 'income',    label: 'Income',       color: 'bg-green-50 border-green-200' },
    { nature: 'expense',   label: 'Expenses',     color: 'bg-red-50 border-red-200' },
  ]

  const totalLedgers = (g: CoAGroup): number =>
    g.ledgers.length + g.children.reduce((s, c) => s + totalLedgers(c), 0)

  const renderGroup = (g: CoAGroup, depth: number = 0): React.ReactNode => {
    const isCollapsed = collapsed.has(g.id)
    const hasContent = g.children.length > 0 || g.ledgers.length > 0
    const indentPx = depth * 20
    const bgClass = depth === 0 ? 'bg-gray-50' : depth === 1 ? 'bg-white' : 'bg-gray-50/40'
    return (
      <div key={g.id}>
        {/* Group header row */}
        <div
          className={`flex items-center gap-2 px-3 py-2 border-b border-gray-100 cursor-pointer hover:bg-blue-50/50 ${bgClass}`}
          style={{ paddingLeft: `${12 + indentPx}px` }}
          onClick={() => hasContent && toggle(g.id)}
        >
          {hasContent ? (
            <span className="text-gray-400 text-xs w-3 flex-shrink-0">{isCollapsed ? '▶' : '▼'}</span>
          ) : (
            <span className="w-3 flex-shrink-0" />
          )}
          <span className={`font-semibold ${depth === 0 ? 'text-sm text-gray-800' : 'text-xs text-gray-700'}`}>
            {g.name}
          </span>
          <NatureBadge nature={g.nature} />
          {totalLedgers(g) > 0 && (
            <span className="ml-auto text-xs text-gray-400">{totalLedgers(g)} ledger{totalLedgers(g) !== 1 ? 's' : ''}</span>
          )}
        </div>
        {/* Children & ledgers */}
        {!isCollapsed && (
          <>
            {g.children.map(child => renderGroup(child, depth + 1))}
            {g.ledgers.map(ldr => (
              <div key={ldr.id}
                className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-50 bg-white hover:bg-gray-50"
                style={{ paddingLeft: `${28 + indentPx}px` }}>
                <span className="text-gray-300 text-xs">•</span>
                <span className="text-xs text-gray-700 flex-1">{ldr.name}</span>
                {ldr.opening_balance !== 0 && (
                  <span className={`text-xs font-medium ${ldr.opening_balance >= 0 ? 'text-blue-600' : 'text-red-500'}`}>
                    {ldr.opening_balance >= 0 ? 'Dr' : 'Cr'} ₹{Math.abs(ldr.opening_balance).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </span>
                )}
              </div>
            ))}
          </>
        )}
      </div>
    )
  }

  if (isLoading) return <div className="p-8 text-center text-gray-400">Loading chart of accounts…</div>
  if (!data) return null

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-gray-800">Chart of Accounts</h3>
          <p className="text-xs text-gray-500 mt-0.5">Hierarchical view of all account groups and ledgers</p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => setCollapsed(new Set())}
            className="px-3 py-1.5 text-xs bg-white border border-gray-200 rounded hover:bg-gray-50">Expand All</button>
          <button onClick={() => setCollapsed(new Set(data.groups.flatMap(g => collectIds(g))))}
            className="px-3 py-1.5 text-xs bg-white border border-gray-200 rounded hover:bg-gray-50">Collapse All</button>
        </div>
      </div>

      {NATURE_SECTIONS.map(({ nature, label, color }) => {
        const sectionGroups = data.groups.filter(g => g.nature === nature)
        if (sectionGroups.length === 0) return null
        return (
          <div key={nature} className={`border rounded-xl overflow-hidden ${color}`}>
            <div className={`px-4 py-2 border-b ${color} flex items-center gap-2`}>
              <span className="text-sm font-bold text-gray-800">{label}</span>
              <span className="text-xs text-gray-500">
                ({sectionGroups.reduce((s, g) => s + totalLedgers(g), 0)} ledgers)
              </span>
            </div>
            <div className="divide-y divide-gray-100">
              {sectionGroups.map(g => renderGroup(g, 0))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function collectIds(g: CoAGroup): number[] {
  return [g.id, ...g.children.flatMap(collectIds)]
}

// ── Trial Balance Tab ──────────────────────────────────────────────
function TrialBalanceTab() {
  const now = new Date()
  const firstOfYear = `${now.getFullYear()}-04-01`  // Indian FY starts April
  const todayStr = now.toISOString().slice(0, 10)
  const [startDate, setStartDate] = useState(firstOfYear)
  const [endDate,   setEndDate]   = useState(todayStr)
  const [search,    setSearch]    = useState('')

  const { data, isLoading, refetch } = useQuery<{
    rows: TBRow[]; total_debit: number; total_credit: number; balanced: boolean
  }>({
    queryKey: ['finance-trial-balance', startDate, endDate],
    queryFn: async () => {
      const { data } = await api.get(`/finance/trial-balance?start_date=${startDate}&end_date=${endDate}`)
      return data
    },
    staleTime: 30 * 1000,
  })

  const filtered = useMemo(() => {
    if (!data?.rows) return []
    const q = search.toLowerCase()
    return q ? data.rows.filter(r => r.ledger.toLowerCase().includes(q) || r.group.toLowerCase().includes(q)) : data.rows
  }, [data, search])

  const NATURE_COLORS: Record<string, string> = {
    asset:     'text-blue-700 bg-blue-50',
    liability: 'text-amber-700 bg-amber-50',
    income:    'text-green-700 bg-green-50',
    expense:   'text-red-700 bg-red-50',
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
        <div className="flex flex-wrap gap-3 items-end">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">From Date</label>
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">To Date</label>
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
          <button onClick={() => refetch()}
            className="px-4 py-1.5 text-xs font-semibold bg-[#002B5B] text-white rounded hover:bg-blue-900">
            Refresh
          </button>
          <div className="flex flex-col gap-1 ml-auto">
            <label className="text-xs text-gray-500">Search Ledger</label>
            <input type="text" value={search} onChange={e => setSearch(e.target.value)} placeholder="Filter by name or group…"
              className="text-xs border border-gray-200 rounded px-2 py-1.5 w-48 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          </div>
        </div>
        {data && (
          <div className={`mt-3 flex items-center gap-2 text-xs font-medium ${data.balanced ? 'text-green-700' : 'text-red-600'}`}>
            <span>{data.balanced ? '✓' : '✗'}</span>
            <span>
              {data.balanced
                ? 'Trial balance tallies — Total Dr = Total Cr'
                : `Difference: ₹${Math.abs(data.total_debit - data.total_credit).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`}
            </span>
          </div>
        )}
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {isLoading ? (
          <div className="p-8 text-center text-gray-400">Loading trial balance…</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Ledger</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">Group</th>
                  <th className="px-4 py-3 text-right font-semibold text-gray-600">Opening Bal</th>
                  <th className="px-4 py-3 text-right font-semibold text-gray-600 bg-blue-50">Period Dr (₹)</th>
                  <th className="px-4 py-3 text-right font-semibold text-gray-600 bg-red-50">Period Cr (₹)</th>
                  <th className="px-4 py-3 text-right font-semibold text-gray-600 border-l-2 border-gray-300">Closing Dr (₹)</th>
                  <th className="px-4 py-3 text-right font-semibold text-gray-600">Closing Cr (₹)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {filtered.map((row, i) => {
                  const closingDr = row.closing >= 0 ? row.closing : 0
                  const closingCr = row.closing < 0 ? Math.abs(row.closing) : 0
                  return (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="px-4 py-2.5 font-medium text-gray-800">{row.ledger}</td>
                      <td className="px-4 py-2.5">
                        <span className={`px-1.5 py-0.5 rounded text-xs ${NATURE_COLORS[row.nature] || 'bg-gray-100 text-gray-600'}`}>
                          {row.group || '—'}
                        </span>
                      </td>
                      <td className={`px-4 py-2.5 text-right ${row.opening_balance >= 0 ? 'text-blue-600' : 'text-red-500'}`}>
                        {row.opening_balance !== 0
                          ? `${row.opening_balance >= 0 ? '' : '('}₹${Math.abs(row.opening_balance).toLocaleString('en-IN', { maximumFractionDigits: 2 })}${row.opening_balance < 0 ? ')' : ''}`
                          : '—'}
                      </td>
                      <td className="px-4 py-2.5 text-right text-blue-700 bg-blue-50/30 font-medium">
                        {row.period_dr > 0 ? `₹${row.period_dr.toLocaleString('en-IN', { maximumFractionDigits: 2 })}` : '—'}
                      </td>
                      <td className="px-4 py-2.5 text-right text-red-600 bg-red-50/30 font-medium">
                        {row.period_cr > 0 ? `₹${row.period_cr.toLocaleString('en-IN', { maximumFractionDigits: 2 })}` : '—'}
                      </td>
                      <td className="px-4 py-2.5 text-right font-semibold text-blue-700 border-l-2 border-gray-200">
                        {closingDr > 0 ? `₹${closingDr.toLocaleString('en-IN', { maximumFractionDigits: 2 })}` : '—'}
                      </td>
                      <td className="px-4 py-2.5 text-right font-semibold text-red-600">
                        {closingCr > 0 ? `₹${closingCr.toLocaleString('en-IN', { maximumFractionDigits: 2 })}` : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
              {data && (
                <tfoot>
                  <tr className="bg-gray-800 text-white font-bold">
                    <td className="px-4 py-3" colSpan={2}>TOTAL</td>
                    <td className="px-4 py-3 text-right">—</td>
                    <td className="px-4 py-3 text-right">
                      ₹{filtered.reduce((s, r) => s + r.period_dr, 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                    </td>
                    <td className="px-4 py-3 text-right">
                      ₹{filtered.reduce((s, r) => s + r.period_cr, 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                    </td>
                    <td className="px-4 py-3 text-right border-l-2 border-gray-600">
                      ₹{filtered.reduce((s, r) => s + (r.closing >= 0 ? r.closing : 0), 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                    </td>
                    <td className="px-4 py-3 text-right">
                      ₹{filtered.reduce((s, r) => s + (r.closing < 0 ? Math.abs(r.closing) : 0), 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                    </td>
                  </tr>
                </tfoot>
              )}
            </table>
            {filtered.length === 0 && !isLoading && (
              <div className="p-8 text-center text-gray-400">No ledger movements found for this period.</div>
            )}
          </div>
        )}
      </div>
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
