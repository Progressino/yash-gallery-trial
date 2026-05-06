import React, { useState, useMemo, useRef, useEffect, useCallback, Fragment } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, Cell,
} from 'recharts'
import api from '../api/client'
import { GlSalesNotesContent } from '../components/finance/GlSalesNotesContent'
import { FinanceUserGuideContent } from '../components/finance/FinanceUserGuideContent'

// ── Types ────────────────────────────────────────────────────────
interface TallyPL {
  id: number; fy: string
  opening_stock: number; purchases: number; direct_expenses: number
  indirect_expenses: number; sales: number; closing_stock: number
  indirect_incomes: number; notes: string; updated_at: string
  cogs: number; gross_profit: number; net_profit: number
}

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
  revenue_source?:   'session' | 'finance_lock'
  cogs_basis_note?:  string
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
  payment_mode:   string
  bank_ledger:    string
  lines:          VoucherLine[]
}
interface SalesUpload {
  id:            number
  platform:      string
  company_name?: string
  seller_gstin?: string
  company_state?: string
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
interface SalesInvoiceRow {
  id: number
  voucher_no: string
  voucher_date: string
  platform: string
  period: string
  invoice_no: string
  order_id: string
  party_name: string
  party_gstin: string
  party_state: string
  ship_to_state: string
  taxable_amount: number
  cgst_amount: number
  sgst_amount: number
  igst_amount: number
  total_amount: number
  net_payable: number
  source_filename: string
  narration?: string
  document_subtype?: 'sales_invoice' | 'sales_credit_memo' | 'upload_summary'
  row_kind?: 'entry' | 'upload_summary'
  sales_upload_id?: number
}

/** BC / D365 Customer Ledger Entries-style row (from posted sales uploads). */
interface CustomerLedgerEntry {
  id: number
  document_date: string
  document_type: string
  document_no: string
  customer_no: string
  customer_name: string
  description: string
  branch_code: string
  taxable_amount: number
  gst_amount: number
  invoice_amount: number
  due_date: string
  gst_customer_type: string
  seller_state_code: string
  seller_gst_reg_no: string
  location_code: string
  location_state_code: string
  gst_jurisdiction_type: string
  external_document_no: string
  location_gst_reg_no: string
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

function downloadCsv(filename: string, rows: (string | number)[][]) {
  const esc = (c: string | number) => `"${String(c).replace(/"/g, '""')}"`
  const body = rows.map(r => r.map(esc).join(',')).join('\n')
  const blob = new Blob(['\ufeff' + body], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  URL.revokeObjectURL(a.href)
}

/** BC CRONUS-style primary nav buckets */
type CronusMega = 'finance' | 'cash' | 'sales' | 'purchasing' | 'india' | 'voucher' | null

function cronusMegaForTab(tab: FinanceTab): CronusMega | 'dash' {
  if (tab === 'dashboard') return 'dash'
  if (tab === 'gstr' || tab === 'gst') return 'india'
  if (tab === 'sales-uploads' || tab === 'sales-invoices' || tab === 'sales-credit-memos' || tab === 'customer-ledger' || tab === 'inventory' || tab === 'revenue') return 'sales'
  if (tab === 'vouchers' || tab === 'daybook' || tab === 'voucher-register') return 'voucher'
  if (tab === 'cash-book' || tab === 'bank-book') return 'cash'
  if (tab === 'expenses') return 'purchasing'
  return 'finance'
}

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

type FinanceTab = 'dashboard' | 'daybook' | 'sales-invoices' | 'sales-credit-memos' | 'customer-ledger' | 'inventory' | 'vouchers' | 'voucher-register' | 'cash-book' | 'bank-book' | 'gstr' | 'pl' | 'gst' | 'expenses' | 'revenue' | 'masters' | 'sales-uploads' | 'help-notes' | 'coa' | 'trial-balance'

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
  bill_no?: string
  bill_date?: string
  ref_number?: string
  party_gstin?: string
  party_state?: string
  supply_type?: string
  meta?: {
    source?: string
    invoice_no?: string
    order_id?: string
    ship_to_state?: string
    platform?: string
    period?: string
    source_filename?: string
    seller_gstin?: string
    seller_company?: string
    seller_state?: string
    line_items?: Array<Record<string, unknown>>
    /** BC / D365-style default dimensions (attribute → code → description). */
    dimension_assignments?: Array<Record<string, unknown>>
  }
}

interface GSTR3BData {
  outward:    { taxable: number; cgst: number; sgst: number; igst: number; total: number }
  inward_itc: { taxable: number; cgst: number; sgst: number; igst: number; total: number }
  net_cgst: number; net_sgst: number; net_igst: number; net_total: number
  breakdown: { voucher_no: string; voucher_date: string; voucher_type: string; party_name: string; taxable_amount: number; cgst_amount: number; sgst_amount: number; igst_amount: number; total_amount: number }[]
}

// ── CRONUS / Business Central–style mega navigation ────────────────
function FinanceCronusNav(props: {
  activeTab: FinanceTab
  setActiveTab: (t: FinanceTab) => void
  megaOpen: CronusMega
  setMegaOpen: (m: CronusMega) => void
  jumpVoucher: (voucherType: string) => void
  jumpMasters: (sub: MastersSubTab) => void
}) {
  const { activeTab, setActiveTab, megaOpen, setMegaOpen, jumpVoucher, jumpMasters } = props
  const panelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!megaOpen) return
    function onDocDown(e: MouseEvent) {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) setMegaOpen(null)
    }
    document.addEventListener('mousedown', onDocDown)
    return () => document.removeEventListener('mousedown', onDocDown)
  }, [megaOpen, setMegaOpen])

  function go(tab: FinanceTab) {
    setActiveTab(tab)
    setMegaOpen(null)
  }

  const hi = (id: CronusMega) =>
    cronusMegaForTab(activeTab) === id
      ? 'text-teal-800 border-b-2 border-teal-600 bg-teal-50/60'
      : 'text-slate-600 border-b-2 border-transparent hover:bg-slate-50'

  function MegaCol({ title, children }: { title: string; children: React.ReactNode }) {
    return (
      <div className="min-w-[140px]">
        <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wide mb-2 border-b border-slate-100 pb-1">{title}</p>
        <div className="flex flex-col gap-1">{children}</div>
      </div>
    )
  }

  function L({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
    return (
      <button
        type="button"
        onClick={onClick}
        className="text-left text-xs text-teal-700 hover:text-teal-950 hover:underline py-1 leading-snug"
      >
        {children}
      </button>
    )
  }

  return (
    <div ref={panelRef} className="bg-white rounded-t-lg border border-slate-200 border-b-0 shadow-sm relative z-40">
      <div className="flex flex-wrap items-stretch gap-0 px-1 border-b border-slate-100">
        <button
          type="button"
          onClick={() => { setMegaOpen(null); setActiveTab('dashboard') }}
          className={`px-3 py-2.5 text-xs font-semibold transition-colors rounded-t-md ${activeTab === 'dashboard' ? 'text-teal-800 border-b-2 border-teal-600 bg-teal-50/80' : 'text-slate-600 hover:bg-slate-50'}`}
        >
          Role Centre
        </button>
        {([
          ['finance', 'Finance'],
          ['cash', 'Cash Management'],
          ['sales', 'Sales'],
          ['purchasing', 'Purchasing'],
          ['india', 'India Taxation'],
          ['voucher', 'Voucher Interface'],
        ] as const).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => setMegaOpen(megaOpen === id ? null : id)}
            className={`px-3 py-2.5 text-xs font-medium transition-colors rounded-t-md flex items-center gap-1 ${hi(id)}`}
          >
            {label}
            <span className="text-[10px] opacity-60" aria-hidden>▾</span>
          </button>
        ))}
      </div>

      {megaOpen && (
        <div className="absolute left-0 right-0 top-full bg-white border border-slate-200 shadow-xl rounded-b-lg p-4 max-h-[min(70vh,520px)] overflow-y-auto">
          {megaOpen === 'finance' && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-left">
              <MegaCol title="General ledger">
                <L onClick={() => jumpVoucher('Journal')}>General Journals</L>
                <L onClick={() => go('coa')}>Chart of Accounts</L>
                <L onClick={() => jumpMasters('ledger-groups')}>G/L groups &amp; categories</L>
              </MegaCol>
              <MegaCol title="Budgets &amp; assets">
                <L onClick={() => go('help-notes')}>G/L budgets (see Help)</L>
                <L onClick={() => go('help-notes')}>Fixed assets (see Help)</L>
                <L onClick={() => go('pl')}>Financial reporting / P&amp;L</L>
              </MegaCol>
              <MegaCol title="Analysis">
                <L onClick={() => go('revenue')}>Sales analysis — platform revenue</L>
                <L onClick={() => go('expenses')}>Purchase / expense register</L>
                <L onClick={() => go('help-notes')}>Inventory analysis (ops)</L>
              </MegaCol>
              <MegaCol title="Tax &amp; local">
                <L onClick={() => go('gstr')}>GST returns — GSTR-3B</L>
                <L onClick={() => go('gst')}>GST summary (Amazon MTR)</L>
                <L onClick={() => go('help-notes')}>VAT / e-invoice notes</L>
              </MegaCol>
              <MegaCol title="Setup">
                <L onClick={() => go('masters')}>Currencies &amp; masters</L>
                <L onClick={() => jumpMasters('voucher-types')}>Voucher types</L>
                <L onClick={() => go('trial-balance')}>Trial balance</L>
              </MegaCol>
              <MegaCol title="Dimensions">
                <L onClick={() => go('vouchers')}>Cost centre on lines</L>
                <L onClick={() => go('help-notes')}>Statistical / allocation (Help)</L>
                <L onClick={() => go('help-notes')}>Microsoft Learn mapping</L>
              </MegaCol>
            </div>
          )}
          {megaOpen === 'cash' && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
              <MegaCol title="Cash flow">
                <L onClick={() => go('pl')}>Cash view — P&amp;L / expenses</L>
                <L onClick={() => go('trial-balance')}>Bank &amp; cash ledgers (T/B)</L>
                <L onClick={() => go('help-notes')}>Cash flow forecasts (Help)</L>
              </MegaCol>
              <MegaCol title="Journals">
                <L onClick={() => jumpVoucher('Receipt')}>Cash receipt journals</L>
                <L onClick={() => jumpVoucher('Payment')}>Payment journals</L>
                <L onClick={() => jumpVoucher('Contra')}>Contra (bank ↔ cash)</L>
              </MegaCol>
              <MegaCol title="Bank">
                <L onClick={() => jumpMasters('ledgers')}>Bank accounts (ledgers)</L>
                <L onClick={() => go('bank-book')}>Bank Book</L>
                <L onClick={() => go('vouchers')}>Payment reconciliation</L>
              </MegaCol>
              <MegaCol title="Terms">
                <L onClick={() => jumpMasters('ledgers')}>Payment terms (party ledgers)</L>
                <L onClick={() => go('expenses')}>Deposits &amp; batches</L>
              </MegaCol>
              <MegaCol title="Reconciliation">
                <L onClick={() => go('trial-balance')}>Bank reconciliation — T/B</L>
                <L onClick={() => go('daybook')}>Find entries</L>
              </MegaCol>
            </div>
          )}
          {megaOpen === 'sales' && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
              <MegaCol title="Master data">
                <L onClick={() => jumpMasters('ledgers')}>Customers (debtor ledgers)</L>
                <L onClick={() => go('help-notes')}>Items (Item Master — ops)</L>
              </MegaCol>
              <MegaCol title="Orders &amp; quotes">
                <L onClick={() => go('sales-invoices')}>Sales invoices</L>
                <L onClick={() => go('sales-credit-memos')}>Sales credit memos</L>
                <L onClick={() => go('customer-ledger')}>Customer ledger</L>
                <L onClick={() => go('inventory')}>Inventory (shipments / returns)</L>
                <L onClick={() => go('sales-uploads')}>Sales uploads</L>
                <L onClick={() => go('daybook')}>Posted sales — Day Book</L>
              </MegaCol>
              <MegaCol title="Returns">
                <L onClick={() => go('sales-credit-memos')}>Sales credit memos (returns)</L>
                <L onClick={() => go('customer-ledger')}>Customer ledger (BC-style log)</L>
                <L onClick={() => go('revenue')}>Returns in platform revenue</L>
              </MegaCol>
              <MegaCol title="Posted documents">
                <L onClick={() => go('daybook')}>Posted invoices (by date)</L>
                <L onClick={() => go('gstr')}>GST on outward supplies</L>
              </MegaCol>
            </div>
          )}
          {megaOpen === 'purchasing' && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
              <MegaCol title="Master data">
                <L onClick={() => jumpMasters('ledgers')}>Vendors (creditor ledgers)</L>
                <L onClick={() => go('help-notes')}>Incoming docs (Help)</L>
              </MegaCol>
              <MegaCol title="Documents">
                <L onClick={() => jumpVoucher('Purchase Invoice')}>Purchase invoices</L>
                <L onClick={() => jumpVoucher('Expense')}>Expense vouchers</L>
              </MegaCol>
              <MegaCol title="Posted">
                <L onClick={() => go('daybook')}>Posted purchases — Day Book</L>
                <L onClick={() => go('voucher-register')}>Voucher register</L>
              </MegaCol>
            </div>
          )}
          {megaOpen === 'india' && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <MegaCol title="Common setup">
                <L onClick={() => jumpMasters('gst-classifications')}>GST HSN / rates</L>
                <L onClick={() => jumpMasters('tds-sections')}>TDS sections</L>
                <L onClick={() => jumpMasters('ledgers')}>Party GSTIN / ledgers</L>
              </MegaCol>
              <MegaCol title="Goods &amp; Services Tax">
                <L onClick={() => go('gstr')}>GSTR-3B (monthly)</L>
                <L onClick={() => go('gst')}>GST summary — MTR</L>
                <L onClick={() => go('sales-uploads')}>Sales uploads / reconciliation</L>
              </MegaCol>
              <MegaCol title="TDS">
                <L onClick={() => jumpMasters('tds-sections')}>TDS master</L>
                <L onClick={() => go('vouchers')}>TDS on vouchers</L>
              </MegaCol>
              <MegaCol title="TCS">
                <L onClick={() => jumpMasters('ledgers')}>TCS on ledgers</L>
                <L onClick={() => go('help-notes')}>TCS process (Help / Notes)</L>
              </MegaCol>
            </div>
          )}
          {megaOpen === 'voucher' && (
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
              <MegaCol title="Payments">
                <L onClick={() => jumpVoucher('Payment')}>Bank payment voucher</L>
                <L onClick={() => jumpVoucher('Payment')}>Cash payment voucher</L>
              </MegaCol>
              <MegaCol title="Receipts">
                <L onClick={() => jumpVoucher('Receipt')}>Cash receipt voucher</L>
                <L onClick={() => jumpVoucher('Receipt')}>Bank receipt voucher</L>
              </MegaCol>
              <MegaCol title="Other">
                <L onClick={() => jumpVoucher('Journal')}>Journal voucher</L>
                <L onClick={() => jumpVoucher('Contra')}>Contra voucher</L>
              </MegaCol>
              <MegaCol title="Books">
                <L onClick={() => go('daybook')}>Day Book</L>
                <L onClick={() => go('cash-book')}>Cash Book</L>
                <L onClick={() => go('bank-book')}>Bank Book</L>
                <L onClick={() => go('trial-balance')}>Ledger / T/B</L>
              </MegaCol>
              <MegaCol title="Register">
                <L onClick={() => go('voucher-register')}>Voucher register</L>
              </MegaCol>
            </div>
          )}
        </div>
      )}

      <p className="px-3 py-1.5 text-[10px] text-slate-400 border-t border-slate-50 bg-slate-50/50">
        BC demo menus mapped to this app — items marked “Help” are definitions or Microsoft Learn pointers in <strong>Help / Notes</strong>.
      </p>
    </div>
  )
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
  const [cronusMegaOpen, setCronusMegaOpen] = useState<CronusMega>(null)
  const [voucherPreset, setVoucherPreset] = useState<{ n: number; type: string } | null>(null)
  const [mastersJump, setMastersJump] = useState<{ n: number; sub: MastersSubTab; ledgerSearch?: string } | null>(null)
  const [tbSearchJump, setTbSearchJump] = useState<{ n: number; q: string } | null>(null)
  const [dateStart,    setDateStart]    = useState(() => daysAgo(90))
  const [dateEnd,      setDateEnd]      = useState(TODAY)
  const [activePreset, setActivePreset] = useState<string>('90D')

  function jumpVoucher(type: string) {
    setVoucherPreset({ n: Date.now(), type })
    setActiveTab('vouchers')
    setCronusMegaOpen(null)
  }
  function jumpMasters(sub: MastersSubTab) {
    setMastersJump({ n: Date.now(), sub })
    setActiveTab('masters')
    setCronusMegaOpen(null)
  }
  const openTrialBalanceForParty = useCallback((q: string) => {
    const s = q.trim()
    if (!s) return
    setTbSearchJump({ n: Date.now(), q: s })
    setActiveTab('trial-balance')
    setCronusMegaOpen(null)
  }, [])
  const openMastersLedgersForParty = useCallback((q: string) => {
    const s = q.trim()
    if (!s) return
    setMastersJump({ n: Date.now(), sub: 'ledgers', ledgerSearch: s })
    setActiveTab('masters')
    setCronusMegaOpen(null)
  }, [])
  /** Finance Sales Uploads (locked) vs operational session (Upload page — Dashboard / PO) */
  const [revenueSource, setRevenueSource] = useState<'finance_lock' | 'session'>('finance_lock')
  const [financeCompany, setFinanceCompany] = useState('')

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

  const plAndRevQ = useMemo(() => {
    const p = new URLSearchParams(dateQ)
    p.set('revenue_source', revenueSource)
    if (financeCompany && revenueSource === 'finance_lock') p.set('finance_company', financeCompany)
    return p.toString()
  }, [dateQ, revenueSource, financeCompany])

  const { data: financeUploadRows = [] } = useQuery<SalesUpload[]>({
    queryKey: ['finance-company-options'],
    queryFn: async () => { const { data } = await api.get('/finance/sales-uploads'); return data },
    staleTime: 5 * 60 * 1000,
  })
  const financeCompanyOptions = useMemo(() => {
    const seen = new Set<string>()
    const out: string[] = []
    for (const r of financeUploadRows) {
      const c = (r.company_name || '').trim()
      const g = (r.seller_gstin || '').trim()
      if (c && !seen.has(c)) { seen.add(c); out.push(c) }
      if (g && !seen.has(g)) { seen.add(g); out.push(g) }
    }
    return out.sort((a, b) => a.localeCompare(b))
  }, [financeUploadRows])

  // ── Queries ──────────────────────────────────────────────────
  const { data: pl, isLoading: loadPL } = useQuery<PLStatement>({
    queryKey: ['finance-pl', dateStart, dateEnd, revenueSource],
    queryFn:  async () => { const { data } = await api.get(`/finance/pl?${plAndRevQ}`); return data },
    staleTime: 2 * 60 * 1000,
  })

  const { data: tallyRows = [], refetch: refetchTally } = useQuery<TallyPL[]>({
    queryKey: ['finance-tally-pl'],
    queryFn:  async () => { const { data } = await api.get('/finance/tally-pl'); return data },
    staleTime: Infinity,
  })

  const { data: gst, isLoading: loadGST } = useQuery<GSTSummary>({
    queryKey: ['finance-gst', dateStart, dateEnd],
    queryFn:  async () => { const { data } = await api.get(`/finance/gst?${dateQ}`); return data },
    staleTime: 2 * 60 * 1000,
    enabled:   activeTab === 'gst',
  })

  const { data: platformRev, isLoading: loadRev } = useQuery<PlatformRev[]>({
    queryKey: ['finance-platform-rev', dateStart, dateEnd, revenueSource],
    queryFn:  async () => { const { data } = await api.get(`/finance/platform-revenue?${plAndRevQ}`); return data },
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
    <div className="max-w-[1400px] mx-auto space-y-4 pb-10 px-1 sm:px-0">
      {/* Header — BC-inspired workspace title */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-2">
        <div>
          <h2 className="text-xl sm:text-2xl font-semibold text-slate-800 tracking-tight">Finance</h2>
          <p className="text-slate-500 text-xs sm:text-sm mt-0.5">
            Activities, posting, and reports
          </p>
        </div>
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
          {(activeTab === 'pl' || activeTab === 'revenue') && (
            <div className="flex items-center gap-2 border-l border-gray-200 pl-3 ml-1">
              <label className="text-xs text-gray-500 whitespace-nowrap">Revenue basis</label>
              <select
                value={revenueSource}
                onChange={e => setRevenueSource(e.target.value as 'finance_lock' | 'session')}
                className="text-xs border border-gray-200 rounded px-2 py-1.5 text-gray-700 max-w-[280px] focus:outline-none focus:ring-1 focus:ring-blue-300"
              >
                <option value="finance_lock">Finance Sales Uploads (locked DB — separate from Dashboard)</option>
                <option value="session">Operational (Upload page — same as Dashboard / PO)</option>
              </select>
              {revenueSource === 'finance_lock' && (
                <select
                  value={financeCompany}
                  onChange={e => setFinanceCompany(e.target.value)}
                  className="text-xs border border-gray-200 rounded px-2 py-1.5 text-gray-700 max-w-[280px] focus:outline-none focus:ring-1 focus:ring-blue-300"
                >
                  <option value="">All companies / GSTIN</option>
                  {financeCompanyOptions.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              )}
            </div>
          )}
        </div>
      )}

      <FinanceCronusNav
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        megaOpen={cronusMegaOpen}
        setMegaOpen={setCronusMegaOpen}
        jumpVoucher={jumpVoucher}
        jumpMasters={jumpMasters}
      />

      <div className="bg-white rounded-b-lg border border-t-0 border-slate-200 shadow-sm overflow-hidden">
        {(activeTab === 'gstr' || activeTab === 'gst' || activeTab === 'sales-uploads' || activeTab === 'sales-invoices' || activeTab === 'sales-credit-memos' || activeTab === 'customer-ledger' || activeTab === 'inventory') && (
          <div className="px-3 sm:px-5 py-2 flex flex-wrap gap-x-1 gap-y-1 items-center border-b border-teal-100 bg-gradient-to-r from-teal-50/90 to-cyan-50/50">
            <span className="text-[10px] font-bold text-teal-900 uppercase tracking-wide mr-2">India Taxation</span>
            <button type="button" onClick={() => setActiveTab('gstr')} className={`text-xs font-medium px-2 py-1 rounded ${activeTab === 'gstr' ? 'bg-teal-600 text-white' : 'text-teal-800 hover:bg-teal-100'}`}>GSTR-3B</button>
            <button type="button" onClick={() => setActiveTab('gst')} className={`text-xs font-medium px-2 py-1 rounded ${activeTab === 'gst' ? 'bg-teal-600 text-white' : 'text-teal-800 hover:bg-teal-100'}`}>GST summary</button>
            <button type="button" onClick={() => setActiveTab('sales-invoices')} className={`text-xs font-medium px-2 py-1 rounded ${activeTab === 'sales-invoices' ? 'bg-teal-600 text-white' : 'text-teal-800 hover:bg-teal-100'}`}>Sales invoices</button>
            <button type="button" onClick={() => setActiveTab('sales-credit-memos')} className={`text-xs font-medium px-2 py-1 rounded ${activeTab === 'sales-credit-memos' ? 'bg-teal-600 text-white' : 'text-teal-800 hover:bg-teal-100'}`}>Sales credit memos</button>
            <button type="button" onClick={() => setActiveTab('customer-ledger')} className={`text-xs font-medium px-2 py-1 rounded ${activeTab === 'customer-ledger' ? 'bg-teal-600 text-white' : 'text-teal-800 hover:bg-teal-100'}`}>Customer ledger</button>
            <button type="button" onClick={() => setActiveTab('inventory')} className={`text-xs font-medium px-2 py-1 rounded ${activeTab === 'inventory' ? 'bg-teal-600 text-white' : 'text-teal-800 hover:bg-teal-100'}`}>Inventory</button>
            <button type="button" onClick={() => setActiveTab('sales-uploads')} className={`text-xs font-medium px-2 py-1 rounded ${activeTab === 'sales-uploads' ? 'bg-teal-600 text-white' : 'text-teal-800 hover:bg-teal-100'}`}>Sales uploads</button>
            <button type="button" onClick={() => jumpMasters('tds-sections')} className="text-xs font-medium px-2 py-1 rounded text-teal-800 hover:bg-teal-100">TDS masters</button>
            <button type="button" onClick={() => jumpMasters('gst-classifications')} className="text-xs font-medium px-2 py-1 rounded text-teal-800 hover:bg-teal-100">GST classifications</button>
            <span className="text-[10px] text-teal-700/80 ml-auto hidden sm:inline">GSTR-1: use government portal; outward detail here via Day Book / uploads.</span>
          </div>
        )}

        <div className="px-3 sm:px-5 py-2 flex flex-wrap gap-2 items-center border-b border-slate-100 bg-slate-50/40 text-[11px]">
          <span className="text-slate-500 font-medium">Open:</span>
          {([
            ['daybook', 'Day Book'],
            ['sales-invoices', 'Sales Invoices'],
            ['sales-credit-memos', 'Sales credit memos'],
            ['customer-ledger', 'Customer ledger'],
            ['inventory', 'Inventory'],
            ['vouchers', 'Vouchers'],
            ['voucher-register', 'Voucher Register'],
            ['cash-book', 'Cash Book'],
            ['bank-book', 'Bank Book'],
            ['coa', 'COA'],
            ['trial-balance', 'Trial Balance'],
            ['pl', 'P&L'],
            ['expenses', 'Expenses'],
            ['revenue', 'Revenue'],
            ['masters', 'Masters'],
            ['help-notes', 'Help'],
          ] as [FinanceTab, string][]).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => { setCronusMegaOpen(null); setActiveTab(id) }}
              className={`px-2 py-0.5 rounded font-medium transition-colors ${
                activeTab === id ? 'bg-slate-700 text-white' : 'text-teal-700 hover:bg-slate-100'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="px-3 sm:px-5 py-5">

      {/* ── Tab: P&L ── */}
      {activeTab === 'pl' && (
        <div className="space-y-4">
          {pl?.revenue_source === 'finance_lock' && pl?.cogs_basis_note && (
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 text-xs text-slate-700 flex gap-2 items-start">
              <span className="mt-0.5">📌</span>
              <div>
                <p className="font-semibold text-slate-800">Revenue from Finance Sales Uploads</p>
                <p className="mt-0.5 text-slate-600">{pl.cogs_basis_note}</p>
              </div>
            </div>
          )}
          {pl?.revenue_source === 'session' && (
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 text-xs text-slate-700">
              <strong>Operational basis:</strong> Uses Upload page session data. Finance Sales Uploads are available across finance tabs by default with Revenue basis set to Finance Sales Uploads.
            </div>
          )}
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

          {/* ── Tally / Accountant P&L ── */}
          <TallyPLSection tallyRows={tallyRows} refetch={refetchTally} appPL={pl ?? null} />
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
          {revenueSource === 'finance_lock' && (
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 text-xs text-slate-700">
              Totals below are auto-synced from <strong>Sales Uploads</strong> (finance.db) for months overlapping the date range, and flow across Finance Dashboard / Day Book / Vouchers / GSTR3B / P&amp;L tabs.
            </div>
          )}
          {revenueSource === 'session' && (
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 text-xs text-slate-700">
              <strong>Operational:</strong> same marketplace files as Dashboard / PO. Finance Sales Uploads are ignored here.
            </div>
          )}
          {!loadRev && (platformRev ?? []).length === 0 && (
            <div className="bg-white rounded-xl border border-dashed border-gray-300 p-8 text-center text-sm text-gray-500">
              {revenueSource === 'finance_lock'
                ? 'No finance-locked sales rows for this period. Upload under Sales Uploads, or choose Operational to see Dashboard data.'
                : 'No operational revenue for this period — load marketplace files on the Upload page.'}
            </div>
          )}
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
              <div className="h-48 flex flex-col items-center justify-center text-gray-400 text-sm px-4 text-center gap-2">
                <span>{revenueSource === 'finance_lock'
                  ? 'No Finance Sales Uploads in this range. Add files under Sales Uploads, or switch Revenue basis to Operational.'
                  : 'No platform data loaded on the Upload page for this range.'}</span>
              </div>
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
      {activeTab === 'daybook' && (
        <DaybookTab
          openTrialBalanceForParty={openTrialBalanceForParty}
          openMastersLedgersForParty={openMastersLedgersForParty}
        />
      )}

      {/* ── Tab: Sales Invoices ── */}
      {activeTab === 'sales-invoices' && (
        <SalesInvoicesTab
          mode="sales"
          openTrialBalanceForParty={openTrialBalanceForParty}
          openMastersLedgersForParty={openMastersLedgersForParty}
        />
      )}

      {activeTab === 'sales-credit-memos' && (
        <SalesInvoicesTab
          mode="credit_memo"
          openTrialBalanceForParty={openTrialBalanceForParty}
          openMastersLedgersForParty={openMastersLedgersForParty}
        />
      )}

      {activeTab === 'customer-ledger' && <CustomerLedgerEntriesTab />}
      {activeTab === 'inventory' && <FinanceInventoryTab />}

      {/* ── Tab: GSTR3B ── */}
      {activeTab === 'gstr' && <GSTR3BTab />}

      {/* ── Tab: Vouchers ── */}
      {activeTab === 'vouchers' && <VouchersTab voucherPreset={voucherPreset} />}

      {/* ── Tab: Voucher Register ── */}
      {activeTab === 'voucher-register' && <VoucherRegisterTab />}

      {/* ── Tab: Cash / Bank Books ── */}
      {activeTab === 'cash-book' && <CashBankBookTab mode='cash' />}
      {activeTab === 'bank-book' && <CashBankBookTab mode='bank' />}

      {/* ── Tab: Masters ── */}
      {activeTab === 'masters' && <MastersTab mastersJump={mastersJump} />}

      {/* ── Tab: Sales Uploads ── */}
      {activeTab === 'sales-uploads' && <SalesUploadsTab />}

      {/* ── Tab: Help / Notes ── */}
      {activeTab === 'help-notes' && <FinanceHelpNotesTab />}

      {/* ── Tab: Chart of Accounts ── */}
      {activeTab === 'coa' && <ChartOfAccountsTab />}

      {/* ── Tab: Trial Balance ── */}
      {activeTab === 'trial-balance' && <TrialBalanceTab searchJump={tbSearchJump} />}
        </div>
      </div>
    </div>
  )
}

function FinanceHelpNotesTab() {
  return (
    <div className="space-y-5">
      <FinanceUserGuideContent />
      <GlSalesNotesContent showAccountantChecklist />
    </div>
  )
}

// ── Dashboard Tab — Business Central–style activities (Role Centre pattern) ──
function DashboardTab({ onNavigate }: { onNavigate: (tab: FinanceTab) => void }) {
  const todayStr = toIso(new Date())
  const monthStart = useMemo(() => {
    const n = new Date()
    return `${n.getFullYear()}-${String(n.getMonth() + 1).padStart(2, '0')}-01`
  }, [])

  const { data: todayVouchers = [], isLoading: loadToday } = useQuery<DaybookVoucher[]>({
    queryKey: ['finance-daybook-today'],
    queryFn: async () => { const { data } = await api.get(`/finance/daybook?date=${todayStr}`); return data },
    staleTime: 30 * 1000,
  })

  const { data: monthVouchers = [], isLoading: loadMonthVouchers } = useQuery<Voucher[]>({
    queryKey: ['finance-vouchers-dashboard-month', monthStart, todayStr],
    queryFn: async () => {
      const { data } = await api.get(`/finance/vouchers?start_date=${monthStart}&end_date=${todayStr}`)
      return data
    },
    staleTime: 60 * 1000,
  })

  const { data: uploads = [], isLoading: loadUploads } = useQuery<SalesUpload[]>({
    queryKey: ['finance-dashboard-uploads'],
    queryFn: async () => { const { data } = await api.get('/finance/sales-uploads'); return data },
    staleTime: 60 * 1000,
  })

  const { data: platformMonth = [], isLoading: loadPlat } = useQuery<PlatformRev[]>({
    queryKey: ['finance-dashboard-platform-m', monthStart, todayStr],
    queryFn: async () => {
      const p = new URLSearchParams({ start_date: monthStart, end_date: todayStr, revenue_source: 'finance_lock' })
      const { data } = await api.get(`/finance/platform-revenue?${p}`)
      return data
    },
    staleTime: 60 * 1000,
  })

  const { data: monthExpenses } = useQuery<Expense[]>({
    queryKey: ['finance-expenses-month'],
    queryFn: async () => {
      const { data } = await api.get(`/finance/expenses?start_date=${monthStart}&end_date=${todayStr}`)
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

  const salesThisMonth = useMemo(
    () => (platformMonth ?? []).filter(p => p.loaded).reduce((s, p) => s + p.net_revenue, 0),
    [platformMonth],
  )
  const salesEntryToday = useMemo(
    () => todayVouchers.filter(v => String(v.voucher_no).startsWith('SUE-')).length,
    [todayVouchers],
  )
  const debtorTotal = useMemo(
    () => (ledgerBalances?.debtors ?? []).reduce((s, d) => s + d.balance, 0),
    [ledgerBalances],
  )
  const creditorTotal = useMemo(
    () => (ledgerBalances?.creditors ?? []).reduce((s, c) => s + c.balance, 0),
    [ledgerBalances],
  )

  const expenseVouchersMt = monthVouchers.filter(v => ['Expense', 'JWO Payment', 'Purchase Invoice'].includes(v.voucher_type)).length
  const paymentCountMt = monthVouchers.filter(v => v.voucher_type === 'Payment').length
  const receiptCountMt = monthVouchers.filter(v => v.voucher_type === 'Receipt').length
  const journalCountMt = monthVouchers.filter(v => v.voucher_type === 'Journal').length
  const monthVoucherTotal = loadMonthVouchers ? null : monthVouchers.length

  const QUICK_LINKS: { label: string; tab: FinanceTab }[] = [
    { label: 'Chart of Accounts', tab: 'coa' },
    { label: 'Day Book', tab: 'daybook' },
    { label: 'Sales Invoices', tab: 'sales-invoices' },
    { label: 'Inventory (Finance)', tab: 'inventory' },
    { label: 'Sales credit memos', tab: 'sales-credit-memos' },
    { label: 'Customer ledger', tab: 'customer-ledger' },
    { label: 'Vouchers', tab: 'vouchers' },
    { label: 'Ledgers & masters', tab: 'masters' },
    { label: 'Sales Uploads', tab: 'sales-uploads' },
    { label: 'Trial Balance', tab: 'trial-balance' },
    { label: 'GSTR-3B', tab: 'gstr' },
    { label: 'P&L', tab: 'pl' },
  ]

  function KpiCard(props: {
    title: string
    value: React.ReactNode
    sub?: string
    alert?: boolean
    loading?: boolean
    onSeeMore?: () => void
  }) {
    const { title, value, sub, alert, loading, onSeeMore } = props
    return (
      <div
        className={`relative bg-white rounded-lg border border-slate-200 shadow-sm p-4 min-h-[108px] flex flex-col ${
          alert ? 'border-l-4 border-l-red-500' : 'border-l-4 border-l-teal-500'
        }`}
      >
        <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">{title}</p>
        {loading ? (
          <div className="h-8 bg-slate-100 rounded animate-pulse mt-2 flex-1" />
        ) : (
          <p className="text-2xl sm:text-[26px] font-semibold text-slate-900 mt-1 tabular-nums leading-tight">{value}</p>
        )}
        {sub && <p className="text-[11px] text-slate-400 mt-1">{sub}</p>}
        {onSeeMore && (
          <button
            type="button"
            onClick={onSeeMore}
            className="mt-auto pt-2 text-left text-[11px] font-medium text-teal-700 hover:text-teal-900 hover:underline"
          >
            See more →
          </button>
        )}
      </div>
    )
  }

  function CueTile({ label, count, onClick }: { label: string; count: number | null; onClick: () => void }) {
    return (
      <button
        type="button"
        onClick={onClick}
        className="flex flex-col items-center justify-center rounded-md bg-teal-600 hover:bg-teal-700 text-white p-4 min-h-[88px] shadow-sm transition-colors text-center focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-400 focus-visible:ring-offset-2"
      >
        <span className="text-2xl font-bold tabular-nums leading-none">{count === null ? '…' : count}</span>
        <span className="text-[10px] sm:text-xs font-medium mt-2 leading-snug opacity-95 px-1">{label}</span>
      </button>
    )
  }

  return (
    <div className="space-y-6">
      <div className="rounded-md border border-sky-200 bg-sky-50/90 px-3 py-2 text-[11px] text-sky-900 flex flex-wrap gap-x-4 gap-y-1 items-center">
        <span className="font-semibold">Activities</span>
        <span className="text-sky-800/90">
          Layout inspired by Dynamics 365 <strong>Business Central</strong> Role Centre — tiles drill through to lists and reports.
        </span>
      </div>

      <div className="rounded-lg border border-teal-200 bg-gradient-to-r from-teal-50/80 to-cyan-50/50 px-3 py-3 space-y-2 shadow-sm">
        <div className="flex flex-wrap items-center gap-2 justify-between">
          <p className="text-[11px] font-bold text-teal-900 uppercase tracking-wide">India Taxation (CRONUS-style)</p>
          <div className="flex flex-wrap gap-1 justify-end">
            <button type="button" onClick={() => onNavigate('gstr')} className="text-xs px-2.5 py-1 rounded-md bg-teal-600 text-white font-semibold hover:bg-teal-700 shadow-sm">
              GSTR-3B
            </button>
            <button type="button" onClick={() => onNavigate('gst')} className="text-xs px-2.5 py-1 rounded-md bg-white border border-teal-300 text-teal-900 font-medium hover:bg-teal-50">
              GST summary
            </button>
            <button type="button" onClick={() => onNavigate('sales-uploads')} className="text-xs px-2.5 py-1 rounded-md bg-white border border-teal-300 text-teal-900 font-medium hover:bg-teal-50">
              Sales uploads
            </button>
            <button type="button" onClick={() => onNavigate('masters')} className="text-xs px-2.5 py-1 rounded-md bg-white border border-teal-300 text-teal-900 font-medium hover:bg-teal-50">
              TDS / GST masters
            </button>
          </div>
        </div>
        <div className="flex flex-wrap gap-2 items-center pt-2 border-t border-teal-100/80">
          <span className="text-[10px] font-bold text-teal-900 uppercase tracking-wide">Export</span>
          <button
            type="button"
            onClick={() => {
              const rows: (string | number)[][] = [['voucher_no', 'voucher_type', 'party', 'net_payable', 'taxable_amount']]
              for (const v of todayVouchers) {
                rows.push([v.voucher_no, v.voucher_type, v.party_name || '', v.net_payable, v.taxable_amount])
              }
              downloadCsv(`finance-daybook-${todayStr}.csv`, rows)
            }}
            className="text-[11px] font-medium px-2 py-1 rounded border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
          >
            Today&apos;s day book (CSV)
          </button>
          <button
            type="button"
            onClick={() => {
              const br = gstr3b?.breakdown ?? []
              const rows: (string | number)[][] = [['voucher_no', 'voucher_date', 'voucher_type', 'party_name', 'taxable_amount', 'cgst', 'sgst', 'igst', 'total_amount']]
              for (const r of br) {
                rows.push([r.voucher_no, r.voucher_date, r.voucher_type, r.party_name, r.taxable_amount, r.cgst_amount, r.sgst_amount, r.igst_amount, r.total_amount])
              }
              downloadCsv(`gstr3b-breakdown-mtd-${todayStr.slice(0, 7)}.csv`, rows)
            }}
            className="text-[11px] font-medium px-2 py-1 rounded border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
          >
            GSTR-3B breakdown (CSV)
          </button>
          <button
            type="button"
            onClick={() => {
              const rows: (string | number)[][] = [['platform', 'net_revenue', 'gross_revenue', 'returns_value']]
              for (const p of platformMonth.filter(x => x.loaded)) {
                rows.push([p.platform, p.net_revenue, p.gross_revenue, p.returns_value])
              }
              downloadCsv(`platform-revenue-mtd-${todayStr.slice(0, 7)}.csv`, rows)
            }}
            className="text-[11px] font-medium px-2 py-1 rounded border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
          >
            Locked sales by platform (CSV)
          </button>
        </div>
        <p className="text-[10px] text-teal-800/80">GSTR-1 and other statutory filings: prepare in the government portal; use Day Book and uploads here for outward line detail.</p>
      </div>

      <div className="flex flex-col xl:flex-row gap-6">
        <div className="flex-1 space-y-5 min-w-0">
          <div>
            <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide mb-2">Quick access</p>
            <div className="flex flex-wrap items-center gap-x-1 gap-y-1 text-xs">
              {QUICK_LINKS.map((link, i) => (
                <Fragment key={link.tab}>
                  {i > 0 && <span className="text-slate-300 select-none" aria-hidden>·</span>}
                  <button
                    type="button"
                    onClick={() => onNavigate(link.tab)}
                    className="text-teal-700 hover:text-teal-900 hover:underline font-medium px-0.5"
                  >
                    {link.label}
                  </button>
                </Fragment>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-slate-200 bg-gradient-to-r from-slate-50 to-white px-4 py-3">
            <h3 className="text-sm font-semibold text-slate-800">Finance workspace</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              KPIs and cues update from your finance database. “Sales this month” uses <strong>Finance Sales Uploads</strong> (locked), not the operational Upload page.
            </p>
          </div>

          <div>
            <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide mb-2">Insights</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
              <KpiCard
                title="Sales this month (locked)"
                value={fmt(Math.round(salesThisMonth))}
                sub="Net revenue · uploads basis"
                loading={loadPlat}
                onSeeMore={() => onNavigate('revenue')}
              />
              <KpiCard
                title="Outstanding receivables"
                value={fmt(Math.round(debtorTotal))}
                sub="Top debtor ledgers (sum)"
                alert={debtorTotal > 0}
                loading={!ledgerBalances}
                onSeeMore={() => onNavigate('trial-balance')}
              />
              <KpiCard
                title="Outstanding payables"
                value={fmt(Math.round(creditorTotal))}
                sub="Top creditor ledgers (sum)"
                alert={creditorTotal > 0}
                loading={!ledgerBalances}
                onSeeMore={() => onNavigate('trial-balance')}
              />
              <KpiCard
                title="GST net (MTD)"
                value={fmt(Math.round(gstPayable))}
                sub="Current month · GSTR-3B basis"
                alert={gstPayable > 0}
                onSeeMore={() => onNavigate('gstr')}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide">Today&apos;s vouchers</p>
              {loadToday ? (
                <div className="h-7 bg-slate-100 rounded animate-pulse mt-2" />
              ) : (
                <p className="text-xl font-bold text-slate-900 mt-1 tabular-nums">{todayCount}</p>
              )}
              <button type="button" className="text-[11px] font-medium text-teal-700 mt-1 hover:underline" onClick={() => onNavigate('daybook')}>
                See more →
              </button>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide">Today&apos;s net</p>
              {loadToday ? (
                <div className="h-7 bg-slate-100 rounded animate-pulse mt-2" />
              ) : (
                <p className="text-xl font-bold text-slate-900 mt-1 tabular-nums">{fmt(todayTotal)}</p>
              )}
              <p className="text-[11px] text-slate-400 mt-1">All voucher types</p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm border-l-4 border-l-rose-500">
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide">Expenses MTD</p>
              <p className="text-xl font-bold text-rose-700 mt-1 tabular-nums">{fmt(Math.round(monthExpTotal))}</p>
              <button type="button" className="text-[11px] font-medium text-teal-700 mt-1 hover:underline" onClick={() => onNavigate('expenses')}>
                See more →
              </button>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide">Sales uploads</p>
              {loadUploads ? (
                <div className="h-7 bg-slate-100 rounded animate-pulse mt-2" />
              ) : (
                <p className="text-xl font-bold text-teal-700 mt-1 tabular-nums">{uploads.length}</p>
              )}
              <button type="button" className="text-[11px] font-medium text-teal-700 mt-1 hover:underline" onClick={() => onNavigate('sales-uploads')}>
                See more →
              </button>
            </div>
          </div>
        </div>

        <div className="w-full xl:w-52 shrink-0 space-y-3">
          <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">Actions</p>
          <div className="rounded-lg border border-slate-200 bg-white divide-y divide-slate-100 overflow-hidden shadow-sm">
            {[
              { label: 'Find entries…', tab: 'daybook' as FinanceTab },
              { label: 'New voucher', tab: 'vouchers' as FinanceTab },
              { label: 'Reports — P&L', tab: 'pl' as FinanceTab },
              { label: 'Platform revenue', tab: 'revenue' as FinanceTab },
              { label: 'Help / Notes', tab: 'help-notes' as FinanceTab },
            ].map(a => (
              <button
                key={a.label}
                type="button"
                onClick={() => onNavigate(a.tab)}
                className="w-full text-left px-3 py-2.5 text-xs font-medium text-slate-700 hover:bg-teal-50 hover:text-teal-900 transition-colors"
              >
                {a.label}
              </button>
            ))}
          </div>
          <p className="text-[10px] text-slate-400 leading-snug">
            Use <strong>Find entries</strong> to open Day Book and pick any date, similar to filtering lists in BC.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <p className="text-sm font-semibold text-slate-800">Vouchers this month</p>
          <p className="text-[11px] text-slate-500 mb-3">Cue tiles — counts of posted voucher types (drill to Vouchers).</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            <CueTile label="All vouchers" count={monthVoucherTotal} onClick={() => onNavigate('vouchers')} />
            <CueTile label="Expense & purchase" count={loadMonthVouchers ? null : expenseVouchersMt} onClick={() => onNavigate('vouchers')} />
            <CueTile label="Payments" count={loadMonthVouchers ? null : paymentCountMt} onClick={() => onNavigate('vouchers')} />
            <CueTile label="Receipts" count={loadMonthVouchers ? null : receiptCountMt} onClick={() => onNavigate('vouchers')} />
            <CueTile label="Journals" count={loadMonthVouchers ? null : journalCountMt} onClick={() => onNavigate('vouchers')} />
            <CueTile label="Sales entries today" count={loadToday ? null : salesEntryToday} onClick={() => onNavigate('daybook')} />
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-white shadow-sm overflow-hidden">
          <div className="px-4 py-2.5 border-b border-slate-100 flex items-center justify-between bg-slate-50/80">
            <h3 className="text-sm font-semibold text-slate-800">Today&apos;s day book</h3>
            <button type="button" onClick={() => onNavigate('daybook')} className="text-[11px] font-medium text-teal-700 hover:underline">
              Open →
            </button>
          </div>
          {loadToday ? (
            <div className="p-8 text-center text-slate-400 text-sm animate-pulse">Loading…</div>
          ) : todayVouchers.length === 0 ? (
            <div className="p-8 text-center text-slate-400 text-sm">No vouchers for {todayStr}.</div>
          ) : (
            <div className="overflow-x-auto max-h-[280px] overflow-y-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-slate-50">
                  <tr className="text-slate-500 uppercase tracking-wide text-[10px]">
                    <th className="px-3 py-2 text-left font-semibold">No.</th>
                    <th className="px-3 py-2 text-left font-semibold">Type</th>
                    <th className="px-3 py-2 text-left font-semibold">Party</th>
                    <th className="px-3 py-2 text-right font-semibold">Net</th>
                  </tr>
                </thead>
                <tbody>
                  {todayVouchers.slice(0, 12).map(v => (
                    <tr key={v.id} className="border-t border-slate-100 hover:bg-teal-50/40">
                      <td className="px-3 py-1.5 font-mono font-semibold text-teal-900">{v.voucher_no}</td>
                      <td className="px-3 py-1.5">
                        <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${VOUCHER_COLORS[v.voucher_type] ?? 'bg-slate-100 text-slate-600'}`}>
                          {v.voucher_type}
                        </span>
                      </td>
                      <td className="px-3 py-1.5 text-slate-600 max-w-[100px] truncate">{v.party_name || '—'}</td>
                      <td className="px-3 py-1.5 text-right font-semibold text-slate-800">{fmt(v.net_payable)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white shadow-sm overflow-hidden">
        <div className="px-4 py-2.5 border-b border-slate-100">
          <h3 className="text-sm font-semibold text-slate-800">Outstanding summary</h3>
          <p className="text-[11px] text-slate-500 mt-0.5">Top ledgers — receivables vs payables (BC-style balance cues).</p>
        </div>
        {!ledgerBalances ? (
          <div className="p-8 text-center text-slate-400 text-sm animate-pulse">Loading…</div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-slate-100">
            <div className="p-4">
              <p className="text-[10px] font-bold text-emerald-800 uppercase tracking-widest mb-2">Debtors</p>
              {(ledgerBalances.debtors ?? []).slice(0, 6).length === 0 ? (
                <p className="text-xs text-slate-400">None</p>
              ) : (
                (ledgerBalances.debtors ?? []).slice(0, 6).map((d, i) => (
                  <div key={i} className="flex justify-between items-center py-1 border-b border-slate-50 last:border-0">
                    <span className="text-xs text-slate-600 truncate max-w-[140px]" title={d.name}>{d.name}</span>
                    <span className="text-xs font-semibold text-emerald-800 tabular-nums">{fmt(d.balance)}</span>
                  </div>
                ))
              )}
            </div>
            <div className="p-4">
              <p className="text-[10px] font-bold text-red-800 uppercase tracking-widest mb-2">Creditors</p>
              {(ledgerBalances.creditors ?? []).slice(0, 6).length === 0 ? (
                <p className="text-xs text-slate-400">None</p>
              ) : (
                (ledgerBalances.creditors ?? []).slice(0, 6).map((c, i) => (
                  <div key={i} className="flex justify-between items-center py-1 border-b border-slate-50 last:border-0">
                    <span className="text-xs text-slate-600 truncate max-w-[140px]" title={c.name}>{c.name}</span>
                    <span className="text-xs font-semibold text-red-700 tabular-nums">{fmt(c.balance)}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}


// ── Voucher Register / Cash / Bank Books ─────────────────────────
function VoucherRegisterTab() {
  const [startDate, setStartDate] = useState(daysAgo(30))
  const [endDate, setEndDate] = useState(TODAY)

  const { data: vouchers = [], isLoading } = useQuery<Voucher[]>({
    queryKey: ['finance-voucher-register', startDate, endDate],
    queryFn: async () => {
      const p = new URLSearchParams()
      if (startDate) p.set('start_date', startDate)
      if (endDate) p.set('end_date', endDate)
      const { data } = await api.get(`/finance/vouchers?${p.toString()}`)
      return data
    },
    staleTime: 30 * 1000,
  })

  const total = vouchers.reduce((s, v) => s + v.net_payable, 0)

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-3 flex flex-wrap gap-2 items-end justify-between">
        <div className="flex items-end gap-2 flex-wrap">
          <div>
            <label className="text-xs text-gray-500">From</label>
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} className="block text-xs border border-gray-200 rounded px-2 py-1.5" />
          </div>
          <div>
            <label className="text-xs text-gray-500">To</label>
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="block text-xs border border-gray-200 rounded px-2 py-1.5" />
          </div>
        </div>
        <button
          type="button"
          onClick={() => {
            const rows: (string | number)[][] = [['date', 'voucher_no', 'voucher_type', 'party_name', 'taxable', 'cgst', 'sgst', 'igst', 'net_payable']]
            for (const v of vouchers) rows.push([v.voucher_date, v.voucher_no, v.voucher_type, v.party_name || '', v.taxable_amount, v.cgst_amount, v.sgst_amount, v.igst_amount, v.net_payable])
            downloadCsv(`voucher-register-${startDate || 'all'}-${endDate || 'all'}.csv`, rows)
          }}
          className="text-xs font-semibold px-3 py-1.5 rounded border border-teal-600 text-teal-800 bg-teal-50 hover:bg-teal-100"
        >
          Export register (CSV)
        </button>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {isLoading ? (
          <div className="p-8 text-center text-gray-400 text-sm">Loading voucher register…</div>
        ) : vouchers.length === 0 ? (
          <div className="p-8 text-center text-gray-400 text-sm">No vouchers in this range.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 text-gray-500 uppercase tracking-wide">
                  <th className="px-3 py-2 text-left">Date</th><th className="px-3 py-2 text-left">No</th><th className="px-3 py-2 text-left">Type</th><th className="px-3 py-2 text-left">Party</th><th className="px-3 py-2 text-right">Net</th>
                </tr>
              </thead>
              <tbody>
                {vouchers.map(v => (
                  <tr key={v.id} className="border-t border-gray-100 hover:bg-gray-50">
                    <td className="px-3 py-1.5">{v.voucher_date}</td>
                    <td className="px-3 py-1.5 font-mono text-[#002B5B]">{v.voucher_no}</td>
                    <td className="px-3 py-1.5"><span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${VOUCHER_COLORS[v.voucher_type] ?? 'bg-gray-100 text-gray-600'}`}>{v.voucher_type}</span></td>
                    <td className="px-3 py-1.5 text-gray-600 max-w-[180px] truncate">{v.party_name || '—'}</td>
                    <td className="px-3 py-1.5 text-right font-semibold">{fmt(v.net_payable)}</td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="bg-slate-800 text-white text-xs font-semibold"><td colSpan={4} className="px-3 py-2">Total ({vouchers.length})</td><td className="px-3 py-2 text-right">{fmt(total)}</td></tr>
              </tfoot>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function CashBankBookTab({ mode }: { mode: 'cash' | 'bank' }) {
  const title = mode === 'cash' ? 'Cash Book' : 'Bank Book'
  const [startDate, setStartDate] = useState(daysAgo(30))
  const [endDate, setEndDate] = useState(TODAY)

  const { data: vouchers = [], isLoading } = useQuery<Voucher[]>({
    queryKey: ['finance-book', mode, startDate, endDate],
    queryFn: async () => {
      const p = new URLSearchParams()
      if (startDate) p.set('start_date', startDate)
      if (endDate) p.set('end_date', endDate)
      const { data } = await api.get(`/finance/vouchers?${p.toString()}`)
      return data
    },
    staleTime: 30 * 1000,
  })

  const rows = useMemo(() => {
    if (mode === 'cash') {
      return vouchers.filter(v => {
        const pm = (v.payment_mode || '').toLowerCase()
        const hasBank = !!(v.bank_ledger || '').trim()
        return pm === 'cash' || (!hasBank && ['Receipt', 'Payment', 'Contra'].includes(v.voucher_type))
      })
    }
    return vouchers.filter(v => {
      const pm = (v.payment_mode || '').toLowerCase()
      const hasBank = !!(v.bank_ledger || '').trim()
      return hasBank || (pm && pm !== 'cash')
    })
  }, [mode, vouchers])

  const total = rows.reduce((s, v) => s + v.net_payable, 0)

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-3 flex flex-wrap gap-2 items-end justify-between">
        <div className="flex items-end gap-2 flex-wrap">
          <p className="text-xs font-semibold text-gray-700 mr-2">{title}</p>
          <div>
            <label className="text-xs text-gray-500">From</label>
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} className="block text-xs border border-gray-200 rounded px-2 py-1.5" />
          </div>
          <div>
            <label className="text-xs text-gray-500">To</label>
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="block text-xs border border-gray-200 rounded px-2 py-1.5" />
          </div>
        </div>
        <button
          type="button"
          onClick={() => {
            const csv: (string | number)[][] = [['date','voucher_no','voucher_type','party_name','payment_mode','bank_ledger','net_payable']]
            for (const v of rows) csv.push([v.voucher_date, v.voucher_no, v.voucher_type, v.party_name || '', v.payment_mode || '', v.bank_ledger || '', v.net_payable])
            downloadCsv(`${mode}-book-${startDate || 'all'}-${endDate || 'all'}.csv`, csv)
          }}
          className="text-xs font-semibold px-3 py-1.5 rounded border border-teal-600 text-teal-800 bg-teal-50 hover:bg-teal-100"
        >
          Export {title} (CSV)
        </button>
      </div>
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {isLoading ? (
          <div className="p-8 text-center text-gray-400 text-sm">Loading {title.toLowerCase()}…</div>
        ) : rows.length === 0 ? (
          <div className="p-8 text-center text-gray-400 text-sm">No entries for {title.toLowerCase()} in this range.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-gray-50 text-gray-500 uppercase tracking-wide">
                  <th className="px-3 py-2 text-left">Date</th><th className="px-3 py-2 text-left">No</th><th className="px-3 py-2 text-left">Type</th><th className="px-3 py-2 text-left">Payment mode</th><th className="px-3 py-2 text-left">Bank ledger</th><th className="px-3 py-2 text-right">Amount</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(v => (
                  <tr key={v.id} className="border-t border-gray-100 hover:bg-gray-50">
                    <td className="px-3 py-1.5">{v.voucher_date}</td><td className="px-3 py-1.5 font-mono text-[#002B5B]">{v.voucher_no}</td><td className="px-3 py-1.5">{v.voucher_type}</td><td className="px-3 py-1.5">{v.payment_mode || '—'}</td><td className="px-3 py-1.5">{v.bank_ledger || '—'}</td><td className="px-3 py-1.5 text-right font-semibold">{fmt(v.net_payable)}</td>
                  </tr>
                ))}
              </tbody>
              <tfoot><tr className="bg-slate-800 text-white text-xs font-semibold"><td colSpan={5} className="px-3 py-2">Total ({rows.length})</td><td className="px-3 py-2 text-right">{fmt(total)}</td></tr></tfoot>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Sales invoice: BC-style line grid + tax summary (Finance + Daybook) ──
function fmtDec(n: number) {
  return n.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function liStr(li: Record<string, unknown>, ...keys: string[]): string {
  for (const k of keys) {
    const v = li[k]
    if (v != null && String(v).trim() !== '' && String(v).toLowerCase() !== 'nan') return String(v).trim()
  }
  return ''
}

function fmtGstRatePct(li: Record<string, unknown>): string {
  const r = Number(li.gst_rate ?? li.GST_Rate ?? 0)
  if (!Number.isFinite(r) || r <= 0) return '—'
  // File may store 5 for 5% or 0.05 for 5%
  const pct = r <= 1 && r > 0 ? r * 100 : r
  return `${pct.toFixed(2)}%`
}

function SalesInvoiceBcDetailPanel({
  detail,
  lineItemsShowSourceInv,
}: {
  detail: DaybookVoucher
  lineItemsShowSourceInv: boolean
}) {
  const lines = (detail.meta?.line_items ?? []) as Record<string, unknown>[]
  const taxNum = detail.cgst_amount + detail.sgst_amount + detail.igst_amount
  const { sumLineInvoiceIncl, sumLineTaxableExcl } = useMemo(() => {
    let inv = 0
    let tex = 0
    for (const li of lines) {
      const invAmt = Number(li.invoice_amount ?? li.Invoice_Amount ?? 0)
      if (Number.isFinite(invAmt)) inv += invAmt
      const t = Number(li.tax_exclusive_gross ?? li.tax_exclusive_amount ?? 0)
      if (Number.isFinite(t)) tex += t
    }
    return { sumLineInvoiceIncl: inv, sumLineTaxableExcl: tex }
  }, [lines])

  const {
    subExcl,
    taxNumDisp,
    totalIncl,
    hdrTaxableBase,
    dispCgst,
    dispSgst,
    dispIgst,
  } = useMemo(() => {
    const subRaw = Number(detail.taxable_amount) || 0
    const totRaw = Number(detail.total_amount) || 0
    const tx = taxNum
    // Taxable total must be compared to tax-exclusive line sums, not invoice-inclusive amounts.
    // If tax-exclusive is missing in the file, we fall back to using invoice-inclusive lines.
    const hasTexOnLines = Math.abs(sumLineTaxableExcl) > 0.01
    const linesComparable = hasTexOnLines ? sumLineTaxableExcl : sumLineInvoiceIncl
    const inclusiveLinePayable =
      lines.length > 0 && Math.abs(linesComparable - subRaw) < 1.5 && Math.abs(linesComparable) > 0.01
    const builtAsExclusivePlusTax =
      tx > 0.01 &&
      subRaw > 0 &&
      Math.abs(subRaw + tx - totRaw) < 0.08 &&
      totRaw > sumLineInvoiceIncl + 0.5
    if (inclusiveLinePayable && builtAsExclusivePlusTax) {
      let r: number | null = null
      for (const li of lines) {
        const gr = Number(li.gst_rate ?? li.GST_Rate ?? 0)
        if (gr > 0 && Number.isFinite(gr)) {
          const dec = gr <= 1 && gr > 0 ? gr : gr / 100
          if (dec > 0.005 && dec < 0.28) {
            r = dec
            break
          }
        }
      }
      if (r == null) {
        r = tx / subRaw
      }
      if (r > 0.005 && r < 0.28) {
        const subE = subRaw / (1 + r)
        const taxD = subRaw - subE
        const sc = tx > 1e-9 ? taxD / tx : 1
        return {
          subExcl: subE,
          taxNumDisp: taxD,
          totalIncl: subRaw,
          hdrTaxableBase: Math.max(0.001, subE),
          dispCgst: detail.cgst_amount * sc,
          dispSgst: detail.sgst_amount * sc,
          dispIgst: detail.igst_amount * sc,
        }
      }
    }
    const hdr = Math.max(0.001, subRaw)
    return {
      subExcl: subRaw,
      taxNumDisp: tx,
      totalIncl: totRaw || subRaw + tx,
      hdrTaxableBase: hdr,
      dispCgst: detail.cgst_amount,
      dispSgst: detail.sgst_amount,
      dispIgst: detail.igst_amount,
    }
  }, [
    detail.cgst_amount,
    detail.sgst_amount,
    detail.igst_amount,
    detail.taxable_amount,
    detail.total_amount,
    lines,
    sumLineInvoiceIncl,
    sumLineTaxableExcl,
    taxNum,
  ])

  const effRate = hdrTaxableBase > 0 ? (100 * taxNumDisp) / hdrTaxableBase : 0
  const hsns = [...new Set(lines.map(li => liStr(li, 'hsn_sac', 'HSN_SAC')).filter(Boolean))]
  const shipState = (detail.meta?.ship_to_state || detail.party_state || '').toString().trim()
  const fromSt = (detail.meta?.seller_state || '').toString().trim()

  return (
    <div className="flex flex-col xl:flex-row gap-4 min-h-0">
      <div className="flex-1 min-w-0 border border-slate-200 rounded-sm bg-white overflow-hidden flex flex-col">
        <div className="px-3 py-2 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
          <p className="text-[11px] font-bold text-slate-600 uppercase tracking-wide">Lines</p>
          <span className="text-[10px] text-slate-500">{lines.length} row{lines.length !== 1 ? 's' : ''}</span>
        </div>
        <div className="overflow-auto max-h-[52vh]">
          <table className="w-full text-[10px] min-w-[2800px]">
            <thead className="bg-slate-50 sticky top-0 z-[1]">
              <tr className="text-left text-slate-600">
                {lineItemsShowSourceInv ? <th className="px-1.5 py-2 whitespace-nowrap">Source inv.</th> : null}
                <th className="px-1.5 py-2 whitespace-nowrap">Ship from state</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Location</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Party name</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Invoice number</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Invoice date</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Transaction type</th>
                <th className="px-1.5 py-2 whitespace-nowrap max-w-[14rem]">Product / item desc.</th>
                <th className="px-1.5 py-2 text-right whitespace-nowrap">Qty</th>
                <th className="px-1.5 py-2 whitespace-nowrap">HSN/SAC</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Item no.</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Ship to state</th>
                <th className="px-1.5 py-2 text-right whitespace-nowrap">Invoice amount</th>
                <th className="px-1.5 py-2 text-right whitespace-nowrap">Tax excl. gross</th>
                <th className="px-1.5 py-2 text-right whitespace-nowrap">Total tax</th>
                <th className="px-1.5 py-2 text-right whitespace-nowrap">GST rate</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Order id</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Shipment id</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Shipment item id</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Order date</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Shipment date</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Credit note no.</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Credit note date</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Customer name</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Customer GST no.</th>
                <th className="px-1.5 py-2 whitespace-nowrap max-w-[8rem]">IRN hash</th>
                <th className="px-1.5 py-2 whitespace-nowrap">Ack. date</th>
                <th className="px-1.5 py-2 text-right whitespace-nowrap">CGST</th>
                <th className="px-1.5 py-2 text-right whitespace-nowrap">SGST</th>
                <th className="px-1.5 py-2 text-right whitespace-nowrap">IGST</th>
                <th className="px-1.5 py-2 text-right whitespace-nowrap">Line total</th>
              </tr>
            </thead>
            <tbody>
              {lines.map((li, i) => {
                const qty = Number(li.quantity ?? li.Quantity ?? 0)
                const tex = Number(li.tax_exclusive_gross ?? li.tax_exclusive_amount ?? 0)
                const invAmt = Number(li.invoice_amount ?? li.Invoice_Amount ?? 0)
                const cg = Number(li.cgst ?? li.CGST ?? 0)
                const sg = Number(li.sgst ?? li.SGST ?? 0)
                const ig = Number(li.igst ?? li.IGST ?? 0)
                const tt = Number(li.total_tax_amount ?? li.total_tax ?? 0) || cg + sg + ig
                const base = Math.abs(tex) > 1e-9 ? tex : invAmt
                const naiveTot = base + (Number.isFinite(tt) ? tt : 0)
                const lineTot =
                  Number.isFinite(invAmt) && Number.isFinite(tt) && Math.abs(invAmt - naiveTot) < 0.5 ? invAmt : naiveTot
                const shipFrom = liStr(li, 'ship_from_state', 'Ship_From_State', 'bill_from_state', 'Bill_From_State')
                const loc = liStr(li, 'location', 'Location_Line')
                const party = liStr(li, 'party_name', 'Party_Name', 'customer_name', 'Customer_Name')
                const invNo = liStr(li, 'invoice_number', 'Invoice_Number', 'source_invoice_no')
                const invDt = liStr(li, 'invoice_date', 'Invoice_Date')
                const txn = liStr(li, 'transaction_type', 'Transaction_Type', 'type', 'Type')
                const prodDesc = liStr(li, 'product_name', 'Product_Name', 'item_description')
                const hsn = liStr(li, 'hsn_sac', 'HSN_SAC')
                const itemNo = liStr(li, 'item_no', 'Item_No', 'sku', 'SKU')
                const shipTo = liStr(li, 'ship_to_state_code', 'Ship_To_State') || liStr(li, 'ship_to_state', 'place_of_supply', 'Place_Of_Supply')
                const oid = liStr(li, 'order_id', 'Order_Id')
                const shipAmazonId = liStr(li, 'shipment_id', 'Shipment_Id')
                const shipItemId = liStr(li, 'order_item_id', 'Order_Item_Id')
                const ordDtFile = liStr(li, 'order_date_text', 'Order_Date_Text')
                const shipDtFile = liStr(li, 'shipment_date_text', 'Shipment_Date_Text')
                const cnNo = liStr(li, 'credit_note_no', 'Credit_Note_No')
                const cnDt = liStr(li, 'credit_note_date', 'Credit_Note_Date')
                const cust = liStr(li, 'customer_name', 'Customer_Name') || party
                const custGst = liStr(li, 'customer_gst_no', 'Customer_Gst_No', 'customer_gstin')
                const irnH = liStr(li, 'irn_hash', 'IRN_Hash')
                const ack = liStr(li, 'acknowledgement_date', 'Acknowledgement_Date')
                return (
                  <tr key={i} className="border-t border-slate-100 hover:bg-slate-50/80">
                    {lineItemsShowSourceInv ? (
                      <td className="px-1.5 py-1.5 font-mono text-slate-600 max-w-[5rem] truncate" title={liStr(li, 'source_invoice_no')}>
                        {liStr(li, 'source_invoice_no') || '—'}
                      </td>
                    ) : null}
                    <td className="px-1.5 py-1.5 text-slate-800 max-w-[5rem] truncate" title={shipFrom}>{shipFrom || '—'}</td>
                    <td className="px-1.5 py-1.5 text-slate-700 max-w-[7rem] truncate" title={loc}>{loc || '—'}</td>
                    <td className="px-1.5 py-1.5 text-slate-800 max-w-[7rem] truncate" title={party}>{party || '—'}</td>
                    <td className="px-1.5 py-1.5 font-mono text-slate-800 max-w-[6rem] truncate" title={invNo}>{invNo || '—'}</td>
                    <td className="px-1.5 py-1.5 font-mono text-slate-700 whitespace-nowrap">{invDt || '—'}</td>
                    <td className="px-1.5 py-1.5 text-slate-700 max-w-[5rem] truncate" title={txn}>{txn || '—'}</td>
                    <td className="px-1.5 py-1.5 text-slate-700 max-w-[14rem] truncate" title={prodDesc}>{prodDesc || '—'}</td>
                    <td className="px-1.5 py-1.5 text-right tabular-nums">{qty}</td>
                    <td className="px-1.5 py-1.5 font-mono text-slate-700">{hsn || '—'}</td>
                    <td className="px-1.5 py-1.5 font-mono text-slate-800 max-w-[6rem] truncate" title={itemNo}>{itemNo || '—'}</td>
                    <td className="px-1.5 py-1.5 text-slate-700 max-w-[5rem] truncate" title={shipTo}>{shipTo || '—'}</td>
                    <td className="px-1.5 py-1.5 text-right tabular-nums">{fmtDec(invAmt)}</td>
                    <td className="px-1.5 py-1.5 text-right tabular-nums">{fmtDec(tex)}</td>
                    <td className="px-1.5 py-1.5 text-right tabular-nums">{fmtDec(tt)}</td>
                    <td className="px-1.5 py-1.5 text-right font-mono text-slate-800">{fmtGstRatePct(li)}</td>
                    <td className="px-1.5 py-1.5 font-mono text-slate-700 max-w-[6rem] truncate" title={oid}>{oid || '—'}</td>
                    <td className="px-1.5 py-1.5 font-mono text-slate-600 max-w-[7rem] truncate text-[9px]" title={shipAmazonId}>{shipAmazonId || '—'}</td>
                    <td className="px-1.5 py-1.5 font-mono text-slate-600 max-w-[6rem] truncate text-[9px]" title={shipItemId}>{shipItemId || '—'}</td>
                    <td className="px-1.5 py-1.5 font-mono text-slate-600 whitespace-nowrap text-[9px]">{ordDtFile || '—'}</td>
                    <td className="px-1.5 py-1.5 font-mono text-slate-600 whitespace-nowrap text-[9px]">{shipDtFile || '—'}</td>
                    <td className="px-1.5 py-1.5 font-mono max-w-[5rem] truncate" title={cnNo}>{cnNo || '—'}</td>
                    <td className="px-1.5 py-1.5 font-mono whitespace-nowrap">{cnDt || '—'}</td>
                    <td className="px-1.5 py-1.5 text-slate-800 max-w-[7rem] truncate" title={cust}>{cust || '—'}</td>
                    <td className="px-1.5 py-1.5 font-mono text-[9px] max-w-[7rem] truncate" title={custGst}>{custGst || '—'}</td>
                    <td className="px-1.5 py-1.5 font-mono text-[9px] max-w-[8rem] break-all" title={irnH}>{irnH || '—'}</td>
                    <td className="px-1.5 py-1.5 font-mono whitespace-nowrap">{ack || '—'}</td>
                    <td className="px-1.5 py-1.5 text-right tabular-nums text-slate-600">{cg ? fmtDec(cg) : '—'}</td>
                    <td className="px-1.5 py-1.5 text-right tabular-nums text-slate-600">{sg ? fmtDec(sg) : '—'}</td>
                    <td className="px-1.5 py-1.5 text-right tabular-nums text-slate-600">{ig ? fmtDec(ig) : '—'}</td>
                    <td className="px-1.5 py-1.5 text-right tabular-nums font-medium text-slate-900">{fmtDec(lineTot)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <div className="px-3 py-2 border-t border-slate-200 bg-slate-50/90 text-[11px] text-slate-600 grid grid-cols-2 sm:grid-cols-6 gap-2">
          <div><span className="text-slate-500">Header taxable (excl. tax)</span><div className="font-semibold text-slate-900">₹{fmtDec(subExcl)}</div></div>
          <div><span className="text-slate-500">Lines taxable (excl. tax)</span><div className="font-semibold text-slate-900">₹{fmtDec(sumLineTaxableExcl)}</div></div>
          <div><span className="text-slate-500">Lines invoice (incl. tax)</span><div className="font-semibold text-slate-700">₹{fmtDec(sumLineInvoiceIncl)}</div></div>
          <div><span className="text-slate-500">Total tax</span><div className="font-semibold text-slate-900">₹{fmtDec(taxNumDisp)}</div></div>
          <div><span className="text-slate-500">Header total incl. tax</span><div className="font-semibold text-[#002B5B]">₹{fmtDec(totalIncl)}</div></div>
          <div><span className="text-slate-500">Effective tax %</span><div className="font-semibold text-slate-700">{effRate > 0 ? `${effRate.toFixed(2)}%` : '—'}</div></div>
        </div>
      </div>
      <aside className="w-full xl:w-[300px] shrink-0 border border-slate-200 rounded-sm bg-white p-3 space-y-4">
        <div>
          <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wide mb-2">Tax information</p>
          <dl className="space-y-2 text-[11px]">
            <div className="flex justify-between gap-2"><dt className="text-slate-500">HSN/SAC</dt><dd className="font-mono text-right text-slate-800 break-all">{hsns.length ? hsns.join(', ') : '—'}</dd></div>
            <div className="flex justify-between gap-2"><dt className="text-slate-500">Place of supply</dt><dd className="text-right text-slate-800">{shipState || '—'}</dd></div>
            <div className="flex justify-between gap-2"><dt className="text-slate-500">Seller branch state</dt><dd className="text-right text-slate-800">{fromSt || '—'}</dd></div>
            <div className="flex justify-between gap-2"><dt className="text-slate-500">Supply</dt><dd className="text-right text-slate-800">{detail.supply_type === 'Inter' ? 'Inter-state (IGST)' : detail.supply_type === 'Intra' ? 'Intra-state (CGST+SGST)' : '—'}</dd></div>
            <div className="flex justify-between gap-2"><dt className="text-slate-500">Effective tax %</dt><dd className="text-right font-mono text-slate-800">{effRate > 0 ? `${effRate.toFixed(2)}%` : '—'}</dd></div>
            <div className="flex justify-between gap-2"><dt className="text-slate-500">GST base (header)</dt><dd className="text-right font-mono">₹{fmtDec(hdrTaxableBase)}</dd></div>
          </dl>
        </div>
        <div>
          <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wide mb-2">Tax components</p>
          <table className="w-full text-[11px]">
            <thead><tr className="text-slate-500 text-left"><th className="py-1">Component</th><th className="py-1 text-right">%</th><th className="py-1 text-right">Amount</th></tr></thead>
            <tbody>
              {dispCgst > 0 ? (
                <tr className="border-t border-slate-100"><td className="py-1">CGST</td><td className="text-right font-mono">{hdrTaxableBase ? ((100 * dispCgst) / hdrTaxableBase).toFixed(2) : '—'}</td><td className="text-right font-mono">₹{fmtDec(dispCgst)}</td></tr>
              ) : null}
              {dispSgst > 0 ? (
                <tr className="border-t border-slate-100"><td className="py-1">SGST</td><td className="text-right font-mono">{hdrTaxableBase ? ((100 * dispSgst) / hdrTaxableBase).toFixed(2) : '—'}</td><td className="text-right font-mono">₹{fmtDec(dispSgst)}</td></tr>
              ) : null}
              {dispIgst > 0 ? (
                <tr className="border-t border-slate-100"><td className="py-1">IGST</td><td className="text-right font-mono">{hdrTaxableBase ? ((100 * dispIgst) / hdrTaxableBase).toFixed(2) : '—'}</td><td className="text-right font-mono">₹{fmtDec(dispIgst)}</td></tr>
              ) : null}
              {taxNumDisp <= 0 ? <tr><td colSpan={3} className="py-2 text-slate-400">No tax on header — line taxes may still show per SKU.</td></tr> : null}
            </tbody>
          </table>
        </div>
        <p className="text-[10px] text-slate-500 border-t border-slate-100 pt-2 leading-snug">
          Columns match your mandatory GST invoice field list when the Amazon MTR (or compatible CSV) includes them. Empty cells mean the column was missing in that file — re-upload after export updates. GST rate is taken from the file or derived from line tax ÷ taxable when absent.
        </p>
      </aside>
    </div>
  )
}

// ── Sales Invoices — BC-style document form fields ─────────────────
function SiFormField(props: {
  label: string
  value: string
  onChange: (v: string) => void
  type?: string
  disabled?: boolean
}) {
  const { label, value, onChange, type = 'text', disabled } = props
  return (
    <div className="grid grid-cols-1 sm:grid-cols-[10rem_1fr] gap-1 sm:gap-2 py-1.5 border-b border-slate-200/80">
      <label className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide pt-1.5">{label}</label>
      <input
        type={type}
        disabled={disabled}
        value={value}
        onChange={e => onChange(e.target.value)}
        className={`text-sm border rounded px-2 py-1.5 w-full ${disabled ? 'bg-slate-50 text-slate-500' : 'border-slate-300 bg-white'}`}
      />
    </div>
  )
}

// ── Sales Invoices Tab (auto from Finance Sales Uploads) ───────────
function salesInvoicesDedupedNet(rows: SalesInvoiceRow[]) {
  const uploadsWithEntry = new Set<number>()
  for (const r of rows) {
    if (r.row_kind === 'entry' && r.sales_upload_id != null && r.sales_upload_id > 0)
      uploadsWithEntry.add(r.sales_upload_id)
  }
  let total = 0
  for (const r of rows) {
    if (r.row_kind === 'upload_summary' || r.voucher_no?.startsWith('SUP-')) {
      const m = /^SUP-(\d+)$/.exec(r.voucher_no || '')
      const uid = m ? parseInt(m[1], 10) : NaN
      if (!Number.isNaN(uid) && uploadsWithEntry.has(uid)) continue
    }
    total += r.net_payable
  }
  return total
}

function salesInvoicesDedupedTax(rows: SalesInvoiceRow[]) {
  const uploadsWithEntry = new Set<number>()
  for (const r of rows) {
    if (r.row_kind === 'entry' && r.sales_upload_id != null && r.sales_upload_id > 0)
      uploadsWithEntry.add(r.sales_upload_id)
  }
  let tax = 0
  for (const r of rows) {
    if (r.row_kind === 'upload_summary' || r.voucher_no?.startsWith('SUP-')) {
      const m = /^SUP-(\d+)$/.exec(r.voucher_no || '')
      const uid = m ? parseInt(m[1], 10) : NaN
      if (!Number.isNaN(uid) && uploadsWithEntry.has(uid)) continue
    }
    tax += r.cgst_amount + r.sgst_amount + r.igst_amount
  }
  return tax
}

type SalesInvoicesListMode = 'sales' | 'credit_memo'

function SalesInvoicesTab({
  mode,
  openTrialBalanceForParty,
  openMastersLedgersForParty,
}: {
  mode: SalesInvoicesListMode
  openTrialBalanceForParty: (q: string) => void
  openMastersLedgersForParty: (q: string) => void
}) {
  const qc = useQueryClient()
  const isCreditTab = mode === 'credit_memo'
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [cardOpen, setCardOpen] = useState(false)
  const [cardTab, setCardTab] = useState<'general' | 'lines' | 'inventory' | 'location'>('general')
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [dimRows, setDimRows] = useState<Array<{ attribute_name: string; value_code: string; value_description: string }>>([
    { attribute_name: '', value_code: '', value_description: '' },
  ])
  const [saveMsg, setSaveMsg] = useState<string | null>(null)
  const [includeUploadSummaries, setIncludeUploadSummaries] = useState(true)

  const q = useMemo(() => {
    const p = new URLSearchParams()
    p.set('document_kind', isCreditTab ? 'credit_memo' : 'sales')
    if (startDate) p.set('start_date', startDate)
    if (endDate) p.set('end_date', endDate)
    if (search.trim()) p.set('search', search.trim())
    if (!isCreditTab && !includeUploadSummaries) p.set('include_upload_summaries', 'false')
    return p.toString()
  }, [startDate, endDate, search, isCreditTab, includeUploadSummaries])

  const { data: rows = [], isLoading } = useQuery<SalesInvoiceRow[]>({
    queryKey: ['finance-sales-invoices', mode, startDate, endDate, search, includeUploadSummaries],
    queryFn: async () => { const { data } = await api.get(`/finance/sales-invoices?${q}`); return data },
    staleTime: 30 * 1000,
  })

  useEffect(() => {
    if (rows.length === 0) {
      setSelectedId(null)
      setCardOpen(false)
      return
    }
    if (selectedId == null || !rows.some(r => r.id === selectedId)) setSelectedId(rows[0].id)
  }, [rows, selectedId])

  const selected = rows.find(r => r.id === selectedId) || null
  const { data: detail, isLoading: detailLoading } = useQuery<DaybookVoucher>({
    queryKey: ['finance-sales-invoice-detail', selectedId],
    queryFn: async () => { const { data } = await api.get(`/finance/vouchers/${selectedId}`); return data },
    enabled: !!selectedId,
    staleTime: 30 * 1000,
  })

  useEffect(() => {
    if (!detail || !cardOpen) return
    const inv = (detail.bill_no || detail.meta?.invoice_no || '').toString()
    setDraft({
      invoice_no: inv,
      voucher_date: (detail.voucher_date || '').toString(),
      bill_date: (detail.bill_date || detail.voucher_date || '').toString(),
      party_name: detail.party_name || '',
      party_gstin: (detail.party_gstin || '').toString(),
      party_state: (detail.party_state || '').toString(),
      ship_to_state: (detail.meta?.ship_to_state || '').toString(),
      order_id: (detail.meta?.order_id || detail.ref_number || '').toString(),
      source_filename: (detail.meta?.source_filename || '').toString(),
      narration: detail.narration || '',
      supply_type: (detail.supply_type || '').toString(),
      platform: (detail.meta?.platform || '').toString(),
      period: (detail.meta?.period || '').toString(),
      taxable_amount: String(detail.taxable_amount ?? ''),
      cgst_amount: String(detail.cgst_amount ?? ''),
      sgst_amount: String(detail.sgst_amount ?? ''),
      igst_amount: String(detail.igst_amount ?? ''),
      total_amount: String(detail.total_amount ?? ''),
      net_payable: String(detail.net_payable ?? ''),
    })
    const rawDims = detail.meta?.dimension_assignments
    if (Array.isArray(rawDims) && rawDims.length > 0) {
      setDimRows(
        rawDims.map((x: Record<string, unknown>) => ({
          attribute_name: String(x.attribute_name ?? x.attribute ?? ''),
          value_code: String(x.value_code ?? x.code ?? ''),
          value_description: String(x.value_description ?? x.description ?? x.financial_tag ?? ''),
        })),
      )
    } else {
      setDimRows([{ attribute_name: '', value_code: '', value_description: '' }])
    }
    setSaveMsg(null)
  }, [detail, cardOpen, selectedId])

  useEffect(() => {
    if (!cardOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setCardOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [cardOpen])

  const saveMutation = useMutation({
    mutationFn: async (body: Record<string, unknown>) => {
      if (selectedId == null) throw new Error('No invoice selected')
      const { data } = await api.patch<{ ok?: boolean; message?: string }>(`/finance/sales-invoices/${selectedId}`, body)
      return data
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['finance-sales-invoices'] })
      if (selectedId != null) void qc.invalidateQueries({ queryKey: ['finance-sales-invoice-detail', selectedId] })
      setSaveMsg('Saved.')
    },
    onError: (e: unknown) => {
      setSaveMsg(e instanceof Error ? e.message : 'Save failed')
    },
  })

  const setD = (key: string, v: string) => {
    setDraft(prev => ({ ...prev, [key]: v }))
  }

  const openRow = (id: number) => {
    const hit = rows.find(r => r.id === id)
    const isUploadSummary =
      hit?.row_kind === 'upload_summary' || (hit?.voucher_no ?? '').toUpperCase().startsWith('SUP-')
    setSelectedId(id)
    // SKU / mandatory columns live under Lines; upload summaries used to open on General only.
    setCardTab(isUploadSummary ? 'lines' : 'general')
    setCardOpen(true)
  }

  const saveInvoice = () => {
    if (selectedId == null) return
    const body: Record<string, unknown> = {}
    const strKeys = [
      'invoice_no', 'voucher_date', 'bill_date', 'party_name', 'party_gstin', 'party_state', 'ship_to_state',
      'order_id', 'source_filename', 'narration', 'supply_type', 'platform', 'period',
    ] as const
    for (const k of strKeys) {
      if (draft[k] !== undefined) body[k] = draft[k]
    }
    const numKeys = ['taxable_amount', 'cgst_amount', 'sgst_amount', 'igst_amount', 'total_amount', 'net_payable'] as const
    for (const k of numKeys) {
      const raw = (draft[k] ?? '').trim()
      if (raw === '') continue
      const n = Number(raw)
      if (!Number.isNaN(n)) body[k] = n
    }
    body.dimension_assignments = dimRows
      .map(d => ({
        attribute_name: d.attribute_name.trim(),
        value_code: d.value_code.trim(),
        value_description: d.value_description.trim(),
      }))
      .filter(d => d.attribute_name || d.value_code || d.value_description)
    saveMutation.mutate(body)
  }

  const stats = useMemo(() => {
    const total = salesInvoicesDedupedNet(rows)
    const tax = salesInvoicesDedupedTax(rows)
    return { total, tax, count: rows.length }
  }, [rows])

  const partyTbQuery = useMemo(
    () => (draft.party_name || '').trim() || (draft.party_gstin || '').trim(),
    [draft.party_name, draft.party_gstin],
  )
  const partyMasterQuery = useMemo(() => {
    const g = (draft.party_gstin || '').replace(/\s/g, '')
    if (g.length >= 15) return g
    return (draft.party_name || '').trim()
  }, [draft.party_gstin, draft.party_name])

  const lineItemsShowSourceInv = useMemo(() => {
    const lis = detail?.meta?.line_items ?? []
    return lis.some(li => String((li as { source_invoice_no?: string }).source_invoice_no || '').trim() !== '')
  }, [detail?.meta?.line_items])

  const lineItemCount = Array.isArray(detail?.meta?.line_items) ? detail.meta!.line_items!.length : 0
  const inventoryRows = useMemo(() => {
    const lis = Array.isArray(detail?.meta?.line_items) ? detail.meta!.line_items! : []
    const bySku = new Map<string, { sku: string; product_name: string; qty_out: number; qty_in: number; net_qty: number }>()
    for (const li of lis) {
      const obj = li as Record<string, unknown>
      const sku = liStr(obj, 'sku', 'SKU', 'item_no', 'Item_No').trim()
      if (!sku) continue
      const prod = liStr(obj, 'product_name', 'Product_Name')
      const qty = Number(obj.quantity ?? obj.Quantity ?? 0) || 0
      const isCredit = (detail?.voucher_type || '').toLowerCase().includes('credit')
      const qOut = isCredit ? 0 : Math.max(0, qty)
      const qIn = isCredit ? Math.max(0, Math.abs(qty)) : 0
      const hit = bySku.get(sku) || { sku, product_name: prod, qty_out: 0, qty_in: 0, net_qty: 0 }
      hit.qty_out += qOut
      hit.qty_in += qIn
      hit.net_qty = hit.qty_out - hit.qty_in
      if (!hit.product_name && prod) hit.product_name = prod
      bySku.set(sku, hit)
    }
    return Array.from(bySku.values()).sort((a, b) => Math.abs(b.net_qty) - Math.abs(a.net_qty))
  }, [detail])
  const locationRows = useMemo(() => {
    const lis = Array.isArray(detail?.meta?.line_items) ? detail.meta!.line_items! : []
    const byLoc = new Map<string, { location: string; ship_to_state: string; qty: number; lines: number }>()
    for (const li of lis) {
      const obj = li as Record<string, unknown>
      const location = liStr(obj, 'location', 'Location_Line')
      const ship = liStr(obj, 'ship_to_state', 'ship_to_state_code', 'Ship_To_State')
      const key = `${location}__${ship}`
      const qty = Math.abs(Number(obj.quantity ?? obj.Quantity ?? 0) || 0)
      const hit = byLoc.get(key) || { location, ship_to_state: ship, qty: 0, lines: 0 }
      hit.qty += qty
      hit.lines += 1
      byLoc.set(key, hit)
    }
    return Array.from(byLoc.values()).sort((a, b) => b.qty - a.qty)
  }, [detail])

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-3 flex flex-wrap gap-2 items-end justify-between">
        <div className="flex flex-wrap gap-2 items-end">
          <div>
            <label className="text-xs text-gray-500">From</label>
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} className="block text-xs border border-gray-200 rounded px-2 py-1.5" />
            <p className="text-[10px] text-gray-400 mt-0.5 max-w-[11rem]">
              {isCreditTab
                ? 'Leave From/To empty to load all posted sales credit memos / returns (SUE lines only).'
                : 'Leave From/To empty to load all uploads and invoice lines (excludes credit memos).'}
            </p>
          </div>
          <div>
            <label className="text-xs text-gray-500">To</label>
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="block text-xs border border-gray-200 rounded px-2 py-1.5" />
          </div>
          <div>
            <label className="text-xs text-gray-500">Search</label>
            <input type="text" value={search} onChange={e => setSearch(e.target.value)} placeholder="Invoice / Order / Customer" className="block text-xs border border-gray-200 rounded px-2 py-1.5 w-56" />
          </div>
          {!isCreditTab ? (
            <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={includeUploadSummaries}
                onChange={e => setIncludeUploadSummaries(e.target.checked)}
                className="rounded border-gray-300"
              />
              Show upload summaries (SUP-)
            </label>
          ) : null}
        </div>
        <button
          type="button"
          onClick={() => {
            const csv: (string | number)[][] = [[
              'voucher_no', 'document_subtype', 'row_kind', 'sales_upload_id', 'invoice_no', 'order_id', 'date', 'customer', 'platform',
              'taxable', 'cgst', 'sgst', 'igst', 'net_payable', 'narration',
            ]]
            for (const r of rows) {
              csv.push([
                r.voucher_no, r.document_subtype ?? '', r.row_kind ?? '', r.sales_upload_id ?? '', r.invoice_no, r.order_id, r.voucher_date, r.party_name, r.platform,
                r.taxable_amount, r.cgst_amount, r.sgst_amount, r.igst_amount, r.net_payable, r.narration ?? '',
              ])
            }
            const fn = isCreditTab
              ? `sales-credit-memos-${startDate || 'all'}-${endDate || 'all'}.csv`
              : `sales-invoices-${startDate || 'all'}-${endDate || 'all'}.csv`
            downloadCsv(fn, csv)
          }}
          className="text-xs font-semibold px-3 py-1.5 rounded border border-teal-600 text-teal-800 bg-teal-50 hover:bg-teal-100"
        >
          {isCreditTab ? 'Export credit memos (CSV)' : 'Export invoices (CSV)'}
        </button>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-4 py-2.5 border-b border-gray-100 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-gray-800">
              {isCreditTab ? 'Sales credit memos (returns / credit notes)' : 'Sales invoices (uploads + parsed lines)'}
            </h3>
            <p className="text-[11px] text-gray-500 mt-0.5">
              {isCreditTab
                ? 'Posted returns and adjustments (negative net or taxable, or narration mentioning refund). Press Esc to close the card.'
                : 'Posted buyer invoices (SUE-) plus optional upload summaries (SUP-). Uncheck “Show upload summaries” to hide SUP-. One Amazon file often creates several SUP- rows — one Finance upload per seller GSTIN branch in the MTR (company name shows that branch), not separate manual uploads per state. Returns: Sales credit memos.'}
            </p>
          </div>
          <span className="text-xs text-gray-500">{stats.count} rows · net {fmt(stats.total)}</span>
        </div>
        {isLoading ? (
          <div className="p-8 text-center text-gray-400 text-sm">{isCreditTab ? 'Loading credit memos…' : 'Loading invoices…'}</div>
        ) : rows.length === 0 ? (
          <div className="p-8 text-center text-gray-400 text-sm">
            {isCreditTab
              ? 'No sales credit memos match your filters. Clear dates or search; credit lines are detected from negative amounts or refund narration.'
              : 'No rows match your filters. Clear dates to show everything from Sales Uploads, or adjust search.'}
          </div>
        ) : (
          <div className="overflow-x-auto max-h-[560px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-gray-50 z-[1] shadow-sm">
                <tr className="text-gray-500 uppercase tracking-wide">
                  <th className="px-3 py-2 text-left">No.</th>
                  <th className="px-3 py-2 text-left">Invoice</th>
                  <th className="px-3 py-2 text-left">Party</th>
                  <th className="px-3 py-2 text-left">Platform</th>
                  <th className="px-3 py-2 text-right">Net</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(r => (
                  <tr
                    key={r.id}
                    className={`border-t border-gray-100 hover:bg-teal-50/50 cursor-pointer ${selectedId === r.id ? 'bg-teal-50/70' : ''}`}
                    onClick={() => openRow(r.id)}
                  >
                    <td className="px-3 py-1.5 font-mono text-teal-900">{r.voucher_no}</td>
                    <td className="px-3 py-1.5">
                      <p className="font-medium text-gray-800">{r.invoice_no || '—'}</p>
                      <p className="text-[10px] text-gray-500">{r.voucher_date} · Order {r.order_id || '—'}</p>
                    </td>
                    <td className="px-3 py-1.5 text-gray-700">
                      <div className="flex flex-col gap-0.5 max-w-[16rem]">
                        {(r.row_kind === 'upload_summary' || (r.voucher_no ?? '').toUpperCase().startsWith('SUP-')) ? (
                          <span className="text-[9px] font-semibold uppercase tracking-wide text-amber-900 bg-amber-50 border border-amber-200 rounded px-1 py-0.5 w-max">
                            Seller · upload summary
                          </span>
                        ) : null}
                        <span className="font-medium text-gray-800">{r.party_name || '—'}</span>
                        {(r.party_state || r.ship_to_state) ? (
                          <span className="text-[10px] text-gray-500">{r.party_state || r.ship_to_state}</span>
                        ) : null}
                      </div>
                    </td>
                    <td className="px-3 py-1.5 text-gray-600">{r.platform || '—'}</td>
                    <td className="px-3 py-1.5 text-right font-semibold">{fmt(r.net_payable)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {cardOpen && selectedId != null && (
        <div
          className="fixed inset-0 z-[120] flex items-center justify-center p-3 sm:p-6 bg-slate-900/50 backdrop-blur-[1px]"
          role="dialog"
          aria-modal="true"
          aria-labelledby="si-card-title"
          onClick={() => setCardOpen(false)}
        >
          <div
            className="bg-[#f9fbfd] w-full max-w-6xl max-h-[92vh] flex flex-col rounded-sm border border-slate-300 shadow-2xl overflow-hidden"
            onClick={e => e.stopPropagation()}
          >
            <header className="shrink-0 flex items-center justify-between px-4 py-2.5 bg-[#002B5B] text-white border-b border-[#001a3d]">
              <div className="min-w-0">
                <p className="text-[10px] uppercase tracking-wider text-blue-200/90">
                  {detail?.voucher_type === 'Sales Credit Memo' || selected?.document_subtype === 'sales_credit_memo'
                    ? 'Sales · Sales credit memo'
                    : 'Sales · Posted document'}
                </p>
                <h2 id="si-card-title" className="text-base font-semibold truncate">
                  {selected?.voucher_no ?? 'Invoice'} · {draft.invoice_no || '—'}
                </h2>
              </div>
              <button
                type="button"
                className="text-sm px-3 py-1.5 rounded border border-white/30 hover:bg-white/10"
                onClick={() => setCardOpen(false)}
              >
                Close
              </button>
            </header>

            <div className="shrink-0 flex gap-1 px-3 pt-2 border-b border-slate-200 bg-white">
              <button
                type="button"
                className={`text-xs font-semibold px-3 py-2 rounded-t ${cardTab === 'general' ? 'bg-[#f9fbfd] text-[#002B5B] border border-b-0 border-slate-200 -mb-px' : 'text-slate-500 hover:text-slate-700'}`}
                onClick={() => setCardTab('general')}
              >
                General
              </button>
              <button
                type="button"
                className={`text-xs font-semibold px-3 py-2 rounded-t ${cardTab === 'lines' ? 'bg-[#f9fbfd] text-[#002B5B] border border-b-0 border-slate-200 -mb-px' : 'text-slate-500 hover:text-slate-700'}`}
                onClick={() => setCardTab('lines')}
              >
                Lines
                {lineItemCount > 0 ? (
                  <span className="ml-1.5 rounded-full bg-teal-100 text-teal-900 px-1.5 py-0.5 text-[10px] tabular-nums">{lineItemCount}</span>
                ) : null}
              </button>
              <button
                type="button"
                className={`text-xs font-semibold px-3 py-2 rounded-t ${cardTab === 'inventory' ? 'bg-[#f9fbfd] text-[#002B5B] border border-b-0 border-slate-200 -mb-px' : 'text-slate-500 hover:text-slate-700'}`}
                onClick={() => setCardTab('inventory')}
              >
                Inventory
              </button>
              <button
                type="button"
                className={`text-xs font-semibold px-3 py-2 rounded-t ${cardTab === 'location' ? 'bg-[#f9fbfd] text-[#002B5B] border border-b-0 border-slate-200 -mb-px' : 'text-slate-500 hover:text-slate-700'}`}
                onClick={() => setCardTab('location')}
              >
                Location
              </button>
            </div>

            <div className="flex-1 min-h-0 overflow-y-auto p-4 bg-[#f9fbfd]">
              {detailLoading && !detail ? (
                <p className="text-sm text-slate-500">Loading document…</p>
              ) : cardTab === 'general' ? (
                <div className="bg-white border border-slate-200 rounded-sm p-4 shadow-sm max-w-3xl">
                  <p className="text-[11px] font-semibold text-slate-500 uppercase mb-3">Invoice &amp; customer</p>
                  {detail ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-4 p-3 rounded border border-slate-200 bg-slate-50/90 text-[11px]">
                      <div>
                        <span className="text-slate-500 font-semibold">Status</span>
                        <p className="text-slate-900">Posted to Finance (upload lock)</p>
                      </div>
                      <div>
                        <span className="text-slate-500 font-semibold">Document type</span>
                        <p className="text-slate-900">{detail.voucher_type}</p>
                      </div>
                      <div className="sm:col-span-2">
                        <span className="text-slate-500 font-semibold">Seller (from upload)</span>
                        <p className="text-slate-900 truncate" title={(detail.meta?.seller_company || '').toString()}>
                          {(detail.meta?.seller_company || '—').toString()}
                          {(detail.meta?.seller_gstin || '').toString() ? (
                            <span className="block font-mono text-[10px] text-slate-600 mt-0.5">{detail.meta?.seller_gstin}</span>
                          ) : null}
                        </p>
                      </div>
                    </div>
                  ) : null}
                  {selected?.row_kind === 'upload_summary' ? (
                    <p className="text-[11px] text-amber-900 bg-amber-50 border border-amber-200 rounded px-3 py-2 mb-3 leading-snug">
                      <strong>SUP- / SUM-</strong> is the <strong>upload file summary</strong> (seller branch / GSTIN slice), not a buyer invoice. One Amazon MTR often creates <strong>several SUP- rows</strong> — one locked upload per <strong>seller GSTIN</strong> in the file — so you may see multiple company names ending in a state even though you uploaded a single file. This screen opens on <strong>Lines</strong> by default; use <strong>General</strong> for file and totals. Open <strong>SUE-</strong> for a single customer invoice.
                    </p>
                  ) : null}
                  {selected?.row_kind === 'upload_summary' && detail && lineItemCount > 0 ? (
                    <div className="mb-4 rounded border border-slate-200 bg-white overflow-hidden shadow-sm">
                      <div className="px-3 py-2 border-b border-slate-100 flex flex-wrap items-center justify-between gap-2 bg-slate-50/90">
                        <p className="text-[11px] font-semibold text-slate-700">SKU preview (first {Math.min(5, lineItemCount)} of {lineItemCount})</p>
                        <button
                          type="button"
                          className="text-[11px] font-semibold text-teal-800 hover:underline"
                          onClick={() => setCardTab('lines')}
                        >
                          All columns on Lines →
                        </button>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-[11px]">
                          <thead className="bg-slate-50 text-slate-600">
                            <tr>
                              <th className="text-left px-2 py-1.5">SKU / item no.</th>
                              <th className="text-right px-2 py-1.5">Qty</th>
                              <th className="text-right px-2 py-1.5">Invoice amt.</th>
                              <th className="text-left px-2 py-1.5">Invoice</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(detail.meta!.line_items as Record<string, unknown>[]).slice(0, 5).map((li, i) => (
                              <tr key={i} className="border-t border-slate-100">
                                <td className="px-2 py-1.5 font-mono text-slate-800 max-w-[14rem] truncate" title={liStr(li, 'sku', 'SKU', 'item_no', 'Item_No')}>
                                  {liStr(li, 'sku', 'SKU', 'item_no', 'Item_No') || '—'}
                                </td>
                                <td className="px-2 py-1.5 text-right tabular-nums">{Number(li.quantity ?? li.Quantity ?? 0) || '—'}</td>
                                <td className="px-2 py-1.5 text-right tabular-nums">
                                  {fmtDec(Number(li.invoice_amount ?? li.Invoice_Amount ?? 0))}
                                </td>
                                <td className="px-2 py-1.5 font-mono text-slate-600 max-w-[8rem] truncate" title={liStr(li, 'source_invoice_no', 'invoice_number', 'Invoice_Number')}>
                                  {liStr(li, 'source_invoice_no', 'invoice_number', 'Invoice_Number') || '—'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : null}
                  <SiFormField label="No. / external" value={draft.invoice_no ?? ''} onChange={v => setD('invoice_no', v)} />
                  <SiFormField label="Posting date" value={draft.voucher_date ?? ''} onChange={v => setD('voucher_date', v)} type="date" />
                  <SiFormField label="Document date" value={draft.bill_date ?? ''} onChange={v => setD('bill_date', v)} type="date" />
                  <div className="grid grid-cols-1 sm:grid-cols-[10rem_1fr] gap-1 sm:gap-2 py-1.5 border-b border-slate-200/80">
                    <div className="flex flex-col gap-1 pt-1">
                      <label className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">Customer name</label>
                      <span className="text-[10px] text-slate-500 normal-case font-normal">Open ledger views (chart party)</span>
                    </div>
                    <div className="flex flex-col sm:flex-row gap-2 sm:items-start">
                      <input
                        type="text"
                        value={draft.party_name ?? ''}
                        onChange={e => setD('party_name', e.target.value)}
                        className="text-sm border rounded px-2 py-1.5 w-full border-slate-300 bg-white min-w-0"
                        aria-label="Customer name from sale file"
                      />
                      <div className="flex flex-wrap gap-1.5 shrink-0">
                        <button
                          type="button"
                          className="text-[11px] font-semibold px-2 py-1.5 rounded border border-teal-600 text-teal-800 bg-teal-50 hover:bg-teal-100 whitespace-nowrap"
                          onClick={() => openTrialBalanceForParty(partyTbQuery)}
                          disabled={!partyTbQuery}
                          title="Filter Trial Balance by this party name / GSTIN"
                        >
                          Trial balance
                        </button>
                        <button
                          type="button"
                          className="text-[11px] font-semibold px-2 py-1.5 rounded border border-slate-400 text-slate-800 bg-slate-50 hover:bg-slate-100 whitespace-nowrap"
                          onClick={() => openMastersLedgersForParty(partyMasterQuery)}
                          disabled={!partyMasterQuery}
                          title="Masters → Ledgers, filtered by GSTIN (if 15 chars) or name"
                        >
                          Ledger master
                        </button>
                      </div>
                    </div>
                  </div>
                  <SiFormField label="Customer GSTIN (B2B, from file)" value={draft.party_gstin ?? ''} onChange={v => setD('party_gstin', v)} />
                  <SiFormField label="Customer / ship-to state (from file)" value={draft.party_state ?? ''} onChange={v => setD('party_state', v)} />
                  <SiFormField label="Ship-to location (city, state from file)" value={draft.ship_to_state ?? ''} onChange={v => setD('ship_to_state', v)} />
                  <SiFormField label="External doc no. (order)" value={draft.order_id ?? ''} onChange={v => setD('order_id', v)} />
                  <SiFormField label="Source file" value={draft.source_filename ?? ''} onChange={v => setD('source_filename', v)} />
                  <SiFormField label="Platform" value={draft.platform ?? ''} onChange={v => setD('platform', v)} />
                  <SiFormField label="Period" value={draft.period ?? ''} onChange={v => setD('period', v)} />
                  <SiFormField label="Supply type" value={draft.supply_type ?? ''} onChange={v => setD('supply_type', v)} />
                  <div className="grid grid-cols-1 sm:grid-cols-[10rem_1fr] gap-1 sm:gap-2 py-1.5 border-b border-slate-200/80">
                    <label className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide pt-1.5">Description</label>
                    <textarea
                      value={draft.narration ?? ''}
                      onChange={e => setD('narration', e.target.value)}
                      rows={3}
                      className="text-sm border border-slate-300 rounded px-2 py-1.5 w-full"
                    />
                  </div>
                  <p className="text-[11px] font-semibold text-slate-500 uppercase mt-5 mb-1">Default dimensions (D365-style)</p>
                  <p className="text-[10px] text-slate-500 mb-2 leading-snug">
                    Attribute (category name) → Value code → Description (friendly tag). Stored on this document and returned with the voucher.
                  </p>
                  <div className="space-y-2 mb-4">
                    {dimRows.map((dr, idx) => (
                      <div
                        key={`dim-${idx}`}
                        className="grid grid-cols-1 sm:grid-cols-[1fr_1fr_1fr_auto] gap-2 items-end rounded border border-slate-200 bg-slate-50/80 p-2"
                      >
                        <div>
                          <label className="block text-[10px] font-semibold text-slate-500 uppercase mb-0.5">Attribute</label>
                          <input
                            type="text"
                            value={dr.attribute_name}
                            onChange={e => {
                              const v = e.target.value
                              setDimRows(prev => prev.map((row, i) => (i === idx ? { ...row, attribute_name: v } : row)))
                            }}
                            className="text-sm border border-slate-300 rounded px-2 py-1.5 w-full bg-white"
                            placeholder="e.g. Department"
                          />
                        </div>
                        <div>
                          <label className="block text-[10px] font-semibold text-slate-500 uppercase mb-0.5">Value code</label>
                          <input
                            type="text"
                            value={dr.value_code}
                            onChange={e => {
                              const v = e.target.value
                              setDimRows(prev => prev.map((row, i) => (i === idx ? { ...row, value_code: v } : row)))
                            }}
                            className="text-sm border border-slate-300 rounded px-2 py-1.5 w-full bg-white"
                            placeholder="e.g. 100"
                          />
                        </div>
                        <div>
                          <label className="block text-[10px] font-semibold text-slate-500 uppercase mb-0.5">Description</label>
                          <input
                            type="text"
                            value={dr.value_description}
                            onChange={e => {
                              const v = e.target.value
                              setDimRows(prev => prev.map((row, i) => (i === idx ? { ...row, value_description: v } : row)))
                            }}
                            className="text-sm border border-slate-300 rounded px-2 py-1.5 w-full bg-white"
                            placeholder="e.g. Sales Department"
                          />
                        </div>
                        <button
                          type="button"
                          className="text-[11px] font-semibold px-2 py-1.5 rounded border border-slate-300 text-slate-700 hover:bg-slate-100 whitespace-nowrap"
                          onClick={() => setDimRows(prev => (prev.length <= 1 ? prev : prev.filter((_, i) => i !== idx)))}
                        >
                          Remove
                        </button>
                      </div>
                    ))}
                    <button
                      type="button"
                      className="text-[11px] font-semibold text-teal-800 hover:underline"
                      onClick={() => setDimRows(prev => [...prev, { attribute_name: '', value_code: '', value_description: '' }])}
                    >
                      + Add dimension row
                    </button>
                  </div>
                  <p className="text-[11px] font-semibold text-slate-500 uppercase mt-6 mb-3">Amounts</p>
                  <SiFormField label="Taxable" value={draft.taxable_amount ?? ''} onChange={v => setD('taxable_amount', v)} />
                  <SiFormField label="CGST" value={draft.cgst_amount ?? ''} onChange={v => setD('cgst_amount', v)} />
                  <SiFormField label="SGST" value={draft.sgst_amount ?? ''} onChange={v => setD('sgst_amount', v)} />
                  <SiFormField label="IGST" value={draft.igst_amount ?? ''} onChange={v => setD('igst_amount', v)} />
                  <SiFormField label="Total" value={draft.total_amount ?? ''} onChange={v => setD('total_amount', v)} />
                  <SiFormField label="Net" value={draft.net_payable ?? ''} onChange={v => setD('net_payable', v)} />
                </div>
              ) : cardTab === 'lines' ? (
                <div className="bg-transparent rounded-sm space-y-2">
                  {selected?.row_kind === 'upload_summary' ? (
                    <p className="text-[11px] text-slate-700 bg-slate-100 border border-slate-200 rounded px-3 py-2 leading-snug">
                      These rows are rolled-up <strong>SKU / item lines</strong> from every saved invoice (<strong>SUE-…</strong>) for this upload. Use <strong>General</strong> for file name, period, and header totals.
                    </p>
                  ) : null}
                  {detail?.meta?.line_items && detail.meta.line_items.length > 0 ? (
                    <SalesInvoiceBcDetailPanel detail={detail} lineItemsShowSourceInv={lineItemsShowSourceInv} />
                  ) : (
                    <div className="bg-white border border-slate-200 rounded-sm p-4 shadow-sm">
                      <p className="text-sm text-slate-600 mb-2">No SKU line items are stored for this document yet.</p>
                      <p className="text-xs text-slate-500 leading-relaxed">
                        For an upload summary (<strong>SUP-</strong>), the grid is filled from parsed MTR rows saved as <strong>SUE-</strong> entries. If this upload predates that step, <strong>re-upload the same CSV</strong> from Finance → Sales uploads. You can also click an <strong>SUE-</strong> row in the list for lines on a single invoice.
                      </p>
                    </div>
                  )}
                </div>
              ) : cardTab === 'inventory' ? (
                <div className="bg-white border border-slate-200 rounded-sm p-4 shadow-sm">
                  <p className="text-[11px] text-slate-500 mb-3">Inventory view from this sales document’s line_items.</p>
                  {inventoryRows.length === 0 ? (
                    <p className="text-sm text-slate-500">No inventory lines found.</p>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-[11px]">
                        <thead className="bg-slate-50 text-slate-600">
                          <tr>
                            <th className="px-2 py-1.5 text-left">SKU</th>
                            <th className="px-2 py-1.5 text-left">Product</th>
                            <th className="px-2 py-1.5 text-right">Out</th>
                            <th className="px-2 py-1.5 text-right">In</th>
                            <th className="px-2 py-1.5 text-right">Net</th>
                          </tr>
                        </thead>
                        <tbody>
                          {inventoryRows.map((r, i) => (
                            <tr key={`${r.sku}-${i}`} className="border-t border-slate-100">
                              <td className="px-2 py-1.5 font-mono">{r.sku}</td>
                              <td className="px-2 py-1.5">{r.product_name || '—'}</td>
                              <td className="px-2 py-1.5 text-right tabular-nums">{fmtQtyCell(r.qty_out)}</td>
                              <td className="px-2 py-1.5 text-right tabular-nums">{fmtQtyCell(r.qty_in)}</td>
                              <td className="px-2 py-1.5 text-right tabular-nums">{fmtQtyCell(r.net_qty)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              ) : (
                <div className="bg-white border border-slate-200 rounded-sm p-4 shadow-sm">
                  <p className="text-[11px] text-slate-500 mb-3">Location view from this sales document’s line_items.</p>
                  {locationRows.length === 0 ? (
                    <p className="text-sm text-slate-500">No location lines found.</p>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-[11px]">
                        <thead className="bg-slate-50 text-slate-600">
                          <tr>
                            <th className="px-2 py-1.5 text-left">Location</th>
                            <th className="px-2 py-1.5 text-left">Ship-to</th>
                            <th className="px-2 py-1.5 text-right">Qty</th>
                            <th className="px-2 py-1.5 text-right">Lines</th>
                          </tr>
                        </thead>
                        <tbody>
                          {locationRows.map((r, i) => (
                            <tr key={`${r.location}-${r.ship_to_state}-${i}`} className="border-t border-slate-100">
                              <td className="px-2 py-1.5">{r.location || '—'}</td>
                              <td className="px-2 py-1.5">{r.ship_to_state || '—'}</td>
                              <td className="px-2 py-1.5 text-right tabular-nums">{fmtQtyCell(r.qty)}</td>
                              <td className="px-2 py-1.5 text-right tabular-nums">{r.lines}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </div>

            <footer className="shrink-0 flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-t border-slate-200 bg-white">
              <p className="text-xs text-slate-500">{saveMsg ? <span className={saveMsg === 'Saved.' ? 'text-green-700' : 'text-red-600'}>{saveMsg}</span> : null}</p>
              <div className="flex gap-2">
                <button type="button" className="text-xs px-4 py-2 rounded border border-slate-300 hover:bg-slate-50" onClick={() => setCardOpen(false)}>
                  Cancel
                </button>
                <button
                  type="button"
                  disabled={saveMutation.isPending || detailLoading}
                  className="text-xs px-4 py-2 rounded bg-[#002B5B] text-white font-semibold hover:bg-blue-900 disabled:opacity-50"
                  onClick={() => void saveInvoice()}
                >
                  {saveMutation.isPending ? 'Saving…' : 'Save'}
                </button>
              </div>
            </footer>
          </div>
        </div>
      )}
    </div>
  )
}

interface InventoryMovementRow {
  sku: string
  product_name: string
  qty_out: number
  qty_in: number
  net_qty: number
  line_count: number
}

function fmtQtyCell(n: number) {
  if (!Number.isFinite(n)) return '—'
  const t = Math.abs(n % 1) < 1e-9 ? n.toFixed(0) : n.toFixed(3)
  return Number(t).toLocaleString('en-IN')
}

// ── Finance inventory (qty from posted sales line_items) ────────────
function FinanceInventoryTab() {
  const [startDate, setStartDate] = useState(() => monthsAgo(1))
  const [endDate, setEndDate] = useState(TODAY)
  const [search, setSearch] = useState('')

  const q = useMemo(() => {
    const p = new URLSearchParams()
    if (startDate) p.set('start_date', startDate)
    if (endDate) p.set('end_date', endDate)
    if (search.trim()) p.set('search', search.trim())
    return p.toString()
  }, [startDate, endDate, search])

  const { data: rows = [], isLoading } = useQuery<InventoryMovementRow[]>({
    queryKey: ['finance-inventory-movements', startDate, endDate, search],
    queryFn: async () => {
      const { data } = await api.get<InventoryMovementRow[]>(`/finance/inventory-movements?${q}`)
      return data
    },
    staleTime: 30 * 1000,
  })

  const totals = useMemo(() => {
    let o = 0
    let i = 0
    for (const r of rows) {
      o += r.qty_out
      i += r.qty_in
    }
    return { out: o, in: i, net: o - i }
  }, [rows])

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-teal-100 bg-teal-50/70 px-4 py-3 text-[11px] text-teal-950 leading-snug">
        <strong>Outbound</strong> totals shipment line quantities from posted Finance sales entries;{' '}
        <strong>Inbound</strong> totals return / credit-memo lines (stock coming back). This view reads only the Finance lock DB — it does not replace the operational Inventory upload or warehouse on-hand.
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-3 flex flex-wrap gap-3 items-end">
        <div>
          <label className="text-xs text-gray-500">From</label>
          <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} className="block text-xs border border-gray-200 rounded px-2 py-1.5" />
        </div>
        <div>
          <label className="text-xs text-gray-500">To</label>
          <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="block text-xs border border-gray-200 rounded px-2 py-1.5" />
        </div>
        <div>
          <label className="text-xs text-gray-500">SKU / product contains</label>
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Filter…"
            className="block text-xs border border-gray-200 rounded px-2 py-1.5 w-52"
          />
        </div>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-4 py-2 border-b border-gray-100 flex flex-wrap justify-between gap-2 items-center">
          <h3 className="text-sm font-semibold text-gray-800">SKU movement (Finance)</h3>
          <span className="text-xs text-gray-500">
            {rows.length} SKU{rows.length !== 1 ? 's' : ''} · out {fmtQtyCell(totals.out)} · in {fmtQtyCell(totals.in)} · net {fmtQtyCell(totals.net)}
          </span>
        </div>
        {isLoading ? (
          <div className="p-8 text-center text-gray-400 text-sm">Loading movements…</div>
        ) : rows.length === 0 ? (
          <div className="p-8 text-center text-gray-400 text-sm">
            No line quantities in range. Post Amazon (or other) sales with line_items from Finance → Sales uploads, then refresh.
          </div>
        ) : (
          <div className="overflow-x-auto max-h-[560px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-gray-50 z-[1] shadow-sm">
                <tr className="text-gray-500 uppercase tracking-wide">
                  <th className="px-3 py-2 text-left">SKU</th>
                  <th className="px-3 py-2 text-left">Product</th>
                  <th className="px-3 py-2 text-right">Outbound</th>
                  <th className="px-3 py-2 text-right">Inbound</th>
                  <th className="px-3 py-2 text-right">Net</th>
                  <th className="px-3 py-2 text-right">Lines</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(r => (
                  <tr key={r.sku} className="border-t border-gray-100 hover:bg-teal-50/40">
                    <td className="px-3 py-1.5 font-mono text-gray-800">{r.sku}</td>
                    <td className="px-3 py-1.5 text-gray-600 max-w-[20rem] truncate" title={r.product_name}>{r.product_name || '—'}</td>
                    <td className="px-3 py-1.5 text-right tabular-nums text-emerald-800 font-medium">{fmtQtyCell(r.qty_out)}</td>
                    <td className="px-3 py-1.5 text-right tabular-nums text-amber-800 font-medium">{fmtQtyCell(r.qty_in)}</td>
                    <td className="px-3 py-1.5 text-right tabular-nums font-semibold text-gray-900">{fmtQtyCell(r.net_qty)}</td>
                    <td className="px-3 py-1.5 text-right tabular-nums text-gray-500">{r.line_count}</td>
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

// ── Customer Ledger (BC / D365-style export log) ───────────────────
function CustomerLedgerEntriesTab() {
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [search, setSearch] = useState('')
  const [docKind, setDocKind] = useState<'all' | 'sales' | 'credit_memo'>('all')

  const q = useMemo(() => {
    const p = new URLSearchParams()
    if (docKind !== 'all') p.set('document_kind', docKind === 'sales' ? 'sales' : 'credit_memo')
    if (startDate) p.set('start_date', startDate)
    if (endDate) p.set('end_date', endDate)
    if (search.trim()) p.set('search', search.trim())
    return p.toString()
  }, [startDate, endDate, search, docKind])

  const { data: rows = [], isLoading } = useQuery<CustomerLedgerEntry[]>({
    queryKey: ['finance-customer-ledger', startDate, endDate, search, docKind],
    queryFn: async () => {
      const { data } = await api.get(`/finance/customer-ledger-entries?${q}`)
      return data
    },
    staleTime: 30 * 1000,
  })

  const stats = useMemo(() => {
    let taxable = 0
    let gst = 0
    let inv = 0
    for (const r of rows) {
      taxable += Number(r.taxable_amount) || 0
      gst += Number(r.gst_amount) || 0
      inv += Number(r.invoice_amount) || 0
    }
    return { taxable, gst, inv, count: rows.length }
  }, [rows])

  const exportBcCsv = () => {
    const headers = [
      'Document Date', 'Document Type', 'Document No.', 'Customer No.', 'Customer Name', 'Description', 'BRANCH CODE',
      'Taxable Amount2', 'Gst Amount', 'Invoice Amount', 'Due Date', 'GST Customer Type', 'Seller State Code',
      'Seller GST Reg. No.', 'Location Code', 'Location State Code', 'GST Jurisdiction Type', 'External Document No.',
      'Location GST Reg. No.',
    ]
    const csv: (string | number)[][] = [headers]
    for (const r of rows) {
      csv.push([
        r.document_date,
        r.document_type,
        r.document_no,
        r.customer_no,
        r.customer_name,
        r.description,
        r.branch_code,
        r.taxable_amount,
        r.gst_amount,
        r.invoice_amount,
        r.due_date,
        r.gst_customer_type,
        r.seller_state_code,
        r.seller_gst_reg_no,
        r.location_code,
        r.location_state_code,
        r.gst_jurisdiction_type,
        r.external_document_no,
        r.location_gst_reg_no,
      ])
    }
    downloadCsv(`customer-ledger-entries-${startDate || 'all'}-${endDate || 'all'}.csv`, csv)
  }

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-3 flex flex-wrap gap-2 items-end justify-between">
        <div className="flex flex-wrap gap-2 items-end">
          <div>
            <label className="text-xs text-gray-500">From</label>
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} className="block text-xs border border-gray-200 rounded px-2 py-1.5" />
          </div>
          <div>
            <label className="text-xs text-gray-500">To</label>
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="block text-xs border border-gray-200 rounded px-2 py-1.5" />
          </div>
          <div>
            <label className="text-xs text-gray-500">Search</label>
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Document / customer / order / GSTIN"
              className="block text-xs border border-gray-200 rounded px-2 py-1.5 w-56"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500">Documents</label>
            <select
              value={docKind}
              onChange={e => setDocKind(e.target.value as 'all' | 'sales' | 'credit_memo')}
              className="block text-xs border border-gray-200 rounded px-2 py-1.5 text-gray-800"
            >
              <option value="all">All (invoices + credit memos)</option>
              <option value="sales">Invoices only</option>
              <option value="credit_memo">Credit memos only</option>
            </select>
          </div>
        </div>
        <button
          type="button"
          onClick={exportBcCsv}
          className="text-xs font-semibold px-3 py-1.5 rounded border border-teal-600 text-teal-800 bg-teal-50 hover:bg-teal-100"
        >
          Export (BC column names, CSV)
        </button>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-4 py-2.5 border-b border-gray-100 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-gray-800">Customer ledger entries</h3>
            <p className="text-[11px] text-gray-500 mt-0.5 max-w-3xl">
              Dynamics-style log from posted Finance sales lines (same column names as a BC Customer Ledger export). Edit headers on Sales invoices if amounts or parties need correction — this view reflects those overlays.
            </p>
          </div>
          <span className="text-xs text-gray-500 whitespace-nowrap">
            {stats.count} rows · taxable {fmtDec(stats.taxable)} · GST {fmtDec(stats.gst)} · invoice {fmtDec(stats.inv)}
          </span>
        </div>
        {isLoading ? (
          <div className="p-8 text-center text-gray-400 text-sm">Loading ledger…</div>
        ) : rows.length === 0 ? (
          <div className="p-8 text-center text-gray-400 text-sm">No posted customer lines match your filters. Parse sales uploads into invoice rows first.</div>
        ) : (
          <div className="overflow-x-auto max-h-[560px] overflow-y-auto">
            <table className="w-full text-[11px] min-w-[1200px]">
              <thead className="sticky top-0 bg-gray-50 z-[1] shadow-sm">
                <tr className="text-gray-500 uppercase tracking-wide">
                  <th className="px-2 py-2 text-left whitespace-nowrap">Doc date</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">Type</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">Document no.</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">Customer</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">Description</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">Branch</th>
                  <th className="px-2 py-2 text-right whitespace-nowrap">Taxable</th>
                  <th className="px-2 py-2 text-right whitespace-nowrap">GST</th>
                  <th className="px-2 py-2 text-right whitespace-nowrap">Invoice amt.</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">Due</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">GST cust.</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">Seller st.</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">Seller GSTIN</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">Loc. code</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">Loc. state</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">Jurisdiction</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">External doc</th>
                  <th className="px-2 py-2 text-left whitespace-nowrap">Loc. GSTIN</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(r => (
                  <tr key={r.id} className="border-t border-gray-100 hover:bg-teal-50/40">
                    <td className="px-2 py-1.5 whitespace-nowrap text-gray-700">{r.document_date}</td>
                    <td className="px-2 py-1.5 whitespace-nowrap text-gray-800">{r.document_type}</td>
                    <td className="px-2 py-1.5 font-mono text-teal-900 whitespace-nowrap">{r.document_no || '—'}</td>
                    <td className="px-2 py-1.5 text-gray-800 max-w-[10rem] truncate" title={r.customer_name}>{r.customer_name || '—'}</td>
                    <td className="px-2 py-1.5 text-gray-600 max-w-[12rem] truncate" title={r.description}>{r.description || '—'}</td>
                    <td className="px-2 py-1.5 text-gray-600 max-w-[8rem] truncate" title={r.branch_code}>{r.branch_code || '—'}</td>
                    <td className="px-2 py-1.5 text-right tabular-nums">{fmtDec(Number(r.taxable_amount) || 0)}</td>
                    <td className="px-2 py-1.5 text-right tabular-nums">{fmtDec(Number(r.gst_amount) || 0)}</td>
                    <td className="px-2 py-1.5 text-right font-semibold tabular-nums">{fmtDec(Number(r.invoice_amount) || 0)}</td>
                    <td className="px-2 py-1.5 whitespace-nowrap text-gray-500">{r.due_date}</td>
                    <td className="px-2 py-1.5 whitespace-nowrap">{r.gst_customer_type}</td>
                    <td className="px-2 py-1.5 whitespace-nowrap">{r.seller_state_code || '—'}</td>
                    <td className="px-2 py-1.5 font-mono text-[10px] text-gray-600 max-w-[7rem] truncate" title={r.seller_gst_reg_no}>{r.seller_gst_reg_no || '—'}</td>
                    <td className="px-2 py-1.5 font-mono whitespace-nowrap">{r.location_code || '—'}</td>
                    <td className="px-2 py-1.5 whitespace-nowrap">{r.location_state_code || '—'}</td>
                    <td className="px-2 py-1.5 whitespace-nowrap">{r.gst_jurisdiction_type}</td>
                    <td className="px-2 py-1.5 font-mono text-[10px] max-w-[9rem] truncate" title={r.external_document_no}>{r.external_document_no || '—'}</td>
                    <td className="px-2 py-1.5 font-mono text-[10px] max-w-[7rem] truncate" title={r.location_gst_reg_no}>{r.location_gst_reg_no || '—'}</td>
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

// ── Day Book Tab ──────────────────────────────────────────────────
function DaybookTab({
  openTrialBalanceForParty,
  openMastersLedgersForParty,
}: {
  openTrialBalanceForParty: (q: string) => void
  openMastersLedgersForParty: (q: string) => void
}) {
  const qc = useQueryClient()
  const [dateStr, setDateStr] = useState(toIso(new Date()))
  const [openEntryId, setOpenEntryId] = useState<number | null>(null)

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

  const { data: entryDetail, isLoading: entryLoading } = useQuery<DaybookVoucher>({
    queryKey: ['finance-voucher-detail', openEntryId],
    queryFn: async () => {
      const { data } = await api.get(`/finance/vouchers/${openEntryId}`)
      return data
    },
    enabled: openEntryId != null && openEntryId > 0,
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

  const entryPartyTb = (entryDetail?.party_name || '').trim() || (entryDetail?.party_gstin || '').trim()
  const entryPartyMaster = (() => {
    const g = (entryDetail?.party_gstin || '').replace(/\s/g, '')
    if (g.length >= 15) return g
    return (entryDetail?.party_name || '').trim()
  })()
  const daybookLineItemsShowInv = (entryDetail?.meta?.line_items ?? []).some(
    li => String((li as { source_invoice_no?: string }).source_invoice_no || '').trim() !== '',
  )

  return (
    <div className="space-y-4">
      {openEntryId != null && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="sales-entry-title"
          onClick={() => setOpenEntryId(null)}
        >
          <div
            className="bg-white rounded-xl shadow-xl max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col border border-gray-200"
            onClick={e => e.stopPropagation()}
          >
            <div className="px-5 py-3 border-b border-gray-100 flex items-start justify-between gap-3 bg-[#002B5B] text-white">
              <div>
                <h3 id="sales-entry-title" className="text-sm font-semibold">
                  {entryDetail?.voucher_no ?? '…'} — Sales entry
                </h3>
                <p className="text-xs text-blue-200 mt-0.5">
                  {entryDetail?.meta?.platform ?? ''}{entryDetail?.meta?.period ? ` · ${entryDetail.meta.period}` : ''}
                </p>
              </div>
              <button
                type="button"
                className="text-xs font-medium text-white/90 hover:text-white px-2 py-1 rounded border border-white/30"
                onClick={() => setOpenEntryId(null)}
              >
                Close
              </button>
            </div>
            {entryLoading || !entryDetail ? (
              <div className="p-10 text-center text-gray-400 text-sm">Loading…</div>
            ) : (
              <div className="overflow-y-auto p-5 space-y-5 text-xs">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="rounded-lg border border-gray-200 p-3 space-y-2">
                    <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wide">General</p>
                    <div className="flex flex-wrap items-start gap-2 justify-between">
                      <p className="min-w-0">
                        <span className="text-gray-500">Customer:</span>{' '}
                        <span className="font-medium text-gray-800">{entryDetail.party_name || '—'}</span>
                      </p>
                      <div className="flex flex-wrap gap-1 shrink-0">
                        <button
                          type="button"
                          className="text-[10px] font-semibold px-2 py-1 rounded border border-teal-600 text-teal-800 bg-teal-50 hover:bg-teal-100"
                          onClick={() => openTrialBalanceForParty(entryPartyTb)}
                          disabled={!entryPartyTb}
                        >
                          Trial balance
                        </button>
                        <button
                          type="button"
                          className="text-[10px] font-semibold px-2 py-1 rounded border border-gray-300 text-gray-800 bg-gray-50 hover:bg-gray-100"
                          onClick={() => openMastersLedgersForParty(entryPartyMaster)}
                          disabled={!entryPartyMaster}
                        >
                          Ledger master
                        </button>
                      </div>
                    </div>
                    <p><span className="text-gray-500">Document date:</span> <span className="font-mono">{entryDetail.voucher_date}</span></p>
                    <p><span className="text-gray-500">Invoice no.:</span> <span className="font-mono">{entryDetail.meta?.invoice_no || entryDetail.bill_no || '—'}</span></p>
                    <p><span className="text-gray-500">Order ID (external):</span> <span className="font-mono break-all">{entryDetail.meta?.order_id || entryDetail.ref_number || '—'}</span></p>
                    <p><span className="text-gray-500">Customer GSTIN:</span> <span className="font-mono">{entryDetail.party_gstin || '—'}</span></p>
                    <p><span className="text-gray-500">Customer / ship-to state:</span> {entryDetail.party_state || '—'}</p>
                    <p><span className="text-gray-500">Ship-to location:</span> {entryDetail.meta?.ship_to_state || '—'}</p>
                    <p><span className="text-gray-500">Seller GSTIN (upload):</span> <span className="font-mono">{entryDetail.meta?.seller_gstin || '—'}</span></p>
                    <p><span className="text-gray-500">Seller company (upload):</span> {entryDetail.meta?.seller_company || '—'}</p>
                    <p><span className="text-gray-500">Seller branch state:</span> {entryDetail.meta?.seller_state || '—'}</p>
                    <p><span className="text-gray-500">Supply:</span> {entryDetail.supply_type === 'Inter' ? 'Inter-state (IGST)' : entryDetail.supply_type === 'Intra' ? 'Intra-state (CGST+SGST)' : '—'}</p>
                    <p><span className="text-gray-500">Source file:</span> <span className="break-all">{entryDetail.meta?.source_filename || '—'}</span></p>
                    {entryDetail.narration ? <p className="text-gray-600 pt-1 border-t border-gray-100">{entryDetail.narration}</p> : null}
                  </div>
                  <div className="rounded-lg border border-gray-200 p-3 space-y-2 bg-gray-50/80">
                    <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wide">Amounts</p>
                    <p><span className="text-gray-500">Taxable:</span> <span className="font-semibold">{fmt(entryDetail.taxable_amount)}</span></p>
                    <p><span className="text-gray-500">CGST:</span> {entryDetail.cgst_amount > 0 ? fmt(entryDetail.cgst_amount) : '—'}</p>
                    <p><span className="text-gray-500">SGST:</span> {entryDetail.sgst_amount > 0 ? fmt(entryDetail.sgst_amount) : '—'}</p>
                    <p><span className="text-gray-500">IGST:</span> {entryDetail.igst_amount > 0 ? fmt(entryDetail.igst_amount) : '—'}</p>
                    <p><span className="text-gray-500">Total:</span> <span className="font-semibold">{fmt(entryDetail.total_amount)}</span></p>
                    <p><span className="text-gray-500">Net payable:</span> <span className="font-bold text-[#002B5B]">{fmt(entryDetail.net_payable)}</span></p>
                  </div>
                </div>
                <div>
                  <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wide mb-2">Lines &amp; tax (BC-style)</p>
                  {(entryDetail.meta?.line_items?.length ?? 0) === 0 && (entryDetail.lines?.length ?? 0) === 0 ? (
                    <p className="text-gray-400">No line detail stored for this entry.</p>
                  ) : (entryDetail.meta?.line_items?.length ?? 0) > 0 ? (
                    <SalesInvoiceBcDetailPanel detail={entryDetail} lineItemsShowSourceInv={daybookLineItemsShowInv} />
                  ) : (
                    <div className="overflow-x-auto rounded-lg border border-gray-200">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="bg-gray-50 text-gray-500 uppercase tracking-wide">
                            <th className="px-3 py-2 text-left">Ledger / head</th>
                            <th className="px-3 py-2 text-right">Amount</th>
                            <th className="px-3 py-2 text-left">Dim / centre</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(entryDetail.lines ?? []).map((ln, i) => (
                            <tr key={`l-${i}`} className="border-t border-gray-100">
                              <td className="px-3 py-1.5 font-mono">{ln.expense_head}</td>
                              <td className="px-3 py-1.5 text-right">{fmt(ln.amount)}</td>
                              <td className="px-3 py-1.5">{ln.cost_centre || ''}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
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
                  <tr
                    key={v.id}
                    className="border-t border-gray-50 hover:bg-gray-50 cursor-pointer"
                    onClick={() => setOpenEntryId(v.id)}
                  >
                    <td className="px-3 py-2 text-gray-500">{v.voucher_date}</td>
                    <td className="px-3 py-2 font-mono font-semibold text-[#002B5B] underline-offset-2 hover:underline">{v.voucher_no}</td>
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
                      {v.id < 1_000_000 ? (
                        <button
                          type="button"
                          onClick={e => {
                            e.stopPropagation()
                            if (window.confirm('Delete voucher ' + v.voucher_no + '?')) delMut.mutate(v.id)
                          }}
                          className="text-red-400 hover:text-red-600"
                        >
                          ✕
                        </button>
                      ) : (
                        <span className="text-gray-300" title="Sales upload entries are removed from Sales Uploads tab">—</span>
                      )}
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
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm px-4 py-3 flex flex-wrap items-center gap-3 justify-between">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Return Period</span>
          <input type="month" value={month} onChange={e => setMonth(e.target.value)}
            className="text-sm font-semibold border border-gray-200 rounded px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-300" />
          <span className="text-xs text-gray-400">{startDate} to {endDate}</span>
        </div>
        {!isLoading && gstr && (
          <button
            type="button"
            onClick={() => {
              const br = gstr.breakdown ?? []
              const rows: (string | number)[][] = [['voucher_no', 'voucher_date', 'voucher_type', 'party_name', 'taxable_amount', 'cgst', 'sgst', 'igst', 'total_amount']]
              for (const r of br) {
                rows.push([r.voucher_no, r.voucher_date, r.voucher_type, r.party_name, r.taxable_amount, r.cgst_amount, r.sgst_amount, r.igst_amount, r.total_amount])
              }
              downloadCsv(`gstr3b-${month}.csv`, rows)
            }}
            className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-teal-600 text-teal-800 bg-teal-50 hover:bg-teal-100"
          >
            Export breakdown (CSV)
          </button>
        )}
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
function VouchersTab({ voucherPreset }: { voucherPreset: { n: number; type: string } | null }) {
  const qc = useQueryClient()
  const [showVoucherHelp, setShowVoucherHelp] = useState(false)
  const lastPresetN = useRef(0)

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

  useEffect(() => {
    if (voucherPreset && voucherPreset.n > lastPresetN.current) {
      lastPresetN.current = voucherPreset.n
      setVType(voucherPreset.type)
    }
  }, [voucherPreset])
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
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-3">
        <button
          type="button"
          onClick={() => setShowVoucherHelp(v => !v)}
          className="w-full flex items-center justify-between text-left"
        >
          <span className="text-xs font-semibold text-blue-900 uppercase tracking-wide">
            Voucher Help: GL, Sales Fields & GST Rules
          </span>
          <span className="text-blue-700 text-xs">{showVoucherHelp ? 'Hide' : 'Show'}</span>
        </button>
        {showVoucherHelp && (
          <div className="mt-3 text-xs text-blue-900 space-y-1">
            <p><strong>Full GL &amp; Sales Report definitions</strong> (warehouse, e-invoice, e-way bill, ledger master, print checklist): open Finance → <strong>Help / Notes</strong>.</p>
            <p><strong>GL:</strong> Main record of posted amounts under sales, purchases, GST, bank/cash, parties, expenses.</p>
            <p><strong>Key fields:</strong> Import Order ID, Invoice No., Ship From / Ship To (and warehouse name on reports), type of sale (B2B/B2C/e-com), HSN, SKU/item name.</p>
            <div className="rounded border border-blue-300 bg-white/70 px-2 py-1.5 mt-2">
              <p className="font-semibold">Quick GST Decision Rule</p>
              <p>Same state (Ship From = Ship To): <strong>CGST + SGST</strong>. Different states: <strong>IGST</strong>.</p>
            </div>
          </div>
        )}
      </div>
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
function MastersTab({ mastersJump }: { mastersJump: { n: number; sub: MastersSubTab; ledgerSearch?: string } | null }) {
  const [sub, setSub] = useState<MastersSubTab>('ledger-groups')
  const lastJumpN = useRef(0)

  useEffect(() => {
    if (mastersJump && mastersJump.n > lastJumpN.current) {
      lastJumpN.current = mastersJump.n
      setSub(mastersJump.sub)
    }
  }, [mastersJump])

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
      {sub === 'ledgers'             && <LedgersSubTab mastersJump={mastersJump} />}
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

function LedgersSubTab({ mastersJump }: { mastersJump: { n: number; sub: MastersSubTab; ledgerSearch?: string } | null }) {
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
  const [listFilter, setListFilter] = useState('')
  const lastLedgerJumpN = useRef(0)
  useEffect(() => {
    if (
      mastersJump?.sub === 'ledgers' &&
      mastersJump.ledgerSearch &&
      mastersJump.n > lastLedgerJumpN.current
    ) {
      lastLedgerJumpN.current = mastersJump.n
      setListFilter(mastersJump.ledgerSearch)
    }
  }, [mastersJump])
  const { data: ledgers = [], isLoading } = useQuery<Ledger[]>({
    queryKey: ['finance-ledgers', filterGroup],
    queryFn: async () => {
      const params = filterGroup ? `?group_id=${filterGroup}` : ''
      const { data } = await api.get(`/finance/masters/ledgers${params}`)
      return data
    },
  })

  const visibleLedgers = useMemo(() => {
    const f = listFilter.trim().toLowerCase()
    if (!f) return ledgers
    return ledgers.filter(l =>
      (l.name || '').toLowerCase().includes(f) ||
      (l.alias || '').toLowerCase().includes(f) ||
      (l.gstin || '').toLowerCase().includes(f),
    )
  }, [ledgers, listFilter])

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
        <div className="px-4 py-3 border-b border-gray-100 flex flex-wrap items-center gap-3">
          <span className="text-xs text-gray-500">Filter by group:</span>
          <select value={filterGroup} onChange={e => setFilterGroup(e.target.value)}
            className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none">
            <option value="">All Groups</option>
            {groups.map(g => <option key={g.id} value={String(g.id)}>{g.name}</option>)}
          </select>
          <div className="flex items-center gap-2 ml-auto">
            <label className="text-xs text-gray-500">Name / GSTIN</label>
            <input
              type="text"
              value={listFilter}
              onChange={e => setListFilter(e.target.value)}
              placeholder="Filter table…"
              className="text-xs border border-gray-200 rounded px-2 py-1 w-44 focus:outline-none focus:ring-1 focus:ring-blue-300"
            />
          </div>
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
                {visibleLedgers.map(l => (
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
  skipped?: string[]
  state_breakdown?: { state: string; orders: number; gross_revenue: number; returns: number; net_revenue: number }[]
  company_breakdown?: { company: string; seller_gstin: string; company_state?: string; orders: number; gross_revenue: number; returns: number; net_revenue: number }[]
  saved_companies?: { id: number; company: string; seller_gstin: string; company_state: string; orders: number; gross_revenue: number; returns: number; net_revenue: number }[]
  // Monthly package fields
  saved?: { platform: string; id: number; orders: number; revenue: number; returns: number; net_revenue: number; note?: string }[]
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
  const [filterCompany,  setFilterCompany]  = useState('')

  const { data: uploads = [], isLoading } = useQuery<SalesUpload[]>({
    queryKey: ['finance-sales-uploads', filterPlatform, filterPeriod, filterCompany],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filterPlatform) params.set('platform', filterPlatform)
      if (filterPeriod)   params.set('period',   filterPeriod)
      if (filterCompany)  params.set('company',  filterCompany)
      const { data } = await api.get(`/finance/sales-uploads?${params}`)
      return data
    },
  })
  const companyOptions = useMemo(() => {
    const seen = new Set<string>()
    const out: string[] = []
    for (const r of uploads) {
      const c = (r.company_name || '').trim()
      const g = (r.seller_gstin || '').trim()
      if (c && !seen.has(c)) { seen.add(c); out.push(c) }
      if (g && !seen.has(g)) { seen.add(g); out.push(g) }
    }
    return out.sort((a, b) => a.localeCompare(b))
  }, [uploads])

  const delMut = useMutation({
    mutationFn: (id: number) => api.delete(`/finance/sales-uploads/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['finance-sales-uploads'] }),
  })

  const isPackage = platform === 'Monthly Package'
  const statsContext = useMemo(() => {
    const company = result?.company_breakdown?.[0]
    const state = result?.state_breakdown?.[0]?.state || 'Unknown'
    return {
      companyName: company?.company || 'Unknown',
      gstin: company?.seller_gstin || 'UNKNOWN',
      state,
    }
  }, [result])

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
          <p className="text-xs text-amber-700 mt-0.5">Records saved here stay in finance database and now automatically reflect across Finance tabs (Dashboard, Day Book, Vouchers, GSTR3B, P&amp;L, Platform Revenue). They do not merge into main Upload page / PO engine.</p>
        </div>
      </div>

      <GlSalesNotesContent />

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
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <span className="text-green-600 text-lg">✅</span>
                    <div>
                      <p className="text-sm font-semibold text-green-800">Upload saved — {(result.rows_parsed ?? result.total_orders).toLocaleString()} rows parsed</p>
                      <p className="text-xs text-green-600">{result.filename} · {result.platform} · {result.period}</p>
                    </div>
                  </div>
                  <div className="shrink-0 bg-white rounded-lg border border-green-200 px-3 py-2 text-[11px] text-right">
                    <p className="font-semibold text-gray-700">{statsContext.companyName}</p>
                    <p className="text-gray-500">GSTIN: {statsContext.gstin}</p>
                    <p className="text-gray-500">State: {statsContext.state}</p>
                  </div>
                </div>
                {result.saved_companies && result.saved_companies.length > 0 && (
                  <div className="mt-2 text-xs text-emerald-700">
                    Saved {result.saved_companies.length} company row(s) for this upload.
                  </div>
                )}
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
                {result.skipped && result.skipped.length > 0 && (
                  <div className="mt-2 text-xs text-amber-700">
                    Skipped: {result.skipped.join(' | ')}
                  </div>
                )}
                {(result.state_breakdown && result.state_breakdown.length > 0) && (
                  <div className="mt-3 bg-white rounded-lg border border-green-100 p-3">
                    <p className="text-xs font-semibold text-gray-700 mb-2">State-wise sales (where sale happened)</p>
                    <div className="overflow-x-auto">
                      <table className="w-full text-[11px]">
                        <thead>
                          <tr className="text-gray-500">
                            <th className="text-left py-1">State</th>
                            <th className="text-right py-1">Orders</th>
                            <th className="text-right py-1">Gross</th>
                            <th className="text-right py-1">Returns</th>
                            <th className="text-right py-1">Net</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.state_breakdown.map((r, i) => (
                            <tr key={`${r.state}-${i}`} className="border-t border-gray-100">
                              <td className="py-1 text-gray-700">{r.state}</td>
                              <td className="py-1 text-right text-gray-600">{r.orders.toLocaleString()}</td>
                              <td className="py-1 text-right text-gray-700">{fmt(r.gross_revenue)}</td>
                              <td className="py-1 text-right text-red-600">{fmt(r.returns)}</td>
                              <td className="py-1 text-right font-semibold text-green-700">{fmt(r.net_revenue)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
                {(result.company_breakdown && result.company_breakdown.length > 0) && (
                  <div className="mt-3 bg-white rounded-lg border border-green-100 p-3">
                    <p className="text-xs font-semibold text-gray-700 mb-2">Company-wise sales (seller GSTIN)</p>
                    <div className="overflow-x-auto">
                      <table className="w-full text-[11px]">
                        <thead>
                          <tr className="text-gray-500">
                            <th className="text-left py-1">Company</th>
                            <th className="text-left py-1">Seller GSTIN</th>
                            <th className="text-right py-1">Orders</th>
                            <th className="text-right py-1">Gross</th>
                            <th className="text-right py-1">Returns</th>
                            <th className="text-right py-1">Net</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.company_breakdown.map((r, i) => (
                            <tr key={`${r.seller_gstin}-${i}`} className="border-t border-gray-100">
                              <td className="py-1 text-gray-700">{r.company}</td>
                              <td className="py-1 text-gray-500">{r.seller_gstin || 'UNKNOWN'}</td>
                              <td className="py-1 text-right text-gray-600">{r.orders.toLocaleString()}</td>
                              <td className="py-1 text-right text-gray-700">{fmt(r.gross_revenue)}</td>
                              <td className="py-1 text-right text-red-600">{fmt(r.returns)}</td>
                              <td className="py-1 text-right font-semibold text-green-700">{fmt(r.net_revenue)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
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
        <select value={filterCompany} onChange={e => setFilterCompany(e.target.value)}
          className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none">
          <option value="">All Companies / GSTIN</option>
          {companyOptions.map(c => <option key={c}>{c}</option>)}
        </select>
        {(filterPlatform || filterPeriod || filterCompany) && (
          <button onClick={() => { setFilterPlatform(''); setFilterPeriod(''); setFilterCompany('') }}
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
                  <th className="px-3 py-2.5 text-left">Company</th>
                  <th className="px-3 py-2.5 text-left">GSTIN</th>
                  <th className="px-3 py-2.5 text-left">State</th>
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
                    <td className="px-3 py-2 text-gray-700">{u.company_name || '—'}</td>
                    <td className="px-3 py-2 text-gray-500">{u.seller_gstin || '—'}</td>
                    <td className="px-3 py-2 text-gray-500">{u.company_state || '—'}</td>
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
                    <td className="px-3 py-2.5" colSpan={6}>Total ({uploads.length} records)</td>
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
  const toggle = (id: number) => setCollapsed(s => {
    const n = new Set(s)
    if (n.has(id)) n.delete(id)
    else n.add(id)
    return n
  })

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
function TrialBalanceTab({ searchJump }: { searchJump?: { n: number; q: string } | null }) {
  const now = new Date()
  const firstOfYear = `${now.getFullYear()}-04-01`  // Indian FY starts April
  const todayStr = now.toISOString().slice(0, 10)
  const [startDate, setStartDate] = useState(firstOfYear)
  const [endDate,   setEndDate]   = useState(todayStr)
  const [search,    setSearch]    = useState('')
  const lastSearchJumpN = useRef(0)
  useEffect(() => {
    if (searchJump && searchJump.n > lastSearchJumpN.current) {
      lastSearchJumpN.current = searchJump.n
      setSearch(searchJump.q)
    }
  }, [searchJump])

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

// ── Tally / Accountant P&L Section ──────────────────────────────
const BLANK_TALLY = {
  fy: '', opening_stock: 0, purchases: 0, direct_expenses: 0,
  indirect_expenses: 0, sales: 0, closing_stock: 0, indirect_incomes: 0, notes: '',
}

function TallyPLSection({
  tallyRows, refetch, appPL,
}: {
  tallyRows: TallyPL[]
  refetch: () => void
  appPL: PLStatement | null
}) {
  const qc = useQueryClient()
  const [form, setForm]     = useState<typeof BLANK_TALLY>({ ...BLANK_TALLY })
  const [editing, setEditing] = useState(false)
  const [saving,  setSaving]  = useState(false)
  const [err,     setErr]     = useState('')

  // Determine which Tally row to show in comparison (default: latest FY)
  const [selectedFY, setSelectedFY] = useState<string>('')
  const tally = tallyRows.find(r => r.fy === selectedFY) ?? tallyRows[0] ?? null

  function openForm(row?: TallyPL) {
    if (row) {
      setForm({
        fy: row.fy, opening_stock: row.opening_stock, purchases: row.purchases,
        direct_expenses: row.direct_expenses, indirect_expenses: row.indirect_expenses,
        sales: row.sales, closing_stock: row.closing_stock,
        indirect_incomes: row.indirect_incomes, notes: row.notes,
      })
    } else {
      setForm({ ...BLANK_TALLY })
    }
    setErr(''); setEditing(true)
  }

  async function handleSave() {
    if (!form.fy) { setErr('Financial Year is required (e.g. 2025-26)'); return }
    setSaving(true); setErr('')
    try {
      await api.post('/finance/tally-pl', form)
      await qc.invalidateQueries({ queryKey: ['finance-tally-pl'] })
      refetch()
      setSelectedFY(form.fy)
      setEditing(false)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setErr(msg || 'Save failed')
    } finally { setSaving(false) }
  }

  async function handleDelete(fy: string) {
    if (!window.confirm(`Delete Tally figures for ${fy}?`)) return
    await api.delete(`/finance/tally-pl/${encodeURIComponent(fy)}`)
    await qc.invalidateQueries({ queryKey: ['finance-tally-pl'] })
    refetch()
    if (selectedFY === fy) setSelectedFY('')
  }

  const f = (v: number) => `₹${Math.abs(v).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`

  type CompRow = { label: string; tally: number | null; app: number | null; indent?: boolean; bold?: boolean; colored?: boolean }
  const compRows: CompRow[] = tally ? [
    { label: 'Sales',              tally: tally.sales,             app: appPL?.gross_revenue ?? null },
    { label: 'Closing Stock',      tally: tally.closing_stock,     app: null,                        indent: true },
    { label: 'Indirect Incomes',   tally: tally.indirect_incomes,  app: null,                        indent: true },
    { label: 'Opening Stock',      tally: -tally.opening_stock,    app: null,                        indent: true },
    { label: 'Purchases',          tally: -tally.purchases,        app: -(appPL?.cogs ?? 0) || null, indent: true },
    { label: 'Direct Expenses',    tally: -tally.direct_expenses,  app: null,                        indent: true },
    { label: 'Gross Profit',       tally: tally.gross_profit,      app: appPL?.gross_profit ?? null, bold: true, colored: true },
    { label: 'Indirect Expenses',  tally: -tally.indirect_expenses,app: -(appPL?.total_expenses ?? 0) || null, indent: true },
    { label: 'Net Profit',         tally: tally.net_profit,        app: appPL?.net_profit ?? null,   bold: true, colored: true },
  ] : []

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-gray-700">Accountant P&amp;L (Tally)</h3>
          <p className="text-xs text-gray-400 mt-0.5">Official figures from your CA / Tally export</p>
        </div>
        <div className="flex items-center gap-2">
          {tallyRows.length > 0 && (
            <select
              value={selectedFY || tally?.fy || ''}
              onChange={e => setSelectedFY(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none"
            >
              {tallyRows.map(r => <option key={r.fy} value={r.fy}>{r.fy}</option>)}
            </select>
          )}
          <button
            onClick={() => openForm(tally ?? undefined)}
            className="text-xs px-3 py-1.5 bg-[#002B5B] text-white rounded-lg hover:bg-[#003f80] transition-colors font-medium"
          >
            {tally ? 'Edit' : '+ Add Tally Figures'}
          </button>
          {tally && (
            <button onClick={() => handleDelete(tally.fy)} className="text-xs px-2 py-1.5 text-red-500 hover:text-red-700 transition-colors">✕</button>
          )}
        </div>
      </div>

      {/* Comparison table */}
      {tally ? (
        <div>
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-100">
                <th className="px-5 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">Particulars</th>
                <th className="px-5 py-2 text-right text-xs font-semibold text-gray-500 uppercase tracking-wide">Tally ({tally.fy})</th>
                <th className="px-5 py-2 text-right text-xs font-semibold text-blue-500 uppercase tracking-wide">App (computed)</th>
              </tr>
            </thead>
            <tbody>
              {compRows.map((row, i) => {
                const tv = row.tally
                const av = row.app
                const tallyColor = row.colored ? (tv !== null && tv >= 0 ? 'text-green-700 font-bold' : 'text-red-600 font-bold') : row.bold ? 'font-semibold text-gray-800' : 'text-gray-600'
                const appColor   = row.colored ? (av !== null && av >= 0 ? 'text-green-700 font-bold' : 'text-red-600 font-bold') : row.bold ? 'font-semibold text-gray-800' : 'text-gray-400'
                // highlight big difference
                const diff = (tv !== null && av !== null) ? Math.abs(tv - av) / (Math.abs(tv) || 1) : 0
                const rowBg = (row.bold && diff > 0.15) ? 'bg-red-50' : row.bold ? 'bg-gray-50' : ''
                return (
                  <tr key={i} className={`border-t border-gray-50 ${rowBg}`}>
                    <td className={`px-5 py-2.5 text-gray-700 ${row.indent ? 'pl-8 text-xs' : ''} ${row.bold ? 'font-semibold' : ''}`}>{row.label}</td>
                    <td className={`px-5 py-2.5 text-right ${tallyColor} ${row.indent ? 'text-xs' : ''}`}>
                      {tv !== null ? (tv < 0 ? `(${f(tv)})` : f(tv)) : '—'}
                    </td>
                    <td className={`px-5 py-2.5 text-right ${appColor} ${row.indent ? 'text-xs' : ''}`}>
                      {av !== null ? (av < 0 ? `(${f(av)})` : f(av)) : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          {tally.notes && (
            <div className="px-5 py-2 text-xs text-gray-400 border-t border-gray-50">Notes: {tally.notes}</div>
          )}
          <div className="px-5 py-2 text-xs text-gray-400 border-t border-gray-50">
            Last updated: {tally.updated_at?.split('T')[0] ?? ''}
          </div>
        </div>
      ) : (
        <div className="p-8 text-center text-sm text-gray-400">
          No Tally figures saved yet. Click &ldquo;+ Add Tally Figures&rdquo; to enter your accountant&rsquo;s P&amp;L.
        </div>
      )}

      {/* Entry Form Modal */}
      {editing && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-lg max-h-[90vh] overflow-y-auto">
            <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-800">Tally P&amp;L Figures</h3>
              <button onClick={() => setEditing(false)} className="text-gray-400 hover:text-gray-600 text-lg">✕</button>
            </div>
            <div className="px-6 py-4 space-y-4">
              <div>
                <label className="text-xs text-gray-500 block mb-1">Financial Year *</label>
                <input value={form.fy} onChange={e => setForm(f => ({ ...f, fy: e.target.value }))}
                  placeholder="e.g. 2025-26"
                  className="w-full text-sm border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                {([
                  ['Opening Stock',    'opening_stock'],
                  ['Purchases',        'purchases'],
                  ['Direct Expenses',  'direct_expenses'],
                  ['Indirect Expenses','indirect_expenses'],
                  ['Sales',            'sales'],
                  ['Closing Stock',    'closing_stock'],
                  ['Indirect Incomes', 'indirect_incomes'],
                ] as [string, keyof typeof BLANK_TALLY][]).map(([label, key]) => (
                  <div key={key}>
                    <label className="text-xs text-gray-500 block mb-1">{label}</label>
                    <input
                      type="number" step="0.01" min="0"
                      value={(form[key] as number) || ''}
                      onChange={e => setForm(f => ({ ...f, [key]: parseFloat(e.target.value) || 0 }))}
                      className="w-full text-sm border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-300"
                    />
                  </div>
                ))}
              </div>
              <div>
                <label className="text-xs text-gray-500 block mb-1">Notes (optional)</label>
                <input value={form.notes} onChange={e => setForm(f => ({ ...f, notes: e.target.value }))}
                  placeholder="e.g. Audited figures"
                  className="w-full text-sm border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-300" />
              </div>
              {err && <p className="text-xs text-red-600">{err}</p>}
            </div>
            <div className="px-6 py-4 border-t border-gray-100 flex gap-2 justify-end">
              <button onClick={() => setEditing(false)}
                className="px-4 py-2 text-sm text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50">Cancel</button>
              <button onClick={handleSave} disabled={saving}
                className="px-4 py-2 text-sm bg-[#002B5B] text-white rounded-lg hover:bg-[#003f80] disabled:opacity-50 font-medium">
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
