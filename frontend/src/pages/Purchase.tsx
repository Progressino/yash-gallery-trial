import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type Tab = 'dashboard' | 'suppliers' | 'processors' | 'pr' | 'po' | 'jwo' | 'grn' | 'min' | 'gate-pass'
type PRSubTab = 'list' | 'new' | 'from-mrp'

interface Stats { open_prs: number; open_pos: number; open_jwos: number; pending_grns: number; total_suppliers: number; total_processors: number }
interface Supplier { id: number; supplier_code: string; supplier_name: string; supplier_type: string; contact_person: string; email: string; phone: string; payment_terms: string; active: number }
interface Processor { id: number; processor_code: string; processor_name: string; processor_type: string; contact_person: string; phone: string }
interface PRLine { id: number; material_code: string; material_name: string; material_type: string; required_qty: number; po_qty: number; unit: string; required_by_date: string }
interface PR { id: number; pr_number: string; pr_date: string; requested_by: string; department: string; priority: string; status: string; so_reference: string; pr_type: string; source: string; required_by_date: string; lines: PRLine[] }
interface PO { id: number; po_number: string; po_date: string; supplier_name: string; status: string; total: number; delivery_date: string; pr_reference: string; so_reference: string; payment_terms?: string; lines: POLine[] }
interface POLine { id: number; material_code: string; material_name: string; material_type: string; po_qty: number; unit: string; rate: number; gst_pct: number; amount: number }
interface JWO { id: number; jwo_number: string; jwo_date: string; processor_name: string; processor_id?: number; status: string; total: number; expected_return_date: string; pr_reference?: string; so_reference?: string; issued_by?: string; remarks?: string; lines: JWOLine[] }
interface JWOLine { id: number; input_material: string; input_qty?: number; output_material: string; output_qty: number; process_type: string; rate: number; amount: number; unit?: string }
interface GRN { id: number; grn_number: string; grn_date: string; grn_type: string; party_name: string; status: string; total_value: number; reference_number?: string; challan_no?: string; invoice_no?: string; lines: GRNLine[] }
interface GRNLine { id: number; material_code: string; material_name?: string; received_qty: number; accepted_qty: number; rejected_qty: number; qc_status: string; rate: number; amount: number; unit?: string }
interface MRPLineItem { material_code: string; material_name: string; material_type: string; required_qty: number; net_req: number; unit: string; inputs?: { material_code: string; material_name: string; quantity: number; unit: string }[] }
interface MRPLinesResult { so_number: string; purchase_items: MRPLineItem[]; sfg_items: MRPLineItem[]; error?: string; warning?: string }
interface POFromPRLine { supplier_id?: number; supplier_name: string; processor_id?: number; processor_name?: string; qty: number; rate: number; gst_pct: number; order_type?: 'PO' | 'JWO' }

const SUP_TYPES = ['Fabric Supplier', 'Accessories Supplier', 'Job Work', 'Others']
const PROC_TYPES = ['Printing Unit', 'Dyeing Unit', 'Embroidery', 'Others']
const DEPTS = ['Production', 'Stores', 'Others']
const MAT_TYPES = ['RM', 'ACC', 'PKG', 'SFG', 'FG', 'GF']
const PROC_TYPES2 = ['Printing', 'Dyeing', 'Embroidery', 'Other']
const GRN_TYPES = ['PO Receipt', 'JWO Receipt']
const PAYMENT_TERMS = ['Immediate', 'Net 15', 'Net 30', 'Net 45', 'Net 60']
const GST_RATES = [0, 5, 12, 18, 28]

const statusColor = (s: string) => {
  if (['Approved', 'Received', 'Verified', 'Posted', 'Closed'].includes(s)) return 'bg-green-100 text-green-700'
  if (['Draft', 'Pending Approval'].includes(s)) return 'bg-yellow-100 text-yellow-700'
  if (s === 'Partial PO') return 'bg-orange-100 text-orange-700'
  if (['PO Created', 'Sent to Supplier', 'In Process', 'Confirmed', 'Issued to Processor'].includes(s)) return 'bg-blue-100 text-blue-700'
  if (['Rejected', 'Cancelled'].includes(s)) return 'bg-red-100 text-red-700'
  return 'bg-gray-100 text-gray-600'
}

const fmt = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN')
const today = () => new Date().toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })

// ── Print helpers ─────────────────────────────────────────────────────────────
const printDocument = (html: string, title: string) => {
  const win = window.open('', '_blank', 'width=900,height=700')
  if (!win) { alert('Please allow popups to print.'); return }
  win.document.write(`<!DOCTYPE html><html><head><title>${title}</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:'Segoe UI',sans-serif;font-size:12px;color:#1a1a1a;padding:24px}
    .header{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:2px solid #002B5B;padding-bottom:12px;margin-bottom:16px}
    .company{font-size:20px;font-weight:700;color:#002B5B}
    .doc-title{font-size:16px;font-weight:600;color:#002B5B;text-align:right}
    .doc-num{font-size:22px;font-weight:800;color:#002B5B;text-align:right}
    .info-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
    .info-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px}
    .info-label{font-size:10px;text-transform:uppercase;color:#64748b;font-weight:600;margin-bottom:4px}
    .info-value{font-size:13px;font-weight:600;color:#1e293b}
    table{width:100%;border-collapse:collapse;margin-bottom:16px}
    th{background:#002B5B;color:white;padding:7px 10px;text-align:left;font-size:11px;font-weight:600}
    th.right,td.right{text-align:right}
    td{padding:6px 10px;border-bottom:1px solid #e2e8f0;font-size:12px}
    tr:nth-child(even) td{background:#f8fafc}
    .totals{margin-left:auto;width:260px;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden}
    .total-row{display:flex;justify-content:space-between;padding:7px 12px;border-bottom:1px solid #e2e8f0;font-size:12px}
    .total-row.grand{background:#002B5B;color:white;font-weight:700;font-size:14px}
    .badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:600;background:#dbeafe;color:#1d4ed8}
    .footer{margin-top:32px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px;border-top:1px solid #e2e8f0;padding-top:16px}
    .sign-box{text-align:center}
    .sign-line{border-top:1px solid #64748b;margin-top:32px;padding-top:6px;font-size:10px;color:#64748b}
    .status-chip{display:inline-block;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:600}
    .qc-pass{background:#dcfce7;color:#166534}
    .qc-fail{background:#fee2e2;color:#991b1b}
    .qc-pending{background:#fef9c3;color:#854d0e}
    @media print{body{padding:12px}button{display:none}}
  </style>
  </head><body>${html}
  <script>window.onload=()=>window.print()<\/script>
  </body></html>`)
  win.document.close()
}

const buildPOPrintHTML = (po: PO) => {
  const subtotal = po.lines.reduce((s, l) => s + l.amount, 0)
  const gstTotal = po.lines.reduce((s, l) => s + (l.amount * (l.gst_pct || 0) / 100), 0)
  const grand = subtotal + gstTotal
  return `
    <div class="header">
      <div><div class="company">🧵 Garment ERP</div><div style="font-size:11px;color:#64748b;margin-top:4px">Purchase Department</div></div>
      <div><div class="doc-title">PURCHASE ORDER</div><div class="doc-num">${po.po_number}</div></div>
    </div>
    <div class="info-grid">
      <div class="info-box">
        <div class="info-label">Supplier</div>
        <div class="info-value">${po.supplier_name}</div>
      </div>
      <div class="info-box">
        <div class="info-label">PO Date</div>
        <div class="info-value">${po.po_date || today()}</div>
        <div class="info-label" style="margin-top:8px">Delivery Date</div>
        <div class="info-value">${po.delivery_date || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">SO Reference</div>
        <div class="info-value">${po.so_reference || '—'}</div>
        <div class="info-label" style="margin-top:8px">PR Reference</div>
        <div class="info-value">${po.pr_reference || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Payment Terms</div>
        <div class="info-value">${po.payment_terms || '—'}</div>
        <div class="info-label" style="margin-top:8px">Status</div>
        <div class="info-value"><span class="badge">${po.status}</span></div>
      </div>
    </div>
    <table>
      <thead><tr>
        <th>#</th><th>Material Code</th><th>Description</th><th>Type</th>
        <th class="right">Qty</th><th>Unit</th><th class="right">Rate (₹)</th>
        <th class="right">GST%</th><th class="right">Amount (₹)</th>
      </tr></thead>
      <tbody>
        ${po.lines.map((l, i) => `<tr>
          <td>${i + 1}</td>
          <td><strong>${l.material_code}</strong></td>
          <td>${l.material_name}</td>
          <td>${l.material_type || '—'}</td>
          <td class="right">${l.po_qty}</td>
          <td>${l.unit || 'PCS'}</td>
          <td class="right">${fmt(l.rate)}</td>
          <td class="right">${l.gst_pct || 0}%</td>
          <td class="right"><strong>${fmt(l.amount)}</strong></td>
        </tr>`).join('')}
      </tbody>
    </table>
    <div style="display:flex;justify-content:flex-end">
      <div class="totals">
        <div class="total-row"><span>Subtotal</span><span>${fmt(subtotal)}</span></div>
        <div class="total-row"><span>GST</span><span>${fmt(gstTotal)}</span></div>
        <div class="total-row grand"><span>Grand Total</span><span>${fmt(grand)}</span></div>
      </div>
    </div>
    <div class="footer">
      <div class="sign-box"><div class="sign-line">Prepared By</div></div>
      <div class="sign-box"><div class="sign-line">Approved By</div></div>
      <div class="sign-box"><div class="sign-line">Supplier Acknowledgement</div></div>
    </div>`
}

const buildJWOPrintHTML = (jwo: JWO) => {
  const total = jwo.lines.reduce((s, l) => s + (l.output_qty * l.rate), 0)
  return `
    <div class="header">
      <div><div class="company">🧵 Garment ERP</div><div style="font-size:11px;color:#64748b;margin-top:4px">Production Department</div></div>
      <div><div class="doc-title">JOB WORK ORDER</div><div class="doc-num">${jwo.jwo_number}</div></div>
    </div>
    <div class="info-grid">
      <div class="info-box">
        <div class="info-label">Processor</div>
        <div class="info-value">${jwo.processor_name}</div>
      </div>
      <div class="info-box">
        <div class="info-label">JWO Date</div>
        <div class="info-value">${jwo.jwo_date || today()}</div>
        <div class="info-label" style="margin-top:8px">Expected Return</div>
        <div class="info-value">${jwo.expected_return_date || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">SO Reference</div>
        <div class="info-value">${jwo.so_reference || '—'}</div>
        <div class="info-label" style="margin-top:8px">PR Reference</div>
        <div class="info-value">${jwo.pr_reference || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Issued By</div>
        <div class="info-value">${jwo.issued_by || '—'}</div>
        <div class="info-label" style="margin-top:8px">Status</div>
        <div class="info-value"><span class="badge">${jwo.status}</span></div>
      </div>
    </div>
    <table>
      <thead><tr>
        <th>#</th><th>Input Material</th><th>Output Material</th><th>Process</th>
        <th class="right">Output Qty</th><th>Unit</th><th class="right">Rate (₹)</th><th class="right">Amount (₹)</th>
      </tr></thead>
      <tbody>
        ${jwo.lines.map((l, i) => `<tr>
          <td>${i + 1}</td>
          <td><strong>${l.input_material}</strong></td>
          <td>${l.output_material}</td>
          <td>${l.process_type}</td>
          <td class="right">${l.output_qty}</td>
          <td>${l.unit || 'PCS'}</td>
          <td class="right">${fmt(l.rate)}</td>
          <td class="right"><strong>${fmt(l.output_qty * l.rate)}</strong></td>
        </tr>`).join('')}
      </tbody>
    </table>
    <div style="display:flex;justify-content:flex-end">
      <div class="totals">
        <div class="total-row grand"><span>Total Job Work Value</span><span>${fmt(total)}</span></div>
      </div>
    </div>
    <div style="background:#fef9c3;border:1px solid #fde047;border-radius:6px;padding:10px;margin-bottom:16px;font-size:11px;color:#713f12">
      <strong>Terms & Conditions:</strong> Material issued on challan. Return within agreed date. Quality must meet specifications. Defective work will not be accepted.
    </div>
    <div class="footer">
      <div class="sign-box"><div class="sign-line">Stores In-charge</div></div>
      <div class="sign-box"><div class="sign-line">Authorized By</div></div>
      <div class="sign-box"><div class="sign-line">Processor Acknowledgement</div></div>
    </div>`
}

const buildGRNPrintHTML = (grn: GRN) => {
  return `
    <div class="header">
      <div><div class="company">🧵 Garment ERP</div><div style="font-size:11px;color:#64748b;margin-top:4px">Stores Department</div></div>
      <div><div class="doc-title">GOODS RECEIPT NOTE</div><div class="doc-num">${grn.grn_number}</div></div>
    </div>
    <div class="info-grid">
      <div class="info-box">
        <div class="info-label">Party Name</div>
        <div class="info-value">${grn.party_name}</div>
        <div class="info-label" style="margin-top:8px">GRN Type</div>
        <div class="info-value">${grn.grn_type}</div>
      </div>
      <div class="info-box">
        <div class="info-label">GRN Date</div>
        <div class="info-value">${grn.grn_date || today()}</div>
        <div class="info-label" style="margin-top:8px">Reference</div>
        <div class="info-value">${grn.reference_number || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Challan No</div>
        <div class="info-value">${grn.challan_no || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Status</div>
        <div class="info-value"><span class="badge">${grn.status}</span></div>
      </div>
    </div>
    <table>
      <thead><tr>
        <th>#</th><th>Material Code</th><th>Description</th>
        <th class="right">Received</th><th class="right">Accepted</th>
        <th class="right">Rejected</th><th>Unit</th><th class="right">Rate</th>
        <th class="right">Amount</th><th>QC Status</th>
      </tr></thead>
      <tbody>
        ${grn.lines.map((l, i) => `<tr>
          <td>${i + 1}</td>
          <td><strong>${l.material_code}</strong></td>
          <td>${l.material_name || '—'}</td>
          <td class="right">${l.received_qty}</td>
          <td class="right" style="color:#166534;font-weight:600">${l.accepted_qty}</td>
          <td class="right" style="color:#991b1b;font-weight:600">${l.rejected_qty}</td>
          <td>${l.unit || 'PCS'}</td>
          <td class="right">${fmt(l.rate)}</td>
          <td class="right"><strong>${fmt(l.amount)}</strong></td>
          <td><span class="status-chip ${l.qc_status === 'Pass' ? 'qc-pass' : l.qc_status === 'Fail' ? 'qc-fail' : 'qc-pending'}">${l.qc_status}</span></td>
        </tr>`).join('')}
      </tbody>
    </table>
    <div style="display:flex;justify-content:flex-end">
      <div class="totals">
        <div class="total-row grand"><span>Total Value</span><span>${fmt(grn.total_value)}</span></div>
      </div>
    </div>
    <div class="footer">
      <div class="sign-box"><div class="sign-line">Received By</div></div>
      <div class="sign-box"><div class="sign-line">QC Checked By</div></div>
      <div class="sign-box"><div class="sign-line">Store Manager</div></div>
    </div>`
}

const buildMINPrintHTML = (min: any) => {
  return `
    <div class="header">
      <div><div class="company">🧵 Garment ERP</div><div style="font-size:11px;color:#64748b;margin-top:4px">Stores Department</div></div>
      <div><div class="doc-title">MATERIAL ISSUE NOTE</div><div class="doc-num">${min.min_number}</div></div>
    </div>
    <div class="info-grid">
      <div class="info-box">
        <div class="info-label">From Location</div><div class="info-value">${min.from_location || 'Grey Warehouse'}</div>
        <div class="info-label" style="margin-top:8px">To Location</div><div class="info-value">${min.to_location || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">To Vendor / Processor</div><div class="info-value">${min.to_vendor || '—'}</div>
        <div class="info-label" style="margin-top:8px">Issue Date</div><div class="info-value">${min.min_date || today()}</div>
      </div>
      <div class="info-box">
        <div class="info-label">JWO Reference</div><div class="info-value">${min.jwo_reference || '—'}</div>
        <div class="info-label" style="margin-top:8px">SO Reference</div><div class="info-value">${min.so_reference || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Issued By</div><div class="info-value">${min.issued_by || '—'}</div>
        <div class="info-label" style="margin-top:8px">Status</div><div class="info-value"><span class="badge">${min.status}</span></div>
      </div>
    </div>
    <table>
      <thead><tr><th>#</th><th>Material Code</th><th>Description</th><th>Type</th>
        <th class="right">Issue Qty</th><th>Unit</th><th class="right">Rate (₹)</th><th class="right">Amount (₹)</th></tr></thead>
      <tbody>${(min.lines || []).map((l: any, i: number) => `<tr>
        <td>${i + 1}</td><td><strong>${l.material_code}</strong></td><td>${l.material_name || '—'}</td>
        <td>${l.material_type || 'GF'}</td><td class="right">${l.issue_qty}</td><td>${l.unit || 'MTR'}</td>
        <td class="right">${fmt(l.rate || 0)}</td><td class="right"><strong>${fmt(l.amount || 0)}</strong></td></tr>`).join('')}</tbody>
    </table>
    <div style="background:#fef9c3;border:1px solid #fde047;border-radius:6px;padding:10px;margin-bottom:16px;font-size:11px;color:#713f12">
      <strong>Note:</strong> Material issued against JWO. Gate pass required for material moving out of factory premises.
    </div>
    <div class="footer">
      <div class="sign-box"><div class="sign-line">Issued By (Stores)</div></div>
      <div class="sign-box"><div class="sign-line">Received By (Processor)</div></div>
      <div class="sign-box"><div class="sign-line">Authorized By</div></div>
    </div>`
}

const buildGatePasePrintHTML = (gp: any) => {
  return `
    <div class="header">
      <div><div class="company">🧵 Garment ERP</div><div style="font-size:11px;color:#64748b;margin-top:4px">Security / Store Department</div></div>
      <div><div class="doc-title">GATE PASS</div><div class="doc-num">${gp.gp_number}</div></div>
    </div>
    <div class="info-grid">
      <div class="info-box">
        <div class="info-label">From</div><div class="info-value">${gp.from_location || 'Factory'}</div>
        <div class="info-label" style="margin-top:8px">To</div><div class="info-value">${gp.to_location || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Party / Processor</div><div class="info-value">${gp.party_name || '—'}</div>
        <div class="info-label" style="margin-top:8px">Purpose</div><div class="info-value">${gp.purpose || 'Job Work'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">Vehicle No</div><div class="info-value">${gp.vehicle_no || '—'}</div>
        <div class="info-label" style="margin-top:8px">Driver</div><div class="info-value">${gp.driver_name || '—'}</div>
      </div>
      <div class="info-box">
        <div class="info-label">MIN Reference</div><div class="info-value">${gp.min_reference || '—'}</div>
        <div class="info-label" style="margin-top:8px">JWO Reference</div><div class="info-value">${gp.jwo_reference || '—'}</div>
      </div>
    </div>
    <table>
      <thead><tr><th>#</th><th>Material Code</th><th>Description</th>
        <th class="right">Qty</th><th>Unit</th><th>Remarks</th></tr></thead>
      <tbody>${(gp.lines || []).map((l: any, i: number) => `<tr>
        <td>${i + 1}</td><td><strong>${l.material_code}</strong></td><td>${l.material_name || '—'}</td>
        <td class="right">${l.qty}</td><td>${l.unit || 'MTR'}</td><td>${l.remarks || '—'}</td></tr>`).join('')}</tbody>
    </table>
    <div style="background:#fef9c3;border:1px solid #fde047;border-radius:6px;padding:10px;margin-bottom:16px;font-size:11px;color:#713f12">
      <strong>Instructions:</strong> Material must be returned within stipulated time. This gate pass is valid for one trip only. Security must verify quantity before allowing material out.
    </div>
    <div class="footer">
      <div class="sign-box"><div class="sign-line">Issued By (Store)</div></div>
      <div class="sign-box"><div class="sign-line">Security Check</div></div>
      <div class="sign-box"><div class="sign-line">Received By (Party)</div></div>
    </div>`
}
void buildGatePasePrintHTML

export default function Purchase() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('dashboard')
  const [expanded, setExpanded] = useState<number | null>(null)
  const [filterStatus, setFilterStatus] = useState('')
  const [prSubTab, setPRSubTab] = useState<PRSubTab>('list')

  // Supplier form
  const [showSupForm, setShowSupForm] = useState(false)
  const [supForm, setSupForm] = useState({ supplier_name: '', supplier_type: 'Others', contact_person: '', email: '', phone: '', address: '', gst_number: '', payment_terms: 'Net 30' })

  // Processor form
  const [showProcForm, setShowProcForm] = useState(false)
  const [procForm, setProcForm] = useState({ processor_name: '', processor_type: 'Others', contact_person: '', email: '', phone: '', address: '' })

  // PR manual form
  const [prForm, setPRForm] = useState({ requested_by: '', department: 'Production', priority: 'Normal', so_reference: '', required_by_date: '', notes: '' })
  const [prLines, setPRLines] = useState<{ material_code: string; material_name: string; material_type: string; required_qty: number; unit: string }[]>([])

  // From MRP
  const [mrpSO, setMrpSO] = useState('')
  const [mrpLinesData, setMrpLinesData] = useState<MRPLinesResult | null>(null)
  const [mrpLoadingLines, setMrpLoadingLines] = useState(false)
  const [selectedPurchaseItems, setSelectedPurchaseItems] = useState<Set<string>>(new Set())
  const [selectedSFGItems, setSelectedSFGItems] = useState<Set<string>>(new Set())
  const [mrpReqDate, setMrpReqDate] = useState('')
  const [generatingPRs, setGeneratingPRs] = useState(false)
  const [generatedPRs, setGeneratedPRs] = useState<string[]>([])

  // Create PO from PR
  const [showCreatePO, setShowCreatePO] = useState<number | null>(null)
  const [poFromPRLines, setPoFromPRLines] = useState<Record<string, POFromPRLine>>({})
  const [poFromPRMeta, setPoFromPRMeta] = useState({ delivery_date: '', payment_terms: 'Immediate' })
  const [createdPOs, setCreatedPOs] = useState<string[]>([])

  // Edit Draft PO
  const [editingPO, setEditingPO] = useState<number | null>(null)
  const [editPOForm, setEditPOForm] = useState({ supplier_id: undefined as number | undefined, supplier_name: '', delivery_date: '', payment_terms: '', so_reference: '', remarks: '' })
  const [editPOLines, setEditPOLines] = useState<{ material_code: string; material_name: string; material_type: string; po_qty: number; unit: string; rate: number; gst_pct: number }[]>([])

  // ── NEW: Edit Draft JWO ────────────────────────────────────────────────────
  const [editingJWO, setEditingJWO] = useState<number | null>(null)
  const [editJWOForm, setEditJWOForm] = useState({ processor_id: undefined as number | undefined, processor_name: '', expected_return_date: '', pr_reference: '', so_reference: '', issued_by: '', remarks: '' })
  const [editJWOLines, setEditJWOLines] = useState<{ input_material: string; input_qty: number; output_material: string; output_qty: number; process_type: string; rate: number }[]>([])

  // ── NEW: GRN Auto-fill state ───────────────────────────────────────────────
  const [grnAutoRef, setGrnAutoRef] = useState('')
  const [grnAutoLoading, setGrnAutoLoading] = useState(false)
  const [grnAutoError, setGrnAutoError] = useState('')

  // PO / JWO / GRN forms
  const [showPOForm, setShowPOForm] = useState(false)
  const [poForm, setPOForm] = useState({ supplier_id: undefined as number | undefined, supplier_name: '', delivery_date: '', payment_terms: '', delivery_location: '', pr_reference: '', so_reference: '', remarks: '' })
  const [poLines, setPOLines] = useState<{ material_code: string; material_name: string; material_type: string; po_qty: number; unit: string; rate: number; gst_pct: number }[]>([])
  const [showJWOForm, setShowJWOForm] = useState(false)
  const [jwoForm, setJWOForm] = useState({ processor_id: undefined as number | undefined, processor_name: '', expected_return_date: '', pr_reference: '', so_reference: '', issued_by: '', remarks: '' })
  const [jwoLines, setJWOLines] = useState<{ input_material: string; input_qty: number; output_material: string; output_qty: number; process_type: string; rate: number }[]>([])
  const [showGRNForm, setShowGRNForm] = useState(false)
  const [grnForm, setGRNForm] = useState({ grn_type: 'PO Receipt', reference_number: '', party_name: '', challan_no: '', invoice_no: '', vehicle_no: '', transporter: '', warehouse: '', remarks: '' })
  const [grnLines, setGRNLines] = useState<{ material_code: string; material_name: string; received_qty: number; accepted_qty: number; rejected_qty: number; unit: string; rate: number; qc_status: string }[]>([])
  // MIN form
  const [showMINForm, setShowMINForm] = useState(false)
  const [minForm, setMINForm] = useState({ jwo_reference: '', so_reference: '', from_location: 'Grey Warehouse', to_location: '', to_vendor: '', issued_by: '', remarks: '' })
  const [minLines, setMINLines] = useState<{ material_code: string; material_name: string; material_type: string; issue_qty: number; unit: string; rate: number }[]>([])
  // Gate Pass form
  const [showGPForm, setShowGPForm] = useState(false)
  const [gpForm, setGPForm] = useState({ min_reference: '', jwo_reference: '', from_location: 'Factory', to_location: '', party_name: '', vehicle_no: '', driver_name: '', material_desc: '', purpose: 'Job Work', remarks: '' })
  const [gpLines, setGPLines] = useState<{ material_code: string; material_name: string; qty: number; unit: string }[]>([])

  const { data: stats } = useQuery<Stats>({ queryKey: ['purchase-stats'], queryFn: () => api.get('/purchase/stats').then(r => r.data) })
  const { data: suppliers = [] } = useQuery<Supplier[]>({ queryKey: ['suppliers'], queryFn: () => api.get('/purchase/suppliers').then(r => r.data), enabled: tab === 'suppliers' || tab === 'po' || tab === 'dashboard' || tab === 'pr' })
  const { data: processors = [] } = useQuery<Processor[]>({ queryKey: ['processors'], queryFn: () => api.get('/purchase/processors').then(r => r.data), enabled: tab === 'processors' || tab === 'jwo' || tab === 'pr' })
  const { data: prs = [] } = useQuery<PR[]>({ queryKey: ['prs', filterStatus], queryFn: () => api.get('/purchase/pr' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data), enabled: tab === 'pr' })
  const { data: pos = [] } = useQuery<PO[]>({ queryKey: ['pos', filterStatus], queryFn: () => api.get('/purchase/po' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data), enabled: tab === 'po' })
  const { data: jwos = [] } = useQuery<JWO[]>({ queryKey: ['jwos', filterStatus], queryFn: () => api.get('/purchase/jwo' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data), enabled: tab === 'jwo' })
  const { data: grns = [] } = useQuery<GRN[]>({ queryKey: ['grns', filterStatus], queryFn: () => api.get('/purchase/grn' + (filterStatus ? `?status=${filterStatus}` : '')).then(r => r.data), enabled: tab === 'grn' })
  const { data: mins = [] } = useQuery<any[]>({ queryKey: ['mins'], queryFn: () => api.get('/purchase/min').then(r => r.data), enabled: tab === 'min' })
  const { data: gatePasses = [] } = useQuery<any[]>({ queryKey: ['gate-passes'], queryFn: () => api.get('/purchase/gate-pass').then(r => r.data), enabled: tab === 'gate-pass' })

  const invalidate = () => qc.invalidateQueries({ queryKey: ['purchase-stats'] })
  const createSupMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/suppliers', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['suppliers'] }); invalidate(); setShowSupForm(false) } })
  const createProcMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/processors', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['processors'] }); setShowProcForm(false) } })
  const createPRMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/pr', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['prs'] }); invalidate(); setPRSubTab('list'); setPRLines([]) } })
  const approvePRMut = useMutation({ mutationFn: ({ id, approver }: { id: number; approver: string }) => api.post(`/purchase/pr/${id}/approve`, { approver }), onSuccess: () => qc.invalidateQueries({ queryKey: ['prs'] }) })
  const rejectPRMut = useMutation({ mutationFn: (id: number) => api.post(`/purchase/pr/${id}/reject`, { remarks: '' }), onSuccess: () => qc.invalidateQueries({ queryKey: ['prs'] }) })
  const createPOMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/po', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['pos'] }); invalidate(); setShowPOForm(false); setPOLines([]) } })
  const updatePOStatusMut = useMutation({ mutationFn: ({ id, status }: { id: number; status: string }) => api.patch(`/purchase/po/${id}/status`, { status }), onSuccess: () => qc.invalidateQueries({ queryKey: ['pos'] }) })
  const createJWOMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/jwo', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['jwos'] }); invalidate(); setShowJWOForm(false); setJWOLines([]) } })
  const updateJWOStatusMut = useMutation({ mutationFn: ({ id, status }: { id: number; status: string }) => api.patch(`/purchase/jwo/${id}/status`, { status }), onSuccess: () => qc.invalidateQueries({ queryKey: ['jwos'] }) })

  // ── NEW: Update JWO mutation ───────────────────────────────────────────────
  const updateJWOMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/purchase/jwo/${id}`, data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['jwos'] }); setEditingJWO(null) }
  })

  const createGRNMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/grn', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['grns'] }); invalidate(); setShowGRNForm(false); setGRNLines([]) } })
  const verifyGRNMut = useMutation({ mutationFn: ({ id, status }: { id: number; status: string }) => api.patch(`/purchase/grn/${id}/verify`, { status }), onSuccess: () => qc.invalidateQueries({ queryKey: ['grns'] }) })
  const updatePOMut = useMutation({ mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/purchase/po/${id}`, data), onSuccess: () => { qc.invalidateQueries({ queryKey: ['pos'] }); setEditingPO(null) } })
  const createMINMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/min', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['mins'] }); setShowMINForm(false); setMINLines([]) } })
  const createGPMut = useMutation({ mutationFn: (b: object) => api.post('/purchase/gate-pass', b), onSuccess: () => { qc.invalidateQueries({ queryKey: ['gate-passes'] }); setShowGPForm(false); setGPLines([]) } })
  // Gate-pass UI is wired through dedicated buttons added later — keep state hooks live.
  void showGPForm; void gpForm; void setGPForm; void gpLines; void gatePasses; void createGPMut

  const TABS: [Tab, string][] = [
    ['dashboard', '📊 Dashboard'],
    ['suppliers', '🏢 Suppliers'],
    ['processors', '🏭 Processors'],
    ['pr', '📝 Requisitions'],
    ['po', '🛒 Purchase Orders'],
    ['jwo', '⚙️ Job Work Orders'],
    ['grn', '📦 GRN'],
    ['min', '📋 Issue Notes'],
    ['gate-pass', '🚪 Gate Pass'],
  ]

  // ── From MRP helpers ──────────────────────────────────────────────────────
  const loadMRPLines = async () => {
    if (!mrpSO.trim()) return
    setMrpLoadingLines(true); setMrpLinesData(null); setGeneratedPRs([])
    try {
      const res = await api.get(`/production/mrp/lines-for-so?so_number=${encodeURIComponent(mrpSO.trim())}`)
      const d: MRPLinesResult = res.data
      setMrpLinesData(d)
      setSelectedPurchaseItems(new Set(d.purchase_items.map(i => i.material_code)))
      setSelectedSFGItems(new Set(d.sfg_items.map(i => i.material_code)))
    } catch {
      setMrpLinesData({ so_number: mrpSO, purchase_items: [], sfg_items: [], error: 'Failed to load MRP lines.' })
    }
    setMrpLoadingLines(false)
  }

  const generatePRsFromMRP = async () => {
    if (!mrpLinesData) return
    setGeneratingPRs(true); setGeneratedPRs([])
    const created: string[] = []
    try {
      const toLine = (i: MRPLineItem) => ({ material_code: i.material_code, material_name: i.material_name, material_type: i.material_type, required_qty: i.net_req || i.required_qty, unit: i.unit })
      const purchaseLines = mrpLinesData.purchase_items.filter(i => selectedPurchaseItems.has(i.material_code)).map(toLine)
      const jwLines = mrpLinesData.sfg_items.filter(i => selectedSFGItems.has(i.material_code)).map(toLine)
      if (purchaseLines.length > 0) {
        const r = await api.post('/purchase/pr', { pr_type: 'Purchase', source: 'MRP', so_reference: mrpSO, required_by_date: mrpReqDate, department: 'Production', priority: 'Normal', lines: purchaseLines })
        created.push(r.data.pr_number)
      }
      if (jwLines.length > 0) {
        const r = await api.post('/purchase/pr', { pr_type: 'Job Work', source: 'MRP', so_reference: mrpSO, required_by_date: mrpReqDate, department: 'Production', priority: 'Normal', lines: jwLines })
        created.push(r.data.pr_number)
      }
      setGeneratedPRs(created)
      qc.invalidateQueries({ queryKey: ['prs'] }); invalidate()
    } catch { /* ignore */ }
    setGeneratingPRs(false)
  }

  // ── Create PO from PR helpers ──────────────────────────────────────────────
  const openCreatePO = (pr: PR) => {
    const init: Record<string, POFromPRLine> = {}
    pr.lines.forEach(l => {
      const pending = l.required_qty - (l.po_qty || 0)
      if (pending > 0) init[`${pr.id}-${l.material_code}`] = {
        supplier_name: '', processor_name: '',
        qty: pending, rate: 0, gst_pct: 12,
        order_type: l.material_type === 'SFG' ? 'JWO' : 'PO',
      }
    })
    setPoFromPRLines(init)
    setPoFromPRMeta({ delivery_date: '', payment_terms: 'Immediate' })
    setCreatedPOs([])
    setShowCreatePO(pr.id)
  }

  const updatePOLine = (prId: number, matCode: string, field: keyof POFromPRLine, value: string | number) => {
    const key = `${prId}-${matCode}`
    setPoFromPRLines(prev => ({ ...prev, [key]: { ...prev[key], [field]: value } }))
  }

  const submitPOFromPR = async (pr: PR) => {
    const pendingLines = pr.lines.filter(l => (l.required_qty - (l.po_qty || 0)) > 0)
    const poLinesList: object[] = []
    const jwoByProcessor: Record<string, { processor_id?: number; processor_name: string; lines: object[] }> = {}

    for (const l of pendingLines) {
      const key = `${pr.id}-${l.material_code}`
      const ld = poFromPRLines[key] || { order_type: 'PO', supplier_name: '', qty: 0, rate: 0, gst_pct: 12 }
      const orderType = ld.order_type || (l.material_type === 'SFG' ? 'JWO' : 'PO')

      if (orderType === 'JWO') {
        if (!ld.processor_id && !ld.processor_name) continue
        const procKey = `${ld.processor_id}-${ld.processor_name}`
        if (!jwoByProcessor[procKey]) jwoByProcessor[procKey] = { processor_id: ld.processor_id, processor_name: ld.processor_name || '', lines: [] }
        jwoByProcessor[procKey].lines.push({ input_material: l.material_code, input_qty: ld.qty, input_unit: l.unit, output_material: l.material_code, output_qty: ld.qty, output_unit: l.unit, process_type: 'Processing', rate: ld.rate, amount: ld.qty * ld.rate })
      } else {
        if (!ld.supplier_id && !ld.supplier_name) continue
        const sup = suppliers.find(s => s.id === ld.supplier_id)
        poLinesList.push({ material_code: l.material_code, material_name: l.material_name, material_type: l.material_type, unit: l.unit, qty: ld.qty, rate: ld.rate, gst_pct: ld.gst_pct, supplier_id: ld.supplier_id, supplier_name: ld.supplier_name || sup?.supplier_name || '' })
      }
    }
    if (poLinesList.length === 0 && Object.keys(jwoByProcessor).length === 0) { alert('Please assign at least one supplier or processor.'); return }
    try {
      const created: string[] = []
      if (poLinesList.length > 0) {
        const r = await api.post('/purchase/po/from-pr', { pr_id: pr.id, ...poFromPRMeta, lines: poLinesList })
        created.push(...(r.data.po_numbers || []))
      }
      for (const { processor_id, processor_name, lines } of Object.values(jwoByProcessor)) {
        const r = await api.post('/purchase/jwo', { processor_id, processor_name, pr_reference: pr.pr_number, so_reference: pr.so_reference, expected_return_date: poFromPRMeta.delivery_date, lines })
        created.push(r.data.jwo_number)
        const jwoUpdates = (lines as { output_material: string; output_qty: number }[]).map(ln => ({ material_code: ln.output_material, qty: ln.output_qty }))
        await api.post(`/purchase/pr/${pr.id}/mark-ordered`, { updates: jwoUpdates })
      }
      setCreatedPOs(created)
      qc.invalidateQueries({ queryKey: ['prs'] }); qc.invalidateQueries({ queryKey: ['pos'] }); qc.invalidateQueries({ queryKey: ['jwos'] }); invalidate()
    } catch { alert('Failed to create orders.') }
  }

  // ── NEW: GRN Auto-fill from PO/JWO ────────────────────────────────────────
  const autoFillGRNFromRef = async () => {
    const ref = grnAutoRef.trim()
    if (!ref) return
    setGrnAutoLoading(true); setGrnAutoError('')
    try {
      const isPO = ref.toUpperCase().startsWith('PO')
      const isJWO = ref.toUpperCase().startsWith('JWO')
      if (!isPO && !isJWO) { setGrnAutoError('Enter a valid PO or JWO number (e.g. PO-0001 or JWO-0001)'); setGrnAutoLoading(false); return }

      const endpoint = isPO ? `/purchase/po?search=${encodeURIComponent(ref)}` : `/purchase/jwo?search=${encodeURIComponent(ref)}`
      const res = await api.get(endpoint)
      const list = res.data as (PO | JWO)[]
      const doc = list.find((d: PO | JWO) => ('po_number' in d ? d.po_number : d.jwo_number) === ref)

      if (!doc) { setGrnAutoError(`Document ${ref} not found.`); setGrnAutoLoading(false); return }

      if (isPO) {
        const po = doc as PO
        setGRNForm(f => ({ ...f, grn_type: 'PO Receipt', reference_number: po.po_number, party_name: po.supplier_name }))
        setGRNLines(po.lines.map(l => ({
          material_code: l.material_code, material_name: l.material_name,
          received_qty: l.po_qty, accepted_qty: l.po_qty, rejected_qty: 0,
          unit: l.unit || 'PCS', rate: l.rate, qc_status: 'Pending'
        })))
      } else {
        const jwo = doc as JWO
        setGRNForm(f => ({ ...f, grn_type: 'JWO Receipt', reference_number: jwo.jwo_number, party_name: jwo.processor_name }))
        setGRNLines(jwo.lines.map(l => ({
          material_code: l.output_material, material_name: l.output_material,
          received_qty: l.output_qty, accepted_qty: l.output_qty, rejected_qty: 0,
          unit: l.unit || 'PCS', rate: l.rate, qc_status: 'Pending'
        })))
      }
      setShowGRNForm(true)
    } catch { setGrnAutoError('Failed to fetch document. Check the reference number.') }
    setGrnAutoLoading(false)
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-gray-800">Purchase Module</h1>
        <p className="text-sm text-gray-500">Suppliers, PR, PO, JWO, GRN management</p>
      </div>

      <div className="flex flex-wrap gap-1 bg-gray-100 p-1 rounded-lg w-fit">
        {TABS.map(([key, label]) => (
          <button key={key} onClick={() => { setTab(key); setFilterStatus(''); setExpanded(null) }}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${tab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* Dashboard */}
      {tab === 'dashboard' && stats && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {[
              { label: 'Open PRs', value: stats.open_prs, color: 'text-yellow-600' },
              { label: 'Open POs', value: stats.open_pos, color: 'text-blue-600' },
              { label: 'Open JWOs', value: stats.open_jwos, color: 'text-purple-600' },
              { label: 'Pending GRNs', value: stats.pending_grns, color: 'text-orange-600' },
              { label: 'Suppliers', value: stats.total_suppliers, color: 'text-green-600' },
              { label: 'Processors', value: stats.total_processors, color: 'text-gray-700' },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
                <p className={`text-2xl font-bold ${color}`}>{value}</p>
                <p className="text-xs text-gray-500 mt-1 font-semibold">{label}</p>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-3">
            {([['pr', '+ New PR'], ['po', '+ New PO'], ['jwo', '+ JWO'], ['grn', '+ GRN']] as [Tab, string][]).map(([t, label]) => (
              <button key={t} onClick={() => setTab(t)}
                className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Suppliers */}
      {tab === 'suppliers' && (
        <div className="space-y-4">
          <div className="flex justify-between">
            <p className="text-sm text-gray-500">{suppliers.length} active suppliers</p>
            <button onClick={() => setShowSupForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ Add Supplier</button>
          </div>
          {showSupForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Supplier</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[['supplier_name','Name *'],['contact_person','Contact'],['email','Email'],['phone','Phone'],['gst_number','GST'],['address','Address'],['payment_terms','Payment Terms']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(supForm as Record<string,string>)[k]} onChange={e => setSupForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                ))}
                <div><label className="text-xs text-gray-500">Type</label>
                  <select value={supForm.supplier_type} onChange={e => setSupForm(f => ({ ...f, supplier_type: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {SUP_TYPES.map(t => <option key={t}>{t}</option>)}
                  </select>
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createSupMut.mutate(supForm)} disabled={createSupMut.isPending || !supForm.supplier_name}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createSupMut.isPending ? 'Saving…' : 'Save'}
                </button>
                <button onClick={() => setShowSupForm(false)} className="px-4 py-2 border border-gray-200 rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}
          <div className="bg-white rounded-xl border border-gray-100 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                <tr>{['Code','Name','Type','Contact','Phone','Payment Terms'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {suppliers.map(s => (
                  <tr key={s.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">{s.supplier_code}</td>
                    <td className="px-4 py-2 text-gray-700">{s.supplier_name}</td>
                    <td className="px-4 py-2 text-gray-500">{s.supplier_type}</td>
                    <td className="px-4 py-2 text-gray-500">{s.contact_person}</td>
                    <td className="px-4 py-2 text-gray-500">{s.phone}</td>
                    <td className="px-4 py-2 text-gray-500">{s.payment_terms}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {suppliers.length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No suppliers yet</p>}
          </div>
        </div>
      )}

      {/* Processors */}
      {tab === 'processors' && (
        <div className="space-y-4">
          <div className="flex justify-between">
            <p className="text-sm text-gray-500">{processors.length} processors</p>
            <button onClick={() => setShowProcForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ Add Processor</button>
          </div>
          {showProcForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Processor</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[['processor_name','Name *'],['contact_person','Contact'],['phone','Phone'],['address','Address']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(procForm as Record<string,string>)[k]} onChange={e => setProcForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                ))}
                <div><label className="text-xs text-gray-500">Type</label>
                  <select value={procForm.processor_type} onChange={e => setProcForm(f => ({ ...f, processor_type: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {PROC_TYPES.map(t => <option key={t}>{t}</option>)}
                  </select>
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createProcMut.mutate(procForm)} disabled={createProcMut.isPending || !procForm.processor_name}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">Save</button>
                <button onClick={() => setShowProcForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}
          <div className="bg-white rounded-xl border border-gray-100 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                <tr>{['Code','Name','Type','Contact','Phone'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {processors.map(p => (
                  <tr key={p.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">{p.processor_code}</td>
                    <td className="px-4 py-2 text-gray-700">{p.processor_name}</td>
                    <td className="px-4 py-2 text-gray-500">{p.processor_type}</td>
                    <td className="px-4 py-2 text-gray-500">{p.contact_person}</td>
                    <td className="px-4 py-2 text-gray-500">{p.phone}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {processors.length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No processors yet</p>}
          </div>
        </div>
      )}

      {/* ── Purchase Requisitions ──────────────────────────────────────────── */}
      {tab === 'pr' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex gap-1 bg-gray-100 p-1 rounded-lg">
              {([['list', 'PR List'], ['new', '+ Create PR'], ['from-mrp', 'From MRP']] as [PRSubTab, string][]).map(([st, lbl]) => (
                <button key={st} onClick={() => { setPRSubTab(st); setGeneratedPRs([]) }}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${prSubTab === st ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
                  {lbl}
                </button>
              ))}
            </div>
            {prSubTab === 'list' && (
              <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
                <option value="">All Statuses</option>
                {['Pending Approval','Approved','Rejected','Partial PO','PO Created','Closed'].map(s => <option key={s}>{s}</option>)}
              </select>
            )}
          </div>

          {/* PR List */}
          {prSubTab === 'list' && (
            <div className="space-y-2">
              {prs.map(pr => {
                const totalPending = pr.lines.reduce((s, l) => s + Math.max(0, l.required_qty - (l.po_qty || 0)), 0)
                const isExpanded = expanded === pr.id
                return (
                  <div key={pr.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                    <div className="flex items-start justify-between p-4 cursor-pointer" onClick={() => { setExpanded(isExpanded ? null : pr.id); if (isExpanded) setShowCreatePO(null) }}>
                      <div>
                        <div className="flex items-center gap-2 flex-wrap">
                          <p className="font-semibold text-sm text-gray-800">{pr.pr_number}</p>
                          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${pr.pr_type === 'Job Work' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'}`}>
                            {pr.pr_type === 'Job Work' ? '✏️ Job Work' : '🛒 Purchase'}
                          </span>
                          {pr.source === 'MRP' && <span className="text-xs px-2 py-0.5 rounded-full bg-orange-100 text-orange-700 font-medium">From MRP</span>}
                        </div>
                        <p className="text-xs text-gray-500 mt-0.5">SO: {pr.so_reference || '—'} · {pr.lines.length} items · Pending: {Math.round(totalPending)}</p>
                      </div>
                      <div className="flex items-center gap-2 ml-2">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(pr.status)}`}>{pr.status}</span>
                        {['Pending Approval', 'Draft'].includes(pr.status) && (
                          <>
                            <button onClick={e => { e.stopPropagation(); approvePRMut.mutate({ id: pr.id, approver: 'Admin' }) }}
                              className="text-xs px-2 py-1 bg-green-600 text-white rounded font-medium hover:bg-green-700">✓ Approve</button>
                            <button onClick={e => { e.stopPropagation(); rejectPRMut.mutate(pr.id) }}
                              className="text-xs px-2 py-1 bg-red-500 text-white rounded font-medium hover:bg-red-600">✗ Reject</button>
                          </>
                        )}
                        <span className="text-gray-400 text-xs">{isExpanded ? '▲' : '▼'}</span>
                      </div>
                    </div>
                    {isExpanded && (
                      <div className="border-t px-4 pb-4 space-y-4">
                        <div className="grid grid-cols-3 gap-4 mt-3">
                          <div><p className="text-xs text-gray-400 uppercase font-semibold mb-1">PR Info</p>
                            <p className="text-xs text-gray-700">Date: {pr.pr_date}</p>
                            <p className="text-xs text-gray-700">Req By: {pr.required_by_date || pr.lines[0]?.required_by_date || '—'}</p>
                          </div>
                          <div><p className="text-xs text-gray-400 uppercase font-semibold mb-1">Source</p>
                            <p className="text-xs text-gray-700">SO: {pr.so_reference || '—'}</p>
                            <p className="text-xs text-gray-700">From: {pr.source || 'Manual'}</p>
                          </div>
                          <div><p className="text-xs text-gray-400 uppercase font-semibold mb-1">Status</p>
                            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(pr.status)}`}>{pr.status}</span>
                          </div>
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-gray-600 mb-2">PR Lines — Pending Qty:</p>
                          <table className="w-full text-xs">
                            <thead><tr className="text-gray-400 uppercase text-left border-b">
                              <th className="pb-1">Material</th><th className="pb-1">Type</th>
                              <th className="pb-1 text-right">Required Qty</th><th className="pb-1 text-right">PO Created</th>
                              <th className="pb-1 text-right">Pending</th><th className="pb-1 text-right">Unit</th>
                            </tr></thead>
                            <tbody>
                              {pr.lines.map(l => {
                                const pending = Math.max(0, l.required_qty - (l.po_qty || 0))
                                return (
                                  <tr key={l.id} className="border-t border-gray-50">
                                    <td className="py-1.5 font-medium text-gray-800">{l.material_name || l.material_code}</td>
                                    <td className="py-1.5 text-gray-500">{l.material_type}</td>
                                    <td className="py-1.5 text-right">{l.required_qty}</td>
                                    <td className="py-1.5 text-right text-blue-600">{l.po_qty || 0}</td>
                                    <td className={`py-1.5 text-right font-semibold ${pending > 0 ? 'text-red-500' : 'text-green-600'}`}>{pending}</td>
                                    <td className="py-1.5 text-right text-gray-500">{l.unit}</td>
                                  </tr>
                                )
                              })}
                            </tbody>
                          </table>
                        </div>
                        {['Approved', 'Partial PO', 'PO Created'].includes(pr.status) && totalPending > 0 && (
                          <div className="border-t pt-4">
                            {showCreatePO !== pr.id ? (
                              <button onClick={() => openCreatePO(pr)}
                                className="flex items-center gap-2 px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">
                                {pr.pr_type === 'Job Work' ? '⚙️ Create Job Work Order from this PR' : '🛒 Create Order from this PR'}
                              </button>
                            ) : (
                              <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                  <h3 className="font-semibold text-gray-800">🛒 Create Purchase Order from this PR</h3>
                                  <button onClick={() => setShowCreatePO(null)} className="text-gray-400 hover:text-gray-600 text-sm">✕ Close</button>
                                </div>
                                <p className="text-xs text-blue-700 bg-blue-50 px-3 py-2 rounded-lg">
                                  For each item assign a supplier (PO) or processor (JWO). SFG items default to JWO. Same supplier lines merge into one PO automatically.
                                </p>
                                <div className="grid grid-cols-2 gap-3">
                                  <div><label className="text-xs text-gray-500">Default Delivery / Return Date</label>
                                    <input type="date" value={poFromPRMeta.delivery_date} onChange={e => setPoFromPRMeta(m => ({ ...m, delivery_date: e.target.value }))}
                                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                                  <div><label className="text-xs text-gray-500">Payment Terms (PO)</label>
                                    <select value={poFromPRMeta.payment_terms} onChange={e => setPoFromPRMeta(m => ({ ...m, payment_terms: e.target.value }))}
                                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                                      {PAYMENT_TERMS.map(t => <option key={t}>{t}</option>)}
                                    </select>
                                  </div>
                                </div>
                                {pr.lines.filter(l => (l.required_qty - (l.po_qty || 0)) > 0).map(l => {
                                  const key = `${pr.id}-${l.material_code}`
                                  const defaultType = l.material_type === 'SFG' ? 'JWO' : 'PO'
                                  const ld = poFromPRLines[key] || { order_type: defaultType, supplier_name: '', processor_name: '', qty: l.required_qty - (l.po_qty || 0), rate: 0, gst_pct: 12 }
                                  const isJWO = (ld.order_type || defaultType) === 'JWO'
                                  return (
                                    <div key={l.id} className="border border-gray-200 rounded-lg p-3 space-y-2">
                                      <div className="flex items-start justify-between gap-2">
                                        <div>
                                          <p className="font-medium text-sm text-gray-800">{l.material_name || l.material_code}</p>
                                          <p className="text-xs text-gray-400">{l.material_type} | Pending: {l.required_qty - (l.po_qty || 0)} {l.unit}</p>
                                        </div>
                                        <div className="flex gap-0.5 bg-gray-100 p-0.5 rounded-lg shrink-0">
                                          {(['PO', 'JWO'] as const).map(type => (
                                            <button key={type}
                                              onClick={() => setPoFromPRLines(prev => ({ ...prev, [key]: { ...(prev[key] || ld), order_type: type, supplier_id: undefined, supplier_name: '', processor_id: undefined, processor_name: '' } }))}
                                              className={`px-2.5 py-0.5 rounded text-xs font-medium transition-colors ${(ld.order_type || defaultType) === type ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
                                              {type === 'PO' ? '🛒 PO' : '⚙️ JWO'}
                                            </button>
                                          ))}
                                        </div>
                                      </div>
                                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                        <div>
                                          <label className="text-xs text-gray-400">{isJWO ? 'Processor' : 'Supplier'}</label>
                                          {isJWO ? (
                                            <select value={ld.processor_id || ''}
                                              onChange={e => { const p = processors.find(x => x.id === +e.target.value); setPoFromPRLines(prev => ({ ...prev, [key]: { ...(prev[key] || ld), processor_id: +e.target.value || undefined, processor_name: p?.processor_name || '' } })) }}
                                              className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-0.5">
                                              <option value="">Select processor</option>
                                              {processors.map(p => <option key={p.id} value={p.id}>{p.processor_name}</option>)}
                                            </select>
                                          ) : (
                                            <select value={ld.supplier_id || ''}
                                              onChange={e => { const s = suppliers.find(x => x.id === +e.target.value); setPoFromPRLines(prev => ({ ...prev, [key]: { ...(prev[key] || ld), supplier_id: +e.target.value || undefined, supplier_name: s?.supplier_name || '' } })) }}
                                              className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-0.5">
                                              <option value="">Select supplier</option>
                                              {suppliers.map(s => <option key={s.id} value={s.id}>{s.supplier_name}</option>)}
                                            </select>
                                          )}
                                        </div>
                                        <div><label className="text-xs text-gray-400">Qty</label>
                                          <div className="flex items-center gap-1 mt-0.5">
                                            <button onClick={() => updatePOLine(pr.id, l.material_code, 'qty', Math.max(0, ld.qty - 1))} className="w-7 h-7 border rounded text-sm hover:bg-gray-100">−</button>
                                            <input type="number" value={ld.qty} onChange={e => updatePOLine(pr.id, l.material_code, 'qty', +e.target.value)} className="flex-1 border border-gray-200 rounded px-2 py-1 text-sm text-center min-w-0" />
                                            <button onClick={() => updatePOLine(pr.id, l.material_code, 'qty', ld.qty + 1)} className="w-7 h-7 border rounded text-sm hover:bg-gray-100">+</button>
                                          </div>
                                        </div>
                                        <div><label className="text-xs text-gray-400">Rate (₹)</label>
                                          <div className="flex items-center gap-1 mt-0.5">
                                            <button onClick={() => updatePOLine(pr.id, l.material_code, 'rate', Math.max(0, ld.rate - 1))} className="w-7 h-7 border rounded text-sm hover:bg-gray-100">−</button>
                                            <input type="number" value={ld.rate} onChange={e => updatePOLine(pr.id, l.material_code, 'rate', +e.target.value)} className="flex-1 border border-gray-200 rounded px-2 py-1 text-sm text-center min-w-0" />
                                            <button onClick={() => updatePOLine(pr.id, l.material_code, 'rate', ld.rate + 1)} className="w-7 h-7 border rounded text-sm hover:bg-gray-100">+</button>
                                          </div>
                                        </div>
                                        <div><label className="text-xs text-gray-400">{isJWO ? 'Process Rate' : 'GST%'}</label>
                                          {isJWO ? <p className="text-xs text-gray-400 mt-1.5">N/A for JWO</p> : (
                                            <select value={ld.gst_pct} onChange={e => updatePOLine(pr.id, l.material_code, 'gst_pct', +e.target.value)}
                                              className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-0.5">
                                              {GST_RATES.map(g => <option key={g} value={g}>{g}%</option>)}
                                            </select>
                                          )}
                                        </div>
                                      </div>
                                    </div>
                                  )
                                })}
                                {createdPOs.length > 0 ? (
                                  <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                                    <p className="text-sm font-semibold text-green-700">POs Created Successfully!</p>
                                    <p className="text-xs text-green-600 mt-1">{createdPOs.join(', ')}</p>
                                  </div>
                                ) : (
                                  <button onClick={() => submitPOFromPR(pr)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">Create PO from PR</button>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
              {prs.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No purchase requisitions found.</p>}
            </div>
          )}

          {/* Create PR manually */}
          {prSubTab === 'new' && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Purchase Requisition</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[['requested_by','Requested By'],['so_reference','SO Reference'],['notes','Notes']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(prForm as Record<string,string>)[k]} onChange={e => setPRForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                ))}
                <div><label className="text-xs text-gray-500">Required By Date</label>
                  <input type="date" value={prForm.required_by_date} onChange={e => setPRForm(f => ({ ...f, required_by_date: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                <div><label className="text-xs text-gray-500">Department</label>
                  <select value={prForm.department} onChange={e => setPRForm(f => ({ ...f, department: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {DEPTS.map(d => <option key={d}>{d}</option>)}
                  </select>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-2"><p className="text-sm font-medium text-gray-600">Material Lines</p>
                  <button onClick={() => setPRLines(l => [...l, { material_code: '', material_name: '', material_type: 'RM', required_qty: 0, unit: 'PCS' }])}
                    className="text-xs text-blue-600 hover:underline">+ Add Line</button></div>
                {prLines.map((ln, i) => (
                  <div key={i} className="grid grid-cols-6 gap-2 mb-2">
                    <input placeholder="Material Code" value={ln.material_code} onChange={e => setPRLines(l => l.map((x, j) => j === i ? { ...x, material_code: e.target.value } : x))} className="border border-gray-200 rounded px-2 py-1.5 text-sm col-span-1" />
                    <input placeholder="Material Name" value={ln.material_name} onChange={e => setPRLines(l => l.map((x, j) => j === i ? { ...x, material_name: e.target.value } : x))} className="border border-gray-200 rounded px-2 py-1.5 text-sm col-span-2" />
                    <select value={ln.material_type} onChange={e => setPRLines(l => l.map((x, j) => j === i ? { ...x, material_type: e.target.value } : x))} className="border border-gray-200 rounded px-2 py-1.5 text-sm">
                      {MAT_TYPES.map(t => <option key={t}>{t}</option>)}
                    </select>
                    <input type="number" placeholder="Qty" value={ln.required_qty} onChange={e => setPRLines(l => l.map((x, j) => j === i ? { ...x, required_qty: +e.target.value } : x))} className="border border-gray-200 rounded px-2 py-1.5 text-sm" />
                    <button onClick={() => setPRLines(l => l.filter((_, j) => j !== i))} className="text-red-400 text-sm">✕</button>
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <button onClick={() => createPRMut.mutate({ ...prForm, source: 'Manual', pr_type: 'Purchase', lines: prLines })} disabled={createPRMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createPRMut.isPending ? 'Saving…' : 'Create PR'}
                </button>
                <button onClick={() => setPRSubTab('list')} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}

          {/* From MRP */}
          {prSubTab === 'from-mrp' && (
            <div className="space-y-4">
              <div className="bg-white rounded-xl border p-4 space-y-4">
                <h3 className="font-semibold text-gray-700">Generate PR from MRP</h3>
                <div className="grid grid-cols-3 gap-3 items-end">
                  <div><label className="text-xs text-gray-500">SO Reference</label>
                    <input value={mrpSO} onChange={e => setMrpSO(e.target.value)} placeholder="SO-0001"
                      onKeyDown={e => e.key === 'Enter' && loadMRPLines()} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                  <div><label className="text-xs text-gray-500">Required By Date</label>
                    <input type="date" value={mrpReqDate} onChange={e => setMrpReqDate(e.target.value)} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                  <button onClick={loadMRPLines} disabled={!mrpSO.trim() || mrpLoadingLines}
                    className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800 disabled:opacity-50">
                    {mrpLoadingLines ? 'Loading…' : 'Load MRP'}
                  </button>
                </div>
                {mrpLinesData?.error && <p className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">{mrpLinesData.error}</p>}
                {mrpLinesData?.warning && <p className="text-sm text-yellow-700 bg-yellow-50 px-3 py-2 rounded-lg">{mrpLinesData.warning}</p>}
                {mrpLinesData && !mrpLinesData.error && (
                  <div className="space-y-5">
                    {mrpLinesData.purchase_items.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-gray-700 mb-2 flex items-center gap-2">🛒 Purchase PR — Raw Materials &amp; Accessories
                          <span className="text-xs text-gray-400 font-normal">{mrpLinesData.purchase_items.filter(i => selectedPurchaseItems.has(i.material_code)).length} selected</span>
                        </h4>
                        <div className="bg-white border border-gray-100 rounded-lg overflow-hidden">
                          <table className="w-full text-sm">
                            <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                              <tr>
                                <th className="px-3 py-2 text-left w-8">
                                  <input type="checkbox" checked={mrpLinesData.purchase_items.every(i => selectedPurchaseItems.has(i.material_code))}
                                    onChange={e => setSelectedPurchaseItems(e.target.checked ? new Set(mrpLinesData.purchase_items.map(i => i.material_code)) : new Set())} />
                                </th>
                                <th className="px-3 py-2 text-left">Code</th><th className="px-3 py-2 text-left">Name</th>
                                <th className="px-3 py-2 text-left">Type</th><th className="px-3 py-2 text-right">Net Required</th><th className="px-3 py-2 text-right">Unit</th>
                              </tr>
                            </thead>
                            <tbody>
                              {mrpLinesData.purchase_items.map(item => (
                                <tr key={item.material_code} className="border-t border-gray-50 hover:bg-gray-50">
                                  <td className="px-3 py-2">
                                    <input type="checkbox" checked={selectedPurchaseItems.has(item.material_code)}
                                      onChange={e => setSelectedPurchaseItems(prev => { const n = new Set(prev); e.target.checked ? n.add(item.material_code) : n.delete(item.material_code); return n })} />
                                  </td>
                                  <td className="px-3 py-2 font-medium text-gray-800">{item.material_code}</td>
                                  <td className="px-3 py-2 text-gray-600">{item.material_name}</td>
                                  <td className="px-3 py-2"><span className="text-xs bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded">{item.material_type}</span></td>
                                  <td className="px-3 py-2 text-right font-semibold text-gray-800">{item.net_req || item.required_qty}</td>
                                  <td className="px-3 py-2 text-right text-gray-500">{item.unit}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                    {mrpLinesData.sfg_items.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-gray-700 mb-1 flex items-center gap-2">✏️ Job Work PR — Semi-Finished Goods
                          <span className="text-xs text-gray-400 font-normal">{selectedSFGItems.size} selected</span>
                        </h4>
                        <p className="text-xs text-gray-500 mb-2">These SFG items will go into a Job Work PR. Uncheck to exclude.</p>
                        <div className="bg-white border border-gray-100 rounded-lg overflow-hidden">
                          <table className="w-full text-sm">
                            <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                              <tr>
                                <th className="px-3 py-2 text-left w-8">
                                  <input type="checkbox" checked={mrpLinesData.sfg_items.every(i => selectedSFGItems.has(i.material_code))}
                                    onChange={e => setSelectedSFGItems(e.target.checked ? new Set(mrpLinesData.sfg_items.map(i => i.material_code)) : new Set())} />
                                </th>
                                <th className="px-3 py-2 text-left">Code</th><th className="px-3 py-2 text-left">Name</th>
                                <th className="px-3 py-2 text-right">Qty</th><th className="px-3 py-2 text-right">Unit</th><th className="px-3 py-2 text-left">BOM Inputs</th>
                              </tr>
                            </thead>
                            <tbody>
                              {mrpLinesData.sfg_items.map(item => (
                                <tr key={item.material_code} className="border-t border-gray-50 hover:bg-gray-50">
                                  <td className="px-3 py-2">
                                    <input type="checkbox" checked={selectedSFGItems.has(item.material_code)}
                                      onChange={e => setSelectedSFGItems(prev => { const n = new Set(prev); e.target.checked ? n.add(item.material_code) : n.delete(item.material_code); return n })} />
                                  </td>
                                  <td className="px-3 py-2 font-medium text-gray-800">{item.material_code}</td>
                                  <td className="px-3 py-2 text-gray-600">{item.material_name}</td>
                                  <td className="px-3 py-2 text-right font-semibold text-gray-800">{item.net_req || item.required_qty}</td>
                                  <td className="px-3 py-2 text-right text-gray-500">{item.unit}</td>
                                  <td className="px-3 py-2 text-xs text-blue-600">{(item.inputs || []).map(inp => `${inp.material_code} (${inp.quantity} ${inp.unit})`).join(', ') || '—'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                    {generatedPRs.length > 0 ? (
                      <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                        <p className="text-sm font-semibold text-green-700">PRs Generated Successfully!</p>
                        <p className="text-xs text-green-600 mt-1">{generatedPRs.join(', ')}</p>
                        <button onClick={() => { setPRSubTab('list'); setMrpLinesData(null) }} className="mt-2 text-xs text-green-700 underline hover:no-underline">View PR List →</button>
                      </div>
                    ) : (
                      <button onClick={generatePRsFromMRP} disabled={generatingPRs || (mrpLinesData.purchase_items.length === 0 && mrpLinesData.sfg_items.length === 0)}
                        className="px-5 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800 disabled:opacity-50">
                        {generatingPRs ? 'Generating PRs…' : 'Generate PRs'}
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Purchase Orders ─────────────────────────────────────────────────── */}
      {tab === 'po' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All</option>
              {['Draft','Sent to Supplier','Confirmed','Partial Received','Received','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={() => setShowPOForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ New PO</button>
          </div>
          {showPOForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Purchase Order</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div><label className="text-xs text-gray-500">Supplier</label>
                  <select value={poForm.supplier_id ?? ''} onChange={e => { const s = suppliers.find(x => x.id === +e.target.value); setPOForm(f => ({ ...f, supplier_id: +e.target.value || undefined, supplier_name: s?.supplier_name || '' })) }}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select supplier</option>
                    {suppliers.map(s => <option key={s.id} value={s.id}>{s.supplier_name}</option>)}
                  </select>
                </div>
                {[['delivery_date','Delivery Date'],['payment_terms','Payment Terms'],['delivery_location','Delivery Location'],['pr_reference','PR Reference'],['so_reference','SO Reference'],['remarks','Remarks']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input type={k === 'delivery_date' ? 'date' : 'text'} value={(poForm as Record<string,unknown>)[k] as string}
                      onChange={e => setPOForm(f => ({ ...f, [k]: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                ))}
              </div>
              <div>
                <div className="flex justify-between mb-2"><p className="text-sm font-medium text-gray-600">PO Lines</p>
                  <button onClick={() => setPOLines(l => [...l, { material_code: '', material_name: '', material_type: 'RM', po_qty: 0, unit: 'PCS', rate: 0, gst_pct: 0 }])}
                    className="text-xs text-blue-600 hover:underline">+ Add Line</button></div>
                {poLines.map((ln, i) => (
                  <div key={i} className="grid grid-cols-7 gap-2 mb-2 text-sm">
                    <input placeholder="Code" value={ln.material_code} onChange={e => setPOLines(l => l.map((x, j) => j === i ? { ...x, material_code: e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input placeholder="Name" value={ln.material_name} onChange={e => setPOLines(l => l.map((x, j) => j === i ? { ...x, material_name: e.target.value } : x))} className="border rounded px-2 py-1.5 col-span-2" />
                    <input type="number" placeholder="Qty" value={ln.po_qty} onChange={e => setPOLines(l => l.map((x, j) => j === i ? { ...x, po_qty: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Rate ₹" value={ln.rate} onChange={e => setPOLines(l => l.map((x, j) => j === i ? { ...x, rate: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <span className="py-1.5 text-right text-gray-600 font-medium">{fmt(ln.po_qty * ln.rate)}</span>
                    <button onClick={() => setPOLines(l => l.filter((_, j) => j !== i))} className="text-red-400 text-sm">✕</button>
                  </div>
                ))}
                {poLines.length > 0 && <p className="text-sm font-semibold text-right text-gray-700 mt-2">Total: {fmt(poLines.reduce((s, l) => s + l.po_qty * l.rate, 0))}</p>}
              </div>
              <div className="flex gap-2">
                <button onClick={() => createPOMut.mutate({ ...poForm, lines: poLines.map(l => ({ ...l, amount: l.po_qty * l.rate })) })} disabled={createPOMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createPOMut.isPending ? 'Saving…' : 'Create PO'}
                </button>
                <button onClick={() => setShowPOForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}
          <div className="space-y-2">
            {pos.map(po => (
              <div key={po.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                <div className="flex items-center justify-between p-4 cursor-pointer" onClick={() => setExpanded(expanded === po.id ? null : po.id)}>
                  <div>
                    <p className="font-semibold text-sm text-gray-800">{po.po_number}</p>
                    <p className="text-xs text-gray-500">{po.supplier_name} · SO: {po.so_reference || '—'} · Delivery: {po.delivery_date || '—'}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-700">{fmt(po.total)}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(po.status)}`}>{po.status}</span>
                    {/* ── Print button for PO ── */}
                    <button onClick={e => { e.stopPropagation(); printDocument(buildPOPrintHTML(po), `PO - ${po.po_number}`) }}
                      className="text-xs px-2 py-1 border border-gray-200 rounded text-gray-500 hover:bg-gray-50 flex items-center gap-1" title="Print PO">
                      🖨️ Print
                    </button>
                    <select value={po.status} onClick={e => e.stopPropagation()} onChange={e => updatePOStatusMut.mutate({ id: po.id, status: e.target.value })}
                      className="border border-gray-200 rounded px-2 py-1 text-xs">
                      {['Draft','Sent to Supplier','Confirmed','Partial Received','Received','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
                    </select>
                    <span className="text-gray-400 text-xs">{expanded === po.id ? '▲' : '▼'}</span>
                  </div>
                </div>
                {expanded === po.id && (
                  <div className="border-t px-4 pb-4 space-y-3 mt-0">
                    {editingPO === po.id ? (
                      <div className="space-y-3 pt-3">
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                          <div><label className="text-xs text-gray-500">Supplier</label>
                            <select value={editPOForm.supplier_id ?? ''} onChange={e => { const s = suppliers.find(x => x.id === +e.target.value); setEditPOForm(f => ({ ...f, supplier_id: +e.target.value || undefined, supplier_name: s?.supplier_name || '' })) }}
                              className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                              <option value="">Select supplier</option>
                              {suppliers.map(s => <option key={s.id} value={s.id}>{s.supplier_name}</option>)}
                            </select>
                          </div>
                          <div><label className="text-xs text-gray-500">Delivery Date</label>
                            <input type="date" value={editPOForm.delivery_date} onChange={e => setEditPOForm(f => ({ ...f, delivery_date: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                          <div><label className="text-xs text-gray-500">Payment Terms</label>
                            <select value={editPOForm.payment_terms} onChange={e => setEditPOForm(f => ({ ...f, payment_terms: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                              <option value="">—</option>{PAYMENT_TERMS.map(t => <option key={t}>{t}</option>)}
                            </select>
                          </div>
                          <div><label className="text-xs text-gray-500">SO Reference</label>
                            <input value={editPOForm.so_reference} onChange={e => setEditPOForm(f => ({ ...f, so_reference: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                          <div className="col-span-2"><label className="text-xs text-gray-500">Remarks</label>
                            <input value={editPOForm.remarks} onChange={e => setEditPOForm(f => ({ ...f, remarks: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                        </div>
                        <div>
                          <div className="flex justify-between mb-2">
                            <p className="text-xs font-semibold text-gray-600">PO Lines</p>
                            <button onClick={() => setEditPOLines(l => [...l, { material_code: '', material_name: '', material_type: 'RM', po_qty: 0, unit: 'PCS', rate: 0, gst_pct: 0 }])} className="text-xs text-blue-600 hover:underline">+ Add Line</button>
                          </div>
                          <table className="w-full text-xs">
                            <thead><tr className="text-gray-400 uppercase border-b">
                              <th className="text-left pb-1">Code</th><th className="text-left pb-1">Name</th><th className="text-left pb-1">Type</th>
                              <th className="text-right pb-1">Qty</th><th className="text-right pb-1">Rate ₹</th><th className="text-right pb-1">GST%</th>
                              <th className="text-right pb-1">Amount</th><th className="pb-1"></th>
                            </tr></thead>
                            <tbody>
                              {editPOLines.map((ln, i) => (
                                <tr key={i} className="border-t border-gray-50">
                                  <td className="py-1"><input value={ln.material_code} onChange={e => setEditPOLines(l => l.map((x, j) => j === i ? { ...x, material_code: e.target.value } : x))} className="w-full border rounded px-1.5 py-1" /></td>
                                  <td className="py-1"><input value={ln.material_name} onChange={e => setEditPOLines(l => l.map((x, j) => j === i ? { ...x, material_name: e.target.value } : x))} className="w-full border rounded px-1.5 py-1" /></td>
                                  <td className="py-1"><select value={ln.material_type} onChange={e => setEditPOLines(l => l.map((x, j) => j === i ? { ...x, material_type: e.target.value } : x))} className="w-full border rounded px-1.5 py-1">{MAT_TYPES.map(t => <option key={t}>{t}</option>)}</select></td>
                                  <td className="py-1"><input type="number" value={ln.po_qty} onChange={e => setEditPOLines(l => l.map((x, j) => j === i ? { ...x, po_qty: +e.target.value } : x))} className="w-16 border rounded px-1.5 py-1 text-right" /></td>
                                  <td className="py-1"><input type="number" value={ln.rate} onChange={e => setEditPOLines(l => l.map((x, j) => j === i ? { ...x, rate: +e.target.value } : x))} className="w-20 border rounded px-1.5 py-1 text-right" /></td>
                                  <td className="py-1"><select value={ln.gst_pct} onChange={e => setEditPOLines(l => l.map((x, j) => j === i ? { ...x, gst_pct: +e.target.value } : x))} className="w-full border rounded px-1.5 py-1">{GST_RATES.map(g => <option key={g} value={g}>{g}%</option>)}</select></td>
                                  <td className="py-1 text-right font-medium pr-1">{fmt(ln.po_qty * ln.rate)}</td>
                                  <td className="py-1"><button onClick={() => setEditPOLines(l => l.filter((_, j) => j !== i))} className="text-red-400 px-1">✕</button></td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {editPOLines.length > 0 && <p className="text-xs font-semibold text-right text-gray-700 mt-1 pr-6">Total: {fmt(editPOLines.reduce((s, l) => s + l.po_qty * l.rate, 0))}</p>}
                        </div>
                        <div className="flex gap-2 pt-1">
                          <button onClick={() => updatePOMut.mutate({ id: po.id, data: { ...editPOForm, lines: editPOLines } })} disabled={updatePOMut.isPending}
                            className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                            {updatePOMut.isPending ? 'Saving…' : 'Save Changes'}
                          </button>
                          <button onClick={() => setEditingPO(null)} className="px-4 py-2 border border-gray-200 rounded-lg text-sm text-gray-600">Cancel</button>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-2 pt-3">
                        {po.status === 'Draft' && (
                          <div className="flex justify-end">
                            <button onClick={() => { const sup = suppliers.find(s => s.supplier_name === po.supplier_name); setEditingPO(po.id); setEditPOForm({ supplier_id: sup?.id, supplier_name: po.supplier_name, delivery_date: po.delivery_date || '', payment_terms: '', so_reference: po.so_reference || '', remarks: '' }); setEditPOLines(po.lines.map(l => ({ material_code: l.material_code, material_name: l.material_name, material_type: l.material_type || 'RM', po_qty: l.po_qty, unit: l.unit, rate: l.rate, gst_pct: l.gst_pct }))) }}
                              className="text-xs px-3 py-1 border border-gray-300 rounded-lg text-gray-600 hover:bg-gray-50">✏️ Edit</button>
                          </div>
                        )}
                        <table className="w-full text-xs">
                          <thead><tr className="text-gray-400 uppercase border-b"><th className="text-left pb-1">Code</th><th className="text-left pb-1">Name</th><th className="text-right pb-1">Qty</th><th className="text-right pb-1">Rate</th><th className="text-right pb-1">Amount</th></tr></thead>
                          <tbody>{po.lines.map(l => (
                            <tr key={l.id} className="border-t border-gray-50">
                              <td className="py-1.5 font-medium">{l.material_code}</td><td className="py-1.5">{l.material_name}</td>
                              <td className="py-1.5 text-right">{l.po_qty}</td><td className="py-1.5 text-right">{fmt(l.rate)}</td>
                              <td className="py-1.5 text-right font-medium">{fmt(l.amount)}</td>
                            </tr>
                          ))}</tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {pos.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No purchase orders found.</p>}
          </div>
        </div>
      )}

      {/* ── Job Work Orders ──────────────────────────────────────────────────── */}
      {tab === 'jwo' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All</option>
              {['Draft','Issued to Processor','In Process','Partial Received','Received','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={() => setShowJWOForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ New JWO</button>
          </div>

          {showJWOForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Job Work Order</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div><label className="text-xs text-gray-500">Processor</label>
                  <select value={jwoForm.processor_id ?? ''} onChange={e => { const p = processors.find(x => x.id === +e.target.value); setJWOForm(f => ({ ...f, processor_id: +e.target.value, processor_name: p?.processor_name || '' })) }}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    <option value="">Select processor</option>
                    {processors.map(p => <option key={p.id} value={p.id}>{p.processor_name}</option>)}
                  </select>
                </div>
                {[['expected_return_date','Return Date','date'],['so_reference','SO Reference','text'],['issued_by','Issued By','text']].map(([k,l,t]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input type={t} value={(jwoForm as Record<string,unknown>)[k] as string}
                      onChange={e => setJWOForm(f => ({ ...f, [k]: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                ))}
              </div>
              <div>
                <div className="flex justify-between mb-2"><p className="text-sm font-medium text-gray-600">Process Lines</p>
                  <button onClick={() => setJWOLines(l => [...l, { input_material: '', input_qty: 0, output_material: '', output_qty: 0, process_type: 'Printing', rate: 0 }])}
                    className="text-xs text-blue-600 hover:underline">+ Add Line</button></div>
                {jwoLines.map((ln, i) => (
                  <div key={i} className="grid grid-cols-6 gap-2 mb-2 text-sm">
                    <input placeholder="Input Material" value={ln.input_material} onChange={e => setJWOLines(l => l.map((x, j) => j === i ? { ...x, input_material: e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Input Qty" value={ln.input_qty} onChange={e => setJWOLines(l => l.map((x, j) => j === i ? { ...x, input_qty: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input placeholder="Output Material" value={ln.output_material} onChange={e => setJWOLines(l => l.map((x, j) => j === i ? { ...x, output_material: e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Output Qty" value={ln.output_qty} onChange={e => setJWOLines(l => l.map((x, j) => j === i ? { ...x, output_qty: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <select value={ln.process_type} onChange={e => setJWOLines(l => l.map((x, j) => j === i ? { ...x, process_type: e.target.value } : x))} className="border rounded px-2 py-1.5">
                      {PROC_TYPES2.map(t => <option key={t}>{t}</option>)}
                    </select>
                    <button onClick={() => setJWOLines(l => l.filter((_, j) => j !== i))} className="text-red-400">✕</button>
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <button onClick={() => createJWOMut.mutate({ ...jwoForm, lines: jwoLines })} disabled={createJWOMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createJWOMut.isPending ? 'Saving…' : 'Create JWO'}
                </button>
                <button onClick={() => setShowJWOForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {jwos.map(jwo => (
              <div key={jwo.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                <div className="flex items-center justify-between p-4 cursor-pointer" onClick={() => { setExpanded(expanded === jwo.id ? null : jwo.id); if (expanded === jwo.id) setEditingJWO(null) }}>
                  <div>
                    <p className="font-semibold text-sm text-gray-800">{jwo.jwo_number}</p>
                    <p className="text-xs text-gray-500">{jwo.processor_name} · Return: {jwo.expected_return_date || '—'} · SO: {jwo.so_reference || '—'}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-700">{fmt(jwo.total)}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(jwo.status)}`}>{jwo.status}</span>
                    {/* ── Print button for JWO ── */}
                    <button onClick={e => { e.stopPropagation(); printDocument(buildJWOPrintHTML(jwo), `JWO - ${jwo.jwo_number}`) }}
                      className="text-xs px-2 py-1 border border-gray-200 rounded text-gray-500 hover:bg-gray-50 flex items-center gap-1" title="Print JWO">
                      🖨️ Print
                    </button>
                    <select value={jwo.status} onClick={e => e.stopPropagation()} onChange={e => updateJWOStatusMut.mutate({ id: jwo.id, status: e.target.value })}
                      className="border border-gray-200 rounded px-2 py-1 text-xs">
                      {['Draft','Issued to Processor','In Process','Partial Received','Received','Closed','Cancelled'].map(s => <option key={s}>{s}</option>)}
                    </select>
                    <span className="text-gray-400 text-xs">{expanded === jwo.id ? '▲' : '▼'}</span>
                  </div>
                </div>

                {/* ── NEW: JWO Expanded view with Edit ── */}
                {expanded === jwo.id && (
                  <div className="border-t px-4 pb-4 space-y-3">
                    {editingJWO === jwo.id ? (
                      /* ── JWO Edit form ── */
                      <div className="space-y-3 pt-3">
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                          <div><label className="text-xs text-gray-500">Processor</label>
                            <select value={editJWOForm.processor_id ?? ''} onChange={e => { const p = processors.find(x => x.id === +e.target.value); setEditJWOForm(f => ({ ...f, processor_id: +e.target.value || undefined, processor_name: p?.processor_name || '' })) }}
                              className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                              <option value="">Select processor</option>
                              {processors.map(p => <option key={p.id} value={p.id}>{p.processor_name}</option>)}
                            </select>
                          </div>
                          <div><label className="text-xs text-gray-500">Expected Return Date</label>
                            <input type="date" value={editJWOForm.expected_return_date} onChange={e => setEditJWOForm(f => ({ ...f, expected_return_date: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                          <div><label className="text-xs text-gray-500">SO Reference</label>
                            <input value={editJWOForm.so_reference} onChange={e => setEditJWOForm(f => ({ ...f, so_reference: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                          <div><label className="text-xs text-gray-500">PR Reference</label>
                            <input value={editJWOForm.pr_reference} onChange={e => setEditJWOForm(f => ({ ...f, pr_reference: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                          <div><label className="text-xs text-gray-500">Issued By</label>
                            <input value={editJWOForm.issued_by} onChange={e => setEditJWOForm(f => ({ ...f, issued_by: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                          <div><label className="text-xs text-gray-500">Remarks</label>
                            <input value={editJWOForm.remarks} onChange={e => setEditJWOForm(f => ({ ...f, remarks: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                        </div>
                        <div>
                          <div className="flex justify-between mb-2">
                            <p className="text-xs font-semibold text-gray-600">Process Lines</p>
                            <button onClick={() => setEditJWOLines(l => [...l, { input_material: '', input_qty: 0, output_material: '', output_qty: 0, process_type: 'Printing', rate: 0 }])} className="text-xs text-blue-600 hover:underline">+ Add Line</button>
                          </div>
                          <table className="w-full text-xs">
                            <thead><tr className="text-gray-400 uppercase border-b">
                              <th className="text-left pb-1">Input Material</th><th className="text-right pb-1">Input Qty</th>
                              <th className="text-left pb-1">Output Material</th><th className="text-right pb-1">Output Qty</th>
                              <th className="text-left pb-1">Process</th><th className="text-right pb-1">Rate ₹</th>
                              <th className="text-right pb-1">Amount</th><th className="pb-1"></th>
                            </tr></thead>
                            <tbody>
                              {editJWOLines.map((ln, i) => (
                                <tr key={i} className="border-t border-gray-50">
                                  <td className="py-1"><input value={ln.input_material} onChange={e => setEditJWOLines(l => l.map((x, j) => j === i ? { ...x, input_material: e.target.value } : x))} className="w-full border rounded px-1.5 py-1" placeholder="Input" /></td>
                                  <td className="py-1"><input type="number" value={ln.input_qty} onChange={e => setEditJWOLines(l => l.map((x, j) => j === i ? { ...x, input_qty: +e.target.value } : x))} className="w-16 border rounded px-1.5 py-1 text-right" /></td>
                                  <td className="py-1"><input value={ln.output_material} onChange={e => setEditJWOLines(l => l.map((x, j) => j === i ? { ...x, output_material: e.target.value } : x))} className="w-full border rounded px-1.5 py-1" placeholder="Output" /></td>
                                  <td className="py-1"><input type="number" value={ln.output_qty} onChange={e => setEditJWOLines(l => l.map((x, j) => j === i ? { ...x, output_qty: +e.target.value } : x))} className="w-16 border rounded px-1.5 py-1 text-right" /></td>
                                  <td className="py-1"><select value={ln.process_type} onChange={e => setEditJWOLines(l => l.map((x, j) => j === i ? { ...x, process_type: e.target.value } : x))} className="w-full border rounded px-1.5 py-1">{PROC_TYPES2.map(t => <option key={t}>{t}</option>)}</select></td>
                                  <td className="py-1"><input type="number" value={ln.rate} onChange={e => setEditJWOLines(l => l.map((x, j) => j === i ? { ...x, rate: +e.target.value } : x))} className="w-20 border rounded px-1.5 py-1 text-right" /></td>
                                  <td className="py-1 text-right font-medium pr-1">{fmt(ln.output_qty * ln.rate)}</td>
                                  <td className="py-1"><button onClick={() => setEditJWOLines(l => l.filter((_, j) => j !== i))} className="text-red-400 px-1">✕</button></td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {editJWOLines.length > 0 && <p className="text-xs font-semibold text-right text-gray-700 mt-1 pr-6">Total: {fmt(editJWOLines.reduce((s, l) => s + l.output_qty * l.rate, 0))}</p>}
                        </div>
                        <div className="flex gap-2 pt-1">
                          <button onClick={() => updateJWOMut.mutate({ id: jwo.id, data: { ...editJWOForm, lines: editJWOLines } })} disabled={updateJWOMut.isPending}
                            className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                            {updateJWOMut.isPending ? 'Saving…' : 'Save Changes'}
                          </button>
                          <button onClick={() => setEditingJWO(null)} className="px-4 py-2 border border-gray-200 rounded-lg text-sm text-gray-600">Cancel</button>
                        </div>
                      </div>
                    ) : (
                      /* ── JWO Read-only view ── */
                      <div className="space-y-2 pt-3">
                        {jwo.status === 'Draft' && (
                          <div className="flex justify-end">
                            <button onClick={() => {
                              const proc = processors.find(p => p.id === jwo.processor_id || p.processor_name === jwo.processor_name)
                              setEditingJWO(jwo.id)
                              setEditJWOForm({
                                processor_id: proc?.id, processor_name: jwo.processor_name,
                                expected_return_date: jwo.expected_return_date || '',
                                pr_reference: jwo.pr_reference || '', so_reference: jwo.so_reference || '',
                                issued_by: jwo.issued_by || '', remarks: jwo.remarks || ''
                              })
                              setEditJWOLines(jwo.lines.map(l => ({
                                input_material: l.input_material, input_qty: l.input_qty || 0,
                                output_material: l.output_material, output_qty: l.output_qty,
                                process_type: l.process_type, rate: l.rate
                              })))
                            }} className="text-xs px-3 py-1 border border-gray-300 rounded-lg text-gray-600 hover:bg-gray-50">✏️ Edit</button>
                          </div>
                        )}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs mb-2">
                          {[['SO', jwo.so_reference], ['PR Ref', jwo.pr_reference], ['Issued By', jwo.issued_by], ['Return Date', jwo.expected_return_date]].map(([label, val]) => (
                            <div key={label}><span className="text-gray-400 uppercase text-xs font-semibold">{label}</span><p className="text-gray-700 mt-0.5">{val || '—'}</p></div>
                          ))}
                        </div>
                        <table className="w-full text-xs">
                          <thead><tr className="text-gray-400 uppercase border-b">
                            <th className="text-left pb-1">Input Material</th><th className="text-left pb-1">Output Material</th>
                            <th className="text-left pb-1">Process</th><th className="text-right pb-1">Output Qty</th>
                            <th className="text-right pb-1">Rate</th><th className="text-right pb-1">Amount</th>
                          </tr></thead>
                          <tbody>{jwo.lines.map(l => (
                            <tr key={l.id} className="border-t border-gray-50">
                              <td className="py-1.5 font-medium">{l.input_material}</td>
                              <td className="py-1.5">{l.output_material}</td>
                              <td className="py-1.5 text-gray-500">{l.process_type}</td>
                              <td className="py-1.5 text-right">{l.output_qty}</td>
                              <td className="py-1.5 text-right">{fmt(l.rate)}</td>
                              <td className="py-1.5 text-right font-medium">{fmt(l.output_qty * l.rate)}</td>
                            </tr>
                          ))}</tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {jwos.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No job work orders found.</p>}
          </div>
        </div>
      )}

      {/* ── GRN ──────────────────────────────────────────────────────────────── */}
      {tab === 'grn' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center flex-wrap gap-2">
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm">
              <option value="">All</option>
              {['Draft','Verified','Posted'].map(s => <option key={s}>{s}</option>)}
            </select>
            <div className="flex items-center gap-2">
              {/* ── NEW: GRN Auto-fill widget ── */}
              <div className="flex items-center gap-2 bg-blue-50 border border-blue-200 rounded-lg px-3 py-1.5">
                <span className="text-xs text-blue-700 font-medium whitespace-nowrap">⚡ Auto-fill from:</span>
                <input value={grnAutoRef} onChange={e => setGrnAutoRef(e.target.value)} onKeyDown={e => e.key === 'Enter' && autoFillGRNFromRef()}
                  placeholder="PO-0001 or JWO-0001" className="border border-blue-200 rounded px-2 py-1 text-xs w-36 bg-white" />
                <button onClick={autoFillGRNFromRef} disabled={grnAutoLoading || !grnAutoRef.trim()}
                  className="px-2 py-1 bg-blue-600 text-white rounded text-xs font-medium hover:bg-blue-700 disabled:opacity-50 whitespace-nowrap">
                  {grnAutoLoading ? '…' : 'Load'}
                </button>
              </div>
              <button onClick={() => { setShowGRNForm(true); setGRNLines([]); setGRNForm({ grn_type: 'PO Receipt', reference_number: '', party_name: '', challan_no: '', invoice_no: '', vehicle_no: '', transporter: '', warehouse: '', remarks: '' }) }}
                className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ New GRN</button>
            </div>
          </div>

          {grnAutoError && <p className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">⚠️ {grnAutoError}</p>}

          {showGRNForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-gray-700">New GRN</h3>
                {grnLines.length > 0 && grnForm.reference_number && (
                  <span className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded-full font-medium">⚡ Auto-filled from {grnForm.reference_number}</span>
                )}
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div><label className="text-xs text-gray-500">GRN Type</label>
                  <select value={grnForm.grn_type} onChange={e => setGRNForm(f => ({ ...f, grn_type: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {GRN_TYPES.map(t => <option key={t}>{t}</option>)}
                  </select>
                </div>
                {[['reference_number','PO/JWO #'],['party_name','Party Name'],['challan_no','Challan No'],['invoice_no','Invoice No'],['vehicle_no','Vehicle No'],['transporter','Transporter'],['warehouse','Warehouse']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(grnForm as Record<string,string>)[k]} onChange={e => setGRNForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                ))}
              </div>
              <div>
                <div className="flex justify-between mb-2"><p className="text-sm font-medium text-gray-600">Material Lines</p>
                  <button onClick={() => setGRNLines(l => [...l, { material_code: '', material_name: '', received_qty: 0, accepted_qty: 0, rejected_qty: 0, unit: 'PCS', rate: 0, qc_status: 'Pending' }])}
                    className="text-xs text-blue-600 hover:underline">+ Add Line</button></div>
                {grnLines.map((ln, i) => (
                  <div key={i} className="grid grid-cols-8 gap-2 mb-2 text-sm items-center">
                    <input placeholder="Code" value={ln.material_code} onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, material_code: e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input placeholder="Name" value={ln.material_name} onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, material_name: e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Received" value={ln.received_qty}
                      onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, received_qty: +e.target.value, accepted_qty: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Accepted" value={ln.accepted_qty} onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, accepted_qty: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Rejected" value={ln.rejected_qty} onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, rejected_qty: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Rate ₹" value={ln.rate} onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, rate: +e.target.value } : x))} className="border rounded px-2 py-1.5" />
                    <select value={ln.qc_status} onChange={e => setGRNLines(l => l.map((x, j) => j === i ? { ...x, qc_status: e.target.value } : x))} className="border rounded px-2 py-1.5">
                      {['Pending','Pass','Fail'].map(s => <option key={s}>{s}</option>)}
                    </select>
                    <button onClick={() => setGRNLines(l => l.filter((_, j) => j !== i))} className="text-red-400">✕</button>
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <button onClick={() => createGRNMut.mutate({ ...grnForm, lines: grnLines.map(l => ({ ...l, amount: l.accepted_qty * l.rate })) })}
                  disabled={createGRNMut.isPending} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createGRNMut.isPending ? 'Saving…' : 'Create GRN'}
                </button>
                <button onClick={() => { setShowGRNForm(false); setGRNLines([]); setGrnAutoRef(''); setGrnAutoError('') }} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {grns.map(grn => (
              <div key={grn.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                <div className="flex items-center justify-between p-4 cursor-pointer" onClick={() => setExpanded(expanded === grn.id ? null : grn.id)}>
                  <div>
                    <p className="font-semibold text-sm text-gray-800">{grn.grn_number} <span className="text-xs font-normal text-gray-400">({grn.grn_type})</span></p>
                    <p className="text-xs text-gray-500">{grn.party_name} · {grn.grn_date}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-700">{fmt(grn.total_value)}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(grn.status)}`}>{grn.status}</span>
                    {/* ── Print button for GRN ── */}
                    <button onClick={e => { e.stopPropagation(); printDocument(buildGRNPrintHTML(grn), `GRN - ${grn.grn_number}`) }}
                      className="text-xs px-2 py-1 border border-gray-200 rounded text-gray-500 hover:bg-gray-50 flex items-center gap-1" title="Print GRN">
                      🖨️ Print
                    </button>
                    {grn.status === 'Draft' && (
                      <button onClick={e => { e.stopPropagation(); verifyGRNMut.mutate({ id: grn.id, status: 'Verified' }) }}
                        className="text-xs px-2 py-0.5 bg-green-50 text-green-700 rounded hover:bg-green-100">Verify</button>
                    )}
                    <span className="text-gray-400 text-xs">{expanded === grn.id ? '▲' : '▼'}</span>
                  </div>
                </div>
                {expanded === grn.id && (
                  <div className="border-t px-4 pb-3">
                    <table className="w-full text-xs mt-3">
                      <thead><tr className="text-gray-400 uppercase"><th className="text-left">Code</th><th className="text-left">Name</th><th className="text-right">Received</th><th className="text-right">Accepted</th><th className="text-right">Rejected</th><th className="text-right">QC</th></tr></thead>
                      <tbody>{grn.lines.map(l => (
                        <tr key={l.id} className="border-t border-gray-50">
                          <td className="py-1.5 font-medium">{l.material_code}</td>
                          <td className="py-1.5 text-gray-600">{l.material_name || '—'}</td>
                          <td className="py-1.5 text-right">{l.received_qty}</td>
                          <td className="py-1.5 text-right text-green-600">{l.accepted_qty}</td>
                          <td className="py-1.5 text-right text-red-500">{l.rejected_qty}</td>
                          <td className="py-1.5 text-right"><span className={`px-1.5 py-0.5 rounded text-xs ${l.qc_status === 'Pass' ? 'bg-green-100 text-green-700' : l.qc_status === 'Fail' ? 'bg-red-100 text-red-600' : 'bg-yellow-100 text-yellow-600'}`}>{l.qc_status}</span></td>
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                )}
              </div>
            ))}
            {grns.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No GRNs found.</p>}
          </div>
        </div>
      )}

      {/* ── Material Issue Notes ── */}
      {tab === 'min' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <p className="text-sm text-gray-500">Material Issue Notes — Grey fabric issue to processor</p>
            <button onClick={() => setShowMINForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ New MIN</button>
          </div>
          {showMINForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Material Issue Note</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {[['jwo_reference','JWO Reference'],['so_reference','SO Reference'],['from_location','From Location'],['to_location','To Location'],['to_vendor','To Vendor/Processor'],['issued_by','Issued By'],['remarks','Remarks']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(minForm as any)[k]} onChange={e => setMINForm(f => ({ ...f, [k]: e.target.value }))} className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" /></div>
                ))}
              </div>
              <div>
                <div className="flex justify-between mb-2"><p className="text-sm font-medium text-gray-600">Material Lines</p>
                  <button onClick={() => setMINLines(l => [...l, { material_code: '', material_name: '', material_type: 'GF', issue_qty: 0, unit: 'MTR', rate: 0 }])} className="text-xs text-blue-600 hover:underline">+ Add Line</button></div>
                {minLines.map((ln, i) => (
                  <div key={i} className="grid grid-cols-6 gap-2 mb-2 text-sm">
                    <input placeholder="Material Code" value={ln.material_code} onChange={e => setMINLines(l => l.map((x,j) => j===i ? {...x, material_code: e.target.value} : x))} className="border rounded px-2 py-1.5" />
                    <input placeholder="Material Name" value={ln.material_name} onChange={e => setMINLines(l => l.map((x,j) => j===i ? {...x, material_name: e.target.value} : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Issue Qty" value={ln.issue_qty} onChange={e => setMINLines(l => l.map((x,j) => j===i ? {...x, issue_qty: +e.target.value} : x))} className="border rounded px-2 py-1.5" />
                    <input placeholder="Unit" value={ln.unit} onChange={e => setMINLines(l => l.map((x,j) => j===i ? {...x, unit: e.target.value} : x))} className="border rounded px-2 py-1.5" />
                    <input type="number" placeholder="Rate ₹" value={ln.rate} onChange={e => setMINLines(l => l.map((x,j) => j===i ? {...x, rate: +e.target.value} : x))} className="border rounded px-2 py-1.5" />
                    <button onClick={() => setMINLines(l => l.filter((_,j) => j!==i))} className="text-red-400">✕</button>
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <button onClick={() => createMINMut.mutate({ ...minForm, lines: minLines })} disabled={createMINMut.isPending} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">{createMINMut.isPending ? 'Saving…' : 'Create MIN'}</button>
                <button onClick={() => setShowMINForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}
          <div className="space-y-2">
            {(mins as any[]).map((min: any) => (
              <div key={min.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                <div className="flex items-center justify-between p-4">
                  <div>
                    <p className="font-semibold text-sm text-gray-800">{min.min_number}</p>
                    <p className="text-xs text-gray-500">JWO: {min.jwo_reference || '—'} · To: {min.to_vendor || min.to_location || '—'} · {min.min_date}</p>
                    <p className="text-xs text-gray-400">{min.lines?.length || 0} items · SO: {min.so_reference || '—'}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusColor(min.status)}`}>{min.status}</span>
                    <button onClick={() => printDocument(buildMINPrintHTML(min), `MIN - ${min.min_number}`)} className="text-xs px-2 py-1 border border-gray-200 rounded text-gray-500 hover:bg-gray-50">🖨️ Print</button>
                  </div>
                </div>
              </div>
            ))}
            {mins.length === 0 && <p className="text-center text-gray-400 py-8 text-sm">No material issue notes found.</p>}
          </div>
        </div>
      )}
    </div>
  )
}