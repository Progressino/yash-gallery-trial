"""
Finance module router.
POST /api/finance/verify-pin       — verify Finance 2FA PIN
GET  /api/finance/pl               — P&L statement
GET  /api/finance/gst              — GST summary (Amazon MTR)
GET  /api/finance/platform-revenue — per-platform revenue reconciliation
GET  /api/finance/expenses         — list expenses
POST /api/finance/expenses         — add expense
DELETE /api/finance/expenses/{id}  — delete expense

Masters:
GET/POST/DELETE /api/finance/masters/ledger-groups
GET/POST/PUT/DELETE /api/finance/masters/ledgers
GET/POST/DELETE /api/finance/masters/gst-classifications
GET/POST/DELETE /api/finance/masters/tds-sections

Vouchers:
GET/POST /api/finance/vouchers
GET/DELETE /api/finance/vouchers/{id}

Sales Uploads:
GET/POST/DELETE /api/finance/sales-uploads
"""
import os
from typing import Optional, List, Any
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel

from ..db.finance_db import (
    add_expense, list_expenses, delete_expense,
    list_ledger_groups, create_ledger_group, delete_ledger_group,
    list_ledgers, create_ledger, update_ledger, delete_ledger,
    list_gst_classifications, create_gst_classification, delete_gst_classification,
    list_tds_sections, create_tds_section, delete_tds_section,
    list_expense_vouchers, get_expense_voucher, create_expense_voucher, delete_expense_voucher,
    list_finance_sales_uploads, create_finance_sales_upload, create_finance_sales_entries, delete_finance_sales_upload,
    list_voucher_types, create_voucher_type, update_voucher_type, delete_voucher_type,
    list_vouchers, list_sales_invoices, list_finance_inventory_movements, list_customer_ledger_entries, get_upload_summary_voucher, get_voucher_summary_by_date, get_gstr3b_data, get_ledger_balances, get_sales_entry_voucher,
    upsert_sales_invoice_edit_patch, get_sales_invoice_edit_patch,
    get_chart_of_accounts, get_trial_balance,
    list_tally_pl, upsert_tally_pl, delete_tally_pl,
)
from ..services.finance import (
    get_pl_statement,
    get_gst_summary,
    get_platform_revenue,
    get_platform_revenue_from_finance_uploads,
)

_FINANCE_PIN = os.environ.get("FINANCE_PIN", "").strip()
_GSTIN_COMPANY_MAP = {
    "08AABCY3804E1ZJ": ("Yash Gallery Pvt Ltd Rajasthan", "Rajasthan"),
    "06AABCY3804E1ZN": ("Yash Gallery Pvt Ltd Haryana", "Haryana"),
    "29AABCY3804E1ZF": ("Yash Gallery Pvt Ltd Karnataka", "Karnataka"),
    "27AABCY3804E1ZJ": ("Yash Gallery Pvt Ltd Maharashtra", "Maharashtra"),
    "33AABCY3804E1ZQ": ("Yash Gallery Pvt Ltd Tamil Nadu", "Tamil Nadu"),
    "36AABCY3804E1ZK": ("Yash Gallery Pvt Ltd Telangana", "Telangana"),
    "09AABCY3804E1ZH": ("Yash Gallery Pvt Ltd Uttar Pradesh", "Uttar Pradesh"),
    "19AABCY3804E1ZG": ("Yash Gallery Pvt Ltd West Bengal", "West Bengal"),
    "18AABCY3804E1ZI": ("Yash Gallery Pvt Ltd Assam", "Assam"),
    "10AABCY3804E1ZY": ("Yash Gallery Pvt Ltd Bihar", "Bihar"),
    "24AABCY3804E1ZP": ("Yash Gallery Pvt Ltd Gujarat", "Gujarat"),
    "23AABCY3804E1ZR": ("Yash Gallery Pvt Ltd Madhya Pradesh", "Madhya Pradesh"),
    "21AABCY3804E1ZV": ("Yash Gallery Pvt Ltd Odisha", "Odisha"),
    "03AABCY3804E1ZT": ("Yash Gallery Pvt Ltd Punjab", "Punjab"),
}


def _format_ship_location(city: str, state: str) -> str:
    """Exact ship-to display: city + state from the sales file when both exist."""
    c = (city or "").strip()
    s = (state or "").strip()
    if c.lower() in ("nan", "none", "-", ""):
        c = ""
    if s.lower() in ("nan", "none", "-", ""):
        s = ""
    if c and s:
        return f"{c}, {s}"
    return s or c or ""


router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────

class ExpenseCreate(BaseModel):
    date:        str
    category:    str
    description: str   = ""
    amount:      float
    gst_amount:  float = 0.0


class PinVerify(BaseModel):
    pin: str


class TallyPLUpsert(BaseModel):
    fy:                str
    opening_stock:     float = 0.0
    purchases:         float = 0.0
    direct_expenses:   float = 0.0
    indirect_expenses: float = 0.0
    sales:             float = 0.0
    closing_stock:     float = 0.0
    indirect_incomes:  float = 0.0
    notes:             str   = ""


class LedgerGroupCreate(BaseModel):
    name:         str
    parent_group: str = ''
    nature:       str = 'expense'


class LedgerCreate(BaseModel):
    name:                  str
    group_id:              Optional[int] = None
    group_name:            str = ''
    gstin:                 str = ''
    pan:                   str = ''
    state:                 str = ''
    state_code:            str = ''
    address:               str = ''
    tds_applicable:        int = 0
    tds_section:           str = ''
    alias:                 str = ''
    credit_period:         int = 0
    maintain_bill_by_bill: int = 0
    is_tcs_applicable:     int = 0
    country:               str = 'India'
    pincode:               str = ''
    registration_type:     str = ''
    bank_name:             str = ''
    bank_account:          str = ''
    bank_ifsc:             str = ''
    opening_balance:       float = 0.0


class LedgerUpdate(BaseModel):
    name:                  Optional[str]   = None
    group_id:              Optional[int]   = None
    group_name:            Optional[str]   = None
    gstin:                 Optional[str]   = None
    pan:                   Optional[str]   = None
    state:                 Optional[str]   = None
    state_code:            Optional[str]   = None
    address:               Optional[str]   = None
    tds_applicable:        Optional[int]   = None
    tds_section:           Optional[str]   = None
    is_active:             Optional[int]   = None
    alias:                 Optional[str]   = None
    credit_period:         Optional[int]   = None
    maintain_bill_by_bill: Optional[int]   = None
    is_tcs_applicable:     Optional[int]   = None
    country:               Optional[str]   = None
    pincode:               Optional[str]   = None
    registration_type:     Optional[str]   = None
    bank_name:             Optional[str]   = None
    bank_account:          Optional[str]   = None
    bank_ifsc:             Optional[str]   = None
    opening_balance:       Optional[float] = None


class VoucherTypeCreate(BaseModel):
    name:             str
    voucher_category: str = 'Sales'
    abbreviation:     str = ''
    allow_narration:  int = 1
    numbering_method: str = 'Auto'


class VoucherTypeUpdate(BaseModel):
    name:             Optional[str] = None
    voucher_category: Optional[str] = None
    abbreviation:     Optional[str] = None
    is_active:        Optional[int] = None
    allow_narration:  Optional[int] = None
    numbering_method: Optional[str] = None


class GSTClassificationCreate(BaseModel):
    name:     str
    hsn_sac:  str  = ''
    gst_rate: float = 18.0
    type:     str  = 'Goods'


class TDSSectionCreate(BaseModel):
    section:         str
    description:     str   = ''
    rate_individual: float = 1.0
    rate_company:    float = 2.0
    threshold:       float = 0.0


class VoucherLineIn(BaseModel):
    expense_head: str
    description:  str   = ''
    amount:       float = 0
    cost_centre:  str   = ''
    is_debit:     int   = 1   # 1=Dr, 0=Cr (for Journal)


class VoucherCreate(BaseModel):
    voucher_date:   Optional[str] = None
    voucher_type:   str   = 'Expense'
    party_name:     str   = ''
    party_gstin:    str   = ''
    party_state:    str   = ''
    bill_no:        str   = ''
    bill_date:      str   = ''
    supply_type:    str   = 'Intra'
    narration:      str   = ''
    taxable_amount: float = 0
    cgst_amount:    float = 0
    sgst_amount:    float = 0
    igst_amount:    float = 0
    tds_section:    str   = ''
    tds_rate:       float = 0
    tds_amount:     float = 0
    total_amount:   float = 0
    net_payable:    float = 0
    payment_mode:   str   = ''   # Cash/Cheque/NEFT/RTGS/IMPS
    bank_ledger:    str   = ''
    cheque_no:      str   = ''
    ref_number:     str   = ''   # UTR/transaction ref
    lines:          List[VoucherLineIn] = []


class SalesUploadCreate(BaseModel):
    platform:      str
    period:        str
    filename:      str   = ''
    total_revenue: float = 0
    total_orders:  int   = 0
    total_returns: float = 0
    net_revenue:   float = 0
    uploaded_by:   str   = ''
    upload_notes:  str   = ''


# ── Helpers ───────────────────────────────────────────────────────

def _sess(request: Request):
    sess = request.state.session
    if sess is None:
        raise HTTPException(status_code=403, detail="No session")
    return sess


# ── Auth ──────────────────────────────────────────────────────────

@router.get("/pin-required")
def pin_required():
    """Check whether Finance 2FA PIN is configured."""
    return {"required": bool(_FINANCE_PIN)}


@router.post("/verify-pin")
def verify_pin(body: PinVerify):
    """Verify the Finance 2FA PIN set via FINANCE_PIN env var."""
    if not _FINANCE_PIN:
        return {"ok": True, "message": "no_pin_set"}
    if body.pin == _FINANCE_PIN:
        return {"ok": True}
    return {"ok": False, "message": "Incorrect PIN"}


# ── Analytics endpoints ───────────────────────────────────────────

@router.get("/pl")
def pl_statement(
    request:    Request,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    revenue_source: str = Query(
        "finance_lock",
        description="finance_lock = Sales Uploads tab (finance.db) only; session = Upload page / Dashboard basis",
        pattern="^(session|finance_lock)$",
    ),
    finance_company: Optional[str] = Query(
        None,
        description="Optional company name or GSTIN filter when revenue_source=finance_lock",
    ),
):
    sess = _sess(request)
    return get_pl_statement(
        mtr_df      = sess.mtr_df,
        myntra_df   = sess.myntra_df,
        meesho_df   = sess.meesho_df,
        flipkart_df = sess.flipkart_df,
        sales_df    = sess.sales_df,
        cogs_df     = sess.cogs_df,
        start_date  = start_date,
        end_date    = end_date,
        revenue_source=revenue_source,
        finance_company=finance_company,
    )


@router.get("/gst")
def gst_summary(
    request:    Request,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
):
    sess = _sess(request)
    return get_gst_summary(sess.mtr_df, start_date=start_date, end_date=end_date)


@router.get("/platform-revenue")
def platform_revenue(
    request:    Request,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    revenue_source: str = Query(
        "finance_lock",
        pattern="^(session|finance_lock)$",
    ),
    finance_company: Optional[str] = Query(None),
):
    sess = _sess(request)
    if revenue_source == "finance_lock":
        return get_platform_revenue_from_finance_uploads(start_date, end_date, company=finance_company)
    return get_platform_revenue(
        mtr_df      = sess.mtr_df,
        myntra_df   = sess.myntra_df,
        meesho_df   = sess.meesho_df,
        flipkart_df = sess.flipkart_df,
        start_date  = start_date,
        end_date    = end_date,
    )


# ── Tally / Accountant P&L ────────────────────────────────────────

@router.get("/tally-pl")
def get_tally_pl():
    rows = list_tally_pl()
    # Compute derived fields for each row
    for r in rows:
        cogs = r["purchases"] + r["direct_expenses"] + r["opening_stock"] - r["closing_stock"]
        gross_profit = r["sales"] + r["indirect_incomes"] - cogs
        net_profit   = gross_profit - r["indirect_expenses"]
        r["cogs"]         = round(cogs, 2)
        r["gross_profit"] = round(gross_profit, 2)
        r["net_profit"]   = round(net_profit, 2)
    return rows


@router.post("/tally-pl")
def save_tally_pl(body: TallyPLUpsert):
    return upsert_tally_pl(
        fy                = body.fy,
        opening_stock     = body.opening_stock,
        purchases         = body.purchases,
        direct_expenses   = body.direct_expenses,
        indirect_expenses = body.indirect_expenses,
        sales             = body.sales,
        closing_stock     = body.closing_stock,
        indirect_incomes  = body.indirect_incomes,
        notes             = body.notes,
    )


@router.delete("/tally-pl/{fy}")
def remove_tally_pl(fy: str):
    if not delete_tally_pl(fy):
        raise HTTPException(status_code=404, detail="FY not found")
    return {"ok": True}


# ── Expenses (original) ───────────────────────────────────────────

@router.get("/expenses")
def get_expenses(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
):
    return list_expenses(start_date=start_date, end_date=end_date)


@router.post("/expenses")
def create_expense(body: ExpenseCreate):
    new_id = add_expense(
        date        = body.date,
        category    = body.category,
        description = body.description,
        amount      = body.amount,
        gst_amount  = body.gst_amount,
    )
    return {"ok": True, "id": new_id}


@router.delete("/expenses/{expense_id}")
def remove_expense(expense_id: int):
    deleted = delete_expense(expense_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Expense not found")
    return {"ok": True}


# ── Masters: Ledger Groups ────────────────────────────────────────

@router.get("/masters/ledger-groups")
def get_ledger_groups():
    return list_ledger_groups()


@router.post("/masters/ledger-groups")
def post_ledger_group(body: LedgerGroupCreate):
    new_id = create_ledger_group(
        name         = body.name,
        parent_group = body.parent_group,
        nature       = body.nature,
    )
    return {"ok": True, "id": new_id}


@router.delete("/masters/ledger-groups/{group_id}")
def del_ledger_group(group_id: int):
    deleted = delete_ledger_group(group_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Ledger group not found")
    return {"ok": True}


# ── Masters: Ledgers ──────────────────────────────────────────────

@router.get("/masters/ledgers")
def get_ledgers(group_id: Optional[int] = None, search: Optional[str] = None):
    return list_ledgers(group_id=group_id, search=search)


@router.post("/masters/ledgers")
def post_ledger(body: LedgerCreate):
    new_id = create_ledger(
        name                  = body.name,
        group_id              = body.group_id,
        group_name            = body.group_name,
        gstin                 = body.gstin,
        pan                   = body.pan,
        state                 = body.state,
        state_code            = body.state_code,
        address               = body.address,
        tds_applicable        = body.tds_applicable,
        tds_section           = body.tds_section,
        alias                 = body.alias,
        credit_period         = body.credit_period,
        maintain_bill_by_bill = body.maintain_bill_by_bill,
        is_tcs_applicable     = body.is_tcs_applicable,
        country               = body.country,
        pincode               = body.pincode,
        registration_type     = body.registration_type,
        bank_name             = body.bank_name,
        bank_account          = body.bank_account,
        bank_ifsc             = body.bank_ifsc,
        opening_balance       = body.opening_balance,
    )
    return {"ok": True, "id": new_id}


@router.put("/masters/ledgers/{ledger_id}")
def put_ledger(ledger_id: int, body: LedgerUpdate):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    updated = update_ledger(ledger_id, **fields)
    if not updated:
        raise HTTPException(status_code=404, detail="Ledger not found")
    return {"ok": True}


@router.delete("/masters/ledgers/{ledger_id}")
def del_ledger(ledger_id: int):
    deleted = delete_ledger(ledger_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Ledger not found")
    return {"ok": True}


# ── Masters: GST Classifications ──────────────────────────────────

@router.get("/masters/gst-classifications")
def get_gst_classifications():
    return list_gst_classifications()


@router.post("/masters/gst-classifications")
def post_gst_classification(body: GSTClassificationCreate):
    new_id = create_gst_classification(
        name     = body.name,
        hsn_sac  = body.hsn_sac,
        gst_rate = body.gst_rate,
        type_    = body.type,
    )
    return {"ok": True, "id": new_id}


@router.delete("/masters/gst-classifications/{classification_id}")
def del_gst_classification(classification_id: int):
    deleted = delete_gst_classification(classification_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="GST classification not found")
    return {"ok": True}


# ── Masters: TDS Sections ─────────────────────────────────────────

@router.get("/masters/tds-sections")
def get_tds_sections():
    return list_tds_sections()


@router.post("/masters/tds-sections")
def post_tds_section(body: TDSSectionCreate):
    new_id = create_tds_section(
        section         = body.section,
        description     = body.description,
        rate_individual = body.rate_individual,
        rate_company    = body.rate_company,
        threshold       = body.threshold,
    )
    return {"ok": True, "id": new_id}


@router.delete("/masters/tds-sections/{section_id}")
def del_tds_section(section_id: int):
    deleted = delete_tds_section(section_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="TDS section not found")
    return {"ok": True}


# ── Masters: Voucher Types ────────────────────────────────────────

@router.get("/masters/voucher-types")
def get_voucher_types(category: Optional[str] = None):
    return list_voucher_types(category=category)


@router.post("/masters/voucher-types")
def post_voucher_type(body: VoucherTypeCreate):
    new_id = create_voucher_type(
        name             = body.name,
        voucher_category = body.voucher_category,
        abbreviation     = body.abbreviation,
        allow_narration  = body.allow_narration,
        numbering_method = body.numbering_method,
    )
    return {"ok": True, "id": new_id}


@router.put("/masters/voucher-types/{vt_id}")
def put_voucher_type(vt_id: int, body: VoucherTypeUpdate):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    updated = update_voucher_type(vt_id, **fields)
    if not updated:
        raise HTTPException(status_code=404, detail="Voucher type not found")
    return {"ok": True}


@router.delete("/masters/voucher-types/{vt_id}")
def del_voucher_type(vt_id: int):
    deleted = delete_voucher_type(vt_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Voucher type not found")
    return {"ok": True}


# ── Expense Vouchers ──────────────────────────────────────────────

@router.get("/vouchers")
def get_vouchers(
    start_date:   Optional[str] = None,
    end_date:     Optional[str] = None,
    voucher_type: Optional[str] = None,
):
    return list_expense_vouchers(
        start_date   = start_date,
        end_date     = end_date,
        voucher_type = voucher_type,
    )


@router.get("/sales-invoices")
def get_sales_invoices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
    document_kind: Optional[str] = Query(
        None,
        description='Omit or "all" = invoices + credit memos + upload summaries. '
        '"sales" = posted invoices/shipments only (no credit memos) + SUP summaries. '
        '"credit_memo" = sales credit memos / returns (SUE only).',
    ),
    include_upload_summaries: bool = Query(
        True,
        description="When false, omit SUP-* upload summary rows. Parsed SUE-* lines remain.",
    ),
):
    """Invoice-level rows persisted from Finance Sales Uploads."""
    return list_sales_invoices(
        start_date=start_date,
        end_date=end_date,
        search=search,
        document_kind=document_kind,
        include_upload_summaries=include_upload_summaries,
    )


@router.get("/inventory-movements")
def get_finance_inventory_movements(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
):
    """SKU outbound (shipments) vs inbound (returns) from posted Finance sales line_items."""
    return list_finance_inventory_movements(
        start_date=start_date,
        end_date=end_date,
        search=search,
    )


@router.get("/customer-ledger-entries")
def get_customer_ledger_entries(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
    document_kind: Optional[str] = Query(
        None,
        description='"all" (default) = invoices + credit memos. "sales" = invoices only. '
        '"credit_memo" = credit memos / returns only. Posted finance_sales_entries only.',
    ),
):
    """BC / D365-style customer ledger lines from posted sales upload entries (for reconciliation exports)."""
    return list_customer_ledger_entries(
        start_date=start_date,
        end_date=end_date,
        search=search,
        document_kind=document_kind,
    )


class SalesInvoicePatch(BaseModel):
    """Partial update for sales upload / entry voucher display (stored as JSON overlay)."""
    invoice_no: Optional[str] = None
    voucher_date: Optional[str] = None
    bill_date: Optional[str] = None
    party_name: Optional[str] = None
    party_gstin: Optional[str] = None
    party_state: Optional[str] = None
    ship_to_state: Optional[str] = None
    order_id: Optional[str] = None
    source_filename: Optional[str] = None
    narration: Optional[str] = None
    supply_type: Optional[str] = None
    platform: Optional[str] = None
    period: Optional[str] = None
    taxable_amount: Optional[float] = None
    cgst_amount: Optional[float] = None
    sgst_amount: Optional[float] = None
    igst_amount: Optional[float] = None
    total_amount: Optional[float] = None
    net_payable: Optional[float] = None
    # BC / D365-style default dimensions on the document (stored in sales_invoice_edits JSON).
    dimension_assignments: Optional[List[dict[str, Any]]] = None


@router.patch("/sales-invoices/{voucher_id}")
def patch_sales_invoice(voucher_id: int, body: SalesInvoicePatch):
    """Persist editable header/amount fields for a sales invoice row (SUP-* or SUE-* synthetic ids)."""
    v = get_sales_entry_voucher(voucher_id) or get_upload_summary_voucher(voucher_id)
    if not v:
        raise HTTPException(status_code=404, detail="Sales invoice not found")
    raw = body.model_dump(exclude_unset=True)
    patch = {}
    for k, val in raw.items():
        if val is None:
            continue
        patch[k] = val
    if not patch:
        return {"ok": True, "patch": get_sales_invoice_edit_patch(voucher_id)}
    merged = upsert_sales_invoice_edit_patch(voucher_id, patch)
    return {"ok": True, "patch": merged}


@router.post("/vouchers")
def post_voucher(body: VoucherCreate):
    data = body.model_dump()
    data['lines'] = [ln.model_dump() for ln in body.lines]
    voucher_no = create_expense_voucher(data)
    return {"ok": True, "voucher_no": voucher_no}


@router.get("/vouchers/{voucher_id}")
def get_voucher(voucher_id: int):
    v = get_sales_entry_voucher(voucher_id)
    if v:
        return v
    v = get_upload_summary_voucher(voucher_id)
    if v:
        return v
    v = get_expense_voucher(voucher_id)
    if not v:
        raise HTTPException(status_code=404, detail="Voucher not found")
    return v


@router.delete("/vouchers/{voucher_id}")
def del_voucher(voucher_id: int):
    deleted = delete_expense_voucher(voucher_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Voucher not found")
    return {"ok": True}


# ── Extra Voucher Endpoints ───────────────────────────────────────

@router.get("/vouchers/types")
def get_voucher_types_in_db():
    """Return distinct voucher_type values present in expense_vouchers table."""
    from ..db.finance_db import _connect
    conn = _connect()
    rows = conn.execute(
        "SELECT DISTINCT voucher_type FROM expense_vouchers ORDER BY voucher_type"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


@router.get("/daybook")
def get_daybook(date: Optional[str] = None):
    """Return all vouchers for a specific date."""
    import datetime
    use_date = date or datetime.date.today().isoformat()
    return get_voucher_summary_by_date(use_date)


@router.get("/gstr3b")
def get_gstr3b(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
):
    """Return GSTR3B computed summary."""
    import datetime
    today = datetime.date.today().isoformat()
    first_of_month = datetime.date.today().replace(day=1).isoformat()
    sd = start_date or first_of_month
    ed = end_date   or today
    return get_gstr3b_data(sd, ed)


@router.get("/ledger-balances")
def ledger_balances():
    """Sum of payments/receipts per ledger."""
    return get_ledger_balances()


@router.get("/chart-of-accounts")
def chart_of_accounts():
    """Return hierarchical Chart of Accounts tree."""
    return get_chart_of_accounts()


@router.get("/trial-balance")
def trial_balance(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
):
    """Return trial balance with Dr/Cr per ledger for a given date range."""
    return get_trial_balance(start_date=start_date, end_date=end_date)


# ── Finance Sales Uploads ─────────────────────────────────────────

@router.get("/sales-uploads")
def get_sales_uploads(
    platform: Optional[str] = None,
    period:   Optional[str] = None,
    company:  Optional[str] = None,
):
    return list_finance_sales_uploads(platform=platform, period=period, company=company)


@router.post("/sales-uploads")
def post_sales_upload(body: SalesUploadCreate):
    new_id = create_finance_sales_upload(body.model_dump())
    return {"ok": True, "id": new_id}


@router.delete("/sales-uploads/{upload_id}")
def del_sales_upload(upload_id: int):
    deleted = delete_finance_sales_upload(upload_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Sales upload not found")
    return {"ok": True}


# ── Finance-specific parsers (revenue/returns without SKU) ────────

def _parse_meesho_gst_for_finance(zip_bytes: bytes) -> dict:
    """Parse Meesho gst_* ZIP → revenue/returns/tax for Finance."""
    import io, zipfile
    import pandas as pd
    result = {"revenue": 0.0, "returns": 0.0, "tax": 0.0, "orders": 0}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for fname in zf.namelist():
                if not fname.lower().endswith(".xlsx"):
                    continue
                df = pd.read_excel(io.BytesIO(zf.read(fname)))
                if "total_invoice_value" not in df.columns:
                    continue
                val = float(df["total_invoice_value"].sum())
                tax = float(df["tax_amount"].sum()) if "tax_amount" in df.columns else 0.0
                if "return" in fname.lower():
                    result["returns"] += val
                else:
                    result["revenue"] += val
                    result["orders"]  += len(df)
                    result["tax"]     += tax
    except Exception:
        pass
    return result


def _parse_snapdeal_settlement_for_finance(file_bytes: bytes) -> dict:
    """Parse Snapdeal settlement XLSX → revenue/returns for Finance."""
    import io
    import pandas as pd
    result = {"revenue": 0.0, "returns": 0.0, "orders": 0}
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        if "Total_Suboders" in xl.sheet_names:
            df = xl.parse("Total_Suboders", dtype=str)
            if "Invoice Amount" in df.columns and "Transaction Type" in df.columns:
                df["_amt"] = pd.to_numeric(df["Invoice Amount"], errors="coerce").fillna(0)
                txn = df["Transaction Type"].fillna("").str.lower()
                result["revenue"] = float(df[txn.str.contains("invoice")]["_amt"].sum())
                result["orders"]  = int(txn.str.contains("invoice").sum())
        if "Returns" in xl.sheet_names:
            df_r = xl.parse("Returns", dtype=str)
            if "Invoice Amount" in df_r.columns:
                df_r["_amt"] = pd.to_numeric(df_r["Invoice Amount"], errors="coerce").fillna(0)
                result["returns"] = float(df_r["_amt"].abs().sum())
    except Exception:
        pass
    return result


def _looks_like_monthly_sales_package(zip_bytes: bytes) -> bool:
    """Detect outer monthly package ZIP by platform-folder signatures."""
    import io, zipfile
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = [n.lower() for n in zf.namelist() if n and not n.endswith("/")]
    except Exception:
        return False
    hits = 0
    for key in ("amazon/", "myntra/", "meesho/", "flipkart/", "snapdeal/"):
        if any(key in n for n in names):
            hits += 1
    # Treat as package when at least two platform folder signatures are present.
    return hits >= 2


def _process_monthly_package_bytes(
    raw: bytes,
    filename: str,
    period: str,
    uploaded_by: str,
    notes: str,
    *,
    dry_run: bool,
) -> dict:
    """Parse full monthly package ZIP and optionally persist Finance rows."""
    import io, zipfile
    import pandas as pd
    from ..services.mtr import load_mtr_from_zip
    from ..services.myntra import load_myntra_from_zip
    from ..services.meesho import load_meesho_from_zip

    results: list = []
    skipped: list = []

    try:
        outer_zf = zipfile.ZipFile(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot open ZIP: {e}")

    names = outer_zf.namelist()

    amazon_files: dict = {}   # basename → bytes
    myntra_csvs: dict = {}    # sub-account → [(fname, bytes)]
    meesho_zips: list = []
    flipkart_zips: list = []
    snapdeal_files: list = []

    for name in names:
        if name.endswith("/"):
            continue
        parts = [p for p in name.split("/") if p]
        basename = parts[-1]
        lower = basename.lower()
        path_lc = name.lower()

        if "amazon" in path_lc and lower.endswith(".zip"):
            amazon_files[basename] = outer_zf.read(name)
        elif "myntra" in path_lc and lower.endswith(".csv"):
            sub = parts[-2] if len(parts) >= 2 else "Myntra"
            if "tp" in sub.lower():
                skipped.append(f"{name} — Myntra TP is duplicate of PPMP, skipped")
                continue
            myntra_csvs.setdefault(sub, []).append((basename, outer_zf.read(name)))
        elif "meesho" in path_lc and lower.endswith(".zip"):
            meesho_zips.append((basename, outer_zf.read(name)))
        elif "flipkart" in path_lc and lower.endswith(".zip"):
            flipkart_zips.append((basename, outer_zf.read(name)))
        elif "snapdeal" in path_lc and lower.endswith((".xlsx", ".xls", ".csv", ".zip")):
            snapdeal_files.append((basename, outer_zf.read(name)))

    outer_zf.close()

    def _record(platform, rev, ret, orders, fname, note=""):
        entry = {
            "platform": platform,
            "orders": orders,
            "revenue": round(rev, 2),
            "returns": round(ret, 2),
            "net_revenue": round(rev - ret, 2),
            "filename": fname,
        }
        if note:
            entry["note"] = note
        if not dry_run:
            nid = create_finance_sales_upload({
                "platform": platform,
                "period": period,
                "filename": fname,
                "total_revenue": round(rev, 2),
                "total_orders": orders,
                "total_returns": round(ret, 2),
                "net_revenue": round(rev - ret, 2),
                "uploaded_by": uploaded_by,
                "upload_notes": notes,
            })
            entry["id"] = nid
        return entry

    # Amazon — MTR uses Transaction_Type (underscore)
    amz_rev = amz_ret = amz_rows = 0
    amazon_dfs: list = []
    amz_result_index: Optional[int] = None
    for fname, fbytes in amazon_files.items():
        if any(x in fname.lower() for x in ("stock", "transfer")):
            continue
        try:
            df, _, _ = load_mtr_from_zip(fbytes)
            if df.empty:
                continue
            amazon_dfs.append(df)
            amz_rows += len(df)
            if "Invoice_Amount" in df.columns and "Transaction_Type" in df.columns:
                amz_rev += float(df[df["Transaction_Type"] == "Shipment"]["Invoice_Amount"].sum())
                # Amazon parser emits Refund; some legacy files may emit Return.
                ret_mask = df["Transaction_Type"].isin(["Return", "Refund"])
                amz_ret += float(df.loc[ret_mask, "Invoice_Amount"].abs().sum())
            elif "Invoice_Amount" in df.columns:
                amz_rev += float(df["Invoice_Amount"].sum())
        except Exception as e:
            skipped.append(f"Amazon {fname}: {e}")
    if amz_rows:
        amz_result_index = len(results)
        results.append(_record("Amazon", amz_rev, amz_ret, amz_rows, filename or "monthly"))

    # Myntra per sub-account — uses TxnType
    for sub, csv_list in myntra_csvs.items():
        try:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as mzf:
                for fn, fb in csv_list:
                    mzf.writestr(fn, fb)
            df, _, _ = load_myntra_from_zip(buf.getvalue(), {})
            if df.empty:
                skipped.append(f"Myntra {sub}: no data")
                continue
            rev = float(df[df["TxnType"] == "Shipment"]["Invoice_Amount"].sum()) if "TxnType" in df.columns else 0.0
            ret = float(df[df["TxnType"] == "Refund"]["Invoice_Amount"].sum()) if "TxnType" in df.columns else 0.0
            results.append(_record(f"Myntra ({sub})", rev, ret, len(df), filename or "monthly"))
        except Exception as e:
            skipped.append(f"Myntra {sub}: {e}")

    # Meesho
    for fname, fbytes in meesho_zips:
        try:
            if fname.lower().startswith("gst_"):
                s = _parse_meesho_gst_for_finance(fbytes)
                if s["orders"] == 0 and s["revenue"] == 0:
                    skipped.append(f"Meesho {fname}: empty gst file")
                    continue
                results.append(_record("Meesho", s["revenue"], s["returns"], s["orders"], fname))
            else:
                df, _, _ = load_meesho_from_zip(fbytes)
                if df.empty:
                    skipped.append(f"Meesho {fname}: no data")
                    continue
                txn_col = next((c for c in df.columns if "transaction" in c.lower() or "txn" in c.lower()), None)
                if txn_col and "Invoice_Amount" in df.columns:
                    rev = float(df[df[txn_col] == "Shipment"]["Invoice_Amount"].sum())
                elif "Invoice_Amount" in df.columns:
                    rev = float(df["Invoice_Amount"].sum())
                else:
                    rev = 0.0
                results.append(_record("Meesho", rev, 0.0, len(df), fname))
        except Exception as e:
            skipped.append(f"Meesho {fname}: {e}")

    # Flipkart — invoice PDFs only
    for fname, fbytes in flipkart_zips:
        try:
            with zipfile.ZipFile(io.BytesIO(fbytes)) as fzf:
                pdf_count = sum(1 for n in fzf.namelist() if n.lower().endswith(".pdf"))
            if pdf_count:
                results.append(
                    _record(
                        "Flipkart", 0.0, 0.0, pdf_count, fname,
                        note=f"{pdf_count} invoice PDFs — revenue not parseable from PDFs",
                    )
                )
            else:
                skipped.append(f"Flipkart {fname}: no parseable data")
        except Exception as e:
            skipped.append(f"Flipkart {fname}: {e}")

    # Snapdeal settlement
    for fname, fbytes in snapdeal_files:
        try:
            s = _parse_snapdeal_settlement_for_finance(fbytes)
            if s["revenue"] == 0 and s["returns"] == 0:
                skipped.append(f"Snapdeal {fname}: no revenue found")
                continue
            results.append(_record("Snapdeal", s["revenue"], s["returns"], s["orders"], fname))
        except Exception as e:
            skipped.append(f"Snapdeal {fname}: {e}")

    if not dry_run and amz_result_index is not None and amazon_dfs:
        combined_amz = pd.concat(amazon_dfs, ignore_index=True)
        entry = results[amz_result_index]
        upload_id = entry.get("id")
        if upload_id and combined_amz is not None and not combined_amz.empty:
            n = _persist_amazon_finance_sales_entries(
                combined_amz,
                sales_upload_id=int(upload_id),
                platform="Amazon",
                period=period,
                source_filename=filename or "monthly",
                company_name="",
                seller_gstin="",
                company_state="",
            )
            entry["sales_entry_rows"] = n

    if not results and not skipped:
        raise HTTPException(status_code=422, detail="No recognisable platform folders found in ZIP.")

    total_rev = sum(r.get("revenue", 0) for r in results)
    total_ret = sum(r.get("returns", 0) for r in results)
    payload = {
        "ok": True,
        "period": period,
        "total_revenue": round(total_rev, 2),
        "total_returns": round(total_ret, 2),
        "net_revenue": round(total_rev - total_ret, 2),
        "skipped": skipped,
    }
    if dry_run:
        payload["preview"] = True
        payload["platforms"] = results
    else:
        payload["saved"] = results
    return payload


def _build_geo_company_breakdowns(df, ship_col: str, amt_col: str, ship_val: str, return_values: set[str]) -> tuple[list[dict], list[dict]]:
    """Build state and company/GSTIN revenue breakdowns for upload diagnostics."""
    import pandas as pd
    if df is None or df.empty or ship_col not in df.columns or amt_col not in df.columns:
        return [], []

    work = df.copy()
    work["_amt"] = pd.to_numeric(work[amt_col], errors="coerce").fillna(0.0)
    work["_is_ship"] = work[ship_col] == ship_val
    work["_is_ret"] = work[ship_col].isin(list(return_values))

    # State-wise (where sale happened)
    state_col = "Ship_To_State" if "Ship_To_State" in work.columns else None
    state_rows: list[dict] = []
    if state_col:
        st = work.copy()
        st["_state"] = st[state_col].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown", "NaN": "Unknown"})
        grp = st.groupby("_state", dropna=False).agg(
            gross_revenue=("_amt", lambda s: float(s[st.loc[s.index, "_is_ship"]].sum())),
            returns=("_amt", lambda s: float(s[st.loc[s.index, "_is_ret"]].abs().sum())),
            orders=("_is_ship", "sum"),
        ).reset_index()
        for _, r in grp.sort_values("gross_revenue", ascending=False).iterrows():
            gross = float(r["gross_revenue"])
            ret = float(r["returns"])
            state_rows.append({
                "state": str(r["_state"]),
                "orders": int(r["orders"]),
                "gross_revenue": round(gross, 2),
                "returns": round(ret, 2),
                "net_revenue": round(gross - ret, 2),
            })

    # Company/GSTIN wise (who sold it)
    has_gstin = "Seller_GSTIN" in work.columns
    has_company = "Seller_Company" in work.columns
    company_rows: list[dict] = []
    if has_gstin or has_company:
        cp = work.copy()
        cp["_gstin"] = cp["Seller_GSTIN"].astype(str).str.strip().str.upper() if has_gstin else "UNKNOWN"
        cp["_company"] = cp["Seller_Company"].astype(str).str.strip() if has_company else ""
        cp.loc[cp["_gstin"].isin(["", "NAN", "NONE"]), "_gstin"] = "UNKNOWN"
        cp.loc[cp["_company"].isin(["", "nan", "NaN", "NONE"]), "_company"] = ""
        grp = cp.groupby(["_gstin", "_company"], dropna=False).agg(
            gross_revenue=("_amt", lambda s: float(s[cp.loc[s.index, "_is_ship"]].sum())),
            returns=("_amt", lambda s: float(s[cp.loc[s.index, "_is_ret"]].abs().sum())),
            orders=("_is_ship", "sum"),
        ).reset_index()
        for _, r in grp.sort_values("gross_revenue", ascending=False).iterrows():
            gross = float(r["gross_revenue"])
            ret = float(r["returns"])
            gstin = str(r["_gstin"])
            mapped = _GSTIN_COMPANY_MAP.get(gstin)
            mapped_name = mapped[0] if mapped else ""
            mapped_state = mapped[1] if mapped else ""
            company_name = str(r["_company"]).strip() if str(r["_company"]).strip() else mapped_name
            if not company_name:
                company_name = gstin if gstin != "UNKNOWN" else "Unknown"
            company_rows.append({
                "company": company_name,
                "seller_gstin": gstin,
                "company_state": mapped_state,
                "orders": int(r["orders"]),
                "gross_revenue": round(gross, 2),
                "returns": round(ret, 2),
                "net_revenue": round(gross - ret, 2),
            })

    return state_rows, company_rows


def _reconcile_amazon_invoice_taxable_inclusive(
    inv_sum: float,
    tex_sum: float,
    cgst: float,
    sgst: float,
    igst: float,
    tax_from_tt: float,
    gst_rate_median: Optional[float],
    *,
    gst_rate_from_column: bool = False,
) -> tuple[float, float, float, float, float, bool]:
    """Return (taxable, cgst, sgst, igst, gst_total, inclusive_reconciled).

    Amazon MTR ``Invoice_Amount`` is often tax-inclusive while ``IGST``/``CGST``/``SGST``
    are the tax component of that same total. Summing ``Invoice_Amount + GST`` double-counts.
    When ``Tax_Exclusive_Gross`` is missing (``tex_sum`` ≈ 0) and we have a GST rate,
    back out GST-exclusive taxable as ``inv / (1 + r)`` and scale tax components to match.
    """
    if abs(tex_sum) > 1e-4:
        if cgst == 0.0 and sgst == 0.0 and igst == 0.0:
            gst_t = tax_from_tt
        else:
            gst_t = cgst + sgst + igst
        return tex_sum, cgst, sgst, igst, gst_t, False

    if cgst == 0.0 and sgst == 0.0 and igst == 0.0:
        gst_t = tax_from_tt
    else:
        gst_t = cgst + sgst + igst

    taxable = inv_sum
    if abs(gst_t) < 1e-9 or abs(inv_sum) < 1e-9:
        return taxable, cgst, sgst, igst, gst_t, False

    r: Optional[float] = None
    if gst_rate_median is not None and gst_rate_median == gst_rate_median:
        r = float(gst_rate_median)
        if r > 1.0:
            r = r / 100.0
        if r <= 0 or r > 0.35:
            r = None

    if r is None:
        abs_inv = abs(inv_sum)
        abs_g = abs(gst_t)
        den = abs_inv - abs_g
        if abs(den) > 1e-6:
            r_try = abs_g / den
            if 0.0005 < r_try < 0.35:
                r = r_try

    if r is None or r <= 0:
        return taxable, cgst, sgst, igst, gst_t, False

    abs_inv = abs(inv_sum)
    gst_if_inclusive = abs_inv * r / (1.0 + r)
    apply_split = gst_rate_from_column
    if not apply_split:
        apply_split = abs(gst_if_inclusive - abs(gst_t)) <= max(2.0, 0.03 * abs(gst_t) + 1e-9)
    if not apply_split:
        return taxable, cgst, sgst, igst, gst_t, False

    sign = -1.0 if inv_sum < 0 else 1.0
    excl_abs = abs_inv / (1.0 + r)
    new_tax_abs = abs_inv - excl_abs
    taxable = round(sign * excl_abs, 2)
    gst_abs_old = abs(cgst) + abs(sgst) + abs(igst)
    if gst_abs_old > 1e-6:
        sc = new_tax_abs / gst_abs_old
        cgst = round(cgst * sc, 2)
        sgst = round(sgst * sc, 2)
        igst = round(igst * sc, 2)
        gst_t = cgst + sgst + igst
    else:
        gst_t = round(sign * new_tax_abs, 2)

    return taxable, cgst, sgst, igst, gst_t, True


def _persist_amazon_finance_sales_entries(
    df,
    *,
    sales_upload_id: int,
    platform: str,
    period: str,
    source_filename: str,
    company_name: str,
    seller_gstin: str,
    company_state: str,
    seller_gstin_filter: Optional[str] = None,
) -> int:
    """
    Build one Day Book row per Amazon invoice (or per order when invoice no missing).
    Refunds/credits are stored as negative amounts. Line-level SKU rows kept in line_items JSON.
    """
    import json
    import pandas as pd

    if df is None or df.empty or "Transaction_Type" not in df.columns:
        return 0
    work = df.copy()
    if seller_gstin_filter and str(seller_gstin_filter).strip().upper() not in ("", "UNKNOWN", "NAN"):
        sg = str(seller_gstin_filter).strip().upper()
        if "Seller_GSTIN" in work.columns:
            work = work[work["Seller_GSTIN"].astype(str).str.strip().str.upper() == sg]
    if work.empty:
        return 0

    work["_dt"] = pd.to_datetime(work["Date"], errors="coerce")
    work["_inv"] = (
        work["Invoice_Number"].astype(str).str.strip()
        if "Invoice_Number" in work.columns
        else pd.Series("", index=work.index, dtype=str)
    )
    work["_oid"] = (
        work["Order_Id"].astype(str).str.strip()
        if "Order_Id" in work.columns
        else pd.Series("", index=work.index, dtype=str)
    )
    work["_buyer"] = (
        work["Buyer_Name"].astype(str).str.strip()
        if "Buyer_Name" in work.columns
        else pd.Series("", index=work.index, dtype=str)
    )
    work["_ship_st"] = (
        work["Ship_To_State"].astype(str).str.strip().str.upper()
        if "Ship_To_State" in work.columns
        else pd.Series("", index=work.index, dtype=str)
    )
    work["_amt"] = pd.to_numeric(work["Invoice_Amount"], errors="coerce").fillna(0.0)
    work["_ttax"] = pd.to_numeric(work.get("Total_Tax", 0), errors="coerce").fillna(0.0)
    work["_cgst"] = pd.to_numeric(work.get("CGST", 0), errors="coerce").fillna(0.0)
    work["_sgst"] = pd.to_numeric(work.get("SGST", 0), errors="coerce").fillna(0.0)
    work["_igst"] = pd.to_numeric(work.get("IGST", 0), errors="coerce").fillna(0.0)
    if "Tax_Exclusive_Gross" in work.columns:
        work["_tex"] = pd.to_numeric(work["Tax_Exclusive_Gross"], errors="coerce").fillna(0.0)
    else:
        work["_tex"] = 0.0

    def _row_key(r) -> str:
        inv = str(r["_inv"]).strip()
        if inv and inv.lower() not in ("nan", "none", ""):
            return inv
        oid = str(r["_oid"]).strip()
        if oid and oid.lower() not in ("nan", "none", ""):
            return f"OID:{oid}"
        return ""

    work["_inv_key"] = work.apply(_row_key, axis=1)

    entries: list[dict] = []
    for txn_type, sign in (("Shipment", 1), ("Refund", -1), ("Return", -1)):
        sub = work[work["Transaction_Type"] == txn_type]
        if sub.empty:
            continue
        bad_key = sub["_inv_key"].eq("")
        if bad_key.any():
            sub = sub.copy()
            sub.loc[bad_key, "_inv_key"] = "ROW_" + sub.index[bad_key].astype(str)
        for inv_key, g in sub.groupby("_inv_key", dropna=False):
            g2 = g.sort_values("_dt")
            first = g2.iloc[0]
            vdt = first["_dt"]
            vdate = vdt.date().isoformat() if pd.notna(vdt) else f"{str(period or '')[:7]}-01"
            buyer = str(first["_buyer"] or "").strip()
            if buyer.lower() in ("nan", "none"):
                buyer = ""
            ship_st = str(first["_ship_st"] or "").strip()
            if ship_st.lower() in ("nan", "none"):
                ship_st = ""
            buyer_gst = ""
            if "Buyer_GSTIN" in g2.columns:
                buyer_gst = str(first.get("Buyer_GSTIN", "") or "").strip().upper()
            if buyer_gst.lower() in ("nan", "none", ""):
                buyer_gst = ""
            ship_city = ""
            if "Ship_To_City" in g2.columns:
                ship_city = str(first.get("Ship_To_City", "") or "").strip()
            if ship_city.lower() in ("nan", "none"):
                ship_city = ""
            ship_location = _format_ship_location(ship_city, ship_st)
            oids = sorted({str(x).strip() for x in g2["_oid"].tolist() if str(x).strip() and str(x).lower() not in ("nan", "none")})
            oid_disp = ", ".join(oids[:3])
            if len(oids) > 3:
                oid_disp += f" (+{len(oids) - 3} more)"
            inv_disp = str(inv_key)
            if inv_disp.startswith("OID:"):
                inv_disp = ""

            inv_sum_pre = float(sign * g2["_amt"].sum())
            tex_sum = float(sign * g2["_tex"].sum())
            cgst = float(sign * g2["_cgst"].sum())
            sgst = float(sign * g2["_sgst"].sum())
            igst = float(sign * g2["_igst"].sum())
            tax_from_tt = float(sign * g2["_ttax"].sum())

            gst_rate_median: Optional[float] = None
            gst_rate_from_column = False
            if "GST_Rate" in g2.columns:
                rt = pd.to_numeric(g2["GST_Rate"], errors="coerce")
                rt = rt[rt.notna() & (rt > 0)]
                if len(rt) > 0:
                    gst_rate_median = float(rt.median())
                    gst_rate_from_column = True

            taxable, cgst, sgst, igst, gst_total, inclusive_reconciled = (
                _reconcile_amazon_invoice_taxable_inclusive(
                    inv_sum_pre,
                    tex_sum,
                    cgst,
                    sgst,
                    igst,
                    tax_from_tt,
                    gst_rate_median,
                    gst_rate_from_column=gst_rate_from_column,
                )
            )
            total_amt = round(taxable + gst_total, 2)
            net_pay = total_amt

            r_line: Optional[float] = None
            if inclusive_reconciled:
                if gst_rate_median is not None and gst_rate_median == gst_rate_median:
                    r_line = float(gst_rate_median)
                    if r_line > 1.0:
                        r_line = r_line / 100.0
                    if r_line <= 0 or r_line > 0.35:
                        r_line = None
                if r_line is None and abs(taxable) > 1e-9:
                    r_line = abs(gst_total) / abs(taxable)

            items: list[dict] = []
            for _, rr in g2.iterrows():
                st_row = str(rr.get("_ship_st") or "").strip()
                city_row = str(rr.get("Ship_To_City", "") or "").strip() if "Ship_To_City" in g2.columns else ""
                if city_row.lower() in ("nan", "none"):
                    city_row = ""
                loc_line = _format_ship_location(city_row, st_row) or st_row
                prod = str(rr.get("Product_Name", "") or "").strip() if "Product_Name" in g2.columns else ""
                if prod.lower() in ("nan", "none"):
                    prod = ""
                qty_l = float(rr.get("Quantity", 0) or 0)
                amt_l = float(sign * float(rr["_amt"]))
                tex_raw = float(rr["_tex"]) if "_tex" in g2.columns else 0.0
                tex_l = float(sign * tex_raw) if abs(tex_raw) > 1e-12 else 0.0
                row_cgst = float(sign * float(rr["_cgst"]))
                row_sgst = float(sign * float(rr["_sgst"]))
                row_igst = float(sign * float(rr["_igst"]))
                row_gst_sum = row_cgst + row_sgst + row_igst
                item_px = float(rr.get("Item_Price", 0) or 0) if "Item_Price" in g2.columns else 0.0
                if abs(tex_l) > 1e-6:
                    line_taxable = tex_l
                    line_tt = float(sign * float(rr["_ttax"]))
                elif inclusive_reconciled and r_line is not None and r_line > 0:
                    line_taxable = round(amt_l / (1.0 + r_line), 2)
                    line_target_gst = round(amt_l - line_taxable, 2)
                    if abs(row_gst_sum) > 1e-9:
                        sc_g = line_target_gst / row_gst_sum
                        row_cgst = round(row_cgst * sc_g, 2)
                        row_sgst = round(row_sgst * sc_g, 2)
                        row_igst = round(row_igst * sc_g, 2)
                    else:
                        row_cgst, row_sgst = 0.0, 0.0
                        row_igst = line_target_gst
                    line_tt = row_cgst + row_sgst + row_igst
                else:
                    line_taxable = tex_l if abs(tex_l) > 1e-6 else amt_l
                    line_tt = float(sign * float(rr["_ttax"]))
                unit_px = (line_taxable / qty_l) if abs(qty_l) > 1e-9 else (item_px if abs(item_px) > 1e-9 else 0.0)
                hsn = str(rr.get("HSN_SAC", "") or "").strip() if "HSN_SAC" in g2.columns else ""
                if hsn.lower() in ("nan", "none"):
                    hsn = ""
                asin = str(rr.get("ASIN", "") or "").strip() if "ASIN" in g2.columns else ""
                fnsku = str(rr.get("FNSKU", "") or "").strip() if "FNSKU" in g2.columns else ""
                oid_it = str(rr.get("Order_Item_Id", "") or "").strip() if "Order_Item_Id" in g2.columns else ""
                ref_parts = [p for p in (asin, fnsku) if p and p.lower() not in ("nan", "none")]
                item_ref = " · ".join(ref_parts) if ref_parts else (oid_it if oid_it and oid_it.lower() not in ("nan", "none") else "")
                bf = str(rr.get("Bill_From_State", "") or "").strip() if "Bill_From_State" in g2.columns else ""
                if bf.lower() in ("nan", "none"):
                    bf = ""
                wh = str(rr.get("Warehouse_Id", "") or "").strip() if "Warehouse_Id" in g2.columns else ""
                ful = str(rr.get("Fulfillment", "") or "").strip() if "Fulfillment" in g2.columns else ""
                pay = str(rr.get("Payment_Method", "") or "").strip() if "Payment_Method" in g2.columns else ""
                irn_status = str(rr.get("IRN_Status", "") or "").strip() if "IRN_Status" in g2.columns else ""
                irn_hash = str(rr.get("IRN_Hash", "") or "").strip() if "IRN_Hash" in g2.columns else ""
                if irn_hash.lower() in ("nan", "none"):
                    irn_hash = ""
                ship_from = str(rr.get("Ship_From_State", "") or "").strip() if "Ship_From_State" in g2.columns else ""
                if ship_from.lower() in ("nan", "none"):
                    ship_from = bf or ""
                loc_ln = str(rr.get("Location_Line", "") or "").strip() if "Location_Line" in g2.columns else ""
                if loc_ln.lower() in ("nan", "none"):
                    loc_ln = ""
                if not loc_ln:
                    loc_ln = " · ".join(p for p in (city_row, st_row, wh) if p and str(p).lower() not in ("nan", "none", ""))
                inv_dt_txt = str(rr.get("Invoice_Date_Text", "") or "").strip() if "Invoice_Date_Text" in g2.columns else ""
                if inv_dt_txt.lower() in ("nan", "none"):
                    inv_dt_txt = ""
                if not inv_dt_txt and "Date" in g2.columns and pd.notna(rr.get("Date")):
                    try:
                        inv_dt_txt = str(pd.Timestamp(rr["Date"]).date())
                    except Exception:
                        inv_dt_txt = ""
                txn_row = str(rr.get("Transaction_Type", "") or "").strip() if "Transaction_Type" in g2.columns else ""
                gst_r = float(rr.get("GST_Rate", 0) or 0) if "GST_Rate" in g2.columns else 0.0
                base_for_rate = abs(line_taxable) if abs(line_taxable) > 1e-9 else abs(amt_l)
                if (gst_r <= 0 or gst_r != gst_r) and base_for_rate > 1e-9 and abs(line_tt) > 1e-9:
                    gst_r = round(100.0 * abs(line_tt) / base_for_rate, 4)
                cn_no = str(rr.get("Credit_Note_No", "") or "").strip() if "Credit_Note_No" in g2.columns else ""
                cn_dt = str(rr.get("Credit_Note_Date", "") or "").strip() if "Credit_Note_Date" in g2.columns else ""
                if cn_no.lower() in ("nan", "none"):
                    cn_no = ""
                if cn_dt.lower() in ("nan", "none"):
                    cn_dt = ""
                ack_dt = str(rr.get("Acknowledgement_Date", "") or "").strip() if "Acknowledgement_Date" in g2.columns else ""
                if ack_dt.lower() in ("nan", "none"):
                    ack_dt = ""
                party_ln = str(rr.get("Buyer_Name", "") or "").strip() if "Buyer_Name" in g2.columns else ""
                if party_ln.lower() in ("nan", "none"):
                    party_ln = ""
                if not party_ln and "Customer_Name_Alt" in g2.columns:
                    party_ln = str(rr.get("Customer_Name_Alt", "") or "").strip()
                cust_gst_ln = str(rr.get("Buyer_GSTIN", "") or "").strip().upper() if "Buyer_GSTIN" in g2.columns else ""
                if cust_gst_ln.lower() in ("nan", "none"):
                    cust_gst_ln = ""
                inv_no_row = str(rr.get("Invoice_Number", "") or "").strip() if "Invoice_Number" in g2.columns else ""
                oid_row = str(rr.get("Order_Id", "") or "").strip() if "Order_Id" in g2.columns else ""
                ship_to_raw = str(rr.get("Ship_To_State", "") or "").strip() if "Ship_To_State" in g2.columns else ""
                if ship_to_raw.lower() in ("nan", "none"):
                    ship_to_raw = ""
                pos = str(rr.get("Place_Of_Supply", "") or "").strip() if "Place_Of_Supply" in g2.columns else ""
                if pos and not ship_to_raw:
                    ship_to_raw = pos
                ship_id_ln = str(rr.get("Shipment_Id", "") or "").strip() if "Shipment_Id" in g2.columns else ""
                if ship_id_ln.lower() in ("nan", "none"):
                    ship_id_ln = ""
                ord_dt_txt = str(rr.get("Order_Date_Text", "") or "").strip() if "Order_Date_Text" in g2.columns else ""
                if ord_dt_txt.lower() in ("nan", "none"):
                    ord_dt_txt = ""
                ship_dt_txt = str(rr.get("Shipment_Date_Text", "") or "").strip() if "Shipment_Date_Text" in g2.columns else ""
                if ship_dt_txt.lower() in ("nan", "none"):
                    ship_dt_txt = ""
                items.append({
                    "type": "Item",
                    "sku": str(rr.get("SKU", "") or ""),
                    "product_name": prod,
                    "variant_code": fnsku if fnsku else "",
                    "item_reference": item_ref,
                    "quantity": qty_l,
                    "unit_price": round(unit_px, 4),
                    "tax_exclusive_amount": round(line_taxable, 2),
                    "invoice_amount": round(amt_l, 2),
                    "total_tax": line_tt,
                    "total_tax_amount": line_tt,
                    "cgst": row_cgst,
                    "sgst": row_sgst,
                    "igst": row_igst,
                    "ship_to_city": city_row,
                    "ship_to_state": loc_line,
                    "ship_from_state": ship_from,
                    "location": loc_ln,
                    "party_name": party_ln,
                    "invoice_number": inv_no_row,
                    "invoice_date": inv_dt_txt,
                    "transaction_type": txn_row,
                    "hsn_sac": hsn,
                    "item_no": str(rr.get("SKU", "") or ""),
                    "ship_to_state_code": ship_to_raw,
                    "tax_exclusive_gross": round(line_taxable, 2),
                    "gst_rate": gst_r,
                    "order_id": oid_row,
                    "shipment_id": ship_id_ln,
                    "order_date_text": ord_dt_txt,
                    "shipment_date_text": ship_dt_txt,
                    "credit_note_no": cn_no,
                    "credit_note_date": cn_dt,
                    "customer_name": party_ln,
                    "customer_gst_no": cust_gst_ln,
                    "irn_hash": irn_hash,
                    "irn_status": irn_status,
                    "acknowledgement_date": ack_dt,
                    "bill_from_state": bf,
                    "place_of_supply": pos,
                    "asin": asin,
                    "fnsku": fnsku,
                    "order_item_id": oid_it,
                    "warehouse_id": wh,
                    "fulfillment": ful,
                    "payment_method": pay,
                })

            narration = f"{platform} {period} — {txn_type}"
            if inv_disp:
                narration += f" — Inv {inv_disp}"
            if oid_disp:
                narration += f" — Order {oid_disp}"

            entries.append({
                "platform": platform,
                "period": period,
                "voucher_date": vdate,
                "invoice_no": inv_disp,
                "order_id": oid_disp,
                # Customer / ship-to from the marketplace file only — never substitute seller company.
                "party_name": buyer,
                "party_gstin": buyer_gst,
                "party_state": ship_st,
                "ship_to_state": ship_location or ship_st,
                "taxable_amount": round(taxable, 2),
                "cgst_amount": round(cgst, 2),
                "sgst_amount": round(sgst, 2),
                "igst_amount": round(igst, 2),
                "total_amount": round(total_amt, 2),
                "net_payable": round(net_pay, 2),
                "narration": narration,
                "source_filename": source_filename or "",
                "line_items": json.dumps(items),
            })

    return create_finance_sales_entries(sales_upload_id, entries)


# ── Finance Sales Upload — parse file and save locked record ──────

@router.post("/sales-uploads/upload-file")
async def upload_sales_file(
    file:        UploadFile = File(...),
    platform:    str        = Form(...),
    period:      str        = Form(...),
    uploaded_by: str        = Form(''),
    notes:       str        = Form(''),
):
    """
    Parse a platform sales file (MTR zip, Myntra zip, Meesho zip, Flipkart zip,
    Snapdeal zip, or plain CSV/Excel) and save a locked record in finance_sales_uploads.
    Returns parsed summary stats + the new record id.

    Does **not** write to the ERP session, Tier-3 daily_sales.db, or affect Dashboard / PO —
    only finance.db (see ``revenue_source=finance_lock`` on /pl and /platform-revenue).
    """
    import pandas as pd
    from ..services.mtr      import load_mtr_from_zip, parse_mtr_csv
    from ..services.myntra   import load_myntra_from_zip
    from ..services.meesho   import load_meesho_from_zip
    from ..services.flipkart import load_flipkart_from_zip
    from ..services.snapdeal import load_snapdeal_from_zip

    raw = await file.read()
    filename = file.filename or ''
    platform_lc = platform.lower()

    # If user drops a full monthly package ZIP in single-file mode, auto-route it.
    if filename.lower().endswith(".zip") and _looks_like_monthly_sales_package(raw):
        return _process_monthly_package_bytes(
            raw=raw,
            filename=filename,
            period=period,
            uploaded_by=uploaded_by,
            notes=notes,
            dry_run=False,
        )

    df = None
    parse_skipped: list[str] = []
    ship_col  = 'Transaction_Type'
    ship_val  = 'Shipment'
    ret_val   = 'Return'
    amt_col   = 'Invoice_Amount'
    order_col = None   # if set, count distinct orders

    try:
        if platform_lc in ('amazon', 'amazon mtr', 'mtr'):
            if filename.lower().endswith(".csv"):
                # Allow direct B2B/B2C CSV uploads in finance flow.
                df_csv, msg = parse_mtr_csv(raw, filename)
                if df_csv.empty:
                    parse_skipped.append(f"{filename}: {msg}")
                else:
                    df = df_csv
                    if msg != "OK":
                        parse_skipped.append(f"{filename}: Partial ({msg})")
            else:
                df, _, parse_skipped = load_mtr_from_zip(raw)
            platform = 'Amazon'
        elif platform_lc == 'myntra':
            # Myntra needs sku_mapping; skip mapping → parse raw directly
            df, _, _ = load_myntra_from_zip(raw, {})
        elif platform_lc == 'meesho':
            # Try gst_ Finance format first (tcs_sales.xlsx), then regular order format
            stats = _parse_meesho_gst_for_finance(raw)
            if stats["orders"] > 0:
                new_id = create_finance_sales_upload({
                    'platform':      'Meesho',
                    'period':        period,
                    'filename':      filename,
                    'total_revenue': round(stats["revenue"], 2),
                    'total_orders':  stats["orders"],
                    'total_returns': round(stats["returns"], 2),
                    'net_revenue':   round(stats["revenue"] - stats["returns"], 2),
                    'uploaded_by':   uploaded_by,
                    'upload_notes':  notes,
                })
                return {
                    'ok': True, 'id': new_id, 'platform': 'Meesho', 'period': period,
                    'filename': filename,
                    'total_revenue': round(stats["revenue"], 2),
                    'total_orders':  stats["orders"],
                    'total_returns': round(stats["returns"], 2),
                    'net_revenue':   round(stats["revenue"] - stats["returns"], 2),
                    'rows_parsed':   stats["orders"],
                }
            df, _, _ = load_meesho_from_zip(raw)
        elif platform_lc == 'flipkart':
            df, _, _ = load_flipkart_from_zip(raw)
        elif platform_lc == 'snapdeal':
            # Snapdeal settlement reports: parse revenue/returns directly
            stats = _parse_snapdeal_settlement_for_finance(raw)
            if stats["revenue"] > 0 or stats["returns"] > 0:
                new_id = create_finance_sales_upload({
                    'platform':      'Snapdeal',
                    'period':        period,
                    'filename':      filename,
                    'total_revenue': round(stats["revenue"], 2),
                    'total_orders':  stats["orders"],
                    'total_returns': round(stats["returns"], 2),
                    'net_revenue':   round(stats["revenue"] - stats["returns"], 2),
                    'uploaded_by':   uploaded_by,
                    'upload_notes':  notes,
                })
                return {
                    'ok': True, 'id': new_id, 'platform': 'Snapdeal', 'period': period,
                    'filename': filename,
                    'total_revenue': round(stats["revenue"], 2),
                    'total_orders':  stats["orders"],
                    'total_returns': round(stats["returns"], 2),
                    'net_revenue':   round(stats["revenue"] - stats["returns"], 2),
                    'rows_parsed':   stats["orders"],
                }
            df, _, _, _ = load_snapdeal_from_zip(raw, {}, filename)
        else:
            # Fallback: try reading as Excel/CSV for unknown platforms
            try:
                import io
                df = pd.read_excel(io.BytesIO(raw))
            except Exception:
                import io
                df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse {platform} file: {e}")

    # Compute summary stats from DataFrame
    total_revenue = 0.0
    total_returns = 0.0
    total_orders  = 0
    state_breakdown: list[dict] = []
    company_breakdown: list[dict] = []

    if df is not None and not df.empty:
        if ship_col in df.columns and amt_col in df.columns:
            total_revenue = float(df[df[ship_col] == ship_val][amt_col].sum())
            if platform_lc in ('amazon', 'amazon mtr', 'mtr'):
                # Amazon parser uses Refund labels; keep returns as positive value.
                ret_mask = df[ship_col].isin([ret_val, 'Refund'])
                total_returns = float(df.loc[ret_mask, amt_col].abs().sum())
            else:
                total_returns = float(df[df[ship_col] == ret_val][amt_col].abs().sum())
        elif amt_col in df.columns:
            total_revenue = float(df[amt_col].sum())

        # Estimate order count from row count (shipments only)
        if ship_col in df.columns:
            total_orders = int((df[ship_col] == ship_val).sum())
        else:
            total_orders = len(df)

        if platform_lc in ('amazon', 'amazon mtr', 'mtr'):
            state_breakdown, company_breakdown = _build_geo_company_breakdowns(
                df=df,
                ship_col=ship_col,
                amt_col=amt_col,
                ship_val=ship_val,
                return_values={ret_val, 'Refund'},
            )
    elif parse_skipped:
        # Surface parser reasons clearly instead of silently saving zero totals.
        raise HTTPException(
            status_code=422,
            detail="No usable rows parsed. " + " | ".join(parse_skipped[:8]),
        )

    net_revenue = total_revenue - total_returns

    # Save one row per company for Amazon when GST/company split is available.
    saved_rows = []
    if platform_lc in ('amazon', 'amazon mtr', 'mtr') and company_breakdown:
        for c in company_breakdown:
            nid = create_finance_sales_upload({
                'platform': 'Amazon',
                'company_name': c.get('company') or '',
                'seller_gstin': c.get('seller_gstin') or '',
                'company_state': c.get('company_state') or '',
                'period': period,
                'filename': filename,
                'total_revenue': round(float(c.get('gross_revenue') or 0), 2),
                'total_orders': int(c.get('orders') or 0),
                'total_returns': round(float(c.get('returns') or 0), 2),
                'net_revenue': round(float(c.get('net_revenue') or 0), 2),
                'uploaded_by': uploaded_by,
                'upload_notes': notes,
            })
            saved_rows.append({
                "id": nid,
                "company": c.get("company"),
                "seller_gstin": c.get("seller_gstin"),
                "company_state": c.get("company_state"),
                "orders": int(c.get("orders") or 0),
                "gross_revenue": round(float(c.get("gross_revenue") or 0), 2),
                "returns": round(float(c.get("returns") or 0), 2),
                "net_revenue": round(float(c.get("net_revenue") or 0), 2),
            })
        new_id = saved_rows[0]["id"] if saved_rows else None
    else:
        new_id = create_finance_sales_upload({
            'platform':      platform,
            'period':        period,
            'filename':      filename,
            'total_revenue': round(total_revenue, 2),
            'total_orders':  total_orders,
            'total_returns': round(total_returns, 2),
            'net_revenue':   round(net_revenue, 2),
            'uploaded_by':   uploaded_by,
            'upload_notes':  notes,
        })

    # Invoice/order-level Day Book rows for Amazon MTR (one row per invoice / order).
    sales_entry_rows = 0
    if platform_lc in ('amazon', 'amazon mtr', 'mtr') and df is not None and not df.empty:
        if saved_rows:
            for row in saved_rows:
                sales_entry_rows += _persist_amazon_finance_sales_entries(
                    df,
                    sales_upload_id=int(row["id"]),
                    platform='Amazon',
                    period=period,
                    source_filename=filename,
                    company_name=str(row.get('company') or ''),
                    seller_gstin=str(row.get('seller_gstin') or ''),
                    company_state=str(row.get('company_state') or ''),
                    seller_gstin_filter=str(row.get('seller_gstin') or ''),
                )
        elif new_id:
            sales_entry_rows += _persist_amazon_finance_sales_entries(
                df,
                sales_upload_id=int(new_id),
                platform='Amazon',
                period=period,
                source_filename=filename,
                company_name='',
                seller_gstin='',
                company_state='',
            )

    return {
        'ok':            True,
        'id':            new_id,
        'platform':      platform,
        'period':        period,
        'filename':      filename,
        'total_revenue': round(total_revenue, 2),
        'total_orders':  total_orders,
        'total_returns': round(total_returns, 2),
        'net_revenue':   round(net_revenue, 2),
        'rows_parsed':   len(df) if df is not None else 0,
        'skipped':       parse_skipped,
        'state_breakdown': state_breakdown,
        'company_breakdown': company_breakdown,
        'saved_companies': saved_rows,
        'sales_entry_rows': sales_entry_rows,
    }


@router.post("/sales-uploads/upload-monthly-package")
async def upload_monthly_package(
    file:        UploadFile = File(...),
    period:      str        = Form(...),
    uploaded_by: str        = Form(''),
    notes:       str        = Form(''),
):
    """
    Accept the full monthly sales ZIP (e.g. Sales Data Feb 2026.zip).
    Auto-detects platform subfolders and parses each with the right parser.
    Creates one finance_sales_upload record per platform/account found.
    """
    raw = await file.read()
    return _process_monthly_package_bytes(
        raw=raw,
        filename=file.filename or "monthly",
        period=period,
        uploaded_by=uploaded_by,
        notes=notes,
        dry_run=False,
    )


@router.post("/sales-uploads/preview-monthly-package")
async def preview_monthly_package(
    file:   UploadFile = File(...),
    period: str        = Form(...),
):
    """
    Dry-run parse of a monthly sales ZIP — same logic as upload_monthly_package
    but nothing is saved to the database. Returns a preview so the user can
    verify numbers before committing.
    """
    raw = await file.read()
    return _process_monthly_package_bytes(
        raw=raw,
        filename=file.filename or "monthly",
        period=period,
        uploaded_by="",
        notes="",
        dry_run=True,
    )
