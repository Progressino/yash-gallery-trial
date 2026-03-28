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
from typing import Optional, List
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from ..db.finance_db import (
    add_expense, list_expenses, delete_expense,
    list_ledger_groups, create_ledger_group, delete_ledger_group,
    list_ledgers, create_ledger, update_ledger, delete_ledger,
    list_gst_classifications, create_gst_classification, delete_gst_classification,
    list_tds_sections, create_tds_section, delete_tds_section,
    list_expense_vouchers, get_expense_voucher, create_expense_voucher, delete_expense_voucher,
    list_finance_sales_uploads, create_finance_sales_upload, delete_finance_sales_upload,
    list_voucher_types, create_voucher_type, update_voucher_type, delete_voucher_type,
    list_vouchers, get_voucher_summary_by_date, get_gstr3b_data, get_ledger_balances,
    get_chart_of_accounts, get_trial_balance,
)
from ..services.finance import get_pl_statement, get_gst_summary, get_platform_revenue

_FINANCE_PIN = os.environ.get("FINANCE_PIN", "").strip()

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
):
    sess = _sess(request)
    return get_platform_revenue(
        mtr_df      = sess.mtr_df,
        myntra_df   = sess.myntra_df,
        meesho_df   = sess.meesho_df,
        flipkart_df = sess.flipkart_df,
        start_date  = start_date,
        end_date    = end_date,
    )


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


@router.post("/vouchers")
def post_voucher(body: VoucherCreate):
    data = body.model_dump()
    data['lines'] = [ln.model_dump() for ln in body.lines]
    voucher_no = create_expense_voucher(data)
    return {"ok": True, "voucher_no": voucher_no}


@router.get("/vouchers/{voucher_id}")
def get_voucher(voucher_id: int):
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
):
    return list_finance_sales_uploads(platform=platform, period=period)


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
    """
    import pandas as pd
    from ..services.mtr      import load_mtr_from_zip
    from ..services.myntra   import load_myntra_from_zip
    from ..services.meesho   import load_meesho_from_zip
    from ..services.flipkart import load_flipkart_from_zip
    from ..services.snapdeal import load_snapdeal_from_zip

    raw = await file.read()
    filename = file.filename or ''
    platform_lc = platform.lower()

    df = None
    ship_col  = 'Transaction_Type'
    ship_val  = 'Shipment'
    ret_val   = 'Return'
    amt_col   = 'Invoice_Amount'
    order_col = None   # if set, count distinct orders

    try:
        if platform_lc in ('amazon', 'amazon mtr', 'mtr'):
            df, _, _ = load_mtr_from_zip(raw)
            platform = 'Amazon'
        elif platform_lc == 'myntra':
            # Myntra needs sku_mapping; skip mapping → parse raw directly
            df, _, _ = load_myntra_from_zip(raw, {})
        elif platform_lc == 'meesho':
            df, _, _ = load_meesho_from_zip(raw)
        elif platform_lc == 'flipkart':
            df, _, _ = load_flipkart_from_zip(raw)
        elif platform_lc == 'snapdeal':
            df, _, _ = load_snapdeal_from_zip(raw)
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

    if df is not None and not df.empty:
        if ship_col in df.columns and amt_col in df.columns:
            total_revenue = float(df[df[ship_col] == ship_val][amt_col].sum())
            total_returns = float(df[df[ship_col] == ret_val][amt_col].sum())
        elif amt_col in df.columns:
            total_revenue = float(df[amt_col].sum())

        # Estimate order count from row count (shipments only)
        if ship_col in df.columns:
            total_orders = int((df[ship_col] == ship_val).sum())
        else:
            total_orders = len(df)

    net_revenue = total_revenue - total_returns

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
    }
