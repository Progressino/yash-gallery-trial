"""
Finance module router.
GET  /api/finance/pl               — P&L statement
GET  /api/finance/gst              — GST summary (Amazon MTR)
GET  /api/finance/platform-revenue — per-platform revenue reconciliation
GET  /api/finance/expenses         — list expenses
POST /api/finance/expenses         — add expense
DELETE /api/finance/expenses/{id}  — delete expense
"""
from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from ..db.finance_db import add_expense, list_expenses, delete_expense
from ..services.finance import get_pl_statement, get_gst_summary, get_platform_revenue

router = APIRouter()


class ExpenseCreate(BaseModel):
    date:        str
    category:    str
    description: str   = ""
    amount:      float
    gst_amount:  float = 0.0


def _sess(request: Request):
    sess = request.state.session
    if sess is None:
        raise HTTPException(status_code=403, detail="No session")
    return sess


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
