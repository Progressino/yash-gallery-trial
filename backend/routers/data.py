"""
Data query router.
GET /api/data/coverage  → what data is currently loaded in this session
"""
from fastapi import APIRouter, Request, HTTPException
from ..models.schemas import CoverageResponse

router = APIRouter()


@router.get("/coverage", response_model=CoverageResponse)
def get_coverage(request: Request):
    sess = request.state.session
    if sess is None:
        raise HTTPException(status_code=500, detail="Session not initialised")

    return CoverageResponse(
        sku_mapping=bool(sess.sku_mapping),
        mtr=not sess.mtr_df.empty,
        sales=not sess.sales_df.empty,
        myntra=not sess.myntra_df.empty,
        meesho=not sess.meesho_df.empty,
        flipkart=not sess.flipkart_df.empty,
        inventory=not sess.inventory_df_variant.empty,
        daily_orders=not sess.daily_orders_df.empty,
        mtr_rows=len(sess.mtr_df),
        sales_rows=len(sess.sales_df),
        myntra_rows=len(sess.myntra_df),
        meesho_rows=len(sess.meesho_df),
        flipkart_rows=len(sess.flipkart_df),
    )
