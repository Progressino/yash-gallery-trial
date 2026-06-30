"""How far back PO ADS / seasonality must read unified sales history."""
from __future__ import annotations

import os


def po_ads_history_horizon_days(
    period_days: int,
    *,
    use_seasonality: bool = False,
    use_ly_fallback: bool = False,
) -> int:
    """
    Minimum calendar span ending at planning_date that must be present in sales_df.

    - Recent ADS: ``period_days``
    - LY (parallel + forward): ``period_days + 365`` each
    - Seasonal (2 prior years × 3-month band): ~760d
    """
    horizon = max(int(period_days), 90)
    if use_ly_fallback:
        horizon = max(horizon, int(period_days) + 365 + 14)
    if use_seasonality:
        try:
            seasonal_span = int(os.environ.get("PO_SEASONAL_LOOKBACK_DAYS", "800"))
        except ValueError:
            seasonal_span = 800
        horizon = max(horizon, seasonal_span)
    return horizon
