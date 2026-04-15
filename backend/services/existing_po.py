"""
Existing PO sheet loader.
Parses an Excel/CSV uploaded by the user that contains open/pending PO quantities.

Returns a DataFrame with:
  - OMS_SKU
  - PO_Pipeline_Total   (total in-pipeline, used for net PO deduction)
  - PO_Qty_Ordered      (original qty ordered with manufacturer, if col found)
  - Pending_Cutting     (units awaiting cutting, if col found)
  - Balance_to_Dispatch (units cut but not dispatched, if col found)
"""
import io
from typing import Optional

import pandas as pd


_SKU_EXACT = [
    "OMS SKU", "OMS_SKU", "OMS SKU Code", "OMS",
    "SKU", "Seller SKU", "Merchant SKU", "Listing SKU", "Parent SKU",
    "Style Code", "Style", "Item Code", "Product Code", "Product ID",
    "ASIN", "Vendor Article Number", "Vendor Article", "Article Code",
    "Article", "Vendor Style", "MRP Article",
]


def _find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    """Case-insensitive exact column finder."""
    lower = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _find_col_fuzzy(cols: list[str], keywords: list[str]) -> Optional[str]:
    """Find a column whose name *contains* any of the given keywords (case-insensitive)."""
    for col in cols:
        col_l = col.lower()
        if any(kw in col_l for kw in keywords):
            return col
    return None


def _resolve_sku_column(cols: list[str]) -> Optional[str]:
    """Exact names first, then a safe fuzzy match (avoid order/sub-order id columns)."""
    hit = _find_col(cols, _SKU_EXACT)
    if hit:
        return hit
    skip_sub = ("order", "sub order", "packet", "line id", "release")
    for col in cols:
        cl = str(col).lower().strip()
        if not cl or cl.startswith("unnamed"):
            continue
        if any(x in cl for x in skip_sub):
            continue
        if "sku" in cl or cl.endswith("sku"):
            return col
        if cl in ("style code", "item code", "article code", "vendor article"):
            return col
    return None


def _dedupe_columns(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.loc[:, ~raw.columns.duplicated(keep="first")].copy()
    out.columns = out.columns.astype(str).str.strip()
    return out


def _read_po_csv(file_bytes: bytes) -> pd.DataFrame:
    try:
        raw = pd.read_csv(
            io.BytesIO(file_bytes), dtype=str, encoding="utf-8", on_bad_lines="skip"
        )
    except UnicodeDecodeError:
        raw = pd.read_csv(
            io.BytesIO(file_bytes), dtype=str, encoding="ISO-8859-1", on_bad_lines="skip"
        )
    return _dedupe_columns(raw)


def _read_po_excel(file_bytes: bytes) -> pd.DataFrame:
    """
    Try every sheet and header row 0..5 — real templates often have a title row
    or put the table on sheet 2, which breaks a naive read_excel(sheet=0).
    """
    bio = io.BytesIO(file_bytes)
    try:
        xl = pd.ExcelFile(bio)
    except Exception as e:
        raise ValueError(f"Cannot open Excel file: {e}") from e

    candidates: list[tuple[int, pd.DataFrame, str, int]] = []
    for sheet in xl.sheet_names:
        for header_row in range(0, 6):
            try:
                raw = pd.read_excel(xl, sheet_name=sheet, header=header_row, dtype=str)
            except Exception:
                continue
            if raw.empty or len(raw.columns) < 2:
                continue
            raw = _dedupe_columns(raw)
            cols = list(raw.columns)
            sku_guess = _resolve_sku_column(cols)
            if not sku_guess:
                continue
            n_ok = (
                raw[sku_guess]
                .astype(str)
                .str.strip()
                .str.len()
                .gt(0)
                .sum()
            )
            candidates.append((int(n_ok), raw, sheet, header_row))

    if not candidates:
        raise ValueError(
            "Could not find a SKU column on any sheet (tried multiple header rows). "
            "Add a column such as: OMS SKU, SKU, Style Code, Item Code, Seller SKU, "
            "Vendor Article — or move the table to start on row 1 with headers in one row."
        )

    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _read_raw_po(file_bytes: bytes, filename: str) -> pd.DataFrame:
    fn_lower = filename.lower()
    try:
        if fn_lower.endswith(".csv"):
            raw = _read_po_csv(file_bytes)
        else:
            raw = _read_po_excel(file_bytes)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Cannot read file: {e}") from e

    if raw.empty:
        raise ValueError("File is empty.")

    return raw


def parse_existing_po(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse an existing PO tracking sheet.
    Accepts .xlsx or .csv.

    Extracts (all optional except SKU):
      - PO_Qty_Ordered      : qty originally ordered with manufacturer
      - Pending_Cutting     : units awaiting cutting
      - Balance_to_Dispatch : units cut but not yet dispatched
      - Total_Balance       : total pipeline → PO_Pipeline_Total

    If none of the specific columns are found, falls back to any
    generic balance/qty column as PO_Pipeline_Total.
    """
    raw = _read_raw_po(file_bytes, filename)
    cols = list(raw.columns)

    sku_col = _resolve_sku_column(cols)
    if sku_col is None:
        raise ValueError(
            f"Cannot find a SKU column. Columns seen: {cols[:25]}"
        )

    # ── Find specific breakdown columns (all optional) ──────────
    cutting_col = _find_col(
        cols,
        [
            "Pending Cutting", "Pending Cut", "Cutting Pending",
            "Pend Cutting", "Cut Pending", "Pending Cuts",
        ],
    )
    if cutting_col is None:
        cutting_col = _find_col_fuzzy(
            cols, ["pending cut", "cut pending", "pending cutting"]
        )

    dispatch_col = _find_col(
        cols,
        [
            "Balance to dispatch", "Dispatch Balance", "Bal to Dispatch",
            "Balance Dispatch", "Pending dispatch", "Pending Dispatch",
        ],
    )
    if dispatch_col is None:
        dispatch_col = _find_col_fuzzy(
            cols,
            [
                "balance to dispatch",
                "dispatch balance",
                "bal to dispatch",
                "qty to dispatch",
                "pending dispatch",
            ],
        )

    total_col = _find_col(
        cols,
        [
            "TOTAL BALANCE From Latest Status", "Total Balance From Latest Status",
            "Total Balance", "Total Bal", "Total Pending", "Net Balance", "TOTAL BALANCE",
        ],
    )
    if total_col is None:
        total_col = _find_col_fuzzy(cols, ["total balance", "total bal"])

    _claimed = {c for c in [cutting_col, dispatch_col, total_col] if c}
    ordered_col = _find_col(
        cols,
        [
            "NEW ORDER", "New Order", "PO Qty", "PO Quantity",
            "Ordered Qty", "Ordered Quantity", "Order Qty",
        ],
    )
    if ordered_col is None or ordered_col in _claimed:
        ordered_col = next(
            (
                col
                for col in cols
                if col not in _claimed
                and any(
                    kw in col.lower()
                    for kw in ["new order", "ordered qty", "order qty"]
                )
            ),
            None,
        )

    fallback_col: Optional[str] = None
    if not any([ordered_col, cutting_col, dispatch_col, total_col]):
        fallback_col = _find_col(
            cols,
            [
                "Balance Qty", "Balance Quantity", "Open Qty", "Open Quantity",
                "Pending Qty", "Pending Quantity", "Units", "Balance",
                "Qty", "Balance to dispatch", "TOTAL BALANCE From Latest Status",
                "Total Balance", "Dispatch Balance", "Pending dispatch",
                "New Order", "NEW ORDER",
            ],
        )
        if fallback_col is None:
            fallback_col = _find_col_fuzzy(
                cols, ["balance", "dispatch", "pending", "open qty", "po qty", "new order"]
            )
        if fallback_col is None:
            raise ValueError(
                f"Cannot find a quantity/balance column. Columns seen: {cols[:25]}"
            )

    specific_cols = [
        c for c in [ordered_col, cutting_col, dispatch_col, total_col, fallback_col] if c
    ]
    all_needed = list(dict.fromkeys([sku_col] + specific_cols))
    try:
        df = raw[all_needed].copy()
    except KeyError as e:
        raise ValueError(f"Missing expected column while building PO table: {e}") from e

    df[sku_col] = df[sku_col].astype(str).str.strip().str.upper()
    df = df[df[sku_col].str.len() > 0]
    df = df[~df[sku_col].isin(["SKU", "OMS_SKU", "NAN", "NONE", ""])]

    if df.empty:
        raise ValueError("No valid SKU rows found after parsing.")

    for c in specific_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    agg_dict = {c: "sum" for c in specific_cols}
    result = df.groupby(sku_col, as_index=False).agg(agg_dict)
    result = result.rename(columns={sku_col: "OMS_SKU"})

    rename_map: dict[str, str] = {}
    if ordered_col:
        rename_map[ordered_col] = "PO_Qty_Ordered"
    if cutting_col:
        rename_map[cutting_col] = "Pending_Cutting"
    if dispatch_col:
        rename_map[dispatch_col] = "Balance_to_Dispatch"
    if total_col:
        rename_map[total_col] = "Total_Balance"
    if fallback_col and fallback_col not in rename_map:
        rename_map[fallback_col] = "PO_Qty_Ordered"
    result = result.rename(columns=rename_map)

    if "Total_Balance" in result.columns:
        result["PO_Pipeline_Total"] = result["Total_Balance"].clip(lower=0).astype(int)
        result = result.drop(columns=["Total_Balance"])
    elif "Pending_Cutting" in result.columns and "Balance_to_Dispatch" in result.columns:
        result["PO_Pipeline_Total"] = (
            result["Pending_Cutting"] + result["Balance_to_Dispatch"]
        ).clip(lower=0).astype(int)
    elif "PO_Qty_Ordered" in result.columns:
        result["PO_Pipeline_Total"] = result["PO_Qty_Ordered"].clip(lower=0).astype(int)
    else:
        result["PO_Pipeline_Total"] = 0

    for c in ["PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch", "PO_Pipeline_Total"]:
        if c in result.columns:
            result[c] = result[c].clip(lower=0).astype(int)

    return result
