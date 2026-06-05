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
import re
from typing import Optional

import pandas as pd

# Heuristic used only for auto-detection (headerless sheets, ambiguous SKU columns).
# Must be permissive: "OMS_SKU" can be size SKUs (1917YKBLUE-3XL), parent-like tokens,
# or vendor/article ids (V1-A). The main goal is to reject non-SKU labels like "New SKU".
_OMS_SKU_RE = re.compile(r"^[A-Z0-9][A-Z0-9-]{2,}$", re.I)
_PO_SIZE_TOKENS = frozenset(
    {"XS", "S", "M", "L", "XL", "XXL", "XXXL", "2XL", "3XL", "4XL", "5XL", "6XL", "7XL", "8XL"}
)

# Accept common unicode dash variants from Excel/WhatsApp copies.
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
_DASH_SPLIT_RE = re.compile(r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]+")


def _normalize_sku_text(value: object) -> str:
    """Normalize SKU tokens for stable grouping/joins (unicode dashes, stray whitespace)."""
    s = str(value or "").strip().upper()
    if not s:
        return ""
    s = _DASH_RE.sub("-", s)
    # Collapse " - " / "--" to "-" and trim around separators.
    parts = [p.strip() for p in s.split("-")]
    parts = [p for p in parts if p]
    return "-".join(parts)


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
            sku_series = raw[sku_guess].astype(str).str.strip()
            n_ok = int(sku_series.map(_looks_like_oms_sku).sum())
            if n_ok == 0:
                # Column name looked like SKU-ish (e.g. "New SKU") but values do not.
                # Skip this candidate; otherwise headerless sheets get mis-detected.
                continue
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


def _looks_like_oms_sku(value: object) -> bool:
    s = str(value or "").strip().upper()
    if len(s) < 3 or s in {"SKU", "OMS_SKU", "NAN", "NONE"}:
        return False
    if _OMS_SKU_RE.match(s):
        parts = s.split("-")
        # Size SKUs (and bundled size ranges).
        if len(parts) >= 2 and parts[-1] in _PO_SIZE_TOKENS:
            return True
        if len(parts) >= 3 and parts[-1] in _PO_SIZE_TOKENS and parts[-2] in _PO_SIZE_TOKENS:
            return True
        # Vendor/article ids usually include digits.
        if any(ch.isdigit() for ch in s):
            return True
        # Generic "ABC-RED-L" style tokens.
        if "-" in s and len(s) >= 6:
            return True
    return False


def _column_numeric_ratio(series: pd.Series) -> float:
    nums = pd.to_numeric(series, errors="coerce")
    if len(nums) == 0:
        return 0.0
    return float(nums.notna().mean())


def _column_looks_numeric_qty(series: pd.Series) -> bool:
    return _column_numeric_ratio(series) >= 0.6


def _read_headerless_po(file_bytes: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Sheets whose first row is data (no header) — common in operator exports."""
    fn_lower = filename.lower()
    try:
        if fn_lower.endswith(".csv"):
            raw = pd.read_csv(
                io.BytesIO(file_bytes), header=None, dtype=str, encoding="utf-8", on_bad_lines="skip"
            )
        else:
            raw = pd.read_excel(io.BytesIO(file_bytes), header=None, dtype=str)
    except Exception:
        return None
    if raw.empty or raw.shape[1] < 4:
        return None
    raw = raw.dropna(how="all").reset_index(drop=True)
    sku_series = raw.iloc[:, 0].astype(str).str.strip()
    hits = sku_series.map(_looks_like_oms_sku).sum()
    if hits < max(2, int(len(raw) * 0.5)):
        return None
    out = raw.copy()
    out.columns = [f"_c{i}" for i in range(out.shape[1])]
    return out


def _infer_quantity_columns(raw: pd.DataFrame, sku_col: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    When named headers are missing, pick numeric columns by data shape.
    Typical Yash layout: SKU | status | style | name | pending | total | total | 0
    """
    numeric: list[tuple[str, float, float]] = []
    for col in raw.columns:
        if col == sku_col:
            continue
        series = raw[col]
        if not _column_looks_numeric_qty(series):
            continue
        nums = pd.to_numeric(series, errors="coerce").fillna(0)
        total = float(nums.sum())
        if total <= 0:
            continue
        numeric.append((col, total, float(nums.max())))

    if not numeric:
        return None, None, None, None

    numeric.sort(key=lambda x: (-x[1], -x[2]))
    total_col = numeric[0][0]
    total_series = pd.to_numeric(raw[total_col], errors="coerce").fillna(0)
    cutting_col = None
    if len(numeric) > 1:
        # Smallest non-zero numeric column that is not a duplicate of the total column.
        rest = []
        for col, sm, mx in numeric[1:]:
            if col == total_col:
                continue
            other = pd.to_numeric(raw[col], errors="coerce").fillna(0)
            if other.equals(total_series):
                continue
            rest.append((col, sm, mx))
        if rest:
            cutting_col = min(rest, key=lambda x: x[1])[0]
    return None, cutting_col, None, total_col


def _split_bundled_po_sku(sku: str) -> list[str]:
    """1917YKBLUE-XXL-3XL → [1917YKBLUE-XXL, 1917YKBLUE-3XL]."""
    s = _normalize_sku_text(sku)
    # Split on any dash variant / repeated separators.
    parts = [p.strip() for p in _DASH_SPLIT_RE.split(s) if p.strip()]
    if len(parts) >= 3 and parts[-1] in _PO_SIZE_TOKENS and parts[-2] in _PO_SIZE_TOKENS:
        base = "-".join(parts[:-2])
        return [f"{base}-{parts[-2]}", f"{base}-{parts[-1]}"]
    return [s]


def is_bundled_size_range_sku(sku: object) -> bool:
    """True when SKU is a combined size band like 1917YKBLUE-XXL-3XL."""
    parts = _DASH_SPLIT_RE.split(_normalize_sku_text(sku))
    return (
        len(parts) >= 3
        and parts[-1] in _PO_SIZE_TOKENS
        and parts[-2] in _PO_SIZE_TOKENS
    )


def expand_bundled_po_skus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fan out combined size-range PO lines (S-M, XXL-3XL) to individual OMS SKUs
    so PO Engine rows like 1917YKBLUE-3XL receive pipeline quantities.
    """
    if df is None or df.empty:
        return df
    breakdown = [
        c
        for c in ("PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch", "PO_Pipeline_Total")
        if c in df.columns
    ]
    rows: list[dict] = []
    for _, r in df.iterrows():
        sku = _normalize_sku_text(r.get("OMS_SKU"))
        if not sku:
            continue
        expanded = _split_bundled_po_sku(sku)
        n = len(expanded)
        for esku in expanded:
            row: dict = {"OMS_SKU": esku}
            for c in breakdown:
                val = pd.to_numeric(r.get(c), errors="coerce")
                val = 0 if pd.isna(val) else int(val)
                row[c] = val // n if n > 1 else val
            rows.append(row)
    if not rows:
        return df
    out = pd.DataFrame(rows)
    if breakdown:
        out = out.groupby("OMS_SKU", as_index=False)[breakdown].sum()
    return out


def prepare_existing_po_for_merge(
    existing_po_df: pd.DataFrame,
    canonical_fn,
) -> pd.DataFrame:
    """Canonicalize + expand bundled size ranges; aggregate to one row per OMS_SKU."""
    if existing_po_df is None or existing_po_df.empty:
        return pd.DataFrame()
    ep = existing_po_df.copy()
    if "PO_Pipeline_Total" not in ep.columns:
        return pd.DataFrame()
    ep["OMS_SKU"] = (
        ep["OMS_SKU"].astype(str).map(canonical_fn).astype(str).str.strip().str.upper()
    )
    ep = ep[ep["OMS_SKU"].str.len() > 0]
    return expand_bundled_po_skus(ep)


def rollup_pipeline_onto_bundled_rows(
    po_df: pd.DataFrame,
    ep: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inventory often uses bundled SKUs (4XL-5XL) while Existing PO lists individual sizes.
    Sum child pipeline onto the bundled inventory row when direct merge missed.
    """
    if po_df is None or po_df.empty or ep is None or ep.empty or "OMS_SKU" not in po_df.columns:
        return po_df
    breakdown = [
        c
        for c in ("PO_Pipeline_Total", "PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch")
        if c in ep.columns
    ]
    if not breakdown:
        return po_df
    ep_idx = ep.set_index("OMS_SKU")
    out = po_df.copy()
    for col in breakdown:
        if col not in out.columns:
            out[col] = 0
    for idx, row in out.iterrows():
        sku = _normalize_sku_text(row.get("OMS_SKU"))
        if not is_bundled_size_range_sku(sku):
            continue
        children = _split_bundled_po_sku(sku)
        if len(children) < 2:
            continue
        for col in breakdown:
            total = 0.0
            found_child = False
            for child in children:
                if child in ep_idx.index:
                    found_child = True
                    total += float(pd.to_numeric(ep_idx.at[child, col], errors="coerce") or 0)
            # Always prefer child totals on bundled inventory rows — the sheet lists
            # individual sizes, not the bundled inventory token.
            if found_child and total > 0:
                out.at[idx, col] = int(total) if col == "PO_Pipeline_Total" else total
    return out


def dedupe_po_rows_by_sku(po_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate OMS_SKU rows — keep the richest row (pipeline / sales / stock)."""
    if po_df is None or po_df.empty or "OMS_SKU" not in po_df.columns:
        return po_df
    if not po_df["OMS_SKU"].astype(str).duplicated().any():
        return po_df
    work = po_df.copy()

    def _num(col: str) -> pd.Series:
        if col not in work.columns:
            return pd.Series(0.0, index=work.index)
        return pd.to_numeric(work[col], errors="coerce").fillna(0)

    work["_dedupe_rank"] = (
        _num("PO_Pipeline_Total")
        + _num("Total_Inventory")
        + _num("Sold_Units")
        + _num("ADS") * 10.0
    )
    work = work.sort_values("_dedupe_rank", ascending=False)
    work = work.drop_duplicates(subset=["OMS_SKU"], keep="first")
    return work.drop(columns=["_dedupe_rank"], errors="ignore")


def _build_po_dataframe(
    raw: pd.DataFrame,
    sku_col: str,
    *,
    ordered_col: Optional[str],
    cutting_col: Optional[str],
    dispatch_col: Optional[str],
    total_col: Optional[str],
    fallback_col: Optional[str],
) -> pd.DataFrame:
    specific_cols = [
        c for c in [ordered_col, cutting_col, dispatch_col, total_col, fallback_col] if c
    ]
    all_needed = list(dict.fromkeys([sku_col] + specific_cols))
    df = raw[all_needed].copy()

    df[sku_col] = df[sku_col].map(_normalize_sku_text)
    df = df[df[sku_col].str.len() > 0]
    df = df[~df[sku_col].isin(["SKU", "OMS_SKU", "NAN", "NONE", ""])]

    if df.empty:
        raise ValueError("No valid SKU rows found after parsing.")

    for c in specific_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    result = df.groupby(sku_col, as_index=False).agg({c: "sum" for c in specific_cols})
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
    elif "Pending_Cutting" in result.columns and "PO_Pipeline_Total" not in result.columns:
        result["PO_Pipeline_Total"] = result["Pending_Cutting"].clip(lower=0).astype(int)
    elif "PO_Qty_Ordered" in result.columns:
        result["PO_Pipeline_Total"] = result["PO_Qty_Ordered"].clip(lower=0).astype(int)
    else:
        result["PO_Pipeline_Total"] = 0

    for c in ["PO_Qty_Ordered", "Pending_Cutting", "Balance_to_Dispatch", "PO_Pipeline_Total"]:
        if c in result.columns:
            result[c] = result[c].clip(lower=0).astype(int)

    return result


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
    try:
        raw = _read_raw_po(file_bytes, filename)
    except ValueError:
        # Headerless sheets can cause the Excel header-row scan to fail entirely.
        headerless = _read_headerless_po(file_bytes, filename)
        if headerless is not None:
            raw = headerless
        else:
            raise
    cols = list(raw.columns)

    sku_col = _resolve_sku_column(cols)
    # Some operator exports omit the header row; pandas then treats the first data row as headers
    # (e.g. columns: ["1917YKBLUE-3XL", "New SKU", ..., "130"]). Detect and re-read as headerless.
    if sku_col is not None:
        try:
            _col_names = [str(c).strip() for c in cols]
            _has_numeric_headers = any(
                bool(re.fullmatch(r"\d+(?:\.\d+)?", n)) for n in _col_names
            )
            if _looks_like_oms_sku(sku_col) and _has_numeric_headers:
                headerless = _read_headerless_po(file_bytes, filename)
                if headerless is not None:
                    raw = headerless
                    cols = list(raw.columns)
                    sku_col = "_c0"
        except Exception:
            pass

    if sku_col is None:
        headerless = _read_headerless_po(file_bytes, filename)
        if headerless is not None:
            raw = headerless
            cols = list(raw.columns)
            sku_col = "_c0"
        else:
            raise ValueError(f"Cannot find a SKU column. Columns seen: {cols[:25]}")

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
                and _column_looks_numeric_qty(raw[col])
            ),
            None,
        )
    elif not _column_looks_numeric_qty(raw[ordered_col]):
        ordered_col = None

    fallback_col: Optional[str] = None
    if not any([ordered_col, cutting_col, dispatch_col, total_col]):
        fallback_col = _find_col(
            cols,
            [
                "Balance Qty", "Balance Quantity", "Open Qty", "Open Quantity",
                "Pending Qty", "Pending Quantity", "Units", "Balance",
                "Qty", "Balance to dispatch", "TOTAL BALANCE From Latest Status",
                "Total Balance", "Dispatch Balance", "Pending dispatch",
            ],
        )
        if fallback_col is None:
            fallback_col = _find_col_fuzzy(
                cols, ["balance", "dispatch", "pending", "open qty", "po qty"]
            )
        if fallback_col is None or not _column_looks_numeric_qty(raw[fallback_col]):
            inferred = _infer_quantity_columns(raw, sku_col)
            ordered_col, cutting_col, dispatch_col, total_col = inferred
            fallback_col = None
            if not any([ordered_col, cutting_col, dispatch_col, total_col]):
                headerless = _read_headerless_po(file_bytes, filename)
                if headerless is not None:
                    raw = headerless
                    sku_col = "_c0"
                    ordered_col, cutting_col, dispatch_col, total_col = _infer_quantity_columns(
                        raw, sku_col
                    )
                if not any([ordered_col, cutting_col, dispatch_col, total_col]):
                    raise ValueError(
                        f"Cannot find a quantity/balance column. Columns seen: {cols[:25]}"
                    )

    return _build_po_dataframe(
        raw,
        sku_col,
        ordered_col=ordered_col,
        cutting_col=cutting_col,
        dispatch_col=dispatch_col,
        total_col=total_col,
        fallback_col=fallback_col,
    )
