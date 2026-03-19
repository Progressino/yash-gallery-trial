"""
SQLite persistence layer for Item Master & BOM module.
DB path: /data/items.db (env ITEM_DB_PATH), fallback ./items_dev.db for local dev.
"""
import json
import os
import sqlite3
from typing import Optional

DB_PATH = os.environ.get("ITEM_DB_PATH", "/data/items.db")

_DEFAULT_TYPES = [
    ("Finished Goods",      "FG"),
    ("Semi-Finished Goods", "SFG"),
    ("Raw Material",        "RM"),
    ("Accessories",         "ACC"),
    ("Packing Materials",   "PKG"),
    ("Fuel & Lubricants",   "FUEL"),
    ("Service / Process",   "SVC"),
]

_DEFAULT_SIZE_GROUPS = [
    ("Standard",  ["S", "M", "L", "XL", "XXL", "3XL"]),
    ("Extended",  ["3XL", "4XL", "5XL", "6XL", "7XL", "8XL"]),
    ("Kids",      ["2Y", "4Y", "6Y", "8Y", "10Y", "12Y"]),
]

_DEFAULT_ROUTING = [
    ("Cutting",   "Fabric cutting as per pattern", 1),
    ("Printing",  "Screen / digital printing",     2),
    ("Stitching", "Assembly and stitching",        3),
    ("Finishing", "Quality check and ironing",     4),
    ("Packing",   "Tagging and packing",           5),
]


# ── Connection ────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(DB_PATH)
    except Exception:
        conn = sqlite3.connect("./items_dev.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables and seed default data. Called on app startup."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS item_types (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            code TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS size_groups (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name  TEXT NOT NULL UNIQUE,
            sizes TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS routing_steps (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            sort_order  INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS items (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            item_code      TEXT NOT NULL UNIQUE,
            item_name      TEXT NOT NULL,
            item_type_id   INTEGER NOT NULL REFERENCES item_types(id),
            hsn_code       TEXT DEFAULT '',
            season         TEXT DEFAULT '',
            merchant_code  TEXT DEFAULT '',
            selling_price  REAL DEFAULT 0,
            purchase_price REAL DEFAULT 0,
            parent_id      INTEGER REFERENCES items(id),
            size_label     TEXT DEFAULT '',
            launch_date    TEXT DEFAULT '',
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS bom_headers (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
            bom_name   TEXT NOT NULL DEFAULT 'Default',
            applies_to TEXT NOT NULL DEFAULT 'all',
            is_default INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS bom_lines (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            bom_id            INTEGER NOT NULL REFERENCES bom_headers(id) ON DELETE CASCADE,
            component_item_id INTEGER REFERENCES items(id),
            component_name    TEXT NOT NULL,
            component_type    TEXT NOT NULL DEFAULT 'RM',
            quantity          REAL NOT NULL DEFAULT 1,
            unit              TEXT DEFAULT 'PCS',
            rate              REAL DEFAULT 0,
            process_id        INTEGER REFERENCES routing_steps(id),
            shrinkage_pct     REAL DEFAULT 0,
            wastage_pct       REAL DEFAULT 0,
            remarks           TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS item_routing (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
            step_id    INTEGER NOT NULL REFERENCES routing_steps(id),
            sort_order INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS merchants (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant_code TEXT NOT NULL UNIQUE,
            merchant_name TEXT NOT NULL,
            created_at    TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS buyers (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            buyer_code TEXT NOT NULL UNIQUE,
            buyer_name TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS item_buyer_packaging (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id           INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
            buyer_id          INTEGER NOT NULL REFERENCES buyers(id) ON DELETE CASCADE,
            packaging_item_id INTEGER NOT NULL REFERENCES items(id),
            quantity          REAL NOT NULL DEFAULT 1,
            unit              TEXT DEFAULT 'PCS',
            remarks           TEXT DEFAULT '',
            created_at        TEXT DEFAULT (datetime('now'))
        );
    """)

    # Migrate existing bom_headers table — add columns if missing
    for col_ddl in [
        "ALTER TABLE bom_headers ADD COLUMN is_certified INTEGER DEFAULT 0",
        "ALTER TABLE bom_headers ADD COLUMN certified_by TEXT DEFAULT ''",
        "ALTER TABLE bom_headers ADD COLUMN certified_at TEXT DEFAULT ''",
        "ALTER TABLE bom_headers ADD COLUMN cmt_cost REAL DEFAULT 0",
        "ALTER TABLE bom_headers ADD COLUMN other_cost REAL DEFAULT 0",
    ]:
        try:
            conn.execute(col_ddl)
        except Exception:
            pass  # column already exists

    # Seed item types
    for name, code in _DEFAULT_TYPES:
        conn.execute(
            "INSERT OR IGNORE INTO item_types (name, code) VALUES (?, ?)", (name, code)
        )

    # Seed size groups
    for name, sizes in _DEFAULT_SIZE_GROUPS:
        conn.execute(
            "INSERT OR IGNORE INTO size_groups (name, sizes) VALUES (?, ?)",
            (name, json.dumps(sizes)),
        )

    # Seed routing steps
    for name, desc, order in _DEFAULT_ROUTING:
        conn.execute(
            "INSERT OR IGNORE INTO routing_steps (name, description, sort_order) VALUES (?, ?, ?)",
            (name, desc, order),
        )

    conn.commit()
    conn.close()


# ── Item Types ────────────────────────────────────────────────────────────────

def list_item_types() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM item_types ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_item_type(name: str, code: str) -> int:
    conn = _connect()
    cur = conn.execute("INSERT INTO item_types (name, code) VALUES (?, ?)", (name, code.upper()))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


# ── Size Groups ───────────────────────────────────────────────────────────────

def list_size_groups() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM size_groups ORDER BY id").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["sizes"] = json.loads(d["sizes"])
        result.append(d)
    return result


def create_size_group(name: str, sizes: list[str]) -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO size_groups (name, sizes) VALUES (?, ?)", (name, json.dumps(sizes))
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


# ── Routing Steps ─────────────────────────────────────────────────────────────

def list_routing_steps() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM routing_steps ORDER BY sort_order, id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_routing_step(name: str, description: str = "", sort_order: int = 0) -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO routing_steps (name, description, sort_order) VALUES (?, ?, ?)",
        (name, description, sort_order),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def delete_routing_step(step_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM routing_steps WHERE id = ?", (step_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── Merchants ──────────────────────────────────────────────────────────────────

def list_merchants() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM merchants ORDER BY merchant_name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_merchant(merchant_code: str, merchant_name: str) -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO merchants (merchant_code, merchant_name) VALUES (?, ?)",
        (merchant_code.strip().upper(), merchant_name.strip()),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def delete_merchant(merchant_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM merchants WHERE id = ?", (merchant_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── Buyers ────────────────────────────────────────────────────────────────────

def list_buyers() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM buyers ORDER BY buyer_name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_buyer(buyer_code: str, buyer_name: str) -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO buyers (buyer_code, buyer_name) VALUES (?, ?)",
        (buyer_code.strip().upper(), buyer_name.strip()),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def delete_buyer(buyer_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM buyers WHERE id = ?", (buyer_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── Buyer Packaging ───────────────────────────────────────────────────────────

def list_item_packaging(item_id: int) -> list[dict]:
    """All packaging lines for an item, across all buyers."""
    conn = _connect()
    rows = conn.execute(
        """SELECT p.*, b.buyer_code, b.buyer_name,
                  i.item_code AS pkg_item_code, i.item_name AS pkg_item_name
           FROM item_buyer_packaging p
           JOIN buyers b ON b.id = p.buyer_id
           JOIN items  i ON i.id = p.packaging_item_id
           WHERE p.item_id = ?
           ORDER BY b.buyer_name, p.id""",
        (item_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_buyer_packaging(item_id: int, buyer_id: int) -> list[dict]:
    """Packaging lines for a specific item + buyer."""
    conn = _connect()
    rows = conn.execute(
        """SELECT p.*, i.item_code AS pkg_item_code, i.item_name AS pkg_item_name,
                  i.item_type_code AS pkg_item_type
           FROM item_buyer_packaging p
           JOIN items i ON i.id = p.packaging_item_id
           WHERE p.item_id = ? AND p.buyer_id = ?
           ORDER BY p.id""",
        (item_id, buyer_id),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_packaging_line(
    item_id: int,
    buyer_id: int,
    packaging_item_id: int,
    quantity: float = 1.0,
    unit: str = "PCS",
    remarks: str = "",
) -> int:
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO item_buyer_packaging
           (item_id, buyer_id, packaging_item_id, quantity, unit, remarks)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (item_id, buyer_id, packaging_item_id, quantity, unit, remarks),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def delete_packaging_line(line_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM item_buyer_packaging WHERE id = ?", (line_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── Items ─────────────────────────────────────────────────────────────────────

def list_items(
    type_id: Optional[int] = None,
    search: Optional[str] = None,
    parent_only: bool = False,
) -> list[dict]:
    conn = _connect()
    q = """
        SELECT i.*, t.name AS item_type_name, t.code AS item_type_code,
               (SELECT COUNT(*) FROM items v WHERE v.parent_id = i.id) AS variant_count
        FROM items i
        JOIN item_types t ON i.item_type_id = t.id
        WHERE 1=1
    """
    params: list = []
    if type_id is not None:
        q += " AND i.item_type_id = ?"
        params.append(type_id)
    if search:
        q += " AND (i.item_code LIKE ? OR i.item_name LIKE ?)"
        params += [f"%{search}%", f"%{search}%"]
    if parent_only:
        q += " AND i.parent_id IS NULL"
    q += " ORDER BY i.created_at DESC, i.id DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_item(item_id: int) -> Optional[dict]:
    conn = _connect()
    row = conn.execute(
        """SELECT i.*, t.name AS item_type_name, t.code AS item_type_code
           FROM items i JOIN item_types t ON i.item_type_id = t.id
           WHERE i.id = ?""",
        (item_id,),
    ).fetchone()
    if row is None:
        conn.close()
        return None
    item = dict(row)
    # Size variants
    variants = conn.execute(
        "SELECT id, item_code, size_label FROM items WHERE parent_id = ? ORDER BY id",
        (item_id,),
    ).fetchall()
    item["variants"] = [dict(v) for v in variants]
    # Routing
    routing = conn.execute(
        """SELECT r.id, r.name, r.sort_order FROM routing_steps r
           JOIN item_routing ir ON ir.step_id = r.id
           WHERE ir.item_id = ? ORDER BY ir.sort_order""",
        (item_id,),
    ).fetchall()
    item["routing"] = [dict(r) for r in routing]
    conn.close()
    return item


def create_item(
    item_code: str,
    item_name: str,
    item_type_id: int,
    hsn_code: str = "",
    season: str = "",
    merchant_code: str = "",
    selling_price: float = 0.0,
    purchase_price: float = 0.0,
    parent_id: Optional[int] = None,
    size_label: str = "",
    launch_date: str = "",
) -> int:
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO items
           (item_code, item_name, item_type_id, hsn_code, season, merchant_code,
            selling_price, purchase_price, parent_id, size_label, launch_date)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (item_code, item_name, item_type_id, hsn_code, season, merchant_code,
         selling_price, purchase_price, parent_id, size_label, launch_date),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def create_size_variants(parent_id: int, sizes: list[str]) -> list[int]:
    """Auto-generate size variant items linked to a parent item."""
    conn = _connect()
    parent = conn.execute(
        "SELECT item_code, item_name, item_type_id, hsn_code, season, merchant_code, selling_price, purchase_price FROM items WHERE id = ?",
        (parent_id,),
    ).fetchone()
    if parent is None:
        conn.close()
        return []
    new_ids: list[int] = []
    for size in sizes:
        variant_code = f"{parent['item_code']}-{size}"
        # Skip if already exists
        existing = conn.execute(
            "SELECT id FROM items WHERE item_code = ?", (variant_code,)
        ).fetchone()
        if existing:
            new_ids.append(existing["id"])
            continue
        cur = conn.execute(
            """INSERT INTO items
               (item_code, item_name, item_type_id, hsn_code, season, merchant_code,
                selling_price, purchase_price, parent_id, size_label)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (variant_code, parent["item_name"], parent["item_type_id"],
             parent["hsn_code"], parent["season"], parent["merchant_code"],
             parent["selling_price"], parent["purchase_price"],
             parent_id, size),
        )
        new_ids.append(cur.lastrowid)
    conn.commit()
    conn.close()
    return new_ids


def update_item(item_id: int, **fields) -> bool:
    allowed = {"item_code", "item_name", "item_type_id", "hsn_code", "season",
               "merchant_code", "selling_price", "purchase_price", "launch_date"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    conn = _connect()
    cur = conn.execute(
        f"UPDATE items SET {set_clause} WHERE id = ?",
        list(updates.values()) + [item_id],
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def delete_item(item_id: int) -> bool:
    conn = _connect()
    # Delete variants first (cascades handle BOM/routing via ON DELETE CASCADE)
    conn.execute("DELETE FROM items WHERE parent_id = ?", (item_id,))
    cur = conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def set_item_routing(item_id: int, step_ids: list[int]) -> None:
    conn = _connect()
    conn.execute("DELETE FROM item_routing WHERE item_id = ?", (item_id,))
    for order, step_id in enumerate(step_ids):
        conn.execute(
            "INSERT INTO item_routing (item_id, step_id, sort_order) VALUES (?, ?, ?)",
            (item_id, step_id, order),
        )
    conn.commit()
    conn.close()


# ── BOM ───────────────────────────────────────────────────────────────────────

def _bom_is_certified(bom_id: int) -> bool:
    conn = _connect()
    row = conn.execute("SELECT is_certified FROM bom_headers WHERE id = ?", (bom_id,)).fetchone()
    conn.close()
    return bool(row and row["is_certified"])


def certify_bom(bom_id: int, certified_by: str = "admin") -> bool:
    conn = _connect()
    from datetime import datetime, timezone
    cur = conn.execute(
        "UPDATE bom_headers SET is_certified=1, certified_by=?, certified_at=? WHERE id=?",
        (certified_by, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), bom_id),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def uncertify_bom(bom_id: int) -> bool:
    conn = _connect()
    cur = conn.execute(
        "UPDATE bom_headers SET is_certified=0, certified_by='', certified_at='' WHERE id=?",
        (bom_id,),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def list_boms(item_id: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM bom_headers WHERE item_id = ? ORDER BY is_default DESC, id",
        (item_id,),
    ).fetchall()
    result = []
    for r in rows:
        bom = dict(r)
        line_count = conn.execute(
            "SELECT COUNT(*) FROM bom_lines WHERE bom_id = ?", (bom["id"],)
        ).fetchone()[0]
        bom["line_count"] = line_count
        result.append(bom)
    conn.close()
    return result


def get_bom_with_lines(bom_id: int) -> Optional[dict]:
    conn = _connect()
    bom_row = conn.execute("SELECT * FROM bom_headers WHERE id = ?", (bom_id,)).fetchone()
    if bom_row is None:
        conn.close()
        return None
    bom = dict(bom_row)
    lines = conn.execute(
        """SELECT bl.*, rs.name AS process_name
           FROM bom_lines bl
           LEFT JOIN routing_steps rs ON rs.id = bl.process_id
           WHERE bl.bom_id = ?
           ORDER BY bl.id""",
        (bom_id,),
    ).fetchall()
    bom["lines"] = [dict(l) for l in lines]
    conn.close()
    return bom


def create_bom(item_id: int, bom_name: str, applies_to: str = "all", is_default: int = 0) -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO bom_headers (item_id, bom_name, applies_to, is_default) VALUES (?, ?, ?, ?)",
        (item_id, bom_name, applies_to, is_default),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def update_bom(bom_id: int, **fields) -> bool:
    if _bom_is_certified(bom_id):
        raise ValueError("BOM is certified and cannot be modified. Uncertify first.")
    allowed = {"bom_name", "applies_to", "is_default", "cmt_cost", "other_cost"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    conn = _connect()
    cur = conn.execute(
        f"UPDATE bom_headers SET {set_clause} WHERE id = ?",
        list(updates.values()) + [bom_id],
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def delete_bom(bom_id: int) -> bool:
    if _bom_is_certified(bom_id):
        raise ValueError("BOM is certified and cannot be deleted. Uncertify first.")
    conn = _connect()
    cur = conn.execute("DELETE FROM bom_headers WHERE id = ?", (bom_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def add_bom_line(
    bom_id: int,
    component_name: str,
    component_type: str = "RM",
    quantity: float = 1.0,
    unit: str = "PCS",
    rate: float = 0.0,
    component_item_id: Optional[int] = None,
    process_id: Optional[int] = None,
    shrinkage_pct: float = 0.0,
    wastage_pct: float = 0.0,
    remarks: str = "",
) -> int:
    if _bom_is_certified(bom_id):
        raise ValueError("BOM is certified and cannot be modified. Uncertify first.")
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO bom_lines
           (bom_id, component_item_id, component_name, component_type,
            quantity, unit, rate, process_id, shrinkage_pct, wastage_pct, remarks)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (bom_id, component_item_id, component_name, component_type,
         quantity, unit, rate, process_id, shrinkage_pct, wastage_pct, remarks),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def update_bom_line(line_id: int, **fields) -> bool:
    # Check if the parent BOM is certified
    conn = _connect()
    row = conn.execute("SELECT bom_id FROM bom_lines WHERE id = ?", (line_id,)).fetchone()
    conn.close()
    if row and _bom_is_certified(row["bom_id"]):
        raise ValueError("BOM is certified and cannot be modified. Uncertify first.")
    allowed = {"component_name", "component_type", "quantity", "unit", "rate",
               "component_item_id", "process_id", "shrinkage_pct", "wastage_pct", "remarks"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    conn = _connect()
    cur = conn.execute(
        f"UPDATE bom_lines SET {set_clause} WHERE id = ?",
        list(updates.values()) + [line_id],
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def delete_bom_line(line_id: int) -> bool:
    conn = _connect()
    row = conn.execute("SELECT bom_id FROM bom_lines WHERE id = ?", (line_id,)).fetchone()
    conn.close()
    if row and _bom_is_certified(row["bom_id"]):
        raise ValueError("BOM is certified and cannot be modified. Uncertify first.")
    conn = _connect()
    cur = conn.execute("DELETE FROM bom_lines WHERE id = ?", (line_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def copy_bom(bom_id: int, target_item_id: int, new_name: str) -> int:
    """Copy a BOM (header + all lines) to a different item."""
    conn = _connect()
    src = conn.execute("SELECT * FROM bom_headers WHERE id = ?", (bom_id,)).fetchone()
    if src is None:
        conn.close()
        raise ValueError(f"BOM {bom_id} not found")
    cur = conn.execute(
        "INSERT INTO bom_headers (item_id, bom_name, applies_to, is_default) VALUES (?, ?, ?, 0)",
        (target_item_id, new_name, src["applies_to"]),
    )
    new_bom_id = cur.lastrowid
    lines = conn.execute("SELECT * FROM bom_lines WHERE bom_id = ?", (bom_id,)).fetchall()
    for l in lines:
        conn.execute(
            """INSERT INTO bom_lines
               (bom_id, component_item_id, component_name, component_type,
                quantity, unit, rate, process_id, shrinkage_pct, wastage_pct, remarks)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (new_bom_id, l["component_item_id"], l["component_name"], l["component_type"],
             l["quantity"], l["unit"], l["rate"], l["process_id"],
             l["shrinkage_pct"], l["wastage_pct"], l["remarks"]),
        )
    conn.commit()
    conn.close()
    return new_bom_id


def bulk_create_items(items: list[dict]) -> dict:
    """
    Bulk insert items from import. Each dict may have 'sizes' list.
    Returns {created, skipped, errors}.
    """
    created = skipped = 0
    errors: list[str] = []
    conn = _connect()
    type_map = {r["code"].upper(): r["id"] for r in
                conn.execute("SELECT id, code FROM item_types").fetchall()}
    conn.close()

    for row in items:
        try:
            code = (row.get("item_code") or "").strip()
            name = (row.get("item_name") or "").strip()
            if not code or not name:
                errors.append(f"Missing item_code or item_name: {row}")
                continue
            type_code = (row.get("item_type") or "FG").strip().upper()
            type_id = type_map.get(type_code) or type_map.get("FG", 1)
            conn = _connect()
            existing = conn.execute("SELECT id FROM items WHERE item_code = ?", (code,)).fetchone()
            conn.close()
            if existing:
                skipped += 1
                continue
            parent_id = create_item(
                item_code=code,
                item_name=name,
                item_type_id=type_id,
                hsn_code=str(row.get("hsn_code") or ""),
                season=str(row.get("season") or ""),
                merchant_code=str(row.get("merchant_code") or ""),
                selling_price=float(row.get("selling_price") or 0),
                purchase_price=float(row.get("purchase_price") or 0),
                launch_date=str(row.get("launch_date") or ""),
            )
            created += 1
            sizes = row.get("sizes") or []
            if isinstance(sizes, str):
                sizes = [s.strip() for s in sizes.split(",") if s.strip()]
            if sizes:
                create_size_variants(parent_id, sizes)
        except Exception as exc:
            errors.append(f"{row.get('item_code', '?')}: {exc}")

    return {"created": created, "skipped": skipped, "errors": errors}


def get_item_stats() -> dict:
    """Return summary counts for the Item Master dashboard."""
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) FROM items WHERE parent_id IS NULL").fetchone()[0]
    by_type = conn.execute(
        """SELECT t.name, t.code, COUNT(i.id) AS cnt
           FROM item_types t
           LEFT JOIN items i ON i.item_type_id = t.id AND i.parent_id IS NULL
           GROUP BY t.id ORDER BY t.id"""
    ).fetchall()
    total_boms = conn.execute("SELECT COUNT(*) FROM bom_headers").fetchone()[0]
    certified_boms = conn.execute(
        "SELECT COUNT(*) FROM bom_headers WHERE is_certified = 1"
    ).fetchone()[0]
    conn.close()
    return {
        "total_items": total,
        "total_boms": total_boms,
        "certified_boms": certified_boms,
        "by_type": [dict(r) for r in by_type],
    }


def list_all_boms() -> list[dict]:
    """Return all BOM headers across all items, with item info."""
    conn = _connect()
    rows = conn.execute(
        """SELECT bh.*, i.item_code, i.item_name,
                  (SELECT COUNT(*) FROM bom_lines bl WHERE bl.bom_id = bh.id) AS line_count,
                  (SELECT SUM(bl.quantity * bl.rate * (1 + bl.shrinkage_pct/100.0) * (1 + bl.wastage_pct/100.0))
                   FROM bom_lines bl WHERE bl.bom_id = bh.id) AS lines_total
           FROM bom_headers bh
           JOIN items i ON i.id = bh.item_id
           ORDER BY bh.item_id, bh.id"""
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["lines_total"] = d["lines_total"] or 0
        d["grand_total"] = d["lines_total"] + (d.get("cmt_cost") or 0) + (d.get("other_cost") or 0)
        result.append(d)
    return result
