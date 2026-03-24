"""
SQLite persistence layer for Finance module.
DB path: /data/finance.db (env FINANCE_DB_PATH), fallback ./finance_dev.db for local dev.
"""
import os
import sqlite3
from typing import Optional

DB_PATH = os.environ.get("FINANCE_DB_PATH", "/data/finance.db")


def _connect() -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(DB_PATH)
    except Exception:
        # Fallback for local dev when /data doesn't exist
        conn = sqlite3.connect("./finance_dev.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Called on app startup."""
    conn = _connect()

    # ── Original table ────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT    NOT NULL,
            category    TEXT    NOT NULL,
            description TEXT    NOT NULL DEFAULT '',
            amount      REAL    NOT NULL,
            gst_amount  REAL    NOT NULL DEFAULT 0,
            created_at  TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── Ledger Groups ─────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ledger_groups (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT    NOT NULL UNIQUE,
            parent_group TEXT    DEFAULT '',
            nature       TEXT    NOT NULL DEFAULT 'expense'
        )
    """)

    # ── Ledgers ───────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ledgers (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT    NOT NULL,
            group_id       INTEGER REFERENCES ledger_groups(id),
            group_name     TEXT    DEFAULT '',
            gstin          TEXT    DEFAULT '',
            pan            TEXT    DEFAULT '',
            state          TEXT    DEFAULT '',
            state_code     TEXT    DEFAULT '',
            address        TEXT    DEFAULT '',
            tds_applicable INTEGER DEFAULT 0,
            tds_section    TEXT    DEFAULT '',
            is_active      INTEGER DEFAULT 1,
            created_at     TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── GST Classifications ───────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gst_classifications (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL UNIQUE,
            hsn_sac  TEXT    DEFAULT '',
            gst_rate REAL    DEFAULT 18.0,
            type     TEXT    DEFAULT 'Goods'
        )
    """)

    # ── TDS Sections ──────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tds_sections (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            section          TEXT    NOT NULL UNIQUE,
            description      TEXT    DEFAULT '',
            rate_individual  REAL    DEFAULT 1.0,
            rate_company     REAL    DEFAULT 2.0,
            threshold        REAL    DEFAULT 0
        )
    """)

    # ── Expense Vouchers ──────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS expense_vouchers (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            voucher_no     TEXT    UNIQUE NOT NULL,
            voucher_date   TEXT    NOT NULL,
            voucher_type   TEXT    DEFAULT 'Expense',
            party_name     TEXT    DEFAULT '',
            party_gstin    TEXT    DEFAULT '',
            party_state    TEXT    DEFAULT '',
            bill_no        TEXT    DEFAULT '',
            bill_date      TEXT    DEFAULT '',
            supply_type    TEXT    DEFAULT 'Intra',
            narration      TEXT    DEFAULT '',
            taxable_amount REAL    DEFAULT 0,
            cgst_amount    REAL    DEFAULT 0,
            sgst_amount    REAL    DEFAULT 0,
            igst_amount    REAL    DEFAULT 0,
            tds_section    TEXT    DEFAULT '',
            tds_rate       REAL    DEFAULT 0,
            tds_amount     REAL    DEFAULT 0,
            total_amount   REAL    DEFAULT 0,
            net_payable    REAL    DEFAULT 0,
            created_at     TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── Expense Voucher Lines ─────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS expense_voucher_lines (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            voucher_id   INTEGER NOT NULL REFERENCES expense_vouchers(id) ON DELETE CASCADE,
            expense_head TEXT    NOT NULL,
            description  TEXT    DEFAULT '',
            amount       REAL    DEFAULT 0,
            cost_centre  TEXT    DEFAULT ''
        )
    """)

    # ── Finance Sales Uploads ─────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS finance_sales_uploads (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            platform      TEXT    NOT NULL,
            period        TEXT    NOT NULL,
            filename      TEXT    DEFAULT '',
            total_revenue REAL    DEFAULT 0,
            total_orders  INTEGER DEFAULT 0,
            total_returns REAL    DEFAULT 0,
            net_revenue   REAL    DEFAULT 0,
            is_locked     INTEGER DEFAULT 1,
            uploaded_by   TEXT    DEFAULT '',
            upload_notes  TEXT    DEFAULT '',
            created_at    TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── Voucher Types ─────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS voucher_types (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            name              TEXT NOT NULL UNIQUE,
            voucher_category  TEXT DEFAULT 'Sales',
            abbreviation      TEXT DEFAULT '',
            is_active         INTEGER DEFAULT 1,
            allow_narration   INTEGER DEFAULT 1,
            numbering_method  TEXT DEFAULT 'Auto',
            created_at        TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()

    # ── Migrate expense_vouchers — add payment/bank columns ───────
    for col_ddl in [
        "ALTER TABLE expense_vouchers ADD COLUMN payment_mode TEXT DEFAULT ''",
        "ALTER TABLE expense_vouchers ADD COLUMN bank_ledger TEXT DEFAULT ''",
        "ALTER TABLE expense_vouchers ADD COLUMN cheque_no TEXT DEFAULT ''",
        "ALTER TABLE expense_vouchers ADD COLUMN ref_number TEXT DEFAULT ''",
    ]:
        try:
            conn.execute(col_ddl)
        except Exception:
            pass  # column already exists
    conn.commit()

    # ── Migrate expense_voucher_lines — add is_debit column ───────
    for col_ddl in [
        "ALTER TABLE expense_voucher_lines ADD COLUMN is_debit INTEGER DEFAULT 1",
    ]:
        try:
            conn.execute(col_ddl)
        except Exception:
            pass  # column already exists
    conn.commit()

    # ── Migrate ledgers table — add new columns if missing ────────
    for col_ddl in [
        "ALTER TABLE ledgers ADD COLUMN alias TEXT DEFAULT ''",
        "ALTER TABLE ledgers ADD COLUMN credit_period INTEGER DEFAULT 0",
        "ALTER TABLE ledgers ADD COLUMN maintain_bill_by_bill INTEGER DEFAULT 0",
        "ALTER TABLE ledgers ADD COLUMN is_tcs_applicable INTEGER DEFAULT 0",
        "ALTER TABLE ledgers ADD COLUMN country TEXT DEFAULT 'India'",
        "ALTER TABLE ledgers ADD COLUMN pincode TEXT DEFAULT ''",
        "ALTER TABLE ledgers ADD COLUMN registration_type TEXT DEFAULT ''",
        "ALTER TABLE ledgers ADD COLUMN bank_name TEXT DEFAULT ''",
        "ALTER TABLE ledgers ADD COLUMN bank_account TEXT DEFAULT ''",
        "ALTER TABLE ledgers ADD COLUMN bank_ifsc TEXT DEFAULT ''",
        "ALTER TABLE ledgers ADD COLUMN opening_balance REAL DEFAULT 0",
    ]:
        try:
            conn.execute(col_ddl)
        except Exception:
            pass  # column already exists
    conn.commit()

    # ── Seed ledger_groups ────────────────────────────────────────
    count = conn.execute("SELECT COUNT(*) FROM ledger_groups").fetchone()[0]
    if count == 0:
        seed_groups = [
            ('Direct Expenses',   '', 'expense'),
            ('Indirect Expenses', '', 'expense'),
            ('Direct Income',     '', 'income'),
            ('Indirect Income',   '', 'income'),
            ('Purchase Accounts', '', 'expense'),
            ('Sales Accounts',    '', 'income'),
            ('Sundry Creditors',  '', 'liability'),
            ('Sundry Debtors',    '', 'asset'),
            ('Bank Accounts',     '', 'asset'),
            ('Cash-in-Hand',      '', 'asset'),
            ('Duties & Taxes',    '', 'liability'),
            ('Capital Account',   '', 'liability'),
        ]
        conn.executemany(
            "INSERT INTO ledger_groups (name, parent_group, nature) VALUES (?,?,?)",
            seed_groups,
        )
        conn.commit()

    # ── Seed additional Tally ledger groups (INSERT OR IGNORE) ────
    extra_groups = [
        ('Reserve & Surplus',           'Capital Account',   'liability'),
        ('Bank Overdraft',              'Loans Liabilities', 'liability'),
        ('Secured Loans',               'Loans Liabilities', 'liability'),
        ('Unsecured Loans',             'Loans Liabilities', 'liability'),
        ('Loans Liabilities',           '',                  'liability'),
        ('Fixed Assets',                '',                  'asset'),
        ('Investments',                 '',                  'asset'),
        ('Current Assets',              '',                  'asset'),
        ('Current Liabilities',         '',                  'liability'),
        ('Deposit Assets',              'Current Assets',    'asset'),
        ('Loans & Advances (Assets)',   'Current Assets',    'asset'),
        ('Stock in Hand',               'Current Assets',    'asset'),
        ('Provision',                   'Current Liabilities', 'liability'),
        ('Miscellaneous Expenses',      '',                  'asset'),
        ('Suspense Account',            '',                  'asset'),
        ('Branch/Division',             '',                  'asset'),
    ]
    for grp in extra_groups:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO ledger_groups (name, parent_group, nature) VALUES (?,?,?)",
                grp,
            )
        except Exception:
            pass
    conn.commit()

    # ── Seed gst_classifications ──────────────────────────────────
    count = conn.execute("SELECT COUNT(*) FROM gst_classifications").fetchone()[0]
    if count == 0:
        seed_gst = [
            ('Nil Rated',      '', 0.0,  'Goods'),
            ('Exempt',         '', 0.0,  'Services'),
            ('5% Goods',       '', 5.0,  'Goods'),
            ('5% Services',    '', 5.0,  'Services'),
            ('12% Goods',      '', 12.0, 'Goods'),
            ('12% Services',   '', 12.0, 'Services'),
            ('18% Goods',      '', 18.0, 'Goods'),
            ('18% Services',   '', 18.0, 'Services'),
            ('28% Goods',      '', 28.0, 'Goods'),
        ]
        conn.executemany(
            "INSERT INTO gst_classifications (name, hsn_sac, gst_rate, type) VALUES (?,?,?,?)",
            seed_gst,
        )
        conn.commit()

    # ── Seed tds_sections ─────────────────────────────────────────
    count = conn.execute("SELECT COUNT(*) FROM tds_sections").fetchone()[0]
    if count == 0:
        seed_tds = [
            ('194C', 'Contractor/Sub-contractor',  1.0,  2.0,  30000),
            ('194J', 'Professional/Technical',     10.0, 10.0, 30000),
            ('194H', 'Commission/Brokerage',        5.0,  5.0, 15000),
            ('194I', 'Rent',                       10.0, 10.0, 240000),
            ('194B', 'Lottery/Game',               30.0, 30.0, 10000),
            ('194D', 'Insurance Commission',        5.0, 10.0, 15000),
        ]
        conn.executemany(
            "INSERT INTO tds_sections (section, description, rate_individual, rate_company, threshold) VALUES (?,?,?,?,?)",
            seed_tds,
        )
        conn.commit()

    # ── Seed voucher_types ────────────────────────────────────────
    count = conn.execute("SELECT COUNT(*) FROM voucher_types").fetchone()[0]
    if count == 0:
        seed_vt = [
            ('Amazon Sales',     'Sales',    'Sale', 1, 1, 'Manual'),
            ('Myntra Sales',     'Sales',    'Myn',  1, 1, 'Manual'),
            ('Meesho Sales',     'Sales',    'Mee',  1, 1, 'Manual'),
            ('Flipkart Sales',   'Sales',    'Flip', 1, 1, 'Manual'),
            ('Cash Payment',     'Payment',  'CPay', 1, 1, 'Auto'),
            ('Bank Payment',     'Payment',  'BPay', 1, 1, 'Auto'),
            ('Bank Receipt',     'Receipt',  'BRec', 1, 1, 'Auto'),
            ('Purchase Invoice', 'Purchase', 'Pur',  1, 1, 'Manual'),
            ('Journal',          'Journal',  'Jnl',  1, 1, 'Auto'),
            ('Contra',           'Contra',   'Con',  1, 1, 'Auto'),
        ]
        conn.executemany(
            """INSERT INTO voucher_types
               (name, voucher_category, abbreviation, is_active, allow_narration, numbering_method)
               VALUES (?,?,?,?,?,?)""",
            seed_vt,
        )
        conn.commit()

    # ── Seed default ledgers (INSERT OR IGNORE) ───────────────────
    def _grp_id(name):
        row = conn.execute("SELECT id FROM ledger_groups WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

    seed_ledgers = [
        # name, group_name, gstin, tds_applicable, tds_section, state
        ('Cash',                         'Cash-in-Hand',      '', 0, '', 'Rajasthan'),
        ('Petty Cash',                   'Cash-in-Hand',      '', 0, '', 'Rajasthan'),
        ('SBI Current Account',          'Bank Accounts',     '', 0, '', 'Rajasthan'),
        ('HDFC Current Account',         'Bank Accounts',     '', 0, '', 'Rajasthan'),
        ('ICICI Current Account',        'Bank Accounts',     '', 0, '', 'Rajasthan'),
        # Purchases
        ('Fabric Purchase A/c',          'Purchase Accounts', '', 0, '', ''),
        ('Accessories Purchase A/c',     'Purchase Accounts', '', 0, '', ''),
        ('Packing Material Purchase',    'Purchase Accounts', '', 0, '', ''),
        ('Job Work Charges',             'Direct Expenses',   '', 1, '194C', ''),
        ('Contract Labour',              'Direct Expenses',   '', 1, '194C', ''),
        ('Freight & Cartage',            'Direct Expenses',   '', 0, '', ''),
        ('Loading & Unloading',          'Direct Expenses',   '', 0, '', ''),
        # Indirect Expenses
        ('Rent A/c',                     'Indirect Expenses', '', 1, '194I', ''),
        ('Electricity Charges',          'Indirect Expenses', '', 0, '', ''),
        ('Telephone & Internet',         'Indirect Expenses', '', 0, '', ''),
        ('Repair & Maintenance',         'Indirect Expenses', '', 0, '', ''),
        ('Printing & Stationery',        'Indirect Expenses', '', 0, '', ''),
        ('Postage & Courier',            'Indirect Expenses', '', 0, '', ''),
        ('Advertisement & Marketing',    'Indirect Expenses', '', 0, '', ''),
        ('Professional Charges',         'Indirect Expenses', '', 1, '194J', ''),
        ('Audit Fees',                   'Indirect Expenses', '', 1, '194J', ''),
        ('Bank Charges',                 'Indirect Expenses', '', 0, '', ''),
        ('Vehicle Running Expenses',     'Indirect Expenses', '', 0, '', ''),
        ('Office Expenses',              'Indirect Expenses', '', 0, '', ''),
        ('Staff Welfare',                'Indirect Expenses', '', 0, '', ''),
        ('Travelling Expenses',          'Indirect Expenses', '', 0, '', ''),
        ('Commission Paid',              'Indirect Expenses', '', 1, '194H', ''),
        # Duties & Taxes
        ('CGST Payable',                 'Duties & Taxes',    '', 0, '', ''),
        ('SGST Payable',                 'Duties & Taxes',    '', 0, '', ''),
        ('IGST Payable',                 'Duties & Taxes',    '', 0, '', ''),
        ('TDS Payable',                  'Duties & Taxes',    '', 0, '', ''),
        ('CGST Receivable (ITC)',        'Current Assets',    '', 0, '', ''),
        ('SGST Receivable (ITC)',        'Current Assets',    '', 0, '', ''),
        ('IGST Receivable (ITC)',        'Current Assets',    '', 0, '', ''),
        # Sales
        ('Amazon Sales A/c',             'Sales Accounts',    '', 0, '', ''),
        ('Myntra Sales A/c',             'Sales Accounts',    '', 0, '', ''),
        ('Meesho Sales A/c',             'Sales Accounts',    '', 0, '', ''),
        ('Flipkart Sales A/c',           'Sales Accounts',    '', 0, '', ''),
        ('Snapdeal Sales A/c',           'Sales Accounts',    '', 0, '', ''),
        # Capital
        ('Capital A/c',                  'Capital Account',   '', 0, '', ''),
        ('Drawing A/c',                  'Capital Account',   '', 0, '', ''),
    ]
    for (name, group_name, gstin, tds_app, tds_sec, state) in seed_ledgers:
        exists = conn.execute("SELECT 1 FROM ledgers WHERE name=?", (name,)).fetchone()
        if not exists:
            gid = _grp_id(group_name)
            conn.execute(
                """INSERT OR IGNORE INTO ledgers
                   (name, alias, group_id, group_name, gstin, tds_applicable, tds_section,
                    is_tcs_applicable, state, country, is_active)
                   VALUES (?,?,?,?,?,?,?,0,?,?,1)""",
                (name, '', gid, group_name, gstin, tds_app, tds_sec, state, 'India'),
            )
    conn.commit()

    conn.close()


# ── Original expense CRUD ─────────────────────────────────────────

def add_expense(
    date: str,
    category: str,
    description: str,
    amount: float,
    gst_amount: float = 0.0,
) -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO expenses (date, category, description, amount, gst_amount) VALUES (?,?,?,?,?)",
        (date, category, description, amount, gst_amount),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def list_expenses(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[dict]:
    conn = _connect()
    query = "SELECT * FROM expenses WHERE 1=1"
    params: list = []
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date DESC, id DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_expense(expense_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── Ledger Groups CRUD ────────────────────────────────────────────

def list_ledger_groups() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM ledger_groups ORDER BY nature, name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_ledger_group(name: str, parent_group: str, nature: str) -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO ledger_groups (name, parent_group, nature) VALUES (?,?,?)",
        (name, parent_group or '', nature or 'expense'),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def delete_ledger_group(group_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM ledger_groups WHERE id = ?", (group_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── Ledgers CRUD ──────────────────────────────────────────────────

def list_ledgers(group_id: Optional[int] = None, search: Optional[str] = None) -> list[dict]:
    conn = _connect()
    query = "SELECT * FROM ledgers WHERE is_active = 1"
    params: list = []
    if group_id is not None:
        query += " AND group_id = ?"
        params.append(group_id)
    if search:
        query += " AND name LIKE ?"
        params.append(f"%{search}%")
    query += " ORDER BY name"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_ledger(
    name: str,
    group_id: Optional[int],
    group_name: str,
    gstin: str,
    pan: str,
    state: str,
    state_code: str,
    address: str,
    tds_applicable: int,
    tds_section: str,
    alias: str = '',
    credit_period: int = 0,
    maintain_bill_by_bill: int = 0,
    is_tcs_applicable: int = 0,
    country: str = 'India',
    pincode: str = '',
    registration_type: str = '',
    bank_name: str = '',
    bank_account: str = '',
    bank_ifsc: str = '',
    opening_balance: float = 0.0,
) -> int:
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO ledgers
           (name, group_id, group_name, gstin, pan, state, state_code, address,
            tds_applicable, tds_section, alias, credit_period, maintain_bill_by_bill,
            is_tcs_applicable, country, pincode, registration_type,
            bank_name, bank_account, bank_ifsc, opening_balance)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            name,
            group_id,
            group_name or '',
            gstin or '',
            pan or '',
            state or '',
            state_code or '',
            address or '',
            tds_applicable or 0,
            tds_section or '',
            alias or '',
            credit_period or 0,
            maintain_bill_by_bill or 0,
            is_tcs_applicable or 0,
            country or 'India',
            pincode or '',
            registration_type or '',
            bank_name or '',
            bank_account or '',
            bank_ifsc or '',
            opening_balance or 0.0,
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def update_ledger(ledger_id: int, **fields) -> bool:
    if not fields:
        return False
    allowed = {
        'name', 'group_id', 'group_name', 'gstin', 'pan', 'state',
        'state_code', 'address', 'tds_applicable', 'tds_section', 'is_active',
        'alias', 'credit_period', 'maintain_bill_by_bill', 'is_tcs_applicable',
        'country', 'pincode', 'registration_type', 'bank_name', 'bank_account',
        'bank_ifsc', 'opening_balance',
    }
    safe = {k: v for k, v in fields.items() if k in allowed}
    if not safe:
        return False
    set_clause = ', '.join(f"{k} = ?" for k in safe)
    values = list(safe.values()) + [ledger_id]
    conn = _connect()
    cur = conn.execute(f"UPDATE ledgers SET {set_clause} WHERE id = ?", values)
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def delete_ledger(ledger_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM ledgers WHERE id = ?", (ledger_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── GST Classifications CRUD ──────────────────────────────────────

def list_gst_classifications() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM gst_classifications ORDER BY gst_rate, name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_gst_classification(name: str, hsn_sac: str, gst_rate: float, type_: str) -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO gst_classifications (name, hsn_sac, gst_rate, type) VALUES (?,?,?,?)",
        (name, hsn_sac or '', gst_rate or 0.0, type_ or 'Goods'),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def delete_gst_classification(classification_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM gst_classifications WHERE id = ?", (classification_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── TDS Sections CRUD ─────────────────────────────────────────────

def list_tds_sections() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM tds_sections ORDER BY section"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_tds_section(
    section: str,
    description: str,
    rate_individual: float,
    rate_company: float,
    threshold: float,
) -> int:
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO tds_sections (section, description, rate_individual, rate_company, threshold)
           VALUES (?,?,?,?,?)""",
        (section, description or '', rate_individual or 0.0, rate_company or 0.0, threshold or 0.0),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def delete_tds_section(section_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM tds_sections WHERE id = ?", (section_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── Expense Vouchers CRUD ─────────────────────────────────────────

def _next_voucher_no(conn: sqlite3.Connection, voucher_type: str) -> str:
    prefix_map = {
        'JWO Payment':      'JWO',
        'Payment':          'PAY',
        'Receipt':          'REC',
        'Journal':          'JNL',
        'Contra':           'CON',
        'Purchase Invoice': 'PUR',
        'Sales Invoice':    'SAL',
    }
    prefix = prefix_map.get(voucher_type) or 'EXP'
    row = conn.execute(
        "SELECT COUNT(*) FROM expense_vouchers WHERE voucher_type = ?",
        (voucher_type,),
    ).fetchone()
    seq = (row[0] or 0) + 1
    return f"{prefix}-{seq:04d}"


def list_expense_vouchers(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    voucher_type: Optional[str] = None,
) -> list[dict]:
    conn = _connect()
    query = "SELECT * FROM expense_vouchers WHERE 1=1"
    params: list = []
    if start_date:
        query += " AND voucher_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND voucher_date <= ?"
        params.append(end_date)
    if voucher_type:
        query += " AND voucher_type = ?"
        params.append(voucher_type)
    query += " ORDER BY voucher_date DESC, id DESC"
    vouchers = [dict(r) for r in conn.execute(query, params).fetchall()]

    for v in vouchers:
        lines = conn.execute(
            "SELECT * FROM expense_voucher_lines WHERE voucher_id = ? ORDER BY id",
            (v['id'],),
        ).fetchall()
        v['lines'] = [dict(ln) for ln in lines]

    conn.close()
    return vouchers


def get_expense_voucher(voucher_id: int) -> Optional[dict]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM expense_vouchers WHERE id = ?", (voucher_id,)
    ).fetchone()
    if not row:
        conn.close()
        return None
    v = dict(row)
    lines = conn.execute(
        "SELECT * FROM expense_voucher_lines WHERE voucher_id = ? ORDER BY id",
        (voucher_id,),
    ).fetchall()
    v['lines'] = [dict(ln) for ln in lines]
    conn.close()
    return v


def create_expense_voucher(data: dict) -> str:
    conn = _connect()
    voucher_type = data.get('voucher_type') or 'Expense'
    voucher_no = _next_voucher_no(conn, voucher_type)
    conn.execute(
        """INSERT INTO expense_vouchers
           (voucher_no, voucher_date, voucher_type, party_name, party_gstin, party_state,
            bill_no, bill_date, supply_type, narration,
            taxable_amount, cgst_amount, sgst_amount, igst_amount,
            tds_section, tds_rate, tds_amount, total_amount, net_payable,
            payment_mode, bank_ledger, cheque_no, ref_number)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            voucher_no,
            data.get('voucher_date') or '',
            voucher_type,
            data.get('party_name') or '',
            data.get('party_gstin') or '',
            data.get('party_state') or '',
            data.get('bill_no') or '',
            data.get('bill_date') or '',
            data.get('supply_type') or 'Intra',
            data.get('narration') or '',
            data.get('taxable_amount') or 0,
            data.get('cgst_amount') or 0,
            data.get('sgst_amount') or 0,
            data.get('igst_amount') or 0,
            data.get('tds_section') or '',
            data.get('tds_rate') or 0,
            data.get('tds_amount') or 0,
            data.get('total_amount') or 0,
            data.get('net_payable') or 0,
            data.get('payment_mode') or '',
            data.get('bank_ledger') or '',
            data.get('cheque_no') or '',
            data.get('ref_number') or '',
        ),
    )
    conn.commit()
    vid = conn.execute(
        "SELECT id FROM expense_vouchers WHERE voucher_no = ?", (voucher_no,)
    ).fetchone()[0]

    for line in data.get('lines') or []:
        conn.execute(
            """INSERT INTO expense_voucher_lines
               (voucher_id, expense_head, description, amount, cost_centre, is_debit)
               VALUES (?,?,?,?,?,?)""",
            (
                vid,
                line.get('expense_head') or '',
                line.get('description') or '',
                line.get('amount') or 0,
                line.get('cost_centre') or '',
                line.get('is_debit') if line.get('is_debit') is not None else 1,
            ),
        )
    conn.commit()
    conn.close()
    return voucher_no


def delete_expense_voucher(voucher_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM expense_vouchers WHERE id = ?", (voucher_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── Finance Sales Uploads CRUD ────────────────────────────────────

def list_finance_sales_uploads(
    platform: Optional[str] = None,
    period: Optional[str] = None,
) -> list[dict]:
    conn = _connect()
    query = "SELECT * FROM finance_sales_uploads WHERE 1=1"
    params: list = []
    if platform:
        query += " AND platform = ?"
        params.append(platform)
    if period:
        query += " AND period = ?"
        params.append(period)
    query += " ORDER BY period DESC, id DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_finance_sales_upload(data: dict) -> int:
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO finance_sales_uploads
           (platform, period, filename, total_revenue, total_orders,
            total_returns, net_revenue, uploaded_by, upload_notes)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            data.get('platform') or '',
            data.get('period') or '',
            data.get('filename') or '',
            data.get('total_revenue') or 0,
            data.get('total_orders') or 0,
            data.get('total_returns') or 0,
            data.get('net_revenue') or 0,
            data.get('uploaded_by') or '',
            data.get('upload_notes') or '',
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def list_vouchers(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    voucher_type: Optional[str] = None,
) -> list[dict]:
    """Return ALL vouchers with their lines, filtered by type/date."""
    return list_expense_vouchers(
        start_date=start_date,
        end_date=end_date,
        voucher_type=voucher_type,
    )


def get_voucher_summary_by_date(date: str) -> list[dict]:
    """Return all vouchers for a specific date (for Day Book)."""
    conn = _connect()
    vouchers = [dict(r) for r in conn.execute(
        "SELECT * FROM expense_vouchers WHERE voucher_date = ? ORDER BY id ASC",
        (date,),
    ).fetchall()]
    for v in vouchers:
        lines = conn.execute(
            "SELECT * FROM expense_voucher_lines WHERE voucher_id = ? ORDER BY id",
            (v['id'],),
        ).fetchall()
        v['lines'] = [dict(ln) for ln in lines]
    conn.close()
    return vouchers


def get_gstr3b_data(start_date: str, end_date: str) -> dict:
    """Compute GSTR3B data from expense_vouchers."""
    conn = _connect()
    # Outward supplies (Sales Invoice)
    out_rows = conn.execute(
        """SELECT SUM(taxable_amount) AS taxable, SUM(cgst_amount) AS cgst,
                  SUM(sgst_amount) AS sgst, SUM(igst_amount) AS igst,
                  SUM(total_amount) AS total
           FROM expense_vouchers
           WHERE voucher_type = 'Sales Invoice'
             AND voucher_date >= ? AND voucher_date <= ?""",
        (start_date, end_date),
    ).fetchone()
    outward = {
        'taxable': round(out_rows['taxable'] or 0, 2),
        'cgst':    round(out_rows['cgst']    or 0, 2),
        'sgst':    round(out_rows['sgst']    or 0, 2),
        'igst':    round(out_rows['igst']    or 0, 2),
        'total':   round(out_rows['total']   or 0, 2),
    }
    # Inward ITC (Purchase Invoice, Expense, JWO Payment)
    itc_rows = conn.execute(
        """SELECT SUM(taxable_amount) AS taxable, SUM(cgst_amount) AS cgst,
                  SUM(sgst_amount) AS sgst, SUM(igst_amount) AS igst,
                  SUM(total_amount) AS total
           FROM expense_vouchers
           WHERE voucher_type IN ('Purchase Invoice', 'Expense', 'JWO Payment')
             AND voucher_date >= ? AND voucher_date <= ?""",
        (start_date, end_date),
    ).fetchone()
    inward_itc = {
        'taxable': round(itc_rows['taxable'] or 0, 2),
        'cgst':    round(itc_rows['cgst']    or 0, 2),
        'sgst':    round(itc_rows['sgst']    or 0, 2),
        'igst':    round(itc_rows['igst']    or 0, 2),
        'total':   round(itc_rows['total']   or 0, 2),
    }
    # Line-by-line voucher breakdown for the period
    breakdown_rows = conn.execute(
        """SELECT voucher_no, voucher_date, voucher_type, party_name,
                  taxable_amount, cgst_amount, sgst_amount, igst_amount, total_amount
           FROM expense_vouchers
           WHERE voucher_type IN ('Sales Invoice','Purchase Invoice','Expense','JWO Payment')
             AND voucher_date >= ? AND voucher_date <= ?
           ORDER BY voucher_date ASC, id ASC""",
        (start_date, end_date),
    ).fetchall()
    breakdown = [dict(r) for r in breakdown_rows]
    conn.close()
    net_cgst  = round(outward['cgst'] - inward_itc['cgst'], 2)
    net_sgst  = round(outward['sgst'] - inward_itc['sgst'], 2)
    net_igst  = round(outward['igst'] - inward_itc['igst'], 2)
    net_total = round(net_cgst + net_sgst + net_igst, 2)
    return {
        'outward':    outward,
        'inward_itc': inward_itc,
        'net_cgst':   net_cgst,
        'net_sgst':   net_sgst,
        'net_igst':   net_igst,
        'net_total':  net_total,
        'breakdown':  breakdown,
    }


def get_ledger_balances() -> list[dict]:
    """Sum of payments/receipts per ledger (party_name)."""
    conn = _connect()
    rows = conn.execute(
        """SELECT party_name,
                  SUM(CASE WHEN voucher_type IN ('Payment','JWO Payment','Expense','Purchase Invoice') THEN net_payable ELSE 0 END) AS total_payments,
                  SUM(CASE WHEN voucher_type IN ('Receipt','Sales Invoice') THEN total_amount ELSE 0 END) AS total_receipts,
                  COUNT(*) AS voucher_count
           FROM expense_vouchers
           WHERE party_name != ''
           GROUP BY party_name
           ORDER BY party_name""",
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_finance_sales_upload(upload_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM finance_sales_uploads WHERE id = ?", (upload_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── Voucher Types CRUD ────────────────────────────────────────────

def list_voucher_types(category: Optional[str] = None) -> list[dict]:
    conn = _connect()
    query = "SELECT * FROM voucher_types WHERE 1=1"
    params: list = []
    if category:
        query += " AND voucher_category = ?"
        params.append(category)
    query += " ORDER BY voucher_category, name"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_voucher_type(
    name: str,
    voucher_category: str,
    abbreviation: str,
    allow_narration: int,
    numbering_method: str,
) -> int:
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO voucher_types
           (name, voucher_category, abbreviation, allow_narration, numbering_method)
           VALUES (?,?,?,?,?)""",
        (
            name,
            voucher_category or 'Sales',
            abbreviation or '',
            allow_narration if allow_narration is not None else 1,
            numbering_method or 'Auto',
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def update_voucher_type(voucher_type_id: int, **fields) -> bool:
    if not fields:
        return False
    allowed = {'name', 'voucher_category', 'abbreviation', 'is_active', 'allow_narration', 'numbering_method'}
    safe = {k: v for k, v in fields.items() if k in allowed}
    if not safe:
        return False
    set_clause = ', '.join(f"{k} = ?" for k in safe)
    values = list(safe.values()) + [voucher_type_id]
    conn = _connect()
    cur = conn.execute(f"UPDATE voucher_types SET {set_clause} WHERE id = ?", values)
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def delete_voucher_type(voucher_type_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM voucher_types WHERE id = ?", (voucher_type_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted
