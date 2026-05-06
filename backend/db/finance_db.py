"""
SQLite persistence layer for Finance module.
DB path: /data/finance.db (env FINANCE_DB_PATH), fallback ./finance_dev.db for local dev.
"""
import json
import os
import sqlite3
from typing import Any, Optional

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
            company_name  TEXT    DEFAULT '',
            seller_gstin  TEXT    DEFAULT '',
            company_state TEXT    DEFAULT '',
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

    # ── Finance Sales Entries (invoice / order level for Day Book) ──
    conn.execute("""
        CREATE TABLE IF NOT EXISTS finance_sales_entries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            sales_upload_id INTEGER NOT NULL REFERENCES finance_sales_uploads(id) ON DELETE CASCADE,
            platform        TEXT    NOT NULL,
            period          TEXT    NOT NULL,
            voucher_date    TEXT    NOT NULL,
            invoice_no      TEXT    DEFAULT '',
            order_id        TEXT    DEFAULT '',
            party_name      TEXT    DEFAULT '',
            party_gstin     TEXT    DEFAULT '',
            party_state     TEXT    DEFAULT '',
            ship_to_state   TEXT    DEFAULT '',
            taxable_amount  REAL    DEFAULT 0,
            cgst_amount     REAL    DEFAULT 0,
            sgst_amount     REAL    DEFAULT 0,
            igst_amount     REAL    DEFAULT 0,
            total_amount    REAL    DEFAULT 0,
            net_payable     REAL    DEFAULT 0,
            narration       TEXT    DEFAULT '',
            source_filename TEXT    DEFAULT '',
            line_items      TEXT    DEFAULT '[]',
            created_at      TEXT    DEFAULT (datetime('now'))
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS finance_sales_invoice_edits (
            voucher_id INTEGER PRIMARY KEY,
            patch_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT DEFAULT (datetime('now'))
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

    # ── Tally / Accountant P&L ─────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tally_pl (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            fy                TEXT    NOT NULL UNIQUE,
            opening_stock     REAL    NOT NULL DEFAULT 0,
            purchases         REAL    NOT NULL DEFAULT 0,
            direct_expenses   REAL    NOT NULL DEFAULT 0,
            indirect_expenses REAL    NOT NULL DEFAULT 0,
            sales             REAL    NOT NULL DEFAULT 0,
            closing_stock     REAL    NOT NULL DEFAULT 0,
            indirect_incomes  REAL    NOT NULL DEFAULT 0,
            notes             TEXT    DEFAULT '',
            updated_at        TEXT    DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    # ── Migrate finance_sales_uploads — add company context columns ─────
    for col_ddl in [
        "ALTER TABLE finance_sales_uploads ADD COLUMN company_name TEXT DEFAULT ''",
        "ALTER TABLE finance_sales_uploads ADD COLUMN seller_gstin TEXT DEFAULT ''",
        "ALTER TABLE finance_sales_uploads ADD COLUMN company_state TEXT DEFAULT ''",
    ]:
        try:
            conn.execute(col_ddl)
        except Exception:
            pass
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

    # ── Migrate ledger_groups — fix missing parent_group assignments ──
    parent_fixes = [
        ('Bank Accounts',   'Current Assets'),
        ('Cash-in-Hand',    'Current Assets'),
        ('Sundry Debtors',  'Current Assets'),
        ('Sundry Creditors','Current Liabilities'),
        ('Duties & Taxes',  'Current Liabilities'),
    ]
    for grp_name, parent_name in parent_fixes:
        row = conn.execute(
            "SELECT id, parent_group FROM ledger_groups WHERE name=?", (grp_name,)
        ).fetchone()
        if row and not (row['parent_group'] or '').strip():
            conn.execute(
                "UPDATE ledger_groups SET parent_group=? WHERE name=?",
                (parent_name, grp_name),
            )
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

    # ── Seed ledger_groups (primary groups, fresh install only) ─────
    count = conn.execute("SELECT COUNT(*) FROM ledger_groups").fetchone()[0]
    if count == 0:
        seed_groups = [
            ('Direct Expenses',   '', 'expense'),
            ('Indirect Expenses', '', 'expense'),
            ('Direct Incomes',    '', 'income'),
            ('Indirect Incomes',  '', 'income'),
            ('Purchase Accounts', '', 'expense'),
            ('Sales Accounts',    '', 'income'),
            ('Sundry Creditors',  '', 'liability'),
            ('Sundry Debtors',    '', 'asset'),
            ('Bank Accounts',     '', 'asset'),
            ('Cash-in-hand',      '', 'asset'),
            ('Duties & Taxes',    '', 'liability'),
            ('Capital Account',   '', 'liability'),
        ]
        conn.executemany(
            "INSERT INTO ledger_groups (name, parent_group, nature) VALUES (?,?,?)",
            seed_groups,
        )
        conn.commit()

    # ── Rename groups to exact Tally names (idempotent) ───────────
    rename_map = [
        ('Direct Income',             'Direct Incomes'),
        ('Indirect Income',           'Indirect Incomes'),
        ('Cash-in-Hand',              'Cash-in-hand'),
        ('Loans Liabilities',         'Loans (Liability)'),
        ('Miscellaneous Expenses',    'Misc. Expenses (ASSET)'),
        ('Suspense Account',          'Suspense A/c'),
        ('Bank Overdraft',            'Bank OD A/c'),
        ('Deposit Assets',            'Deposits (Asset)'),
        ('Loans & Advances (Assets)', 'Loans & Advances (Asset)'),
        ('Stock in Hand',             'Stock-in-hand'),
        ('Provision',                 'Provisions'),
        ('Reserve & Surplus',         'Reserves & Surplus'),
        ('Branch/Division',           'Branch / Divisions'),
    ]
    for old_name, new_name in rename_map:
        exists_old = conn.execute("SELECT 1 FROM ledger_groups WHERE name=?", (old_name,)).fetchone()
        exists_new = conn.execute("SELECT 1 FROM ledger_groups WHERE name=?", (new_name,)).fetchone()
        if exists_old and not exists_new:
            conn.execute("UPDATE ledger_groups SET name=? WHERE name=?", (new_name, old_name))
    conn.commit()

    # ── Insert all 71 Tally groups (INSERT OR IGNORE) ─────────────
    all_71_groups = [
        # (name, parent_group, nature)
        ('Branch / Divisions',          '',                    'asset'),
        ('Capital Account',             '',                    'liability'),
        ('Reserves & Surplus',          'Capital Account',     'liability'),
        ('Current Assets',              '',                    'asset'),
        ('Bank Accounts',               'Current Assets',      'asset'),
        ('Cash-in-hand',                'Current Assets',      'asset'),
        ('Deposits (Asset)',             'Current Assets',      'asset'),
        ('Loans & Advances (Asset)',     'Current Assets',      'asset'),
        ('Prepaid Expenses',            'Current Assets',      'asset'),
        ('Stock-in-hand',               'Current Assets',      'asset'),
        ('Sundry Debtors',              'Current Assets',      'asset'),
        ('Online Portal',               'Sundry Debtors',      'asset'),
        ('Portal TDS',                  'Sundry Debtors',      'asset'),
        ('Shopfiy',                     'Sundry Debtors',      'asset'),
        ('Current Liabilities',         '',                    'liability'),
        ('Duties & Taxes',              'Current Liabilities', 'liability'),
        ('CGST',                        'Duties & Taxes',      'liability'),
        ('IGST',                        'Duties & Taxes',      'liability'),
        ('RCM Tax',                     'Duties & Taxes',      'liability'),
        ('SGST',                        'Duties & Taxes',      'liability'),
        ('TDS Working',                 'Duties & Taxes',      'liability'),
        ('PF Or ESIC',                  'Current Liabilities', 'liability'),
        ('ESIC',                        'PF Or ESIC',          'liability'),
        ('Esic and PF Other Charges',   'PF Or ESIC',          'liability'),
        ('PF',                          'PF Or ESIC',          'liability'),
        ('Provisions',                  'Current Liabilities', 'liability'),
        ('Salary Exempt',               'Current Liabilities', 'liability'),
        ('Salary & Wages Payable',      'Current Liabilities', 'liability'),
        ('Sundry Creditors',            'Current Liabilities', 'liability'),
        ('Creditors for Expenses',      'Sundry Creditors',    'liability'),
        ('In House Job Work',           'Sundry Creditors',    'liability'),
        ('MSME',                        'Sundry Creditors',    'liability'),
        ('Non MSME',                    'Sundry Creditors',    'liability'),
        ('Transport Contractors',       'Sundry Creditors',    'liability'),
        ('Direct Expenses',             '',                    'expense'),
        ('Electricity Direct Exp.',     'Direct Expenses',     'expense'),
        ('FREIGHT EXPENSES',            'Direct Expenses',     'expense'),
        ('Job Work',                    'Direct Expenses',     'expense'),
        ('PACKING MATERIAL',            'Direct Expenses',     'expense'),
        ('Direct Incomes',              '',                    'income'),
        ('Fixed Assets',                '',                    'asset'),
        ('Computers & Peripherals',     'Fixed Assets',        'asset'),
        ('Furniture',                   'Fixed Assets',        'asset'),
        ('Office Equipment',            'Fixed Assets',        'asset'),
        ('Plant & Machinery',           'Fixed Assets',        'asset'),
        ('Sewing Machine',              'Fixed Assets',        'asset'),
        ('Solar & Power Backup Assets', 'Fixed Assets',        'asset'),
        ('Indirect Expenses',           '',                    'expense'),
        ('Advertisements',              'Indirect Expenses',   'expense'),
        ('Bank Charges',                'Indirect Expenses',   'expense'),
        ('Commission Charges',          'Indirect Expenses',   'expense'),
        ('Contribution PF Or Esic',     'Indirect Expenses',   'expense'),
        ('E-Commerce Portal Expenses',  'Indirect Expenses',   'expense'),
        ('Fixed Fee',                   'Indirect Expenses',   'expense'),
        ('Interest',                    'Indirect Expenses',   'expense'),
        ('Internet & Recharges Expe.',  'Indirect Expenses',   'expense'),
        ('Legal and Professional Exp.', 'Indirect Expenses',   'expense'),
        ('Office Expenses',             'Indirect Expenses',   'expense'),
        ('PRINTING & STATIONERY',       'Indirect Expenses',   'expense'),
        ('Repaire & Maintenance Exp.',  'Indirect Expenses',   'expense'),
        ('Indirect Incomes',            '',                    'income'),
        ('Investments',                 '',                    'asset'),
        ('Loans (Liability)',            '',                    'liability'),
        ('Bank OD A/c',                 'Loans (Liability)',    'liability'),
        ('Secured Loans',               'Loans (Liability)',    'liability'),
        ('Unsecured Loans',             'Loans (Liability)',    'liability'),
        ('Misc. Expenses (ASSET)',       '',                    'asset'),
        ('Purchase Accounts',           '',                    'expense'),
        ('Raw Material',                'Purchase Accounts',   'expense'),
        ('Sales Accounts',              '',                    'income'),
        ('Suspense A/c',                '',                    'liability'),
    ]
    for grp in all_71_groups:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO ledger_groups (name, parent_group, nature) VALUES (?,?,?)",
                grp,
            )
        except Exception:
            pass
    conn.commit()

    # ── Fix parent_group for all groups to match Tally hierarchy ──
    parent_corrections = [
        ('Reserves & Surplus',          'Capital Account'),
        ('Bank Accounts',               'Current Assets'),
        ('Cash-in-hand',                'Current Assets'),
        ('Deposits (Asset)',             'Current Assets'),
        ('Loans & Advances (Asset)',     'Current Assets'),
        ('Prepaid Expenses',            'Current Assets'),
        ('Stock-in-hand',               'Current Assets'),
        ('Sundry Debtors',              'Current Assets'),
        ('Online Portal',               'Sundry Debtors'),
        ('Portal TDS',                  'Sundry Debtors'),
        ('Shopfiy',                     'Sundry Debtors'),
        ('Duties & Taxes',              'Current Liabilities'),
        ('PF Or ESIC',                  'Current Liabilities'),
        ('Provisions',                  'Current Liabilities'),
        ('Salary Exempt',               'Current Liabilities'),
        ('Salary & Wages Payable',      'Current Liabilities'),
        ('Sundry Creditors',            'Current Liabilities'),
        ('CGST',                        'Duties & Taxes'),
        ('IGST',                        'Duties & Taxes'),
        ('RCM Tax',                     'Duties & Taxes'),
        ('SGST',                        'Duties & Taxes'),
        ('TDS Working',                 'Duties & Taxes'),
        ('ESIC',                        'PF Or ESIC'),
        ('Esic and PF Other Charges',   'PF Or ESIC'),
        ('PF',                          'PF Or ESIC'),
        ('Creditors for Expenses',      'Sundry Creditors'),
        ('In House Job Work',           'Sundry Creditors'),
        ('MSME',                        'Sundry Creditors'),
        ('Non MSME',                    'Sundry Creditors'),
        ('Transport Contractors',       'Sundry Creditors'),
        ('Electricity Direct Exp.',     'Direct Expenses'),
        ('FREIGHT EXPENSES',            'Direct Expenses'),
        ('Job Work',                    'Direct Expenses'),
        ('PACKING MATERIAL',            'Direct Expenses'),
        ('Computers & Peripherals',     'Fixed Assets'),
        ('Furniture',                   'Fixed Assets'),
        ('Office Equipment',            'Fixed Assets'),
        ('Plant & Machinery',           'Fixed Assets'),
        ('Sewing Machine',              'Fixed Assets'),
        ('Solar & Power Backup Assets', 'Fixed Assets'),
        ('Advertisements',              'Indirect Expenses'),
        ('Bank Charges',                'Indirect Expenses'),
        ('Commission Charges',          'Indirect Expenses'),
        ('Contribution PF Or Esic',     'Indirect Expenses'),
        ('E-Commerce Portal Expenses',  'Indirect Expenses'),
        ('Fixed Fee',                   'Indirect Expenses'),
        ('Interest',                    'Indirect Expenses'),
        ('Internet & Recharges Expe.',  'Indirect Expenses'),
        ('Legal and Professional Exp.', 'Indirect Expenses'),
        ('Office Expenses',             'Indirect Expenses'),
        ('PRINTING & STATIONERY',       'Indirect Expenses'),
        ('Repaire & Maintenance Exp.',  'Indirect Expenses'),
        ('Bank OD A/c',                 'Loans (Liability)'),
        ('Secured Loans',               'Loans (Liability)'),
        ('Unsecured Loans',             'Loans (Liability)'),
        ('Raw Material',                'Purchase Accounts'),
    ]
    for grp_name, correct_parent in parent_corrections:
        conn.execute(
            "UPDATE ledger_groups SET parent_group=? WHERE name=? AND parent_group!=?",
            (correct_parent, grp_name, correct_parent),
        )
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
    # Ensure Sales Upload voucher type exists on existing installs too.
    try:
        conn.execute(
            """INSERT OR IGNORE INTO voucher_types
               (name, voucher_category, abbreviation, is_active, allow_narration, numbering_method)
               VALUES (?,?,?,?,?,?)""",
            ("Sales Upload", "Sales", "SU", 1, 1, "Auto"),
        )
    except Exception:
        pass
    conn.commit()

    # ── Seed default ledgers (INSERT OR IGNORE) ───────────────────
    def _grp_id(name):
        row = conn.execute("SELECT id FROM ledger_groups WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

    seed_ledgers = [
        # name, group_name, gstin, tds_applicable, tds_section, state
        ('Cash',                         'Cash-in-hand',      '', 0, '', 'Rajasthan'),
        ('Petty Cash',                   'Cash-in-hand',      '', 0, '', 'Rajasthan'),
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
        ('Finance Sales Upload A/c',     'Sales Accounts',    '', 0, '', ''),
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

    # ── Fix ledger group_name/group_id for renamed groups ─────────
    ledger_group_renames = [
        ('Cash-in-Hand', 'Cash-in-hand'),
        ('Direct Income', 'Direct Incomes'),
        ('Indirect Income', 'Indirect Incomes'),
        ('Loans Liabilities', 'Loans (Liability)'),
        ('Miscellaneous Expenses', 'Misc. Expenses (ASSET)'),
        ('Suspense Account', 'Suspense A/c'),
        ('Bank Overdraft', 'Bank OD A/c'),
        ('Deposit Assets', 'Deposits (Asset)'),
        ('Loans & Advances (Assets)', 'Loans & Advances (Asset)'),
        ('Stock in Hand', 'Stock-in-hand'),
        ('Provision', 'Provisions'),
        ('Reserve & Surplus', 'Reserves & Surplus'),
        ('Branch/Division', 'Branch / Divisions'),
    ]
    for old_gname, new_gname in ledger_group_renames:
        new_gid = _grp_id(new_gname)
        conn.execute(
            "UPDATE ledgers SET group_name=?, group_id=? WHERE group_name=?",
            (new_gname, new_gid, old_gname),
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
    period_from: Optional[str] = None,
    period_to: Optional[str] = None,
    company: Optional[str] = None,
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
    # period is stored as YYYY-MM — lexicographic range works
    if period_from:
        query += " AND period >= ?"
        params.append(period_from[:7])
    if period_to:
        query += " AND period <= ?"
        params.append(period_to[:7])
    if company:
        query += " AND (company_name = ? OR seller_gstin = ?)"
        params.extend([company, company])
    query += " ORDER BY period DESC, id DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_finance_sales_upload(data: dict) -> int:
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO finance_sales_uploads
           (platform, company_name, seller_gstin, company_state, period, filename, total_revenue, total_orders,
            total_returns, net_revenue, uploaded_by, upload_notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            data.get('platform') or '',
            data.get('company_name') or '',
            data.get('seller_gstin') or '',
            data.get('company_state') or '',
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


def create_finance_sales_entries(sales_upload_id: int, entries: list[dict]) -> int:
    """Persist invoice/order-level rows for a finance sales upload."""
    if not entries:
        return 0
    conn = _connect()
    # Rebuilding the same upload (e.g. parse fix + re-save) should replace rows, not append
    # duplicates that look like "extra invoices" in Sales Invoices / Customer Ledger.
    conn.execute("DELETE FROM finance_sales_entries WHERE sales_upload_id = ?", (int(sales_upload_id),))
    count = 0
    seen: set[tuple] = set()
    for e in entries:
        try:
            _line_items_raw = e.get("line_items") or "[]"
            _line_items_obj = json.loads(_line_items_raw) if isinstance(_line_items_raw, str) else _line_items_raw
            _line_items_norm = json.dumps(_line_items_obj, sort_keys=True, separators=(",", ":"))
        except Exception:
            _line_items_norm = str(e.get("line_items") or "[]")

        sig = (
            str(e.get("platform") or "").strip().upper(),
            str(e.get("period") or "").strip(),
            str(e.get("voucher_date") or "").strip(),
            str(e.get("invoice_no") or "").strip().upper(),
            str(e.get("order_id") or "").strip().upper(),
            str(e.get("party_name") or "").strip().upper(),
            round(float(e.get("taxable_amount") or 0.0), 2),
            round(float(e.get("cgst_amount") or 0.0), 2),
            round(float(e.get("sgst_amount") or 0.0), 2),
            round(float(e.get("igst_amount") or 0.0), 2),
            round(float(e.get("total_amount") or 0.0), 2),
            round(float(e.get("net_payable") or 0.0), 2),
            str(e.get("narration") or "").strip(),
            _line_items_norm,
        )
        if sig in seen:
            continue
        seen.add(sig)
        conn.execute(
            """INSERT INTO finance_sales_entries
               (sales_upload_id, platform, period, voucher_date, invoice_no, order_id,
                party_name, party_gstin, party_state, ship_to_state,
                taxable_amount, cgst_amount, sgst_amount, igst_amount,
                total_amount, net_payable, narration, source_filename, line_items)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sales_upload_id,
                e.get("platform") or "",
                e.get("period") or "",
                e.get("voucher_date") or "",
                e.get("invoice_no") or "",
                e.get("order_id") or "",
                e.get("party_name") or "",
                e.get("party_gstin") or "",
                e.get("party_state") or "",
                e.get("ship_to_state") or "",
                float(e.get("taxable_amount") or 0.0),
                float(e.get("cgst_amount") or 0.0),
                float(e.get("sgst_amount") or 0.0),
                float(e.get("igst_amount") or 0.0),
                float(e.get("total_amount") or 0.0),
                float(e.get("net_payable") or 0.0),
                e.get("narration") or "",
                e.get("source_filename") or "",
                e.get("line_items") or "[]",
            ),
        )
        count += 1
    conn.commit()
    conn.close()
    return count


# Synthetic voucher id for upload rows that have no persisted finance_sales_entries (summary-only).
# Keep well above 2_000_000 + max(finance_sales_entries.id) to avoid colliding with SUE-* voucher ids.
UPLOAD_SUMMARY_VOUCHER_BASE = 10_000_000

_NUM_PATCH_KEYS = frozenset(
    {"taxable_amount", "cgst_amount", "sgst_amount", "igst_amount", "total_amount", "net_payable"}
)


def get_sales_invoice_edit_patch(voucher_id: int) -> dict:
    """User overrides for synthetic / sales-upload voucher payloads (JSON merge)."""
    conn = _connect()
    r = conn.execute(
        "SELECT patch_json FROM finance_sales_invoice_edits WHERE voucher_id = ?",
        (int(voucher_id),),
    ).fetchone()
    conn.close()
    if not r or not r[0]:
        return {}
    try:
        return json.loads(r[0])
    except Exception:
        return {}


def _invoice_edit_map(conn: sqlite3.Connection, voucher_ids: list[int]) -> dict[int, dict]:
    if not voucher_ids:
        return {}
    ph = ",".join("?" * len(voucher_ids))
    rows = conn.execute(
        f"SELECT voucher_id, patch_json FROM finance_sales_invoice_edits WHERE voucher_id IN ({ph})",
        [int(x) for x in voucher_ids],
    ).fetchall()
    out: dict[int, dict] = {}
    for row in rows:
        try:
            out[int(row["voucher_id"])] = json.loads(row["patch_json"] or "{}")
        except Exception:
            out[int(row["voucher_id"])] = {}
    return out


def upsert_sales_invoice_edit_patch(voucher_id: int, patch: dict[str, Any]) -> dict:
    """Merge `patch` into stored JSON for this voucher id. Returns merged dict."""
    vid = int(voucher_id)
    conn = _connect()
    old = conn.execute(
        "SELECT patch_json FROM finance_sales_invoice_edits WHERE voucher_id = ?",
        (vid,),
    ).fetchone()
    merged: dict = {}
    if old and old[0]:
        try:
            merged = json.loads(old[0])
        except Exception:
            merged = {}
    for k, v in patch.items():
        if v is None:
            continue
        merged[str(k)] = v
    conn.execute(
        """INSERT INTO finance_sales_invoice_edits (voucher_id, patch_json, updated_at)
           VALUES (?,?,datetime('now'))
           ON CONFLICT(voucher_id) DO UPDATE SET
             patch_json = excluded.patch_json,
             updated_at = excluded.updated_at""",
        (vid, json.dumps(merged)),
    )
    conn.commit()
    conn.close()
    return merged


def _apply_sales_invoice_row_patch(row: dict, patch: dict) -> dict:
    if not patch:
        return row
    r = dict(row)
    if patch.get("invoice_no") is not None:
        r["invoice_no"] = str(patch["invoice_no"])
    if patch.get("voucher_date") is not None:
        r["voucher_date"] = str(patch["voucher_date"])
    if patch.get("bill_date") is not None:
        r["bill_date"] = str(patch["bill_date"])
    if patch.get("party_name") is not None:
        r["party_name"] = str(patch["party_name"])
    if patch.get("party_gstin") is not None:
        r["party_gstin"] = str(patch["party_gstin"])
    if patch.get("party_state") is not None:
        r["party_state"] = str(patch["party_state"])
    if patch.get("ship_to_state") is not None:
        r["ship_to_state"] = str(patch["ship_to_state"])
    if patch.get("order_id") is not None:
        r["order_id"] = str(patch["order_id"])
    if patch.get("source_filename") is not None:
        r["source_filename"] = str(patch["source_filename"])
    if patch.get("platform") is not None:
        r["platform"] = str(patch["platform"])
    if patch.get("period") is not None:
        r["period"] = str(patch["period"])
    if patch.get("narration") is not None:
        r["narration"] = str(patch["narration"])
    if patch.get("dimension_assignments") is not None:
        r["dimension_assignments"] = patch["dimension_assignments"]
    for nk in _NUM_PATCH_KEYS:
        if nk in patch and patch[nk] is not None:
            try:
                r[nk] = round(float(patch[nk]), 2)
            except (TypeError, ValueError):
                pass
    return r


def _apply_sales_invoice_patch_to_voucher(v: dict, patch: dict) -> dict:
    if not patch:
        return v
    out = dict(v)
    meta = dict(out.get("meta") or {})
    if patch.get("invoice_no") is not None:
        inv = str(patch["invoice_no"])
        out["bill_no"] = inv
        meta["invoice_no"] = inv
    if patch.get("voucher_date") is not None:
        out["voucher_date"] = str(patch["voucher_date"])
        if patch.get("bill_date") is None:
            out["bill_date"] = str(patch["voucher_date"])
    if patch.get("bill_date") is not None:
        out["bill_date"] = str(patch["bill_date"])
    if patch.get("party_name") is not None:
        out["party_name"] = str(patch["party_name"])
    if patch.get("party_gstin") is not None:
        out["party_gstin"] = str(patch["party_gstin"])
    if patch.get("party_state") is not None:
        out["party_state"] = str(patch["party_state"])
    if patch.get("ship_to_state") is not None:
        meta["ship_to_state"] = str(patch["ship_to_state"])
    if patch.get("order_id") is not None:
        meta["order_id"] = str(patch["order_id"])
        out["ref_number"] = str(patch["order_id"])
    if patch.get("source_filename") is not None:
        meta["source_filename"] = str(patch["source_filename"])
    if patch.get("platform") is not None:
        meta["platform"] = str(patch["platform"])
    if patch.get("period") is not None:
        meta["period"] = str(patch["period"])
    if patch.get("narration") is not None:
        out["narration"] = str(patch["narration"])
    if patch.get("supply_type") is not None:
        out["supply_type"] = str(patch["supply_type"])
    if patch.get("dimension_assignments") is not None:
        meta["dimension_assignments"] = patch["dimension_assignments"]
    for nk in _NUM_PATCH_KEYS:
        if nk in patch and patch[nk] is not None:
            try:
                out[nk] = round(float(patch[nk]), 2)
            except (TypeError, ValueError):
                pass
    out["meta"] = meta
    return out


def get_sales_entry_voucher(voucher_id: int) -> Optional[dict]:
    """Synthetic voucher payload for a finance_sales_entries row (id offset 2_000_000)."""
    if voucher_id < 2_000_000:
        return None
    entry_id = voucher_id - 2_000_000
    conn = _connect()
    r = conn.execute(
        "SELECT * FROM finance_sales_entries WHERE id = ?",
        (entry_id,),
    ).fetchone()
    if not r:
        conn.close()
        return None
    rr = dict(r)
    upload_meta: dict = {}
    su_id = rr.get("sales_upload_id")
    if su_id:
        urow = conn.execute(
            "SELECT seller_gstin, company_name, company_state FROM finance_sales_uploads WHERE id = ?",
            (int(su_id),),
        ).fetchone()
        if urow:
            upload_meta = {
                "seller_gstin": str(urow["seller_gstin"] or ""),
                "seller_company": str(urow["company_name"] or ""),
                "seller_state": str(urow["company_state"] or ""),
            }
    conn.close()
    import json

    raw_lines = rr.get("line_items") or "[]"
    try:
        line_items = json.loads(raw_lines) if isinstance(raw_lines, str) else raw_lines
    except Exception:
        line_items = []

    voucher_lines = []
    for i, li in enumerate(line_items):
        sku = str(li.get("sku") or li.get("SKU") or "").strip() or "Item"
        qty = li.get("quantity") or li.get("Quantity") or ""
        prod = str(li.get("product_name") or li.get("Product_Name") or "").strip()
        desc_parts = [sku]
        if prod:
            desc_parts.append(prod[:120] + ("…" if len(prod) > 120 else ""))
        if qty != "":
            desc_parts.append(f"Qty {qty}")
        description = " — ".join(desc_parts)
        voucher_lines.append({
            "id": i + 1,
            "expense_head": sku,
            "description": description,
            "amount": float(li.get("invoice_amount") or li.get("Invoice_Amount") or 0.0),
            "cost_centre": str(li.get("ship_to_state") or li.get("Ship_To_State") or ""),
            "is_debit": 1,
        })
    vdate = rr.get("voucher_date") or f"{str(rr.get('period') or '')[:7]}-01"
    vid = 2_000_000 + int(rr["id"])
    _np = float(rr.get("net_payable") or 0)
    _tb = float(rr.get("taxable_amount") or 0)
    _nlow = str(rr.get("narration") or "").lower()
    _is_credit = _np < 0 or _tb < 0 or ("refund" in _nlow)
    vtype = "Sales Credit Memo" if _is_credit else "Sales Invoice"
    v = {
        "id": vid,
        "voucher_no": f"SUE-{rr['id']}",
        "voucher_date": vdate,
        "voucher_type": vtype,
        "party_name": rr.get("party_name") or "",
        "party_gstin": rr.get("party_gstin") or "",
        "party_state": rr.get("party_state") or "",
        "bill_no": rr.get("invoice_no") or "",
        "bill_date": vdate,
        "supply_type": "Inter" if float(rr.get("igst_amount") or 0) > 0 else "Intra",
        "narration": rr.get("narration") or "",
        "taxable_amount": float(rr.get("taxable_amount") or 0.0),
        "cgst_amount": float(rr.get("cgst_amount") or 0.0),
        "sgst_amount": float(rr.get("sgst_amount") or 0.0),
        "igst_amount": float(rr.get("igst_amount") or 0.0),
        "tds_section": "",
        "tds_rate": 0.0,
        "tds_amount": 0.0,
        "total_amount": float(rr.get("total_amount") or 0.0),
        "net_payable": float(rr.get("net_payable") or 0.0),
        "payment_mode": "",
        "bank_ledger": "",
        "cheque_no": "",
        "ref_number": rr.get("order_id") or "",
        "lines": voucher_lines,
        "meta": {
            "source": "finance_sales_entry",
            "platform": rr.get("platform"),
            "period": rr.get("period"),
            "invoice_no": rr.get("invoice_no"),
            "order_id": rr.get("order_id"),
            "ship_to_state": rr.get("ship_to_state"),
            "source_filename": rr.get("source_filename"),
            "line_items": line_items,
            **upload_meta,
        },
    }
    patch = get_sales_invoice_edit_patch(vid)
    return _apply_sales_invoice_patch_to_voucher(v, patch)


def _rollup_line_items_from_sales_entries(upload_id: int) -> tuple[list, list]:
    """Flatten SKU line_items from all finance_sales_entries for this upload; build synthetic voucher lines."""
    conn = _connect()
    rows = conn.execute(
        """SELECT invoice_no, order_id, line_items FROM finance_sales_entries
           WHERE sales_upload_id = ? ORDER BY id""",
        (int(upload_id),),
    ).fetchall()
    conn.close()
    merged: list = []
    for row in rows:
        inv = str(row["invoice_no"] or "").strip()
        oid = str(row["order_id"] or "").strip()
        raw = row["line_items"] or "[]"
        try:
            items = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            items = []
        if not isinstance(items, list):
            continue
        for li in items:
            if not isinstance(li, dict):
                continue
            d = dict(li)
            if inv:
                d.setdefault("source_invoice_no", inv)
            if oid:
                d.setdefault("source_order_id", oid)
            merged.append(d)
    voucher_lines: list = []
    for i, li in enumerate(merged):
        sku = str(li.get("sku") or li.get("SKU") or "").strip() or "Item"
        qty = li.get("quantity") or li.get("Quantity") or ""
        prod = str(li.get("product_name") or li.get("Product_Name") or "").strip()
        desc_parts = [sku]
        if prod:
            desc_parts.append(prod[:120] + ("…" if len(prod) > 120 else ""))
        if qty != "":
            desc_parts.append(f"Qty {qty}")
        inv_tag = str(li.get("source_invoice_no") or "").strip()
        if inv_tag:
            desc_parts.append(f"Inv {inv_tag}")
        description = " — ".join(desc_parts)
        voucher_lines.append({
            "id": i + 1,
            "expense_head": sku,
            "description": description,
            "amount": float(li.get("invoice_amount") or li.get("Invoice_Amount") or 0.0),
            "cost_centre": str(li.get("ship_to_state") or li.get("Ship_To_State") or ""),
            "is_debit": 1,
        })
    return merged, voucher_lines


def get_upload_summary_voucher(voucher_id: int) -> Optional[dict]:
    """Voucher-shaped payload for a finance_sales_uploads row (synthetic id base UPLOAD_SUMMARY_VOUCHER_BASE)."""
    if voucher_id < UPLOAD_SUMMARY_VOUCHER_BASE:
        return None
    upload_id = voucher_id - UPLOAD_SUMMARY_VOUCHER_BASE
    if upload_id <= 0:
        return None
    conn = _connect()
    r = conn.execute("SELECT * FROM finance_sales_uploads WHERE id = ?", (upload_id,)).fetchone()
    if not r:
        conn.close()
        return None
    rr = dict(r)
    conn.close()
    p = str(rr.get("period") or "").strip()
    vdate = f"{p[:7]}-01" if len(p) >= 7 else (str(rr.get("created_at") or "")[:10] or "")
    gross = float(rr.get("total_revenue") or 0.0)
    ret = float(rr.get("total_returns") or 0.0)
    net = float(rr.get("net_revenue") or 0.0)
    party = (rr.get("company_name") or "").strip() or (rr.get("seller_gstin") or "").strip() or (rr.get("platform") or "")
    uid = int(rr["id"])
    doc_no = f"SUM-{uid}"
    cstate = str(rr.get("company_state") or "")
    vid = UPLOAD_SUMMARY_VOUCHER_BASE + uid
    rolled_lines, voucher_lines = _rollup_line_items_from_sales_entries(uid)
    narr = (
        f"Upload summary · {int(rr.get('total_orders') or 0)} orders in file; "
        f"SUE-… rows are per-invoice postings. "
    )
    if rolled_lines:
        narr += f"Lines tab lists {len(rolled_lines)} SKU row(s) rolled up from parsed invoices in this upload."
    else:
        narr += "Product lines appear when MTR rows are parsed into finance_sales_entries for this upload."
    v = {
        "id": vid,
        "voucher_no": f"SUP-{uid}",
        "voucher_date": vdate,
        "voucher_type": "Sales Upload",
        "party_name": party,
        "party_gstin": str(rr.get("seller_gstin") or ""),
        "party_state": cstate,
        "bill_no": doc_no,
        "bill_date": vdate,
        "supply_type": "Intra",
        "narration": narr,
        "taxable_amount": round(gross - ret, 2),
        "cgst_amount": 0.0,
        "sgst_amount": 0.0,
        "igst_amount": 0.0,
        "tds_section": "",
        "tds_rate": 0.0,
        "tds_amount": 0.0,
        "total_amount": round(gross, 2),
        "net_payable": round(net, 2),
        "payment_mode": "",
        "bank_ledger": "",
        "cheque_no": "",
        "ref_number": f"UPLOAD-{uid}",
        "lines": voucher_lines,
        "meta": {
            "source": "finance_sales_upload_summary",
            "sales_upload_id": uid,
            "platform": rr.get("platform"),
            "period": rr.get("period"),
            "filename": rr.get("filename"),
            "invoice_no": doc_no,
            "order_id": f"UPLOAD-{uid}",
            "ship_to_state": cstate,
            "source_filename": str(rr.get("filename") or ""),
            "total_orders": int(rr.get("total_orders") or 0),
            "total_revenue": round(gross, 2),
            "total_returns": round(ret, 2),
            "line_items": rolled_lines,
            "line_items_rollup_from_entries": bool(rolled_lines),
        },
    }
    patch = get_sales_invoice_edit_patch(vid)
    return _apply_sales_invoice_patch_to_voucher(v, patch)


def _entry_is_sales_credit_memo_sql() -> str:
    """SQLite predicate: posted credit / return lines (negative AR movement or Refund narration)."""
    return (
        "(COALESCE(net_payable,0) < 0 OR COALESCE(taxable_amount,0) < 0 "
        "OR instr(lower(COALESCE(narration,'')),'refund') > 0)"
    )


def _entry_is_sales_credit_memo_sql_se_alias() -> str:
    """Same as _entry_is_sales_credit_memo_sql but for queries using table alias `se`."""
    return (
        "(COALESCE(se.net_payable,0) < 0 OR COALESCE(se.taxable_amount,0) < 0 "
        "OR instr(lower(COALESCE(se.narration,'')),'refund') > 0)"
    )


def list_sales_invoices(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
    document_kind: Optional[str] = None,
    include_upload_summaries: bool = True,
) -> list[dict]:
    """All finance_sales_entries rows plus every finance_sales_uploads summary (same coverage as Sales Uploads).

    document_kind:
      - None / \"all\" — legacy combined list (invoices + credit memos + upload summaries).
      - \"sales\" — invoices & shipments only (exclude credit-memo SUE rows); include SUP summaries.
      - \"credit_memo\" — sales credit memos / returns only (SUE rows matching credit predicate); no SUP rows.

    include_upload_summaries:
      When False (with document_kind other than credit_memo), SUP-* upload summary rows are omitted.
      Use this to hide one-summary-per-upload rows; Amazon uploads with multiple seller GSTINs still
      create one Finance upload (and one SUP) per GSTIN from the same file.
    """
    conn = _connect()
    dk = (document_kind or "").strip().lower()
    credit_pred = _entry_is_sales_credit_memo_sql()
    q = """
        SELECT id, sales_upload_id, voucher_date, platform, period, invoice_no, order_id,
               party_name, party_gstin, party_state, ship_to_state,
               taxable_amount, cgst_amount, sgst_amount, igst_amount,
               total_amount, net_payable, source_filename, narration
        FROM finance_sales_entries
        WHERE 1=1
    """
    params: list = []
    if dk == "sales":
        q += f" AND NOT ({credit_pred})"
    elif dk == "credit_memo":
        q += f" AND ({credit_pred})"
    if start_date:
        q += " AND voucher_date >= ?"
        params.append(start_date)
    if end_date:
        q += " AND voucher_date <= ?"
        params.append(end_date)
    if search:
        like = f"%{search}%"
        q += """ AND (
            invoice_no LIKE ? OR order_id LIKE ? OR party_name LIKE ? OR platform LIKE ?
            OR party_gstin LIKE ? OR source_filename LIKE ? OR ship_to_state LIKE ?
        )"""
        params.extend([like, like, like, like, like, like, like])
    q += " ORDER BY voucher_date DESC, id DESC"
    rows = conn.execute(q, params).fetchall()

    out: list[dict] = []
    for row in rows:
        rr = dict(row)
        rid = int(rr.get("id") or 0)
        _np = float(rr.get("net_payable") or 0)
        _tb = float(rr.get("taxable_amount") or 0)
        _narr = str(rr.get("narration") or "").lower()
        is_credit = _np < 0 or _tb < 0 or ("refund" in _narr)
        out.append({
            "id": 2_000_000 + rid,
            "voucher_no": f"SUE-{rid}",
            "row_kind": "entry",
            "document_subtype": "sales_credit_memo" if is_credit else "sales_invoice",
            "sales_upload_id": int(rr.get("sales_upload_id") or 0),
            "voucher_date": rr.get("voucher_date") or "",
            "platform": rr.get("platform") or "",
            "period": rr.get("period") or "",
            "invoice_no": rr.get("invoice_no") or "",
            "order_id": rr.get("order_id") or "",
            "party_name": rr.get("party_name") or "",
            "party_gstin": rr.get("party_gstin") or "",
            "party_state": rr.get("party_state") or "",
            "ship_to_state": rr.get("ship_to_state") or "",
            "taxable_amount": round(float(rr.get("taxable_amount") or 0), 2),
            "cgst_amount": round(float(rr.get("cgst_amount") or 0), 2),
            "sgst_amount": round(float(rr.get("sgst_amount") or 0), 2),
            "igst_amount": round(float(rr.get("igst_amount") or 0), 2),
            "total_amount": round(float(rr.get("total_amount") or 0), 2),
            "net_payable": round(float(rr.get("net_payable") or 0), 2),
            "source_filename": rr.get("source_filename") or "",
            "narration": rr.get("narration") or "",
        })

    # Upload summaries for every row in finance_sales_uploads (same grid as Sales Uploads), in addition to entries.
    q2 = """
        SELECT su.id, su.platform, su.period, su.company_name, su.seller_gstin, su.company_state,
               su.filename, su.total_revenue, su.total_returns, su.net_revenue, su.total_orders,
               su.created_at
        FROM finance_sales_uploads su
        WHERE 1=1
    """
    params2: list = []
    if start_date:
        q2 += """
          AND date(
            CASE WHEN length(trim(COALESCE(su.period, ''))) >= 7 THEN trim(su.period) || '-01'
                 ELSE substr(COALESCE(su.created_at, ''), 1, 10) END
          ) >= date(?)
        """
        params2.append(start_date)
    if end_date:
        q2 += """
          AND date(
            CASE WHEN length(trim(COALESCE(su.period, ''))) >= 7 THEN trim(su.period) || '-01'
                 ELSE substr(COALESCE(su.created_at, ''), 1, 10) END
          ) <= date(?)
        """
        params2.append(end_date)
    if search:
        like = f"%{search}%"
        q2 += """ AND (
            su.platform LIKE ? OR su.company_name LIKE ? OR su.seller_gstin LIKE ?
            OR su.filename LIKE ? OR CAST(su.id AS TEXT) LIKE ?
        )"""
        params2.extend([like, like, like, like, like])
    q2 += " ORDER BY su.period DESC, su.id DESC"
    if dk != "credit_memo" and include_upload_summaries:
        for row in conn.execute(q2, params2).fetchall():
            su = dict(row)
            uid = int(su["id"])
            p = str(su.get("period") or "").strip()
            vdate = f"{p[:7]}-01" if len(p) >= 7 else (str(su.get("created_at") or "")[:10] or "")
            gross = float(su.get("total_revenue") or 0.0)
            ret = float(su.get("total_returns") or 0.0)
            net = float(su.get("net_revenue") or 0.0)
            party = (su.get("company_name") or "").strip() or (su.get("seller_gstin") or "").strip() or (su.get("platform") or "")
            doc_no = f"SUM-{uid}"
            cst = str(su.get("company_state") or "")
            out.append({
                "id": UPLOAD_SUMMARY_VOUCHER_BASE + uid,
                "voucher_no": f"SUP-{uid}",
                "row_kind": "upload_summary",
                "document_subtype": "upload_summary",
                "sales_upload_id": uid,
                "voucher_date": vdate,
                "platform": str(su.get("platform") or ""),
                "period": str(su.get("period") or ""),
                "invoice_no": doc_no,
                "order_id": f"UPLOAD-{uid}",
                "party_name": party,
                "party_gstin": str(su.get("seller_gstin") or ""),
                "party_state": cst,
                "ship_to_state": cst,
                "taxable_amount": round(gross - ret, 2),
                "cgst_amount": 0.0,
                "sgst_amount": 0.0,
                "igst_amount": 0.0,
                "total_amount": round(gross, 2),
                "net_payable": round(net, 2),
                "source_filename": str(su.get("filename") or ""),
                "narration": "",
            })

    ids = [int(r["id"]) for r in out]
    pmap = _invoice_edit_map(conn, ids)
    out = [_apply_sales_invoice_row_patch(r, pmap.get(int(r["id"]), {})) for r in out]
    conn.close()

    out.sort(key=lambda r: (r.get("voucher_date") or "", r.get("id")), reverse=True)
    return out


def list_finance_inventory_movements(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
) -> list[dict[str, Any]]:
    """SKU quantities from ``finance_sales_entries.line_items`` (Finance uploads only).

    * **qty_out** — shipment / invoice entries (not classified as sales credit memo).
    * **qty_in** — returns / credits (negative net/taxable or refund narration), quantity as positive.

    This does not read operational inventory sheets; it is a movement ledger derived from
    posted Finance sales lines so you can see what left (sales) and came back (returns).
    """
    conn = _connect()
    q = """
        SELECT line_items, net_payable, taxable_amount, narration
        FROM finance_sales_entries
        WHERE COALESCE(trim(line_items), '') NOT IN ('', '[]', 'null')
    """
    params: list[Any] = []
    if start_date:
        q += " AND voucher_date >= ?"
        params.append(start_date)
    if end_date:
        q += " AND voucher_date <= ?"
        params.append(end_date)
    rows = conn.execute(q, params).fetchall()
    conn.close()

    sk = (search or "").strip().lower()
    agg: dict[str, dict[str, Any]] = {}

    for row in rows:
        np_ = float(row["net_payable"] or 0)
        tb = float(row["taxable_amount"] or 0)
        narr = str(row["narration"] or "").lower()
        is_credit = np_ < -1e-6 or tb < -1e-6 or ("refund" in narr)
        raw = row["line_items"] or "[]"
        try:
            items = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if not isinstance(items, list):
            continue
        for li in items:
            if not isinstance(li, dict):
                continue
            sku = str(li.get("sku") or li.get("SKU") or "").strip()
            if not sku or sku.lower() in ("nan", "none"):
                continue
            try:
                qty = float(li.get("quantity") or li.get("Quantity") or 0)
            except (TypeError, ValueError):
                qty = 0.0
            if abs(qty) < 1e-9:
                continue
            prod = str(li.get("product_name") or li.get("Product_Name") or "").strip()
            if sk:
                hay = f"{sku} {prod}".lower()
                if sk not in hay:
                    continue
            if sku not in agg:
                agg[sku] = {"sku": sku, "product_name": prod, "qty_out": 0.0, "qty_in": 0.0, "line_count": 0}
            elif prod and not (agg[sku].get("product_name") or ""):
                agg[sku]["product_name"] = prod
            qv = abs(qty)
            if is_credit:
                agg[sku]["qty_in"] += qv
            else:
                agg[sku]["qty_out"] += qv
            agg[sku]["line_count"] = int(agg[sku].get("line_count", 0)) + 1

    out: list[dict[str, Any]] = []
    for v in agg.values():
        qo = round(float(v["qty_out"]), 3)
        qi = round(float(v["qty_in"]), 3)
        out.append({
            "sku": v["sku"],
            "product_name": (v.get("product_name") or "")[:200],
            "qty_out": qo,
            "qty_in": qi,
            "net_qty": round(qo - qi, 3),
            "line_count": int(v.get("line_count", 0)),
        })
    out.sort(key=lambda r: (r["qty_out"] + r["qty_in"], r["sku"]), reverse=True)
    return out


def _bc_location_code_from_invoice_no(invoice_no: str) -> str:
    """BC-style FC / warehouse code prefix before hyphen (e.g. BLR8-22106 → BLR8)."""
    s = (invoice_no or "").strip()
    if not s:
        return ""
    if "-" in s:
        return s.split("-", 1)[0].strip()
    return ""


def _gst_customer_type_bc(party_gstin: str) -> str:
    g = (party_gstin or "").replace(" ", "").upper()
    return "Registered" if len(g) >= 15 else "Unregistered"


def _gst_jurisdiction_bc(igst: float) -> str:
    return "Interstate" if float(igst or 0) > 0.001 else "Intrastate"


def list_customer_ledger_entries(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
    document_kind: Optional[str] = None,
) -> list[dict]:
    """Dynamics 365 / BC-style customer ledger lines (posted sales upload entries only).

    Joins seller context from finance_sales_uploads. Rows match the common
    "Customer Ledger Entries" export shape (document, customer, GST, amounts).

    document_kind:
      - None / \"all\" — invoices and credit memos.
      - \"sales\" — positive invoices only (exclude credit-memo predicate).
      - \"credit_memo\" — sales credit memos / returns only.
    """
    conn = _connect()
    dk = (document_kind or "all").strip().lower()
    credit_pred_se = _entry_is_sales_credit_memo_sql_se_alias()
    q = """
        SELECT se.id AS entry_id, se.sales_upload_id, se.voucher_date, se.platform, se.period,
               se.invoice_no, se.order_id, se.party_name, se.party_gstin, se.party_state, se.ship_to_state,
               se.taxable_amount, se.cgst_amount, se.sgst_amount, se.igst_amount,
               se.total_amount, se.net_payable, se.source_filename, se.narration, se.line_items,
               su.company_name, su.company_state, su.seller_gstin
        FROM finance_sales_entries se
        JOIN finance_sales_uploads su ON se.sales_upload_id = su.id
        WHERE 1=1
    """
    params: list = []
    if dk == "sales":
        q += f" AND NOT ({credit_pred_se})"
    elif dk == "credit_memo":
        q += f" AND ({credit_pred_se})"
    if start_date:
        q += " AND se.voucher_date >= ?"
        params.append(start_date)
    if end_date:
        q += " AND se.voucher_date <= ?"
        params.append(end_date)
    if search:
        like = f"%{search}%"
        q += """ AND (
            se.invoice_no LIKE ? OR se.order_id LIKE ? OR se.party_name LIKE ? OR se.platform LIKE ?
            OR se.party_gstin LIKE ? OR se.source_filename LIKE ? OR se.ship_to_state LIKE ?
            OR se.narration LIKE ?
        )"""
        params.extend([like, like, like, like, like, like, like, like])
    q += " ORDER BY se.voucher_date DESC, se.id DESC"
    raw_rows = conn.execute(q, params).fetchall()

    semi: list[dict] = []
    for row in raw_rows:
        rr = dict(row)
        rid = int(rr.get("entry_id") or 0)
        vid = 2_000_000 + rid
        company = str(rr.get("company_name") or "").strip()
        plat = str(rr.get("platform") or "").strip()
        branch_core = company or plat
        semi.append({
            "id": vid,
            "voucher_date": rr.get("voucher_date") or "",
            "invoice_no": str(rr.get("invoice_no") or ""),
            "order_id": str(rr.get("order_id") or ""),
            "party_name": str(rr.get("party_name") or ""),
            "party_gstin": str(rr.get("party_gstin") or ""),
            "party_state": str(rr.get("party_state") or ""),
            "ship_to_state": str(rr.get("ship_to_state") or ""),
            "taxable_amount": round(float(rr.get("taxable_amount") or 0), 2),
            "cgst_amount": round(float(rr.get("cgst_amount") or 0), 2),
            "sgst_amount": round(float(rr.get("sgst_amount") or 0), 2),
            "igst_amount": round(float(rr.get("igst_amount") or 0), 2),
            "total_amount": round(float(rr.get("total_amount") or 0), 2),
            "net_payable": round(float(rr.get("net_payable") or 0), 2),
            "narration": str(rr.get("narration") or ""),
            "line_items": rr.get("line_items") or "[]",
            "_branch_core": branch_core,
            "seller_company_state": str(rr.get("company_state") or ""),
            "seller_gstin": str(rr.get("seller_gstin") or ""),
        })

    ids = [int(r["id"]) for r in semi]
    pmap = _invoice_edit_map(conn, ids)
    patched = [_apply_sales_invoice_row_patch(r, pmap.get(int(r["id"]), {})) for r in semi]
    conn.close()

    out: list[dict] = []
    def _normalize_doc_date(v: Any) -> str:
        s = str(v or "").strip()
        if not s:
            return ""
        # Keep plain YYYY-MM-DD unchanged for speed/stability.
        if len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-":
            return s[:10]
        # Common ERP/date export patterns.
        from datetime import datetime
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(s[:10], fmt).date().isoformat()
            except Exception:
                continue
        return s[:10]

    def _line_item_invoice_date(raw_line_items: Any) -> str:
        try:
            lis = json.loads(raw_line_items) if isinstance(raw_line_items, str) else raw_line_items
        except Exception:
            return ""
        if not isinstance(lis, list):
            return ""
        for li in lis:
            if not isinstance(li, dict):
                continue
            for k in ("invoice_date", "Invoice_Date", "invoice_date_text", "Invoice_Date_Text"):
                v = str(li.get(k) or "").strip()
                if v and v.lower() not in ("nan", "none"):
                    return _normalize_doc_date(v)
        return ""
    for r in patched:
        cgst = float(r.get("cgst_amount") or 0)
        sgst = float(r.get("sgst_amount") or 0)
        igst = float(r.get("igst_amount") or 0)
        gst_amount = round(cgst + sgst + igst, 2)
        taxable = round(float(r.get("taxable_amount") or 0), 2)
        inv = str(r.get("invoice_no") or "")
        vdate = _normalize_doc_date(r.get("voucher_date"))
        bill_date = str(r.get("bill_date") or "").strip()
        if not bill_date:
            bill_date = _line_item_invoice_date(r.get("line_items"))
        doc_date = _normalize_doc_date(bill_date) or vdate
        party_gstin = str(r.get("party_gstin") or "")
        ship = str(r.get("ship_to_state") or "").strip()
        loc_state = ship or str(r.get("party_state") or "")
        narration = str(r.get("narration") or "").strip()
        desc = narration if narration else (f"Posted {inv}".strip() if inv else "Posted entry")
        core = str(r.get("_branch_core") or "").strip()
        pst = str(r.get("party_state") or "").strip()
        branch_code = f"{core} {pst}".strip()
        _np = float(r.get("net_payable") or 0)
        _tb = float(r.get("taxable_amount") or 0)
        narr_l = str(r.get("narration") or "").lower()
        is_credit = _np < 0 or _tb < 0 or ("refund" in narr_l)
        out.append({
            "id": int(r["id"]),
            "document_date": doc_date,
            "document_type": "Credit Memo" if is_credit else "Invoice",
            "document_no": inv,
            "customer_no": "",
            "customer_name": str(r.get("party_name") or ""),
            "description": desc,
            "branch_code": branch_code,
            "taxable_amount": taxable,
            "gst_amount": gst_amount,
            "invoice_amount": round(float(r.get("total_amount") or 0), 2),
            "due_date": doc_date,
            "gst_customer_type": _gst_customer_type_bc(party_gstin),
            "seller_state_code": str(r.get("seller_company_state") or ""),
            "seller_gst_reg_no": str(r.get("seller_gstin") or ""),
            "location_code": _bc_location_code_from_invoice_no(inv),
            "location_state_code": loc_state,
            "gst_jurisdiction_type": _gst_jurisdiction_bc(igst),
            "external_document_no": str(r.get("order_id") or ""),
            "location_gst_reg_no": party_gstin,
        })
    return out


def list_vouchers(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    voucher_type: Optional[str] = None,
) -> list[dict]:
    """Return vouchers including synthetic rows from finance sales uploads."""
    base = list_expense_vouchers(
        start_date=start_date,
        end_date=end_date,
        voucher_type=voucher_type,
    )
    if voucher_type and voucher_type not in ("Sales Upload", "Sales Invoice"):
        return base

    conn = _connect()
    synthetic: list[dict] = []

    eq = "SELECT * FROM finance_sales_entries WHERE 1=1"
    eparams: list = []
    if start_date:
        eq += " AND voucher_date >= ?"
        eparams.append(start_date)
    if end_date:
        eq += " AND voucher_date <= ?"
        eparams.append(end_date)
    eq += " ORDER BY voucher_date DESC, id DESC"
    for rr in conn.execute(eq, eparams).fetchall():
        rr = dict(rr)
        synthetic.append({
            "id": 2_000_000 + int(rr["id"]),
            "voucher_no": f"SUE-{rr['id']}",
            "voucher_date": rr.get("voucher_date") or f"{str(rr.get('period') or '')[:7]}-01",
            "voucher_type": "Sales Upload",
            "party_name": rr.get("party_name") or rr.get("platform") or "",
            "party_gstin": rr.get("party_gstin") or "",
            "party_state": rr.get("party_state") or "",
            "bill_no": rr.get("invoice_no") or "",
            "bill_date": rr.get("voucher_date") or "",
            "supply_type": "",
            "narration": rr.get("narration") or "",
            "taxable_amount": float(rr.get("taxable_amount") or 0.0),
            "cgst_amount": float(rr.get("cgst_amount") or 0.0),
            "sgst_amount": float(rr.get("sgst_amount") or 0.0),
            "igst_amount": float(rr.get("igst_amount") or 0.0),
            "tds_section": "",
            "tds_rate": 0.0,
            "tds_amount": 0.0,
            "total_amount": float(rr.get("total_amount") or 0.0),
            "net_payable": float(rr.get("net_payable") or 0.0),
            "payment_mode": "",
            "bank_ledger": "",
            "cheque_no": "",
            "ref_number": rr.get("order_id") or "",
            "lines": [],
        })

    q = """SELECT * FROM finance_sales_uploads su WHERE 1=1
           AND NOT EXISTS (SELECT 1 FROM finance_sales_entries se WHERE se.sales_upload_id = su.id)"""
    params: list = []
    if start_date:
        q += " AND period >= ?"
        params.append(start_date[:7])
    if end_date:
        q += " AND period <= ?"
        params.append(end_date[:7])
    q += " ORDER BY created_at DESC, id DESC"
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()

    for r in rows:
        synthetic.append({
            "id": 1_000_000 + int(r["id"]),
            "voucher_no": f"SU-{r['id']}",
            "voucher_date": f"{str(r.get('period') or '')[:7]}-01",
            "voucher_type": "Sales Upload",
            "party_name": r.get("company_name") or r.get("seller_gstin") or r.get("platform") or "",
            "party_gstin": r.get("seller_gstin") or "",
            "party_state": r.get("company_state") or "",
            "bill_no": r.get("filename") or "",
            "bill_date": f"{str(r.get('period') or '')[:7]}-01",
            "supply_type": "",
            "narration": f"Finance sales upload ({r.get('platform','')}) {r.get('period','')}",
            "taxable_amount": float(r.get("total_revenue") or 0.0),
            "cgst_amount": 0.0,
            "sgst_amount": 0.0,
            "igst_amount": 0.0,
            "tds_section": "",
            "tds_rate": 0.0,
            "tds_amount": 0.0,
            "total_amount": float(r.get("total_revenue") or 0.0),
            "net_payable": float(r.get("net_revenue") or 0.0),
            "payment_mode": "",
            "bank_ledger": "",
            "cheque_no": "",
            "ref_number": "",
            "lines": [],
        })
    return base + synthetic


def get_voucher_summary_by_date(date: str) -> list[dict]:
    """Return all vouchers for a specific date including finance sales uploads."""
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
    entries = [dict(r) for r in conn.execute(
        "SELECT * FROM finance_sales_entries WHERE voucher_date = ? ORDER BY id ASC",
        (date,),
    ).fetchall()]
    uploads = [dict(r) for r in conn.execute(
        """SELECT * FROM finance_sales_uploads su WHERE period = ?
           AND NOT EXISTS (SELECT 1 FROM finance_sales_entries se WHERE se.sales_upload_id = su.id)
           ORDER BY id ASC""",
        (date[:7],),
    ).fetchall()]
    conn.close()
    for rr in entries:
        vouchers.append({
            "id": 2_000_000 + int(rr["id"]),
            "voucher_no": f"SUE-{rr['id']}",
            "voucher_date": rr.get("voucher_date") or date,
            "voucher_type": "Sales Upload",
            "party_name": rr.get("party_name") or rr.get("platform") or "",
            "party_gstin": rr.get("party_gstin") or "",
            "party_state": rr.get("party_state") or "",
            "bill_no": rr.get("invoice_no") or "",
            "bill_date": rr.get("voucher_date") or date,
            "supply_type": "",
            "narration": rr.get("narration") or "",
            "taxable_amount": float(rr.get("taxable_amount") or 0.0),
            "cgst_amount": float(rr.get("cgst_amount") or 0.0),
            "sgst_amount": float(rr.get("sgst_amount") or 0.0),
            "igst_amount": float(rr.get("igst_amount") or 0.0),
            "tds_section": "",
            "tds_rate": 0.0,
            "tds_amount": 0.0,
            "total_amount": float(rr.get("total_amount") or 0.0),
            "net_payable": float(rr.get("net_payable") or 0.0),
            "payment_mode": "",
            "bank_ledger": "",
            "cheque_no": "",
            "ref_number": rr.get("order_id") or "",
            "lines": [],
        })
    for r in uploads:
        vouchers.append({
            "id": 1_000_000 + int(r["id"]),
            "voucher_no": f"SU-{r['id']}",
            "voucher_date": f"{str(r.get('period') or '')[:7]}-01",
            "voucher_type": "Sales Upload",
            "party_name": r.get("company_name") or r.get("seller_gstin") or r.get("platform") or "",
            "party_gstin": r.get("seller_gstin") or "",
            "party_state": r.get("company_state") or "",
            "bill_no": r.get("filename") or "",
            "bill_date": f"{str(r.get('period') or '')[:7]}-01",
            "supply_type": "",
            "narration": f"Finance sales upload ({r.get('platform','')}) {r.get('period','')}",
            "taxable_amount": float(r.get("total_revenue") or 0.0),
            "cgst_amount": 0.0,
            "sgst_amount": 0.0,
            "igst_amount": 0.0,
            "tds_section": "",
            "tds_rate": 0.0,
            "tds_amount": 0.0,
            "total_amount": float(r.get("total_revenue") or 0.0),
            "net_payable": float(r.get("net_revenue") or 0.0),
            "payment_mode": "",
            "bank_ledger": "",
            "cheque_no": "",
            "ref_number": "",
            "lines": [],
        })
    return vouchers


def get_gstr3b_data(start_date: str, end_date: str) -> dict:
    """Compute GSTR3B data from vouchers + finance sales uploads."""
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
    # Finance: invoice-level entries (Amazon etc.) + summary uploads without line detail.
    ent = conn.execute(
        """SELECT SUM(taxable_amount) AS taxable, SUM(cgst_amount) AS cgst,
                  SUM(sgst_amount) AS sgst, SUM(igst_amount) AS igst,
                  SUM(total_amount) AS total
           FROM finance_sales_entries
           WHERE voucher_date >= ? AND voucher_date <= ?""",
        (start_date, end_date),
    ).fetchone()
    ent_taxable = round(float((ent["taxable"] if ent and ent["taxable"] is not None else 0) or 0), 2)
    ent_cgst = round(float((ent["cgst"] if ent and ent["cgst"] is not None else 0) or 0), 2)
    ent_sgst = round(float((ent["sgst"] if ent and ent["sgst"] is not None else 0) or 0), 2)
    ent_igst = round(float((ent["igst"] if ent and ent["igst"] is not None else 0) or 0), 2)
    ent_total = round(float((ent["total"] if ent and ent["total"] is not None else 0) or 0), 2)
    if ent_taxable or ent_total:
        outward["taxable"] = round(outward["taxable"] + ent_taxable, 2)
        outward["cgst"] = round(outward["cgst"] + ent_cgst, 2)
        outward["sgst"] = round(outward["sgst"] + ent_sgst, 2)
        outward["igst"] = round(outward["igst"] + ent_igst, 2)
        outward["total"] = round(outward["total"] + ent_total, 2)
    su = conn.execute(
        """SELECT SUM(su.total_revenue) AS gross
           FROM finance_sales_uploads su
           WHERE su.period >= ? AND su.period <= ?
             AND NOT EXISTS (SELECT 1 FROM finance_sales_entries se WHERE se.sales_upload_id = su.id)""",
        (start_date[:7], end_date[:7]),
    ).fetchone()
    su_gross = round(float((su["gross"] if su and su["gross"] is not None else 0) or 0), 2)
    if su_gross:
        outward["taxable"] = round(outward["taxable"] + su_gross, 2)
        outward["total"] = round(outward["total"] + su_gross, 2)
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
    ent_rows = conn.execute(
        """SELECT id, voucher_date, party_name, invoice_no, order_id,
                  taxable_amount, cgst_amount, sgst_amount, igst_amount, total_amount
           FROM finance_sales_entries
           WHERE voucher_date >= ? AND voucher_date <= ?
           ORDER BY voucher_date ASC, id ASC""",
        (start_date, end_date),
    ).fetchall()
    for r in ent_rows:
        rr = dict(r)
        breakdown.append({
            "voucher_no": f"SUE-{rr.get('id')}",
            "voucher_date": rr.get("voucher_date") or "",
            "voucher_type": "Sales Upload",
            "party_name": rr.get("party_name") or "",
            "taxable_amount": round(float(rr.get("taxable_amount") or 0), 2),
            "cgst_amount": round(float(rr.get("cgst_amount") or 0), 2),
            "sgst_amount": round(float(rr.get("sgst_amount") or 0), 2),
            "igst_amount": round(float(rr.get("igst_amount") or 0), 2),
            "total_amount": round(float(rr.get("total_amount") or 0), 2),
        })
    su_rows = conn.execute(
        """SELECT su.id, su.platform, su.period, su.company_name, su.seller_gstin, su.filename,
                  su.total_revenue, su.created_at
           FROM finance_sales_uploads su
           WHERE su.period >= ? AND su.period <= ?
             AND NOT EXISTS (SELECT 1 FROM finance_sales_entries se WHERE se.sales_upload_id = su.id)
           ORDER BY su.created_at ASC, su.id ASC""",
        (start_date[:7], end_date[:7]),
    ).fetchall()
    for r in su_rows:
        rr = dict(r)
        breakdown.append({
            "voucher_no": f"SU-{rr.get('id')}",
            "voucher_date": f"{str(rr.get('period') or '')[:7]}-01",
            "voucher_type": "Sales Upload",
            "party_name": rr.get("company_name") or rr.get("seller_gstin") or rr.get("platform") or "",
            "taxable_amount": round(float(rr.get("total_revenue") or 0), 2),
            "cgst_amount": 0.0,
            "sgst_amount": 0.0,
            "igst_amount": 0.0,
            "total_amount": round(float(rr.get("total_revenue") or 0), 2),
        })
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
    conn.execute("DELETE FROM finance_sales_entries WHERE sales_upload_id = ?", (upload_id,))
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


# ── Chart of Accounts ─────────────────────────────────────────────

def get_chart_of_accounts() -> dict:
    """Return a hierarchical Chart of Accounts tree: primary groups → sub-groups → ledgers."""
    conn = _connect()
    groups = [dict(r) for r in conn.execute(
        "SELECT * FROM ledger_groups ORDER BY name"
    ).fetchall()]
    ledgers = [dict(r) for r in conn.execute(
        "SELECT id, name, group_id, group_name, opening_balance FROM ledgers WHERE is_active=1 ORDER BY name"
    ).fetchall()]
    conn.close()

    group_by_name = {g['name']: g for g in groups}
    ledgers_by_gid: dict = {}
    for ldr in ledgers:
        gid = ldr.get('group_id')
        if gid:
            ledgers_by_gid.setdefault(gid, []).append(ldr)

    def build_node(g: dict) -> dict:
        node = dict(g)
        node['children'] = sorted(
            [build_node(child) for child in groups
             if (child.get('parent_group') or '') == g['name']],
            key=lambda c: c['name'],
        )
        node['ledgers'] = ledgers_by_gid.get(g['id'], [])
        return node

    # Primary groups = no parent_group or parent not in group_by_name
    nature_order = {'asset': 0, 'liability': 1, 'income': 2, 'expense': 3}
    primary = sorted(
        [build_node(g) for g in groups
         if not (g.get('parent_group') or '').strip()
         or g.get('parent_group') not in group_by_name],
        key=lambda n: (nature_order.get(n['nature'], 9), n['name']),
    )

    return {'groups': primary}


# ── Trial Balance ─────────────────────────────────────────────────

def get_trial_balance(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Return trial balance: opening balance + period Dr/Cr movements per ledger."""
    conn = _connect()

    ledgers = [dict(r) for r in conn.execute("""
        SELECT l.id, l.name, l.group_id, l.group_name, l.opening_balance,
               lg.nature AS group_nature
        FROM ledgers l
        LEFT JOIN ledger_groups lg ON l.group_id = lg.id
        WHERE l.is_active = 1
        ORDER BY lg.nature, l.name
    """).fetchall()]

    q = """
        SELECT evl.expense_head,
               SUM(CASE WHEN evl.is_debit=1 THEN evl.amount ELSE 0 END) AS dr,
               SUM(CASE WHEN evl.is_debit=0 THEN evl.amount ELSE 0 END) AS cr
        FROM expense_voucher_lines evl
        JOIN expense_vouchers ev ON evl.voucher_id = ev.id
        WHERE 1=1
    """
    params: list = []
    if start_date:
        q += " AND ev.voucher_date >= ?"
        params.append(start_date)
    if end_date:
        q += " AND ev.voucher_date <= ?"
        params.append(end_date)
    q += " GROUP BY evl.expense_head"

    movements: dict = {}
    for row in conn.execute(q, params).fetchall():
        movements[row['expense_head']] = {
            'dr': float(row['dr'] or 0),
            'cr': float(row['cr'] or 0),
        }

    # Finance sales: invoice-level net + summary uploads without line detail (no double count).
    ent_q = """
        SELECT SUM(net_payable) AS net_rev
        FROM finance_sales_entries
        WHERE 1=1
    """
    ent_params: list = []
    if start_date:
        ent_q += " AND voucher_date >= ?"
        ent_params.append(start_date)
    if end_date:
        ent_q += " AND voucher_date <= ?"
        ent_params.append(end_date)
    ent_row = conn.execute(ent_q, ent_params).fetchone()
    ent_credit = float((ent_row["net_rev"] if ent_row else 0) or 0)
    su_q = """
        SELECT SUM(su.net_revenue) AS net_rev
        FROM finance_sales_uploads su
        WHERE 1=1
          AND NOT EXISTS (SELECT 1 FROM finance_sales_entries se WHERE se.sales_upload_id = su.id)
    """
    su_params: list = []
    if start_date:
        su_q += " AND su.period >= ?"
        su_params.append(start_date[:7])
    if end_date:
        su_q += " AND su.period <= ?"
        su_params.append(end_date[:7])
    su_row = conn.execute(su_q, su_params).fetchone()
    su_credit = float((su_row["net_rev"] if su_row else 0) or 0)
    total_fin = ent_credit + su_credit
    if total_fin:
        mv = movements.setdefault("Finance Sales Upload A/c", {"dr": 0.0, "cr": 0.0})
        mv["cr"] += total_fin
    conn.close()

    rows = []
    total_dr = 0.0
    total_cr = 0.0

    for ldr in ledgers:
        ob = float(ldr.get('opening_balance') or 0)
        mvmt = movements.get(ldr['name'], {'dr': 0.0, 'cr': 0.0})

        # Opening balance: positive → Dr side (asset/expense convention)
        ob_dr = ob if ob >= 0 else 0.0
        ob_cr = abs(ob) if ob < 0 else 0.0

        dr = round(ob_dr + mvmt['dr'], 2)
        cr = round(ob_cr + mvmt['cr'], 2)

        if dr == 0 and cr == 0:
            continue

        rows.append({
            'ledger':          ldr['name'],
            'group':           ldr.get('group_name') or '',
            'nature':          ldr.get('group_nature') or '',
            'opening_balance': ob,
            'period_dr':       round(mvmt['dr'], 2),
            'period_cr':       round(mvmt['cr'], 2),
            'debit':           dr,
            'credit':          cr,
            'closing':         round(dr - cr, 2),
        })
        total_dr += dr
        total_cr += cr

    return {
        'rows':         rows,
        'total_debit':  round(total_dr, 2),
        'total_credit': round(total_cr, 2),
        'balanced':     abs(total_dr - total_cr) < 0.01,
    }


# ── Tally / Accountant P&L ────────────────────────────────────────

def list_tally_pl() -> list:
    conn = _connect()
    rows = conn.execute("SELECT * FROM tally_pl ORDER BY fy DESC").fetchall()
    return [dict(r) for r in rows]


def upsert_tally_pl(fy: str, opening_stock: float, purchases: float,
                    direct_expenses: float, indirect_expenses: float,
                    sales: float, closing_stock: float,
                    indirect_incomes: float, notes: str = "") -> dict:
    conn = _connect()
    conn.execute("""
        INSERT INTO tally_pl
            (fy, opening_stock, purchases, direct_expenses,
             indirect_expenses, sales, closing_stock, indirect_incomes,
             notes, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(fy) DO UPDATE SET
            opening_stock     = excluded.opening_stock,
            purchases         = excluded.purchases,
            direct_expenses   = excluded.direct_expenses,
            indirect_expenses = excluded.indirect_expenses,
            sales             = excluded.sales,
            closing_stock     = excluded.closing_stock,
            indirect_incomes  = excluded.indirect_incomes,
            notes             = excluded.notes,
            updated_at        = datetime('now')
    """, (fy, opening_stock, purchases, direct_expenses,
          indirect_expenses, sales, closing_stock, indirect_incomes, notes))
    conn.commit()
    row = conn.execute("SELECT * FROM tally_pl WHERE fy=?", (fy,)).fetchone()
    return dict(row)


def delete_tally_pl(fy: str) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM tally_pl WHERE fy=?", (fy,))
    conn.commit()
    return cur.rowcount > 0
