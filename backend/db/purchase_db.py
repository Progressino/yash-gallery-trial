"""Purchase Module DB — Suppliers, Processors, PR, PO, JWO, GRN"""
import sqlite3, os
from datetime import datetime

_DB = os.path.join(os.path.dirname(__file__), "..", "purchase.db")

def _connect():
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = _connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS suppliers (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        supplier_code  TEXT UNIQUE NOT NULL,
        supplier_name  TEXT NOT NULL,
        supplier_type  TEXT DEFAULT 'Others',
        contact_person TEXT, email TEXT, phone TEXT,
        address TEXT, gst_number TEXT,
        payment_terms  TEXT DEFAULT 'Net 30',
        active         INTEGER DEFAULT 1,
        created_at     TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS processors (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        processor_code TEXT UNIQUE NOT NULL,
        processor_name TEXT NOT NULL,
        processor_type TEXT DEFAULT 'Others',
        contact_person TEXT, email TEXT, phone TEXT,
        address TEXT,
        active         INTEGER DEFAULT 1,
        created_at     TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS pr_headers (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        pr_number    TEXT UNIQUE NOT NULL,
        pr_date      TEXT NOT NULL,
        requested_by TEXT,
        department   TEXT DEFAULT 'Production',
        priority     TEXT DEFAULT 'Normal',
        status       TEXT DEFAULT 'Draft',
        so_reference TEXT,
        approver     TEXT, approval_date TEXT, approval_remarks TEXT,
        notes        TEXT,
        created_at   TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS pr_lines (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        pr_id            INTEGER NOT NULL REFERENCES pr_headers(id) ON DELETE CASCADE,
        material_code    TEXT NOT NULL,
        material_name    TEXT,
        material_type    TEXT DEFAULT 'RM',
        required_qty     REAL DEFAULT 0,
        unit             TEXT DEFAULT 'PCS',
        required_by_date TEXT,
        purpose          TEXT,
        remarks          TEXT
    );
    CREATE TABLE IF NOT EXISTS po_headers (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        po_number       TEXT UNIQUE NOT NULL,
        po_date         TEXT NOT NULL,
        supplier_id     INTEGER REFERENCES suppliers(id),
        supplier_name   TEXT,
        currency        TEXT DEFAULT 'INR',
        payment_terms   TEXT,
        delivery_location TEXT,
        delivery_date   TEXT,
        pr_reference    TEXT,
        so_reference    TEXT,
        status          TEXT DEFAULT 'Draft',
        subtotal        REAL DEFAULT 0,
        gst_amount      REAL DEFAULT 0,
        total           REAL DEFAULT 0,
        remarks         TEXT,
        created_at      TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS po_lines (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        po_id          INTEGER NOT NULL REFERENCES po_headers(id) ON DELETE CASCADE,
        material_code  TEXT NOT NULL,
        material_name  TEXT,
        material_type  TEXT DEFAULT 'RM',
        po_qty         REAL DEFAULT 0,
        unit           TEXT DEFAULT 'PCS',
        rate           REAL DEFAULT 0,
        gst_pct        INTEGER DEFAULT 0,
        amount         REAL DEFAULT 0,
        remarks        TEXT
    );
    CREATE TABLE IF NOT EXISTS jwo_headers (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        jwo_number           TEXT UNIQUE NOT NULL,
        jwo_date             TEXT NOT NULL,
        processor_id         INTEGER REFERENCES processors(id),
        processor_name       TEXT,
        pr_reference         TEXT,
        so_reference         TEXT,
        expected_return_date TEXT,
        status               TEXT DEFAULT 'Draft',
        total                REAL DEFAULT 0,
        remarks              TEXT,
        issued_by            TEXT,
        created_at           TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS jwo_lines (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        jwo_id           INTEGER NOT NULL REFERENCES jwo_headers(id) ON DELETE CASCADE,
        input_material   TEXT NOT NULL,
        input_qty        REAL DEFAULT 0,
        input_unit       TEXT DEFAULT 'MTR',
        output_material  TEXT NOT NULL,
        output_qty       REAL DEFAULT 0,
        output_unit      TEXT DEFAULT 'MTR',
        process_type     TEXT DEFAULT 'Printing',
        rate             REAL DEFAULT 0,
        amount           REAL DEFAULT 0,
        remarks          TEXT
    );
    CREATE TABLE IF NOT EXISTS grn_headers (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        grn_number       TEXT UNIQUE NOT NULL,
        grn_date         TEXT NOT NULL,
        grn_type         TEXT DEFAULT 'PO Receipt',
        reference_number TEXT,
        party_name       TEXT,
        challan_no       TEXT,
        invoice_no       TEXT,
        invoice_date     TEXT,
        vehicle_no       TEXT,
        transporter      TEXT,
        warehouse        TEXT,
        so_reference     TEXT,
        total_value      REAL DEFAULT 0,
        status           TEXT DEFAULT 'Draft',
        qc_checked_by    TEXT, qc_date TEXT,
        remarks          TEXT,
        created_at       TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS grn_lines (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        grn_id          INTEGER NOT NULL REFERENCES grn_headers(id) ON DELETE CASCADE,
        material_code   TEXT NOT NULL,
        material_name   TEXT,
        material_type   TEXT DEFAULT 'RM',
        po_qty          REAL DEFAULT 0,
        received_qty    REAL DEFAULT 0,
        accepted_qty    REAL DEFAULT 0,
        rejected_qty    REAL DEFAULT 0,
        unit            TEXT DEFAULT 'PCS',
        rate            REAL DEFAULT 0,
        amount          REAL DEFAULT 0,
        qc_status       TEXT DEFAULT 'Pending',
        rejection_reason TEXT
    );
    """)
    conn.commit(); conn.close()

def _next_num(conn, table, col, prefix):
    row = conn.execute(f"SELECT {col} FROM {table} ORDER BY id DESC LIMIT 1").fetchone()
    n = 1
    if row:
        try: n = int(row[0].split('-')[-1]) + 1
        except: pass
    return f"{prefix}-{n:04d}"

# ── Suppliers ──────────────────────────────────────────────────────────────────
def list_suppliers(active_only=True):
    conn = _connect()
    q = "SELECT * FROM suppliers WHERE active=1 ORDER BY supplier_name" if active_only else "SELECT * FROM suppliers ORDER BY supplier_name"
    rows = conn.execute(q).fetchall()
    conn.close(); return [dict(r) for r in rows]

def create_supplier(data: dict):
    conn = _connect()
    code = data.get('supplier_code') or f"SUP-{conn.execute('SELECT COUNT(*)+1 FROM suppliers').fetchone()[0]:03d}"
    conn.execute("""INSERT INTO suppliers(supplier_code,supplier_name,supplier_type,contact_person,email,phone,address,gst_number,payment_terms)
        VALUES(?,?,?,?,?,?,?,?,?)""",
        (code, data['supplier_name'], data.get('supplier_type','Others'),
         data.get('contact_person',''), data.get('email',''), data.get('phone',''),
         data.get('address',''), data.get('gst_number',''), data.get('payment_terms','Net 30')))
    conn.commit(); conn.close()

def update_supplier(sid: int, data: dict):
    allowed = ['supplier_name','supplier_type','contact_person','email','phone','address','gst_number','payment_terms','active']
    sets = ', '.join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed] + [sid]
    if not sets: return
    conn = _connect(); conn.execute(f"UPDATE suppliers SET {sets} WHERE id=?", vals)
    conn.commit(); conn.close()

# ── Processors ────────────────────────────────────────────────────────────────
def list_processors(active_only=True):
    conn = _connect()
    q = "SELECT * FROM processors WHERE active=1 ORDER BY processor_name" if active_only else "SELECT * FROM processors ORDER BY processor_name"
    rows = conn.execute(q).fetchall()
    conn.close(); return [dict(r) for r in rows]

def create_processor(data: dict):
    conn = _connect()
    code = data.get('processor_code') or f"PRC-{conn.execute('SELECT COUNT(*)+1 FROM processors').fetchone()[0]:03d}"
    conn.execute("""INSERT INTO processors(processor_code,processor_name,processor_type,contact_person,email,phone,address)
        VALUES(?,?,?,?,?,?,?)""",
        (code, data['processor_name'], data.get('processor_type','Others'),
         data.get('contact_person',''), data.get('email',''), data.get('phone',''), data.get('address','')))
    conn.commit(); conn.close()

# ── Purchase Requisitions ─────────────────────────────────────────────────────
def list_prs(status=None):
    conn = _connect()
    q = "SELECT * FROM pr_headers WHERE status=? ORDER BY id DESC" if status else "SELECT * FROM pr_headers ORDER BY id DESC"
    rows = conn.execute(q, (status,) if status else ()).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d['lines'] = [dict(l) for l in conn.execute("SELECT * FROM pr_lines WHERE pr_id=?", (d['id'],)).fetchall()]
        result.append(d)
    conn.close(); return result

def create_pr(data: dict):
    conn = _connect()
    num = _next_num(conn, 'pr_headers', 'pr_number', 'PR')
    conn.execute("""INSERT INTO pr_headers(pr_number,pr_date,requested_by,department,priority,status,so_reference,notes)
        VALUES(?,?,?,?,?,?,?,?)""",
        (num, data.get('pr_date') or datetime.now().strftime('%Y-%m-%d'),
         data.get('requested_by') or '', data.get('department') or 'Production',
         data.get('priority') or 'Normal', 'Draft', data.get('so_reference') or '', data.get('notes') or ''))
    prid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for ln in data.get('lines', []):
        conn.execute("""INSERT INTO pr_lines(pr_id,material_code,material_name,material_type,required_qty,unit,required_by_date,purpose,remarks)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (prid, ln['material_code'], ln.get('material_name',''), ln.get('material_type','RM'),
             ln.get('required_qty',0), ln.get('unit','PCS'), ln.get('required_by_date',''),
             ln.get('purpose',''), ln.get('remarks','')))
    conn.commit(); conn.close(); return num

def approve_pr(prid: int, approver: str, remarks: str = ''):
    conn = _connect()
    conn.execute("UPDATE pr_headers SET status='Approved',approver=?,approval_date=?,approval_remarks=? WHERE id=?",
        (approver, datetime.now().strftime('%Y-%m-%d'), remarks, prid))
    conn.commit(); conn.close()

def update_pr_status(prid: int, status: str):
    conn = _connect(); conn.execute("UPDATE pr_headers SET status=? WHERE id=?", (status, prid))
    conn.commit(); conn.close()

# ── Purchase Orders ───────────────────────────────────────────────────────────
def list_pos(status=None):
    conn = _connect()
    q = "SELECT * FROM po_headers WHERE status=? ORDER BY id DESC" if status else "SELECT * FROM po_headers ORDER BY id DESC"
    rows = conn.execute(q, (status,) if status else ()).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d['lines'] = [dict(l) for l in conn.execute("SELECT * FROM po_lines WHERE po_id=?", (d['id'],)).fetchall()]
        result.append(d)
    conn.close(); return result

def create_po(data: dict):
    conn = _connect()
    num = _next_num(conn, 'po_headers', 'po_number', 'PO')
    lines = data.get('lines', [])
    subtotal = sum(l.get('amount', l.get('po_qty',0)*l.get('rate',0)) for l in lines)
    gst = sum(l.get('amount',0) * l.get('gst_pct',0) / 100 for l in lines)
    conn.execute("""INSERT INTO po_headers(po_number,po_date,supplier_id,supplier_name,currency,payment_terms,
        delivery_location,delivery_date,pr_reference,so_reference,status,subtotal,gst_amount,total,remarks)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (num, data.get('po_date') or datetime.now().strftime('%Y-%m-%d'),
         data.get('supplier_id'), data.get('supplier_name') or '',
         data.get('currency') or 'INR', data.get('payment_terms') or '',
         data.get('delivery_location') or '', data.get('delivery_date') or '',
         data.get('pr_reference') or '', data.get('so_reference') or '',
         'Draft', subtotal, gst, subtotal+gst, data.get('remarks') or ''))
    poid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for ln in lines:
        amt = ln.get('amount', ln.get('po_qty',0)*ln.get('rate',0))
        conn.execute("""INSERT INTO po_lines(po_id,material_code,material_name,material_type,po_qty,unit,rate,gst_pct,amount,remarks)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (poid, ln['material_code'], ln.get('material_name',''), ln.get('material_type','RM'),
             ln.get('po_qty',0), ln.get('unit','PCS'), ln.get('rate',0), ln.get('gst_pct',0), amt, ln.get('remarks','')))
    conn.commit(); conn.close(); return num

def update_po_status(poid: int, status: str):
    conn = _connect(); conn.execute("UPDATE po_headers SET status=? WHERE id=?", (status, poid))
    conn.commit(); conn.close()

# ── Job Work Orders ───────────────────────────────────────────────────────────
def list_jwos(status=None):
    conn = _connect()
    q = "SELECT * FROM jwo_headers WHERE status=? ORDER BY id DESC" if status else "SELECT * FROM jwo_headers ORDER BY id DESC"
    rows = conn.execute(q, (status,) if status else ()).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d['lines'] = [dict(l) for l in conn.execute("SELECT * FROM jwo_lines WHERE jwo_id=?", (d['id'],)).fetchall()]
        result.append(d)
    conn.close(); return result

def create_jwo(data: dict):
    conn = _connect()
    num = _next_num(conn, 'jwo_headers', 'jwo_number', 'JWO')
    lines = data.get('lines', [])
    total = sum(l.get('amount', l.get('output_qty',0)*l.get('rate',0)) for l in lines)
    conn.execute("""INSERT INTO jwo_headers(jwo_number,jwo_date,processor_id,processor_name,pr_reference,so_reference,
        expected_return_date,status,total,remarks,issued_by) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (num, data.get('jwo_date') or datetime.now().strftime('%Y-%m-%d'),
         data.get('processor_id'), data.get('processor_name') or '',
         data.get('pr_reference') or '', data.get('so_reference') or '',
         data.get('expected_return_date') or '', 'Draft', total,
         data.get('remarks') or '', data.get('issued_by') or ''))
    jwoid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for ln in lines:
        amt = ln.get('amount', ln.get('output_qty',0)*ln.get('rate',0))
        conn.execute("""INSERT INTO jwo_lines(jwo_id,input_material,input_qty,input_unit,output_material,output_qty,output_unit,process_type,rate,amount,remarks)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (jwoid, ln['input_material'], ln.get('input_qty',0), ln.get('input_unit','MTR'),
             ln['output_material'], ln.get('output_qty',0), ln.get('output_unit','MTR'),
             ln.get('process_type','Printing'), ln.get('rate',0), amt, ln.get('remarks','')))
    conn.commit(); conn.close(); return num

def update_jwo_status(jwoid: int, status: str):
    conn = _connect(); conn.execute("UPDATE jwo_headers SET status=? WHERE id=?", (status, jwoid))
    conn.commit(); conn.close()

# ── GRN ───────────────────────────────────────────────────────────────────────
def list_grns(status=None):
    conn = _connect()
    q = "SELECT * FROM grn_headers WHERE status=? ORDER BY id DESC" if status else "SELECT * FROM grn_headers ORDER BY id DESC"
    rows = conn.execute(q, (status,) if status else ()).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d['lines'] = [dict(l) for l in conn.execute("SELECT * FROM grn_lines WHERE grn_id=?", (d['id'],)).fetchall()]
        result.append(d)
    conn.close(); return result

def create_grn(data: dict):
    conn = _connect()
    num = _next_num(conn, 'grn_headers', 'grn_number', 'GRN')
    lines = data.get('lines', [])
    total = sum(l.get('amount', l.get('accepted_qty',0)*l.get('rate',0)) for l in lines)
    conn.execute("""INSERT INTO grn_headers(grn_number,grn_date,grn_type,reference_number,party_name,challan_no,
        invoice_no,invoice_date,vehicle_no,transporter,warehouse,so_reference,total_value,status,remarks)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (num, data.get('grn_date') or datetime.now().strftime('%Y-%m-%d'),
         data.get('grn_type') or 'PO Receipt', data.get('reference_number') or '',
         data.get('party_name') or '', data.get('challan_no') or '',
         data.get('invoice_no') or '', data.get('invoice_date') or '',
         data.get('vehicle_no') or '', data.get('transporter') or '',
         data.get('warehouse') or '', data.get('so_reference') or '',
         total, 'Draft', data.get('remarks') or ''))
    grnid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for ln in lines:
        amt = ln.get('amount', ln.get('accepted_qty',0)*ln.get('rate',0))
        conn.execute("""INSERT INTO grn_lines(grn_id,material_code,material_name,material_type,po_qty,received_qty,accepted_qty,rejected_qty,unit,rate,amount,qc_status,rejection_reason)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (grnid, ln['material_code'], ln.get('material_name',''), ln.get('material_type','RM'),
             ln.get('po_qty',0), ln.get('received_qty',0), ln.get('accepted_qty',0),
             ln.get('rejected_qty',0), ln.get('unit','PCS'), ln.get('rate',0), amt,
             ln.get('qc_status','Pending'), ln.get('rejection_reason','')))
    conn.commit(); conn.close(); return num

def update_grn_status(grnid: int, status: str, qc_by: str = ''):
    conn = _connect()
    conn.execute("UPDATE grn_headers SET status=?,qc_checked_by=?,qc_date=? WHERE id=?",
        (status, qc_by, datetime.now().strftime('%Y-%m-%d'), grnid))
    conn.commit(); conn.close()

def get_purchase_stats():
    conn = _connect()
    stats = {
        'open_prs': conn.execute("SELECT COUNT(*) FROM pr_headers WHERE status NOT IN ('Closed','Rejected')").fetchone()[0],
        'open_pos': conn.execute("SELECT COUNT(*) FROM po_headers WHERE status NOT IN ('Closed','Cancelled')").fetchone()[0],
        'open_jwos': conn.execute("SELECT COUNT(*) FROM jwo_headers WHERE status NOT IN ('Closed','Cancelled')").fetchone()[0],
        'pending_grns': conn.execute("SELECT COUNT(*) FROM grn_headers WHERE status='Draft'").fetchone()[0],
        'total_suppliers': conn.execute("SELECT COUNT(*) FROM suppliers WHERE active=1").fetchone()[0],
        'total_processors': conn.execute("SELECT COUNT(*) FROM processors WHERE active=1").fetchone()[0],
    }
    conn.close(); return stats
