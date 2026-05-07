"""Purchase Module DB — Suppliers, Processors, PR, PO, JWO, GRN"""
import sqlite3, os
from datetime import datetime

_DB = os.environ.get("PURCHASE_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "purchase.db"))

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
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        pr_number        TEXT UNIQUE NOT NULL,
        pr_date          TEXT NOT NULL,
        requested_by     TEXT,
        department       TEXT DEFAULT 'Production',
        priority         TEXT DEFAULT 'Normal',
        status           TEXT DEFAULT 'Pending Approval',
        so_reference     TEXT,
        pr_type          TEXT DEFAULT 'Purchase',
        source           TEXT DEFAULT 'Manual',
        required_by_date TEXT,
        approver         TEXT, approval_date TEXT, approval_remarks TEXT,
        notes            TEXT,
        created_at       TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS pr_lines (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        pr_id            INTEGER NOT NULL REFERENCES pr_headers(id) ON DELETE CASCADE,
        material_code    TEXT NOT NULL,
        material_name    TEXT,
        material_type    TEXT DEFAULT 'RM',
        required_qty     REAL DEFAULT 0,
        po_qty           REAL DEFAULT 0,
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
    CREATE TABLE IF NOT EXISTS material_issue_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        min_number TEXT UNIQUE NOT NULL,
        min_date TEXT NOT NULL,
        jwo_reference TEXT, so_reference TEXT,
        from_location TEXT DEFAULT 'Grey Warehouse',
        to_location TEXT, to_vendor TEXT,
        issued_by TEXT, status TEXT DEFAULT 'Draft',
        remarks TEXT, created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS min_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        min_id INTEGER NOT NULL REFERENCES material_issue_notes(id) ON DELETE CASCADE,
        material_code TEXT NOT NULL, material_name TEXT,
        material_type TEXT DEFAULT 'GF', issue_qty REAL DEFAULT 0,
        unit TEXT DEFAULT 'MTR', rate REAL DEFAULT 0,
        amount REAL DEFAULT 0, remarks TEXT
    );
    CREATE TABLE IF NOT EXISTS gate_passes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gp_number TEXT UNIQUE NOT NULL,
        gp_date TEXT NOT NULL,
        min_reference TEXT, jwo_reference TEXT,
        from_location TEXT DEFAULT 'Factory',
        to_location TEXT, party_name TEXT,
        vehicle_no TEXT, driver_name TEXT,
        material_desc TEXT, total_qty REAL DEFAULT 0,
        unit TEXT DEFAULT 'MTR',
        purpose TEXT DEFAULT 'Job Work',
        status TEXT DEFAULT 'Open',
        remarks TEXT, created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS gate_pass_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gp_id INTEGER NOT NULL REFERENCES gate_passes(id) ON DELETE CASCADE,
        material_code TEXT NOT NULL, material_name TEXT,
        qty REAL DEFAULT 0, unit TEXT DEFAULT 'MTR', remarks TEXT
    );
    """)
    conn.commit()
    for sql in [
        "ALTER TABLE pr_headers ADD COLUMN pr_type TEXT DEFAULT 'Purchase'",
        "ALTER TABLE pr_headers ADD COLUMN source TEXT DEFAULT 'Manual'",
        "ALTER TABLE pr_headers ADD COLUMN required_by_date TEXT",
        "ALTER TABLE pr_lines ADD COLUMN po_qty REAL DEFAULT 0",
    ]:
        try: conn.execute(sql)
        except: pass
    conn.commit()
    conn.close()

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
    conn.execute("""INSERT INTO pr_headers(pr_number,pr_date,requested_by,department,priority,status,so_reference,pr_type,source,required_by_date,notes)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (num, data.get('pr_date') or datetime.now().strftime('%Y-%m-%d'),
        data.get('requested_by') or '', data.get('department') or 'Production',
        data.get('priority') or 'Normal', 'Pending Approval',
        data.get('so_reference') or '',
        data.get('pr_type') or 'Purchase',
        data.get('source') or 'Manual',
        data.get('required_by_date') or '',
        data.get('notes') or ''))
    prid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for ln in data.get('lines', []):
        conn.execute("""INSERT INTO pr_lines(pr_id,material_code,material_name,material_type,required_qty,unit,required_by_date,purpose,remarks)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (prid, ln['material_code'], ln.get('material_name',''), ln.get('material_type','RM'),
            ln.get('required_qty',0), ln.get('unit','PCS'),
            ln.get('required_by_date', data.get('required_by_date', '')),
            ln.get('purpose',''), ln.get('remarks','')))
    conn.commit(); conn.close(); return num

def approve_pr(prid: int, approver: str, remarks: str = ''):
    conn = _connect()
    conn.execute("UPDATE pr_headers SET status='Approved',approver=?,approval_date=?,approval_remarks=? WHERE id=?",
        (approver, datetime.now().strftime('%Y-%m-%d'), remarks, prid))
    conn.commit(); conn.close()

def reject_pr(prid: int, remarks: str = ''):
    conn = _connect()
    conn.execute("UPDATE pr_headers SET status='Rejected',approval_remarks=?,approval_date=? WHERE id=?",
        (remarks, datetime.now().strftime('%Y-%m-%d'), prid))
    conn.commit(); conn.close()

def update_pr_status(prid: int, status: str):
    conn = _connect(); conn.execute("UPDATE pr_headers SET status=? WHERE id=?", (status, prid))
    conn.commit(); conn.close()

def create_pos_from_pr(pr_id: int, lines_data: list, delivery_date: str = '', payment_terms: str = 'Immediate') -> list:
    from collections import defaultdict
    by_supplier: dict = defaultdict(list)
    for ld in lines_data:
        sup_name = (ld.get('supplier_name') or '').strip()
        sup_id = ld.get('supplier_id') or None
        if not sup_name and not sup_id:
            continue
        key = (sup_id, sup_name)
        by_supplier[key].append(ld)
    if not by_supplier:
        return []
    conn = _connect()
    pr = conn.execute("SELECT * FROM pr_headers WHERE id=?", (pr_id,)).fetchone()
    if not pr:
        conn.close(); return []
    pr = dict(pr)
    po_numbers = []
    for (sup_id, sup_name), lines in by_supplier.items():
        num = _next_num(conn, 'po_headers', 'po_number', 'PO')
        subtotal = sum(l.get('qty', 0) * l.get('rate', 0) for l in lines)
        gst = sum(l.get('qty', 0) * l.get('rate', 0) * (l.get('gst_pct') or 0) / 100 for l in lines)
        conn.execute("""INSERT INTO po_headers(po_number,po_date,supplier_id,supplier_name,currency,payment_terms,
            delivery_date,pr_reference,so_reference,status,subtotal,gst_amount,total,remarks)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (num, datetime.now().strftime('%Y-%m-%d'), sup_id, sup_name, 'INR',
            payment_terms or 'Immediate', delivery_date or '',
            pr['pr_number'], pr.get('so_reference') or '',
            'Draft', subtotal, gst, subtotal + gst, ''))
        poid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for ln in lines:
            qty = ln.get('qty', 0); rate = ln.get('rate', 0)
            conn.execute("""INSERT INTO po_lines(po_id,material_code,material_name,material_type,po_qty,unit,rate,gst_pct,amount)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (poid, ln['material_code'], ln.get('material_name', ''), ln.get('material_type', 'RM'),
                qty, ln.get('unit', 'PCS'), rate, ln.get('gst_pct', 0), qty * rate))
            conn.execute("UPDATE pr_lines SET po_qty = po_qty + ? WHERE pr_id=? AND material_code=?",
                (qty, pr_id, ln['material_code']))
        po_numbers.append(num)
        gf_lines = [l for l in lines if l.get('material_type','').upper() == 'GF']
        if gf_lines:
            try:
                import os as _os
                _grey_db = _os.environ.get("GREY_DB_PATH",
                    _os.path.join(_os.path.dirname(__file__), "..", "grey.db"))
                gconn = sqlite3.connect(_grey_db)
                gconn.row_factory = sqlite3.Row
                for ln in gf_lines:
                    mat_code = ln.get('material_code','')
                    existing = gconn.execute(
                        "SELECT id FROM grey_tracker WHERE po_number=? AND material_code=?",
                        (num, mat_code)).fetchone()
                    if not existing:
                        cnt = gconn.execute("SELECT COUNT(*) FROM grey_tracker").fetchone()[0]
                        gt_key = f"GT-{cnt+1:04d}"
                        gconn.execute("""INSERT INTO grey_tracker(tracker_key,po_number,
                            material_code,material_name,supplier,so_reference,ordered_qty,rate,status)
                            VALUES(?,?,?,?,?,?,?,?,?)""",
                            (gt_key, num, mat_code, ln.get('material_name',''),
                            pr.get('supplier_name',''), pr.get('so_reference',''),
                            float(ln.get('qty',0)), float(ln.get('rate',0)), 'PO Created'))
                gconn.commit(); gconn.close()
            except Exception: pass
    all_lines = conn.execute("SELECT required_qty, po_qty FROM pr_lines WHERE pr_id=?", (pr_id,)).fetchall()
    all_covered = all((l['po_qty'] or 0) >= (l['required_qty'] or 0) for l in all_lines)
    conn.execute("UPDATE pr_headers SET status=? WHERE id=?", ('PO Created' if all_covered else 'Partial PO', pr_id))
    conn.commit(); conn.close()
    return po_numbers

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
    gf_lines = [ln for ln in lines if ln.get('material_type','').upper() == 'GF']
    if gf_lines:
        try:
            import os as _os
            _grey_db = _os.environ.get("GREY_DB_PATH",
                _os.path.join(_os.path.dirname(__file__), "..", "grey.db"))
            gconn = sqlite3.connect(_grey_db)
            gconn.row_factory = sqlite3.Row
            for ln in gf_lines:
                mat_code = ln.get('material_code','')
                existing = gconn.execute(
                    "SELECT id FROM grey_tracker WHERE po_number=? AND material_code=?",
                    (num, mat_code)).fetchone()
                if not existing:
                    cnt = gconn.execute("SELECT COUNT(*) FROM grey_tracker").fetchone()[0]
                    gt_key = f"GT-{cnt+1:04d}"
                    gconn.execute("""INSERT INTO grey_tracker(tracker_key,po_number,material_code,
                        material_name,supplier,so_reference,ordered_qty,rate,delivery_location,status)
                        VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (gt_key, num, mat_code, ln.get('material_name',''),
                        data.get('supplier_name',''), data.get('so_reference',''),
                        float(ln.get('po_qty',0)), float(ln.get('rate',0)),
                        data.get('delivery_location',''), 'PO Created'))
                    tid = gconn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    gconn.execute("""INSERT INTO grey_ledger(entry_date,tracker_id,material_code,
                        material_name,transaction_type,qty,unit,from_location,to_location,reference_no,remarks)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (datetime.now().strftime('%Y-%m-%d'), tid, mat_code, ln.get('material_name',''),
                        'PO Created', float(ln.get('po_qty',0)), 'MTR','','Ordered', num, 'Auto from PO'))
            gconn.commit(); gconn.close()
        except Exception: pass
    conn.commit(); conn.close(); return num

def update_po_status(poid: int, status: str):
    conn = _connect(); conn.execute("UPDATE po_headers SET status=? WHERE id=?", (status, poid))
    conn.commit(); conn.close()

def update_po(poid: int, data: dict):
    conn = _connect()
    allowed = ['supplier_id','supplier_name','payment_terms','delivery_location','delivery_date','so_reference','remarks']
    sets = ', '.join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed]
    if sets:
        conn.execute(f"UPDATE po_headers SET {sets} WHERE id=? AND status='Draft'", vals + [poid])
    if 'lines' in data:
        conn.execute("DELETE FROM po_lines WHERE po_id=?", (poid,))
        subtotal = 0; gst_total = 0
        for ln in data['lines']:
            qty = ln.get('po_qty', 0); rate = ln.get('rate', 0)
            amt = qty * rate; gst = amt * (ln.get('gst_pct', 0) / 100)
            subtotal += amt; gst_total += gst
            conn.execute("""INSERT INTO po_lines(po_id,material_code,material_name,material_type,po_qty,unit,rate,gst_pct,amount,remarks)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (poid, ln['material_code'], ln.get('material_name',''), ln.get('material_type','RM'),
                qty, ln.get('unit','PCS'), rate, ln.get('gst_pct',0), amt, ln.get('remarks','')))
        conn.execute("UPDATE po_headers SET subtotal=?,gst_amount=?,total=? WHERE id=?",
            (subtotal, gst_total, subtotal + gst_total, poid))
    conn.commit(); conn.close()

def mark_pr_lines_ordered(pr_id: int, updates: list):
    conn = _connect()
    for u in updates:
        conn.execute("UPDATE pr_lines SET po_qty = po_qty + ? WHERE pr_id=? AND material_code=?",
            (u['qty'], pr_id, u['material_code']))
    all_lines = conn.execute("SELECT required_qty, po_qty FROM pr_lines WHERE pr_id=?", (pr_id,)).fetchall()
    all_covered = all((l['po_qty'] or 0) >= (l['required_qty'] or 0) for l in all_lines)
    conn.execute("UPDATE pr_headers SET status=? WHERE id=?", ('PO Created' if all_covered else 'Partial PO', pr_id))
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
    if lines:
        try:
            import os as _os
            _grey_db = _os.environ.get("GREY_DB_PATH",
                _os.path.join(_os.path.dirname(__file__), "..", "grey.db"))
            gconn = sqlite3.connect(_grey_db)
            gconn.row_factory = sqlite3.Row
            so_ref = data.get('so_reference','')
            for ln in lines:
                input_mat = ln.get('input_material','')
                tracker = gconn.execute("""
                    SELECT id, material_code, material_name FROM grey_tracker
                    WHERE material_code=? AND (so_reference=? OR so_reference='')
                    AND status NOT IN ('Closed','QC Done','Return to Vendor')
                    ORDER BY id DESC LIMIT 1""",
                    (input_mat, so_ref)).fetchone()
                if tracker:
                    gconn.execute("""UPDATE grey_tracker SET job_work_order_no=?, status=?, updated_at=? WHERE id=?""",
                        (num, 'Sent to Printer', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), tracker['id']))
                    gconn.execute("""INSERT INTO grey_ledger(entry_date,tracker_id,material_code,
                        material_name,transaction_type,qty,unit,from_location,to_location,reference_no,remarks)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (datetime.now().strftime('%Y-%m-%d'), tracker['id'],
                        tracker['material_code'], tracker['material_name'],
                        'JWO Created - Sent to Printer', float(ln.get('input_qty',0)),
                        'MTR', 'Factory/Warehouse', 'Printer', num, f"JWO: {num}"))
            gconn.commit(); gconn.close()
        except Exception: pass
    conn.commit(); conn.close(); return num

def update_jwo_status(jwoid: int, status: str):
    conn = _connect(); conn.execute("UPDATE jwo_headers SET status=? WHERE id=?", (status, jwoid))
    conn.commit(); conn.close()

def update_jwo(jwoid: int, data: dict):
    conn = _connect()
    allowed = ['processor_id','processor_name','pr_reference','so_reference','expected_return_date','remarks','issued_by']
    sets = ', '.join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed]
    if sets:
        conn.execute(f"UPDATE jwo_headers SET {sets} WHERE id=? AND status='Draft'", vals + [jwoid])
    if 'lines' in data:
        conn.execute("DELETE FROM jwo_lines WHERE jwo_id=?", (jwoid,))
        total = 0
        for ln in data['lines']:
            out_qty = ln.get('output_qty', 0); rate = ln.get('rate', 0)
            amt = ln.get('amount', out_qty * rate); total += amt
            conn.execute("""INSERT INTO jwo_lines(jwo_id,input_material,input_qty,input_unit,
                output_material,output_qty,output_unit,process_type,rate,amount,remarks)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (jwoid, ln['input_material'], ln.get('input_qty',0), ln.get('input_unit','MTR'),
                ln['output_material'], out_qty, ln.get('output_unit','MTR'),
                ln.get('process_type','Printing'), rate, amt, ln.get('remarks','')))
        conn.execute("UPDATE jwo_headers SET total=? WHERE id=?", (total, jwoid))
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

    # ── PO Receipt → Grey Tracker sync ────────────────────────────────────────
    if data.get('grn_type') == 'PO Receipt':
        try:
            import os as _os
            _grey_db = _os.environ.get("GREY_DB_PATH",
                _os.path.join(_os.path.dirname(__file__), "..", "grey.db"))
            gconn = sqlite3.connect(_grey_db)
            gconn.row_factory = sqlite3.Row
            ref = data.get('reference_number', '')
            for ln in lines:
                mat_code = ln.get('material_code', '')
                tracker = gconn.execute("""SELECT id, material_code, material_name
                    FROM grey_tracker WHERE po_number=? AND material_code=?
                    ORDER BY id DESC LIMIT 1""", (ref, mat_code)).fetchone()
                if tracker:
                    accepted = float(ln.get('accepted_qty', 0))
                    gconn.execute("""UPDATE grey_tracker SET
                        factory_qty = COALESCE(factory_qty,0) + ?,
                        received_qty = COALESCE(received_qty,0) + ?,
                        status = 'At Factory', updated_at = ? WHERE id=?""",
                        (accepted, accepted, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), tracker['id']))
                    gconn.execute("""INSERT INTO grey_ledger(entry_date,tracker_id,material_code,
                        material_name,transaction_type,qty,unit,from_location,to_location,reference_no,remarks)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (datetime.now().strftime('%Y-%m-%d'), tracker['id'],
                        tracker['material_code'], tracker['material_name'],
                        'GRN - Received at Factory', accepted,
                        'MTR', 'Supplier', 'Factory', num, f"GRN: {num}"))
            gconn.commit(); gconn.close()
        except Exception as e:
            print(f"PO GRN sync error: {e}")

    # ── JWO Receipt → Printed Fabric Stock sync ───────────────────────────────
    if data.get('grn_type') == 'JWO Receipt':
        try:
            import os as _os
            _grey_db = _os.environ.get("GREY_DB_PATH",
                _os.path.join(_os.path.dirname(__file__), "..", "grey.db"))
            gconn = sqlite3.connect(_grey_db)
            gconn.row_factory = sqlite3.Row
            ref = data.get('reference_number','')
            # Create table once before loop
            gconn.execute("""CREATE TABLE IF NOT EXISTS printed_fabric_stock (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fabric_code TEXT NOT NULL, fabric_name TEXT DEFAULT '',
                printer TEXT DEFAULT '', qty REAL DEFAULT 0,
                jwo_ref TEXT DEFAULT '', grn_ref TEXT DEFAULT '',
                receive_date TEXT DEFAULT '', status TEXT DEFAULT 'Unchecked',
                created_at TEXT DEFAULT (datetime('now')))""")
            for ln in lines:
                mat_code = ln.get('material_code','')
                tracker = gconn.execute("""SELECT id, material_code, material_name FROM grey_tracker
                    WHERE job_work_order_no=? OR material_code=?
                    ORDER BY id DESC LIMIT 1""", (ref, mat_code)).fetchone()
                if tracker:
                    gconn.execute("""UPDATE grey_tracker SET
                        printed_fabric_qty = COALESCE(printed_fabric_qty,0) + ?,
                        status='Printed Fabric Received', updated_at=? WHERE id=?""",
                        (float(ln.get('accepted_qty',0)),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), tracker['id']))
                    gconn.execute("""INSERT INTO grey_ledger(entry_date,tracker_id,material_code,
                        material_name,transaction_type,qty,unit,from_location,to_location,reference_no,remarks)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (datetime.now().strftime('%Y-%m-%d'), tracker['id'],
                        tracker['material_code'], tracker['material_name'],
                        'Printed Fabric Received', float(ln.get('accepted_qty',0)),
                        'MTR', 'Printer', 'Checking Warehouse', num, f"GRN: {num}"))
                # Insert to printed_fabric_stock for EVERY line (outside if tracker)
                gconn.execute("""INSERT INTO printed_fabric_stock
                    (fabric_code, fabric_name, printer, qty, jwo_ref, grn_ref, receive_date)
                    VALUES (?,?,?,?,?,?,?)""",
                    (ln.get('material_code',''), ln.get('material_name',''),
                    data.get('party_name',''), float(ln.get('accepted_qty',0)),
                    ref, num, datetime.now().strftime('%Y-%m-%d')))
            gconn.commit(); gconn.close()
        except Exception as e:
            print(f"JWO GRN sync error: {e}")

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

def get_po_by_number(po_number: str):
    conn = _connect()
    row = conn.execute("SELECT * FROM po_headers WHERE po_number=?", (po_number,)).fetchone()
    if not row: conn.close(); return None
    d = dict(row)
    d['lines'] = [dict(l) for l in conn.execute("SELECT * FROM po_lines WHERE po_id=?", (d['id'],)).fetchall()]
    conn.close(); return d

def get_jwo_by_number(jwo_number: str):
    conn = _connect()
    row = conn.execute("SELECT * FROM jwo_headers WHERE jwo_number=?", (jwo_number,)).fetchone()
    if not row: conn.close(); return None
    d = dict(row)
    d['lines'] = [dict(l) for l in conn.execute("SELECT * FROM jwo_lines WHERE jwo_id=?", (d['id'],)).fetchall()]
    conn.close(); return d

# ── Material Issue Note (MIN) ─────────────────────────────────────────────────
def init_min_table():
    conn = _connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS material_issue_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, min_number TEXT UNIQUE NOT NULL,
        min_date TEXT NOT NULL, jwo_reference TEXT, so_reference TEXT,
        from_location TEXT DEFAULT 'Grey Warehouse', to_location TEXT,
        to_vendor TEXT, issued_by TEXT, status TEXT DEFAULT 'Draft',
        remarks TEXT, created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS min_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        min_id INTEGER NOT NULL REFERENCES material_issue_notes(id) ON DELETE CASCADE,
        material_code TEXT NOT NULL, material_name TEXT,
        material_type TEXT DEFAULT 'GF', issue_qty REAL DEFAULT 0,
        unit TEXT DEFAULT 'MTR', rate REAL DEFAULT 0, amount REAL DEFAULT 0, remarks TEXT
    );
    """)
    conn.commit(); conn.close()

def list_mins(status=None):
    conn = _connect()
    q = "SELECT * FROM material_issue_notes WHERE status=? ORDER BY id DESC" if status else "SELECT * FROM material_issue_notes ORDER BY id DESC"
    rows = conn.execute(q, (status,) if status else ()).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d['lines'] = [dict(l) for l in conn.execute("SELECT * FROM min_lines WHERE min_id=?", (d['id'],)).fetchall()]
        result.append(d)
    conn.close(); return result

def create_min(data: dict):
    conn = _connect()
    conn.execute("""CREATE TABLE IF NOT EXISTS material_issue_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, min_number TEXT UNIQUE NOT NULL,
        min_date TEXT NOT NULL, jwo_reference TEXT, so_reference TEXT,
        from_location TEXT DEFAULT 'Grey Warehouse', to_location TEXT,
        to_vendor TEXT, issued_by TEXT, status TEXT DEFAULT 'Draft',
        remarks TEXT, created_at TEXT DEFAULT (datetime('now')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS min_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        min_id INTEGER NOT NULL REFERENCES material_issue_notes(id) ON DELETE CASCADE,
        material_code TEXT NOT NULL, material_name TEXT,
        material_type TEXT DEFAULT 'GF', issue_qty REAL DEFAULT 0,
        unit TEXT DEFAULT 'MTR', rate REAL DEFAULT 0, amount REAL DEFAULT 0, remarks TEXT)""")
    num = _next_num(conn, 'material_issue_notes', 'min_number', 'MIN')
    lines = data.get('lines', [])
    conn.execute("""INSERT INTO material_issue_notes(min_number,min_date,jwo_reference,
        so_reference,from_location,to_location,to_vendor,issued_by,status,remarks)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (num, data.get('min_date') or datetime.now().strftime('%Y-%m-%d'),
        data.get('jwo_reference',''), data.get('so_reference',''),
        data.get('from_location','Grey Warehouse'), data.get('to_location',''),
        data.get('to_vendor',''), data.get('issued_by',''), 'Draft', data.get('remarks','')))
    minid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for ln in lines:
        amt = ln.get('issue_qty',0) * ln.get('rate',0)
        conn.execute("""INSERT INTO min_lines(min_id,material_code,material_name,
            material_type,issue_qty,unit,rate,amount,remarks)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (minid, ln['material_code'], ln.get('material_name',''),
            ln.get('material_type','GF'), ln.get('issue_qty',0),
            ln.get('unit','MTR'), ln.get('rate',0), amt, ln.get('remarks','')))
    conn.commit(); conn.close(); return num

def update_min_status(minid: int, status: str):
    conn = _connect()
    conn.execute("UPDATE material_issue_notes SET status=? WHERE id=?", (status, minid))
    conn.commit(); conn.close()

def get_min_by_number(min_number: str):
    conn = _connect()
    row = conn.execute("SELECT * FROM material_issue_notes WHERE min_number=?", (min_number,)).fetchone()
    if not row: conn.close(); return None
    d = dict(row)
    d['lines'] = [dict(l) for l in conn.execute("SELECT * FROM min_lines WHERE min_id=?", (d['id'],)).fetchall()]
    conn.close(); return d

# ── Gate Pass ──────────────────────────────────────────────────────────────
def create_gate_pass(data: dict):
    conn = _connect()
    conn.execute("""CREATE TABLE IF NOT EXISTS gate_passes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, gp_number TEXT UNIQUE NOT NULL,
        gp_date TEXT NOT NULL, min_reference TEXT, jwo_reference TEXT,
        from_location TEXT DEFAULT 'Factory', to_location TEXT, party_name TEXT,
        vehicle_no TEXT, driver_name TEXT, material_desc TEXT, total_qty REAL DEFAULT 0,
        unit TEXT DEFAULT 'MTR', purpose TEXT DEFAULT 'Job Work',
        status TEXT DEFAULT 'Open', remarks TEXT, created_at TEXT DEFAULT (datetime('now')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS gate_pass_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gp_id INTEGER NOT NULL REFERENCES gate_passes(id) ON DELETE CASCADE,
        material_code TEXT NOT NULL, material_name TEXT,
        qty REAL DEFAULT 0, unit TEXT DEFAULT 'MTR', remarks TEXT)""")
    num = _next_num(conn, 'gate_passes', 'gp_number', 'GP')
    lines = data.get('lines', [])
    total = sum(l.get('qty', 0) for l in lines)
    conn.execute("""INSERT INTO gate_passes(gp_number,gp_date,min_reference,jwo_reference,
        from_location,to_location,party_name,vehicle_no,driver_name,
        material_desc,total_qty,unit,purpose,status,remarks)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (num, data.get('gp_date') or datetime.now().strftime('%Y-%m-%d'),
        data.get('min_reference',''), data.get('jwo_reference',''),
        data.get('from_location','Factory'), data.get('to_location',''),
        data.get('party_name',''), data.get('vehicle_no',''),
        data.get('driver_name',''), data.get('material_desc',''),
        total, data.get('unit','MTR'), data.get('purpose','Job Work'), 'Open', data.get('remarks','')))
    gpid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for ln in lines:
        conn.execute("""INSERT INTO gate_pass_lines(gp_id,material_code,material_name,qty,unit,remarks)
            VALUES(?,?,?,?,?,?)""",
            (gpid, ln['material_code'], ln.get('material_name',''),
            ln.get('qty',0), ln.get('unit','MTR'), ln.get('remarks','')))
    conn.commit(); conn.close(); return num

def list_gate_passes():
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM gate_passes ORDER BY id DESC").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d['lines'] = [dict(l) for l in conn.execute(
                "SELECT * FROM gate_pass_lines WHERE gp_id=?", (d['id'],)).fetchall()]
            result.append(d)
        conn.close(); return result
    except Exception:
        conn.close(); return []

def get_gate_pass_by_number(gp_number: str):
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM gate_passes WHERE gp_number=?", (gp_number,)).fetchone()
        if not row: conn.close(); return None
        d = dict(row)
        d['lines'] = [dict(l) for l in conn.execute(
            "SELECT * FROM gate_pass_lines WHERE gp_id=?", (d['id'],)).fetchall()]
        conn.close(); return d
    except Exception:
        conn.close(); return None