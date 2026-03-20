"""Grey Fabric Module DB — Transit tracking, QC, Hard Reservations"""
import sqlite3, os
from datetime import datetime

_DB = os.path.join(os.path.dirname(__file__), "..", "grey.db")

def _connect():
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

STATUSES = [
    'PO Created', 'Vendor Dispatch Pending', 'In Transit', 'At Transport Location',
    'Sent to Factory', 'At Factory', 'Sent to Printer', 'At Printer',
    'Printed Fabric Received', 'QC Pass', 'QC Reject', 'Rework', 'Closed'
]

def init_db():
    conn = _connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS grey_tracker (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        tracker_key      TEXT UNIQUE NOT NULL,
        po_number        TEXT,
        material_code    TEXT,
        material_name    TEXT,
        supplier         TEXT,
        so_reference     TEXT,
        ordered_qty      REAL DEFAULT 0,
        dispatched_qty   REAL DEFAULT 0,
        received_qty     REAL DEFAULT 0,
        transport_qty    REAL DEFAULT 0,
        factory_qty      REAL DEFAULT 0,
        printer_qty      REAL DEFAULT 0,
        checked_qty      REAL DEFAULT 0,
        rejected_qty     REAL DEFAULT 0,
        rework_qty       REAL DEFAULT 0,
        dispatch_date    TEXT,
        bilty_no         TEXT,
        vendor_invoice   TEXT,
        vendor_challan   TEXT,
        vehicle_no       TEXT,
        transporter      TEXT,
        expected_arrival TEXT,
        status           TEXT DEFAULT 'PO Created',
        qc_status        TEXT DEFAULT 'Pending',
        qc_checked_by    TEXT,
        qc_date          TEXT,
        qc_remarks       TEXT,
        created_at       TEXT DEFAULT (datetime('now')),
        updated_at       TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS grey_ledger (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_date       TEXT NOT NULL,
        tracker_id       INTEGER REFERENCES grey_tracker(id),
        material_code    TEXT,
        material_name    TEXT,
        transaction_type TEXT,
        qty              REAL DEFAULT 0,
        unit             TEXT DEFAULT 'MTR',
        from_location    TEXT,
        to_location      TEXT,
        reference_no     TEXT,
        remarks          TEXT,
        created_at       TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS hard_reservations (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        fabric_code      TEXT NOT NULL,
        fabric_name      TEXT,
        so_number        TEXT,
        sku              TEXT,
        qty              REAL DEFAULT 0,
        unit             TEXT DEFAULT 'MTR',
        reserved_date    TEXT DEFAULT (datetime('now')),
        status           TEXT DEFAULT 'Active',
        remarks          TEXT
    );
    """)
    conn.commit(); conn.close()

def _next_key(conn, po_number, material_code):
    existing = conn.execute("SELECT id FROM grey_tracker WHERE po_number=? AND material_code=?",
        (po_number, material_code)).fetchone()
    if existing: return None  # already exists
    row = conn.execute("SELECT COUNT(*) FROM grey_tracker").fetchone()
    return f"GT-{int(row[0])+1:04d}"

def list_grey(status=None):
    conn = _connect()
    q = "SELECT * FROM grey_tracker WHERE status=? ORDER BY id DESC" if status else "SELECT * FROM grey_tracker ORDER BY id DESC"
    rows = conn.execute(q, (status,) if status else ()).fetchall()
    conn.close(); return [dict(r) for r in rows]

def create_grey_entry(data: dict):
    conn = _connect()
    key = _next_key(conn, data.get('po_number',''), data.get('material_code',''))
    if not key:
        conn.close(); return None
    conn.execute("""INSERT INTO grey_tracker(tracker_key,po_number,material_code,material_name,supplier,so_reference,
        ordered_qty,status) VALUES(?,?,?,?,?,?,?,?)""",
        (key, data.get('po_number',''), data.get('material_code',''), data.get('material_name',''),
         data.get('supplier',''), data.get('so_reference',''), data.get('ordered_qty',0), 'PO Created'))
    _add_ledger_entry(conn, data.get('material_code',''), data.get('material_name',''), 'PO Created',
        data.get('ordered_qty',0), 'MTR', '', 'Ordered', data.get('po_number',''), 'PO created')
    conn.commit(); conn.close(); return key

def update_grey_status(gid: int, data: dict):
    allowed = ['status','dispatched_qty','received_qty','transport_qty','factory_qty','printer_qty',
               'checked_qty','rejected_qty','rework_qty','dispatch_date','bilty_no','vendor_invoice',
               'vendor_challan','vehicle_no','transporter','expected_arrival',
               'qc_status','qc_checked_by','qc_date','qc_remarks']
    sets = ', '.join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed]
    if not sets: return
    sets += ", updated_at=?"
    vals += [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), gid]
    conn = _connect()
    conn.execute(f"UPDATE grey_tracker SET {sets} WHERE id=?", vals)
    # Log to ledger if status changed
    if 'status' in data:
        row = conn.execute("SELECT material_code,material_name FROM grey_tracker WHERE id=?", (gid,)).fetchone()
        if row:
            qty = data.get('dispatched_qty') or data.get('received_qty') or data.get('factory_qty') or 0
            _add_ledger_entry(conn, row['material_code'], row['material_name'], data['status'],
                qty, 'MTR', '', data['status'], '', f"Status → {data['status']}")
    conn.commit(); conn.close()

def _add_ledger_entry(conn, material_code, material_name, txn_type, qty, unit,
                      from_loc, to_loc, ref, remarks):
    conn.execute("""INSERT INTO grey_ledger(entry_date,material_code,material_name,transaction_type,qty,unit,from_location,to_location,reference_no,remarks)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (datetime.now().strftime('%Y-%m-%d'), material_code, material_name, txn_type, qty, unit,
         from_loc, to_loc, ref, remarks))

def list_ledger(material_code=None):
    conn = _connect()
    if material_code:
        rows = conn.execute("SELECT * FROM grey_ledger WHERE material_code=? ORDER BY id DESC", (material_code,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM grey_ledger ORDER BY id DESC LIMIT 200").fetchall()
    conn.close(); return [dict(r) for r in rows]

# ── Hard Reservations ──────────────────────────────────────────────────────────
def list_hard_reservations(status='Active'):
    conn = _connect()
    rows = conn.execute("SELECT * FROM hard_reservations WHERE status=? ORDER BY id DESC", (status,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def create_hard_reservation(data: dict):
    conn = _connect()
    conn.execute("""INSERT INTO hard_reservations(fabric_code,fabric_name,so_number,sku,qty,unit,status,remarks)
        VALUES(?,?,?,?,?,?,?,?)""",
        (data['fabric_code'], data.get('fabric_name',''), data.get('so_number',''),
         data.get('sku',''), data.get('qty',0), data.get('unit','MTR'), 'Active', data.get('remarks','')))
    conn.commit(); conn.close()

def release_hard_reservation(rid: int):
    conn = _connect(); conn.execute("UPDATE hard_reservations SET status='Released' WHERE id=?", (rid,))
    conn.commit(); conn.close()

def get_grey_stats():
    conn = _connect()
    stats = {
        'total_trackers': conn.execute("SELECT COUNT(*) FROM grey_tracker").fetchone()[0],
        'in_transit': conn.execute("SELECT COUNT(*) FROM grey_tracker WHERE status='In Transit'").fetchone()[0],
        'at_factory': conn.execute("SELECT COUNT(*) FROM grey_tracker WHERE status='At Factory'").fetchone()[0],
        'at_printer': conn.execute("SELECT COUNT(*) FROM grey_tracker WHERE status='At Printer'").fetchone()[0],
        'pending_qc': conn.execute("SELECT COUNT(*) FROM grey_tracker WHERE qc_status='Pending' AND status='At Factory'").fetchone()[0],
        'hard_reserved': conn.execute("SELECT COUNT(*) FROM hard_reservations WHERE status='Active'").fetchone()[0],
        'transit_meters': conn.execute("SELECT COALESCE(SUM(dispatched_qty),0) FROM grey_tracker WHERE status='In Transit'").fetchone()[0],
    }
    conn.close(); return stats
