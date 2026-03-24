"""Production Module DB — Job Orders + MRP Soft Reservations"""
import sqlite3, os, json
from datetime import datetime

_DB = os.path.join(os.path.dirname(__file__), "..", "production.db")

def _connect():
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = _connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS job_orders (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        jo_number          TEXT UNIQUE NOT NULL,
        jo_date            TEXT NOT NULL,
        so_number          TEXT,
        sku                TEXT,
        sku_name           TEXT,
        process            TEXT,
        exec_type          TEXT DEFAULT 'Inhouse',
        vendor_name        TEXT,
        so_qty             INTEGER DEFAULT 0,
        planned_qty        INTEGER DEFAULT 0,
        output_qty         INTEGER DEFAULT 0,
        status             TEXT DEFAULT 'Created',
        expected_completion TEXT,
        completed_date     TEXT,
        issued_to          TEXT,
        remarks            TEXT,
        created_at         TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS jo_lines (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        jo_id          INTEGER NOT NULL REFERENCES job_orders(id) ON DELETE CASCADE,
        sku            TEXT,
        sku_name       TEXT,
        planned_qty    INTEGER DEFAULT 0,
        output_qty     INTEGER DEFAULT 0,
        input_material TEXT,
        input_qty      REAL DEFAULT 0,
        input_unit     TEXT DEFAULT 'PCS',
        process_yield  REAL DEFAULT 100,
        remarks        TEXT
    );
    CREATE TABLE IF NOT EXISTS soft_reservations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        material_code   TEXT NOT NULL,
        material_name   TEXT,
        reserved_qty    REAL DEFAULT 0,
        unit            TEXT DEFAULT 'PCS',
        against_so      TEXT,
        reservation_date TEXT DEFAULT (datetime('now')),
        status          TEXT DEFAULT 'Active',
        remarks         TEXT
    );
    CREATE TABLE IF NOT EXISTS mrp_soft_reservations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        material_code   TEXT NOT NULL,
        material_name   TEXT,
        unit            TEXT DEFAULT 'PCS',
        so_no           TEXT NOT NULL,
        sku             TEXT,
        qty             REAL DEFAULT 0,
        status          TEXT DEFAULT 'Active',
        created_at      TEXT DEFAULT (datetime('now')),
        UNIQUE(material_code, so_no, sku)
    );
    CREATE TABLE IF NOT EXISTS mrp_last_run (
        id          INTEGER PRIMARY KEY,
        run_time    TEXT NOT NULL,
        so_numbers  TEXT NOT NULL,
        result_json TEXT NOT NULL
    );
    """)
    conn.commit(); conn.close()

def _next_jo(conn):
    row = conn.execute("SELECT jo_number FROM job_orders ORDER BY id DESC LIMIT 1").fetchone()
    n = 1
    if row:
        try: n = int(row[0].split('-')[-1]) + 1
        except: pass
    return f"PJO-{n:04d}"

def list_jos(status=None, so_number=None):
    conn = _connect()
    conditions = []
    params = []
    if status:
        conditions.append("status=?"); params.append(status)
    if so_number:
        conditions.append("so_number=?"); params.append(so_number)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = conn.execute(f"SELECT * FROM job_orders {where} ORDER BY id DESC", params).fetchall()
    result = []
    for r in rows:
        jo = dict(r)
        jo['lines'] = [dict(l) for l in conn.execute("SELECT * FROM jo_lines WHERE jo_id=?", (jo['id'],)).fetchall()]
        result.append(jo)
    conn.close(); return result

def create_jo(data: dict):
    conn = _connect()
    num = _next_jo(conn)
    conn.execute("""INSERT INTO job_orders(jo_number,jo_date,so_number,sku,sku_name,process,exec_type,vendor_name,
        so_qty,planned_qty,status,expected_completion,issued_to,remarks)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (num, data.get('jo_date') or datetime.now().strftime('%Y-%m-%d'),
         data.get('so_number') or '', data.get('sku') or '', data.get('sku_name') or '',
         data.get('process') or 'Cutting', data.get('exec_type') or 'Inhouse',
         data.get('vendor_name') or '', data.get('so_qty') or 0, data.get('planned_qty') or 0,
         'Created', data.get('expected_completion') or '',
         data.get('issued_to') or '', data.get('remarks') or ''))
    joid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for ln in data.get('lines', []):
        conn.execute("""INSERT INTO jo_lines(jo_id,sku,sku_name,planned_qty,input_material,input_qty,input_unit,remarks)
            VALUES(?,?,?,?,?,?,?,?)""",
            (joid, ln.get('sku',''), ln.get('sku_name',''), ln.get('planned_qty',0),
             ln.get('input_material',''), ln.get('input_qty',0), ln.get('input_unit','PCS'), ln.get('remarks','')))
    conn.commit(); conn.close(); return num

def update_jo(joid: int, data: dict):
    allowed = ['status','output_qty','completed_date','remarks','issued_to','exec_type','vendor_name']
    sets = ', '.join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed] + [joid]
    if not sets: return
    conn = _connect(); conn.execute(f"UPDATE job_orders SET {sets} WHERE id=?", vals)
    conn.commit(); conn.close()

# ── MRP Soft Reservations ──────────────────────────────────────────────────────
def list_reservations(status='Active'):
    conn = _connect()
    rows = conn.execute("SELECT * FROM soft_reservations WHERE status=? ORDER BY id DESC", (status,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def create_reservation(data: dict):
    conn = _connect()
    conn.execute("""INSERT INTO soft_reservations(material_code,material_name,reserved_qty,unit,against_so,status,remarks)
        VALUES(?,?,?,?,?,?,?)""",
        (data['material_code'], data.get('material_name',''), data.get('reserved_qty',0),
         data.get('unit','PCS'), data.get('against_so',''), 'Active', data.get('remarks','')))
    conn.commit(); conn.close()

def release_reservation(rid: int):
    conn = _connect(); conn.execute("UPDATE soft_reservations SET status='Released' WHERE id=?", (rid,))
    conn.commit(); conn.close()

def get_reserved_qty(material_code: str) -> float:
    conn = _connect()
    row = conn.execute("SELECT COALESCE(SUM(reserved_qty),0) FROM soft_reservations WHERE material_code=? AND status='Active'",
        (material_code,)).fetchone()
    conn.close(); return float(row[0])

def get_production_stats():
    conn = _connect()
    stats = {
        'total_jos': conn.execute("SELECT COUNT(*) FROM job_orders").fetchone()[0],
        'open_jos': conn.execute("SELECT COUNT(*) FROM job_orders WHERE status NOT IN ('Completed','Closed','Cancelled')").fetchone()[0],
        'in_progress': conn.execute("SELECT COUNT(*) FROM job_orders WHERE status='In Progress'").fetchone()[0],
        'completed_today': conn.execute("SELECT COUNT(*) FROM job_orders WHERE status='Completed' AND completed_date=?",
            (datetime.now().strftime('%Y-%m-%d'),)).fetchone()[0],
        'soft_reservations': conn.execute("SELECT COUNT(*) FROM soft_reservations WHERE status='Active'").fetchone()[0],
    }
    conn.close(); return stats


# ── MRP Last Run ───────────────────────────────────────────────────────────────

def save_mrp_result(so_numbers: list, result_dict: dict):
    """Save (or replace) the last MRP run result."""
    conn = _connect()
    conn.execute("DELETE FROM mrp_last_run")
    conn.execute(
        "INSERT INTO mrp_last_run(id, run_time, so_numbers, result_json) VALUES(1,?,?,?)",
        (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
         json.dumps(so_numbers),
         json.dumps(result_dict))
    )
    conn.commit(); conn.close()


def get_last_mrp_result() -> dict | None:
    """Returns {so_numbers, run_time, result} or None."""
    conn = _connect()
    row = conn.execute("SELECT * FROM mrp_last_run WHERE id=1").fetchone()
    conn.close()
    if not row:
        return None
    return {
        'run_time': row['run_time'],
        'so_numbers': json.loads(row['so_numbers']),
        'result': json.loads(row['result_json']),
    }


# ── MRP Soft Reservations v2 ───────────────────────────────────────────────────

def soft_reserve_all(material_reservations: list):
    """
    Each dict: {material_code, material_name, unit, so_no, sku, qty}
    INSERT OR REPLACE into mrp_soft_reservations.
    """
    conn = _connect()
    for r in material_reservations:
        conn.execute(
            """INSERT OR REPLACE INTO mrp_soft_reservations
               (material_code, material_name, unit, so_no, sku, qty, status, created_at)
               VALUES(?,?,?,?,?,?,'Active',datetime('now'))""",
            (r['material_code'], r.get('material_name', ''), r.get('unit', 'PCS'),
             r['so_no'], r.get('sku', ''), float(r.get('qty', 0)))
        )
    conn.commit(); conn.close()


def release_so_reservations(so_no: str):
    """Set status='Released' for all active reservations for this SO."""
    conn = _connect()
    conn.execute(
        "UPDATE mrp_soft_reservations SET status='Released' WHERE so_no=? AND status='Active'",
        (so_no,)
    )
    conn.commit(); conn.close()


def list_soft_reservations_v2() -> list:
    """Returns all active mrp_soft_reservations."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM mrp_soft_reservations WHERE status='Active' ORDER BY material_code, so_no"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_soft_reserved_by_material(material_code: str) -> float:
    """Sum of active qty for a given material code in mrp_soft_reservations."""
    conn = _connect()
    row = conn.execute(
        "SELECT COALESCE(SUM(qty),0) FROM mrp_soft_reservations WHERE material_code=? AND status='Active'",
        (material_code,)
    ).fetchone()
    conn.close()
    return float(row[0])
