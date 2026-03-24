"""Sales Orders & Demand Management DB"""
import sqlite3, os
from datetime import datetime

_DB = os.environ.get("SALES_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "sales.db"))

def _connect():
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = _connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS demands (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        demand_number TEXT UNIQUE NOT NULL,
        demand_date   TEXT NOT NULL,
        demand_source TEXT DEFAULT 'Sales Team',
        buyer         TEXT,
        priority      TEXT DEFAULT 'Normal',
        status        TEXT DEFAULT 'Draft',
        notes         TEXT,
        created_at    TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS demand_lines (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        demand_id  INTEGER NOT NULL REFERENCES demands(id) ON DELETE CASCADE,
        sku        TEXT NOT NULL,
        sku_name   TEXT,
        demand_qty INTEGER DEFAULT 0,
        delivered_qty INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS sales_orders (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        so_number     TEXT UNIQUE NOT NULL,
        so_date       TEXT NOT NULL,
        buyer         TEXT,
        warehouse     TEXT,
        sales_team    TEXT,
        source_type   TEXT DEFAULT 'Sales Team Demand',
        ref_demand    TEXT,
        delivery_date TEXT,
        payment_terms TEXT,
        status        TEXT DEFAULT 'Draft',
        notes         TEXT,
        created_at    TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS so_lines (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        so_id         INTEGER NOT NULL REFERENCES sales_orders(id) ON DELETE CASCADE,
        sku           TEXT NOT NULL,
        sku_name      TEXT,
        qty           INTEGER DEFAULT 0,
        produced_qty  INTEGER DEFAULT 0,
        dispatch_qty  INTEGER DEFAULT 0,
        received_qty  INTEGER DEFAULT 0,
        unit          TEXT DEFAULT 'PCS',
        rate          REAL DEFAULT 0,
        delivery_date TEXT DEFAULT '',
        remarks       TEXT DEFAULT ''
    );
    """)
    conn.commit()

    # Migrate so_lines — add new columns if missing
    for col_ddl in [
        "ALTER TABLE so_lines ADD COLUMN rate REAL DEFAULT 0",
        "ALTER TABLE so_lines ADD COLUMN delivery_date TEXT DEFAULT ''",
        "ALTER TABLE so_lines ADD COLUMN remarks TEXT DEFAULT ''",
        "ALTER TABLE so_lines ADD COLUMN hsn_code TEXT DEFAULT ''",
        "ALTER TABLE so_lines ADD COLUMN gst_pct REAL DEFAULT 0",
        "ALTER TABLE so_lines ADD COLUMN merchant_code TEXT DEFAULT ''",
        "ALTER TABLE so_lines ADD COLUMN priority TEXT DEFAULT 'Normal'",
        "ALTER TABLE so_lines ADD COLUMN line_delivery_date TEXT DEFAULT ''",
    ]:
        try:
            conn.execute(col_ddl)
        except Exception:
            pass
    conn.commit()
    conn.close()

# ── auto-number helpers ────────────────────────────────────────────────────────
def _next_num(conn, table, col, prefix):
    row = conn.execute(f"SELECT {col} FROM {table} ORDER BY id DESC LIMIT 1").fetchone()
    if row:
        try:
            n = int(row[0].split('-')[-1]) + 1
        except Exception:
            n = 1
    else:
        n = 1
    return f"{prefix}-{n:04d}"

# ── Demands ───────────────────────────────────────────────────────────────────
def list_demands(status: str = None):
    conn = _connect()
    if status:
        rows = conn.execute("SELECT * FROM demands WHERE status=? ORDER BY id DESC", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM demands ORDER BY id DESC").fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d['lines'] = [dict(l) for l in conn.execute("SELECT * FROM demand_lines WHERE demand_id=?", (d['id'],)).fetchall()]
        result.append(d)
    conn.close(); return result

def create_demand(data: dict):
    conn = _connect()
    num = _next_num(conn, 'demands', 'demand_number', 'DEM')
    conn.execute("""INSERT INTO demands(demand_number,demand_date,demand_source,buyer,priority,status,notes)
        VALUES(?,?,?,?,?,?,?)""",
        (num, data.get('demand_date') or datetime.now().strftime('%Y-%m-%d'),
         data.get('demand_source') or 'Sales Team', data.get('buyer') or '',
         data.get('priority') or 'Normal', data.get('status') or 'Draft', data.get('notes') or ''))
    did = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for ln in data.get('lines', []):
        conn.execute("INSERT INTO demand_lines(demand_id,sku,sku_name,demand_qty) VALUES(?,?,?,?)",
            (did, ln['sku'], ln.get('sku_name',''), ln.get('demand_qty',0)))
    conn.commit(); conn.close(); return num

def update_demand_status(did: int, status: str):
    conn = _connect()
    conn.execute("UPDATE demands SET status=? WHERE id=?", (status, did))
    conn.commit(); conn.close()

def get_demand_by_number(demand_number: str):
    conn = _connect()
    row = conn.execute("SELECT * FROM demands WHERE demand_number=?", (demand_number,)).fetchone()
    if not row:
        conn.close(); return None
    d = dict(row)
    d['lines'] = [dict(l) for l in conn.execute("SELECT * FROM demand_lines WHERE demand_id=?", (d['id'],)).fetchall()]
    conn.close(); return d

# ── Sales Orders ──────────────────────────────────────────────────────────────
def list_orders(status: str = None):
    conn = _connect()
    if status:
        rows = conn.execute("SELECT * FROM sales_orders WHERE status=? ORDER BY id DESC", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM sales_orders ORDER BY id DESC").fetchall()
    result = []
    for r in rows:
        so = dict(r)
        so['lines'] = [dict(l) for l in conn.execute("SELECT * FROM so_lines WHERE so_id=?", (so['id'],)).fetchall()]
        result.append(so)
    conn.close(); return result

def create_order(data: dict):
    conn = _connect()
    num = _next_num(conn, 'sales_orders', 'so_number', 'SO')
    conn.execute("""INSERT INTO sales_orders(so_number,so_date,buyer,warehouse,sales_team,source_type,
        ref_demand,delivery_date,payment_terms,status,notes)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (num, data.get('so_date') or datetime.now().strftime('%Y-%m-%d'),
         data.get('buyer') or '', data.get('warehouse') or '', data.get('sales_team') or '',
         data.get('source_type') or 'Sales Team Demand', data.get('ref_demand') or '',
         data.get('delivery_date') or '', data.get('payment_terms') or '',
         data.get('status') or 'Draft', data.get('notes') or ''))
    soid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for ln in data.get('lines', []):
        conn.execute(
            """INSERT INTO so_lines
               (so_id,sku,sku_name,qty,unit,rate,delivery_date,remarks,
                hsn_code,gst_pct,merchant_code,priority,line_delivery_date)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (soid, ln['sku'], ln.get('sku_name',''), ln.get('qty',0), ln.get('unit','PCS'),
             ln.get('rate') or 0, ln.get('delivery_date',''), ln.get('remarks',''),
             ln.get('hsn_code',''), ln.get('gst_pct') or 0,
             ln.get('merchant_code',''), ln.get('priority') or 'Normal',
             ln.get('line_delivery_date','')))
    conn.commit(); conn.close(); return num

def update_order(soid: int, data: dict):
    allowed = ['buyer','warehouse','delivery_date','payment_terms','status','notes','sales_team']
    sets = ', '.join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed] + [soid]
    if not sets: return
    conn = _connect()
    conn.execute(f"UPDATE sales_orders SET {sets} WHERE id=?", vals)
    conn.commit(); conn.close()

def update_so_line(lid: int, data: dict):
    allowed = ['produced_qty','dispatch_qty','received_qty','qty','rate','delivery_date','remarks',
               'hsn_code','gst_pct','merchant_code','priority','line_delivery_date']
    sets = ', '.join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed] + [lid]
    if not sets: return
    conn = _connect()
    conn.execute(f"UPDATE so_lines SET {sets} WHERE id=?", vals)
    conn.commit(); conn.close()

def get_open_orders():
    """Returns open SOs suitable for MRP/Production"""
    conn = _connect()
    rows = conn.execute("""SELECT so.*, sl.sku, sl.sku_name, sl.qty, sl.produced_qty, sl.unit
        FROM sales_orders so JOIN so_lines sl ON sl.so_id=so.id
        WHERE so.status NOT IN ('Closed','Cancelled')
        ORDER BY so.delivery_date, so.id""").fetchall()
    conn.close(); return [dict(r) for r in rows]
