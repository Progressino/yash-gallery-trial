"""Production Module DB — Dynamic Routing, Multi-line JO, Stage Stock"""
import sqlite3, os, json
from datetime import datetime
from typing import Optional

_DB = os.environ.get("PRODUCTION_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "production.db"))
_ITEM_DB = os.environ.get("ITEM_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "items_dev.db"))

def _connect():
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def _item_connect():
    path = _ITEM_DB
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "..", "items_dev.db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # Clean up WAL files if they exist
    for ext in ['-wal', '-shm']:
        wal_path = _DB + ext
        if os.path.exists(wal_path):
            try:
                os.remove(wal_path)
            except:
                pass
    conn = _connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS job_orders (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        jo_number           TEXT UNIQUE NOT NULL,
        jo_date             TEXT NOT NULL,
        so_number           TEXT DEFAULT '',
        sku                 TEXT DEFAULT '',
        sku_name            TEXT DEFAULT '',
        process             TEXT DEFAULT 'Cutting',
        stage               TEXT DEFAULT 'Cutting',
        exec_type           TEXT DEFAULT 'Inhouse',
        vendor_name         TEXT DEFAULT '',
        vendor_rate         REAL DEFAULT 0,
        so_qty              INTEGER DEFAULT 0,
        planned_qty         INTEGER DEFAULT 0,
        issued_qty          INTEGER DEFAULT 0,
        received_qty        INTEGER DEFAULT 0,
        rejected_qty        INTEGER DEFAULT 0,
        output_qty          INTEGER DEFAULT 0,
        balance_qty         INTEGER DEFAULT 0,
        status              TEXT DEFAULT 'Created',
        expected_completion TEXT DEFAULT '',
        completed_date      TEXT DEFAULT '',
        issued_to           TEXT DEFAULT '',
        remarks             TEXT DEFAULT '',
        fabric_code         TEXT DEFAULT '',
        fabric_qty          REAL DEFAULT 0,
        fabric_unit         TEXT DEFAULT 'MTR',
        fabric_issued_qty   REAL DEFAULT 0,
        fabric_received_qty REAL DEFAULT 0,
        fabric_consumption  REAL DEFAULT 0,
        process_cost        REAL DEFAULT 0,
        total_cost          REAL DEFAULT 0,
        parent_jo_id        INTEGER REFERENCES job_orders(id),
        next_stage_jo_id    INTEGER REFERENCES job_orders(id),
        created_at          TEXT DEFAULT (datetime('now')),
        updated_at          TEXT DEFAULT (datetime('now'))
    );

    -- Multi-line JO: each SO line (sku+style) is a separate line in JO
    CREATE TABLE IF NOT EXISTS jo_lines (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        jo_id           INTEGER NOT NULL REFERENCES job_orders(id) ON DELETE CASCADE,
        so_number       TEXT DEFAULT '',
        sku             TEXT DEFAULT '',
        sku_name        TEXT DEFAULT '',
        style           TEXT DEFAULT '',
        planned_qty     INTEGER DEFAULT 0,
        issued_qty      INTEGER DEFAULT 0,
        received_qty    INTEGER DEFAULT 0,
        rejected_qty    INTEGER DEFAULT 0,
        balance_qty     INTEGER DEFAULT 0,
        vendor_rate     REAL DEFAULT 0,
        process_cost    REAL DEFAULT 0,
        remarks         TEXT DEFAULT ''
    );

    -- Fabric issue per JO (Cutting only)
    CREATE TABLE IF NOT EXISTS jo_fabric_issues (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        jo_id           INTEGER NOT NULL REFERENCES job_orders(id),
        jo_line_id      INTEGER REFERENCES jo_lines(id),
        issue_date      TEXT DEFAULT (date('now')),
        fabric_code     TEXT NOT NULL,
        fabric_name     TEXT DEFAULT '',
        issued_qty      REAL DEFAULT 0,
        unit            TEXT DEFAULT 'MTR',
        issued_by       TEXT DEFAULT '',
        remarks         TEXT DEFAULT '',
        created_at      TEXT DEFAULT (datetime('now'))
    );

    -- Fabric return per JO
    CREATE TABLE IF NOT EXISTS jo_fabric_returns (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        jo_id           INTEGER NOT NULL REFERENCES job_orders(id),
        return_date     TEXT DEFAULT (date('now')),
        fabric_code     TEXT NOT NULL,
        returned_qty    REAL DEFAULT 0,
        unit            TEXT DEFAULT 'MTR',
        returned_by     TEXT DEFAULT '',
        remarks         TEXT DEFAULT '',
        created_at      TEXT DEFAULT (datetime('now'))
    );

    -- Issue pieces from one process to next (per line)
    CREATE TABLE IF NOT EXISTS jo_piece_issues (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        jo_id           INTEGER NOT NULL REFERENCES job_orders(id),
        jo_line_id      INTEGER REFERENCES jo_lines(id),
        from_process    TEXT NOT NULL,
        to_process      TEXT NOT NULL,
        so_number       TEXT DEFAULT '',
        sku             TEXT DEFAULT '',
        issue_date      TEXT DEFAULT (date('now')),
        issued_qty      INTEGER DEFAULT 0,
        issued_by       TEXT DEFAULT '',
        remarks         TEXT DEFAULT '',
        created_at      TEXT DEFAULT (datetime('now'))
    );

    -- Receive pieces at a process (per line)
    CREATE TABLE IF NOT EXISTS jo_piece_receipts (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        jo_id           INTEGER NOT NULL REFERENCES job_orders(id),
        jo_line_id      INTEGER REFERENCES jo_lines(id),
        process         TEXT NOT NULL,
        so_number       TEXT DEFAULT '',
        sku             TEXT DEFAULT '',
        receipt_date    TEXT DEFAULT (date('now')),
        received_qty    INTEGER DEFAULT 0,
        rejected_qty    INTEGER DEFAULT 0,
        received_by     TEXT DEFAULT '',
        remarks         TEXT DEFAULT '',
        created_at      TEXT DEFAULT (datetime('now'))
    );

    -- Cost entries per JO per process
    CREATE TABLE IF NOT EXISTS jo_cost_entries (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        jo_id           INTEGER NOT NULL REFERENCES job_orders(id),
        cost_date       TEXT DEFAULT (date('now')),
        process         TEXT NOT NULL,
        cost_type       TEXT DEFAULT 'Labour',
        amount          REAL DEFAULT 0,
        description     TEXT DEFAULT '',
        created_at      TEXT DEFAULT (datetime('now'))
    );

    -- Stage/process stock per SO+SKU+process
    CREATE TABLE IF NOT EXISTS process_stock (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        so_number       TEXT NOT NULL,
        sku             TEXT NOT NULL,
        process         TEXT NOT NULL,
        available_qty   INTEGER DEFAULT 0,
        total_in        INTEGER DEFAULT 0,
        total_out       INTEGER DEFAULT 0,
        updated_at      TEXT DEFAULT (datetime('now')),
        UNIQUE(so_number, sku, process)
    );

    CREATE TABLE IF NOT EXISTS soft_reservations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        material_code   TEXT NOT NULL,
        material_name   TEXT DEFAULT '',
        reserved_qty    REAL DEFAULT 0,
        unit            TEXT DEFAULT 'PCS',
        against_so      TEXT DEFAULT '',
        reservation_date TEXT DEFAULT (datetime('now')),
        status          TEXT DEFAULT 'Active',
        remarks         TEXT DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS mrp_soft_reservations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        material_code   TEXT NOT NULL,
        material_name   TEXT DEFAULT '',
        unit            TEXT DEFAULT 'PCS',
        so_no           TEXT NOT NULL,
        sku             TEXT DEFAULT '',
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

    # Migrations for existing DB
    migrations = [
        ("job_orders", "vendor_rate", "REAL DEFAULT 0"),
        ("job_orders", "issued_qty", "INTEGER DEFAULT 0"),
        ("job_orders", "rejected_qty", "INTEGER DEFAULT 0"),
        ("job_orders", "balance_qty", "INTEGER DEFAULT 0"),
        ("job_orders", "process_cost", "REAL DEFAULT 0"),
        ("job_orders", "fabric_code", "TEXT DEFAULT ''"),
        ("job_orders", "fabric_qty", "REAL DEFAULT 0"),
        ("job_orders", "fabric_unit", "TEXT DEFAULT 'MTR'"),
        ("job_orders", "fabric_issued_qty", "REAL DEFAULT 0"),
        ("job_orders", "fabric_received_qty", "REAL DEFAULT 0"),
        ("job_orders", "fabric_consumption", "REAL DEFAULT 0"),
        ("job_orders", "stage", "TEXT DEFAULT 'Cutting'"),
        ("job_orders", "received_qty", "INTEGER DEFAULT 0"),
        ("job_orders", "total_cost", "REAL DEFAULT 0"),
        ("job_orders", "parent_jo_id", "INTEGER"),
        ("job_orders", "next_stage_jo_id", "INTEGER"),
        ("job_orders", "updated_at", "TEXT DEFAULT (datetime('now'))"),
        ("jo_lines", "so_number", "TEXT DEFAULT ''"),
        ("jo_lines", "style", "TEXT DEFAULT ''"),
        ("jo_lines", "issued_qty", "INTEGER DEFAULT 0"),
        ("jo_lines", "rejected_qty", "INTEGER DEFAULT 0"),
        ("jo_lines", "balance_qty", "INTEGER DEFAULT 0"),
        ("jo_lines", "vendor_rate", "REAL DEFAULT 0"),
        ("jo_lines", "process_cost", "REAL DEFAULT 0"),
    ]
    for table, col, decl in migrations:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        except:
            pass
    conn.commit()
    conn.close()


def _next_jo(conn):
    row = conn.execute("SELECT jo_number FROM job_orders ORDER BY id DESC LIMIT 1").fetchone()
    n = 1
    if row:
        try:
            n = int(row[0].split('-')[-1]) + 1
        except:
            pass
    return f"PJO-{n:04d}"


# ── Item Routing from items_dev.db ────────────────────────────────────────────

def get_item_routing(sku: str) -> list:
    """Get ordered process list for an item from item_routing + routing_steps."""
    try:
        conn = _item_connect()
        item = conn.execute("SELECT id FROM items WHERE item_code=?", (sku,)).fetchone()
        if not item:
            conn.close()
            return ['Cutting', 'Stitching', 'Finishing']  # default
        rows = conn.execute("""
            SELECT rs.name, rs.id as step_id, ir.sort_order
            FROM item_routing ir
            JOIN routing_steps rs ON rs.id = ir.step_id
            WHERE ir.item_id = ?
            ORDER BY ir.sort_order ASC
        """, (item['id'],)).fetchall()
        conn.close()
        if rows:
            return [r['name'] for r in rows]
        return ['Cutting', 'Stitching', 'Finishing']
    except Exception:
        return ['Cutting', 'Stitching', 'Finishing']


def get_next_process(sku: str, current_process: str) -> Optional[str]:
    """Get next process in routing for an item."""
    routing = get_item_routing(sku)
    try:
        idx = routing.index(current_process)
        if idx + 1 < len(routing):
            return routing[idx + 1]
    except ValueError:
        pass
    return None


def get_all_routing_steps() -> list:
    """Get all available routing steps."""
    try:
        conn = _item_connect()
        rows = conn.execute("SELECT name FROM routing_steps ORDER BY sort_order").fetchall()
        conn.close()
        return [r['name'] for r in rows]
    except:
        return ['Cutting', 'Printing', 'Embroidery', 'Stitching', 'Finishing', 'Packing']


# ── Process Stock ──────────────────────────────────────────────────────────────

def get_process_stock(so_number: str, sku: str, process: str) -> int:
    conn = _connect()
    row = conn.execute(
        "SELECT available_qty FROM process_stock WHERE so_number=? AND sku=? AND process=?",
        (so_number, sku, process)
    ).fetchone()
    conn.close()
    return int(row['available_qty']) if row else 0


def get_all_process_stocks(so_number: str, sku: str) -> dict:
    conn = _connect()
    rows = conn.execute(
        "SELECT process, available_qty, total_in, total_out FROM process_stock WHERE so_number=? AND sku=?",
        (so_number, sku)
    ).fetchall()
    conn.close()
    return {r['process']: {'available': int(r['available_qty']), 'in': int(r['total_in']), 'out': int(r['total_out'])} for r in rows}


def _update_process_stock(conn, so_number: str, sku: str, process: str, qty_in: int = 0, qty_out: int = 0):
    conn.execute("""
        INSERT INTO process_stock(so_number, sku, process, available_qty, total_in, total_out, updated_at)
        VALUES(?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(so_number, sku, process) DO UPDATE SET
            available_qty = MAX(0, available_qty + ? - ?),
            total_in = total_in + ?,
            total_out = total_out + ?,
            updated_at = datetime('now')
    """, (so_number, sku, process, qty_in - qty_out, qty_in, qty_out,
          qty_in - qty_out, qty_in, qty_out))


# ── Ready to Process Lists ─────────────────────────────────────────────────────

def get_ready_to_process(process: str) -> list:
    """
    Get lines ready for a process:
    - Cutting: from printed_fabric_reservations in grey.db
    - Other: from process_stock of previous process
    """
    if process == 'Cutting':
        return _get_ready_to_cut()
    else:
        return _get_ready_for_process(process)


def _get_ready_to_cut() -> list:
    """Get printed fabric reservations ready for cutting — deduct already planned JO qty."""
    grey_db_path = os.environ.get("GREY_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "grey.db"))
    try:
        gconn = sqlite3.connect(grey_db_path)
        gconn.row_factory = sqlite3.Row
        rows = gconn.execute("""
            SELECT r.so_number, r.sku, r.fabric_code, r.fabric_name,
                   r.qty as reserved_qty, s.available_qty as fabric_available,
                   r.status
            FROM printed_fabric_reservations r
            LEFT JOIN printed_fabric_checked_stock s ON s.fabric_code = r.fabric_code
            WHERE r.status = 'Active'
            ORDER BY r.so_number, r.sku
        """).fetchall()
        # Convert to plain dicts BEFORE closing grey connection
        raw = [dict(r) for r in rows]
        gconn.close()

        # Now open production.db separately
        conn = _connect()
        result = []
        try:
            for d in raw:
                existing = conn.execute("""
                    SELECT COALESCE(SUM(planned_qty), 0) as already_planned
                    FROM job_orders
                    WHERE so_number=? AND sku=? AND process='Cutting'
                    AND status NOT IN ('Cancelled')
                """, (d['so_number'], d.get('sku',''))).fetchone()
                already_planned = int(existing[0]) if existing else 0
                reserved = float(d.get('reserved_qty') or 0)
                remaining = max(0, reserved - already_planned)
                if remaining > 0:
                    d['already_planned'] = already_planned
                    d['available_qty'] = remaining
                    d['routing'] = get_item_routing(d.get('sku', ''))
                    result.append(d)
        finally:
            conn.close()
        return result
    except Exception as e:
        return []


def _get_ready_for_process(process: str) -> list:
    """Get SO+SKU lines with available pieces at previous process."""
    conn = _connect()
    rows = conn.execute("""
        SELECT so_number, sku, process, available_qty, total_in, total_out
        FROM process_stock
        WHERE process=? AND available_qty > 0
        ORDER BY so_number, sku
    """, (process,)).fetchall()
    conn.close()

    # Find previous process for each sku
    result = []
    for r in rows:
        d = dict(r)
        routing = get_item_routing(d['sku'])
        # Check if this process feeds into the requested process
        try:
            idx = routing.index(process)
            if idx > 0:
                prev_process = routing[idx - 1]
                if d['process'] == prev_process:
                    d['routing'] = routing
                    d['next_process'] = get_next_process(d['sku'], process)
                    result.append(d)
        except ValueError:
            pass

    # Actually simpler — just get process_stock for the process BEFORE target
    # Re-query correctly
    conn2 = _connect()
    all_stocks = conn2.execute("""
        SELECT so_number, sku, process, available_qty
        FROM process_stock WHERE available_qty > 0
        ORDER BY so_number, sku
    """).fetchall()
    conn2.close()

    result = []
    seen = set()
    for r in all_stocks:
        d = dict(r)
        routing = get_item_routing(d['sku'])
        next_p = get_next_process(d['sku'], d['process'])
        if next_p == process and d['available_qty'] > 0:
            key = (d['so_number'], d['sku'])
            if key not in seen:
                seen.add(key)
                result.append({
                    'so_number': d['so_number'],
                    'sku': d['sku'],
                    'available_qty': d['available_qty'],
                    'from_process': d['process'],
                    'to_process': process,
                    'routing': routing,
                })
    return result


# ── Job Order CRUD ─────────────────────────────────────────────────────────────

def list_jos(status=None, so_number=None, process=None):
    conn = _connect()
    conditions, params = [], []
    if status: conditions.append("status=?"); params.append(status)
    if so_number: conditions.append("so_number=?"); params.append(so_number)
    if process: conditions.append("process=?"); params.append(process)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    rows = conn.execute(f"SELECT * FROM job_orders {where} ORDER BY id DESC", params).fetchall()
    result = []
    for r in rows:
        jo = dict(r)
        jo['lines'] = _get_jo_lines_with_stats(conn, jo['id'])
        jo['fabric_issues'] = [dict(l) for l in conn.execute("SELECT * FROM jo_fabric_issues WHERE jo_id=?", (jo['id'],)).fetchall()]
        jo['fabric_returns'] = [dict(l) for l in conn.execute("SELECT * FROM jo_fabric_returns WHERE jo_id=?", (jo['id'],)).fetchall()]
        jo['cost_entries'] = [dict(l) for l in conn.execute("SELECT * FROM jo_cost_entries WHERE jo_id=?", (jo['id'],)).fetchall()]
        stocks_rows = conn.execute(
            "SELECT process, available_qty, total_in, total_out FROM process_stock WHERE so_number=? AND sku=?",
            (jo.get('so_number',''), jo.get('sku',''))
        ).fetchall()
        jo['process_stocks'] = {r['process']: {'available': int(r['available_qty']), 'in': int(r['total_in']), 'out': int(r['total_out'])} for r in stocks_rows}
        result.append(jo)
    conn.close()
    # Add routing info after connection is closed
    for jo in result:
        jo['routing'] = get_item_routing(jo.get('sku', ''))
        jo['next_process'] = get_next_process(jo.get('sku', ''), jo.get('process', ''))
    return result


def _get_jo_lines_with_stats(conn, jo_id: int) -> list:
    lines = [dict(l) for l in conn.execute("SELECT * FROM jo_lines WHERE jo_id=?", (jo_id,)).fetchall()]
    for ln in lines:
        # Get issue/receipt totals per line
        issued = conn.execute(
            "SELECT COALESCE(SUM(issued_qty),0) FROM jo_piece_issues WHERE jo_line_id=?", (ln['id'],)
        ).fetchone()[0]
        received = conn.execute(
            "SELECT COALESCE(SUM(received_qty),0) FROM jo_piece_receipts WHERE jo_line_id=?", (ln['id'],)
        ).fetchone()[0]
        rejected = conn.execute(
            "SELECT COALESCE(SUM(rejected_qty),0) FROM jo_piece_receipts WHERE jo_line_id=?", (ln['id'],)
        ).fetchone()[0]
        ln['issued_qty'] = int(issued)
        ln['received_qty'] = int(received)
        ln['rejected_qty'] = int(rejected)
        ln['balance_qty'] = int(ln.get('planned_qty', 0)) - int(received)
    return lines


def get_jo(joid: int):
    conn = _connect()
    row = conn.execute("SELECT * FROM job_orders WHERE id=?", (joid,)).fetchone()
    if not row:
        conn.close()
        return None
    jo = dict(row)
    jo['lines'] = _get_jo_lines_with_stats(conn, jo['id'])
    jo['fabric_issues'] = [dict(l) for l in conn.execute("SELECT * FROM jo_fabric_issues WHERE jo_id=?", (jo['id'],)).fetchall()]
    jo['fabric_returns'] = [dict(l) for l in conn.execute("SELECT * FROM jo_fabric_returns WHERE jo_id=?", (jo['id'],)).fetchall()]
    jo['piece_issues'] = [dict(l) for l in conn.execute("SELECT * FROM jo_piece_issues WHERE jo_id=?", (jo['id'],)).fetchall()]
    jo['piece_receipts'] = [dict(l) for l in conn.execute("SELECT * FROM jo_piece_receipts WHERE jo_id=?", (jo['id'],)).fetchall()]
    jo['cost_entries'] = [dict(l) for l in conn.execute("SELECT * FROM jo_cost_entries WHERE jo_id=?", (jo['id'],)).fetchall()]
    # Get process stocks inline without new connection
    stocks_rows = conn.execute(
        "SELECT process, available_qty, total_in, total_out FROM process_stock WHERE so_number=? AND sku=?",
        (jo.get('so_number',''), jo.get('sku',''))
    ).fetchall()
    jo['process_stocks'] = {r['process']: {'available': int(r['available_qty']), 'in': int(r['total_in']), 'out': int(r['total_out'])} for r in stocks_rows}
    conn.close()
    # These open their own connections separately - safe after main is closed
    jo['routing'] = get_item_routing(jo.get('sku', ''))
    jo['next_process'] = get_next_process(jo.get('sku', ''), jo.get('process', ''))
    return jo


def validate_jo_creation(process: str, so_number: str, sku: str, planned_qty: int) -> dict:
    if process == 'Cutting':
        return {'ok': True, 'available': 99999, 'message': ''}
    routing = get_item_routing(sku)
    try:
        idx = routing.index(process)
        if idx == 0:
            return {'ok': True, 'available': 99999, 'message': ''}
        prev_process = routing[idx - 1]
    except ValueError:
        return {'ok': True, 'available': 99999, 'message': ''}
    available = get_process_stock(so_number, sku, prev_process)
    if available <= 0:
        return {'ok': False, 'available': 0,
                'message': f'No pieces available at {prev_process} for {sku}. Complete {prev_process} first.'}
    if planned_qty > available:
        return {'ok': False, 'available': available,
                'message': f'Only {available} pieces available at {prev_process}. Cannot plan {planned_qty}.'}
    return {'ok': True, 'available': available, 'message': ''}


def create_jo(data: dict) -> str:
    conn = _connect()
    num = _next_jo(conn)
    process = data.get('process') or data.get('stage') or 'Cutting'
    planned = int(data.get('planned_qty') or 0)
    conn.execute("""INSERT INTO job_orders(
        jo_number, jo_date, so_number, sku, sku_name, process, stage,
        exec_type, vendor_name, vendor_rate, so_qty, planned_qty, balance_qty,
        status, expected_completion, issued_to, remarks,
        fabric_code, fabric_qty, fabric_unit, updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        (num, data.get('jo_date') or datetime.now().strftime('%Y-%m-%d'),
         data.get('so_number',''), data.get('sku',''), data.get('sku_name',''),
         process, process,
         data.get('exec_type','Inhouse'), data.get('vendor_name',''),
         float(data.get('vendor_rate') or 0),
         int(data.get('so_qty') or 0), planned, planned,
         'Created', data.get('expected_completion',''),
         data.get('issued_to',''), data.get('remarks',''),
         data.get('fabric_code',''), float(data.get('fabric_qty') or 0),
         data.get('fabric_unit','MTR')))
    joid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for ln in data.get('lines', []):
        pq = int(ln.get('planned_qty', 0))
        conn.execute("""INSERT INTO jo_lines(jo_id,so_number,sku,sku_name,style,planned_qty,balance_qty,vendor_rate,remarks)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (joid, ln.get('so_number', data.get('so_number','')),
             ln.get('sku', data.get('sku','')),
             ln.get('sku_name', data.get('sku_name','')),
             ln.get('style',''), pq, pq,
             float(ln.get('vendor_rate') or 0),
             ln.get('remarks','')))
    conn.commit()
    conn.close()
    return num


def update_jo(joid: int, data: dict):
    allowed = ['status','output_qty','received_qty','rejected_qty','balance_qty',
               'completed_date','remarks','issued_to','exec_type','vendor_name',
               'vendor_rate','fabric_issued_qty','fabric_received_qty',
               'fabric_consumption','process_cost','total_cost','next_stage_jo_id']
    sets = ', '.join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed]
    if not sets: return
    vals += [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), joid]
    conn = _connect()
    conn.execute(f"UPDATE job_orders SET {sets}, updated_at=? WHERE id=?", vals)
    conn.commit()
    conn.close()


# ── Fabric Issue ───────────────────────────────────────────────────────────────

def issue_fabric(joid: int, data: dict):
    conn = _connect()
    jo = dict(conn.execute("SELECT * FROM job_orders WHERE id=?", (joid,)).fetchone() or {})
    if not jo:
        conn.close()
        raise ValueError("JO not found")
    issued = float(data.get('issued_qty', 0))
    conn.execute("""INSERT INTO jo_fabric_issues(jo_id,jo_line_id,issue_date,fabric_code,fabric_name,issued_qty,unit,issued_by,remarks)
        VALUES(?,?,?,?,?,?,?,?,?)""",
        (joid, data.get('jo_line_id'),
         data.get('issue_date') or datetime.now().strftime('%Y-%m-%d'),
         data.get('fabric_code',''), data.get('fabric_name',''),
         issued, data.get('unit','MTR'), data.get('issued_by',''), data.get('remarks','')))
    conn.execute("""UPDATE job_orders SET
        fabric_issued_qty = COALESCE(fabric_issued_qty,0) + ?,
        status = CASE WHEN status='Created' THEN 'In Progress' ELSE status END,
        updated_at = datetime('now') WHERE id=?""", (issued, joid))
    conn.commit()
    conn.close()
    # Deduct from grey.db AFTER closing production.db
    grey_db = os.environ.get("GREY_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "grey.db"))
    try:
        gc = sqlite3.connect(grey_db)
        gc.execute("UPDATE printed_fabric_checked_stock SET available_qty = MAX(0, available_qty - ?) WHERE fabric_code=?",
                   (issued, data.get('fabric_code','')))
        gc.commit()
        gc.close()
    except: pass


def return_fabric(joid: int, data: dict):
    conn = _connect()
    returned = float(data.get('returned_qty', 0))
    conn.execute("""INSERT INTO jo_fabric_returns(jo_id,return_date,fabric_code,returned_qty,unit,returned_by,remarks)
        VALUES(?,?,?,?,?,?,?)""",
        (joid, data.get('return_date') or datetime.now().strftime('%Y-%m-%d'),
         data.get('fabric_code',''), returned, data.get('unit','MTR'),
         data.get('returned_by',''), data.get('remarks','')))
    conn.execute("""UPDATE job_orders SET
        fabric_received_qty = COALESCE(fabric_received_qty,0) + ?,
        fabric_consumption = COALESCE(fabric_issued_qty,0) - (COALESCE(fabric_received_qty,0) + ?),
        updated_at = datetime('now') WHERE id=?""", (returned, returned, joid))
    conn.commit()
    conn.close()
    # Return to grey.db AFTER closing production.db
    grey_db = os.environ.get("GREY_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "grey.db"))
    try:
        gc = sqlite3.connect(grey_db)
        gc.execute("UPDATE printed_fabric_checked_stock SET available_qty = available_qty + ? WHERE fabric_code=?",
                   (returned, data.get('fabric_code','')))
        gc.commit()
        gc.close()
    except: pass


# ── Issue Pieces (process → next process) ─────────────────────────────────────

def issue_pieces(joid: int, data: dict):
    conn = _connect()
    jo = dict(conn.execute("SELECT * FROM job_orders WHERE id=?", (joid,)).fetchone() or {})
    if not jo:
        conn.close()
        raise ValueError("JO not found")
    issued = int(data.get('issued_qty', 0))
    jo_line_id = data.get('jo_line_id')
    from_process = data.get('from_process') or jo.get('process','Cutting')
    to_process = data.get('to_process') or get_next_process(jo.get('sku',''), from_process)
    so_number = jo.get('so_number','')
    sku = data.get('sku') or jo.get('sku','')

    # Validate stock inline (no separate connection)
    stock_row = conn.execute(
        "SELECT COALESCE(available_qty,0) FROM process_stock WHERE so_number=? AND sku=? AND process=?",
        (so_number, sku, from_process)
    ).fetchone()
    available = int(stock_row[0]) if stock_row else 0
    if issued > available:
        conn.close()
        raise ValueError(f"Only {available} pieces available at {from_process}. Cannot issue {issued}.")

    conn.execute("""INSERT INTO jo_piece_issues(jo_id,jo_line_id,from_process,to_process,so_number,sku,issue_date,issued_qty,issued_by,remarks)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (joid, jo_line_id, from_process, to_process, so_number, sku,
         data.get('issue_date') or datetime.now().strftime('%Y-%m-%d'),
         issued, data.get('issued_by',''), data.get('remarks','')))

    # Update line issued_qty
    if jo_line_id:
        conn.execute("UPDATE jo_lines SET issued_qty = COALESCE(issued_qty,0) + ?, balance_qty = planned_qty - received_qty WHERE id=?",
                     (issued, jo_line_id))

    # Move stock: deduct from current, add to next process
    _update_process_stock(conn, so_number, sku, from_process, qty_out=issued)
    _update_process_stock(conn, so_number, sku, to_process, qty_in=issued)

    conn.commit()
    conn.close()


# ── Receive Pieces ─────────────────────────────────────────────────────────────

def receive_pieces(joid: int, data: dict):
    conn = _connect()
    jo = dict(conn.execute("SELECT * FROM job_orders WHERE id=?", (joid,)).fetchone() or {})
    if not jo:
        conn.close()
        raise ValueError("JO not found")
    received = int(data.get('received_qty', 0))
    rejected = int(data.get('rejected_qty', 0))
    jo_line_id = data.get('jo_line_id')
    process = data.get('process') or jo.get('process','Cutting')
    so_number = jo.get('so_number','')
    sku = data.get('sku') or jo.get('sku','')
    conn.execute("""INSERT INTO jo_piece_receipts(jo_id,jo_line_id,process,so_number,sku,receipt_date,received_qty,rejected_qty,received_by,remarks)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (joid, jo_line_id, process, so_number, sku,
         data.get('receipt_date') or datetime.now().strftime('%Y-%m-%d'),
         received, rejected, data.get('received_by',''), data.get('remarks','')))
    conn.execute("""UPDATE job_orders SET
        received_qty = COALESCE(received_qty,0) + ?,
        output_qty = COALESCE(output_qty,0) + ?,
        updated_at = datetime('now') WHERE id=?""", (received, received, joid))
    if jo_line_id:
        conn.execute("""UPDATE jo_lines SET
            received_qty = COALESCE(received_qty,0) + ?,
            rejected_qty = COALESCE(rejected_qty,0) + ?,
            balance_qty = planned_qty - (COALESCE(received_qty,0) + ?)
            WHERE id=?""", (received, rejected, received, jo_line_id))
    _update_process_stock(conn, so_number, sku, process, qty_in=received)
    conn.commit()
    conn.close()


# ── Cost Entry ─────────────────────────────────────────────────────────────────

def add_cost(joid: int, data: dict):
    conn = _connect()
    amount = float(data.get('amount', 0))
    process = data.get('process', 'Cutting')
    conn.execute("""INSERT INTO jo_cost_entries(jo_id,cost_date,process,cost_type,amount,description)
        VALUES(?,?,?,?,?,?)""",
        (joid, data.get('cost_date') or datetime.now().strftime('%Y-%m-%d'),
         process, data.get('cost_type','Labour'), amount, data.get('description','')))
    conn.execute("""UPDATE job_orders SET
        process_cost = COALESCE(process_cost,0) + ?,
        total_cost = COALESCE(total_cost,0) + ?,
        updated_at = datetime('now') WHERE id=?""", (amount, amount, joid))
    conn.commit()
    conn.close()


# ── Next Process JO ────────────────────────────────────────────────────────────

def create_next_process_jo(parent_joid: int) -> dict:
    conn = _connect()
    parent = conn.execute("SELECT * FROM job_orders WHERE id=?", (parent_joid,)).fetchone()
    if not parent:
        conn.close()
        return {'ok': False, 'message': 'JO not found'}
    parent = dict(parent)
    sku = parent.get('sku','')
    so_number = parent.get('so_number','')
    current_process = parent.get('process','Cutting')
    next_process = get_next_process(sku, current_process)
    if not next_process:
        conn.close()
        return {'ok': False, 'message': f'{current_process} is the last process for this item'}
    available = get_process_stock(so_number, sku, current_process)
    if available <= 0:
        conn.close()
        return {'ok': False, 'available': 0,
                'message': f'No pieces at {current_process}. Receive pieces first.'}
    num = _next_jo(conn)
    conn.execute("""INSERT INTO job_orders(
        jo_number, jo_date, so_number, sku, sku_name, process, stage,
        exec_type, so_qty, planned_qty, balance_qty, status,
        expected_completion, fabric_code, parent_jo_id, updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        (num, datetime.now().strftime('%Y-%m-%d'),
         so_number, sku, parent.get('sku_name',''),
         next_process, next_process,
         parent.get('exec_type','Inhouse'),
         parent.get('so_qty',0), available, available,
         'Created', parent.get('expected_completion',''),
         parent.get('fabric_code',''), parent_joid))
    new_joid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Copy lines from parent with available qty
    parent_lines = conn.execute("SELECT * FROM jo_lines WHERE jo_id=?", (parent_joid,)).fetchall()
    for pl in parent_lines:
        pl = dict(pl)
        line_avail = get_process_stock(so_number, pl.get('sku', sku), current_process)
        if line_avail > 0:
            conn.execute("""INSERT INTO jo_lines(jo_id,so_number,sku,sku_name,style,planned_qty,balance_qty,vendor_rate,remarks)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (new_joid, pl.get('so_number', so_number),
                 pl.get('sku', sku), pl.get('sku_name',''),
                 pl.get('style',''), line_avail, line_avail,
                 pl.get('vendor_rate',0), ''))

    conn.execute("UPDATE job_orders SET next_stage_jo_id=?, updated_at=datetime('now') WHERE id=?",
                 (new_joid, parent_joid))
    conn.commit()
    conn.close()
    return {'ok': True, 'jo_number': num, 'process': next_process, 'planned_qty': available}


# ── Reports ────────────────────────────────────────────────────────────────────

def get_process_report() -> list:
    """Issue/Receive/Balance report per process per SO+SKU."""
    conn = _connect()
    rows = conn.execute("""
        SELECT j.process, j.so_number, j.sku, j.sku_name,
               SUM(j.planned_qty) as planned,
               SUM(j.issued_qty) as issued,
               SUM(j.received_qty) as received,
               SUM(j.rejected_qty) as rejected,
               SUM(j.balance_qty) as balance
        FROM job_orders j
        WHERE j.status NOT IN ('Cancelled')
        GROUP BY j.process, j.so_number, j.sku
        ORDER BY j.process, j.so_number, j.sku
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Production Stats ───────────────────────────────────────────────────────────

def get_production_stats():
    conn = _connect()
    # Dynamic process counts
    process_counts = {}
    rows = conn.execute("""
        SELECT process, COUNT(*) as cnt FROM job_orders
        WHERE status NOT IN ('Completed','Closed','Cancelled')
        GROUP BY process
    """).fetchall()
    for r in rows:
        process_counts[r['process']] = r['cnt']
    stats = {
        'total_jos': conn.execute("SELECT COUNT(*) FROM job_orders").fetchone()[0],
        'open_jos': conn.execute("SELECT COUNT(*) FROM job_orders WHERE status NOT IN ('Completed','Closed','Cancelled')").fetchone()[0],
        'in_progress': conn.execute("SELECT COUNT(*) FROM job_orders WHERE status='In Progress'").fetchone()[0],
        'completed_today': conn.execute("SELECT COUNT(*) FROM job_orders WHERE status='Completed' AND completed_date=?",
            (datetime.now().strftime('%Y-%m-%d'),)).fetchone()[0],
        'process_counts': process_counts,
        'soft_reservations': conn.execute("SELECT COUNT(*) FROM soft_reservations WHERE status='Active'").fetchone()[0],
    }
    conn.close()
    return stats


# ── MRP functions ──────────────────────────────────────────────────────────────

def save_mrp_result(so_numbers: list, result_dict: dict):
    conn = _connect()
    conn.execute("DELETE FROM mrp_last_run")
    conn.execute("INSERT INTO mrp_last_run(id,run_time,so_numbers,result_json) VALUES(1,?,?,?)",
        (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), json.dumps(so_numbers), json.dumps(result_dict)))
    conn.commit(); conn.close()

def get_last_mrp_result():
    conn = _connect()
    row = conn.execute("SELECT * FROM mrp_last_run WHERE id=1").fetchone()
    conn.close()
    if not row: return None
    return {'run_time': row['run_time'], 'so_numbers': json.loads(row['so_numbers']), 'result': json.loads(row['result_json'])}

def list_reservations(status='Active'):
    conn = _connect()
    rows = conn.execute("SELECT * FROM soft_reservations WHERE status=? ORDER BY id DESC", (status,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def create_reservation(data: dict):
    conn = _connect()
    conn.execute("INSERT INTO soft_reservations(material_code,material_name,reserved_qty,unit,against_so,status,remarks) VALUES(?,?,?,?,?,?,?)",
        (data['material_code'], data.get('material_name',''), data.get('reserved_qty',0),
         data.get('unit','PCS'), data.get('against_so',''), 'Active', data.get('remarks','')))
    conn.commit(); conn.close()

def release_reservation(rid: int):
    conn = _connect()
    conn.execute("UPDATE soft_reservations SET status='Released' WHERE id=?", (rid,))
    conn.commit(); conn.close()

def get_reserved_qty(material_code: str) -> float:
    conn = _connect()
    row = conn.execute("SELECT COALESCE(SUM(reserved_qty),0) FROM soft_reservations WHERE material_code=? AND status='Active'", (material_code,)).fetchone()
    conn.close(); return float(row[0])

def soft_reserve_all(material_reservations: list):
    conn = _connect()
    for r in material_reservations:
        conn.execute("INSERT OR REPLACE INTO mrp_soft_reservations(material_code,material_name,unit,so_no,sku,qty,status,created_at) VALUES(?,?,?,?,?,?,'Active',datetime('now'))",
            (r['material_code'], r.get('material_name',''), r.get('unit','PCS'), r['so_no'], r.get('sku',''), float(r.get('qty',0))))
    conn.commit(); conn.close()

def release_so_reservations(so_no: str):
    conn = _connect()
    conn.execute("UPDATE mrp_soft_reservations SET status='Released' WHERE so_no=? AND status='Active'", (so_no,))
    conn.commit(); conn.close()

def list_soft_reservations_v2() -> list:
    conn = _connect()
    rows = conn.execute("SELECT * FROM mrp_soft_reservations WHERE status='Active' ORDER BY material_code,so_no").fetchall()
    conn.close(); return [dict(r) for r in rows]

def get_soft_reserved_by_material(material_code: str) -> float:
    conn = _connect()
    row = conn.execute("SELECT COALESCE(SUM(qty),0) FROM mrp_soft_reservations WHERE material_code=? AND status='Active'", (material_code,)).fetchone()
    conn.close(); return float(row[0])
