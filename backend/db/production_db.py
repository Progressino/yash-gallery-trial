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

    CREATE TABLE IF NOT EXISTS mrp_material_commitments (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        so_number       TEXT NOT NULL,
        material_code   TEXT NOT NULL,
        material_name   TEXT DEFAULT '',
        unit            TEXT DEFAULT 'PCS',
        mrp_qty         REAL NOT NULL DEFAULT 0,
        po_committed_qty REAL DEFAULT 0,
        jo_committed_qty REAL DEFAULT 0,
        status          TEXT DEFAULT 'Open',
        updated_at      TEXT DEFAULT (datetime('now')),
        UNIQUE(so_number, material_code)
    );

    -- Set BOM: style-level recipe of garment components (Top/Pant/Dupatta…)
    CREATE TABLE IF NOT EXISTS set_bom_headers (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        style_key       TEXT NOT NULL UNIQUE,
        style_name      TEXT DEFAULT '',
        active          INTEGER DEFAULT 1,
        remarks         TEXT DEFAULT '',
        created_at      TEXT DEFAULT (datetime('now')),
        updated_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS set_bom_lines (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        header_id           INTEGER NOT NULL REFERENCES set_bom_headers(id) ON DELETE CASCADE,
        component_code      TEXT NOT NULL,
        component_name      TEXT DEFAULT '',
        qty_per_set         INTEGER NOT NULL DEFAULT 1,
        default_next_process TEXT DEFAULT '',
        sort_order          INTEGER DEFAULT 0,
        UNIQUE(header_id, component_code)
    );

    -- Per-component material consumption (fabric, thread, etc.)
    CREATE TABLE IF NOT EXISTS set_bom_material_lines (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        set_bom_line_id     INTEGER NOT NULL REFERENCES set_bom_lines(id) ON DELETE CASCADE,
        material_code       TEXT NOT NULL,
        material_name       TEXT DEFAULT '',
        quantity            REAL NOT NULL DEFAULT 0,
        unit                TEXT DEFAULT 'MTR',
        sort_order          INTEGER DEFAULT 0
    );

    -- Audit: Cutting receive → component stock split
    CREATE TABLE IF NOT EXISTS set_split_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        jo_id           INTEGER REFERENCES job_orders(id),
        jo_line_id      INTEGER REFERENCES jo_lines(id),
        so_number       TEXT NOT NULL,
        main_sku        TEXT NOT NULL,
        process         TEXT NOT NULL DEFAULT 'Cutting',
        split_qty       INTEGER NOT NULL,
        components_json TEXT NOT NULL DEFAULT '[]',
        created_at      TEXT DEFAULT (datetime('now'))
    );

    -- Audit: Finishing set-match → main SKU Packing stock
    CREATE TABLE IF NOT EXISTS set_match_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        so_number       TEXT NOT NULL,
        main_sku        TEXT NOT NULL,
        from_process    TEXT NOT NULL DEFAULT 'Finishing',
        to_process      TEXT NOT NULL DEFAULT 'Packing',
        match_qty       INTEGER NOT NULL,
        components_json TEXT NOT NULL DEFAULT '[]',
        matched_by      TEXT DEFAULT '',
        remarks         TEXT DEFAULT '',
        created_at      TEXT DEFAULT (datetime('now'))
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
        ("jo_lines", "received_qty", "INTEGER DEFAULT 0"),
        ("jo_lines", "issued_qty", "INTEGER DEFAULT 0"),
        ("jo_lines", "rejected_qty", "INTEGER DEFAULT 0"),
        ("jo_lines", "balance_qty", "INTEGER DEFAULT 0"),
        ("jo_lines", "vendor_rate", "REAL DEFAULT 0"),
        ("jo_lines", "process_cost", "REAL DEFAULT 0"),
        ("jo_lines", "parent_sku", "TEXT DEFAULT ''"),
        ("jo_lines", "sku_role", "TEXT DEFAULT 'MAIN'"),
        ("jo_lines", "component_code", "TEXT DEFAULT ''"),
        ("job_orders", "main_sku", "TEXT DEFAULT ''"),
        ("job_orders", "component_code", "TEXT DEFAULT ''"),
        ("job_orders", "sku_role", "TEXT DEFAULT 'MAIN'"),
        # Operation-based routing / partial WIP
        ("set_bom_headers", "stitching_requires_complete_set", "INTEGER DEFAULT 1"),
        ("set_bom_headers", "bundle_gate_process", "TEXT DEFAULT 'Cutting'"),
        ("set_bom_lines", "routing", "TEXT DEFAULT ''"),
        ("set_bom_lines", "requires_embroidery", "INTEGER DEFAULT 0"),
        ("set_bom_lines", "component_role", "TEXT DEFAULT 'SET_COMPONENT'"),
        ("set_bom_lines", "parent_component_code", "TEXT DEFAULT ''"),
        ("set_bom_lines", "creates_cutting_jo", "INTEGER DEFAULT 1"),
    ]
    for table, col, decl in migrations:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        except Exception:
            pass
    # Ensure Embroidery exists as a process step in item master (best-effort).
    try:
        ic = _item_connect()
        ic.execute(
            "INSERT OR IGNORE INTO routing_steps (name, description, sort_order) VALUES (?,?,?)",
            ("Embroidery", "Partial panel / fabric embroidery (child of Cutting)", 15),
        )
        ic.commit()
        ic.close()
    except Exception:
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


def get_component_routing(sku: str) -> list:
    """Ordered process path for a SKU, honoring Set BOM component routing when present."""
    from ..services.operation_routing import resolve_component_routing
    from ..services.set_components import parse_component_sku

    item_path = get_item_routing(sku)
    main, comp = parse_component_sku(sku)
    if not comp:
        return item_path
    bom = get_set_bom_for_sku(main or sku)
    if not bom:
        return item_path
    for ln in bom.get("lines") or []:
        if str(ln.get("component_code") or "").strip().upper() != str(comp).upper():
            continue
        return resolve_component_routing(
            routing=ln.get("routing"),
            default_next_process=ln.get("default_next_process"),
            item_routing=item_path,
        )
    return item_path


def get_next_process(sku: str, current_process: str) -> Optional[str]:
    """Next process for a SKU — uses component-level routing when Set BOM defines it."""
    from ..services.operation_routing import next_process_in_path, normalize_process_name

    path = get_component_routing(sku)
    cur = normalize_process_name(current_process)
    # Returning from Embroidery to Cutting: next after Cutting is Stitching (post-embroidery hop).
    if cur == "Cutting" and "Embroidery" in path:
        post = next_process_in_path(path, "Cutting", after_process="Embroidery")
        pre = next_process_in_path(path, "Cutting")
        # Default child hop is Embroidery; callers issuing after return pass to_process=Stitching.
        return pre or post
    nxt = next_process_in_path(path, current_process)
    if nxt:
        return nxt
    try:
        idx = [normalize_process_name(p) for p in path].index(cur)
        if idx + 1 < len(path):
            return path[idx + 1]
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
    delta = int(qty_in) - int(qty_out)
    conn.execute("""
        INSERT INTO process_stock(so_number, sku, process, available_qty, total_in, total_out, updated_at)
        VALUES(?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(so_number, sku, process) DO UPDATE SET
            available_qty = MAX(0, available_qty + excluded.available_qty),
            total_in = total_in + excluded.total_in,
            total_out = total_out + excluded.total_out,
            updated_at = datetime('now')
    """, (so_number, sku, process, delta, int(qty_in), int(qty_out)))


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
    try:
        from ..services.jo_issue_notes import get_issue_note_by_jo_id
        jo['issue_note'] = get_issue_note_by_jo_id(jo['id'])
    except Exception:
        jo['issue_note'] = None
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


def create_jo(data: dict) -> str | list[str]:
    """Create one JO, or multiple component Cutting JOs when Set BOM applies."""
    from ..services.component_bom import (
        resolve_cutting_main_sku,
        should_auto_create_component_jos,
    )
    from ..services.set_components import parse_component_sku

    data = dict(data)
    main = resolve_cutting_main_sku(data)
    if main:
        data["sku"] = main

    active_lines = [
        dict(ln)
        for ln in (data.get("lines") or [])
        if int(ln.get("planned_qty") or 0) > 0 and str(ln.get("sku") or "").strip()
    ]

    def _payload_for_line(ln: dict) -> dict:
        s = str(ln.get("sku") or data.get("sku") or "").strip().upper()
        main_sku, comp = parse_component_sku(s)
        if comp:
            raise ValueError(
                f"Cannot create a Cutting JO on component SKU {s}; use the main size SKU (e.g. {main_sku})."
            )
        pl = dict(data)
        pl["sku"] = main_sku or s
        pl["planned_qty"] = int(ln.get("planned_qty") or pl.get("planned_qty") or 0)
        pl["lines"] = [{**ln, "sku": pl["sku"], "planned_qty": pl["planned_qty"]}]
        return pl

    if active_lines:
        probe = _payload_for_line(active_lines[0])
        if should_auto_create_component_jos(probe):
            if len(active_lines) == 1:
                return create_component_cutting_jos(probe)
            nums: list[str] = []
            for ln in active_lines:
                nums.extend(create_component_cutting_jos(_payload_for_line(ln)))
            return nums
        if len(active_lines) > 1:
            return [_create_single_jo(_payload_for_line(ln)) for ln in active_lines]

    if should_auto_create_component_jos(data):
        return create_component_cutting_jos(data)
    return _create_single_jo(data)


def create_component_cutting_jos(data: dict) -> list[str]:
    """Explode a main-SKU Cutting plan into one JO per Set BOM set-component.

    Panel rows (Front/Back/…) do not get Cutting Job Orders.
    """
    from ..services.component_bom import (
        effective_set_bom_for_cutting,
        resolve_cutting_main_sku,
        set_component_lines,
    )
    from ..services.set_components import component_sku

    main_sku = resolve_cutting_main_sku(data) or str(data.get("sku") or "").strip().upper()
    bom = effective_set_bom_for_cutting(main_sku)
    lines = set_component_lines(bom)
    if not lines:
        return [_create_single_jo(data)]

    base_planned = int(data.get("planned_qty") or 0)
    jo_numbers: list[str] = []
    for ln in lines:
        code = ln["component_code"]
        ratio = max(int(ln.get("qty_per_set") or 1), 1)
        comp_qty = base_planned * ratio
        csku = component_sku(main_sku, code)
        comp_name = str(ln.get("component_name") or code).strip()
        comp_data = dict(data)
        comp_data.update(
            {
                "sku": csku,
                "sku_name": f"{data.get('sku_name') or main_sku} {comp_name}".strip(),
                "planned_qty": comp_qty,
                "main_sku": main_sku,
                "component_code": code,
                "sku_role": "COMPONENT",
                "create_component_jos": False,
                "lines": [
                    {
                        "so_number": data.get("so_number", ""),
                        "sku": csku,
                        "sku_name": comp_name,
                        "style": (data.get("lines") or [{}])[0].get("style", "")
                        if data.get("lines")
                        else "",
                        "planned_qty": comp_qty,
                        "parent_sku": main_sku,
                        "component_code": code,
                        "sku_role": "COMPONENT",
                        "vendor_rate": float(data.get("vendor_rate") or 0),
                        "remarks": data.get("remarks", ""),
                    }
                ],
            }
        )
        mats = ln.get("materials") or []
        if mats and not (comp_data.get("fabric_code") or "").strip():
            first = mats[0]
            comp_data["fabric_code"] = str(first.get("material_code") or "").strip()
            comp_data["fabric_qty"] = round(
                float(first.get("quantity") or 0) * comp_qty, 3
            )
            comp_data["fabric_unit"] = str(first.get("unit") or "MTR")
        jo_numbers.append(_create_single_jo(comp_data))
    return jo_numbers


def _create_single_jo(data: dict) -> str:
    so_number = (data.get("so_number") or "").strip()
    fabric_code = (data.get("fabric_code") or "").strip()
    fabric_qty = float(data.get("fabric_qty") or 0)
    if so_number and fabric_code and fabric_qty > 0:
        check_mrp_commitment(so_number, fabric_code, fabric_qty)
    conn = _connect()
    num = _next_jo(conn)
    process = data.get('process') or data.get('stage') or 'Cutting'
    planned = int(data.get('planned_qty') or 0)
    conn.execute("""INSERT INTO job_orders(
        jo_number, jo_date, so_number, sku, sku_name, process, stage,
        exec_type, vendor_name, vendor_rate, so_qty, planned_qty, balance_qty,
        status, expected_completion, issued_to, remarks,
        fabric_code, fabric_qty, fabric_unit, main_sku, component_code, sku_role, updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        (num, data.get('jo_date') or datetime.now().strftime('%Y-%m-%d'),
         data.get('so_number',''), data.get('sku',''), data.get('sku_name',''),
         process, process,
         data.get('exec_type','Inhouse'), data.get('vendor_name',''),
         float(data.get('vendor_rate') or 0),
         int(data.get('so_qty') or 0), planned, planned,
         'Created', data.get('expected_completion',''),
         data.get('issued_to',''), data.get('remarks',''),
         data.get('fabric_code',''), float(data.get('fabric_qty') or 0),
         data.get('fabric_unit','MTR'),
         str(data.get('main_sku') or '').strip().upper(),
         str(data.get('component_code') or '').strip().upper(),
         str(data.get('sku_role') or 'MAIN').strip().upper() or 'MAIN'))
    joid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for ln in data.get('lines', []):
        pq = int(ln.get('planned_qty', 0))
        conn.execute("""INSERT INTO jo_lines(jo_id,so_number,sku,sku_name,style,planned_qty,balance_qty,vendor_rate,remarks,parent_sku,sku_role,component_code)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (joid, ln.get('so_number', data.get('so_number','')),
             ln.get('sku', data.get('sku','')),
             ln.get('sku_name', data.get('sku_name','')),
             ln.get('style',''), pq, pq,
             float(ln.get('vendor_rate') or 0),
             ln.get('remarks',''),
             str(ln.get('parent_sku') or data.get('main_sku') or '').strip().upper(),
             str(ln.get('sku_role') or data.get('sku_role') or 'MAIN').strip().upper() or 'MAIN',
             str(ln.get('component_code') or data.get('component_code') or '').strip().upper()))
    jo_snapshot = {
        "jo_date": data.get("jo_date") or datetime.now().strftime("%Y-%m-%d"),
        "so_number": data.get("so_number", ""),
        "sku": data.get("sku", ""),
        "sku_name": data.get("sku_name", ""),
        "process": process,
        "planned_qty": planned,
        "fabric_code": data.get("fabric_code", ""),
        "fabric_qty": float(data.get("fabric_qty") or 0),
        "fabric_unit": data.get("fabric_unit", "MTR"),
        "main_sku": str(data.get("main_sku") or "").strip().upper(),
        "component_code": str(data.get("component_code") or "").strip().upper(),
        "sku_role": str(data.get("sku_role") or "MAIN").strip().upper() or "MAIN",
    }
    line_snapshots = list(data.get("lines") or [])
    conn.commit()
    conn.close()
    try:
        from ..services.jo_issue_notes import create_issue_note_for_jo

        create_issue_note_for_jo(joid, num, jo_snapshot, line_snapshots)
    except Exception:
        pass
    if so_number and fabric_code and fabric_qty > 0:
        record_mrp_jo_commitment(so_number, fabric_code, fabric_qty)
    if process == "Cutting" and so_number and data.get("sku"):
        try:
            from .grey_db import close_printed_reservations_when_fully_planned

            close_printed_reservations_when_fully_planned(
                so_number, (data.get("sku") or "").strip(), fabric_code
            )
        except Exception:
            pass
    return num


def update_jo(joid: int, data: dict):
    conn = _connect()
    prev = conn.execute("SELECT so_number, fabric_code, fabric_qty, status FROM job_orders WHERE id=?", (joid,)).fetchone()
    conn.close()
    allowed = ['status','output_qty','received_qty','rejected_qty','balance_qty',
               'completed_date','remarks','issued_to','exec_type','vendor_name',
               'vendor_rate','fabric_issued_qty','fabric_received_qty',
               'fabric_consumption','process_cost','total_cost','next_stage_jo_id']
    sets = ', '.join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed]
    if not sets:
        return
    new_status = data.get("status")
    if prev and new_status == "Cancelled" and (prev["status"] or "") != "Cancelled":
        release_mrp_jo_commitment(
            prev["so_number"] or "",
            prev["fabric_code"] or "",
            float(prev["fabric_qty"] or 0),
        )
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
    if issued <= 0:
        conn.close()
        raise ValueError("Issued qty must be greater than 0")
    jo_line_id = data.get('jo_line_id')
    from_process = data.get('from_process') or jo.get('process','Cutting')
    to_process = data.get('to_process') or get_next_process(jo.get('sku',''), from_process)
    so_number = jo.get('so_number','')
    sku = data.get('sku') or jo.get('sku','')

    try:
        _assert_set_issue_allowed(conn, so_number, sku, from_process, to_process)
    except ValueError:
        conn.close()
        raise

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

    # Auto-create / grow the destination process JO (e.g. Embroidery work order for FRONT).
    child_jo = None
    if to_process and str(to_process).strip() and str(to_process) != str(from_process):
        child_jo = _ensure_downstream_jo_for_issue(
            conn,
            parent_joid=joid,
            parent_jo=jo,
            sku=str(sku or "").strip().upper(),
            to_process=str(to_process).strip(),
            qty=issued,
        )

    # Mark parent in progress when pieces leave.
    conn.execute(
        """UPDATE job_orders SET
            status = CASE WHEN status='Created' THEN 'In Progress' ELSE status END,
            updated_at=datetime('now') WHERE id=?""",
        (joid,),
    )

    conn.commit()
    conn.close()
    return {"ok": True, "child_jo": child_jo}


def _ensure_downstream_jo_for_issue(
    conn,
    *,
    parent_joid: int,
    parent_jo: dict,
    sku: str,
    to_process: str,
    qty: int,
) -> Optional[dict]:
    """Create or grow a destination-process JO when pieces are issued into it.

    Embroidery (and other intermediate processes) need their own pending work order
    so the department can receive / process / return against a document.
    """
    from ..services.operation_routing import normalize_process_name
    from ..services.set_components import parse_component_sku

    process = normalize_process_name(to_process) or str(to_process or "").strip()
    if not process or qty <= 0 or not sku:
        return None
    # Only spawn work orders for intermediate specialist processes.
    # Returning panels to Cutting (or moving to Packing) must not create a new JO.
    if process not in {"Embroidery", "Printing"}:
        return None

    so_number = str(parent_jo.get("so_number") or "").strip()
    main, comp = parse_component_sku(sku)
    sku_name = str(parent_jo.get("sku_name") or "").strip()
    if comp:
        sku_name = f"{sku_name} {comp}".strip() if sku_name else sku

    existing = conn.execute(
        """SELECT id, jo_number, planned_qty, received_qty, status FROM job_orders
           WHERE parent_jo_id=? AND process=? AND UPPER(TRIM(sku))=UPPER(TRIM(?))
             AND so_number=? AND status NOT IN ('Closed','Cancelled')
           ORDER BY id DESC LIMIT 1""",
        (parent_joid, process, sku, so_number),
    ).fetchone()

    if existing:
        existing = dict(existing)
        new_planned = int(existing.get("planned_qty") or 0) + int(qty)
        received = int(existing.get("received_qty") or 0)
        conn.execute(
            """UPDATE job_orders SET
                planned_qty=?,
                balance_qty=MAX(0, ? - COALESCE(received_qty,0)),
                status = CASE WHEN status='Closed' THEN 'In Progress' ELSE status END,
                updated_at=datetime('now')
               WHERE id=?""",
            (new_planned, new_planned, existing["id"]),
        )
        line = conn.execute(
            "SELECT id, planned_qty FROM jo_lines WHERE jo_id=? AND UPPER(TRIM(sku))=UPPER(TRIM(?)) LIMIT 1",
            (existing["id"], sku),
        ).fetchone()
        if line:
            line = dict(line)
            line_planned = int(line.get("planned_qty") or 0) + int(qty)
            conn.execute(
                """UPDATE jo_lines SET planned_qty=?, balance_qty=MAX(0, ? - COALESCE(received_qty,0))
                   WHERE id=?""",
                (line_planned, line_planned, line["id"]),
            )
        else:
            conn.execute(
                """INSERT INTO jo_lines(jo_id,so_number,sku,sku_name,style,planned_qty,balance_qty,
                    parent_sku,sku_role,component_code)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    existing["id"],
                    so_number,
                    sku,
                    sku_name or sku,
                    "",
                    qty,
                    qty,
                    main or "",
                    "PANEL" if comp else (str(parent_jo.get("sku_role") or "COMPONENT").upper()),
                    comp or "",
                ),
            )
        return {
            "id": existing["id"],
            "jo_number": existing.get("jo_number"),
            "process": process,
            "sku": sku,
            "planned_qty": new_planned,
            "created": False,
            "message": f"Updated {process} JO {existing.get('jo_number')} (+{qty} pcs)",
        }

    num = _next_jo(conn)
    conn.execute(
        """INSERT INTO job_orders(
            jo_number, jo_date, so_number, sku, sku_name, process, stage,
            exec_type, vendor_name, vendor_rate, so_qty, planned_qty, balance_qty, status,
            expected_completion, fabric_code, parent_jo_id, main_sku, component_code, sku_role,
            remarks, updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        (
            num,
            datetime.now().strftime("%Y-%m-%d"),
            so_number,
            sku,
            sku_name or sku,
            process,
            process,
            parent_jo.get("exec_type") or "Inhouse",
            parent_jo.get("vendor_name") or "",
            float(parent_jo.get("vendor_rate") or 0),
            int(parent_jo.get("so_qty") or 0),
            int(qty),
            int(qty),
            "Created",
            parent_jo.get("expected_completion") or "",
            parent_jo.get("fabric_code") or "",
            parent_joid,
            main or str(parent_jo.get("main_sku") or "").strip().upper(),
            comp or "",
            "PANEL" if comp else (str(parent_jo.get("sku_role") or "COMPONENT").upper() or "COMPONENT"),
            f"Auto-created from {parent_jo.get('jo_number') or parent_joid} issue → {process}",
        ),
    )
    new_joid = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.execute(
        """INSERT INTO jo_lines(jo_id,so_number,sku,sku_name,style,planned_qty,balance_qty,
            parent_sku,sku_role,component_code,remarks)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (
            new_joid,
            so_number,
            sku,
            sku_name or sku,
            "",
            int(qty),
            int(qty),
            main or "",
            "PANEL" if comp else "COMPONENT",
            comp or "",
            f"Issued from parent JO {parent_jo.get('jo_number') or parent_joid}",
        ),
    )
    # Only set next_stage_jo_id when parent has none yet (multi-panel may spawn several).
    if not parent_jo.get("next_stage_jo_id"):
        conn.execute(
            "UPDATE job_orders SET next_stage_jo_id=?, updated_at=datetime('now') WHERE id=?",
            (new_joid, parent_joid),
        )
    return {
        "id": new_joid,
        "jo_number": num,
        "process": process,
        "sku": sku,
        "planned_qty": int(qty),
        "created": True,
        "message": f"Created {process} JO {num} for {sku} ({qty} pcs)",
    }

# ── Receive Pieces ─────────────────────────────────────────────────────────────

def receive_pieces(joid: int, data: dict):
    from ..services.document_qty_control import max_allowed_receive

    conn = _connect()
    jo = dict(conn.execute("SELECT * FROM job_orders WHERE id=?", (joid,)).fetchone() or {})
    if not jo:
        conn.close()
        raise ValueError("JO not found")
    if (jo.get("status") or "") == "Closed":
        conn.close()
        raise ValueError("Job order is closed — further receive not allowed")
    received = int(data.get('received_qty', 0))
    if received <= 0:
        conn.close()
        raise ValueError("Received qty must be greater than 0")
    rejected = int(data.get('rejected_qty', 0))
    jo_line_id = data.get('jo_line_id')
    process = data.get('process') or jo.get('process', 'Cutting')
    so_number = jo.get('so_number', '')
    sku = data.get('sku') or jo.get('sku', '')
    jo_tol = 0.0
    if jo_line_id:
        line = conn.execute(
            "SELECT id, sku, planned_qty FROM jo_lines WHERE id=? AND jo_id=?",
            (jo_line_id, joid),
        ).fetchone()
        if not line:
            conn.close()
            raise ValueError("JO line not found for this job order")
        line = dict(line)
        sku = (line.get('sku') or sku).strip()
        already = int(conn.execute(
            "SELECT COALESCE(SUM(received_qty),0) FROM jo_piece_receipts WHERE jo_line_id=?",
            (jo_line_id,),
        ).fetchone()[0])
        planned = int(line.get('planned_qty') or 0)
        cap = int(max_allowed_receive(planned, jo_tol) + 0.999)
        if already + received > cap:
            conn.close()
            raise ValueError(
                f"Cannot receive {received} pcs — max {cap} allowed on this line "
                f"(planned {planned}, already {already})"
            )
    else:
        already = int(jo.get("received_qty") or 0)
        planned = int(jo.get("planned_qty") or 0)
        cap = int(max_allowed_receive(planned, jo_tol) + 0.999)
        if already + received > cap:
            conn.close()
            raise ValueError(
                f"Cannot receive {received} pcs — max {cap} allowed "
                f"(planned {planned}, already {already})"
            )

    try:
        _assert_set_receive_allowed(conn, so_number, sku, process)
    except ValueError:
        conn.close()
        raise

    conn.execute("""INSERT INTO jo_piece_receipts(jo_id,jo_line_id,process,so_number,sku,receipt_date,received_qty,rejected_qty,received_by,remarks)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (joid, jo_line_id, process, so_number, sku,
         data.get('receipt_date') or datetime.now().strftime('%Y-%m-%d'),
         received, rejected, data.get('received_by', ''), data.get('remarks', '')))
    conn.execute("""UPDATE job_orders SET
        received_qty = COALESCE(received_qty,0) + ?,
        output_qty = COALESCE(output_qty,0) + ?,
        status = CASE WHEN status='Created' THEN 'In Progress' ELSE status END,
        updated_at = datetime('now') WHERE id=?""", (received, received, joid))
    if jo_line_id:
        conn.execute("""UPDATE jo_lines SET
            received_qty = COALESCE(received_qty,0) + ?,
            rejected_qty = COALESCE(rejected_qty,0) + ?,
            balance_qty = planned_qty - (COALESCE(received_qty,0) + ?)
            WHERE id=? AND jo_id=?""", (received, rejected, received, jo_line_id, joid))
    # Child process JOs (e.g. Embroidery spawned from Cutting issue) already hold
    # stock from the parent issue transfer — do not double-count on receive.
    stock_preloaded = bool(jo.get("parent_jo_id")) and str(process) == str(jo.get("process") or "")
    if not stock_preloaded:
        _update_process_stock(conn, so_number, sku, process, qty_in=received)
    elif rejected > 0:
        # Rejected pcs leave the preloaded available pool.
        _update_process_stock(conn, so_number, sku, process, qty_out=rejected)

    split_info = None
    split_flag = data.get("split_components", True)
    if process == "Cutting" and split_flag is not False and str(split_flag).lower() not in ("0", "false", "no"):
        split_info = _split_cutting_receive(
            conn,
            joid=joid,
            jo_line_id=jo_line_id,
            so_number=so_number,
            main_sku=sku,
            process=process,
            split_qty=received,
        )
        # Parent set-component Cutting JO receive: explode FRONT/BACK panel stock
        # under that parent (panels never get their own Cutting JO).
        if split_info is None:
            split_info = _split_panels_on_component_receive(
                conn,
                so_number=so_number,
                component_sku_in=sku,
                process=process,
                split_qty=received,
            )

    from ..services.document_qty_control import jo_should_auto_close

    jo_after = dict(conn.execute("SELECT planned_qty, received_qty, status FROM job_orders WHERE id=?", (joid,)).fetchone())
    if jo_after and jo_should_auto_close(int(jo_after["planned_qty"] or 0), int(jo_after["received_qty"] or 0), jo_tol):
        conn.execute(
            """UPDATE job_orders SET status='Closed', completed_date=?, updated_at=datetime('now') WHERE id=?""",
            (datetime.now().strftime("%Y-%m-%d"), joid),
        )
    conn.commit()
    conn.close()
    return {"ok": True, "split": split_info}


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
        exec_type, vendor_name, vendor_rate, so_qty, planned_qty, balance_qty, status,
        expected_completion, fabric_code, parent_jo_id, updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        (num, datetime.now().strftime('%Y-%m-%d'),
         so_number, sku, parent.get('sku_name',''),
         next_process, next_process,
         parent.get('exec_type','Inhouse'),
         parent.get('vendor_name',''),
         float(parent.get('vendor_rate') or 0),
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
    try:
        from ..services.jo_issue_notes import create_issue_note_for_jo

        child = get_jo(new_joid)
        if child:
            create_issue_note_for_jo(new_joid, num, child, child.get("lines") or [])
    except Exception:
        pass
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


def _commitment_status(mrp_qty: float, po_qty: float, jo_qty: float) -> str:
    committed = float(po_qty or 0) + float(jo_qty or 0)
    mrp = float(mrp_qty or 0)
    if mrp <= 0:
        return "Open"
    if committed <= 0:
        return "Open"
    if committed >= mrp - 1e-9:
        return "Fully Processed"
    return "Partially Processed"


def _so_material_qty_from_breakdown(mat: dict, *, so_no: str) -> float:
    """Net requirement for one SO — sum all BOM breakdown lines (not last line only)."""
    total_req = float(mat.get("total_req") or 0)
    net_req = float(mat.get("net_req_with_soft", mat.get("net_req", 0)) or 0)
    gross = sum(
        float(bd.get("qty_req") or 0)
        for bd in (mat.get("breakdown") or [])
        if (bd.get("so_no") or "").strip() == so_no
    )
    if gross <= 0:
        return 0.0
    if total_req > 1e-9 and net_req > 0:
        return round(net_req * (gross / total_req), 3)
    return round(gross, 3)


def sync_mrp_commitments_from_run(so_numbers: list, materials: dict) -> None:
    """Upsert per-SO material requirements from normalized MRP payload."""
    conn = _connect()
    # Aggregate across all BOM breakdown rows — one material/SO can appear on many SKUs.
    aggregated: dict[tuple[str, str], float] = {}
    meta: dict[tuple[str, str], dict] = {}
    for mat_code, mat in (materials or {}).items():
        for bd in mat.get("breakdown") or []:
            so_no = (bd.get("so_no") or "").strip()
            if not so_no:
                continue
            if so_numbers and so_no not in so_numbers:
                continue
            gross = float(bd.get("qty_req") or 0)
            if gross <= 0:
                continue
            total_req = float(mat.get("total_req") or 0)
            net_req = float(mat.get("net_req_with_soft", mat.get("net_req", 0)) or 0)
            if total_req > 1e-9 and net_req > 0:
                slice_qty = round(net_req * (gross / total_req), 6)
            else:
                slice_qty = gross
            key = (so_no, mat_code)
            aggregated[key] = aggregated.get(key, 0.0) + slice_qty
            meta[key] = mat
    for (so_no, mat_code), qty in aggregated.items():
        qty = round(qty, 3)
        if qty <= 0:
            continue
        mat = meta.get((so_no, mat_code), {})
        row = conn.execute(
            "SELECT id, po_committed_qty, jo_committed_qty FROM mrp_material_commitments WHERE so_number=? AND material_code=?",
            (so_no, mat_code),
        ).fetchone()
        if row:
            po_c = float(row["po_committed_qty"] or 0)
            jo_c = float(row["jo_committed_qty"] or 0)
            conn.execute(
                """UPDATE mrp_material_commitments SET material_name=?, unit=?, mrp_qty=?,
                status=?, updated_at=datetime('now') WHERE id=?""",
                (
                    mat.get("name", mat_code),
                    mat.get("unit", "PCS"),
                    qty,
                    _commitment_status(qty, po_c, jo_c),
                    row["id"],
                ),
            )
        else:
            conn.execute(
                """INSERT INTO mrp_material_commitments
                (so_number, material_code, material_name, unit, mrp_qty, status)
                VALUES(?,?,?,?,?,?)""",
                (so_no, mat_code, mat.get("name", mat_code), mat.get("unit", "PCS"), qty, "Open"),
            )
    conn.commit()
    conn.close()


def check_mrp_commitment(so_number: str, material_code: str, qty: float) -> None:
    if not (so_number or "").strip() or not (material_code or "").strip():
        return
    conn = _connect()
    row = conn.execute(
        "SELECT mrp_qty, po_committed_qty, jo_committed_qty FROM mrp_material_commitments WHERE so_number=? AND material_code=?",
        (so_number.strip(), material_code.strip()),
    ).fetchone()
    conn.close()
    if not row:
        return
    mrp = float(row["mrp_qty"] or 0)
    committed = float(row["po_committed_qty"] or 0) + float(row["jo_committed_qty"] or 0)
    remaining = mrp - committed
    if float(qty or 0) > remaining + 1e-9:
        raise ValueError(
            f"Material planning limit for {material_code} on {so_number}: need {qty}, only {max(0, remaining):.3f} remaining "
            f"(planned {mrp:.3f}, already committed PO+JO {committed:.3f})"
        )


def check_mrp_po_commitment(so_number: str, material_code: str, qty: float) -> None:
    check_mrp_commitment(so_number, material_code, qty)


def record_mrp_po_commitment(so_number: str, lines: list, *, doc_ref: str = "") -> None:
    if not (so_number or "").strip():
        return
    conn = _connect()
    for ln in lines or []:
        code = (ln.get("material_code") or "").strip()
        if not code:
            continue
        qty = float(ln.get("po_qty") or ln.get("required_qty") or ln.get("qty") or 0)
        if qty <= 0:
            continue
        row = conn.execute(
            "SELECT id, mrp_qty, po_committed_qty, jo_committed_qty FROM mrp_material_commitments WHERE so_number=? AND material_code=?",
            (so_number.strip(), code),
        ).fetchone()
        if row:
            new_po = float(row["po_committed_qty"] or 0) + qty
            mrp = float(row["mrp_qty"] or 0)
            conn.execute(
                """UPDATE mrp_material_commitments SET po_committed_qty=?, status=?, updated_at=datetime('now')
                WHERE id=?""",
                (new_po, _commitment_status(mrp, new_po, row["jo_committed_qty"]), row["id"]),
            )
        else:
            conn.execute(
                """INSERT INTO mrp_material_commitments
                (so_number, material_code, material_name, unit, mrp_qty, po_committed_qty, status)
                VALUES(?,?,?,?,?,?,?)""",
                (
                    so_number.strip(),
                    code,
                    ln.get("material_name", ""),
                    ln.get("unit", "PCS"),
                    qty,
                    qty,
                    "Fully Processed",
                ),
            )
    conn.commit()
    conn.close()


def record_mrp_jo_commitment(so_number: str, fabric_code: str, fabric_qty: float) -> None:
    if not (so_number or "").strip() or not (fabric_code or "").strip():
        return
    qty = float(fabric_qty or 0)
    if qty <= 0:
        return
    conn = _connect()
    row = conn.execute(
        "SELECT id, mrp_qty, po_committed_qty, jo_committed_qty FROM mrp_material_commitments WHERE so_number=? AND material_code=?",
        (so_number.strip(), fabric_code.strip()),
    ).fetchone()
    if row:
        new_jo = float(row["jo_committed_qty"] or 0) + qty
        mrp = float(row["mrp_qty"] or 0)
        conn.execute(
            """UPDATE mrp_material_commitments SET jo_committed_qty=?, status=?, updated_at=datetime('now')
            WHERE id=?""",
            (new_jo, _commitment_status(mrp, row["po_committed_qty"], new_jo), row["id"]),
        )
    else:
        conn.execute(
            """INSERT INTO mrp_material_commitments
            (so_number, material_code, unit, mrp_qty, jo_committed_qty, status)
            VALUES(?,?,?,?,?,?)""",
            (so_number.strip(), fabric_code.strip(), "MTR", qty, qty, "Fully Processed"),
        )
    conn.commit()
    conn.close()


def release_mrp_jo_commitment(so_number: str, fabric_code: str, fabric_qty: float) -> None:
    if not (so_number or "").strip() or not (fabric_code or "").strip():
        return
    qty = float(fabric_qty or 0)
    if qty <= 0:
        return
    conn = _connect()
    row = conn.execute(
        "SELECT id, mrp_qty, po_committed_qty, jo_committed_qty FROM mrp_material_commitments WHERE so_number=? AND material_code=?",
        (so_number.strip(), fabric_code.strip()),
    ).fetchone()
    if row:
        new_jo = max(0.0, float(row["jo_committed_qty"] or 0) - qty)
        mrp = float(row["mrp_qty"] or 0)
        conn.execute(
            """UPDATE mrp_material_commitments SET jo_committed_qty=?, status=?, updated_at=datetime('now')
            WHERE id=?""",
            (new_jo, _commitment_status(mrp, row["po_committed_qty"], new_jo), row["id"]),
        )
    conn.commit()
    conn.close()


def get_mrp_commitments_for_so(so_number: str) -> list:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM mrp_material_commitments WHERE so_number=? ORDER BY material_code",
        (so_number.strip(),),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        mrp = float(d.get("mrp_qty") or 0)
        po_c = float(d.get("po_committed_qty") or 0)
        jo_c = float(d.get("jo_committed_qty") or 0)
        d["remaining_qty"] = round(max(0.0, mrp - po_c - jo_c), 6)
        d["can_create_po"] = d["remaining_qty"] > 1e-9
        d["can_create_jo"] = d["remaining_qty"] > 1e-9
        out.append(d)
    return out


# ── Set BOM + Cutting split + Finishing set-match ─────────────────────────────

def _set_bom_header_row(conn, style_key: str):
    return conn.execute(
        "SELECT * FROM set_bom_headers WHERE style_key=? AND COALESCE(active,1)=1",
        (str(style_key or "").strip().upper(),),
    ).fetchone()


def _set_bom_material_lines(conn, set_bom_line_id: int) -> list:
    rows = conn.execute(
        """SELECT * FROM set_bom_material_lines WHERE set_bom_line_id=?
           ORDER BY sort_order, id""",
        (int(set_bom_line_id),),
    ).fetchall()
    return [dict(r) for r in rows]


def _set_bom_lines(conn, header_id: int) -> list:
    rows = conn.execute(
        """SELECT * FROM set_bom_lines WHERE header_id=?
           ORDER BY sort_order, id""",
        (header_id,),
    ).fetchall()
    out = []
    for r in rows:
        ln = dict(r)
        ln["materials"] = _set_bom_material_lines(conn, ln["id"])
        out.append(ln)
    return out


def _hydrate_set_bom(conn, header_row) -> Optional[dict]:
    if not header_row:
        return None
    h = dict(header_row)
    h["lines"] = _set_bom_lines(conn, h["id"])
    from ..services.component_bom import annotate_set_bom_roles

    return annotate_set_bom_roles(h)


def list_set_boms(active_only: bool = True) -> list:
    conn = _connect()
    q = "SELECT * FROM set_bom_headers"
    if active_only:
        q += " WHERE COALESCE(active,1)=1"
    q += " ORDER BY style_key"
    headers = [dict(r) for r in conn.execute(q).fetchall()]
    out = []
    for h in headers:
        h["lines"] = _set_bom_lines(conn, h["id"])
        out.append(h)
    conn.close()
    return out


def get_set_bom(style_key: str) -> Optional[dict]:
    conn = _connect()
    bom = _hydrate_set_bom(conn, _set_bom_header_row(conn, style_key))
    conn.close()
    return bom


def get_set_bom_for_sku(sku: str) -> Optional[dict]:
    """Resolve Set BOM for a size SKU or component SKU."""
    from ..services.set_components import style_key_for_set_bom, parse_component_sku

    raw = str(sku or "").strip().upper()
    if not raw:
        return None
    conn = _connect()
    # Prefer exact style_key match, then stripped parent, then main of component.
    candidates = []
    main, _comp = parse_component_sku(raw)
    if main:
        candidates.append(main)
        candidates.append(style_key_for_set_bom(main))
    candidates.append(raw)
    candidates.append(style_key_for_set_bom(raw))
    seen = set()
    for key in candidates:
        k = str(key or "").strip().upper()
        if not k or k in seen:
            continue
        seen.add(k)
        bom = _hydrate_set_bom(conn, _set_bom_header_row(conn, k))
        if bom:
            conn.close()
            return bom
    conn.close()
    return None


def upsert_set_bom(data: dict) -> dict:
    from ..services.component_bom import (
        ROLE_PANEL,
        ROLE_SET_COMPONENT,
        normalize_line_role,
    )
    from ..services.set_components import normalize_component_code

    style_key = str(data.get("style_key") or "").strip().upper()
    if not style_key:
        raise ValueError("style_key is required")
    lines_in = data.get("lines") or []
    if not lines_in:
        raise ValueError("At least one component line is required")
    cleaned = []
    seen_codes = set()
    for i, ln in enumerate(lines_in):
        code = normalize_component_code(ln.get("component_code") or ln.get("code") or "")
        if code in seen_codes:
            raise ValueError(f"Duplicate component code: {code}")
        seen_codes.add(code)
        qty = max(int(ln.get("qty_per_set") or 1), 1)
        role = normalize_line_role(ln)
        parent = str(ln.get("parent_component_code") or "").strip().upper()
        if role == ROLE_PANEL and not parent:
            parent = ""  # filled after all codes known
        cleaned.append(
            {
                "component_code": code,
                "component_name": str(ln.get("component_name") or code).strip(),
                "qty_per_set": qty,
                "default_next_process": str(ln.get("default_next_process") or "").strip(),
                "routing": str(ln.get("routing") or "").strip(),
                "requires_embroidery": 1 if ln.get("requires_embroidery") else 0,
                "component_role": role,
                "parent_component_code": parent,
                "creates_cutting_jo": 0 if role == ROLE_PANEL else 1,
                "sort_order": int(ln.get("sort_order") if ln.get("sort_order") is not None else i),
                "materials": [
                    {
                        "material_code": str(m.get("material_code") or "").strip().upper(),
                        "material_name": str(m.get("material_name") or "").strip(),
                        "quantity": float(m.get("quantity") or 0),
                        "unit": str(m.get("unit") or "MTR").strip() or "MTR",
                        "sort_order": int(m.get("sort_order") if m.get("sort_order") is not None else j),
                    }
                    for j, m in enumerate(ln.get("materials") or [])
                    if str(m.get("material_code") or "").strip()
                ],
            }
        )
    from ..services.component_bom import _infer_panel_parent

    set_codes = {
        ln["component_code"]
        for ln in cleaned
        if ln["component_role"] == ROLE_SET_COMPONENT
    }
    for ln in cleaned:
        if ln["component_role"] == ROLE_PANEL and not ln["parent_component_code"]:
            ln["parent_component_code"] = _infer_panel_parent(ln["component_code"], set_codes)
    if not any(ln["component_role"] == ROLE_SET_COMPONENT for ln in cleaned):
        raise ValueError(
            "At least one Set Component (Top / Bottom / …) is required — "
            "Front/Back panels alone cannot create Cutting Job Orders"
        )
    conn = _connect()
    existing = conn.execute(
        "SELECT id FROM set_bom_headers WHERE style_key=?", (style_key,)
    ).fetchone()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    style_name = str(data.get("style_name") or "").strip()
    remarks = str(data.get("remarks") or "").strip()
    active = 1 if data.get("active", True) else 0
    stitching_gate = 1 if data.get("stitching_requires_complete_set", True) else 0
    bundle_gate = str(data.get("bundle_gate_process") or "Cutting").strip() or "Cutting"
    if existing:
        hid = int(existing["id"])
        conn.execute(
            """UPDATE set_bom_headers
               SET style_name=?, active=?, remarks=?, updated_at=?,
                   stitching_requires_complete_set=?, bundle_gate_process=?
               WHERE id=?""",
            (style_name, active, remarks, now, stitching_gate, bundle_gate, hid),
        )
        conn.execute("DELETE FROM set_bom_lines WHERE header_id=?", (hid,))
    else:
        conn.execute(
            """INSERT INTO set_bom_headers(
                   style_key, style_name, active, remarks, created_at, updated_at,
                   stitching_requires_complete_set, bundle_gate_process)
               VALUES(?,?,?,?,?,?,?,?)""",
            (style_key, style_name, active, remarks, now, now, stitching_gate, bundle_gate),
        )
        hid = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    for ln in cleaned:
        # Derive routing from default_next_process when routing blank.
        routing = ln["routing"]
        if not routing and ln["default_next_process"]:
            from ..services.operation_routing import resolve_component_routing, routing_path_to_string

            routing = routing_path_to_string(
                resolve_component_routing(
                    default_next_process=ln["default_next_process"],
                    item_routing=["Cutting", "Stitching", "Finishing"],
                )
            )
        requires_emb = ln["requires_embroidery"]
        if not requires_emb and "Embroidery" in routing:
            requires_emb = 1
        conn.execute(
            """INSERT INTO set_bom_lines
               (header_id, component_code, component_name, qty_per_set,
                default_next_process, routing, requires_embroidery,
                component_role, parent_component_code, creates_cutting_jo, sort_order)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                hid,
                ln["component_code"],
                ln["component_name"],
                ln["qty_per_set"],
                ln["default_next_process"],
                routing,
                requires_emb,
                ln["component_role"],
                ln["parent_component_code"],
                ln["creates_cutting_jo"],
                ln["sort_order"],
            ),
        )
        line_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        for mat in ln.get("materials") or []:
            conn.execute(
                """INSERT INTO set_bom_material_lines
                   (set_bom_line_id, material_code, material_name, quantity, unit, sort_order)
                   VALUES(?,?,?,?,?,?)""",
                (
                    line_id,
                    mat["material_code"],
                    mat["material_name"],
                    mat["quantity"],
                    mat["unit"],
                    mat["sort_order"],
                ),
            )
    conn.commit()
    bom = _hydrate_set_bom(conn, conn.execute("SELECT * FROM set_bom_headers WHERE id=?", (hid,)).fetchone())
    conn.close()
    return bom


def delete_set_bom(style_key: str) -> bool:
    conn = _connect()
    cur = conn.execute(
        "DELETE FROM set_bom_headers WHERE style_key=?",
        (str(style_key or "").strip().upper(),),
    )
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def _was_set_split(conn, so_number: str, main_sku: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM set_split_events WHERE so_number=? AND main_sku=? LIMIT 1",
        (so_number, str(main_sku or "").strip().upper()),
    ).fetchone()
    return bool(row)


def _assert_set_issue_allowed(conn, so_number: str, sku: str, from_process: str, to_process: str) -> None:
    from ..services.operation_routing import normalize_process_name
    from ..services.set_components import parse_component_sku

    sku_u = str(sku or "").strip().upper()
    to_p = normalize_process_name(to_process)
    from_p = normalize_process_name(from_process)
    main, comp = parse_component_sku(sku_u)
    if to_p == "Packing":
        if comp:
            raise ValueError(
                "Component SKUs cannot be issued to Packing — use Set Match to form complete sets"
            )
        # After Cutting split, Packing stock for the main SKU comes only from Set Match
        if _was_set_split(conn, so_number, sku_u):
            raise ValueError(
                "This style was split into components — use Set Match to move complete sets to Packing"
            )

    # Stitching gate is per set-component (TOP panels, or PANT alone, etc.) —
    # Top / Bottom / Dupatta may stitch independently.
    if to_p == "Stitching" and from_p != "Stitching":
        style_sku = main or sku_u
        bom = get_set_bom_for_sku(style_sku)
        if not bom or not bom.get("lines"):
            return
        if not int(bom.get("stitching_requires_complete_set") if bom.get("stitching_requires_complete_set") is not None else 1):
            return
        scope = _stitching_gate_scope(bom, comp)
        ready = preview_bundle_ready(
            so_number,
            style_sku,
            conn=conn,
            parent_component_code=scope,
        )
        if not ready.get("bundle_complete"):
            msg = ready.get("message") or "Bundle incomplete"
            raise ValueError(
                f"Stitching blocked — {msg}. "
                "All mandatory panels for this component must be at the gate process with embroidery complete."
            )


def _stitching_gate_scope(bom: dict | None, component_code: str | None) -> str | None:
    """Resolve which set-component's panel bundle must be ready for stitching.

    FRONT/BACK → parent TOP; TOP itself → TOP; PANT with no panels → PANT.
    """
    from ..services.component_bom import ROLE_PANEL, ROLE_SET_COMPONENT, normalize_line_role

    code = str(component_code or "").strip().upper()
    if not code or not bom:
        return None
    for ln in bom.get("lines") or []:
        if str(ln.get("component_code") or "").strip().upper() != code:
            continue
        role = normalize_line_role(ln)
        if role == ROLE_PANEL:
            return str(ln.get("parent_component_code") or "").strip().upper() or None
        if role == ROLE_SET_COMPONENT:
            return code
    return code


def _assert_set_receive_allowed(conn, so_number: str, sku: str, process: str) -> None:
    from ..services.set_components import parse_component_sku

    sku_u = str(sku or "").strip().upper()
    proc = str(process or "").strip()
    _main, comp = parse_component_sku(sku_u)
    if proc == "Packing" and not comp and _was_set_split(conn, so_number, sku_u):
        raise ValueError(
            "For split set styles, Packing stock is created via Set Match — cannot receive Packing on main SKU"
        )


def preview_bundle_ready(
    so_number: str,
    main_sku: str,
    *,
    conn=None,
    parent_component_code: str | None = None,
) -> dict:
    """WIP board + stitching gate for a style (optionally scoped to one set component).

    When ``parent_component_code`` is set (e.g. TOP), only that component's panels
    (FRONT/BACK) must be ready — PANT/DUPATTA do not block TOP stitching.
    """
    from ..services.operation_routing import compute_bundle_readiness, resolve_component_routing
    from ..services.set_components import component_sku, parse_component_sku

    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        main = str(main_sku or "").strip().upper()
        parsed_main, parsed_comp = parse_component_sku(main)
        if parsed_main:
            main = parsed_main
        bom = get_set_bom_for_sku(main)
        if not bom or not bom.get("lines"):
            return {
                "bundle_complete": True,
                "complete_sets": 0,
                "gate_process": "Cutting",
                "components": [],
                "blockers": [],
                "message": "No Set BOM — stitching gate not enforced",
                "so_number": so_number,
                "main_sku": main,
            }
        gate = str(bom.get("bundle_gate_process") or "Cutting").strip() or "Cutting"
        from ..services.component_bom import ROLE_PANEL, ROLE_SET_COMPONENT, normalize_line_role, panel_lines as _panel_lines, set_component_lines

        # Auto-scope when caller passed a component SKU (TEST SKU-TOP → TOP).
        scope = str(parent_component_code or "").strip().upper() or None
        if not scope and parsed_comp:
            scope = _stitching_gate_scope(bom, parsed_comp)

        all_set_lines = set_component_lines(bom) or [
            ln for ln in (bom.get("lines") or []) if normalize_line_role(ln) == ROLE_SET_COMPONENT
        ]
        if scope:
            gate_lines = [ln for ln in all_set_lines if str(ln.get("component_code") or "").upper() == scope]
            if not gate_lines:
                # Scope may be a panel parent that is still a set component code.
                gate_lines = [
                    ln for ln in (bom.get("lines") or [])
                    if str(ln.get("component_code") or "").upper() == scope
                ]
        else:
            gate_lines = all_set_lines or list(bom.get("lines") or [])

        comps = []
        for ln in gate_lines:
            code = ln["component_code"]
            csku = component_sku(main, code)
            path = resolve_component_routing(
                routing=ln.get("routing"),
                default_next_process=ln.get("default_next_process"),
                item_routing=["Cutting", "Stitching", "Finishing"],
            )
            gate_row = conn.execute(
                "SELECT COALESCE(available_qty,0) AS q FROM process_stock WHERE so_number=? AND sku=? AND process=?",
                (so_number, csku, gate),
            ).fetchone()
            emb_row = conn.execute(
                "SELECT COALESCE(available_qty,0) AS q FROM process_stock WHERE so_number=? AND sku=? AND process=?",
                (so_number, csku, "Embroidery"),
            ).fetchone()
            avail_gate = int(gate_row["q"] if gate_row else 0)
            emb_out = int(emb_row["q"] if emb_row else 0)
            location = "Embroidery" if emb_out > 0 else (gate if avail_gate > 0 else "")
            if not location:
                stocks = conn.execute(
                    "SELECT process, available_qty FROM process_stock WHERE so_number=? AND sku=? AND available_qty>0",
                    (so_number, csku),
                ).fetchall()
                if stocks:
                    location = str(stocks[0]["process"])
            comps.append(
                {
                    "component_code": code,
                    "component_name": ln.get("component_name") or code,
                    "component_sku": csku,
                    "qty_per_set": ln.get("qty_per_set") or 1,
                    "available_at_gate": avail_gate,
                    "embroidery_outstanding": emb_out,
                    "location": location,
                    "routing": path,
                    "component_role": ln.get("component_role") or "SET_COMPONENT",
                    "parent_component_code": ln.get("parent_component_code") or "",
                }
            )

        panel_comps = []
        for ln in _panel_lines(bom, parent_component_code=scope):
            code = ln["component_code"]
            csku = component_sku(main, code)
            path = resolve_component_routing(
                routing=ln.get("routing"),
                default_next_process=ln.get("default_next_process"),
                item_routing=["Cutting", "Embroidery", "Cutting", "Stitching"],
            )
            emb_row = conn.execute(
                "SELECT COALESCE(available_qty,0) AS q FROM process_stock WHERE so_number=? AND sku=? AND process=?",
                (so_number, csku, "Embroidery"),
            ).fetchone()
            gate_row = conn.execute(
                "SELECT COALESCE(available_qty,0) AS q FROM process_stock WHERE so_number=? AND sku=? AND process=?",
                (so_number, csku, gate),
            ).fetchone()
            # Panels already issued to Stitching (or later) still satisfy the TOP
            # panel-bundle gate so siblings can follow.
            downstream = conn.execute(
                """SELECT COALESCE(SUM(available_qty),0) AS q FROM process_stock
                   WHERE so_number=? AND sku=? AND process IN ('Stitching','Finishing','Packing','QC')""",
                (so_number, csku),
            ).fetchone()
            avail_gate = int(gate_row["q"] if gate_row else 0)
            emb_out = int(emb_row["q"] if emb_row else 0)
            satisfied = avail_gate + int(downstream["q"] if downstream else 0)
            location = "Embroidery" if emb_out > 0 else (gate if avail_gate > 0 else "")
            if not location and satisfied > avail_gate:
                location = "Stitching"
            panel_comps.append(
                {
                    "component_code": code,
                    "component_name": ln.get("component_name") or code,
                    "component_sku": csku,
                    "qty_per_set": ln.get("qty_per_set") or 1,
                    "available_at_gate": avail_gate,
                    "available_at_cutting": avail_gate,
                    "satisfied_qty": satisfied,
                    "embroidery_outstanding": emb_out,
                    "location": location,
                    "routing": path,
                    "component_role": ROLE_PANEL,
                    "parent_component_code": ln.get("parent_component_code") or "",
                    "gate_optional": True,
                }
            )

        # When this set-component has panels, the stitching gate is the panel bundle
        # (FRONT/BACK), not the sibling set components (PANT/DUPATTA).
        if panel_comps and scope:
            # Cutting + already-stitched qty both satisfy the panel-bundle gate.
            gate_for_math = [
                {
                    **pc,
                    "available_at_gate": int(pc.get("satisfied_qty") or 0),
                    "gate_optional": False,
                }
                for pc in panel_comps
            ]
            out = compute_bundle_readiness(gate_for_math, gate_process=gate)
            for c in comps:
                out.setdefault("components", []).insert(0, {
                    **c,
                    "ready": bool(out.get("bundle_complete")),
                    "status": "Ready" if out.get("bundle_complete") else "Waiting for panels",
                    "gate_optional": True,
                })
        else:
            # No panels under scope: set-component itself must be at the gate.
            emb_by_parent: dict[str, int] = {}
            for pc in panel_comps:
                parent = str(pc.get("parent_component_code") or "").strip().upper()
                if not parent:
                    continue
                emb_by_parent[parent] = emb_by_parent.get(parent, 0) + int(pc.get("embroidery_outstanding") or 0)
            for c in comps:
                code = str(c.get("component_code") or "").strip().upper()
                extra = emb_by_parent.get(code, 0)
                if extra:
                    c["embroidery_outstanding"] = int(c.get("embroidery_outstanding") or 0) + extra
            comps.extend(panel_comps)
            out = compute_bundle_readiness(
                [c for c in comps if not c.get("gate_optional")],
                gate_process=gate,
            )

        # Append panel WIP rows for UI (already in math when scoped).
        panel_rows = panel_comps
        for c in panel_rows:
            from ..services.operation_routing import panel_wip_status, routing_path_to_string

            emb_out = int(c.get("embroidery_outstanding") or 0)
            avail = int(c.get("available_at_gate") or 0)
            path = c.get("routing") or []
            status = panel_wip_status(
                location=str(c.get("location") or ""),
                available_qty=emb_out if emb_out > 0 else avail,
                path=path if isinstance(path, list) else [],
                bundle_ready=bool(out.get("bundle_complete")),
            )
            # Avoid duplicating panels already included in gate math.
            already = {
                str(x.get("component_code") or "").upper()
                for x in (out.get("components") or [])
            }
            if str(c.get("component_code") or "").upper() in already:
                # Refresh status on existing row
                for x in out.get("components") or []:
                    if str(x.get("component_code") or "").upper() == str(c.get("component_code") or "").upper():
                        x["status"] = status
                        x["available_at_gate"] = avail
                        x["ready"] = emb_out == 0 and int(c.get("satisfied_qty") or avail) > 0
                        x["gate_optional"] = True
                continue
            out.setdefault("components", []).append(
                {
                    "component_code": c.get("component_code"),
                    "component_name": c.get("component_name"),
                    "component_sku": c.get("component_sku"),
                    "qty_per_set": c.get("qty_per_set") or 1,
                    "available_at_gate": avail,
                    "embroidery_outstanding": emb_out,
                    "location": c.get("location") or "",
                    "status": status,
                    "ready": emb_out == 0 and avail > 0,
                    "routing": routing_path_to_string(path) if isinstance(path, list) else str(path or ""),
                    "component_role": ROLE_PANEL,
                    "parent_component_code": c.get("parent_component_code") or "",
                    "gate_optional": True,
                }
            )
        if scope and out.get("bundle_complete"):
            out["message"] = (
                f"{out.get('complete_sets') or 0} complete {scope} bundle(s) ready at {gate}"
            )
        elif scope and out.get("blockers"):
            out["message"] = f"{scope} bundle incomplete — " + "; ".join((out.get("blockers") or [])[:3])
        out["so_number"] = so_number
        out["main_sku"] = main
        out["parent_component_code"] = scope or ""
        out["stitching_requires_complete_set"] = bool(
            int(bom.get("stitching_requires_complete_set") if bom.get("stitching_requires_complete_set") is not None else 1)
        )
        return out
    finally:
        if own_conn:
            conn.close()


def get_jo_panel_wip(joid: int) -> dict:
    """Panel rows + live stock for a Cutting JO (parent component or main set SKU)."""
    from ..services.component_bom import panel_lines
    from ..services.operation_routing import (
        panel_wip_status,
        resolve_component_routing,
        routing_path_to_string,
    )
    from ..services.set_components import component_sku, parse_component_sku

    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, sku, process, so_number FROM job_orders WHERE id=?",
            (joid,),
        ).fetchone()
        if not row:
            return {"has_panels": False}
        jo = dict(row)
        if str(jo.get("process") or "") != "Cutting":
            return {"has_panels": False, "jo_sku": jo.get("sku"), "reason": "not_cutting"}

        sku = str(jo.get("sku") or "").strip().upper()
        so_number = str(jo.get("so_number") or "").strip()
        main, comp = parse_component_sku(sku)
        bom = get_set_bom_for_sku(sku)
        if not bom:
            return {"has_panels": False, "jo_sku": sku, "so_number": so_number}

        style_main = main or sku
        if comp:
            panels_bom = panel_lines(bom, parent_component_code=comp)
            parent_code = comp
        else:
            panels_bom = panel_lines(bom)
            parent_code = None

        if not panels_bom:
            return {
                "has_panels": False,
                "jo_sku": sku,
                "main_sku": style_main,
                "so_number": so_number,
                "parent_component_code": parent_code,
            }

        gate = str(bom.get("bundle_gate_process") or "Cutting").strip() or "Cutting"
        panel_rows: list[dict] = []
        for ln in panels_bom:
            code = ln["component_code"]
            psku = component_sku(style_main, code)
            path = resolve_component_routing(
                routing=ln.get("routing"),
                default_next_process=ln.get("default_next_process"),
                item_routing=["Cutting", "Embroidery", "Cutting", "Stitching"],
            )
            stock_rows = conn.execute(
                "SELECT process, available_qty FROM process_stock WHERE so_number=? AND sku=?",
                (so_number, psku),
            ).fetchall()
            stock_map = {
                str(r["process"]): int(r["available_qty"] or 0)
                for r in stock_rows
                if int(r["available_qty"] or 0) > 0
            }
            emb_out = int(stock_map.get("Embroidery", 0))
            gate_qty = int(stock_map.get(gate, 0))
            cutting_qty = int(stock_map.get("Cutting", 0))
            avail = gate_qty if gate_qty > 0 else cutting_qty
            if emb_out > 0:
                location = "Embroidery"
            elif avail > 0:
                location = gate if gate_qty > 0 else "Cutting"
            elif stock_map:
                location = next(iter(stock_map.keys()))
            else:
                location = ""
            status = panel_wip_status(
                location=location or gate,
                available_qty=emb_out if emb_out > 0 else avail,
                path=path,
                bundle_ready=False,
            )
            issue_from = "Embroidery" if emb_out > 0 else "Cutting"
            issue_to = get_next_process(psku, issue_from)
            child = conn.execute(
                """SELECT id, jo_number, process, status, planned_qty, received_qty
                   FROM job_orders
                   WHERE parent_jo_id=? AND process='Embroidery'
                     AND UPPER(TRIM(sku))=UPPER(TRIM(?))
                     AND status NOT IN ('Cancelled')
                   ORDER BY id DESC LIMIT 1""",
                (joid, psku),
            ).fetchone()
            child_jo = None
            if child:
                child = dict(child)
                child_jo = {
                    "id": child["id"],
                    "jo_number": child.get("jo_number"),
                    "process": child.get("process"),
                    "status": child.get("status"),
                    "planned_qty": int(child.get("planned_qty") or 0),
                    "received_qty": int(child.get("received_qty") or 0),
                }
            # Path Cutting>Embroidery>Cutting>Stitching: after emb is done, next is Stitching.
            emb_done = emb_out == 0 and child_jo and (
                str(child_jo.get("status") or "").strip().lower() in ("closed", "complete", "completed")
                or (
                    int(child_jo.get("planned_qty") or 0) > 0
                    and int(child_jo.get("received_qty") or 0) >= int(child_jo.get("planned_qty") or 0)
                )
            )
            if emb_done and issue_from == "Cutting" and issue_to == "Embroidery":
                issue_to = "Stitching"
            panel_rows.append(
                {
                    "component_code": code,
                    "component_name": ln.get("component_name") or code,
                    "component_sku": psku,
                    "parent_component_code": ln.get("parent_component_code") or "",
                    "routing": routing_path_to_string(path),
                    "default_next_process": ln.get("default_next_process") or "",
                    "process_stocks": stock_map,
                    "available_qty": avail,
                    "embroidery_outstanding": emb_out,
                    "current_location": location or "—",
                    "status": status,
                    "next_process": get_next_process(psku, "Cutting"),
                    "issue_from_process": issue_from,
                    "issue_to_process": issue_to,
                    "issueable_qty": emb_out if emb_out > 0 else avail,
                    "embroidery_jo": child_jo,
                }
            )
    finally:
        conn.close()

    # Scope stitching readiness to this JO's set component (TOP panels only —
    # do not require PANT/DUPATTA on a TOP Cutting JO).
    bundle = (
        preview_bundle_ready(
            so_number,
            style_main,
            parent_component_code=parent_code,
        )
        if so_number and style_main
        else {}
    )
    from ..services.operation_routing import parse_routing_path as _parse_path

    bundle_ok = bool(bundle.get("bundle_complete"))
    for p in panel_rows:
        p["status"] = panel_wip_status(
            location=str(p.get("current_location") or gate),
            available_qty=int(p.get("embroidery_outstanding") or 0)
            if int(p.get("embroidery_outstanding") or 0) > 0
            else int(p.get("available_qty") or 0),
            path=_parse_path(p.get("routing")),
            bundle_ready=bundle_ok,
        )
    return {
        "has_panels": True,
        "jo_id": joid,
        "jo_sku": sku,
        "main_sku": style_main,
        "so_number": so_number,
        "parent_component_code": parent_code,
        "gate_process": gate,
        "panels": panel_rows,
        "bundle_complete": bundle.get("bundle_complete"),
        "complete_sets": bundle.get("complete_sets"),
        "bundle_message": bundle.get("message"),
        "stitching_requires_complete_set": bundle.get("stitching_requires_complete_set"),
        "hint": (
            "Receive this Cutting JO to create panel stock (FRONT/BACK), "
            "then issue each panel to its routed next process. "
            "Stitching for this component only needs its own panels ready — "
            "sibling pieces (Pant/Dupatta) stitch separately."
        ),
    }


def get_partial_wip_board(so_number: str, main_sku: str) -> dict:
    """Panel-level WIP locations/status for the Production UI."""
    ready = preview_bundle_ready(so_number, main_sku)
    return {
        "so_number": so_number,
        "main_sku": str(main_sku or "").strip().upper(),
        "bundle_complete": ready.get("bundle_complete"),
        "complete_sets": ready.get("complete_sets"),
        "gate_process": ready.get("gate_process"),
        "message": ready.get("message"),
        "stitching_requires_complete_set": ready.get("stitching_requires_complete_set"),
        "items": [
            {
                "item": c.get("component_name") or c.get("component_code"),
                "component_code": c.get("component_code"),
                "component_sku": c.get("component_sku"),
                "current_location": c.get("location") or "—",
                "status": c.get("status"),
                "available_at_gate": c.get("available_at_gate"),
                "embroidery_outstanding": c.get("embroidery_outstanding"),
                "routing": c.get("routing"),
                "ready": c.get("ready"),
            }
            for c in ready.get("components") or []
        ],
        "blockers": ready.get("blockers") or [],
    }


def _load_set_bom_for_split(conn, main_sku: str) -> Optional[dict]:
    from ..services.set_components import style_key_for_set_bom

    main = str(main_sku or "").strip().upper()
    if not main:
        return None
    style_key = style_key_for_set_bom(main)
    bom = _hydrate_set_bom(conn, _set_bom_header_row(conn, style_key))
    if not bom:
        bom = _hydrate_set_bom(conn, _set_bom_header_row(conn, main))
    return bom


def _create_panel_stocks_under_parents(
    conn,
    *,
    so_number: str,
    main_sku: str,
    process: str,
    parent_qty_by_code: dict[str, int],
    bom: dict,
) -> list[dict]:
    """Create Cutting process_stock for PANEL SKUs under each parent set component."""
    from ..services.component_bom import panel_lines
    from ..services.set_components import component_sku

    main = str(main_sku or "").strip().upper()
    panels_out: list[dict] = []
    for parent_code, parent_qty in parent_qty_by_code.items():
        pq = int(parent_qty or 0)
        if pq <= 0:
            continue
        parent = str(parent_code or "").strip().upper()
        for ln in panel_lines(bom, parent_component_code=parent):
            code = ln["component_code"]
            ratio = max(int(ln.get("qty_per_set") or 1), 1)
            qty = pq * ratio
            psku = component_sku(main, code)
            _update_process_stock(conn, so_number, psku, process, qty_in=qty)
            panels_out.append(
                {
                    "component_code": code,
                    "component_name": ln.get("component_name") or code,
                    "component_sku": psku,
                    "parent_component_code": parent,
                    "qty_per_set": ratio,
                    "qty": qty,
                    "default_next_process": ln.get("default_next_process") or "",
                    "component_role": "PANEL",
                }
            )
    return panels_out


def _split_panels_on_component_receive(
    conn,
    *,
    so_number: str,
    component_sku_in: str,
    process: str,
    split_qty: int,
) -> Optional[dict]:
    """When a parent set-component Cutting JO receives, explode its child panels."""
    from ..services.component_bom import panel_lines, set_component_lines
    from ..services.set_components import parse_component_sku

    sku_u = str(component_sku_in or "").strip().upper()
    if not sku_u or split_qty <= 0:
        return None
    main, comp = parse_component_sku(sku_u)
    if not main or not comp:
        return None

    bom = _load_set_bom_for_split(conn, main)
    if not bom:
        return None

    # Only explode from SET_COMPONENT parents — never from a panel SKU itself.
    set_codes = {str(ln["component_code"]).upper() for ln in set_component_lines(bom)}
    if comp not in set_codes:
        return None
    children = panel_lines(bom, parent_component_code=comp)
    if not children:
        return None

    panels_out = _create_panel_stocks_under_parents(
        conn,
        so_number=so_number,
        main_sku=main,
        process=process,
        parent_qty_by_code={comp: int(split_qty)},
        bom=bom,
    )
    if not panels_out:
        return None
    return {
        "main_sku": main,
        "parent_component_code": comp,
        "parent_sku": sku_u,
        "style_key": bom.get("style_key"),
        "split_qty": split_qty,
        "panels": panels_out,
        "components": panels_out,
        "message": (
            f"Split {split_qty} of {sku_u} into {len(panels_out)} panel SKU(s) "
            f"(FRONT/BACK WIP under parent Cutting JO)"
        ),
    }


def _split_cutting_receive(
    conn,
    *,
    joid: int,
    jo_line_id,
    so_number: str,
    main_sku: str,
    process: str,
    split_qty: int,
) -> Optional[dict]:
    """Move Cutting stock from main SKU onto set-component + panel SKUs per Set BOM."""
    from ..services.set_components import component_sku

    main = str(main_sku or "").strip().upper()
    if not main or split_qty <= 0:
        return None
    # Do not re-split a component receive (panel explosion handled separately)
    from ..services.set_components import parse_component_sku

    if parse_component_sku(main)[1]:
        return None

    bom = _load_set_bom_for_split(conn, main)
    from ..services.component_bom import set_component_lines

    lines = set_component_lines(bom)
    if not lines:
        return None

    components_out = []
    parent_qty_by_code: dict[str, int] = {}
    for ln in lines:
        code = ln["component_code"]
        ratio = max(int(ln.get("qty_per_set") or 1), 1)
        qty = int(split_qty) * ratio
        csku = component_sku(main, code)
        _update_process_stock(conn, so_number, csku, process, qty_in=qty)
        parent_qty_by_code[str(code).upper()] = qty
        components_out.append(
            {
                "component_code": code,
                "component_name": ln.get("component_name") or code,
                "component_sku": csku,
                "qty_per_set": ratio,
                "qty": qty,
                "default_next_process": ln.get("default_next_process") or "",
                "component_role": "SET_COMPONENT",
            }
        )

    panels_out = _create_panel_stocks_under_parents(
        conn,
        so_number=so_number,
        main_sku=main,
        process=process,
        parent_qty_by_code=parent_qty_by_code,
        bom=bom,
    )

    # Remove the just-received qty from main Cutting stock (components own it now)
    _update_process_stock(conn, so_number, main, process, qty_out=split_qty)

    if jo_line_id:
        conn.execute(
            """UPDATE jo_lines SET parent_sku=?, sku_role='MAIN', component_code=''
               WHERE id=?""",
            (main, jo_line_id),
        )

    event_payload = components_out + panels_out
    conn.execute(
        """INSERT INTO set_split_events
           (jo_id, jo_line_id, so_number, main_sku, process, split_qty, components_json)
           VALUES(?,?,?,?,?,?,?)""",
        (
            joid,
            jo_line_id,
            so_number,
            main,
            process,
            split_qty,
            json.dumps(event_payload),
        ),
    )
    n_comp = len(components_out)
    n_panel = len(panels_out)
    msg = f"Split {split_qty} of {main} into {n_comp} component SKU(s)"
    if n_panel:
        msg += f" and {n_panel} panel SKU(s)"
    return {
        "main_sku": main,
        "style_key": bom.get("style_key"),
        "split_qty": split_qty,
        "components": components_out,
        "panels": panels_out,
        "message": msg,
    }


def preview_set_match(
    so_number: str,
    main_sku: str,
    from_process: str = "Finishing",
) -> dict:
    from ..services.set_components import component_sku, compute_complete_sets, parse_component_sku

    so = str(so_number or "").strip()
    main = str(main_sku or "").strip().upper()
    if not so or not main:
        raise ValueError("so_number and main_sku are required")
    # If caller passed a component SKU, normalize to main
    parsed_main, _ = parse_component_sku(main)
    if parsed_main:
        main = parsed_main

    bom = get_set_bom_for_sku(main)
    from ..services.component_bom import set_component_lines

    lines = set_component_lines(bom)
    if not lines:
        raise ValueError(f"No Set BOM components defined for {main}")

    conn = _connect()
    avails = []
    for ln in lines:
        code = ln["component_code"]
        csku = component_sku(main, code)
        stock_row = conn.execute(
            "SELECT COALESCE(available_qty,0) FROM process_stock WHERE so_number=? AND sku=? AND process=?",
            (so, csku, from_process),
        ).fetchone()
        avail = int(stock_row[0]) if stock_row else 0
        avails.append(
            {
                "component_code": code,
                "component_name": ln.get("component_name") or code,
                "component_sku": csku,
                "qty_per_set": max(int(ln.get("qty_per_set") or 1), 1),
                "available_qty": avail,
            }
        )
    conn.close()
    result = compute_complete_sets(avails)
    result["so_number"] = so
    result["main_sku"] = main
    result["from_process"] = from_process
    result["style_key"] = bom.get("style_key")
    return result


def commit_set_match(data: dict) -> dict:
    from ..services.set_components import component_sku

    so = str(data.get("so_number") or "").strip()
    main = str(data.get("main_sku") or "").strip().upper()
    from_process = str(data.get("from_process") or "Finishing").strip() or "Finishing"
    to_process = str(data.get("to_process") or "Packing").strip() or "Packing"
    preview = preview_set_match(so, main, from_process)
    complete = int(preview.get("complete_sets") or 0)
    requested = data.get("match_qty")
    match_qty = int(requested) if requested is not None else complete
    if match_qty <= 0:
        raise ValueError("match_qty must be greater than 0")
    if match_qty > complete:
        raise ValueError(
            f"Only {complete} complete set(s) available at {from_process} — cannot match {match_qty}"
        )

    conn = _connect()
    consumed = []
    for row in preview.get("components") or []:
        code = row["component_code"]
        ratio = max(int(row.get("qty_per_set") or 1), 1)
        need = match_qty * ratio
        csku = row.get("component_sku") or component_sku(main, code)
        stock_row = conn.execute(
            "SELECT COALESCE(available_qty,0) FROM process_stock WHERE so_number=? AND sku=? AND process=?",
            (so, csku, from_process),
        ).fetchone()
        avail = int(stock_row[0]) if stock_row else 0
        if need > avail:
            conn.close()
            raise ValueError(f"Insufficient {csku} at {from_process}: need {need}, have {avail}")
        _update_process_stock(conn, so, csku, from_process, qty_out=need)
        consumed.append(
            {
                "component_code": code,
                "component_sku": csku,
                "qty_per_set": ratio,
                "consumed_qty": need,
                "remaining_qty": avail - need,
            }
        )

    _update_process_stock(conn, so, main, to_process, qty_in=match_qty)
    conn.execute(
        """INSERT INTO set_match_events
           (so_number, main_sku, from_process, to_process, match_qty, components_json, matched_by, remarks)
           VALUES(?,?,?,?,?,?,?,?)""",
        (
            so,
            main,
            from_process,
            to_process,
            match_qty,
            json.dumps(consumed),
            str(data.get("matched_by") or ""),
            str(data.get("remarks") or ""),
        ),
    )
    conn.commit()
    conn.close()
    extras = [
        {
            "component_code": c["component_code"],
            "component_sku": c["component_sku"],
            "extra_wip_qty": int(c["remaining_qty"]),
        }
        for c in consumed
        if int(c["remaining_qty"]) > 0
    ]
    return {
        "ok": True,
        "so_number": so,
        "main_sku": main,
        "match_qty": match_qty,
        "from_process": from_process,
        "to_process": to_process,
        "components": consumed,
        "extra_wip": extras,
        "message": f"Matched {match_qty} complete set(s) → {main} at {to_process}",
    }


def list_set_split_events(so_number: str = "", main_sku: str = "") -> list:
    conn = _connect()
    q = "SELECT * FROM set_split_events WHERE 1=1"
    params: list = []
    if so_number:
        q += " AND so_number=?"
        params.append(so_number.strip())
    if main_sku:
        q += " AND main_sku=?"
        params.append(main_sku.strip().upper())
    q += " ORDER BY id DESC LIMIT 200"
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()
    for r in rows:
        try:
            r["components"] = json.loads(r.get("components_json") or "[]")
        except Exception:
            r["components"] = []
    return rows


def list_set_match_events(so_number: str = "", main_sku: str = "") -> list:
    conn = _connect()
    q = "SELECT * FROM set_match_events WHERE 1=1"
    params: list = []
    if so_number:
        q += " AND so_number=?"
        params.append(so_number.strip())
    if main_sku:
        q += " AND main_sku=?"
        params.append(main_sku.strip().upper())
    q += " ORDER BY id DESC LIMIT 200"
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()
    for r in rows:
        try:
            r["components"] = json.loads(r.get("components_json") or "[]")
        except Exception:
            r["components"] = []
    return rows
