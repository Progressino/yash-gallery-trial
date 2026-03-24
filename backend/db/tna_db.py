"""TNA (Time & Action Calendar) DB"""
import sqlite3, os
from datetime import datetime, timedelta

_DB = os.environ.get("TNA_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "tna.db"))

def _connect():
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

# Built-in TNA templates (activity, group, lead_days_before_delivery)
TEMPLATES = {
    "Domestic Order TNA": [
        ("BOM Final",               "Sampling",       44),
        ("Accessories Final",        "Purchase",       37),
        ("Grey Fabric Booking",      "Fabric",         35),
        ("Accessories Purchase",     "Purchase",       34),
        ("Grey Fabric Inward",       "Fabric",         28),
        ("Accessories Inward",       "Purchase",       26),
        ("PP Sample",                "Sampling",       25),
        ("Sample Approval",          "Sampling",       22),
        ("Artwork Release",          "CAD",            20),
        ("Cutting",                  "Cutting",         7),
        ("Stitching",                "Stitching",       4),
        ("Finishing",                "Finishing",       2),
        ("Packing",                  "Packing",         1),
        ("Dispatch",                 "Dispatch",        0),
    ],
    "Export Order TNA": [
        ("BOM Final",               "Sampling",       80),
        ("Tech Pack",               "Sampling",       75),
        ("Accessories Final",        "Purchase",       70),
        ("Grey Fabric Booking",      "Fabric",         65),
        ("Accessories Purchase",     "Purchase",       60),
        ("Grey Fabric Inward",       "Fabric",         50),
        ("Accessories Inward",       "Purchase",       48),
        ("PP Sample",               "Sampling",       45),
        ("Sample Approval",         "Sampling",       40),
        ("Artwork Release",         "CAD",            38),
        ("Counter Sample",          "Sampling",       35),
        ("Counter Sample Approval", "Sampling",       30),
        ("Bulk Fabric Booking",     "Fabric",         25),
        ("Cutting",                 "Cutting",        14),
        ("Stitching",               "Stitching",       8),
        ("Finishing",               "Finishing",       4),
        ("Packing",                 "Packing",         2),
        ("Shipment / Dispatch",     "Dispatch",        0),
    ],
    "Printed Fabric TNA": [
        ("Design Finalisation",     "CAD",            60),
        ("Grey Fabric Booking",     "Fabric",         55),
        ("Grey Fabric Inward",      "Fabric",         45),
        ("Dispatch to Printer",     "Fabric",         40),
        ("Printing",                "Printing",       25),
        ("Printed Fabric Receipt",  "Fabric",         20),
        ("QC Check",                "Quality",        18),
        ("Accessories Final",       "Purchase",       15),
        ("Accessories Inward",      "Purchase",       13),
        ("PP Sample",               "Sampling",       12),
        ("Sample Approval",         "Sampling",       10),
        ("Cutting",                 "Cutting",         7),
        ("Stitching",               "Stitching",       4),
        ("Finishing",               "Finishing",       2),
        ("Dispatch",                "Dispatch",        0),
    ],
}

def init_db():
    conn = _connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS tna_list (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        tna_number    TEXT UNIQUE NOT NULL,
        so_number     TEXT,
        style_name    TEXT,
        buyer         TEXT,
        po_number     TEXT,
        merchandiser  TEXT,
        season        TEXT,
        order_qty     INTEGER DEFAULT 0,
        delivery_date TEXT,
        exfactory_date TEXT,
        shipment_date  TEXT,
        priority      TEXT DEFAULT 'Normal',
        status        TEXT DEFAULT 'Active',
        template_used TEXT,
        created_at    TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS tna_lines (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        tna_id           INTEGER NOT NULL REFERENCES tna_list(id) ON DELETE CASCADE,
        sr               INTEGER,
        activity         TEXT NOT NULL,
        activity_group   TEXT,
        planned_start    TEXT,
        planned_end      TEXT,
        actual_start     TEXT,
        actual_end       TEXT,
        status           TEXT DEFAULT 'Not Started',
        responsible      TEXT,
        backup_person    TEXT,
        delay_days       INTEGER DEFAULT 0,
        delay_reason     TEXT,
        remarks          TEXT
    );
    """)
    conn.commit(); conn.close()

def _next_num(conn):
    row = conn.execute("SELECT tna_number FROM tna_list ORDER BY id DESC LIMIT 1").fetchone()
    n = 1
    if row:
        try: n = int(row[0].split('-')[-1]) + 1
        except: pass
    return f"TNA-{n:04d}"

def _add_days(date_str: str, days: int) -> str:
    try:
        d = datetime.strptime(date_str, '%Y-%m-%d')
        return (d + timedelta(days=days)).strftime('%Y-%m-%d')
    except:
        return date_str

def list_tnas(status=None):
    conn = _connect()
    q = "SELECT * FROM tna_list WHERE status=? ORDER BY id DESC" if status else "SELECT * FROM tna_list ORDER BY id DESC"
    rows = conn.execute(q, (status,) if status else ()).fetchall()
    result = []
    for r in rows:
        t = dict(r)
        lines = conn.execute("SELECT * FROM tna_lines WHERE tna_id=? ORDER BY sr", (t['id'],)).fetchall()
        t['lines'] = [dict(l) for l in lines]
        # calc delays
        today = datetime.now().strftime('%Y-%m-%d')
        t['delayed_count'] = sum(1 for l in t['lines'] if l['status'] not in ('Completed','Cancelled') and l['planned_end'] and l['planned_end'] < today)
        t['completed_count'] = sum(1 for l in t['lines'] if l['status'] == 'Completed')
        t['total_activities'] = len(t['lines'])
        result.append(t)
    conn.close(); return result

def get_tna(tid: int):
    conn = _connect()
    row = conn.execute("SELECT * FROM tna_list WHERE id=?", (tid,)).fetchone()
    if not row: return None
    t = dict(row)
    t['lines'] = [dict(l) for l in conn.execute("SELECT * FROM tna_lines WHERE tna_id=? ORDER BY sr", (tid,)).fetchall()]
    conn.close(); return t

def create_tna(data: dict):
    conn = _connect()
    num = _next_num(conn)
    delivery = data.get('delivery_date') or ''
    template = data.get('template_used') or 'Domestic Order TNA'
    conn.execute("""INSERT INTO tna_list(tna_number,so_number,style_name,buyer,po_number,merchandiser,season,
        order_qty,delivery_date,exfactory_date,shipment_date,priority,status,template_used)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (num, data.get('so_number') or '', data.get('style_name') or '',
         data.get('buyer') or '', data.get('po_number') or '',
         data.get('merchandiser') or '', data.get('season') or '',
         data.get('order_qty') or 0, delivery,
         data.get('exfactory_date') or '', data.get('shipment_date') or '',
         data.get('priority') or 'Normal', 'Active', template))
    tid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    # Generate lines from template
    tmpl_activities = data.get('custom_lines') or TEMPLATES.get(template, TEMPLATES["Domestic Order TNA"])
    for i, act in enumerate(tmpl_activities):
        if isinstance(act, dict):
            activity, group, lead_days = act.get('activity',''), act.get('group',''), act.get('lead_days',0)
        else:
            activity, group, lead_days = act[0], act[1], act[2]
        planned_end = _add_days(delivery, -lead_days) if delivery else ''
        planned_start = _add_days(planned_end, -2) if planned_end else ''
        conn.execute("""INSERT INTO tna_lines(tna_id,sr,activity,activity_group,planned_start,planned_end,status)
            VALUES(?,?,?,?,?,?,?)""",
            (tid, i+1, activity, group, planned_start, planned_end, 'Not Started'))
    conn.commit(); conn.close(); return num

def update_tna_line(lid: int, data: dict):
    allowed = ['actual_start','actual_end','status','responsible','backup_person','delay_reason','remarks']
    sets = ', '.join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed] + [lid]
    if not sets: return
    conn = _connect(); conn.execute(f"UPDATE tna_lines SET {sets} WHERE id=?", vals)
    # recalculate delay_days
    row = conn.execute("SELECT planned_end,actual_end FROM tna_lines WHERE id=?", (lid,)).fetchone()
    if row and row['planned_end'] and row['actual_end']:
        try:
            p = datetime.strptime(row['planned_end'], '%Y-%m-%d')
            a = datetime.strptime(row['actual_end'], '%Y-%m-%d')
            delay = max(0, (a - p).days)
            conn.execute("UPDATE tna_lines SET delay_days=? WHERE id=?", (delay, lid))
        except: pass
    conn.commit(); conn.close()

def update_tna_status(tid: int, status: str):
    conn = _connect(); conn.execute("UPDATE tna_list SET status=? WHERE id=?", (status, tid))
    conn.commit(); conn.close()

def get_tna_stats():
    conn = _connect()
    today = datetime.now().strftime('%Y-%m-%d')
    stats = {
        'total': conn.execute("SELECT COUNT(*) FROM tna_list").fetchone()[0],
        'active': conn.execute("SELECT COUNT(*) FROM tna_list WHERE status='Active'").fetchone()[0],
        'delayed_activities': conn.execute("""
            SELECT COUNT(*) FROM tna_lines l JOIN tna_list t ON t.id=l.tna_id
            WHERE t.status='Active' AND l.status NOT IN ('Completed','Cancelled')
            AND l.planned_end < ?""", (today,)).fetchone()[0],
        'completed': conn.execute("SELECT COUNT(*) FROM tna_list WHERE status='Completed'").fetchone()[0],
    }
    conn.close(); return stats
