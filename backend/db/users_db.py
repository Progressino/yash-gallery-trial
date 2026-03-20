"""Admin / Users / Roles DB"""
import sqlite3, os, bcrypt
from datetime import datetime

_DB = os.path.join(os.path.dirname(__file__), "..", "users.db")

def _connect():
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

DEFAULT_ROLES = [
    ('Admin',     'Full access to all modules'),
    ('Manager',   'Create/edit/approve in assigned departments'),
    ('Executive', 'Create documents; view reports'),
    ('Clerk',     'Data entry only; limited view'),
    ('Viewer',    'Read-only access to reports'),
]

DEPARTMENTS = ['Sales', 'Merchandising', 'Stores', 'Production', 'Quality', 'Logistics', 'Finance', 'Admin']

def init_db():
    conn = _connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS roles (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        role_name   TEXT UNIQUE NOT NULL,
        description TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS erp_users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT UNIQUE NOT NULL,
        email         TEXT UNIQUE,
        password_hash TEXT NOT NULL,
        full_name     TEXT,
        role_id       INTEGER REFERENCES roles(id),
        department    TEXT,
        active        INTEGER DEFAULT 1,
        created_at    TEXT DEFAULT (datetime('now')),
        updated_at    TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS activity_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       INTEGER REFERENCES erp_users(id),
        username      TEXT,
        action        TEXT,
        document_type TEXT,
        document_no   TEXT,
        details       TEXT,
        created_at    TEXT DEFAULT (datetime('now'))
    );
    """)
    # Seed default roles
    for name, desc in DEFAULT_ROLES:
        try:
            conn.execute("INSERT OR IGNORE INTO roles(role_name,description) VALUES(?,?)", (name, desc))
        except: pass
    conn.commit(); conn.close()

# ── Roles ──────────────────────────────────────────────────────────────────────
def list_roles():
    conn = _connect()
    rows = conn.execute("SELECT * FROM roles ORDER BY id").fetchall()
    conn.close(); return [dict(r) for r in rows]

def create_role(role_name: str, description: str = ''):
    conn = _connect()
    conn.execute("INSERT INTO roles(role_name,description) VALUES(?,?)", (role_name, description))
    conn.commit(); conn.close()

# ── Users ──────────────────────────────────────────────────────────────────────
def list_users(active_only=True):
    conn = _connect()
    q = """SELECT u.*, r.role_name FROM erp_users u
           LEFT JOIN roles r ON r.id=u.role_id
           WHERE u.active=1 ORDER BY u.id""" if active_only else \
        "SELECT u.*, r.role_name FROM erp_users u LEFT JOIN roles r ON r.id=u.role_id ORDER BY u.id"
    rows = conn.execute(q).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d.pop('password_hash', None)  # never return hash
        result.append(d)
    conn.close(); return result

def create_user(data: dict):
    conn = _connect()
    pw = data.get('password', 'changeme123')
    hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    conn.execute("""INSERT INTO erp_users(username,email,password_hash,full_name,role_id,department,active)
        VALUES(?,?,?,?,?,?,?)""",
        (data['username'], data.get('email',''), hashed,
         data.get('full_name',''), data.get('role_id'), data.get('department',''), 1))
    conn.commit(); conn.close()

def update_user(uid: int, data: dict):
    allowed = ['email','full_name','role_id','department','active']
    sets = ', '.join(f"{k}=?" for k in data if k in allowed)
    vals = [data[k] for k in data if k in allowed]
    if 'password' in data:
        hashed = bcrypt.hashpw(data['password'].encode(), bcrypt.gensalt()).decode()
        sets += ', password_hash=?, updated_at=?'
        vals += [hashed, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    elif sets:
        sets += ', updated_at=?'
        vals += [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    vals += [uid]
    if not sets: return
    conn = _connect(); conn.execute(f"UPDATE erp_users SET {sets} WHERE id=?", vals)
    conn.commit(); conn.close()

def deactivate_user(uid: int):
    conn = _connect(); conn.execute("UPDATE erp_users SET active=0 WHERE id=?", (uid,))
    conn.commit(); conn.close()

def verify_erp_user(username: str, password: str):
    conn = _connect()
    row = conn.execute("SELECT * FROM erp_users WHERE username=? AND active=1", (username,)).fetchone()
    conn.close()
    if not row: return None
    if bcrypt.checkpw(password.encode(), row['password_hash'].encode()):
        d = dict(row); d.pop('password_hash'); return d
    return None

# ── Activity Log ───────────────────────────────────────────────────────────────
def log_activity(username: str, action: str, doc_type: str, doc_no: str, details: str = ''):
    conn = _connect()
    conn.execute("""INSERT INTO activity_log(username,action,document_type,document_no,details)
        VALUES(?,?,?,?,?)""", (username, action, doc_type, doc_no, details))
    conn.commit(); conn.close()

def list_activity(limit=100):
    conn = _connect()
    rows = conn.execute("SELECT * FROM activity_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def get_admin_stats():
    conn = _connect()
    stats = {
        'total_users': conn.execute("SELECT COUNT(*) FROM erp_users WHERE active=1").fetchone()[0],
        'total_roles': conn.execute("SELECT COUNT(*) FROM roles").fetchone()[0],
        'recent_activity': conn.execute("SELECT COUNT(*) FROM activity_log WHERE created_at >= datetime('now','-1 day')").fetchone()[0],
        'by_role': [dict(r) for r in conn.execute("""
            SELECT r.role_name, COUNT(u.id) as cnt FROM roles r
            LEFT JOIN erp_users u ON u.role_id=r.id AND u.active=1
            GROUP BY r.id ORDER BY r.id""").fetchall()],
    }
    conn.close(); return stats
