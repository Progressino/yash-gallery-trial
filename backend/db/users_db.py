"""Admin / Users / Roles DB"""
import sqlite3, os, bcrypt
from datetime import datetime

_DB = os.environ.get("USERS_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "users.db"))

def _connect():
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

DEFAULT_ROLES = [
    ('Super Admin', 'Full system access — OTP on new devices; manages all users'),
    ('Admin',     'Full access to all modules'),
    ('Sir',       'Management — full ERP access'),
    ('HOD',       'Head of department — HRM module, team data only'),
    ('Employee',  'HRM module — own tasks and performance only'),
    ('Manager',   'Create/edit/approve in assigned departments'),
    ('Executive', 'Create documents; view reports'),
    ('Clerk',     'Data entry only; limited view'),
    ('Viewer',    'Read-only access to reports'),
    ('Karigar',   'Stitching production entry only (mobile floor users)'),
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
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS login_otps (
        challenge_id  TEXT PRIMARY KEY,
        user_id       INTEGER,
        username      TEXT NOT NULL,
        phone         TEXT NOT NULL,
        code_hash     TEXT NOT NULL,
        expires_at    TEXT NOT NULL,
        verified      INTEGER DEFAULT 0,
        attempts      INTEGER DEFAULT 0,
        created_at    TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS trusted_devices (
        device_id     TEXT PRIMARY KEY,
        user_id       INTEGER,
        username      TEXT NOT NULL,
        fingerprint   TEXT NOT NULL,
        user_agent    TEXT,
        expires_at    TEXT NOT NULL,
        created_at    TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_trusted_devices_user ON trusted_devices(username, fingerprint);
    """)
    for table, col, decl in [
        ("erp_users", "karigar_id", "TEXT DEFAULT ''"),
        ("erp_users", "employee_id", "INTEGER"),
        ("erp_users", "hrm_department_id", "INTEGER"),
        ("erp_users", "reporting_hod_user_id", "INTEGER"),
        ("erp_users", "module_access", "TEXT"),
        ("erp_users", "phone", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        except Exception:
            pass
    # Empty string counted as a value under UNIQUE(email); only one '' was allowed.
    # Normalize to NULL so many users can omit email.
    try:
        conn.execute(
            "UPDATE erp_users SET email = NULL "
            "WHERE email IS NOT NULL AND trim(email) = ''"
        )
    except Exception:
        pass
    ensure_super_admin(conn)
    ensure_user_has_modules(conn, "harsh", ["hrm"])
    conn.commit(); conn.close()

# ── Super Admin bootstrap ─────────────────────────────────────────────────────
def ensure_super_admin(conn=None):
    """Ensure env admin exists in erp_users as Super Admin with phone for OTP."""
    from ..services.login_otp import normalize_india_phone

    close = False
    if conn is None:
        conn = _connect()
        close = True
    row = conn.execute("SELECT id FROM roles WHERE role_name='Super Admin'").fetchone()
    if not row:
        conn.execute(
            "INSERT OR IGNORE INTO roles(role_name,description) VALUES(?,?)",
            ("Super Admin", "Full system access — OTP on new devices; manages all users"),
        )
        row = conn.execute("SELECT id FROM roles WHERE role_name='Super Admin'").fetchone()
    if not row:
        if close:
            conn.commit()
            conn.close()
        return
    role_id = row["id"]
    username = (os.environ.get("SUPER_ADMIN_USERNAME") or os.environ.get("AUTH_USERNAME") or "admin").strip()
    phone_raw = os.environ.get("SUPER_ADMIN_PHONE", "").strip()
    phone = normalize_india_phone(phone_raw) if phone_raw else None
    pw_hash = os.environ.get("AUTH_PASSWORD_HASH", "").strip()
    if not username:
        if close:
            conn.commit()
            conn.close()
        return
    existing = conn.execute("SELECT id, phone FROM erp_users WHERE username=?", (username,)).fetchone()
    if existing:
        sets = ["role_id=?", "active=1", "updated_at=?"]
        vals = [role_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        if phone and not (existing["phone"] or "").strip():
            sets.append("phone=?")
            vals.append(phone)
        vals.append(existing["id"])
        conn.execute(f"UPDATE erp_users SET {', '.join(sets)} WHERE id=?", vals)
    elif pw_hash:
        conn.execute(
            """INSERT INTO erp_users(username,email,password_hash,full_name,role_id,department,phone,active)
               VALUES(?,?,?,?,?,?,?,1)""",
            (
                username,
                None,
                pw_hash,
                "Super Administrator",
                role_id,
                "Admin",
                phone,
            ),
        )
        if close:
            conn.commit()
            conn.close()


def ensure_user_has_modules(conn, username: str, modules: list[str]) -> None:
    """Merge module keys into a user's module_access (e.g. grant HRM to an existing ERP user)."""
    import json

    from ..services.rbac import resolve_module_access

    uname = (username or "").strip()
    if not uname or not modules:
        return
    row = conn.execute(
        """SELECT u.id, u.module_access, r.role_name
           FROM erp_users u LEFT JOIN roles r ON r.id = u.role_id
           WHERE lower(u.username) = lower(?) AND u.active = 1""",
        (uname,),
    ).fetchone()
    if not row:
        return
    current = resolve_module_access(row["role_name"] or "", row["module_access"])
    if all(m in current for m in modules):
        return
    merged = sorted(set(current) | set(modules))
    conn.execute(
        "UPDATE erp_users SET module_access=?, updated_at=? WHERE id=?",
        (json.dumps(merged), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row["id"]),
    )


def get_role_id(role_name: str) -> int | None:
    conn = _connect()
    row = conn.execute("SELECT id FROM roles WHERE role_name=?", (role_name,)).fetchone()
    conn.close()
    return int(row["id"]) if row else None

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

def _normalize_email(value) -> str | None:
    """Blank / whitespace-only email → NULL in DB (avoids UNIQUE collisions on '')."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def create_user(data: dict):
    conn = _connect()
    username = (data.get("username") or "").strip()
    if not username:
        conn.close()
        raise ValueError("username is required")
    pw = data.get('password', 'changeme123')
    hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    email = _normalize_email(data.get("email"))
    from ..services.login_otp import normalize_india_phone
    phone = normalize_india_phone(data.get("phone") or "") if data.get("phone") else None
    conn.execute("""INSERT INTO erp_users(
        username,email,password_hash,full_name,role_id,department,karigar_id,
        employee_id,hrm_department_id,reporting_hod_user_id,module_access,phone,active)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (username, email, hashed,
         data.get('full_name',''), data.get('role_id'), data.get('department',''),
         data.get('karigar_id', '') or '',
         data.get('employee_id'), data.get('hrm_department_id'),
         data.get('reporting_hod_user_id'), data.get('module_access'), phone,
         1))
    conn.commit(); conn.close()

def update_user(uid: int, data: dict):
    if "email" in data:
        data = {**data, "email": _normalize_email(data.get("email"))}
    allowed = [
        'email', 'full_name', 'role_id', 'department', 'active', 'karigar_id',
        'employee_id', 'hrm_department_id', 'reporting_hod_user_id', 'module_access', 'phone',
    ]
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
    row = conn.execute(
        """SELECT u.*, r.role_name FROM erp_users u
           LEFT JOIN roles r ON r.id = u.role_id
           WHERE u.username=? AND u.active=1""",
        (username,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    if bcrypt.checkpw(password.encode(), row['password_hash'].encode()):
        d = dict(row)
        d.pop('password_hash', None)
        return d
    return None


def get_user_auth_profile(username: str) -> dict | None:
    """Load user + role for /auth/me (no password)."""
    conn = _connect()
    row = conn.execute(
        """SELECT u.id, u.username, u.email, u.full_name, u.role_id, u.department,
                  u.karigar_id, u.employee_id, u.hrm_department_id, u.reporting_hod_user_id,
                  u.module_access, u.phone, u.active, r.role_name
           FROM erp_users u
           LEFT JOIN roles r ON r.id = u.role_id
           WHERE u.username=? AND u.active=1""",
        (username,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None

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


# ── OTP challenges ─────────────────────────────────────────────────────────────
def create_otp_challenge(*, challenge_id: str, user_id: int | None, username: str, phone: str,
                         code_hash: str, expires_at: str) -> None:
    conn = _connect()
    conn.execute(
        """INSERT OR REPLACE INTO login_otps
           (challenge_id, user_id, username, phone, code_hash, expires_at, verified, attempts, created_at)
           VALUES (?,?,?,?,?,?,0,0,datetime('now'))""",
        (challenge_id, user_id, username, phone, code_hash, expires_at),
    )
    conn.commit()
    conn.close()


def get_otp_challenge(challenge_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM login_otps WHERE challenge_id=?", (challenge_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def increment_otp_attempts(challenge_id: str) -> None:
    conn = _connect()
    conn.execute(
        "UPDATE login_otps SET attempts = attempts + 1 WHERE challenge_id=?",
        (challenge_id,),
    )
    conn.commit()
    conn.close()


def mark_otp_verified(challenge_id: str) -> None:
    conn = _connect()
    conn.execute("UPDATE login_otps SET verified=1 WHERE challenge_id=?", (challenge_id,))
    conn.commit()
    conn.close()


# ── Trusted devices ────────────────────────────────────────────────────────────
def register_trusted_device(*, device_id: str, user_id: int | None, username: str,
                            fingerprint: str, user_agent: str, trust_days: int) -> None:
    from datetime import timedelta, timezone
    exp = (datetime.now(tz=timezone.utc) + timedelta(days=trust_days)).strftime("%Y-%m-%d %H:%M:%S")
    conn = _connect()
    conn.execute(
        """INSERT OR REPLACE INTO trusted_devices
           (device_id, user_id, username, fingerprint, user_agent, expires_at, created_at)
           VALUES (?,?,?,?,?,?,datetime('now'))""",
        (device_id, user_id, username, fingerprint, user_agent, exp),
    )
    conn.commit()
    conn.close()


def trusted_device_exists(*, device_id: str, username: str) -> bool:
    conn = _connect()
    row = conn.execute(
        """SELECT 1 FROM trusted_devices
           WHERE device_id=? AND username=? AND datetime(expires_at) > datetime('now')""",
        (device_id, username),
    ).fetchone()
    conn.close()
    return row is not None
