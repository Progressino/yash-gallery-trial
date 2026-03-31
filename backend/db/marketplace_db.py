"""
Marketplace credentials + sync log storage.
Credentials are encrypted at rest using Fernet symmetric encryption.
"""
import os
import sqlite3
from typing import Optional

# ── Encryption helpers ────────────────────────────────────────────────────────

def _get_fernet():
    """Return a Fernet instance using MARKETPLACE_ENCRYPTION_KEY env var."""
    from cryptography.fernet import Fernet
    key = os.environ.get("MARKETPLACE_ENCRYPTION_KEY", "").strip()
    if not key:
        # Auto-generate a temporary key (warns in logs — not suitable for production)
        import logging
        logging.getLogger("erp.marketplace").warning(
            "MARKETPLACE_ENCRYPTION_KEY not set — credentials will not survive restarts. "
            "Set this env var in .env to persist credentials."
        )
        key = Fernet.generate_key().decode()
    return Fernet(key.encode() if isinstance(key, str) else key)


def _encrypt(value: str) -> str:
    return _get_fernet().encrypt(value.encode()).decode()


def _decrypt(value: str) -> str:
    try:
        return _get_fernet().decrypt(value.encode()).decode()
    except Exception:
        return ""  # corrupted / wrong key → return empty


# ── DB path ───────────────────────────────────────────────────────────────────

def _db_path() -> str:
    return os.environ.get("MARKETPLACE_DB_PATH", "./marketplace_dev.db")


def _conn():
    return sqlite3.connect(_db_path(), check_same_thread=False)


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db():
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS marketplace_credentials (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                platform       TEXT NOT NULL UNIQUE,
                client_id      TEXT NOT NULL,
                client_secret  TEXT NOT NULL,
                refresh_token  TEXT NOT NULL,
                seller_id      TEXT NOT NULL,
                marketplace_id TEXT NOT NULL DEFAULT 'A21TJRUUN4KGV',
                created_at     TEXT DEFAULT (datetime('now')),
                updated_at     TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS marketplace_sync_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                platform    TEXT NOT NULL,
                synced_at   TEXT DEFAULT (datetime('now')),
                status      TEXT NOT NULL,
                rows_added  INTEGER DEFAULT 0,
                date_from   TEXT DEFAULT '',
                date_to     TEXT DEFAULT '',
                message     TEXT DEFAULT ''
            );
        """)


# ── Credentials CRUD ─────────────────────────────────────────────────────────

def save_credentials(
    platform: str,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    seller_id: str,
    marketplace_id: str = "A21TJRUUN4KGV",
) -> None:
    """Upsert credentials (client_secret + refresh_token encrypted)."""
    enc_secret = _encrypt(client_secret)
    enc_token  = _encrypt(refresh_token)
    with _conn() as con:
        con.execute("""
            INSERT INTO marketplace_credentials
                (platform, client_id, client_secret, refresh_token, seller_id, marketplace_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(platform) DO UPDATE SET
                client_id      = excluded.client_id,
                client_secret  = excluded.client_secret,
                refresh_token  = excluded.refresh_token,
                seller_id      = excluded.seller_id,
                marketplace_id = excluded.marketplace_id,
                updated_at     = datetime('now')
        """, (platform, client_id, enc_secret, enc_token, seller_id, marketplace_id))


def get_credentials(platform: str) -> Optional[dict]:
    """Return decrypted credentials dict, or None if not found."""
    with _conn() as con:
        row = con.execute(
            "SELECT client_id, client_secret, refresh_token, seller_id, marketplace_id, updated_at "
            "FROM marketplace_credentials WHERE platform = ?", (platform,)
        ).fetchone()
    if not row:
        return None
    client_id, enc_secret, enc_token, seller_id, marketplace_id, updated_at = row
    return {
        "platform":       platform,
        "client_id":      client_id,
        "client_secret":  _decrypt(enc_secret),
        "refresh_token":  _decrypt(enc_token),
        "seller_id":      seller_id,
        "marketplace_id": marketplace_id,
        "updated_at":     updated_at,
    }


def delete_credentials(platform: str) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM marketplace_credentials WHERE platform = ?", (platform,))
    return cur.rowcount > 0


def list_connected_platforms() -> list[dict]:
    """Return list of connected platform names (no secrets)."""
    with _conn() as con:
        rows = con.execute(
            "SELECT platform, seller_id, marketplace_id, updated_at FROM marketplace_credentials"
        ).fetchall()
    return [
        {"platform": r[0], "seller_id": r[1], "marketplace_id": r[2], "updated_at": r[3]}
        for r in rows
    ]


# ── Sync log CRUD ─────────────────────────────────────────────────────────────

def save_sync_log(
    platform: str,
    status: str,
    rows_added: int = 0,
    date_from: str = "",
    date_to: str = "",
    message: str = "",
) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO marketplace_sync_log (platform, status, rows_added, date_from, date_to, message) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (platform, status, rows_added, date_from, date_to, message),
        )


def list_sync_log(platform: str, limit: int = 20) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, platform, synced_at, status, rows_added, date_from, date_to, message "
            "FROM marketplace_sync_log WHERE platform = ? ORDER BY id DESC LIMIT ?",
            (platform, limit),
        ).fetchall()
    return [
        {
            "id": r[0], "platform": r[1], "synced_at": r[2], "status": r[3],
            "rows_added": r[4], "date_from": r[5], "date_to": r[6], "message": r[7],
        }
        for r in rows
    ]


def get_last_sync(platform: str) -> Optional[dict]:
    rows = list_sync_log(platform, limit=1)
    return rows[0] if rows else None
