"""Session RAM cap — evict LRU sessions when over SESSION_MAX_ACTIVE."""
from __future__ import annotations

from backend.session import AppSession, SessionStore


def test_enforce_active_cap_evicts_oldest(monkeypatch):
    monkeypatch.setenv("SESSION_MAX_ACTIVE", "3")
    st = SessionStore()
    ids = []
    for i in range(5):
        sid, sess = st.get_or_empty(None)
        ids.append(sid)
        sess.last_accessed = float(i)
    assert st.count == 3
    assert ids[0] not in st._sessions
    assert ids[1] not in st._sessions
    assert ids[2] in st._sessions
    assert ids[3] in st._sessions
    assert ids[4] in st._sessions


def test_enforce_active_cap_protects_current_session(monkeypatch):
    monkeypatch.setenv("SESSION_MAX_ACTIVE", "2")
    st = SessionStore()
    a, sa = st.get_or_empty(None)
    b, sb = st.get_or_empty(None)
    sa.last_accessed = 1.0
    sb.last_accessed = 2.0
    c, _ = st.get_or_empty(None)
    st._sessions[a].last_accessed = 0.0
    st._enforce_active_cap(protect_sid=c)
    assert st.count == 2
    assert c in st._sessions
    assert b in st._sessions
    assert a not in st._sessions
