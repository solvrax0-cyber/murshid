"""SQLite persistence: conversations (memory for follow-ups), question log (analytics),
and feedback. PII (emails, phones, ID numbers) is redacted before anything is stored."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import Counter

from . import config
from .arabic import normalize, redact_pii

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, ts REAL);
        CREATE INDEX IF NOT EXISTS ix_msg_session ON messages(session_id);
        CREATE TABLE IF NOT EXISTS logs(id INTEGER PRIMARY KEY, ts REAL, session_id TEXT, question TEXT,
            normalized TEXT, rewritten TEXT, intent TEXT, refused INTEGER, top_source TEXT,
            latency_ms INTEGER, feedback INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS users(student_id TEXT PRIMARY KEY, name TEXT, created_at REAL, last_login REAL);
        CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, user_id TEXT, title TEXT, created_at REAL, updated_at REAL);
        CREATE INDEX IF NOT EXISTS ix_conv_user ON conversations(user_id);
        CREATE TABLE IF NOT EXISTS llm_cache(key TEXT PRIMARY KEY, text TEXT, model TEXT, usage TEXT, ts REAL);
        """)
    _migrate()


def _migrate() -> None:
    """Safe, idempotent migration for databases created before conversations existed."""
    with _lock, _conn() as c:
        try:
            c.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists


def upsert_user(student_id: str, name: str) -> None:
    now = time.time()
    with _lock, _conn() as c:
        c.execute("INSERT INTO users(student_id, name, created_at, last_login) VALUES(?,?,?,?) "
                  "ON CONFLICT(student_id) DO UPDATE SET name=excluded.name, last_login=excluded.last_login",
                  (student_id, name, now, now))


def conversation_owner(conv_id: str) -> str | None:
    with _conn() as c:
        r = c.execute("SELECT user_id FROM conversations WHERE id=?", (conv_id,)).fetchone()
    return r["user_id"] if r else None


def touch_conversation(conv_id: str, user_id: str, first_question: str) -> None:
    now = time.time()
    with _lock, _conn() as c:
        c.execute("INSERT INTO conversations(id, user_id, title, created_at, updated_at) VALUES(?,?,?,?,?) "
                  "ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at",
                  (conv_id, user_id, redact_pii(first_question)[:40], now, now))


def list_conversations(user_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE user_id=? "
                         "ORDER BY updated_at DESC LIMIT 100", (user_id,)).fetchall()
    return [dict(r) for r in rows]


def delete_conversation(conv_id: str, user_id: str) -> bool:
    with _lock, _conn() as c:
        n = c.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (conv_id, user_id)).rowcount
        if n:
            c.execute("DELETE FROM messages WHERE session_id=?", (conv_id,))
    return bool(n)


def cache_get(key: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT text, model, usage FROM llm_cache WHERE key=?", (key,)).fetchone()
    return {"text": r["text"], "model": r["model"], "usage": json.loads(r["usage"] or "{}")} if r else None


def cache_put(key: str, text: str, model: str, usage: dict) -> None:
    with _lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO llm_cache(key, text, model, usage, ts) VALUES(?,?,?,?,?)",
                  (key, text, model, json.dumps(usage, ensure_ascii=False, default=str), time.time()))


def add_message(session_id: str, role: str, content: str) -> None:
    with _lock, _conn() as c:
        c.execute("INSERT INTO messages(session_id, conversation_id, role, content, ts) VALUES(?,?,?,?,?)",
                  (session_id, session_id, role, redact_pii(content), time.time()))


def history(session_id: str, turns: int = config.HISTORY_TURNS) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                         (session_id, turns * 2)).fetchall()
    return [dict(r) for r in reversed(rows)]


def log_question(session_id, question, rewritten, intent, refused, top_source, latency_ms) -> int:
    q = redact_pii(question)
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO logs(ts, session_id, question, normalized, rewritten, intent, refused, top_source, latency_ms)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (time.time(), session_id, q, normalize(q), redact_pii(rewritten or ""), intent, int(refused), top_source, latency_ms))
        return cur.lastrowid


def set_feedback(log_id: int, value: int) -> None:
    with _lock, _conn() as c:
        c.execute("UPDATE logs SET feedback=? WHERE id=?", (1 if value > 0 else -1, log_id))


def _count_users() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def stats(limit: int = 15) -> dict:
    with _conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 5000").fetchall()]
    academic = [r for r in rows if r["intent"] == "academic"]
    # group near-identical questions by their normalised standalone form
    key = lambda r: normalize(r["rewritten"] or r["question"])
    counts = Counter(key(r) for r in academic)
    example = {}
    for r in academic:
        example.setdefault(key(r), r["rewritten"] or r["question"])
    top = [{"question": example[k], "count": n} for k, n in counts.most_common(limit)]
    unanswered = Counter(key(r) for r in academic if r["refused"])
    gaps = [{"question": example[k], "count": n} for k, n in unanswered.most_common(limit)]
    sources = Counter(r["top_source"] for r in academic if r["top_source"] and not r["refused"])
    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
    return {
        "total_questions": len(rows),
        "registered_students": _count_users(),
        "academic_questions": len(academic),
        "sessions": len({r["session_id"] for r in rows}),
        "refusal_rate": round(sum(r["refused"] for r in academic) / len(academic), 3) if academic else 0,
        "thumbs_up": sum(r["feedback"] == 1 for r in rows),
        "thumbs_down": sum(r["feedback"] == -1 for r in rows),
        "p50_latency_ms": lat[len(lat) // 2] if lat else 0,
        "top_questions": top,
        "knowledge_gaps": gaps,
        "top_sources": [{"source": s, "count": n} for s, n in sources.most_common(8)],
    }
