"""Checkout-scoped SQLite state (ARCH §15, §17, §21).

Ordinals are allocated here and only here: git refs are not authoritative
sequence state, because they can be pruned, partially cleaned or created
concurrently. Every write is a short transaction — no connection is held
across a git call or a subprocess.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from . import CONTRACT_VERSION, SCHEMA_VERSION, artifacts, deadline, paths

BUSY_TIMEOUT_MS = 5000
_RETRIES = 6


class StateCompatibilityError(sqlite3.DatabaseError):
    """The state database belongs to unsupported semantics or another checkout."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    host            TEXT NOT NULL,
    host_session_id TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    contract_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_events (
    session_id  TEXT NOT NULL,
    turn_key    TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    at          TEXT NOT NULL,
    tool        TEXT NOT NULL,
    detail      TEXT,
    tool_use_id TEXT,
    outcome     TEXT,
    PRIMARY KEY (session_id, turn_key, seq)
);
CREATE INDEX IF NOT EXISTS tool_events_use_id
    ON tool_events (session_id, tool_use_id);
CREATE TABLE IF NOT EXISTS refusals (
    at     TEXT NOT NULL,
    kind   TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ordinals (
    session_id TEXT NOT NULL,
    scope      TEXT NOT NULL,
    class      TEXT NOT NULL,
    next       INTEGER NOT NULL,
    PRIMARY KEY (session_id, scope, class)
);
"""

_HAS_RETURNING = sqlite3.sqlite_version_info >= (3, 35)


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    wait_s = deadline.remaining(BUSY_TIMEOUT_MS / 1000)
    conn = sqlite3.connect(str(db_path), timeout=wait_s, isolation_level=None)
    try:
        conn.execute(f"PRAGMA busy_timeout = {max(0, int(wait_s * 1000))}")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        yield conn
    finally:
        conn.close()


def retrying(fn, *args, **kw):
    """SQLITE_BUSY survives the pragma timeout under heavy concurrency."""
    delay = 0.02
    for attempt in range(_RETRIES):
        if deadline.expired():
            raise sqlite3.OperationalError("capture deadline exceeded while waiting for state")
        try:
            return fn(*args, **kw)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) and "busy" not in str(exc):
                raise
            if attempt == _RETRIES - 1:
                raise
            remaining = deadline.remaining(delay)
            if remaining <= 0:
                raise sqlite3.OperationalError(
                    "capture deadline exceeded while waiting for state") from exc
            time.sleep(remaining)
            delay *= 2


def validate_metadata(conn: sqlite3.Connection, checkout_id: str) -> None:
    """Fail closed before interpreting or changing an existing state database."""
    try:
        values = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.DatabaseError as exc:
        raise StateCompatibilityError(f"state metadata is unreadable: {exc}") from exc
    expected_schema = str(SCHEMA_VERSION)
    if values.get("schema_version") != expected_schema:
        raise StateCompatibilityError(
            f"state schema_version {values.get('schema_version')!r} is unsupported; "
            f"expected {expected_schema!r}")
    contract = values.get("contract_version")
    if contract not in artifacts.SUPPORTED_CONTRACTS:
        raise StateCompatibilityError(
            f"state contract_version {contract!r} is unsupported")
    if values.get("checkout_id") != checkout_id:
        raise StateCompatibilityError(
            f"state checkout_id {values.get('checkout_id')!r} does not match "
            f"{checkout_id!r}")


def validate_sessions(conn: sqlite3.Connection) -> None:
    """Validate semantic identity rows, including rows written by older Core."""
    try:
        rows = conn.execute(
            "SELECT session_id, host, host_session_id, contract_version FROM sessions"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise StateCompatibilityError(f"session index is unreadable: {exc}") from exc
    for session_id, host, host_session_id, contract in rows:
        if not all(isinstance(value, str) and value for value in
                   (session_id, host, host_session_id)):
            raise StateCompatibilityError(
                f"session index has an invalid identity row for {session_id!r}")
        if contract not in artifacts.SUPPORTED_CONTRACTS:
            raise StateCompatibilityError(
                f"session index {session_id!r} has unsupported contract_version "
                f"{contract!r}")


def validate_state(conn: sqlite3.Connection, checkout_id: str) -> None:
    """Read-only health check for every persisted coordination structure."""
    validate_metadata(conn, checkout_id)
    try:
        integrity = conn.execute("PRAGMA quick_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise StateCompatibilityError(
                f"state integrity check failed: {integrity[0] if integrity else 'no result'}")
        required = {"meta", "sessions", "tool_events", "refusals", "ordinals"}
        present = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        missing = sorted(required - present)
        if missing:
            raise StateCompatibilityError(
                "state database is missing table(s): " + ", ".join(missing))
        validate_sessions(conn)
        for detail, outcome in conn.execute("SELECT detail, outcome FROM tool_events"):
            _loads(detail)
            _loads(outcome)
    except StateCompatibilityError:
        raise
    except sqlite3.DatabaseError as exc:
        raise StateCompatibilityError(f"state contents are unreadable: {exc}") from exc


def _prepare(conn: sqlite3.Connection, checkout_id: str) -> None:
    """Create new state or migrate known state only after validating its identity."""
    # The metadata rows and their table must become visible together. Otherwise
    # a concurrent first hook can observe the table between CREATE and INSERT,
    # mistake an in-progress initialization for corruption, and lose its event.
    conn.execute("BEGIN IMMEDIATE")
    try:
        has_meta = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
        if has_meta:
            validate_metadata(conn, checkout_id)
        for statement in SCHEMA.split(";"):
            if statement.strip():
                conn.execute(statement)
        conn.executemany("INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
                         [("schema_version", str(SCHEMA_VERSION)),
                          ("contract_version", CONTRACT_VERSION),
                          ("checkout_id", checkout_id)])
        validate_metadata(conn, checkout_id)
        validate_sessions(conn)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


@contextmanager
def checked(checkout_id: str, db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """Open existing state only after its schema and checkout identity agree."""
    with connect(db_path or paths.state_db(checkout_id)) as conn:
        validate_metadata(conn, checkout_id)
        yield conn


def ensure_schema(checkout_id: str) -> None:
    """Idempotent. A checkout created by an older Core lacks newer tables and
    columns, and a reader must neither crash on one nor silently ignore it."""
    db = paths.state_db(checkout_id)

    def _ensure():
        with connect(db) as conn:
            _prepare(conn, checkout_id)

    retrying(_ensure)


def initialize(checkout_id: str) -> Path:
    db = paths.state_db(checkout_id)

    def _init():
        with connect(db) as conn:
            _prepare(conn, checkout_id)
        return db

    return retrying(_init)


def ensure_session(checkout_id: str, session_id: str, host: str,
                   host_session_id: str, started_at: str) -> None:
    """Schema and registration in one connection. Idempotent, because concurrent
    hooks on one event all reach it (§14)."""
    db = paths.state_db(checkout_id)

    def _ensure():
        with connect(db) as conn:
            _prepare(conn, checkout_id)
            conn.execute(
                "INSERT OR IGNORE INTO sessions"
                "(session_id, host, host_session_id, started_at, contract_version)"
                " VALUES(?,?,?,?,?)",
                (session_id, host, host_session_id, started_at, CONTRACT_VERSION))
            stored = conn.execute(
                "SELECT host, host_session_id, contract_version FROM sessions "
                "WHERE session_id=?", (session_id,)).fetchone()
            identity = stored[:2] if stored else None
            stored_contract = stored[2] if stored else None
            if (identity != (host, host_session_id)
                    or stored_contract not in artifacts.SUPPORTED_CONTRACTS):
                raise StateCompatibilityError(
                    f"session index {session_id!r} claims identity {stored!r}; "
                    f"expected host identity {(host, host_session_id)!r} and a "
                    "supported contract")

    retrying(_ensure)


MAX_REFUSALS = 500
_FALLBACK_REFUSALS = "capture-refusals.jsonl"
_MAX_FALLBACK_BYTES = 512 * 1024


def _fallback_path(checkout_id: str) -> Path:
    return paths.checkout_dir(checkout_id) / _FALLBACK_REFUSALS


def _append_fallback_refusal(checkout_id: str, row: dict) -> bool:
    """Last-resort append when SQLite itself is the failed capture component."""
    target = _fallback_path(checkout_id)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if target.stat().st_size >= _MAX_FALLBACK_BYTES:
                return False
        except FileNotFoundError:
            pass
        payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                   ).encode("utf-8")
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False


def _fallback_refusals(checkout_id: str) -> List[dict]:
    target = _fallback_path(checkout_id)
    try:
        lines = target.read_bytes().splitlines()[-MAX_REFUSALS:]
    except FileNotFoundError:
        return []
    except OSError as exc:
        return [{"at": "", "kind": "refusal-log",
                 "reason": f"cannot read fallback refusal log: {exc}"}]
    rows = []
    for payload in lines:
        try:
            line = payload.decode("utf-8")
            row = json.loads(line)
            if (isinstance(row, dict)
                    and all(isinstance(row.get(name), str)
                            for name in ("at", "kind", "reason"))):
                rows.append(row)
            else:
                rows.append({"at": "", "kind": "refusal-log",
                             "reason": "malformed fallback refusal entry"})
        except (UnicodeError, ValueError):
            rows.append({"at": "", "kind": "refusal-log",
                         "reason": "malformed fallback refusal entry"})
    return rows


def record_refusal(checkout_id: str, kind: str, reason: str, at: str) -> bool:
    """An event refused before any session is known has no record to be written
    into, so it is counted here. Otherwise a dropped event looks like a quiet
    checkout rather than lost evidence."""
    try:
        ensure_schema(checkout_id)

        def _write():
            with checked(checkout_id) as conn:
                conn.execute("INSERT INTO refusals(at, kind, reason) VALUES(?,?,?)",
                             (at, kind, str(reason)[:300]))
                conn.execute("DELETE FROM refusals WHERE rowid NOT IN "
                             "(SELECT rowid FROM refusals ORDER BY rowid DESC LIMIT ?)",
                             (MAX_REFUSALS,))

        retrying(_write)
        return True
    except (sqlite3.DatabaseError, OSError):
        # §9.2: capture degrades, it never disrupts the session. SQLite failure
        # is itself evidence loss, so preserve it in a bounded append-only file.
        return _append_fallback_refusal(
            checkout_id, {"at": at, "kind": kind, "reason": str(reason)[:300]})


def refusals(checkout_id: str) -> List[dict]:
    db = paths.state_db(checkout_id)
    fallback = _fallback_refusals(checkout_id)
    try:
        present = paths.present(db)
    except OSError as exc:
        return [{"at": "", "kind": "state", "reason": f"state cannot be read: {exc}"}
                ] + list(reversed(fallback))
    if not present:
        return list(reversed(fallback))

    def _read():
        with checked(checkout_id, db) as conn:
            rows = conn.execute("SELECT at, kind, reason FROM refusals "
                                "ORDER BY rowid DESC").fetchall()
        return [{"at": r[0], "kind": r[1], "reason": r[2]} for r in rows]

    try:
        rows = retrying(_read) + list(reversed(fallback))
        return sorted(rows, key=lambda row: row["at"], reverse=True)[:MAX_REFUSALS]
    except sqlite3.DatabaseError:
        return list(reversed(fallback))


def allocate_ordinal(checkout_id: str, session_id: str, scope: str, cls: str) -> int:
    """Atomic and monotonic. Never reused: a deleted ref does not roll this back."""
    db = paths.state_db(checkout_id)

    def _alloc():
        with checked(checkout_id, db) as conn:
            if _HAS_RETURNING:
                row = conn.execute(
                    "INSERT INTO ordinals(session_id, scope, class, next) VALUES(?,?,?,1)"
                    " ON CONFLICT(session_id, scope, class)"
                    " DO UPDATE SET next = next + 1 RETURNING next",
                    (session_id, scope, cls)).fetchone()
                return int(row[0])
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT next FROM ordinals WHERE session_id=? AND scope=? AND class=?",
                    (session_id, scope, cls)).fetchone()
                nxt = 1 if row is None else int(row[0]) + 1
                conn.execute(
                    "INSERT INTO ordinals(session_id, scope, class, next) VALUES(?,?,?,?)"
                    " ON CONFLICT(session_id, scope, class) DO UPDATE SET next=excluded.next",
                    (session_id, scope, cls, nxt))
                conn.execute("COMMIT")
                return nxt
            except Exception:
                conn.execute("ROLLBACK")
                raise

    return retrying(_alloc)


def append_tool_event(checkout_id: str, session_id: str, turn_key: str, at: str,
                      tool: str, detail: Optional[str], tool_use_id: Optional[str],
                      cap: int) -> Optional[int]:
    """One short transaction per tool call. PreToolUse fires constantly, so this
    never touches the session record — folding happens once at turn close.

    Returns None once the per-turn cap is reached: storage is bounded, so a
    runaway turn cannot grow the database without limit.
    """
    db = paths.state_db(checkout_id)

    def _append():
        with checked(checkout_id, db) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM tool_events"
                    " WHERE session_id=? AND turn_key=?", (session_id, turn_key)).fetchone()
                seq = int(row[0]) + 1
                if seq > cap:
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    "INSERT INTO tool_events"
                    "(session_id, turn_key, seq, at, tool, detail, tool_use_id, outcome)"
                    " VALUES(?,?,?,?,?,?,?,NULL)",
                    (session_id, turn_key, seq, at, tool, detail, tool_use_id))
                conn.execute("COMMIT")
                return seq
            except Exception:
                conn.execute("ROLLBACK")
                raise

    return retrying(_append)


def set_tool_outcome(checkout_id: str, session_id: str, tool_use_id: str,
                     outcome: str) -> bool:
    """PostToolUse completes the row PreToolUse opened. False when no row matched,
    which means the call was never recorded (over cap, or capture started late)."""
    db = paths.state_db(checkout_id)

    def _set():
        with checked(checkout_id, db) as conn:
            cur = conn.execute(
                "UPDATE tool_events SET outcome=? WHERE session_id=? AND tool_use_id=?",
                (outcome, session_id, tool_use_id))
            return cur.rowcount > 0

    return bool(retrying(_set))


def _loads(blob):
    """Decode staged JSON; corruption must become visible capture evidence."""
    if not blob:
        return None
    try:
        return json.loads(blob)
    except ValueError as exc:
        raise StateCompatibilityError(
            f"tool event contains malformed JSON: {exc}") from exc


def tool_events(checkout_id: str, session_id: str):
    """All tool rows for a session, grouped by turn key, in observed order."""
    db = paths.state_db(checkout_id)
    if not paths.present(db):
        return {}
    out: Dict[str, list] = {}
    with checked(checkout_id, db) as conn:
        for turn_key, seq, at, tool, detail, outcome in conn.execute(
                "SELECT turn_key, seq, at, tool, detail, outcome FROM tool_events"
                " WHERE session_id=? ORDER BY turn_key, seq", (session_id,)):
            out.setdefault(turn_key, []).append(
                {"seq": seq, "at": at, "tool": tool,
                 "detail": _loads(detail), "outcome": _loads(outcome)})
    return out


def clear_tool_events(checkout_id: str, session_id: str) -> int:
    """Staging only: once folded into the immutable record, the rows have served
    their purpose and SQLite goes back to being coordination state (ARCH §21.1)."""
    db = paths.state_db(checkout_id)
    if not paths.present(db):
        return 0

    def _clear():
        with checked(checkout_id, db) as conn:
            cur = conn.execute("DELETE FROM tool_events WHERE session_id=?", (session_id,))
            return cur.rowcount

    return int(retrying(_clear) or 0)


def ordinal_count(checkout_id: str) -> Optional[int]:
    """None when there is no state at all; used to spot refs outliving their state."""
    db = paths.state_db(checkout_id)
    try:
        if not paths.present(db):
            return None
        with checked(checkout_id, db) as conn:
            return int(conn.execute("SELECT count(*) FROM ordinals").fetchone()[0])
    except (sqlite3.Error, OSError):
        return None


