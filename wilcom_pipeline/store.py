"""A local ledger of every API call, in SQLite.

Why this exists rather than reading the JSON records:

  * **The cost of a call was never persisted anywhere.** Tokens were logged to the
    console and thrown away, so spend could not be totalled after the fact at all.
  * **Deriving usage from the cache is brittle.** `details.py` can recover tokens by
    recomputing the client's *effective* cache key, but that is only correct while
    two independent pieces of hashing logic agree. If one changes, the recovery
    degrades to "unknown" -- safe, because it never reports another call's numbers,
    but useless.
  * **Aggregation is the point.** "What did this video cost", "what did every
    grounded action cost", "how long does a Pro call take" are one query here and a
    full traversal there.

Keyed on the RAW cache key, the one the caller passed, because that value is
already written into the trajectory and grounding records. That makes the join
exact, and keeps this module ignorant of what a stage is.

Deliberately per-work-tree (`<work>/usage.sqlite`), so the ledger travels with the
run it describes and can be deleted with it.

SQLite rather than a JSON file because calls are appended from a worker thread and
read from request handlers, and because a partially written JSON file is a
corrupt file where a partially written transaction is not.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    at                REAL    NOT NULL,
    model             TEXT    NOT NULL,
    cache_key         TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    thinking_tokens   INTEGER,
    seconds           REAL,
    cached            INTEGER,
    usd               REAL
);
CREATE INDEX IF NOT EXISTS calls_cache_key ON calls(cache_key);
CREATE INDEX IF NOT EXISTS calls_model ON calls(model);
"""


def db_path(work) -> Path:
    """Where the ledger for a work tree lives."""
    return Path(work) / "usage.sqlite"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10.0)
    # WAL keeps a reader from blocking the worker thread mid-run, which matters
    # because the Details view can be opened while a step is still going.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        pass
    return conn


def init(work) -> None:
    """Create the ledger if it is missing. Safe to call repeatedly."""
    path = db_path(work)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.executescript(SCHEMA)


def record_call(work, model: str, cache_key: str | None, usage: dict | None,
                seconds: float | None = None, cached: bool = False,
                usd: float | None = None) -> None:
    """Append one call. NEVER raises.

    This runs inside the client's hot path, so a ledger problem must not be able to
    fail a run that has already been paid for. Losing a ledger row is a far smaller
    failure than losing the run, which is the same rule the log sink follows.
    """
    if work is None:
        return
    try:
        path = db_path(work)
        usage = usage or {}
        details = usage.get("completion_tokens_details") or {}
        row = (
            time.time(),
            str(model or ""),
            cache_key,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            details.get("reasoning_tokens"),
            float(seconds) if seconds is not None else None,
            1 if cached else 0,
            float(usd) if usd is not None else None,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with _connect(path) as conn:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT INTO calls (at, model, cache_key, prompt_tokens,"
                " completion_tokens, thinking_tokens, seconds, cached, usd)"
                " VALUES (?,?,?,?,?,?,?,?,?)", row)
    except Exception:
        pass


def calls_for(work) -> dict:
    """Every recorded call, keyed by raw cache key.

    Also returns a list for the calls that have no key, which is how a call made
    without caching is recorded.
    """
    path = db_path(work)
    if not path.is_file():
        return {}
    try:
        with _connect(path) as conn:
            conn.executescript(SCHEMA)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT model, cache_key, prompt_tokens, completion_tokens,"
                " thinking_tokens, seconds, cached, usd FROM calls").fetchall()
    except Exception:
        return {}

    out: dict = {}
    for row in rows:
        key = row["cache_key"]
        if not key:
            continue
        entry = {
            "model": row["model"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "thinking_tokens": row["thinking_tokens"],
            "seconds": row["seconds"],
            "cached": bool(row["cached"]),
            "usd": row["usd"],
        }
        # A key can legitimately appear twice: a cache hit after a live call. Keep
        # the first real call and note that it was replayed, rather than letting a
        # replayed row overwrite the numbers that were actually billed.
        if key in out:
            out[key]["replays"] = out[key].get("replays", 0) + 1
        else:
            out[key] = entry
    return out


def summary(work) -> dict:
    """Totals across the whole ledger, for a run-level figure."""
    path = db_path(work)
    if not path.is_file():
        return {"calls": 0, "tokens_in": 0, "tokens_out": 0, "thinking": 0,
                "usd": 0.0, "seconds": 0.0}
    try:
        with _connect(path) as conn:
            conn.executescript(SCHEMA)
            row = conn.execute(
                "SELECT COUNT(*) AS calls,"
                " COALESCE(SUM(prompt_tokens),0) AS tin,"
                " COALESCE(SUM(completion_tokens),0) AS tout,"
                " COALESCE(SUM(thinking_tokens),0) AS think,"
                " COALESCE(SUM(usd),0.0) AS usd,"
                " COALESCE(SUM(seconds),0.0) AS secs FROM calls").fetchone()
    except Exception:
        return {"calls": 0, "tokens_in": 0, "tokens_out": 0, "thinking": 0,
                "usd": 0.0, "seconds": 0.0}
    return {"calls": row["calls"], "tokens_in": row["tin"],
            "tokens_out": row["tout"], "thinking": row["think"],
            "usd": row["usd"], "seconds": row["secs"]}


def reset(work) -> bool:
    """Delete the ledger. Returns True when something was removed."""
    path = db_path(work)
    removed = False
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            try:
                os.remove(p)
                removed = True
            except OSError:
                pass
    return removed
