"""
Local SQLite cache so we don't hit the OpenF1 API every time we need data
we've already fetched (training the model, generating a chart, etc. would
otherwise all re-download the same laps over and over).

Design: one SQLite table per OpenF1 endpoint (laps, weather, sessions, ...),
holding the raw rows OpenF1 returned. A separate `_cache_log` table tracks
which (table, cache_key) pairs we've already fetched, so we know whether to
hit the network or just read from SQLite.

`cache_key` is our own bookkeeping id (e.g. "session_9158" or "year_2023"),
NOT an OpenF1 field — it's how we know "have we already fetched everything
for this session/year?" without re-deriving it from the data itself.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "storage" / "cache.sqlite"


def get_connection() -> sqlite3.Connection:
    """Open the cache DB, creating the storage folder and bookkeeping table if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _cache_log (
            table_name TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            PRIMARY KEY (table_name, cache_key)
        )
        """
    )
    return conn


def _is_cached(conn: sqlite3.Connection, table: str, cache_key: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM _cache_log WHERE table_name = ? AND cache_key = ?",
        (table, cache_key),
    ).fetchone()
    return row is not None


def get_or_fetch(conn: sqlite3.Connection, table: str, cache_key: str, fetch_fn) -> pd.DataFrame:
    """
    The one function everything else in the data layer calls.

    - If `(table, cache_key)` was already fetched, read it back from SQLite
      (no network call).
    - Otherwise call `fetch_fn()` (a zero-arg function that hits the OpenF1
      API), store the result, and return it.

    `table` is always one of our own hardcoded endpoint names (never user
    input), so building the SQL with an f-string here is safe — there's no
    injection surface since callers can't pass arbitrary table names.
    """
    if _is_cached(conn, table, cache_key):
        return pd.read_sql(
            f"SELECT * FROM {table} WHERE _cache_key = ?",
            conn,
            params=(cache_key,),
        ).drop(columns=["_cache_key"])

    records = fetch_fn()
    df = pd.DataFrame(records)

    if df.empty:
        # OpenF1 can legitimately return an empty list (e.g. no race_control
        # messages in a quiet session). Still mark it cached so we don't
        # keep re-requesting an endpoint that has nothing to say — but we
        # must still create the table, otherwise the *next* call's cache-hit
        # read (SELECT FROM table) fails because the table was never made.
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (_cache_key TEXT)")
        conn.execute(
            "INSERT INTO _cache_log (table_name, cache_key) VALUES (?, ?)",
            (table, cache_key),
        )
        conn.commit()
        return df

    df["_cache_key"] = cache_key  # tag every row so future reads can filter back to this batch

    # Some OpenF1 fields are nested (e.g. /laps' segments_sector_1 is a list
    # of per-mini-sector status codes). SQLite columns are scalar-only, so
    # any list/dict cell has to become a JSON string before we can insert it.
    df = df.map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)

    # A table can already exist with FEWER columns than this batch has —
    # e.g. the first session ever queried for an endpoint had no rows (see
    # the empty-result branch above, which creates a bare _cache_key-only
    # table), or an OpenF1 endpoint's schema genuinely grew a new field
    # between older and newer sessions (confirmed in practice: `pit`'s
    # stop_duration is only populated from the 2024 US GP onward). Add
    # whatever's missing before inserting, or to_sql's append raises
    # "table X has no column named Y".
    table_exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None
    if table_exists:
        existing_columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for column in df.columns:
            if column not in existing_columns:
                conn.execute(f'ALTER TABLE {table} ADD COLUMN "{column}"')
    # else: no widening needed — to_sql below creates the table fresh from
    # this DataFrame's own columns, the same as it always has.

    df.to_sql(table, conn, if_exists="append", index=False)
    conn.execute(
        "INSERT INTO _cache_log (table_name, cache_key) VALUES (?, ?)",
        (table, cache_key),
    )
    conn.commit()
    return df.drop(columns=["_cache_key"])
