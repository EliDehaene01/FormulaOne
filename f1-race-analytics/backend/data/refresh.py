"""
Incremental refresh: pull any session that's completed since the last time
this ran, without re-downloading everything ingest.py already backfilled.

Meant for the scheduled workflow (.github/workflows/data-refresh.yml), not
manual backfills -- use ingest.py directly for those.

WHY NOT JUST RE-RUN ingest.py?
--------------------------------
cache.get_or_fetch() caches a (table, cache_key) pair forever once fetched.
meetings/sessions are cached at YEAR granularity -- "sessions_year_2026" is
one opaque cache key covering every session in that year. Re-running
ingest.py against the current season would keep reading last week's session
list back out of the cache and never notice a new race weekend happened.
This script always re-fetches the current year's session list LIVE, diffs
it against what's already cached per-session, and only pulls what's new --
then refreshes the meetings/sessions cache entries too, so later reads of
those tables (features.py, llm/tools.py, the frontend API) see the new
session immediately instead of on the next full backfill.

Run with:
    python -m backend.data.refresh                  # current year only
    python -m backend.data.refresh --years 2025 2026
"""

import argparse
import datetime as dt

from backend.data import cache, openf1_client as client
from backend.data.ingest import ingest_session


def _already_ingested(conn, session_key: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM _cache_log WHERE table_name = 'laps' AND cache_key = ?",
        (f"session_{session_key}",),
    ).fetchone()
    return row is not None


def _force_refetch(conn, table: str, cache_key: str, fetch_fn) -> None:
    """Bypass the hard cache for one (table, cache_key) pair. Only used for
    meetings/sessions -- the only tables cached at year (not per-session)
    granularity, so the only ones that can go stale mid-season."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None
    if table_exists:
        conn.execute(f"DELETE FROM {table} WHERE _cache_key = ?", (cache_key,))
        conn.execute("DELETE FROM _cache_log WHERE table_name = ? AND cache_key = ?", (table, cache_key))
        conn.commit()
    cache.get_or_fetch(conn, table, cache_key, fetch_fn)


def refresh(years: list[int]) -> int:
    """Ingests any newly-completed session in `years`. Returns how many."""
    conn = cache.get_connection()
    now = dt.datetime.now(dt.timezone.utc)
    new_count = 0
    try:
        for year in years:
            sessions = client.get_sessions(year=year)
            new_sessions = [
                s for s in sessions
                if dt.datetime.fromisoformat(s["date_start"]) <= now and not _already_ingested(conn, s["session_key"])
            ]
            if not new_sessions:
                print(f"{year}: nothing new.")
                continue

            print(f"=== {year}: {len(new_sessions)} new session(s) ===")
            for session in new_sessions:
                print(f"  session {session['session_key']} ({session['session_name']})")
                ingest_session(conn, session["session_key"])
                new_count += 1

            _force_refetch(conn, "meetings", f"meetings_year_{year}", lambda y=year: client.get_meetings(year=y))
            _force_refetch(conn, "sessions", f"sessions_year_{year}", lambda y=year: client.get_sessions(year=y))
    finally:
        conn.close()

    print(f"\nDone. {new_count} new session(s) ingested." if new_count else "\nDone. Nothing new.")
    return new_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull any OpenF1 session completed since the last update.")
    parser.add_argument("--years", type=int, nargs="+", default=[dt.date.today().year])
    args = parser.parse_args()
    refresh(args.years)


if __name__ == "__main__":
    main()
