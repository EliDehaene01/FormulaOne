"""
Command-line script: pull a set of past races from OpenF1 and cache them
locally, so every later phase (feature engineering, charts, LLM tools)
reads from SQLite instead of the network.

Usage:
    python -m backend.data.ingest --years 2023
    python -m backend.data.ingest --years 2023 2024 --session-types Race Qualifying
    python -m backend.data.ingest --years 2023 --limit 2      # just the first 2 sessions, for a quick test
"""

import argparse
import time

from backend.data import cache, openf1_client as client

# Being polite to a free, unauthenticated API: a pause between per-session
# requests avoids hammering it during a big backfill. 0.2s was too aggressive
# in practice — OpenF1 started returning 429 (Too Many Requests) after only
# ~30 requests. openf1_client retries on 429 with backoff regardless, but a
# larger gap up front means we trigger that far less often.
REQUEST_DELAY_SECONDS = 0.5


def ingest_year(conn, year: int, session_types: list[str] | None, limit: int | None) -> None:
    print(f"\n=== {year} ===")

    # /meetings gives us the race weekends; we cache it mainly so the
    # frontend can later show "Bahrain GP", "Monaco GP" etc. as race names.
    cache.get_or_fetch(conn, "meetings", f"meetings_year_{year}", lambda: client.get_meetings(year=year))

    # /sessions?year=Y returns every Practice/Qualifying/Race session that year
    # in one call — cheaper than asking meeting-by-meeting.
    sessions_df = cache.get_or_fetch(conn, "sessions", f"sessions_year_{year}", lambda: client.get_sessions(year=year))

    if session_types:
        sessions_df = sessions_df[sessions_df["session_type"].isin(session_types)]

    session_keys = sessions_df["session_key"].tolist()
    if limit is not None:
        session_keys = session_keys[:limit]

    for i, session_key in enumerate(session_keys, start=1):
        session_name = sessions_df.loc[sessions_df["session_key"] == session_key, "session_name"].iloc[0]
        print(f"  [{i}/{len(session_keys)}] session {session_key} ({session_name})")
        ingest_session(conn, session_key)


def ingest_session(conn, session_key: int) -> None:
    """Cache every per-session endpoint we need for charts/features/LLM tools."""
    # Each lambda takes `sk=session_key` as a default argument, not a free
    # variable — Python closures capture variables by reference, so without
    # this every lambda in the loop would see whatever session_key the loop
    # variable ends up holding *last*, not the value at the time it was
    # defined. Binding it as a default argument freezes the value now.
    endpoints = {
        "laps": lambda sk=session_key: client.get_laps(session_key=sk),
        "position": lambda sk=session_key: client.get_position(session_key=sk),
        "stints": lambda sk=session_key: client.get_stints(session_key=sk),
        "weather": lambda sk=session_key: client.get_weather(session_key=sk),
        "race_control": lambda sk=session_key: client.get_race_control(session_key=sk),
        "drivers": lambda sk=session_key: client.get_drivers(session_key=sk),
        "pit": lambda sk=session_key: client.get_pit(session_key=sk),
        "intervals": lambda sk=session_key: client.get_intervals(session_key=sk),
        "overtakes": lambda sk=session_key: client.get_overtakes(session_key=sk),
        "team_radio": lambda sk=session_key: client.get_team_radio(session_key=sk),
        "session_result": lambda sk=session_key: client.get_session_result(session_key=sk),
        "starting_grid": lambda sk=session_key: client.get_starting_grid(session_key=sk),
        "championship_drivers": lambda sk=session_key: client.get_championship_drivers(session_key=sk),
        "championship_teams": lambda sk=session_key: client.get_championship_teams(session_key=sk),
        # car_data is deliberately excluded here — see get_car_data's docstring
        # in openf1_client.py (~3.7Hz telemetry, not bulk-cache-sized).
    }
    for table, fetch_fn in endpoints.items():
        cache.get_or_fetch(conn, table, f"session_{session_key}", fetch_fn)
        time.sleep(REQUEST_DELAY_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache OpenF1 race data.")
    parser.add_argument("--years", type=int, nargs="+", default=[2023], help="Season year(s) to ingest.")
    parser.add_argument(
        "--session-types",
        nargs="+",
        default=None,
        help="Only these session types, e.g. Race Qualifying. Default: all (Practice 1/2/3, Qualifying, Sprint, Race).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest the first N sessions per year — handy for a quick smoke test before running a full season.",
    )
    args = parser.parse_args()

    conn = cache.get_connection()
    try:
        for year in args.years:
            ingest_year(conn, year, args.session_types, args.limit)
    finally:
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
