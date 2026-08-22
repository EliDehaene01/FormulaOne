"""
Smoke test for the caching logic in backend/data/cache.py.

This is the one piece of Phase 1 with real branching behavior (cached vs.
not cached, empty vs. non-empty results), so it's the one piece worth a
runnable check. No network calls, no pytest — just asserts against a
throwaway in-memory SQLite database.

Run with:
    python -m backend.tests.test_cache
"""

import sqlite3

from backend.data import cache


def _connection_with_log_table() -> sqlite3.Connection:
    """An in-memory DB (not the real cache.sqlite) with the same bookkeeping table cache.py expects."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE _cache_log (
            table_name TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            PRIMARY KEY (table_name, cache_key)
        )
        """
    )
    return conn


def test_second_call_does_not_refetch():
    conn = _connection_with_log_table()
    call_count = {"n": 0}

    def fake_fetch():
        call_count["n"] += 1
        return [{"lap_number": 1, "lap_duration": 91.234}, {"lap_number": 2, "lap_duration": 90.876}]

    first = cache.get_or_fetch(conn, "laps", "session_9999", fake_fetch)
    second = cache.get_or_fetch(conn, "laps", "session_9999", fake_fetch)

    assert call_count["n"] == 1, "fetch_fn should only run once — the second call must hit the cache"
    assert first.to_dict("records") == second.to_dict("records"), "cached read should return the same rows as the original fetch"
    assert list(first.columns) == ["lap_number", "lap_duration"], "internal _cache_key column should not leak to callers"


def test_empty_result_is_still_marked_cached():
    conn = _connection_with_log_table()
    call_count = {"n": 0}

    def fake_fetch_empty():
        call_count["n"] += 1
        return []  # e.g. a session with no race_control messages

    first = cache.get_or_fetch(conn, "race_control", "session_9999", fake_fetch_empty)
    second = cache.get_or_fetch(conn, "race_control", "session_9999", fake_fetch_empty)

    assert first.empty and second.empty
    assert call_count["n"] == 1, "an empty API result should still be cached, not re-fetched forever"


def test_different_cache_keys_are_independent():
    conn = _connection_with_log_table()
    calls = []

    def fake_fetch():
        calls.append(1)
        return [{"driver_number": 1}]

    cache.get_or_fetch(conn, "drivers", "session_1", fake_fetch)
    cache.get_or_fetch(conn, "drivers", "session_2", fake_fetch)

    assert len(calls) == 2, "different session keys must each trigger their own fetch"


if __name__ == "__main__":
    test_second_call_does_not_refetch()
    test_empty_result_is_still_marked_cached()
    test_different_cache_keys_are_independent()
    print("All cache tests passed.")
