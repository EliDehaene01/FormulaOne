"""
Sanity checks for the deterministic parts of backend/llm/summary.py, run
against the real cache. Doesn't call the LLM (costs real tokens — that part
is verified by manually running build_race_summary(), same approach as
Phase 3's client.py).

Run with:
    python -m backend.tests.test_summary
"""

import sqlite3

import pandas as pd

from backend.data.cache import DB_PATH
from backend.llm.summary import _compute_key_moments, _lap_time_deltas


def _real_race_session_key() -> int:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT session_key FROM sessions WHERE session_type='Race' AND session_name='Race' AND year=2024 LIMIT 1"
    ).fetchone()
    conn.close()
    return row[0]


def _laps_for(session_key: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT driver_number, lap_number, lap_duration, is_pit_out_lap FROM laps WHERE session_key = ?",
        conn, params=[session_key],
    )
    conn.close()
    return df


def test_key_moments_sane():
    session_key = _real_race_session_key()
    laps = _laps_for(session_key)
    moments = _compute_key_moments(session_key, laps)

    assert moments["fastest_lap"] is not None
    assert moments["fastest_lap"]["lap_duration_s"] > 0

    for mover in (moments["biggest_gainer"], moments["biggest_loser"]):
        assert mover is None or isinstance(mover["positions_changed"], int)

    assert moments["biggest_gainer"]["positions_changed"] >= 0
    assert moments["biggest_loser"]["positions_changed"] <= 0
    assert all(count >= 0 for count in moments["pit_stop_counts_by_driver"].values())


def test_lap_time_deltas_never_negative():
    session_key = _real_race_session_key()
    laps = _laps_for(session_key)
    deltas = _lap_time_deltas(laps)

    assert len(deltas) > 0
    # every row's delta is relative to the FASTEST lap set on that lap number,
    # so nothing can be faster than its own lap's reference point.
    assert all(row["delta_to_fastest_s"] >= -1e-9 for row in deltas)


if __name__ == "__main__":
    test_key_moments_sane()
    test_lap_time_deltas_never_negative()
    print("All summary tests passed.")
