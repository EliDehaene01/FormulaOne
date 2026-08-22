"""
Sanity checks for backend/models/features.py, run against the REAL cached
data (not synthetic fixtures) — feature engineering is exactly the kind of
code where a subtly wrong join silently produces plausible-looking numbers,
so checking against real data is more valuable here than a hand-built
example would be.

Run with:
    python -m backend.tests.test_features
"""

import sqlite3

from backend.data.cache import DB_PATH
from backend.models.features import NUMERIC_FEATURE_COLUMNS, build_feature_table


def test_feature_table_sane():
    conn = sqlite3.connect(DB_PATH)
    df = build_feature_table(conn)
    conn.close()

    assert len(df) > 0, "expected at least some real Qualifying sessions in the cache"

    assert df[NUMERIC_FEATURE_COLUMNS + ["target_lap_time_s"]].isna().sum().sum() == 0, \
        "no NaNs should reach the model — features.py is responsible for filling them"

    assert (df["target_lap_time_s"] > 0).all(), "lap times must be positive"

    # A driver's qualifying gap to pole should never be negative — pole IS
    # the fastest time in the session, everyone else's gap is >= 0.
    session_min = df.groupby("session_key")["target_lap_time_s"].transform("min")
    assert ((df["target_lap_time_s"] - session_min) >= -1e-9).all(), "no one should be faster than the session's own pole time"

    # Every one-hot compound row should sum to exactly 1 (each row belongs
    # to exactly one compound category, including the OTHER catch-all).
    compound_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c.startswith("compound_")]
    assert (df[compound_cols].sum(axis=1) == 1).all(), "one-hot compound columns should sum to 1 per row"

    # Every session should have at least 2 drivers (a "session" with 1 row
    # would mean our per-session grouping logic is broken somewhere).
    assert (df.groupby("session_key").size() >= 2).all()


if __name__ == "__main__":
    test_feature_table_sane()
    print("All feature tests passed.")
