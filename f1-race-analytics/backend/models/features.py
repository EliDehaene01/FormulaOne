"""
Feature engineering: turns the raw OpenF1 cache into one training row per
(Qualifying session, driver) — the unit our model predicts on.

THE CORE IDEA
--------------
We frame "predict the fastest qualifying time and who sets it" as ONE
regression problem: predict every driver's own best qualifying lap time.
Once we have that per-driver prediction for a session, the session-level
answers fall out for free:
    predicted fastest time  = min(predicted times in that session)
    predicted pole driver   = the driver with that min

This is simpler than training two separate models (one regression, one
classification) and it's how you'd actually reason about it as a fan: you
don't guess "who's on pole" directly, you estimate everyone's pace and see
who comes out fastest.

WHY A CIRCUIT EMBEDDING (not just driver/team)
------------------------------------------------
Raw lap times differ enormously by track (Monaco ~71s, Spa ~103s). Without
something that tells the model *which* track a row belongs to, it can't
tell a great Monaco lap from a mediocre Spa lap just from the number of
seconds. We solve most of this with `practice_gap_to_best_s` (see below —
it's relative, not absolute, so it's already track-independent), but we
still give the model a circuit identity via an embedding (same technique as
driver/team) so it can learn each track's own baseline pace and quirks.

AVOIDING LEAKAGE
-----------------
Every "form" feature below (recent/season qualifying form) is computed with
`.shift(1)` before any rolling/expanding average — that excludes the
CURRENT session from its own feature, using only sessions that happened
strictly earlier. Without this, we'd be handing the model a sneak peek at
the answer it's supposed to predict.
"""

import sqlite3

import pandas as pd

# Fixed, not derived from whatever happens to appear in the data — so the
# one-hot columns are always the same shape whether we're building a
# training set, a validation set, or a single row for live inference.
COMPOUND_CATEGORIES = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "OTHER"]

NUMERIC_FEATURE_COLUMNS = [
    "practice_best_time_s",
    "practice_gap_to_best_s",
    "practice_sessions_completed",
    "recent_quali_gap_to_pole_avg_s",
    "season_quali_gap_to_pole_avg_s",
    "air_temperature",
    "track_temperature",
    "humidity",
    "rainfall",
] + [f"compound_{c}" for c in COMPOUND_CATEGORIES]


def _best_lap_per_session(laps: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (session_key, driver_number): their fastest *clean* lap.

    "Clean" = not an out-lap (is_pit_out_lap == 0, out-laps are slow and
    untimed-in-the-meaningful-sense) and has a recorded lap_duration (some
    laps are null — red flags, DNFs mid-lap, etc.).
    """
    clean = laps[(laps["is_pit_out_lap"] == 0) & laps["lap_duration"].notna()]
    best_idx = clean.groupby(["session_key", "driver_number"])["lap_duration"].idxmin()
    best = clean.loc[best_idx, ["session_key", "driver_number", "lap_duration"]]
    return best.rename(columns={"lap_duration": "best_lap_s"})


def _qualifying_targets(sessions: pd.DataFrame, best_laps: pd.DataFrame) -> pd.DataFrame:
    """
    The label we're predicting, one row per (driver, real Qualifying session).

    Joining against `best_laps` (inner join) automatically drops sessions
    with no lap data at all — which is exactly the scheduled-but-not-yet-run
    2026 sessions OpenF1 already lists. No laps recorded means no row here.
    """
    quali_sessions = sessions[
        (sessions["session_type"] == "Qualifying") & (sessions["session_name"] == "Qualifying")
    ]
    quali = best_laps.merge(
        quali_sessions[["session_key", "meeting_key", "year", "date_start", "circuit_short_name"]],
        on="session_key",
    )
    quali = quali.rename(columns={"best_lap_s": "target_lap_time_s"})

    # Pole time = the fastest time set in that session. gap_to_pole is how
    # far off pole each driver was — a track-independent measure of pace
    # that we can meaningfully average/roll across different circuits.
    quali["session_pole_time_s"] = quali.groupby("session_key")["target_lap_time_s"].transform("min")
    quali["gap_to_pole_s"] = quali["target_lap_time_s"] - quali["session_pole_time_s"]
    return quali


def _add_form_features(quali: pd.DataFrame) -> pd.DataFrame:
    """Rolling recent form + season-to-date form, per driver, leakage-safe (see module docstring)."""
    quali = quali.sort_values(["driver_number", "date_start"]).copy()

    quali["recent_quali_gap_to_pole_avg_s"] = quali.groupby("driver_number")["gap_to_pole_s"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    quali["season_quali_gap_to_pole_avg_s"] = quali.groupby(["driver_number", "year"])["gap_to_pole_s"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    return quali


def _add_practice_features(quali: pd.DataFrame, sessions: pd.DataFrame, best_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Practice pace for the SAME race weekend (joined by meeting_key — this is
    what naturally excludes pre-season testing days too, since testing has
    its own meeting_key that no Qualifying session shares).
    """
    practice_sessions = sessions[sessions["session_type"] == "Practice"]
    practice = best_laps.merge(practice_sessions[["session_key", "meeting_key"]], on="session_key")

    per_driver = (
        practice.groupby(["meeting_key", "driver_number"])
        .agg(
            practice_best_time_s=("best_lap_s", "min"),
            # Number of practice SESSIONS (FP1/FP2/FP3) the driver set a
            # valid lap in — not total laps run. A rough "did they get
            # normal running" signal, not a precise lap count.
            practice_sessions_completed=("best_lap_s", "count"),
        )
        .reset_index()
    )
    meeting_best = practice.groupby("meeting_key")["best_lap_s"].min().rename("meeting_practice_best_s")
    per_driver = per_driver.merge(meeting_best, on="meeting_key")
    per_driver["practice_gap_to_best_s"] = per_driver["practice_best_time_s"] - per_driver["meeting_practice_best_s"]

    return quali.merge(
        per_driver[["meeting_key", "driver_number", "practice_best_time_s", "practice_gap_to_best_s", "practice_sessions_completed"]],
        on=["meeting_key", "driver_number"],
        how="left",
    )


def _add_weather_features(quali: pd.DataFrame, conn: sqlite3.Connection, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    We can't know qualifying-day weather in advance for a real "upcoming
    race" prediction, so we use the LAST practice session's weather as the
    closest available proxy (same day, similar conditions, no leakage).
    """
    practice_sessions = sessions[sessions["session_type"] == "Practice"]
    last_practice = practice_sessions.sort_values("date_start").groupby("meeting_key").tail(1)

    weather = pd.read_sql("SELECT session_key, air_temperature, track_temperature, humidity, rainfall FROM weather", conn)
    weather_avg = weather.groupby("session_key")[["air_temperature", "track_temperature", "humidity", "rainfall"]].mean()

    meeting_weather = last_practice[["meeting_key", "session_key"]].merge(weather_avg, on="session_key", how="left")
    meeting_weather = meeting_weather.drop(columns="session_key")

    return quali.merge(meeting_weather, on="meeting_key", how="left")


def _add_compound_features(quali: pd.DataFrame, conn: sqlite3.Connection, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot: which tire compound the driver ran MOST during practice this
    weekend (by lap count, not the compound of their single fastest lap —
    a simpler join for the same "what were they mostly running" signal;
    good enough for a first baseline).
    """
    practice_sessions = sessions[sessions["session_type"] == "Practice"]
    stints = pd.read_sql("SELECT session_key, driver_number, compound, lap_start, lap_end FROM stints", conn)
    stints["lap_count"] = (stints["lap_end"] - stints["lap_start"] + 1).clip(lower=1)
    stints["compound"] = stints["compound"].where(stints["compound"].isin(COMPOUND_CATEGORIES), "OTHER")

    practice_stints = stints.merge(practice_sessions[["session_key", "meeting_key"]], on="session_key")
    laps_by_compound = practice_stints.groupby(["meeting_key", "driver_number", "compound"])["lap_count"].sum().reset_index()
    top_idx = laps_by_compound.groupby(["meeting_key", "driver_number"])["lap_count"].idxmax()
    top_compound = laps_by_compound.loc[top_idx, ["meeting_key", "driver_number", "compound"]]

    quali = quali.merge(top_compound, on=["meeting_key", "driver_number"], how="left")
    # Fixed categories (not just whatever's present) so one-hot columns are
    # identical across train/val/test/live-inference builds of this table.
    quali["compound"] = pd.Categorical(quali["compound"].fillna("OTHER"), categories=COMPOUND_CATEGORIES)
    dummies = pd.get_dummies(quali["compound"], prefix="compound").astype(float)
    return pd.concat([quali.drop(columns="compound"), dummies], axis=1)


def _add_team(quali: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Team AT THE TIME of that qualifying session, not a fixed per-driver
    mapping — a driver's team can and does change mid-season (confirmed in
    this dataset: 2 mid-season swaps), so this must be a per-session join.
    """
    drivers = pd.read_sql("SELECT session_key, driver_number, team_name FROM drivers", conn)
    quali = quali.merge(drivers, on=["session_key", "driver_number"], how="left")
    quali["team_name"] = quali["team_name"].fillna("UNKNOWN_TEAM")
    return quali


def build_feature_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    The single entry point: cached OpenF1 data in, one row per
    (Qualifying session, driver) with target + engineered features out.
    """
    sessions = pd.read_sql(
        "SELECT session_key, session_name, session_type, meeting_key, year, date_start, circuit_short_name FROM sessions",
        conn,
    )
    sessions["date_start"] = pd.to_datetime(sessions["date_start"])

    laps = pd.read_sql("SELECT session_key, driver_number, lap_duration, is_pit_out_lap FROM laps", conn)
    best_laps = _best_lap_per_session(laps)

    quali = _qualifying_targets(sessions, best_laps)
    quali = _add_form_features(quali)
    quali = _add_practice_features(quali, sessions, best_laps)
    quali = _add_weather_features(quali, conn, sessions)
    quali = _add_compound_features(quali, conn, sessions)
    quali = _add_team(quali, conn)

    # Simplification (documented, not hidden): missing numeric values are
    # filled with the whole-table mean rather than a train-only mean like
    # the feature scaler in train.py uses. That's a tiny amount of leakage
    # (val/test rows influence the fill value a little) — acceptable for a
    # first baseline since missing values are rare (mostly reserve-driver
    # sessions with no practice running). Upgrade path if it ever matters:
    # move imputation into train.py and fit it on the train split only,
    # the same way the numeric scaler already is.
    fill_cols = [
        "practice_best_time_s",
        "practice_gap_to_best_s",
        "practice_sessions_completed",
        "air_temperature",
        "track_temperature",
        "humidity",
        "rainfall",
    ]
    quali[fill_cols] = quali[fill_cols].fillna(quali[fill_cols].mean())

    # A rookie's first-ever qualifying has no prior sessions to average, so
    # rolling/expanding form is NaN. Filling with the overall mean gap
    # assumes "average midfield pace, no information yet" — a more honest
    # default than filling with 0, which would assume pole-level pace.
    mean_gap = quali["gap_to_pole_s"].mean()
    quali["recent_quali_gap_to_pole_avg_s"] = quali["recent_quali_gap_to_pole_avg_s"].fillna(mean_gap)
    quali["season_quali_gap_to_pole_avg_s"] = quali["season_quali_gap_to_pole_avg_s"].fillna(mean_gap)

    id_cols = ["session_key", "driver_number", "date_start", "year", "circuit_short_name", "team_name"]
    return quali[id_cols + ["target_lap_time_s"] + NUMERIC_FEATURE_COLUMNS].reset_index(drop=True)
