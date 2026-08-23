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
Every "form"/"history" feature below (recent/season qualifying form, race
form, circuit history, teammate comparison) is computed with `.shift(1)`
before any rolling/expanding aggregate — that excludes the CURRENT session
from its own feature, using only sessions/races that happened strictly
earlier. Race-derived features (pit stops, overtakes, grid-to-finish gain,
race points/position) additionally only ever look at PAST race weekends —
never the race for the weekend being predicted, since that race hasn't
happened yet relative to its own qualifying session. Without these rules
we'd be handing the model a sneak peek at the answer it's supposed to
predict.

WHY THE NUMERIC FEATURE COUNT IS LARGE (50+)
------------------------------------------------
This started as a small baseline (~15 features: practice pace, basic form,
weather, tires, championship standing). It's since been extended to use
every OpenF1 endpoint this project ingests: per-session practice pace
(FP1/FP2/FP3 individually, not just aggregated), sector-level pace, prior
race results and racecraft (grid-to-finish gain, pit stops, overtakes),
circuit-specific history, team-level form, team and driver championship
standing, teammate head-to-head form, and weekend scheduling context. More
features isn't automatically better (see backend/models/README.md for the
model's honest baseline comparison) — this is deliberately cast a wide net
so the network has real signal to learn FROM, not a claim that all 50+ are
equally predictive.
"""

import sqlite3

import pandas as pd

# Fixed, not derived from whatever happens to appear in the data — so the
# one-hot columns are always the same shape whether we're building a
# training set, a validation set, or a single row for live inference.
COMPOUND_CATEGORIES = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "OTHER"]

NUMERIC_FEATURE_COLUMNS = [
    # --- practice pace (weekend-aggregated) ---
    "practice_best_time_s",
    "practice_gap_to_best_s",
    "practice_sessions_completed",
    # --- practice pace (per FP session) ---
    "practice_fp1_best_time_s",
    "practice_fp1_gap_to_best_s",
    "practice_fp2_best_time_s",
    "practice_fp2_gap_to_best_s",
    "practice_fp3_best_time_s",
    "practice_fp3_gap_to_best_s",
    "practice_total_laps",
    # --- practice pace (sector-level) ---
    "practice_best_sector1_gap_s",
    "practice_best_sector2_gap_s",
    "practice_best_sector3_gap_s",
    # --- qualifying form ---
    "recent_quali_gap_to_pole_avg_s",
    "season_quali_gap_to_pole_avg_s",
    # --- circuit-specific history ---
    "circuit_history_avg_gap_to_pole_s",
    "circuit_history_best_gap_to_pole_s",
    "circuit_history_sessions_count",
    # --- team-level form ---
    "team_recent_quali_gap_to_pole_avg_s",
    "team_season_quali_gap_to_pole_avg_s",
    # --- teammate comparison ---
    "recent_teammate_quali_diff_avg_s",
    # --- weather ---
    "air_temperature",
    "track_temperature",
    "track_temperature_range",
    "humidity",
    "rainfall",
    "wind_speed",
    # --- championship standing entering the weekend ---
    "championship_position_entering_weekend",
    "championship_points_entering_weekend",
    "team_championship_position_entering_weekend",
    "team_championship_points_entering_weekend",
    # --- recent RACE form (prior race weekends only) ---
    "recent_race_finish_position_avg",
    "season_race_finish_position_avg",
    "recent_race_points_avg",
    "season_race_points_avg",
    "recent_dnf_rate",
    "recent_grid_to_finish_gain_avg",
    "recent_pit_stop_count_avg",
    "recent_pit_stop_duration_avg_s",
    "recent_overtakes_made_avg",
    "recent_overtakes_lost_avg",
    "recent_race_gap_to_winner_avg_s",
    # --- weekend scheduling / experience ---
    "driver_career_sessions_count",
    "meeting_round_number",
    "days_since_last_session",
    "is_sprint_weekend",
    "practice_red_flags_count",
    "practice_yellow_flags_count",
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


def _add_circuit_history_features(quali: pd.DataFrame) -> pd.DataFrame:
    """
    Driver's own history AT THIS CIRCUIT from earlier visits only — same
    shift(1) rule as _add_form_features, just grouped by (driver, circuit)
    instead of driver alone, so a return to a track they've raced before
    carries "how have I usually done here" without seeing this visit's own
    result.
    """
    quali = quali.sort_values(["driver_number", "circuit_short_name", "date_start"]).copy()
    g = quali.groupby(["driver_number", "circuit_short_name"])["gap_to_pole_s"]
    quali["circuit_history_avg_gap_to_pole_s"] = g.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    quali["circuit_history_best_gap_to_pole_s"] = g.transform(lambda s: s.shift(1).expanding(min_periods=1).min())
    quali["circuit_history_sessions_count"] = g.transform(lambda s: s.shift(1).expanding(min_periods=1).count())
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


def _add_practice_by_session_features(quali: pd.DataFrame, sessions: pd.DataFrame, best_laps: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """
    Same practice pace as _add_practice_features but broken out PER FP
    session instead of aggregated across the weekend — lets the model see
    whether pace improved/declined session to session, not just the best of
    the three. practice_total_laps is real running volume (every lap, not
    just the ones that produced a session-best), a reliability/track-time
    signal distinct from practice_sessions_completed above.
    """
    practice_sessions = sessions[sessions["session_type"] == "Practice"][["session_key", "meeting_key", "session_name"]]
    practice = best_laps.merge(practice_sessions, on="session_key")
    meeting_best = practice.groupby("meeting_key")["best_lap_s"].min().rename("meeting_practice_best_s")
    practice = practice.merge(meeting_best, on="meeting_key")
    practice["gap_s"] = practice["best_lap_s"] - practice["meeting_practice_best_s"]

    for fp_num, fp_name in [(1, "Practice 1"), (2, "Practice 2"), (3, "Practice 3")]:
        fp = practice[practice["session_name"] == fp_name][["meeting_key", "driver_number", "best_lap_s", "gap_s"]].rename(
            columns={
                "best_lap_s": f"practice_fp{fp_num}_best_time_s",
                "gap_s": f"practice_fp{fp_num}_gap_to_best_s",
            }
        )
        quali = quali.merge(fp, on=["meeting_key", "driver_number"], how="left")

    practice_laps = laps.merge(practice_sessions[["session_key", "meeting_key"]], on="session_key")
    lap_counts = practice_laps.groupby(["meeting_key", "driver_number"]).size().rename("practice_total_laps").reset_index()
    quali = quali.merge(lap_counts, on=["meeting_key", "driver_number"], how="left")
    return quali


def _add_sector_features(quali: pd.DataFrame, sessions: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Gap to the weekend's best practice sector time, per sector — finer-grained pace signal than the whole-lap practice_gap_to_best_s."""
    practice_sessions = sessions[sessions["session_type"] == "Practice"][["session_key", "meeting_key"]]
    practice_laps = laps.merge(practice_sessions, on="session_key")
    clean = practice_laps[practice_laps["is_pit_out_lap"] == 0]

    for i in (1, 2, 3):
        col = f"duration_sector_{i}"
        best_per_driver = clean.groupby(["meeting_key", "driver_number"])[col].min().rename(f"best_{col}").reset_index()
        meeting_best = best_per_driver.groupby("meeting_key")[f"best_{col}"].transform("min")
        best_per_driver[f"practice_best_sector{i}_gap_s"] = best_per_driver[f"best_{col}"] - meeting_best
        quali = quali.merge(
            best_per_driver[["meeting_key", "driver_number", f"practice_best_sector{i}_gap_s"]],
            on=["meeting_key", "driver_number"],
            how="left",
        )
    return quali


def _add_weather_features(quali: pd.DataFrame, conn: sqlite3.Connection, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    We can't know qualifying-day weather in advance for a real "upcoming
    race" prediction, so we use the LAST practice session's weather as the
    closest available proxy (same day, similar conditions, no leakage).
    track_temperature_range (max - min during that practice session) is a
    cheap proxy for "were conditions changing" beyond just the average.
    """
    practice_sessions = sessions[sessions["session_type"] == "Practice"]
    last_practice = practice_sessions.sort_values("date_start").groupby("meeting_key").tail(1)

    weather = pd.read_sql(
        "SELECT session_key, air_temperature, track_temperature, humidity, rainfall, wind_speed FROM weather", conn
    )
    weather_avg = weather.groupby("session_key")[["air_temperature", "track_temperature", "humidity", "rainfall", "wind_speed"]].mean()
    weather_range = weather.groupby("session_key")["track_temperature"].agg(lambda s: s.max() - s.min()).rename("track_temperature_range")

    meeting_weather = last_practice[["meeting_key", "session_key"]].merge(weather_avg, on="session_key", how="left")
    meeting_weather = meeting_weather.merge(weather_range, on="session_key", how="left")
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


def _add_team_form_features(quali: pd.DataFrame) -> pd.DataFrame:
    """
    Team-level (both cars) rolling quali form — same shift(1) rule, grouped
    by team_name instead of driver_number. A wider window (10 vs 5) since a
    team's combined session count per weekend is ~2x a single driver's.
    Must run AFTER _add_team (needs team_name).
    """
    quali = quali.sort_values(["team_name", "date_start"]).copy()
    quali["team_recent_quali_gap_to_pole_avg_s"] = quali.groupby("team_name")["gap_to_pole_s"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    quali["team_season_quali_gap_to_pole_avg_s"] = quali.groupby(["team_name", "year"])["gap_to_pole_s"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    return quali


def _add_teammate_features(quali: pd.DataFrame) -> pd.DataFrame:
    """
    Recent head-to-head qualifying form vs teammate: for each past session,
    (this driver's gap_to_pole - teammate's gap_to_pole) — negative means
    they out-qualified their teammate that session. Averaged over the last
    5 PRIOR sessions (shift(1) again) so the current session's own result
    (which is the target we're predicting) never leaks in.
    """
    pair = quali[["session_key", "date_start", "team_name", "driver_number", "gap_to_pole_s"]].copy()
    joined = pair.merge(pair, on=["session_key", "team_name"], suffixes=("", "_mate"))
    joined = joined[joined["driver_number"] != joined["driver_number_mate"]]
    joined["head_to_head_diff"] = joined["gap_to_pole_s"] - joined["gap_to_pole_s_mate"]

    per_driver_session = joined.groupby(["driver_number", "date_start"])["head_to_head_diff"].mean().reset_index()
    per_driver_session = per_driver_session.sort_values(["driver_number", "date_start"])
    per_driver_session["recent_teammate_quali_diff_avg_s"] = per_driver_session.groupby("driver_number")["head_to_head_diff"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    return quali.merge(
        per_driver_session[["driver_number", "date_start", "recent_teammate_quali_diff_avg_s"]],
        on=["driver_number", "date_start"],
        how="left",
    )


def _add_championship_features(quali: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Driver's championship standing ENTERING the race weekend — the earliest
    championship_drivers snapshot for that meeting, so it reflects standing
    before any session that weekend (including this quali) ran. Same
    leakage rule as everywhere else in this module: this must not depend on
    anything that happens IN the session we're predicting.
    """
    champ = pd.read_sql(
        "SELECT session_key, driver_number, points_start, position_start FROM championship_drivers", conn
    )
    if champ.empty:
        quali["championship_position_entering_weekend"] = None
        quali["championship_points_entering_weekend"] = None
        return quali

    sessions_meta = pd.read_sql("SELECT session_key, meeting_key, date_start FROM sessions", conn)
    sessions_meta["date_start"] = pd.to_datetime(sessions_meta["date_start"])
    champ = champ.merge(sessions_meta, on="session_key")
    earliest_idx = champ.groupby(["meeting_key", "driver_number"])["date_start"].idxmin()
    earliest = champ.loc[earliest_idx, ["meeting_key", "driver_number", "points_start", "position_start"]].rename(
        columns={
            "points_start": "championship_points_entering_weekend",
            "position_start": "championship_position_entering_weekend",
        }
    )
    return quali.merge(earliest, on=["meeting_key", "driver_number"], how="left")


def _add_team_championship_features(quali: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """Constructors'-championship mirror of _add_championship_features. Must run after _add_team (needs team_name)."""
    champ = pd.read_sql("SELECT session_key, team_name, points_start, position_start FROM championship_teams", conn)
    if champ.empty:
        quali["team_championship_position_entering_weekend"] = None
        quali["team_championship_points_entering_weekend"] = None
        return quali

    sessions_meta = pd.read_sql("SELECT session_key, meeting_key, date_start FROM sessions", conn)
    sessions_meta["date_start"] = pd.to_datetime(sessions_meta["date_start"])
    champ = champ.merge(sessions_meta, on="session_key")
    earliest_idx = champ.groupby(["meeting_key", "team_name"])["date_start"].idxmin()
    earliest = champ.loc[earliest_idx, ["meeting_key", "team_name", "points_start", "position_start"]].rename(
        columns={
            "points_start": "team_championship_points_entering_weekend",
            "position_start": "team_championship_position_entering_weekend",
        }
    )
    return quali.merge(earliest, on=["meeting_key", "team_name"], how="left")


def _race_level_table(conn: sqlite3.Connection, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (driver_number, meeting_key) summarizing that meeting's RACE
    session — the raw material for _add_race_form_features' rolling
    averages. NOT used directly as features itself (this meeting's own race
    happens AFTER its qualifying, so using it un-shifted for that same
    meeting's quali row would be leakage) — only ever consumed through a
    shift(1) rolling/expanding aggregate in the caller.
    """
    race_sessions = sessions[(sessions["session_type"] == "Race") & (sessions["session_name"] == "Race")][
        ["session_key", "meeting_key", "year", "date_start"]
    ]
    if race_sessions.empty:
        return pd.DataFrame(columns=["meeting_key", "driver_number", "year", "date_start"])
    race_key_list = race_sessions["session_key"].tolist()

    result = pd.read_sql("SELECT session_key, driver_number, position, gap_to_leader, dnf FROM session_result", conn)
    result = result[result["session_key"].isin(race_key_list)].merge(race_sessions, on="session_key")

    grid = pd.read_sql("SELECT session_key, driver_number, position AS grid_position FROM starting_grid", conn)
    grid = grid[grid["session_key"].isin(race_key_list)]

    # stop_duration is only populated from the 2024 US GP onward per OpenF1
    # docs — earlier races fill as 0 stops/0s here, a known simplification
    # (upgrade path: fall back to lane_duration/pit_duration for older races).
    pit = pd.read_sql("SELECT session_key, driver_number, stop_duration FROM pit", conn)
    pit = pit[pit["session_key"].isin(race_key_list)]
    pit_agg = (
        pit.groupby(["session_key", "driver_number"])
        .agg(pit_stop_count=("stop_duration", "size"), pit_stop_duration_avg_s=("stop_duration", "mean"))
        .reset_index()
    )

    overtakes = pd.read_sql("SELECT session_key, overtaking_driver_number, overtaken_driver_number FROM overtakes", conn)
    overtakes = overtakes[overtakes["session_key"].isin(race_key_list)]
    made = (
        overtakes.groupby(["session_key", "overtaking_driver_number"])
        .size()
        .rename("overtakes_made")
        .reset_index()
        .rename(columns={"overtaking_driver_number": "driver_number"})
    )
    lost = (
        overtakes.groupby(["session_key", "overtaken_driver_number"])
        .size()
        .rename("overtakes_lost")
        .reset_index()
        .rename(columns={"overtaken_driver_number": "driver_number"})
    )

    champ = pd.read_sql("SELECT session_key, driver_number, points_start, points_current FROM championship_drivers", conn)
    champ = champ[champ["session_key"].isin(race_key_list)].copy()
    champ["points_scored"] = champ["points_current"] - champ["points_start"]

    race = result.merge(grid, on=["session_key", "driver_number"], how="left")
    race = race.merge(pit_agg, on=["session_key", "driver_number"], how="left")
    race = race.merge(made, on=["session_key", "driver_number"], how="left")
    race = race.merge(lost, on=["session_key", "driver_number"], how="left")
    race = race.merge(champ[["session_key", "driver_number", "points_scored"]], on=["session_key", "driver_number"], how="left")

    race["grid_to_finish_gain"] = race["grid_position"] - race["position"]
    for col in ("pit_stop_count", "pit_stop_duration_avg_s", "overtakes_made", "overtakes_lost", "points_scored"):
        race[col] = race[col].fillna(0)

    return race[
        [
            "meeting_key", "driver_number", "year", "date_start", "position", "gap_to_leader", "dnf",
            "grid_to_finish_gain", "pit_stop_count", "pit_stop_duration_avg_s", "overtakes_made", "overtakes_lost", "points_scored",
        ]
    ]


_RACE_FORM_FEATURE_COLUMNS = [
    "recent_race_finish_position_avg",
    "season_race_finish_position_avg",
    "recent_race_points_avg",
    "season_race_points_avg",
    "recent_dnf_rate",
    "recent_grid_to_finish_gain_avg",
    "recent_pit_stop_count_avg",
    "recent_pit_stop_duration_avg_s",
    "recent_overtakes_made_avg",
    "recent_overtakes_lost_avg",
    "recent_race_gap_to_winner_avg_s",
]


def _add_race_form_features(quali: pd.DataFrame, conn: sqlite3.Connection, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling "how has this driver's racecraft/reliability been lately"
    features, built from PAST race weekends only — see _race_level_table's
    docstring for why that table alone isn't leakage-safe until shifted here.
    """
    race = _race_level_table(conn, sessions)
    if race.empty:
        for col in _RACE_FORM_FEATURE_COLUMNS:
            quali[col] = None
        return quali

    race = race.sort_values(["driver_number", "date_start"]).copy()
    g = race.groupby("driver_number")

    race["recent_race_finish_position_avg"] = g["position"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    race["recent_race_points_avg"] = g["points_scored"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    race["recent_dnf_rate"] = g["dnf"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    race["recent_grid_to_finish_gain_avg"] = g["grid_to_finish_gain"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    race["recent_pit_stop_count_avg"] = g["pit_stop_count"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    race["recent_pit_stop_duration_avg_s"] = g["pit_stop_duration_avg_s"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    race["recent_overtakes_made_avg"] = g["overtakes_made"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    race["recent_overtakes_lost_avg"] = g["overtakes_lost"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    race["recent_race_gap_to_winner_avg_s"] = g["gap_to_leader"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    race["season_race_finish_position_avg"] = race.groupby(["driver_number", "year"])["position"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    race["season_race_points_avg"] = race.groupby(["driver_number", "year"])["points_scored"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    return quali.merge(race[["meeting_key", "driver_number"] + _RACE_FORM_FEATURE_COLUMNS], on=["meeting_key", "driver_number"], how="left")


def _add_experience_features(quali: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Weekend scheduling / experience context. driver_career_sessions_count
    and days_since_last_session are proxies computed over QUALIFYING
    sessions only (not every session type) — a simplification, not a true
    career-session count, but a consistent, leakage-safe "how much recent
    experience" signal within the data this table already has in scope.
    """
    quali = quali.sort_values(["driver_number", "date_start"]).copy()
    quali["driver_career_sessions_count"] = quali.groupby("driver_number").cumcount()
    quali["days_since_last_session"] = quali.groupby("driver_number")["date_start"].diff().dt.total_seconds() / 86400

    meeting_order = sessions.sort_values("date_start").drop_duplicates("meeting_key")[["meeting_key", "year"]].copy()
    meeting_order["meeting_round_number"] = meeting_order.groupby("year").cumcount() + 1
    quali = quali.merge(meeting_order[["meeting_key", "meeting_round_number"]], on="meeting_key", how="left")

    sprint_meetings = set(sessions.loc[sessions["session_name"] == "Sprint", "meeting_key"])
    quali["is_sprint_weekend"] = quali["meeting_key"].isin(sprint_meetings).astype(float)
    return quali


def _add_practice_flag_features(quali: pd.DataFrame, conn: sqlite3.Connection, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Red/yellow flag counts during THIS weekend's practice sessions — known
    before qualifying happens (unlike flags during quali or the race
    itself), so safe to use as "how disrupted was this weekend's running".
    """
    practice_sessions = sessions[sessions["session_type"] == "Practice"][["session_key", "meeting_key"]]
    rc = pd.read_sql("SELECT session_key, flag FROM race_control", conn)
    rc = rc.merge(practice_sessions, on="session_key")

    red = rc[rc["flag"] == "RED"].groupby("meeting_key").size().rename("practice_red_flags_count")
    yellow = rc[rc["flag"].isin(["YELLOW", "DOUBLE YELLOW"])].groupby("meeting_key").size().rename("practice_yellow_flags_count")

    quali = quali.merge(red, on="meeting_key", how="left")
    quali = quali.merge(yellow, on="meeting_key", how="left")
    quali["practice_red_flags_count"] = quali["practice_red_flags_count"].fillna(0)
    quali["practice_yellow_flags_count"] = quali["practice_yellow_flags_count"].fillna(0)
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

    laps = pd.read_sql(
        "SELECT session_key, driver_number, lap_duration, is_pit_out_lap, duration_sector_1, duration_sector_2, duration_sector_3 FROM laps",
        conn,
    )
    best_laps = _best_lap_per_session(laps)

    quali = _qualifying_targets(sessions, best_laps)
    quali = _add_circuit_history_features(quali)
    quali = _add_form_features(quali)
    quali = _add_practice_features(quali, sessions, best_laps)
    quali = _add_practice_by_session_features(quali, sessions, best_laps, laps)
    quali = _add_sector_features(quali, sessions, laps)
    quali = _add_weather_features(quali, conn, sessions)
    quali = _add_compound_features(quali, conn, sessions)
    quali = _add_team(quali, conn)
    quali = _add_team_form_features(quali)
    quali = _add_teammate_features(quali)
    quali = _add_championship_features(quali, conn)
    quali = _add_team_championship_features(quali, conn)
    quali = _add_race_form_features(quali, conn, sessions)
    quali = _add_experience_features(quali, sessions)
    quali = _add_practice_flag_features(quali, conn, sessions)

    # Simplification (documented, not hidden): missing numeric values are
    # filled with the whole-table mean rather than a train-only mean like
    # the feature scaler in train.py uses. That's a tiny amount of leakage
    # (val/test rows influence the fill value a little) — acceptable for a
    # first baseline since missing values are rare (mostly reserve-driver
    # sessions, debut weekends, or drivers/tracks/teams with no prior
    # history yet). Upgrade path if it ever matters: move imputation into
    # train.py and fit it on the train split only, the same way the
    # numeric scaler already is.
    fill_cols = [
        "practice_best_time_s",
        "practice_gap_to_best_s",
        "practice_sessions_completed",
        "practice_fp1_best_time_s",
        "practice_fp1_gap_to_best_s",
        "practice_fp2_best_time_s",
        "practice_fp2_gap_to_best_s",
        "practice_fp3_best_time_s",
        "practice_fp3_gap_to_best_s",
        "practice_total_laps",
        "practice_best_sector1_gap_s",
        "practice_best_sector2_gap_s",
        "practice_best_sector3_gap_s",
        "circuit_history_avg_gap_to_pole_s",
        "circuit_history_best_gap_to_pole_s",
        "circuit_history_sessions_count",
        "team_recent_quali_gap_to_pole_avg_s",
        "team_season_quali_gap_to_pole_avg_s",
        "recent_teammate_quali_diff_avg_s",
        "air_temperature",
        "track_temperature",
        "track_temperature_range",
        "humidity",
        "rainfall",
        "wind_speed",
        "championship_position_entering_weekend",
        "championship_points_entering_weekend",
        "team_championship_position_entering_weekend",
        "team_championship_points_entering_weekend",
        "recent_race_finish_position_avg",
        "season_race_finish_position_avg",
        "recent_race_points_avg",
        "season_race_points_avg",
        "recent_dnf_rate",
        "recent_grid_to_finish_gain_avg",
        "recent_pit_stop_count_avg",
        "recent_pit_stop_duration_avg_s",
        "recent_overtakes_made_avg",
        "recent_overtakes_lost_avg",
        "recent_race_gap_to_winner_avg_s",
        "days_since_last_session",
        "meeting_round_number",
    ]
    quali[fill_cols] = quali[fill_cols].fillna(quali[fill_cols].mean())

    # A rookie's first-ever qualifying has no prior sessions to average, so
    # rolling/expanding form is NaN. Filling with the overall mean gap
    # assumes "average midfield pace, no information yet" — a more honest
    # default than filling with 0, which would assume pole-level pace.
    mean_gap = quali["gap_to_pole_s"].mean()
    quali["recent_quali_gap_to_pole_avg_s"] = quali["recent_quali_gap_to_pole_avg_s"].fillna(mean_gap)
    quali["season_quali_gap_to_pole_avg_s"] = quali["season_quali_gap_to_pole_avg_s"].fillna(mean_gap)
    quali["circuit_history_avg_gap_to_pole_s"] = quali["circuit_history_avg_gap_to_pole_s"].fillna(mean_gap)
    quali["circuit_history_best_gap_to_pole_s"] = quali["circuit_history_best_gap_to_pole_s"].fillna(mean_gap)
    quali["circuit_history_sessions_count"] = quali["circuit_history_sessions_count"].fillna(0)
    quali["recent_teammate_quali_diff_avg_s"] = quali["recent_teammate_quali_diff_avg_s"].fillna(0)

    # Final safety net, not the normal path: quali[fill_cols].mean() is itself
    # NaN for any column with ZERO real values anywhere in this build (e.g. a
    # --limit-truncated smoke-test ingest with no repeat circuit visits at
    # all) — fillna(NaN) is then a no-op and NaN would reach the network.
    # 0 isn't semantically meaningful for every column here, but this only
    # ever fires in that degenerate small-data case, never on a real ingest.
    quali[NUMERIC_FEATURE_COLUMNS] = quali[NUMERIC_FEATURE_COLUMNS].fillna(0)

    id_cols = ["session_key", "driver_number", "date_start", "year", "circuit_short_name", "team_name"]
    return quali[id_cols + ["target_lap_time_s"] + NUMERIC_FEATURE_COLUMNS].reset_index(drop=True)
