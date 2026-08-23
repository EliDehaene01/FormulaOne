"""
The tools the LLM can call. Each one is a plain Python function that reads
from the local SQLite cache (or the trained model checkpoint) and returns
JSON-serializable data — the LLM never touches the database directly, it
asks us to run one of these and we hand back the result.

WHY TOOL CALLING INSTEAD OF RAG
-----------------------------------
RAG (retrieval-augmented generation) makes sense when your knowledge is
unstructured text and you need to find the most SEMANTICALLY SIMILAR
chunk to a question. Our data is the opposite: structured, tabular, and
exactly queryable — "driver 44's lap times in session 9468" isn't a
similarity search, it's a SQL WHERE clause. Tool calling lets the model
ask for precisely that, get back exact numbers, and reason over them —
instead of hoping the right chunk of embedded text happened to mention
the right lap time.

Every tool function is deliberately a thin, stateless wrapper: open a
connection, run one or two queries/pandas operations, close it, return
plain dicts/lists. No shared state between calls — simplest thing that
works at this scale (the cache is a local SQLite file, not a service under
load).
"""

import sqlite3

import pandas as pd

from backend.data.cache import DB_PATH


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def search_knowledge_base(query: str, k: int = 3) -> list[dict]:
    """
    Semantic search over Pit Wall's F1 knowledge base (glossary terms, tyre
    strategy concepts, regulations/race control rules, circuit guides) — for
    CONCEPTUAL/rules questions, never race data (which the other tools own).
    """
    from backend.rag.store import search_knowledge_base as _search

    return _search(query, k)


def get_meetings(year: int | None = None) -> list[dict]:
    """Race weekends (Grand Prix or pre-season testing) — name, location, circuit, dates. Use find_sessions instead when you need to resolve a specific race into a session_key; this is for browsing weekends themselves."""
    conn = _connect()
    query = "SELECT meeting_key, meeting_name, meeting_official_name, location, country_name, circuit_short_name, year, date_start FROM meetings"
    params: list = []
    if year is not None:
        query += " WHERE year = ?"
        params.append(year)
    query += " ORDER BY date_start"
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df.to_dict("records")


def get_car_telemetry(session_key: int, driver_number: int, lap_number: int) -> dict:
    """
    Speed/throttle/brake/RPM/gear/DRS telemetry for one driver's one lap,
    summarized (not the raw ~3.7Hz samples, which would burn a lot of
    tokens for one lap). Same "live, bounded to one lap" approach as
    get_car_location — see that function's docstring for why this isn't
    bulk-cached.
    """
    from backend.data.openf1_client import get_car_data

    conn = _connect()
    laps = pd.read_sql(
        "SELECT date_start, lap_duration FROM laps WHERE session_key = ? AND driver_number = ? AND lap_number = ?",
        conn,
        params=[session_key, driver_number, lap_number],
    )
    conn.close()
    if laps.empty:
        return {"error": f"no lap {lap_number} recorded for driver {driver_number} in session {session_key}"}

    start = pd.Timestamp(laps.iloc[0]["date_start"])
    duration = laps.iloc[0]["lap_duration"]
    seconds = float(duration) if pd.notna(duration) else 180.0
    end = start + pd.Timedelta(seconds=seconds)

    samples = get_car_data(session_key, driver_number, start.isoformat(), end.isoformat())
    if not samples:
        return {"session_key": session_key, "driver_number": driver_number, "lap_number": lap_number, "note": "no telemetry samples available for this lap"}

    df = pd.DataFrame(samples)
    return {
        "session_key": session_key,
        "driver_number": driver_number,
        "lap_number": lap_number,
        "sample_count": len(df),
        "top_speed_kph": float(df["speed"].max()) if "speed" in df else None,
        "avg_speed_kph": round(float(df["speed"].mean()), 1) if "speed" in df else None,
        "avg_throttle_pct": round(float(df["throttle"].mean()), 1) if "throttle" in df else None,
        "time_braking_pct": round(float((df["brake"] > 0).mean() * 100), 1) if "brake" in df else None,
        "drs_active_pct": round(float((df["drs"] > 0).mean() * 100), 1) if "drs" in df else None,
    }


def find_sessions(
    year: int | None = None,
    meeting_name: str | None = None,
    session_type: str | None = None,
    session_name: str | None = None,
) -> list[dict]:
    """
    Resolve a race the user mentions by name/year into the session_key every
    other tool needs.

    Two filters that look similar but aren't: `session_type` is one of the
    3 broad OpenF1 categories ("Practice", "Qualifying", "Race") — but
    "Race" also matches Sprint races, and "Qualifying" also matches Sprint
    Qualifying. `session_name` is the precise one ("Race" vs "Sprint",
    "Qualifying" vs "Sprint Qualifying") — pass this when you specifically
    want the main event, not the sprint weekend variant.

    Only returns sessions we actually have lap data for — the `sessions`
    table itself includes every session OpenF1 has ever scheduled,
    including ones this project's cache never fetched practice/lap data
    for (or that haven't happened yet), which would otherwise look
    selectable but return empty data from every other tool.
    """
    conn = _connect()
    query = (
        "SELECT s.session_key, s.session_name, s.session_type, s.year, s.date_start, "
        "s.circuit_short_name, m.meeting_name "
        "FROM sessions s LEFT JOIN meetings m ON s.meeting_key = m.meeting_key "
        "WHERE s.session_key IN (SELECT DISTINCT session_key FROM laps)"
    )
    params: list = []
    if year is not None:
        query += " AND s.year = ?"
        params.append(year)
    if session_type is not None:
        query += " AND s.session_type = ?"
        params.append(session_type)
    if session_name is not None:
        query += " AND s.session_name = ?"
        params.append(session_name)
    if meeting_name is not None:
        query += " AND m.meeting_name LIKE ?"
        params.append(f"%{meeting_name}%")
    query += " ORDER BY s.date_start"

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df.to_dict("records")


def get_lap_times(session_key: int, driver_number: int | None = None) -> list[dict]:
    """Lap-by-lap times for a session, optionally for one driver."""
    conn = _connect()
    query = (
        "SELECT driver_number, lap_number, lap_duration, duration_sector_1, duration_sector_2, "
        "duration_sector_3, is_pit_out_lap FROM laps WHERE session_key = ?"
    )
    params: list = [session_key]
    if driver_number is not None:
        query += " AND driver_number = ?"
        params.append(driver_number)
    query += " ORDER BY driver_number, lap_number"

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df.to_dict("records")


def get_position_changes(session_key: int, driver_number: int | None = None) -> list[dict]:
    """Track position over time. Prefer passing driver_number for a full race — unfiltered can be a few hundred rows."""
    conn = _connect()
    query = "SELECT driver_number, date, position FROM position WHERE session_key = ?"
    params: list = [session_key]
    if driver_number is not None:
        query += " AND driver_number = ?"
        params.append(driver_number)
    query += " ORDER BY date"

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df.to_dict("records")


def get_tire_strategy(session_key: int, driver_number: int | None = None) -> list[dict]:
    """Tire stints: which compound each driver ran and over which lap range."""
    conn = _connect()
    query = (
        "SELECT driver_number, stint_number, compound, lap_start, lap_end, tyre_age_at_start "
        "FROM stints WHERE session_key = ?"
    )
    params: list = [session_key]
    if driver_number is not None:
        query += " AND driver_number = ?"
        params.append(driver_number)
    query += " ORDER BY driver_number, stint_number"

    df = pd.read_sql(query, conn, params=params)
    conn.close()

    # OpenF1 data quality gap, found by actually running this against
    # Monaco 2024: a handful of drivers' FIRST stint has null lap_start/
    # lap_end (likely early-race incidents — Monaco 2024 had a first-lap
    # crash). NaN isn't valid JSON, so this would otherwise break every
    # endpoint that returns stint data. Stint 1 always starts at lap 1;
    # lacking any better signal for when it ended, lap_end falls back to
    # lap_start too (renders as a zero-length stint — honest about "we
    # don't know how long this lasted" rather than fabricating a number).
    df["lap_start"] = df["lap_start"].fillna(1)
    df["lap_end"] = df["lap_end"].fillna(df["lap_start"])

    return df.to_dict("records")


def get_drivers(session_key: int) -> list[dict]:
    """Driver number -> name/team lookup for a session. Call this before referring to a driver by name — don't guess names from memory."""
    conn = _connect()
    df = pd.read_sql(
        "SELECT driver_number, full_name, name_acronym, team_name, team_colour FROM drivers WHERE session_key = ? ORDER BY driver_number",
        conn,
        params=[session_key],
    )
    conn.close()
    return df.to_dict("records")


def get_car_location(session_key: int, driver_number: int, lap_number: int) -> list[dict]:
    """
    x/y car coordinates for one driver's one lap — chart-only helper, not
    an LLM tool (spatial coordinates aren't useful for the model to reason
    over in text). A full lap's trace IS the track layout; the frontend
    plots it directly rather than needing a separate "track shape" source.

    Not from the cache — OpenF1's /location endpoint is called live, bounded
    to just this lap's time window (position data at ~3.7Hz would be tens
    of thousands of rows per session if bulk-cached like the other
    endpoints).
    """
    from backend.data.openf1_client import get_location

    conn = _connect()
    laps = pd.read_sql(
        "SELECT date_start, lap_duration FROM laps WHERE session_key = ? AND driver_number = ? AND lap_number = ?",
        conn,
        params=[session_key, driver_number, lap_number],
    )
    conn.close()
    if laps.empty:
        return []

    start = pd.Timestamp(laps.iloc[0]["date_start"])
    duration = laps.iloc[0]["lap_duration"]
    # Some laps have no recorded duration (red flag, DNF mid-lap) — fall
    # back to a generous 3-minute window rather than failing outright.
    seconds = float(duration) if pd.notna(duration) else 180.0
    end = start + pd.Timedelta(seconds=seconds)

    points = get_location(session_key, driver_number, start.isoformat(), end.isoformat())
    return [{"x": p["x"], "y": p["y"]} for p in points]


def get_car_telemetry_series(session_key: int, driver_number: int, lap_number: int) -> list[dict]:
    """
    Raw speed/throttle/brake samples for one lap — chart-only helper (not an
    LLM tool; get_car_telemetry's summary is what the model gets). Same
    live/bounded-per-lap approach as get_car_location.
    """
    from backend.data.openf1_client import get_car_data

    conn = _connect()
    laps = pd.read_sql(
        "SELECT date_start, lap_duration FROM laps WHERE session_key = ? AND driver_number = ? AND lap_number = ?",
        conn,
        params=[session_key, driver_number, lap_number],
    )
    conn.close()
    if laps.empty:
        return []

    start = pd.Timestamp(laps.iloc[0]["date_start"])
    duration = laps.iloc[0]["lap_duration"]
    seconds = float(duration) if pd.notna(duration) else 180.0
    end = start + pd.Timedelta(seconds=seconds)

    samples = get_car_data(session_key, driver_number, start.isoformat(), end.isoformat())
    return [
        {
            "date": s.get("date"),
            "speed": s.get("speed"),
            "throttle": s.get("throttle"),
            "brake": s.get("brake"),
        }
        for s in samples
    ]


def get_weather(session_key: int) -> dict:
    """
    Summarized, not raw — a session can have hundreds of individual weather
    samples (one every minute or so); handing all of them to the model
    would burn tokens for no benefit when "was it raining, how hot" is
    what actually gets asked.
    """
    conn = _connect()
    df = pd.read_sql(
        "SELECT air_temperature, track_temperature, humidity, rainfall, wind_speed FROM weather WHERE session_key = ?",
        conn,
        params=[session_key],
    )
    conn.close()
    if df.empty:
        return {"session_key": session_key, "note": "no weather data recorded for this session"}
    return {
        "session_key": session_key,
        "air_temperature_avg_c": round(df["air_temperature"].mean(), 1),
        "track_temperature_avg_c": round(df["track_temperature"].mean(), 1),
        "humidity_avg_pct": round(df["humidity"].mean(), 1),
        "rain_detected": bool((df["rainfall"] > 0).any()),
        "wind_speed_avg_ms": round(df["wind_speed"].mean(), 1),
    }


def get_weather_timeseries(session_key: int) -> list[dict]:
    """
    Raw weather samples over time — chart-only helper (not registered as an
    LLM tool; get_weather's summary is what the model gets, to keep its
    context small). Used by the frontend's weather-over-the-session chart.
    """
    conn = _connect()
    df = pd.read_sql(
        "SELECT date, air_temperature, track_temperature, rainfall FROM weather WHERE session_key = ? ORDER BY date",
        conn,
        params=[session_key],
    )
    conn.close()
    return df.to_dict("records")


def get_driver_lap_stats(session_key: int) -> list[dict]:
    """
    Per-driver aggregate lap stats (fastest/average/consistency/top speed) —
    chart-only helper, not registered as an LLM tool (get_lap_times already
    covers the same underlying data for the model; this is a specific
    aggregation shape for the frontend's bar charts).
    """
    conn = _connect()
    laps = pd.read_sql(
        "SELECT driver_number, lap_duration, is_pit_out_lap, st_speed FROM laps WHERE session_key = ?",
        conn,
        params=[session_key],
    )
    conn.close()

    clean = laps[(laps["is_pit_out_lap"] == 0) & laps["lap_duration"].notna()]
    stats = clean.groupby("driver_number").agg(
        fastest_lap_s=("lap_duration", "min"),
        average_lap_s=("lap_duration", "mean"),
        consistency_std_s=("lap_duration", "std"),
        top_speed_kph=("st_speed", "max"),
    )
    stats = stats.round(3).fillna(0).reset_index()
    return stats.to_dict("records")


def get_race_control_messages(session_key: int) -> list[dict]:
    """Flags, safety cars, penalties, and other official messages, in order."""
    conn = _connect()
    df = pd.read_sql(
        "SELECT date, category, message, flag, lap_number, driver_number FROM race_control WHERE session_key = ? ORDER BY date",
        conn,
        params=[session_key],
    )
    conn.close()
    return df.to_dict("records")


def get_pit_stops(session_key: int, driver_number: int | None = None) -> list[dict]:
    """Pit lane visits: which lap, and how long the car spent in the pit lane / stationary."""
    conn = _connect()
    # OpenF1 renamed pit_duration -> lane_duration at some point but kept
    # pit_duration as a deprecated alias with the same value — select *
    # and coalesce rather than hardcode one name that might not be the
    # column this particular cache actually has.
    df = pd.read_sql("SELECT * FROM pit WHERE session_key = ?", conn, params=[session_key])
    conn.close()
    if df.empty:
        return []
    if driver_number is not None:
        df = df[df["driver_number"] == driver_number]
    df["pit_lane_duration_s"] = df["pit_duration"] if "pit_duration" in df.columns else df.get("lane_duration")
    if "lane_duration" in df.columns:
        df["pit_lane_duration_s"] = df["pit_lane_duration_s"].fillna(df["lane_duration"])
    cols = [c for c in ["driver_number", "lap_number", "pit_lane_duration_s", "stop_duration"] if c in df.columns or c == "pit_lane_duration_s"]
    return df.sort_values(["driver_number", "lap_number"])[cols].to_dict("records")


def get_intervals(session_key: int, driver_number: int | None = None) -> list[dict]:
    """Real-time gap to the driver ahead and to the race leader, over time (Race/Sprint sessions only). Prefer passing driver_number — unfiltered can be a few hundred rows."""
    conn = _connect()
    query = "SELECT driver_number, date, gap_to_leader, interval FROM intervals WHERE session_key = ?"
    params: list = [session_key]
    if driver_number is not None:
        query += " AND driver_number = ?"
        params.append(driver_number)
    query += " ORDER BY date"
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df.to_dict("records")


def get_overtakes(session_key: int) -> list[dict]:
    """On-track and pit-related position exchanges between two drivers, in chronological order."""
    conn = _connect()
    df = pd.read_sql(
        "SELECT date, overtaking_driver_number, overtaken_driver_number, position FROM overtakes WHERE session_key = ? ORDER BY date",
        conn,
        params=[session_key],
    )
    conn.close()
    return df.to_dict("records")


def get_team_radio(session_key: int, driver_number: int | None = None) -> list[dict]:
    """Driver-to-pit-wall radio message timestamps + recording URLs. The model can't listen to the audio — only note when a message happened and to/from whom."""
    conn = _connect()
    query = "SELECT driver_number, date, recording_url FROM team_radio WHERE session_key = ?"
    params: list = [session_key]
    if driver_number is not None:
        query += " AND driver_number = ?"
        params.append(driver_number)
    query += " ORDER BY date"
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df.to_dict("records")


def get_session_result(session_key: int) -> list[dict]:
    """Final classification: position, race/best-lap duration, gap to leader, DNF/DNS/DSQ flags."""
    conn = _connect()
    df = pd.read_sql(
        "SELECT driver_number, position, duration, gap_to_leader, number_of_laps, dnf, dns, dsq "
        "FROM session_result WHERE session_key = ? ORDER BY position",
        conn,
        params=[session_key],
    )
    conn.close()
    return df.to_dict("records")


def get_starting_grid(session_key: int) -> list[dict]:
    """Starting grid order for a Race/Sprint session, with the qualifying lap time that earned each slot."""
    conn = _connect()
    df = pd.read_sql(
        "SELECT driver_number, position, lap_duration FROM starting_grid WHERE session_key = ? ORDER BY position",
        conn,
        params=[session_key],
    )
    conn.close()
    return df.to_dict("records")


def get_driver_championship_standings(session_key: int) -> list[dict]:
    """Drivers'-championship standing before and after this session."""
    conn = _connect()
    df = pd.read_sql(
        "SELECT driver_number, position_start, position_current, points_start, points_current "
        "FROM championship_drivers WHERE session_key = ? ORDER BY position_current",
        conn,
        params=[session_key],
    )
    conn.close()
    return df.to_dict("records")


def get_team_championship_standings(session_key: int) -> list[dict]:
    """Constructors'-championship standing before and after this session."""
    conn = _connect()
    df = pd.read_sql(
        "SELECT team_name, position_start, position_current, points_start, points_current "
        "FROM championship_teams WHERE session_key = ? ORDER BY position_current",
        conn,
        params=[session_key],
    )
    conn.close()
    return df.to_dict("records")


def get_driver_season_summary(driver_number: int, year: int) -> dict:
    """Aggregate stats for a driver's season: finishing positions and qualifying pace relative to pole."""
    from backend.models.features import _best_lap_per_session  # reuse the same "clean fastest lap" logic training uses

    conn = _connect()
    sessions = pd.read_sql(
        "SELECT session_key, session_name, session_type FROM sessions WHERE year = ?", conn, params=[year]
    )
    races = sessions[(sessions["session_type"] == "Race") & (sessions["session_name"] == "Race")]
    quali = sessions[(sessions["session_type"] == "Qualifying") & (sessions["session_name"] == "Qualifying")]

    positions = pd.read_sql(
        "SELECT session_key, position, date FROM position WHERE driver_number = ?", conn, params=[driver_number]
    )
    race_positions = positions.merge(races[["session_key"]], on="session_key")
    # "Final classified position" = the last position update recorded in that race — a proxy, not
    # an official result (doesn't distinguish a DNF from a clean finish in P20).
    final_positions = race_positions.sort_values("date").groupby("session_key").tail(1)

    laps = pd.read_sql("SELECT session_key, driver_number, lap_duration, is_pit_out_lap FROM laps", conn)
    conn.close()

    best_laps = _best_lap_per_session(laps)
    quali_best = best_laps.merge(quali[["session_key"]], on="session_key")
    pole_times = quali_best.groupby("session_key")["best_lap_s"].min()
    driver_quali = quali_best[quali_best["driver_number"] == driver_number].set_index("session_key")["best_lap_s"]
    gaps = (driver_quali - pole_times.reindex(driver_quali.index)).dropna()

    return {
        "driver_number": driver_number,
        "year": year,
        "races_with_recorded_finish": int(final_positions.shape[0]),
        "average_finishing_position": round(final_positions["position"].mean(), 1) if not final_positions.empty else None,
        "best_finishing_position": int(final_positions["position"].min()) if not final_positions.empty else None,
        "qualifying_sessions_recorded": int(len(driver_quali)),
        "average_qualifying_gap_to_pole_seconds": round(gaps.mean(), 3) if not gaps.empty else None,
    }


def predict_qualifying_pace(session_key: int) -> dict:
    """
    Runs the trained PyTorch qualifying-pace model for a session already in
    the cache. Returns a predicted time per driver and the predicted pole
    sitter, alongside the actual result for comparison.

    LIMITATION: only works for sessions the cache already has BOTH practice
    and qualifying data for (i.e. backtesting an already-run weekend) — it
    can't yet predict a genuinely upcoming qualifying session before it
    happens. See backend/models/README.md for why and the upgrade path.
    """
    # Lazy import: torch doesn't import natively on this dev machine outside
    # WSL (see backend/models/README.md) — keeping the import inside this
    # function means every OTHER tool in this file keeps working from a
    # plain Windows Python process; only this specific tool needs the
    # WSL/torch environment, and only once it's actually called.
    import torch

    from backend.models.features import NUMERIC_FEATURE_COLUMNS, build_feature_table
    from backend.models.model import QualifyingLapTimePredictor
    from backend.models.train import CHECKPOINT_PATH, _apply_category_index

    if not CHECKPOINT_PATH.exists():
        return {"error": "no trained model checkpoint found — run `python -m backend.models.train` first"}

    # weights_only=False: safe here because this checkpoint is one we wrote
    # ourselves (see train.py), not a file from an untrusted source — it
    # contains plain dicts and pandas Series (the category maps and scaler
    # stats) alongside the tensor weights, which PyTorch's default
    # weights_only=True loading would refuse to unpickle.
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

    conn = _connect()
    df = build_feature_table(conn)
    conn.close()
    session_df = df[df["session_key"] == session_key]
    if session_df.empty:
        return {"error": f"session {session_key} has no cached practice+qualifying data to predict from"}

    model = QualifyingLapTimePredictor(
        num_drivers=len(checkpoint["driver_map"]),
        num_teams=len(checkpoint["team_map"]),
        num_circuits=len(checkpoint["circuit_map"]),
        num_numeric_features=len(checkpoint["numeric_feature_columns"]),
        # Must match training exactly or load_state_dict below fails on a
        # shape mismatch — tune.py can pick different hidden_sizes per run,
        # so this can't rely on the constructor's own default anymore.
        hidden_sizes=tuple(checkpoint.get("hidden_sizes", (128, 64, 32))),
        dropout=checkpoint.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    driver_idx = torch.tensor(_apply_category_index(session_df["driver_number"], checkpoint["driver_map"]), dtype=torch.long)
    team_idx = torch.tensor(_apply_category_index(session_df["team_name"], checkpoint["team_map"]), dtype=torch.long)
    circuit_idx = torch.tensor(_apply_category_index(session_df["circuit_short_name"], checkpoint["circuit_map"]), dtype=torch.long)
    numeric = (session_df[NUMERIC_FEATURE_COLUMNS] - checkpoint["numeric_mean"]) / checkpoint["numeric_std"]
    numeric_tensor = torch.tensor(numeric.clip(lower=-5, upper=5).to_numpy(), dtype=torch.float32)

    with torch.no_grad():
        predictions = model(driver_idx, team_idx, circuit_idx, numeric_tensor).numpy()

    results = session_df[["driver_number", "target_lap_time_s"]].rename(columns={"target_lap_time_s": "actual_lap_time_s"}).copy()
    results["predicted_lap_time_s"] = predictions.round(3)
    results = results.sort_values("predicted_lap_time_s")

    predicted_pole = results.iloc[0]
    actual_pole = results.sort_values("actual_lap_time_s").iloc[0]

    return {
        "session_key": session_key,
        "predicted_pole_driver_number": int(predicted_pole["driver_number"]),
        "predicted_pole_time_s": float(predicted_pole["predicted_lap_time_s"]),
        "actual_pole_driver_number": int(actual_pole["driver_number"]),
        "actual_pole_time_s": float(actual_pole["actual_lap_time_s"]),
        "per_driver_predictions": results.to_dict("records"),
        "caveat": "This first-pass model has not yet been shown to beat a simple baseline (see backend/models/README.md) — treat it as a rough, uncertain estimate, not a reliable forecast.",
    }


def explain_qualifying_prediction(session_key: int, driver_number: int, top_n: int = 8) -> dict:
    """
    Explains ONE driver's qualifying prediction: how much each input (driver
    identity, team identity, circuit identity, and individual numeric
    features like practice pace) pushed the predicted lap time up or down,
    via Captum Integrated Gradients against a neutral baseline (unknown
    driver/team/circuit, average conditions). Call this when the user asks
    WHY the model predicted what it did — not on every prediction request.

    LIMITATION: same underlying model as predict_qualifying_pace — see that
    caveat and backend/models/README.md. Explaining an unreliable model's
    reasoning is still useful (it shows what it actually leaned on), but
    isn't a claim that the reasoning is sound.
    """
    from backend.models.explain import explain_prediction
    from backend.models.train import CHECKPOINT_PATH

    if not CHECKPOINT_PATH.exists():
        return {"error": "no trained model checkpoint found — run `python -m backend.models.train` first"}

    try:
        result = explain_prediction(session_key, driver_number)
    except ValueError as exc:
        return {"error": str(exc)}

    result["contributions"] = result["contributions"][:top_n]
    result["caveat"] = (
        "This first-pass model has not yet been shown to beat a simple baseline (see backend/models/README.md) — "
        "these attributions show what the model leaned on, not proof its reasoning is sound. Frame as an estimate, never certainty."
    )
    return result


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Semantic search over Pit Wall's F1 knowledge base: glossary terms, tyre strategy concepts, "
                "regulations & race control rules, and circuit characteristics. Use this ONLY for conceptual/rules "
                "questions in plain language (e.g. \"what's an undercut\", \"why can't you overtake under blue "
                "flags\", \"why is Monaco hard to pass at\") — never for race data like lap times, positions, "
                "standings, or session results, which always come from the data tools instead. Don't call this on "
                "every message, only when the question genuinely needs conceptual grounding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The concept or rule being asked about, in plain language."},
                    "k": {"type": "integer", "description": "Number of chunks to retrieve (default 3)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_meetings",
            "description": "Race weekends (Grand Prix or pre-season testing) — name, location, circuit, dates. Use find_sessions instead to resolve a specific race into a session_key.",
            "parameters": {
                "type": "object",
                "properties": {"year": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_car_location",
            "description": "x/y track coordinates for one driver's one lap — useful for describing the racing line or where on track something happened.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_key": {"type": "integer"},
                    "driver_number": {"type": "integer"},
                    "lap_number": {"type": "integer"},
                },
                "required": ["session_key", "driver_number", "lap_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_car_telemetry",
            "description": "Summarized speed/throttle/brake/DRS telemetry for one driver's one lap (top/avg speed, % time braking, % DRS active).",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_key": {"type": "integer"},
                    "driver_number": {"type": "integer"},
                    "lap_number": {"type": "integer"},
                },
                "required": ["session_key", "driver_number", "lap_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_sessions",
            "description": "Find race weekend sessions we have data for, by year, Grand Prix/meeting name, or session type/name. Use this first to resolve a race the user mentions by name into a session_key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "Season year, e.g. 2024"},
                    "meeting_name": {"type": "string", "description": "Partial or full Grand Prix name, e.g. 'Bahrain', 'Monaco'"},
                    "session_type": {"type": "string", "enum": ["Practice", "Qualifying", "Race"], "description": "Broad category — 'Race' also matches Sprint races, 'Qualifying' also matches Sprint Qualifying"},
                    "session_name": {"type": "string", "description": "Precise session, e.g. 'Race', 'Sprint', 'Qualifying', 'Sprint Qualifying' — use this to exclude sprint-weekend variants"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_drivers",
            "description": "Driver number -> full name/team lookup for a session. Call this before naming a driver in your answer — never guess a name from memory.",
            "parameters": {
                "type": "object",
                "properties": {"session_key": {"type": "integer"}},
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lap_times",
            "description": "Lap-by-lap times for a session (sector times, lap duration, out-lap flag), optionally for one driver.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_key": {"type": "integer"},
                    "driver_number": {"type": "integer", "description": "Optional: limit to one driver"},
                },
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_position_changes",
            "description": "Track position over time for a session. Pass driver_number when asking about one driver's race — unfiltered can be a few hundred rows for a full field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_key": {"type": "integer"},
                    "driver_number": {"type": "integer"},
                },
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tire_strategy",
            "description": "Tire stints for a session: which compound each driver ran and over which lap range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_key": {"type": "integer"},
                    "driver_number": {"type": "integer"},
                },
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Summarized weather for a session: average temperatures, humidity, whether it rained, wind speed.",
            "parameters": {
                "type": "object",
                "properties": {"session_key": {"type": "integer"}},
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_race_control_messages",
            "description": "Official race control messages for a session: flags, safety cars, penalties, incidents, in chronological order.",
            "parameters": {
                "type": "object",
                "properties": {"session_key": {"type": "integer"}},
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_driver_season_summary",
            "description": "A driver's aggregate stats for a season: average/best finishing position, average qualifying gap to pole.",
            "parameters": {
                "type": "object",
                "properties": {
                    "driver_number": {"type": "integer"},
                    "year": {"type": "integer"},
                },
                "required": ["driver_number", "year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pit_stops",
            "description": "Pit lane visits for a session: which lap, and how long the car spent in the pit lane / stationary.",
            "parameters": {
                "type": "object",
                "properties": {"session_key": {"type": "integer"}, "driver_number": {"type": "integer"}},
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_intervals",
            "description": "Real-time gap to the driver ahead and to the race leader over time (Race/Sprint sessions only). Pass driver_number for a full race — unfiltered can be a few hundred rows.",
            "parameters": {
                "type": "object",
                "properties": {"session_key": {"type": "integer"}, "driver_number": {"type": "integer"}},
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_overtakes",
            "description": "On-track and pit-related position exchanges between two drivers, in chronological order.",
            "parameters": {
                "type": "object",
                "properties": {"session_key": {"type": "integer"}},
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_radio",
            "description": "Driver-to-pit-wall radio message timestamps for a session. You cannot listen to the audio — only report when a message happened and who it involved.",
            "parameters": {
                "type": "object",
                "properties": {"session_key": {"type": "integer"}, "driver_number": {"type": "integer"}},
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_result",
            "description": "Final classification for a session: position, race/best-lap duration, gap to leader, DNF/DNS/DSQ flags.",
            "parameters": {
                "type": "object",
                "properties": {"session_key": {"type": "integer"}},
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_starting_grid",
            "description": "Starting grid order for a Race/Sprint session, with the qualifying lap time that earned each slot.",
            "parameters": {
                "type": "object",
                "properties": {"session_key": {"type": "integer"}},
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_driver_championship_standings",
            "description": "Drivers'-championship standing (position + points) before and after a session.",
            "parameters": {
                "type": "object",
                "properties": {"session_key": {"type": "integer"}},
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_championship_standings",
            "description": "Constructors'-championship standing (position + points) before and after a session.",
            "parameters": {
                "type": "object",
                "properties": {"session_key": {"type": "integer"}},
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_qualifying_pace",
            "description": "Run the trained model's qualifying pace prediction for a session (only works for sessions with cached practice+qualifying data — good for 'what did the model predict for this race', not yet for genuinely future races). Always present the result as an uncertain estimate, never as fact.",
            "parameters": {
                "type": "object",
                "properties": {"session_key": {"type": "integer"}},
                "required": ["session_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_qualifying_prediction",
            "description": (
                "Explain WHY the qualifying model predicted what it did for one driver in one session — which "
                "inputs (driver identity, team identity, circuit identity, practice pace, weather, form, etc.) "
                "pushed the predicted lap time up or down, ranked by influence. Call this only when the user asks "
                "for the reasoning behind a prediction (e.g. 'why do you think he'll be fastest'), not on every "
                "prediction request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_key": {"type": "integer"},
                    "driver_number": {"type": "integer"},
                },
                "required": ["session_key", "driver_number"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "search_knowledge_base": search_knowledge_base,
    "get_meetings": get_meetings,
    "get_car_location": get_car_location,
    "get_car_telemetry": get_car_telemetry,
    "find_sessions": find_sessions,
    "get_drivers": get_drivers,
    "get_lap_times": get_lap_times,
    "get_position_changes": get_position_changes,
    "get_tire_strategy": get_tire_strategy,
    "get_weather": get_weather,
    "get_race_control_messages": get_race_control_messages,
    "get_pit_stops": get_pit_stops,
    "get_intervals": get_intervals,
    "get_overtakes": get_overtakes,
    "get_team_radio": get_team_radio,
    "get_session_result": get_session_result,
    "get_starting_grid": get_starting_grid,
    "get_driver_championship_standings": get_driver_championship_standings,
    "get_team_championship_standings": get_team_championship_standings,
    "get_driver_season_summary": get_driver_season_summary,
    "predict_qualifying_pace": predict_qualifying_pace,
    "explain_qualifying_prediction": explain_qualifying_prediction,
}
