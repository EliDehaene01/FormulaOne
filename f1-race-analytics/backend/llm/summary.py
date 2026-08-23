"""
Race summary generator: deterministic Python assembles the data + chart
payloads for one race, then exactly ONE LLM call writes the narrative
recap, grounded strictly in what we hand it.

WHY NOT THE TOOL-CALLING LOOP FROM client.py
------------------------------------------------
Tool calling exists because in open-ended chat we don't know in
advance what the user will ask, so the model has to decide what data it
needs and request it, possibly over several round trips. A race summary is
the opposite: a FIXED task — we already know exactly what a good recap
needs (race control events, who gained/lost the most positions, the
fastest lap, the pit strategy). Precomputing that ourselves and handing it
to the model in one shot is simpler, cheaper (one API call instead of
several tool round trips), and more reliable: the model can't forget to
check something we've already guaranteed is sitting in front of it.

STRUCTURED OUTPUTS
----------------------
The LLM call below uses `response_format={"type": "json_schema", ...}`
instead of asking for free text and hoping it comes back parseable. This
constrains the model's response to match an exact JSON shape (here:
`{"headline": str, "narrative": str}`) — the API itself guarantees the
fields exist and are the right type, so downstream code doesn't need
fragile regex/string-splitting to pull a headline out of prose. This is
the right tool for "I need this LLM output to plug into a UI"; it would be
the WRONG tool for the Phase 3 chat loop, where we want the model free to
just talk.

CHART DATA NEVER GOES THROUGH THE LLM
------------------------------------------
Lap times, positions, and stints are exact numbers from the cache — the
LLM only ever sees a compact SUMMARY of them (key moments + race control +
strategy), not the raw per-lap data, both to keep the prompt small and
because there's no reason to let a language model "retype" numbers we
already have exactly right.
"""

import json
import os

import pandas as pd

from backend.llm.client import SYSTEM_PROMPT, _get_client
from backend.llm.tools import _connect, get_drivers, get_position_changes, get_race_control_messages, get_tire_strategy

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "race_summary",
        "schema": {
            "type": "object",
            "properties": {
                "headline": {"type": "string", "description": "A punchy one-line headline for the race"},
                "narrative": {
                    "type": "string",
                    "description": (
                        "A long, detailed recap — roughly one A4 page (about 600-800 words, "
                        "8-12 paragraphs). Cover the race chronologically: the start and any "
                        "early incidents, how the tyre strategies played out stint by stint for "
                        "the notable drivers, key overtakes and position changes, every race "
                        "control event worth mentioning (flags, penalties, safety cars), and a "
                        "closing summary of the result. Don't pad with repetition — go into more "
                        "specific, concrete detail than a short recap would, using the real data "
                        "provided."
                    ),
                },
            },
            "required": ["headline", "narrative"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def _extreme(series: pd.Series, kind: str) -> dict | None:
    """Picks the biggest (kind='max') or worst (kind='min') mover from a driver_number-indexed series."""
    if series.empty:
        return None
    idx = series.idxmax() if kind == "max" else series.idxmin()
    return {"driver_number": int(idx), "positions_changed": int(series[idx])}


def _compute_key_moments(session_key: int, laps: pd.DataFrame) -> dict:
    conn = _connect()
    positions = pd.read_sql(
        "SELECT driver_number, date, position FROM position WHERE session_key = ? ORDER BY date",
        conn, params=[session_key],
    )
    stints = pd.read_sql("SELECT driver_number, stint_number FROM stints WHERE session_key = ?", conn, params=[session_key])
    conn.close()

    clean_laps = laps[(laps["is_pit_out_lap"] == 0) & laps["lap_duration"].notna()]
    fastest = None if clean_laps.empty else clean_laps.loc[clean_laps["lap_duration"].idxmin()]

    # First recorded position per driver = grid/start proxy, last = finish proxy
    # (same "last position update = final classified position" approximation
    # used in tools.py's get_driver_season_summary).
    start_positions = positions.sort_values("date").groupby("driver_number").head(1).set_index("driver_number")["position"]
    finish_positions = positions.sort_values("date").groupby("driver_number").tail(1).set_index("driver_number")["position"]
    position_change = (start_positions - finish_positions).dropna()  # positive = gained positions

    pit_stop_counts = (stints.groupby("driver_number")["stint_number"].max() - 1).astype(int)  # N stints = N-1 stops

    return {
        "fastest_lap": None if fastest is None else {
            "driver_number": int(fastest["driver_number"]),
            "lap_number": int(fastest["lap_number"]),
            "lap_duration_s": float(fastest["lap_duration"]),
        },
        "biggest_gainer": _extreme(position_change, "max"),
        "biggest_loser": _extreme(position_change, "min"),
        "pit_stop_counts_by_driver": pit_stop_counts.to_dict(),
        # Every driver's start->finish, not just the two extremes above —
        # gives the model enough real detail across the whole grid to write
        # a genuinely longer recap instead of padding with repetition.
        "grid_summary": [
            {
                "driver_number": int(driver),
                "start_position": int(start_positions.get(driver)) if pd.notna(start_positions.get(driver)) else None,
                "finish_position": int(finish_positions.get(driver)) if pd.notna(finish_positions.get(driver)) else None,
                "positions_changed": int(position_change.get(driver)) if driver in position_change.index else None,
            }
            for driver in sorted(set(start_positions.index) | set(finish_positions.index))
        ],
    }


def _lap_time_deltas(laps: pd.DataFrame) -> list[dict]:
    """Per lap, per driver: how far off the fastest time set BY ANYONE on that specific lap they were."""
    clean = laps[(laps["is_pit_out_lap"] == 0) & laps["lap_duration"].notna()].copy()
    fastest_per_lap = clean.groupby("lap_number")["lap_duration"].transform("min")
    clean["delta_to_fastest_s"] = (clean["lap_duration"] - fastest_per_lap).round(3)
    return clean[["driver_number", "lap_number", "lap_duration", "delta_to_fastest_s"]].to_dict("records")


def _laps_for(session_key: int) -> pd.DataFrame:
    conn = _connect()
    laps = pd.read_sql(
        "SELECT driver_number, lap_number, lap_duration, is_pit_out_lap FROM laps WHERE session_key = ?",
        conn, params=[session_key],
    )
    conn.close()
    return laps


def get_lap_time_deltas(session_key: int) -> list[dict]:
    """Public wrapper so the API layer can expose this chart's data directly, without duplicating the computation in TypeScript."""
    return _lap_time_deltas(_laps_for(session_key))


def build_race_summary(session_key: int) -> dict:
    laps = _laps_for(session_key)

    key_moments = _compute_key_moments(session_key, laps)
    race_control = get_race_control_messages(session_key)
    tire_strategy = get_tire_strategy(session_key)
    position_changes = get_position_changes(session_key)
    lap_time_deltas = _lap_time_deltas(laps)
    drivers = get_drivers(session_key)  # so the narrative can name drivers, not just cite numbers

    prompt_data = {
        "drivers": drivers,
        "key_moments": key_moments,
        "race_control_messages": race_control,
        "tire_strategy": tire_strategy,
    }

    client = _get_client()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Write a long, detailed race recap using ONLY the data below (see the "
                    "response schema for exactly how long and what to cover). Do not invent any "
                    "lap times, positions, drivers, or events not present here. Use the `drivers` "
                    "list to refer to drivers by full name and number together, per your "
                    "instructions.\n\n" + json.dumps(prompt_data, default=str)
                ),
            },
        ],
        response_format=RESPONSE_SCHEMA,
    )
    narrative = json.loads(response.choices[0].message.content)

    return {
        "session_key": session_key,
        "headline": narrative["headline"],
        "narrative": narrative["narrative"],
        "key_moments": key_moments,
        "charts": {
            "lap_time_deltas": lap_time_deltas,
            "position_changes": position_changes,
            "tire_strategy": tire_strategy,
        },
    }
