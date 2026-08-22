"""
Sanity checks for backend/llm/tools.py, run against the real cache — same
reasoning as test_features.py: these queries/joins are exactly the kind of
code where a wrong column or filter silently returns plausible-looking
nonsense, so real data catches more than a hand-built fixture would.

Doesn't touch the network/Azure — that's covered by manually running
backend/llm/client.py against the real API (costs real tokens, so it's not
part of the automated suite).

Run with:
    python -m backend.tests.test_tools
"""

import math

from backend.llm.tools import TOOL_DISPATCH, TOOL_SCHEMAS, find_sessions, get_driver_season_summary, get_tire_strategy, get_weather


def test_tool_schemas_match_dispatch():
    """Every advertised tool must have a real handler, and vice versa — otherwise the model can request a tool that silently fails."""
    schema_names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    dispatch_names = set(TOOL_DISPATCH.keys())
    assert schema_names == dispatch_names, f"mismatch: {schema_names ^ dispatch_names}"


def test_find_sessions_filters_correctly():
    results = find_sessions(year=2024, meeting_name="Bahrain", session_type="Race")
    assert len(results) == 1
    assert results[0]["session_name"] == "Race"
    assert results[0]["year"] == 2024
    assert "Bahrain" in results[0]["meeting_name"]


def test_find_sessions_excludes_sprint_when_session_name_given():
    results = find_sessions(year=2024, session_name="Race")
    assert all(r["session_name"] == "Race" for r in results)


def test_tire_strategy_has_no_nan():
    """Regression test: Monaco 2024 (session 9523) has stints with null lap_start/lap_end in the raw OpenF1 data, which used to crash JSON serialization."""
    monaco_2024_race_session_key = 9523
    stints = get_tire_strategy(monaco_2024_race_session_key)
    assert len(stints) > 0
    for stint in stints:
        assert not math.isnan(stint["lap_start"])
        assert not math.isnan(stint["lap_end"])


def test_get_weather_is_summarized_not_raw():
    session_key = find_sessions(year=2024, meeting_name="Bahrain", session_type="Race")[0]["session_key"]
    result = get_weather(session_key)
    assert set(result.keys()) >= {"air_temperature_avg_c", "track_temperature_avg_c", "rain_detected"}
    assert isinstance(result["rain_detected"], bool)


def test_driver_season_summary_is_sane():
    result = get_driver_season_summary(driver_number=1, year=2024)  # Verstappen, a full-season driver in a fully-ingested year
    assert result["races_with_recorded_finish"] > 0
    assert 1 <= result["average_finishing_position"] <= 20
    assert result["average_qualifying_gap_to_pole_seconds"] >= 0


if __name__ == "__main__":
    test_tool_schemas_match_dispatch()
    test_find_sessions_filters_correctly()
    test_find_sessions_excludes_sprint_when_session_name_given()
    test_tire_strategy_has_no_nan()
    test_get_weather_is_summarized_not_raw()
    test_driver_season_summary_is_sane()
    print("All tool tests passed.")
