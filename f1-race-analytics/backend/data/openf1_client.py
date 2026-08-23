"""
Thin wrapper around the OpenF1 API (https://api.openf1.org).

OpenF1 has no auth and no client library — every endpoint is just
`GET /v1/<endpoint>?param=value`, returning a JSON array of flat objects
(one dict per row: one lap, one position update, one weather sample, etc.).

This module ONLY talks to the network. It has no idea about caching or
SQLite — that's cache.py's job. Keeping them separate means we can test
or reuse the API wrapper without spinning up a database, and swap the
caching strategy later without touching this file.
"""

import time

import requests

BASE_URL = "https://api.openf1.org/v1"

# OpenF1 throttles bursts (we hit HTTP 429 after ~30 requests in a few
# seconds during a real ingest run). Retry with exponential backoff instead
# of crashing the whole ingest over one rate-limited request.
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0


def _get(endpoint: str, **params) -> list[dict]:
    """
    Call one OpenF1 endpoint and return the parsed JSON list.

    `**params` are passed straight through as query string params, e.g.
    _get("laps", session_key=9158, driver_number=1) ->
        GET /v1/laps?session_key=9158&driver_number=1

    None values are dropped so callers can pass optional filters without
    each wrapper function needing its own "if X is not None" branch.
    """
    query = {k: v for k, v in params.items() if v is not None}
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(MAX_RETRIES + 1):
        response = requests.get(f"{BASE_URL}/{endpoint}", params=query, timeout=30)

        if response.status_code == 404:
            # Seen in practice: some sessions have no data published yet for
            # a given endpoint (e.g. laps for one specific session). That's
            # a data-availability gap, not a bug in our request — treat it
            # the same as "empty result" rather than crashing a multi-hour
            # ingest over one missing session.
            return []

        if response.status_code != 429:
            response.raise_for_status()  # turn other HTTP errors into a Python exception now, not a confusing KeyError later
            return response.json()

        if attempt == MAX_RETRIES:
            response.raise_for_status()  # out of retries — surface the 429 as a real error instead of retrying forever

        # Respect the server's Retry-After header if it sends one, otherwise
        # back off exponentially (1s, 2s, 4s, ...).
        wait_seconds = float(response.headers.get("Retry-After", backoff))
        time.sleep(wait_seconds)
        backoff *= 2


def get_meetings(year: int | None = None) -> list[dict]:
    """A 'meeting' is a race weekend (e.g. '2023 Bahrain Grand Prix')."""
    return _get("meetings", year=year)


def get_sessions(
    year: int | None = None,
    session_type: str | None = None,
    meeting_key: int | None = None,
) -> list[dict]:
    """
    A 'session' is one part of a race weekend: Practice 1/2/3, Qualifying,
    Sprint, or Race. Each session has a unique session_key, which every
    other endpoint below uses to scope its results to that session.
    """
    return _get("sessions", year=year, session_type=session_type, meeting_key=meeting_key)


def get_laps(session_key: int) -> list[dict]:
    """Every timed lap in the session: driver, lap number, sector times, lap_duration."""
    return _get("laps", session_key=session_key)


def get_position(session_key: int) -> list[dict]:
    """Track-position changes over time (who's running where, and when it changed)."""
    return _get("position", session_key=session_key)


def get_location(session_key: int, driver_number: int, date_start: str, date_end: str) -> list[dict]:
    """
    Car x/y/z coordinates over time — sampled at ~3.7Hz, so even one lap is
    a few hundred points. NOT bulk-cached like the other endpoints (that
    would mean caching tens of thousands of rows per session): called live,
    on demand, for one driver's one lap at a time, bounded by that lap's
    start/end timestamps.
    """
    # OpenF1's range filters are literal operators in the query param name
    # (date>=..., date<=...) — not valid Python keyword argument names, so
    # these go in via dict-unpacking instead of named kwargs.
    return _get(
        "location",
        session_key=session_key,
        driver_number=driver_number,
        **{"date>=": date_start, "date<=": date_end},
    )


def get_stints(session_key: int) -> list[dict]:
    """Tire stints: which compound each driver ran, and over which lap range."""
    return _get("stints", session_key=session_key)


def get_weather(session_key: int) -> list[dict]:
    """Track/air temperature, humidity, wind, rainfall — sampled periodically through the session."""
    return _get("weather", session_key=session_key)


def get_race_control(session_key: int) -> list[dict]:
    """Flags, safety cars, penalties, and other official messages issued during the session."""
    return _get("race_control", session_key=session_key)


def get_drivers(session_key: int) -> list[dict]:
    """The driver/team lineup for this specific session (teams change year to year, so this is per-session, not global)."""
    return _get("drivers", session_key=session_key)


def get_pit(session_key: int) -> list[dict]:
    """Pit lane visits: which lap, and how long the car spent in the pit lane."""
    return _get("pit", session_key=session_key)


def get_intervals(session_key: int) -> list[dict]:
    """Real-time gap to the driver ahead and to the race leader (Race/Sprint sessions only)."""
    return _get("intervals", session_key=session_key)


def get_overtakes(session_key: int) -> list[dict]:
    """On-track and pit-related position exchanges between two drivers."""
    return _get("overtakes", session_key=session_key)


def get_team_radio(session_key: int) -> list[dict]:
    """Metadata + recording URLs for driver-to-pit-wall radio messages."""
    return _get("team_radio", session_key=session_key)


def get_session_result(session_key: int) -> list[dict]:
    """Final classification (position, points, gap, status) at the end of the session."""
    return _get("session_result", session_key=session_key)


def get_starting_grid(session_key: int) -> list[dict]:
    """Starting grid order for a Race/Sprint session."""
    return _get("starting_grid", session_key=session_key)


def get_championship_drivers(session_key: int) -> list[dict]:
    """Running drivers'-championship standings as of this session."""
    return _get("championship_drivers", session_key=session_key)


def get_championship_teams(session_key: int) -> list[dict]:
    """Running constructors'-championship standings as of this session."""
    return _get("championship_teams", session_key=session_key)


def get_car_data(session_key: int, driver_number: int, date_start: str, date_end: str) -> list[dict]:
    """
    Per-car telemetry (speed, throttle, brake, RPM, gear, DRS) at ~3.7Hz.

    Same deal as get_location: NOT bulk-cached (a whole session for every
    driver would be hundreds of millions of rows across the existing 378
    cached sessions) — called live, one driver/one time-window at a time.
    """
    return _get(
        "car_data",
        session_key=session_key,
        driver_number=driver_number,
        **{"date>=": date_start, "date<=": date_end},
    )
