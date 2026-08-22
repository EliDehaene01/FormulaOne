"""
FastAPI backend: exposes the data/model/LLM layers built in Phases 1-4 over
HTTP so the frontend has something to call.

Run with:
    uvicorn backend.api.main:app --reload --port 8000

STATELESS BY DESIGN: no server-side session/conversation storage. The
frontend holds the chat conversation in its own state and sends the full
message history on every /api/chat request — simplest thing that works
without needing a database table or session store for a portfolio project.

Endpoints mostly just call straight through to the tool functions from
Phase 3 (backend/llm/tools.py) — no separate "API layer" logic duplicating
what those already do, they're already the right shape (plain dicts/lists,
JSON-serializable).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.llm import tools
from backend.llm.client import chat as run_chat
from backend.llm.summary import build_race_summary, get_lap_time_deltas

app = FastAPI(title="F1 Race Analytics API")

# Vite's default dev port. Wide open for local dev only — if this app is
# ever actually deployed, this should be the real frontend origin, not "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/races")
def list_races(
    year: int | None = None,
    meeting_name: str | None = None,
    session_type: str | None = None,
    session_name: str | None = None,
):
    return tools.find_sessions(year=year, meeting_name=meeting_name, session_type=session_type, session_name=session_name)


@app.get("/api/races/{session_key}/laps")
def race_laps(session_key: int, driver_number: int | None = None):
    return tools.get_lap_times(session_key, driver_number)


@app.get("/api/races/{session_key}/lap-time-deltas")
def race_lap_time_deltas(session_key: int):
    return get_lap_time_deltas(session_key)


@app.get("/api/races/{session_key}/positions")
def race_positions(session_key: int, driver_number: int | None = None):
    return tools.get_position_changes(session_key, driver_number)


@app.get("/api/races/{session_key}/stints")
def race_stints(session_key: int, driver_number: int | None = None):
    return tools.get_tire_strategy(session_key, driver_number)


@app.get("/api/races/{session_key}/weather")
def race_weather(session_key: int):
    return tools.get_weather(session_key)


@app.get("/api/races/{session_key}/weather-timeseries")
def race_weather_timeseries(session_key: int):
    return tools.get_weather_timeseries(session_key)


@app.get("/api/races/{session_key}/driver-lap-stats")
def race_driver_lap_stats(session_key: int):
    return tools.get_driver_lap_stats(session_key)


@app.get("/api/races/{session_key}/race-control")
def race_control(session_key: int):
    return tools.get_race_control_messages(session_key)


@app.get("/api/races/{session_key}/drivers")
def race_drivers(session_key: int):
    return tools.get_drivers(session_key)


@app.get("/api/races/{session_key}/track-layout")
def race_track_layout(session_key: int, driver_number: int, lap_number: int):
    try:
        return tools.get_car_location(session_key, driver_number, lap_number)
    except Exception as exc:
        # OpenF1's /location endpoint returns 401 for this project's
        # unauthenticated access — unlike every other endpoint used
        # elsewhere in this app, it needs a paid/authenticated OpenF1 tier
        # we don't have. Surface that plainly instead of a generic 500.
        raise HTTPException(status_code=502, detail=f"OpenF1 location data unavailable: {exc}") from exc


@app.get("/api/races/{session_key}/prediction")
def race_prediction(session_key: int):
    # predict_qualifying_pace lazily imports torch, which doesn't import
    # natively on this dev machine outside WSL (see backend/models/README.md)
    # — surface that as a clear API error instead of a raw 500 traceback.
    try:
        return tools.predict_qualifying_pace(session_key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/races/{session_key}/summary")
def race_summary(session_key: int):
    try:
        return build_race_summary(session_key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/drivers/{driver_number}/season/{year}")
def driver_season(driver_number: int, year: int):
    return tools.get_driver_season_summary(driver_number, year)


class ChatRequest(BaseModel):
    # Plain dicts, not a typed ChatMessage(role, content) model: a real
    # conversation has heterogeneous message shapes (assistant messages
    # carry a tool_calls array, tool-result messages carry a
    # tool_call_id) — a strict Pydantic model naming only role/content
    # would silently DROP those fields on every round trip (Pydantic
    # ignores unknown fields by default), which breaks the second message
    # in any conversation that involved a tool call: Azure rejects a
    # "tool" role message whose tool_call_id no longer has a matching
    # tool_calls entry on the preceding assistant message.
    messages: list[dict]


@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    try:
        return {"messages": run_chat(request.messages)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
