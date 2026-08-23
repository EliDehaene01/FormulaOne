// Thin fetch wrappers around the FastAPI backend (backend/api/main.py).
// Vite's dev server proxies /api -> http://localhost:8000 (see vite.config.ts),
// so these can just use relative paths in dev and prod alike.

export interface RaceSession {
  session_key: number
  session_name: string
  session_type: string
  year: number
  date_start: string
  circuit_short_name: string
  meeting_name: string
}

export interface LapTimeDelta {
  driver_number: number
  lap_number: number
  lap_duration: number
  delta_to_fastest_s: number
}

export interface PositionChange {
  driver_number: number
  date: string
  position: number
}

export interface TireStint {
  driver_number: number
  stint_number: number
  compound: string
  lap_start: number
  lap_end: number
  tyre_age_at_start: number
}

export interface DriverLapStats {
  driver_number: number
  fastest_lap_s: number
  average_lap_s: number
  consistency_std_s: number
  top_speed_kph: number
}

export interface WeatherPoint {
  date: string
  air_temperature: number
  track_temperature: number
  rainfall: number
}

export interface Driver {
  driver_number: number
  full_name: string
  name_acronym: string
  team_name: string
  team_colour: string | null
}

export interface CarLocationPoint {
  x: number
  y: number
}

export interface QualifyingPrediction {
  session_key: number
  predicted_pole_driver_number: number
  predicted_pole_time_s: number
  actual_pole_driver_number: number
  actual_pole_time_s: number
  per_driver_predictions: { driver_number: number; predicted_lap_time_s: number; actual_lap_time_s: number }[]
  caveat: string
  error?: string
}

export interface FeatureImportance {
  feature: string
  mae_increase_s: number
  baseline_val_mae_s: number
}

export interface PredictionContribution {
  feature_name: string
  contribution_s: number
}

export interface PredictionExplanation {
  session_key: number
  driver_number: number
  predicted_lap_time_s: number
  actual_lap_time_s: number
  baseline_prediction_s: number
  attribution_total_s: number
  prediction_minus_baseline_s: number
  contributions: PredictionContribution[]
  caveat: string
  error?: string
}

export interface RaceSummary {
  session_key: number
  headline: string
  narrative: string
  key_moments: {
    fastest_lap: { driver_number: number; lap_number: number; lap_duration_s: number } | null
    biggest_gainer: { driver_number: number; positions_changed: number } | null
    biggest_loser: { driver_number: number; positions_changed: number } | null
    pit_stop_counts_by_driver: Record<string, number>
  }
}

// A real conversation has heterogeneous message shapes (assistant messages
// carry tool_calls, tool-result messages carry a tool_call_id) — typing
// this strictly as {role, content} and nothing else would encourage
// stripping those fields, which is exactly the bug that broke the second
// chat message (see backend/api/main.py's ChatRequest for the full story).
// The index signature keeps whatever extra fields a message round-trips
// with intact.
export interface Meeting {
  meeting_key: number
  meeting_name: string
  meeting_official_name: string
  location: string
  country_name: string
  circuit_short_name: string
  year: number
  date_start: string
}

export interface RaceControlMessage {
  date: string
  category: string
  message: string
  flag: string | null
  lap_number: number | null
  driver_number: number | null
}

export interface CarTelemetryPoint {
  date: string
  speed: number | null
  throttle: number | null
  brake: number | null
}

export interface PitStop {
  driver_number: number
  lap_number: number
  pit_lane_duration_s: number | null
  // Real stop_duration is only reported by OpenF1 from a certain race
  // onward; stop_duration_estimated is true when this value was instead
  // approximated from pit_lane_duration_s (see backend/llm/tools.py).
  stop_duration: number | null
  stop_duration_estimated: boolean
}

export interface IntervalPoint {
  driver_number: number
  date: string
  gap_to_leader: number | null
  interval: number | null
}

export interface Overtake {
  date: string
  overtaking_driver_number: number
  overtaken_driver_number: number
  position: number
}

export interface TeamRadioMessage {
  driver_number: number
  date: string
  recording_url: string
}

export interface SessionResultRow {
  driver_number: number
  position: number | null
  duration: number | null
  // OpenF1 reports this as a number of seconds usually, but as text for a
  // lapped-down race finisher ("+1 LAP") or a per-segment array during
  // Qualifying ("[0.3, 0.2, 0.0]") — see backend/models/features.py's
  // _race_level_table for the same data quirk on the training side.
  gap_to_leader: number | string | null
  number_of_laps: number | null
  dnf: boolean
  dns: boolean
  dsq: boolean
}

export interface StartingGridRow {
  driver_number: number
  position: number
  lap_duration: number | null
}

export interface ChampionshipStanding {
  driver_number?: number
  team_name?: string
  position_start: number | null
  position_current: number | null
  points_start: number | null
  points_current: number | null
}

export interface ChatMessage {
  role: "user" | "assistant" | "tool" | "system"
  content?: string
  [key: string]: unknown
}

async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) {
    // FastAPI error responses are {"detail": "..."} — surface that actual
    // reason (e.g. an upstream OpenF1 outage/rate-limit message) instead of
    // just a status code, so a chart's error state can explain WHY instead
    // of looking identical to "there's genuinely no data here".
    let detail: string | null = null
    try {
      detail = (await response.json())?.detail ?? null
    } catch {
      // body wasn't JSON — fall through to the generic message below
    }
    throw new Error(detail || `${path} failed: ${response.status}`)
  }
  return response.json()
}

export function fetchRaces(): Promise<RaceSession[]> {
  // Unfiltered — the season/weekend/session cascade (RaceSelector) needs
  // every session type to populate its three dropdowns, not just Races.
  return getJSON(`/api/races`)
}

export function fetchCarLocation(sessionKey: number, driverNumber: number, lapNumber: number): Promise<CarLocationPoint[]> {
  return getJSON(`/api/races/${sessionKey}/track-layout?driver_number=${driverNumber}&lap_number=${lapNumber}`)
}

export function fetchPrediction(sessionKey: number): Promise<QualifyingPrediction> {
  return getJSON(`/api/races/${sessionKey}/prediction`)
}

export function fetchRaceSummary(sessionKey: number): Promise<RaceSummary> {
  return getJSON(`/api/races/${sessionKey}/summary`)
}

export function fetchFeatureImportance(): Promise<FeatureImportance[]> {
  return getJSON(`/api/model/feature-importance`)
}

export function fetchPredictionExplanation(sessionKey: number, driverNumber: number): Promise<PredictionExplanation> {
  return getJSON(`/api/races/${sessionKey}/prediction/explain?driver_number=${driverNumber}`)
}

export function fetchLapTimeDeltas(sessionKey: number): Promise<LapTimeDelta[]> {
  return getJSON(`/api/races/${sessionKey}/lap-time-deltas`)
}

export function fetchPositions(sessionKey: number): Promise<PositionChange[]> {
  return getJSON(`/api/races/${sessionKey}/positions`)
}

export function fetchStints(sessionKey: number): Promise<TireStint[]> {
  return getJSON(`/api/races/${sessionKey}/stints`)
}

export function fetchDriverLapStats(sessionKey: number): Promise<DriverLapStats[]> {
  return getJSON(`/api/races/${sessionKey}/driver-lap-stats`)
}

export function fetchWeatherTimeseries(sessionKey: number): Promise<WeatherPoint[]> {
  return getJSON(`/api/races/${sessionKey}/weather-timeseries`)
}

export function fetchDrivers(sessionKey: number): Promise<Driver[]> {
  return getJSON(`/api/races/${sessionKey}/drivers`)
}

export function fetchMeetings(year: number): Promise<Meeting[]> {
  return getJSON(`/api/meetings?year=${year}`)
}

export function fetchRaceControl(sessionKey: number): Promise<RaceControlMessage[]> {
  return getJSON(`/api/races/${sessionKey}/race-control`)
}

export function fetchCarTelemetry(sessionKey: number, driverNumber: number, lapNumber: number): Promise<CarTelemetryPoint[]> {
  return getJSON(`/api/races/${sessionKey}/car-telemetry?driver_number=${driverNumber}&lap_number=${lapNumber}`)
}

export function fetchPitStops(sessionKey: number): Promise<PitStop[]> {
  return getJSON(`/api/races/${sessionKey}/pit-stops`)
}

export function fetchIntervals(sessionKey: number): Promise<IntervalPoint[]> {
  return getJSON(`/api/races/${sessionKey}/intervals`)
}

export function fetchOvertakes(sessionKey: number): Promise<Overtake[]> {
  return getJSON(`/api/races/${sessionKey}/overtakes`)
}

export function fetchTeamRadio(sessionKey: number): Promise<TeamRadioMessage[]> {
  return getJSON(`/api/races/${sessionKey}/team-radio`)
}

export function fetchSessionResult(sessionKey: number): Promise<SessionResultRow[]> {
  return getJSON(`/api/races/${sessionKey}/session-result`)
}

export function fetchStartingGrid(sessionKey: number): Promise<StartingGridRow[]> {
  return getJSON(`/api/races/${sessionKey}/starting-grid`)
}

export function fetchDriverStandings(sessionKey: number): Promise<ChampionshipStanding[]> {
  return getJSON(`/api/races/${sessionKey}/standings/drivers`)
}

export function fetchTeamStandings(sessionKey: number): Promise<ChampionshipStanding[]> {
  return getJSON(`/api/races/${sessionKey}/standings/teams`)
}

export async function sendChatMessage(messages: ChatMessage[]): Promise<ChatMessage[]> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  })
  if (!response.ok) throw new Error(`chat failed: ${response.status}`)
  const data = await response.json()
  return data.messages
}
