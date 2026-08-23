import { useQuery } from "@tanstack/react-query"
import { useMemo, useState } from "react"

import { CarTelemetryChart } from "@/components/charts/CarTelemetryChart"
import { CompoundUsageChart } from "@/components/charts/CompoundUsageChart"
import { DataTable } from "@/components/charts/DataTable"
import { DriverBarChart } from "@/components/charts/DriverBarChart"
import { IntervalsChart } from "@/components/charts/IntervalsChart"
import { LapTimeDeltaChart } from "@/components/charts/LapTimeDeltaChart"
import { PositionChart } from "@/components/charts/PositionChart"
import { TireStrategyChart } from "@/components/charts/TireStrategyChart"
import { TrackLayoutChart } from "@/components/charts/TrackLayoutChart"
import { WeatherChart } from "@/components/charts/WeatherChart"
import { DriverFilter } from "@/components/DriverFilter"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  fetchDriverLapStats,
  fetchDriverStandings,
  fetchDrivers,
  fetchIntervals,
  fetchLapTimeDeltas,
  fetchMeetings,
  fetchOvertakes,
  fetchPitStops,
  fetchPositions,
  fetchRaceControl,
  fetchRaces,
  fetchSessionResult,
  fetchStartingGrid,
  fetchStints,
  fetchTeamRadio,
  fetchTeamStandings,
  fetchWeatherTimeseries,
  type ChampionshipStanding,
  type Meeting,
  type Overtake,
  type PitStop,
  type PositionChange,
  type RaceControlMessage,
  type SessionResultRow,
  type StartingGridRow,
  type TeamRadioMessage,
} from "@/lib/api"

// Separate from SummaryView on purpose: this hits the plain data endpoints
// directly (fast, free), not build_race_summary (which makes a real LLM
// call every time it's requested). Lets you browse charts for any race
// without spending API tokens just to look at a graph.
export function ChartsView({ sessionKey }: { sessionKey: number }) {
  const [selectedDrivers, setSelectedDrivers] = useState<Set<number> | null>(null)

  const drivers = useQuery({ queryKey: ["drivers", sessionKey], queryFn: () => fetchDrivers(sessionKey) })
  const laps = useQuery({ queryKey: ["laps", sessionKey], queryFn: () => fetchLapTimeDeltas(sessionKey) })
  const positions = useQuery({ queryKey: ["positions", sessionKey], queryFn: () => fetchPositions(sessionKey) })
  const stints = useQuery({ queryKey: ["stints", sessionKey], queryFn: () => fetchStints(sessionKey) })
  const lapStats = useQuery({ queryKey: ["lap-stats", sessionKey], queryFn: () => fetchDriverLapStats(sessionKey) })
  const weather = useQuery({ queryKey: ["weather-ts", sessionKey], queryFn: () => fetchWeatherTimeseries(sessionKey) })
  const pitStops = useQuery({ queryKey: ["pit-stops", sessionKey], queryFn: () => fetchPitStops(sessionKey) })
  const intervals = useQuery({ queryKey: ["intervals", sessionKey], queryFn: () => fetchIntervals(sessionKey) })
  const overtakes = useQuery({ queryKey: ["overtakes", sessionKey], queryFn: () => fetchOvertakes(sessionKey) })
  const teamRadio = useQuery({ queryKey: ["team-radio", sessionKey], queryFn: () => fetchTeamRadio(sessionKey) })
  const sessionResult = useQuery({ queryKey: ["session-result", sessionKey], queryFn: () => fetchSessionResult(sessionKey) })
  const startingGrid = useQuery({ queryKey: ["starting-grid", sessionKey], queryFn: () => fetchStartingGrid(sessionKey) })
  const driverStandings = useQuery({ queryKey: ["driver-standings", sessionKey], queryFn: () => fetchDriverStandings(sessionKey) })
  const teamStandings = useQuery({ queryKey: ["team-standings", sessionKey], queryFn: () => fetchTeamStandings(sessionKey) })
  const raceControl = useQuery({ queryKey: ["race-control", sessionKey], queryFn: () => fetchRaceControl(sessionKey) })
  const allRaces = useQuery({ queryKey: ["all-races"], queryFn: fetchRaces })
  const currentYear = allRaces.data?.find((r) => r.session_key === sessionKey)?.year
  const meetings = useQuery({
    queryKey: ["meetings", currentYear],
    queryFn: () => fetchMeetings(currentYear!),
    enabled: currentYear != null,
  })

  const driverList = drivers.data ?? []
  const nameMap = useMemo(() => new Map(driverList.map((d) => [d.driver_number, d])), [driverList])
  const nameOf = (n: number) => nameMap.get(n)?.full_name ?? `#${n}`

  // Derived, driver-filtered client-side from data already fetched above —
  // no extra backend endpoints needed for these three.
  const filteredLapStats = useMemo(
    () => (selectedDrivers ? (lapStats.data ?? []).filter((r) => selectedDrivers.has(r.driver_number)) : (lapStats.data ?? [])),
    [lapStats.data, selectedDrivers]
  )
  const pitStopsPerDriver = useMemo(() => {
    const counts = new Map<number, number>()
    for (const stint of stints.data ?? []) counts.set(stint.driver_number, Math.max(counts.get(stint.driver_number) ?? 0, stint.stint_number))
    return [...counts.entries()]
      .filter(([d]) => !selectedDrivers || selectedDrivers.has(d))
      .map(([driver_number, maxStint]) => ({ driver_number, value: Math.max(0, maxStint - 1) }))
  }, [stints.data, selectedDrivers])
  const positionsChangedPerDriver = useMemo(() => {
    const byDriver = new Map<number, PositionChange[]>()
    for (const row of positions.data ?? []) {
      if (!byDriver.has(row.driver_number)) byDriver.set(row.driver_number, [])
      byDriver.get(row.driver_number)!.push(row)
    }
    return [...byDriver.entries()]
      .filter(([d]) => !selectedDrivers || selectedDrivers.has(d))
      .map(([driver_number, rows]) => {
        const sorted = [...rows].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
        return { driver_number, value: sorted[0].position - sorted[sorted.length - 1].position }
      })
  }, [positions.data, selectedDrivers])
  const filteredStints = useMemo(
    () => (selectedDrivers ? (stints.data ?? []).filter((s) => selectedDrivers.has(s.driver_number)) : (stints.data ?? [])),
    [stints.data, selectedDrivers]
  )

  return (
    <div className="flex flex-col gap-6">
      {driverList.length > 0 && <DriverFilter drivers={driverList} selected={selectedDrivers} onChange={setSelectedDrivers} />}

      <ChartCard title="Position changes">
        {positions.isLoading ? <ChartSkeleton /> : positions.data && <PositionChart data={positions.data} drivers={driverList} selectedDrivers={selectedDrivers} />}
      </ChartCard>

      <ChartCard title="Lap time deltas">
        {laps.isLoading ? <ChartSkeleton /> : laps.data && <LapTimeDeltaChart data={laps.data} drivers={driverList} selectedDrivers={selectedDrivers} />}
      </ChartCard>

      <ChartCard title="Tire strategy">
        {stints.isLoading ? <ChartSkeleton /> : stints.data && <TireStrategyChart data={stints.data} drivers={driverList} selectedDrivers={selectedDrivers} />}
      </ChartCard>

      <ChartCard title="Track layout & racing line">
        {drivers.isLoading || laps.isLoading ? <ChartSkeleton /> : driverList.length > 0 && <TrackLayoutChart sessionKey={sessionKey} drivers={driverList} laps={laps.data ?? []} />}
      </ChartCard>

      <ChartCard title="Car telemetry (speed / throttle / brake)">
        {drivers.isLoading || laps.isLoading ? <ChartSkeleton /> : driverList.length > 0 && <CarTelemetryChart sessionKey={sessionKey} drivers={driverList} laps={laps.data ?? []} />}
      </ChartCard>

      <ChartCard title="Race control messages">
        {raceControl.isLoading ? (
          <ChartSkeleton />
        ) : (
          <DataTable<RaceControlMessage>
            rows={raceControl.data ?? []}
            rowKey={(r) => `${r.date}-${r.category}-${r.message}`}
            columns={[
              { key: "date", label: "Time", render: (r) => new Date(r.date).toLocaleTimeString() },
              { key: "lap_number", label: "Lap" },
              { key: "category", label: "Category" },
              { key: "flag", label: "Flag" },
              { key: "message", label: "Message" },
            ]}
          />
        )}
      </ChartCard>

      <ChartCard title="Season calendar">
        {meetings.isLoading ? (
          <ChartSkeleton />
        ) : (
          <DataTable<Meeting>
            rows={meetings.data ?? []}
            rowKey={(r) => r.meeting_key}
            columns={[
              { key: "date_start", label: "Date", render: (r) => new Date(r.date_start).toLocaleDateString() },
              { key: "meeting_name", label: "Grand Prix" },
              { key: "location", label: "Location" },
              { key: "circuit_short_name", label: "Circuit" },
            ]}
          />
        )}
      </ChartCard>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <ChartCard title="Fastest lap per driver">
          {lapStats.isLoading ? (
            <ChartSkeleton />
          ) : (
            <DriverBarChart data={filteredLapStats.map((r) => ({ driver_number: r.driver_number, value: r.fastest_lap_s }))} drivers={driverList} valueLabel="Fastest lap (s)" unit="s" decimals={3} />
          )}
        </ChartCard>

        <ChartCard title="Average lap time per driver">
          {lapStats.isLoading ? (
            <ChartSkeleton />
          ) : (
            <DriverBarChart data={filteredLapStats.map((r) => ({ driver_number: r.driver_number, value: r.average_lap_s }))} drivers={driverList} valueLabel="Average lap (s)" unit="s" decimals={2} />
          )}
        </ChartCard>

        <ChartCard title="Lap time consistency (lower = more consistent)">
          {lapStats.isLoading ? (
            <ChartSkeleton />
          ) : (
            <DriverBarChart data={filteredLapStats.map((r) => ({ driver_number: r.driver_number, value: r.consistency_std_s }))} drivers={driverList} valueLabel="Std dev (s)" unit="s" decimals={2} />
          )}
        </ChartCard>

        <ChartCard title="Top speed per driver">
          {lapStats.isLoading ? (
            <ChartSkeleton />
          ) : (
            <DriverBarChart data={filteredLapStats.map((r) => ({ driver_number: r.driver_number, value: r.top_speed_kph }))} drivers={driverList} valueLabel="Top speed (km/h)" unit=" km/h" decimals={0} />
          )}
        </ChartCard>

        <ChartCard title="Pit stops per driver">
          {stints.isLoading ? <ChartSkeleton /> : <DriverBarChart data={pitStopsPerDriver} drivers={driverList} valueLabel="Pit stops" decimals={0} />}
        </ChartCard>

        <ChartCard title="Positions gained / lost">
          {positions.isLoading ? <ChartSkeleton /> : <DriverBarChart data={positionsChangedPerDriver} drivers={driverList} valueLabel="Positions changed" decimals={0} />}
        </ChartCard>
      </div>

      <ChartCard title="Tire compound usage (whole field)">
        {stints.isLoading ? <ChartSkeleton /> : <CompoundUsageChart data={filteredStints} />}
      </ChartCard>

      <ChartCard title="Weather & rainfall over the session">
        {weather.isLoading ? <ChartSkeleton /> : weather.data && <WeatherChart data={weather.data} />}
      </ChartCard>

      <ChartCard title="Gap to leader over the race">
        {intervals.isLoading ? <ChartSkeleton /> : intervals.data && <IntervalsChart data={intervals.data} drivers={driverList} selectedDrivers={selectedDrivers} />}
      </ChartCard>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <ChartCard title="Starting grid">
          {startingGrid.isLoading ? (
            <ChartSkeleton />
          ) : (
            <DataTable<StartingGridRow>
              rows={startingGrid.data ?? []}
              rowKey={(r) => r.driver_number}
              columns={[
                { key: "position", label: "Grid" },
                { key: "driver_number", label: "Driver", render: (r) => nameOf(r.driver_number) },
                { key: "lap_duration", label: "Quali time (s)", align: "right", render: (r) => (r.lap_duration != null ? r.lap_duration.toFixed(3) : "—") },
              ]}
            />
          )}
        </ChartCard>

        <ChartCard title="Final classification">
          {sessionResult.isLoading ? (
            <ChartSkeleton />
          ) : (
            <DataTable<SessionResultRow>
              rows={sessionResult.data ?? []}
              rowKey={(r) => r.driver_number}
              columns={[
                { key: "position", label: "Pos", render: (r) => (r.dnf ? "DNF" : r.dns ? "DNS" : r.dsq ? "DSQ" : (r.position ?? "—")) },
                { key: "driver_number", label: "Driver", render: (r) => nameOf(r.driver_number) },
                { key: "number_of_laps", label: "Laps", align: "right" },
                {
                  key: "gap_to_leader",
                  label: "Gap",
                  align: "right",
                  render: (r) => (typeof r.gap_to_leader === "number" ? `+${r.gap_to_leader.toFixed(3)}s` : (r.gap_to_leader ?? "—")),
                },
              ]}
            />
          )}
        </ChartCard>

        <ChartCard title="Pit stops">
          {pitStops.isLoading ? (
            <ChartSkeleton />
          ) : (
            <DataTable<PitStop>
              rows={pitStops.data ?? []}
              rowKey={(r) => `${r.driver_number}-${r.lap_number}`}
              columns={[
                { key: "driver_number", label: "Driver", render: (r) => nameOf(r.driver_number) },
                { key: "lap_number", label: "Lap", align: "right" },
                { key: "pit_lane_duration_s", label: "Lane time (s)", align: "right", render: (r) => (r.pit_lane_duration_s != null ? r.pit_lane_duration_s.toFixed(1) : "—") },
                { key: "stop_duration", label: "Stop time (s)", align: "right", render: (r) => (r.stop_duration != null ? r.stop_duration.toFixed(1) : "—") },
              ]}
            />
          )}
        </ChartCard>

        <ChartCard title="Overtakes">
          {overtakes.isLoading ? (
            <ChartSkeleton />
          ) : (
            <DataTable<Overtake>
              rows={overtakes.data ?? []}
              rowKey={(r) => `${r.date}-${r.overtaking_driver_number}-${r.overtaken_driver_number}`}
              columns={[
                { key: "overtaking_driver_number", label: "Passed by", render: (r) => nameOf(r.overtaking_driver_number) },
                { key: "overtaken_driver_number", label: "Passed", render: (r) => nameOf(r.overtaken_driver_number) },
                { key: "position", label: "New pos.", align: "right" },
              ]}
            />
          )}
        </ChartCard>

        <ChartCard title="Drivers' championship standing">
          {driverStandings.isLoading ? (
            <ChartSkeleton />
          ) : (
            <DataTable<ChampionshipStanding>
              rows={driverStandings.data ?? []}
              rowKey={(r) => r.driver_number!}
              columns={[
                { key: "position_current", label: "Pos" },
                { key: "driver_number", label: "Driver", render: (r) => nameOf(r.driver_number!) },
                { key: "points_current", label: "Points", align: "right" },
                {
                  key: "points_delta",
                  label: "This session",
                  align: "right",
                  render: (r) => (r.points_current != null && r.points_start != null ? `+${r.points_current - r.points_start}` : "—"),
                },
              ]}
            />
          )}
        </ChartCard>

        <ChartCard title="Constructors' championship standing">
          {teamStandings.isLoading ? (
            <ChartSkeleton />
          ) : (
            <DataTable<ChampionshipStanding>
              rows={teamStandings.data ?? []}
              rowKey={(r) => r.team_name!}
              columns={[
                { key: "position_current", label: "Pos" },
                { key: "team_name", label: "Team" },
                { key: "points_current", label: "Points", align: "right" },
                {
                  key: "points_delta",
                  label: "This session",
                  align: "right",
                  render: (r) => (r.points_current != null && r.points_start != null ? `+${r.points_current - r.points_start}` : "—"),
                },
              ]}
            />
          )}
        </ChartCard>
      </div>

      <ChartCard title="Team radio">
        {teamRadio.isLoading ? (
          <ChartSkeleton />
        ) : (
          <DataTable<TeamRadioMessage>
            rows={teamRadio.data ?? []}
            rowKey={(r) => `${r.driver_number}-${r.date}`}
            columns={[
              { key: "driver_number", label: "Driver", render: (r) => nameOf(r.driver_number) },
              { key: "date", label: "Time", render: (r) => new Date(r.date).toLocaleTimeString() },
              { key: "recording_url", label: "Recording", render: (r) => <audio controls preload="none" src={r.recording_url} className="h-8" /> },
            ]}
          />
        )}
      </ChartCard>
    </div>
  )
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  )
}

function ChartSkeleton() {
  return <div className="bg-muted/40 h-[280px] animate-pulse rounded-lg" />
}
