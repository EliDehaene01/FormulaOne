import { useQuery } from "@tanstack/react-query"
import { useMemo, useState } from "react"

import { CompoundUsageChart } from "@/components/charts/CompoundUsageChart"
import { DriverBarChart } from "@/components/charts/DriverBarChart"
import { LapTimeDeltaChart } from "@/components/charts/LapTimeDeltaChart"
import { PositionChart } from "@/components/charts/PositionChart"
import { TireStrategyChart } from "@/components/charts/TireStrategyChart"
import { TrackLayoutChart } from "@/components/charts/TrackLayoutChart"
import { WeatherChart } from "@/components/charts/WeatherChart"
import { DriverFilter } from "@/components/DriverFilter"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  fetchDriverLapStats,
  fetchDrivers,
  fetchLapTimeDeltas,
  fetchPositions,
  fetchStints,
  fetchWeatherTimeseries,
  type PositionChange,
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

  const driverList = drivers.data ?? []

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
