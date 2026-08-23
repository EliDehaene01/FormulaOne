import { useQueries } from "@tanstack/react-query"
import { useMemo, useState } from "react"
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import { DriverFilter } from "@/components/DriverFilter"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { fetchCarTelemetry, type Driver, type LapTimeDelta } from "@/lib/api"
import { buildDriverColorMap } from "@/lib/colors"

// Same driver/lap picker shape as TrackLayoutChart, charting speed/throttle/
// brake over the lap instead of x/y position — see get_car_telemetry_series
// in backend/llm/tools.py for why this is a live, per-lap OpenF1 call
// rather than a bulk-cached dataset (car_data is ~3.7Hz per car per session).
//
// COMPARING MULTIPLE DRIVERS: each driver's telemetry is sampled at its own
// rate over its own lap duration, so raw sample index isn't a fair shared
// x-axis across drivers — this normalizes each driver's samples to % lap
// progress (0-100) instead, the same way real-world telemetry overlays
// compare drivers regardless of exact lap time. With 2+ drivers selected,
// only speed is plotted (throttle+brake for everyone at once is unreadable
// clutter); a single driver still gets the full speed/throttle/brake view.
export function CarTelemetryChart({ sessionKey, drivers, laps }: { sessionKey: number; drivers: Driver[]; laps: LapTimeDelta[] }) {
  const alphabetical = useMemo(() => [...drivers].sort((a, b) => a.full_name.localeCompare(b.full_name)), [drivers])
  const [selectedDrivers, setSelectedDrivers] = useState<Set<number> | null>(
    alphabetical[0] ? new Set([alphabetical[0].driver_number]) : null
  )
  const [lapNumber, setLapNumber] = useState<number | null>(null)

  const colorMap = useMemo(() => buildDriverColorMap(drivers), [drivers])
  const nameMap = useMemo(() => new Map(drivers.map((d) => [d.driver_number, d])), [drivers])
  const activeDrivers = selectedDrivers ? [...selectedDrivers] : drivers.map((d) => d.driver_number)
  const comparing = activeDrivers.length > 1

  const availableLaps = useMemo(
    () => [...new Set(laps.filter((l) => activeDrivers.includes(l.driver_number)).map((l) => l.lap_number))].sort((a, b) => a - b),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [laps, activeDrivers.join(",")]
  )
  const effectiveLap = lapNumber ?? availableLaps[0] ?? null

  const driverQueries = useQueries({
    queries: activeDrivers.map((driverNumber) => ({
      queryKey: ["car-telemetry", sessionKey, driverNumber, effectiveLap],
      queryFn: () => fetchCarTelemetry(sessionKey, driverNumber, effectiveLap!),
      enabled: effectiveLap != null,
      retry: false,
    })),
  })

  const series = activeDrivers.map((driverNumber, i) => {
    const data = driverQueries[i]?.data ?? []
    return { driverNumber, points: data.map((p, idx) => ({ ...p, pct: data.length > 1 ? (idx / (data.length - 1)) * 100 : 0 })) }
  })

  const anyLoading = driverQueries.some((q) => q.isLoading)
  const anyData = series.some((s) => s.points.length > 0)
  // Same reasoning as TrackLayoutChart: a failed request (e.g. OpenF1
  // rate-limiting/blocking unauthenticated access) otherwise looks
  // identical to "no telemetry recorded for this lap".
  const failedQuery = driverQueries.find((q) => q.isError)

  return (
    <div className="flex flex-col gap-3">
      <DriverFilter drivers={drivers} selected={selectedDrivers} onChange={setSelectedDrivers} />
      <Select value={effectiveLap ? String(effectiveLap) : undefined} onValueChange={(v) => setLapNumber(Number(v))}>
        <SelectTrigger className="w-28">
          <SelectValue placeholder="Lap" />
        </SelectTrigger>
        <SelectContent>
          {availableLaps.map((lap) => (
            <SelectItem key={lap} value={String(lap)}>
              Lap {lap}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {anyLoading ? (
        <div className="bg-muted/40 h-[280px] animate-pulse rounded-lg" />
      ) : failedQuery ? (
        <div className="text-destructive flex h-[280px] flex-col items-center justify-center gap-1 px-6 text-center text-sm">
          <span>Couldn't load telemetry for this lap.</span>
          <span className="text-muted-foreground text-xs">{failedQuery.error instanceof Error ? failedQuery.error.message : "Unknown error."}</span>
        </div>
      ) : !anyData ? (
        <div className="text-muted-foreground flex h-[280px] items-center justify-center text-sm">No telemetry for this lap.</div>
      ) : (
        <ResponsiveContainer width="100%" height={280}>
          <LineChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis
              dataKey="pct"
              type="number"
              domain={[0, 100]}
              tickFormatter={(v) => `${v}%`}
              stroke="var(--color-muted-foreground)"
              label={{ value: "Lap progress", position: "insideBottom", offset: -4, fill: "var(--color-muted-foreground)" }}
            />
            <YAxis yAxisId="speed" stroke="var(--color-muted-foreground)" label={{ value: "km/h", angle: -90, position: "insideLeft", fill: "var(--color-muted-foreground)" }} />
            {!comparing && <YAxis yAxisId="pct" orientation="right" domain={[0, 100]} stroke="var(--color-muted-foreground)" />}
            <Tooltip contentStyle={{ background: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: 8 }} labelFormatter={() => ""} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {comparing
              ? series.map(
                  (s) =>
                    s.points.length > 0 && (
                      <Line
                        key={s.driverNumber}
                        yAxisId="speed"
                        data={s.points}
                        dataKey="speed"
                        type="monotone"
                        name={`${nameMap.get(s.driverNumber)?.full_name ?? `#${s.driverNumber}`} — speed`}
                        stroke={colorMap.get(s.driverNumber)}
                        dot={false}
                        isAnimationActive={false}
                      />
                    )
                )
              : series[0] && (
                  <>
                    <Line yAxisId="speed" data={series[0].points} dataKey="speed" type="monotone" name="Speed (km/h)" stroke="#ef4444" dot={false} isAnimationActive={false} />
                    <Line yAxisId="pct" data={series[0].points} dataKey="throttle" type="monotone" name="Throttle %" stroke="#22c55e" dot={false} isAnimationActive={false} />
                    <Line yAxisId="pct" data={series[0].points} dataKey="brake" type="monotone" name="Brake %" stroke="#eab308" dot={false} isAnimationActive={false} />
                  </>
                )}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
