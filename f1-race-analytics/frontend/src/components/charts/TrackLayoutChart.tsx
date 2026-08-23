import { useQueries, useQuery } from "@tanstack/react-query"
import { useMemo, useState } from "react"
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import { DriverFilter } from "@/components/DriverFilter"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { fetchCarLocation, type Driver, type LapTimeDelta } from "@/lib/api"
import { buildDriverColorMap } from "@/lib/colors"

// A full lap's x/y trace IS the track layout (a car driving one lap
// necessarily traces the whole circuit shape). Uses the DriverFilter badge
// picker (same component as the page-level filter) so comparing 1-N
// drivers' racing lines on the same lap works the same way everywhere else
// in the app lets you pick drivers. See backend/llm/tools.py's
// get_car_location for why this is a live OpenF1 call per lap rather than
// a cached/bulk dataset.
export function TrackLayoutChart({ sessionKey, drivers, laps }: { sessionKey: number; drivers: Driver[]; laps: LapTimeDelta[] }) {
  const alphabetical = useMemo(() => [...drivers].sort((a, b) => a.full_name.localeCompare(b.full_name)), [drivers])
  const [selectedDrivers, setSelectedDrivers] = useState<Set<number> | null>(
    alphabetical[0] ? new Set([alphabetical[0].driver_number]) : null
  )
  const [lapNumber, setLapNumber] = useState<number | null>(null)

  const colorMap = useMemo(() => buildDriverColorMap(drivers), [drivers])
  const nameMap = useMemo(() => new Map(drivers.map((d) => [d.driver_number, d])), [drivers])
  const activeDrivers = selectedDrivers ? [...selectedDrivers] : drivers.map((d) => d.driver_number)

  const availableLaps = useMemo(
    () => [...new Set(laps.filter((l) => activeDrivers.includes(l.driver_number)).map((l) => l.lap_number))].sort((a, b) => a - b),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- activeDrivers is derived fresh each render; its content is what should trigger recompute, not its identity
    [laps, activeDrivers.join(",")]
  )
  const effectiveLap = lapNumber ?? availableLaps[0] ?? null

  // Faint track-shape reference behind the colored racing lines: the
  // session's single fastest lap (any driver), fetched once regardless of
  // who's selected for comparison, so switching drivers doesn't reset it.
  const fastestLap = useMemo(() => (laps.length === 0 ? null : [...laps].sort((a, b) => a.lap_duration - b.lap_duration)[0]), [laps])
  const referenceQuery = useQuery({
    queryKey: ["track-reference", sessionKey, fastestLap?.driver_number, fastestLap?.lap_number],
    queryFn: () => fetchCarLocation(sessionKey, fastestLap!.driver_number, fastestLap!.lap_number),
    enabled: fastestLap != null,
    retry: false,
  })

  const driverQueries = useQueries({
    queries: activeDrivers.map((driverNumber) => ({
      queryKey: ["track-layout", sessionKey, driverNumber, effectiveLap],
      queryFn: () => fetchCarLocation(sessionKey, driverNumber, effectiveLap!),
      enabled: effectiveLap != null,
      retry: false,
    })),
  })

  const allPoints = useMemo(() => {
    const pts = [...(referenceQuery.data ?? [])]
    for (const q of driverQueries) pts.push(...(q.data ?? []))
    return pts
  }, [referenceQuery.data, driverQueries])

  const anyLoading = driverQueries.some((q) => q.isLoading)
  const anyData = allPoints.length > 0
  // A failed request (e.g. OpenF1 rate-limiting/blocking unauthenticated
  // access, which does happen — see fetchCarLocation's error message)
  // otherwise looks IDENTICAL to "no data was recorded for this lap" once
  // it falls through to the empty-state branch. Surface it distinctly.
  const failedQuery = driverQueries.find((q) => q.isError) ?? (referenceQuery.isError ? referenceQuery : undefined)
  const xs = allPoints.map((p) => p.x)
  const ys = allPoints.map((p) => p.y)
  const xDomain: [number, number] = anyData ? [Math.min(...xs), Math.max(...xs)] : [0, 1]
  const yDomain: [number, number] = anyData ? [Math.min(...ys), Math.max(...ys)] : [0, 1]

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
        <div className="bg-muted/40 h-[360px] animate-pulse rounded-lg" />
      ) : failedQuery ? (
        <div className="text-destructive flex h-[360px] flex-col items-center justify-center gap-1 px-6 text-center text-sm">
          <span>Couldn't load position data for this lap.</span>
          <span className="text-muted-foreground text-xs">{failedQuery.error instanceof Error ? failedQuery.error.message : "Unknown error."}</span>
        </div>
      ) : !anyData ? (
        <div className="text-muted-foreground flex h-[360px] items-center justify-center text-sm">No location data for this lap.</div>
      ) : (
        <ResponsiveContainer width="100%" height={360}>
          <LineChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="x" type="number" domain={xDomain} tick={false} stroke="var(--color-muted-foreground)" />
            <YAxis dataKey="y" type="number" domain={yDomain} tick={false} stroke="var(--color-muted-foreground)" />
            <Tooltip
              contentStyle={{ background: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: 8 }}
              formatter={(value, name) => [Number(value).toFixed(0), name]}
            />
            {activeDrivers.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
            {referenceQuery.data && referenceQuery.data.length > 0 && (
              <>
                {/* A thick, rounded "road" line reads much more clearly as a track
                    surface than a thin outline would — two layers (a wide muted
                    base, a slightly narrower lighter one on top) approximate an
                    asphalt-with-racing-surface look without needing real track SVG data. */}
                <Line
                  data={referenceQuery.data}
                  type="monotone"
                  dataKey="y"
                  stroke="#1e2330"
                  strokeWidth={16}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  dot={false}
                  isAnimationActive={false}
                  name="Track"
                  legendType="none"
                  tooltipType="none"
                />
                <Line
                  data={referenceQuery.data}
                  type="monotone"
                  dataKey="y"
                  stroke="#3a4156"
                  strokeWidth={11}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  dot={false}
                  isAnimationActive={false}
                  name="Track outline"
                  legendType="none"
                  tooltipType="none"
                />
              </>
            )}
            {activeDrivers.map((driverNumber, i) => {
              const data = driverQueries[i]?.data
              if (!data || data.length === 0) return null
              return (
                <Line
                  key={driverNumber}
                  data={data}
                  type="monotone"
                  dataKey="y"
                  stroke={colorMap.get(driverNumber)}
                  dot={false}
                  strokeWidth={2}
                  isAnimationActive={false}
                  name={nameMap.get(driverNumber)?.full_name ?? `#${driverNumber}`}
                />
              )
            })}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
