import { useQuery } from "@tanstack/react-query"
import { useMemo, useState } from "react"
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { fetchCarLocation, type Driver, type LapTimeDelta } from "@/lib/api"
import { buildDriverColorMap } from "@/lib/colors"

// A full lap's x/y trace IS the track layout (a car driving one lap
// necessarily traces the whole circuit shape) — so "track layout" and
// "driver's racing line for a lap" are the same chart, not two separate
// features. See backend/llm/tools.py's get_car_location for why this is a
// live OpenF1 call per lap rather than a cached/bulk dataset.
export function TrackLayoutChart({ sessionKey, drivers, laps }: { sessionKey: number; drivers: Driver[]; laps: LapTimeDelta[] }) {
  const [driverNumber, setDriverNumber] = useState<number | null>(drivers[0]?.driver_number ?? null)
  const [lapNumber, setLapNumber] = useState<number | null>(null)

  const colorMap = useMemo(() => buildDriverColorMap(drivers), [drivers])
  const availableLaps = useMemo(
    () => [...new Set(laps.filter((l) => l.driver_number === driverNumber).map((l) => l.lap_number))].sort((a, b) => a - b),
    [laps, driverNumber]
  )
  const effectiveLap = lapNumber ?? availableLaps[0] ?? null

  const { data, isLoading, error } = useQuery({
    queryKey: ["track-layout", sessionKey, driverNumber, effectiveLap],
    queryFn: () => fetchCarLocation(sessionKey, driverNumber!, effectiveLap!),
    enabled: driverNumber != null && effectiveLap != null,
    retry: false, // a 401 won't succeed on retry — don't burn requests re-asking
  })

  const driver = drivers.find((d) => d.driver_number === driverNumber)

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-2">
        <Select value={driverNumber ? String(driverNumber) : undefined} onValueChange={(v) => { setDriverNumber(Number(v)); setLapNumber(null) }}>
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Driver" />
          </SelectTrigger>
          <SelectContent>
            {drivers.map((d) => (
              <SelectItem key={d.driver_number} value={String(d.driver_number)}>
                {d.full_name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
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
      </div>

      {isLoading ? (
        <div className="bg-muted/40 h-[360px] animate-pulse rounded-lg" />
      ) : error ? (
        <div className="text-destructive flex h-[360px] flex-col items-center justify-center gap-1 text-center text-sm">
          <span>OpenF1's location data requires a paid/authenticated API tier this project doesn't have.</span>
          <span className="text-muted-foreground text-xs">Everything else in this app uses OpenF1's free endpoints — only car position (x/y) data is gated.</span>
        </div>
      ) : !data || data.length === 0 ? (
        <div className="text-muted-foreground flex h-[360px] items-center justify-center text-sm">No location data for this lap.</div>
      ) : (
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} tick={false} stroke="var(--color-muted-foreground)" />
            <YAxis dataKey="y" type="number" domain={["dataMin", "dataMax"]} tick={false} stroke="var(--color-muted-foreground)" />
            <Tooltip
              contentStyle={{ background: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: 8 }}
              formatter={(value, name) => [Number(value).toFixed(0), name]}
            />
            <Line
              type="monotone"
              dataKey="y"
              stroke={driverNumber ? colorMap.get(driverNumber) : "#ef4444"}
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
              name={driver?.full_name ?? "Car"}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
