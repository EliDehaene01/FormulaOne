import { useQuery } from "@tanstack/react-query"
import { useMemo, useState } from "react"
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { fetchCarTelemetry, type Driver, type LapTimeDelta } from "@/lib/api"

// Same driver/lap picker shape as TrackLayoutChart, charting speed/throttle/
// brake over the lap instead of x/y position — see get_car_telemetry_series
// in backend/llm/tools.py for why this is a live, per-lap OpenF1 call
// rather than a bulk-cached dataset (car_data is ~3.7Hz per car per session).
export function CarTelemetryChart({ sessionKey, drivers, laps }: { sessionKey: number; drivers: Driver[]; laps: LapTimeDelta[] }) {
  const [driverNumber, setDriverNumber] = useState<number | null>(drivers[0]?.driver_number ?? null)
  const [lapNumber, setLapNumber] = useState<number | null>(null)

  const availableLaps = useMemo(
    () => [...new Set(laps.filter((l) => l.driver_number === driverNumber).map((l) => l.lap_number))].sort((a, b) => a - b),
    [laps, driverNumber]
  )
  const effectiveLap = lapNumber ?? availableLaps[0] ?? null

  const { data, isLoading, error } = useQuery({
    queryKey: ["car-telemetry", sessionKey, driverNumber, effectiveLap],
    queryFn: () => fetchCarTelemetry(sessionKey, driverNumber!, effectiveLap!),
    enabled: driverNumber != null && effectiveLap != null,
    retry: false,
  })

  const points = (data ?? []).map((p, i) => ({ ...p, i }))

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
        <div className="bg-muted/40 h-[280px] animate-pulse rounded-lg" />
      ) : error ? (
        <div className="text-destructive flex h-[280px] flex-col items-center justify-center gap-1 text-center text-sm">
          <span>OpenF1's car telemetry requires a paid/authenticated API tier this project doesn't have.</span>
          <span className="text-muted-foreground text-xs">Everything else in this app uses OpenF1's free endpoints — only live car telemetry is gated.</span>
        </div>
      ) : points.length === 0 ? (
        <div className="text-muted-foreground flex h-[280px] items-center justify-center text-sm">No telemetry for this lap.</div>
      ) : (
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={points} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="i" tick={false} stroke="var(--color-muted-foreground)" label={{ value: "Lap progress", position: "insideBottom", offset: -4, fill: "var(--color-muted-foreground)" }} />
            <YAxis yAxisId="speed" stroke="var(--color-muted-foreground)" label={{ value: "km/h", angle: -90, position: "insideLeft", fill: "var(--color-muted-foreground)" }} />
            <YAxis yAxisId="pct" orientation="right" domain={[0, 100]} stroke="var(--color-muted-foreground)" />
            <Tooltip contentStyle={{ background: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: 8 }} labelFormatter={() => ""} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line yAxisId="speed" type="monotone" dataKey="speed" name="Speed (km/h)" stroke="#ef4444" dot={false} isAnimationActive={false} />
            <Line yAxisId="pct" type="monotone" dataKey="throttle" name="Throttle %" stroke="#22c55e" dot={false} isAnimationActive={false} />
            <Line yAxisId="pct" type="monotone" dataKey="brake" name="Brake %" stroke="#eab308" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
