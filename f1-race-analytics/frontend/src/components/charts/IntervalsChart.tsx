import { useMemo } from "react"
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import type { Driver, IntervalPoint } from "@/lib/api"
import { buildDriverColorMap } from "@/lib/colors"

// Same per-driver-lines-over-time shape as PositionChart, just charting
// gap-to-leader (seconds, lower = closer to the front) instead of position.
export function IntervalsChart({
  data,
  drivers: driverInfo,
  selectedDrivers,
}: {
  data: IntervalPoint[]
  drivers: Driver[]
  selectedDrivers?: Set<number> | null
}) {
  const withTimestamp = data.map((row) => ({ ...row, t: new Date(row.date).getTime() }))

  const byDriver = new Map<number, typeof withTimestamp>()
  for (const row of withTimestamp) {
    if (!byDriver.has(row.driver_number)) byDriver.set(row.driver_number, [])
    byDriver.get(row.driver_number)!.push(row)
  }
  const allDrivers = [...byDriver.keys()].sort((a, b) => a - b)
  const drivers = selectedDrivers ? allDrivers.filter((d) => selectedDrivers.has(d)) : allDrivers

  const colorMap = useMemo(() => buildDriverColorMap(driverInfo), [driverInfo])
  const nameMap = useMemo(() => new Map(driverInfo.map((d) => [d.driver_number, d])), [driverInfo])
  const labelOf = (n: number) => nameMap.get(n)?.name_acronym ?? `#${n}`

  if (data.length === 0) {
    return <p className="text-sm text-muted-foreground">No interval data for this session (only recorded for Race/Sprint sessions).</p>
  }

  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={withTimestamp} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis
          dataKey="t"
          type="number"
          domain={["dataMin", "dataMax"]}
          tick={false}
          stroke="var(--color-muted-foreground)"
          label={{ value: "Race progress", position: "insideBottom", offset: -4, fill: "var(--color-muted-foreground)" }}
        />
        <YAxis
          stroke="var(--color-muted-foreground)"
          label={{ value: "Gap to leader (s)", angle: -90, position: "insideLeft", fill: "var(--color-muted-foreground)" }}
        />
        <Tooltip
          contentStyle={{ background: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: 8 }}
          labelFormatter={() => ""}
          formatter={(value, _name, item) => [`+${Number(value).toFixed(1)}s`, nameMap.get(item.payload.driver_number)?.full_name ?? `#${item.payload.driver_number}`]}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {drivers.map((driverNumber) => (
          <Line
            key={driverNumber}
            data={byDriver.get(driverNumber)}
            dataKey="gap_to_leader"
            name={labelOf(driverNumber)}
            stroke={colorMap.get(driverNumber)}
            dot={false}
            strokeWidth={1.5}
            type="monotone"
            isAnimationActive={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
