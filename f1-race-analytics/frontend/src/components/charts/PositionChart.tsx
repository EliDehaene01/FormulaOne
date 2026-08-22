import { useMemo } from "react"
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import type { Driver, PositionChange } from "@/lib/api"
import { buildDriverColorMap } from "@/lib/colors"

export function PositionChart({
  data,
  drivers: driverInfo,
  selectedDrivers,
}: {
  data: PositionChange[]
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
        {/* Position 1 (the lead) at the top of the chart, like a real leaderboard. */}
        <YAxis
          reversed
          allowDecimals={false}
          domain={[1, "dataMax"]}
          stroke="var(--color-muted-foreground)"
          label={{ value: "Position", angle: -90, position: "insideLeft", fill: "var(--color-muted-foreground)" }}
        />
        <Tooltip
          contentStyle={{ background: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: 8 }}
          labelFormatter={() => ""}
          formatter={(value, _name, item) => [`P${value}`, nameMap.get(item.payload.driver_number)?.full_name ?? `#${item.payload.driver_number}`]}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {drivers.map((driverNumber) => (
          <Line
            key={driverNumber}
            data={byDriver.get(driverNumber)}
            dataKey="position"
            name={labelOf(driverNumber)}
            stroke={colorMap.get(driverNumber)}
            dot={false}
            strokeWidth={1.5}
            type="stepAfter"
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
