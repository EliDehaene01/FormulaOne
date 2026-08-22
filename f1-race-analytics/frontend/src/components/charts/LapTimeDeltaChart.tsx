import { useMemo } from "react"
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import type { Driver, LapTimeDelta } from "@/lib/api"
import { buildDriverColorMap } from "@/lib/colors"

export function LapTimeDeltaChart({
  data,
  drivers: driverInfo,
  selectedDrivers,
}: {
  data: LapTimeDelta[]
  drivers: Driver[]
  selectedDrivers?: Set<number> | null
}) {
  const byDriver = new Map<number, LapTimeDelta[]>()
  for (const row of data) {
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
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis
          dataKey="lap_number"
          type="number"
          allowDuplicatedCategory={false}
          stroke="var(--color-muted-foreground)"
          label={{ value: "Lap", position: "insideBottom", offset: -4, fill: "var(--color-muted-foreground)" }}
        />
        <YAxis
          stroke="var(--color-muted-foreground)"
          label={{ value: "Delta to fastest (s)", angle: -90, position: "insideLeft", fill: "var(--color-muted-foreground)" }}
        />
        <Tooltip
          contentStyle={{ background: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: 8 }}
          labelFormatter={(lap) => `Lap ${lap}`}
          formatter={(value, _name, item) => [`${Number(value).toFixed(3)}s`, nameMap.get(item.payload.driver_number)?.full_name ?? `#${item.payload.driver_number}`]}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {drivers.map((driverNumber) => (
          <Line
            key={driverNumber}
            data={byDriver.get(driverNumber)}
            dataKey="delta_to_fastest_s"
            name={labelOf(driverNumber)}
            stroke={colorMap.get(driverNumber)}
            dot={false}
            strokeWidth={1.5}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
