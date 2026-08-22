import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import type { Driver, TireStint } from "@/lib/api"
import { compoundColor } from "@/lib/colors"

// Recharts has no built-in Gantt chart, so this builds one out of a
// stacked horizontal BarChart: one row per driver, one Bar series per
// "stint slot" (stint 1, stint 2, ...), each bar's length = laps run on
// that stint, each bar's COLOR set per-row via <Cell> since two drivers'
// stint 1 can be different compounds. Stints are contiguous (no gaps
// between them for one driver), so no invisible spacer bars are needed —
// stacking the real stint-length bars end to end IS the strategy timeline.
export function TireStrategyChart({
  data,
  drivers: driverInfo,
  selectedDrivers,
}: {
  data: TireStint[]
  drivers: Driver[]
  selectedDrivers?: Set<number> | null
}) {
  const nameMap = new Map(driverInfo.map((d) => [d.driver_number, d.name_acronym]))
  const filtered = selectedDrivers ? data.filter((s) => selectedDrivers.has(s.driver_number)) : data
  const byDriver = new Map<number, TireStint[]>()
  for (const stint of filtered) {
    if (!byDriver.has(stint.driver_number)) byDriver.set(stint.driver_number, [])
    byDriver.get(stint.driver_number)!.push(stint)
  }

  const maxStints = Math.max(0, ...[...byDriver.values()].map((stints) => stints.length))

  const rows = [...byDriver.entries()]
    .sort(([a], [b]) => a - b)
    .map(([driverNumber, stints]) => {
      const sorted = [...stints].sort((a, b) => a.stint_number - b.stint_number)
      const row: Record<string, number | string> = { driver: nameMap.get(driverNumber) ?? `#${driverNumber}` }
      sorted.forEach((stint, i) => {
        row[`stint_${i}_length`] = stint.lap_end - stint.lap_start + 1
        row[`stint_${i}_compound`] = stint.compound
      })
      return row
    })

  return (
    <ResponsiveContainer width="100%" height={Math.max(320, rows.length * 28)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" horizontal={false} />
        <XAxis
          type="number"
          stroke="var(--color-muted-foreground)"
          label={{ value: "Lap", position: "insideBottom", offset: -4, fill: "var(--color-muted-foreground)" }}
        />
        <YAxis dataKey="driver" type="category" width={48} stroke="var(--color-muted-foreground)" />
        <Tooltip
          contentStyle={{ background: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: 8 }}
          formatter={(value, name, item) => {
            const index = String(name).match(/stint_(\d+)_length/)?.[1]
            const compound = index != null ? item.payload[`stint_${index}_compound`] : ""
            return [`${value} laps`, compound]
          }}
        />
        {Array.from({ length: maxStints }, (_, i) => (
          <Bar key={i} dataKey={`stint_${i}_length`} stackId="strategy" isAnimationActive={false}>
            {rows.map((row, rowIndex) => (
              <Cell key={rowIndex} fill={compoundColor(String(row[`stint_${i}_compound`] ?? "OTHER"))} />
            ))}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}
