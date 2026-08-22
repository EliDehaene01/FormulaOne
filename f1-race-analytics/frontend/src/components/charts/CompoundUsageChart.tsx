import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import type { TireStint } from "@/lib/api"
import { compoundColor } from "@/lib/colors"

// How many total laps the field ran on each compound — a field-wide view
// to complement the per-driver TireStrategyChart timeline.
export function CompoundUsageChart({ data }: { data: TireStint[] }) {
  const totals = new Map<string, number>()
  for (const stint of data) {
    const laps = stint.lap_end - stint.lap_start + 1
    totals.set(stint.compound, (totals.get(stint.compound) ?? 0) + laps)
  }
  const rows = [...totals.entries()].map(([compound, laps]) => ({ compound, laps })).sort((a, b) => b.laps - a.laps)

  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis dataKey="compound" stroke="var(--color-muted-foreground)" />
        <YAxis
          stroke="var(--color-muted-foreground)"
          label={{ value: "Total laps run", angle: -90, position: "insideLeft", fill: "var(--color-muted-foreground)" }}
        />
        <Tooltip
          contentStyle={{ background: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: 8 }}
          formatter={(value) => [`${value} laps`, "laps run"]}
        />
        <Bar dataKey="laps" isAnimationActive={false}>
          {rows.map((row) => (
            <Cell key={row.compound} fill={compoundColor(row.compound)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
