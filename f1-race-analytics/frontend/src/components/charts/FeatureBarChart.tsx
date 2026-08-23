import { useMemo } from "react"
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

// One reusable horizontal bar chart for "named value" model-explainability
// visualizations — permutation importance (magnitude only, one hue) and
// per-prediction Captum contributions (signed, diverging) share this exact
// shape. Horizontal, not vertical like DriverBarChart: feature names
// ("practice_best_sector2_gap_s") are long, and a horizontal layout reads
// them without rotating labels.
const MAGNITUDE_COLOR = "var(--color-primary)"
const POSITIVE_COLOR = "var(--color-destructive)" // pushes the predicted lap time UP (slower)
const NEGATIVE_COLOR = "#22c55e" // pushes it DOWN (faster) — same green this app already uses for intermediate tyres

export function FeatureBarChart({
  data,
  valueLabel,
  unit = "s",
  diverging = false,
  topN = 10,
}: {
  data: { name: string; value: number }[]
  valueLabel: string
  unit?: string
  diverging?: boolean
  topN?: number
}) {
  // Backend already returns these sorted most-to-least influential — just
  // reverse for display, since a horizontal bar chart reads top-to-bottom
  // but Recharts' category axis renders the first entry at the bottom.
  const rows = useMemo(() => [...data.slice(0, topN)].reverse(), [data, topN])

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, rows.length * 32)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" horizontal={false} />
        <XAxis type="number" stroke="var(--color-muted-foreground)" tickFormatter={(v) => `${v}${unit}`} />
        <YAxis type="category" dataKey="name" width={230} stroke="var(--color-muted-foreground)" tick={{ fontSize: 11 }} />
        {diverging && <ReferenceLine x={0} stroke="var(--color-border)" />}
        <Tooltip
          contentStyle={{ background: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: 8 }}
          formatter={(value) => {
            const n = Number(value)
            return [`${diverging && n > 0 ? "+" : ""}${n.toFixed(3)}${unit}`, valueLabel]
          }}
        />
        <Bar dataKey="value" isAnimationActive={false} radius={2}>
          {rows.map((row) => (
            <Cell key={row.name} fill={diverging ? (row.value >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR) : MAGNITUDE_COLOR} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
