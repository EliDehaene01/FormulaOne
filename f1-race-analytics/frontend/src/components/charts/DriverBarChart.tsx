import { useMemo } from "react"
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import type { Driver } from "@/lib/api"
import { buildDriverColorMap } from "@/lib/colors"

// One reusable chart for every "single number per driver" visualization
// (fastest lap, average lap, consistency, top speed, pit stop count,
// positions gained/lost) — six of the ~11 charts share this exact shape,
// so this is written once and parameterized rather than copy-pasted six
// times with only the data/label changed.
export function DriverBarChart({
  data,
  drivers: driverInfo,
  valueLabel,
  unit = "",
  decimals = 1,
}: {
  data: { driver_number: number; value: number }[]
  drivers: Driver[]
  valueLabel: string
  unit?: string
  decimals?: number
}) {
  const sorted = [...data].sort((a, b) => a.driver_number - b.driver_number)
  const colorMap = useMemo(() => buildDriverColorMap(driverInfo), [driverInfo])
  const nameMap = useMemo(() => new Map(driverInfo.map((d) => [d.driver_number, d])), [driverInfo])
  const labelOf = (n: number) => nameMap.get(n)?.name_acronym ?? `#${n}`

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={sorted} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis dataKey="driver_number" tickFormatter={labelOf} stroke="var(--color-muted-foreground)" />
        <YAxis
          stroke="var(--color-muted-foreground)"
          label={{ value: valueLabel, angle: -90, position: "insideLeft", fill: "var(--color-muted-foreground)" }}
        />
        <Tooltip
          contentStyle={{ background: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: 8 }}
          formatter={(value) => [`${Number(value).toFixed(decimals)}${unit}`, valueLabel]}
          labelFormatter={(d) => nameMap.get(Number(d))?.full_name ?? `#${d}`}
        />
        <Bar dataKey="value" isAnimationActive={false}>
          {sorted.map((row) => (
            <Cell key={row.driver_number} fill={colorMap.get(row.driver_number)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
