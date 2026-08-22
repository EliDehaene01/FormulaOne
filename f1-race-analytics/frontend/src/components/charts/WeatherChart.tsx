import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

import type { WeatherPoint } from "@/lib/api"

// "Rain forecast" isn't something OpenF1 (or this app) can provide — it's
// historical race data, not a live meteorological feed, so there's no
// forward-looking prediction to show. This charts the ACTUAL observed
// rainfall alongside temperature, which is the real signal we have.
export function WeatherChart({ data }: { data: WeatherPoint[] }) {
  const points = data.map((row) => ({ ...row, t: new Date(row.date).getTime() }))

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={points} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tick={false} stroke="var(--color-muted-foreground)" />
        <YAxis
          yAxisId="temp"
          stroke="var(--color-muted-foreground)"
          label={{ value: "°C", angle: -90, position: "insideLeft", fill: "var(--color-muted-foreground)" }}
        />
        <YAxis
          yAxisId="rain"
          orientation="right"
          stroke="var(--color-muted-foreground)"
          label={{ value: "Rainfall", angle: 90, position: "insideRight", fill: "var(--color-muted-foreground)" }}
        />
        <Tooltip
          contentStyle={{ background: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: 8 }}
          labelFormatter={() => ""}
          formatter={(value, name) => (name === "Rainfall" ? [value, name] : [`${Number(value).toFixed(1)}°C`, name])}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Area yAxisId="rain" type="monotone" dataKey="rainfall" name="Rainfall" fill="#3b82f6" stroke="#3b82f6" fillOpacity={0.25} isAnimationActive={false} />
        <Line yAxisId="temp" type="monotone" dataKey="air_temperature" name="Air temp" stroke="#eab308" dot={false} isAnimationActive={false} />
        <Line yAxisId="temp" type="monotone" dataKey="track_temperature" name="Track temp" stroke="#ef4444" dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
