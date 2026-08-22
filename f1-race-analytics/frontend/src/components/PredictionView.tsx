import { useQuery } from "@tanstack/react-query"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { fetchDrivers, fetchPrediction } from "@/lib/api"

// IMPORTANT SCOPE NOTE: the trained model (backend/models/) predicts
// QUALIFYING PACE — a driver's likely qualifying lap time, given their
// practice pace, form, weather, and tire compound (see
// backend/models/README.md). It does NOT predict "the next lap" during a
// race — that would be a different, sequential prediction problem this
// project hasn't built a model for. This view is honest about that rather
// than relabeling the qualifying model as something it isn't.
export function PredictionView({ sessionKey }: { sessionKey: number }) {
  const prediction = useQuery({ queryKey: ["prediction", sessionKey], queryFn: () => fetchPrediction(sessionKey) })
  const drivers = useQuery({ queryKey: ["drivers", sessionKey], queryFn: () => fetchDrivers(sessionKey) })

  if (prediction.isLoading) {
    return <div className="text-muted-foreground py-12 text-center text-sm">Running the model… (needs a PyTorch environment — see backend/models/README.md if this errors)</div>
  }
  if (prediction.error || !prediction.data || prediction.data.error) {
    return (
      <div className="text-destructive py-12 text-center text-sm">
        {prediction.data?.error ?? "Couldn't get a prediction for this session — it may not have both practice and qualifying data cached."}
      </div>
    )
  }

  const data = prediction.data
  const nameOf = (n: number) => drivers.data?.find((d) => d.driver_number === n)?.full_name ?? `#${n}`

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Qualifying pace prediction</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <p className="text-muted-foreground text-sm">
            Predicted pole: <span className="text-foreground font-medium">{nameOf(data.predicted_pole_driver_number)}</span> ({data.predicted_pole_time_s.toFixed(3)}s) — actual pole was{" "}
            <span className="text-foreground font-medium">{nameOf(data.actual_pole_driver_number)}</span> ({data.actual_pole_time_s.toFixed(3)}s).
          </p>
          <p className="text-destructive text-sm">{data.caveat}</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Predicted vs. actual, per driver</CardTitle>
        </CardHeader>
        <CardContent>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-muted-foreground border-border border-b text-left">
                <th className="py-2 font-normal">Driver</th>
                <th className="py-2 font-normal">Predicted</th>
                <th className="py-2 font-normal">Actual</th>
                <th className="py-2 font-normal">Error</th>
              </tr>
            </thead>
            <tbody>
              {[...data.per_driver_predictions]
                .sort((a, b) => a.predicted_lap_time_s - b.predicted_lap_time_s)
                .map((row) => (
                  <tr key={row.driver_number} className="border-border/50 border-b">
                    <td className="py-2">{nameOf(row.driver_number)}</td>
                    <td className="py-2">{row.predicted_lap_time_s.toFixed(3)}s</td>
                    <td className="py-2">{row.actual_lap_time_s.toFixed(3)}s</td>
                    <td className="py-2">{(row.predicted_lap_time_s - row.actual_lap_time_s).toFixed(3)}s</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  )
}
