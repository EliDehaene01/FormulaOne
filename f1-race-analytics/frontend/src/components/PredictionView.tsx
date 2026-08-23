import { useQuery } from "@tanstack/react-query"
import { useState } from "react"

import { FeatureBarChart } from "@/components/charts/FeatureBarChart"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { fetchDrivers, fetchFeatureImportance, fetchPrediction, fetchPredictionExplanation } from "@/lib/api"

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
  // Global, not session-scoped — a property of the trained model itself, so
  // this stays cached across race switches instead of refetching each time.
  const importance = useQuery({ queryKey: ["feature-importance"], queryFn: fetchFeatureImportance })

  const [explainDriver, setExplainDriver] = useState<number | null>(null)
  // Defaults to the predicted pole driver so "why this prediction" has
  // something to show immediately, without requiring a click first.
  const activeExplainDriver = explainDriver ?? prediction.data?.predicted_pole_driver_number ?? null
  const explanation = useQuery({
    queryKey: ["prediction-explain", sessionKey, activeExplainDriver],
    queryFn: () => fetchPredictionExplanation(sessionKey, activeExplainDriver as number),
    enabled: activeExplainDriver != null,
  })

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

      <Card>
        <CardHeader>
          <CardTitle className="text-base">What the model relies on overall</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground mb-3 text-sm">
            Permutation importance on the validation set: how much worse the model's predictions get when one input is shuffled. A property of
            the trained model, not this specific race.
          </p>
          {importance.isLoading ? (
            <div className="bg-muted/40 h-[320px] animate-pulse rounded-lg" />
          ) : importance.error || !importance.data ? (
            <p className="text-destructive text-sm">Couldn't load feature importance.</p>
          ) : (
            <FeatureBarChart
              data={importance.data.map((r) => ({ name: r.feature, value: r.mae_increase_s }))}
              valueLabel="Val MAE increase"
              unit="s"
              topN={12}
            />
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Why this prediction</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground text-sm">Driver:</span>
            <Select value={activeExplainDriver != null ? String(activeExplainDriver) : undefined} onValueChange={(value) => setExplainDriver(Number(value))}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Pick a driver" />
              </SelectTrigger>
              <SelectContent>
                {[...data.per_driver_predictions]
                  .sort((a, b) => a.predicted_lap_time_s - b.predicted_lap_time_s)
                  .map((row) => (
                    <SelectItem key={row.driver_number} value={String(row.driver_number)}>
                      {nameOf(row.driver_number)}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>

          {explanation.isLoading ? (
            <div className="bg-muted/40 h-[280px] animate-pulse rounded-lg" />
          ) : explanation.error || !explanation.data || explanation.data.error ? (
            <p className="text-destructive text-sm">{explanation.data?.error ?? "Couldn't load an explanation for this driver."}</p>
          ) : (
            <>
              <p className="text-muted-foreground text-sm">
                Predicted {explanation.data.predicted_lap_time_s.toFixed(3)}s vs. a neutral baseline (unknown driver/team/circuit, average
                conditions) of {explanation.data.baseline_prediction_s.toFixed(3)}s. Bars show what pushed the prediction{" "}
                <span style={{ color: "var(--color-destructive)" }}>slower</span> or <span style={{ color: "#22c55e" }}>faster</span> than that
                baseline — they sum to {explanation.data.attribution_total_s.toFixed(3)}s (actual difference:{" "}
                {explanation.data.prediction_minus_baseline_s.toFixed(3)}s).
              </p>
              <FeatureBarChart
                data={explanation.data.contributions.map((c) => ({ name: c.feature_name, value: c.contribution_s }))}
                valueLabel="Contribution"
                unit="s"
                diverging
                topN={10}
              />
              <p className="text-destructive text-sm">{explanation.data.caveat}</p>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
