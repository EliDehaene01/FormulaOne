import { useQuery } from "@tanstack/react-query"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { fetchRaceSummary } from "@/lib/api"

export function SummaryView({ sessionKey }: { sessionKey: number }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["summary", sessionKey],
    queryFn: () => fetchRaceSummary(sessionKey),
  })

  if (isLoading) {
    return <div className="text-muted-foreground py-12 text-center text-sm">Generating recap… (this calls the LLM, can take a few seconds)</div>
  }
  if (error || !data) {
    return <div className="text-destructive py-12 text-center text-sm">Couldn't generate a summary for this race.</div>
  }

  const { key_moments } = data

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xl">{data.headline}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <p className="text-foreground/90 whitespace-pre-line leading-relaxed">{data.narrative}</p>
        <div className="flex flex-wrap gap-2">
          {key_moments.fastest_lap && (
            <Badge variant="secondary">
              Fastest lap: #{key_moments.fastest_lap.driver_number} — {key_moments.fastest_lap.lap_duration_s.toFixed(3)}s (lap {key_moments.fastest_lap.lap_number})
            </Badge>
          )}
          {key_moments.biggest_gainer && (
            <Badge variant="secondary">
              Biggest gainer: #{key_moments.biggest_gainer.driver_number} (+{key_moments.biggest_gainer.positions_changed})
            </Badge>
          )}
          {key_moments.biggest_loser && (
            <Badge variant="secondary">
              Biggest loser: #{key_moments.biggest_loser.driver_number} ({key_moments.biggest_loser.positions_changed})
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
