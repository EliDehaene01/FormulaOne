import { useQuery } from "@tanstack/react-query"
import { useMemo, useState } from "react"

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { fetchRaces, type RaceSession } from "@/lib/api"

// OpenF1's session_name is the raw value ("Practice 1", "Race") — this is
// just how it's labeled in the UI dropdown, per the user's requested
// wording ("Free Practice 1", "Grand Prix").
const SESSION_LABELS: Record<string, string> = {
  "Practice 1": "Free Practice 1",
  "Practice 2": "Free Practice 2",
  "Practice 3": "Free Practice 3",
  Qualifying: "Qualifying",
  "Sprint Qualifying": "Sprint Qualifying",
  Sprint: "Sprint",
  Race: "Grand Prix",
}

export function RaceSelector({
  selected,
  onSelect,
}: {
  selected: number | null
  onSelect: (sessionKey: number, label: string) => void
}) {
  const { data: allSessions, isLoading, error } = useQuery({ queryKey: ["races"], queryFn: fetchRaces })

  const [year, setYear] = useState<string | null>(null)
  const [meeting, setMeeting] = useState<string | null>(null)

  const years = useMemo(() => [...new Set((allSessions ?? []).map((s) => s.year))].sort((a, b) => b - a), [allSessions])

  const meetings = useMemo(() => {
    if (!year) return []
    const inYear = (allSessions ?? []).filter((s) => String(s.year) === year)
    const seen = new Map<string, RaceSession>()
    for (const s of inYear.sort((a, b) => a.date_start.localeCompare(b.date_start))) {
      if (!seen.has(s.meeting_name)) seen.set(s.meeting_name, s)
    }
    return [...seen.keys()]
  }, [allSessions, year])

  const sessions = useMemo(() => {
    if (!year || !meeting) return []
    return (allSessions ?? [])
      .filter((s) => String(s.year) === year && s.meeting_name === meeting)
      .sort((a, b) => a.date_start.localeCompare(b.date_start))
  }, [allSessions, year, meeting])

  if (isLoading) return <div className="text-muted-foreground text-sm">Loading races…</div>
  if (error) return <div className="text-destructive text-sm">Couldn't load races — is the backend running?</div>

  return (
    <div className="flex flex-wrap gap-2">
      <Select
        value={year ?? undefined}
        onValueChange={(value) => {
          setYear(value)
          setMeeting(null)
        }}
      >
        <SelectTrigger className="w-28">
          <SelectValue placeholder="Season" />
        </SelectTrigger>
        <SelectContent>
          {years.map((y) => (
            <SelectItem key={y} value={String(y)}>
              {y}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Select
        value={meeting ?? undefined}
        onValueChange={(value) => setMeeting(value)}
        disabled={!year}
      >
        <SelectTrigger className="w-56">
          <SelectValue placeholder="Race weekend" />
        </SelectTrigger>
        <SelectContent>
          {meetings.map((m) => (
            <SelectItem key={m} value={m}>
              {m}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Select
        value={selected ? String(selected) : undefined}
        onValueChange={(value) => {
          const session = sessions.find((s) => s.session_key === Number(value))
          if (session) onSelect(session.session_key, `${session.year} ${session.meeting_name} — ${SESSION_LABELS[session.session_name] ?? session.session_name}`)
        }}
        disabled={!meeting}
      >
        <SelectTrigger className="w-44">
          <SelectValue placeholder="Session" />
        </SelectTrigger>
        <SelectContent>
          {sessions.map((s) => (
            <SelectItem key={s.session_key} value={String(s.session_key)}>
              {SESSION_LABELS[s.session_name] ?? s.session_name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
