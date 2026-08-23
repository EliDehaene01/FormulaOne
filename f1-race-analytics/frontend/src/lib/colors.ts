// Shared color helpers so every chart colors the same driver consistently,
// and tire compounds always match their real-world F1 broadcast colors
// (red=soft, yellow=medium, white=hard, green=intermediate, blue=wet) —
// familiar to anyone who's watched a race, unlike an arbitrary palette.

const DRIVER_PALETTE = [
  "#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4",
  "#3b82f6", "#8b5cf6", "#ec4899", "#84cc16", "#14b8a6",
]

export function driverColor(index: number): string {
  return DRIVER_PALETTE[index % DRIVER_PALETTE.length]
}

const COMPOUND_COLORS: Record<string, string> = {
  SOFT: "#ef4444",
  MEDIUM: "#eab308",
  HARD: "#f8fafc",
  INTERMEDIATE: "#22c55e",
  WET: "#3b82f6",
  OTHER: "#9ca3af",
}

export function compoundColor(compound: string): string {
  return COMPOUND_COLORS[compound] ?? COMPOUND_COLORS.OTHER
}

interface DriverLike {
  driver_number: number
  team_name: string
  team_colour: string | null
}

// Lightens a #rrggbb hex color by blending it toward white — used to tell
// teammates apart, since OpenF1's team_colour is one color PER TEAM, not
// per driver, and two identically-colored lines on the same chart would be
// indistinguishable.
function lighten(hex: string, amount: number): string {
  const n = parseInt(hex.replace("#", ""), 16)
  const r = Math.min(255, Math.round(((n >> 16) & 255) + (255 - ((n >> 16) & 255)) * amount))
  const g = Math.min(255, Math.round(((n >> 8) & 255) + (255 - ((n >> 8) & 255)) * amount))
  const b = Math.min(255, Math.round((n & 255) + (255 - (n & 255)) * amount))
  return `#${[r, g, b].map((c) => c.toString(16).padStart(2, "0")).join("")}`
}

// Real team colors (per user request) instead of an arbitrary palette,
// with the second driver on each team lightened so teammates are still
// distinguishable on the same chart. Falls back to the generic palette for
// any driver missing a team_colour (shouldn't normally happen, but the
// data is third-party and not guaranteed complete for every session).
export function buildDriverColorMap(drivers: DriverLike[]): Map<number, string> {
  const seenTeams = new Map<string, number>()
  const map = new Map<number, string>()
  drivers.forEach((driver, i) => {
    if (!driver.team_colour) {
      map.set(driver.driver_number, driverColor(i))
      return
    }
    const hex = `#${driver.team_colour.replace("#", "")}`
    const teammateIndex = seenTeams.get(driver.team_name) ?? 0
    seenTeams.set(driver.team_name, teammateIndex + 1)
    map.set(driver.driver_number, teammateIndex === 0 ? hex : lighten(hex, 0.45))
  })
  return map
}
