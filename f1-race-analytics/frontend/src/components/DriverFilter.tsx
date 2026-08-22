import { useMemo } from "react"

import { Badge } from "@/components/ui/badge"
import type { Driver } from "@/lib/api"
import { buildDriverColorMap } from "@/lib/colors"

// The single shared "select things in the chart" control (per user
// request) — toggling a driver here filters EVERY chart on the page at
// once, rather than each of ~11 charts needing its own bespoke
// click-to-isolate logic. null selection = show everyone.
export function DriverFilter({
  drivers,
  selected,
  onChange,
}: {
  drivers: Driver[]
  selected: Set<number> | null
  onChange: (next: Set<number> | null) => void
}) {
  const colorMap = useMemo(() => buildDriverColorMap(drivers), [drivers])

  function toggle(driverNumber: number) {
    const current = selected ?? new Set(drivers.map((d) => d.driver_number))
    const next = new Set(current)
    if (next.has(driverNumber)) next.delete(driverNumber)
    else next.add(driverNumber)
    onChange(next.size === drivers.length ? null : next)
  }

  const isActive = (d: number) => selected === null || selected.has(d)

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-muted-foreground text-sm">Drivers:</span>
      {drivers.map((driver) => (
        <Badge
          key={driver.driver_number}
          variant={isActive(driver.driver_number) ? "default" : "outline"}
          className="cursor-pointer select-none"
          style={
            isActive(driver.driver_number)
              ? { backgroundColor: colorMap.get(driver.driver_number), color: "#0b0d12", borderColor: colorMap.get(driver.driver_number) }
              : undefined
          }
          onClick={() => toggle(driver.driver_number)}
          title={driver.full_name}
        >
          {driver.name_acronym}
        </Badge>
      ))}
      {selected !== null && (
        <button type="button" className="text-primary text-sm underline underline-offset-2" onClick={() => onChange(null)}>
          Show all
        </button>
      )}
    </div>
  )
}
