// One reusable table for every "list of rows" visualization added alongside
// the pit/overtakes/radio/session-result/starting-grid/standings endpoints —
// same reasoning as DriverBarChart: these are all the same shape (a plain
// table with a render override per column), so this is written once and
// parameterized rather than six near-identical table components.
export interface DataTableColumn<T> {
  key: string
  label: string
  align?: "left" | "right"
  render?: (row: T) => React.ReactNode
}

export function DataTable<T>({
  columns,
  rows,
  rowKey,
}: {
  columns: DataTableColumn<T>[]
  rows: T[]
  rowKey: (row: T) => string | number
}) {
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">No data available for this session.</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-muted-foreground">
            {columns.map((col) => (
              <th key={col.key} className={`px-3 py-2 font-medium ${col.align === "right" ? "text-right" : "text-left"}`}>
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={rowKey(row)} className="border-b border-border/50 last:border-0">
              {columns.map((col) => (
                <td key={col.key} className={`px-3 py-2 ${col.align === "right" ? "text-right" : "text-left"}`}>
                  {col.render ? col.render(row) : String((row as Record<string, unknown>)[col.key] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
