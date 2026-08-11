/** Pure helpers for New JO form line list (Manual SO multi-size). */

export type DraftJoLine = {
  so_number: string
  sku: string
  sku_name: string
  style: string
  planned_qty: number
  so_qty?: number
  vendor_rate: number
  remarks: string
}

/** Append a line or replace qty for the same SKU — never wipe the list. */
export function appendManualJoLine(lines: DraftJoLine[], next: DraftJoLine): DraftJoLine[] {
  const sku = (next.sku || '').trim()
  if (!sku || !(Number(next.planned_qty) > 0)) return lines
  const key = sku.toUpperCase()
  const idx = lines.findIndex(l => (l.sku || '').trim().toUpperCase() === key)
  if (idx >= 0) {
    return lines.map((x, j) =>
      j === idx
        ? {
            ...x,
            planned_qty: next.planned_qty,
            so_qty: next.so_qty ?? next.planned_qty,
            sku_name: next.sku_name || x.sku_name,
            vendor_rate: next.vendor_rate || x.vendor_rate,
          }
        : x,
    )
  }
  return [...lines, { ...next, sku }]
}
