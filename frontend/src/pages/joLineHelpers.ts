/** Pure helpers for New JO form line list (Manual SO multi-size) and JO export. */

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

export type JoExportLine = {
  sku?: string
  sku_name?: string
  style?: string
  planned_qty?: number
  received_qty?: number
  issued_qty?: number
  balance_qty?: number
  remarks?: string
}

export type JoExportSource = {
  jo_number: string
  jo_date?: string
  so_number?: string
  so_source?: string
  sku?: string
  sku_name?: string
  process?: string
  status?: string
  planned_qty?: number
  received_qty?: number
  issued_qty?: number
  balance_qty?: number
  vendor_name?: string
  fabric_code?: string
  fabric_qty?: number
  exec_type?: string
  remarks?: string
  lines?: JoExportLine[]
}

export const JO_EXPORT_HEADERS = [
  'jo_number', 'jo_date', 'so_number', 'so_source', 'sku', 'sku_name', 'style', 'size',
  'process', 'status', 'planned_qty', 'received_qty', 'issued_qty', 'balance_qty',
  'vendor_name', 'fabric_code', 'fabric_qty', 'exec_type', 'remarks',
] as const

/** Size token from style field, else last hyphen segment of SKU (e.g. 1303YKBLACK-4XL → 4XL). */
export function sizeFromSku(sku: string, style?: string): string {
  const st = String(style || '').trim()
  if (st) return st
  const s = String(sku || '').trim()
  const i = s.lastIndexOf('-')
  if (i > 0 && i < s.length - 1) return s.slice(i + 1)
  return ''
}

export function joLineSkus(jo: { sku?: string; lines?: { sku?: string }[] }): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const l of jo.lines || []) {
    const s = String(l.sku || '').trim()
    if (!s) continue
    const k = s.toUpperCase()
    if (seen.has(k)) continue
    seen.add(k)
    out.push(s)
  }
  const header = String(jo.sku || '').trim()
  if (header && !seen.has(header.toUpperCase())) out.unshift(header)
  return out
}

/** Header label listing every size SKU so a multi-line JO is not shown as one size. */
export function formatJoHeaderSku(jo: {
  sku?: string
  sku_name?: string
  lines?: { sku?: string; sku_name?: string; style?: string; planned_qty?: number }[]
}): string {
  const lines = (jo.lines || []).filter(l => String(l.sku || '').trim())
  const unique: { sku: string; sku_name: string; style: string; planned_qty: number }[] = []
  const seen = new Set<string>()
  for (const l of lines) {
    const sku = String(l.sku || '').trim()
    const k = sku.toUpperCase()
    if (seen.has(k)) continue
    seen.add(k)
    unique.push({
      sku,
      sku_name: String(l.sku_name || '').trim(),
      style: String(l.style || '').trim(),
      planned_qty: Number(l.planned_qty) || 0,
    })
  }
  if (unique.length <= 1) {
    const sku = unique[0]?.sku || String(jo.sku || '').trim()
    const name = unique[0]?.sku_name || String(jo.sku_name || '').trim()
    if (!sku) return name
    return name ? `${sku} — ${name}` : sku
  }
  return unique
    .map(l => {
      const size = sizeFromSku(l.sku, l.style)
      const qty = l.planned_qty > 0 ? ` (${l.planned_qty})` : ''
      return size && l.sku.toUpperCase().endsWith(`-${size.toUpperCase()}`)
        ? `${l.sku}${qty}`
        : `${l.sku}${size ? ` ${size}` : ''}${qty}`
    })
    .join(' · ')
}

export function joMatchesSkuFilter(
  jo: { sku?: string; lines?: { sku?: string }[] },
  filterSku: string,
): boolean {
  const want = String(filterSku || '').trim().toUpperCase()
  if (!want) return true
  if (String(jo.sku || '').trim().toUpperCase() === want) return true
  return (jo.lines || []).some(l => String(l.sku || '').trim().toUpperCase() === want)
}

/**
 * One CSV row per JO line SKU. Header-only / empty-line JOs stay a single row.
 * Does not collapse multi-size Cutting JOs onto the header SKU.
 */
export function expandJoExportRows(jo: JoExportSource): Array<Array<string | number>> {
  const lines = (jo.lines || []).filter(l => String(l.sku || '').trim())
  const sources: JoExportLine[] = lines.length > 0
    ? lines
    : [{
        sku: jo.sku,
        sku_name: jo.sku_name,
        style: '',
        planned_qty: jo.planned_qty,
        received_qty: jo.received_qty,
        issued_qty: jo.issued_qty,
        balance_qty: jo.balance_qty,
        remarks: jo.remarks,
      }]
  return sources.map(s => {
    const sku = String(s.sku || jo.sku || '').trim()
    const planned = Number(s.planned_qty ?? jo.planned_qty) || 0
    const received = Number(s.received_qty ?? 0) || 0
    const issued = Number(s.issued_qty ?? jo.issued_qty) || 0
    const balance = s.balance_qty != null && s.balance_qty !== undefined
      ? Number(s.balance_qty) || 0
      : planned - received
    const style = String(s.style || '').trim()
    const size = sizeFromSku(sku, style)
    return [
      jo.jo_number,
      jo.jo_date || '',
      jo.so_number || '',
      (jo.so_source || 'system'),
      sku,
      String(s.sku_name || jo.sku_name || ''),
      style,
      size,
      jo.process || '',
      jo.status || '',
      planned,
      received,
      issued,
      balance,
      jo.vendor_name || '',
      jo.fabric_code || '',
      jo.fabric_qty ?? '',
      jo.exec_type || '',
      String(s.remarks || jo.remarks || ''),
    ]
  })
}
