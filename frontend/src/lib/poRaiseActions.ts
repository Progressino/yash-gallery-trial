import api from '../api/client'
import { archivePoExportOnServer } from './archivePoExport'
import { calendarDateIST } from './dates'
import { downloadCsvBlob } from './poDisplay'

export type PoRaiseRow = Record<string, unknown> & { Final_PO_Qty: number }

export function exportRaisePoCsv(rows: PoRaiseRow[], poNumber: string) {
  const cols = ['PO_No', 'OMS_SKU', 'Priority', 'Days_Left', 'ADS', 'Gross_PO_Qty', 'PO_Pipeline_Total', 'Final_PO_Qty']
  const header = cols.join(',')
  const body = rows
    .map(r =>
      cols
        .map(c => {
          if (c === 'PO_No') return JSON.stringify(poNumber)
          return JSON.stringify(r[c] ?? '')
        })
        .join(','),
    )
    .join('\n')
  const csv = header + '\n' + body
  const safeNo = poNumber.replace(/[^\w-]+/g, '_')
  downloadCsvBlob(csv, `${safeNo}_raise.csv`)
  void archivePoExportOnServer(csv, calendarDateIST())
}

export async function confirmPoRaiseOnServer(opts: {
  rows: PoRaiseRow[]
  raisedDate: string
  groupByParent: boolean
}): Promise<{
  ok: boolean
  message?: string
  poNumber?: string
  raisedDate?: string
  totalQty?: number
}> {
  const payloadRows = opts.rows
    .map(r => ({
      oms_sku: String(r.OMS_SKU ?? ''),
      qty: Math.max(0, Math.floor(Number(r.Final_PO_Qty ?? 0))),
    }))
    .filter(r => r.oms_sku.length > 0 && r.qty > 0)
  if (payloadRows.length === 0) {
    return { ok: false, message: 'No positive quantities to record.' }
  }
  const { data } = await api.post<{
    ok?: boolean
    message?: string
    po_number?: string
    raised_date?: string
    total_qty?: number
  }>('/po/raise-confirm', {
    rows: payloadRows,
    raised_date: opts.raisedDate,
    group_by_parent: opts.groupByParent,
  })
  if (!data?.ok) {
    return { ok: false, message: data?.message || 'Could not save raise ledger.' }
  }
  return {
    ok: true,
    message: data.message,
    poNumber: data.po_number || `PO-${opts.raisedDate}`,
    raisedDate: data.raised_date || opts.raisedDate,
    totalQty: data.total_qty ?? payloadRows.reduce((s, r) => s + r.qty, 0),
  }
}
