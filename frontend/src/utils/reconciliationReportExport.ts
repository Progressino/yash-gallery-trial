import type { UploadReconciliationReport } from '../api/client'
import { downloadCsvBlob } from '../lib/poDisplay'

function csvRow(cells: (string | number | boolean | undefined | null)[]): string {
  return cells.map(c => JSON.stringify(c ?? '')).join(',')
}

/** Download full daily vs monthly reconciliation (mismatches + file inventory) as CSV. */
export function downloadReconciliationReportCsv(report: UploadReconciliationReport): void {
  const lines: string[] = []
  const stamp = new Date().toISOString().slice(0, 19).replace('T', ' ')

  lines.push(csvRow(['Daily vs Monthly Reconciliation Report']))
  lines.push(csvRow(['Generated', stamp]))
  lines.push(csvRow(['Files tracked', report.file_count]))
  lines.push(csvRow(['Parquets loaded for compare', report.parquet_files_loaded ?? '']))
  lines.push(csvRow(['Mismatches', report.mismatch_count]))
  lines.push(csvRow(['Deduped rows (not double-counted)', report.total_collapsible_rows ?? 0]))
  if (report.return_overlay?.units) {
    lines.push(csvRow(['Return overlay units', report.return_overlay.units]))
    lines.push(csvRow(['Return overlay SKUs', report.return_overlay.skus ?? 0]))
  }
  if (report.elapsed_ms != null) {
    lines.push(csvRow(['Report runtime (ms)', report.elapsed_ms]))
  }

  lines.push('')
  lines.push(csvRow(['MISMATCHES — daily vs monthly by month, company, txn type']))
  lines.push(
    csvRow([
      'month',
      'company',
      'txn_type',
      'daily_units',
      'monthly_units',
      'unit_diff',
      'daily_amount',
      'monthly_amount',
      'amount_diff',
    ]),
  )
  for (const m of report.mismatches) {
    lines.push(
      csvRow([
        m.month,
        m.segment,
        m.txn,
        m.daily_units,
        m.monthly_units,
        m.unit_diff,
        m.daily_amount,
        m.monthly_amount,
        m.amount_diff,
      ]),
    )
  }

  lines.push('')
  lines.push(csvRow(['FILE INVENTORY — all Tier-3 uploads']))
  lines.push(
    csvRow(['platform', 'filename', 'upload_kind', 'rows', 'date_from', 'date_to', 'uploaded_at']),
  )
  for (const f of report.files) {
    lines.push(
      csvRow([
        f.platform,
        f.filename,
        f.upload_kind,
        f.rows,
        f.date_from,
        f.date_to,
        f.uploaded_at,
      ]),
    )
  }

  const dailyMonthly = report.files.filter(f => f.upload_kind === 'daily' || f.upload_kind === 'monthly')
  if (dailyMonthly.length) {
    lines.push('')
    lines.push(csvRow(['DAILY + MONTHLY FILES ONLY']))
    lines.push(
      csvRow(['platform', 'filename', 'upload_kind', 'rows', 'date_from', 'date_to', 'uploaded_at']),
    )
    for (const f of dailyMonthly) {
      lines.push(
        csvRow([
          f.platform,
          f.filename,
          f.upload_kind,
          f.rows,
          f.date_from,
          f.date_to,
          f.uploaded_at,
        ]),
      )
    }
  }

  const dedup = report.dedup_by_platform ?? {}
  const dedupRows = Object.entries(dedup).filter(([, v]) => (v.raw_rows ?? 0) > 0)
  if (dedupRows.length) {
    lines.push('')
    lines.push(csvRow(['DEDUP BY PLATFORM']))
    lines.push(csvRow(['platform', 'raw_rows', 'deduped_rows', 'collapsible_rows', 'note']))
    for (const [plat, v] of dedupRows) {
      lines.push(
        csvRow([
          plat,
          v.raw_rows,
          v.deduped_rows,
          v.collapsible_rows ?? 0,
          (v as { note?: string }).note ?? '',
        ]),
      )
    }
  }

  const date = new Date().toISOString().slice(0, 10)
  downloadCsvBlob(lines.join('\n'), `daily-vs-monthly-reconciliation-${date}.csv`)
}
