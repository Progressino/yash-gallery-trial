import type { CoverageResponse } from '../api/client'
import { operationalDataComplete } from './localSessionHint'

export function coverageJobsRunning(c: CoverageResponse | undefined): boolean {
  if (!c) return false
  return (
    c.inventory_upload_status === 'running' ||
    c.daily_inventory_upload_status === 'running' ||
    c.daily_auto_ingest_status === 'running' ||
    c.tier1_bulk_status === 'running' ||
    c.sales_rebuild === 'running' ||
    c.session_restore_status === 'running'
  )
}

export function coverageNeedsSync(c: CoverageResponse): boolean {
  const empty =
    !c.mtr && !c.sales && !c.myntra && !c.meesho && !c.flipkart && !c.snapdeal
  const needsSales =
    !c.sales &&
    (c.mtr || c.myntra || c.meesho || c.flipkart || c.snapdeal) &&
    (c.sales_rebuild === 'running' || c.daily_auto_ingest_status === 'running')
  return empty || needsSales
}

export function coveragePollIntervalMs(c: CoverageResponse | undefined): number | false {
  if (!c) return 8_000
  if (coverageJobsRunning(c)) return 3_000
  if (!operationalDataComplete(c)) return 8_000
  if (coverageNeedsSync(c)) return 15_000
  return 60_000
}
