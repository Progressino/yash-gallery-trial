import { todayIsoIST, salesDataLagDays } from './reportingDates'

export type InventoryStalenessCoverage = {
  inventory?: boolean
  inventory_snapshot_date?: string | null
  daily_inventory_history?: boolean
  daily_inventory_history_max_date?: string | null
  inventory_staleness_warnings?: string[] | null
  inventory_snapshot_stale?: boolean
  daily_inventory_history_stale?: boolean
}

export function inventoryDataLagDays(referenceIso: string, inventoryDateIso: string): number | null {
  return salesDataLagDays(referenceIso, inventoryDateIso)
}

export function inventoryDataGapNeedsWarning(
  referenceIso: string,
  inventoryDateIso: string,
  maxExpectedLagDays = 1,
): boolean {
  const lag = inventoryDataLagDays(referenceIso, inventoryDateIso)
  if (lag == null) return false
  return lag > maxExpectedLagDays
}

/** User-facing warnings when snapshot inventory or PO history matrix is behind today. */
export function buildInventoryStalenessWarnings(
  cov: InventoryStalenessCoverage,
  referenceIso = todayIsoIST(),
): string[] {
  if (cov.inventory_staleness_warnings?.length) {
    return [...cov.inventory_staleness_warnings]
  }
  const warnings: string[] = []
  if (cov.inventory && cov.inventory_snapshot_date) {
    if (inventoryDataGapNeedsWarning(referenceIso, cov.inventory_snapshot_date)) {
      warnings.push(
        `Daily snapshot inventory is from ${cov.inventory_snapshot_date}. Upload today's file on Upload → Daily uploads.`,
      )
    }
  }
  if (cov.daily_inventory_history && cov.daily_inventory_history_max_date) {
    if (inventoryDataGapNeedsWarning(referenceIso, cov.daily_inventory_history_max_date)) {
      warnings.push(
        `Inventory history matrix ends ${cov.daily_inventory_history_max_date}. Re-upload the wide Excel on Upload → History & setup — Eff_Days in PO may be wrong.`,
      )
    }
  } else if (cov.inventory && !cov.daily_inventory_history) {
    warnings.push(
      'No daily inventory history matrix loaded. Upload the wide Inventory History Excel (OMS + Amazon sheets) for accurate Eff_Days in PO.',
    )
  }
  return warnings
}

export function inventoryStalenessIsBlocking(cov: InventoryStalenessCoverage): boolean {
  return (
    cov.inventory_snapshot_stale === true ||
    cov.daily_inventory_history_stale === true ||
    buildInventoryStalenessWarnings(cov).length > 0
  )
}
