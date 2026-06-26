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
    const histMax = cov.daily_inventory_history_max_date || ''
    const histCurrent =
      cov.daily_inventory_history &&
      histMax &&
      !inventoryDataGapNeedsWarning(referenceIso, histMax)
    const filtered = cov.inventory_staleness_warnings.filter(w => {
      if (!histCurrent) return true
      return !/snapshot inventory is from unknown/i.test(w)
    })
    if (filtered.length) return filtered
  }
  const warnings: string[] = []
  const histMax = cov.daily_inventory_history_max_date || ''
  const histStale =
    cov.daily_inventory_history &&
    histMax &&
    inventoryDataGapNeedsWarning(referenceIso, histMax)
  if (cov.daily_inventory_history && histMax) {
    if (histStale) {
      warnings.push(
        `Inventory history matrix ends ${histMax}. Re-upload the wide Excel on Upload → History & setup — Eff_Days in PO may be wrong.`,
      )
    }
  } else if (cov.inventory && !cov.daily_inventory_history) {
    warnings.push(
      'No daily inventory history matrix loaded. Upload the wide Inventory History Excel (OMS + Amazon sheets) for accurate Eff_Days in PO.',
    )
  }
  if (cov.inventory && cov.inventory_snapshot_date && !histStale) {
    if (inventoryDataGapNeedsWarning(referenceIso, cov.inventory_snapshot_date)) {
      warnings.push(
        `Daily snapshot inventory is from ${cov.inventory_snapshot_date}. Upload today's file on Upload → Daily uploads.`,
      )
    }
  } else if (cov.inventory && !cov.inventory_snapshot_date && !cov.daily_inventory_history && !histMax) {
    warnings.push(
      "Daily snapshot inventory date is unknown. Upload today's OMS + marketplace inventory on Upload → Daily uploads.",
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
