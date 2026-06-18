/**
 * Lightweight browser snapshot of server session coverage (not full sales/inventory data).
 *
 * Full cache (millions of rows) lives on the server: warm cache, PostgreSQL session, GitHub Release.
 * Storing that in localStorage/IndexedDB would be hundreds of MB and is not feasible.
 *
 * This hint lets the app skip slow hydrate-warm + GitHub cacheLoad on return visits when the
 * same browser still has a valid session cookie and the server already holds the data.
 */
import type { CoverageResponse } from '../api/client'

const STORAGE_KEY = 'erp_local_session_hint_v1'
const DEFAULT_TTL_MS = 6 * 60 * 60 * 1000 // 6 hours

export type LocalSessionHint = {
  savedAt: number
  sales: boolean
  sales_rows: number
  mtr: boolean
  myntra: boolean
  meesho: boolean
  flipkart: boolean
  snapdeal: boolean
  inventory: boolean
}

function hintTtlMs(): number {
  try {
    const raw = localStorage.getItem('erp_local_session_hint_ttl_hours')
    if (raw) {
      const h = parseFloat(raw)
      if (h > 0) return h * 60 * 60 * 1000
    }
  } catch {
    /* ignore */
  }
  return DEFAULT_TTL_MS
}

export function sessionLooksLoaded(c: CoverageResponse): boolean {
  return !!(
    c.sales ||
    c.mtr ||
    c.myntra ||
    c.meesho ||
    c.flipkart ||
    c.snapdeal ||
    c.inventory
  )
}

export function readLocalSessionHint(): LocalSessionHint | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as LocalSessionHint
    if (!parsed?.savedAt) return null
    return parsed
  } catch {
    return null
  }
}

export function persistLocalSessionHint(c: CoverageResponse): void {
  if (!operationalDataComplete(c)) return
  try {
    const hint: LocalSessionHint = {
      savedAt: Date.now(),
      sales: !!c.sales,
      sales_rows: c.sales_rows ?? 0,
      mtr: !!c.mtr,
      myntra: !!c.myntra,
      meesho: !!c.meesho,
      flipkart: !!c.flipkart,
      snapdeal: !!c.snapdeal,
      inventory: !!c.inventory,
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(hint))
  } catch {
    /* private mode / quota */
  }
}

export function clearLocalSessionHint(): void {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {
    /* ignore */
  }
}

const OPERATIONAL_DATASET_KEYS = [
  'sku_mapping',
  'mtr',
  'sales',
  'inventory',
  'myntra',
  'meesho',
  'flipkart',
  'snapdeal',
] as const

/** PO / dashboard essentials — Snapdeal optional (often unused). */
export const PO_OPERATIONAL_KEYS = [
  'sku_mapping',
  'mtr',
  'sales',
  'inventory',
  'myntra',
  'meesho',
  'flipkart',
] as const

export const OPERATIONAL_DATA_TOTAL = OPERATIONAL_DATASET_KEYS.length
export const PO_OPERATIONAL_TOTAL = PO_OPERATIONAL_KEYS.length

export function operationalDataLoaded(c: CoverageResponse): number {
  return OPERATIONAL_DATASET_KEYS.filter(k => Boolean(c[k])).length
}

export function poOperationalLoaded(c: CoverageResponse): number {
  return PO_OPERATIONAL_KEYS.filter(k => Boolean(c[k])).length
}

/** True when PO Engine can run (does not require Snapdeal). */
export function poOperationalReady(c: CoverageResponse): boolean {
  return poOperationalLoaded(c) === PO_OPERATIONAL_TOTAL
}

/** @deprecated Returns a new object on every call — use operationalDataLoaded() + OPERATIONAL_DATA_TOTAL instead to avoid useSyncExternalStore looping. */
export function operationalDataLoadedCount(c: CoverageResponse): { loaded: number; total: number } {
  return { loaded: operationalDataLoaded(c), total: OPERATIONAL_DATA_TOTAL }
}

export function operationalDataComplete(c: CoverageResponse): boolean {
  const { loaded, total } = operationalDataLoadedCount(c)
  return loaded === total
}

/** Minimum unified sales rows before PO page mounts (prod catalog ~1.56M). */
export const PO_MIN_SALES_ROWS = 1_000_000

/** Minimum inventory SKUs before PO page mounts (prod ~6.7k). */
export const PO_MIN_INVENTORY_ROWS = 5_000

/** True when PO routes may mount — uses po_ready semantics (see GET /api/po/readiness). */
export function poPageHydrationReady(c: CoverageResponse): boolean {
  if (typeof c.po_ready === 'boolean') {
    return c.po_ready
  }
  if (c.critical_restore_running) {
    return false
  }
  if (!operationalDataComplete(c)) return false
  if (!c.inventory || !c.sales) return false
  if ((c.sales_rows ?? 0) < PO_MIN_SALES_ROWS) return false
  if ((c.inventory_rows ?? 0) < PO_MIN_INVENTORY_ROWS) return false
  return true
}

/** True when we can skip GitHub hydrate + full cache download on login. */
export function canSkipHeavyServerRestore(
  coverage: CoverageResponse,
  hint: LocalSessionHint | null,
): boolean {
  if (!hint) return false
  if (Date.now() - hint.savedAt > hintTtlMs()) return false
  if (coverage.pause_auto_data_restore) return false
  if (!operationalDataComplete(coverage)) return false
  if (coverage.inventory_upload_status === 'running') return false
  if (coverage.daily_auto_ingest_status === 'running') return false
  if (coverage.tier1_bulk_status === 'running') return false
  if (coverage.sales_rebuild === 'running') return false
  // Server lost most of the data (new session / deploy) — do not skip restore.
  if (hint.sales_rows > 500 && (coverage.sales_rows ?? 0) < hint.sales_rows * 0.4) {
    return false
  }
  return true
}

export function formatLocalHintAge(hint: LocalSessionHint): string {
  const min = Math.round((Date.now() - hint.savedAt) / 60_000)
  if (min < 2) return 'just now'
  if (min < 60) return `${min}m ago`
  const h = Math.round(min / 60)
  return `${h}h ago`
}
