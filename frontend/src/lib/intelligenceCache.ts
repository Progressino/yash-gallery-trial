/**
 * Browser cache for Intelligence dashboard aggregates (summary + platforms + top SKUs).
 * Full sales history stays on the server; this stores only the small JSON the UI needs.
 */

export interface DsrBrandMonthlyRow {
  month: string
  month_display: string
  YG: number
  Akiko: number
  Other: number
  Untagged: number
  leader: string
  delta: number
}

export interface DsrBrandMonthlyResponse {
  rows: DsrBrandMonthlyRow[]
  totals: { YG: number; Akiko: number; Other: number; Untagged: number }
  note: string
}

export interface PlatformSummaryItem {
  platform: string
  loaded: boolean
  total_units: number
  total_returns: number
  net_units?: number
  return_rate: number
  top_sku: string
  trend_direction: 'up' | 'down' | 'flat'
  trend_direction_net?: 'up' | 'down' | 'flat'
  monthly: { month: string; shipments: number; refunds: number; net?: number }[]
  daily?: { date: string; shipments: number; refunds: number; net?: number }[]
  by_state: { state: string; units: number; net_units?: number }[]
}

export interface AnomalyItem {
  type: string
  severity: 'critical' | 'warning' | 'info'
  platform: string
  message: string
  sku?: string
}

export interface SalesSummary {
  total_units: number
  total_returns: number
  net_units: number
  return_rate: number
  return_sheet_units?: number
  marketplace_return_units?: number
  active_months?: number
  date_basis_note?: string
}

export interface TopSku {
  sku: string
  units: number
}

export type IntelligenceBundle = {
  sales_summary: SalesSummary
  platform_summary: PlatformSummaryItem[]
  top_skus: TopSku[]
  anomalies: AnomalyItem[]
  dsr_brand_monthly: DsrBrandMonthlyResponse
}

export type CachedIntelligenceBundle = IntelligenceBundle & {
  cached_at: number
  start_date: string
  end_date: string
  basis: 'gross' | 'net'
}

const STORAGE_PREFIX = 'erp_intelligence_bundle_v1'
const DEFAULT_TTL_MS = 24 * 60 * 60 * 1000

function ttlMs(): number {
  try {
    const raw = localStorage.getItem('erp_intelligence_cache_ttl_hours')
    if (raw) {
      const h = parseFloat(raw)
      if (h > 0) return h * 60 * 60 * 1000
    }
  } catch {
    /* ignore */
  }
  return DEFAULT_TTL_MS
}

function storageKey(start: string, end: string, basis: 'gross' | 'net'): string {
  return `${STORAGE_PREFIX}:${start || '_'}:${end || '_'}:${basis}`
}

export function readIntelligenceCache(
  start: string,
  end: string,
  basis: 'gross' | 'net',
): CachedIntelligenceBundle | null {
  try {
    const raw = localStorage.getItem(storageKey(start, end, basis))
    if (!raw) return null
    const parsed = JSON.parse(raw) as CachedIntelligenceBundle
    if (!parsed?.cached_at || !parsed.platform_summary) return null
    if (Date.now() - parsed.cached_at > ttlMs()) return null
    return parsed
  } catch {
    return null
  }
}

export function writeIntelligenceCache(
  start: string,
  end: string,
  basis: 'gross' | 'net',
  bundle: IntelligenceBundle,
): void {
  try {
    const payload: CachedIntelligenceBundle = {
      ...bundle,
      cached_at: Date.now(),
      start_date: start,
      end_date: end,
      basis,
    }
    localStorage.setItem(storageKey(start, end, basis), JSON.stringify(payload))
  } catch {
    /* quota / private mode */
  }
}

export function clearIntelligenceCache(): void {
  try {
    const keys: string[] = []
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)
      if (k?.startsWith(STORAGE_PREFIX)) keys.push(k)
    }
    keys.forEach(k => localStorage.removeItem(k))
  } catch {
    /* ignore */
  }
}

export function bundleHasDisplayData(bundle: IntelligenceBundle | null | undefined): boolean {
  if (!bundle) return false
  if ((bundle.sales_summary?.total_units ?? 0) > 0) return true
  return (bundle.platform_summary ?? []).some(p => p.loaded && (p.total_units ?? 0) > 0)
}
