/**
 * Versioned browser cache for Intelligence dashboard aggregates.
 * Coordinates with server ``/data/intelligence/version`` — instant render when versions match.
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
  status?: 'warming' | 'ready'
  data_completeness?: 'partial' | 'full'
  message?: string
  busy?: boolean
  empty_window?: boolean
  session_data_range?: { min: string; max: string }
  sales_summary: SalesSummary
  platform_summary: PlatformSummaryItem[]
  top_skus: TopSku[]
  anomalies: AnomalyItem[]
  dsr_brand_monthly: DsrBrandMonthlyResponse
  version?: string
  stale?: boolean
}

export type CachedIntelligenceBundle = IntelligenceBundle & {
  cached_at: number
  start_date: string
  end_date: string
  basis: 'gross' | 'net'
  version: string
}

const STORAGE_PREFIX = 'erp_intelligence_bundle_v5'
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

function versionMatches(cached: CachedIntelligenceBundle, serverVersion: string | undefined): boolean {
  if (!serverVersion) return true
  if (!cached.version) return false
  return cached.version === serverVersion
}

export function readIntelligenceCache(
  start: string,
  end: string,
  basis: 'gross' | 'net',
  serverVersion?: string,
): CachedIntelligenceBundle | null {
  try {
    const raw = localStorage.getItem(storageKey(start, end, basis))
    if (!raw) return null
    const parsed = JSON.parse(raw) as CachedIntelligenceBundle
    if (!parsed?.cached_at || !parsed.platform_summary) return null
    if (Date.now() - parsed.cached_at > ttlMs()) return null
    if (serverVersion && !versionMatches(parsed, serverVersion)) return null
    return parsed
  } catch {
    return null
  }
}

/** Stale-while-revalidate: return cached bundle for instant first paint. */
export function readIntelligenceCacheStale(
  start: string,
  end: string,
  basis: 'gross' | 'net',
  serverVersion?: string,
): { bundle: CachedIntelligenceBundle; expired: boolean; versionMismatch: boolean } | null {
  try {
    const raw = localStorage.getItem(storageKey(start, end, basis))
    if (!raw) return null
    const parsed = JSON.parse(raw) as CachedIntelligenceBundle
    if (!parsed?.cached_at || !parsed.platform_summary) return null
    const expired = Date.now() - parsed.cached_at > ttlMs()
    const versionMismatch = Boolean(serverVersion && parsed.version && parsed.version !== serverVersion)
    if (versionMismatch && serverVersion) {
      return { bundle: parsed, expired: true, versionMismatch: true }
    }
    return { bundle: parsed, expired, versionMismatch: false }
  } catch {
    return null
  }
}

export function writeIntelligenceCache(
  start: string,
  end: string,
  basis: 'gross' | 'net',
  bundle: IntelligenceBundle,
  version?: string,
): void {
  try {
    const payload: CachedIntelligenceBundle = {
      ...bundle,
      cached_at: Date.now(),
      start_date: start,
      end_date: end,
      basis,
      version: version || bundle.version || '',
    }
    localStorage.setItem(storageKey(start, end, basis), JSON.stringify(payload))
  } catch {
    /* quota / private mode */
  }
}

export function clearIntelligenceCacheForRange(
  start: string,
  end: string,
  basis: 'gross' | 'net',
): void {
  try {
    localStorage.removeItem(storageKey(start, end, basis))
  } catch {
    /* ignore */
  }
}

export function clearIntelligenceCache(): void {
  try {
    const prefixes = [
      'erp_intelligence_bundle_v1',
      'erp_intelligence_bundle_v2',
      'erp_intelligence_bundle_v3',
      'erp_intelligence_bundle_v4',
      STORAGE_PREFIX,
    ]
    const keys: string[] = []
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)
      if (k && prefixes.some(p => k.startsWith(p))) keys.push(k)
    }
    keys.forEach(k => localStorage.removeItem(k))
  } catch {
    /* ignore */
  }
}

function hasLoadedPlatformUnits(bundle: IntelligenceBundle): boolean {
  return (bundle.platform_summary ?? []).some(p => p.loaded && (p.total_units ?? 0) > 0)
}

export function bundleHasDisplayData(bundle: IntelligenceBundle | null | undefined): boolean {
  if (!bundle) return false
  if (bundle.status === 'warming') return false
  if (bundle.status === 'ready') return true
  if (bundle.data_completeness === 'partial' && hasLoadedPlatformUnits(bundle)) return true
  const platforms = bundle.platform_summary ?? []
  if (platforms.length > 0) {
    if (hasLoadedPlatformUnits(bundle)) return true
    if ((bundle.sales_summary?.total_units ?? 0) > 0) return true
    return false
  }
  return (bundle.sales_summary?.total_units ?? 0) > 0
}

export function summaryToCachedBundle(
  summary: {
    platform_summary?: PlatformSummaryItem[]
    sales_summary?: Partial<SalesSummary>
    top_skus?: Array<{ sku: string; units?: number }>
    data_completeness?: string
    version?: string
    stale?: boolean
  },
): IntelligenceBundle | null {
  const platforms = summary.platform_summary ?? []
  const units = Number(summary.sales_summary?.total_units ?? 0)
  if (units <= 0 && platforms.length === 0) return null
  return {
    status: 'ready',
    data_completeness: summary.data_completeness === 'full' ? 'full' : 'partial',
    sales_summary: {
      total_units: units,
      total_returns: Number(summary.sales_summary?.total_returns ?? 0),
      net_units: Number(summary.sales_summary?.net_units ?? units),
      return_rate: Number(summary.sales_summary?.return_rate ?? 0),
    },
    platform_summary: platforms,
    top_skus: (summary.top_skus ?? []).map(t => ({
      sku: t.sku,
      units: Number(t.units ?? 0),
    })),
    anomalies: [],
    dsr_brand_monthly: { rows: [], totals: { YG: 0, Akiko: 0, Other: 0, Untagged: 0 }, note: '' },
    version: summary.version,
    stale: summary.stale,
  }
}

function purgeLegacyIntelligenceCache(): void {
  try {
    const prefixes = [
      'erp_intelligence_bundle_v1',
      'erp_intelligence_bundle_v2',
      'erp_intelligence_bundle_v3',
      'erp_intelligence_bundle_v4',
    ]
    const keys: string[] = []
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)
      if (k && prefixes.some(p => k.startsWith(p))) keys.push(k)
    }
    keys.forEach(k => localStorage.removeItem(k))
  } catch {
    /* ignore */
  }
}

purgeLegacyIntelligenceCache()
