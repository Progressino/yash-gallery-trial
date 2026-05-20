/**
 * Axios instance — automatically sends the session_id cookie.
 * In dev, Vite proxies /api → http://localhost:8000.
 * In production, nginx proxies /api → FastAPI.
 */
import axios from 'axios'
import { isUploadBusy } from '../store/uploadActivity'

export const api = axios.create({
  baseURL: '/api',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
})

// ── Types ─────────────────────────────────────────────────────

export interface UploadResponse {
  ok: boolean
  message: string
  rows?: number
  parsed_rows?: number
  kept_rows?: number
  dropped_rows?: number
  dropped_reasons?: string[]
  validation_warnings?: string[]
  years?: number[]
  sku_count?: number
  detected_platforms?: string[]
  /** Present after SKU master upload when some sales SKUs are not keys or OMS values in the map */
  unmapped_skus?: string[]
}

export interface CoverageResponse {
  sku_mapping: boolean
  mtr: boolean
  sales: boolean
  myntra: boolean
  meesho: boolean
  flipkart: boolean
  snapdeal: boolean
  inventory: boolean
  daily_orders: boolean
  existing_po: boolean
  sku_status_lead?: boolean
  daily_inventory_history?: boolean
  po_raise_ledger?: boolean
  return_sheet?: boolean
  mtr_rows: number
  sales_rows: number
  myntra_rows: number
  meesho_rows: number
  flipkart_rows: number
  snapdeal_rows: number
  sku_status_lead_rows?: number
  daily_inventory_history_rows?: number
  daily_inventory_history_skus?: number
  po_raise_ledger_rows?: number
  return_sheet_skus?: number
  /** True after "Clear all app data" until an upload or explicit Load Cache / Fresh reload. */
  pause_auto_data_restore?: boolean
  /** Tier-3 daily-auto background sales rebuild */
  sales_rebuild?: 'idle' | 'running' | 'done' | 'error'
  sales_rebuild_message?: string
  daily_auto_ingest_status?: 'idle' | 'running' | 'done' | 'error'
  daily_auto_ingest_message?: string
}

// ── Upload helpers ────────────────────────────────────────────

/** Tier-1 multi-year ZIPs can take several minutes to parse; align with nginx proxy_read_timeout (e.g. 900s). */
const UPLOAD_TIMEOUT_MS = 900_000
// Coverage/cache calls may queue behind uploads or session restore.
const CACHE_TIMEOUT_MS = 120_000
/** Status polls while server parses inventory or runs PO math (per request). */
const POLL_TIMEOUT_MS = 180_000
/** Fetching the full PO table after calculate completes (can be 10k+ rows). */
const PO_RESULT_TIMEOUT_MS = 900_000
/** POST /po/calculate should return immediately; allow headroom if the worker is busy. */
export const PO_REQUEST_TIMEOUT_MS = 30_000

function _errMessage(e: unknown, fallback: string): string {
  if (axios.isAxiosError(e)) {
    const data = e.response?.data as { message?: string; detail?: string } | undefined
    if (typeof data?.message === 'string' && data.message.trim()) return data.message
    if (typeof data?.detail === 'string' && data.detail.trim()) return data.detail
    if (e.code === 'ECONNABORTED') return 'Request timed out. File may be too large or server is busy.'
    if (e.response?.status === 502) {
      return 'Server gateway error (502). A large job may still be running — wait 1–2 minutes and try again, or hard-refresh the page.'
    }
    if (!e.response) return 'Network error. Check connection/VPN and try again.'
  }
  if (e instanceof Error && e.message.trim()) return e.message
  return fallback
}


async function uploadFile(endpoint: string, file: File, extraFields?: Record<string, string>): Promise<UploadResponse> {
  const fd = new FormData()
  fd.append('file', file)
  if (extraFields) Object.entries(extraFields).forEach(([k, v]) => fd.append(k, v))
  try {
    const { data } = await api.post<UploadResponse>(endpoint, fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: UPLOAD_TIMEOUT_MS,
    })
    return data
  } catch (e: unknown) {
    throw new Error(_errMessage(e, 'Upload failed'))
  }
}

export const uploadSkuMapping = (file: File) => uploadFile('/upload/sku-mapping', file)
export const uploadMtr        = (file: File) => uploadFile('/upload/mtr', file)
export const uploadMyntra     = (file: File) => uploadFile('/upload/myntra', file)
export const uploadMeesho     = (file: File) => uploadFile('/upload/meesho', file)
export const uploadFlipkart   = (file: File) => uploadFile('/upload/flipkart', file)
export const uploadAmazonB2C  = (file: File) => uploadFile('/upload/amazon-b2c', file)
export const uploadAmazonB2B  = (file: File) => uploadFile('/upload/amazon-b2b', file)
export const uploadExistingPO = (file: File) => uploadFile('/upload/existing-po', file)
export const uploadSnapdeal   = (file: File) => uploadFile('/upload/snapdeal', file)

export interface ReturnSheetImportResponse {
  ok: boolean
  message?: string
  skus?: number
  total_units?: number
  sales_rebuilt?: boolean
}

/** Return sheet (CSV / Excel) — merged into unified sales (net) and PO return overlay. Upload tab only in UI. */
export async function uploadReturnSheet(
  file: File,
  opts?: { groupByParent?: boolean; replace?: boolean },
): Promise<ReturnSheetImportResponse> {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('group_by_parent', opts?.groupByParent ? 'true' : 'false')
  fd.append('replace', opts?.replace === false ? 'false' : 'true')
  try {
    const { data } = await api.post<ReturnSheetImportResponse>('/po/returns/import-file', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: UPLOAD_TIMEOUT_MS,
    })
    return data
  } catch (e: unknown) {
    throw new Error(_errMessage(e, 'Return sheet upload failed'))
  }
}

export async function clearReturnSheet(): Promise<{ ok: boolean; message?: string }> {
  const { data } = await api.delete<{ ok: boolean; message?: string }>('/po/returns/overlay')
  return data
}

export async function uploadInventoryAuto(
  files: File[]
): Promise<{ ok: boolean; message: string; rows?: number; debug?: Record<string, unknown>; detected?: string[] }> {
  const fd = new FormData()
  files.forEach(f => fd.append('files', f))
  const { data } = await api.post('/upload/inventory-auto', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: UPLOAD_TIMEOUT_MS,
  })
  return data
}

export async function uploadInventory(files: {
  oms?: File[]; fk?: File; myntra?: File; amz?: File
}): Promise<{ ok: boolean; message: string; rows?: number; debug?: Record<string, unknown> }> {
  const fd = new FormData()
  if (files.oms)    files.oms.forEach(f => fd.append('oms_file', f))
  if (files.fk)     fd.append('fk_file',     files.fk)
  if (files.myntra) fd.append('myntra_file', files.myntra)
  if (files.amz)    fd.append('amz_file',    files.amz)
  const { data } = await api.post('/upload/inventory', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: UPLOAD_TIMEOUT_MS,
  })
  return data
}

export async function uploadDailyAuto(
  files: File[]
): Promise<{
  ok: boolean
  message: string
  ingest_async?: boolean
  detected_platforms?: string[]
  warnings?: string[]
  processed_files?: number
  detected_files?: number
  unknown_files?: number
  sales_rebuild?: 'inline' | 'pending'
}> {
  const fd = new FormData()
  files.forEach(f => fd.append('files', f))
  try {
    const { data } = await api.post('/upload/daily-auto', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: UPLOAD_TIMEOUT_MS,
    })
    return data
  } catch (e: unknown) {
    throw new Error(_errMessage(e, 'Daily upload failed'))
  }
}

/** Poll until Tier-3 background sales rebuild finishes (daily-auto). */
export async function waitForSalesRebuild(
  onTick?: (message: string) => void,
  maxMs = UPLOAD_TIMEOUT_MS,
): Promise<CoverageResponse> {
  const start = Date.now()
  while (Date.now() - start < maxMs) {
    const cov = await getCoverage()
    const st = cov.sales_rebuild ?? 'idle'
    if (st === 'running') {
      onTick?.(cov.sales_rebuild_message || 'Rebuilding combined sales…')
      await new Promise(r => setTimeout(r, 2500))
      continue
    }
    if (st === 'error') {
      throw new Error(cov.sales_rebuild_message || 'Sales rebuild failed')
    }
    return cov
  }
  throw new Error('Sales rebuild timed out — refresh the page in a minute.')
}

/** Poll until Tier-3 background ingest (RAR / large multi-file) finishes. */
export async function waitForDailyAutoIngest(
  onTick?: (message: string) => void,
  maxMs = UPLOAD_TIMEOUT_MS,
): Promise<CoverageResponse> {
  const start = Date.now()
  while (Date.now() - start < maxMs) {
    const cov = await getCoverage()
    const st = cov.daily_auto_ingest_status ?? 'idle'
    if (st === 'running') {
      onTick?.(cov.daily_auto_ingest_message || 'Processing archives on server…')
      await new Promise(r => setTimeout(r, 2500))
      continue
    }
    if (st === 'error') {
      throw new Error(cov.daily_auto_ingest_message || 'Tier-3 ingest failed')
    }
    return cov
  }
  throw new Error('Tier-3 ingest timed out — refresh the page in a minute.')
}

export interface POCalculateResult {
  ok: boolean
  status?: string
  message?: string
  rows?: Record<string, unknown>[]
  columns?: string[]
  sales_through?: string | null
  planning_date?: string | null
  raise_ledger_rows?: number
  ledger_auto_import?: string | null
}

export type DailyInventoryUploadResult = {
  ok: boolean
  status?: string
  message?: string
  rows?: number
  skus?: number
  days?: number
  min_date?: string
  max_date?: string
}

/** Poll after POST /po/daily-inventory-history (parses in background on the server). */
export async function waitForDailyInventoryUpload(
  onTick?: (message: string) => void,
  maxMs = 900_000,
): Promise<DailyInventoryUploadResult> {
  const start = Date.now()
  while (Date.now() - start < maxMs) {
    const { data } = await api.get<DailyInventoryUploadResult>(
      '/po/daily-inventory-history/upload-status',
      { timeout: POLL_TIMEOUT_MS },
    )
    const st = data.status ?? 'idle'
    if (st === 'running') {
      onTick?.(data.message || 'Parsing daily inventory sheet…')
      await new Promise(r => setTimeout(r, 2000))
      continue
    }
    if (st === 'error') {
      throw new Error(data.message || 'Daily inventory upload failed')
    }
    if (st === 'done') {
      return data
    }
    await new Promise(r => setTimeout(r, 1500))
  }
  throw new Error('Daily inventory upload timed out — try again in a minute.')
}

function _isAxiosTimeout(e: unknown): boolean {
  return axios.isAxiosError(e) && e.code === 'ECONNABORTED'
}

function _isGateway502(e: unknown): boolean {
  return axios.isAxiosError(e) && e.response?.status === 502
}

async function _sleep(ms: number): Promise<void> {
  await new Promise(r => setTimeout(r, ms))
}

/** Poll after POST /po/calculate (runs in background on the server). */
export async function waitForPoCalculate(
  onTick?: (message: string) => void,
  maxMs = 900_000,
): Promise<POCalculateResult> {
  const start = Date.now()
  let gatewayRetries = 0
  while (Date.now() - start < maxMs) {
    let data: POCalculateResult & { row_count?: number }
    try {
      ;({ data } = await api.get<POCalculateResult & { row_count?: number }>(
        '/po/calculate/status',
        { timeout: POLL_TIMEOUT_MS },
      ))
    } catch (e: unknown) {
      if (_isGateway502(e) && gatewayRetries < 40) {
        gatewayRetries += 1
        onTick?.('Server busy (502) — still calculating…')
        await _sleep(3000)
        continue
      }
      throw e
    }
    const st = data.status ?? 'idle'
    if (st === 'running') {
      onTick?.(data.message || 'Calculating PO recommendations…')
      await new Promise(r => setTimeout(r, 2000))
      continue
    }
    if (st === 'error') {
      throw new Error(data.message || 'PO calculation failed')
    }
    if (st === 'done') {
      const pageSize = 1200
      let offset = 0
      let columns: string[] | undefined
      const allRows: Record<string, unknown>[] = []
      let meta: POCalculateResult = { ok: true }
      while (true) {
        let page: POCalculateResult & {
          offset?: number
          total?: number
          has_more?: boolean
        }
        try {
          ;({ data: page } = await api.get<
            POCalculateResult & {
              offset?: number
              total?: number
              has_more?: boolean
            }
          >('/po/calculate/result', {
            params: { offset, limit: pageSize },
            timeout: PO_RESULT_TIMEOUT_MS,
          }))
        } catch (e: unknown) {
          if (_isGateway502(e) && gatewayRetries < 40) {
            gatewayRetries += 1
            onTick?.('Server busy (502) — loading results…')
            await _sleep(3000)
            continue
          }
          throw e
        }
        if (!page.ok) {
          throw new Error(page.message || 'Failed to load PO results')
        }
        if (!columns?.length && page.columns?.length) columns = page.columns
        if (page.rows?.length) allRows.push(...page.rows)
        meta = {
          ok: true,
          columns: columns ?? page.columns,
          sales_through: page.sales_through ?? meta.sales_through,
          planning_date: page.planning_date ?? meta.planning_date,
          raise_ledger_rows: page.raise_ledger_rows ?? meta.raise_ledger_rows,
          ledger_auto_import: page.ledger_auto_import ?? meta.ledger_auto_import,
        }
        const total = page.total ?? allRows.length
        onTick?.(`Loading PO results… ${Math.min(allRows.length, total).toLocaleString()} / ${total.toLocaleString()}`)
        if (!page.has_more) break
        offset += pageSize
      }
      return { ...meta, rows: allRows }
    }
    await new Promise(r => setTimeout(r, 1500))
  }
  throw new Error('PO calculation timed out — try again in a minute.')
}

/** Start PO calculate; if the POST times out or 502s, poll anyway (server may still be running). */
export async function startPoCalculate(
  body: Record<string, unknown>,
  onTick?: (message: string) => void,
): Promise<POCalculateResult> {
  try {
    const { data } = await api.post<POCalculateResult & { status?: string }>(
      '/po/calculate',
      body,
      { timeout: PO_REQUEST_TIMEOUT_MS },
    )
    if (!data.ok) {
      return data
    }
    if (data.status === 'running' || !data.rows) {
      return waitForPoCalculate(onTick)
    }
    return data
  } catch (e: unknown) {
    if (_isAxiosTimeout(e)) {
      onTick?.('Still calculating on server…')
      return waitForPoCalculate(onTick)
    }
    // 502 on the initial POST: the server may have received the request and started
    // the calculation before the gateway timed out. Poll the status endpoint to find out.
    if (axios.isAxiosError(e) && e.response?.status === 502) {
      onTick?.('Server busy — checking calculation status…')
      await new Promise(r => setTimeout(r, 3000))
      return waitForPoCalculate(onTick)
    }
    throw e
  }
}

export async function buildSales(): Promise<{
  ok: boolean
  message: string
  rows?: number
  unmapped_skus?: string[]
}> {
  const { data } = await api.post('/upload/build-sales', undefined, {
    timeout: UPLOAD_TIMEOUT_MS,
  })
  return data
}

export async function clearPlatform(platform: string): Promise<{ ok: boolean; message: string }> {
  const { data } = await api.delete(`/upload/clear/${platform}`)
  return data
}

// ── Coverage ──────────────────────────────────────────────────

export async function getCoverage(opts?: {
  timeout?: number
  /** Skip slow SQLite restore on the server (use after PO Engine uploads). */
  light?: boolean
}): Promise<CoverageResponse> {
  try {
    const params = opts?.light ? { light: '1' } : undefined
    const { data } = await api.get<CoverageResponse>('/data/coverage', {
      params,
      timeout: opts?.timeout ?? CACHE_TIMEOUT_MS,
    })
    return data
  } catch (e: unknown) {
    throw new Error(_errMessage(e, 'Coverage refresh failed'))
  }
}

/** YG vs Akiko monthly comparison CSV (uses same date params as dashboard summary). */
export async function downloadDsrBrandMonthlyCsv(params: string): Promise<void> {
  const q = params.trim() ? `?${params}` : ''
  const res = await fetch(`/api/data/dsr-brand-monthly-export${q}`, { credentials: 'include' })
  if (!res.ok) {
    let msg = `Export failed (${res.status})`
    try {
      const j = (await res.json()) as { detail?: string }
      if (typeof j.detail === 'string') msg = j.detail
    } catch {
      /* ignore */
    }
    throw new Error(msg)
  }
  const blob = await res.blob()
  const cd = res.headers.get('Content-Disposition')
  const m = cd?.match(/filename="([^"]+)"/i)
  const filename = m?.[1] ?? 'dsr-yg-akiko-monthly.csv'
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

/** Daily DSR table as CSV (same rows as the dashboard DSR block). */
export async function downloadDailyDsrCsv(isoDate: string): Promise<void> {
  const p = new URLSearchParams()
  if (isoDate) p.set('date', isoDate)
  const res = await fetch(`/api/data/daily-dsr-export?${p}`, { credentials: 'include' })
  if (!res.ok) {
    let msg = `Export failed (${res.status})`
    try {
      const j = (await res.json()) as { detail?: string }
      if (typeof j.detail === 'string') msg = j.detail
    } catch {
      /* ignore */
    }
    throw new Error(msg)
  }
  const blob = await res.blob()
  const cd = res.headers.get('Content-Disposition')
  const m = cd?.match(/filename="([^"]+)"/i)
  const filename = m?.[1] ?? 'daily-dsr.csv'
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

/** Line-level unified sales CSV for Intelligence dashboard (date range + optional platforms). */
export async function downloadIntelligenceSalesCsv(opts: {
  startDate?: string
  endDate?: string
  platforms?: string[]
}): Promise<void> {
  const p = new URLSearchParams()
  p.set('months', '0')
  if (opts.startDate) p.set('start_date', opts.startDate)
  if (opts.endDate) p.set('end_date', opts.endDate)
  if (opts.platforms?.length) p.set('platforms', opts.platforms.join(','))
  const res = await fetch(`/api/data/sales-export?${p}`, { credentials: 'include' })
  if (!res.ok) {
    let msg = `Export failed (${res.status})`
    try {
      const j = (await res.json()) as { detail?: string }
      if (typeof j.detail === 'string') msg = j.detail
    } catch {
      /* ignore */
    }
    throw new Error(msg)
  }
  const blob = await res.blob()
  const cd = res.headers.get('Content-Disposition')
  const m = cd?.match(/filename="([^"]+)"/i)
  const filename = m?.[1] ?? 'intelligence-sales.csv'
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

// ── Daily sales management ────────────────────────────────────

export interface DailyUpload {
  id: number
  platform: string
  file_date: string
  /** Actual min/max row dates inside the file (when stored). */
  date_from?: string | null
  date_to?: string | null
  filename: string
  uploaded_at: string
  rows: number
}

export interface DailySummary {
  [platform: string]: {
    min_date: string
    max_date: string
    total_rows: number
    file_count: number
  }
}

export const getDailySummary  = (): Promise<DailySummary>    => api.get('/data/daily-summary').then(r => r.data)
export const getDailyUploads  = (): Promise<DailyUpload[]>   => api.get('/data/daily-uploads').then(r => r.data)
export const deleteDailyUpload = (id: number)                => api.delete(`/data/daily-uploads/${id}`).then(r => r.data)

// ── Cache ─────────────────────────────────────────────────────

export async function cacheStatus() {
  try {
    const { data } = await api.get('/cache/status', { timeout: CACHE_TIMEOUT_MS })
    return data
  } catch (e: unknown) {
    throw new Error(_errMessage(e, 'Cache status request failed'))
  }
}
export async function cacheSave() {
  try {
    const { data } = await api.post('/cache/save', undefined, { timeout: CACHE_TIMEOUT_MS })
    return data
  } catch (e: unknown) {
    throw new Error(_errMessage(e, 'Cache save failed'))
  }
}
export async function cacheLoad() {
  try {
    const { data } = await api.post('/cache/load', undefined, { timeout: CACHE_TIMEOUT_MS })
    return data
  } catch (e: unknown) {
    throw new Error(_errMessage(e, 'Cache load failed'))
  }
}

/** Clear warm + session, re-download GitHub cache, merge Tier-3 SQLite, rebuild sales, re-save GitHub (background). */
export async function cacheReloadFresh(): Promise<{ ok: boolean; message: string; sales_rows?: number }> {
  // Large datasets: many parquet downloads + sales rebuild can exceed default proxy limits.
  const { data } = await api.post('/cache/reload-fresh', undefined, { timeout: 900_000 })
  return data
}

export async function cacheClear(includeWarm = false) {
  const { data } = await api.delete('/cache', { params: includeWarm ? { include_warm: true } : {} })
  return data
}

/** Wipe session + warm cache + optional Tier-3 SQLite + optional GitHub Release cache. */
export async function resetAllAppData(opts?: {
  clearTier3Sqlite?: boolean
  clearWarmCache?: boolean
  clearGithubCache?: boolean
}): Promise<{ ok: boolean; message: string; tier3_deleted?: number }> {
  const { data } = await api.post('/cache/reset-all', {
    clear_tier3_sqlite: opts?.clearTier3Sqlite ?? false,
    clear_warm_cache: opts?.clearWarmCache !== false,
    clear_github_cache: opts?.clearGithubCache ?? false,
  })
  return data
}

export async function getDataQuality(): Promise<{
  loaded: boolean
  checks: Record<string, unknown>
  hints: string[]
}> {
  const { data } = await api.get('/data/data-quality')
  return data
}

// ── 401 interceptor — only hard-logout when token is truly invalid ─────────────
api.interceptors.response.use(
  res => res,
  err => {
    const url: string = err.config?.url ?? ''
    const isTimeout =
      err.code === 'ECONNABORTED' ||
      err.response?.status === 502 ||
      err.response?.status === 503 ||
      err.response?.status === 504
    let hasStoredAuth = false
    try {
      hasStoredAuth = !!sessionStorage.getItem('erp_auth_profile_v1')
    } catch {
      hasStoredAuth = false
    }
    if (
      err.response?.status === 401 &&
      !isTimeout &&
      !url.includes('/auth/') &&
      !window.location.pathname.startsWith('/login') &&
      !isUploadBusy() &&
      !hasStoredAuth
    ) {
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

/** Invalidate data queries after upload without refetching auth (avoids false logout). */
export function invalidateDataQueries(
  qc: { invalidateQueries: (opts?: { predicate?: (q: { queryKey: unknown }) => boolean }) => void },
) {
  qc.invalidateQueries({
    predicate: q => {
      const key = Array.isArray(q.queryKey) ? q.queryKey[0] : q.queryKey
      return key !== 'auth-me'
    },
  })
}

export default api
