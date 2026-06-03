/**
 * Axios instance — automatically sends the session_id cookie.
 * In dev, Vite proxies /api → http://localhost:8000.
 * In production, nginx proxies /api → FastAPI.
 */
import axios from 'axios'
import { clearIntelligenceCache } from '../lib/intelligenceCache'
import { isUploadBusy } from '../store/uploadActivity'
import {
  shouldUseChunkedUpload,
  uploadFilesChunked,
  type ChunkUploadProgress,
} from '../lib/chunkedUpload'

export type { ChunkUploadProgress }

export const api = axios.create({
  baseURL: '/api',
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
  // Prevent accidental tiny inherited timeouts (e.g. 3000ms) on upload/status calls.
  timeout: 120_000,
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
  return_sheet_units?: number
  /** True after "Clear all app data" until an upload or explicit Load Cache / Fresh reload. */
  pause_auto_data_restore?: boolean
  /** Tier-3 daily-auto background sales rebuild */
  sales_rebuild?: 'idle' | 'running' | 'done' | 'error'
  sales_rebuild_message?: string
  /** Background full restore (Upload → Restore all from server) */
  session_restore_status?: 'idle' | 'running' | 'done' | 'error'
  session_restore_message?: string
  session_restore_step?: string
  session_restore_progress?: number
  daily_auto_ingest_status?: 'idle' | 'running' | 'done' | 'error'
  daily_auto_ingest_message?: string
  /** Set after background daily-auto ingest finishes (see session.daily_auto_ingest_result). */
  daily_auto_ingest_detected_platforms?: string[]
  daily_auto_ingest_warnings?: string[]
  daily_auto_ingest_processed_files?: number
  daily_auto_ingest_detected_files?: number
  daily_auto_ingest_unknown_files?: number
  daily_auto_ingest_expanded_files?: number
  daily_auto_ingest_saved_files?: number
  daily_auto_ingest_file_results?: Array<{
    filename: string
    status: 'saved' | 'skipped'
    reason?: string
    platform?: string
    rows?: number
  }>
  inventory_upload_status?: 'idle' | 'running' | 'done' | 'error'
  inventory_upload_message?: string
  inventory_upload_progress?: number
  inventory_upload_rows?: number
  inventory_upload_warnings?: string[]
  inventory_upload_file_results?: Array<{
    filename: string
    category: string
    status: 'loaded' | 'skipped'
    reason?: string
  }>
  inventory_upload_sources?: string[]
  inventory_upload_amz_disclaimer?: Record<string, unknown>
  inventory_snapshot_date?: string | null
  inventory_snapshot_date_label?: string | null
  inventory_snapshot_date_sources?: string[] | null
  inventory_snapshot_uploaded_at?: string | null
  daily_inventory_upload_status?: 'idle' | 'running' | 'done' | 'error'
  daily_inventory_upload_message?: string
}

export type DailyAutoIngestSummary = {
  detected_platforms: string[]
  warnings: string[]
  processed_files: number
  detected_files: number
  unknown_files: number
  saved_files?: number
  expanded_files?: number
  file_results?: CoverageResponse['daily_auto_ingest_file_results']
  message: string
}

export function dailyAutoSummaryFromCoverage(c: CoverageResponse): DailyAutoIngestSummary | null {
  const hasResult =
    (c.daily_auto_ingest_detected_platforms?.length ?? 0) > 0 ||
    (c.daily_auto_ingest_warnings?.length ?? 0) > 0 ||
    c.daily_auto_ingest_processed_files != null
  if (!hasResult) return null
  return {
    detected_platforms: c.daily_auto_ingest_detected_platforms ?? [],
    warnings: c.daily_auto_ingest_warnings ?? [],
    processed_files: c.daily_auto_ingest_processed_files ?? 0,
    detected_files: c.daily_auto_ingest_detected_files ?? 0,
    unknown_files: c.daily_auto_ingest_unknown_files ?? 0,
    saved_files: c.daily_auto_ingest_saved_files ?? undefined,
    expanded_files: c.daily_auto_ingest_expanded_files ?? undefined,
    file_results: c.daily_auto_ingest_file_results ?? undefined,
    message: c.daily_auto_ingest_message ?? '',
  }
}

export function dailyAutoSummaryFromUpload(res: {
  detected_platforms?: string[]
  warnings?: string[]
  processed_files?: number
  detected_files?: number
  unknown_files?: number
  saved_files?: number
  expanded_files?: number
  file_results?: CoverageResponse['daily_auto_ingest_file_results']
  message?: string
}): DailyAutoIngestSummary {
  return {
    detected_platforms: res.detected_platforms ?? [],
    warnings: res.warnings ?? [],
    processed_files: res.processed_files ?? 0,
    detected_files: res.detected_files ?? 0,
    unknown_files: res.unknown_files ?? 0,
    saved_files: res.saved_files,
    expanded_files: res.expanded_files,
    file_results: res.file_results,
    message: res.message ?? '',
  }
}

/** User-facing completion line: what parsed vs skipped. */
export function formatDailyAutoCompleteToast(
  s: DailyAutoIngestSummary,
  salesRows?: number,
): string {
  const recognized = s.saved_files ?? (s.detected_files || s.detected_platforms.length)
  const total = s.expanded_files ?? (s.processed_files || recognized + s.unknown_files)
  const platforms = [...new Set(s.detected_platforms.map(d => d.split('(')[0].trim()))]
  const parts: string[] = []
  parts.push(`Upload complete: ${recognized} of ${total} file(s) saved.`)
  if (platforms.length) parts.push(`Parsed: ${platforms.join(', ')}.`)
  if (s.unknown_files > 0) {
    parts.push(`${s.unknown_files} file(s) skipped — see reasons below.`)
  }
  if (salesRows != null && salesRows > 0) {
    parts.push(`Combined sales: ${salesRows.toLocaleString()} rows.`)
  }
  if (s.warnings.length) {
    const preview = s.warnings.slice(0, 3).join('; ')
    parts.push(
      s.warnings.length > 3
        ? `Warnings: ${preview} (+${s.warnings.length - 3} more — see Import completeness below)`
        : `Warnings: ${preview}`,
    )
  } else if (s.message && !parts.some(p => p.includes(s.message.slice(0, 20)))) {
    parts.push(s.message)
  }
  return parts.join(' ')
}

// ── Upload helpers ────────────────────────────────────────────

/** Tier-1 multi-year ZIPs can take several minutes to parse; align with nginx proxy_read_timeout (e.g. 900s). */
const UPLOAD_TIMEOUT_MS = 900_000
/** Background parsing/rebuild can exceed request upload timeout on very large bundles. */
const UPLOAD_BACKGROUND_WAIT_MS = 1_800_000
// Coverage/cache calls may queue behind uploads or session restore.
const CACHE_TIMEOUT_MS = 120_000
/** Status polls while server parses inventory or runs PO math (per request). */
const POLL_TIMEOUT_MS = 180_000
/** Status polls during PO calc — short timeout so Cloudflare/nginx 502s retry quickly. */
const PO_STATUS_POLL_TIMEOUT_MS = 25_000
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

export async function uploadInventoryAuto(
  files: File[],
  onProgress?: (p: ChunkUploadProgress) => void,
): Promise<{
  ok: boolean
  message: string
  ingest_async?: boolean
  chunked?: boolean
  rows?: number
  debug?: Record<string, unknown>
  detected?: string[]
}> {
  try {
    if (shouldUseChunkedUpload(files)) {
      return await uploadFilesChunked('inventory-auto', files, onProgress)
    }
    const fd = new FormData()
    files.forEach(f => fd.append('files', f))
    const { data } = await api.post<{
      ok: boolean
      message: string
      require_chunked?: boolean
      ingest_async?: boolean
    }>('/upload/inventory-auto', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: UPLOAD_TIMEOUT_MS,
    })
    if (data?.require_chunked) {
      return await uploadFilesChunked('inventory-auto', files, onProgress)
    }
    return data
  } catch (e: unknown) {
    if (isUploadGateway502(e)) {
      return { ...dailyUploadPendingAfter502(files.length), ingest_async: true }
    }
    throw new Error(_errMessage(e, 'Inventory upload failed'))
  }
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

/** Import return units overlay for PO (same as PO Engine → Import returns). */
export async function uploadPoReturnsImport(
  file: File,
  opts?: { groupByParent?: boolean; replace?: boolean },
): Promise<{ ok: boolean; message: string; sales_rebuild?: string }> {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('group_by_parent', opts?.groupByParent ? 'true' : 'false')
  fd.append('replace', opts?.replace === false ? 'false' : 'true')
  try {
    const { data } = await api.post<{ ok?: boolean; message?: string; sales_rebuild?: string }>(
      '/po/returns/import-file',
      fd,
      { headers: { 'Content-Type': 'multipart/form-data' }, timeout: UPLOAD_TIMEOUT_MS },
    )
    return {
      ok: !!data?.ok,
      message: data?.message || (data?.ok ? 'Returns imported.' : 'Import failed.'),
      sales_rebuild: data?.sales_rebuild,
    }
  } catch (e: unknown) {
    throw new Error(_errMessage(e, 'Returns import failed'))
  }
}

/** True when Cloudflare/nginx returned 502 but the server may still be processing. */
export function isUploadGateway502(e: unknown): boolean {
  if (axios.isAxiosError(e) && e.response?.status === 502) return true
  if (e instanceof Error && (e as Error & { gateway502?: boolean }).gateway502) return true
  if (e instanceof Error && e.message.includes('GATEWAY_502_CHUNK_COMPLETE')) return true
  return (
    e instanceof Error &&
    /gateway error \(502\)|status code 502/i.test(e.message)
  )
}

async function getCoverageResilient(opts?: {
  timeout?: number
  light?: boolean
}): Promise<CoverageResponse> {
  let lastErr: unknown
  for (let attempt = 0; attempt < 8; attempt++) {
    try {
      return await getCoverage(opts)
    } catch (e) {
      lastErr = e
      if (isUploadGateway502(e) || (axios.isAxiosError(e) && !e.response)) {
        await new Promise(r => setTimeout(r, 1500 * (attempt + 1)))
        continue
      }
      throw e
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error('Coverage refresh failed')
}

function dailyUploadPendingAfter502(fileCount: number): {
  ok: boolean
  message: string
  ingest_async: boolean
  sales_rebuild: 'pending'
  processed_files: number
  detected_files: number
  unknown_files: number
  detected_platforms: string[]
  warnings: string[]
} {
  return {
    ok: true,
    message:
      'Upload may have been accepted (gateway timed out). Checking server status — stay on this page.',
    ingest_async: true,
    sales_rebuild: 'pending',
    processed_files: fileCount,
    detected_files: 0,
    unknown_files: 0,
    detected_platforms: [],
    warnings: [],
  }
}

/** Chunked daily upload with 502 → treat as async pending (server often still ingesting). */
export async function uploadDailyAutoChunked(
  files: File[],
  onProgress?: (p: ChunkUploadProgress) => void,
): Promise<{
  ok: boolean
  message: string
  ingest_async?: boolean
  chunked?: boolean
  detected_platforms?: string[]
  warnings?: string[]
  processed_files?: number
  detected_files?: number
  unknown_files?: number
  saved_files?: number
  expanded_files?: number
  file_results?: CoverageResponse['daily_auto_ingest_file_results']
  sales_rebuild?: 'inline' | 'pending'
}> {
  try {
    return await uploadFilesChunked('daily-auto', files, onProgress)
  } catch (e: unknown) {
    if (isUploadGateway502(e)) {
      return { ...dailyUploadPendingAfter502(files.length), chunked: true }
    }
    throw e
  }
}

export async function uploadDailyAuto(
  files: File[],
  onProgress?: (p: ChunkUploadProgress) => void,
): Promise<{
  ok: boolean
  message: string
  ingest_async?: boolean
  chunked?: boolean
  detected_platforms?: string[]
  warnings?: string[]
  processed_files?: number
  detected_files?: number
  unknown_files?: number
  saved_files?: number
  expanded_files?: number
  file_results?: CoverageResponse['daily_auto_ingest_file_results']
  sales_rebuild?: 'inline' | 'pending'
}> {
  try {
    if (shouldUseChunkedUpload(files)) {
      return await uploadDailyAutoChunked(files, onProgress)
    }
    const fd = new FormData()
    files.forEach(f => fd.append('files', f))
    const { data } = await api.post<{
      ok: boolean
      message: string
      require_chunked?: boolean
      ingest_async?: boolean
      detected_platforms?: string[]
      warnings?: string[]
      processed_files?: number
      detected_files?: number
      unknown_files?: number
      sales_rebuild?: 'inline' | 'pending'
    }>('/upload/daily-auto', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: UPLOAD_TIMEOUT_MS,
    })
    if (data?.require_chunked) {
      return await uploadDailyAutoChunked(files, onProgress)
    }
    return data
  } catch (e: unknown) {
    if (isUploadGateway502(e)) {
      return dailyUploadPendingAfter502(files.length)
    }
    throw new Error(_errMessage(e, 'Daily upload failed'))
  }
}

/** Clear a session stuck on “parsing…” after daily upload (server-side reset). */
export async function resetStuckDailyUpload(): Promise<{ ok: boolean; cleared: boolean; message: string }> {
  const { data } = await api.post('/upload/daily-auto/reset-stuck', {}, { timeout: 30_000 })
  return data
}

/** Poll until Tier-3 background ingest finishes (daily-auto). */
export async function waitForDailyAutoIngest(
  onTick?: (message: string) => void,
  maxMs = UPLOAD_BACKGROUND_WAIT_MS,
): Promise<CoverageResponse> {
  const start = Date.now()
  let sawRunning = false
  while (Date.now() - start < maxMs) {
    const cov = await getCoverageResilient({ light: true, timeout: POLL_TIMEOUT_MS })
    const st = cov.daily_auto_ingest_status ?? 'idle'
    const salesSt = cov.sales_rebuild ?? 'idle'
    if (st === 'running' || salesSt === 'running') {
      sawRunning = true
      let tickMsg =
        cov.daily_auto_ingest_message ||
        cov.sales_rebuild_message ||
        'Parsing daily files…'
      try {
        const job = await getJobStatus()
        if (job.upload_memory_lock_held && st === 'running') {
          tickMsg = `${cov.daily_auto_ingest_message || 'Parsing…'} (server cache loading — your files are queued)`
        } else if (job.daily_auto_ingest_message) {
          tickMsg = job.daily_auto_ingest_message
        } else if (job.sales_rebuild_message && salesSt === 'running') {
          tickMsg = job.sales_rebuild_message
        }
      } catch {
        /* use coverage message */
      }
      onTick?.(tickMsg)
      await new Promise(r => setTimeout(r, 2000))
      continue
    }
    if (st === 'error') {
      throw new Error(cov.daily_auto_ingest_message || 'Daily ingest failed')
    }
    if (st === 'done' || (sawRunning && st === 'idle')) {
      const salesBusy = String(cov.sales_rebuild ?? 'idle') === 'running'
      if (salesBusy) {
        onTick?.(cov.sales_rebuild_message || 'Rebuilding combined sales…')
        await new Promise(r => setTimeout(r, 2000))
        continue
      }
      return cov
    }
    onTick?.('Waiting for server to start processing…')
    await new Promise(r => setTimeout(r, 1500))
  }
  throw new Error('Daily ingest timed out — refresh the page in a minute.')
}

/** Poll until snapshot inventory-auto finishes parsing. */
export async function resetStuckInventoryUpload(): Promise<{
  ok: boolean
  cleared: boolean
  message: string
}> {
  const { data } = await api.post('/upload/inventory-auto/reset-stuck', {}, { timeout: 30_000 })
  return data
}

export async function waitForInventoryUpload(
  onTick?: (message: string, progressPct?: number) => void,
  maxMs = UPLOAD_TIMEOUT_MS,
): Promise<CoverageResponse> {
  const start = Date.now()
  while (Date.now() - start < maxMs) {
    const cov = await getCoverageResilient({ light: true, timeout: POLL_TIMEOUT_MS })
    const st = cov.inventory_upload_status ?? 'idle'
    if (st === 'running') {
      const pct = cov.inventory_upload_progress ?? 0
      onTick?.(cov.inventory_upload_message || 'Parsing inventory…', pct)
      await new Promise(r => setTimeout(r, 1500))
      continue
    }
    if (st === 'error') {
      throw new Error(cov.inventory_upload_message || 'Inventory upload failed')
    }
    if (st === 'done') {
      onTick?.(cov.inventory_upload_message || 'Inventory snapshot updated.')
      return cov
    }
    await new Promise(r => setTimeout(r, 1500))
  }
  throw new Error('Inventory upload timed out — refresh the page in a minute.')
}

/** Poll until Tier-3 background sales rebuild finishes (daily-auto). */
export async function waitForSalesRebuild(
  onTick?: (message: string) => void,
  maxMs = UPLOAD_BACKGROUND_WAIT_MS,
): Promise<CoverageResponse> {
  const start = Date.now()
  while (Date.now() - start < maxMs) {
    const cov = await getCoverageResilient({ light: true, timeout: POLL_TIMEOUT_MS })
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
  from_shared_cache?: boolean
  shared_cache_at?: string
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

/** SKU / status / lead-time sheet for PO rules (POST /po/sku-status-lead). */
export async function uploadPoSkuStatusLead(file: File): Promise<{ ok: boolean; message?: string; rows?: number }> {
  const fd = new FormData()
  fd.append('file', file)
  const { data } = await api.post<{ ok?: boolean; message?: string; rows?: number }>(
    '/po/sku-status-lead',
    fd,
    { headers: { 'Content-Type': 'multipart/form-data' }, timeout: UPLOAD_TIMEOUT_MS },
  )
  return { ok: !!data?.ok, message: data?.message, rows: data?.rows }
}

/** Wide daily inventory matrix for PO effective-days (POST /po/daily-inventory-history). */
export async function uploadPoDailyInventoryHistoryFile(
  file: File,
  onTick?: (message: string) => void,
): Promise<DailyInventoryUploadResult> {
  const fd = new FormData()
  fd.append('file', file)
  const { data } = await api.post<DailyInventoryUploadResult>(
    '/po/daily-inventory-history',
    fd,
    { headers: { 'Content-Type': 'multipart/form-data' }, timeout: UPLOAD_TIMEOUT_MS },
  )
  if (!data?.ok) {
    return { ok: false, message: data?.message || 'Upload failed.' }
  }
  if (data.status === 'running' || !data.rows) {
    return waitForDailyInventoryUpload(onTick)
  }
  return data
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

const PO_RESULT_PAGE_SIZE = 400

function _rowsFromPoResultPage(
  page: {
    columns?: string[]
    rows?: Record<string, unknown>[]
    rows_matrix?: unknown[][]
  },
  columns: string[] | undefined,
): { rows: Record<string, unknown>[]; columns: string[] | undefined } {
  const cols = columns?.length ? columns : page.columns
  if (page.rows_matrix?.length && cols?.length) {
    const rows = page.rows_matrix.map(row =>
      Object.fromEntries(cols.map((c, i) => [c, row[i] ?? ''])),
    )
    return { rows, columns: cols }
  }
  if (page.rows?.length) {
    return { rows: page.rows, columns: cols ?? page.columns }
  }
  return { rows: [], columns: cols }
}

/** Poll after POST /po/calculate (runs in background on the server). */
export async function waitForPoCalculate(
  onTick?: (message: string, progress?: number) => void,
  maxMs = 900_000,
): Promise<POCalculateResult> {
  const start = Date.now()
  let statusGatewayRetries = 0
  let lastServerMessage = 'Calculating PO recommendations…'
  let lastServerProgress: number | undefined
  while (Date.now() - start < maxMs) {
    let data: POCalculateResult & { row_count?: number; progress?: number; columns?: string[] }
    try {
      ;({ data } = await api.get<
        POCalculateResult & { row_count?: number; progress?: number; columns?: string[] }
      >('/po/calculate/status', { timeout: PO_STATUS_POLL_TIMEOUT_MS }))
    } catch (e: unknown) {
      if (_isGateway502(e) && statusGatewayRetries < 120) {
        statusGatewayRetries += 1
        const pct =
          lastServerProgress ??
          Math.min(98, 82 + Math.floor(statusGatewayRetries / 4))
        const busyMsg = lastServerMessage
          ? `${lastServerMessage} (server busy — retrying…)`
          : 'Server busy (502) — still calculating…'
        onTick?.(busyMsg, pct)
        await _sleep(statusGatewayRetries < 20 ? 2500 : 4000)
        continue
      }
      throw e
    }
    const st = data.status ?? 'idle'
    if (st === 'running') {
      const prog =
        typeof data.progress === 'number' && Number.isFinite(data.progress)
          ? Math.max(0, Math.min(100, Math.round(data.progress)))
          : undefined
      if (data.message) lastServerMessage = data.message
      if (prog != null) lastServerProgress = prog
      onTick?.(data.message || lastServerMessage, prog)
      await new Promise(r => setTimeout(r, 2000))
      continue
    }
    if (st === 'error') {
      throw new Error(data.message || 'PO calculation failed')
    }
    if (st === 'done') {
      const pageSize = PO_RESULT_PAGE_SIZE
      let offset = 0
      let columns: string[] | undefined = data.columns?.length ? [...data.columns] : undefined
      const allRows: Record<string, unknown>[] = []
      let meta: POCalculateResult = { ok: true, columns }
      let resultGatewayRetries = 0
      const expectedTotal = data.row_count ?? 0
      while (true) {
        let page: POCalculateResult & {
          offset?: number
          total?: number
          has_more?: boolean
          rows_matrix?: unknown[][]
        }
        try {
          ;({ data: page } = await api.get<
            POCalculateResult & {
              offset?: number
              total?: number
              has_more?: boolean
              rows_matrix?: unknown[][]
            }
          >('/po/calculate/result', {
            params: { offset, limit: pageSize, compact: 1 },
            timeout: PO_RESULT_TIMEOUT_MS,
          }))
          resultGatewayRetries = 0
        } catch (e: unknown) {
          if (_isGateway502(e) && resultGatewayRetries < 60) {
            resultGatewayRetries += 1
            const total = expectedTotal || allRows.length + pageSize
            const loadPct =
              total > 0
                ? 92 + Math.round((Math.min(allRows.length, total) / total) * 8)
                : 95
            const pageNum = Math.floor(offset / pageSize) + 1
            const pageTotal = total > 0 ? Math.ceil(total / pageSize) : '?'
            onTick?.(
              `Server busy (502) — retrying results page ${pageNum}/${pageTotal} (attempt ${resultGatewayRetries})…`,
              loadPct,
            )
            await _sleep(2500)
            continue
          }
          throw e
        }
        if (!page.ok) {
          throw new Error(page.message || 'Failed to load PO results')
        }
        const batch = _rowsFromPoResultPage(page, columns)
        if (batch.columns?.length) columns = batch.columns
        if (batch.rows.length) allRows.push(...batch.rows)
        meta = {
          ok: true,
          columns: columns ?? page.columns,
          sales_through: page.sales_through ?? meta.sales_through,
          planning_date: page.planning_date ?? meta.planning_date,
          raise_ledger_rows: page.raise_ledger_rows ?? meta.raise_ledger_rows,
          ledger_auto_import: page.ledger_auto_import ?? meta.ledger_auto_import,
        }
        const total = (page.total ?? expectedTotal) || allRows.length
        const loaded = Math.min(allRows.length, total)
        const loadPct =
          total > 0 ? 92 + Math.round((loaded / total) * 8) : 95
        onTick?.(
          `Loading PO results… ${loaded.toLocaleString()} / ${total.toLocaleString()}`,
          loadPct,
        )
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
export async function deleteRaiseLedgerDay(raisedDate: string): Promise<{ ok: boolean; message: string }> {
  const { data } = await api.delete<{ ok: boolean; message: string }>('/po/raise-ledger/day', {
    params: { raised_date: raisedDate },
  })
  return data
}

export async function deleteRaiseLedgerSkus(
  raisedDate: string,
  omsSkus: string[],
): Promise<{ ok: boolean; message: string }> {
  const { data } = await api.post<{ ok: boolean; message: string }>('/po/raise-ledger/delete-skus', {
    raised_date: raisedDate,
    oms_skus: omsSkus,
  })
  return data
}

export async function startPoCalculate(
  body: Record<string, unknown>,
  onTick?: (message: string, progress?: number) => void,
): Promise<POCalculateResult> {
  try {
    const { data } = await api.post<
      POCalculateResult & { status?: string; from_shared_cache?: boolean }
    >(
      '/po/calculate',
      body,
      { timeout: PO_REQUEST_TIMEOUT_MS },
    )
    if (!data.ok) {
      return data
    }
    if (data.from_shared_cache) {
      onTick?.(
        (data.message as string) || 'Loaded shared PO from an earlier run today…',
        100,
      )
    }
    if (data.status === 'running' || (!data.rows && data.status !== 'done')) {
      return waitForPoCalculate(onTick)
    }
    if (data.status === 'done' && !data.rows) {
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
  sales_rebuild?: 'pending'
}> {
  const { data } = await api.post('/upload/build-sales', undefined, {
    timeout: 60_000,
  })
  return data
}

export async function clearPlatform(platform: string): Promise<{ ok: boolean; message: string }> {
  const { data } = await api.delete(`/upload/clear/${platform}`)
  return data
}

// ── Coverage ──────────────────────────────────────────────────

export type JobStatusResponse = {
  server_time: string
  warm_cache: boolean
  warm_cache_generation: number
  upload_memory_lock_held: boolean
  daily_auto_ingest_status: string
  daily_auto_ingest_message?: string
  sales_rebuild_status: string
  sales_rebuild_message?: string
  session_restore_status: string
  inventory_upload_status: string
  daily_inventory_upload_status: string
}

export type DailyUploadVerifyResponse = {
  ok: boolean
  date: string
  message: string
  tier3_platforms: string[]
  tier3_upload_count: number
  recent_uploads: Array<Record<string, unknown>>
  session_sales_rows: number
  sales_ready: boolean
  dashboard_ready: boolean
  daily_auto_ingest_status?: string
  sales_rebuild_status?: string
  last_ingest_message?: string
}

export async function verifyDailyUpload(date: string): Promise<DailyUploadVerifyResponse> {
  const { data } = await api.get<DailyUploadVerifyResponse>('/upload/daily-auto/verify', {
    params: { date },
    timeout: 30_000,
  })
  return data
}

export async function getJobStatus(): Promise<JobStatusResponse> {
  const { data } = await api.get<JobStatusResponse>('/data/job-status', { timeout: 15_000 })
  return data
}

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

/** Full restore: warm + disk + Tier-3 + GitHub for missing platforms (Upload page). */
export type RestoreFullResponse = CoverageResponse & {
  ok: boolean
  message: string
  missing_platforms: string[]
  restore_steps: string[]
  restore_async?: boolean
}

export type RestoreProgressTick = {
  message: string
  progress: number
  step: string
}

function restoreTickFromCoverage(cov: CoverageResponse): RestoreProgressTick {
  const st = cov.session_restore_status ?? 'idle'
  let progress = cov.session_restore_progress ?? 0
  let step = cov.session_restore_step ?? ''
  let message = cov.session_restore_message || 'Restoring from server…'
  if (st === 'done' && (cov.sales_rebuild ?? 'idle') === 'running') {
    progress = Math.max(progress, 96)
    step = step || 'sales'
    message = cov.sales_rebuild_message || message
  }
  return { message, progress, step }
}

/** Poll until background restore (+ optional sales rebuild) finishes. */
export async function waitForSessionRestore(
  onTick?: (tick: RestoreProgressTick) => void,
): Promise<RestoreFullResponse> {
  const deadline = Date.now() + 20 * 60_000
  while (Date.now() < deadline) {
    const cov = await getCoverageResilient({ light: true, timeout: 60_000 })
    const tick = restoreTickFromCoverage(cov)
    onTick?.(tick)
    const st = cov.session_restore_status ?? 'idle'
    const salesSt = cov.sales_rebuild ?? 'idle'
    if (st === 'running') {
      await new Promise(r => setTimeout(r, 2500))
      continue
    }
    if (st === 'error') {
      throw new Error(cov.session_restore_message || 'Restore from server failed')
    }
    if (st === 'done') {
      if (salesSt === 'running') {
        await new Promise(r => setTimeout(r, 2500))
        continue
      }
      if (salesSt === 'error') {
        throw new Error(cov.sales_rebuild_message || 'Sales rebuild failed')
      }
      const full = await getCoverage({ light: false, timeout: 180_000 })
      const essentialMissing: string[] = []
      if (!full.mtr) essentialMissing.push('amazon')
      if (!full.myntra) essentialMissing.push('myntra')
      if (!full.meesho) essentialMissing.push('meesho')
      if (!full.flipkart) essentialMissing.push('flipkart')
      const ok =
        essentialMissing.length === 0 &&
        full.sku_mapping &&
        full.mtr &&
        full.sales
      return {
        ...full,
        ok,
        message: cov.session_restore_message || 'Full restore complete.',
        missing_platforms: full.snapdeal ? [] : ['snapdeal'].filter(() => !full.snapdeal),
        restore_steps: [],
        restore_async: false,
      }
    }
    await new Promise(r => setTimeout(r, 3500))
  }
  throw new Error('Restore timed out after 20 minutes — refresh the page and try again.')
}

export async function restoreFullFromServer(
  onTick?: (tick: RestoreProgressTick) => void,
): Promise<RestoreFullResponse> {
  try {
    const { data } = await api.post<RestoreFullResponse>('/data/restore-full', undefined, {
      timeout: 90_000,
    })
    if (data.restore_async || data.session_restore_status === 'running') {
      onTick?.(restoreTickFromCoverage(data))
      return waitForSessionRestore(onTick)
    }
    return data
  } catch (e: unknown) {
    if (isUploadGateway502(e) || (axios.isAxiosError(e) && !e.response)) {
      onTick?.({
        message: 'Gateway timed out — restore may still be running on the server…',
        progress: 5,
        step: 'queued',
      })
      return waitForSessionRestore(onTick)
    }
    throw new Error(_errMessage(e, 'Full restore from server failed'))
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
/** Fast path after login: copy server warm cache into session (Tier-3 merges in background). */
export async function cacheHydrateWarm() {
  try {
    const { data } = await api.post('/cache/hydrate-warm', undefined, { timeout: CACHE_TIMEOUT_MS })
    return data
  } catch (e: unknown) {
    throw new Error(_errMessage(e, 'Warm cache hydrate failed'))
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

export async function resetErpModuleData(module: string): Promise<{
  ok: boolean
  module: string
  rows_deleted: number
  message: string
}> {
  const { data } = await api.post('/erp-admin/reset-module-data', { module }, { timeout: 120_000 })
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
  clearIntelligenceCache()
  qc.invalidateQueries({
    predicate: q => {
      const key = Array.isArray(q.queryKey) ? q.queryKey[0] : q.queryKey
      return key !== 'auth-me'
    },
  })
}

export default api
