/**
 * Axios instance — automatically sends the session_id cookie.
 * In dev, Vite proxies /api → http://localhost:8000.
 * In production, nginx proxies /api → FastAPI.
 */
import axios from 'axios'
import { clearIntelligenceCache, type PlatformSummaryItem } from '../lib/intelligenceCache'
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
  existing_po_uploaded_at?: string
  existing_po_generation?: number
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
  inventory_rows?: number
  sku_status_lead_rows?: number
  daily_inventory_history_rows?: number
  daily_inventory_history_skus?: number
  daily_inventory_history_min_date?: string | null
  daily_inventory_history_max_date?: string | null
  daily_inventory_history_uploaded_at?: string | null
  daily_inventory_history_filename?: string | null
  inventory_snapshot_stale?: boolean
  inventory_snapshot_lag_days?: number | null
  daily_inventory_history_stale?: boolean
  daily_inventory_history_lag_days?: number | null
  inventory_staleness_warnings?: string[] | null
  manual_intransit_sheet?: boolean
  manual_intransit_skus?: number
  manual_intransit_units?: number
  manual_not_in_inventory_units?: number
  manual_intransit_uploaded_at?: string | null
  manual_intransit_filename?: string | null
  manual_intransit_parse_report?: ManualIntransitParseReport | null
  finishing_receipt_uploaded_at?: string | null
  finishing_receipt_filename?: string | null
  finishing_receipt_report?: FinishingReceiptReport | null
  po_raise_ledger_rows?: number
  return_sheet_skus?: number
  return_sheet_units?: number
  return_overlay_uploaded_at?: string | null
  return_overlay_filename?: string | null
  return_overlay_sources?: Array<{
    filename: string
    uploaded_at?: string
    platform?: string
    brand?: string
    skus?: number
    units?: number
  }>
  return_overlay_by_platform?: Array<{
    platform: string
    skus: number
    units: number
  }>
  returns_import_status?: 'idle' | 'running' | 'done' | 'error'
  returns_import_message?: string
  returns_import_progress?: number
  returns_import_warnings?: string[]
  /** True after "Clear all app data" until an upload or explicit Load Cache / Fresh reload. */
  pause_auto_data_restore?: boolean
  /** Tier-3 daily-auto background sales rebuild */
  sales_rebuild?: 'idle' | 'running' | 'done' | 'error'
  sales_rebuild_message?: string
  sales_data_revision?: number
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
    status: 'saved' | 'skipped' | 'error'
    reason?: string
    platform?: string
    rows?: number
  }>
  /** Tier-1 bulk history ZIP/RAR (MTR, Myntra PPMP, …) — async parse */
  tier1_bulk_status?: 'idle' | 'running' | 'done' | 'error'
  tier1_bulk_message?: string
  tier1_bulk_platform?: string
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
  existing_po_uploaded_at?: string | null
  existing_po_filename?: string | null
  existing_po_generation?: number
  existing_po_rows?: number
  existing_po_needs_recalc?: boolean
  existing_po_per_size_skus?: number
  existing_po_pipeline_skus?: number
  existing_po_new_order_skus?: number
  existing_po_looks_aggregated?: boolean
  existing_po_upload_status?: 'idle' | 'running' | 'done' | 'error'
  existing_po_upload_message?: string
  existing_po_upload_progress?: number
  daily_inventory_upload_status?: 'idle' | 'running' | 'done' | 'error'
  daily_inventory_upload_message?: string
  /** Session holds PO essentials (8/8 + row floors) — ignores background jobs. */
  data_ready?: boolean
  /** PO page may mount — data_ready and no critical session/inventory restore. */
  po_ready?: boolean
  /** Non-critical and critical background jobs currently running. */
  background_tasks?: Record<string, boolean>
  critical_restore_running?: boolean
  /** Intelligence dashboard — platform history available (PG/Tier-3 or session frames). */
  platforms_loaded?: boolean
  /** Session warm hydrate finished (not inflight). */
  hydration_complete?: boolean
  /** Full intelligence gate — sales + inventory + platforms + hydration settled. */
  intelligence_ready?: boolean
  /** Dashboard may fetch aggregates — 8/8 + platforms_loaded + hydration_complete. */
  dashboard_ready?: boolean
}

export interface IntelligenceReadinessResponse {
  intelligence_ready: boolean
  dashboard_ready: boolean
  precomputed_bundle_ready?: boolean
  tier3_platforms_in_window?: string[]
  data_ready?: boolean
  platforms_loaded: boolean
  hydration_complete: boolean
  hydration_inflight: boolean
  sales_available: boolean
  inventory_available: boolean
  sales_rows: number
  inventory_rows: number
  platform_rows: Record<string, number>
  data_source: string
  background_jobs: string[]
  background_tasks?: Record<string, boolean>
}

export interface DashboardSummaryResponse {
  source: string
  platforms: Record<string, { sales?: number; returns?: number; net?: number; loaded?: boolean }>
  platform_summary?: PlatformSummaryItem[]
  top_skus: Array<{ sku: string; units?: number }>
  sales_summary: Record<string, number>
  data_completeness?: 'partial' | 'full'
  message?: string
  version?: string
  stale?: boolean
  refresh_queued?: boolean
}

export interface IntelligenceVersionResponse {
  ok: boolean
  version: string
  start_date?: string
  end_date?: string
  basis?: string
  hot_ready?: boolean
  deep_ready?: boolean
  message?: string
}

export interface PoReadinessResponse {
  po_ready: boolean
  data_ready?: boolean
  sales_rows: number
  inventory_rows: number
  data_source: string
  hydration: string
  background_jobs: string[]
  background_tasks?: Record<string, boolean>
  critical_restore_running?: boolean
  calculate_allowed?: boolean
  pipeline_blockers?: string[]
  pipeline_warnings?: string[]
  pipeline_version?: number
  snapshot_id?: string
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
export async function uploadExistingPO(
  file: File,
  onProgress?: (pct: number, phase: 'upload' | 'server') => void,
): Promise<UploadResponse & { ingest_async?: boolean }> {
  const fd = new FormData()
  fd.append('file', file)
  try {
    const { data } = await api.post<UploadResponse>('/upload/existing-po', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: UPLOAD_TIMEOUT_MS,
      onUploadProgress: (e) => {
        if (!onProgress || !e.total) return
        onProgress(Math.round((e.loaded / e.total) * 100), 'upload')
      },
    })
    onProgress?.(100, 'server')
    return data
  } catch (e: unknown) {
    throw new Error(_errMessage(e, 'Existing PO upload failed'))
  }
}

export async function uploadFinishingReceipt(file: File): Promise<UploadResponse> {
  const fd = new FormData()
  fd.append('file', file)
  try {
    const { data } = await api.post<UploadResponse>('/upload/finishing-receipt', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: UPLOAD_TIMEOUT_MS,
    })
    return data
  } catch (e: unknown) {
    throw new Error(_errMessage(e, 'Finishing receipt upload failed'))
  }
}

export const uploadSnapdeal   = (file: File) => uploadFile('/upload/snapdeal', file)

export async function uploadInventoryAuto(
  files: File[],
  onProgress?: (p: ChunkUploadProgress) => void,
): Promise<{
  ok: boolean
  message: string
  ingest_async?: boolean
  chunked?: boolean
  wrong_upload_target?: boolean
  suggested_section?: string
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
    if (isUploadRequestInterrupted(e)) {
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
  opts?: {
    groupByParent?: boolean
    replace?: boolean
    onUploadProgress?: (pct: number, loaded: number, total: number) => void
  },
): Promise<{ ok: boolean; message: string; sales_rebuild?: string; returns_import?: string }> {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('group_by_parent', opts?.groupByParent ? 'true' : 'false')
  fd.append('replace', opts?.replace ? 'true' : 'false')
  try {
    const { data } = await api.post<{
      ok?: boolean
      message?: string
      sales_rebuild?: string
      returns_import?: string
      status?: string
    }>(
      '/po/returns/import-file',
      fd,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120_000,
        onUploadProgress: (evt) => {
          if (!opts?.onUploadProgress) return
          const total = evt.total || file.size || 0
          const loaded = evt.loaded || 0
          const pct = total > 0 ? Math.min(40, Math.round((loaded / total) * 40)) : 5
          opts.onUploadProgress(pct, loaded, total)
        },
      },
    )
    return {
      ok: !!data?.ok,
      message: data?.message || (data?.ok ? 'Returns import queued.' : 'Import failed.'),
      sales_rebuild: data?.sales_rebuild,
      returns_import: data?.returns_import || data?.status,
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

/** Upload may have reached the server before the client/proxy gave up (502/504/timeout). */
export function isUploadRequestInterrupted(e: unknown): boolean {
  if (isUploadGateway502(e)) return true
  if (axios.isAxiosError(e)) {
    if (e.code === 'ECONNABORTED') return true
    const st = e.response?.status
    if (st === 504 || st === 503) return true
    if (!e.response) return true
  }
  if (e instanceof Error && /timed out|timeout/i.test(e.message)) return true
  return false
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
      const retryable =
        isUploadGateway502(e)
        || (axios.isAxiosError(e) && !e.response)
        || (axios.isAxiosError(e) && e.code === 'ECONNABORTED')
        || (e instanceof Error && /timed out|timeout/i.test(e.message))
      if (retryable) {
        await new Promise(r => setTimeout(r, 1500 * (attempt + 1)))
        continue
      }
      throw e
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error('Coverage refresh failed')
}

export { getCoverageResilient }

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
      'Upload may have been accepted (connection timed out). Checking server status — stay on this page.',
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
    if (isUploadRequestInterrupted(e)) {
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
    if (isUploadRequestInterrupted(e)) {
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
  shouldAbort?: () => boolean,
): Promise<CoverageResponse> {
  const start = Date.now()
  let sawRunning = false
  while (Date.now() - start < maxMs) {
    if (shouldAbort?.()) {
      throw new Error('Inventory upload cancelled — you can upload again.')
    }
    const cov = await getCoverageResilient({ light: true, timeout: POLL_TIMEOUT_MS })
    const st = cov.inventory_upload_status ?? 'idle'
    if (st === 'running') {
      sawRunning = true
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
    if (sawRunning && st === 'idle') {
      return cov
    }
    await new Promise(r => setTimeout(r, 1500))
  }
  throw new Error('Inventory upload timed out — use “Clear stuck”, then upload again.')
}

/** Poll until Tier-1 bulk history parse finishes (MTR / Myntra / Meesho / Flipkart / Snapdeal). */
export async function waitForTier1Bulk(
  onTick?: (message: string) => void,
  maxMs = UPLOAD_BACKGROUND_WAIT_MS,
): Promise<CoverageResponse> {
  const start = Date.now()
  let sawRunning = false
  while (Date.now() - start < maxMs) {
    const cov = await getCoverageResilient({ light: true, timeout: POLL_TIMEOUT_MS })
    const st = cov.tier1_bulk_status ?? 'idle'
    if (st === 'running') {
      sawRunning = true
      const plat = cov.tier1_bulk_platform ? `${cov.tier1_bulk_platform}: ` : ''
      onTick?.(cov.tier1_bulk_message || `${plat}Parsing archive on server…`)
      await new Promise(r => setTimeout(r, 2000))
      continue
    }
    if (st === 'error') {
      throw new Error(cov.tier1_bulk_message || 'Bulk history upload failed')
    }
    if (st === 'done' || (sawRunning && st === 'idle')) {
      return cov
    }
    onTick?.('Waiting for server to start parsing…')
    await new Promise(r => setTimeout(r, 1500))
  }
  throw new Error('Bulk history upload timed out — refresh the page; row counts may still be updating.')
}

/** Poll until async return overlay import finishes and return_sheet is loaded. */
export async function waitForReturnsImport(
  onTick?: (message: string, progress?: number) => void,
  maxMs = UPLOAD_BACKGROUND_WAIT_MS,
): Promise<CoverageResponse> {
  const start = Date.now()
  while (Date.now() - start < maxMs) {
    const cov = await getCoverageResilient({ light: true, timeout: POLL_TIMEOUT_MS })
    const st = cov.returns_import_status ?? 'idle'
    if (st === 'running') {
      const prog = cov.returns_import_progress ?? 0
      onTick?.(cov.returns_import_message || 'Importing return data…', prog > 0 ? prog : undefined)
      await new Promise(r => setTimeout(r, 2500))
      continue
    }
    if (st === 'error') {
      throw new Error(cov.returns_import_message || 'Return import failed')
    }
    if (cov.return_sheet) {
      return cov
    }
    if (st === 'done' && !cov.return_sheet) {
      onTick?.('Finalizing return sheet…')
      await new Promise(r => setTimeout(r, 2000))
      continue
    }
    await new Promise(r => setTimeout(r, 2000))
  }
  throw new Error('Return import timed out — refresh the page in a minute.')
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
  po_merge_version?: number
  total_rows?: number
  row_count?: number
  summary?: {
    new_po_qty_sum?: number
    new_po_sku_count?: number
    pipeline_qty_sum?: number
    pipeline_sku_count?: number
    sheet_po_ordered_sum?: number
    existing_po_applied?: boolean
    existing_po_generation?: number
    existing_po_filename?: string
  }
}

export type DailyInventoryUploadResult = {
  ok: boolean
  status?: string
  message?: string
  wrong_upload_target?: boolean
  suggested_section?: string
  rows?: number
  skus?: number
  days?: number
  min_date?: string
  max_date?: string
}

export type UploadSkipDetail = {
  sheet?: string
  kind?: string
  reason?: string
  rows_affected?: number
  filename?: string
  status?: string
}

export type FinishingReceiptReport = {
  filename?: string
  sheet_format?: 'issue' | 'receive'
  rows_read?: number
  skus?: number
  issued_units?: number
  received_units?: number
  balance_units?: number
  left_units?: number
  non_clear_skus?: number
  issue_numbers?: string[]
  receive_numbers?: string[]
  report_date?: string
  updated_skus?: number
  added_skus?: number
  replaced_previous?: boolean
}

export type ManualIntransitParseReport = {
  filename?: string
  sheets_found?: Array<{ sheet: string; kind: string; rows: number }>
  sheets_skipped?: Array<{ sheet: string; reason: string }>
  skip_details?: UploadSkipDetail[]
  warnings?: string[]
  intransit_units?: number
  not_in_inventory_units?: number
  intransit_skus?: number
  not_in_inventory_skus?: number
  error?: string
}

export type ManualIntransitUploadResult = {
  ok: boolean
  message?: string
  skus?: number
  intransit_units?: number
  not_in_inventory_units?: number
  parse_report?: ManualIntransitParseReport
  replaced_previous?: boolean
}

/** Admin: Intrasit + Not In Inventory workbook (POST /po/manual-intransit-sheet). */
export async function uploadPoManualIntransitSheet(
  file: File,
): Promise<ManualIntransitUploadResult> {
  const fd = new FormData()
  fd.append('file', file)
  const { data } = await api.post<ManualIntransitUploadResult>(
    '/po/manual-intransit-sheet',
    fd,
    { headers: { 'Content-Type': 'multipart/form-data' }, timeout: UPLOAD_TIMEOUT_MS },
  )
  return {
    ok: !!data?.ok,
    message: data?.message,
    skus: data?.skus,
    intransit_units: data?.intransit_units,
    not_in_inventory_units: data?.not_in_inventory_units,
    parse_report: data?.parse_report,
    replaced_previous: data?.replaced_previous,
  }
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
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: UPLOAD_TIMEOUT_MS,
      onUploadProgress: e => {
        if (e.total && e.total > 0) {
          const pct = Math.min(99, Math.round((e.loaded / e.total) * 100))
          onTick?.(`Uploading file… ${pct}%`)
        } else if (e.loaded > 0) {
          onTick?.(`Uploading file… ${(e.loaded / (1024 * 1024)).toFixed(1)} MB`)
        }
      },
    },
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
      await new Promise(r => setTimeout(r, 1000))
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

export async function resetStuckDailyInventoryUpload(): Promise<{
  ok: boolean
  cleared: boolean
  message: string
}> {
  const { data } = await api.post('/po/daily-inventory-history/reset-stuck', {}, { timeout: 30_000 })
  return data
}

export type DailyInventoryHistoryDate = { date: string; rows: number; skus: number }

export async function getPoDailyInventoryHistoryDates(limit = 120) {
  const { data } = await api.get<{ ok: boolean; dates: DailyInventoryHistoryDate[] }>(
    '/po/daily-inventory-history/dates',
    { params: { limit } },
  )
  return data
}

export async function getPoDailyInventoryHistoryByDate(
  date: string,
  q = '',
  limit = 500,
  offset = 0,
) {
  const { data } = await api.get<{
    ok: boolean
    date: string
    rows: Array<{ sku: string; qty: number; in_stock: boolean; source: string }>
    total: number
  }>('/po/daily-inventory-history/by-date', {
    params: { date, q, limit, offset },
  })
  return data
}

export async function getPoDailyInventoryHistorySku(sku: string, windowDays = 30) {
  const { data } = await api.get<{
    ok: boolean
    loaded: boolean
    sku: string
    rows: Array<{ date: string; qty: number; in_stock: boolean; source: string }>
    in_stock_days?: number
    window_days?: number
    window_start?: string
    window_end?: string
  }>('/po/daily-inventory-history/sku', {
    params: { sku, window_days: windowDays },
  })
  return data
}

export type InventoryHistoryMatrixRow = { sku: string; qtys: number[] }

export async function getPoDailyInventoryHistoryMatrix(
  q = '',
  limit = 150,
  offset = 0,
  opts?: { days?: number; endDate?: string },
) {
  const { data } = await api.get<{
    ok: boolean
    loaded: boolean
    dates: string[]
    rows: InventoryHistoryMatrixRow[]
    total: number
    limit: number
    offset: number
    in_stock_min_qty?: number
    window_days?: number
    window_end?: string
  }>('/po/daily-inventory-history/matrix', {
    params: {
      q,
      limit,
      offset,
      days: opts?.days ?? 30,
      ...(opts?.endDate ? { end_date: opts.endDate } : {}),
    },
    timeout: 120_000,
  })
  return data
}

function _isAxiosTimeout(e: unknown): boolean {
  return axios.isAxiosError(e) && e.code === 'ECONNABORTED'
}

function _isGateway502(e: unknown): boolean {
  return axios.isAxiosError(e) && e.response?.status === 502
}

/** Status polls during long PO runs — gateway timeouts and network blips are common. */
function _isRetryablePoPollError(e: unknown): boolean {
  if (!axios.isAxiosError(e)) return false
  const st = e.response?.status
  if (st === 502 || st === 503 || st === 504) return true
  if (e.code === 'ECONNABORTED') return true
  return !e.response
}

function _poProgressWhileUnreachable(
  pollStart: number,
  lastProgress: number | undefined,
  retryCount: number,
): number {
  if (lastProgress != null && lastProgress > 0) {
    // Creep forward slowly so the bar does not look frozen during gateway blips.
    return Math.min(94, lastProgress + Math.min(8, Math.floor(retryCount / 4)))
  }
  const mins = (Date.now() - pollStart) / 60_000
  return Math.min(88, 12 + Math.round(mins * 10))
}

/** User-facing status while polls retry automatically (no "server busy" / attempt counts). */
function _poPollDisplayMessage(lastServerMessage: string, elapsedMs: number): string {
  const trimmed = (lastServerMessage || '')
    .replace(/\s*[—-]\s*server busy.*$/i, '')
    .replace(/\s*\(check \d+\).*$/i, '')
    .trim()
  const base =
    trimmed &&
    !/status check unavailable|retrying \(/i.test(trimmed)
      ? trimmed.replace(/\.\.\.$/, '')
      : 'Running PO calculation engine (large files may take 3–8 minutes)'
  const mins = Math.floor(elapsedMs / 60_000)
  if (mins >= 1) return `${base} — ${mins} min elapsed`
  return base
}

function _poPollRetryDelayMs(retryCount: number): number {
  return Math.min(5000, 1500 + retryCount * 250)
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

export type PoCalculateStatus = {
  job_id?: string
  status?: string
  message?: string
  progress?: number
  ok?: boolean
  row_count?: number
  columns?: string[]
  from_shared_cache?: boolean
}

function _poStatusPath(jobId?: string): string {
  return jobId ? `/po/calculate/status/${jobId}` : '/po/calculate/status'
}

function _poResultPath(jobId?: string): string {
  return jobId ? `/po/calculate/result/${jobId}` : '/po/calculate/result'
}

/** Lightweight status probe (used to auto-resume after refresh). */
export async function getPoCalculateStatus(jobId?: string): Promise<PoCalculateStatus> {
  const { data } = await api.get<PoCalculateStatus>(_poStatusPath(jobId), {
    timeout: PO_STATUS_POLL_TIMEOUT_MS,
  })
  return data
}

/** Fetch all paginated PO result rows (session must already be ``done``). */
export async function fetchAllPoResultPages(
  onTick?: (message: string, progress?: number) => void,
  hint?: { columns?: string[]; row_count?: number },
  jobId?: string,
): Promise<POCalculateResult> {
  const pageSize = PO_RESULT_PAGE_SIZE
  let offset = 0
  let columns: string[] | undefined = hint?.columns?.length ? [...hint.columns] : undefined
  const allRows: Record<string, unknown>[] = []
  let meta: POCalculateResult = { ok: true, columns }
  let resultGatewayRetries = 0
  const expectedTotal = hint?.row_count ?? 0
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
      >(_poResultPath(jobId), {
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
        onTick?.(
          `Loading PO results… ${Math.min(allRows.length, total).toLocaleString()} / ${total.toLocaleString()} rows`,
          loadPct,
        )
        await _sleep(_poPollRetryDelayMs(resultGatewayRetries))
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
      po_merge_version: page.po_merge_version ?? meta.po_merge_version,
      summary: page.summary ?? meta.summary,
      from_shared_cache: page.from_shared_cache ?? meta.from_shared_cache,
      shared_cache_at: page.shared_cache_at ?? meta.shared_cache_at,
    }
    const total = (page.total ?? expectedTotal) || allRows.length
    const loaded = Math.min(allRows.length, total)
    const loadPct = total > 0 ? 92 + Math.round((loaded / total) * 8) : 95
    onTick?.(
      `Loading PO results… ${loaded.toLocaleString()} / ${total.toLocaleString()}`,
      loadPct,
    )
    if (!page.has_more) break
    offset += pageSize
  }
  return { ...meta, rows: allRows }
}

/** Load PO table from the current session when a prior run finished (tab switch / refresh). */
export async function loadPoCalculateResultFromSession(
  onTick?: (message: string, progress?: number) => void,
): Promise<POCalculateResult | null> {
  try {
    const st = await getPoCalculateStatus()
    if (st.status === 'running') return null
    if (st.status === 'error') return null
    if (st.status !== 'done') {
      const { data: probe } = await api.get<
        POCalculateResult & { status?: string; total?: number; rows_matrix?: unknown[][] }
      >(
        '/po/calculate/result',
        { params: { offset: 0, limit: 1, compact: 1 }, timeout: PO_STATUS_POLL_TIMEOUT_MS },
      )
      if (!probe.ok || probe.status === 'stale') return null
      if ((probe.total ?? 0) <= 0 && !(probe.rows?.length || probe.rows_matrix?.length)) return null
    }
    onTick?.('Restoring PO results from server…', 90)
    return await fetchAllPoResultPages(onTick, {
      columns: st.columns,
      row_count: st.row_count,
    })
  } catch {
    return null
  }
}

export type PoSharedCacheAvailability = {
  available: boolean
  planning_date?: string
  row_count?: number
  computed_at?: string
  sales_through?: string
  po_merge_version?: number
}

/** Check if another user already computed PO today with the same settings. */
export async function getPoSharedCacheAvailability(
  body: Record<string, unknown>,
): Promise<PoSharedCacheAvailability> {
  const { data } = await api.get<PoSharedCacheAvailability>('/po/calculate/shared-cache', {
    params: body,
    timeout: PO_STATUS_POLL_TIMEOUT_MS,
  })
  return data
}

/** If the server is still running PO calc, poll until done and return results. */
export async function resumePoCalculateIfRunning(
  onTick?: (message: string, progress?: number) => void,
): Promise<POCalculateResult | null> {
  try {
    const st = await getPoCalculateStatus()
    if (st.status !== 'running') return null
    onTick?.(st.message || 'Resuming PO calculation…', st.progress ?? 10)
    return waitForPoCalculate(onTick)
  } catch {
    return null
  }
}

/** Poll after POST /po/calculate (runs in background on the server). */
export async function waitForPoCalculate(
  onTick?: (message: string, progress?: number) => void,
  maxMs = 900_000,
  jobId?: string,
): Promise<POCalculateResult> {
  const start = Date.now()
  const statusPath = _poStatusPath(jobId)
  let statusGatewayRetries = 0
  let idlePolls = 0
  let lastServerMessage = 'Calculating PO recommendations…'
  let lastServerProgress: number | undefined
  let sawRunning = false
  while (Date.now() - start < maxMs) {
    let data: POCalculateResult & { row_count?: number; progress?: number; columns?: string[] }
    try {
      ;({ data } = await api.get<
        POCalculateResult & { row_count?: number; progress?: number; columns?: string[] }
      >(statusPath, { timeout: PO_STATUS_POLL_TIMEOUT_MS }))
      statusGatewayRetries = 0
    } catch (e: unknown) {
      if (_isRetryablePoPollError(e) && statusGatewayRetries < 180) {
        statusGatewayRetries += 1
        const pct = _poProgressWhileUnreachable(start, lastServerProgress, statusGatewayRetries)
        onTick?.(_poPollDisplayMessage(lastServerMessage, Date.now() - start), pct)
        await _sleep(_poPollRetryDelayMs(statusGatewayRetries))
        continue
      }
      throw e
    }
    let st = data.status ?? 'idle'
    if (st === 'running') {
      sawRunning = true
      idlePolls = 0
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
    if (st === 'idle' && sawRunning) {
      try {
        const { data: probe } = await api.get<
          POCalculateResult & { status?: string; total?: number }
        >(_poResultPath(jobId), {
          params: { offset: 0, limit: 1, compact: 1 },
          timeout: PO_STATUS_POLL_TIMEOUT_MS,
        })
        if (probe.ok && (probe.rows?.length || (probe.total ?? 0) > 0)) {
          data = {
            status: 'done',
            ok: true,
            row_count: probe.total,
            columns: probe.columns,
            message: 'PO calculation complete.',
          }
          st = 'done'
        } else if (probe.status === 'running') {
          onTick?.('Still calculating on server…', lastServerProgress ?? 40)
          await _sleep(3000)
          continue
        } else {
          await _sleep(2500)
          continue
        }
      } catch {
        await _sleep(2500)
        continue
      }
    }
    if (st === 'idle') {
      idlePolls += 1
      onTick?.(
        sawRunning
          ? 'Waiting for server to finish…'
          : lastServerMessage || 'Waiting for PO calculation to start…',
        lastServerProgress ?? Math.min(12 + idlePolls, 40),
      )
      if (!sawRunning && idlePolls >= 24) {
        throw new Error(
          'PO calculation did not start on the server. Hard-refresh the page and click Calculate PO again. If it persists, restart the local PO stack.',
        )
      }
      if (sawRunning && idlePolls >= 40) {
        throw new Error(
          'PO calculation lost server progress. Hard-refresh and click Calculate PO again.',
        )
      }
      await new Promise(r => setTimeout(r, 1500))
      continue
    }
    if (st === 'done') {
      const loaded = await fetchAllPoResultPages(
        onTick,
        {
          columns: data.columns,
          row_count: data.row_count,
        },
        jobId,
      )
      return {
        ...loaded,
        from_shared_cache: data.from_shared_cache ?? loaded.from_shared_cache,
        shared_cache_at: data.shared_cache_at ?? loaded.shared_cache_at,
        message: data.message ?? loaded.message,
      }
    }
  }
  throw new Error(
    'PO calculation timed out — the server may still be finishing. Wait 2 minutes, hard-refresh, and check if the table loaded.',
  )
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
      POCalculateResult & { status?: string; from_shared_cache?: boolean; job_id?: string }
    >(
      '/po/calculate',
      body,
      { timeout: PO_REQUEST_TIMEOUT_MS },
    )
    if (!data.ok) {
      return data
    }
    const jobId = data.job_id
    if (jobId) {
      return waitForPoCalculate(onTick, 900_000, jobId)
    }
    if (data.from_shared_cache) {
      onTick?.(
        (data.message as string) || 'Loaded shared PO from an earlier run today…',
        100,
      )
    }
    if (data.status === 'done' && !data.rows?.length) {
      onTick?.('Loading PO results…', 92)
      const loaded = await fetchAllPoResultPages(onTick, {
        columns: data.columns,
        row_count: data.total_rows ?? data.row_count,
      })
      return {
        ...loaded,
        from_shared_cache: data.from_shared_cache ?? loaded.from_shared_cache,
        shared_cache_at: data.shared_cache_at ?? loaded.shared_cache_at,
        message: data.message ?? loaded.message,
        po_merge_version: data.po_merge_version ?? loaded.po_merge_version,
      }
    }
    if (data.status === 'running' || (!data.rows && data.status !== 'done')) {
      return waitForPoCalculate(onTick)
    }
    return data
  } catch (e: unknown) {
    if (_isAxiosTimeout(e)) {
      onTick?.('Still calculating on server…')
      return waitForPoCalculate(onTick)
    }
    if (axios.isAxiosError(e) && e.response?.status === 502) {
      onTick?.('Still calculating on server — checking progress…', 8)
      await _sleep(2000)
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
  tier1_bulk_status: string
  tier1_bulk_message?: string
}

export type DailyUploadVerifyResponse = {
  ok: boolean
  date: string
  message: string
  tier3_platforms: string[]
  tier3_upload_count: number
  tier3_summary?: DailySummary
  recent_uploads: Array<Record<string, unknown>>
  session_sales_rows: number
  session_sales_range?: { min: string | null; max: string | null }
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

/** Wait until server warm cache is ready (post-deploy startup can take several minutes). */
export async function waitForWarmCacheReady(opts?: {
  maxWaitMs?: number
  pollMs?: number
}): Promise<boolean> {
  const maxWaitMs = opts?.maxWaitMs ?? 180_000
  const pollMs = opts?.pollMs ?? 3_000
  const deadline = Date.now() + maxWaitMs
  while (Date.now() < deadline) {
    try {
      const j = await getJobStatus()
      if (j.warm_cache && j.warm_cache_generation >= 1) return true
    } catch {
      /* server may still be starting */
    }
    await new Promise(r => setTimeout(r, pollMs))
  }
  return false
}

export async function getPoReadiness(opts?: { timeout?: number }): Promise<PoReadinessResponse> {
  const { data } = await api.get<PoReadinessResponse>('/po/readiness', {
    timeout: opts?.timeout ?? 45_000,
  })
  return data
}

export async function getIntelligenceReadiness(opts?: {
  timeout?: number
}): Promise<IntelligenceReadinessResponse> {
  const { data } = await api.get<IntelligenceReadinessResponse>('/data/intelligence/readiness', {
    timeout: opts?.timeout ?? 90_000,
  })
  return data
}

export async function getIntelligenceVersion(opts: {
  startDate: string
  endDate: string
  basis?: 'gross' | 'net'
}): Promise<IntelligenceVersionResponse> {
  const { data } = await api.get<IntelligenceVersionResponse>('/data/intelligence/version', {
    params: {
      start_date: opts.startDate,
      end_date: opts.endDate,
      basis: opts.basis ?? 'gross',
    },
    timeout: 15_000,
  })
  return data
}

export async function getDashboardSummary(opts?: {
  startDate?: string
  endDate?: string
  limit?: number
  timeout?: number
}): Promise<DashboardSummaryResponse> {
  const params: Record<string, string | number> = {}
  if (opts?.startDate) params.start_date = opts.startDate
  if (opts?.endDate) params.end_date = opts.endDate
  if (opts?.limit != null) params.limit = opts.limit
  const { data } = await api.get<DashboardSummaryResponse>('/data/dashboard/summary', {
    params,
    timeout: opts?.timeout ?? 60_000,
  })
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

export type DataParityReport = {
  ok: boolean
  planning_date: string
  sales_through: string
  tier3_file_count: number
  tier3_sync_mismatch: boolean
  tier3_platforms_mismatch: string[]
  warnings: string[]
}

export async function getDataParity(planningDate?: string): Promise<DataParityReport> {
  const params = planningDate ? { planning_date: planningDate } : undefined
  const { data } = await api.get<DataParityReport>('/data/parity', { params, timeout: 30_000 })
  return data
}

export type PoSkuAuditCheck = {
  field: string
  engine: number
  tier3_window: number
  delta: number
  ok: boolean
  note: string
}

export type PoSkuAuditResponse = {
  ok: boolean
  sku: string
  planning_date?: string
  period_days?: number
  ads_window?: { start: string; end: string }
  po_row?: Record<string, unknown> | null
  shared_cache?: { created_at_ist?: string; total_rows?: number; sales_through?: string }
  tier3?: { sold_units: number; return_units: number; net_units: number; platforms?: Record<string, { sold: number; returns: number }> }
  checks?: PoSkuAuditCheck[]
  message?: string
}

/** Cross-check PO row vs Tier-3 sales in the ADS window (same params as PO calculate). */
export async function getPoSkuAudit(
  sku: string,
  calcParams: Record<string, string | number | boolean | undefined>,
): Promise<PoSkuAuditResponse> {
  const params: Record<string, string | number | boolean> = { sku }
  for (const [k, v] of Object.entries(calcParams)) {
    if (v !== undefined && v !== '') params[k] = v
  }
  const { data } = await api.get<PoSkuAuditResponse>('/po/sku-audit', { params, timeout: 90_000 })
  return data
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
    const { data } = await api.post('/cache/hydrate-warm', undefined, { timeout: 30_000 })
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

/** Lightweight: sync Tier-3 SQLite into session + rebuild sales. No GitHub download. */
export async function cacheSyncTier3(): Promise<{ ok: boolean; message: string; sales_rows?: number }> {
  const { data } = await api.post('/cache/sync-tier3', undefined, { timeout: 300_000 })
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

export type UploadReconciliationReport = {
  ok: boolean
  file_count: number
  files: {
    platform: string
    filename: string
    upload_kind: string
    rows: number
    date_from?: string
    date_to?: string
    uploaded_at?: string
  }[]
  mismatches: {
    month: string
    segment: string
    txn: string
    daily_units: number
    monthly_units: number
    unit_diff: number
    daily_amount: number
    monthly_amount: number
    amount_diff: number
  }[]
  mismatch_count: number
  dedup_by_platform: Record<string, { raw_rows: number; deduped_rows: number; collapsible_rows: number }>
  total_collapsible_rows: number
  return_overlay: { skus?: number; units?: number }
  hints: string[]
  parquet_files_loaded?: number
  parquet_files_skipped?: number
  elapsed_ms?: number
  cached?: boolean
}

export async function getUploadReconciliation(
  startMonth?: string,
  endMonth?: string,
): Promise<UploadReconciliationReport> {
  const params: Record<string, string> = {}
  if (startMonth) params.start_month = startMonth
  if (endMonth) params.end_month = endMonth
  const { data } = await api.get<UploadReconciliationReport>('/data/upload-reconciliation', {
    params,
    timeout: 300_000,
  })
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
