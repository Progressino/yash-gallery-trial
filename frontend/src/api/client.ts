/**
 * Axios instance — automatically sends the session_id cookie.
 * In dev, Vite proxies /api → http://localhost:8000.
 * In production, nginx proxies /api → FastAPI.
 */
import axios from 'axios'

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
  years?: number[]
  sku_count?: number
  detected_platforms?: string[]
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
  mtr_rows: number
  sales_rows: number
  myntra_rows: number
  meesho_rows: number
  flipkart_rows: number
  snapdeal_rows: number
  /** True after "Clear all app data" until an upload or explicit Load Cache / Fresh reload. */
  pause_auto_data_restore?: boolean
}

// ── Upload helpers ────────────────────────────────────────────

async function uploadFile(endpoint: string, file: File, extraFields?: Record<string, string>): Promise<UploadResponse> {
  const fd = new FormData()
  fd.append('file', file)
  if (extraFields) Object.entries(extraFields).forEach(([k, v]) => fd.append(k, v))
  const { data } = await api.post<UploadResponse>(endpoint, fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
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
  files: File[]
): Promise<{ ok: boolean; message: string; rows?: number; debug?: Record<string, unknown>; detected?: string[] }> {
  const fd = new FormData()
  files.forEach(f => fd.append('files', f))
  const { data } = await api.post('/upload/inventory-auto', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
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
  })
  return data
}

export async function uploadDailyAuto(
  files: File[]
): Promise<{ ok: boolean; message: string; detected_platforms?: string[] }> {
  const fd = new FormData()
  files.forEach(f => fd.append('files', f))
  const { data } = await api.post('/upload/daily-auto', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function buildSales(): Promise<{ ok: boolean; message: string; rows?: number }> {
  const { data } = await api.post('/upload/build-sales')
  return data
}

export async function clearPlatform(platform: string): Promise<{ ok: boolean; message: string }> {
  const { data } = await api.delete(`/upload/clear/${platform}`)
  return data
}

// ── Coverage ──────────────────────────────────────────────────

export async function getCoverage(): Promise<CoverageResponse> {
  const { data } = await api.get<CoverageResponse>('/data/coverage')
  return data
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
  const { data } = await api.get('/cache/status')
  return data
}
export async function cacheSave() {
  const { data } = await api.post('/cache/save')
  return data
}
export async function cacheLoad() {
  const { data } = await api.post('/cache/load')
  return data
}

/** Clear warm + session, re-download GitHub cache, merge Tier-3 SQLite, rebuild sales, re-save GitHub (background). */
export async function cacheReloadFresh(): Promise<{ ok: boolean; message: string; sales_rows?: number }> {
  const { data } = await api.post('/cache/reload-fresh')
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

// ── 401 interceptor — redirect to /login on expired/missing token ─────────────
api.interceptors.response.use(
  res => res,
  err => {
    const url: string = err.config?.url ?? ''
    if (
      err.response?.status === 401 &&
      !url.includes('/auth/') &&
      !window.location.pathname.startsWith('/login')
    ) {
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export default api
