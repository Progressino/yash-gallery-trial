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
