/**
 * Axios instance — automatically sends the session_id cookie.
 * In dev, Vite proxies /api → http://localhost:8000.
 * In production, nginx proxies /api → FastAPI.
 */
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  withCredentials: true,   // send session_id cookie
  headers: { 'Content-Type': 'application/json' },
})

// ── Typed API functions ───────────────────────────────────────

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
  inventory: boolean
  daily_orders: boolean
  mtr_rows: number
  sales_rows: number
  myntra_rows: number
  meesho_rows: number
  flipkart_rows: number
}

/** Upload SKU Mapping Excel file */
export async function uploadSkuMapping(file: File): Promise<UploadResponse> {
  const fd = new FormData()
  fd.append('file', file)
  const { data } = await api.post<UploadResponse>('/upload/sku-mapping', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

/** Upload Amazon MTR ZIP */
export async function uploadMtr(file: File): Promise<UploadResponse> {
  const fd = new FormData()
  fd.append('file', file)
  const { data } = await api.post<UploadResponse>('/upload/mtr', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

/** Get what data is loaded in the current session */
export async function getCoverage(): Promise<CoverageResponse> {
  const { data } = await api.get<CoverageResponse>('/data/coverage')
  return data
}

export default api
