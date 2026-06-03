import axios from 'axios'
import api from '../api/client'

export type HourEntryPayload = {
  hour_col: string
  operation: string
  pieces: number
  sticker_in: number
  sticker_out: number
  manual_pieces: boolean
}

export type ProductionEntrySaveBody = {
  date: string
  karigar_id: string
  karigar_name: string
  challan_no: string
  style: string
  hour_entries: HourEntryPayload[]
}

export type ProductionEntrySaveResult = {
  ok: boolean
  message?: string
  rows_added?: number
  save_time?: string
}

const SAVE_TIMEOUT_MS = 60_000

function detailToString(detail: unknown): string {
  if (detail == null) return ''
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map(item => {
        if (typeof item === 'string') return item
        if (item && typeof item === 'object' && 'msg' in item) return String((item as { msg: string }).msg)
        return JSON.stringify(item)
      })
      .join('; ')
  }
  if (typeof detail === 'object') return JSON.stringify(detail)
  return String(detail)
}

/** User-facing message for production-entry save failures. */
export function formatProductionEntrySaveError(err: unknown): string {
  if (!axios.isAxiosError(err)) {
    return err instanceof Error ? err.message : 'Save failed — check connection and try again'
  }
  const status = err.response?.status
  const detail = detailToString(err.response?.data?.detail ?? err.response?.data?.message)
  if (detail) return detail
  if (err.code === 'ECONNABORTED' || status === 504) {
    return 'Save timed out — server may be busy. Wait a moment and tap Save again.'
  }
  if (status === 502 || status === 503) {
    return 'Server is temporarily unavailable. Wait a moment and try again.'
  }
  if (status === 401) return 'Session expired — sign in again.'
  if (status === 403) return 'You do not have permission to save production entry.'
  if (status && status >= 500) return `Server error (${status}). Try again in a moment.`
  if (err.message) return err.message
  return 'Save failed — check connection and try again'
}

function isRetryableStatus(status: number | undefined): boolean {
  return status === 502 || status === 503 || status === 504
}

/** POST /stitching/production-entry with timeout + one retry on gateway errors. */
export async function saveProductionEntry(
  body: ProductionEntrySaveBody,
): Promise<ProductionEntrySaveResult> {
  let lastErr: unknown
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const { data } = await api.post<ProductionEntrySaveResult>('/stitching/production-entry', body, {
        timeout: SAVE_TIMEOUT_MS,
      })
      return data
    } catch (err) {
      lastErr = err
      if (
        attempt === 0 &&
        axios.isAxiosError(err) &&
        (err.code === 'ECONNABORTED' || isRetryableStatus(err.response?.status))
      ) {
        await new Promise(r => setTimeout(r, 800))
        continue
      }
      throw err
    }
  }
  throw lastErr
}
