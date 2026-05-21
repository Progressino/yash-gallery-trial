/**
 * Chunked multipart upload for large daily / inventory batches.
 * Each chunk is a small POST so slow links and proxies are less likely to time out.
 */
import axios, { type AxiosError } from 'axios'

const chunkApi = axios.create({
  baseURL: '/api',
  withCredentials: true,
})

export type ChunkUploadTarget = 'daily-auto' | 'inventory-auto'

export type ChunkUploadProgress = {
  phase: 'init' | 'upload' | 'complete'
  filesTotal: number
  fileIndex: number
  chunkIndex: number
  chunksTotal: number
  bytesSent: number
  bytesTotal: number
  message: string
}

/** Use chunked path when any file or the whole batch is large enough to risk timeouts. */
export const CHUNK_UPLOAD_FILE_THRESHOLD = 1.5 * 1024 * 1024
export const CHUNK_UPLOAD_BATCH_THRESHOLD = 4 * 1024 * 1024

const CHUNK_REQUEST_TIMEOUT_MS = 120_000
const COMPLETE_TIMEOUT_MS = 45_000
const CHUNK_CONCURRENCY = 3
const CHUNK_RETRY_MAX = 4

export function shouldUseChunkedUpload(files: File[]): boolean {
  if (!files.length) return false
  // Multi-file batches always use chunks (direct POST closes UploadFile streams in background).
  if (files.length >= 2) return true
  const total = files.reduce((s, f) => s + f.size, 0)
  return (
    total >= CHUNK_UPLOAD_BATCH_THRESHOLD ||
    files.some(f => f.size >= CHUNK_UPLOAD_FILE_THRESHOLD)
  )
}

function fileChunkCount(size: number, chunkSize: number): number {
  return Math.max(1, Math.ceil(size / chunkSize))
}

function sleep(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms))
}

function isRetryableChunkError(e: unknown): boolean {
  if (!axios.isAxiosError(e)) return false
  const err = e as AxiosError
  if (err.code === 'ECONNABORTED') return true
  const st = err.response?.status
  return st === 502 || st === 503 || st === 504
}

async function postChunkWithRetry(fd: FormData): Promise<void> {
  let lastErr: unknown
  for (let attempt = 0; attempt < CHUNK_RETRY_MAX; attempt++) {
    try {
      const { data: part } = await chunkApi.post<{ ok: boolean; message?: string }>(
        '/upload/chunk',
        fd,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: CHUNK_REQUEST_TIMEOUT_MS,
        },
      )
      if (!part?.ok) {
        throw new Error(part?.message || 'Chunk upload failed')
      }
      return
    } catch (e) {
      lastErr = e
      if (attempt < CHUNK_RETRY_MAX - 1 && isRetryableChunkError(e)) {
        await sleep(1500 * (attempt + 1))
        continue
      }
      throw e
    }
  }
  throw lastErr
}

/** Run async tasks with a fixed concurrency limit. */
async function runPool<T>(items: (() => Promise<T>)[], concurrency: number): Promise<T[]> {
  const results: T[] = new Array(items.length)
  let next = 0
  async function worker(): Promise<void> {
    while (next < items.length) {
      const i = next++
      results[i] = await items[i]()
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, () => worker()))
  return results
}

export async function uploadFilesChunked<T extends Record<string, unknown>>(
  target: ChunkUploadTarget,
  files: File[],
  onProgress?: (p: ChunkUploadProgress) => void,
): Promise<T> {
  if (!files.length) {
    throw new Error('No files selected')
  }

  const bytesTotal = files.reduce((s, f) => s + f.size, 0)
  let bytesSent = 0

  onProgress?.({
    phase: 'init',
    filesTotal: files.length,
    fileIndex: 0,
    chunkIndex: 0,
    chunksTotal: 0,
    bytesSent: 0,
    bytesTotal,
    message: 'Starting chunked upload…',
  })

  const { data: initData } = await chunkApi.post<{
    ok: boolean
    message?: string
    upload_id?: string
    chunk_size?: number
  }>(
    '/upload/chunk/init',
    {
      target,
      files: files.map(f => ({ name: f.name, size: f.size })),
    },
    { timeout: 60_000 },
  )

  if (!initData?.ok || !initData.upload_id || !initData.chunk_size) {
    throw new Error(initData?.message || 'Could not start chunked upload')
  }

  const uploadId = initData.upload_id
  const chunkSize = initData.chunk_size

  type ChunkJob = { fileIndex: number; file: File; chunkIndex: number; totalChunks: number; blob: Blob }

  try {
    const jobs: ChunkJob[] = []
    for (let fileIndex = 0; fileIndex < files.length; fileIndex++) {
      const file = files[fileIndex]
      const totalChunks = fileChunkCount(file.size, chunkSize)
      for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
        const start = chunkIndex * chunkSize
        const end = Math.min(start + chunkSize, file.size)
        jobs.push({
          fileIndex,
          file,
          chunkIndex,
          totalChunks,
          blob: file.slice(start, end),
        })
      }
    }

    const jobFns = jobs.map(
      ({ fileIndex, file, chunkIndex, totalChunks, blob }) => async () => {
        const fd = new FormData()
        fd.append('upload_id', uploadId)
        fd.append('file_index', String(fileIndex))
        fd.append('chunk_index', String(chunkIndex))
        fd.append('total_chunks', String(totalChunks))
        fd.append('chunk', blob, `${file.name}.part${chunkIndex}`)

        onProgress?.({
          phase: 'upload',
          filesTotal: files.length,
          fileIndex,
          chunkIndex,
          chunksTotal: totalChunks,
          bytesSent,
          bytesTotal,
          message: `Uploading ${file.name} (${chunkIndex + 1}/${totalChunks})…`,
        })

        await postChunkWithRetry(fd)
        bytesSent += blob.size
      },
    )

    await runPool(jobFns, CHUNK_CONCURRENCY)

    onProgress?.({
      phase: 'complete',
      filesTotal: files.length,
      fileIndex: files.length - 1,
      chunkIndex: 0,
      chunksTotal: 0,
      bytesSent: bytesTotal,
      bytesTotal,
      message: 'Finishing upload on server…',
    })

    let done: (T & { ok?: boolean; message?: string }) | undefined
    for (let attempt = 0; attempt < CHUNK_RETRY_MAX; attempt++) {
      try {
        const res = await chunkApi.post<T & { ok?: boolean; message?: string }>(
          '/upload/chunk/complete',
          { upload_id: uploadId },
          { timeout: COMPLETE_TIMEOUT_MS },
        )
        done = res.data
        break
      } catch (e) {
        if (attempt < CHUNK_RETRY_MAX - 1 && isRetryableChunkError(e)) {
          await sleep(2000 * (attempt + 1))
          continue
        }
        throw e
      }
    }

    if (!done || done.ok === false) {
      throw new Error(done?.message || 'Chunked upload finalize failed')
    }
    return done as T
  } catch (e) {
    try {
      await chunkApi.delete(`/upload/chunk/${uploadId}`, { timeout: 15_000 })
    } catch {
      /* ignore cleanup errors */
    }
    if (axios.isAxiosError(e) && e.code === 'ECONNABORTED') {
      throw new Error('Chunk upload timed out. Check connection and try again.')
    }
    if (isRetryableChunkError(e)) {
      const err = new Error(
        'GATEWAY_502_CHUNK_COMPLETE: Server gateway timed out while finishing upload. '
        + 'Chunks were likely received — poll server status.',
      ) as Error & { gateway502?: boolean }
      err.gateway502 = true
      throw err
    }
    throw e
  }
}
