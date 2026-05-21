/**
 * Chunked multipart upload for large daily / inventory batches.
 * Each chunk is a small POST so slow links and proxies are less likely to time out.
 */
import axios from 'axios'

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

export function shouldUseChunkedUpload(files: File[]): boolean {
  if (!files.length) return false
  const total = files.reduce((s, f) => s + f.size, 0)
  return (
    total >= CHUNK_UPLOAD_BATCH_THRESHOLD ||
    files.some(f => f.size >= CHUNK_UPLOAD_FILE_THRESHOLD) ||
    files.length >= 4
  )
}

function fileChunkCount(size: number, chunkSize: number): number {
  return Math.max(1, Math.ceil(size / chunkSize))
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

  try {
    for (let fileIndex = 0; fileIndex < files.length; fileIndex++) {
      const file = files[fileIndex]
      const totalChunks = fileChunkCount(file.size, chunkSize)
      for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
        const start = chunkIndex * chunkSize
        const end = Math.min(start + chunkSize, file.size)
        const blob = file.slice(start, end)
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
        bytesSent += blob.size
      }
    }

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

    const { data: done } = await chunkApi.post<T & { ok?: boolean; message?: string }>(
      '/upload/chunk/complete',
      { upload_id: uploadId },
      { timeout: 120_000 },
    )
    if (!done || (done as { ok?: boolean }).ok === false) {
      throw new Error((done as { message?: string })?.message || 'Chunked upload finalize failed')
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
    throw e
  }
}
