/** Client-side filename hints — server still validates content before ingest. */

const HISTORY_NAME =
  /inventory[\s_-]*history|daily[\s_-]*inv(?:entory)?|inv[\s_-]*matrix|inventory-matrix/i
const SNAPSHOT_NAME =
  /(^oms\b|\boms[\s_-]|flipkart|myntra|amazon|ppmp|seller[\s_-]*inventory|current[\s_-]*inventory|\.rar$)/i

export type UploadFilenameHint = 'history_matrix' | 'snapshot' | 'neutral'

export function sniffUploadFilename(filename: string): {
  hint: UploadFilenameHint
  likelyHistoryMatrix: boolean
  likelySnapshot: boolean
} {
  const name = filename.trim()
  const likelyHistoryMatrix = HISTORY_NAME.test(name)
  const likelySnapshot = SNAPSHOT_NAME.test(name)
  let hint: UploadFilenameHint = 'neutral'
  if (likelyHistoryMatrix && !likelySnapshot) hint = 'history_matrix'
  else if (likelySnapshot && !likelyHistoryMatrix) hint = 'snapshot'
  return { hint, likelyHistoryMatrix, likelySnapshot }
}

export const WRONG_TARGET_MESSAGES = {
  historyOnSnapshot: (filename: string) =>
    `“${filename}” looks like the Daily Inventory History matrix (wide Excel with many date columns). ` +
    'Upload it under History & setup → Daily inventory history matrix (PO), not Snapshot inventory.',
  snapshotOnHistory: (filename: string) =>
    `“${filename}” looks like a single-day snapshot (OMS / marketplace / RAR), not the wide history matrix. ` +
    'Upload it under Daily uploads → Snapshot inventory.',
} as const

export function findMisplacedHistoryFiles(files: File[]): File[] {
  return files.filter(f => {
    const { likelyHistoryMatrix, likelySnapshot } = sniffUploadFilename(f.name)
    return likelyHistoryMatrix && !likelySnapshot
  })
}

export function findMisplacedSnapshotFile(file: File): boolean {
  const { likelyHistoryMatrix, likelySnapshot } = sniffUploadFilename(file.name)
  return likelySnapshot && !likelyHistoryMatrix
}
