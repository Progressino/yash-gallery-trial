/** Client-side filename hints — server still validates content before ingest. */

export type UploadTarget =
  | 'snapshot_inventory'
  | 'daily_inventory_history'
  | 'daily_sales'
  | 'returns'
  | 'sku_status_lead'
  | 'existing_po'

export type UploadDocCategory =
  | 'daily_inventory_history_matrix'
  | 'snapshot_inventory'
  | 'daily_sales'
  | 'returns'
  | 'sku_status_lead'
  | 'existing_po'
  | 'neutral'

const HISTORY_NAME =
  /inventory[\s_-]*history|daily[\s_-]*inv(?:entory)?|inv[\s_-]*matrix|inventory-matrix/i
const SNAPSHOT_NAME =
  /(^oms\b|\boms[\s_-]|flipkart|myntra|amazon|ppmp|seller[\s_-]*inventory|current[\s_-]*inventory)/i
const RETURN_NAME =
  /return|seller_returns|tcs_sales_return|last\s*\d+\s*days?\s*return|return\s*data/i
const SALES_NAME =
  /shipment|seller.?orders|orders_\d{4}|mtr|tax[\s_-]*report|b2c|b2b|daily[\s_-]*order|\bsales\b/i
const EXISTING_PO_NAME =
  /existing[\s_-]*po|open[\s_-]*po|pipeline[\s_-]*po|po[\s_-]*pipeline/i
const SKU_STATUS_NAME =
  /sku[\s_-]*status|status[\s_-]*lead|lead[\s_-]*time/i

const TARGET_SECTION: Record<UploadTarget, string> = {
  snapshot_inventory: 'Daily uploads → Snapshot inventory',
  daily_inventory_history: 'History & setup → Daily inventory history matrix (PO)',
  daily_sales: 'Daily uploads → Daily order upload',
  returns: 'Daily uploads → Returns (for PO)',
  sku_status_lead: 'History & setup → SKU status & lead time',
  existing_po: 'History & setup → Existing PO',
}

const CATEGORY_LABEL: Record<UploadDocCategory, string> = {
  daily_inventory_history_matrix: 'wide daily inventory history matrix',
  snapshot_inventory: "today's snapshot inventory export",
  daily_sales: 'daily sales / shipment report',
  returns: 'returns report',
  sku_status_lead: 'SKU status & lead time sheet',
  existing_po: 'existing PO / pipeline sheet',
  neutral: 'this file type',
}

function scorePatterns(filename: string, re: RegExp, weight = 1): number {
  return re.test(filename) ? weight : 0
}

export function classifyUploadFilename(filename: string): UploadDocCategory {
  const scores: Array<[UploadDocCategory, number]> = [
    ['daily_inventory_history_matrix', scorePatterns(filename, HISTORY_NAME, 3)],
    ['returns', scorePatterns(filename, RETURN_NAME, 3)],
    ['daily_sales', scorePatterns(filename, SALES_NAME, 2)],
    ['snapshot_inventory', scorePatterns(filename, SNAPSHOT_NAME, 2)],
    ['existing_po', scorePatterns(filename, EXISTING_PO_NAME, 3)],
    ['sku_status_lead', scorePatterns(filename, SKU_STATUS_NAME, 3)],
  ]
  scores.sort((a, b) => b[1] - a[1])
  if (scores[0][1] <= 0) return 'neutral'
  if (scores[0][1] === scores[1][1]) return 'neutral'
  return scores[0][0]
}

/** @deprecated use classifyUploadFilename */
export function sniffUploadFilename(filename: string): {
  hint: UploadDocCategory
  likelyHistoryMatrix: boolean
  likelySnapshot: boolean
} {
  const category = classifyUploadFilename(filename)
  return {
    hint: category,
    likelyHistoryMatrix: category === 'daily_inventory_history_matrix',
    likelySnapshot: category === 'snapshot_inventory',
  }
}

const BLOCKED: Record<UploadTarget, Set<UploadDocCategory>> = {
  snapshot_inventory: new Set([
    'daily_inventory_history_matrix',
    'returns',
    'daily_sales',
    'sku_status_lead',
    'existing_po',
  ]),
  daily_inventory_history: new Set([
    'snapshot_inventory',
    'returns',
    'daily_sales',
    'sku_status_lead',
    'existing_po',
  ]),
  daily_sales: new Set([
    'daily_inventory_history_matrix',
    'returns',
    'snapshot_inventory',
    'sku_status_lead',
    'existing_po',
  ]),
  returns: new Set([
    'daily_inventory_history_matrix',
    'snapshot_inventory',
    'daily_sales',
    'sku_status_lead',
    'existing_po',
  ]),
  sku_status_lead: new Set([
    'daily_inventory_history_matrix',
    'snapshot_inventory',
    'returns',
    'daily_sales',
    'existing_po',
  ]),
  existing_po: new Set([
    'daily_inventory_history_matrix',
    'snapshot_inventory',
    'returns',
    'daily_sales',
    'sku_status_lead',
  ]),
}

const CATEGORY_TO_TARGET: Partial<Record<UploadDocCategory, UploadTarget>> = {
  daily_inventory_history_matrix: 'daily_inventory_history',
  snapshot_inventory: 'snapshot_inventory',
  daily_sales: 'daily_sales',
  returns: 'returns',
  sku_status_lead: 'sku_status_lead',
  existing_po: 'existing_po',
}

export function wrongTargetMessage(filename: string, detected: UploadDocCategory, correctTarget: UploadTarget): string {
  return (
    `“${filename}” looks like a ${CATEGORY_LABEL[detected]}, not the file for this card. ` +
    `Use ${TARGET_SECTION[correctTarget]} instead.`
  )
}

export function checkFileForUploadTarget(filename: string, currentTarget: UploadTarget): string | null {
  const detected = classifyUploadFilename(filename)
  if (detected === 'neutral') return null
  if (!BLOCKED[currentTarget].has(detected)) return null
  const correctTarget = CATEGORY_TO_TARGET[detected]
  if (!correctTarget) return null
  return wrongTargetMessage(filename, detected, correctTarget)
}

export function checkFilesForUploadTarget(files: File[], target: UploadTarget): string | null {
  for (const f of files) {
    const msg = checkFileForUploadTarget(f.name, target)
    if (msg) return msg
  }
  return null
}

/** @deprecated */
export const WRONG_TARGET_MESSAGES = {
  historyOnSnapshot: (filename: string) =>
    wrongTargetMessage(filename, 'daily_inventory_history_matrix', 'daily_inventory_history'),
  snapshotOnHistory: (filename: string) =>
    wrongTargetMessage(filename, 'snapshot_inventory', 'snapshot_inventory'),
} as const

/** @deprecated */
export function findMisplacedHistoryFiles(files: File[]): File[] {
  return files.filter(f => classifyUploadFilename(f.name) === 'daily_inventory_history_matrix')
}

/** @deprecated */
export function findMisplacedSnapshotFile(file: File): boolean {
  return classifyUploadFilename(file.name) === 'snapshot_inventory'
}

export function misplacedFilesForTarget(files: File[], target: UploadTarget): File[] {
  return files.filter(f => {
    const detected = classifyUploadFilename(f.name)
    return detected !== 'neutral' && BLOCKED[target].has(detected)
  })
}

const DAILY_TAB_TARGETS = new Set<UploadTarget>(['daily_sales', 'snapshot_inventory', 'returns'])

/** Split files across Daily uploads cards — auto-route instead of blocking. */
export function partitionFilesByUploadTarget(
  files: File[],
  requestedTarget: UploadTarget,
): { buckets: Partial<Record<UploadTarget, File[]>>; routedNotes: string[] } {
  const buckets: Partial<Record<UploadTarget, File[]>> = {}
  const routedNotes: string[] = []

  for (const f of files) {
    const detected = classifyUploadFilename(f.name)
    const routedTarget = CATEGORY_TO_TARGET[detected]

    if (
      detected !== 'neutral' &&
      routedTarget &&
      routedTarget !== requestedTarget &&
      DAILY_TAB_TARGETS.has(requestedTarget) &&
      DAILY_TAB_TARGETS.has(routedTarget)
    ) {
      buckets[routedTarget] = [...(buckets[routedTarget] ?? []), f]
      routedNotes.push(`“${f.name}” → ${TARGET_SECTION[routedTarget]}`)
    } else {
      buckets[requestedTarget] = [...(buckets[requestedTarget] ?? []), f]
    }
  }

  return { buckets, routedNotes }
}
