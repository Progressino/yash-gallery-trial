/** Hour-wise production entry: sticker in − out → pieces (shared with Production Entry UI). */

export type HourEntryState = {
  operation: string
  pieces: number
  sticker_in: number
  sticker_out: number
  manual_pieces: boolean
}

export function emptyHourEntry(): HourEntryState {
  return { operation: '', pieces: 0, sticker_in: 0, sticker_out: 0, manual_pieces: false }
}

/** True when pieces should be computed from sticker in − out. */
export function isStickerMode(st: HourEntryState | undefined): boolean {
  if (!st || st.manual_pieces) return false
  return st.sticker_in !== 0 || st.sticker_out !== 0
}

/** Sticker in − out when in sticker mode; otherwise manual pieces. */
export function resolveHourPieces(st: HourEntryState | undefined): number {
  if (!st) return 0
  if (isStickerMode(st)) {
    return Math.max(0, Number(st.sticker_in) - Number(st.sticker_out))
  }
  return Number(st.pieces) || 0
}

/** Merge a field patch into one hour row (mirrors Production Entry setHour). */
export function applyHourEntryPatch(
  prev: HourEntryState | undefined,
  patch: Partial<HourEntryState>,
): HourEntryState {
  const base: HourEntryState = { ...emptyHourEntry(), ...prev, ...patch }
  const stickerTouched = 'sticker_in' in patch || 'sticker_out' in patch
  const piecesTouched = 'pieces' in patch && patch.pieces !== undefined

  if (stickerTouched && patch.manual_pieces !== true) {
    base.manual_pieces = false
    base.pieces = Math.max(0, Number(base.sticker_in) - Number(base.sticker_out))
  } else if (piecesTouched) {
    base.manual_pieces = true
  }

  return base
}

export function normalizeLoadedHourEntry(raw: Partial<HourEntryState>): HourEntryState {
  const si = Number(raw.sticker_in) || 0
  const so = Number(raw.sticker_out) || 0
  const pieces = Number(raw.pieces) || 0
  return {
    operation: String(raw.operation || ''),
    pieces,
    sticker_in: si,
    sticker_out: so,
    manual_pieces: si === 0 && so === 0,
  }
}

export function piecesInputValue(st: HourEntryState | undefined): string {
  if (!st) return ''
  if (isStickerMode(st)) return String(resolveHourPieces(st))
  return st.pieces > 0 ? String(st.pieces) : ''
}
