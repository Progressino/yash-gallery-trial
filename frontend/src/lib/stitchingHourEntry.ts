/** Hour-wise production entry: sticker in/out → pieces (shared with Production Entry UI). */

export type HourEntryState = {
  operation: string
  pieces: number
  sticker_in: number
  sticker_out: number
  manual_pieces: boolean
}

export type HourColRef = { col: string }

export function emptyHourEntry(): HourEntryState {
  return { operation: '', pieces: 0, sticker_in: 0, sticker_out: 0, manual_pieces: false }
}

/** True when pieces should be computed from stickers (not manual pieces). */
export function isStickerMode(st: HourEntryState | undefined): boolean {
  if (!st || st.manual_pieces) return false
  return st.sticker_in !== 0 || st.sticker_out !== 0
}

/** Single-hour resolve (no cumulative context). */
export function resolveHourPieces(st: HourEntryState | undefined): number {
  if (!st) return 0
  if (st.manual_pieces) return Number(st.pieces) || 0
  const si = Number(st.sticker_in) || 0
  const so = Number(st.sticker_out) || 0
  if (si === 0 && so === 0) return Number(st.pieces) || 0
  if (si === 0 && so > 0) return so
  return Math.abs(si - so)
}

/**
 * Resolve pieces for all hours in time order.
 * Sticker-out-only with rising counts = cumulative counter (hourly delta).
 */
export function resolveSessionHourPieces(
  hours: HourColRef[],
  hourState: Record<string, HourEntryState | undefined>,
): Record<string, number> {
  const out: Record<string, number> = {}
  let prevOut = 0
  for (const h of hours) {
    if (h.col === 'H_13_14') continue
    const st = hourState[h.col]
    if (!st) {
      out[h.col] = 0
      continue
    }
    if (st.manual_pieces) {
      out[h.col] = Number(st.pieces) || 0
      continue
    }
    const si = Number(st.sticker_in) || 0
    const so = Number(st.sticker_out) || 0
    if (si === 0 && so === 0) {
      out[h.col] = Number(st.pieces) || 0
      continue
    }
    if (si === 0 && so > 0) {
      let pcs = so
      if (prevOut > 0 && so >= prevOut) pcs = so - prevOut
      out[h.col] = pcs
      prevOut = Math.max(prevOut, so)
      continue
    }
    out[h.col] = Math.abs(si - so)
  }
  return out
}

/** Merge a field patch into one hour row (mirrors Production Entry setHour). */
export function applyHourEntryPatch(
  prev: HourEntryState | undefined,
  patch: Partial<HourEntryState>,
  sessionPieces?: Record<string, number>,
  hourCol?: string,
): HourEntryState {
  const base: HourEntryState = { ...emptyHourEntry(), ...prev, ...patch }
  const stickerTouched = 'sticker_in' in patch || 'sticker_out' in patch
  const piecesTouched = 'pieces' in patch && patch.pieces !== undefined

  if (stickerTouched && patch.manual_pieces !== true) {
    base.manual_pieces = false
    if (sessionPieces && hourCol && hourCol in sessionPieces) {
      base.pieces = sessionPieces[hourCol]
    } else {
      base.pieces = resolveHourPieces(base)
    }
  } else if (piecesTouched) {
    base.manual_pieces = true
  }

  return base
}

export function normalizeLoadedHourEntry(raw: Partial<HourEntryState>): HourEntryState {
  const si = Number(raw.sticker_in) || 0
  const so = Number(raw.sticker_out) || 0
  const pieces = Number(raw.pieces) || 0
  const manual = Boolean(raw.manual_pieces) || (si === 0 && so === 0 && pieces > 0)
  return {
    operation: String(raw.operation || ''),
    pieces,
    sticker_in: si,
    sticker_out: so,
    manual_pieces: manual,
  }
}

export function piecesInputValue(
  st: HourEntryState | undefined,
  sessionPieces?: Record<string, number>,
  hourCol?: string,
): string {
  if (!st) return ''
  const pcs =
    sessionPieces && hourCol && hourCol in sessionPieces
      ? sessionPieces[hourCol]
      : resolveHourPieces(st)
  if (pcs > 0) return String(pcs)
  if (isStickerMode(st)) return '0'
  return st.pieces > 0 ? String(st.pieces) : ''
}
