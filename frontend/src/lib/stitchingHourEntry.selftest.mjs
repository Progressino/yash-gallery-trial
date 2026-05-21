/**
 * Run: node frontend/src/lib/stitchingHourEntry.selftest.mjs
 * Lightweight checks for sticker in/out → pieces (no vitest required).
 */
import {
  applyHourEntryPatch,
  emptyHourEntry,
  isStickerMode,
  normalizeLoadedHourEntry,
  piecesInputValue,
  resolveHourPieces,
} from './stitchingHourEntry.ts'

function assert(cond, msg) {
  if (!cond) throw new Error(msg)
}

let st = emptyHourEntry()
st = applyHourEntryPatch(st, { sticker_in: 10, manual_pieces: false })
assert(resolveHourPieces(st) === 10, `expected 10-0=10, got ${resolveHourPieces(st)}`)
assert(piecesInputValue(st) === '10', `pieces display should be 10, got ${piecesInputValue(st)}`)
assert(isStickerMode(st), 'sticker mode should be on')

st = applyHourEntryPatch(st, { sticker_out: 5 })
assert(resolveHourPieces(st) === 5, `expected 10-5=5, got ${resolveHourPieces(st)}`)

st = applyHourEntryPatch(emptyHourEntry(), { pieces: 12, manual_pieces: true })
assert(resolveHourPieces(st) === 12, 'manual pieces')
assert(!isStickerMode(st), 'not sticker mode when manual')

st = applyHourEntryPatch(st, { sticker_in: 10, sticker_out: 0, manual_pieces: false })
assert(resolveHourPieces(st) === 10, 'sticker patch should clear manual and compute 10')

const loaded = normalizeLoadedHourEntry({ sticker_in: 10, sticker_out: 0, pieces: 0, manual_pieces: true })
assert(!loaded.manual_pieces, 'load with stickers must not stay manual_pieces')
assert(resolveHourPieces(loaded) === 10, 'loaded row should resolve 10 from stickers')

console.log('stitchingHourEntry.selftest: OK')
