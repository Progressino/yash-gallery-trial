/**
 * Run: node frontend/src/lib/stitchingHourEntry.selftest.mjs
 */
import {
  applyHourEntryPatch,
  emptyHourEntry,
  resolveHourPieces,
  resolveSessionHourPieces,
} from './stitchingHourEntry.ts'

function assert(cond, msg) {
  if (!cond) throw new Error(msg)
}

const hours = [
  { col: 'H_09_10' },
  { col: 'H_10_11' },
  { col: 'H_11_12' },
]
const state = {
  H_09_10: { operation: 'Astin', sticker_in: 0, sticker_out: 30, pieces: 0, manual_pieces: false },
  H_10_11: { operation: 'Astin', sticker_in: 0, sticker_out: 55, pieces: 0, manual_pieces: false },
  H_11_12: { operation: 'Astin', sticker_in: 0, sticker_out: 95, pieces: 0, manual_pieces: false },
}
const pcs = resolveSessionHourPieces(hours, state)
assert(pcs.H_09_10 === 30, `h1 ${pcs.H_09_10}`)
assert(pcs.H_10_11 === 25, `h2 ${pcs.H_10_11}`)
assert(pcs.H_11_12 === 40, `h3 ${pcs.H_11_12}`)

assert(resolveHourPieces({ sticker_in: 10, sticker_out: 20, manual_pieces: false }) === 10, 'abs in-out')

console.log('stitchingHourEntry.selftest: OK')
