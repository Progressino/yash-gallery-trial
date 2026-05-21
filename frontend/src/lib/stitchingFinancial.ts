/** Benchmark-based production financial audit (₹480 / operation target). */

export const BENCHMARK_DAILY_RATE_RS = 480

export type FinancialAudit = {
  budgetRatePerPiece: number
  budgetedAmount: number
  actualRatePerPiece: number
  actualAmount: number
  pl: number
}

export function computeFinancialAudit(
  baseTarget: number,
  pieces: number,
  dailyRate: number,
  opts?: { allocatedActualAmount?: number; allocatedBudgetedAmount?: number },
): FinancialAudit {
  const bt = Math.max(baseTarget || 0, 1)
  const pcs = Math.max(pieces || 0, 0)
  const dr = dailyRate || 0
  const budgetRatePerPiece = BENCHMARK_DAILY_RATE_RS / bt
  const budgetedAmount =
    opts?.allocatedBudgetedAmount ?? pcs * budgetRatePerPiece
  const actualAmount = opts?.allocatedActualAmount ?? dr
  const actualRatePerPiece = pcs > 0 ? actualAmount / pcs : 0
  return {
    budgetRatePerPiece: Math.round(budgetRatePerPiece * 100) / 100,
    budgetedAmount: Math.round(budgetedAmount * 100) / 100,
    actualRatePerPiece: Math.round(actualRatePerPiece * 100) / 100,
    actualAmount: Math.round(actualAmount * 100) / 100,
    pl: Math.round((budgetedAmount - actualAmount) * 100) / 100,
  }
}

/** Cap total benchmark budget at ₹480/day; split actual wage by piece share. */
export function computeDayFinancialSummary(
  opTotals: Record<string, { pieces: number; target: number }>,
  dailyRate: number,
): {
  totalPcs: number
  totalBudgeted: number
  totalActual: number
  pl: number
  opFin: Record<string, FinancialAudit>
} {
  const totalPcs = Object.values(opTotals).reduce((s, d) => s + d.pieces, 0)
  const rawBudgetByOp: Record<string, number> = {}
  let rawBudgetTotal = 0
  for (const [op, d] of Object.entries(opTotals)) {
    const raw = d.pieces * (BENCHMARK_DAILY_RATE_RS / Math.max(d.target, 1))
    rawBudgetByOp[op] = raw
    rawBudgetTotal += raw
  }
  const budgetScale =
    rawBudgetTotal > BENCHMARK_DAILY_RATE_RS
      ? BENCHMARK_DAILY_RATE_RS / rawBudgetTotal
      : 1

  let totalBudgeted = 0
  let totalActual = 0
  const opFin: Record<string, FinancialAudit> = {}
  for (const [op, d] of Object.entries(opTotals)) {
    const share = totalPcs > 0 ? d.pieces / totalPcs : 1
    const fin = computeFinancialAudit(d.target, d.pieces, dailyRate, {
      allocatedActualAmount: dailyRate * share,
      allocatedBudgetedAmount: rawBudgetByOp[op] * budgetScale,
    })
    opFin[op] = fin
    totalBudgeted += fin.budgetedAmount
    totalActual += fin.actualAmount
  }
  return {
    totalPcs,
    totalBudgeted: Math.round(totalBudgeted * 100) / 100,
    totalActual: Math.round(totalActual * 100) / 100,
    pl: Math.round((totalBudgeted - totalActual) * 100) / 100,
    opFin,
  }
}

export function formatProfitLoss(pl: number): string {
  const sign = pl >= 0 ? '+' : ''
  return `${sign}₹${pl.toFixed(2)}`
}
