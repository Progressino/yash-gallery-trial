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
  opts?: { allocatedActualAmount?: number },
): FinancialAudit {
  const bt = Math.max(baseTarget || 0, 1)
  const pcs = Math.max(pieces || 0, 0)
  const dr = dailyRate || 0
  const budgetRatePerPiece = BENCHMARK_DAILY_RATE_RS / bt
  const budgetedAmount = pcs * budgetRatePerPiece
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

export function formatProfitLoss(pl: number): string {
  const sign = pl >= 0 ? '+' : ''
  return `${sign}₹${pl.toFixed(2)}`
}
