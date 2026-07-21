/** India reporting calendar — matches backend Asia/Kolkata dashboard filters. */
const IST = 'Asia/Kolkata'

export function todayIsoIST(date = new Date()): string {
  return new Intl.DateTimeFormat('en-CA', { timeZone: IST }).format(date)
}

export function addDaysIsoIST(iso: string, delta: number): string {
  const [y, m, d] = iso.split('-').map(Number)
  const utc = Date.UTC(y, m - 1, d) + delta * 86_400_000
  return todayIsoIST(new Date(utc))
}

export function daysAgoIsoIST(n: number): string {
  return addDaysIsoIST(todayIsoIST(), -n)
}

/** Monday of the current ISO week (IST). */
export function startOfWeekIsoIST(): string {
  const today = todayIsoIST()
  const [y, m, d] = today.split('-').map(Number)
  const utc = new Date(Date.UTC(y, m - 1, d))
  const dow = utc.getUTCDay()
  const mondayOffset = dow === 0 ? -6 : 1 - dow
  return addDaysIsoIST(today, mondayOffset)
}

/** First day of calendar month containing `iso` (or today). */
export function startOfMonthIsoIST(iso?: string): string {
  const base = (iso || todayIsoIST()).slice(0, 7)
  return `${base}-01`
}

export function reportingSpanDays(startIso: string, endIso: string): number | null {
  if (!startIso || !endIso) return null
  const d0 = new Date(`${startIso}T12:00:00`)
  const d1 = new Date(`${endIso}T12:00:00`)
  if (Number.isNaN(d0.getTime()) || Number.isNaN(d1.getTime())) return null
  const span = Math.round((d1.getTime() - d0.getTime()) / 86_400_000) + 1
  return span >= 1 ? span : null
}

/** Days sales_through is before planning_date (1 = yesterday's upload — normal). */
export function salesDataLagDays(planningIso: string, salesThroughIso: string): number | null {
  if (!planningIso || !salesThroughIso) return null
  const plan = new Date(`${planningIso.slice(0, 10)}T12:00:00`)
  const thru = new Date(`${salesThroughIso.slice(0, 10)}T12:00:00`)
  if (Number.isNaN(plan.getTime()) || Number.isNaN(thru.getTime())) return null
  return Math.round((plan.getTime() - thru.getTime()) / 86_400_000)
}

export function salesDataGapNeedsWarning(
  planningIso: string,
  salesThroughIso: string,
  maxExpectedLagDays = 1,
): boolean {
  const lag = salesDataLagDays(planningIso, salesThroughIso)
  if (lag == null) return false
  return lag > maxExpectedLagDays
}

/** Deploy/build time from API or Docker (ISO UTC) → readable IST date + time for sidebar. */
export function formatBuildDeployedAt(iso: string): string {
  const raw = (iso || '').trim()
  if (!raw) return ''
  const normalized = /[zZ]|[+-]\d{2}:?\d{2}$/.test(raw) ? raw : `${raw}Z`
  const d = new Date(normalized)
  if (Number.isNaN(d.getTime())) return raw.slice(0, 16).replace('T', ' ')
  return new Intl.DateTimeFormat('en-IN', {
    timeZone: IST,
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  }).format(d)
}

export function formatBuildVersionLabel(sha: string, builtAt?: string): string {
  const built = builtAt ? formatBuildDeployedAt(builtAt) : ''
  return built ? `${sha} · ${built}` : sha
}
