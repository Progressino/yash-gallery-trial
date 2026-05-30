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

export function reportingSpanDays(startIso: string, endIso: string): number | null {
  if (!startIso || !endIso) return null
  const d0 = new Date(`${startIso}T12:00:00`)
  const d1 = new Date(`${endIso}T12:00:00`)
  if (Number.isNaN(d0.getTime()) || Number.isNaN(d1.getTime())) return null
  const span = Math.round((d1.getTime() - d0.getTime()) / 86_400_000) + 1
  return span >= 1 ? span : null
}
