/** Calendar date YYYY-MM-DD in India (PO planning / raise ledger). */
export function calendarDateIST(d: Date = new Date()): string {
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Kolkata' }).format(d)
}

export function addCalendarDaysIST(isoDate: string, deltaDays: number): string {
  const [y, m, day] = isoDate.split('-').map(Number)
  const utc = new Date(Date.UTC(y, m - 1, day + deltaDays))
  return utc.toISOString().slice(0, 10)
}

export function yesterdayIST(): string {
  return addCalendarDaysIST(calendarDateIST(), -1)
}
