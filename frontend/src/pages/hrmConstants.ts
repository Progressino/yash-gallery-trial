export const FREQUENCIES = [
  'Daily',
  'Weekly',
  'Fortnightly',
  'Monthly',
  'Quarterly',
  'Yearly',
  'Whenever Required',
] as const

export const PRIORITIES = ['High', 'Medium', 'Low', 'Critical'] as const

export const TIME_PERIODS = [
  'Morning',
  'Afternoon',
  'Evening',
  'Full Day',
  'Shift-A',
  'Shift-B',
  'Custom',
] as const

export const WEEKDAYS = [
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
  'Sunday',
] as const

export const MONTHS = [
  { value: 1, label: 'January' },
  { value: 2, label: 'February' },
  { value: 3, label: 'March' },
  { value: 4, label: 'April' },
  { value: 5, label: 'May' },
  { value: 6, label: 'June' },
  { value: 7, label: 'July' },
  { value: 8, label: 'August' },
  { value: 9, label: 'September' },
  { value: 10, label: 'October' },
  { value: 11, label: 'November' },
  { value: 12, label: 'December' },
] as const

export const priorityStyle = (p: string) => {
  if (p === 'Critical') return 'bg-red-100 text-red-800'
  if (p === 'High') return 'bg-orange-100 text-orange-800'
  if (p === 'Medium') return 'bg-blue-100 text-blue-800'
  return 'bg-gray-100 text-gray-700'
}

/** Calendar date in IST (the HRM server's timezone), YYYY-MM-DD. */
export const todayIst = () => new Date(Date.now() + 330 * 60_000).toISOString().slice(0, 10)
export const tomorrowIst = () => new Date(Date.now() + 330 * 60_000 + 86_400_000).toISOString().slice(0, 10)

export const fmtHM = (sec?: number | null) => {
  const s = Math.max(0, Math.floor(Number(sec || 0)))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  if (!h && !m) return s > 0 ? '<1m' : '0m'
  return h ? `${h}h ${m}m` : `${m}m`
}

/** Frequencies that accept a Dynamic Schedule Rule (e.g. "1st Monday", "Last Working Day"). */
export const SCHEDULE_RULE_FREQS = ['Monthly', 'Fortnightly', 'Yearly']
