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
