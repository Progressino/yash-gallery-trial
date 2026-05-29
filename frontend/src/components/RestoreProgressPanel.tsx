/** Progress UI for Upload → Restore all from server (polls coverage session_restore_*). */

/** Order matches backend restore priority (GitHub before disk / Tier-3). */
export const RESTORE_UI_STEPS: { id: string; label: string }[] = [
  { id: 'queued', label: 'Queued' },
  { id: 'waiting', label: 'Waiting for server' },
  { id: 'sku', label: 'SKU mapping' },
  { id: 'warm', label: 'Warm cache (platforms + sales)' },
  { id: 'github_download', label: 'GitHub — downloading (priority)' },
  { id: 'github_amazon', label: 'GitHub — Amazon (MTR)' },
  { id: 'github_myntra', label: 'GitHub — Myntra' },
  { id: 'github_meesho', label: 'GitHub — Meesho' },
  { id: 'github_flipkart', label: 'GitHub — Flipkart' },
  { id: 'github_snapdeal', label: 'GitHub — Snapdeal' },
  { id: 'github_inventory', label: 'GitHub — inventory' },
  { id: 'disk', label: 'On-disk backup' },
  { id: 'inventory', label: 'Inventory snapshot' },
  { id: 'tier3', label: 'Tier-3 daily history' },
  { id: 'daily_store', label: 'Daily upload store' },
  { id: 'publish', label: 'Saving warm cache' },
  { id: 'sales_queue', label: 'Queuing sales rebuild' },
  { id: 'sales', label: 'Combined sales rebuild' },
  { id: 'done', label: 'Complete' },
]

const STEP_ORDER = RESTORE_UI_STEPS.map(s => s.id)

function stepStatus(
  stepId: string,
  currentStep: string,
): 'done' | 'current' | 'pending' {
  const cur = STEP_ORDER.indexOf(currentStep)
  const idx = STEP_ORDER.indexOf(stepId)
  if (idx < 0 || cur < 0) return stepId === currentStep ? 'current' : 'pending'
  if (idx < cur) return 'done'
  if (idx === cur) return 'current'
  return 'pending'
}

type Props = {
  message: string
  progress: number
  step: string
}

export default function RestoreProgressPanel({ message, progress, step }: Props) {
  const pct = Math.max(0, Math.min(100, progress))
  const current = step || 'queued'

  return (
    <div className="rounded-xl border border-[#002B5B]/25 bg-gradient-to-br from-blue-50 to-slate-50 px-4 py-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <p className="text-sm font-semibold text-[#002B5B]">Restoring all data from server</p>
        <span className="text-sm font-bold tabular-nums text-[#002B5B]">{pct}%</span>
      </div>
      <div className="h-2.5 w-full rounded-full bg-slate-200 overflow-hidden mb-2">
        <div
          className="h-full rounded-full bg-[#002B5B] transition-all duration-500 ease-out"
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
      <p className="text-xs text-slate-700 mb-3 font-medium">{message}</p>
      <p className="text-[10px] uppercase tracking-wide text-slate-500 mb-1.5">Steps</p>
      <ul className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1 max-h-44 overflow-y-auto text-xs">
        {RESTORE_UI_STEPS.filter(s => s.id !== 'done').map(s => {
          const st = stepStatus(s.id, current)
          return (
            <li
              key={s.id}
              className={`flex items-center gap-1.5 ${
                st === 'current'
                  ? 'font-semibold text-[#002B5B]'
                  : st === 'done'
                    ? 'text-green-700'
                    : 'text-slate-400'
              }`}
            >
              <span aria-hidden className="w-4 text-center shrink-0">
                {st === 'done' ? '✓' : st === 'current' ? '●' : '○'}
              </span>
              <span>{s.label}</span>
            </li>
          )
        })}
      </ul>
      <p className="mt-2 text-[10px] text-slate-500">
        Full Amazon (MTR) loads from GitHub first. Large history can take 10–15 minutes — the timer
        in the message updates every few seconds while each step runs.
      </p>
    </div>
  )
}
