/** Slim indeterminate bar for route-level and long async work. */

export function IndeterminateBar({ className = '' }: { className?: string }) {
  return (
    <div
      className={`app-progress-track relative h-1.5 w-full overflow-hidden rounded-full bg-slate-200/90 ${className}`}
      role="progressbar"
      aria-valuetext="Loading"
    >
      <div className="app-progress-chunk absolute top-0 h-full w-[32%] rounded-full bg-[#002B5B]" />
    </div>
  )
}

/** Bordered card stripe + optional caption (sticky-friendly via className). */
export function PageLoadingStripe({
  active,
  label,
  className = '',
}: {
  active: boolean
  label?: string
  className?: string
}) {
  if (!active) return null
  return (
    <div
      className={`rounded-lg border border-slate-200/90 bg-white/95 px-3 py-2 shadow-sm ${className}`}
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <IndeterminateBar />
      {label ? <p className="mt-1.5 text-xs font-medium text-slate-600">{label}</p> : null}
    </div>
  )
}

/** Fixed under browser chrome — cache / global shell loads. */
export function FixedTopLoadingBar({ active, label }: { active: boolean; label?: string }) {
  if (!active) return null
  return (
    <div
      className="pointer-events-none fixed inset-x-0 top-0 z-[300]"
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <IndeterminateBar className="h-1 rounded-none" />
      {label ? (
        <div className="bg-[#002B5B] px-2 py-0.5 text-center text-[10px] font-semibold uppercase tracking-wide text-white">
          {label}
        </div>
      ) : null}
    </div>
  )
}
