/** Slim indeterminate bar for route-level and long async work. */

export function DeterminateBar({
  percent,
  className = '',
}: {
  percent: number
  className?: string
}) {
  const pct = Math.max(0, Math.min(100, Math.round(percent)))
  return (
    <div
      className={`h-2 w-full overflow-hidden rounded-full bg-slate-200/90 ${className}`}
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className="h-full rounded-full bg-[#002B5B] transition-[width] duration-300 ease-out"
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

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
  percent,
  className = '',
}: {
  active: boolean
  label?: string
  /** When set (0–100), show a determinate progress bar instead of the indeterminate stripe. */
  percent?: number | null
  className?: string
}) {
  if (!active) return null
  const showPct = percent != null && Number.isFinite(percent)
  const pct = showPct ? Math.max(0, Math.min(100, Math.round(percent))) : null
  return (
    <div
      className={`rounded-lg border border-slate-200/90 bg-white/95 px-3 py-2 shadow-sm ${className}`}
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      {showPct && pct != null ? <DeterminateBar percent={pct} /> : <IndeterminateBar />}
      {label ? (
        <p className="mt-1.5 text-xs font-medium text-slate-600">
          {label}
          {showPct && pct != null ? (
            <span className="float-right tabular-nums text-slate-500">{pct}%</span>
          ) : null}
        </p>
      ) : showPct && pct != null ? (
        <p className="mt-1 text-right text-xs tabular-nums text-slate-500">{pct}%</p>
      ) : null}
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
