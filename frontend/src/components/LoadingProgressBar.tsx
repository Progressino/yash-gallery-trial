/** Slim indeterminate bar for route-level and long async work. */

import type { PlatformSummaryItem } from '../lib/intelligenceCache'

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

/** Estimated first Intelligence bundle build time (ms) by reporting window length. */
export function estimateIntelligenceBundleLoadMs(spanDays: number | null | undefined): number {
  const span = spanDays ?? 999
  if (span <= 1) return 12_000
  if (span <= 7) return 20_000
  if (span <= 45) return 45_000
  if (span <= 120) return 90_000
  return 150_000
}

function bundleLoadPhaseLabel(
  percent: number,
  opts: { dateStart: string; dateEnd: string; serverMessage?: string },
): string {
  if (opts.serverMessage?.trim()) return opts.serverMessage.trim()
  if (percent < 18) return `Checking uploads for ${opts.dateStart} → ${opts.dateEnd}…`
  if (percent < 45) return 'Loading Amazon, Myntra, Meesho, Flipkart, Snapdeal from saved data…'
  if (percent < 78) return 'Aggregating shipments and returns — first load can take up to a minute'
  return 'Finalizing charts and platform cards…'
}

/** Platform readiness checklist for Intelligence loading (replaces fake % progress). */
export function IntelligencePlatformChecklist({
  platforms,
  tier3Platforms,
  backgroundJobs,
  className = '',
}: {
  platforms: PlatformSummaryItem[]
  tier3Platforms?: string[]
  backgroundJobs?: string[]
  className?: string
}) {
  const specs: { key: string; label: string }[] = [
    { key: 'amazon', label: 'Amazon' },
    { key: 'myntra', label: 'Myntra' },
    { key: 'meesho', label: 'Meesho' },
    { key: 'flipkart', label: 'Flipkart' },
    { key: 'snapdeal', label: 'Snapdeal' },
  ]
  const tier3 = new Set((tier3Platforms ?? []).map(p => p.toLowerCase()))
  const busy = (backgroundJobs ?? []).length > 0

  return (
    <div className={`intel-platform-checklist ${className}`} role="status" aria-live="polite">
      <p className="intel-platform-checklist-title">Loading marketplace channels</p>
      <div className="intel-platform-checklist-row">
        {specs.map(({ key, label }) => {
          const card = platforms.find(p => p.platform.toLowerCase() === label.toLowerCase())
          const hasUnits = Boolean(card?.loaded && ((card?.total_units ?? 0) > 0 || (card?.total_returns ?? 0) > 0))
          const inTier3 = tier3.has(key)
          const state = hasUnits
            ? 'ready'
            : card?.loaded
              ? 'empty'
              : inTier3 || busy
                ? 'loading'
                : 'offline'
          return (
            <span key={key} className={`intel-platform-chip intel-platform-chip--${state}`}>
              {state === 'ready' ? '✓' : state === 'loading' ? '…' : '○'} {label}
            </span>
          )
        })}
      </div>
    </div>
  )
}

/** Full-width banner while the first Intelligence bundle is building on the server. */
export function IntelligenceBundleLoadPanel({
  active,
  percent,
  elapsedSec,
  dateStart,
  dateEnd,
  serverMessage,
  pollNote,
  className = '',
}: {
  active: boolean
  percent: number
  elapsedSec: number
  dateStart: string
  dateEnd: string
  serverMessage?: string
  /** Shown when polling after a quick empty/warming response. */
  pollNote?: string
  className?: string
}) {
  if (!active) return null
  const pct = Math.max(0, Math.min(99, Math.round(percent)))
  const detail = bundleLoadPhaseLabel(pct, { dateStart, dateEnd, serverMessage })
  return (
    <div
      className={`intel-bundle-load ${className}`}
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <div className="intel-bundle-load-head">
        <span className="intel-bundle-load-pulse" aria-hidden />
        <div>
          <p className="intel-bundle-load-title">Building your Intelligence dashboard</p>
          <p className="intel-bundle-load-detail">{detail}</p>
          {pollNote ? <p className="intel-bundle-load-poll">{pollNote}</p> : null}
        </div>
        <span className="intel-bundle-load-elapsed">{elapsedSec}s</span>
      </div>
      <DeterminateBar percent={pct} />
      <p className="intel-bundle-load-foot">
        Please keep this tab open — numbers will appear when the server finishes. Cached loads are much faster.
      </p>
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
