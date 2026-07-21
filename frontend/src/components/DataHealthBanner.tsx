import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getDataHealthChecks } from '../api/client'

/**
 * Global automated data-health banner. The backend re-runs its check suite
 * after every upload and on a schedule; this surfaces failures (duplicate
 * sales files, matrix vs upload drift, inventory double counts) without the
 * user having to spot them manually.
 */
export default function DataHealthBanner() {
  const [dismissedAt, setDismissedAt] = useState<string | null>(null)
  const q = useQuery({
    queryKey: ['data-health-checks'],
    queryFn: () => getDataHealthChecks(),
    refetchInterval: 5 * 60_000,
    staleTime: 4 * 60_000,
    retry: 1,
  })

  const report = q.data
  if (!report || report.status === 'running') return null

  const fails = report.checks.filter(c => !c.ok && c.severity === 'fail')
  const warns = report.checks.filter(c => !c.ok && c.severity === 'warn')
  if (fails.length === 0 && warns.length === 0) return null
  if (dismissedAt === report.generated_at && fails.length === 0) return null

  const isFail = fails.length > 0
  const items = isFail ? fails : warns
  const tone = isFail
    ? 'border-red-300 bg-red-50 text-red-950'
    : 'border-amber-300 bg-amber-50 text-amber-950'

  return (
    <div className={`rounded-xl border p-3 mb-4 text-sm ${tone}`}>
      <div className="flex items-start justify-between gap-3">
        <p className="font-semibold">
          {isFail
            ? `Automatic data check failed (${fails.length}) — fix before raising POs`
            : `Data warnings (${warns.length})`}
        </p>
        {!isFail && (
          <button
            type="button"
            className="text-xs underline shrink-0"
            onClick={() => setDismissedAt(report.generated_at ?? 'x')}
          >
            Dismiss
          </button>
        )}
      </div>
      <ul className="list-disc pl-5 mt-1 space-y-0.5 text-xs">
        {items.slice(0, 8).map(c => (
          <li key={c.id}>
            <span className="font-medium">{c.title}:</span> {c.detail}
          </li>
        ))}
        {items.length > 8 && <li>…and {items.length - 8} more</li>}
      </ul>
      {isFail && warns.length > 0 && (
        <p className="mt-1 text-xs opacity-80">Plus {warns.length} warning(s).</p>
      )}
    </div>
  )
}
