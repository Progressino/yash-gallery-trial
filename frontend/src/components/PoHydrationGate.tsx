/**
 * Blocks PO routes until session hydration reaches 8/8 coverage with full row counts.
 * POFresh / POEngine must not mount while warm-cache copy or sales rebuild is in flight.
 */
import { useEffect, useRef } from 'react'
import { Outlet } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { cacheHydrateWarm, getCoverage, type CoverageResponse } from '../api/client'
import {
  OPERATIONAL_DATA_TOTAL,
  operationalDataLoaded,
  poPageHydrationReady,
  PO_MIN_INVENTORY_ROWS,
  PO_MIN_SALES_ROWS,
} from '../lib/localSessionHint'
import { coverageJobsRunning } from '../lib/coverageJobs'
import { useSession } from '../store/session'
import { PageLoadingStripe } from './LoadingProgressBar'

function hydrationLabel(c: CoverageResponse | undefined): string {
  if (!c) return 'Connecting to server…'
  if (c.session_restore_status === 'running') {
    return c.session_restore_message || 'Restoring session from server…'
  }
  if (c.sales_rebuild === 'running') {
    return c.sales_rebuild_message || 'Rebuilding unified sales…'
  }
  if (coverageJobsRunning(c)) {
    return 'Syncing datasets into your session…'
  }
  const loaded = operationalDataLoaded(c)
  const sales = (c.sales_rows ?? 0).toLocaleString()
  const inv = (c.inventory_rows ?? 0).toLocaleString()
  if (loaded < OPERATIONAL_DATA_TOTAL) {
    return `Loading datasets (${loaded}/${OPERATIONAL_DATA_TOTAL}) — ${sales} sales · ${inv} inventory SKUs`
  }
  if ((c.sales_rows ?? 0) < PO_MIN_SALES_ROWS) {
    return `Waiting for full sales history (${sales} / ${PO_MIN_SALES_ROWS.toLocaleString()}+ rows)…`
  }
  if ((c.inventory_rows ?? 0) < PO_MIN_INVENTORY_ROWS) {
    return `Waiting for inventory (${inv} / ${PO_MIN_INVENTORY_ROWS.toLocaleString()}+ SKUs)…`
  }
  return 'Finalizing session data…'
}

function LoadingDataScreen({ coverage }: { coverage: CoverageResponse | undefined }) {
  const loaded = coverage ? operationalDataLoaded(coverage) : 0
  const pct =
    coverage && OPERATIONAL_DATA_TOTAL > 0
      ? Math.min(95, Math.round((loaded / OPERATIONAL_DATA_TOTAL) * 70))
      : null
  const restorePct =
    coverage?.session_restore_status === 'running'
      ? Math.max(pct ?? 0, coverage.session_restore_progress ?? 0)
      : pct

  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center px-4 py-12">
      <div className="w-full max-w-md space-y-4">
        <div className="text-center space-y-1">
          <h1 className="text-lg font-semibold text-[#002B5B]">Loading PO data</h1>
          <p className="text-sm text-slate-600">
            Hydrating sales, inventory, and platform history before the PO engine opens.
          </p>
        </div>
        <PageLoadingStripe active label={hydrationLabel(coverage)} percent={restorePct} />
        {coverage ? (
          <dl className="grid grid-cols-2 gap-2 text-xs text-slate-600">
            <div className="rounded-lg border border-slate-200 bg-white px-3 py-2">
              <dt className="text-slate-400">Coverage</dt>
              <dd className="font-semibold tabular-nums">
                {loaded}/{OPERATIONAL_DATA_TOTAL} datasets
              </dd>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white px-3 py-2">
              <dt className="text-slate-400">Sales rows</dt>
              <dd className="font-semibold tabular-nums">{(coverage.sales_rows ?? 0).toLocaleString()}</dd>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 col-span-2">
              <dt className="text-slate-400">Inventory SKUs</dt>
              <dd className="font-semibold tabular-nums">{(coverage.inventory_rows ?? 0).toLocaleString()}</dd>
            </div>
          </dl>
        ) : null}
      </div>
    </div>
  )
}

export default function PoHydrationGate() {
  const setCoverage = useSession(s => s.setCoverage)
  const sessionCoverage = useSession()
  const hydrateRequested = useRef(false)

  const { data: polled } = useQuery({
    queryKey: ['po-hydration-gate'],
    queryFn: async () => {
      const c = await getCoverage({ light: true, timeout: 45_000 })
      setCoverage(c)
      return c
    },
    refetchInterval: q => {
      const c = q.state.data ?? sessionCoverage
      return poPageHydrationReady(c) ? false : 2_000
    },
    staleTime: 0,
  })

  const coverage = polled ?? sessionCoverage
  const ready = poPageHydrationReady(coverage)

  useEffect(() => {
    if (ready || hydrateRequested.current) return
    if (coverageJobsRunning(coverage)) return
    hydrateRequested.current = true
    void cacheHydrateWarm()
      .then(() => getCoverage({ light: true, timeout: 45_000 }))
      .then(c => setCoverage(c))
      .catch(() => {
        hydrateRequested.current = false
      })
  }, [ready, coverage, setCoverage])

  if (!ready) {
    return <LoadingDataScreen coverage={coverage} />
  }

  return <Outlet />
}
