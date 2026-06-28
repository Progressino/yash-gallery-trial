/**
 * Blocks PO routes until GET /api/po/readiness reports po_ready.
 * Non-critical jobs (e.g. sales_rebuild) do not block the PO engine.
 */
import { useEffect, useRef } from 'react'
import { Outlet } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { cacheHydrateWarm, getPoReadiness, type PoReadinessResponse } from '../api/client'
import { PageLoadingStripe } from './LoadingProgressBar'
import { useSession } from '../store/session'

function hydrationLabel(r: PoReadinessResponse | undefined): string {
  if (!r) return 'Connecting to server…'
  if (r.critical_restore_running) {
    return 'Restoring session or inventory…'
  }
  if (r.hydration === 'inflight') {
    return 'Syncing datasets into your session…'
  }
  const sales = (r.sales_rows ?? 0).toLocaleString()
  const inv = (r.inventory_rows ?? 0).toLocaleString()
  if (!r.data_ready) {
    return `Loading PO data — ${sales} sales · ${inv} inventory SKUs…`
  }
  if (r.background_jobs?.length) {
    return `PO data ready (${r.background_jobs.join(', ')} running in background)…`
  }
  return 'Opening PO engine…'
}

function LoadingDataScreen({ readiness }: { readiness: PoReadinessResponse | undefined }) {
  const pct = readiness?.data_ready ? 85 : readiness ? 40 : null

  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center px-4 py-12">
      <div className="w-full max-w-md space-y-4">
        <div className="text-center space-y-1">
          <h1 className="text-lg font-semibold text-[#002B5B]">Loading PO data</h1>
          <p className="text-sm text-slate-600">
            Waiting for sales and inventory required by the PO engine.
          </p>
        </div>
        <PageLoadingStripe active label={hydrationLabel(readiness)} percent={pct} />
        {readiness ? (
          <dl className="grid grid-cols-2 gap-2 text-xs text-slate-600">
            <div className="rounded-lg border border-slate-200 bg-white px-3 py-2">
              <dt className="text-slate-400">PO ready</dt>
              <dd className="font-semibold">{readiness.po_ready ? 'Yes' : 'No'}</dd>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white px-3 py-2">
              <dt className="text-slate-400">Data source</dt>
              <dd className="font-semibold">{readiness.data_source}</dd>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white px-3 py-2">
              <dt className="text-slate-400">Sales rows</dt>
              <dd className="font-semibold tabular-nums">{readiness.sales_rows.toLocaleString()}</dd>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white px-3 py-2">
              <dt className="text-slate-400">Inventory SKUs</dt>
              <dd className="font-semibold tabular-nums">{readiness.inventory_rows.toLocaleString()}</dd>
            </div>
            {readiness.background_jobs?.length ? (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 col-span-2">
                <dt className="text-amber-700">Background (non-blocking)</dt>
                <dd className="font-medium text-amber-900">{readiness.background_jobs.join(', ')}</dd>
              </div>
            ) : null}
          </dl>
        ) : null}
      </div>
    </div>
  )
}

export default function PoHydrationGate() {
  const hydrateRequested = useRef(false)
  const coveragePoReady = useSession(s => s.po_ready === true)

  const { data: readiness } = useQuery({
    queryKey: ['po-readiness-gate'],
    queryFn: () => getPoReadiness({ timeout: 20_000 }),
    refetchInterval: q => (q.state.data?.po_ready || coveragePoReady ? false : 2_000),
    staleTime: 0,
  })

  const ready = readiness?.po_ready === true || coveragePoReady

  useEffect(() => {
    if (hydrateRequested.current) return
    if (!ready) {
      if (readiness?.critical_restore_running) return
      return
    }
    hydrateRequested.current = true
    void cacheHydrateWarm().catch(() => {
      hydrateRequested.current = false
    })
  }, [ready, readiness])

  if (!ready) {
    return <LoadingDataScreen readiness={readiness} />
  }

  return <Outlet />
}
