import { useRef } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { cacheHydrateWarm, getCoverage, invalidateDataQueries } from '../api/client'
import { coverageJobsRunning, coveragePollIntervalMs } from '../lib/coverageJobs'
import { operationalDataComplete } from '../lib/localSessionHint'
import { usePOFreshStore } from '../store/poFresh'
import { useSession } from '../store/session'

/** Single shared coverage poll — replaces per-page duplicate intervals. */
export default function CoverageProvider({
  enabled,
  children,
}: {
  enabled: boolean
  children: React.ReactNode
}) {
  const setCoverage = useSession(s => s.setCoverage)
  const lastHydrateAt = useRef(0)
  const prevSalesRevision = useRef<number | null>(null)
  const prevDailyIngest = useRef('idle')
  const prevSalesRebuild = useRef('idle')
  const qc = useQueryClient()

  useQuery({
    queryKey: ['coverage-poll'],
    queryFn: async () => {
      let c = await getCoverage({ light: true, timeout: 45_000 })
      const totallyEmpty =
        !c.mtr && !c.sales && !c.myntra && !c.meesho && !c.flipkart && !c.inventory
      if (
        !operationalDataComplete(c) &&
        !coverageJobsRunning(c) &&
        totallyEmpty &&
        Date.now() - lastHydrateAt.current > 15_000
      ) {
        lastHydrateAt.current = Date.now()
        try {
          await cacheHydrateWarm()
          c = await getCoverage({ light: true, timeout: 45_000 })
        } catch {
          /* server may be busy — next poll retries */
        }
      }
      setCoverage(c)

      const rev = c.sales_data_revision ?? 0
      const ingest = c.daily_auto_ingest_status ?? 'idle'
      const rebuild = c.sales_rebuild ?? 'idle'
      const ingestDone =
        prevDailyIngest.current === 'running' && ingest !== 'running' && ingest !== 'error'
      const rebuildDone =
        prevSalesRebuild.current === 'running' && rebuild !== 'running' && rebuild !== 'error'
      const revisionBumped =
        prevSalesRevision.current != null && rev > prevSalesRevision.current

      if (revisionBumped || ingestDone || rebuildDone) {
        invalidateDataQueries(qc)
        if (revisionBumped) {
          const po = usePOFreshStore.getState()
          if (po.result?.ok) {
            const note = 'Daily sales updated — recalculate PO for latest ADS.'
            const msg = po.result.message?.includes(note)
              ? po.result.message
              : (po.result.message ? `${po.result.message} ` : '') + note
            po.setResult({ ...po.result, message: msg })
            po.setFromSharedCache(false)
          }
        }
      }

      prevSalesRevision.current = rev
      prevDailyIngest.current = ingest
      prevSalesRebuild.current = rebuild
      return c
    },
    enabled,
    refetchInterval: q => coveragePollIntervalMs(q.state.data),
    retry: 2,
    staleTime: 5_000,
  })

  return <>{children}</>
}
