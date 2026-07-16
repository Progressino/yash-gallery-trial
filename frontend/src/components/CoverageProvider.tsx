import { useRef } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { cacheHydrateWarm, getCoverage, invalidateDataQueries } from '../api/client'
import { coverageJobsRunning, coveragePollIntervalMs } from '../lib/coverageJobs'
import { operationalDataComplete } from '../lib/localSessionHint'
import { usePOFreshStore } from '../store/poFresh'
import { useSession } from '../store/session'

function poRecalcNote(existing: string | undefined, note: string): string {
  return existing?.includes(note) ? existing : (existing ? `${existing} ` : '') + note
}

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
  const prevInventoryRevision = useRef<number | null>(null)
  const prevInvUploadedAt = useRef<string | null>(null)
  const prevDailyIngest = useRef('idle')
  const prevSalesRebuild = useRef('idle')
  const prevInvUpload = useRef('idle')
  const prevDailyInvUpload = useRef('idle')
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
      const invRev = c.inventory_data_revision ?? 0
      const invUploadedAt = c.inventory_snapshot_uploaded_at ?? ''
      const ingest = c.daily_auto_ingest_status ?? 'idle'
      const rebuild = c.sales_rebuild ?? 'idle'
      const invUpload = c.inventory_upload_status ?? 'idle'
      const dailyInvUpload = c.daily_inventory_upload_status ?? 'idle'
      const ingestDone =
        prevDailyIngest.current === 'running' && ingest !== 'running' && ingest !== 'error'
      const rebuildDone =
        prevSalesRebuild.current === 'running' && rebuild !== 'running' && rebuild !== 'error'
      const revisionBumped =
        prevSalesRevision.current != null && rev > prevSalesRevision.current
      const invRevisionBumped =
        prevInventoryRevision.current != null && invRev > prevInventoryRevision.current
      // Revision is often 0 after hydrate (not always persisted historically) — also
      // watch uploaded_at so other tabs/sessions refresh when a new snapshot lands.
      const invSnapshotChanged =
        prevInvUploadedAt.current != null &&
        Boolean(invUploadedAt) &&
        invUploadedAt !== prevInvUploadedAt.current
      const invUploadDone =
        prevInvUpload.current === 'running' &&
        invUpload !== 'running' &&
        invUpload !== 'error'
      const dailyInvUploadDone =
        prevDailyInvUpload.current === 'running' &&
        dailyInvUpload !== 'running' &&
        dailyInvUpload !== 'error'

      if (
        revisionBumped ||
        ingestDone ||
        rebuildDone ||
        invRevisionBumped ||
        invSnapshotChanged ||
        invUploadDone ||
        dailyInvUploadDone
      ) {
        invalidateDataQueries(qc)
        const po = usePOFreshStore.getState()
        if (po.result?.ok) {
          let msg = po.result.message
          if (revisionBumped || ingestDone || rebuildDone) {
            msg = poRecalcNote(msg, 'Daily sales updated — recalculate PO for latest ADS.')
          }
          if (invRevisionBumped || invSnapshotChanged || invUploadDone || dailyInvUploadDone) {
            msg = poRecalcNote(msg, 'Inventory updated — recalculate PO for latest stock.')
          }
          if (msg !== po.result.message) {
            po.setResult({ ...po.result, message: msg })
            po.setFromSharedCache(false)
          }
        }
      }

      prevSalesRevision.current = rev
      prevInventoryRevision.current = invRev
      prevInvUploadedAt.current = invUploadedAt
      prevDailyIngest.current = ingest
      prevSalesRebuild.current = rebuild
      prevInvUpload.current = invUpload
      prevDailyInvUpload.current = dailyInvUpload
      return c
    },
    enabled,
    refetchInterval: q => coveragePollIntervalMs(q.state.data),
    retry: 2,
    staleTime: 5_000,
  })

  return <>{children}</>
}
