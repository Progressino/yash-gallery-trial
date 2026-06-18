import { useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { cacheHydrateWarm, getCoverage } from '../api/client'
import { coverageJobsRunning, coveragePollIntervalMs } from '../lib/coverageJobs'
import { operationalDataComplete } from '../lib/localSessionHint'
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
        Date.now() - lastHydrateAt.current > 60_000
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
      return c
    },
    enabled,
    refetchInterval: q => coveragePollIntervalMs(q.state.data),
    retry: 2,
    staleTime: 5_000,
  })

  return <>{children}</>
}
