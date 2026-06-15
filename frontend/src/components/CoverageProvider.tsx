import { useQuery } from '@tanstack/react-query'
import { getCoverage } from '../api/client'
import { coveragePollIntervalMs } from '../lib/coverageJobs'
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

  useQuery({
    queryKey: ['coverage-poll'],
    queryFn: async () => {
      const c = await getCoverage({ light: true, timeout: 45_000 })
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
