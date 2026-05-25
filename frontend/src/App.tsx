import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import axios from 'axios'
import Layout from './components/Layout'
import KarigarLayout from './components/KarigarLayout'
import { KarigarGate, StaffGate, ModuleAccessGate } from './components/RouteGuards'
import { isHrmOnlyUser } from './store/auth'
import Login from './pages/Login'
import api, { cacheHydrateWarm, cacheLoad, getCoverage, invalidateDataQueries } from './api/client'
import { useSession } from './store/session'
import { useAuth, isKarigarUser, type AuthUser } from './store/auth'

const Dashboard   = lazy(() => import('./pages/Dashboard'))
const Upload      = lazy(() => import('./pages/Upload'))
const MTR         = lazy(() => import('./pages/MTR'))
const Myntra      = lazy(() => import('./pages/Myntra'))
const Meesho      = lazy(() => import('./pages/Meesho'))
const Flipkart    = lazy(() => import('./pages/Flipkart'))
const Inventory   = lazy(() => import('./pages/Inventory'))
const POEngine    = lazy(() => import('./pages/POEngine'))
const Forecast    = lazy(() => import('./pages/Forecast'))
const Finance     = lazy(() => import('./pages/Finance'))
const ItemMaster  = lazy(() => import('./pages/ItemMaster'))
const Snapdeal    = lazy(() => import('./pages/Snapdeal'))
const SalesOrders = lazy(() => import('./pages/SalesOrders'))
const Purchase    = lazy(() => import('./pages/Purchase'))
const TNA         = lazy(() => import('./pages/TNA'))
const Production  = lazy(() => import('./pages/Production'))
const StitchingCosting = lazy(() => import('./pages/StitchingCosting'))
const GreyFabric  = lazy(() => import('./pages/GreyFabric'))
const HRM         = lazy(() => import('./pages/HRM'))
const Admin                  = lazy(() => import('./pages/Admin'))
const MarketplaceConnections = lazy(() => import('./pages/MarketplaceConnections'))
const SKUDeepDive            = lazy(() => import('./pages/SKUDeepDive'))

const qc = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
    },
  },
})

async function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  let t: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race<T>([
      p,
      new Promise<T>((_, reject) => {
        t = setTimeout(() => reject(new Error('Auto-restore timeout')), ms)
      }),
    ])
  } finally {
    if (t) clearTimeout(t)
  }
}

function ProtectedRoute() {
  const setCoverage = useSession(s => s.setCoverage)
  const invUploadRunning = useSession(s => s.inventory_upload_status === 'running')
  const setUser = useAuth(s => s.setUser)
  const cachedUser = useAuth(s => s.user)

  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ['auth-me'],
    queryFn: async () => {
      const { data: me } = await api.get<AuthUser>('/auth/me', { timeout: 60_000 })
      setUser(me)
      return me
    },
    initialData: cachedUser ?? undefined,
    initialDataUpdatedAt: cachedUser ? Date.now() : undefined,
    retry: (failureCount, err) => {
      if (axios.isAxiosError(err) && err.response?.status === 401) return false
      return failureCount < 3
    },
    refetchOnWindowFocus: false,
    staleTime: 30 * 60 * 1000,
  })

  const authUnauthorized =
    axios.isAxiosError(error) && error.response?.status === 401

  const authUnreachable =
    !!error &&
    !authUnauthorized &&
    axios.isAxiosError(error) &&
    (!error.response ||
      error.code === 'ECONNABORTED' ||
      [502, 503, 504].includes(error.response?.status ?? 0))

  /** Keep last known profile when the server is busy (upload/restart) but cookie may still be valid. */
  const activeUser = data ?? (authUnreachable ? cachedUser : null) ?? cachedUser
  const isKarigar = isKarigarUser(activeUser)
  const hrmOnly = isHrmOnlyUser(activeUser)

  const coverageEmpty = (c: Awaited<ReturnType<typeof getCoverage>>) =>
    !c.mtr && !c.sales && !c.myntra && !c.meesho && !c.flipkart && !c.snapdeal

  /** Server has platform history but unified sales not built yet — dashboard stays blank until fixed. */
  const sessionNeedsSales = (c: Awaited<ReturnType<typeof getCoverage>>) =>
    !c.sales &&
    (c.mtr || c.myntra || c.meesho || c.flipkart || c.snapdeal) &&
    (c.sales_rebuild === 'running' || c.daily_auto_ingest_status === 'running')

  const sessionNeedsSync = (c: Awaited<ReturnType<typeof getCoverage>>) =>
    coverageEmpty(c) || sessionNeedsSales(c)

  const backgroundJobRunning = (c: Awaited<ReturnType<typeof getCoverage>>) =>
    c.inventory_upload_status === 'running' ||
    c.daily_inventory_upload_status === 'running' ||
    c.daily_auto_ingest_status === 'running' ||
    c.sales_rebuild === 'running'

  const { isPending: isRestoring } = useQuery({
    queryKey: ['session-auto-restore'],
    queryFn: async () => {
      try {
        let coverage = await getCoverage({ light: true, timeout: 45_000 })
        setCoverage(coverage)
        if (coverage.inventory_upload_status === 'running') {
          return true
        }
        const inventoryOnly =
          coverage.inventory && coverageEmpty(coverage) && !sessionNeedsSales(coverage)
        if (inventoryOnly) {
          if (!sessionNeedsSync(coverage)) invalidateDataQueries(qc)
          return true
        }
        // Sales loaded from PG but inventory only in warm cache — coverage already restores it.
        if (!coverage.inventory && !coverageEmpty(coverage)) {
          try {
            coverage = await getCoverage({ light: true, timeout: 60_000 })
            setCoverage(coverage)
            if (coverage.inventory && !sessionNeedsSync(coverage)) {
              invalidateDataQueries(qc)
            }
          } catch {
            /* warm cache may still be loading after deploy */
          }
        }
        const hasAnyPlatform =
          coverage.mtr || coverage.myntra || coverage.meesho || coverage.flipkart || coverage.snapdeal
        if (coverageEmpty(coverage) && !hasAnyPlatform) {
          try {
            await withTimeout(cacheHydrateWarm(), 90_000)
            coverage = await getCoverage({ light: true, timeout: 60_000 })
            setCoverage(coverage)
            if (coverage.inventory_upload_status === 'running') return true
          } catch {
            /* warm cache may still be starting after deploy */
          }
        }
        if (coverageEmpty(coverage) && !hasAnyPlatform) {
          try {
            await withTimeout(cacheLoad(), 90_000)
            coverage = await getCoverage({ timeout: 90_000 })
            setCoverage(coverage)
          } catch {
            /* GitHub cache optional */
          }
        } else if (sessionNeedsSales(coverage)) {
          coverage = await getCoverage({ timeout: 90_000 })
          setCoverage(coverage)
        }
        if (!sessionNeedsSync(coverage)) {
          invalidateDataQueries(qc)
        }
      } catch {
        /* server busy during upload — coverage polling will retry */
      }
      return true
    },
    enabled: !!activeUser && !isKarigar && !hrmOnly,
    retry: 3,
    retryDelay: 8_000,
    staleTime: Infinity,
  })

  useQuery({
    queryKey: ['coverage-empty-retry'],
    queryFn: async () => {
      const prev = qc.getQueryData<Awaited<ReturnType<typeof getCoverage>>>(['coverage-empty-retry'])
      const useLight = !!prev && !coverageEmpty(prev)
      const c = await getCoverage({ timeout: useLight ? 45_000 : 120_000, light: useLight })
      setCoverage(c)
      if (c.sales) invalidateDataQueries(qc)
      return c
    },
    enabled: !!activeUser && !isKarigar && !hrmOnly && !isRestoring,
    refetchInterval: (q) => {
      const c = q.state.data
      if (!c) return 8_000
      if (backgroundJobRunning(c)) return 3_000
      return sessionNeedsSync(c) ? 15_000 : false
    },
    retry: 2,
  })

  if (isLoading && !cachedUser) {
    return (
      <div className="min-h-screen flex items-center justify-center text-gray-400 text-sm">
        Loading…
      </div>
    )
  }
  if (authUnauthorized) return <Navigate to="/login" replace />
  if (!activeUser) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-3 p-6 text-center">
        <p className="text-gray-600 text-sm max-w-md">
          Could not reach the server right now. The app may be restarting or a large upload is running.
          Wait a minute, then sign in again.
        </p>
        <div className="flex flex-wrap gap-2 justify-center">
          <button
            type="button"
            onClick={() => void refetch()}
            disabled={isFetching}
            className="px-4 py-2 rounded-lg bg-[#002B5B] text-white text-sm font-medium disabled:opacity-50"
          >
            {isFetching ? 'Retrying…' : 'Retry'}
          </button>
          <a
            href="/login"
            className="px-4 py-2 rounded-lg border border-gray-300 text-gray-700 text-sm font-medium hover:bg-gray-50"
          >
            Sign in
          </a>
        </div>
      </div>
    )
  }
  return (
    <>
      {authUnreachable && activeUser && (
        <div className="fixed top-0 left-0 right-0 z-[9999] flex items-center justify-center gap-2 bg-amber-600 text-white text-xs py-1.5 px-3 shadow-md">
          Server is slow to respond — using your saved sign-in. Data sync may catch up in a moment.
          <button
            type="button"
            className="underline font-semibold"
            onClick={() => void refetch()}
          >
            Retry
          </button>
        </div>
      )}
      {isRestoring && !isKarigar && !invUploadRunning && (
        <div className="fixed top-0 left-0 right-0 z-[9999] flex items-center justify-center gap-2 bg-[#002B5B] text-white text-xs py-1.5 shadow-md">
          <svg className="animate-spin h-3 w-3 shrink-0" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          Syncing your data…
        </div>
      )}
      <Suspense fallback={<div className="min-h-screen flex items-center justify-center text-gray-400 text-sm">Loading…</div>}>
        <Outlet />
      </Suspense>
    </>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route element={<ProtectedRoute />}>
            <Route element={<KarigarGate />}>
              <Route element={<KarigarLayout />}>
                <Route path="/production-entry" element={<StitchingCosting karigarOnly />} />
                <Route path="*" element={<Navigate to="/production-entry" replace />} />
              </Route>
            </Route>
            <Route element={<StaffGate />}>
            <Route path="/" element={<Layout />}>
              <Route element={<ModuleAccessGate />}>
              <Route index element={<Dashboard />} />
              <Route path="upload"    element={<Upload />} />
              <Route path="mtr"       element={<MTR />} />
              <Route path="myntra"    element={<Myntra />} />
              <Route path="meesho"    element={<Meesho />} />
              <Route path="flipkart"  element={<Flipkart />} />
              <Route path="inventory" element={<Inventory />} />
              <Route path="po" element={<POEngine />} />
              <Route path="po-dashboard" element={<Navigate to="/po?tab=dashboard" replace />} />
              <Route path="forecast"  element={<Forecast />} />
              <Route path="finance"   element={<Finance />} />
              <Route path="items"      element={<ItemMaster />} />
              <Route path="snapdeal"  element={<Snapdeal />} />
              <Route path="sales"     element={<SalesOrders />} />
              <Route path="purchase"  element={<Purchase />} />
              <Route path="tna"       element={<TNA />} />
              <Route path="production" element={<Production />} />
              <Route path="stitching-costing" element={<StitchingCosting />} />
              <Route path="grey"      element={<GreyFabric />} />
              <Route path="hrm"       element={<HRM />} />
              <Route path="admin"       element={<Admin />} />
              <Route path="marketplace-connections" element={<MarketplaceConnections />} />
              <Route path="sku-deepdive" element={<SKUDeepDive />} />
              </Route>
            </Route>
            </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
