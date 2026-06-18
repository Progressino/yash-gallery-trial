import { Component, lazy, Suspense, type ReactNode } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import axios from 'axios'
import Layout from './components/Layout'
import KarigarLayout from './components/KarigarLayout'
import { KarigarGate, StaffGate, ModuleAccessGate } from './components/RouteGuards'
import { isHrmOnlyUser } from './store/auth'
import Login from './pages/Login'
import api, { cacheHydrateWarm, cacheLoad, getCoverage, invalidateDataQueries, waitForWarmCacheReady } from './api/client'
import CoverageProvider from './components/CoverageProvider'
import { canSkipHeavyServerRestore, poOperationalReady, poOperationalLoaded, PO_OPERATIONAL_TOTAL, readLocalSessionHint } from './lib/localSessionHint'
import { coverageJobsRunning, coverageNeedsSync } from './lib/coverageJobs'
import { useSession } from './store/session'
import { useAuth, isKarigarUser, type AuthUser } from './store/auth'
const Dashboard   = lazy(() => import('./pages/Dashboard'))
const POFresh     = lazy(() => import('./pages/POFresh'))
const PO2         = lazy(() => import('./pages/PO2'))
const POEngine    = lazy(() => import('./pages/POEngine'))
const Upload      = lazy(() => import('./pages/Upload'))
const MTR         = lazy(() => import('./pages/MTR'))
const Myntra      = lazy(() => import('./pages/Myntra'))
const Meesho      = lazy(() => import('./pages/Meesho'))
const Flipkart    = lazy(() => import('./pages/Flipkart'))
const Inventory   = lazy(() => import('./pages/Inventory'))
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

class AppErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean; message: string }> {
  constructor(props: { children: ReactNode }) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  static getDerivedStateFromError(error: unknown) {
    return {
      hasError: true,
      message: error instanceof Error ? error.message : String(error ?? 'unknown-client-error'),
    }
  }

  componentDidCatch(error: unknown, info: unknown) {
    console.error('[AppErrorBoundary]', error, info)
  }

  render() {
    if (!this.state.hasError) return this.props.children
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 p-6">
        <div className="max-w-xl w-full rounded-xl border border-red-200 bg-white shadow-sm p-5 space-y-3">
          <h1 className="text-lg font-semibold text-red-700">Page failed to load</h1>
          <p className="text-sm text-slate-700">
            The app hit an unexpected UI error. Please reload once. If this repeats, share this error text.
          </p>
          <p className="text-xs font-mono rounded bg-slate-100 border border-slate-200 px-2 py-1 text-slate-700 break-all">
            {this.state.message || 'unknown-client-error'}
          </p>
          <button
            type="button"
            className="px-4 py-2 rounded-lg bg-[#002B5B] text-white text-sm font-semibold hover:bg-blue-800"
            onClick={() => window.location.reload()}
          >
            Reload app
          </button>
        </div>
      </div>
    )
  }
}

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

  const { data, isLoading, error, refetch } = useQuery({
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

  const { isPending: isRestoring } = useQuery({
    queryKey: ['session-auto-restore'],
    queryFn: async () => {
      try {
        return await withTimeout((async () => {
        let coverage = await getCoverage({ light: true, timeout: 45_000 })
        setCoverage(coverage)
        const localHint = readLocalSessionHint()
        if (canSkipHeavyServerRestore(coverage, localHint)) {
          if (!coverageNeedsSync(coverage)) invalidateDataQueries(qc)
          return true
        }
        if (poOperationalReady(coverage)) {
          invalidateDataQueries(qc)
          return true
        }
        if (coverageJobsRunning(coverage)) {
          return true
        }
        const hasAnyPlatform =
          coverage.mtr || coverage.myntra || coverage.meesho || coverage.flipkart || coverage.snapdeal
        if (!poOperationalReady(coverage)) {
          try {
            await waitForWarmCacheReady({ maxWaitMs: 45_000, pollMs: 2_000 })
            await withTimeout(cacheHydrateWarm(), 30_000)
            coverage = await getCoverage({ light: true, timeout: 45_000 })
            setCoverage(coverage)
            if (poOperationalReady(coverage)) {
              invalidateDataQueries(qc)
              return true
            }
          } catch {
            /* warm cache may still be starting after deploy */
          }
        }
        if (coverageEmpty(coverage) && !hasAnyPlatform) {
          try {
            await withTimeout(cacheLoad(), 90_000)
            coverage = await getCoverage({ light: true, timeout: 45_000 })
            setCoverage(coverage)
          } catch {
            /* GitHub cache optional */
          }
        }
        if (!coverageNeedsSync(coverage)) {
          invalidateDataQueries(qc)
        }
        return true
        })(), 90_000)
      } catch {
        /* server busy during upload — coverage polling will retry */
      }
      return true
    },
    enabled: !!activeUser && !isKarigar && !hrmOnly,
    retry: 1,
    retryDelay: 5_000,
    staleTime: Infinity,
  })

  const pollCoverage = !!activeUser && !isKarigar && !hrmOnly
  const dataStillLoading = useSession(s => !poOperationalReady(s))
  const dataLoadLoaded = useSession(s => poOperationalLoaded(s))
  const dataLoadTotal = PO_OPERATIONAL_TOTAL

  if (isLoading && !cachedUser) {
    return (
      <div className="min-h-screen flex items-center justify-center text-gray-400 text-sm">
        Loading…
      </div>
    )
  }
  if (authUnauthorized) return <Navigate to="/login" replace />
  if (!activeUser && !isLoading) {
    return <Navigate to="/login" replace state={{ serverUnreachable: authUnreachable }} />
  }
  return (
    <CoverageProvider enabled={pollCoverage}>
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
      {!isRestoring && dataStillLoading && !isKarigar && !invUploadRunning && (
        <div className="fixed top-0 left-0 right-0 z-[9999] flex items-center justify-center gap-2 bg-[#002B5B] text-white text-xs py-1.5 shadow-md">
          <svg className="animate-spin h-3 w-3 shrink-0" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          Loading all datasets from server ({dataLoadLoaded}/{dataLoadTotal})…
        </div>
      )}
      <Suspense fallback={<div className="min-h-screen flex items-center justify-center text-gray-400 text-sm">Loading…</div>}>
        <Outlet />
      </Suspense>
    </CoverageProvider>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <AppErrorBoundary>
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
              <Route path="po-fresh" element={<POFresh />} />
              <Route path="po2" element={<PO2 />} />
              <Route path="po" element={<Navigate to="/po-fresh" replace />} />
              <Route path="po-legacy" element={<POEngine />} />
              <Route path="po-dashboard" element={<Navigate to="/po-fresh" replace />} />
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
      </AppErrorBoundary>
    </QueryClientProvider>
  )
}
