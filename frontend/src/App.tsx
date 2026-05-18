import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import axios from 'axios'
import Layout from './components/Layout'
import KarigarLayout from './components/KarigarLayout'
import { KarigarGate, StaffGate } from './components/RouteGuards'
import Login from './pages/Login'
import api, { cacheLoad, getCoverage } from './api/client'
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

const AUTO_RESTORE_TIMEOUT_MS = 120_000

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
    axios.isAxiosError(error) && error.response?.status === 401 && !cachedUser

  const activeUser = data ?? cachedUser
  const isKarigar = isKarigarUser(activeUser)

  const coverageEmpty = (c: Awaited<ReturnType<typeof getCoverage>>) =>
    !c.mtr && !c.sales && !c.myntra && !c.meesho && !c.flipkart && !c.snapdeal

  const { isFetching: isRestoring } = useQuery({
    queryKey: ['session-auto-restore'],
    queryFn: async () => {
      try {
        let coverage = await getCoverage({ timeout: 90_000 })
        setCoverage(coverage)
        if (coverageEmpty(coverage)) {
          try {
            await withTimeout(cacheLoad(), AUTO_RESTORE_TIMEOUT_MS)
            coverage = await getCoverage({ timeout: 90_000 })
            setCoverage(coverage)
          } catch {
            /* warm cache may still be loading on server */
          }
        }
      } catch {
        /* server busy during upload — coverage polling on Upload page will retry */
      }
      return true
    },
    enabled: !!activeUser && !isKarigar,
    retry: 3,
    retryDelay: 8_000,
    staleTime: Infinity,
  })

  useQuery({
    queryKey: ['coverage-empty-retry'],
    queryFn: async () => {
      const c = await getCoverage({ timeout: 90_000 })
      setCoverage(c)
      return c
    },
    enabled: !!activeUser && !isKarigar && !isRestoring,
    refetchInterval: (q) => {
      const c = q.state.data
      if (!c) return 8_000
      return coverageEmpty(c) ? 8_000 : false
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
          Could not reach the server right now. If a large upload is running, wait and retry.
        </p>
        <button
          type="button"
          onClick={() => void refetch()}
          disabled={isFetching}
          className="px-4 py-2 rounded-lg bg-[#002B5B] text-white text-sm font-medium disabled:opacity-50"
        >
          {isFetching ? 'Retrying…' : 'Retry'}
        </button>
      </div>
    )
  }
  return (
    <>
      {isRestoring && !isKarigar && (
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
              <Route path="admin"       element={<Admin />} />
              <Route path="marketplace-connections" element={<MarketplaceConnections />} />
              <Route path="sku-deepdive" element={<SKUDeepDive />} />
            </Route>
            </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
