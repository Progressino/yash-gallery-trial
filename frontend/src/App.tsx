import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import Layout from './components/Layout'
import Login from './pages/Login'
import api, { cacheLoad, getCoverage } from './api/client'
import { useSession } from './store/session'

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
const GreyFabric  = lazy(() => import('./pages/GreyFabric'))
const Admin                  = lazy(() => import('./pages/Admin'))
const MarketplaceConnections = lazy(() => import('./pages/MarketplaceConnections'))
const SKUDeepDive            = lazy(() => import('./pages/SKUDeepDive'))

const qc = new QueryClient()

function ProtectedRoute() {
  const setCoverage = useSession(s => s.setCoverage)

  const { data, isLoading, isError } = useQuery({
    queryKey: ['auth-me'],
    queryFn: async () => {
      const { data } = await api.get('/auth/me')
      return data
    },
    retry: false,
    staleTime: 5 * 60 * 1000,
  })

  // Auto-restore session data after server restart or session expiry.
  // Runs in the background — the app renders immediately and shows a banner
  // while restoring so the user isn't stuck on the login screen.
  const { isFetching: isRestoring } = useQuery({
    queryKey: ['session-auto-restore'],
    queryFn: async () => {
      const coverage = await getCoverage()
      if (!coverage.mtr && !coverage.sales && !coverage.pause_auto_data_restore) {
        await cacheLoad()
        const refreshed = await getCoverage()
        setCoverage(refreshed)
      } else {
        setCoverage(coverage)
      }
      return true
    },
    enabled: !!data,   // only run once authenticated
    retry: false,
    staleTime: Infinity,  // run once per app load, not on every navigation
  })

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-gray-400 text-sm">
        Loading…
      </div>
    )
  }
  if (isError || !data) return <Navigate to="/login" replace />
  return (
    <>
      {isRestoring && (
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
            <Route path="/" element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="upload"    element={<Upload />} />
              <Route path="mtr"       element={<MTR />} />
              <Route path="myntra"    element={<Myntra />} />
              <Route path="meesho"    element={<Meesho />} />
              <Route path="flipkart"  element={<Flipkart />} />
              <Route path="inventory" element={<Inventory />} />
              <Route path="po"        element={<POEngine />} />
              <Route path="forecast"  element={<Forecast />} />
              <Route path="finance"   element={<Finance />} />
              <Route path="items"      element={<ItemMaster />} />
              <Route path="snapdeal"  element={<Snapdeal />} />
              <Route path="sales"     element={<SalesOrders />} />
              <Route path="purchase"  element={<Purchase />} />
              <Route path="tna"       element={<TNA />} />
              <Route path="production" element={<Production />} />
              <Route path="grey"      element={<GreyFabric />} />
              <Route path="admin"       element={<Admin />} />
              <Route path="marketplace-connections" element={<MarketplaceConnections />} />
              <Route path="sku-deepdive" element={<SKUDeepDive />} />
            </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
