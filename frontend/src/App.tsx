import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import Layout from './components/Layout'
import Login from './pages/Login'
import api from './api/client'

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
const Admin       = lazy(() => import('./pages/Admin'))

const qc = new QueryClient()

function ProtectedRoute() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['auth-me'],
    queryFn: async () => {
      const { data } = await api.get('/auth/me')
      return data
    },
    retry: false,
    staleTime: 5 * 60 * 1000,
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
    <Suspense fallback={<div className="min-h-screen flex items-center justify-center text-gray-400 text-sm">Loading…</div>}>
      <Outlet />
    </Suspense>
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
              <Route path="admin"     element={<Admin />} />
            </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
