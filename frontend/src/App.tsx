import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import MTR from './pages/MTR'
import Myntra from './pages/Myntra'
import Meesho from './pages/Meesho'
import Flipkart from './pages/Flipkart'
import Inventory from './pages/Inventory'
import POEngine from './pages/POEngine'
import Forecast from './pages/Forecast'
import Finance from './pages/Finance'
import ItemMaster from './pages/ItemMaster'
import Login from './pages/Login'
import api from './api/client'

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
  return <Outlet />
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
              <Route path="items"     element={<ItemMaster />} />
            </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
