import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import MTR from './pages/MTR'
import Myntra from './pages/Myntra'
import Meesho from './pages/Meesho'
import Flipkart from './pages/Flipkart'
import Inventory from './pages/Inventory'
import POEngine from './pages/POEngine'
import Forecast from './pages/Forecast'

const qc = new QueryClient()

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="mtr"       element={<MTR />} />
            <Route path="myntra"    element={<Myntra />} />
            <Route path="meesho"    element={<Meesho />} />
            <Route path="flipkart"  element={<Flipkart />} />
            <Route path="inventory" element={<Inventory />} />
            <Route path="po"        element={<POEngine />} />
            <Route path="forecast"  element={<Forecast />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
