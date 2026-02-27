import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Placeholder from './pages/Placeholder'

const qc = new QueryClient()

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="mtr"       element={<Placeholder title="📑 MTR Analytics" />} />
            <Route path="myntra"    element={<Placeholder title="🛍️ Myntra" />} />
            <Route path="meesho"    element={<Placeholder title="🛒 Meesho" />} />
            <Route path="flipkart"  element={<Placeholder title="🟡 Flipkart" />} />
            <Route path="inventory" element={<Placeholder title="📦 Inventory" />} />
            <Route path="po"        element={<Placeholder title="🎯 PO Engine" />} />
            <Route path="forecast"  element={<Placeholder title="📈 AI Forecast" />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
