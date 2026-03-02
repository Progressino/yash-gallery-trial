import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { useSession } from '../store/session'
import { cacheLoad, cacheSave, getCoverage } from '../api/client'

const NAV_ITEMS = [
  { to: '/',          label: '📊 Dashboard' },
  { to: '/mtr',       label: '📑 MTR Analytics' },
  { to: '/myntra',    label: '🛍️ Myntra' },
  { to: '/meesho',    label: '🛒 Meesho' },
  { to: '/flipkart',  label: '🟡 Flipkart' },
  { to: '/inventory', label: '📦 Inventory' },
  { to: '/po',        label: '🎯 PO Engine' },
  { to: '/forecast',  label: '📈 AI Forecast' },
]

export default function Layout() {
  const { sku_mapping, mtr, sales, myntra, meesho, flipkart, setCoverage } = useSession()
  const qc = useQueryClient()
  const [cacheMsg, setCacheMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)
  const [cacheLoading, setCacheLoading] = useState<'load' | 'save' | null>(null)

  const flash = (type: 'ok' | 'err', text: string) => {
    setCacheMsg({ type, text })
    setTimeout(() => setCacheMsg(null), 5000)
  }

  const handleLoad = async () => {
    setCacheLoading('load')
    try {
      const res = await cacheLoad()
      if (res.ok) {
        const c = await getCoverage()
        setCoverage(c)
        qc.invalidateQueries()
        flash('ok', res.message)
      } else {
        flash('err', res.message)
      }
    } catch {
      flash('err', 'Load failed')
    } finally {
      setCacheLoading(null)
    }
  }

  const handleSave = async () => {
    setCacheLoading('save')
    try {
      const res = await cacheSave()
      flash(res.ok ? 'ok' : 'err', res.message)
    } catch {
      flash('err', 'Save failed')
    } finally {
      setCacheLoading(null)
    }
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 bg-white border-r border-gray-200 flex flex-col shrink-0">
        <div className="px-4 py-5 border-b border-gray-100">
          <h1 className="text-sm font-bold text-[#002B5B] leading-tight">
            🚀 Yash Gallery<br />
            <span className="font-normal text-gray-500 text-xs">ERP Command Center</span>
          </h1>
        </div>

        <nav className="flex-1 overflow-y-auto py-2">
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `block px-4 py-2 text-sm transition-colors ${
                  isActive
                    ? 'bg-[#002B5B] text-white font-semibold'
                    : 'text-gray-600 hover:bg-gray-50'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Data coverage badges */}
        <div className="px-3 py-3 border-t border-gray-100 text-xs space-y-1">
          <p className="font-semibold text-gray-500 uppercase tracking-wide mb-2">Loaded</p>
          <Badge label="SKU Map"  active={sku_mapping} />
          <Badge label="MTR"      active={mtr} />
          <Badge label="Sales"    active={sales} />
          <Badge label="Myntra"   active={myntra} />
          <Badge label="Meesho"   active={meesho} />
          <Badge label="Flipkart" active={flipkart} />
        </div>

        {/* Cache controls */}
        <div className="px-3 py-3 border-t border-gray-100 space-y-2">
          <button
            onClick={handleLoad}
            disabled={cacheLoading !== null}
            className="w-full py-1.5 rounded text-xs font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-50"
          >
            {cacheLoading === 'load' ? 'Loading…' : '📥 Load Cache'}
          </button>
          <button
            onClick={handleSave}
            disabled={cacheLoading !== null}
            className="w-full py-1.5 rounded text-xs font-semibold text-[#002B5B] border border-[#002B5B] hover:bg-gray-50 disabled:opacity-50"
          >
            {cacheLoading === 'save' ? 'Saving…' : '💾 Save Cache'}
          </button>
          {cacheMsg && (
            <p className={`text-xs leading-tight ${cacheMsg.type === 'ok' ? 'text-green-600' : 'text-red-500'}`}>
              {cacheMsg.text}
            </p>
          )}
        </div>
      </aside>

      {/* Main area */}
      <main className="flex-1 overflow-y-auto bg-gray-50 p-6">
        <Outlet />
      </main>
    </div>
  )
}

function Badge({ label, active }: { label: string; active: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <span className={`w-2 h-2 rounded-full ${active ? 'bg-green-500' : 'bg-gray-300'}`} />
      <span className={active ? 'text-gray-700' : 'text-gray-400'}>{label}</span>
    </div>
  )
}
