import { useState, useEffect, useRef } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { useSession } from '../store/session'
import { cacheLoad, cacheSave, cacheReloadFresh, resetAllAppData, getCoverage } from '../api/client'
import api from '../api/client'

const NAV_GROUPS = [
  {
    label: 'Sales & Analytics',
    items: [
      { to: '/',          label: '📊 Intelligence' },
      { to: '/upload',    label: '📁 Upload Data' },
      { to: '/mtr',       label: '📦 Amazon' },
      { to: '/myntra',    label: '🛍️ Myntra' },
      { to: '/meesho',    label: '🛒 Meesho' },
      { to: '/flipkart',  label: '🟡 Flipkart' },
      { to: '/snapdeal',  label: '🔴 Snapdeal' },
      { to: '/forecast',    label: '📈 AI Forecast' },
      { to: '/finance',     label: '💰 Finance' },
      { to: '/sku-deepdive', label: '🔬 SKU Deepdive' },
    ],
  },
  {
    label: 'ERP',
    items: [
      { to: '/sales',      label: '🧾 Sales Orders' },
      { to: '/items',      label: '🏭 Item Master' },
      { to: '/purchase',   label: '🛒 Purchase' },
      { to: '/tna',        label: '📅 TNA Calendar' },
      { to: '/production', label: '⚙️ Production' },
      { to: '/grey',       label: '🧵 Grey Fabric' },
      { to: '/inventory',  label: '📦 Inventory' },
      { to: '/po',         label: '🎯 PO Engine' },
      { to: '/admin',      label: '🔐 Admin' },
      { to: '/marketplace-connections', label: '🔗 Marketplace API' },
    ],
  },
]

export default function Layout() {
  const { sku_mapping, mtr, sales, myntra, meesho, flipkart, snapdeal, setCoverage } =
    useSession()
  const qc = useQueryClient()
  const [cacheMsg, setCacheMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)
  const [cacheLoading, setCacheLoading] = useState<'load' | 'save' | 'reload' | 'delete' | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const autoLoadAttempted = useRef(false)

  // On mount: just refresh badges from server session state.
  // Auto-restore is handled once in ProtectedRoute to avoid duplicate cacheLoad calls.
  useEffect(() => {
    if (autoLoadAttempted.current) return
    autoLoadAttempted.current = true

    getCoverage()
      .then(c => {
        setCoverage(c)
      })
      .catch(() => {})
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

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

  const handleLogout = async () => {
    await api.post('/auth/logout')
    window.location.href = '/login'
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

  /** Full pipeline: clears server warm cache + session, GitHub download, SQLite daily merge, rebuild sales (no browser file picking). */
  const handleReloadFresh = async () => {
    setCacheLoading('reload')
    try {
      const res = await cacheReloadFresh()
      if (res.ok) {
        const c = await getCoverage()
        setCoverage(c)
        qc.invalidateQueries()
        flash('ok', res.message)
      } else {
        flash('err', res.message)
      }
    } catch (e: unknown) {
      const msg =
        e && typeof e === 'object' && 'message' in e && typeof (e as { message: unknown }).message === 'string'
          ? (e as { message: string }).message
          : 'Fresh reload failed (timeout or network — try again; if it persists, check server logs).'
      flash('err', msg)
    } finally {
      setCacheLoading(null)
    }
  }

  const handleDeleteAll = async () => {
    if (!window.confirm('Delete ALL data including GitHub cache? You will need to re-upload everything.')) return
    setCacheLoading('delete')
    try {
      const res = await resetAllAppData({ clearTier3Sqlite: true, clearWarmCache: true, clearGithubCache: true })
      if (res.ok) {
        const c = await getCoverage()
        setCoverage(c)
        qc.clear()
        flash('ok', res.message)
      } else {
        flash('err', res.message)
      }
    } catch {
      flash('err', 'Delete failed')
    } finally {
      setCacheLoading(null)
    }
  }

  const closeSidebar = () => setSidebarOpen(false)

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-20 md:hidden"
          onClick={closeSidebar}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed md:static inset-y-0 left-0 z-30
          w-56 bg-white border-r border-gray-200 flex flex-col shrink-0
          transition-transform duration-200 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}
      >
        <div className="px-4 py-4 border-b border-gray-100 flex items-center justify-between">
          <div>
            <img src="/logo.png" alt="Progressino" className="h-8 w-auto" />
            <p className="text-[10px] text-gray-400 font-medium mt-0.5 tracking-wide">Yash Gallery ERP</p>
          </div>
          {/* Close button — mobile only */}
          <button
            className="md:hidden text-gray-400 hover:text-gray-600 text-lg leading-none"
            onClick={closeSidebar}
          >
            ✕
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto py-2">
          {NAV_GROUPS.map(({ label, items }) => (
            <div key={label}>
              <p className="px-4 pt-3 pb-1 text-[10px] font-bold text-gray-400 uppercase tracking-widest">{label}</p>
              {items.map(({ to, label: navLabel }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === '/'}
                  onClick={closeSidebar}
                  className={({ isActive }) =>
                    `block px-4 py-1.5 text-sm transition-colors ${
                      isActive
                        ? 'bg-[#002B5B] text-white font-semibold'
                        : 'text-gray-600 hover:bg-gray-50'
                    }`
                  }
                >
                  {navLabel}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        {/* Data coverage badges */}
        <div className="px-3 py-3 border-t border-gray-100 text-xs space-y-1">
          <p className="font-semibold text-gray-500 uppercase tracking-wide mb-2">Loaded</p>
          <Badge label="SKU Map"  active={sku_mapping} />
          <Badge label="Amazon"   active={mtr} />
          <Badge label="Sales"    active={sales} />
          <Badge label="Myntra"   active={myntra} />
          <Badge label="Meesho"   active={meesho} />
          <Badge label="Flipkart" active={flipkart} />
          <Badge label="Snapdeal" active={snapdeal} />
        </div>

        {/* Logout */}
        <div className="px-3 pb-2">
          <button
            onClick={handleLogout}
            className="w-full py-1.5 rounded text-xs font-semibold text-gray-500 hover:text-red-600 hover:bg-red-50 border border-gray-200 transition-colors"
          >
            🚪 Sign Out
          </button>
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
          <button
            type="button"
            onClick={handleReloadFresh}
            disabled={cacheLoading !== null}
            title="Full server reload from GitHub + Tier-3 merge (operators)."
            className="w-full py-1.5 rounded text-xs font-semibold text-[#002B5B] border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
          >
            {cacheLoading === 'reload' ? 'Rebuilding…' : '↻ Fresh reload (server)'}
          </button>
          <button
            type="button"
            onClick={handleDeleteAll}
            disabled={cacheLoading !== null}
            title="Wipes session, server warm cache, Tier-3 SQLite, AND GitHub Release cache. Re-upload required."
            className="w-full py-1.5 rounded text-xs font-semibold text-white bg-red-600 hover:bg-red-700 disabled:opacity-50"
          >
            {cacheLoading === 'delete' ? 'Deleting…' : '🗑️ Delete All Data'}
          </button>
          {cacheMsg && (
            <p className={`text-xs leading-tight ${cacheMsg.type === 'ok' ? 'text-green-600' : 'text-red-500'}`}>
              {cacheMsg.text}
            </p>
          )}
        </div>

        {/* Built by Progressino */}
        <div className="px-3 py-3 border-t border-gray-100 flex items-center justify-center gap-1.5">
          <img src="/logo.png" alt="Progressino" className="h-4 w-auto opacity-40" />
          <span className="text-[10px] text-gray-300 font-medium">Built by Progressino</span>
        </div>
      </aside>

      {/* Main column */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        {/* Mobile top bar */}
        <header className="md:hidden flex items-center gap-3 px-4 py-3 bg-white border-b border-gray-200 shrink-0">
          <button
            onClick={() => setSidebarOpen(true)}
            className="text-[#002B5B] text-xl font-bold leading-none"
            aria-label="Open menu"
          >
            ☰
          </button>
          <span className="text-sm font-bold text-[#002B5B]">🚀 Yash Gallery ERP</span>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto bg-gray-50 p-4 md:p-6">
          <Outlet />
        </main>
      </div>
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
