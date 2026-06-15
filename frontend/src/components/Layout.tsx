import { useState, useEffect, useRef, useMemo } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { useSession } from '../store/session'
import { cacheLoad, cacheSave, cacheReloadFresh, resetAllAppData, getCoverage, invalidateDataQueries } from '../api/client'
import api from '../api/client'
import { useAuth, mayResetSharedData, mayUploadHistorical, canAccessModule, isHrmOnlyUser } from '../store/auth'
import type { ModuleKey } from '../lib/modules'
import { FixedTopLoadingBar } from './LoadingProgressBar'
import { clearLocalSessionHint, formatLocalHintAge, readLocalSessionHint } from '../lib/localSessionHint'

type NavItem = { to: string; label: string; module: ModuleKey; short?: string }

const NAV_GROUPS: { label: string; items: NavItem[] }[] = [
  {
    label: 'Sales & Analytics',
    items: [
      { to: '/', module: 'intelligence', label: 'Intelligence', short: '📊' },
      { to: '/upload', module: 'upload', label: 'Upload Data', short: '📁' },
      { to: '/mtr', module: 'amazon', label: 'Amazon', short: '📦' },
      { to: '/myntra', module: 'myntra', label: 'Myntra', short: '🛍️' },
      { to: '/meesho', module: 'meesho', label: 'Meesho', short: '🛒' },
      { to: '/flipkart', module: 'flipkart', label: 'Flipkart', short: '🟡' },
      { to: '/snapdeal', module: 'snapdeal', label: 'Snapdeal', short: '🔴' },
      { to: '/forecast', module: 'forecast', label: 'AI Forecast', short: '📈' },
      { to: '/finance', module: 'finance', label: 'Finance', short: '💰' },
      { to: '/sku-deepdive', module: 'sku_deepdive', label: 'SKU Deepdive', short: '🔬' },
    ],
  },
  {
    label: 'ERP',
    items: [
      { to: '/sales', module: 'sales', label: 'Sales Orders', short: '🧾' },
      { to: '/items', module: 'items', label: 'Item Master', short: '🏭' },
      { to: '/purchase', module: 'purchase', label: 'Purchase', short: '🛒' },
      { to: '/tna', module: 'tna', label: 'TNA Calendar', short: '📅' },
      { to: '/production', module: 'production', label: 'Production', short: '⚙️' },
      { to: '/stitching-costing', module: 'stitching', label: 'Stitching Costing', short: '🧵' },
      { to: '/grey', module: 'grey', label: 'Grey Fabric', short: '🧵' },
      { to: '/hrm', module: 'hrm', label: 'HRM', short: '👥' },
      { to: '/inventory', module: 'inventory', label: 'Inventory', short: '📦' },
      { to: '/po', module: 'po', label: 'PO Engine', short: '🎯' },
      { to: '/admin', module: 'admin', label: 'Admin', short: '🔐' },
      { to: '/marketplace-connections', module: 'marketplace', label: 'Marketplace API', short: '🔗' },
    ],
  },
]

const LOADED_PANEL_KEY = 'erp_sidebar_loaded_open'
const TOOLS_PANEL_KEY = 'erp_sidebar_tools_open'

function filterNavGroups(user: ReturnType<typeof useAuth.getState>['user']) {
  return NAV_GROUPS.map(g => ({
    ...g,
    items: g.items.filter(item => canAccessModule(user, item.module)),
  })).filter(g => g.items.length > 0)
}

function readPanelPref(key: string, defaultOpen: boolean): boolean {
  try {
    const v = localStorage.getItem(key)
    if (v === '1') return true
    if (v === '0') return false
  } catch {
    /* ignore */
  }
  return defaultOpen
}

function writePanelPref(key: string, open: boolean) {
  try {
    localStorage.setItem(key, open ? '1' : '0')
  } catch {
    /* ignore */
  }
}

type DatasetRow = { id: string; label: string; active: boolean }

function LoadedDataPanel({
  datasets,
  open,
  onToggle,
}: {
  datasets: DatasetRow[]
  open: boolean
  onToggle: () => void
}) {
  const loaded = datasets.filter(d => d.active).length
  const total = datasets.length
  const allOk = loaded === total

  return (
    <div className="border-t border-slate-200/80 shrink-0">
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-slate-50 transition-colors"
        aria-expanded={open}
      >
        <span
          className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-bold ${
            allOk ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-900'
          }`}
        >
          {loaded}/{total}
        </span>
        <span className="flex-1 min-w-0">
          <span className="block text-[11px] font-semibold text-slate-700">Data loaded</span>
          <span className="block text-[10px] text-slate-500 truncate">
            {open ? 'Tap to hide' : allOk ? 'All sheets ready' : `${total - loaded} missing`}
          </span>
        </span>
        <span className={`text-slate-400 text-xs transition-transform ${open ? 'rotate-180' : ''}`}>▼</span>
      </button>
      {open && (
        <div className="px-3 pb-2">
          <div className="grid grid-cols-2 gap-1">
            {datasets.map(d => (
              <div
                key={d.id}
                className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-[10px] ${
                  d.active ? 'bg-emerald-50 text-emerald-900' : 'bg-slate-50 text-slate-400'
                }`}
                title={d.active ? `${d.label} loaded` : `${d.label} not loaded`}
              >
                <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${d.active ? 'bg-emerald-500' : 'bg-slate-300'}`} />
                <span className="truncate font-medium">{d.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function CacheToolsPanel({
  open,
  onToggle,
  cacheLoading,
  allowHistorical,
  allowReset,
  localHintAge,
  cacheMsg,
  onLoad,
  onSave,
  onReload,
  onDelete,
}: {
  open: boolean
  onToggle: () => void
  cacheLoading: 'load' | 'save' | 'reload' | 'delete' | null
  allowHistorical: boolean
  allowReset: boolean
  localHintAge: string | null
  cacheMsg: { type: 'ok' | 'err'; text: string } | null
  onLoad: () => void
  onSave: () => void
  onReload: () => void
  onDelete: () => void
}) {
  const busy = cacheLoading !== null

  return (
    <div className="border-t border-slate-200/80 shrink-0">
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-slate-50 transition-colors"
        aria-expanded={open}
      >
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-[#002B5B]/10 text-sm">⚡</span>
        <span className="flex-1 min-w-0">
          <span className="block text-[11px] font-semibold text-slate-700">Server &amp; cache</span>
          <span className="block text-[10px] text-slate-500 truncate">
            {busy ? 'Working…' : open ? 'Tap to hide' : 'Load · save · reload'}
          </span>
        </span>
        <span className={`text-slate-400 text-xs transition-transform ${open ? 'rotate-180' : ''}`}>▼</span>
      </button>
      {open && (
        <div className="px-3 pb-3 space-y-1.5">
          <button
            type="button"
            onClick={onLoad}
            disabled={busy}
            className="w-full py-2 rounded-lg text-xs font-semibold text-white bg-gradient-to-r from-[#002B5B] to-[#1e5a9a] hover:from-[#003d7a] hover:to-[#2563a8] disabled:opacity-50 shadow-sm"
          >
            {cacheLoading === 'load' ? 'Loading…' : '📥 Load cache'}
          </button>
          {allowHistorical && (
            <div className="grid grid-cols-2 gap-1.5">
              <button
                type="button"
                onClick={onSave}
                disabled={busy}
                className="py-1.5 rounded-lg text-[10px] font-semibold text-[#002B5B] border border-[#002B5B]/30 bg-white hover:bg-slate-50 disabled:opacity-50"
              >
                {cacheLoading === 'save' ? '…' : '💾 Save'}
              </button>
              <button
                type="button"
                onClick={onReload}
                disabled={busy}
                title="GitHub + warm snapshot + Tier-3"
                className="py-1.5 rounded-lg text-[10px] font-semibold text-[#002B5B] border border-slate-200 bg-white hover:bg-slate-50 disabled:opacity-50"
              >
                {cacheLoading === 'reload' ? '…' : '↻ Reload'}
              </button>
            </div>
          )}
          {allowReset && (
            <button
              type="button"
              onClick={onDelete}
              disabled={busy}
              className="w-full py-1.5 rounded-lg text-[10px] font-semibold text-white bg-red-600 hover:bg-red-700 disabled:opacity-50"
            >
              {cacheLoading === 'delete' ? 'Deleting…' : '🗑️ Delete all data'}
            </button>
          )}
          {!allowHistorical && (
            <p className="text-[10px] text-slate-500 leading-snug pt-0.5">
              Daily uploads only for your role. Use Upload for new sales files.
            </p>
          )}
          {!allowReset && allowHistorical && (
            <p className="text-[10px] text-amber-800 leading-snug pt-0.5">
              Uploaded data is locked — clear and delete-all are owner-only.
            </p>
          )}
          {localHintAge && (
            <p className="text-[10px] text-slate-400">Browser hint: {localHintAge}</p>
          )}
          {cacheMsg && (
            <p className={`text-[10px] leading-tight rounded px-2 py-1 ${cacheMsg.type === 'ok' ? 'bg-emerald-50 text-emerald-800' : 'bg-red-50 text-red-700'}`}>
              {cacheMsg.text}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

export default function Layout() {
  const { sku_mapping, mtr, sales, myntra, meesho, flipkart, snapdeal, inventory, setCoverage } =
    useSession()
  const qc = useQueryClient()
  const [cacheMsg, setCacheMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)
  const [cacheLoading, setCacheLoading] = useState<'load' | 'save' | 'reload' | 'delete' | null>(null)
  const [loadedPanelOpen, setLoadedPanelOpen] = useState(() => readPanelPref(LOADED_PANEL_KEY, false))
  const [toolsPanelOpen, setToolsPanelOpen] = useState(() => readPanelPref(TOOLS_PANEL_KEY, false))

  const cacheBarLabel =
    cacheLoading === 'load'
      ? 'Loading cache…'
      : cacheLoading === 'save'
        ? 'Saving cache…'
        : cacheLoading === 'reload'
          ? 'Rebuilding from server…'
          : cacheLoading === 'delete'
            ? 'Deleting data…'
            : undefined

  const [sidebarOpen, setSidebarOpen] = useState(false)
  const embeddedSha = (import.meta.env.VITE_APP_GIT_SHA as string | undefined)?.trim()
  const embeddedBuilt = (import.meta.env.VITE_APP_BUILT_AT as string | undefined)?.trim()
  const embeddedVersion =
    embeddedSha && embeddedSha !== 'dev'
      ? `${embeddedSha}${embeddedBuilt ? ` · ${embeddedBuilt.slice(0, 10)}` : ''}`
      : null
  const [appVersion, setAppVersion] = useState<string | null>(embeddedVersion)
  const localHint = readLocalSessionHint()
  const navScrollRef = useRef<HTMLElement>(null)
  const [navCanScroll, setNavCanScroll] = useState(false)

  const datasets = useMemo<DatasetRow[]>(
    () => [
      { id: 'sku', label: 'SKU Map', active: sku_mapping },
      { id: 'mtr', label: 'Amazon', active: mtr },
      { id: 'sales', label: 'Sales', active: sales },
      { id: 'inv', label: 'Inventory', active: inventory },
      { id: 'myntra', label: 'Myntra', active: myntra },
      { id: 'meesho', label: 'Meesho', active: meesho },
      { id: 'flipkart', label: 'Flipkart', active: flipkart },
      { id: 'snapdeal', label: 'Snapdeal', active: snapdeal },
    ],
    [sku_mapping, mtr, sales, inventory, myntra, meesho, flipkart, snapdeal],
  )

  useEffect(() => {
    let cancelled = false
    const load = (attempt: number) => {
      api
        .get<{ label?: string; git_sha?: string; version?: string; built_at?: string }>('/health', {
          timeout: 8_000,
        })
        .then((r) => {
          if (cancelled) return
          const sha = r.data.git_sha || r.data.label || r.data.version
          if (!sha || sha === 'dev') return
          const built = r.data.built_at ? ` · ${r.data.built_at.slice(0, 10)}` : ''
          setAppVersion(`${sha}${built}`)
        })
        .catch(() => {
          if (!cancelled && attempt < 4) {
            window.setTimeout(() => load(attempt + 1), 4_000)
          }
        })
    }
    load(0)
    return () => {
      cancelled = true
    }
  }, [])

  // Coverage polling is centralized in CoverageProvider (App.tsx).

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
        invalidateDataQueries(qc)
        flash('ok', res.message)
      } else flash('err', res.message)
    } catch {
      flash('err', 'Load failed')
    } finally {
      setCacheLoading(null)
    }
  }

  const clearAuth = useAuth(s => s.clear)
  const authUser = useAuth(s => s.user)
  const allowReset = mayResetSharedData(authUser)
  const allowHistorical = mayUploadHistorical(authUser)
  const navGroups = useMemo(() => filterNavGroups(authUser), [authUser])
  const hrmOnly = isHrmOnlyUser(authUser)

  useEffect(() => {
    const el = navScrollRef.current
    if (!el) return
    const check = () => {
      setNavCanScroll(el.scrollHeight > el.clientHeight + 4)
    }
    check()
    const ro = new ResizeObserver(check)
    ro.observe(el)
    return () => ro.disconnect()
  }, [navGroups])

  const handleLogout = async () => {
    try {
      await api.post('/auth/logout')
    } catch {
      /* still clear local session */
    }
    clearAuth()
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

  const handleReloadFresh = async () => {
    setCacheLoading('reload')
    try {
      const res = await cacheReloadFresh()
      if (res.ok) {
        const c = await getCoverage()
        setCoverage(c)
        invalidateDataQueries(qc)
        flash('ok', res.message)
      } else flash('err', res.message)
    } catch (e: unknown) {
      const msg =
        e && typeof e === 'object' && 'message' in e && typeof (e as { message: unknown }).message === 'string'
          ? (e as { message: string }).message
          : 'Fresh reload failed (timeout or network — try again).'
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
      clearLocalSessionHint()
      if (res.ok) {
        const c = await getCoverage()
        setCoverage(c)
        qc.clear()
        flash('ok', res.message)
      } else flash('err', res.message)
    } catch {
      flash('err', 'Delete failed')
    } finally {
      setCacheLoading(null)
    }
  }

  const closeSidebar = () => setSidebarOpen(false)
  const toggleLoaded = () => {
    setLoadedPanelOpen(v => {
      const next = !v
      writePanelPref(LOADED_PANEL_KEY, next)
      return next
    })
  }
  const toggleTools = () => {
    setToolsPanelOpen(v => {
      const next = !v
      writePanelPref(TOOLS_PANEL_KEY, next)
      return next
    })
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <FixedTopLoadingBar active={cacheLoading !== null} label={cacheBarLabel} />
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/40 z-20 md:hidden" onClick={closeSidebar} aria-hidden />
      )}

      <aside
        className={`
          fixed md:static inset-y-0 left-0 z-30
          w-[15.5rem] md:w-60
          bg-gradient-to-b from-white via-white to-slate-50
          border-r border-slate-200/90 flex flex-col shrink-0 min-h-0
          shadow-xl md:shadow-none
          transition-transform duration-200 ease-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}
      >
        {/* Header */}
        <div className="shrink-0 px-3 py-3 border-b border-slate-200/80 bg-gradient-to-r from-[#002B5B] to-[#1a4a7c] text-white">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <img src="/logo.png" alt="" className="h-8 w-auto brightness-0 invert opacity-95" />
              <div className="min-w-0">
                <p className="text-xs font-bold leading-tight truncate">Yash Gallery</p>
                <p className="text-[10px] text-white/70">ERP · Progressino</p>
              </div>
            </div>
            <button
              type="button"
              className="md:hidden shrink-0 h-8 w-8 rounded-lg bg-white/10 hover:bg-white/20 text-sm"
              onClick={closeSidebar}
              aria-label="Close menu"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Navigation — primary scroll region */}
        <div className="relative flex-1 min-h-0 flex flex-col">
          {navCanScroll && (
            <div
              className="pointer-events-none absolute top-0 left-0 right-0 h-6 z-10 bg-gradient-to-b from-white to-transparent"
              aria-hidden
            />
          )}
          <nav
            ref={navScrollRef}
            className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden py-2 sidebar-nav-scroll"
            aria-label="Main navigation"
          >
            {navGroups.map(({ label, items }) => (
              <div key={label} className="mb-1">
                <p className="sticky top-0 z-[1] mx-2 mb-0.5 px-2 py-1 text-[10px] font-bold text-slate-400 uppercase tracking-widest bg-white/95 backdrop-blur-sm rounded-md">
                  {label}
                </p>
                <ul className="space-y-0.5 px-2">
                  {items.map(({ to, label: navLabel, short }) => (
                    <li key={to}>
                      <NavLink
                        to={to}
                        end={to === '/'}
                        onClick={closeSidebar}
                        title={navLabel}
                        className={({ isActive }) =>
                          `flex items-center gap-2 rounded-lg px-2.5 py-2 text-[13px] transition-all ${
                            isActive
                              ? 'bg-[#002B5B] text-white font-semibold shadow-md shadow-[#002B5B]/20'
                              : 'text-slate-600 hover:bg-slate-100 hover:text-[#002B5B]'
                          }`
                        }
                      >
                        <span className="text-base leading-none shrink-0 w-5 text-center" aria-hidden>
                          {short}
                        </span>
                        <span className="truncate">{navLabel}</span>
                      </NavLink>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </nav>
          {navCanScroll && (
            <div
              className="pointer-events-none absolute bottom-0 left-0 right-0 h-8 z-10 bg-gradient-to-t from-slate-50 to-transparent flex items-end justify-center pb-0.5"
              aria-hidden
            >
              <span className="text-[9px] text-slate-400 font-medium tracking-wide">SCROLL</span>
            </div>
          )}
        </div>

        {/* Footer stack — compact, collapsible */}
        <div className="shrink-0 bg-white/80 backdrop-blur-sm border-t border-slate-200/80">
          <button
            type="button"
            onClick={handleLogout}
            className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium text-slate-600 hover:bg-red-50 hover:text-red-700 transition-colors border-b border-slate-100"
          >
            <span className="text-sm">🚪</span>
            <span>Sign out</span>
            {(authUser?.full_name || authUser?.username) && (
              <span
                className="ml-auto text-[10px] text-slate-400 truncate max-w-[5rem]"
                title={authUser.full_name || authUser.username}
              >
                {(authUser.full_name || authUser.username).split(' ')[0]}
              </span>
            )}
          </button>

          {!hrmOnly && (
            <>
              <LoadedDataPanel datasets={datasets} open={loadedPanelOpen} onToggle={toggleLoaded} />
              <CacheToolsPanel
                open={toolsPanelOpen}
                onToggle={toggleTools}
                cacheLoading={cacheLoading}
                allowHistorical={allowHistorical}
                allowReset={allowReset}
                localHintAge={localHint ? formatLocalHintAge(localHint) : null}
                cacheMsg={cacheMsg}
                onLoad={handleLoad}
                onSave={handleSave}
                onReload={handleReloadFresh}
                onDelete={handleDeleteAll}
              />
            </>
          )}

          <div className="px-3 py-2 border-t border-slate-200 bg-slate-50 shrink-0 space-y-1">
            <div className="flex items-center gap-1.5 min-w-0">
              <img src="/logo.png" alt="" className="h-3.5 w-auto shrink-0" />
              <span className="text-[10px] text-slate-500 font-medium truncate">Progressino</span>
            </div>
            <div
              className="text-[11px] font-mono text-slate-700 bg-white border border-slate-300 rounded px-2 py-1 truncate"
              title={appVersion ? `Build ${appVersion}` : 'Loading build…'}
            >
              {appVersion ? `Build ${appVersion}` : 'Build …'}
            </div>
          </div>
        </div>
      </aside>

      <div className="flex-1 flex flex-col overflow-hidden min-w-0 relative">
        <header className="md:hidden flex items-center gap-3 px-4 py-3 bg-white border-b border-gray-200 shrink-0">
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            className="flex h-10 w-10 items-center justify-center rounded-lg border border-slate-200 text-[#002B5B] text-lg"
            aria-label="Open menu"
          >
            ☰
          </button>
          <span className="text-sm font-bold text-[#002B5B]">Yash Gallery ERP</span>
        </header>

        <main className="flex-1 overflow-y-auto bg-gray-50 p-4 md:p-6 min-h-0">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
