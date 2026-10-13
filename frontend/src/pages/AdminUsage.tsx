import { useMemo, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import api from '../api/client'
import { useAuth } from '../store/auth'
import { MODULE_LABELS, moduleForPath, type ModuleKey } from '../lib/modules'
import { downloadCsv } from '../lib/exportCsv'

interface UsageUser {
  username: string
  full_name: string
  role: string
  department: string
  active: boolean
  last_seen: string | null
  last_login: string | null
  logins: number
  days_active: number
  active_minutes: number
  page_views: number
  actions: number
  top_module: string
}

interface TopRow { key: string; module: string; hits: number; users: number; last_at: string | null }

interface UsageReport {
  generated_at: string
  days: number
  from: string
  to: string
  tracking_since: string | null
  summary: {
    total_users: number
    active_today: number
    active_7d: number
    active_30d: number
    active_in_window: number
    page_views: number
    actions: number
    active_hours: number
    logins: number
    never_seen: number
  }
  trend: { day: string; active_users: number; active_minutes: number; page_views: number; actions: number }[]
  by_hour: { hour: number; active_users: number; active_minutes: number }[]
  modules: { module: string; views: number; actions: number; users: number }[]
  pages: TopRow[]
  tabs: TopRow[]
  actions: TopRow[]
  users: UsageUser[]
  recent_logins: { username: string; role: string; method: string; ip: string; user_agent: string; created_at: string }[]
  roles: string[]
  departments: string[]
}

type UserFilter = 'all' | 'active' | 'idle7' | 'never'

const PERIODS = [7, 30, 90, 180]

function moduleLabel(m: string): string {
  return MODULE_LABELS[m as ModuleKey] ?? (m ? m.replace(/[-_]/g, ' ') : '—')
}

function pageLabel(path: string): string {
  const m = moduleForPath(path)
  const base = m ? MODULE_LABELS[m] : path
  if (path === '/admin/usage') return 'Admin › Usage report'
  if (path === '/admin/performance') return 'Admin › Performance'
  return base
}

function tabLabel(key: string, module: string): string {
  const [, tab = key] = key.split(' › ')
  return `${moduleLabel(module)} › ${tab.replace(/[-_]/g, ' ')}`
}

const VERB: Record<string, string> = { POST: 'Create / run', PUT: 'Update', PATCH: 'Update', DELETE: 'Delete' }

function actionLabel(key: string): { verb: string; path: string } {
  const [method, path = ''] = key.split(' ', 2)
  return { verb: VERB[method] ?? method, path }
}

function fmtMinutes(min: number): string {
  if (!min) return '—'
  const h = Math.floor(min / 60)
  const m = min % 60
  return h ? `${h}h ${m.toString().padStart(2, '0')}m` : `${m}m`
}

function daysAgo(ts: string | null, today: string): number | null {
  if (!ts) return null
  const d = new Date(`${ts.slice(0, 10)}T00:00:00`)
  const t = new Date(`${today}T00:00:00`)
  return Math.round((t.getTime() - d.getTime()) / 86_400_000)
}

function fmtSeen(ts: string | null, today: string): string {
  if (!ts) return 'Never'
  const n = daysAgo(ts, today)
  const time = ts.slice(11, 16)
  if (n === 0) return `Today ${time}`
  if (n === 1) return `Yesterday ${time}`
  return `${ts.slice(0, 10)} (${n}d ago)`
}

function Card({ label, value, hint, tone = 'text-[#002B5B]' }: { label: string; value: string | number; hint?: string; tone?: string }) {
  return (
    <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
      <p className="text-xs font-semibold tracking-wide text-gray-500 uppercase">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${tone}`}>{value}</p>
      {hint && <p className="text-xs text-gray-400 mt-1">{hint}</p>}
    </div>
  )
}

function TopTable({ title, rows, render, empty }: {
  title: string
  rows: TopRow[]
  render: (r: TopRow) => ReactNode
  empty: string
}) {
  const max = Math.max(1, ...rows.map(r => r.hits))
  return (
    <div className="bg-white rounded-xl border p-4">
      <h3 className="font-semibold text-gray-700 mb-3 text-sm">{title}</h3>
      {rows.length === 0 ? (
        <p className="text-sm text-gray-400">{empty}</p>
      ) : (
        <div className="overflow-x-auto max-h-96 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-white">
              <tr className="text-left text-xs text-gray-400 border-b">
                <th className="py-2 pr-3">Name</th>
                <th className="py-2 pr-3 text-right">Count</th>
                <th className="py-2 text-right">Users</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.key} className="border-b border-gray-50 last:border-0">
                  <td className="py-1.5 pr-3">
                    {render(r)}
                    <div className="h-1 mt-1 rounded bg-indigo-100">
                      <div className="h-1 rounded bg-indigo-500" style={{ width: `${(r.hits / max) * 100}%` }} />
                    </div>
                  </td>
                  <td className="py-1.5 pr-3 text-right font-medium text-gray-700">{r.hits.toLocaleString()}</td>
                  <td className="py-1.5 text-right text-gray-500">{r.users}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default function AdminUsage() {
  const authUser = useAuth(s => s.user)
  const allowed = authUser?.role === 'Super Admin' || authUser?.role === 'Admin'
  const [days, setDays] = useState(30)
  const [role, setRole] = useState('')
  const [department, setDepartment] = useState('')
  const [username, setUsername] = useState('')
  const [userFilter, setUserFilter] = useState<UserFilter>('all')

  const { data, isLoading, isError, refetch, isFetching } = useQuery<UsageReport>({
    queryKey: ['admin-usage', days, role, department, username],
    queryFn: () => api.get('/admin/usage', { params: { days, role, department, username } }).then(r => r.data),
    enabled: allowed,
    staleTime: 60_000,
  })

  const today = data?.to ?? new Date().toISOString().slice(0, 10)

  const users = useMemo(() => {
    const rows = data?.users ?? []
    if (userFilter === 'active') return rows.filter(u => u.days_active > 0)
    if (userFilter === 'never') return rows.filter(u => !u.last_seen)
    if (userFilter === 'idle7') return rows.filter(u => { const n = daysAgo(u.last_seen, today); return n == null || n >= 7 })
    return rows
  }, [data?.users, userFilter, today])

  const trend = useMemo(
    () => (data?.trend ?? []).map(t => ({ ...t, label: t.day.slice(5), active_hours: +(t.active_minutes / 60).toFixed(1) })),
    [data?.trend],
  )
  const hours = useMemo(
    () => (data?.by_hour ?? []).filter(h => h.hour >= 6 && h.hour <= 23).map(h => ({ ...h, label: `${h.hour}:00`, active_hours: +(h.active_minutes / 60).toFixed(1) })),
    [data?.by_hour],
  )
  const moduleMax = Math.max(1, ...(data?.modules ?? []).map(m => m.views + m.actions))

  if (!allowed) {
    return (
      <div className="p-8 text-center text-gray-600">
        <p className="font-medium">Usage report is available to Super Admin and Admin only.</p>
      </div>
    )
  }

  const exportUsers = () => {
    if (!data) return
    downloadCsv(
      `user-usage-${data.from}-to-${data.to}`,
      ['Username', 'Full name', 'Role', 'Department', 'Last seen (IST)', 'Last login (IST)', 'Logins', 'Days active', 'Active minutes', 'Screen views', 'Actions', 'Top module'],
      data.users.map(u => [
        u.username, u.full_name, u.role, u.department, u.last_seen ?? '', u.last_login ?? '', u.logins,
        u.days_active, u.active_minutes, u.page_views, u.actions, moduleLabel(u.top_module),
      ]),
    )
  }

  const exportFeatures = () => {
    if (!data) return
    const rows: Array<Array<string | number>> = [
      ...data.modules.map(m => ['Module', moduleLabel(m.module), m.views, m.actions, m.users]),
      ...data.pages.map(p => ['Page', pageLabel(p.key), p.hits, 0, p.users]),
      ...data.tabs.map(t => ['Tab', tabLabel(t.key, t.module), t.hits, 0, t.users]),
      ...data.actions.map(a => ['Action', a.key, 0, a.hits, a.users]),
    ]
    downloadCsv(`feature-usage-${data.from}-to-${data.to}`, ['Type', 'Name', 'Views', 'Actions', 'Users'], rows)
  }

  const s = data?.summary

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <Link to="/admin" className="hover:text-[#002B5B]">Admin</Link>
            <span>/</span>
            <span>Usage</span>
          </div>
          <h1 className="text-xl font-bold text-gray-800">User Usage Report</h1>
          <p className="text-sm text-gray-500">
            Who is using the app, how much, and which modules, screens and actions are used most. Times in IST.
            {data?.tracking_since ? ` Tracking since ${data.tracking_since}.` : ' Tracking starts from this release.'}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select value={days} onChange={e => setDays(Number(e.target.value))}
            className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm">
            {PERIODS.map(p => <option key={p} value={p}>Last {p} days</option>)}
          </select>
          <select value={role} onChange={e => setRole(e.target.value)}
            className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm">
            <option value="">All roles</option>
            {(data?.roles ?? []).map(r => <option key={r} value={r}>{r}</option>)}
          </select>
          <select value={department} onChange={e => setDepartment(e.target.value)}
            className="border border-gray-200 rounded-lg px-2 py-1.5 text-sm">
            <option value="">All departments</option>
            {(data?.departments ?? []).map(d => <option key={d} value={d}>{d}</option>)}
          </select>
          {username && (
            <button type="button" onClick={() => setUsername('')}
              className="px-2 py-1.5 rounded-lg bg-indigo-50 text-indigo-700 text-sm border border-indigo-200">
              User: {username} ✕
            </button>
          )}
          <button type="button" onClick={() => refetch()} disabled={isFetching}
            className="px-3 py-1.5 border border-gray-200 rounded-lg text-sm text-gray-600 hover:bg-gray-50 disabled:opacity-50">
            {isFetching ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {isLoading && <p className="text-sm text-gray-500">Loading usage…</p>}
      {isError && <p className="text-sm text-red-600">Failed to load usage report.</p>}

      {data && s && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3">
            <Card label="Users" value={s.total_users} hint={`${s.never_seen} never seen`} />
            <Card label="Active today" value={s.active_today} tone="text-emerald-700" />
            <Card label="Active 7 days" value={s.active_7d} tone="text-emerald-700" />
            <Card label="Active 30 days" value={s.active_30d} tone="text-emerald-700" />
            <Card label="Active hours" value={s.active_hours} hint={`last ${data.days} days`} />
            <Card label="Screen views" value={s.page_views.toLocaleString()} hint={`last ${data.days} days`} />
            <Card label="Actions" value={s.actions.toLocaleString()} hint="saves / creates / deletes" />
            <Card label="Logins" value={s.logins} hint={`last ${data.days} days`} />
          </div>

          <div className="grid lg:grid-cols-3 gap-4">
            <div className="bg-white rounded-xl border p-4 lg:col-span-2">
              <h3 className="font-semibold text-gray-700 mb-3 text-sm">Daily active users & hours</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={trend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="u" allowDecimals={false} tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="h" orientation="right" tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend />
                    <Bar yAxisId="u" dataKey="active_users" name="Active users" fill="#6366f1" radius={[3, 3, 0, 0]} />
                    <Line yAxisId="h" dataKey="active_hours" name="Active hours" stroke="#059669" dot={false} strokeWidth={2} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="bg-white rounded-xl border p-4">
              <h3 className="font-semibold text-gray-700 mb-3 text-sm">Busiest hours (IST)</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={hours}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="label" tick={{ fontSize: 10 }} interval={2} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="active_hours" name="Active hours" fill="#0ea5e9" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl border p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-gray-700 text-sm">Most used modules</h3>
              <button type="button" onClick={exportFeatures}
                className="px-3 py-1 border border-gray-200 rounded-lg text-xs text-gray-600 hover:bg-gray-50">
                Export feature usage CSV
              </button>
            </div>
            {data.modules.length === 0 ? (
              <p className="text-sm text-gray-400">No activity recorded in this period yet.</p>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-gray-400 border-b">
                    <th className="py-2 pr-3">Module</th>
                    <th className="py-2 pr-3 w-1/3">Share</th>
                    <th className="py-2 pr-3 text-right">Screen views</th>
                    <th className="py-2 pr-3 text-right">Actions</th>
                    <th className="py-2 text-right">Users</th>
                  </tr>
                </thead>
                <tbody>
                  {data.modules.map(m => (
                    <tr key={m.module} className="border-b border-gray-50 last:border-0">
                      <td className="py-1.5 pr-3 font-medium text-gray-700">{moduleLabel(m.module)}</td>
                      <td className="py-1.5 pr-3">
                        <div className="h-2 rounded bg-indigo-100">
                          <div className="h-2 rounded bg-indigo-500" style={{ width: `${((m.views + m.actions) / moduleMax) * 100}%` }} />
                        </div>
                      </td>
                      <td className="py-1.5 pr-3 text-right">{m.views.toLocaleString()}</td>
                      <td className="py-1.5 pr-3 text-right">{m.actions.toLocaleString()}</td>
                      <td className="py-1.5 text-right text-gray-500">{m.users}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="bg-white rounded-xl border p-4">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <h3 className="font-semibold text-gray-700 text-sm">Users ({users.length})</h3>
              <div className="flex flex-wrap items-center gap-2">
                {([
                  ['all', 'All'],
                  ['active', 'Active in period'],
                  ['idle7', 'Not seen 7+ days'],
                  ['never', 'Never seen'],
                ] as [UserFilter, string][]).map(([k, label]) => (
                  <button key={k} type="button" onClick={() => setUserFilter(k)}
                    className={`px-2.5 py-1 rounded-full text-xs border ${userFilter === k ? 'bg-[#002B5B] text-white border-[#002B5B]' : 'border-gray-200 text-gray-600 hover:bg-gray-50'}`}>
                    {label}
                  </button>
                ))}
                <button type="button" onClick={exportUsers}
                  className="px-3 py-1 border border-gray-200 rounded-lg text-xs text-gray-600 hover:bg-gray-50">
                  Export users CSV
                </button>
              </div>
            </div>
            <div className="overflow-x-auto max-h-[32rem] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-white">
                  <tr className="text-left text-xs text-gray-400 border-b">
                    <th className="py-2 pr-3">User</th>
                    <th className="py-2 pr-3">Role / Dept</th>
                    <th className="py-2 pr-3">Last seen</th>
                    <th className="py-2 pr-3">Last login</th>
                    <th className="py-2 pr-3 text-right">Days active</th>
                    <th className="py-2 pr-3 text-right">Active time</th>
                    <th className="py-2 pr-3 text-right">Views</th>
                    <th className="py-2 pr-3 text-right">Actions</th>
                    <th className="py-2">Top module</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map(u => {
                    const idle = daysAgo(u.last_seen, today)
                    return (
                      <tr key={u.username}
                        onClick={() => setUsername(u.username === username ? '' : u.username)}
                        className={`border-b border-gray-50 last:border-0 cursor-pointer hover:bg-indigo-50/40 ${u.username === username ? 'bg-indigo-50' : ''}`}>
                        <td className="py-1.5 pr-3">
                          <p className="font-medium text-gray-700">{u.full_name || u.username}</p>
                          <p className="text-xs text-gray-400">{u.username}</p>
                        </td>
                        <td className="py-1.5 pr-3 text-xs text-gray-600">{u.role}{u.department ? ` · ${u.department}` : ''}</td>
                        <td className={`py-1.5 pr-3 text-xs ${idle == null ? 'text-red-600' : idle >= 7 ? 'text-amber-700' : 'text-gray-700'}`}>
                          {fmtSeen(u.last_seen, today)}
                        </td>
                        <td className="py-1.5 pr-3 text-xs text-gray-600">{u.last_login ? fmtSeen(u.last_login, today) : '—'}</td>
                        <td className="py-1.5 pr-3 text-right">{u.days_active || '—'}</td>
                        <td className="py-1.5 pr-3 text-right">{fmtMinutes(u.active_minutes)}</td>
                        <td className="py-1.5 pr-3 text-right">{u.page_views || '—'}</td>
                        <td className="py-1.5 pr-3 text-right">{u.actions || '—'}</td>
                        <td className="py-1.5 text-xs text-gray-600">{u.top_module ? moduleLabel(u.top_module) : '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-gray-400 mt-2">Click a user to filter the whole report to that person. Active time counts 5-minute blocks with clicks, typing or scrolling on a visible tab.</p>
          </div>

          <div className="grid lg:grid-cols-3 gap-4">
            <TopTable
              title="Top pages"
              rows={data.pages}
              empty="No page views yet."
              render={r => (
                <span className="text-gray-700">{pageLabel(r.key)} <span className="text-xs text-gray-400 font-mono">{r.key}</span></span>
              )}
            />
            <TopTable
              title="Top tabs inside pages"
              rows={data.tabs}
              empty="No tab views yet."
              render={r => <span className="text-gray-700">{tabLabel(r.key, r.module)}</span>}
            />
            <TopTable
              title="Top actions (saves / creates / deletes)"
              rows={data.actions}
              empty="No actions yet."
              render={r => {
                const a = actionLabel(r.key)
                return (
                  <span className="text-gray-700">
                    <span className="text-xs font-semibold text-indigo-700 mr-1">{a.verb}</span>
                    <span className="font-mono text-xs break-all">{a.path}</span>
                  </span>
                )
              }}
            />
          </div>

          <div className="bg-white rounded-xl border p-4">
            <h3 className="font-semibold text-gray-700 mb-3 text-sm">Recent logins</h3>
            {data.recent_logins.length === 0 ? (
              <p className="text-sm text-gray-400">No logins recorded in this period yet.</p>
            ) : (
              <div className="overflow-x-auto max-h-80 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-white">
                    <tr className="text-left text-xs text-gray-400 border-b">
                      <th className="py-2 pr-3">When (IST)</th>
                      <th className="py-2 pr-3">User</th>
                      <th className="py-2 pr-3">Role</th>
                      <th className="py-2 pr-3">Method</th>
                      <th className="py-2">Device</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.recent_logins.map((l, i) => (
                      <tr key={`${l.username}-${l.created_at}-${i}`} className="border-b border-gray-50 last:border-0">
                        <td className="py-1.5 pr-3 text-xs text-gray-700">{l.created_at}</td>
                        <td className="py-1.5 pr-3">{l.username}</td>
                        <td className="py-1.5 pr-3 text-xs text-gray-600">{l.role}</td>
                        <td className="py-1.5 pr-3 text-xs text-gray-600">{l.method}</td>
                        <td className="py-1.5 text-xs text-gray-400 max-w-md truncate" title={l.user_agent}>{l.user_agent}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
