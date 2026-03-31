import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

interface PlatformStatus {
  connected:    boolean
  last_sync:    string | null
  last_status:  string | null
  last_rows:    number
  last_message: string
}

interface StatusResponse {
  amazon:   PlatformStatus
  flipkart: PlatformStatus
  myntra:   PlatformStatus
  meesho:   PlatformStatus
}

interface SyncLogEntry {
  id:         number
  platform:   string
  synced_at:  string
  status:     string
  rows_added: number
  date_from:  string
  date_to:    string
  message:    string
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtDate(iso: string | null) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('en-IN', {
      day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  } catch { return iso }
}

function StatusBadge({ status }: { status: string | null }) {
  if (!status) return <span className="text-gray-400 text-xs">—</span>
  const cls: Record<string, string> = {
    success: 'bg-green-100 text-green-700',
    partial: 'bg-yellow-100 text-yellow-700',
    error:   'bg-red-100 text-red-700',
  }
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${cls[status] ?? 'bg-gray-100 text-gray-500'}`}>
      {status}
    </span>
  )
}

function DayPicker({ value, onChange }: { value: number; onChange: (n: number) => void }) {
  return (
    <select
      value={value}
      onChange={e => onChange(Number(e.target.value))}
      className="border border-gray-200 rounded-lg text-xs px-2 py-1.5 text-gray-600 focus:outline-none"
    >
      {[1, 3, 7, 14, 30].map(d => (
        <option key={d} value={d}>Last {d} day{d > 1 ? 's' : ''}</option>
      ))}
    </select>
  )
}

function SyncHistory({ entries }: { entries: SyncLogEntry[] }) {
  if (!entries.length) return null
  return (
    <div className="mt-4">
      <p className="text-xs font-medium text-gray-500 mb-2">Recent Syncs</p>
      <div className="space-y-1.5 max-h-44 overflow-y-auto">
        {entries.slice(0, 8).map(e => (
          <div key={e.id} className="flex flex-wrap items-center gap-2 text-xs bg-gray-50 rounded-lg px-3 py-2">
            <StatusBadge status={e.status} />
            <span className="text-gray-400">{fmtDate(e.synced_at)}</span>
            <span className="font-medium ml-auto">{e.rows_added.toLocaleString()} rows</span>
            {e.date_from && <span className="text-gray-400">{e.date_from} → {e.date_to}</span>}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Generic Connect Modal ─────────────────────────────────────────────────────

interface FieldDef { key: string; label: string; placeholder: string; type?: string }

function ConnectModal({
  title, helpUrl, helpText, fields, onClose, onSave,
}: {
  title: string
  helpUrl: string
  helpText: string
  fields: FieldDef[]
  onClose: () => void
  onSave: (values: Record<string, string>) => Promise<void>
}) {
  const [form, setForm] = useState<Record<string, string>>(
    Object.fromEntries(fields.map(f => [f.key, '']))
  )
  const [err, setErr]   = useState('')
  const [busy, setBusy] = useState(false)

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm(f => ({ ...f, [k]: e.target.value }))

  const handleSave = async () => {
    setErr(''); setBusy(true)
    try { await onSave(form) }
    catch (e: any) { setErr(e?.response?.data?.detail ?? e?.message ?? 'Connection failed') }
    finally { setBusy(false) }
  }

  const allFilled = fields.every(f => form[f.key].trim())

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-1">{title}</h2>
        <p className="text-sm text-gray-500 mb-5">
          {helpText}{' '}
          <a href={helpUrl} target="_blank" rel="noreferrer" className="text-blue-600 underline">
            Get credentials →
          </a>
        </p>
        <div className="space-y-3">
          {fields.map(({ key, label, placeholder, type }) => (
            <div key={key}>
              <label className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
              <input
                type={type ?? 'text'}
                value={form[key]}
                onChange={set(key)}
                placeholder={placeholder}
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
              />
            </div>
          ))}
        </div>
        {err && <p className="mt-3 text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">{err}</p>}
        <div className="mt-5 flex gap-3 justify-end">
          <button onClick={onClose} className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition">
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={busy || !allFilled}
            className="px-5 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition"
          >
            {busy ? 'Verifying…' : 'Save & Verify'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Platform Card ─────────────────────────────────────────────────────────────

function PlatformCard({
  platform, icon, name, accentColor, status, syncLog,
  connectFields, helpUrl, helpText,
  onConnect,
}: {
  platform:     string
  icon:         string
  name:         string
  accentColor:  string
  status:       PlatformStatus
  syncLog:      SyncLogEntry[]
  connectFields: FieldDef[]
  helpUrl:      string
  helpText:     string
  onConnect:    (values: Record<string, string>) => Promise<void>
}) {
  const qc = useQueryClient()
  const [showModal, setShowModal] = useState(false)
  const [daysBack, setDaysBack]   = useState(7)
  const [msg, setMsg]             = useState<{ ok: boolean; text: string } | null>(null)

  const { mutate: disconnect } = useMutation({
    mutationFn: () => api.delete(`/marketplace/${platform}/disconnect`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['marketplace-status'] })
      qc.invalidateQueries({ queryKey: ['marketplace-sync-log', platform] })
    },
  })

  const { mutate: syncNow, isPending: syncing } = useMutation({
    mutationFn: () => api.post(`/marketplace/${platform}/sync?days_back=${daysBack}`),
    onSuccess: res => {
      setMsg({ ok: true, text: res.data.message })
      setTimeout(() => {
        setMsg(null)
        qc.invalidateQueries({ queryKey: ['marketplace-sync-log', platform] })
      }, 5000)
    },
    onError: (e: any) => setMsg({ ok: false, text: e?.response?.data?.detail ?? 'Sync failed' }),
  })

  const handleConnect = async (values: Record<string, string>) => {
    await onConnect(values)
    setShowModal(false)
    qc.invalidateQueries({ queryKey: ['marketplace-status'] })
  }

  return (
    <>
      {showModal && (
        <ConnectModal
          title={`Connect ${name}`}
          helpUrl={helpUrl}
          helpText={helpText}
          fields={connectFields}
          onClose={() => setShowModal(false)}
          onSave={handleConnect}
        />
      )}

      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-xl ${accentColor} flex items-center justify-center text-xl`}>
              {icon}
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">{name}</h3>
              <p className="text-xs text-gray-400">Auto-sync daily at 6:00 AM IST</p>
            </div>
          </div>
          <span className={`flex items-center gap-1.5 text-xs font-medium px-3 py-1 rounded-full ${
            status.connected ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${status.connected ? 'bg-green-500' : 'bg-gray-400'}`} />
            {status.connected ? 'Connected' : 'Not connected'}
          </span>
        </div>

        {/* Stats row */}
        {status.connected && status.last_sync && (
          <div className="grid grid-cols-3 gap-2 mb-4 p-3 bg-gray-50 rounded-xl text-xs">
            <div><span className="text-gray-400 block">Last sync</span><span className="text-gray-700 font-medium">{fmtDate(status.last_sync)}</span></div>
            <div><span className="text-gray-400 block">Status</span><StatusBadge status={status.last_status} /></div>
            <div><span className="text-gray-400 block">Rows added</span><span className="text-gray-700 font-medium">{status.last_rows.toLocaleString()}</span></div>
          </div>
        )}

        {/* Actions */}
        <div className="flex flex-wrap gap-2">
          {status.connected ? (
            <>
              <DayPicker value={daysBack} onChange={setDaysBack} />
              <button
                onClick={() => syncNow()}
                disabled={syncing}
                className="px-4 py-1.5 text-xs font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition"
              >
                {syncing ? 'Starting…' : '⚡ Sync Now'}
              </button>
              <button
                onClick={() => setShowModal(true)}
                className="px-3 py-1.5 text-xs text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200 transition"
              >
                ✏️ Edit
              </button>
              <button
                onClick={() => { if (confirm(`Disconnect ${name}?`)) disconnect() }}
                className="px-3 py-1.5 text-xs text-red-600 bg-red-50 rounded-lg hover:bg-red-100 transition"
              >
                🔌 Disconnect
              </button>
            </>
          ) : (
            <button
              onClick={() => setShowModal(true)}
              className="px-5 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
            >
              🔗 Connect {name}
            </button>
          )}
        </div>

        {msg && (
          <p className={`mt-3 text-xs rounded-lg px-3 py-2 ${msg.ok ? 'bg-blue-50 text-blue-700' : 'bg-red-50 text-red-700'}`}>
            {msg.text}
          </p>
        )}

        <SyncHistory entries={syncLog} />
      </div>
    </>
  )
}

// ── Platform configs ──────────────────────────────────────────────────────────

const PLATFORM_CONFIGS = {
  amazon: {
    icon: '🟠', name: 'Amazon', accentColor: 'bg-orange-50',
    helpUrl: 'https://sellercentral.amazon.in/apps/manage',
    helpText: 'Register as SP-API developer on Seller Central, create a self-authorised app.',
    connectFields: [
      { key: 'client_id',      label: 'Client ID',      placeholder: 'amzn1.application-oa2-client.xxxx' },
      { key: 'client_secret',  label: 'Client Secret',  placeholder: '••••••••••', type: 'password' },
      { key: 'refresh_token',  label: 'Refresh Token',  placeholder: 'Atzr|IwEBIxxxxxx', type: 'password' },
      { key: 'seller_id',      label: 'Seller ID',      placeholder: 'A1B2C3D4E5F6G7' },
      { key: 'marketplace_id', label: 'Marketplace ID', placeholder: 'A21TJRUUN4KGV' },
    ] as FieldDef[],
    buildPayload: (v: Record<string, string>) => v,
    endpoint: '/marketplace/amazon/connect',
  },
  flipkart: {
    icon: '🟡', name: 'Flipkart', accentColor: 'bg-yellow-50',
    helpUrl: 'https://seller.flipkart.com/api-docs/FMSAPI.html',
    helpText: 'Go to Seller Dashboard → Manage Profile → Developer Access to create an app.',
    connectFields: [
      { key: 'app_id',     label: 'App ID (Client ID)',     placeholder: 'FK_App_xxxxxxxxx' },
      { key: 'app_secret', label: 'App Secret',             placeholder: '••••••••••', type: 'password' },
      { key: 'seller_id',  label: 'Seller ID (optional)',   placeholder: 'Your Flipkart seller ID' },
    ] as FieldDef[],
    buildPayload: (v: Record<string, string>) => v,
    endpoint: '/marketplace/flipkart/connect',
  },
  myntra: {
    icon: '🛍️', name: 'Myntra', accentColor: 'bg-pink-50',
    helpUrl: 'https://mmip.myntrainfo.com',
    helpText: 'Contact your Myntra account manager to enable API access and get credentials.',
    connectFields: [
      { key: 'username', label: 'Username',    placeholder: 'Your Myntra seller username' },
      { key: 'password', label: 'Password',    placeholder: '••••••••••', type: 'password' },
      { key: 'api_key',  label: 'API Key',     placeholder: 'Provided by Myntra account manager' },
      { key: 'seller_id', label: 'Seller ID (optional)', placeholder: 'Your Myntra seller ID' },
    ] as FieldDef[],
    buildPayload: (v: Record<string, string>) => v,
    endpoint: '/marketplace/myntra/connect',
  },
  meesho: {
    icon: '🛒', name: 'Meesho', accentColor: 'bg-purple-50',
    helpUrl: 'https://merchant.meesho.com',
    helpText: 'Email meesholink-integration@meesho.com to request API credentials.',
    connectFields: [
      { key: 'client_id',     label: 'Client ID',     placeholder: 'Your Meesho client ID' },
      { key: 'client_secret', label: 'Client Secret', placeholder: '••••••••••', type: 'password' },
      { key: 'supplier_id',   label: 'Supplier ID',   placeholder: 'Your Meesho supplier ID' },
    ] as FieldDef[],
    buildPayload: (v: Record<string, string>) => v,
    endpoint: '/marketplace/meesho/connect',
  },
}

// ── Setup Guide ───────────────────────────────────────────────────────────────

function SetupGuide() {
  const [open, setOpen] = useState(false)
  return (
    <div className="bg-blue-50 rounded-2xl border border-blue-100">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between p-5 text-left"
      >
        <span className="font-semibold text-blue-900">📋 How to get API credentials for each marketplace</span>
        <span className="text-blue-400 text-lg">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="px-5 pb-5 grid grid-cols-1 sm:grid-cols-2 gap-5">
          {[
            {
              name: '🟠 Amazon', color: 'border-orange-200',
              steps: [
                'Go to sellercentral.amazon.in → Apps & Services → Develop Apps',
                'Register as SP-API developer (1–3 business days approval)',
                'Create a Private Seller Application (self-authorised)',
                'Authorise the app under your seller account',
                'Copy Client ID, Client Secret, Refresh Token, Seller ID',
              ],
            },
            {
              name: '🟡 Flipkart', color: 'border-yellow-200',
              steps: [
                'Go to seller.flipkart.com → Manage Profile → Developer Access',
                'Create a new application',
                'Verify as a Partner if sharing with third parties (72 hrs)',
                'Copy App ID and App Secret',
              ],
            },
            {
              name: '🛍️ Myntra', color: 'border-pink-200',
              steps: [
                'Contact your dedicated Myntra account manager',
                'Request API access for the MMIP (Merchant Integration Platform)',
                'They will provide Username, Password, and API Key',
                'Use the credentials provided to connect',
              ],
            },
            {
              name: '🛒 Meesho', color: 'border-purple-200',
              steps: [
                'Email meesholink-integration@meesho.com',
                'Subject: "Request for Meesho Seller API Credentials"',
                'Include your Supplier ID and registered email',
                'Meesho will respond with Client ID and Client Secret',
              ],
            },
          ].map(({ name, color, steps }) => (
            <div key={name} className={`bg-white rounded-xl border ${color} p-4`}>
              <p className="font-medium text-gray-800 mb-3">{name}</p>
              <ol className="space-y-1.5">
                {steps.map((s, i) => (
                  <li key={i} className="flex gap-2 text-xs text-gray-600">
                    <span className="flex-shrink-0 w-4 h-4 bg-gray-100 text-gray-500 text-xs font-bold rounded-full flex items-center justify-center mt-0.5">
                      {i + 1}
                    </span>
                    {s}
                  </li>
                ))}
              </ol>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function MarketplaceConnections() {
  const qc = useQueryClient()

  const { data: status, isLoading } = useQuery<StatusResponse>({
    queryKey: ['marketplace-status'],
    queryFn:  async () => { const { data } = await api.get('/marketplace/status'); return data },
    refetchInterval: 30_000,
  })

  // Fetch sync logs for each platform
  const syncLogs: Record<string, SyncLogEntry[]> = {}
  for (const platform of ['amazon', 'flipkart', 'myntra', 'meesho']) {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const { data = [] } = useQuery<SyncLogEntry[]>({
      queryKey: ['marketplace-sync-log', platform],
      queryFn:  async () => { const { data } = await api.get(`/marketplace/${platform}/sync-log?limit=10`); return data },
    })
    syncLogs[platform] = data
  }

  const defaultStatus: PlatformStatus = {
    connected: false, last_sync: null, last_status: null, last_rows: 0, last_message: '',
  }

  const makeConnectFn = (endpoint: string) => async (values: Record<string, string>) => {
    await api.post(endpoint, values)
    qc.invalidateQueries({ queryKey: ['marketplace-status'] })
  }

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Marketplace Connections</h1>
        <p className="text-sm text-gray-500 mt-1">
          Connect your selling accounts directly — data syncs automatically every day at 6:00 AM IST.
          Manual file uploads remain available as a fallback.
        </p>
      </div>

      {isLoading ? (
        <div className="text-sm text-gray-400 py-8 text-center">Loading…</div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {(Object.entries(PLATFORM_CONFIGS) as [keyof StatusResponse, typeof PLATFORM_CONFIGS.amazon][]).map(
            ([key, cfg]) => (
              <PlatformCard
                key={key}
                platform={key}
                icon={cfg.icon}
                name={cfg.name}
                accentColor={cfg.accentColor}
                status={status?.[key] ?? defaultStatus}
                syncLog={syncLogs[key] ?? []}
                connectFields={cfg.connectFields}
                helpUrl={cfg.helpUrl}
                helpText={cfg.helpText}
                onConnect={makeConnectFn(cfg.endpoint)}
              />
            )
          )}
        </div>
      )}

      <SetupGuide />
    </div>
  )
}
