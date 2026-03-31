import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

interface PlatformStatus {
  connected:     boolean
  last_sync:     string | null
  last_status:   string | null
  last_rows:     number
  last_message:  string
}

interface StatusResponse {
  amazon:   PlatformStatus
  myntra:   PlatformStatus
  meesho:   PlatformStatus
  flipkart: PlatformStatus
  snapdeal: PlatformStatus
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
    const d = new Date(iso)
    return d.toLocaleString('en-IN', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })
  } catch { return iso }
}

function StatusBadge({ status }: { status: string | null }) {
  if (!status) return <span className="text-gray-400 text-xs">—</span>
  const map: Record<string, string> = {
    success: 'bg-green-100 text-green-700',
    partial: 'bg-yellow-100 text-yellow-700',
    error:   'bg-red-100   text-red-700',
  }
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${map[status] ?? 'bg-gray-100 text-gray-600'}`}>
      {status}
    </span>
  )
}

// ── Amazon Connect Form ────────────────────────────────────────────────────────

function AmazonConnectModal({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const [form, setForm] = useState({
    client_id:      '',
    client_secret:  '',
    refresh_token:  '',
    seller_id:      '',
    marketplace_id: 'A21TJRUUN4KGV',
  })
  const [err, setErr] = useState('')

  const { mutate: save, isPending } = useMutation({
    mutationFn: () => api.post('/marketplace/amazon/connect', form),
    onSuccess: () => { onSaved(); onClose() },
    onError:   (e: any) => setErr(e?.response?.data?.detail ?? 'Connection failed'),
  })

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm(f => ({ ...f, [k]: e.target.value }))

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-1">Connect Amazon Seller Central</h2>
        <p className="text-sm text-gray-500 mb-5">
          Enter your SP-API credentials. You can get these from{' '}
          <a
            href="https://sellercentral.amazon.in/apps/manage"
            target="_blank" rel="noreferrer"
            className="text-blue-600 underline"
          >
            Seller Central → Apps & Services
          </a>
          .
        </p>

        <div className="space-y-3">
          {[
            { key: 'client_id',      label: 'Client ID',      placeholder: 'amzn1.application-oa2-client.xxxxx' },
            { key: 'client_secret',  label: 'Client Secret',  placeholder: '••••••••••••••••', type: 'password' },
            { key: 'refresh_token',  label: 'Refresh Token',  placeholder: 'Atzr|IwEBIxxxxxxxx',  type: 'password' },
            { key: 'seller_id',      label: 'Seller ID',      placeholder: 'A1B2C3D4E5F6G7' },
            { key: 'marketplace_id', label: 'Marketplace ID', placeholder: 'A21TJRUUN4KGV' },
          ].map(({ key, label, placeholder, type }) => (
            <div key={key}>
              <label className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
              <input
                type={type ?? 'text'}
                value={(form as any)[key]}
                onChange={set(key)}
                placeholder={placeholder}
                className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-orange-400"
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
            onClick={() => { setErr(''); save() }}
            disabled={isPending || !form.client_id || !form.client_secret || !form.refresh_token || !form.seller_id}
            className="px-5 py-2 text-sm font-medium bg-orange-500 text-white rounded-lg hover:bg-orange-600 disabled:opacity-50 transition"
          >
            {isPending ? 'Verifying…' : 'Save & Verify'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Amazon Card ───────────────────────────────────────────────────────────────

function AmazonCard({ status, syncLog }: { status: PlatformStatus; syncLog: SyncLogEntry[] }) {
  const qc = useQueryClient()
  const [showModal, setShowModal] = useState(false)
  const [syncMsg, setSyncMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const [daysBack, setDaysBack] = useState(7)

  const { mutate: disconnect } = useMutation({
    mutationFn: () => api.delete('/marketplace/amazon/disconnect'),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['marketplace-status'] }),
  })

  const { mutate: syncNow, isPending: syncing } = useMutation({
    mutationFn: () => api.post(`/marketplace/amazon/sync?days_back=${daysBack}`),
    onSuccess: (res) => {
      setSyncMsg({ ok: true, text: res.data.message })
      setTimeout(() => { setSyncMsg(null); qc.invalidateQueries({ queryKey: ['marketplace-sync-log'] }) }, 4000)
    },
    onError: (e: any) => setSyncMsg({ ok: false, text: e?.response?.data?.detail ?? 'Sync failed' }),
  })

  return (
    <>
      {showModal && (
        <AmazonConnectModal
          onClose={() => setShowModal(false)}
          onSaved={() => qc.invalidateQueries({ queryKey: ['marketplace-status'] })}
        />
      )}

      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-orange-50 flex items-center justify-center text-xl">🟠</div>
            <div>
              <h3 className="font-semibold text-gray-900">Amazon Seller Central</h3>
              <p className="text-xs text-gray-500">via SP-API (Reports API)</p>
            </div>
          </div>
          <span className={`flex items-center gap-1.5 text-xs font-medium px-3 py-1 rounded-full ${
            status.connected ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${status.connected ? 'bg-green-500' : 'bg-gray-400'}`} />
            {status.connected ? 'Connected' : 'Not connected'}
          </span>
        </div>

        {/* Last sync info */}
        {status.connected && (
          <div className="grid grid-cols-2 gap-3 mb-4 p-3 bg-gray-50 rounded-xl text-xs">
            <div>
              <span className="text-gray-400">Last sync</span>
              <p className="text-gray-700 font-medium mt-0.5">{fmtDate(status.last_sync)}</p>
            </div>
            <div>
              <span className="text-gray-400">Status</span>
              <p className="mt-0.5"><StatusBadge status={status.last_status} /></p>
            </div>
            <div>
              <span className="text-gray-400">Rows added</span>
              <p className="text-gray-700 font-medium mt-0.5">{status.last_rows.toLocaleString()}</p>
            </div>
            <div>
              <span className="text-gray-400">Auto-sync</span>
              <p className="text-gray-700 font-medium mt-0.5">Daily 6:00 AM IST</p>
            </div>
          </div>
        )}

        {/* Action buttons */}
        <div className="flex flex-wrap gap-2">
          {status.connected ? (
            <>
              <div className="flex items-center gap-2">
                <select
                  value={daysBack}
                  onChange={e => setDaysBack(Number(e.target.value))}
                  className="border border-gray-200 rounded-lg text-xs px-2 py-1.5 text-gray-600 focus:outline-none"
                >
                  <option value={1}>Last 1 day</option>
                  <option value={3}>Last 3 days</option>
                  <option value={7}>Last 7 days</option>
                  <option value={14}>Last 14 days</option>
                  <option value={30}>Last 30 days</option>
                </select>
                <button
                  onClick={() => syncNow()}
                  disabled={syncing}
                  className="px-4 py-1.5 text-xs font-medium bg-orange-500 text-white rounded-lg hover:bg-orange-600 disabled:opacity-50 transition"
                >
                  {syncing ? 'Starting…' : '⚡ Sync Now'}
                </button>
              </div>
              <button
                onClick={() => setShowModal(true)}
                className="px-4 py-1.5 text-xs font-medium bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 transition"
              >
                ✏️ Edit Credentials
              </button>
              <button
                onClick={() => { if (confirm('Remove Amazon connection?')) disconnect() }}
                className="px-4 py-1.5 text-xs font-medium bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition"
              >
                🔌 Disconnect
              </button>
            </>
          ) : (
            <button
              onClick={() => setShowModal(true)}
              className="px-5 py-2 text-sm font-medium bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition"
            >
              🔗 Connect Amazon
            </button>
          )}
        </div>

        {syncMsg && (
          <p className={`mt-3 text-xs rounded-lg px-3 py-2 ${syncMsg.ok ? 'bg-blue-50 text-blue-700' : 'bg-red-50 text-red-700'}`}>
            {syncMsg.text}
          </p>
        )}

        {/* Sync history */}
        {syncLog.length > 0 && (
          <div className="mt-4">
            <p className="text-xs font-medium text-gray-500 mb-2">Recent Sync History</p>
            <div className="space-y-1.5 max-h-48 overflow-y-auto">
              {syncLog.slice(0, 8).map(entry => (
                <div key={entry.id} className="flex items-center gap-2 text-xs text-gray-600 bg-gray-50 rounded-lg px-3 py-2">
                  <StatusBadge status={entry.status} />
                  <span className="text-gray-400">{fmtDate(entry.synced_at)}</span>
                  <span className="ml-auto font-medium">{entry.rows_added.toLocaleString()} rows</span>
                  {entry.date_from && (
                    <span className="text-gray-400">{entry.date_from} → {entry.date_to}</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  )
}

// ── Coming Soon Card ──────────────────────────────────────────────────────────

function ComingSoonCard({ icon, name, note }: { icon: string; name: string; note: string }) {
  return (
    <div className="bg-white rounded-2xl border border-dashed border-gray-200 p-6 opacity-60">
      <div className="flex items-center gap-3 mb-3">
        <div className="w-10 h-10 rounded-xl bg-gray-50 flex items-center justify-center text-xl">{icon}</div>
        <div>
          <h3 className="font-semibold text-gray-700">{name}</h3>
          <p className="text-xs text-gray-400">{note}</p>
        </div>
      </div>
      <span className="inline-flex items-center text-xs text-gray-400 bg-gray-100 px-3 py-1 rounded-full">
        Coming soon
      </span>
    </div>
  )
}

// ── How It Works Section ──────────────────────────────────────────────────────

function HowItWorks() {
  return (
    <div className="bg-blue-50 rounded-2xl border border-blue-100 p-6">
      <h3 className="font-semibold text-blue-900 mb-3">📋 How to get Amazon SP-API credentials</h3>
      <ol className="space-y-2 text-sm text-blue-800">
        {[
          'Go to sellercentral.amazon.in → Apps & Services → Develop Apps',
          'Register as an SP-API developer (approval takes 1–3 business days)',
          'Create a Private Seller Application (self-authorised — no Appstore listing needed)',
          'Authorise the app under your seller account to get the Refresh Token',
          'Copy the Client ID, Client Secret, Refresh Token, and Seller ID',
          'Paste them into the "Connect Amazon" form above',
        ].map((step, i) => (
          <li key={i} className="flex gap-3">
            <span className="flex-shrink-0 w-5 h-5 bg-blue-200 text-blue-800 text-xs font-bold rounded-full flex items-center justify-center mt-0.5">
              {i + 1}
            </span>
            <span>{step}</span>
          </li>
        ))}
      </ol>
      <p className="mt-4 text-xs text-blue-600 bg-blue-100 rounded-lg px-3 py-2">
        Once connected, the app will automatically pull your MTR (tax report) data every day at 6:00 AM IST.
        You can also trigger a manual sync at any time.
      </p>
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function MarketplaceConnections() {
  const { data: status, isLoading } = useQuery<StatusResponse>({
    queryKey: ['marketplace-status'],
    queryFn: async () => { const { data } = await api.get('/marketplace/status'); return data },
    refetchInterval: 30_000,
  })

  const { data: syncLog = [] } = useQuery<SyncLogEntry[]>({
    queryKey: ['marketplace-sync-log'],
    queryFn: async () => { const { data } = await api.get('/marketplace/amazon/sync-log?limit=20'); return data },
  })

  const amazonStatus: PlatformStatus = status?.amazon ?? {
    connected: false, last_sync: null, last_status: null, last_rows: 0, last_message: '',
  }

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Marketplace Connections</h1>
        <p className="text-sm text-gray-500 mt-1">
          Connect your selling accounts directly — no more manual file uploads.
          Data syncs automatically every day.
        </p>
      </div>

      {isLoading ? (
        <div className="text-sm text-gray-400 py-8 text-center">Loading connection status…</div>
      ) : (
        <div className="space-y-4">
          <AmazonCard status={amazonStatus} syncLog={syncLog} />

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <ComingSoonCard icon="🛍️" name="Myntra"   note="Requires account manager API access" />
            <ComingSoonCard icon="🛒" name="Meesho"   note="Email request to Meesho team" />
            <ComingSoonCard icon="🟡" name="Flipkart" note="Partner verification required" />
          </div>
        </div>
      )}

      <HowItWorks />
    </div>
  )
}
