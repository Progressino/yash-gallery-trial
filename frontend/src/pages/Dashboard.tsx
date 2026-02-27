import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import FileUpload from '../components/FileUpload'
import {
  uploadSkuMapping, uploadMtr, uploadMyntra, uploadMeesho,
  uploadFlipkart, uploadInventory, buildSales, getCoverage,
} from '../api/client'
import { useSession } from '../store/session'

type Toast = { type: 'success' | 'error'; msg: string }

export default function Dashboard() {
  const setCoverage = useSession((s) => s.setCoverage)
  const coverage    = useSession()
  const qc          = useQueryClient()

  const [toast, setToast]           = useState<Toast | null>(null)
  const [loading, setLoading]       = useState<Record<string, boolean>>({})
  const [buildingMsg, setBuildingMsg] = useState('')

  useQuery({
    queryKey: ['coverage'],
    queryFn: async () => { const c = await getCoverage(); setCoverage(c); return c },
    refetchInterval: 5000,
  })

  const showToast = (type: 'success' | 'error', msg: string) => {
    setToast({ type, msg })
    setTimeout(() => setToast(null), 5000)
  }

  const setL = (key: string, v: boolean) => setLoading(prev => ({ ...prev, [key]: v }))

  const refresh = async () => { const c = await getCoverage(); setCoverage(c); qc.invalidateQueries() }

  const handle = (key: string, fn: () => Promise<{ ok: boolean; message: string }>) => async () => {
    setL(key, true)
    try {
      const res = await fn()
      if (res.ok) { showToast('success', res.message); await refresh() }
      else showToast('error', res.message)
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Upload failed')
    } finally { setL(key, false) }
  }

  const handleBuildSales = async () => {
    setL('build', true)
    setBuildingMsg('Building combined sales dataset…')
    try {
      const res = await buildSales()
      if (res.ok) { showToast('success', res.message); await refresh() }
      else showToast('error', res.message)
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Build failed')
    } finally { setL('build', false); setBuildingMsg('') }
  }

  const anyLoaded = coverage.mtr || coverage.myntra || coverage.meesho || coverage.flipkart

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-[#002B5B]">📊 Dashboard</h2>
        <p className="text-gray-500 text-sm mt-1">Upload your data files to get started.</p>
      </div>

      {/* Toast */}
      {toast && (
        <div className={`fixed top-4 right-4 z-50 rounded-lg px-5 py-3 shadow-lg text-sm text-white max-w-sm
          ${toast.type === 'success' ? 'bg-green-600' : 'bg-red-600'}`}>
          {toast.msg}
        </div>
      )}

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard label="SKU Mapping"  value={coverage.sku_mapping ? '✅ Loaded' : '— Not loaded'} />
        <KpiCard label="MTR Rows"     value={coverage.mtr_rows > 0 ? coverage.mtr_rows.toLocaleString() : '—'} />
        <KpiCard label="Sales Rows"   value={coverage.sales_rows > 0 ? coverage.sales_rows.toLocaleString() : '—'} />
        <KpiCard label="Platforms"    value={[
          coverage.myntra && 'Myntra',
          coverage.meesho && 'Meesho',
          coverage.flipkart && 'Flipkart',
        ].filter(Boolean).join(', ') || '—'} />
      </div>

      {/* Tier 1 — Required */}
      <Section title="Tier 1 — Required">
        <UploadCard title="1️⃣ SKU Mapping" subtitle="Upload Excel (.xlsx)" loaded={coverage.sku_mapping}>
          <FileUpload
            label="Upload .xlsx"
            accept={{ 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'] }}
            onUpload={handle('sku', (file: File) => uploadSkuMapping(file))}
            uploading={loading['sku']}
          />
        </UploadCard>

        <UploadCard title="2️⃣ Amazon MTR" subtitle="Historical ZIP (all months)" loaded={coverage.mtr}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload .zip"
            accept={{ 'application/zip': ['.zip'] }}
            onUpload={handle('mtr', (file: File) => uploadMtr(file))}
            uploading={loading['mtr']}
          />
        </UploadCard>
      </Section>

      {/* Tier 1 — Platforms */}
      <Section title="Tier 1 — Platform History">
        <UploadCard title="🛍️ Myntra PPMP" subtitle="Master ZIP (all months)" loaded={coverage.myntra}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload .zip"
            accept={{ 'application/zip': ['.zip'] }}
            onUpload={handle('myntra', (file: File) => uploadMyntra(file))}
            uploading={loading['myntra']}
          />
        </UploadCard>

        <UploadCard title="🛒 Meesho" subtitle="Master ZIP (all months)" loaded={coverage.meesho}>
          <FileUpload
            label="Upload .zip"
            accept={{ 'application/zip': ['.zip'] }}
            onUpload={handle('meesho', (file: File) => uploadMeesho(file))}
            uploading={loading['meesho']}
          />
        </UploadCard>

        <UploadCard title="🟡 Flipkart" subtitle="Master ZIP (all months)" loaded={coverage.flipkart}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload .zip"
            accept={{ 'application/zip': ['.zip'] }}
            onUpload={handle('flipkart', (file: File) => uploadFlipkart(file))}
            uploading={loading['flipkart']}
          />
        </UploadCard>

        <UploadCard title="📦 Inventory" subtitle="Upload OMS + platform CSVs" loaded={coverage.inventory}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <InventoryUploader
            skuLoaded={coverage.sku_mapping}
            uploading={loading['inv']}
            onUpload={async (files) => {
              setL('inv', true)
              try {
                const res = await uploadInventory(files)
                if (res.ok) { showToast('success', res.message); await refresh() }
                else showToast('error', res.message)
              } catch (e: unknown) {
                showToast('error', e instanceof Error ? e.message : 'Upload failed')
              } finally { setL('inv', false) }
            }}
          />
        </UploadCard>
      </Section>

      {/* Build Sales */}
      {anyLoaded && (
        <div className="bg-white rounded-xl border border-gray-200 p-5 flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-[#002B5B]">🔄 Build Combined Sales Dataset</h3>
            <p className="text-sm text-gray-400 mt-0.5">
              Merges MTR + Myntra + Meesho + Flipkart into a single deduplicated sales_df.
              {coverage.sales && ` (${coverage.sales_rows.toLocaleString()} rows currently loaded)`}
            </p>
            {buildingMsg && <p className="text-xs text-blue-600 mt-1">{buildingMsg}</p>}
          </div>
          <button
            onClick={handleBuildSales}
            disabled={loading['build']}
            className="ml-4 px-5 py-2.5 rounded-lg text-sm font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-50 shrink-0"
          >
            {loading['build'] ? 'Building…' : coverage.sales ? '↻ Rebuild' : 'Build Sales'}
          </button>
        </div>
      )}
    </div>
  )
}

// ── Sub-components ─────────────────────────────────────────────

function KpiCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white rounded-xl border-l-4 border-[#002B5B] border border-gray-200 p-4 shadow-sm">
      <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">{label}</p>
      <p className="text-lg font-bold text-gray-800 mt-0.5 truncate">{value}</p>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">{title}</h3>
      <div className="grid grid-cols-2 gap-4">{children}</div>
    </div>
  )
}

function UploadCard({
  title, subtitle, loaded, children,
}: { title: string; subtitle: string; loaded: boolean; children: React.ReactNode }) {
  return (
    <div className={`bg-white rounded-xl border p-5 space-y-3 shadow-sm ${loaded ? 'border-green-300' : 'border-gray-200'}`}>
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-semibold text-[#002B5B] text-sm">{title}</h3>
          <p className="text-xs text-gray-400">{subtitle}</p>
        </div>
        {loaded && <span className="text-green-600 text-xs font-medium bg-green-50 px-2 py-0.5 rounded-full">✓ Loaded</span>}
      </div>
      {children}
    </div>
  )
}

function Warn({ children }: { children: React.ReactNode }) {
  return <p className="text-xs text-amber-600 bg-amber-50 rounded p-2">{children}</p>
}

function InventoryUploader({
  skuLoaded, uploading, onUpload,
}: {
  skuLoaded: boolean
  uploading: boolean
  onUpload: (files: { oms?: File; fk?: File; myntra?: File; amz?: File }) => Promise<void>
}) {
  const [files, setFiles] = useState<{ oms?: File; fk?: File; myntra?: File; amz?: File }>({})
  const set = (key: keyof typeof files) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setFiles(prev => ({ ...prev, [key]: e.target.files?.[0] }))
  }
  const hasAny = Object.values(files).some(Boolean)

  return (
    <div className="space-y-2">
      {(['oms', 'fk', 'myntra', 'amz'] as const).map(k => (
        <div key={k} className="flex items-center gap-2">
          <label className="text-xs text-gray-500 w-16 shrink-0">{k.toUpperCase()}</label>
          <input
            type="file" accept=".csv"
            onChange={set(k)}
            className="text-xs text-gray-600 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-gray-100"
          />
        </div>
      ))}
      <button
        onClick={() => onUpload(files)}
        disabled={!hasAny || !skuLoaded || uploading}
        className="w-full mt-1 py-2 rounded-lg text-xs font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-40"
      >
        {uploading ? 'Uploading…' : 'Upload Inventory'}
      </button>
    </div>
  )
}
