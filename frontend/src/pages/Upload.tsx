import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import FileUpload from '../components/FileUpload'
import {
  uploadSkuMapping, uploadMtr, uploadMyntra, uploadMeesho,
  uploadFlipkart, uploadInventory, buildSales, getCoverage,
  uploadAmazonB2C, uploadAmazonB2B, uploadExistingPO, uploadDailyAuto,
} from '../api/client'
import { useSession } from '../store/session'

type Toast = { type: 'success' | 'error'; msg: string }

export default function Upload() {
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

  const handle = (key: string, fn: (file: File) => Promise<{ ok: boolean; message: string }>) => async (file: File) => {
    setL(key, true)
    try {
      const res = await fn(file)
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

  const [dailyDetected, setDailyDetected] = useState<string[]>([])

  const handleDailyAuto = async (files: File[]) => {
    setL('daily', true)
    try {
      const res = await uploadDailyAuto(files)
      if (res.ok) {
        setDailyDetected(res.detected_platforms ?? [])
        showToast('success', res.message)
        await refresh()
      } else {
        showToast('error', res.message)
      }
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Upload failed')
    } finally { setL('daily', false) }
  }

  const anyLoaded = coverage.mtr || coverage.myntra || coverage.meesho || coverage.flipkart

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-[#002B5B]">📁 Upload Data</h2>
        <p className="text-gray-500 text-sm mt-1">Manage your data files and build the sales dataset.</p>
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

      {/* Tier 2 — Amazon Individual CSVs + Existing PO */}
      <Section title="Tier 2 — Amazon Orders & PO Pipeline">
        <UploadCard title="📄 Amazon MTR CSV" subtitle="Single-month MTR or FBA shipment CSV" loaded={false}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload MTR .csv"
            accept={{ 'text/csv': ['.csv'] }}
            onUpload={handle('b2c', (file: File) => uploadAmazonB2C(file))}
            uploading={loading['b2c']}
          />
        </UploadCard>

        <UploadCard title="📋 Amazon B2B CSV" subtitle="Single-month B2B report CSV" loaded={false}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload B2B .csv"
            accept={{ 'text/csv': ['.csv'] }}
            onUpload={handle('b2b', (file: File) => uploadAmazonB2B(file))}
            uploading={loading['b2b']}
          />
        </UploadCard>

        <UploadCard title="📦 Existing PO Sheet" subtitle="Open/pending POs (XLSX or CSV)" loaded={coverage.existing_po}>
          <FileUpload
            label="Upload PO Sheet"
            accept={{
              'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
              'text/csv': ['.csv'],
            }}
            onUpload={handle('existingpo', (file: File) => uploadExistingPO(file))}
            uploading={loading['existingpo']}
          />
        </UploadCard>
      </Section>

      {/* Tier 3 — Daily Orders */}
      <Section title="Tier 3 — Daily Orders (auto-detect)">
        <div className="col-span-2 bg-white rounded-xl border border-gray-200 p-5 shadow-sm space-y-3">
          <div>
            <h3 className="font-semibold text-[#002B5B] text-sm">📅 Daily Order Upload</h3>
            <p className="text-xs text-gray-400">
              Drop <strong>any mix</strong> of daily report files — platform is auto-detected from each file.
              Accepted: Amazon MTR/FBA CSV, Myntra PPMP CSV, Meesho CSV or ZIP, Flipkart Sales Report or Payment XLSX.
              Sales dataset is rebuilt automatically after upload.
            </p>
          </div>
          <DailyDropzone
            uploading={loading['daily']}
            onUpload={handleDailyAuto}
          />
          {dailyDetected.length > 0 && (
            <p className="text-xs text-green-600">
              ✓ Last upload detected: {dailyDetected.map(d => d.split('(')[0].trim()).join(', ')}
            </p>
          )}
          {coverage.daily_orders && (
            <p className="text-xs text-blue-600">Daily orders loaded ✓ — included in sales dataset.</p>
          )}
        </div>
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

function DailyDropzone({ uploading, onUpload }: {
  uploading: boolean
  onUpload: (files: File[]) => Promise<void>
}) {
  const [queued, setQueued] = useState<File[]>([])

  const onDrop = useCallback((accepted: File[]) => {
    setQueued(prev => {
      const names = new Set(prev.map(f => f.name))
      return [...prev, ...accepted.filter(f => !names.has(f.name))]
    })
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/zip': ['.zip'],
    },
    multiple: true,
    disabled: uploading,
  })

  const remove = (name: string) => setQueued(prev => prev.filter(f => f.name !== name))

  const submit = async () => {
    if (!queued.length) return
    await onUpload(queued)
    setQueued([])
  }

  return (
    <div className="space-y-2">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400 bg-white'}
          ${uploading ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        {uploading
          ? <p className="text-sm text-blue-600 animate-pulse">Uploading & detecting…</p>
          : isDragActive
            ? <p className="text-sm text-blue-600">Drop files here</p>
            : <p className="text-sm text-gray-500">
                Drag & drop daily report files here, or{' '}
                <span className="text-blue-600 underline">browse</span>
                <br />
                <span className="text-xs text-gray-400">Accepts .csv / .xlsx / .zip — platform auto-detected</span>
              </p>
        }
      </div>
      {queued.length > 0 && (
        <div className="space-y-1">
          {queued.map(f => (
            <div key={f.name} className="flex items-center justify-between bg-gray-50 rounded px-3 py-1.5 text-xs">
              <span className="text-gray-700 truncate max-w-xs">{f.name}</span>
              <button onClick={() => remove(f.name)} className="text-gray-400 hover:text-red-500 ml-2 shrink-0">✕</button>
            </div>
          ))}
          <button
            onClick={submit}
            disabled={uploading}
            className="w-full mt-1 py-2 rounded-lg text-xs font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-40"
          >
            {uploading ? 'Uploading…' : `⬆ Upload ${queued.length} file${queued.length > 1 ? 's' : ''}`}
          </button>
        </div>
      )}
    </div>
  )
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
