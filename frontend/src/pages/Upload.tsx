import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query'
import FileUpload from '../components/FileUpload'
import {
  uploadSkuMapping, uploadMtr, uploadMyntra, uploadMeesho,
  uploadFlipkart, uploadSnapdeal, uploadInventory, buildSales, getCoverage,
  uploadAmazonB2C, uploadAmazonB2B, uploadExistingPO, uploadDailyAuto,
  getDailySummary, getDailyUploads, deleteDailyUpload, clearPlatform,
  type DailyUpload, type DailySummary,
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

  const handleClear = (platform: string) => async () => {
    setL(`clear_${platform}`, true)
    try {
      const res = await clearPlatform(platform)
      if (res.ok) { showToast('success', res.message); await refresh() }
      else showToast('error', res.message)
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Clear failed')
    } finally { setL(`clear_${platform}`, false) }
  }

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
        <KpiCard label="Amazon Rows"  value={coverage.mtr_rows > 0 ? coverage.mtr_rows.toLocaleString() : '—'} />
        <KpiCard label="Sales Rows"   value={coverage.sales_rows > 0 ? coverage.sales_rows.toLocaleString() : '—'} />
        <KpiCard label="Platforms"    value={[
          coverage.myntra && 'Myntra',
          coverage.meesho && 'Meesho',
          coverage.snapdeal && 'Snapdeal',
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

        <UploadCard title="2️⃣ Amazon" subtitle="Upload multiple company ZIPs — data stacks" loaded={coverage.mtr} rows={coverage.mtr_rows} onClear={handleClear('mtr')} clearing={loading['clear_mtr']}>
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
        <UploadCard title="🛍️ Myntra PPMP" subtitle="Upload multiple company ZIPs — data stacks" loaded={coverage.myntra} rows={coverage.myntra_rows} onClear={handleClear('myntra')} clearing={loading['clear_myntra']}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload .zip"
            accept={{ 'application/zip': ['.zip'] }}
            onUpload={handle('myntra', (file: File) => uploadMyntra(file))}
            uploading={loading['myntra']}
          />
        </UploadCard>

        <UploadCard title="🛒 Meesho" subtitle="Upload multiple company ZIPs — data stacks" loaded={coverage.meesho} rows={coverage.meesho_rows} onClear={handleClear('meesho')} clearing={loading['clear_meesho']}>
          <FileUpload
            label="Upload .zip"
            accept={{ 'application/zip': ['.zip'] }}
            onUpload={handle('meesho', (file: File) => uploadMeesho(file))}
            uploading={loading['meesho']}
          />
        </UploadCard>

        <UploadCard title="🟡 Flipkart" subtitle="Upload multiple company ZIPs — data stacks" loaded={coverage.flipkart} rows={coverage.flipkart_rows} onClear={handleClear('flipkart')} clearing={loading['clear_flipkart']}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload .zip"
            accept={{ 'application/zip': ['.zip'] }}
            onUpload={handle('flipkart', (file: File) => uploadFlipkart(file))}
            uploading={loading['flipkart']}
          />
        </UploadCard>

        <UploadCard title="🔴 Snapdeal" subtitle="OMS order reports (CSV/ZIP) or AG/PE/YG ZIPs" loaded={coverage.snapdeal} rows={coverage.snapdeal_rows} onClear={handleClear('snapdeal')} clearing={loading['clear_snapdeal']}>
          <FileUpload
            label="Upload files (select multiple)"
            accept={{ 'application/zip': ['.zip'], 'text/csv': ['.csv'], 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'] }}
            onUpload={handle('snapdeal', (file: File) => uploadSnapdeal(file))}
            uploading={loading['snapdeal']}
            multiple={true}
          />
        </UploadCard>

        <UploadCard title="📦 Inventory" subtitle="OMS/FK/Myntra CSVs + Amazon RAR (Amazon SELLABLE, FBA in-transit, Myntra other WH, OMS+buffer stock, combos)" loaded={coverage.inventory}>
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

        <UploadCard title="🗜️ Monthly Sales RAR" subtitle="Drop Monthly.rar — Amazon, Flipkart, Meesho, Myntra auto-detected inside" loaded={false}>
          <div className="space-y-2">
            <MonthlyRarUploader
              uploading={loading['monthly_rar']}
              onUpload={async (files) => {
                setL('monthly_rar', true)
                try {
                  const res = await uploadDailyAuto(files)
                  if (res.ok) { showToast('success', res.message); await refresh() }
                  else showToast('error', res.message)
                } catch (e: unknown) {
                  showToast('error', e instanceof Error ? e.message : 'Upload failed')
                } finally { setL('monthly_rar', false) }
              }}
            />
          </div>
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
            onUpload={async (files) => { await handleDailyAuto(files); qc.invalidateQueries({ queryKey: ['daily-summary'] }); qc.invalidateQueries({ queryKey: ['daily-uploads'] }) }}
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
        <div className="col-span-2">
          <DailyHistory />
        </div>
      </Section>

      {/* Build Sales */}
      {anyLoaded && (
        <div className="bg-white rounded-xl border border-gray-200 p-5 flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-[#002B5B]">🔄 Build Combined Sales Dataset</h3>
            <p className="text-sm text-gray-400 mt-0.5">
              Merges MTR + Myntra + Meesho + Flipkart + Snapdeal into a single deduplicated sales_df.
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
  title, subtitle, loaded, rows, onClear, clearing, children,
}: {
  title: string; subtitle: string; loaded: boolean
  rows?: number; onClear?: () => void; clearing?: boolean
  children: React.ReactNode
}) {
  return (
    <div className={`bg-white rounded-xl border p-5 space-y-3 shadow-sm ${loaded ? 'border-green-300' : 'border-gray-200'}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="font-semibold text-[#002B5B] text-sm">{title}</h3>
          <p className="text-xs text-gray-400">{subtitle}</p>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {loaded && rows !== undefined && rows > 0 && (
            <span className="text-green-700 text-xs font-medium bg-green-50 px-2 py-0.5 rounded-full">
              ✓ {rows.toLocaleString()} rows
            </span>
          )}
          {loaded && onClear && (
            <button
              onClick={onClear}
              disabled={clearing}
              title="Clear this platform's data"
              className="text-xs text-gray-400 hover:text-red-500 border border-gray-200 hover:border-red-300 px-1.5 py-0.5 rounded transition-colors disabled:opacity-40"
            >
              {clearing ? '…' : '✕ Clear'}
            </button>
          )}
        </div>
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
      'application/x-rar-compressed': ['.rar'],
      'application/vnd.rar': ['.rar'],
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
                <span className="text-xs text-gray-400">Accepts .csv / .xlsx / .zip / .rar — platform auto-detected</span>
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

const PLATFORM_LABELS: Record<string, { label: string; color: string }> = {
  amazon:   { label: 'Amazon',   color: 'bg-orange-100 text-orange-700' },
  myntra:   { label: 'Myntra',   color: 'bg-pink-100 text-pink-700' },
  meesho:   { label: 'Meesho',   color: 'bg-purple-100 text-purple-700' },
  flipkart: { label: 'Flipkart', color: 'bg-yellow-100 text-yellow-700' },
}

function DailyHistory() {
  const qc = useQueryClient()

  const { data: summary, isLoading: summaryLoading } = useQuery<DailySummary>({
    queryKey: ['daily-summary'],
    queryFn: getDailySummary,
    refetchInterval: 10000,
  })

  const { data: uploads, isLoading: uploadsLoading } = useQuery<DailyUpload[]>({
    queryKey: ['daily-uploads'],
    queryFn: getDailyUploads,
    refetchInterval: 10000,
  })

  const deleteMut = useMutation({
    mutationFn: (id: number) => deleteDailyUpload(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['daily-summary'] })
      qc.invalidateQueries({ queryKey: ['daily-uploads'] })
    },
  })

  const hasSummary = summary && Object.keys(summary).length > 0
  const hasUploads = uploads && uploads.length > 0

  if (summaryLoading && uploadsLoading) {
    return <p className="text-xs text-gray-400 py-2">Loading saved daily data…</p>
  }

  if (!hasSummary && !hasUploads) {
    return (
      <div className="bg-gray-50 rounded-xl border border-dashed border-gray-200 p-4 text-center">
        <p className="text-xs text-gray-400">No daily uploads saved yet. Files uploaded above are persisted automatically.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Per-platform summary cards */}
      {hasSummary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Object.entries(summary!).map(([platform, s]) => {
            const meta = PLATFORM_LABELS[platform] ?? { label: platform, color: 'bg-gray-100 text-gray-700' }
            return (
              <div key={platform} className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${meta.color}`}>{meta.label}</span>
                  <span className="text-xs text-gray-400">{s.file_count} file{s.file_count !== 1 ? 's' : ''}</span>
                </div>
                <p className="text-lg font-bold text-gray-800">{s.total_rows.toLocaleString()}</p>
                <p className="text-xs text-gray-400">rows saved</p>
                <p className="text-xs text-gray-500 mt-1 truncate">
                  {s.min_date === s.max_date ? s.min_date : `${s.min_date} → ${s.max_date}`}
                </p>
              </div>
            )
          })}
        </div>
      )}

      {/* Upload history table */}
      {hasUploads && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
            <h4 className="text-sm font-semibold text-[#002B5B]">Saved Daily Uploads</h4>
            <span className="text-xs text-gray-400">{uploads!.length} file{uploads!.length !== 1 ? 's' : ''} · last 30 per platform</span>
          </div>
          <div className="divide-y divide-gray-50 max-h-64 overflow-y-auto">
            {uploads!.map(u => {
              const meta = PLATFORM_LABELS[u.platform] ?? { label: u.platform, color: 'bg-gray-100 text-gray-700' }
              return (
                <div key={u.id} className="flex items-center gap-3 px-5 py-2.5 hover:bg-gray-50 transition-colors">
                  <span className={`text-xs font-medium px-2 py-0.5 rounded-full shrink-0 ${meta.color}`}>{meta.label}</span>
                  <span className="text-xs font-mono text-gray-500 shrink-0 w-24">{u.file_date}</span>
                  <span className="text-xs text-gray-600 truncate flex-1 min-w-0">{u.filename}</span>
                  <span className="text-xs text-gray-400 shrink-0">{u.rows.toLocaleString()} rows</span>
                  <button
                    onClick={() => deleteMut.mutate(u.id)}
                    disabled={deleteMut.isPending}
                    title="Delete this upload"
                    className="text-gray-300 hover:text-red-500 transition-colors shrink-0 ml-1 disabled:opacity-40"
                  >
                    🗑
                  </button>
                </div>
              )
            })}
          </div>
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
  onUpload: (files: { oms?: File[]; fk?: File; myntra?: File; amz?: File }) => Promise<void>
}) {
  const [files, setFiles] = useState<{ oms?: File[]; fk?: File; myntra?: File; amz?: File }>({})
  const set = (key: keyof typeof files) => (e: React.ChangeEvent<HTMLInputElement>) => {
    if (key === 'oms') {
      const selected = e.target.files ? Array.from(e.target.files) : undefined
      setFiles(prev => ({ ...prev, oms: selected }))
    } else {
      setFiles(prev => ({ ...prev, [key]: e.target.files?.[0] }))
    }
  }
  const hasAny = !!(files.oms?.length || files.fk || files.myntra || files.amz)

  return (
    <div className="space-y-2">
      {(['oms', 'fk', 'myntra', 'amz'] as const).map(k => (
        <div key={k} className="flex items-center gap-2">
          <label className="text-xs text-gray-500 w-16 shrink-0">{k.toUpperCase()}</label>
          <input
            type="file"
            accept={k === 'amz' ? '*' : '.csv'}
            multiple={k === 'oms'}
            onChange={set(k)}
            className="text-xs text-gray-600 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-gray-100"
          />
          {k === 'oms' && files.oms && files.oms.length > 1 && (
            <span className="text-xs text-blue-600">{files.oms.length} files</span>
          )}
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

function MonthlyRarUploader({ uploading, onUpload }: {
  uploading: boolean
  onUpload: (files: File[]) => Promise<void>
}) {
  const [file, setFile] = useState<File | null>(null)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] ?? null)
  }

  const submit = async () => {
    if (!file) return
    await onUpload([file])
    setFile(null)
  }

  return (
    <div className="space-y-2">
      <input
        type="file"
        accept="*"
        onChange={handleChange}
        disabled={uploading}
        className="text-xs text-gray-600 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:bg-gray-100 w-full"
      />
      {file && (
        <p className="text-xs text-gray-500 truncate">📦 {file.name}</p>
      )}
      <button
        onClick={submit}
        disabled={!file || uploading}
        className="w-full py-2 rounded-lg text-xs font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-40"
      >
        {uploading ? 'Processing RAR…' : '⬆ Upload Monthly RAR'}
      </button>
    </div>
  )
}
