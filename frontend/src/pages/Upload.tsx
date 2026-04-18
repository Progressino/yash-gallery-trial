import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query'
import FileUpload from '../components/FileUpload'
import {
  uploadSkuMapping, uploadMtr, uploadMyntra, uploadMeesho,
  uploadFlipkart, uploadSnapdeal, uploadInventoryAuto, buildSales, getCoverage,
  uploadAmazonB2C, uploadAmazonB2B, uploadExistingPO, uploadDailyAuto,
  getDailySummary, getDailyUploads, deleteDailyUpload, clearPlatform,
  resetAllAppData, getDataQuality,
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
      setSkuMapGaps(res.unmapped_skus ?? [])
      if (res.ok) { showToast('success', res.message); await refresh() }
      else showToast('error', res.message)
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Build failed')
    } finally { setL('build', false); setBuildingMsg('') }
  }

  const [dailyDetected, setDailyDetected] = useState<string[]>([])
  const [skuMapGaps, setSkuMapGaps] = useState<string[]>([])
  const [resetClearTier3, setResetClearTier3] = useState(false)
  const [resetClearWarm, setResetClearWarm] = useState(true)
  const [qualityReport, setQualityReport] = useState<Awaited<ReturnType<typeof getDataQuality>> | null>(null)

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

  const handleResetAllAppData = async () => {
    const msg =
      'Remove ALL data from this session (mapping, platforms, inventory, sales)?' +
      (resetClearTier3 ? ' Tier-3 daily files on the server will be deleted too.' : '') +
      (resetClearWarm ? ' Server warm cache will be cleared.' : '') +
      ' Cloud GitHub cache is NOT deleted.'
    if (!window.confirm(msg)) return
    setL('reset_all', true)
    try {
      const res = await resetAllAppData({
        clearTier3Sqlite: resetClearTier3,
        clearWarmCache: resetClearWarm,
      })
      if (res.ok) {
        showToast('success', res.message)
        setQualityReport(null)
        await refresh()
        qc.invalidateQueries()
      } else showToast('error', res.message)
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Reset failed')
    } finally {
      setL('reset_all', false)
    }
  }

  const handleDataQuality = async () => {
    setL('quality', true)
    try {
      const r = await getDataQuality()
      setQualityReport(r)
      showToast('success', 'Quality report loaded — see below.')
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Could not load report')
    } finally {
      setL('quality', false)
    }
  }

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
        <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
          <p className="font-semibold text-[#002B5B] text-xs uppercase tracking-wide">Typical workflow</p>
          <ul className="mt-2 space-y-1.5 text-xs text-gray-600 list-disc list-inside">
            <li>
              <strong>Tier 1</strong> — Load SKU mapping, then <em>historical bulk</em> data (e.g. ~2 years: Amazon MTR ZIP/RAR,
              Myntra/Meesho/Flipkart/Snapdeal archives, inventory). Use this when onboarding or replacing base history.
            </li>
            <li>
              <strong>Tier 3</strong> — <em>Daily</em> file drops (auto-detect). Append recent reports without re-uploading full history;
              sales are rebuilt after each batch.
            </li>
          </ul>
        </div>
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
        <UploadCard
          title="1️⃣ SKU Mapping"
          subtitle="Master Yash map (~all panels) ships with the app. Upload your .xlsx to replace or extend it."
          loaded={coverage.sku_mapping}
        >
          <FileUpload
            label="Upload .xlsx"
            accept={{ 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'] }}
            onUpload={async (file: File) => {
              setL('sku', true)
              try {
                const res = await uploadSkuMapping(file)
                setSkuMapGaps(res.unmapped_skus ?? [])
                if (res.ok) {
                  showToast('success', res.message)
                  await refresh()
                } else showToast('error', res.message)
              } catch (e: unknown) {
                setSkuMapGaps([])
                showToast('error', e instanceof Error ? e.message : 'Upload failed')
              } finally {
                setL('sku', false)
              }
            }}
            uploading={loading['sku']}
          />
          {skuMapGaps.length > 0 && (
            <div className="mt-3 rounded border border-amber-300/80 bg-amber-50/90 p-3 text-sm text-amber-950 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-100">
              <div className="font-medium">
                Sales SKUs not in this mapping (missing as seller key or OMS value)
              </div>
              <p className="mt-1 text-xs opacity-90">
                Add or fix these on the master sheet, re-upload mapping, then rebuild sales if needed.
              </p>
              <ul className="mt-2 max-h-40 list-inside list-disc overflow-y-auto font-mono text-xs">
                {skuMapGaps.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </div>
          )}
        </UploadCard>

        <UploadCard title="2️⃣ Amazon" subtitle="MTR master ZIP or RAR — upload multiple; data stacks" loaded={coverage.mtr} rows={coverage.mtr_rows} onClear={handleClear('mtr')} clearing={loading['clear_mtr']}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload .zip or .rar"
            accept={{
              'application/zip': ['.zip'],
              'application/vnd.rar': ['.rar'],
              'application/x-rar-compressed': ['.rar'],
            }}
            onUpload={handle('mtr', (file: File) => uploadMtr(file))}
            uploading={loading['mtr']}
          />
        </UploadCard>
      </Section>

      {/* Tier 1 — Platforms */}
      <Section title="Tier 1 — Platform history (bulk / multi-year)">
        <UploadCard title="🛍️ Myntra PPMP" subtitle="Upload multiple company ZIPs — data stacks" loaded={coverage.myntra} rows={coverage.myntra_rows} onClear={handleClear('myntra')} clearing={loading['clear_myntra']}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload .zip"
            accept={{ 'application/zip': ['.zip'] }}
            onUpload={handle('myntra', (file: File) => uploadMyntra(file))}
            uploading={loading['myntra']}
          />
        </UploadCard>

        <UploadCard title="🛒 Meesho" subtitle="ZIP (TCS/ledger), Order CSV, or unified sales Excel (.xlsx/.xls) — select multiple" loaded={coverage.meesho} rows={coverage.meesho_rows} onClear={handleClear('meesho')} clearing={loading['clear_meesho']}>
          <FileUpload
            label="Upload .zip, .csv, .xlsx, or .xls (select multiple)"
            accept={{
              'application/zip': ['.zip'],
              'text/csv': ['.csv'],
              'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
              'application/vnd.ms-excel': ['.xls'],
              // Some browsers report Excel as generic binary when picking files
              'application/octet-stream': ['.xlsx', '.xls'],
            }}
            onUpload={handle('meesho', (file: File) => uploadMeesho(file))}
            uploading={loading['meesho']}
            multiple
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

        <UploadCard title="📦 Inventory" subtitle="Drop any mix: OMS CSV, Flipkart CSV, Myntra CSV, Amazon RAR — auto-detected" loaded={coverage.inventory}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <InventoryDropzone
            disabled={!coverage.sku_mapping}
            uploading={loading['inv']}
            onUpload={async (files) => {
              setL('inv', true)
              try {
                const res = await uploadInventoryAuto(files)
                if (res.ok) {
                  const debugStr = res.debug ? '\n' + JSON.stringify(res.debug, null, 2) : ''
                  showToast('success', res.message + debugStr)
                  await refresh()
                } else showToast('error', res.message)
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
      <Section title="Tier 3 — Daily orders (incremental; auto-detect)">
        <div className="col-span-2 bg-white rounded-xl border border-gray-200 p-5 shadow-sm space-y-3">
          <div>
            <h3 className="font-semibold text-[#002B5B] text-sm">📅 Daily order upload</h3>
            <p className="text-xs text-gray-400">
              For <strong>ongoing</strong> refreshes after Tier 1 history is loaded. Drop <strong>any mix</strong> of recent files —
              platform is auto-detected per file. Accepted: Amazon MTR/FBA CSV, Myntra PPMP CSV, Meesho CSV / ZIP / unified XLSX, Flipkart XLSX,
              Snapdeal paths containing <code className="text-gray-600">snapdeal</code>, RAR bundles. Sales dataset rebuilds automatically.
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

      {/* Reset & data verification */}
      <div>
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">Start fresh & verify</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-xl border border-red-200 bg-red-50/60 p-5 space-y-3">
            <h4 className="font-semibold text-red-900 text-sm">Clear all app data</h4>
            <p className="text-xs text-red-900/80 leading-relaxed">
              Wipes <strong>everything</strong> in this browser session: SKU map, all platform files, inventory,
              existing PO sheet imports, combined sales, and PO quarterly cache. Use this before a clean Tier‑1 re-upload.
              Your saved <strong>GitHub</strong> data cache is not deleted — use <strong>Save Cache</strong> after new uploads if you want the cloud updated.
            </p>
            <label className="flex items-center gap-2 text-xs text-red-900 cursor-pointer">
              <input
                type="checkbox"
                checked={resetClearWarm}
                onChange={e => setResetClearWarm(e.target.checked)}
              />
              Also clear <strong>server warm cache</strong> (recommended — otherwise other tabs may still show old data until reload)
            </label>
            <label className="flex items-center gap-2 text-xs text-red-900 cursor-pointer">
              <input
                type="checkbox"
                checked={resetClearTier3}
                onChange={e => setResetClearTier3(e.target.checked)}
              />
              Also delete <strong>Tier‑3 daily files</strong> stored on this server (SQLite)
            </label>
            <button
              type="button"
              onClick={() => void handleResetAllAppData()}
              disabled={loading['reset_all']}
              className="w-full py-2 rounded-lg text-sm font-semibold text-white bg-red-700 hover:bg-red-800 disabled:opacity-50"
            >
              {loading['reset_all'] ? 'Working…' : '🗑️ Clear all app data…'}
            </button>
          </div>
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-5 space-y-3">
            <h4 className="font-semibold text-emerald-900 text-sm">Check for duplicates / sanity</h4>
            <p className="text-xs text-emerald-900/80 leading-relaxed">
              Runs a read-only report: overlapping Amazon lines (Tier‑1 ZIP overlap), identical rows in unified sales,
              shipment totals by channel, and Tier‑3 file counts. Compare a sample SKU in{' '}
              <strong>SKU Deepdive</strong> (use <strong>one channel at a time</strong>) to your marketplace export.
            </p>
            <button
              type="button"
              onClick={() => void handleDataQuality()}
              disabled={loading['quality']}
              className="w-full py-2 rounded-lg text-sm font-semibold text-white bg-emerald-700 hover:bg-emerald-800 disabled:opacity-50"
            >
              {loading['quality'] ? 'Running…' : '📋 Run data quality report'}
            </button>
            {qualityReport && (
              <div className="mt-3 rounded-lg border border-emerald-200 bg-white p-3 text-xs space-y-2 max-h-72 overflow-y-auto">
                <QualityReportView report={qualityReport} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function QualityReportView({ report }: { report: Awaited<ReturnType<typeof getDataQuality>> }) {
  const amz = report.checks.amazon_mtr as {
    rows_in_session?: number
    rows_after_dedup_key?: number
    rows_collapsible?: number
  } | undefined
  const sales = report.checks.unified_sales_df as {
    rows?: number
    exact_duplicate_rows?: number
    shipment_units_sum?: number
    by_source?: { source: string; units: number }[]
  } | undefined
  const tier3 = report.checks.tier3_sqlite_summary as Record<
    string,
    { min_date?: string; max_date?: string; total_rows?: number; file_count?: number }
  > | undefined

  return (
    <div className="space-y-3 text-gray-800">
      <div>
        <p className="font-semibold text-gray-600 text-[11px] uppercase tracking-wide mb-1">How to verify</p>
        <ul className="list-disc list-inside text-gray-600 space-y-0.5 leading-relaxed">
          {report.hints.map((h, i) => (
            <li key={i}>{h}</li>
          ))}
        </ul>
      </div>
      {amz && (
        <div className="border-t border-gray-100 pt-2">
          <p className="font-semibold text-[#002B5B]">Amazon MTR (current session)</p>
          <p className="mt-0.5">
            {amz.rows_in_session?.toLocaleString()} rows ·{' '}
            <span
              className={
                (amz.rows_collapsible ?? 0) > 0 ? 'text-amber-700 font-medium' : 'text-green-700'
              }
            >
              {(amz.rows_collapsible ?? 0).toLocaleString()} would collapse
            </span>{' '}
            if dedup rules re-ran (overlapping ZIP lines).
          </p>
        </div>
      )}
      {sales && (
        <div className="border-t border-gray-100 pt-2">
          <p className="font-semibold text-[#002B5B]">Unified sales_df</p>
          <p className="mt-0.5">
            {sales.rows?.toLocaleString()} rows · {(sales.exact_duplicate_rows ?? 0).toLocaleString()} exact duplicate
            lines · {(sales.shipment_units_sum ?? 0).toLocaleString()} shipment units (all channels)
          </p>
          {sales.by_source && sales.by_source.length > 0 && (
            <ul className="mt-1 text-gray-600">
              {sales.by_source.map(r => (
                <li key={r.source}>
                  {r.source}: {r.units.toLocaleString()} units
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {tier3 && Object.keys(tier3).length > 0 && (
        <div className="border-t border-gray-100 pt-2">
          <p className="font-semibold text-[#002B5B]">Tier‑3 files on server (SQLite)</p>
          <ul className="mt-1 text-gray-600">
            {Object.entries(tier3).map(([plat, s]) => (
              <li key={plat}>
                {plat}: {s.file_count ?? 0} file(s), ~{(s.total_rows ?? 0).toLocaleString()} rows
              </li>
            ))}
          </ul>
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
      'application/vnd.ms-excel': ['.xls'],
      'application/zip': ['.zip'],
      'application/x-rar-compressed': ['.rar'],
      'application/vnd.rar': ['.rar'],
      'application/octet-stream': ['.xlsx', '.xls'],
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
                <span className="text-xs text-gray-400">Accepts .csv / .xlsx / .xls / .zip / .rar — platform auto-detected</span>
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
            <span className="text-xs text-gray-400">
              {uploads!.length} file{uploads!.length !== 1 ? 's' : ''} · last 30 per platform · row counts are parsed lines (not Excel grid rows); data range shows min–max dates in the file
            </span>
          </div>
          <div className="divide-y divide-gray-50 max-h-64 overflow-y-auto">
            {uploads!.map(u => {
              const meta = PLATFORM_LABELS[u.platform] ?? { label: u.platform, color: 'bg-gray-100 text-gray-700' }
              return (
                <div key={u.id} className="flex items-center gap-3 px-5 py-2.5 hover:bg-gray-50 transition-colors">
                  <span className={`text-xs font-medium px-2 py-0.5 rounded-full shrink-0 ${meta.color}`}>{meta.label}</span>
                  <span className="text-xs font-mono text-gray-500 shrink-0 w-28" title="Sort key date from filename (or first row date)">
                    {u.date_from && u.date_to
                      ? (u.date_from === u.date_to ? u.date_from : `${u.date_from}→${u.date_to}`)
                      : u.file_date}
                  </span>
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

function InventoryDropzone({ disabled, uploading, onUpload }: {
  disabled: boolean
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
      'application/x-rar-compressed': ['.rar'],
      'application/vnd.rar': ['.rar'],
      'application/octet-stream': ['.rar'],
    },
    multiple: true,
    disabled: uploading || disabled,
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
        className={`border-2 border-dashed rounded-lg p-5 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400 bg-white'}
          ${(uploading || disabled) ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        {uploading
          ? <p className="text-sm text-blue-600 animate-pulse">Uploading & detecting…</p>
          : isDragActive
            ? <p className="text-sm text-blue-600">Drop files here</p>
            : <p className="text-sm text-gray-500">
                Drag & drop inventory files here, or <span className="text-blue-600 underline">browse</span>
                <br /><span className="text-xs text-gray-400">OMS CSV, Flipkart CSV, Myntra CSV, Amazon RAR — auto-detected</span>
              </p>
        }
      </div>
      {queued.length > 0 && (
        <ul className="text-xs space-y-1">
          {queued.map(f => (
            <li key={f.name} className="flex items-center justify-between bg-gray-50 rounded px-2 py-1">
              <span className="truncate text-gray-700">{f.name}</span>
              <button onClick={() => remove(f.name)} className="ml-2 text-gray-400 hover:text-red-500">✕</button>
            </li>
          ))}
        </ul>
      )}
      <button
        onClick={submit}
        disabled={!queued.length || uploading || disabled}
        className="w-full py-2 rounded-lg text-xs font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-40"
      >
        {uploading ? 'Uploading…' : `Upload Inventory${queued.length ? ` (${queued.length} file${queued.length > 1 ? 's' : ''})` : ''}`}
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
