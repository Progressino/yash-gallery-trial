import { useState, useCallback, useEffect, useRef } from 'react'
import { useDropzone, type FileRejection } from 'react-dropzone'
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query'
import FileUpload from '../components/FileUpload'
import RestoreProgressPanel from '../components/RestoreProgressPanel'
import type { RestoreProgressTick } from '../api/client'
import {
  uploadSkuMapping, uploadMtr, uploadMyntra, uploadMeesho,
  uploadFlipkart, uploadSnapdeal, uploadInventoryAuto, waitForInventoryUpload, resetStuckInventoryUpload, buildSales, getCoverage, restoreFullFromServer,
  uploadAmazonB2C, uploadAmazonB2B, uploadExistingPO, uploadFinishingReceipt, uploadDailyAuto, uploadPoReturnsImport,
  uploadPoSkuStatusLead, uploadPoDailyInventoryHistoryFile, uploadPoManualIntransitSheet,
  type ManualIntransitParseReport,
  waitForDailyAutoIngest, waitForReturnsImport, waitForSalesRebuild, waitForTier1Bulk, verifyDailyUpload,
  dailyAutoSummaryFromCoverage, dailyAutoSummaryFromUpload, formatDailyAutoCompleteToast,
  type DailyUploadVerifyResponse,
  type CoverageResponse,
  getDailySummary, getDailyUploads, deleteDailyUpload, clearPlatform,
  resetAllAppData, getDataQuality, invalidateDataQueries,
  type DailyUpload, type DailySummary, type UploadResponse,
} from '../api/client'
import { useSession } from '../store/session'
import { usePOStore } from '../store/po'
import { useUploadActivity } from '../store/uploadActivity'
import { useAuth, mayUploadHistorical, mayResetSharedData, mayUploadPoBaseline, mayDeleteDailyUploadFile, mayClearPlatformData } from '../store/auth'

type Toast = { type: 'success' | 'error'; msg: string }
type UploadAlert = {
  at: string
  complete: boolean
  title: string
  parsed?: number
  kept?: number
  dropped?: number
  droppedReasons: string[]
  validationWarnings: string[]
  fileResults?: Array<{
    filename: string
    status: 'saved' | 'skipped' | 'loaded' | 'error'
    reason?: string
    platform?: string
    category?: string
    rows?: number
  }>
}
type InventoryAmazonDisclaimer = {
  raw_total_units?: number
  excluded_non_sellable_units?: number
  excluded_znne_units?: number
  excluded_older_date_units?: number
  latest_report_units?: number
  latest_report_date?: string
}

export default function Upload() {
  const setCoverage = useSession((s) => s.setCoverage)
  const coverage    = useSession()
  const qc          = useQueryClient()
  const authUser    = useAuth((s) => s.user)
  const allowHistorical = mayUploadHistorical(authUser)
  const allowReset = mayResetSharedData(authUser)
  const allowAdminBaseline = mayUploadPoBaseline(authUser)
  const mayDeleteDailyFiles = mayDeleteDailyUploadFile(authUser)
  const mayClearPlatform = mayClearPlatformData(authUser)
  const showAdminTab = allowHistorical
  const [uploadTab, setUploadTab] = useState<'daily' | 'admin'>('daily')

  const [toast, setToast]           = useState<Toast | null>(null)
  const [loading, setLoading]       = useState<Record<string, boolean>>({})
  const [buildingMsg, setBuildingMsg] = useState('')
  const [chunkProgress, setChunkProgress] = useState<{ pct: number; sent: number; total: number; msg: string } | null>(null)
  const [invProgress, setInvProgress] = useState<{ pct: number; msg: string; phase: 'upload' | 'parse' } | null>(null)
  const invWaitAbortRef = useRef(false)
  const [existingPoProgress, setExistingPoProgress] = useState<{
    pct: number
    msg: string
    phase: 'upload' | 'parse' | 'refresh'
    startedAt: number
  } | null>(null)
  const [uploadAlertsBySource, setUploadAlertsBySource] = useState<Record<string, UploadAlert>>({})
  const [restoreProgress, setRestoreProgress] = useState<RestoreProgressTick>({
    message: '',
    progress: 0,
    step: '',
  })
  const uploadBegin = useUploadActivity(s => s.begin)
  const uploadEnd = useUploadActivity(s => s.end)
  const restoreBusy = coverage.session_restore_status === 'running'
  const showRestorePanel =
    loading['refresh_all'] || restoreBusy || (restoreProgress.progress > 0 && restoreProgress.progress < 100)
  const uploadBusy =
    Object.values(loading).some(Boolean) ||
    !!buildingMsg ||
    !!invProgress ||
    !!existingPoProgress ||
    restoreBusy

  // Shared coverage polling: CoverageProvider in App.tsx (light, 3s while jobs run).

  // Keep progress bar in sync when coverage polls (not only during restoreFullFromServer callback).
  useEffect(() => {
    if (coverage.session_restore_status !== 'running') return
    setRestoreProgress({
      message: coverage.session_restore_message || 'Restoring from server…',
      progress: coverage.session_restore_progress ?? 0,
      step: coverage.session_restore_step || 'queued',
    })
  }, [
    coverage.session_restore_status,
    coverage.session_restore_message,
    coverage.session_restore_progress,
    coverage.session_restore_step,
  ])

  const handleRefreshAllData = async () => {
    const { operationalDataComplete } = await import('../lib/localSessionHint')
    if (operationalDataComplete(coverage)) {
      const forceFull = window.confirm(
        'All 8 datasets are already loaded in this session.\n\n' +
          '• Quick refresh (~1 min): re-copy from server warm cache\n' +
          '• Full restore (10–15 min): re-downloads from GitHub\n\n' +
          'Click OK for quick refresh, or Cancel to run the slow full restore.',
      )
      if (forceFull) {
        setL('refresh_all', true)
        setRestoreProgress({ message: 'Quick refresh from warm cache…', progress: 0, step: 'warm' })
        try {
          const { cacheHydrateWarm } = await import('../api/client')
          await cacheHydrateWarm()
          const c = await getCoverage({ light: true, timeout: 45_000 })
          setCoverage(c)
          setRestoreProgress({ message: 'Quick refresh complete', progress: 100, step: 'done' })
          invalidateDataQueries(qc)
          showToast('success', 'Data refreshed from server warm cache.', 6000)
        } catch (e) {
          showToast('error', e instanceof Error ? e.message : 'Quick refresh failed', 8000)
        } finally {
          setL('refresh_all', false)
        }
        return
      }
    }
    setL('refresh_all', true)
    setRestoreProgress({ message: 'Starting restore…', progress: 0, step: 'queued' })
    try {
      const c = await restoreFullFromServer(tick => setRestoreProgress(tick))
      setCoverage(c)
      setRestoreProgress({
        message: c.message || 'Restore complete',
        progress: 100,
        step: 'done',
      })
      invalidateDataQueries(qc)
      const loaded = [
        c.sku_mapping && 'SKU map',
        c.mtr && `Amazon (${c.mtr_rows.toLocaleString()})`,
        c.myntra && `Myntra (${c.myntra_rows.toLocaleString()})`,
        c.meesho && `Meesho (${c.meesho_rows.toLocaleString()})`,
        c.flipkart && `Flipkart (${c.flipkart_rows.toLocaleString()})`,
        c.snapdeal && `Snapdeal (${c.snapdeal_rows.toLocaleString()})`,
        c.sales && `Sales (${c.sales_rows.toLocaleString()})`,
        c.inventory && 'Inventory',
      ].filter(Boolean)
      const essentialMissing = (c.missing_platforms ?? []).filter(p => p !== 'snapdeal')
      if (essentialMissing.length) {
        showToast(
          'error',
          c.message ||
            `Still missing: ${essentialMissing.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(', ')}`,
          14_000,
        )
      } else {
        showToast(
          'success',
          c.message ||
            (loaded.length
              ? `Full restore: ${loaded.join(' · ')}`
              : 'No bulk data on server — upload Tier 1 files or use Load Cache.'),
          12_000,
        )
      }
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Restore failed')
    } finally {
      setL('refresh_all', false)
    }
  }

  const showToast = (type: 'success' | 'error', msg: string, durationMs = 5000) => {
    setToast({ type, msg })
    setTimeout(() => setToast(null), durationMs)
  }

  const finalizeDailyAutoUpload = (
    source: string,
    cov: CoverageResponse | null,
    initialRes?: Awaited<ReturnType<typeof uploadDailyAuto>>,
  ) => {
    const pendingOnly =
      !!(initialRes as { parsing_pending?: boolean })?.parsing_pending &&
      !cov &&
      (initialRes?.detected_files ?? 0) === 0
    const summary =
      (cov ? dailyAutoSummaryFromCoverage(cov) : null) ??
      (pendingOnly
        ? null
        : initialRes?.detected_platforms?.length || initialRes?.warnings?.length
          ? dailyAutoSummaryFromUpload(initialRes)
          : null)
    const detected = summary?.detected_platforms ?? initialRes?.detected_platforms ?? []
    const warnings = summary?.warnings ?? initialRes?.warnings ?? []
    if (source === 'daily') setDailyDetected(detected)
    if (!pendingOnly) {
      captureGenericAlert(source, warnings, {
        parsed: summary?.expanded_files ?? summary?.processed_files ?? initialRes?.processed_files,
        kept: summary?.saved_files ?? summary?.detected_files ?? initialRes?.detected_files,
        dropped: summary?.unknown_files ?? initialRes?.unknown_files,
        saved: summary?.saved_files ?? initialRes?.saved_files,
        fileResults: summary?.file_results ?? initialRes?.file_results,
      })
    }
    const toastMsg = summary
      ? formatDailyAutoCompleteToast(summary, cov?.sales_rows)
      : (initialRes?.message ?? 'Daily upload complete.')
    const failed = (summary?.detected_files ?? initialRes?.detected_files ?? 0) === 0
      && (summary?.processed_files ?? initialRes?.processed_files ?? 0) > 0
    const hasIssues = (summary?.unknown_files ?? initialRes?.unknown_files ?? 0) > 0
      || (warnings.length > 0)
    showToast(failed ? 'error' : 'success', toastMsg, hasIssues ? 14_000 : 10_000)
    return toastMsg
  }

  const captureUploadAlerts = (source: string, res: UploadResponse) => {
    const dropped = Number(res.dropped_rows ?? 0)
    const hasDropped = dropped > 0
    const hasValidation = (res.validation_warnings?.length ?? 0) > 0
    const next: UploadAlert = {
      at: new Date().toLocaleString(),
      complete: !hasDropped && !hasValidation,
      title: !hasDropped && !hasValidation ? 'Import completeness: Complete' : 'Import completeness: Issues found',
      parsed: res.parsed_rows,
      kept: res.kept_rows,
      dropped: res.dropped_rows,
      droppedReasons: res.dropped_reasons ?? [],
      validationWarnings: res.validation_warnings ?? [],
    }
    setUploadAlertsBySource(prev => ({ ...prev, [source]: next }))
  }

  const captureGenericAlert = (
    source: string,
    warnings: string[] | undefined,
    info?: {
      parsed?: number
      kept?: number
      dropped?: number
      saved?: number
      fileResults?: UploadAlert['fileResults']
    },
  ) => {
    const issues = warnings ?? []
    const dropped = Number(info?.dropped ?? 0)
    const complete = issues.length === 0 && dropped <= 0
    const saved = info?.saved ?? info?.kept
    const parsed = info?.parsed
    const titleExtra =
      saved != null && parsed != null
        ? ` (${saved} of ${parsed} saved)`
        : ''
    const next: UploadAlert = {
      at: new Date().toLocaleString(),
      complete,
      title: complete
        ? `Import completeness: Complete${titleExtra}`
        : `Import completeness: Issues found${titleExtra}`,
      parsed: info?.parsed,
      kept: saved ?? info?.kept,
      dropped: info?.dropped,
      droppedReasons: [],
      validationWarnings: issues,
      fileResults: info?.fileResults,
    }
    setUploadAlertsBySource(prev => ({ ...prev, [source]: next }))
  }

  const clearUploadAlert = (source: string) => {
    setUploadAlertsBySource(prev => {
      if (!prev[source]) return prev
      const next = { ...prev }
      delete next[source]
      return next
    })
  }

  const setL = (key: string, v: boolean) => setLoading(prev => ({ ...prev, [key]: v }))

  // Refresh coverage after an upload. We intentionally swallow errors so a
  // slow /data/coverage (queued behind a big upload) doesn't replace the
  // success toast with "timeout of 20000ms exceeded". The polling useQuery
  // below will pick up the fresh coverage on the next 5s tick.
  const refresh = async (opts?: { light?: boolean }) => {
    try {
      const c = await getCoverage({
        light: opts?.light ?? true,
        timeout: opts?.light === false ? 120_000 : 20_000,
      })
      setCoverage(c)
    } catch (err) {
      console.warn('Post-upload coverage refresh failed; polling will retry.', err)
    }
    invalidateDataQueries(qc)
  }

  const withUploadGuard = async <T,>(fn: () => Promise<T>): Promise<T> => {
    uploadBegin()
    try {
      return await fn()
    } finally {
      uploadEnd()
    }
  }

  const uploadReturnsFile = async (
    file: File,
    onProgress?: (p: { pct: number; msg: string; loaded?: number; total?: number }) => void,
  ): Promise<UploadResponse> => {
    const data = await uploadPoReturnsImport(file, {
      replace: false,
      onUploadProgress: (pct, loaded, total) => {
        onProgress?.({
          pct,
          msg: `Uploading ${file.name}…`,
          loaded,
          total,
        })
      },
    })
    if (data.ok) {
      try {
        if (data.returns_import === 'running') {
          await waitForReturnsImport((msg, prog) => {
            setBuildingMsg(msg)
            onProgress?.({
              pct: Math.max(45, prog ?? 50),
              msg: msg || 'Importing return data…',
            })
          })
        }
        if (data.sales_rebuild === 'pending') {
          await waitForSalesRebuild(msg => setBuildingMsg(msg))
        }
      } catch {
        /* coverage poll will pick up rebuild */
      } finally {
        setBuildingMsg('')
      }
    }
    const msg =
      data.ok && (data.returns_import === 'running' || data.sales_rebuild === 'pending')
        ? `${data.message} Return data saves automatically for PO — run Calculate PO when done.`
        : data.message
    return { ok: data.ok, message: msg } as UploadResponse
  }

  const handleReturnsUpload = async (
    file: File,
    onProgress?: (p: { pct: number; msg: string; loaded?: number; total?: number }) => void,
  ): Promise<UploadResponse> => {
    setL('returns_po', true)
    try {
      let outcome: UploadResponse = { ok: false, message: 'Upload failed' }
      await withUploadGuard(async () => {
        const res = await uploadReturnsFile(file, onProgress)
        outcome = res
        if (res.ok) {
          captureUploadAlerts('returns_po', res)
          showToast('success', res.message)
          await refresh()
        } else {
          showToast('error', res.message)
        }
      })
      return outcome
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Upload failed'
      showToast('error', msg)
      return { ok: false, message: msg }
    } finally {
      setL('returns_po', false)
    }
  }

  const handle = (
    key: string,
    fn: (file: File) => Promise<UploadResponse>,
    opts?: { tier1?: boolean },
  ) => async (file: File) => {
    setL(key, true)
    if (opts?.tier1) setBuildingMsg(`Uploading ${file.name}…`)
    try {
      await withUploadGuard(async () => {
        let res: UploadResponse
        try {
          res = await fn(file)
        } catch (e: unknown) {
          if (opts?.tier1 && e instanceof Error && /timed out|502|gateway/i.test(e.message)) {
            showToast('success', 'Upload may still be processing on the server — watching row counts…', 8000)
            const cov = await waitForTier1Bulk(msg => setBuildingMsg(msg))
            showToast('success', cov.tier1_bulk_message || 'Archive parsed.')
            await refresh()
            return
          }
          throw e
        }
        if (!res.ok) {
          showToast('error', res.message)
          return
        }
        if (opts?.tier1 && /background|accepted/i.test(res.message)) {
          showToast('success', res.message, 8000)
          const cov = await waitForTier1Bulk(msg => setBuildingMsg(msg))
          if (cov.tier1_bulk_status === 'error') {
            showToast('error', cov.tier1_bulk_message || 'Bulk upload failed')
          } else {
            showToast('success', cov.tier1_bulk_message || res.message)
          }
          await refresh()
          return
        }
        captureUploadAlerts(key, res)
        showToast('success', res.message)
        await refresh()
      })
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Upload failed')
    } finally { setL(key, false); setBuildingMsg('') }
  }

  const handleBuildSales = async () => {
    setL('build', true)
    setBuildingMsg('Building combined sales dataset…')
    try {
      await withUploadGuard(async () => {
        const res = await buildSales()
        if (res.ok && res.sales_rebuild === 'pending') {
          showToast('success', res.message, 6000)
          const cov = await waitForSalesRebuild(msg => setBuildingMsg(msg))
          setSkuMapGaps([])
          showToast('success', cov.sales_rebuild_message || `Sales rebuilt (${(cov.sales_rows ?? 0).toLocaleString()} rows).`)
          await refresh()
          return
        }
        setSkuMapGaps(res.unmapped_skus ?? [])
        if (res.ok) { showToast('success', res.message); await refresh() }
        else showToast('error', res.message)
      })
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Build failed')
    } finally { setL('build', false); setBuildingMsg('') }
  }

  const [dailyDetected, setDailyDetected] = useState<string[]>([])
  const [skuMapGaps, setSkuMapGaps] = useState<string[]>([])
  const [resetClearTier3, setResetClearTier3] = useState(false)
  const [resetClearWarm, setResetClearWarm] = useState(true)
  const [showImportCompleteness, setShowImportCompleteness] = useState(true)
  const [inventoryAmzDisclaimer, setInventoryAmzDisclaimer] = useState<InventoryAmazonDisclaimer | null>(null)
  const [qualityReport, setQualityReport] = useState<Awaited<ReturnType<typeof getDataQuality>> | null>(null)
  const [verifyDate, setVerifyDate] = useState(() => {
    const d = new Date()
    d.setDate(d.getDate() - 1)
    return d.toISOString().slice(0, 10)
  })
  const [verifyResult, setVerifyResult] = useState<DailyUploadVerifyResponse | null>(null)
  const [verifyBusy, setVerifyBusy] = useState(false)

  const runUploadVerify = async (dateOverride?: string) => {
    const day = (dateOverride || verifyDate).trim()
    if (!day) return
    setVerifyBusy(true)
    try {
      const v = await verifyDailyUpload(day)
      setVerifyResult(v)
      if (v.ok && v.sales_ready) {
        showToast('success', v.message, 10_000)
        await refresh({ light: true })
      } else if (v.ok) {
        showToast('success', `${v.message} Tap ↻ Rebuild if dashboard is still empty.`, 12_000)
      } else {
        showToast('error', v.message, 12_000)
      }
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Verify failed')
    } finally {
      setVerifyBusy(false)
    }
  }

  const handleDailyAuto = async (files: File[]) => {
    setL('daily', true)
    setBuildingMsg('')
    setChunkProgress(null)
    try {
      await withUploadGuard(async () => {
        const res = await uploadDailyAuto(files, p => {
          setBuildingMsg(p.message)
          if (p.bytesTotal > 0) {
            if (p.phase === 'complete') {
              setChunkProgress({
                pct: 100,
                sent: p.bytesTotal,
                total: p.bytesTotal,
                msg: 'Upload complete — processing on server…',
              })
            } else {
              setChunkProgress({
                pct: Math.min(99, Math.round((p.bytesSent / p.bytesTotal) * 100)),
                sent: p.bytesSent,
                total: p.bytesTotal,
                msg: p.message,
              })
            }
          }
        })
        if (res.ok) {
          if (res.ingest_async || res.sales_rebuild === 'pending') {
            showToast('success', `${res.message} Processing on server…`, 6000)
            try {
              let cov: CoverageResponse | null = null
              if (res.ingest_async) {
                cov = await waitForDailyAutoIngest(msg => {
                  setBuildingMsg(msg)
                  setChunkProgress(prev => {
                    const total = prev?.total ?? 100
                    const prevPct = prev?.pct ?? 85
                    const pct = _bumpProgress(prevPct, 96, 2)
                    return {
                      pct,
                      sent: Math.round((total * pct) / 100),
                      total,
                      msg,
                    }
                  })
                })
              }
              if (res.sales_rebuild === 'pending') {
                setBuildingMsg('Rebuilding combined sales…')
                setChunkProgress(prev => {
                  const total = prev?.total ?? 100
                  const prevPct = prev?.pct ?? 96
                  const pct = Math.max(96, prevPct)
                  return {
                    pct,
                    sent: Math.round((total * pct) / 100),
                    total,
                    msg: 'Rebuilding combined sales…',
                  }
                })
                cov = await waitForSalesRebuild(msg => {
                  setBuildingMsg(msg)
                  setChunkProgress(prev => {
                    const total = prev?.total ?? 100
                    const prevPct = prev?.pct ?? 96
                    const pct = _bumpProgress(prevPct, 99, 1)
                    return {
                      pct,
                      sent: Math.round((total * pct) / 100),
                      total,
                      msg,
                    }
                  })
                })
              }
              setChunkProgress(prev => {
                const total = prev?.total ?? 100
                return {
                  pct: 100,
                  sent: total,
                  total,
                  msg: 'Done',
                }
              })
              setBuildingMsg('')
              finalizeDailyAutoUpload('daily', cov, res)
              const dateFromName = files
                .map(f => f.name.match(/(\d{1,2})[-_./](\d{1,2})[-_./](\d{2,4})/))
                .find(Boolean)
              if (dateFromName) {
                const [, d, m, y] = dateFromName
                const yr = y.length === 2 ? `20${y}` : y
                const iso = `${yr}-${m.padStart(2, '0')}-${d.padStart(2, '0')}`
                setVerifyDate(iso)
                void runUploadVerify(iso)
              } else if (verifyDate) {
                void runUploadVerify(verifyDate)
              }
            } catch (pollErr: unknown) {
              const msg = pollErr instanceof Error ? pollErr.message : 'Upload status unknown'
              if (/timed out/i.test(msg)) {
                showToast(
                  'success',
                  'Upload accepted and still processing on server. You can continue working; status will update below.',
                  12_000,
                )
                await refresh({ light: true })
              } else {
                // Ingest ended with error status — fetch final coverage so the
                // import-completeness box is shown with file-level details.
                try {
                  const { getCoverage } = await import('../api/client')
                  const finalCov = await getCoverage({ light: true, timeout: 10_000 })
                  finalizeDailyAutoUpload('daily', finalCov, res)
                } catch {
                  captureGenericAlert('daily', [msg], {
                    parsed: res.processed_files,
                    kept: res.detected_files ?? 0,
                    dropped: res.unknown_files,
                  })
                }
                throw pollErr
              }
            }
          } else {
            finalizeDailyAutoUpload('daily', null, res)
          }
          await refresh()
        } else {
          const stuck = !!(res as { stuck?: boolean }).stuck
          captureGenericAlert('daily', res.warnings?.length ? res.warnings : [res.message], {
            parsed: res.processed_files,
            kept: res.detected_files,
            dropped: res.unknown_files,
          })
          showToast('error', stuck ? `${res.message} Use “Clear stuck upload” below.` : res.message, 12_000)
        }
      })
    } catch (e: unknown) {
      const { isUploadRequestInterrupted, waitForDailyAutoIngest, waitForSalesRebuild } = await import('../api/client')
      if (isUploadRequestInterrupted(e)) {
        showToast('success', 'Upload may still be processing on the server…', 6000)
        try {
          let cov = await waitForDailyAutoIngest(msg => setBuildingMsg(msg))
          cov = await waitForSalesRebuild(msg => setBuildingMsg(msg))
          setBuildingMsg('')
          finalizeDailyAutoUpload('daily', cov, { ok: true, message: 'Daily upload complete.' })
          await refresh()
        } catch (pollErr: unknown) {
          const msg = pollErr instanceof Error ? pollErr.message : 'Upload status unknown'
          if (/timed out/i.test(msg)) {
            showToast(
              'success',
              'Upload accepted and still processing on server. You can continue working; status will update below.',
              12_000,
            )
            await refresh({ light: true })
          } else {
            showToast(
              'error',
              `${msg} Try “Clear stuck upload”, wait a minute, then Load Cache.`,
              12_000,
            )
          }
        }
      } else {
        const msg = e instanceof Error ? e.message : 'Upload failed'
        showToast(
          'error',
          /timed out/i.test(msg)
            ? `${msg} Files may still be on the server — try “Clear stuck upload”, then Load Cache.`
            : msg,
          12_000,
        )
      }
    } finally { setL('daily', false); setChunkProgress(null); setBuildingMsg('') }
  }

  const handleClearStuckDaily = async () => {
    setL('daily_reset', true)
    try {
      const { resetStuckDailyUpload } = await import('../api/client')
      const res = await resetStuckDailyUpload()
      showToast(res.cleared ? 'success' : 'success', res.message)
      setBuildingMsg('')
      await refresh()
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Could not reset')
    } finally { setL('daily_reset', false) }
  }

  const handleClearStuckInventory = async () => {
    invWaitAbortRef.current = true
    setL('inv', false)
    setL('inv_reset', true)
    try {
      const res = await resetStuckInventoryUpload()
      showToast('success', res.message)
      setBuildingMsg('')
      setInvProgress(null)
      await refresh()
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Could not reset')
    } finally { setL('inv_reset', false) }
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
          {showAdminTab ? (
            <ul className="mt-2 space-y-1.5 text-xs text-gray-600 list-disc list-inside">
              <li>
                <strong>Daily uploads</strong> tab — <em>everyone</em>: daily sales files (auto-detect), snapshot inventory (OMS / marketplace files),
                and return reports for PO. Data saves automatically; only the owner account can delete saved daily files or clear history.
              </li>
              <li>
                <strong>History &amp; setup</strong> tab — <em>Admin / Manager</em>: SKU map, multi-year bulk archives, Amazon extras, and monthly RAR drops.
                SKU master map, existing PO baseline sheet, and PO Engine baseline sheets (SKU status / wide inventory matrix) are <strong>Admin-only</strong> while the lock is on — see <em>History &amp; setup</em> below.
              </li>
            </ul>
          ) : (
            <p className="mt-2 text-xs text-gray-600">
              Use the <strong>Daily uploads</strong> section below: <strong>daily sales</strong> (auto-detect), <strong>snapshot inventory</strong>, and{' '}
              <strong>returns</strong>. Files are saved automatically; contact an Admin to change base history or SKU mapping.
            </p>
          )}
          <label className="mt-3 inline-flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={showImportCompleteness}
              onChange={(e) => setShowImportCompleteness(e.target.checked)}
            />
            Show import completeness status (complete vs missing/issues) after each upload
          </label>
        </div>
      </div>

      {showRestorePanel && (
        <RestoreProgressPanel
          message={restoreProgress.message || coverage.session_restore_message || 'Restoring…'}
          progress={restoreProgress.progress || coverage.session_restore_progress || 0}
          step={restoreProgress.step || coverage.session_restore_step || 'queued'}
        />
      )}

      {uploadBusy && !showRestorePanel && (
        <div className="rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-900 flex items-center gap-2">
          <svg className="animate-spin h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
          </svg>
          <span>
            <strong>Upload in progress</strong>
            {buildingMsg ? ` — ${buildingMsg}` : ''}. Stay on this page; you will not be logged out.
          </span>
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div className={`fixed top-4 right-4 z-50 rounded-lg px-5 py-3 shadow-lg text-sm text-white max-w-sm
          ${toast.type === 'success' ? 'bg-green-600' : 'bg-red-600'}`}>
          {toast.msg}
        </div>
      )}

      {!anyLoaded && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
          {coverage.pause_auto_data_restore ? (
            <p>
              <strong>No data in this browser session.</strong> Click <strong>Load Cache</strong> in the left sidebar
              (shared server history is not deleted).{showAdminTab ? ' Open History & setup if you may replace Tier 1.' : ''}
            </p>
          ) : (
            <p>
              <strong>Data is loading…</strong> The server copies shared history into your session (usually a few seconds
              after refresh). If rows stay empty after a minute, click <strong>Load Cache</strong> in the sidebar.
            </p>
          )}
        </div>
      )}

      {!showAdminTab && (
        <div className="rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-950">
          <p className="font-semibold">Daily uploads</p>
          <p className="mt-1 text-xs">
            Bulk history, platform clear, and delete-all are not available for your role. Everything you need is in{' '}
            <strong>Daily uploads</strong> below (sales, snapshot inventory, returns).
          </p>
        </div>
      )}

      {showAdminTab && authUser?.historical_upload_locked && !allowAdminBaseline && (
        <div className="rounded-xl border border-amber-200 bg-amber-50/80 px-4 py-3 text-sm text-amber-950">
          <p className="font-semibold">Manager access</p>
          <p className="mt-1 text-xs">
            You can upload multi-year platform files on <strong>History &amp; setup</strong>. Replacing the SKU master map, existing PO baseline, or the wide
            daily-inventory matrix is <strong>Admin-only</strong> while data is locked.
          </p>
        </div>
      )}

      {/* KPI strip */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <p className="text-xs text-gray-500">Session data status — scroll down for upload cards per platform.</p>
        <button
          type="button"
          onClick={() => void handleRefreshAllData()}
          disabled={loading['refresh_all'] || uploadBusy}
          className="text-xs px-3 py-1.5 rounded-lg border border-[#002B5B] text-[#002B5B] font-semibold hover:bg-blue-50 disabled:opacity-50"
        >
          {loading['refresh_all'] || restoreBusy
            ? `Restoring… ${restoreProgress.progress || coverage.session_restore_progress || 0}%`
            : '↻ Restore all from server'}
        </button>
      </div>
      <DataLoadedSummary coverage={coverage} />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard label="SKU Mapping"  value={coverage.sku_mapping ? '✅ Loaded' : '— Not loaded'} />
        <KpiCard label="Amazon Rows"  value={coverage.mtr_rows > 0 ? coverage.mtr_rows.toLocaleString() : '—'} />
        <KpiCard label="Sales Rows"   value={coverage.sales_rows > 0 ? coverage.sales_rows.toLocaleString() : '—'} />
        <KpiCard label="Inventory"    value={coverage.inventory ? '✅ Loaded' : '— Not loaded'} />
      </div>

      {showAdminTab && (
        <div className="flex gap-1 border-b border-slate-200 mb-1" role="tablist" aria-label="Upload sections">
          <button
            type="button"
            role="tab"
            aria-selected={uploadTab === 'daily'}
            className={`px-4 py-2.5 text-sm font-semibold rounded-t-lg border-b-2 -mb-px transition-colors ${
              uploadTab === 'daily'
                ? 'border-[#002B5B] text-[#002B5B] bg-white'
                : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-200'
            }`}
            onClick={() => setUploadTab('daily')}
          >
            Daily uploads
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={uploadTab === 'admin'}
            className={`px-4 py-2.5 text-sm font-semibold rounded-t-lg border-b-2 -mb-px transition-colors ${
              uploadTab === 'admin'
                ? 'border-[#002B5B] text-[#002B5B] bg-white'
                : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-200'
            }`}
            onClick={() => setUploadTab('admin')}
          >
            History &amp; setup
          </button>
        </div>
      )}

      {/* Tier 1 — Required (Admin / Manager only when historical lock is on) */}
      {showAdminTab && uploadTab === 'admin' && allowHistorical && <Section title="Tier 1 — Required">
        {allowAdminBaseline ? (
        <UploadCard
          title="1️⃣ SKU Mapping"
          subtitle="Master Yash map (~all panels) ships with the app. Upload your .xlsx to replace or extend it."
          loaded={coverage.sku_mapping}
          alert={showImportCompleteness ? uploadAlertsBySource['sku_mapping'] : undefined}
          onClearAlert={() => clearUploadAlert('sku_mapping')}
        >
          <FileUpload
            label="Upload .xlsx"
            accept={{ 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'] }}
            onUpload={async (file: File) => {
              setL('sku', true)
              try {
                await withUploadGuard(async () => {
                  const res = await uploadSkuMapping(file)
                  setSkuMapGaps(res.unmapped_skus ?? [])
                  if (res.ok) {
                    captureUploadAlerts('sku_mapping', res)
                    showToast('success', res.message)
                    await refresh()
                  } else showToast('error', res.message)
                })
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
        ) : (
          <div className="col-span-2 bg-white rounded-xl border border-amber-100 p-5 shadow-sm space-y-2">
            <h3 className="font-semibold text-[#002B5B] text-sm">1️⃣ SKU Mapping</h3>
            <p className="text-xs text-gray-600">
              Replacing the master map is <strong>Admin-only</strong> while historical data is locked. Ask an Admin to upload a new map.
            </p>
            <p className="text-xs text-gray-500">
              {coverage.sku_mapping ? '✓ This session reports a loaded SKU map.' : '— No SKU map reported in coverage yet.'}
            </p>
          </div>
        )}

        <UploadCard title="2️⃣ Amazon" subtitle="MTR master ZIP or RAR — upload multiple; data stacks" loaded={coverage.mtr} rows={coverage.mtr_rows} onClear={mayClearPlatform ? handleClear('mtr') : undefined} clearing={loading['clear_mtr']} alert={showImportCompleteness ? uploadAlertsBySource['mtr'] : undefined} onClearAlert={() => clearUploadAlert('mtr')}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload .zip or .rar"
            accept={{
              'application/zip': ['.zip'],
              'application/vnd.rar': ['.rar'],
              'application/x-rar-compressed': ['.rar'],
            }}
            onUpload={handle('mtr', (file: File) => uploadMtr(file), { tier1: true })}
            uploading={loading['mtr']}
          />
        </UploadCard>
      </Section>}

      {showAdminTab && uploadTab === 'admin' && allowHistorical && <Section title="Tier 1 — Platform history (bulk / multi-year)">
        {(coverage.tier1_bulk_status === 'running' || (buildingMsg && /uploading|parsing/i.test(buildingMsg))) && (
          <div className="col-span-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
            <div className="flex items-center gap-2 text-xs text-amber-800">
              <svg className="animate-spin h-3 w-3 text-amber-600 shrink-0" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
              </svg>
              <span className="flex-1 truncate">
                {coverage.tier1_bulk_message || buildingMsg || 'Parsing bulk history on server…'}
              </span>
            </div>
          </div>
        )}
        <UploadCard title="🛍️ Myntra PPMP" subtitle="Upload multiple company ZIPs — data stacks" loaded={coverage.myntra} rows={coverage.myntra_rows} onClear={mayClearPlatform ? handleClear('myntra') : undefined} clearing={loading['clear_myntra']} alert={showImportCompleteness ? uploadAlertsBySource['myntra'] : undefined} onClearAlert={() => clearUploadAlert('myntra')}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload .zip"
            accept={{ 'application/zip': ['.zip'] }}
            onUpload={handle('myntra', (file: File) => uploadMyntra(file), { tier1: true })}
            uploading={loading['myntra']}
          />
        </UploadCard>

        <UploadCard title="🛒 Meesho" subtitle="ZIP (TCS/ledger), Order CSV, or unified sales Excel (.xlsx/.xls) — select multiple" loaded={coverage.meesho} rows={coverage.meesho_rows} onClear={mayClearPlatform ? handleClear('meesho') : undefined} clearing={loading['clear_meesho']} alert={showImportCompleteness ? uploadAlertsBySource['meesho'] : undefined} onClearAlert={() => clearUploadAlert('meesho')}>
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
            onUpload={handle('meesho', (file: File) => uploadMeesho(file), { tier1: true })}
            uploading={loading['meesho']}
            multiple
          />
        </UploadCard>

        <UploadCard title="🟡 Flipkart" subtitle="Upload multiple company ZIPs — data stacks" loaded={coverage.flipkart} rows={coverage.flipkart_rows} onClear={mayClearPlatform ? handleClear('flipkart') : undefined} clearing={loading['clear_flipkart']} alert={showImportCompleteness ? uploadAlertsBySource['flipkart'] : undefined} onClearAlert={() => clearUploadAlert('flipkart')}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload .zip"
            accept={{ 'application/zip': ['.zip'] }}
            onUpload={handle('flipkart', (file: File) => uploadFlipkart(file), { tier1: true })}
            uploading={loading['flipkart']}
          />
        </UploadCard>

        <UploadCard title="🔴 Snapdeal" subtitle="OMS order reports (CSV/ZIP) or AG/PE/YG ZIPs" loaded={coverage.snapdeal} rows={coverage.snapdeal_rows} onClear={mayClearPlatform ? handleClear('snapdeal') : undefined} clearing={loading['clear_snapdeal']} alert={showImportCompleteness ? uploadAlertsBySource['snapdeal'] : undefined} onClearAlert={() => clearUploadAlert('snapdeal')}>
          <FileUpload
            label="Upload files (select multiple)"
            accept={{ 'application/zip': ['.zip'], 'text/csv': ['.csv'], 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'] }}
            onUpload={handle('snapdeal', (file: File) => uploadSnapdeal(file), { tier1: true })}
            uploading={loading['snapdeal']}
            multiple={true}
          />
        </UploadCard>
      </Section>}

      {showAdminTab && uploadTab === 'admin' && allowHistorical && <Section title="Tier 2 — Amazon Orders & PO Pipeline">
        <UploadCard title="📄 Amazon MTR CSV" subtitle="Single-month MTR or FBA shipment CSV" loaded={false} alert={showImportCompleteness ? uploadAlertsBySource['b2c'] : undefined} onClearAlert={() => clearUploadAlert('b2c')}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload MTR .csv"
            accept={{ 'text/csv': ['.csv'] }}
            onUpload={handle('b2c', (file: File) => uploadAmazonB2C(file))}
            uploading={loading['b2c']}
          />
        </UploadCard>

        <UploadCard title="📋 Amazon B2B CSV" subtitle="Single-month B2B report CSV" loaded={false} alert={showImportCompleteness ? uploadAlertsBySource['b2b'] : undefined} onClearAlert={() => clearUploadAlert('b2b')}>
          {!coverage.sku_mapping && <Warn>Upload SKU Mapping first.</Warn>}
          <FileUpload
            label="Upload B2B .csv"
            accept={{ 'text/csv': ['.csv'] }}
            onUpload={handle('b2b', (file: File) => uploadAmazonB2B(file))}
            uploading={loading['b2b']}
          />
        </UploadCard>

        {allowAdminBaseline ? (
        <>
        <UploadCard
          title="📦 Existing PO Sheet"
          subtitle={
            coverage.existing_po_rows && coverage.existing_po_rows > 0
              ? `${coverage.existing_po_rows.toLocaleString()} SKUs saved on server — drop a newer file below to replace`
              : 'Open/pending POs (XLSX or CSV)'
          }
          loaded={coverage.existing_po}
          rows={coverage.existing_po_rows}
          rowsUnit="SKUs"
          alert={showImportCompleteness ? uploadAlertsBySource['existingpo'] : undefined}
          onClearAlert={() => clearUploadAlert('existingpo')}
        >
          {(coverage.existing_po_filename || coverage.existing_po_uploaded_at) && (
            <div className="text-[11px] text-gray-500 mb-2">
              Last uploaded:{' '}
              <span className="font-semibold text-gray-700">
                {coverage.existing_po_filename || 'Existing PO sheet'}
              </span>
              {coverage.existing_po_rows && coverage.existing_po_rows > 0 ? (
                <span className="text-gray-600">
                  {' '}
                  · {coverage.existing_po_rows.toLocaleString()} SKUs
                </span>
              ) : null}
              {coverage.existing_po_uploaded_at ? (
                <span className="text-gray-400">
                  {' '}
                  ({new Date(coverage.existing_po_uploaded_at).toLocaleString()})
                </span>
              ) : null}
            </div>
          )}
          <p className="text-[11px] text-slate-600 mb-2 leading-relaxed">
            <strong>Process:</strong>{' '}
            {coverage.existing_po
              ? 'Replace anytime — new file overwrites the previous sheet.'
              : 'Upload your open-PO export.'}{' '}
            Server parses every SKU (Pending Cutting, Balance, Pipeline), clears old PO cache, then click{' '}
            <strong>Calculate PO</strong> on PO Engine for per-size pipeline.
            {coverage.existing_po_rows && coverage.existing_po_rows > 0
              ? null
              : ' Large sheets can take 30–90 seconds.'}
          </p>
          {existingPoProgress ? <ExistingPoUploadProgress progress={existingPoProgress} /> : null}
          <FileUpload
            label={
              coverage.existing_po
                ? 'Replace PO sheet (upload updated file)'
                : 'Upload PO Sheet'
            }
            accept={{
              'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
              'text/csv': ['.csv'],
            }}
            onUpload={async (file: File) => {
              setL('existingpo', true)
              const startedAt = Date.now()
              setExistingPoProgress({ pct: 0, msg: `Uploading ${file.name}…`, phase: 'upload', startedAt })
              try {
                await withUploadGuard(async () => {
                  const res = await uploadExistingPO(file, (pct, phase) => {
                    if (phase === 'upload') {
                      setExistingPoProgress({
                        pct,
                        msg:
                          pct < 100
                            ? `Uploading ${file.name}… ${pct}%`
                            : `Upload complete — parsing ${file.name} on server…`,
                        phase: 'upload',
                        startedAt,
                      })
                    } else {
                      setExistingPoProgress({
                        pct: 55,
                        msg: `Parsing ${file.name} — reading SKU rows (30–90s for large sheets)…`,
                        phase: 'parse',
                        startedAt,
                      })
                    }
                  })
                  if (res.ok) {
                    captureUploadAlerts('existingpo', res)
                    usePOStore.getState().setResult(null)
                    usePOStore.getState().setSkipSharedCacheOnce(true)
                    const skuNote =
                      res.rows != null ? ` ${res.rows.toLocaleString()} SKUs loaded.` : ''
                    showToast(
                      'success',
                      `${res.message}${skuNote} Click Calculate PO on PO Engine to refresh pipeline.`,
                      8000,
                    )
                    setExistingPoProgress({
                      pct: 96,
                      msg: `Saving ${res.rows != null ? `${res.rows.toLocaleString()} SKUs` : 'sheet'} to session…`,
                      phase: 'refresh',
                      startedAt,
                    })
                    await refresh()
                  } else {
                    showToast('error', res.message)
                  }
                })
              } catch (e: unknown) {
                showToast('error', e instanceof Error ? e.message : 'Upload failed')
              } finally {
                setL('existingpo', false)
                setExistingPoProgress(null)
              }
            }}
            uploading={loading['existingpo']}
          />
        </UploadCard>

        <UploadCard
          title="✨ Finishing Receipt (production)"
          subtitle={
            coverage.finishing_receipt_report?.balance_units
              ? `${(coverage.finishing_receipt_report.balance_units ?? 0).toLocaleString()} units still at finishing — updates PO Balance to Dispatch`
              : 'Finishing Dept export — issue (IssQty/BalQty) or receive (ReceiveQty) layout'
          }
          loaded={!!coverage.finishing_receipt_uploaded_at}
          rows={coverage.finishing_receipt_report?.skus}
          rowsUnit="SKUs"
        >
          {(coverage.finishing_receipt_filename || coverage.finishing_receipt_uploaded_at) && (
            <div className="text-[11px] text-gray-500 mb-2">
              Last uploaded:{' '}
              <span className="font-semibold text-gray-700">
                {coverage.finishing_receipt_filename || 'Finishing receipt'}
              </span>
              {coverage.finishing_receipt_report?.receive_numbers?.length ? (
                <span className="text-gray-600">
                  {' '}
                  · Receive #{coverage.finishing_receipt_report.receive_numbers.join(', ')}
                </span>
              ) : coverage.finishing_receipt_report?.issue_numbers?.length ? (
                <span className="text-gray-600">
                  {' '}
                  · Issue #{coverage.finishing_receipt_report.issue_numbers.join(', ')}
                </span>
              ) : null}
              {coverage.finishing_receipt_uploaded_at ? (
                <span className="text-gray-400">
                  {' '}
                  ({new Date(coverage.finishing_receipt_uploaded_at).toLocaleString()})
                </span>
              ) : null}
            </div>
          )}
          {coverage.finishing_receipt_report?.left_units ? (
            <p className="text-[11px] text-amber-800 bg-amber-50 border border-amber-200 rounded px-2 py-1.5 mb-2">
              <strong>{(coverage.finishing_receipt_report.left_units ?? 0).toLocaleString()}</strong> units still
              at finishing across{' '}
              <strong>{(coverage.finishing_receipt_report.non_clear_skus ?? 0).toLocaleString()}</strong> SKUs.
              Re-upload when more goods are received.
            </p>
          ) : null}
          <p className="text-[11px] text-slate-600 mb-2 leading-relaxed">
            Upload after production issues goods to Finishing. <strong>BalQty</strong> becomes{' '}
            <strong>Balance to Dispatch</strong> on the Existing PO sheet. Click <strong>Calculate PO</strong>{' '}
            to refresh requirements. MTR invoice dates will reduce pipeline when stock is dispatched to marketplaces.
          </p>
          <FileUpload
            label="Upload Finishing .xls / .xlsx"
            accept={{
              'application/vnd.ms-excel': ['.xls'],
              'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
            }}
            onUpload={async (file: File) => {
              setL('finishing_receipt', true)
              try {
                await withUploadGuard(async () => {
                  const res = await uploadFinishingReceipt(file)
                  if (res.ok) {
                    showToast('success', res.message || 'Finishing receipt applied.', 8000)
                    setCoverage({
                      ...coverage,
                      existing_po: true,
                      existing_po_generation: res.existing_po_generation ?? coverage.existing_po_generation,
                      existing_po_needs_recalc: true,
                    })
                    await refresh()
                  } else {
                    showToast('error', res.message || 'Upload failed', 10_000)
                  }
                })
              } catch (e: unknown) {
                showToast('error', e instanceof Error ? e.message : 'Upload failed')
              } finally {
                setL('finishing_receipt', false)
              }
            }}
            uploading={loading['finishing_receipt']}
          />
        </UploadCard>
        </>
        ) : (
          <div className="col-span-2 bg-white rounded-xl border border-amber-100 p-5 shadow-sm space-y-2">
            <h3 className="font-semibold text-[#002B5B] text-sm">📦 Existing PO Sheet</h3>
            <p className="text-xs text-gray-600">
              Updating the baseline open-PO sheet is <strong>Admin-only</strong> while historical data is locked.
            </p>
          </div>
        )}

        <UploadCard title="🗜️ Monthly Sales RAR" subtitle="Drop Monthly.rar — Amazon, Flipkart, Meesho, Myntra auto-detected inside" loaded={false} alert={showImportCompleteness ? uploadAlertsBySource['monthly_rar'] : undefined} onClearAlert={() => clearUploadAlert('monthly_rar')}>
          <div className="space-y-2">
            <MonthlyRarUploader
              uploading={loading['monthly_rar']}
              onUpload={async (files) => {
                setL('monthly_rar', true)
                setBuildingMsg('')
                try {
                  await withUploadGuard(async () => {
                    const res = await uploadDailyAuto(files, p => setBuildingMsg(p.message))
                    if (res.ok) {
                      if (res.ingest_async || res.sales_rebuild === 'pending') {
                        showToast('success', `${res.message} Processing on server…`, 6000)
                        let cov: CoverageResponse | null = null
                        if (res.ingest_async) {
                          cov = await waitForDailyAutoIngest(msg => setBuildingMsg(msg))
                        }
                        if (res.sales_rebuild === 'pending') {
                          setBuildingMsg('Rebuilding combined sales…')
                          cov = await waitForSalesRebuild(msg => setBuildingMsg(msg))
                        }
                        setBuildingMsg('')
                        finalizeDailyAutoUpload('monthly_rar', cov, res)
                      } else {
                        finalizeDailyAutoUpload('monthly_rar', null, res)
                      }
                      await refresh()
                    } else {
                      captureGenericAlert('monthly_rar', res.warnings?.length ? res.warnings : [res.message], {
                        parsed: res.processed_files,
                        kept: res.detected_files,
                        dropped: res.unknown_files,
                      })
                      showToast('error', res.message, 10_000)
                    }
                  })
                } catch (e: unknown) {
                  showToast('error', e instanceof Error ? e.message : 'Upload failed')
                } finally { setL('monthly_rar', false); setBuildingMsg('') }
              }}
            />
          </div>
        </UploadCard>
      </Section>}

      {showAdminTab && uploadTab === 'admin' && allowHistorical && (
        <Section title="PO Engine — baseline sheets (optional)">
          {allowAdminBaseline ? (
            <>
              <UploadCard
                title="SKU status &amp; lead time (PO)"
                subtitle="Excel/CSV with SKU, Status, and Lead time. Closed SKUs get zero PO; missing lead blocks that row."
                loaded={!!coverage.sku_status_lead}
                rows={coverage.sku_status_lead_rows}
                alert={showImportCompleteness ? uploadAlertsBySource['po_sku_status'] : undefined}
                onClearAlert={() => clearUploadAlert('po_sku_status')}
              >
                <FileUpload
                  label="Upload .xlsx, .xls, or .csv"
                  accept={{
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
                    'application/vnd.ms-excel': ['.xls'],
                    'text/csv': ['.csv'],
                    'application/octet-stream': ['.xlsx', '.xls'],
                  }}
                  onUpload={async (file: File) => {
                    setL('po_sku_status', true)
                    try {
                      await withUploadGuard(async () => {
                        const data = await uploadPoSkuStatusLead(file)
                        const res = {
                          ok: !!data.ok,
                          message:
                            data.message ||
                            (data.ok ? `Loaded ${Number(data.rows ?? 0).toLocaleString()} SKU rows.` : 'Upload failed.'),
                        } as UploadResponse
                        if (res.ok) {
                          captureUploadAlerts('po_sku_status', res)
                          showToast('success', res.message)
                          await refresh()
                        } else {
                          showToast('error', res.message)
                        }
                      })
                    } catch (e: unknown) {
                      showToast('error', e instanceof Error ? e.message : 'Upload failed')
                    } finally {
                      setL('po_sku_status', false)
                    }
                  }}
                  uploading={loading['po_sku_status']}
                />
              </UploadCard>
              <UploadCard
                title="Daily inventory history matrix (PO)"
                subtitle="Wide Excel: rows = SKUs, columns = daily snapshot totals (OMS / Amazon Inventory sheets). For speed, the server keeps only the latest 30 calendar days of history (set env DAILY_INV_MAX_DAYS to keep more)."
                loaded={!!coverage.daily_inventory_history}
                rows={coverage.daily_inventory_history_rows}
                alert={showImportCompleteness ? uploadAlertsBySource['po_daily_inv'] : undefined}
                onClearAlert={() => clearUploadAlert('po_daily_inv')}
              >
                <FileUpload
                  label="Upload .xlsx, .xls, or .csv"
                  accept={{
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
                    'application/vnd.ms-excel': ['.xls'],
                    'text/csv': ['.csv'],
                    'application/octet-stream': ['.xlsx', '.xls'],
                  }}
                  onUpload={async (file: File) => {
                    setL('po_daily_inv', true)
                    setBuildingMsg('')
                    try {
                      await withUploadGuard(async () => {
                        const result = await uploadPoDailyInventoryHistoryFile(file, msg => setBuildingMsg(msg))
                        setBuildingMsg('')
                        if (result.ok) {
                          captureGenericAlert('po_daily_inv', [], {
                            parsed: result.rows,
                            kept: result.skus,
                          })
                          showToast('success', result.message || `Loaded ${(result.rows ?? 0).toLocaleString()} matrix rows.`)
                          setCoverage({
                            ...coverage,
                            daily_inventory_history: true,
                            daily_inventory_history_rows: result.rows ?? coverage.daily_inventory_history_rows,
                            daily_inventory_history_skus: result.skus ?? coverage.daily_inventory_history_skus,
                          })
                          await refresh({ light: true })
                        } else {
                          showToast('error', result.message || 'Upload failed')
                        }
                      })
                    } catch (e: unknown) {
                      setBuildingMsg('')
                      showToast('error', e instanceof Error ? e.message : 'Upload failed')
                    } finally {
                      setL('po_daily_inv', false)
                    }
                  }}
                  uploading={loading['po_daily_inv']}
                />
              </UploadCard>
              <UploadCard
                title="Intrasit &amp; Not In Inventory (admin)"
                subtitle={
                  coverage.manual_intransit_skus && coverage.manual_intransit_skus > 0
                    ? `${coverage.manual_intransit_skus.toLocaleString()} SKUs · ${(coverage.manual_intransit_units ?? 0).toLocaleString()} in-transit · ${(coverage.manual_not_in_inventory_units ?? 0).toLocaleString()} not-in-inventory — re-upload replaces prior file (no duplicate rows)`
                    : 'Excel with sheets "Intrasit Inventory" and "Not In Inventory" (Sku + Qty). Re-uploading the same file updates counts — duplicate SKU rows are merged, not double-counted.'
                }
                loaded={!!coverage.manual_intransit_sheet}
                rows={coverage.manual_intransit_skus}
                rowsUnit="SKUs"
              >
                {(coverage.manual_intransit_parse_report?.skip_details?.length ?? 0) > 0 ||
                (coverage.manual_intransit_parse_report?.sheets_skipped?.length ?? 0) > 0 ? (
                  <UploadSkipDetailsPanel report={coverage.manual_intransit_parse_report} />
                ) : null}
                <FileUpload
                  label="Upload .xlsx or .xls"
                  accept={{
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
                    'application/vnd.ms-excel': ['.xls'],
                    'application/octet-stream': ['.xlsx', '.xls'],
                  }}
                  onUpload={async (file: File) => {
                    setL('po_manual_intransit', true)
                    try {
                      await withUploadGuard(async () => {
                        const res = await uploadPoManualIntransitSheet(file)
                        if (res.ok) {
                          const skipItems = [
                            ...(res.parse_report?.warnings ?? []),
                            ...(res.parse_report?.skip_details ?? []).map(
                              d =>
                                `${d.sheet || 'Sheet'}: ${d.reason || 'skipped'}${d.rows_affected ? ` (${d.rows_affected} rows)` : ''}`,
                            ),
                            ...(res.parse_report?.sheets_skipped ?? []).map(
                              s => `${s.sheet}: ${s.reason}`,
                            ),
                          ]
                          captureGenericAlert(
                            'po_manual_intransit',
                            skipItems.length ? skipItems : undefined,
                            {
                              parsed: res.skus,
                              kept: res.skus,
                            },
                          )
                          showToast('success', res.message || 'Manual in-transit sheet loaded.')
                          setCoverage({
                            ...coverage,
                            manual_intransit_sheet: true,
                            manual_intransit_skus: res.skus ?? coverage.manual_intransit_skus,
                            manual_intransit_units:
                              res.intransit_units ?? coverage.manual_intransit_units,
                            manual_not_in_inventory_units:
                              res.not_in_inventory_units ?? coverage.manual_not_in_inventory_units,
                            manual_intransit_parse_report: res.parse_report ?? null,
                            manual_intransit_filename: res.parse_report?.filename,
                          })
                          await refresh({ light: true })
                          qc.invalidateQueries({ queryKey: ['inventory'] })
                        } else {
                          const errMsg =
                            res.message ||
                            res.parse_report?.error ||
                            'Upload failed — check sheet names and columns.'
                          captureGenericAlert('po_manual_intransit', [errMsg], {
                            parsed: 0,
                            kept: 0,
                          })
                          showToast('error', errMsg)
                        }
                      })
                    } catch (e: unknown) {
                      showToast('error', e instanceof Error ? e.message : 'Upload failed')
                    } finally {
                      setL('po_manual_intransit', false)
                    }
                  }}
                  uploading={loading['po_manual_intransit']}
                />
              </UploadCard>
            </>
          ) : (
            <div className="col-span-2 rounded-xl border border-amber-100 bg-amber-50/60 p-5 text-sm text-amber-950 space-y-2">
              <p className="font-semibold text-[#002B5B]">PO baseline sheets</p>
              <p className="text-xs">
                SKU status / lead and the wide daily inventory matrix can only be uploaded by an <strong>Admin</strong> while data is locked.
              </p>
            </div>
          )}
        </Section>
      )}

      {(uploadTab === 'daily' || !showAdminTab) && (
      <Section title="Daily uploads — sales, snapshot inventory, returns">
        <div className="col-span-2 bg-white rounded-xl border border-gray-200 p-5 shadow-sm space-y-3">
          <div>
            <h3 className="font-semibold text-[#002B5B] text-sm">📅 Daily order upload</h3>
            <p className="text-xs text-gray-400">
              For <strong>ongoing</strong> refreshes after Tier 1 history is loaded. Drop <strong>any mix</strong> of recent files —
              platform is auto-detected per file. Accepted: Amazon MTR/FBA CSV, Myntra PPMP CSV, Meesho CSV / ZIP / unified XLSX, Flipkart XLSX,
              Snapdeal paths containing <code className="text-gray-600">snapdeal</code>, RAR bundles. Sales dataset rebuilds automatically.
            </p>
          </div>

          <div className="rounded-lg border border-blue-100 bg-blue-50/60 px-3 py-3">
            <p className="text-xs font-semibold text-[#002B5B] mb-2">Saved daily uploads (server)</p>
            <DailyHistory allowDelete={mayDeleteDailyFiles} />
          </div>

          <DailyDropzone
            uploading={loading['daily'] || loading['daily_reset']}
            chunkProgress={chunkProgress}
            onReject={(msg) => showToast('error', msg)}
            onUpload={async (files) => { await handleDailyAuto(files); qc.invalidateQueries({ queryKey: ['daily-summary'] }); qc.invalidateQueries({ queryKey: ['daily-uploads'] }) }}
          />
          {/* Server-side processing banner — shown while server parses & rebuilds sales */}
          {!chunkProgress && (coverage.daily_auto_ingest_status === 'running' || (loading['daily'] && !!buildingMsg)) && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 space-y-1.5">
              <div className="flex items-center gap-2 text-xs text-amber-800">
                <svg className="animate-spin h-3 w-3 text-amber-600 shrink-0" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
                <span className="flex-1 truncate">
                  {coverage.daily_auto_ingest_message || buildingMsg || 'Processing daily files on server…'}
                </span>
                <button
                  type="button"
                  onClick={handleClearStuckDaily}
                  disabled={loading['daily_reset']}
                  className="shrink-0 px-2.5 py-0.5 rounded border border-amber-300 bg-white text-amber-900 hover:bg-amber-100 disabled:opacity-50 text-xs"
                >
                  {loading['daily_reset'] ? 'Clearing…' : 'Clear stuck'}
                </button>
              </div>
            </div>
          )}
          {dailyDetected.length > 0 && (
            <p className="text-xs text-green-700">
              ✓ Recognized: {dailyDetected.join(' · ')}
            </p>
          )}
          <div className="rounded-lg border border-slate-200 bg-slate-50/80 px-3 py-3 space-y-2">
            <p className="text-xs font-semibold text-[#002B5B]">Verify upload on server</p>
            <p className="text-xs text-gray-600">
              After upload, confirm Tier-3 saved your date and sales rebuilt (e.g. 2 Jun 2026).
            </p>
            <div className="flex flex-wrap items-end gap-2">
              <label className="text-xs text-gray-600">
                Date
                <input
                  type="date"
                  className="block border rounded px-2 py-1 mt-0.5 text-sm"
                  value={verifyDate}
                  onChange={e => setVerifyDate(e.target.value)}
                />
              </label>
              <button
                type="button"
                disabled={verifyBusy || !verifyDate}
                onClick={() => void runUploadVerify()}
                className="px-3 py-1.5 rounded-lg bg-[#002B5B] text-white text-xs font-semibold disabled:opacity-50"
              >
                {verifyBusy ? 'Checking…' : 'Check data saved'}
              </button>
            </div>
            {verifyResult && (
              <div
                className={`text-xs rounded border px-2 py-2 ${
                  verifyResult.ok && verifyResult.sales_ready
                    ? 'border-green-200 bg-green-50 text-green-900'
                    : verifyResult.ok
                      ? 'border-amber-200 bg-amber-50 text-amber-900'
                      : 'border-red-200 bg-red-50 text-red-900'
                }`}
              >
                <p>{verifyResult.message}</p>
                {verifyResult.tier3_platforms.length > 0 && (
                  <p className="mt-1">
                    Tier-3: {verifyResult.tier3_platforms.join(', ')} ({verifyResult.tier3_upload_count} file
                    {verifyResult.tier3_upload_count === 1 ? '' : 's'})
                  </p>
                )}
                {verifyResult.session_sales_rows > 0 && (
                  <p className="mt-0.5">Session sales rows: {verifyResult.session_sales_rows.toLocaleString()}</p>
                )}
                {verifyResult.session_sales_range?.min && verifyResult.session_sales_range?.max && (
                  <p className="mt-0.5 text-gray-700">
                    Session sales dates: {verifyResult.session_sales_range.min}
                    {verifyResult.session_sales_range.min !== verifyResult.session_sales_range.max
                      ? ` → ${verifyResult.session_sales_range.max}`
                      : ''}
                  </p>
                )}
                {verifyResult.tier3_summary && Object.keys(verifyResult.tier3_summary).length > 0 && (
                  <p className="mt-1 text-gray-700">
                    Tier-3 saved:{' '}
                    {Object.entries(verifyResult.tier3_summary)
                      .map(([p, s]) => `${p} ${s.min_date}→${s.max_date}`)
                      .join(' · ')}
                  </p>
                )}
              </div>
            )}
          </div>
          {showImportCompleteness && uploadAlertsBySource['daily'] && (
            <div className={`rounded border p-2 text-xs ${
              uploadAlertsBySource['daily'].complete
                ? 'border-green-200 bg-green-50 text-green-900'
                : 'border-amber-200 bg-amber-50 text-amber-900'
            }`}>
              <p className="font-medium">{uploadAlertsBySource['daily'].title} · {uploadAlertsBySource['daily'].at}</p>
              <p className="mt-0.5">
                Processed: {uploadAlertsBySource['daily'].parsed ?? '—'} · Saved: {uploadAlertsBySource['daily'].kept ?? '—'} · Skipped: {uploadAlertsBySource['daily'].dropped ?? 0}
              </p>
              {/* Per-file breakdown — skipped/failed files */}
              {(uploadAlertsBySource['daily'].fileResults?.length ?? 0) > 0 && (
                <div className="mt-1 space-y-0.5">
                  {uploadAlertsBySource['daily'].fileResults!
                    .filter(r => r.status !== 'saved')
                    .slice(0, 6)
                    .map((r, i) => (
                      <p key={i} className="text-amber-700">
                        ⚠ {r.filename?.split('/').pop() ?? r.filename} — {r.reason ?? r.status}
                      </p>
                    ))}
                  {uploadAlertsBySource['daily'].fileResults!.filter(r => r.status === 'saved').length > 0 && (
                    <p className="text-green-700">
                      ✓ {uploadAlertsBySource['daily'].fileResults!.filter(r => r.status === 'saved').length} file(s) saved successfully
                    </p>
                  )}
                </div>
              )}
              {/* Warnings (RAR extract errors, parse issues, etc.) */}
              {uploadAlertsBySource['daily'].validationWarnings.length > 0 && (
                <div className="mt-1 space-y-0.5">
                  {uploadAlertsBySource['daily'].validationWarnings.slice(0, 4).map((w, i) => (
                    <p key={i} className="truncate" title={w}>{w}</p>
                  ))}
                  {uploadAlertsBySource['daily'].validationWarnings.length > 4 && (
                    <p>…and {uploadAlertsBySource['daily'].validationWarnings.length - 4} more warning(s)</p>
                  )}
                </div>
              )}
            </div>
          )}
          {coverage.daily_orders && (
            <p className="text-xs text-blue-600">Daily orders loaded ✓ — included in sales dataset.</p>
          )}
        </div>

        <UploadCard
          title="📦 Snapshot inventory"
          subtitle="Drop the daily RAR plus any separate files. Include OMS CSV, Amazon, Flipkart & Myntra PPMP inventory CSVs inside the bundle for full marketplace stock."
          loaded={coverage.inventory}
          alert={showImportCompleteness ? uploadAlertsBySource['inv'] : undefined}
          onClearAlert={() => clearUploadAlert('inv')}
        >
          {coverage.inventory && coverage.inventory_snapshot_date_label && (
            <p className="text-xs text-blue-800 bg-blue-50 border border-blue-200 rounded-lg px-3 py-2">
              <span className="font-semibold">Current snapshot date: </span>
              {coverage.inventory_snapshot_date_label}
              {coverage.inventory_snapshot_uploaded_at && (
                <span className="text-blue-700/80">
                  {' '}
                  · last updated{' '}
                  {new Date(coverage.inventory_snapshot_uploaded_at).toLocaleString(undefined, {
                    day: 'numeric',
                    month: 'short',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </span>
              )}
            </p>
          )}
          {!coverage.sku_mapping && <Warn>SKU map must be loaded on the server (ask Admin if missing).</Warn>}
          {(coverage.inventory_upload_status === 'running' || invProgress || (loading['inv'] && !!buildingMsg)) && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 space-y-1.5">
              <div className="flex items-center gap-2 text-xs text-amber-800">
                <svg className="animate-spin h-3 w-3 text-amber-600 shrink-0" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
                <span className="flex-1 truncate">
                  {invProgress?.msg || buildingMsg || coverage.inventory_upload_message || 'Parsing inventory on server…'}
                </span>
                <button
                  type="button"
                  onClick={handleClearStuckInventory}
                  disabled={loading['inv_reset']}
                  className="shrink-0 px-2.5 py-0.5 rounded border border-amber-300 bg-white text-amber-900 hover:bg-amber-100 disabled:opacity-50 text-xs"
                >
                  {loading['inv_reset'] ? 'Clearing…' : 'Clear stuck'}
                </button>
              </div>
              {(() => {
                const pct = invProgress?.phase === 'upload'
                  ? invProgress.pct
                  : Math.max(
                      invProgress?.pct ?? 0,
                      coverage.inventory_upload_progress ?? 0,
                    )
                if (pct <= 0) return null
                return (
                  <div className="space-y-0.5">
                    <div className="h-2 rounded-full bg-amber-100 overflow-hidden">
                      <div
                        className="h-full bg-amber-500 transition-all duration-300"
                        style={{ width: `${Math.min(100, pct)}%` }}
                      />
                    </div>
                    <p className="text-xs text-amber-600 text-right tabular-nums">{pct}%</p>
                  </div>
                )
              })()}
            </div>
          )}
          <InventoryDropzone
            disabled={!coverage.sku_mapping || coverage.inventory_upload_status === 'running'}
            uploading={loading['inv']}
            onUpload={async (files) => {
              invWaitAbortRef.current = false
              setL('inv', true)
              setBuildingMsg('Uploading inventory files…')
              setInvProgress({ pct: 0, msg: 'Uploading inventory files…', phase: 'upload' })
              try {
                await withUploadGuard(async () => {
                  const res = await uploadInventoryAuto(files, p => {
                    setBuildingMsg(p.message)
                    if (p.bytesTotal > 0 && p.phase !== 'complete') {
                      setInvProgress({
                        pct: Math.min(40, Math.round((p.bytesSent / p.bytesTotal) * 40)),
                        msg: p.message,
                        phase: 'upload',
                      })
                    }
                  })
                  if (res.ok) {
                    if (res.ingest_async) {
                      showToast('success', `${res.message}`, 6000)
                      setInvProgress({ pct: 5, msg: 'Upload complete — parsing on server…', phase: 'parse' })
                      const cov = await waitForInventoryUpload(
                        (msg, pct) => {
                          setBuildingMsg(msg)
                          setInvProgress({
                            pct: Math.max(5, pct ?? 5),
                            msg,
                            phase: 'parse',
                          })
                        },
                        undefined,
                        () => invWaitAbortRef.current,
                      )
                      setBuildingMsg('')
                      setInvProgress(null)
                      setCoverage(cov)
                      const results = cov.inventory_upload_file_results ?? []
                      const saved = results.filter(r => r.status === 'loaded').length
                      const skipped = results.filter(r => r.status === 'skipped')
                      const warnings = [
                        ...(cov.inventory_upload_warnings ?? []),
                        ...skipped.map(r => `${r.filename}: ${r.reason ?? 'skipped'}`),
                      ]
                      captureGenericAlert('inv', warnings.length ? warnings : undefined, {
                        parsed: results.length || files.length,
                        kept: saved || cov.inventory_upload_rows,
                        dropped: skipped.length,
                        fileResults: results,
                      })
                      const srcLine = (cov.inventory_upload_sources ?? []).join(' · ')
                      showToast(
                        'success',
                        `${cov.inventory_upload_message || 'Inventory updated.'}${srcLine ? ` (${srcLine})` : ''}`,
                        skipped.length ? 14_000 : 10_000,
                      )
                      setInventoryAmzDisclaimer(
                        (cov.inventory_upload_amz_disclaimer as InventoryAmazonDisclaimer | undefined) ?? null,
                      )
                    } else {
                      setInventoryAmzDisclaimer((res.debug?.amz_disclaimer as InventoryAmazonDisclaimer | undefined) ?? null)
                      const issues = (res.debug && 'warnings' in res.debug && Array.isArray(res.debug.warnings))
                        ? (res.debug.warnings as string[])
                        : []
                      captureGenericAlert('inv', issues)
                      showToast('success', res.message)
                    }
                    await refresh({ light: true })
                    qc.invalidateQueries({ queryKey: ['inventory'] })
                  } else showToast('error', res.message)
                })
              } catch (e: unknown) {
                const { isUploadRequestInterrupted, waitForInventoryUpload } = await import('../api/client')
                if (isUploadRequestInterrupted(e)) {
                  showToast('success', 'Upload may still be processing on the server…', 6000)
                  setInvProgress({ pct: 5, msg: 'Checking server status…', phase: 'parse' })
                  try {
                    const cov = await waitForInventoryUpload(
                      (msg, pct) => {
                        setBuildingMsg(msg)
                        setInvProgress({ pct: pct ?? 5, msg, phase: 'parse' })
                      },
                      undefined,
                      () => invWaitAbortRef.current,
                    )
                    setBuildingMsg('')
                    setInvProgress(null)
                    setCoverage(cov)
                    showToast('success', cov.inventory_upload_message || 'Inventory updated.')
                    await refresh({ light: true })
                    qc.invalidateQueries({ queryKey: ['inventory'] })
                  } catch (pollErr: unknown) {
                    showToast(
                      'error',
                      pollErr instanceof Error
                        ? `${pollErr.message} Try “Clear stuck” below, then upload again.`
                        : 'Upload status unknown',
                    )
                  }
                } else {
                  showToast('error', e instanceof Error ? e.message : 'Upload failed')
                }
              } finally { setL('inv', false); setBuildingMsg(''); setInvProgress(null) }
            }}
          />
          {inventoryAmzDisclaimer && (
            <div className="rounded-lg border border-amber-200 bg-amber-50/70 p-3 text-xs text-amber-900">
              <p className="font-medium">Amazon inventory disclaimer</p>
              <p className="mt-1">
                Only latest 1 report day is used
                {inventoryAmzDisclaimer.latest_report_date ? ` (${inventoryAmzDisclaimer.latest_report_date})` : ''}.
              </p>
              <p className="mt-1">
                Raw: {Number(inventoryAmzDisclaimer.raw_total_units ?? 0).toLocaleString()} ·
                Non-sellable excluded: {Number(inventoryAmzDisclaimer.excluded_non_sellable_units ?? 0).toLocaleString()} ·
                ZNNE excluded: {Number(inventoryAmzDisclaimer.excluded_znne_units ?? 0).toLocaleString()} ·
                Older-date excluded: {Number(inventoryAmzDisclaimer.excluded_older_date_units ?? 0).toLocaleString()} ·
                Included: {Number(inventoryAmzDisclaimer.latest_report_units ?? 0).toLocaleString()}
              </p>
            </div>
          )}
        </UploadCard>

        <UploadCard
          title="↩ Returns (for PO)"
          subtitle={
            coverage.return_sheet_skus && coverage.return_sheet_skus > 0
              ? `${coverage.return_sheet_skus.toLocaleString()} SKUs · ${(coverage.return_sheet_units ?? 0).toLocaleString()} return units combined — upload more files for other brands/platforms below`
              : 'Upload return files per platform — Flipkart: monthly Sales Report XLSX (Return rows), Amazon/MTR, Myntra, Meesho, etc. Each file is merged; re-uploading the same filename replaces that file only.'
          }
          loaded={!!coverage.return_sheet}
          rows={coverage.return_sheet_skus}
          rowsUnit="SKUs"
          alert={showImportCompleteness ? uploadAlertsBySource['returns_po'] : undefined}
          onClearAlert={() => clearUploadAlert('returns_po')}
        >
          <ReturnsUploadPanel
            coverage={coverage}
            uploading={loading['returns_po']}
            onUpload={handleReturnsUpload}
          />
        </UploadCard>

      </Section>
      )}

      {(uploadTab === 'daily' || !showAdminTab) && anyLoaded && (
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
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">Verify data</h3>
        <div className={`grid grid-cols-1 gap-4 ${allowReset && (!showAdminTab || uploadTab === 'admin') ? 'md:grid-cols-2' : ''}`}>
          {allowReset && (!showAdminTab || uploadTab === 'admin') && (
          <div className="rounded-xl border border-red-200 bg-red-50/60 p-5 space-y-3">
            <h4 className="font-semibold text-red-900 text-sm">Clear all app data (Admin)</h4>
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
          )}
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

type CoverageLike = {
  sku_mapping: boolean
  mtr: boolean
  sales: boolean
  myntra: boolean
  meesho: boolean
  flipkart: boolean
  snapdeal: boolean
  inventory: boolean
  daily_orders?: boolean
  mtr_rows: number
  sales_rows: number
  myntra_rows: number
  meesho_rows: number
  flipkart_rows: number
  snapdeal_rows: number
}

function DataLoadedSummary({ coverage: c }: { coverage: CoverageLike }) {
  const rows: { key: string; label: string; loaded: boolean; detail: string }[] = [
    { key: 'sku', label: 'SKU mapping', loaded: c.sku_mapping, detail: c.sku_mapping ? 'Ready' : 'Upload master map' },
    { key: 'mtr', label: 'Amazon (MTR)', loaded: c.mtr, detail: c.mtr_rows > 0 ? `${c.mtr_rows.toLocaleString()} rows` : '—' },
    { key: 'myntra', label: 'Myntra PPMP', loaded: c.myntra, detail: c.myntra_rows > 0 ? `${c.myntra_rows.toLocaleString()} rows` : '—' },
    { key: 'meesho', label: 'Meesho', loaded: c.meesho, detail: c.meesho_rows > 0 ? `${c.meesho_rows.toLocaleString()} rows` : '—' },
    { key: 'flipkart', label: 'Flipkart', loaded: c.flipkart, detail: c.flipkart_rows > 0 ? `${c.flipkart_rows.toLocaleString()} rows` : '—' },
    { key: 'snapdeal', label: 'Snapdeal', loaded: c.snapdeal, detail: c.snapdeal_rows > 0 ? `${c.snapdeal_rows.toLocaleString()} rows` : '—' },
    { key: 'sales', label: 'Combined sales', loaded: c.sales, detail: c.sales_rows > 0 ? `${c.sales_rows.toLocaleString()} rows` : 'Build sales' },
    { key: 'inv', label: 'Snapshot inventory', loaded: c.inventory, detail: c.inventory ? 'Loaded' : 'Daily uploads tab' },
    { key: 'daily', label: 'Daily orders (Tier 3)', loaded: !!c.daily_orders, detail: c.daily_orders ? 'On server' : '—' },
  ]
  const missing = rows.filter(r => !r.loaded && r.key !== 'daily')
  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden mb-2">
      <div className="px-4 py-2 bg-slate-50 border-b border-gray-100 flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-semibold text-[#002B5B] uppercase tracking-wide">All datasets in this session</p>
        {missing.length > 0 && (
          <p className="text-[11px] text-amber-800">
            Missing: {missing.map(m => m.label).join(', ')} — upload below or click Restore all from server
          </p>
        )}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-px bg-gray-100">
        {rows.map(r => (
          <div
            key={r.key}
            id={`upload-dataset-${r.key}`}
            className={`px-3 py-2.5 bg-white ${r.loaded ? '' : 'opacity-80'}`}
          >
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide truncate">{r.label}</p>
            <p className={`text-sm font-bold mt-0.5 ${r.loaded ? 'text-green-700' : 'text-gray-400'}`}>
              {r.loaded ? '✓ Loaded' : '—'}
            </p>
            <p className="text-[10px] text-gray-500 truncate" title={r.detail}>
              {r.detail}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="scroll-mt-4">
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">{title}</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">{children}</div>
    </div>
  )
}

type ExistingPoProgressState = {
  pct: number
  msg: string
  phase: 'upload' | 'parse' | 'refresh'
  startedAt: number
}

function ExistingPoUploadProgress({ progress }: { progress: ExistingPoProgressState }) {
  const [elapsed, setElapsed] = useState(0)
  const [barPct, setBarPct] = useState(progress.pct)

  useEffect(() => {
    setElapsed(Math.max(0, Math.round((Date.now() - progress.startedAt) / 1000)))
    const id = window.setInterval(() => {
      setElapsed(Math.max(0, Math.round((Date.now() - progress.startedAt) / 1000)))
    }, 1000)
    return () => window.clearInterval(id)
  }, [progress.startedAt])

  useEffect(() => {
    if (progress.phase === 'upload') {
      setBarPct(progress.pct)
      return
    }
    if (progress.phase === 'parse') {
      setBarPct(prev => Math.max(prev, progress.pct, 55))
      const id = window.setInterval(() => {
        setBarPct(prev => (prev < 92 ? prev + 1 : prev))
      }, 900)
      return () => window.clearInterval(id)
    }
    setBarPct(98)
  }, [progress.phase, progress.pct])

  const steps: { key: ExistingPoProgressState['phase']; label: string }[] = [
    { key: 'upload', label: 'Upload' },
    { key: 'parse', label: 'Parse' },
    { key: 'refresh', label: 'Save' },
  ]
  const stepIndex = progress.phase === 'upload' ? 0 : progress.phase === 'parse' ? 1 : 2

  return (
    <div className="rounded-lg border border-sky-200 bg-sky-50 px-3 py-2.5 mb-2 space-y-2">
      <div className="flex items-center gap-2 text-xs text-sky-900">
        <svg className="animate-spin h-3.5 w-3.5 text-sky-600 shrink-0" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
        </svg>
        <span className="flex-1 font-medium">{progress.msg}</span>
        <span className="text-sky-600 tabular-nums shrink-0">{elapsed}s</span>
      </div>
      <div className="flex items-center gap-1 text-[10px] text-sky-800">
        {steps.map((s, i) => (
          <span
            key={s.key}
            className={
              i < stepIndex
                ? 'text-sky-600 font-semibold'
                : i === stepIndex
                  ? 'font-semibold text-sky-900'
                  : 'text-sky-400'
            }
          >
            {i > 0 ? ' → ' : ''}
            {s.label}
            {i === stepIndex ? '…' : i < stepIndex ? ' ✓' : ''}
          </span>
        ))}
      </div>
      <div className="space-y-0.5">
        <div className="h-2.5 rounded-full bg-sky-100 overflow-hidden">
          <div
            className={`h-full bg-sky-500 transition-all duration-300 ${
              progress.phase === 'parse' && barPct < 92 ? 'animate-pulse' : ''
            }`}
            style={{ width: `${Math.min(100, Math.max(2, barPct))}%` }}
          />
        </div>
        <p className="text-[10px] text-sky-700 text-right tabular-nums">
          {progress.phase === 'upload' ? `${barPct}%` : progress.phase === 'parse' ? 'Parsing…' : 'Almost done…'}
        </p>
      </div>
    </div>
  )
}

function UploadCard({
  title, subtitle, loaded, rows, rowsUnit = 'rows', onClear, clearing, alert, onClearAlert, children,
}: {
  title: string; subtitle: string; loaded: boolean
  rows?: number; rowsUnit?: string; onClear?: () => void; clearing?: boolean
  alert?: UploadAlert; onClearAlert?: () => void
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
          {loaded && (
            <span className="text-green-700 text-xs font-medium bg-green-50 px-2 py-0.5 rounded-full">
              {rows !== undefined && rows > 0 ? `✓ ${rows.toLocaleString()} ${rowsUnit}` : '✓ Loaded'}
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
      {alert && (
        <div className={`rounded-lg border p-3 text-xs ${
          alert.complete
            ? 'border-green-200 bg-green-50/70'
            : 'border-amber-200 bg-amber-50/70'
        }`}>
          <div className="flex items-start justify-between gap-2">
            <p className={`font-medium ${alert.complete ? 'text-green-900' : 'text-amber-900'}`}>
              {alert.title} · {alert.at}
            </p>
            {onClearAlert && (
              <button
                type="button"
                className={`shrink-0 px-1.5 py-0.5 rounded border ${
                  alert.complete
                    ? 'border-green-300 text-green-800 hover:bg-green-100'
                    : 'border-amber-300 text-amber-800 hover:bg-amber-100'
                }`}
                onClick={onClearAlert}
              >
                Clear
              </button>
            )}
          </div>
          <p className="text-gray-700 mt-0.5">
            Parsed: {alert.parsed ?? '—'} · Kept: {alert.kept ?? '—'} · Dropped: {alert.dropped ?? 0}
          </p>
          {alert.droppedReasons.length > 0 && (
            <p className={`${alert.complete ? 'text-green-900' : 'text-amber-900'} mt-1`}>Dropped reasons: {alert.droppedReasons.join(' | ')}</p>
          )}
          {alert.validationWarnings.length > 0 && (
            <ul className={`mt-1 list-disc list-inside ${alert.complete ? 'text-green-900' : 'text-amber-900'}`}>
              {alert.validationWarnings.map((w, wi) => <li key={`${alert.at}-${wi}`}>{w}</li>)}
            </ul>
          )}
          {alert.fileResults && alert.fileResults.some(f => f.status === 'skipped') && (
            <div className="mt-2 overflow-x-auto">
              <p className="font-medium text-amber-900 mb-1">Files not saved:</p>
              <table className="w-full text-[11px] border-collapse">
                <thead>
                  <tr className="text-left text-gray-600">
                    <th className="pr-2 py-0.5">File</th>
                    <th className="py-0.5">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {alert.fileResults.filter(f => f.status === 'skipped').map((f, i) => (
                    <tr key={`${f.filename}-${i}`} className="border-t border-amber-100">
                      <td className="pr-2 py-0.5 font-mono truncate max-w-[200px]" title={f.filename}>{f.filename}</td>
                      <td className="py-0.5">{f.reason ?? 'Not saved'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
      {children}
    </div>
  )
}

function Warn({ children }: { children: React.ReactNode }) {
  return <p className="text-xs text-amber-600 bg-amber-50 rounded p-2">{children}</p>
}

function _fmtBytes(b: number): string {
  if (b >= 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(1)} MB`
  if (b >= 1024) return `${Math.round(b / 1024)} KB`
  return `${b} B`
}

function _bumpProgress(prevPct: number, cap: number, step = 2): number {
  return Math.min(cap, Math.max(prevPct, prevPct + step))
}

function _isJunkUploadFile(name: string): boolean {
  const norm = name.replace(/\\/g, '/').trim().toLowerCase()
  const base = norm.includes('/') ? norm.slice(norm.lastIndexOf('/') + 1) : norm
  if (!base || norm.includes('__macosx/')) return true
  if (base === '.ds_store' || base === 'thumbs.db' || base === 'desktop.ini') return true
  if (base.startsWith('._')) return true
  return false
}

const _RETURN_PLATFORM_LABEL: Record<string, string> = {
  amazon: 'Amazon',
  flipkart: 'Flipkart',
  myntra: 'Myntra',
  meesho: 'Meesho',
  snapdeal: 'Snapdeal',
  unknown: 'Other',
}

function ReturnsUploadPanel({
  coverage,
  uploading,
  onUpload,
}: {
  coverage: CoverageResponse
  uploading: boolean
  onUpload: (
    file: File,
    onProgress?: (p: { pct: number; msg: string; loaded?: number; total?: number }) => void,
  ) => Promise<UploadResponse>
}) {
  const [queued, setQueued] = useState<File[]>([])
  const [progress, setProgress] = useState<{
    pct: number
    msg: string
    loaded?: number
    total?: number
  } | null>(null)

  const sources = coverage.return_overlay_sources ?? []
  const byPlatform = coverage.return_overlay_by_platform ?? []

  const onDrop = useCallback((accepted: File[]) => {
    if (!accepted.length) return
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
      'application/octet-stream': ['.xlsx', '.xls', '.rar', '.zip'],
      'application/x-rar-compressed': ['.rar'],
      'application/zip': ['.zip'],
    },
    multiple: true,
    disabled: uploading,
  })

  const submit = async () => {
    if (!queued.length) return
    for (let i = 0; i < queued.length; i++) {
      const file = queued[i]
      setProgress({
        pct: 2,
        msg: `Starting ${i + 1}/${queued.length}: ${file.name}`,
      })
      const res = await onUpload(file, p => setProgress(p))
      if (!res.ok) break
      setProgress({ pct: 100, msg: `Finished ${file.name}` })
    }
    setQueued([])
    setProgress(null)
  }

  return (
    <div className="space-y-3">
      {sources.length > 0 ? (
        <div className="rounded-lg border border-emerald-100 bg-emerald-50/40 p-2.5 space-y-2">
          <p className="text-[11px] font-semibold text-emerald-900">
            Loaded return files ({sources.length})
          </p>
          <div className="max-h-36 overflow-y-auto space-y-1">
            {sources.map(src => (
              <div
                key={src.filename}
                className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-[11px] text-gray-700 bg-white/80 rounded px-2 py-1"
              >
                <span className="font-medium truncate max-w-[55%]" title={src.filename}>
                  {src.filename}
                </span>
                <span className="text-gray-500">
                  {_RETURN_PLATFORM_LABEL[src.platform || 'unknown'] || src.platform || 'Other'}
                  {src.brand ? ` · ${src.brand}` : ''}
                </span>
                <span className="text-gray-600 tabular-nums">
                  {(src.skus ?? 0).toLocaleString()} SKUs · {(src.units ?? 0).toLocaleString()} units
                </span>
                {src.uploaded_at ? (
                  <span className="text-gray-400">
                    {new Date(src.uploaded_at).toLocaleString()}
                  </span>
                ) : null}
              </div>
            ))}
          </div>
          {byPlatform.length > 0 ? (
            <p className="text-[10px] text-gray-500 pt-1 border-t border-emerald-100">
              By platform:{' '}
              {byPlatform
                .map(p => {
                  const label = _RETURN_PLATFORM_LABEL[p.platform] || p.platform
                  return `${label} ${(p.units ?? 0).toLocaleString()} units`
                })
                .join(' · ')}
            </p>
          ) : null}
          {(coverage.returns_import_warnings?.length ?? 0) > 0 ? (
            <div className="rounded border border-amber-200 bg-amber-50/80 px-2 py-1.5 text-[10px] text-amber-900 space-y-0.5">
              <p className="font-semibold">Import warnings</p>
              {coverage.returns_import_warnings!.slice(0, 6).map((w, i) => (
                <p key={i}>{w}</p>
              ))}
              {(coverage.returns_import_warnings!.length ?? 0) > 6 ? (
                <p className="text-amber-700">
                  …and {coverage.returns_import_warnings!.length - 6} more
                </p>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : coverage.return_sheet ? (
        <p className="text-[11px] text-gray-500">
          {(coverage.return_sheet_skus ?? 0).toLocaleString()} SKUs ·{' '}
          {(coverage.return_sheet_units ?? 0).toLocaleString()} return units on server.
        </p>
      ) : null}

      <div className="space-y-1">
        <p className="text-sm font-medium text-gray-700">Upload returns file(s)</p>
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-5 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-blue-400 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400 bg-white'
          } ${uploading ? 'opacity-60 cursor-not-allowed' : ''}`}
        >
          <input {...getInputProps()} />
          {uploading && progress ? (
            <div className="space-y-2 text-left px-1">
              <div className="flex items-center justify-between text-xs text-blue-700">
                <span className="font-medium truncate max-w-[70%]">{progress.msg}</span>
                <span className="shrink-0 ml-2 tabular-nums">{progress.pct}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(100, Math.max(0, progress.pct))}%` }}
                />
              </div>
              {progress.total && progress.total > 0 ? (
                <p className="text-[10px] text-gray-500 tabular-nums">
                  {_fmtBytes(progress.loaded ?? 0)} / {_fmtBytes(progress.total)}
                </p>
              ) : null}
            </div>
          ) : isDragActive ? (
            <p className="text-sm text-blue-600">Drop return file(s) here</p>
          ) : (
            <p className="text-sm text-gray-500">
              Drag & drop one or more files or{' '}
              <span className="text-blue-600 underline">browse</span>
            </p>
          )}
        </div>
      </div>

      {queued.length > 0 && !uploading ? (
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-2 space-y-2">
          <p className="text-[11px] font-medium text-gray-700">
            Queued ({queued.length} file{queued.length > 1 ? 's' : ''})
          </p>
          <ul className="text-[11px] text-gray-600 space-y-0.5 max-h-24 overflow-y-auto">
            {queued.map(f => (
              <li key={f.name} className="flex justify-between gap-2">
                <span className="truncate">{f.name}</span>
                <button
                  type="button"
                  className="text-red-500 hover:underline shrink-0"
                  onClick={() => setQueued(q => q.filter(x => x.name !== f.name))}
                >
                  remove
                </button>
              </li>
            ))}
          </ul>
          <button
            type="button"
            className="w-full text-sm font-medium rounded-lg bg-[#002B5B] text-white py-2 hover:bg-[#003d7a] disabled:opacity-50"
            disabled={uploading}
            onClick={submit}
          >
            Upload {queued.length} return file{queued.length > 1 ? 's' : ''}
          </button>
        </div>
      ) : null}
    </div>
  )
}

function DailyDropzone({ uploading, chunkProgress, onUpload, onReject }: {
  uploading: boolean
  chunkProgress?: { pct: number; sent: number; total: number; msg: string } | null
  onUpload: (files: File[]) => Promise<void>
  onReject?: (message: string) => void
}) {
  const [queued, setQueued] = useState<File[]>([])

  const onDrop = useCallback((accepted: File[]) => {
    const sales = accepted.filter(f => !_isJunkUploadFile(f.name))
    const junk = accepted.length - sales.length
    if (junk > 0 && onReject) {
      onReject(
        `Ignored ${junk} system file(s) (.DS_Store, etc.). ` +
          `${sales.length ? 'Sales file(s) queued.' : 'Select your Sales RAR/ZIP or CSV/XLSX exports.'}`,
      )
    }
    if (!sales.length) return
    setQueued(prev => {
      const names = new Set(prev.map(f => f.name))
      return [...prev, ...sales.filter(f => !names.has(f.name))]
    })
  }, [onReject])

  const onDropRejected = useCallback((rej: FileRejection[]) => {
    if (!rej.length || !onReject) return
    const bit = rej
      .map((r) => `${r.file.name} (${r.errors.map((e) => e.message).join(', ')})`)
      .join('; ')
    onReject(
      `Some files were not accepted (browser MIME type). ` +
        `Try again or use “browse” — ${bit}. ` +
        `RAR/ZIP often report as application/octet-stream; the dropzone now allows that.`,
    )
  }, [onReject])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: {
      'text/csv': ['.csv'],
      'text/plain': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls', '.xlsx'],
      'application/zip': ['.zip'],
      'application/x-zip-compressed': ['.zip'],
      'application/x-rar-compressed': ['.rar'],
      'application/vnd.rar': ['.rar'],
      // Browsers / OS often use generic type for RAR, XLSB, and some CSV/ZIP exports
      'application/octet-stream': ['.rar', '.zip', '.csv', '.xlsx', '.xls', '.xlsb'],
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
        className={`border-2 border-dashed rounded-lg px-6 py-5 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400 bg-white'}
          ${uploading ? 'cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        {uploading && chunkProgress && chunkProgress.total > 0
          ? (
            <div className="space-y-2 text-left px-1">
              <div className="flex items-center justify-between text-xs text-blue-700">
                <span className="font-medium truncate max-w-[70%]">{chunkProgress.msg}</span>
                <span className="shrink-0 ml-2 tabular-nums">
                  {_fmtBytes(chunkProgress.sent)} / {_fmtBytes(chunkProgress.total)}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${chunkProgress.pct}%` }}
                />
              </div>
              <p className="text-xs text-blue-500 text-right tabular-nums">{chunkProgress.pct}%</p>
            </div>
          )
            : uploading
            ? (
              <div className="flex items-center justify-center gap-2 text-sm text-blue-600">
                <svg className="animate-spin h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
                <span>{chunkProgress?.msg || 'Sending to server…'}</span>
              </div>
            )
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

function DailyHistory({ allowDelete = false }: { allowDelete?: boolean }) {
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
                  {allowDelete ? (
                    <button
                      onClick={() => deleteMut.mutate(u.id)}
                      disabled={deleteMut.isPending}
                      title="Delete this upload"
                      className="text-gray-300 hover:text-red-500 transition-colors shrink-0 ml-1 disabled:opacity-40"
                    >
                      🗑
                    </button>
                  ) : null}
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

function UploadSkipDetailsPanel({
  report,
  title = 'Import skip details',
}: {
  report?: ManualIntransitParseReport | null
  title?: string
}) {
  if (!report) return null
  const details = report.skip_details ?? []
  const skippedSheets = report.sheets_skipped ?? []
  const warnings = report.warnings ?? []
  if (!details.length && !skippedSheets.length && !warnings.length) return null

  return (
    <div className="mb-3 rounded-lg border border-amber-200 bg-amber-50/90 px-3 py-2 text-[11px] text-amber-950 space-y-1.5">
      <p className="font-semibold text-amber-900">{title}</p>
      {warnings.map((w, i) => (
        <p key={`w-${i}`} className="text-amber-800">
          {w}
        </p>
      ))}
      {skippedSheets.map((s, i) => (
        <p key={`ss-${i}`}>
          <span className="font-medium">{s.sheet}</span>: {s.reason}
        </p>
      ))}
      {details.length > 0 ? (
        <div className="max-h-32 overflow-y-auto space-y-1 pt-1 border-t border-amber-200/80">
          {details.map((d, i) => (
            <p key={`d-${i}`}>
              <span className="font-medium">{d.sheet || d.kind || 'Row'}</span>
              {d.kind && d.sheet ? ` (${d.kind})` : ''}: {d.reason}
              {d.rows_affected != null && d.rows_affected > 0 ? (
                <span className="text-amber-700"> — {d.rows_affected.toLocaleString()} row(s)</span>
              ) : null}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  )
}
