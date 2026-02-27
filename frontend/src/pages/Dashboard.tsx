/**
 * Dashboard page — home screen with upload cards.
 * This is the "working" Phase 1 end-to-end: upload SKU mapping → success toast.
 */
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import FileUpload from '../components/FileUpload'
import { uploadSkuMapping, uploadMtr, getCoverage } from '../api/client'
import { useSession } from '../store/session'

type Toast = { type: 'success' | 'error'; msg: string }

export default function Dashboard() {
  const setCoverage = useSession((s) => s.setCoverage)
  const coverage = useSession()

  const [toast, setToast] = useState<Toast | null>(null)
  const [loadingSkuMap, setLoadingSkuMap] = useState(false)
  const [loadingMtr, setLoadingMtr] = useState(false)

  // Poll coverage every 5 s
  useQuery({
    queryKey: ['coverage'],
    queryFn: async () => {
      const c = await getCoverage()
      setCoverage(c)
      return c
    },
    refetchInterval: 5000,
  })

  const showToast = (type: 'success' | 'error', msg: string) => {
    setToast({ type, msg })
    setTimeout(() => setToast(null), 4000)
  }

  const handleSkuMap = async (file: File) => {
    setLoadingSkuMap(true)
    try {
      const res = await uploadSkuMapping(file)
      if (res.ok) {
        showToast('success', res.message)
        const c = await getCoverage()
        setCoverage(c)
      } else {
        showToast('error', res.message)
      }
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setLoadingSkuMap(false)
    }
  }

  const handleMtr = async (file: File) => {
    setLoadingMtr(true)
    try {
      const res = await uploadMtr(file)
      if (res.ok) {
        showToast('success', res.message)
        const c = await getCoverage()
        setCoverage(c)
      } else {
        showToast('error', res.message)
      }
    } catch (e: unknown) {
      showToast('error', e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setLoadingMtr(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-[#002B5B]">📊 Dashboard</h2>
        <p className="text-gray-500 text-sm mt-1">
          Upload your data files to get started.
        </p>
      </div>

      {/* Toast */}
      {toast && (
        <div
          className={`fixed top-4 right-4 z-50 rounded-lg px-5 py-3 shadow-lg text-sm text-white ${
            toast.type === 'success' ? 'bg-green-600' : 'bg-red-600'
          }`}
        >
          {toast.msg}
        </div>
      )}

      {/* Coverage summary */}
      <div className="grid grid-cols-3 gap-4">
        <KpiCard label="SKU Mapping" value={coverage.sku_mapping ? '✅ Loaded' : '— Not loaded'} />
        <KpiCard label="MTR Rows"    value={coverage.mtr_rows > 0 ? coverage.mtr_rows.toLocaleString() : '—'} />
        <KpiCard label="Sales Rows"  value={coverage.sales_rows > 0 ? coverage.sales_rows.toLocaleString() : '—'} />
      </div>

      {/* Upload cards */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4">
          <h3 className="font-semibold text-[#002B5B]">1️⃣ SKU Mapping (required first)</h3>
          <FileUpload
            label="Upload Excel (.xlsx)"
            accept={{ 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'] }}
            onUpload={handleSkuMap}
            uploading={loadingSkuMap}
          />
        </div>

        <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4">
          <h3 className="font-semibold text-[#002B5B]">2️⃣ Amazon MTR (historical ZIP)</h3>
          {!coverage.sku_mapping && (
            <p className="text-xs text-amber-600 bg-amber-50 rounded p-2">
              Upload SKU Mapping first.
            </p>
          )}
          <FileUpload
            label="Upload ZIP"
            accept={{ 'application/zip': ['.zip'] }}
            onUpload={handleMtr}
            uploading={loadingMtr}
          />
        </div>
      </div>

      {/* Coming soon placeholders */}
      <div className="grid grid-cols-2 gap-4 opacity-50 pointer-events-none">
        {['Myntra PPMP ZIP', 'Meesho ZIP', 'Flipkart ZIP', 'Inventory CSVs'].map((lbl) => (
          <div key={lbl} className="bg-white rounded-xl border border-gray-200 p-5">
            <h3 className="font-semibold text-gray-400">{lbl}</h3>
            <p className="text-xs text-gray-400 mt-1">Coming soon</p>
          </div>
        ))}
      </div>
    </div>
  )
}

function KpiCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-white rounded-xl border border-l-4 border-[#002B5B] p-4 shadow-sm">
      <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">{label}</p>
      <p className="text-xl font-bold text-gray-800 mt-1">{value}</p>
    </div>
  )
}
