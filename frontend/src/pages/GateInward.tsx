import { useCallback, useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type ScanLine = {
  line_key?: string
  material_code?: string
  material_name?: string
  sku?: string
  planned_qty?: number
  already_received_qty?: number
  pending_qty?: number
  received_qty?: number
  unit?: string
  jo_id?: number
  jo_line_id?: number
}

type ScanResult = {
  source_type: string
  source_number: string
  party_name?: string
  stage?: string
  barcode_payload?: string
  lines?: ScanLine[]
  lookup_only?: boolean
  message?: string
  gin?: { gin_number: string }
}

export default function GateInward() {
  const qc = useQueryClient()
  const inputRef = useRef<HTMLInputElement>(null)
  const [code, setCode] = useState('')
  const [scan, setScan] = useState<ScanResult | null>(null)
  const [lines, setLines] = useState<ScanLine[]>([])
  const [vehicle, setVehicle] = useState('')
  const [challan, setChallan] = useState('')
  const [remarks, setRemarks] = useState('')
  const [error, setError] = useState('')
  const [lastGin, setLastGin] = useState<string>('')

  const { data: recent = [] } = useQuery({
    queryKey: ['gins'],
    queryFn: () => api.get('/gate/gin?limit=20').then(r => r.data),
  })

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const doScan = useCallback(async (raw?: string) => {
    const c = (raw ?? code).trim()
    if (!c) return
    setError('')
    try {
      const { data } = await api.get<ScanResult>('/gate/scan', { params: { code: c } })
      setScan(data)
      setLines(
        (data.lines || []).map(l => ({
          ...l,
          received_qty: l.received_qty ?? l.pending_qty ?? 0,
        })),
      )
      if (data.lookup_only) {
        setError(data.message || 'Document is lookup-only (not a gate receive).')
      }
    } catch (e: any) {
      setScan(null)
      setLines([])
      setError(e?.response?.data?.detail || e?.message || 'Scan failed')
    }
  }, [code])

  const saveMut = useMutation({
    mutationFn: async () => {
      if (!scan) throw new Error('Scan a document first')
      const payload = {
        source_type: scan.source_type,
        source_number: scan.source_number,
        party_name: scan.party_name || '',
        stage: scan.stage || '',
        vehicle_no: vehicle,
        challan_no: challan,
        remarks,
        created_by: 'Gate',
        lines: lines.map(l => ({
          line_key: l.line_key || '',
          material_code: l.material_code || l.sku || '',
          material_name: l.material_name || '',
          sku: l.sku || l.material_code || '',
          planned_qty: l.planned_qty || 0,
          already_received_qty: l.already_received_qty || 0,
          pending_qty: l.pending_qty || 0,
          received_qty: Number(l.received_qty || 0),
          unit: l.unit || 'PCS',
          jo_id: l.jo_id,
          jo_line_id: l.jo_line_id,
        })),
      }
      const { data } = await api.post('/gate/gin', payload)
      return data
    },
    onSuccess: (data) => {
      const num = data.gin_number || data.gin?.gin_number
      setLastGin(num || '')
      qc.invalidateQueries({ queryKey: ['gins'] })
      setScan(null)
      setLines([])
      setCode('')
      setVehicle('')
      setChallan('')
      setRemarks('')
      inputRef.current?.focus()
      if (data.gin?.id) {
        window.open(`/api/gate/gin/${data.gin.id}/print`, '_blank')
      }
      alert(
        `GIN ${num} saved` +
          (data.grn_number ? ` · GRN ${data.grn_number}` : '') +
          (data.jo_receipt_ids?.length ? ` · JO receipts ${data.jo_receipt_ids.length}` : ''),
      )
    },
    onError: (e: any) => {
      setError(e?.response?.data?.detail || e?.message || 'Save failed')
    },
  })

  return (
    <div className="p-4 md:p-6 max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-[#002B5B]">Gate Inward (GIN)</h1>
        <p className="text-sm text-gray-500 mt-1">
          Scan PO / Job Order / JWO barcode — enter received qty only — save to create GIN + GRN / JO receive.
        </p>
      </div>

      <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-3">
        <label className="block text-xs font-semibold uppercase text-gray-500">Scan barcode / paste code</label>
        <div className="flex gap-2">
          <input
            ref={inputRef}
            value={code}
            onChange={e => setCode(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') {
                e.preventDefault()
                void doScan()
              }
            }}
            placeholder="PO:PO-0001  or  JO:JO-0042"
            className="flex-1 text-lg px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#002B5B]/outline-none"
            autoComplete="off"
          />
          <button
            type="button"
            onClick={() => void doScan()}
            className="px-5 py-3 rounded-lg bg-[#002B5B] text-white font-semibold hover:bg-[#003d7a]"
          >
            Scan
          </button>
        </div>
        {lastGin && (
          <div className="text-sm text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-2">
            Last GIN: <strong>{lastGin}</strong>
          </div>
        )}
        {error && (
          <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">{error}</div>
        )}
      </div>

      {scan && !scan.lookup_only && (
        <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div>
              <div className="text-xs text-gray-500 uppercase font-semibold">Document</div>
              <div className="font-bold text-[#002B5B]">{scan.source_type}: {scan.source_number}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500 uppercase font-semibold">Vendor</div>
              <div className="font-semibold">{scan.party_name || '—'}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500 uppercase font-semibold">Stage</div>
              <div className="font-semibold">{scan.stage || '—'}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500 uppercase font-semibold">Payload</div>
              <div className="font-mono text-xs">{scan.barcode_payload || '—'}</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <input
              value={challan}
              onChange={e => setChallan(e.target.value)}
              placeholder="Challan No"
              className="px-3 py-2 border rounded-lg text-sm"
            />
            <input
              value={vehicle}
              onChange={e => setVehicle(e.target.value)}
              placeholder="Vehicle No"
              className="px-3 py-2 border rounded-lg text-sm"
            />
            <input
              value={remarks}
              onChange={e => setRemarks(e.target.value)}
              placeholder="Remarks"
              className="px-3 py-2 border rounded-lg text-sm"
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-[#002B5B] text-white text-left">
                  <th className="px-3 py-2">SKU / Code</th>
                  <th className="px-3 py-2">Name</th>
                  <th className="px-3 py-2 text-right">Planned</th>
                  <th className="px-3 py-2 text-right">Already</th>
                  <th className="px-3 py-2 text-right">Pending</th>
                  <th className="px-3 py-2 text-right">Received Qty</th>
                </tr>
              </thead>
              <tbody>
                {lines.map((l, i) => (
                  <tr key={l.line_key || i} className="border-b border-gray-100">
                    <td className="px-3 py-2 font-semibold">{l.sku || l.material_code}</td>
                    <td className="px-3 py-2 text-gray-600">{l.material_name || '—'}</td>
                    <td className="px-3 py-2 text-right">{l.planned_qty ?? 0}</td>
                    <td className="px-3 py-2 text-right">{l.already_received_qty ?? 0}</td>
                    <td className="px-3 py-2 text-right font-semibold text-amber-700">{l.pending_qty ?? 0}</td>
                    <td className="px-3 py-2 text-right">
                      <input
                        type="number"
                        min={0}
                        max={l.pending_qty}
                        step="any"
                        value={l.received_qty ?? 0}
                        onChange={e => {
                          const v = e.target.value === '' ? 0 : Number(e.target.value)
                          setLines(prev => prev.map((row, idx) => (idx === i ? { ...row, received_qty: v } : row)))
                        }}
                        className="w-28 text-right px-2 py-1.5 border-2 border-[#002B5B] rounded-lg font-bold"
                      />
                    </td>
                  </tr>
                ))}
                {lines.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-3 py-6 text-center text-gray-500">
                      No pending quantity on this document.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => {
                setScan(null)
                setLines([])
                inputRef.current?.focus()
              }}
              className="px-4 py-2 border rounded-lg text-sm"
            >
              Clear
            </button>
            <button
              type="button"
              disabled={saveMut.isPending || lines.length === 0}
              onClick={() => saveMut.mutate()}
              className="px-6 py-2 rounded-lg bg-emerald-600 text-white font-semibold hover:bg-emerald-700 disabled:opacity-50"
            >
              {saveMut.isPending ? 'Saving…' : 'Complete GIN'}
            </button>
          </div>
        </div>
      )}

      <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm">
        <h2 className="font-semibold text-gray-800 mb-3">Recent GINs</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase text-gray-500 border-b">
                <th className="py-2">GIN</th>
                <th>Source</th>
                <th>Party</th>
                <th>Stage</th>
                <th>Date</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {(recent as any[]).map((g: any) => (
                <tr key={g.id} className="border-b border-gray-50">
                  <td className="py-2 font-semibold text-[#002B5B]">{g.gin_number}</td>
                  <td>
                    {g.source_type}:{g.source_number}
                  </td>
                  <td>{g.party_name || '—'}</td>
                  <td>{g.stage || '—'}</td>
                  <td>{g.gin_date}</td>
                  <td>
                    <a
                      className="text-xs text-blue-600 hover:underline"
                      href={`/api/gate/gin/${g.id}/print`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Print
                    </a>
                  </td>
                </tr>
              ))}
              {!(recent as any[]).length && (
                <tr>
                  <td colSpan={6} className="py-4 text-center text-gray-400">
                    No GINs yet
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
