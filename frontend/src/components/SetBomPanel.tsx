import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'
import api from '../api/client'

function apiErrorMessage(err: unknown, fallback: string): string {
  if (!axios.isAxiosError(err)) return fallback
  const detail = err.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  if (err.message) return err.message
  return fallback
}

const DEFAULT_PROCESSES = ['Cutting', 'Printing', 'Embroidery', 'Stitching', 'Finishing', 'Packing']

const SIZE_TOKENS = new Set(['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL', '2XL', '3XL', '4XL', '5XL', '6XL', '7XL', '8XL'])

export function parentStyleKeyFromSku(sku: string): string {
  const s = String(sku || '').trim().toUpperCase()
  if (!s.includes('-')) return s
  const parts = s.split('-')
  const last = parts[parts.length - 1]
  if (SIZE_TOKENS.has(last) || /^\d{1,2}$/.test(last) || last.endsWith('XL')) {
    return parts.slice(0, -1).join('-')
  }
  return s
}

type SetBomLine = {
  component_code: string
  component_name: string
  qty_per_set: number
  default_next_process: string
}

type Props = {
  processes?: string[]
  initialStyleKey?: string
  initialStyleName?: string
  /** When true, show Set Match board (Production). Item Master hides this. */
  showSetMatch?: boolean
}

export default function SetBomPanel({
  processes = DEFAULT_PROCESSES,
  initialStyleKey = '',
  initialStyleName = '',
  showSetMatch = false,
}: Props) {
  const qc = useQueryClient()
  const [setBomForm, setSetBomForm] = useState({
    style_key: initialStyleKey,
    style_name: initialStyleName,
    lines: [
      { component_code: 'TOP', component_name: 'Top', qty_per_set: 1, default_next_process: 'Stitching' },
      { component_code: 'PANT', component_name: 'Pant', qty_per_set: 1, default_next_process: 'Stitching' },
      { component_code: 'DUPATTA', component_name: 'Dupatta', qty_per_set: 1, default_next_process: 'Embroidery' },
    ] as SetBomLine[],
  })
  const [setMatchForm, setSetMatchForm] = useState({
    so_number: '',
    main_sku: '',
    from_process: 'Finishing',
    match_qty: 0,
  })
  const [setMatchPreview, setSetMatchPreview] = useState<any>(null)

  const { data: setBoms = [] } = useQuery({
    queryKey: ['set-boms'],
    queryFn: () => api.get('/production/set-bom').then(r => r.data || []),
  })

  const saveSetBomMut = useMutation({
    mutationFn: (data: object) => api.post('/production/set-bom', data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['set-boms'] })
      alert('Set BOM saved')
    },
    onError: (e: unknown) => alert(apiErrorMessage(e, 'Failed to save Set BOM')),
  })

  const deleteSetBomMut = useMutation({
    mutationFn: (styleKey: string) => api.delete(`/production/set-bom/${encodeURIComponent(styleKey)}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['set-boms'] }),
    onError: (e: unknown) => alert(apiErrorMessage(e, 'Delete failed')),
  })

  const commitSetMatchMut = useMutation({
    mutationFn: (data: object) => api.post('/production/set-match', data),
    onSuccess: (res) => {
      alert(res.data?.message || 'Set match committed')
      setSetMatchPreview(null)
      qc.invalidateQueries({ queryKey: ['jos-process'] })
      qc.invalidateQueries({ queryKey: ['jos-all'] })
    },
    onError: (e: unknown) => alert(apiErrorMessage(e, 'Set match failed')),
  })

  const loadStyleKey = async (styleKey: string, styleName = '') => {
    const key = parentStyleKeyFromSku(styleKey)
    setSetBomForm(f => ({ ...f, style_key: key, style_name: styleName || f.style_name }))
    try {
      const res = await api.get(`/production/set-bom/${encodeURIComponent(key)}`)
      const bom = res.data
      if (bom?.lines?.length) {
        setSetBomForm({
          style_key: bom.style_key,
          style_name: bom.style_name || styleName,
          lines: bom.lines.map((l: any) => ({
            component_code: l.component_code,
            component_name: l.component_name || l.component_code,
            qty_per_set: l.qty_per_set || 1,
            default_next_process: l.default_next_process || '',
          })),
        })
      } else {
        setSetBomForm(f => ({ ...f, style_key: key, style_name: styleName || f.style_name }))
      }
    } catch {
      setSetBomForm(f => ({ ...f, style_key: key, style_name: styleName || f.style_name }))
    }
  }

  useEffect(() => {
    if (initialStyleKey) {
      void loadStyleKey(initialStyleKey, initialStyleName)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialStyleKey, initialStyleName])

  return (
    <div className={`grid grid-cols-1 ${showSetMatch ? 'lg:grid-cols-2' : ''} gap-4`}>
      <div className="bg-white rounded-xl border p-4 space-y-3">
        <h3 className="font-semibold text-gray-800">Set BOM (component recipe)</h3>
        <p className="text-xs text-gray-500">
          Define Top / Pant / Dupatta for a style parent (e.g. <span className="font-mono">1001YKBEIGE</span>).
          After Cutting receive, size SKUs split into <span className="font-mono">1001YKBEIGE-XS-TOP</span> etc.
        </p>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-xs text-gray-500">Style key *</label>
            <input
              value={setBomForm.style_key}
              onChange={e => setSetBomForm(f => ({ ...f, style_key: e.target.value.toUpperCase() }))}
              placeholder="e.g. 1001YKBEIGE"
              className="w-full border rounded px-2 py-1.5 text-sm mt-1 font-mono"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500">Style name</label>
            <input
              value={setBomForm.style_name}
              onChange={e => setSetBomForm(f => ({ ...f, style_name: e.target.value }))}
              className="w-full border rounded px-2 py-1.5 text-sm mt-1"
            />
          </div>
        </div>
        <div className="border rounded-lg overflow-hidden">
          <table className="w-full text-xs">
            <thead className="bg-gray-50 text-gray-500">
              <tr>
                <th className="text-left px-2 py-1.5">Code</th>
                <th className="text-left px-2 py-1.5">Name</th>
                <th className="text-right px-2 py-1.5">Qty/set</th>
                <th className="text-left px-2 py-1.5">Next process</th>
                <th className="px-2 py-1.5" />
              </tr>
            </thead>
            <tbody>
              {setBomForm.lines.map((ln, i) => (
                <tr key={i} className="border-t">
                  <td className="px-2 py-1">
                    <input
                      value={ln.component_code}
                      onChange={e => setSetBomForm(f => ({
                        ...f,
                        lines: f.lines.map((x, j) => j === i ? { ...x, component_code: e.target.value.toUpperCase() } : x),
                      }))}
                      className="w-full border rounded px-1.5 py-0.5 font-mono"
                    />
                  </td>
                  <td className="px-2 py-1">
                    <input
                      value={ln.component_name}
                      onChange={e => setSetBomForm(f => ({
                        ...f,
                        lines: f.lines.map((x, j) => j === i ? { ...x, component_name: e.target.value } : x),
                      }))}
                      className="w-full border rounded px-1.5 py-0.5"
                    />
                  </td>
                  <td className="px-2 py-1">
                    <input
                      type="number"
                      value={ln.qty_per_set}
                      onChange={e => setSetBomForm(f => ({
                        ...f,
                        lines: f.lines.map((x, j) => j === i ? { ...x, qty_per_set: +e.target.value } : x),
                      }))}
                      className="w-16 border rounded px-1.5 py-0.5 text-right"
                    />
                  </td>
                  <td className="px-2 py-1">
                    <select
                      value={ln.default_next_process}
                      onChange={e => setSetBomForm(f => ({
                        ...f,
                        lines: f.lines.map((x, j) => j === i ? { ...x, default_next_process: e.target.value } : x),
                      }))}
                      className="w-full border rounded px-1.5 py-0.5"
                    >
                      <option value="">—</option>
                      {processes.map(p => <option key={p}>{p}</option>)}
                    </select>
                  </td>
                  <td className="px-2 py-1 text-center">
                    <button
                      onClick={() => setSetBomForm(f => ({ ...f, lines: f.lines.filter((_, j) => j !== i) }))}
                      className="text-red-400 hover:text-red-600"
                    >
                      ✕
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => setSetBomForm(f => ({
              ...f,
              lines: [...f.lines, { component_code: '', component_name: '', qty_per_set: 1, default_next_process: '' }],
            }))}
            className="px-3 py-1.5 text-xs border rounded-lg"
          >
            + Component
          </button>
          <button
            onClick={() => saveSetBomMut.mutate(setBomForm)}
            disabled={saveSetBomMut.isPending || !setBomForm.style_key || setBomForm.lines.length === 0}
            className="px-3 py-1.5 text-xs bg-[#002B5B] text-white rounded-lg disabled:opacity-50"
          >
            {saveSetBomMut.isPending ? 'Saving…' : 'Save Set BOM'}
          </button>
        </div>
        <div className="border-t pt-3 space-y-1">
          <p className="text-xs font-semibold text-gray-600">Saved Set BOMs</p>
          {(setBoms as any[]).length === 0 && <p className="text-xs text-gray-400">None yet.</p>}
          {(setBoms as any[]).map((b: any) => (
            <div key={b.style_key} className="flex items-center justify-between text-xs border rounded-lg px-2 py-1.5">
              <button
                type="button"
                className="text-left"
                onClick={() => setSetBomForm({
                  style_key: b.style_key,
                  style_name: b.style_name || '',
                  lines: (b.lines || []).map((l: any) => ({
                    component_code: l.component_code,
                    component_name: l.component_name || l.component_code,
                    qty_per_set: l.qty_per_set || 1,
                    default_next_process: l.default_next_process || '',
                  })),
                })}
              >
                <span className="font-mono font-semibold text-[#002B5B]">{b.style_key}</span>
                <span className="text-gray-500 ml-2">{(b.lines || []).map((l: any) => l.component_code).join(' + ')}</span>
              </button>
              <button
                type="button"
                onClick={() => { if (confirm(`Delete Set BOM ${b.style_key}?`)) deleteSetBomMut.mutate(b.style_key) }}
                className="text-red-500 text-xs"
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      </div>

      {showSetMatch && (
        <div className="bg-white rounded-xl border p-4 space-y-3">
          <h3 className="font-semibold text-gray-800">Set Match (Finishing → Packing)</h3>
          <p className="text-xs text-gray-500">Complete sets = min(component avail at Finishing). Extras stay as component WIP.</p>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-gray-500">SO number *</label>
              <input
                value={setMatchForm.so_number}
                onChange={e => setSetMatchForm(f => ({ ...f, so_number: e.target.value }))}
                className="w-full border rounded px-2 py-1.5 text-sm mt-1 font-mono"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500">Main size SKU *</label>
              <input
                value={setMatchForm.main_sku}
                onChange={e => setSetMatchForm(f => ({ ...f, main_sku: e.target.value.toUpperCase() }))}
                placeholder="1001-XS"
                className="w-full border rounded px-2 py-1.5 text-sm mt-1 font-mono"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500">From process</label>
              <select
                value={setMatchForm.from_process}
                onChange={e => setSetMatchForm(f => ({ ...f, from_process: e.target.value }))}
                className="w-full border rounded px-2 py-1.5 text-sm mt-1"
              >
                {processes.map(p => <option key={p}>{p}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500">Match qty</label>
              <input
                type="number"
                value={setMatchForm.match_qty || ''}
                onChange={e => setSetMatchForm(f => ({ ...f, match_qty: +e.target.value }))}
                placeholder="max complete"
                className="w-full border rounded px-2 py-1.5 text-sm mt-1"
              />
            </div>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={async () => {
                try {
                  const res = await api.get('/production/set-match', {
                    params: {
                      so_number: setMatchForm.so_number,
                      main_sku: setMatchForm.main_sku,
                      from_process: setMatchForm.from_process,
                    },
                  })
                  setSetMatchPreview(res.data)
                  setSetMatchForm(f => ({ ...f, match_qty: res.data.complete_sets || 0 }))
                } catch (e) {
                  alert(apiErrorMessage(e, 'Preview failed'))
                  setSetMatchPreview(null)
                }
              }}
              disabled={!setMatchForm.so_number || !setMatchForm.main_sku}
              className="px-3 py-1.5 text-xs border rounded-lg disabled:opacity-50"
            >
              Preview
            </button>
            <button
              type="button"
              onClick={() => commitSetMatchMut.mutate({
                so_number: setMatchForm.so_number,
                main_sku: setMatchForm.main_sku,
                from_process: setMatchForm.from_process,
                to_process: 'Packing',
                match_qty: setMatchForm.match_qty || undefined,
              })}
              disabled={commitSetMatchMut.isPending || !setMatchForm.so_number || !setMatchForm.main_sku}
              className="px-3 py-1.5 text-xs bg-green-700 text-white rounded-lg disabled:opacity-50"
            >
              {commitSetMatchMut.isPending ? 'Matching…' : 'Commit Set Match → Packing'}
            </button>
          </div>
          {setMatchPreview && (
            <div className="border rounded-lg p-3 space-y-2 bg-green-50/40">
              <p className="text-sm font-semibold text-green-800">
                Complete sets available: {setMatchPreview.complete_sets}
              </p>
              <table className="w-full text-xs">
                <thead className="text-gray-500">
                  <tr>
                    <th className="text-left py-1">Component</th>
                    <th className="text-right py-1">Avail</th>
                    <th className="text-right py-1">Matched</th>
                    <th className="text-right py-1">Extra WIP</th>
                    <th className="text-right py-1">Shortfall</th>
                  </tr>
                </thead>
                <tbody>
                  {(setMatchPreview.components || []).map((c: any) => (
                    <tr key={c.component_code} className="border-t border-green-100">
                      <td className="py-1 font-mono">{c.component_sku || c.component_code}</td>
                      <td className="py-1 text-right">{c.available_qty}</td>
                      <td className="py-1 text-right">{c.matched_qty}</td>
                      <td className="py-1 text-right text-amber-700 font-semibold">{c.extra_qty}</td>
                      <td className="py-1 text-right text-red-600">{c.shortfall_to_max_peer || 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export { SetBomPanel }
