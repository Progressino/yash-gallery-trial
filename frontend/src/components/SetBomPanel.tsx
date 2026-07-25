import { Fragment, useEffect, useState } from 'react'
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

type SetBomMaterial = {
  material_code: string
  material_name: string
  quantity: number
  unit: string
}

type ItemSearchHit = {
  id: number
  item_code: string
  item_name: string
  item_type_code?: string
  uom?: string
}

const MATERIAL_ITEM_TYPES = new Set(['RM', 'GF', 'SFG', 'ACC', 'PKG', 'FUEL'])

function MaterialPicker({
  materialCode,
  materialName,
  onSelect,
  onClear,
}: {
  materialCode: string
  materialName: string
  onSelect: (code: string, name: string, unit: string) => void
  onClear: () => void
}) {
  const [q, setQ] = useState('')
  const { data: rawResults = [] } = useQuery<ItemSearchHit[]>({
    queryKey: ['item-search-set-bom-material', q],
    queryFn: async () => {
      const { data } = await api.get(`/items/search?q=${encodeURIComponent(q)}`)
      return data
    },
    enabled: q.trim().length >= 2,
    staleTime: 30_000,
  })
  const results = rawResults.filter(
    it => !it.item_type_code || MATERIAL_ITEM_TYPES.has(String(it.item_type_code).toUpperCase()),
  )

  if (materialCode) {
    return (
      <div className="flex items-center gap-1 min-w-[200px] flex-1">
        <div className="flex-1 bg-blue-50 border border-blue-200 rounded px-2 py-1 text-[11px]">
          <div className="font-mono font-medium text-[#002B5B]">{materialCode}</div>
          {materialName && materialName !== materialCode && (
            <div className="text-gray-500 truncate">{materialName}</div>
          )}
        </div>
        <button type="button" onClick={onClear} className="text-gray-400 hover:text-red-500 text-xs px-1">
          ✕
        </button>
      </div>
    )
  }

  return (
    <div className="relative min-w-[200px] flex-1">
      <input
        type="text"
        value={q}
        onChange={e => setQ(e.target.value)}
        placeholder="Search material / fabric…"
        className="w-full border rounded px-1.5 py-0.5 text-[11px]"
      />
      {q.trim().length >= 2 && results.length > 0 && (
        <div className="absolute z-20 top-full left-0 right-0 mt-0.5 border border-gray-200 rounded-lg divide-y divide-gray-100 max-h-40 overflow-y-auto bg-white shadow-lg">
          {results.map(it => (
            <button
              key={it.id}
              type="button"
              onClick={() => {
                onSelect(
                  String(it.item_code || '').toUpperCase(),
                  String(it.item_name || '').trim(),
                  String(it.uom || 'MTR').toUpperCase() || 'MTR',
                )
                setQ('')
              }}
              className="w-full text-left px-2 py-1.5 hover:bg-blue-50 transition-colors"
            >
              <div className="text-[11px] font-mono font-medium text-[#002B5B]">{it.item_code}</div>
              <div className="text-[10px] text-gray-500">
                {it.item_name}
                {it.item_type_code ? ` · ${it.item_type_code}` : ''}
              </div>
            </button>
          ))}
        </div>
      )}
      {q.trim().length >= 2 && results.length === 0 && (
        <div className="absolute z-20 top-full left-0 right-0 mt-0.5 border border-gray-200 rounded-lg bg-white shadow-lg px-2 py-1.5 text-[10px] text-gray-400">
          No materials found — try another code or name
        </div>
      )}
    </div>
  )
}

type SetBomLine = {
  component_code: string
  component_name: string
  qty_per_set: number
  default_next_process: string
  routing: string
  requires_embroidery: boolean
  component_role: 'SET_COMPONENT' | 'PANEL'
  parent_component_code: string
  materials: SetBomMaterial[]
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
    stitching_requires_complete_set: true,
    bundle_gate_process: 'Cutting',
    lines: [
      { component_code: 'TOP', component_name: 'Top', qty_per_set: 1, default_next_process: 'Stitching', routing: 'Cutting>Stitching', requires_embroidery: false, component_role: 'SET_COMPONENT', parent_component_code: '', materials: [] },
      { component_code: 'PANT', component_name: 'Pant', qty_per_set: 1, default_next_process: 'Stitching', routing: 'Cutting>Stitching', requires_embroidery: false, component_role: 'SET_COMPONENT', parent_component_code: '', materials: [] },
      { component_code: 'DUPATTA', component_name: 'Dupatta', qty_per_set: 1, default_next_process: 'Embroidery', routing: 'Cutting>Embroidery>Cutting>Stitching', requires_embroidery: true, component_role: 'SET_COMPONENT', parent_component_code: '', materials: [] },
    ] as SetBomLine[],
  })
  const [setMatchForm, setSetMatchForm] = useState({
    so_number: '',
    main_sku: '',
    from_process: 'Finishing',
    match_qty: 0,
  })
  const [setMatchPreview, setSetMatchPreview] = useState<any>(null)
  const [wipBoard, setWipBoard] = useState<any>(null)

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
          stitching_requires_complete_set: bom.stitching_requires_complete_set !== 0 && bom.stitching_requires_complete_set !== false,
          bundle_gate_process: bom.bundle_gate_process || 'Cutting',
          lines: bom.lines.map((l: any) => ({
            component_code: l.component_code,
            component_name: l.component_name || l.component_code,
            qty_per_set: l.qty_per_set || 1,
            default_next_process: l.default_next_process || '',
            routing: l.routing || '',
            requires_embroidery: Boolean(l.requires_embroidery) || String(l.routing || '').includes('Embroidery'),
            component_role: (String(l.component_role || '').toUpperCase() === 'PANEL' ? 'PANEL' : 'SET_COMPONENT') as 'SET_COMPONENT' | 'PANEL',
            parent_component_code: l.parent_component_code || '',
            materials: (l.materials || []).map((m: any) => ({
              material_code: m.material_code || '',
              material_name: m.material_name || '',
              quantity: Number(m.quantity) || 0,
              unit: m.unit || 'MTR',
            })),
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
        <h3 className="font-semibold text-gray-800">Component BOM (Set recipe + materials)</h3>
        <p className="text-xs text-gray-500">
          <strong>Set Components</strong> (Top / Bottom / Dupatta) each get one Cutting Job Order.
          <strong> Panels</strong> (Front / Back) belong under a parent (usually Top) for embroidery WIP —
          they do <em>not</em> create separate Cutting JOs.
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
          <div className="col-span-2 flex flex-wrap gap-3 items-center text-xs text-gray-700">
            <label className="inline-flex items-center gap-1.5">
              <input
                type="checkbox"
                checked={setBomForm.stitching_requires_complete_set}
                onChange={e => setSetBomForm(f => ({ ...f, stitching_requires_complete_set: e.target.checked }))}
              />
              Stitching requires complete bundle
            </label>
            <label className="inline-flex items-center gap-1.5">
              Bundle gate
              <select
                value={setBomForm.bundle_gate_process}
                onChange={e => setSetBomForm(f => ({ ...f, bundle_gate_process: e.target.value }))}
                className="border rounded px-1.5 py-0.5"
              >
                {processes.map(p => <option key={p}>{p}</option>)}
              </select>
            </label>
          </div>
        </div>
        <div className="border rounded-lg overflow-hidden">
          <table className="w-full text-xs">
            <thead className="bg-gray-50 text-gray-500">
              <tr>
                <th className="text-left px-2 py-1.5">Role</th>
                <th className="text-left px-2 py-1.5">Code</th>
                <th className="text-left px-2 py-1.5">Name</th>
                <th className="text-left px-2 py-1.5">Parent</th>
                <th className="text-right px-2 py-1.5">Qty/set</th>
                <th className="text-left px-2 py-1.5">Routing (partial WIP)</th>
                <th className="text-left px-2 py-1.5">Next</th>
                <th className="px-2 py-1.5" />
              </tr>
            </thead>
            <tbody>
              {setBomForm.lines.map((ln, i) => (
                <Fragment key={i}>
                <tr className="border-t">
                  <td className="px-2 py-1">
                    <select
                      value={ln.component_role || 'SET_COMPONENT'}
                      onChange={e => setSetBomForm(f => ({
                        ...f,
                        lines: f.lines.map((x, j) => j === i ? {
                          ...x,
                          component_role: e.target.value as 'SET_COMPONENT' | 'PANEL',
                          parent_component_code: e.target.value === 'PANEL' ? (x.parent_component_code || 'TOP') : '',
                        } : x),
                      }))}
                      className="border rounded px-1 py-0.5 text-[10px]"
                      title="Set Component = Cutting JO; Panel = Front/Back under parent"
                    >
                      <option value="SET_COMPONENT">Set Component</option>
                      <option value="PANEL">Panel</option>
                    </select>
                  </td>
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
                    {ln.component_role === 'PANEL' ? (
                      <select
                        value={ln.parent_component_code || 'TOP'}
                        onChange={e => setSetBomForm(f => ({
                          ...f,
                          lines: f.lines.map((x, j) => j === i ? { ...x, parent_component_code: e.target.value } : x),
                        }))}
                        className="w-full border rounded px-1 py-0.5 font-mono text-[10px]"
                      >
                        {setBomForm.lines
                          .filter(x => x.component_role !== 'PANEL' && x.component_code)
                          .map(x => (
                            <option key={x.component_code} value={x.component_code}>{x.component_code}</option>
                          ))}
                        {!setBomForm.lines.some(x => x.component_role !== 'PANEL' && x.component_code === 'TOP') && (
                          <option value="TOP">TOP</option>
                        )}
                      </select>
                    ) : (
                      <span className="text-gray-300 text-[10px]">—</span>
                    )}
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
                    <input
                      value={ln.routing || ''}
                      onChange={e => setSetBomForm(f => ({
                        ...f,
                        lines: f.lines.map((x, j) => j === i ? {
                          ...x,
                          routing: e.target.value,
                          requires_embroidery: e.target.value.includes('Embroidery'),
                        } : x),
                      }))}
                      placeholder="Cutting>Embroidery>Cutting>Stitching"
                      className="w-full border rounded px-1.5 py-0.5 font-mono text-[10px]"
                      title="Component-level route. Embroidery between Cutting hops = temporary child WIP."
                    />
                  </td>
                  <td className="px-2 py-1">
                    <select
                      value={ln.default_next_process}
                      onChange={e => setSetBomForm(f => ({
                        ...f,
                        lines: f.lines.map((x, j) => j === i ? {
                          ...x,
                          default_next_process: e.target.value,
                          routing: x.routing || (e.target.value === 'Embroidery'
                            ? 'Cutting>Embroidery>Cutting>Stitching'
                            : e.target.value
                              ? `Cutting>${e.target.value}`
                              : x.routing),
                          requires_embroidery: e.target.value === 'Embroidery' || (x.routing || '').includes('Embroidery'),
                        } : x),
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
                <tr className="border-t bg-gray-50/80">
                  <td colSpan={8} className="px-2 py-2">
                    <p className="text-[10px] font-semibold text-gray-500 mb-1">
                      {ln.component_role === 'PANEL' ? 'Panel' : 'Materials'} for {ln.component_code || 'component'}
                      {ln.component_role === 'PANEL' ? ' (managed inside parent Cutting JO — no separate JO)' : ''}
                    </p>
                    <div className="space-y-1">
                      {(ln.materials || []).map((mat, mi) => (
                        <div key={mi} className="flex gap-1 items-center flex-wrap">
                          <MaterialPicker
                            materialCode={mat.material_code}
                            materialName={mat.material_name}
                            onSelect={(code, name, unit) => setSetBomForm(f => ({
                              ...f,
                              lines: f.lines.map((x, j) => j === i ? {
                                ...x,
                                materials: (x.materials || []).map((m, k) => k === mi ? {
                                  ...m,
                                  material_code: code,
                                  material_name: name,
                                  unit: m.unit || unit,
                                } : m),
                              } : x),
                            }))}
                            onClear={() => setSetBomForm(f => ({
                              ...f,
                              lines: f.lines.map((x, j) => j === i ? {
                                ...x,
                                materials: (x.materials || []).map((m, k) => k === mi ? {
                                  ...m,
                                  material_code: '',
                                  material_name: '',
                                } : m),
                              } : x),
                            }))}
                          />
                          <input
                            type="number"
                            step="0.01"
                            value={mat.quantity}
                            onChange={e => setSetBomForm(f => ({
                              ...f,
                              lines: f.lines.map((x, j) => j === i ? {
                                ...x,
                                materials: (x.materials || []).map((m, k) => k === mi ? { ...m, quantity: +e.target.value } : m),
                              } : x),
                            }))}
                            placeholder="Qty/pc"
                            className="border rounded px-1.5 py-0.5 text-[11px] w-20 text-right"
                          />
                          <select
                            value={mat.unit}
                            onChange={e => setSetBomForm(f => ({
                              ...f,
                              lines: f.lines.map((x, j) => j === i ? {
                                ...x,
                                materials: (x.materials || []).map((m, k) => k === mi ? { ...m, unit: e.target.value } : m),
                              } : x),
                            }))}
                            className="border rounded px-1.5 py-0.5 text-[11px] w-16"
                          >
                            {['MTR', 'KG', 'PCS', 'LTR', 'ROLL', 'SET'].map(u => (
                              <option key={u} value={u}>{u}</option>
                            ))}
                          </select>
                          <button
                            type="button"
                            onClick={() => setSetBomForm(f => ({
                              ...f,
                              lines: f.lines.map((x, j) => j === i ? {
                                ...x,
                                materials: (x.materials || []).filter((_, k) => k !== mi),
                              } : x),
                            }))}
                            className="text-red-400 text-xs px-1"
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                      <button
                        type="button"
                        onClick={() => setSetBomForm(f => ({
                          ...f,
                          lines: f.lines.map((x, j) => j === i ? {
                            ...x,
                            materials: [...(x.materials || []), { material_code: '', material_name: '', quantity: 0, unit: 'MTR' }],
                          } : x),
                        }))}
                        className="text-[10px] text-[#002B5B] hover:underline"
                      >
                        + Material
                      </button>
                    </div>
                  </td>
                </tr>
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => setSetBomForm(f => ({
              ...f,
              lines: [...f.lines, {
                component_code: '',
                component_name: '',
                qty_per_set: 1,
                default_next_process: 'Stitching',
                routing: 'Cutting>Stitching',
                requires_embroidery: false,
                component_role: 'SET_COMPONENT' as const,
                parent_component_code: '',
                materials: [],
              }],
            }))}
            className="px-3 py-1.5 text-xs border rounded-lg"
          >
            + Set Component
          </button>
          <button
            type="button"
            onClick={() => setSetBomForm(f => ({
              ...f,
              lines: [...f.lines, {
                component_code: 'FRONT',
                component_name: 'Top Front',
                qty_per_set: 1,
                default_next_process: 'Embroidery',
                routing: 'Cutting>Embroidery>Cutting>Stitching',
                requires_embroidery: true,
                component_role: 'PANEL' as const,
                parent_component_code: f.lines.find(x => x.component_role !== 'PANEL' && /TOP/i.test(x.component_code))?.component_code || 'TOP',
                materials: [],
              }],
            }))}
            className="px-3 py-1.5 text-xs border border-amber-300 text-amber-900 bg-amber-50 rounded-lg"
          >
            + Panel (Front/Back)
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
                  stitching_requires_complete_set: b.stitching_requires_complete_set !== 0 && b.stitching_requires_complete_set !== false,
                  bundle_gate_process: b.bundle_gate_process || 'Cutting',
                  lines: (b.lines || []).map((l: any) => ({
                    component_code: l.component_code,
                    component_name: l.component_name || l.component_code,
                    qty_per_set: l.qty_per_set || 1,
                    default_next_process: l.default_next_process || '',
                    routing: l.routing || '',
                    requires_embroidery: Boolean(l.requires_embroidery),
                    component_role: (String(l.component_role || '').toUpperCase() === 'PANEL' ? 'PANEL' : 'SET_COMPONENT') as 'SET_COMPONENT' | 'PANEL',
                    parent_component_code: l.parent_component_code || '',
                    materials: (l.materials || []).map((m: any) => ({
                      material_code: m.material_code || '',
                      material_name: m.material_name || '',
                      quantity: Number(m.quantity) || 0,
                      unit: m.unit || 'MTR',
                    })),
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
          <h3 className="font-semibold text-gray-800">Partial WIP board (bundle gate)</h3>
          <p className="text-xs text-gray-500">
            Embroidery is a temporary child of Cutting. Stitching stays blocked until all panels return and the bundle is complete.
          </p>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-gray-500">SO number</label>
              <input
                value={setMatchForm.so_number}
                onChange={e => setSetMatchForm(f => ({ ...f, so_number: e.target.value }))}
                className="w-full border rounded px-2 py-1.5 text-sm mt-1 font-mono"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500">Main size SKU</label>
              <input
                value={setMatchForm.main_sku}
                onChange={e => setSetMatchForm(f => ({ ...f, main_sku: e.target.value.toUpperCase() }))}
                className="w-full border rounded px-2 py-1.5 text-sm mt-1 font-mono"
              />
            </div>
          </div>
          <button
            type="button"
            className="px-3 py-1.5 text-xs rounded bg-[#002B5B] text-white"
            onClick={async () => {
              try {
                const res = await api.get('/production/wip-board', {
                  params: { so_number: setMatchForm.so_number, main_sku: setMatchForm.main_sku },
                })
                setWipBoard(res.data)
              } catch (e) {
                alert(apiErrorMessage(e, 'WIP board failed'))
              }
            }}
          >
            Refresh WIP board
          </button>
          {wipBoard && (
            <div className="text-xs space-y-2">
              <p className={wipBoard.bundle_complete ? 'text-emerald-700 font-semibold' : 'text-amber-700 font-semibold'}>
                {wipBoard.message}
              </p>
              <table className="w-full border text-[11px]">
                <thead className="bg-gray-50 text-gray-500">
                  <tr>
                    <th className="text-left px-2 py-1">Item</th>
                    <th className="text-left px-2 py-1">Location</th>
                    <th className="text-left px-2 py-1">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(wipBoard.items || []).map((row: any) => (
                    <tr key={row.component_sku} className="border-t">
                      <td className="px-2 py-1 font-mono">{row.item}</td>
                      <td className="px-2 py-1">{row.current_location}</td>
                      <td className="px-2 py-1">{row.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <h3 className="font-semibold text-gray-800 pt-2">Set Match (Finishing → Packing)</h3>
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
