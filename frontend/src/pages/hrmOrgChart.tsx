import { useEffect, useMemo, useState } from 'react'
import api from '../api/client'

/** Reporting lines from the Hierarchy tab org chart. */
export type OrgNode = { name: string; children?: OrgNode[] }

/** Fallback seed if the API has not returned a chart yet. */
export const ORG_CHART: OrgNode = {
  name: 'Director',
  children: [],
}

function cloneTree(node: OrgNode): OrgNode {
  return {
    name: node.name,
    children: (node.children || []).map(cloneTree),
  }
}

function norm(name: string) {
  return name.toLowerCase().replace(/\s+/g, ' ').trim()
}

function lookupKeys(name: string) {
  const n = norm(name)
  const keys = [n]
  const stripped = n.replace(/ (ji|master)$/, '')
  if (stripped !== n) keys.push(stripped)
  if (n === 'dipatch staff') keys.push('dispatch staff')
  if (n === 'dispatch staff') keys.push('dipatch staff')
  return keys
}

function indexEmployees(employees: any[]) {
  const map = new Map<string, any[]>()
  for (const e of employees) {
    for (const key of lookupKeys(String(e.name || ''))) {
      const list = map.get(key) || []
      if (!list.some(x => x.id === e.id)) list.push(e)
      map.set(key, list)
    }
  }
  return map
}

function matchOne(map: Map<string, any[]>, name: string) {
  for (const key of lookupKeys(name)) {
    const hits = map.get(key) || []
    if (hits.length === 1) return hits[0]
  }
  return null
}

function nodePath(parentPath: string, name: string) {
  return parentPath ? `${parentPath}/${name}` : name
}

function collectExpandablePaths(node: OrgNode, parentPath = '', out: string[] = []) {
  const path = nodePath(parentPath, node.name)
  if (node.children?.length) {
    out.push(path)
    for (const child of node.children) collectExpandablePaths(child, path, out)
  }
  return out
}

function findNode(root: OrgNode, path: string): OrgNode | null {
  if (path === root.name) return root
  if (!path.startsWith(`${root.name}/`)) return null
  const segs = path.slice(root.name.length + 1).split('/').filter(Boolean)
  let cur: OrgNode = root
  for (const seg of segs) {
    const child: OrgNode | undefined = (cur.children || []).find(c => c.name === seg)
    if (!child) return null
    cur = child
  }
  return cur
}

function updateAtPath(root: OrgNode, path: string, mutator: (node: OrgNode) => OrgNode | null): OrgNode {
  const next = cloneTree(root)
  if (path === next.name) {
    const replaced = mutator(next)
    if (!replaced) throw new Error('Cannot remove the root box')
    return replaced
  }
  const segs = path.slice(next.name.length + 1).split('/').filter(Boolean)
  let parent: OrgNode = next
  for (let i = 0; i < segs.length - 1; i++) {
    const child = (parent.children || []).find(c => c.name === segs[i])
    if (!child) return next
    parent = child
  }
  const leafName = segs[segs.length - 1]
  const idx = (parent.children || []).findIndex(c => c.name === leafName)
  if (idx < 0) return next
  const kids = [...(parent.children || [])]
  const replaced = mutator(kids[idx])
  if (!replaced) kids.splice(idx, 1)
  else kids[idx] = replaced
  parent.children = kids
  return next
}

function Box({
  node,
  people,
  depth,
  path,
  collapsed,
  toggle,
  canEdit,
  onRename,
  onAddChild,
  onRemove,
}: {
  node: OrgNode
  people: Map<string, any[]>
  depth: number
  path: string
  collapsed: Set<string>
  toggle: (path: string) => void
  canEdit: boolean
  onRename: (path: string, name: string) => void
  onAddChild: (path: string) => void
  onRemove: (path: string) => void
}) {
  const emp = matchOne(people, node.name)
  const tone = depth <= 1 ? 'bg-[#1f7a4d] text-white border-[#166534]' : emp ? 'bg-[#d9f2d9] border-[#3d8b40] text-gray-900' : 'bg-[#eef6ee] border-[#8fbc8f] text-gray-700'
  const hasKids = !!node.children?.length
  const isCollapsed = hasKids && collapsed.has(path)
  const isRoot = depth === 0
  return (
    <li>
      <div className={`org-box min-w-[8rem] max-w-[11rem] rounded-md border px-2 py-1.5 text-center shadow-sm ${tone}`}>
        <div className="flex items-start justify-center gap-1">
          {hasKids && (
            <button
              type="button"
              title={isCollapsed ? 'Expand' : 'Collapse'}
              onClick={() => toggle(path)}
              className={`mt-0.5 shrink-0 w-4 h-4 rounded text-[10px] leading-none border ${
                depth <= 1 ? 'border-green-200 text-green-50 hover:bg-green-600' : 'border-green-700/40 text-green-900 hover:bg-green-100'
              }`}
            >
              {isCollapsed ? '+' : '−'}
            </button>
          )}
          <div className="min-w-0 flex-1">
            {canEdit ? (
              <input
                value={node.name}
                onChange={e => onRename(path, e.target.value)}
                className={`w-full text-[11px] font-semibold leading-tight text-center bg-transparent border-b border-transparent focus:border-current outline-none ${
                  depth <= 1 ? 'text-white placeholder:text-green-100' : 'text-gray-900'
                }`}
                title="Edit box name"
              />
            ) : (
              <p className="text-[11px] font-semibold leading-tight">{node.name}</p>
            )}
            {emp && <p className={`text-[9px] mt-0.5 ${depth <= 1 ? 'text-green-100' : 'text-gray-500'}`}>{emp.department_name || emp.designation || 'Employee'}</p>}
            {canEdit && (
              <div className="flex justify-center gap-1 mt-1">
                <button type="button" onClick={() => onAddChild(path)} className={`text-[9px] px-1 rounded ${depth <= 1 ? 'bg-green-600/80 text-white' : 'bg-white/80 text-green-900 border'}`} title="Add child">
                  + Child
                </button>
                {!isRoot && (
                  <button type="button" onClick={() => onRemove(path)} className={`text-[9px] px-1 rounded ${depth <= 1 ? 'bg-red-500/80 text-white' : 'bg-white/80 text-red-700 border'}`} title="Remove box">
                    Remove
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      {hasKids && !isCollapsed && (
        <ul>
          {node.children!.map(child => (
            <Box
              key={nodePath(path, child.name)}
              node={child}
              people={people}
              depth={depth + 1}
              path={nodePath(path, child.name)}
              collapsed={collapsed}
              toggle={toggle}
              canEdit={canEdit}
              onRename={onRename}
              onAddChild={onAddChild}
              onRemove={onRemove}
            />
          ))}
        </ul>
      )}
    </li>
  )
}

function reportingPairs(node: OrgNode, parent: string | null, out: { person: string; manager: string }[]) {
  if (parent) out.push({ person: node.name, manager: parent })
  for (const child of node.children || []) reportingPairs(child, node.name, out)
  return out
}

export function OrgChart({
  employees,
  canApply,
  initialChart,
  onApplied,
  onChartSaved,
}: {
  employees: any[]
  canApply: boolean
  initialChart?: OrgNode | null
  onApplied: () => void
  onChartSaved?: () => void
}) {
  const people = indexEmployees(employees)
  const [tree, setTree] = useState<OrgNode>(() => cloneTree(initialChart || ORG_CHART))
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')

  useEffect(() => {
    if (initialChart?.name && !dirty) {
      setTree(cloneTree(initialChart))
    }
  }, [initialChart, dirty])

  const pairs = reportingPairs(tree, null, [])
  const ready = pairs.filter(p => matchOne(people, p.person) && matchOne(people, p.manager))
  const allPaths = useMemo(() => collectExpandablePaths(tree), [tree])
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set())
  const toggle = (path: string) => {
    setCollapsed(prev => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  const markDirty = (next: OrgNode) => {
    setTree(next)
    setDirty(true)
    setMsg('')
  }

  const onRename = (path: string, name: string) => {
    markDirty(updateAtPath(tree, path, n => ({ ...n, name })))
  }
  const onAddChild = (path: string) => {
    const label = window.prompt('Name for the new box')
    if (!label?.trim()) return
    markDirty(updateAtPath(tree, path, n => ({
      ...n,
      children: [...(n.children || []), { name: label.trim() }],
    })))
  }
  const onRemove = (path: string) => {
    if (path === tree.name) return
    if (!window.confirm(`Remove “${findNode(tree, path)?.name || path}” and everyone under it from the chart?`)) return
    markDirty(updateAtPath(tree, path, () => null))
  }

  async function saveChart() {
    setSaving(true)
    setMsg('')
    try {
      await api.put('/hrm/hierarchy/org-chart', tree)
      setDirty(false)
      setMsg('Chart saved.')
      onChartSaved?.()
    } catch (e: any) {
      setMsg(e?.response?.data?.detail || 'Could not save chart')
    } finally {
      setSaving(false)
    }
  }

  async function apply() {
    if (dirty) {
      alert('Save the chart first, then apply reporting lines.')
      return
    }
    if (!window.confirm(`Save ${ready.length} reporting lines from this chart onto the matching employee records?`)) return
    for (const pair of ready) {
      const person = matchOne(people, pair.person)
      const manager = matchOne(people, pair.manager)
      if (!person || !manager || Number(person.reports_to_employee_id) === Number(manager.id)) continue
      await api.patch('/hrm/hierarchy/reporting', { employee_id: person.id, reports_to_employee_id: manager.id })
    }
    onApplied()
  }

  return (
    <div className="bg-white rounded-xl border p-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="font-semibold text-gray-800">Organization chart</h3>
          <p className="text-xs text-gray-500">
            {canApply
              ? 'Admin can rename boxes, add a child, or remove a box, then Save chart. Save reporting lines copies chart managers onto employee records. HOD sees the full company chart (view only).'
              : 'Full company chart (view only). Use + / − to expand or collapse a branch.'}
          </p>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          <button type="button" onClick={() => setCollapsed(new Set())} className="px-2.5 py-1 rounded-lg border text-xs text-gray-700 hover:bg-gray-50">
            Expand tree
          </button>
          <button type="button" onClick={() => setCollapsed(new Set(allPaths))} className="px-2.5 py-1 rounded-lg border text-xs text-gray-700 hover:bg-gray-50">
            Collapse tree
          </button>
          {canApply && (
            <>
              <button
                type="button"
                onClick={saveChart}
                disabled={!dirty || saving}
                className="px-3 py-1.5 rounded-lg bg-[#002B5B] text-white text-xs disabled:opacity-40"
              >
                {saving ? 'Saving…' : dirty ? 'Save chart' : 'Chart saved'}
              </button>
              <button type="button" onClick={apply} disabled={!ready.length || dirty} className="px-3 py-1.5 rounded-lg bg-[#1f7a4d] text-white text-xs disabled:opacity-40">
                Save reporting lines ({ready.length})
              </button>
            </>
          )}
        </div>
      </div>
      {msg && <p className="text-xs text-gray-600">{msg}</p>}
      <style>{`
        .org-chart ul { display: flex; justify-content: center; padding-top: 16px; position: relative; }
        .org-chart > ul { padding-top: 0; }
        .org-chart li { display: flex; flex-direction: column; align-items: center; position: relative; padding: 16px 4px 0; }
        .org-chart > ul > li { padding-top: 0; }
        .org-chart li::before, .org-chart li::after {
          content: ''; position: absolute; top: 0; width: 50%; height: 16px; border-top: 1px solid #15803d;
        }
        .org-chart li::before { right: 50%; }
        .org-chart li::after { left: 50%; border-left: 1px solid #15803d; }
        .org-chart > ul > li::before, .org-chart > ul > li::after { display: none; }
        .org-chart li:first-child::before, .org-chart li:last-child::after { border: 0; }
        .org-chart li:last-child::before { border-right: 1px solid #15803d; }
        .org-chart li:only-child::before { border: 0; }
        .org-chart ul::before {
          content: ''; position: absolute; top: 0; left: 50%; border-left: 1px solid #15803d; height: 16px;
        }
        .org-chart > ul::before { display: none; }
      `}</style>
      <div className="org-chart overflow-x-auto pb-2">
        <ul className="w-max min-w-full">
          <Box
            node={tree}
            people={people}
            depth={0}
            path={tree.name}
            collapsed={collapsed}
            toggle={toggle}
            canEdit={canApply}
            onRename={onRename}
            onAddChild={onAddChild}
            onRemove={onRemove}
          />
        </ul>
      </div>
    </div>
  )
}
