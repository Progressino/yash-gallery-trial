import { useMemo, useState } from 'react'
import api from '../api/client'

/** Reporting lines from the Hierarchy tab org chart. */
export type OrgNode = { name: string; children?: OrgNode[] }

export const ORG_CHART: OrgNode = {
  name: 'Director',
  children: [
    {
      name: 'Suresh Kundnani',
      children: [
        {
          name: 'Ajay Kumar Verma',
          children: [
            { name: 'Sunil Yogi' },
            { name: 'Farhan Khan' },
            { name: 'Deepak Kumhar' },
            { name: 'Ravi ji' },
            { name: 'Prahalad' },
          ],
        },
        {
          name: 'Irfan Ali',
          children: [
            { name: 'Sandeep Saini' },
            { name: 'Manisha Kanwar' },
            { name: 'Sadhana Verma' },
            { name: 'Vikash Yadav', children: [{ name: 'Priyanka Kumawat' }] },
          ],
        },
        {
          name: 'Chajuram master',
          children: [
            { name: 'Suresh Sharma' },
            { name: 'Bhimaram ji' },
            { name: 'Ramkaran ji' },
            { name: 'Rajkumar Sharma' },
            {
              name: 'Rohitash ji',
              children: [{ name: 'Mahtab' }, { name: 'Jitendra Yadav' }],
            },
            {
              name: 'Tayub Master',
              children: [{ name: 'Chain Management Staff (6)' }, { name: 'Chain Workers' }],
            },
            { name: 'Babulal ji' },
          ],
        },
        {
          name: 'Madan Singh',
          children: [{ name: 'Vishwajeet' }, { name: 'Bhanu ji/Hari ji' }],
        },
        {
          name: 'Rajni Devi',
          children: [{ name: 'Pinky' }, { name: 'All Finishing staff' }],
        },
        {
          name: 'Admin',
          children: [
            { name: 'Cleaning Staff' },
            { name: 'Maintenance Staff' },
            { name: 'Field Staff' },
            { name: 'Security' },
          ],
        },
      ],
    },
    {
      name: 'Yogesh Kundnani',
      children: [
        { name: 'Sanjay Sodani', children: [{ name: 'Rohit' }] },
        {
          name: 'Suraj Chaurasia',
          children: [
            { name: 'Bharti' },
            { name: 'Rimjhim' },
            { name: 'Dinesh Kumar' },
            { name: 'Sandeep kumar' },
            { name: 'Tannu' },
          ],
        },
        { name: 'Irfan Dewshwali' },
        { name: 'Harsh' },
        {
          name: 'Vijay Kumar Saini',
          children: [
            { name: 'Ashu Bhatt' },
            { name: 'Banwari' },
            { name: 'Guddi' },
            { name: 'HImanshu Saini' },
            { name: 'Sanjay Saini' },
            { name: 'Nemichand Saini' },
          ],
        },
        { name: 'Manish Saini', children: [{ name: 'Dispatch Staff' }] },
      ],
    },
  ],
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

function Box({
  node,
  people,
  depth,
  path,
  collapsed,
  toggle,
}: {
  node: OrgNode
  people: Map<string, any[]>
  depth: number
  path: string
  collapsed: Set<string>
  toggle: (path: string) => void
}) {
  const emp = matchOne(people, node.name)
  const tone = depth <= 1 ? 'bg-[#1f7a4d] text-white border-[#166534]' : emp ? 'bg-[#d9f2d9] border-[#3d8b40] text-gray-900' : 'bg-[#eef6ee] border-[#8fbc8f] text-gray-700'
  const hasKids = !!node.children?.length
  const isCollapsed = hasKids && collapsed.has(path)
  return (
    <li>
      <div className={`org-box min-w-[7.5rem] max-w-[9.5rem] rounded-md border px-2 py-1.5 text-center shadow-sm ${tone}`}>
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
          <div className="min-w-0">
            <p className="text-[11px] font-semibold leading-tight">{node.name}</p>
            {emp && <p className={`text-[9px] mt-0.5 ${depth <= 1 ? 'text-green-100' : 'text-gray-500'}`}>{emp.department_name || emp.designation || 'Employee'}</p>}
          </div>
        </div>
      </div>
      {hasKids && !isCollapsed && (
        <ul>
          {node.children!.map(child => (
            <Box
              key={child.name}
              node={child}
              people={people}
              depth={depth + 1}
              path={nodePath(path, child.name)}
              collapsed={collapsed}
              toggle={toggle}
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

export function OrgChart({ employees, canApply, onApplied }: { employees: any[]; canApply: boolean; onApplied: () => void }) {
  const people = indexEmployees(employees)
  const pairs = reportingPairs(ORG_CHART, null, [])
  const ready = pairs.filter(p => matchOne(people, p.person) && matchOne(people, p.manager))
  const allPaths = useMemo(() => collectExpandablePaths(ORG_CHART), [])
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set())
  const toggle = (path: string) => {
    setCollapsed(prev => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }
  async function apply() {
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
          <p className="text-xs text-gray-500">Director at the top. A darker box is a manager. A name with a department is an employee record. Use + / − to expand or collapse a branch.</p>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          <button type="button" onClick={() => setCollapsed(new Set())} className="px-2.5 py-1 rounded-lg border text-xs text-gray-700 hover:bg-gray-50">
            Expand tree
          </button>
          <button type="button" onClick={() => setCollapsed(new Set(allPaths))} className="px-2.5 py-1 rounded-lg border text-xs text-gray-700 hover:bg-gray-50">
            Collapse tree
          </button>
          {canApply && (
            <button type="button" onClick={apply} disabled={!ready.length} className="px-3 py-1.5 rounded-lg bg-[#1f7a4d] text-white text-xs disabled:opacity-40">
              Save reporting lines ({ready.length})
            </button>
          )}
        </div>
      </div>
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
          <Box node={ORG_CHART} people={people} depth={0} path={ORG_CHART.name} collapsed={collapsed} toggle={toggle} />
        </ul>
      </div>
    </div>
  )
}
