import { Link } from 'react-router-dom'
import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'
import api from '../api/client'
import { resetErpModuleData, clearPlatform } from '../api/client'
import { ALL_MODULE_KEYS, MODULE_LABELS } from '../lib/modules'
import { mayAccessErpAdmin, mayClearPlatformData, mayDeleteUploadData, mayManageErpDepartments, useAuth } from '../store/auth'

type Tab = 'dashboard' | 'users' | 'roles' | 'activity' | 'data-reset'

interface AdminStats { total_users: number; total_roles: number; recent_activity: number; by_role: { role_name: string; cnt: number }[] }
interface ERPUser {
  id: number
  username: string
  email: string | null
  phone?: string | null
  full_name: string
  role_id: number
  role_name: string
  department: string
  karigar_id?: string
  employee_id?: number | null
  hrm_department_id?: number | null
  reporting_hod_user_id?: number | null
  module_access?: string | null
  active: number
  created_at: string
}
interface Role { id: number; role_name: string; description: string; created_at: string }
interface ActivityLog { id: number; username: string; action: string; document_type: string; document_no: string; details: string; created_at: string }
interface ErpDepartment { id: number; name: string; active?: number }

const FALLBACK_DEPARTMENTS = ['Sales', 'Merchandising', 'Stores', 'Production', 'Quality', 'Logistics', 'Finance', 'Admin']

const actionColor = (a: string) => {
  if (a === 'create') return 'bg-green-100 text-green-700'
  if (a === 'update') return 'bg-blue-100 text-blue-700'
  if (a === 'delete') return 'bg-red-100 text-red-700'
  if (a === 'approve') return 'bg-purple-100 text-purple-700'
  return 'bg-gray-100 text-gray-600'
}

function formatUserCreateError(err: unknown): string {
  if (!axios.isAxiosError(err)) return 'Request failed'
  const d = err.response?.data as { detail?: unknown } | undefined
  const detail = d?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail))
    return detail.map((x: { msg?: string }) => x?.msg).filter(Boolean).join('; ') || 'Request failed'
  return 'Request failed'
}

const HRM_ROLES = new Set(['HOD', 'Employee'])

const EMPTY_USER_FORM = {
  username: '', email: '', phone: '', password: 'changeme123', full_name: '', role_id: 1,
  department: 'Production', karigar_id: '',
  employee_id: '' as number | '', hrm_department_id: '' as number | '',
  reporting_hod_user_id: '' as number | '', module_access: '',
}

export default function Admin() {
  const authUser = useAuth(s => s.user)
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('dashboard')
  const [showUserForm, setShowUserForm] = useState(false)
  const [showRoleForm, setShowRoleForm] = useState(false)
  const [editUser, setEditUser] = useState<ERPUser | null>(null)
  const [editData, setEditData] = useState<Record<string, string | number | null>>({})
  const [resetBusyModule, setResetBusyModule] = useState<string | null>(null)
  const [clearInventoryBusy, setClearInventoryBusy] = useState(false)
  const [clearInventoryMsg, setClearInventoryMsg] = useState<string | null>(null)
  const mayClearPlatform = mayClearPlatformData(authUser)
  const mayDeleteData = mayDeleteUploadData(authUser)

  const payloadFromUserForm = (form: typeof EMPTY_USER_FORM, modules: string[]) => {
    const body: Record<string, unknown> = {
      username: form.username,
      email: form.email,
      password: form.password,
      full_name: form.full_name,
      role_id: form.role_id,
      department: form.department,
      karigar_id: form.karigar_id,
    }
    if (form.phone.trim()) body.phone = form.phone.trim()
    if (form.employee_id !== '') body.employee_id = Number(form.employee_id)
    if (form.hrm_department_id !== '') body.hrm_department_id = Number(form.hrm_department_id)
    if (form.reporting_hod_user_id !== '') body.reporting_hod_user_id = Number(form.reporting_hod_user_id)
    if (modules.length) body.module_access = JSON.stringify(modules)
    else if (form.module_access) body.module_access = form.module_access
    return body
  }

  const [userForm, setUserForm] = useState({ ...EMPTY_USER_FORM })
  const [modulePick, setModulePick] = useState<string[]>([])
  const [roleForm, setRoleForm] = useState({ role_name: '', description: '' })
  const [includeInactiveUsers, setIncludeInactiveUsers] = useState(false)
  const [newDepartmentName, setNewDepartmentName] = useState('')
  const mayManageDepartments = mayManageErpDepartments(authUser)

  const { data: stats } = useQuery<AdminStats>({ queryKey: ['admin-stats'], queryFn: () => api.get('/erp-admin/stats').then(r => r.data) })
  const { data: users = [] } = useQuery<ERPUser[]>({
    queryKey: ['erp-users', includeInactiveUsers],
    queryFn: () =>
      api
        .get('/erp-admin/users', { params: { include_inactive: includeInactiveUsers } })
        .then(r => r.data),
    enabled: tab === 'users' || tab === 'dashboard',
  })
  const { data: roles = [] } = useQuery<Role[]>({ queryKey: ['erp-roles'], queryFn: () => api.get('/erp-admin/roles').then(r => r.data) })
  const { data: erpDepartments = [] } = useQuery<ErpDepartment[]>({
    queryKey: ['erp-departments'],
    queryFn: () => api.get('/erp-admin/departments').then(r => r.data),
    enabled: tab === 'users' && !!mayAccessErpAdmin(authUser),
  })
  const departmentNames = useMemo(() => {
    const fromApi = erpDepartments.map(d => d.name).filter(Boolean)
    return fromApi.length ? fromApi : FALLBACK_DEPARTMENTS
  }, [erpDepartments])
  const { data: hrmDepts = [] } = useQuery({
    queryKey: ['hrm-depts-admin'],
    queryFn: () => api.get('/hrm/departments').then(r => r.data),
    enabled: tab === 'users' && !!mayAccessErpAdmin(authUser),
  })
  const { data: hrmEmployees = [] } = useQuery({
    queryKey: ['hrm-emps-admin'],
    queryFn: () => api.get('/hrm/employees').then(r => r.data),
    enabled: tab === 'users' && !!mayAccessErpAdmin(authUser),
  })
  const { data: activity = [] } = useQuery<ActivityLog[]>({ queryKey: ['activity-log'], queryFn: () => api.get('/erp-admin/activity?limit=100').then(r => r.data), enabled: tab === 'activity' })

  const createUserMut = useMutation({
    mutationFn: (b: object) => api.post('/erp-admin/users', b),
      onSuccess: () => {
        qc.invalidateQueries({ queryKey: ['erp-users'] })
        qc.invalidateQueries({ queryKey: ['admin-stats'] })
        setShowUserForm(false)
        setModulePick([])
        setUserForm({ ...EMPTY_USER_FORM })
      }
  })
  const updateUserMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: object }) => api.patch(`/erp-admin/users/${id}`, data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['erp-users'] }); setEditUser(null) }
  })
  const deactivateUserMut = useMutation({
    mutationFn: (id: number) => api.delete(`/erp-admin/users/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['erp-users'] })
  })
  const createRoleMut = useMutation({
    mutationFn: (b: object) => api.post('/erp-admin/roles', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['erp-roles'] }); qc.invalidateQueries({ queryKey: ['admin-stats'] }); setShowRoleForm(false); setRoleForm({ role_name: '', description: '' }) }
  })
  const createDepartmentMut = useMutation({
    mutationFn: (name: string) => api.post('/erp-admin/departments', { name }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['erp-departments'] })
      setNewDepartmentName('')
    },
  })
  const resetModuleMut = useMutation({
    mutationFn: (module: string) => resetErpModuleData(module),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['admin-stats'] })
    },
  })

  const TABS: [Tab, string][] = [['dashboard', '📊 Dashboard'], ['users', '👤 Users'], ['roles', '🔑 Roles'], ['activity', '📜 Activity Log'], ['data-reset', '🧹 Data Reset']]
  const RESET_MODULES: Array<{ key: string; label: string; detail: string }> = [
    { key: 'sales_orders', label: 'Sales Orders', detail: 'Sales orders + demand rows' },
    { key: 'item_master', label: 'Item Master', detail: 'Items, BOMs, buyers, merchants' },
    { key: 'purchase', label: 'Purchase', detail: 'PR/PO/JWO/GRN + suppliers/processors' },
    { key: 'tna', label: 'TNA', detail: 'TNA list + activity lines' },
    { key: 'production', label: 'Production', detail: 'JOs, process stock, material requirement planning' },
    { key: 'grey_fabric', label: 'Grey Fabric', detail: 'Tracker, ledger, checked/printed stock' },
  ]

  const newUserRole = roles.find(r => r.id === userForm.role_id)?.role_name
  const editRoleId = editData.role_id !== undefined ? Number(editData.role_id) : editUser?.role_id
  const editUserRole = roles.find(r => r.id === editRoleId)?.role_name

  const openNewUserForm = () => {
    const karigarRole = roles.find(r => r.role_name === 'Karigar')
    setUserForm({ ...EMPTY_USER_FORM, role_id: karigarRole?.id ?? 1 })
    setShowUserForm(true)
  }

  const runModuleReset = (module: string, label: string) => {
    const ok = window.confirm(
      `Remove testing data from ${label}?\n\nThis deletes records from production DB and cannot be undone.`,
    )
    if (!ok) return
    setResetBusyModule(module)
    resetModuleMut.mutate(module, {
      onSettled: () => setResetBusyModule(null),
    })
  }

  const runClearInventorySnapshot = async () => {
    const ok = window.confirm(
      'Remove the current snapshot inventory from the server?\n\n'
        + 'Use this when the wrong RAR/CSV was uploaded (e.g. inflated totals). '
        + 'Re-upload the correct file under Upload Data → Daily uploads → Snapshot inventory.',
    )
    if (!ok) return
    setClearInventoryBusy(true)
    setClearInventoryMsg(null)
    try {
      const res = await clearPlatform('inventory')
      setClearInventoryMsg(res.message)
      qc.invalidateQueries({ queryKey: ['coverage'] })
      qc.invalidateQueries({ queryKey: ['inventory'] })
    } catch (e: unknown) {
      setClearInventoryMsg(formatUserCreateError(e))
    } finally {
      setClearInventoryBusy(false)
    }
  }

  if (!mayAccessErpAdmin(authUser)) {
    return (
      <div className="p-8 text-center text-gray-600">
        <p className="font-medium">Admin access required</p>
        <p className="text-sm mt-2">Your role cannot manage ERP users.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-gray-800">Admin</h1>
        <p className="text-sm text-gray-500">Users, roles (Admin / Sir / HOD / Employee), HRM links, and module access</p>
      </div>

      <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit">
        {TABS.map(([key, label]) => (
          <button key={key} onClick={() => setTab(key)}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${tab === key ? 'bg-white text-[#002B5B] shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* Dashboard */}
      {tab === 'dashboard' && stats && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {[
              { label: 'ACTIVE USERS', value: stats.total_users, color: 'text-blue-600' },
              { label: 'ROLES', value: stats.total_roles, color: 'text-gray-700' },
              { label: "ACTIVITY (24H)", value: stats.recent_activity, color: 'text-green-600' },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
                <p className={`text-2xl font-bold ${color}`}>{value}</p>
                <p className="text-xs text-gray-500 mt-1 font-semibold tracking-wide">{label}</p>
              </div>
            ))}
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white rounded-xl border p-4">
              <h3 className="font-semibold text-gray-700 mb-3 text-sm">Users by Role</h3>
              {stats.by_role.map(r => (
                <div key={r.role_name} className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
                  <span className="text-sm text-gray-600">{r.role_name}</span>
                  <span className="text-sm font-semibold text-gray-700">{r.cnt}</span>
                </div>
              ))}
            </div>
            <div className="bg-white rounded-xl border p-4">
              <h3 className="font-semibold text-gray-700 mb-3 text-sm">Recent Users</h3>
              {users.slice(0, 5).map(u => (
                <div key={u.id} className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
                  <div>
                    <p className="text-sm font-medium text-gray-700">{u.full_name || u.username}</p>
                    <p className="text-xs text-gray-400">{u.role_name} · {u.department}</p>
                  </div>
                  <span className={`w-2 h-2 rounded-full ${u.active ? 'bg-green-500' : 'bg-gray-300'}`} />
                </div>
              ))}
            </div>
          </div>
          <div className="flex gap-3 flex-wrap">
            <Link to="/admin/performance"
              className="px-4 py-2 bg-emerald-700 text-white rounded-lg text-sm font-medium hover:bg-emerald-800">
              Performance dashboard
            </Link>
            <button onClick={() => { setTab('users'); openNewUserForm() }}
              className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ Add User</button>
            <button onClick={() => setTab('activity')}
              className="px-4 py-2 border border-[#002B5B] text-[#002B5B] rounded-lg text-sm font-medium hover:bg-gray-50">View Activity Log</button>
          </div>
        </div>
      )}

      {/* Users */}
      {tab === 'users' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center flex-wrap gap-2">
            <div className="flex items-center gap-3">
              <p className="text-sm text-gray-500">{users.length} users</p>
              <label className="flex items-center gap-1.5 text-xs text-gray-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeInactiveUsers}
                  onChange={e => setIncludeInactiveUsers(e.target.checked)}
                  className="rounded border-gray-300"
                />
                Show inactive
              </label>
            </div>
            <button onClick={openNewUserForm} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ Add User</button>
          </div>

          <div className="bg-white rounded-xl border border-gray-100 p-4 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
              <h3 className="font-semibold text-gray-700 text-sm">User departments</h3>
              <p className="text-xs text-gray-400">{departmentNames.length} options in the Department dropdown</p>
            </div>
            <div className="flex flex-wrap gap-2 mb-3">
              {departmentNames.map(d => (
                <span key={d} className="text-xs px-2 py-1 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
                  {d}
                </span>
              ))}
            </div>
            {mayManageDepartments ? (
              <div className="flex flex-wrap items-end gap-2">
                <div className="min-w-[12rem] flex-1">
                  <label className="text-xs text-gray-500">Add department</label>
                  <input
                    value={newDepartmentName}
                    onChange={e => setNewDepartmentName(e.target.value)}
                    placeholder="e.g. E-commerce"
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1"
                  />
                </div>
                <button
                  type="button"
                  onClick={() => {
                    const name = newDepartmentName.trim()
                    if (name) createDepartmentMut.mutate(name)
                  }}
                  disabled={createDepartmentMut.isPending || !newDepartmentName.trim()}
                  className="px-4 py-2 bg-emerald-700 text-white rounded-lg text-sm font-medium disabled:opacity-50"
                >
                  {createDepartmentMut.isPending ? 'Adding…' : 'Add department'}
                </button>
              </div>
            ) : (
              <p className="text-xs text-gray-400">Ask an Admin or Manager to add new departments.</p>
            )}
            {createDepartmentMut.isError && (
              <p className="text-xs text-red-500 mt-2">Error: {formatUserCreateError(createDepartmentMut.error)}</p>
            )}
          </div>

          {showUserForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New User</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {[['username','Username *'],['email','Email'],['phone','Mobile (+91)'],['password','Password'],['full_name','Full Name']].map(([k,l]) => (
                  <div key={k}><label className="text-xs text-gray-500">{l}</label>
                    <input value={(userForm as Record<string,string|number>)[k] as string}
                      type={k === 'password' ? 'password' : 'text'}
                      onChange={e => setUserForm(f => ({ ...f, [k]: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                  </div>
                ))}
                <div><label className="text-xs text-gray-500">Role</label>
                  <select value={userForm.role_id} onChange={e => setUserForm(f => ({ ...f, role_id: +e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {roles.map(r => <option key={r.id} value={r.id}>{r.role_name}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">Department</label>
                  <select value={userForm.department} onChange={e => setUserForm(f => ({ ...f, department: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {departmentNames.map(d => <option key={d}>{d}</option>)}
                  </select>
                </div>
                {newUserRole === 'Karigar' && (
                  <div><label className="text-xs text-gray-500">Karigar ID (master)</label>
                    <input value={userForm.karigar_id} placeholder="e.g. K001"
                      onChange={e => setUserForm(f => ({ ...f, karigar_id: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1 font-mono" />
                    <p className="text-[10px] text-gray-400 mt-0.5">Must match karigar_master (e.g. K001)</p>
                  </div>
                )}
                {newUserRole && HRM_ROLES.has(newUserRole) && (
                  <>
                    <div><label className="text-xs text-gray-500">HRM department</label>
                      <select value={userForm.hrm_department_id} onChange={e => setUserForm(f => ({ ...f, hrm_department_id: e.target.value ? +e.target.value : '' }))}
                        className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                        <option value="">—</option>
                        {(hrmDepts as { id: number; name: string }[]).map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                      </select></div>
                    <div><label className="text-xs text-gray-500">Linked HRM employee</label>
                      <select value={userForm.employee_id} onChange={e => setUserForm(f => ({ ...f, employee_id: e.target.value ? +e.target.value : '' }))}
                        className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                        <option value="">—</option>
                        {(hrmEmployees as { id: number; name: string; emp_code: string }[]).map(e => (
                          <option key={e.id} value={e.id}>{e.name} ({e.emp_code})</option>
                        ))}
                      </select></div>
                    <div><label className="text-xs text-gray-500">Reporting HOD (user)</label>
                      <select value={userForm.reporting_hod_user_id} onChange={e => setUserForm(f => ({ ...f, reporting_hod_user_id: e.target.value ? +e.target.value : '' }))}
                        className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                        <option value="">—</option>
                        {users.filter(u => u.role_name === 'HOD' && u.active).map(u => (
                          <option key={u.id} value={u.id}>{u.full_name || u.username}</option>
                        ))}
                      </select></div>
                  </>
                )}
                <div className="col-span-2 md:col-span-3">
                  <label className="text-xs text-gray-500">Module access override (optional)</label>
                  <p className="text-[10px] text-gray-400 mb-1">Leave empty for role default. HOD/Employee → HRM only; Admin/Sir → full ERP.</p>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {ALL_MODULE_KEYS.map(m => (
                      <label key={m} className="flex items-center gap-1 text-xs border rounded px-2 py-1 cursor-pointer">
                        <input type="checkbox" checked={modulePick.includes(m)}
                          onChange={e => setModulePick(prev => e.target.checked ? [...prev, m] : prev.filter(x => x !== m))} />
                        {MODULE_LABELS[m]}
                      </label>
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createUserMut.mutate(payloadFromUserForm(userForm, modulePick))} disabled={createUserMut.isPending || !userForm.username}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createUserMut.isPending ? 'Saving…' : 'Create User'}
                </button>
                <button onClick={() => setShowUserForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
              {createUserMut.isError && (
                <p className="text-xs text-red-500">Error: {formatUserCreateError(createUserMut.error)}</p>
              )}
            </div>
          )}

          {/* Edit User Panel */}
          {editUser && (
            <div className="bg-blue-50 rounded-xl border border-blue-200 p-4 space-y-3">
              <div className="flex justify-between">
                <h3 className="font-semibold text-gray-700">Edit: {editUser.username}</h3>
                <button onClick={() => setEditUser(null)} className="text-gray-400 hover:text-gray-600">✕</button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div><label className="text-xs text-gray-500">Full Name</label>
                  <input value={editData.full_name as string ?? editUser.full_name}
                    onChange={e => setEditData(d => ({ ...d, full_name: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">Email</label>
                  <input value={editData.email as string ?? editUser.email}
                    onChange={e => setEditData(d => ({ ...d, email: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">Role</label>
                  <select value={editData.role_id as number ?? editUser.role_id}
                    onChange={e => setEditData(d => ({ ...d, role_id: +e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {roles.map(r => <option key={r.id} value={r.id}>{r.role_name}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">Department</label>
                  <select value={editData.department as string ?? editUser.department}
                    onChange={e => setEditData(d => ({ ...d, department: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1">
                    {departmentNames.map(d => <option key={d}>{d}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">New Password (optional)</label>
                  <input type="password" placeholder="Leave blank to keep" value={editData.password as string ?? ''}
                    onChange={e => setEditData(d => ({ ...d, password: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                {editUserRole === 'Karigar' && (
                  <div><label className="text-xs text-gray-500">Karigar ID (master)</label>
                    <input value={(editData.karigar_id as string) ?? editUser.karigar_id ?? ''} placeholder="e.g. K001"
                      onChange={e => setEditData(d => ({ ...d, karigar_id: e.target.value }))}
                      className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1 font-mono" />
                  </div>
                )}
                {editUserRole && HRM_ROLES.has(editUserRole) && (
                  <>
                    <div><label className="text-xs text-gray-500">HRM department ID</label>
                      <input type="number" value={(editData.hrm_department_id as number) ?? editUser.hrm_department_id ?? ''}
                        onChange={e => setEditData(d => ({ ...d, hrm_department_id: e.target.value ? +e.target.value : null }))}
                        className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                    <div><label className="text-xs text-gray-500">HRM employee ID</label>
                      <input type="number" value={(editData.employee_id as number) ?? editUser.employee_id ?? ''}
                        onChange={e => setEditData(d => ({ ...d, employee_id: e.target.value ? +e.target.value : null }))}
                        className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                    <div><label className="text-xs text-gray-500">Reporting HOD user ID</label>
                      <input type="number" value={(editData.reporting_hod_user_id as number) ?? editUser.reporting_hod_user_id ?? ''}
                        onChange={e => setEditData(d => ({ ...d, reporting_hod_user_id: e.target.value ? +e.target.value : null }))}
                        className="w-full border rounded px-2 py-1.5 text-sm mt-1" /></div>
                  </>
                )}
              </div>
              <div className="flex gap-2">
                <button onClick={() => updateUserMut.mutate({ id: editUser.id, data: editData })} disabled={updateUserMut.isPending}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {updateUserMut.isPending ? 'Saving…' : 'Save Changes'}
                </button>
                <button onClick={() => setEditUser(null)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}

          <div className="bg-white rounded-xl border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                <tr>{['Username','Full Name','Role','HRM Emp','HRM Dept','Department','Email','Status','Actions'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {users.map(u => (
                  <tr key={u.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">{u.username}</td>
                    <td className="px-4 py-2 text-gray-600">{u.full_name || '—'}</td>
                    <td className="px-4 py-2">
                      <span className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">{u.role_name}</span>
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-gray-600">{u.employee_id ?? '—'}</td>
                    <td className="px-4 py-2 text-gray-500">{u.hrm_department_id ?? '—'}</td>
                    <td className="px-4 py-2 text-gray-500">{u.department}</td>
                    <td className="px-4 py-2 text-gray-400">{u.email || '—'}</td>
                    <td className="px-4 py-2">
                      <span className={`w-2 h-2 rounded-full inline-block ${u.active ? 'bg-green-500' : 'bg-gray-300'}`} />
                      <span className={`ml-1.5 text-xs ${u.active ? 'text-green-700' : 'text-gray-400'}`}>{u.active ? 'Active' : 'Inactive'}</span>
                    </td>
                    <td className="px-4 py-2">
                      <div className="flex gap-2">
                        <button onClick={() => { setEditUser(u); setEditData({}) }} className="text-xs text-blue-500 hover:text-blue-700">Edit</button>
                        {u.active ? (
                          <button onClick={() => deactivateUserMut.mutate(u.id)} className="text-xs text-red-400 hover:text-red-600">Deactivate</button>
                        ) : (
                          <button
                            type="button"
                            onClick={() => updateUserMut.mutate({ id: u.id, data: { active: 1 } })}
                            className="text-xs text-green-600 hover:text-green-800"
                          >
                            Reactivate
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {users.length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No users yet</p>}
          </div>
        </div>
      )}

      {/* Roles */}
      {tab === 'roles' && (
        <div className="space-y-4">
          <div className="flex justify-between">
            <p className="text-sm text-gray-500">{roles.length} roles</p>
            <button onClick={() => setShowRoleForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ Add Role</button>
          </div>
          {showRoleForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New Role</h3>
              <div className="grid grid-cols-2 gap-3">
                <div><label className="text-xs text-gray-500">Role Name *</label>
                  <input value={roleForm.role_name} onChange={e => setRoleForm(f => ({ ...f, role_name: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
                <div><label className="text-xs text-gray-500">Description</label>
                  <input value={roleForm.description} onChange={e => setRoleForm(f => ({ ...f, description: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createRoleMut.mutate(roleForm)} disabled={createRoleMut.isPending || !roleForm.role_name}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">Save</button>
                <button onClick={() => setShowRoleForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
            </div>
          )}
          <div className="bg-white rounded-xl border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-400 text-xs uppercase">
                <tr>{['Role','Description','Created'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {roles.map(r => (
                  <tr key={r.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">{r.role_name}</td>
                    <td className="px-4 py-2 text-gray-500">{r.description}</td>
                    <td className="px-4 py-2 text-gray-400 text-xs">{r.created_at?.split('T')[0]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Activity Log */}
      {tab === 'activity' && (
        <div className="bg-white rounded-xl border overflow-hidden">
          <div className="px-4 py-3 border-b bg-gray-50">
            <p className="text-sm font-semibold text-gray-700">Activity Log — Last 100 actions</p>
          </div>
          <table className="w-full text-sm">
            <thead className="text-gray-400 text-xs uppercase">
              <tr>{['Time','User','Action','Document','No.','Details'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
            </thead>
            <tbody>
              {activity.map(a => (
                <tr key={a.id} className="border-t border-gray-50 hover:bg-gray-50">
                  <td className="px-4 py-2 text-gray-400 text-xs">{a.created_at?.replace('T', ' ').split('.')[0]}</td>
                  <td className="px-4 py-2 font-medium text-gray-700">{a.username}</td>
                  <td className="px-4 py-2">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${actionColor(a.action)}`}>{a.action}</span>
                  </td>
                  <td className="px-4 py-2 text-gray-600">{a.document_type}</td>
                  <td className="px-4 py-2 text-gray-600">{a.document_no}</td>
                  <td className="px-4 py-2 text-gray-400 text-xs">{a.details}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {activity.length === 0 && <p className="text-center text-gray-400 py-6 text-sm">No activity logged yet</p>}
        </div>
      )}

      {tab === 'data-reset' && (
        <div className="space-y-4">
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
            <p className="text-sm font-semibold text-amber-800">Super Admin only — dangerous action</p>
            <p className="text-xs text-amber-700 mt-1">
              Use only to remove testing data in production. Each action permanently deletes one module&apos;s records.
            </p>
          </div>
          {!mayDeleteData && (
            <p className="text-sm text-gray-500 bg-gray-50 border border-gray-200 rounded-xl p-4">
              Only Super Admin can delete or reset module data. Admin and other roles can upload but not remove saved data.
            </p>
          )}
          {mayDeleteData && mayClearPlatform && (
            <div className="bg-white rounded-xl border border-red-200 p-4">
              <p className="text-sm font-semibold text-gray-800">Snapshot inventory (Upload Data)</p>
              <p className="text-xs text-gray-500 mt-1">
                Clears the daily OMS + marketplace inventory snapshot (wrong RAR/CSV). Does not delete sales history,
                SKU map, or PO data. After clearing, upload the correct file on Upload Data → Daily uploads.
              </p>
              <button
                type="button"
                onClick={() => void runClearInventorySnapshot()}
                disabled={clearInventoryBusy}
                className="mt-3 px-3 py-1.5 text-xs font-medium rounded-lg border border-red-300 text-red-700 hover:bg-red-50 disabled:opacity-50"
              >
                {clearInventoryBusy ? 'Clearing…' : 'Clear snapshot inventory'}
              </button>
              {clearInventoryMsg && (
                <p className="text-xs mt-2 text-gray-600">{clearInventoryMsg}</p>
              )}
            </div>
          )}
          {mayDeleteData && (
          <div className="grid md:grid-cols-2 gap-3">
            {RESET_MODULES.map(m => (
              <div key={m.key} className="bg-white rounded-xl border p-4">
                <p className="text-sm font-semibold text-gray-800">{m.label}</p>
                <p className="text-xs text-gray-500 mt-1">{m.detail}</p>
                <button
                  type="button"
                  onClick={() => runModuleReset(m.key, m.label)}
                  disabled={resetModuleMut.isPending}
                  className="mt-3 px-3 py-1.5 text-xs font-medium rounded-lg border border-red-300 text-red-700 hover:bg-red-50 disabled:opacity-50"
                >
                  {resetBusyModule === m.key ? 'Removing…' : `Remove ${m.label} Data`}
                </button>
              </div>
            ))}
          </div>
          )}
          {resetModuleMut.isSuccess && (
            <p className="text-sm text-green-700">
              {resetModuleMut.data?.message} ({resetModuleMut.data?.rows_deleted ?? 0} rows deleted)
            </p>
          )}
          {resetModuleMut.isError && (
            <p className="text-sm text-red-600">
              {formatUserCreateError(resetModuleMut.error)}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
