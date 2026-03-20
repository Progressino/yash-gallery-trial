import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../api/client'

type Tab = 'dashboard' | 'users' | 'roles' | 'activity'

interface AdminStats { total_users: number; total_roles: number; recent_activity: number; by_role: { role_name: string; cnt: number }[] }
interface ERPUser { id: number; username: string; email: string; full_name: string; role_id: number; role_name: string; department: string; active: number; created_at: string }
interface Role { id: number; role_name: string; description: string; created_at: string }
interface ActivityLog { id: number; username: string; action: string; document_type: string; document_no: string; details: string; created_at: string }

const DEPARTMENTS = ['Sales', 'Merchandising', 'Stores', 'Production', 'Quality', 'Logistics', 'Finance', 'Admin']

const actionColor = (a: string) => {
  if (a === 'create') return 'bg-green-100 text-green-700'
  if (a === 'update') return 'bg-blue-100 text-blue-700'
  if (a === 'delete') return 'bg-red-100 text-red-700'
  if (a === 'approve') return 'bg-purple-100 text-purple-700'
  return 'bg-gray-100 text-gray-600'
}

export default function Admin() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('dashboard')
  const [showUserForm, setShowUserForm] = useState(false)
  const [showRoleForm, setShowRoleForm] = useState(false)
  const [editUser, setEditUser] = useState<ERPUser | null>(null)
  const [editData, setEditData] = useState<Record<string, string | number>>({})

  const [userForm, setUserForm] = useState({ username: '', email: '', password: 'changeme123', full_name: '', role_id: 1, department: 'Production' })
  const [roleForm, setRoleForm] = useState({ role_name: '', description: '' })

  const { data: stats } = useQuery<AdminStats>({ queryKey: ['admin-stats'], queryFn: () => api.get('/erp-admin/stats').then(r => r.data) })
  const { data: users = [] } = useQuery<ERPUser[]>({ queryKey: ['erp-users'], queryFn: () => api.get('/erp-admin/users').then(r => r.data), enabled: tab === 'users' || tab === 'dashboard' })
  const { data: roles = [] } = useQuery<Role[]>({ queryKey: ['erp-roles'], queryFn: () => api.get('/erp-admin/roles').then(r => r.data) })
  const { data: activity = [] } = useQuery<ActivityLog[]>({ queryKey: ['activity-log'], queryFn: () => api.get('/erp-admin/activity?limit=100').then(r => r.data), enabled: tab === 'activity' })

  const createUserMut = useMutation({
    mutationFn: (b: object) => api.post('/erp-admin/users', b),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['erp-users'] }); qc.invalidateQueries({ queryKey: ['admin-stats'] }); setShowUserForm(false); setUserForm({ username: '', email: '', password: 'changeme123', full_name: '', role_id: 1, department: 'Production' }) }
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

  const TABS: [Tab, string][] = [['dashboard', '📊 Dashboard'], ['users', '👤 Users'], ['roles', '🔑 Roles'], ['activity', '📜 Activity Log']]

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-gray-800">Admin</h1>
        <p className="text-sm text-gray-500">User management, roles, and activity audit log</p>
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
          <div className="flex gap-3">
            <button onClick={() => { setTab('users'); setShowUserForm(true) }}
              className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ Add User</button>
            <button onClick={() => setTab('activity')}
              className="px-4 py-2 border border-[#002B5B] text-[#002B5B] rounded-lg text-sm font-medium hover:bg-gray-50">View Activity Log</button>
          </div>
        </div>
      )}

      {/* Users */}
      {tab === 'users' && (
        <div className="space-y-4">
          <div className="flex justify-between">
            <p className="text-sm text-gray-500">{users.length} users</p>
            <button onClick={() => setShowUserForm(true)} className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium hover:bg-blue-800">+ Add User</button>
          </div>

          {showUserForm && (
            <div className="bg-white rounded-xl border p-4 space-y-3">
              <h3 className="font-semibold text-gray-700">New User</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {[['username','Username *'],['email','Email'],['password','Password'],['full_name','Full Name']].map(([k,l]) => (
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
                    {DEPARTMENTS.map(d => <option key={d}>{d}</option>)}
                  </select>
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => createUserMut.mutate(userForm)} disabled={createUserMut.isPending || !userForm.username}
                  className="px-4 py-2 bg-[#002B5B] text-white rounded-lg text-sm font-medium disabled:opacity-50">
                  {createUserMut.isPending ? 'Saving…' : 'Create User'}
                </button>
                <button onClick={() => setShowUserForm(false)} className="px-4 py-2 border rounded-lg text-sm text-gray-600">Cancel</button>
              </div>
              {createUserMut.isError && <p className="text-xs text-red-500">Error: username or email already exists</p>}
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
                    {DEPARTMENTS.map(d => <option key={d}>{d}</option>)}
                  </select>
                </div>
                <div><label className="text-xs text-gray-500">New Password (optional)</label>
                  <input type="password" placeholder="Leave blank to keep" value={editData.password as string ?? ''}
                    onChange={e => setEditData(d => ({ ...d, password: e.target.value }))}
                    className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm mt-1" />
                </div>
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
                <tr>{['Username','Full Name','Role','Department','Email','Status','Actions'].map(h => <th key={h} className="text-left px-4 py-2">{h}</th>)}</tr>
              </thead>
              <tbody>
                {users.map(u => (
                  <tr key={u.id} className="border-t border-gray-50 hover:bg-gray-50">
                    <td className="px-4 py-2 font-medium text-gray-700">{u.username}</td>
                    <td className="px-4 py-2 text-gray-600">{u.full_name || '—'}</td>
                    <td className="px-4 py-2">
                      <span className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">{u.role_name}</span>
                    </td>
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
                        ) : null}
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
    </div>
  )
}
