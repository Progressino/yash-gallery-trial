import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth, isKarigarUser, canAccessModule, userModules } from '../store/auth'
import { moduleForPath } from '../lib/modules'

export function KarigarGate() {
  const user = useAuth(s => s.user)
  if (!user) return null
  if (!isKarigarUser(user)) return <Navigate to="/" replace />
  return <Outlet />
}

export function StaffGate() {
  const user = useAuth(s => s.user)
  if (!user) return null
  if (isKarigarUser(user)) return <Navigate to="/production-entry" replace />
  return <Outlet />
}

/** Block routes outside the user's allowed ERP modules (e.g. HOD/Employee → HRM only). */
export function ModuleAccessGate() {
  const user = useAuth(s => s.user)
  const { pathname } = useLocation()
  if (!user) return null
  const mod = moduleForPath(pathname)
  if (mod && !canAccessModule(user, mod)) {
    const mods = userModules(user)
    const dest = mods.includes('hrm') ? '/hrm' : mods.includes('intelligence') ? '/' : '/hrm'
    return <Navigate to={dest} replace />
  }
  return <Outlet />
}
