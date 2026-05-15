import { Navigate, Outlet } from 'react-router-dom'
import { useAuth, isKarigarUser } from '../store/auth'

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
