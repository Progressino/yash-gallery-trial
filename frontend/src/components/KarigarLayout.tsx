import { Outlet } from 'react-router-dom'
import api from '../api/client'
import { useAuth } from '../store/auth'

export default function KarigarLayout() {
  const user = useAuth(s => s.user)

  const clearAuth = useAuth(s => s.clear)

  const logout = async () => {
    try {
      await api.post('/auth/logout')
    } catch {
      /* still clear local session */
    }
    clearAuth()
    window.location.href = '/login'
  }

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col overflow-x-hidden">
      <header className="bg-[#002B5B] text-white px-4 py-3 pt-[max(0.75rem,env(safe-area-inset-top))] shadow-md sticky top-0 z-20">
        <div className="max-w-lg mx-auto flex items-center justify-between gap-3">
          <div>
            <h1 className="text-base font-bold leading-tight">Production Entry</h1>
            <p className="text-xs text-blue-200/90 mt-0.5">
              {user?.full_name || user?.username}
              {user?.karigar_id ? ` · ${user.karigar_id}` : ''}
            </p>
          </div>
          <button
            type="button"
            onClick={() => void logout()}
            className="text-xs px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20 touch-manipulation min-h-[40px]"
          >
            Log out
          </button>
        </div>
      </header>
      <main className="flex-1 p-3 pb-28 max-w-lg mx-auto w-full min-w-0 overflow-x-hidden">
        <Outlet />
      </main>
    </div>
  )
}
