import { create } from 'zustand'

const AUTH_STORAGE_KEY = 'erp_auth_profile_v1'

export interface AuthUser {
  username: string
  role: string
  full_name?: string
  karigar_id?: string
  user_id?: number
  department?: string
  permissions?: string[]
  is_karigar?: boolean
  historical_upload_locked?: boolean
  may_upload_historical?: boolean
  may_upload_daily?: boolean
  may_clear_platform?: boolean
  may_reset_all?: boolean
  may_save_shared_cache?: boolean
  may_reload_shared_cache?: boolean
  may_delete_daily_upload?: boolean
}

function readStoredUser(): AuthUser | null {
  try {
    const raw = sessionStorage.getItem(AUTH_STORAGE_KEY)
    if (!raw) return null
    const u = JSON.parse(raw) as AuthUser
    return u?.username ? u : null
  } catch {
    return null
  }
}

export function persistAuthUser(user: AuthUser | null) {
  try {
    if (user?.username) sessionStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(user))
    else sessionStorage.removeItem(AUTH_STORAGE_KEY)
  } catch {
    /* private mode */
  }
}

interface AuthState {
  user: AuthUser | null
  setUser: (user: AuthUser | null) => void
  clear: () => void
}

export const useAuth = create<AuthState>(set => ({
  user: readStoredUser(),
  setUser: user => {
    persistAuthUser(user)
    set({ user })
  },
  clear: () => {
    persistAuthUser(null)
    set({ user: null })
  },
}))

export function isKarigarUser(user: AuthUser | null | undefined): boolean {
  return user?.role === 'Karigar' || !!user?.is_karigar
}

/** Bulk history / platform clears — Admin & Manager when org lock is on. */
export function mayUploadHistorical(user: AuthUser | null | undefined): boolean {
  if (!user) return false
  if (user.may_upload_historical === true) return true
  if (user.may_upload_historical === false) return false
  return user.role === 'Admin' || user.role === 'Manager'
}

export function mayResetSharedData(user: AuthUser | null | undefined): boolean {
  if (!user) return false
  if (user.may_reset_all === true) return true
  if (user.may_reset_all === false) return false
  return user.role === 'Admin'
}
