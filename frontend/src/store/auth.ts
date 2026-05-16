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
