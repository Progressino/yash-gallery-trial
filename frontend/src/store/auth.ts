import { create } from 'zustand'

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

interface AuthState {
  user: AuthUser | null
  setUser: (user: AuthUser | null) => void
  clear: () => void
}

export const useAuth = create<AuthState>(set => ({
  user: null,
  setUser: user => set({ user }),
  clear: () => set({ user: null }),
}))

export function isKarigarUser(user: AuthUser | null | undefined): boolean {
  return user?.role === 'Karigar' || !!user?.is_karigar
}
