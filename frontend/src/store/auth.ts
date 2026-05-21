import { create } from 'zustand'

const AUTH_STORAGE_KEY = 'erp_auth_profile_v1'

export interface HrmScopeInfo {
  level: 'all' | 'department' | 'self'
  department_id?: number | null
  employee_id?: number | null
  can_manage_org?: boolean
}

export interface AuthUser {
  username: string
  role: string
  full_name?: string
  karigar_id?: string
  user_id?: number
  department?: string
  employee_id?: number | null
  hrm_department_id?: number | null
  reporting_hod_user_id?: number | null
  module_access?: string
  modules?: string[]
  hrm_scope?: HrmScopeInfo
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
  /** Wide PO inventory history + SKU status/lead; Admin-only when org historical lock is on. */
  may_upload_po_baseline?: boolean
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

const FULL_ERP_ROLES = new Set(['Admin', 'Sir', 'Manager', 'Executive', 'Clerk', 'Viewer'])

export function userModules(user: AuthUser | null | undefined): string[] {
  if (!user) return []
  if (user.modules?.length) return user.modules
  if (user.role === 'Karigar') return ['stitching']
  if (user.role === 'HOD' || user.role === 'Employee') return ['hrm']
  if (FULL_ERP_ROLES.has(user.role || '')) return ['*']
  return ['hrm']
}

export function canAccessModule(user: AuthUser | null | undefined, moduleKey: string): boolean {
  const mods = userModules(user)
  if (mods.includes('*')) return true
  return mods.includes(moduleKey)
}

export function isHrmOnlyUser(user: AuthUser | null | undefined): boolean {
  const mods = userModules(user)
  return mods.length === 1 && mods[0] === 'hrm'
}

export function mayAccessErpAdmin(user: AuthUser | null | undefined): boolean {
  return user?.role === 'Admin' || user?.role === 'Manager' || user?.role === 'Sir'
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

/** PO baseline sheets (daily inventory history matrix, SKU status/lead) and related Admin-only uploads when locked. */
export function mayUploadPoBaseline(user: AuthUser | null | undefined): boolean {
  if (!user) return false
  if (user.may_upload_po_baseline === true) return true
  if (user.may_upload_po_baseline === false) return false
  return user.role === 'Admin' || user.role === 'Manager'
}

/** Removing saved Tier-3 daily files from the server. */
export function mayDeleteDailyUploadFile(user: AuthUser | null | undefined): boolean {
  if (!user) return false
  if (user.may_delete_daily_upload === true) return true
  if (user.may_delete_daily_upload === false) return false
  return user.role === 'Admin'
}
