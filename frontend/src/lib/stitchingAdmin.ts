import { useCallback, useState } from 'react'
import api from '../api/client'

const UNLOCK_KEY = 'stitching_admin_unlocked'
const PW_KEY = 'stitching_admin_pw'

export function useStitchingAdmin() {
  const [unlocked, setUnlocked] = useState(() => sessionStorage.getItem(UNLOCK_KEY) === '1')

  const unlock = useCallback(async (password: string) => {
    const { data } = await api.post<{ ok: boolean; message?: string }>('/stitching/admin/unlock', {
      password,
    })
    if (data.ok) {
      sessionStorage.setItem(UNLOCK_KEY, '1')
      sessionStorage.setItem(PW_KEY, password)
      setUnlocked(true)
    }
    return data
  }, [])

  const lock = useCallback(() => {
    sessionStorage.removeItem(UNLOCK_KEY)
    sessionStorage.removeItem(PW_KEY)
    setUnlocked(false)
  }, [])

  const adminPassword = () => sessionStorage.getItem(PW_KEY) || ''

  return { unlocked, unlock, lock, adminPassword }
}
