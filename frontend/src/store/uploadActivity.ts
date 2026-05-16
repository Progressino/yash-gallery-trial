import { create } from 'zustand'

/** True while a file upload or background sales rebuild is in progress — suppress auth logout. */
interface UploadActivityState {
  busyCount: number
  begin: () => void
  end: () => void
}

export const useUploadActivity = create<UploadActivityState>((set, get) => ({
  busyCount: 0,
  begin: () => set({ busyCount: get().busyCount + 1 }),
  end: () => set({ busyCount: Math.max(0, get().busyCount - 1) }),
}))

export function isUploadBusy(): boolean {
  return useUploadActivity.getState().busyCount > 0
}
