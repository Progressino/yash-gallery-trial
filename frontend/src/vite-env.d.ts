/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_APP_GIT_SHA?: string
  readonly VITE_APP_BUILT_AT?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
