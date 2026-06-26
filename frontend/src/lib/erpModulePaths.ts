/** ERP modules backed by SQLite — no PO / warm-cache datasets required. */
export const ERP_MODULE_PATH_PREFIXES = [
  '/purchase',
  '/production',
  '/items',
  '/sales',
  '/grey',
  '/tna',
  '/stitching-costing',
] as const

export function isErpModulePath(pathname: string): boolean {
  const p = pathname.replace(/\/$/, '') || '/'
  return ERP_MODULE_PATH_PREFIXES.some(
    prefix => p === prefix || p.startsWith(`${prefix}/`),
  )
}
