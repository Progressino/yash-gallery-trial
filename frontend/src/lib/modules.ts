/** ERP sidebar module keys — must match backend/services/rbac.py ALL_MODULES */

export type ModuleKey =
  | 'intelligence'
  | 'upload'
  | 'amazon'
  | 'myntra'
  | 'meesho'
  | 'flipkart'
  | 'snapdeal'
  | 'forecast'
  | 'finance'
  | 'sku_deepdive'
  | 'sales'
  | 'items'
  | 'purchase'
  | 'tna'
  | 'production'
  | 'stitching'
  | 'grey'
  | 'hrm'
  | 'inventory'
  | 'po'
  | 'admin'
  | 'marketplace'

export const MODULE_LABELS: Record<ModuleKey, string> = {
  intelligence: 'Intelligence',
  upload: 'Upload Data',
  amazon: 'Amazon',
  myntra: 'Myntra',
  meesho: 'Meesho',
  flipkart: 'Flipkart',
  snapdeal: 'Snapdeal',
  forecast: 'AI Forecast',
  finance: 'Finance',
  sku_deepdive: 'SKU Deepdive',
  sales: 'Sales Orders',
  items: 'Item Master',
  purchase: 'Purchase',
  tna: 'TNA Calendar',
  production: 'Production',
  stitching: 'Stitching Costing',
  grey: 'Grey Fabric',
  hrm: 'HRM',
  inventory: 'Inventory',
  po: 'PO Engine',
  admin: 'Admin',
  marketplace: 'Marketplace API',
}

/** Nav path → module key */
export const PATH_MODULE: Record<string, ModuleKey> = {
  '/': 'intelligence',
  '/upload': 'upload',
  '/mtr': 'amazon',
  '/myntra': 'myntra',
  '/meesho': 'meesho',
  '/flipkart': 'flipkart',
  '/snapdeal': 'snapdeal',
  '/forecast': 'forecast',
  '/finance': 'finance',
  '/sku-deepdive': 'sku_deepdive',
  '/sales': 'sales',
  '/items': 'items',
  '/purchase': 'purchase',
  '/tna': 'tna',
  '/production': 'production',
  '/stitching-costing': 'stitching',
  '/grey': 'grey',
  '/hrm': 'hrm',
  '/inventory': 'inventory',
  '/inventory-history': 'po',
  '/sales-history': 'po',
  '/po-fresh': 'po',
  '/po-legacy': 'po',
  '/po2': 'po',
  '/po': 'po',
  '/admin': 'admin',
  '/marketplace-connections': 'marketplace',
}

export function moduleForPath(path: string): ModuleKey | null {
  if (path === '/') return 'intelligence'
  const entry = Object.entries(PATH_MODULE)
    .filter(([p]) => p !== '/')
    .sort((a, b) => b[0].length - a[0].length)
    .find(([p]) => path === p || path.startsWith(`${p}/`))
  return entry ? entry[1] : null
}

export const ALL_MODULE_KEYS = Object.keys(MODULE_LABELS) as ModuleKey[]
