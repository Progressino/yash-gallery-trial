/** Shared Yash Gallery brand block for printable ERP documents. */

import { YASH_GALLERY_LOGO_DATA_URL } from './yashGalleryLogoDataUrl'

export const YASH_GALLERY = {
  name: 'Yash Gallery Pvt. Ltd.',
  shortName: 'Yash Gallery',
  address: '55 TO 64, Tatiyawas, Brij Vihar, Amber, Jaipur (Raj.) 303704',
  gst: '08AABCY3804E1ZJ',
  phone: '',
  email: 'purchase@yashgallery.com',
}

/** Absolute URL to /logo.png (works in app UI). */
export function brandLogoUrl(): string {
  if (typeof window === 'undefined') return '/logo.png'
  return `${window.location.origin}/logo.png`
}

/**
 * Logo as an embedded data URL (no network / Cloudflare / about:blank issues).
 * Sync-friendly: always available for print HTML builders.
 */
export async function brandLogoDataUrl(): Promise<string> {
  return YASH_GALLERY_LOGO_DATA_URL
}

/** Sync accessor for builders that already await other work. */
export function brandLogoDataUrlSync(): string {
  return YASH_GALLERY_LOGO_DATA_URL
}

export type BrandHeaderOpts = {
  department?: string
  docTitle: string
  docNumber: string
  barcodeHtml?: string
  /** Extra HTML under company meta (e.g. style thumb) */
  extraLeft?: string
  /** Logo as data URL or absolute URL */
  logoSrc?: string
}

/** Left company block + right document title/number. */
export function brandPrintHeaderHtml(opts: BrandHeaderOpts): string {
  const logo = opts.logoSrc || YASH_GALLERY_LOGO_DATA_URL
  const dept = opts.department || ''
  const barcode = opts.barcodeHtml || ''
  return `
    <div class="header">
      <div class="company-block">
        <img src="${logo}" alt="Yash Gallery" class="company-logo" />
        <div>
          <div class="company-name">${YASH_GALLERY.name}</div>
          <div class="company-meta company-address">${YASH_GALLERY.address.replace(/\n/g, '<br/>')}</div>
          ${YASH_GALLERY.gst ? `<div class="company-meta"><strong>GSTIN:</strong> ${YASH_GALLERY.gst}</div>` : ''}
          ${YASH_GALLERY.phone ? `<div class="company-meta">Tel: ${YASH_GALLERY.phone}</div>` : ''}
          ${YASH_GALLERY.email ? `<div class="company-meta">${YASH_GALLERY.email}</div>` : ''}
          ${dept ? `<div class="company-meta company-dept">${dept}</div>` : ''}
          ${opts.extraLeft || ''}
        </div>
      </div>
      <div>
        <div class="doc-title">${opts.docTitle}</div>
        <div class="doc-num">${opts.docNumber}</div>
        ${barcode ? `<div style="margin-top:8px;display:flex;justify-content:flex-end">${barcode}</div>` : ''}
      </div>
    </div>`
}

/** Shared CSS snippet for brand header (include in print window styles). */
export const BRAND_PRINT_CSS = `
    .header{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:2px solid #002B5B;padding-bottom:12px;margin-bottom:16px;gap:16px}
    .company-block{display:flex;gap:14px;align-items:flex-start;min-width:0;flex:1}
    .company-logo{height:64px;width:auto;max-width:180px;object-fit:contain;display:block;flex-shrink:0}
    .company-name{font-size:18px;font-weight:700;color:#002B5B;line-height:1.25}
    .company-meta{font-size:11px;color:#334155;line-height:1.45;margin-top:3px}
    .company-address{font-size:11px;color:#1e293b;font-weight:500}
    .company-dept{margin-top:6px;font-weight:700;color:#002B5B}
    .doc-title{font-size:16px;font-weight:600;color:#002B5B;text-align:right}
    .doc-num{font-size:22px;font-weight:800;color:#002B5B;text-align:right}
`
