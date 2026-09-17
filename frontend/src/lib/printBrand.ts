/** Shared Yash Gallery brand block for printable ERP documents. */

export const YASH_GALLERY = {
  name: 'Yash Gallery Pvt. Ltd.',
  shortName: 'Yash Gallery',
  address: 'Bhiwandi, Thane District, Maharashtra — 421302, India',
  gst: '',
  phone: '',
  email: 'purchase@yashgallery.com',
}

let _logoDataUrl: string | null = null
let _logoPromise: Promise<string> | null = null

/** Absolute URL to /logo.png (works in app UI). */
export function brandLogoUrl(): string {
  if (typeof window === 'undefined') return '/logo.png'
  return `${window.location.origin}/logo.png`
}

/**
 * Load logo as a data URL so print popups always render the image
 * (even when relative paths / cookies fail in about:blank windows).
 */
export async function brandLogoDataUrl(): Promise<string> {
  if (_logoDataUrl) return _logoDataUrl
  if (_logoPromise) return _logoPromise
  _logoPromise = (async () => {
    try {
      const res = await fetch(brandLogoUrl(), { credentials: 'same-origin', cache: 'force-cache' })
      if (!res.ok) throw new Error(`logo ${res.status}`)
      const blob = await res.blob()
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(String(reader.result || ''))
        reader.onerror = () => reject(reader.error)
        reader.readAsDataURL(blob)
      })
      _logoDataUrl = dataUrl
      return dataUrl
    } catch {
      // Fall back to absolute URL
      return brandLogoUrl()
    }
  })()
  return _logoPromise
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
  const logo = opts.logoSrc || brandLogoUrl()
  const dept = opts.department || ''
  const barcode = opts.barcodeHtml || ''
  return `
    <div class="header">
      <div class="company-block">
        <img src="${logo}" alt="Yash Gallery" class="company-logo" onerror="this.style.display='none'" />
        <div>
          <div class="company-name">${YASH_GALLERY.name}</div>
          <div class="company-meta">${YASH_GALLERY.address.replace(/\n/g, '<br/>')}</div>
          ${YASH_GALLERY.phone ? `<div class="company-meta">Tel: ${YASH_GALLERY.phone}</div>` : ''}
          ${YASH_GALLERY.email ? `<div class="company-meta">${YASH_GALLERY.email}</div>` : ''}
          ${dept ? `<div class="company-meta" style="margin-top:4px;font-weight:600;color:#002B5B">${dept}</div>` : ''}
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
    .header{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:2px solid #002B5B;padding-bottom:12px;margin-bottom:16px}
    .company-block{display:flex;gap:14px;align-items:flex-start}
    .company-logo{height:56px;width:auto;max-width:160px;object-fit:contain}
    .company-name{font-size:18px;font-weight:700;color:#002B5B;line-height:1.2}
    .company-meta{font-size:10px;color:#475569;line-height:1.5;margin-top:2px}
    .doc-title{font-size:16px;font-weight:600;color:#002B5B;text-align:right}
    .doc-num{font-size:22px;font-weight:800;color:#002B5B;text-align:right}
`
