/** Item Master product image helpers for UI + print HTML. */

export function itemImageUrl(code: string, bust?: string | number): string {
  const c = String(code || '').trim()
  if (!c) return ''
  const q = bust != null && String(bust) !== '' ? `?t=${encodeURIComponent(String(bust))}` : ''
  return `/api/items/by-code/${encodeURIComponent(c)}/image${q}`
}

export async function fetchItemImageDataUrl(code: string): Promise<string | null> {
  const c = String(code || '').trim()
  if (!c) return null
  try {
    const res = await fetch(itemImageUrl(c), { credentials: 'include' })
    if (!res.ok) return null
    const blob = await res.blob()
    if (!blob || blob.size === 0) return null
    return await new Promise<string | null>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const result = String(reader.result || '')
        resolve(result.startsWith('data:') ? result : null)
      }
      reader.onerror = () => reject(reader.error)
      reader.readAsDataURL(blob)
    })
  } catch {
    return null
  }
}

/** Parallel resolve of item codes → data URLs (missing codes omitted). */
export async function fetchItemImageDataUrlMap(codes: string[]): Promise<Record<string, string>> {
  const unique = [
    ...new Set(
      codes
        .map(c => String(c || '').trim())
        .filter(Boolean),
    ),
  ]
  const out: Record<string, string> = {}
  await Promise.all(
    unique.map(async code => {
      const url = await fetchItemImageDataUrl(code)
      if (url) out[code] = url
    }),
  )
  return out
}

export function printThumbHtml(dataUrl: string | undefined | null, size = 48): string {
  if (!dataUrl) return ''
  return `<img src="${dataUrl}" alt="" style="width:${size}px;height:${size}px;object-fit:cover;border-radius:4px;border:1px solid #e2e8f0;vertical-align:middle;margin-right:6px" />`
}
