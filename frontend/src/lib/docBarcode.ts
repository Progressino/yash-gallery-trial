/** Fetch QR (+ optional Code128) assets for document prints. */
export async function fetchDocBarcode(
  type: string,
  number: string,
): Promise<{ payload: string; qr_data_url: string; code128_data_url?: string | null }> {
  const { default: api } = await import('../api/client')
  const { data } = await api.get('/gate/barcode', { params: { type, number } })
  return data
}

/** HTML snippet to embed in print windows (right side of header). */
export function barcodePrintBlock(bundle: {
  payload: string
  qr_data_url: string
  code128_data_url?: string | null
}): string {
  const c128 = bundle.code128_data_url
    ? `<img src="${bundle.code128_data_url}" alt="barcode" style="height:36px;max-width:180px;display:block;margin:6px auto 0" />`
    : ''
  return `<div style="text-align:center">
    <img src="${bundle.qr_data_url}" width="100" height="100" alt="QR" />
    ${c128}
    <div style="font-size:9px;font-family:monospace;margin-top:4px;color:#334155">${bundle.payload}</div>
  </div>`
}
