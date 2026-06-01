import api from '../api/client'

/** Open print dialog for full stitching report pack (user can Save as PDF). */
export async function printStitchingReportsPack(dateFrom: string, dateTo: string): Promise<void> {
  const { data } = await api.get<string>('/stitching/reports/print', {
    params: { date_from: dateFrom, date_to: dateTo },
    responseType: 'text',
    headers: { Accept: 'text/html' },
  })
  const win = window.open('', '_blank', 'width=1100,height=900')
  if (!win) {
    alert('Allow pop-ups to print or save the report as PDF.')
    return
  }
  win.document.open()
  win.document.write(typeof data === 'string' ? data : String(data))
  win.document.close()
}
