import api from '../api/client'

/** Best-effort: save export on server so next-day Calculate PO can auto-import the ledger. */
export async function archivePoExportOnServer(csv: string, raisedDate: string): Promise<void> {
  try {
    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' })
    const fd = new FormData()
    fd.append('file', blob, 'po_recommendation.csv')
    fd.append('raised_date', raisedDate)
    await api.post('/po/raise-ledger/archive-export', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60_000,
    })
  } catch {
    // Non-blocking — download still succeeded for the operator.
  }
}
