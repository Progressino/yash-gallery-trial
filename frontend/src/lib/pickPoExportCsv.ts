/**
 * Pick a PO recommendation CSV, preferring the browser Downloads folder when supported
 * (Chrome / Edge File System Access API). Other browsers fall back to a normal file input.
 */

type FilePickerAccept = Record<string, string[]>

interface OpenFilePickerOptions {
  multiple?: boolean
  startIn?: 'desktop' | 'documents' | 'downloads' | 'music' | 'pictures' | 'videos'
  types?: Array<{ description: string; accept: FilePickerAccept }>
}

type FileSystemFileHandle = {
  getFile: () => Promise<File>
}

type WindowWithPicker = Window & {
  showOpenFilePicker?: (options?: OpenFilePickerOptions) => Promise<FileSystemFileHandle[]>
}

const PO_CSV_PICKER: OpenFilePickerOptions = {
  multiple: false,
  startIn: 'downloads',
  types: [
    {
      description: 'PO recommendation CSV',
      accept: {
        'text/csv': ['.csv'],
        'application/vnd.ms-excel': ['.csv'],
      },
    },
  ],
}

/** Returns a file from Downloads picker, or `null` if unsupported, cancelled, or user aborted. */
export async function pickPoExportCsvFromDownloads(): Promise<File | null> {
  const picker = (window as WindowWithPicker).showOpenFilePicker
  if (typeof picker !== 'function') return null

  try {
    const handles = await picker(PO_CSV_PICKER)
    const handle = handles?.[0]
    if (!handle) return null
    return await handle.getFile()
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === 'AbortError') return null
    throw e
  }
}

export function looksLikePoExportCsv(name: string): boolean {
  const n = name.toLowerCase()
  return n.endsWith('.csv') && (n.includes('po') || n.includes('recommendation') || n.includes('raise'))
}
