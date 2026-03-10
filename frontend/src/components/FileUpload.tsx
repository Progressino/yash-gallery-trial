/**
 * Drag-and-drop file uploader component.
 * Accepts a list of allowed extensions and calls onUpload(file) per file.
 * When multiple=true, the dropzone allows selecting multiple files and calls
 * onUpload sequentially for each one.
 */
import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

interface Props {
  label: string
  accept: Record<string, string[]>        // e.g. { 'application/zip': ['.zip'] }
  onUpload: (file: File) => Promise<void>
  uploading?: boolean
  multiple?: boolean
}

export default function FileUpload({ label, accept, onUpload, uploading, multiple = false }: Props) {
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<string | null>(null)

  const onDrop = useCallback(
    async (accepted: File[]) => {
      if (!accepted.length) return
      setError(null)
      setProgress(null)
      try {
        if (multiple && accepted.length > 1) {
          for (let i = 0; i < accepted.length; i++) {
            setProgress(`Uploading ${i + 1}/${accepted.length}: ${accepted[i].name}…`)
            await onUpload(accepted[i])
          }
          setProgress(null)
        } else {
          await onUpload(accepted[0])
        }
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e))
        setProgress(null)
      }
    },
    [onUpload, multiple],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    multiple,
    disabled: uploading,
  })

  return (
    <div className="space-y-1">
      <p className="text-sm font-medium text-gray-700">{label}</p>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400 bg-white'
        } ${uploading ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        {uploading || progress ? (
          <p className="text-sm text-blue-600 animate-pulse">{progress || 'Uploading…'}</p>
        ) : isDragActive ? (
          <p className="text-sm text-blue-600">Drop file{multiple ? 's' : ''} here</p>
        ) : (
          <p className="text-sm text-gray-500">
            Drag & drop{multiple ? ' one or more files' : ''} or{' '}
            <span className="text-blue-600 underline">browse</span>
          </p>
        )}
      </div>
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  )
}
