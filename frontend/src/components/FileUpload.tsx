/**
 * Drag-and-drop file uploader component.
 * Accepts a list of allowed extensions and calls onUpload(file).
 */
import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

interface Props {
  label: string
  accept: Record<string, string[]>        // e.g. { 'application/zip': ['.zip'] }
  onUpload: (file: File) => Promise<void>
  uploading?: boolean
}

export default function FileUpload({ label, accept, onUpload, uploading }: Props) {
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback(
    async (accepted: File[]) => {
      if (!accepted.length) return
      setError(null)
      try {
        await onUpload(accepted[0])
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e))
      }
    },
    [onUpload],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    multiple: false,
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
        {uploading ? (
          <p className="text-sm text-blue-600 animate-pulse">Uploading…</p>
        ) : isDragActive ? (
          <p className="text-sm text-blue-600">Drop file here</p>
        ) : (
          <p className="text-sm text-gray-500">
            Drag & drop or <span className="text-blue-600 underline">browse</span>
          </p>
        )}
      </div>
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  )
}
