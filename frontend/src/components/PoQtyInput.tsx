import { memo, useState, useEffect } from 'react'

export const PoQtyInput = memo(function PoQtyInput({
  value,
  computed,
  onChange,
  onReset,
}: {
  value: number
  computed: number
  onChange: (v: number) => void
  onReset: () => void
}) {
  // Local raw string lets the user type freely; we only round+commit on blur or Enter
  const [raw, setRaw] = useState(String(value))

  // Sync display when the external value changes (e.g. reset or recalc from parent)
  useEffect(() => {
    setRaw(String(value))
  }, [value])

  const commit = (s: string) => {
    const n = parseFloat(s)
    if (!isNaN(n)) {
      onChange(Math.max(0, Math.round(n / 5) * 5))
    } else {
      setRaw(String(value)) // restore on invalid input
    }
  }

  const isEdited = value !== computed
  return (
    <div
      className="flex items-center gap-1 min-w-[90px]"
      onClick={e => e.stopPropagation()}
    >
      <input
        type="number"
        value={raw}
        min={0}
        step={5}
        onChange={e => setRaw(e.target.value)}
        onBlur={e => commit(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter') {
            commit((e.target as HTMLInputElement).value)
            ;(e.target as HTMLInputElement).blur()
          }
        }}
        className={`w-20 border rounded px-2 py-1 text-xs text-right font-bold focus:outline-none focus:ring-1
          ${isEdited
            ? 'border-orange-400 bg-orange-50 text-orange-700 ring-orange-300 focus:ring-orange-400'
            : 'border-[var(--po-outline-ghost)] bg-white text-[var(--po-primary)] focus:ring-[var(--po-secondary)]'
          }`}
      />
      {isEdited && (
        <button
          type="button"
          onClick={onReset}
          title={`Reset to ${computed}`}
          className="text-[var(--po-outline)] hover:text-gray-600 text-sm leading-none"
        >
          ↩
        </button>
      )}
    </div>
  )
})
