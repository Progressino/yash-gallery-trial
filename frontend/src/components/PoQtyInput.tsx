import { memo } from 'react'

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
  const isEdited = value !== computed
  return (
    <div
      className="flex items-center gap-1 min-w-[90px]"
      onClick={e => e.stopPropagation()}
    >
      <input
        type="number"
        value={value}
        min={0}
        step={5}
        onChange={e => onChange(Math.max(0, Math.round(+e.target.value / 5) * 5))}
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
