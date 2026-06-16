import type { ReactNode } from 'react'
import {
  buildPORowBreakdown,
  getPOFormulaDef,
  type POFormulaContext,
  type POFormulaDef,
} from '../lib/poFormulaHelp'

export interface POFormulaModalState {
  key: string
  row?: Record<string, unknown>
  finalPoQty?: number
}

interface POFormulaModalProps {
  state: POFormulaModalState | null
  ctx: POFormulaContext
  onClose: () => void
  headerExtra?: ReactNode
}

export function POFormulaModal({ state, ctx, onClose, headerExtra }: POFormulaModalProps) {
  if (!state) return null

  const def = getPOFormulaDef(state.key)
  if (!def) return null

  const rowCtx: POFormulaContext = {
    ...ctx,
    finalPoQty: state.finalPoQty ?? ctx.finalPoQty,
  }
  const breakdown =
    state.row != null ? buildPORowBreakdown(state.key, state.row, rowCtx) : null

  return (
    <div
      className="fixed inset-0 z-[60] bg-black/40 flex items-center justify-center p-4"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="bg-white rounded-xl shadow-2xl w-full max-w-lg max-h-[85vh] flex flex-col"
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-labelledby="po-formula-title"
      >
        <div className="px-5 py-4 border-b border-gray-200 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 id="po-formula-title" className="text-lg font-bold text-gray-900">
              {def.title}
            </h3>
            <p className="text-sm text-gray-600 mt-1">{def.summary}</p>
            {headerExtra}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-700 text-xl leading-none shrink-0"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="flex-1 overflow-auto px-5 py-4 space-y-4 text-sm">
          <FormulaBlock def={def} />

          {breakdown && breakdown.length > 1 && (
            <div className="border-t border-gray-100 pt-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
                This row
              </p>
              <dl className="space-y-2">
                {breakdown.map((line, i) => (
                  <div
                    key={`${line.label}-${i}`}
                    className={`rounded-lg px-3 py-2 ${line.highlight ? 'bg-sky-50 border border-sky-100' : 'bg-gray-50'}`}
                  >
                    <dt className="text-xs text-gray-500">{line.label}</dt>
                    {line.expression && (
                      <dd className="font-mono text-[11px] text-gray-600 mt-0.5">{line.expression}</dd>
                    )}
                    <dd className={`font-semibold ${line.highlight ? 'text-sky-900' : 'text-gray-800'}`}>
                      {line.value}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
        </div>

        <div className="px-5 py-3 border-t border-gray-100 bg-gray-50 text-[11px] text-gray-500">
          Column headers show the general formula; click a cell value for SKU-specific numbers.
        </div>
      </div>
    </div>
  )
}

function FormulaBlock({ def }: { def: POFormulaDef }) {
  return (
    <div className="space-y-3">
      <div className="rounded-lg bg-slate-900 text-slate-50 px-3 py-2.5 font-mono text-xs leading-relaxed whitespace-pre-wrap">
        {def.formula}
      </div>
      {def.steps && def.steps.length > 0 && (
        <ol className="list-decimal list-inside space-y-1.5 text-gray-700">
          {def.steps.map((step, i) => (
            <li key={i} className="leading-snug">
              {step}
            </li>
          ))}
        </ol>
      )}
      {def.sources && def.sources.length > 0 && (
        <p className="text-xs text-gray-500">
          <span className="font-semibold">Data source:</span> {def.sources.join(' · ')}
        </p>
      )}
    </div>
  )
}

/** Clickable column header — opens formula popup. */
export function POFormulaHeaderButton({
  col,
  label,
  onOpen,
  className = '',
}: {
  col: string
  label: ReactNode
  onOpen: (col: string) => void
  className?: string
}) {
  if (!getPOFormulaDef(col)) {
    return <span className={className}>{label}</span>
  }
  return (
    <button
      type="button"
      onClick={() => onOpen(col)}
      className={`group inline-flex items-center gap-1 text-left hover:text-[#002B5B] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#002B5B]/40 rounded ${className}`}
      title="Click to see how this column is calculated"
    >
      <span>{label}</span>
      <span className="text-[10px] opacity-40 group-hover:opacity-100" aria-hidden>
        ⓘ
      </span>
    </button>
  )
}

/** Wraps a cell value so clicking opens row-specific formula breakdown. */
export function POFormulaCellTrigger({
  col,
  row,
  finalPoQty,
  onOpen,
  children,
  className = '',
}: {
  col: string
  row: Record<string, unknown>
  finalPoQty?: number
  onOpen: (col: string, row: Record<string, unknown>, finalPoQty?: number) => void
  children: ReactNode
  className?: string
}) {
  if (!getPOFormulaDef(col)) {
    return <>{children}</>
  }
  return (
    <button
      type="button"
      onClick={() => onOpen(col, row, finalPoQty)}
      className={`text-left hover:bg-sky-50/80 hover:ring-1 hover:ring-sky-200 rounded px-0.5 -mx-0.5 transition-colors ${className}`}
      title="Click to see how this value was calculated"
    >
      {children}
    </button>
  )
}
