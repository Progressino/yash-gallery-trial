import { Link } from 'react-router-dom'
import { useSession } from '../store/session'
import { buildInventoryStalenessWarnings } from '../lib/inventoryStaleness'

export function InventoryStalenessBanner({
  className = '',
  showHistoryLink = true,
}: {
  className?: string
  showHistoryLink?: boolean
}) {
  const coverage = useSession()
  const warnings = buildInventoryStalenessWarnings(coverage)
  if (!warnings.length) return null

  return (
    <div
      className={`rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-950 ${className}`}
      role="status"
    >
      <p className="font-semibold mb-1">⚠ Inventory data may be out of date</p>
      <ul className="list-disc pl-5 space-y-1 text-amber-900">
        {warnings.map(w => (
          <li key={w}>{w}</li>
        ))}
      </ul>
      {showHistoryLink && coverage.daily_inventory_history && (
        <p className="mt-2 text-xs text-amber-800">
          <Link to="/inventory-history" className="font-semibold underline">
            Open Inventory History
          </Link>{' '}
          to verify on-hand qty by day (same layout as your Excel export).
        </p>
      )}
    </div>
  )
}
