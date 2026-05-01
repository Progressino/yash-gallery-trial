/**
 * In-app user guide: how to use the Finance module, mapped to Tally Prime and
 * Microsoft Dynamics 365 Business Central concepts (including Microsoft Learn wording).
 */
const MS_LEARN_FINANCE = 'https://learn.microsoft.com/en-us/dynamics365/business-central/finance-setup-finance'
const MS_LEARN_COA = 'https://learn.microsoft.com/en-us/dynamics365/business-central/finance-chart-of-accounts'
const MS_LEARN_GL = 'https://learn.microsoft.com/en-us/dynamics365/business-central/finance-general-ledger'
const MS_LEARN_POSTING_GROUPS = 'https://learn.microsoft.com/en-us/dynamics365/business-central/finance-posting-groups'
const MS_LEARN_DIMENSIONS = 'https://learn.microsoft.com/en-us/dynamics365/business-central/finance-dimensions'
const MS_LEARN_VAT = 'https://learn.microsoft.com/en-us/dynamics365/business-central/finance-setup-vat'
const MS_LEARN_ACCOUNT_PERIODS = 'https://learn.microsoft.com/en-us/dynamics365/business-central/finance-accounting-periods-and-fiscal-years'
const MS_LEARN_BC_FINANCE = 'https://learn.microsoft.com/en-us/dynamics365/business-central/finance'

export function FinanceUserGuideContent() {
  return (
    <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-4 text-xs text-slate-800">
      <div>
        <p className="font-semibold text-slate-900 uppercase tracking-wide text-[11px]">
          Finance module — user guide
        </p>
        <p className="mt-1 text-slate-700 leading-relaxed">
          This area is designed for <strong>accounting-style workflows</strong> similar to{' '}
          <strong>Tally Prime</strong> (India-focused books, vouchers, GST) and{' '}
          <strong>Microsoft Dynamics 365 Business Central</strong> (chart of accounts, general ledger
          mindset, trial balance). Use it to keep <strong>finance-locked</strong> sales and vouchers
          separate from day-to-day operational uploads on the Upload page.
        </p>
        <p className="mt-2 text-[11px] text-slate-600 leading-relaxed">
          This product is <strong>not</strong> Business Central. The BC sections below summarize{' '}
          <strong>official Microsoft Learn</strong> terminology so your team can train with the same
          language used in Dynamics 365 documentation.
        </p>
      </div>

      <div className="rounded-lg border border-indigo-200 bg-indigo-50/60 px-3 py-3 space-y-2">
        <p className="font-semibold text-indigo-950 text-[11px] uppercase tracking-wide">
          Dynamics 365 Business Central — concepts from Microsoft Learn
        </p>
        <p className="text-slate-800 leading-relaxed">
          Per{' '}
          <a href={MS_LEARN_FINANCE} target="_blank" rel="noopener noreferrer" className="text-indigo-800 underline font-medium">
            Setting up finance
          </a>
          , you first define the core of accounting: the <strong>chart of accounts (COA)</strong>, then{' '}
          <strong>posting groups</strong> so customers, vendors, items, and documents default to the right{' '}
          <strong>general ledger (G/L)</strong> accounts. Optional setup covers dimensions, VAT, currencies,
          fiscal periods, payment terms, and financial reporting.
        </p>
        <p className="text-slate-800 leading-relaxed">
          Per{' '}
          <a href={MS_LEARN_COA} target="_blank" rel="noopener noreferrer" className="text-indigo-800 underline font-medium">
            Understanding the Chart of Accounts
          </a>
          , a COA is the directory of <strong>G/L accounts</strong> (with identifiers and descriptions). It groups{' '}
          <strong>balance sheet</strong> accounts (assets, liabilities, equity) and{' '}
          <strong>income statement</strong> accounts (revenue, expenses). You post transactions to the G/L through
          these accounts. BC also describes <strong>dimensions</strong> as categories on entries (e.g. department,
          project) so you can avoid exploding the COA with one account per department.
        </p>
        <p className="text-slate-800 leading-relaxed">
          Per{' '}
          <a href={MS_LEARN_GL} target="_blank" rel="noopener noreferrer" className="text-indigo-800 underline font-medium">
            Understand the general ledger and Chart of Accounts
          </a>
          , the <strong>G/L stores financial data</strong>; the <strong>COA lists the accounts</strong> you post G/L
          entries to. BC highlights <strong>General Ledger Setup</strong> and{' '}
          <strong>General Posting Setup</strong> as central configuration (invoice rounding, reporting, and which G/L
          accounts apply to which posting group combinations). In this app, maintain the same discipline via{' '}
          <strong>Masters</strong> (groups, ledgers, tax masters) and consistent voucher lines.
        </p>
        <p className="text-slate-800 leading-relaxed">
          <strong>Tax:</strong> BC documents{' '}
          <a href={MS_LEARN_VAT} target="_blank" rel="noopener noreferrer" className="text-indigo-800 underline font-medium">
            Value-Added Tax (VAT) setup
          </a>{' '}
          for reporting to tax authorities. In India your compliance focus here is <strong>GST</strong> (e.g.{' '}
          <strong>GSTR-3B</strong> working from posted sales and vouchers), not VAT — use the BC article only as a{' '}
          <em>parallel</em> for “how tax setup relates to G/L and posting.”
        </p>
        <p className="text-slate-800 leading-relaxed">
          <strong>How our tabs map to BC ideas:</strong>{' '}
          <strong>Chart of Accounts</strong> ≈ COA tree; <strong>Day Book</strong> ≈ dated view of posted activity
          (like reviewing G/L entries by date); <strong>Vouchers</strong> ≈ general journals / voucher entry;{' '}
          <strong>Trial Balance</strong> ≈ period debits, credits, and balancing check;{' '}
          <strong>P&amp;L</strong> ≈ income statement view; <strong>Sales Uploads</strong> ≈ bringing external sales
          data into the finance store (similar in spirit to journals or migration, but file-based here).{' '}
          <strong>Posting groups</strong> in BC map conceptually to how you assign ledgers and defaults in{' '}
          <strong>Masters</strong> — see{' '}
          <a href={MS_LEARN_POSTING_GROUPS} target="_blank" rel="noopener noreferrer" className="text-indigo-800 underline font-medium">
            Set Up Posting Groups
          </a>
          . Use{' '}
          <a href={MS_LEARN_DIMENSIONS} target="_blank" rel="noopener noreferrer" className="text-indigo-800 underline font-medium">
            Work with Dimensions
          </a>{' '}
          as reference if you use cost centres or analytic tags on voucher lines.
        </p>
        <p className="text-slate-800 leading-relaxed">
          <strong>BC best practices</strong> (from Microsoft Learn COA guidance): design the COA for management
          reporting; start simple and add detail later; use dimensions instead of one G/L account per product or
          department; add accounts as needed but plan careful cleanup at period-end. Apply the same thinking to
          ledger groups and ledgers here.
        </p>
        <p className="text-[11px] text-slate-600 leading-relaxed">
          More:{' '}
          <a href={MS_LEARN_ACCOUNT_PERIODS} target="_blank" rel="noopener noreferrer" className="text-indigo-800 underline">
            Accounting periods and fiscal years
          </a>
          {' · '}
          <a href={MS_LEARN_BC_FINANCE} target="_blank" rel="noopener noreferrer" className="text-indigo-800 underline">
            Finance overview (Business Central)
          </a>
        </p>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 space-y-2">
        <p className="font-semibold text-slate-900">Concept map: Tally / Business Central → this app</p>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-[11px]">
            <thead>
              <tr className="border-b border-slate-200 text-slate-600">
                <th className="py-1 pr-2 font-medium">Tally (typical)</th>
                <th className="py-1 pr-2 font-medium">Business Central (typical)</th>
                <th className="py-1 font-medium">Here</th>
              </tr>
            </thead>
            <tbody className="text-slate-700">
              <tr className="border-b border-slate-100 align-top">
                <td className="py-1.5 pr-2">Gateway of Tally → Day Book</td>
                <td className="py-1.5 pr-2">General Ledger entries (filtered by date)</td>
                <td className="py-1.5"><strong>Day Book</strong> tab — vouchers for a chosen date; click a row for detail.</td>
              </tr>
              <tr className="border-b border-slate-100 align-top">
                <td className="py-1.5 pr-2">Accounting vouchers (Payment, Receipt, Journal, …)</td>
                <td className="py-1.5 pr-2">General journals / purchase documents</td>
                <td className="py-1.5"><strong>Vouchers</strong> tab — create expense-style vouchers and lines; types follow your master setup.</td>
              </tr>
              <tr className="border-b border-slate-100 align-top">
                <td className="py-1.5 pr-2">Masters → Groups / Ledgers</td>
                <td className="py-1.5 pr-2">Chart of Accounts / G/L accounts</td>
                <td className="py-1.5"><strong>Masters</strong> (groups, ledgers, GST, TDS, voucher types) + <strong>Chart of Accounts</strong> tree.</td>
              </tr>
              <tr className="border-b border-slate-100 align-top">
                <td className="py-1.5 pr-2">Balance Sheet / Trial Balance</td>
                <td className="py-1.5 pr-2">Trial Balance report</td>
                <td className="py-1.5"><strong>Trial Balance</strong> tab for a period (opening + movements).</td>
              </tr>
              <tr className="border-b border-slate-100 align-top">
                <td className="py-1.5 pr-2">Profit &amp; Loss</td>
                <td className="py-1.5 pr-2">Income Statement</td>
                <td className="py-1.5"><strong>P&amp;L</strong> tab; optional <strong>Tally-style manual P&amp;L</strong> (FY buckets) when you maintain parallel books.</td>
              </tr>
              <tr className="border-b border-slate-100 align-top">
                <td className="py-1.5 pr-2">GST returns / GSTR-3B working</td>
                <td className="py-1.5 pr-2">Tax reports (locale-dependent)</td>
                <td className="py-1.5"><strong>GSTR-3B</strong> tab from posted finance sales and vouchers in range.</td>
              </tr>
              <tr className="align-top">
                <td className="py-1.5 pr-2">Import / manual sales books</td>
                <td className="py-1.5 pr-2">Sales journals / data migration</td>
                <td className="py-1.5"><strong>Sales Uploads</strong> — lock marketplace files into the finance DB; flows to Day Book, TB, GSTR views.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="space-y-2">
        <p className="font-semibold text-slate-900">Security (PIN)</p>
        <p className="text-slate-700 leading-relaxed">
          If your administrator set <code className="font-mono bg-slate-100 px-1 rounded">FINANCE_PIN</code> on the
          server, you must enter that PIN when opening Finance. This is similar to restricting sensitive
          areas in Tally or BC with permissions — keep the PIN confidential.
        </p>
      </div>

      <div className="space-y-2">
        <p className="font-semibold text-slate-900">Two revenue bases (important)</p>
        <ul className="list-disc pl-5 space-y-1 text-slate-700 leading-relaxed">
          <li>
            <strong>Finance Sales Uploads (locked)</strong> — figures come only from files you upload under{' '}
            <strong>Sales Uploads</strong>. Use this for accounts-final numbers and to avoid mixing with ops
            preview data (like BC “posted” vs “operational” slices).
          </li>
          <li>
            <strong>Operational (session / Upload page)</strong> — ties to the same session data as the
            Dashboard and Upload page. Useful for reconciliation, not a substitute for locked books if your
            process requires a separate finance close.
          </li>
        </ul>
      </div>

      <div className="space-y-2">
        <p className="font-semibold text-slate-900">Recommended first-time setup</p>
        <ol className="list-decimal pl-5 space-y-1.5 text-slate-700 leading-relaxed">
          <li>Open <strong>Masters</strong> and confirm <strong>ledger groups</strong> and core <strong>ledgers</strong> (debtors, creditors, sales, bank, GST output) match your chart.</li>
          <li>Add or edit <strong>GST classifications</strong> and <strong>TDS sections</strong> you use regularly.</li>
          <li>Review <strong>Voucher types</strong> so voucher entry labels match how your team works.</li>
          <li>In <strong>Chart of Accounts</strong>, expand the tree and sanity-check parent/child groups (Tally-style hierarchy).</li>
        </ol>
      </div>

      <div className="space-y-2">
        <p className="font-semibold text-slate-900">Tab-by-tab: what to do</p>
        <dl className="space-y-2 text-slate-700 leading-relaxed">
          <div>
            <dt className="font-medium text-slate-900">Dashboard</dt>
            <dd>Snapshot: today’s vouchers, month expenses, GST payable hint, quick links — like a finance home in BC.</dd>
          </div>
          <div>
            <dt className="font-medium text-slate-900">Day Book</dt>
            <dd>Pick a date; see every voucher that day (manual + finance sales entries). Open a line for invoice/SKU detail when the row is from a sales upload.</dd>
          </div>
          <div>
            <dt className="font-medium text-slate-900">Vouchers</dt>
            <dd>Create and list accounting vouchers; use narrative, party, tax splits, and line expense heads like Tally voucher lines.</dd>
          </div>
          <div>
            <dt className="font-medium text-slate-900">Masters</dt>
            <dd>Maintain groups, ledgers, GST/TDS masters, voucher types — same discipline as Tally masters or BC cards.</dd>
          </div>
          <div>
            <dt className="font-medium text-slate-900">Sales Uploads</dt>
            <dd>Upload marketplace or monthly packages; data persists in the finance database and drives Day Book / TB / GSTR-3B for those periods.</dd>
          </div>
          <div>
            <dt className="font-medium text-slate-900">GSTR-3B</dt>
            <dd>Select the return period; review outward taxable values and tax breakout driven from posted entries.</dd>
          </div>
          <div>
            <dt className="font-medium text-slate-900">P&amp;L / GST / Expenses / Revenue</dt>
            <dd>Reporting slices: profit and loss, GST summary from ops session data where applicable, expense register, platform revenue.</dd>
          </div>
          <div>
            <dt className="font-medium text-slate-900">Chart of Accounts &amp; Trial Balance</dt>
            <dd>COA tree for structure; Trial Balance for period debits/credits and balancing check before you sign off a month.</dd>
          </div>
          <div>
            <dt className="font-medium text-slate-900">Help / Notes (this page)</dt>
            <dd>User guide (above) plus field-level GL and sales definitions and an accountant UAT checklist below.</dd>
          </div>
        </dl>
      </div>

      <div className="rounded-lg border border-amber-200 bg-amber-50/80 px-3 py-2 space-y-1">
        <p className="font-semibold text-amber-950 text-[11px] uppercase tracking-wide">Daily &amp; month-end rhythm</p>
        <ul className="list-disc pl-5 text-slate-800 space-y-0.5 leading-relaxed">
          <li><strong>Daily:</strong> Post vouchers (payments, receipts, journals) → verify in Day Book.</li>
          <li><strong>When statements arrive:</strong> Upload sales/settlement files → reconcile Day Book totals to the file.</li>
          <li><strong>Month-end:</strong> Trial Balance for the month → GSTR-3B → P&amp;L (locked basis) → tie to Tally manual P&amp;L if you maintain both.</li>
        </ul>
      </div>

      <p className="text-[11px] text-slate-500 leading-relaxed">
        For column-level definitions (ship-from vs ship-to, B2B/B2C, HSN, e-invoice notes), scroll to{' '}
        <strong>Sales Report &amp; General Ledger: Field Definitions</strong> below.
      </p>
    </div>
  )
}
