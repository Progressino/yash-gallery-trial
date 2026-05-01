type Props = {
  /** Full checklist for accountants (Help / Notes tab). */
  showAccountantChecklist?: boolean
}

/**
 * In-app copy: Sales report & GL field definitions (aligned with accountant-facing notes).
 */
export function GlSalesNotesContent({ showAccountantChecklist = false }: Props) {
  return (
    <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 space-y-3">
      <p className="text-xs font-semibold text-blue-900 uppercase tracking-wide">
        Sales Report &amp; General Ledger: Field Definitions (Notes)
      </p>
      <p className="text-xs text-blue-800">
        <strong>Purpose:</strong> Use these notes to understand what each column/field means in the General Ledger (GL) and Sales Report, and how to determine the correct GST tax type.
      </p>
      <div className="text-xs text-blue-900 space-y-2">
        <p><strong>1) General Ledger (GL)</strong></p>
        <p>General Ledger is the main accounting record that contains all financial transactions posted under different accounts (sales, purchase, GST output, bank, cash, customer/vendor, expenses, etc.).</p>

        <p><strong>2) Sales Report: Common Fields</strong></p>
        <p><strong>Import Order ID:</strong> The unique reference number of the order imported from another system/platform (for example: e-commerce marketplace order ID). Used to trace the sale back to the original order.</p>
        <p><strong>Invoice No.:</strong> The tax invoice number generated for the sale. This is the primary reference for accounting and GST compliance.</p>
        <p><strong>Ship From:</strong> The dispatch/origin location (warehouse/store/state) from where goods are shipped.</p>
        <p><strong>Ship To:</strong> The delivery/destination location (customer address/state) where goods are shipped.</p>
        <p><strong>Warehouse / Location (Ship From Warehouse):</strong> The specific warehouse/store name (location) from where the goods are issued/sold. The sales report should always capture the warehouse location name (for example: WH-Delhi, WH-Jaipur, WH-Mumbai) to track stock movement and state-wise GST reporting.</p>
        <p><strong>Type of Sale:</strong> Classification of the customer/transaction type, typically:</p>
        <ul className="list-disc pl-5 space-y-0.5">
          <li><strong>Registered:</strong> Buyer has GSTIN (B2B). Invoice includes buyer GSTIN.</li>
          <li><strong>Unregistered:</strong> Buyer does not have GSTIN (B2C).</li>
          <li><strong>E-commerce:</strong> Sale through an e-commerce operator/marketplace (may involve TCS and platform order references, depending on setup).</li>
        </ul>
        <p><strong>GST Type (Tax Split):</strong> Whether the sale attracts CGST + SGST or IGST, determined mainly by the place of supply (usually the customer/delivery state) and the location of supplier (ship-from state / supplier&apos;s registered state).</p>
        <ul className="list-disc pl-5 space-y-0.5">
          <li><strong>Intra-state supply</strong> (Ship From state = Ship To state): Apply CGST + SGST.</li>
          <li><strong>Inter-state supply</strong> (Ship From state ≠ Ship To state): Apply IGST.</li>
        </ul>
        <p><strong>HSN Code:</strong> Harmonized System of Nomenclature code for the product. Used to identify the product category and applicable GST rate. Also critical for e-Invoicing, because item-level details (including HSN) are part of the e-invoice data sent for IRN generation.</p>
        <p><strong>Item Sold / Product Name:</strong> The name/description of the product being sold (SKU/item). This should match the item master/inventory naming used in the system.</p>

        <div className="rounded-lg border border-blue-300 bg-white/70 px-3 py-2">
          <p className="font-semibold">3) Quick GST Decision Rule (Summary)</p>
          <p className="mt-0.5">If Ship From and Ship To are in the same state: <strong>CGST + SGST</strong>. If they are in different states: <strong>IGST</strong>.</p>
        </div>

        <p><strong>3A) E-Invoicing &amp; e-Way Bill (B2B Compliance Notes)</strong></p>
        <p><strong>HSN for e-Invoice:</strong> HSN is an important mandatory item-level field for e-invoicing. Correct HSN helps ensure the e-invoice payload is valid and tax classification is correct.</p>
        <p><strong>IRN &amp; ACK details on Sales Invoice:</strong> For applicable B2B invoices, the system should store and print the IRN (Invoice Reference Number) and Acknowledgement (ACK) Number (and ACK date/time, if available), because these are generated from the e-invoice system and are required for compliance and audit trail.</p>
        <p><strong>Applicability (turnover threshold):</strong> E-invoicing applicability is generally based on the PAN-based aggregate turnover (all firms/registrations under the same PAN). If the PAN crosses the notified threshold, e-invoicing becomes mandatory for applicable invoice types.</p>
        <p><strong>e-Way Bill data capture:</strong> If e-way bill is applicable/required, capture e-way bill details in the system (e.g., EWB No., EWB date, valid till, transport document no., vehicle no., transporter name/ID). This is necessary to generate correct operational and compliance reports.</p>

        <p><strong>4) Static Log (Platform Mapping) for E-commerce Sales</strong></p>
        <p>Static Log means a fixed/master mapping that helps identify sales coming from different channels (Amazon, Flipkart, Meesho, website, POS, etc.) and ensures the system posts them consistently in reports and accounts.</p>
        <ul className="list-disc pl-5 space-y-0.5">
          <li>Sales channel / platform name (e-commerce operator name)</li>
          <li>Channel order identifier format (how Import Order ID is captured)</li>
          <li>Default customer/party name (if platform sales are booked under a common party)</li>
          <li>Default sales ledger to be used for that platform/channel</li>
          <li>GST/TCS settings (if applicable in your process)</li>
          <li>Return/refund mapping (how credit notes/returns are identified)</li>
        </ul>

        <p><strong>5) Sales Ledger Type (Which ledger the invoice posts to)</strong></p>
        <p>Each sales invoice should have (or derive) a Sales Ledger Type, meaning the specific revenue ledger/account where the sale amount will be posted in the General Ledger. This helps separate reporting by channel and tax nature.</p>
        <ul className="list-disc pl-5 space-y-0.5">
          <li>B2B Sales (Registered) ledger</li>
          <li>B2C Sales (Unregistered) ledger</li>
          <li>E-commerce Sales ledger (platform-wise if needed, e.g., Amazon Sales, Flipkart Sales)</li>
          <li>Inter-state Sales ledger vs Intra-state Sales ledger (if your accounts are maintained separately)</li>
        </ul>

        <p><strong>6) Master Tab: Ledger Creation (Important Setup Points)</strong></p>
        <p>The ledger master (Master Tab) controls how transactions post into the General Ledger and how GST/e-invoice reporting behaves. Create and maintain ledgers carefully, especially when you have multiple GST registrations (multiple states) and multiple sales channels.</p>
        <p><strong>Ledger Name:</strong> Standard naming so reports stay consistent (example: B2B Sales, Amazon Sales, Output CGST, Output IGST, Customer - ABC Traders).</p>
        <p><strong>Ledger Group / Type:</strong> Sales, Sundry Debtors (Customer), Duties &amp; Taxes (GST Output), Bank/Cash, or Clearing/Receivable (for e-commerce settlements).</p>
        <p><strong>GST Registration / Branch linkage:</strong> If your system supports it, map the ledger to the correct company/branch/state GSTIN rules so invoices pick the correct GSTIN automatically.</p>
        <p><strong>GST/TAX Settings:</strong> For GST ledgers, confirm the correct tax type (CGST/SGST/IGST), rate handling, and whether it is Output (sales) or Input (purchase).</p>
        <p><strong>Party / Customer Settings (for debtor ledgers):</strong> GSTIN, State, Address, and customer type (Registered/Unregistered). These drive place of supply and B2B/B2C classification.</p>
        <p><strong>Opening Balance &amp; Credit Limit (if used):</strong> Keep master data aligned with accounting controls.</p>
        <p><strong>Recommended ledgers to create (examples):</strong></p>
        <ul className="list-disc pl-5 space-y-0.5">
          <li><strong>Sales ledgers:</strong> B2B Sales, B2C Sales, Inter-state Sales, Intra-state Sales, and/or platform-wise sales ledgers (Amazon Sales, Flipkart Sales, etc.) as per your reporting.</li>
          <li><strong>Customer (Debtor) ledgers:</strong> Create customer ledgers with correct GSTIN/state/address so B2B invoices and e-invoice data are correct.</li>
          <li><strong>GST Output ledgers:</strong> Output CGST, Output SGST, Output IGST (and Cess if applicable) for correct tax posting and GST return reconciliation.</li>
          <li><strong>E-commerce settlement / clearing ledgers (if applicable):</strong> Marketplace Receivable/Clearing, Commission Charges, Shipping Charges, and TCS Receivable/Payable—so that settlements match bank receipts and platform reports.</li>
          <li><strong>Freight/Transport ledgers (if charged separately):</strong> Freight Outward / Delivery Charges, so e-way bill related transport amounts can be tracked in reports if required.</li>
        </ul>

        <p><strong>7) Sales Invoice Print: What you should see</strong></p>
        <ul className="list-disc pl-5 space-y-0.5">
          <li>Customer name and complete billing/shipping address</li>
          <li>Invoice date (date when the sale is made)</li>
          <li>Invoice number</li>
          <li>Ship From state (state from which you are selling/dispatching)</li>
          <li>Warehouse / dispatch location name (important when you have multiple warehouses with different names)</li>
          <li>Ship To state (delivery state)</li>
          <li>GSTIN used on the invoice (correct branch/state GSTIN when registered in multiple states)</li>
          <li>Customer GSTIN (for registered/B2B sales), if applicable</li>
          <li>Place of supply (generally based on Ship To)</li>
          <li>IRN &amp; ACK details (for applicable B2B invoices): IRN, ACK number, and ACK date/time (if provided)</li>
          <li>e-Way Bill details (when applicable): EWB No., EWB date, valid till, vehicle no., transporter details, transport document no.</li>
          <li>Item details line-wise: Product name/SKU, HSN code, quantity and unit, taxable value, GST rate (%), tax amount (CGST/SGST/IGST breakup), line total and invoice total</li>
        </ul>

        <p><strong>8) Inventory Effect (Stock impact of sales invoice)</strong></p>
        <p>The product name/SKU on the invoice should match the inventory item master. When the sales invoice is posted, the system reduces stock based on the item and quantity (and usually the warehouse/location). Correct item mapping ensures inventory reports show accurate stock, sales quantity, and item-wise taxable value.</p>
      </div>

      {showAccountantChecklist && (
        <div className="rounded-lg border border-blue-400 bg-white/90 px-3 py-3 text-xs text-blue-950 space-y-2">
          <p className="font-semibold text-blue-900 uppercase tracking-wide">Accountant verification checklist (UAT)</p>
          <p className="text-blue-800">Use a <strong>non-production month</strong> or small test file first. After each step, confirm figures in Finance only (finance-locked data).</p>
          <ol className="list-decimal pl-5 space-y-1.5 text-blue-900">
            <li><strong>Read definitions</strong> — Skim sections 1–8 above; note Ship From / Ship To vs CGST+SGST vs IGST.</li>
            <li><strong>Masters</strong> — Under Finance → Masters → Ledgers, confirm a few sample ledgers (sales, debtors, GST output) exist with sensible names; add a test ledger if your process requires it.</li>
            <li><strong>Manual voucher (dry run)</strong> — Finance → Vouchers: create a small <strong>Expense</strong> (or <strong>Payment</strong>) with one line, save. Open <strong>Day Book</strong> for that voucher date and confirm the row appears with correct amounts.</li>
            <li><strong>Sales upload (Amazon / monthly package)</strong> — Finance → Sales Uploads: upload a known file for a test period; confirm success panel and <code className="font-mono bg-blue-100 px-0.5 rounded">sales_entry_rows</code> / totals if shown.</li>
            <li><strong>Day Book (invoice-level)</strong> — Pick a calendar day that appears in your Amazon/MTR data. Confirm <strong>SUE-</strong> rows (one per invoice/order where line detail exists). Click a row: entry modal should show invoice no., order ID, SKU lines, and tax split if present.</li>
            <li><strong>GSTR-3B / P&amp;L / Platform Revenue</strong> — Select the same month; confirm uploaded sales move totals in a consistent direction (no double-count vs Day Book for the same logic).</li>
            <li><strong>Trial Balance / Chart of Accounts</strong> — Open Trial Balance for the period; confirm <strong>Finance Sales Upload A/c</strong> (or equivalent) reflects finance-upload sales; spot-check one ledger balance.</li>
            <li><strong>Cleanup</strong> — If this was test data, delete the test voucher from Day Book (real expense rows only) and remove test sales uploads from Sales Uploads if your policy allows.</li>
          </ol>
        </div>
      )}
    </div>
  )
}
