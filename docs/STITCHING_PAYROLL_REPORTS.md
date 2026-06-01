# Stitching Costing — Payroll & Reports

## Payroll calculation

| Component | Source | Rules |
|-----------|--------|--------|
| **Attendance pay** | Karigar Attendance tab / biometric upload | 09:00–18:00 work blocks; lunch/tea deductions; late after 09:00; early leave; OT after 18:00 (whole hours, rate = daily÷8) |
| **Other work pay** | Karigar Expenses tab | Part change, alter, trainee, etc. — amount per row, often tied to challan |
| **Total payroll** | Payroll tab / Reports | `Attendance_Pay + Other_Work_Pay` per karigar for the date range |

Production Entry **costing** (₹480 benchmark, budgeted vs actual) is separate from cash payroll but used for factory P&L and LTL efficiency.

## Reports tab (Dynamics-style pack)

**Stitching Costing → Reports → Run all reports**

One hub loads:

1. **Karigar profitability** — payroll paid vs piece value vs benchmark P&L; LTL column; profitable yes/no on payroll and on benchmark  
2. **Challan labour** — per challan/style/karigar: budget, expense on challan, attendance allocated by piece share, total paid  
3. **Payroll register** — all expense lines  
4. **Performance** — piece vs full payroll with CSV download  

**P&L Compare** tab also has CSV downloads per section.

## API

- `GET /api/stitching/reports/hub?date_from=&date_to=`
- `GET /api/stitching/reports/karigar-profitability`
- `GET /api/stitching/reports/challan-labour`
- `GET /api/stitching/payroll`
- `GET /api/stitching/performance`
- `GET /api/stitching/comparison-dashboard`

## Shared across users

Payroll and reports use the **server stitching database** (`stitching_costing.db`). Any user on the same server sees the same attendance, expenses, and production — reports update when anyone enters data for that day.
