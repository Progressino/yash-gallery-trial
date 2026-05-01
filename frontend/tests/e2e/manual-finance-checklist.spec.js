import crypto from "node:crypto";
import fs from "node:fs/promises";
import { test, expect } from "@playwright/test";

function createAuthToken(username = "qa-user") {
  const secret = process.env.JWT_SECRET || "change-me-set-jwt-secret-in-env";
  const header = Buffer.from(JSON.stringify({ alg: "HS256", typ: "JWT" })).toString("base64url");
  const exp = Math.floor(Date.now() / 1000) + 60 * 60;
  const payload = Buffer.from(JSON.stringify({ sub: username, exp })).toString("base64url");
  const body = `${header}.${payload}`;
  const sig = crypto.createHmac("sha256", secret).update(body).digest("base64url");
  return `${body}.${sig}`;
}

async function expectFinanceLoaded(page) {
  await expect(page.getByRole("heading", { name: "Finance", exact: true })).toBeVisible();
}

async function assertCsvHeadersFromDownload(page, clickTarget, expectedHeaders) {
  const dlPromise = page.waitForEvent("download");
  await clickTarget.click();
  const dl = await dlPromise;
  const filePath = await dl.path();
  expect(filePath).toBeTruthy();
  const content = await fs.readFile(filePath, "utf8");
  const firstLine = content.split(/\r?\n/)[0] || "";
  for (const h of expectedHeaders) expect(firstLine).toContain(`"${h}"`);
}

test("manual finance checklist across CRONUS menu paths", async ({ page, context, baseURL }) => {
  test.setTimeout(120_000);
  const appBase = baseURL || "http://127.0.0.1:4173";
  await context.addCookies([
    {
      name: "auth_token",
      value: createAuthToken(),
      url: appBase,
      httpOnly: true,
      sameSite: "Lax",
    },
  ]);

  await page.goto("/finance", { waitUntil: "domcontentloaded" });
  await expectFinanceLoaded(page);
  await expect(page.getByText("Finance workspace")).toBeVisible();

  // Quick-open row (tab-level coverage)
  const quickOpenExpectations = [
    ["Day Book", /Day Book —/],
    ["Sales Invoices", /Sales invoices \(auto-picked from uploaded sales\)/],
    ["Vouchers", /Saved Vouchers/],
    ["Voucher Register", /Export register \(CSV\)/],
    ["Cash Book", /Export Cash Book \(CSV\)/],
    ["Bank Book", /Export Bank Book \(CSV\)/],
    ["COA", /Chart of Accounts/],
    ["Trial Balance", /Search Ledger/],
    ["P&L", /Profit & Loss Statement/],
    ["Expenses", /Add Expense/],
    ["Revenue", /Gross vs Net Revenue by Platform/],
    ["Masters", /Ledger Groups/],
    ["Help", /Finance module — user guide/],
  ];

  for (const [btn, marker] of quickOpenExpectations) {
    await page.getByRole("button", { name: btn, exact: true }).first().click();
    await expect(page.getByText(marker)).toBeVisible();
  }

  // India taxation strip visibility and jumps
  await page.getByRole("button", { name: /India Taxation/i }).first().click();
  await page.getByRole("button", { name: "GSTR-3B (monthly)" }).click();
  await expect(page.getByRole("button", { name: "GST summary", exact: true })).toBeVisible();
  await expect(page.getByText(/GSTR-1: use government portal/i)).toBeVisible();
  await page.getByRole("button", { name: "GST summary", exact: true }).first().click();
  await expect(page.getByText(/Amazon MTR only/i)).toBeVisible();

  // CRONUS mega menus: representative path checks
  await page.getByRole("button", { name: /Finance/i }).first().click();
  await page.getByRole("button", { name: "Chart of Accounts" }).click();
  await expect(page.getByText(/Chart of Accounts/)).toBeVisible();

  await page.getByRole("button", { name: /Finance/i }).first().click();
  await page.getByRole("button", { name: "GST returns — GSTR-3B" }).click();
  await expect(page.getByText(/GSTR-3B —/)).toBeVisible();

  await page.getByRole("button", { name: /Cash Management/i }).first().click();
  await page.getByRole("button", { name: "Bank Book" }).first().click();
  await expect(page.getByText(/Export Bank Book \(CSV\)/)).toBeVisible();

  await page.getByRole("button", { name: /^Sales/i }).first().click();
  await page.getByRole("button", { name: "Sales invoices", exact: true }).first().click();
  await expect(page.getByText(/Sales invoices \(auto-picked from uploaded sales\)/)).toBeVisible();

  await page.getByRole("button", { name: /Purchasing/i }).first().click();
  await page.getByRole("button", { name: "Purchase invoices" }).click();
  await expect(page.getByText(/Saved Vouchers/)).toBeVisible();

  await page.getByRole("button", { name: /India Taxation/i }).first().click();
  await page.getByRole("button", { name: "TDS master" }).click();
  await expect(page.getByText(/TDS Sections/)).toBeVisible();

  await page.getByRole("button", { name: /Voucher Interface/i }).first().click();
  await page.getByRole("button", { name: "Cash receipt voucher" }).click();
  await expect(page.getByText(/New Receipt Voucher/)).toBeVisible();

  await page.getByRole("button", { name: /Voucher Interface/i }).first().click();
  await page.getByRole("button", { name: "Voucher register", exact: true }).first().click();
  await expect(page.getByText(/Export register \(CSV\)/)).toBeVisible();

  // Deeper assertions: row counts before/after filters (Voucher Register)
  const registerRows = page.locator("tbody tr");
  const beforeCount = await registerRows.count();
  const fromInputs = page.locator('input[type="date"]');
  await fromInputs.first().fill("2099-01-01");
  await fromInputs.nth(1).fill("2099-12-31");
  await expect(page.getByText(/No vouchers in this range\./)).toBeVisible();
  const afterCount = await registerRows.count();
  expect(afterCount).toBeLessThanOrEqual(beforeCount);

  // Deeper assertions: CSV content headers
  await assertCsvHeadersFromDownload(
    page,
    page.getByRole("button", { name: "Export register (CSV)" }),
    ["date", "voucher_no", "voucher_type", "party_name", "taxable", "cgst", "sgst", "igst", "net_payable"],
  );

  await page.getByRole("button", { name: /India Taxation/i }).first().click();
  await page.getByRole("button", { name: "GSTR-3B (monthly)" }).click();
  await assertCsvHeadersFromDownload(
    page,
    page.getByRole("button", { name: "Export breakdown (CSV)" }),
    ["voucher_no", "voucher_date", "voucher_type", "party_name", "taxable_amount", "cgst", "sgst", "igst", "total_amount"],
  );

  // Deeper assertions: voucher preset value is actually set in form
  await page.getByRole("button", { name: /Voucher Interface/i }).first().click();
  await page.getByRole("button", { name: "Cash receipt voucher" }).click();
  const voucherTypeSelect = page.locator('label:has-text("Voucher Type")').locator('xpath=following::select[1]');
  await expect(voucherTypeSelect).toHaveValue("Receipt");

  await page.getByRole("button", { name: /Voucher Interface/i }).first().click();
  await page.getByRole("button", { name: "Journal voucher" }).click();
  await expect(voucherTypeSelect).toHaveValue("Journal");
});

