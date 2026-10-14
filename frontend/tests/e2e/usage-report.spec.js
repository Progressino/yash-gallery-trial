import { test, expect } from "@playwright/test";

// Needs a running backend and two accounts:
//   E2E_USAGE_ADMIN_USER / E2E_USAGE_ADMIN_PASS  (role Admin or Super Admin)
//   E2E_USAGE_MANAGER_USER / E2E_USAGE_MANAGER_PASS  (role Manager)
const ADMIN = [process.env.E2E_USAGE_ADMIN_USER, process.env.E2E_USAGE_ADMIN_PASS];
const MANAGER = [process.env.E2E_USAGE_MANAGER_USER, process.env.E2E_USAGE_MANAGER_PASS];

async function login(page, [username, password]) {
  const res = await page.request.post("/api/auth/login", { data: { username, password } });
  expect(res.ok()).toBeTruthy();
  expect((await res.json()).ok).toBe(true);
}

test.describe("Admin usage report", () => {
  test.describe.configure({ timeout: 120_000 });

  test("admin sees sidebar link + header button and the report loads", async ({ page }) => {
    test.skip(!ADMIN[0], "E2E_USAGE_ADMIN_* credentials not set");
    await login(page, ADMIN);
    await page.goto("/admin", { waitUntil: "domcontentloaded" });

    const nav = page.getByRole("navigation", { name: "Main navigation" });
    await expect(nav.getByRole("link", { name: "Usage Report" })).toBeVisible({ timeout: 30_000 });
    const headerBtn = page.getByRole("link", { name: /User usage report/ });
    await expect(headerBtn).toBeVisible();

    await headerBtn.click();
    await expect(page).toHaveURL(/\/admin\/usage$/);
    await expect(page.getByRole("heading", { name: "User Usage Report" })).toBeVisible({ timeout: 60_000 });
    await expect(page.getByText("Active today", { exact: true })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole("heading", { name: "Most used modules" })).toBeVisible();

    const api = await page.request.get("/api/admin/usage?days=7");
    expect(api.status()).toBe(200);
    const body = await api.json();
    expect(body.summary.active_today).toBeGreaterThanOrEqual(1);
  });

  test("manager does not see the report", async ({ page }) => {
    test.skip(!MANAGER[0], "E2E_USAGE_MANAGER_* credentials not set");
    await login(page, MANAGER);
    await page.goto("/admin", { waitUntil: "domcontentloaded" });

    const nav = page.getByRole("navigation", { name: "Main navigation" });
    await expect(nav.getByRole("link", { name: "Admin", exact: true })).toBeVisible({ timeout: 30_000 });
    await expect(nav.getByRole("link", { name: "Usage Report" })).toHaveCount(0);
    await expect(page.getByRole("link", { name: /User usage report/ })).toHaveCount(0);

    await page.goto("/admin/usage", { waitUntil: "domcontentloaded" });
    await expect(page.getByText("Usage report is available to Super Admin and Admin only.")).toBeVisible({ timeout: 60_000 });
    const api = await page.request.get("/api/admin/usage");
    expect(api.status()).toBe(403);
  });
});
