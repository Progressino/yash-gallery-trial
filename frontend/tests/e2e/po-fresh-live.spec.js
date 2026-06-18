import { test, expect } from "@playwright/test";

async function loginIfNeeded(page) {
  const welcome = page.getByRole("heading", { name: "Welcome back" });
  if (!(await welcome.isVisible().catch(() => false))) return;
  const username = process.env.E2E_AUTH_USERNAME || process.env.AUTH_USERNAME || "admin";
  const password = process.env.E2E_AUTH_PASSWORD || process.env.AUTH_PASSWORD || "";
  if (!password) throw new Error("Set E2E_AUTH_PASSWORD or AUTH_PASSWORD for live PO test");
  await page.getByPlaceholder("admin").fill(username);
  await page.getByPlaceholder("••••••••").fill(password);
  await page.getByRole("button", { name: "Sign In" }).click();
  await page.waitForURL((url) => !url.pathname.endsWith("/login"), { timeout: 45_000 });
  await page.goto("/po-fresh", { waitUntil: "domcontentloaded" });
}

test("PO Fresh loads after login without React depth error", async ({ page }) => {
  const pageErrors = [];
  page.on("pageerror", (err) => pageErrors.push(String(err?.message || err)));

  await page.goto("/po-fresh", { waitUntil: "domcontentloaded" });
  await loginIfNeeded(page);

  await expect(page.getByTestId("po-fresh-root")).toBeVisible({ timeout: 45_000 });
  await expect(page.getByRole("heading", { name: /PO Engine \(Fresh\)/ })).toBeVisible();
  await expect(page.getByRole("button", { name: "Calculate PO" })).toBeVisible();

  const depthErrors = pageErrors.filter((m) => /maximum update depth/i.test(m));
  expect(depthErrors).toEqual([]);
});
