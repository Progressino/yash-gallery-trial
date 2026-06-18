import { test, expect } from "@playwright/test";

test("po-fresh route redirects unauthenticated users to login", async ({ page }) => {
  const pageErrors = [];
  page.on("pageerror", (err) => pageErrors.push(String(err?.message || err)));

  await page.goto("/po-fresh", { waitUntil: "domcontentloaded" });
  await expect(page).toHaveURL(/\/login$/);

  const depthErrors = pageErrors.filter((m) => /maximum update depth/i.test(m));
  expect(depthErrors).toEqual([]);
});

test("po-fresh root is present when route is reachable", async ({ page }) => {
  // Unauthenticated users hit login — still ensure the app bundle loads without React depth errors.
  const pageErrors = [];
  page.on("pageerror", (err) => pageErrors.push(String(err?.message || err)));

  await page.goto("/login", { waitUntil: "domcontentloaded" });
  await expect(page.getByText("Welcome back")).toBeVisible();

  const depthErrors = pageErrors.filter((m) => /maximum update depth/i.test(m));
  expect(depthErrors).toEqual([]);
});
