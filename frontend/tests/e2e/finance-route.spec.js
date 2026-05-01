import { test, expect } from "@playwright/test";

test("finance route sends unauthenticated users to login", async ({ page }) => {
  await page.goto("/finance", { waitUntil: "domcontentloaded" });
  await expect(page).toHaveURL(/\/login$/);
});
