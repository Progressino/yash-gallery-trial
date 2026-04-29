import { test, expect } from "@playwright/test";

test("login screen loads in chromium", async ({ page }) => {
  await page.goto("/login", { waitUntil: "domcontentloaded" });
  await expect(page.getByText("Welcome back")).toBeVisible();
  await expect(page.getByPlaceholder("admin")).toBeVisible();
  await expect(page.getByPlaceholder("••••••••")).toBeVisible();
  await expect(page.getByRole("button", { name: "Sign In" })).toBeVisible();
});
