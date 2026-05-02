import { defineConfig } from "@playwright/test";

const _pw = process.env.PW_WORKERS;
const pwWorkers =
  _pw && /^\d+$/.test(String(_pw).trim()) ? Number.parseInt(String(_pw).trim(), 10) : undefined;

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 45_000,
  retries: process.env.CI ? 1 : 0,
  // Self-hosted runners often OOM (exit 137) if Playwright spawns many Chromiums.
  workers: pwWorkers !== undefined && pwWorkers > 0 ? pwWorkers : undefined,
  expect: { timeout: 10_000 },
  use: {
    baseURL: process.env.E2E_BASE_URL || "http://127.0.0.1:4173",
    headless: true,
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run preview -- --host 127.0.0.1 --port 4173",
    url: "http://127.0.0.1:4173/login",
    reuseExistingServer: true,
    timeout: 120_000,
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
