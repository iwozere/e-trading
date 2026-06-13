/**
 * E2E smoke test: alert create → list → delete flow
 *
 * Prerequisites:
 *   npm install --save-dev @playwright/test
 *   npx playwright install chromium
 *   npm run dev   (Vite dev server on :5173)
 *   The backend API must be running with a seeded admin user.
 */

import { test, expect, Page } from '@playwright/test';

const ADMIN_USER = process.env.E2E_ADMIN_USER || 'admin';
const ADMIN_PASS = process.env.E2E_ADMIN_PASS || 'admin';

async function login(page: Page) {
  await page.goto('/login');
  await page.fill('input[name="username"], input[type="text"]', ADMIN_USER);
  await page.fill('input[name="password"], input[type="password"]', ADMIN_PASS);
  await page.click('button[type="submit"]');
  await page.waitForURL('**/dashboard', { timeout: 10_000 });
}

test.describe('Alert Management — smoke tests', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('navigates to alert management page', async ({ page }) => {
    await page.goto('/telegram/alerts');
    await expect(page.getByRole('heading', { name: /Alert Management/i })).toBeVisible();
  });

  test('create complex alert via ConfigBuilder → appears in list → delete', async ({ page }) => {
    await page.goto('/telegram/alerts');

    // Open ConfigBuilder
    await page.getByRole('button', { name: /Create Complex Alert/i }).click();
    await expect(page.getByText(/Logic Engine Builder/i)).toBeVisible();

    // Step through the builder (default state is valid)
    const nextBtn = page.getByRole('button', { name: /Next|Finalize/i }).first();
    if (await nextBtn.isVisible()) {
      await nextBtn.click();
    }

    // Click Save (the builder's final save button)
    const saveBtn = page.getByRole('button', { name: /Save|Generate/i }).first();
    await saveBtn.click();

    // Dialog should close and a success toast should appear
    await expect(page.getByText(/created successfully/i)).toBeVisible({ timeout: 8_000 });

    // The new alert should appear in the table
    await expect(page.locator('table tbody tr').first()).toBeVisible({ timeout: 5_000 });

    // Delete the first alert in the list
    const deleteBtn = page.locator('table tbody tr').first().getByRole('button').last();
    await deleteBtn.click();

    // Confirm deletion toast
    await expect(page.getByText(/deleted successfully/i)).toBeVisible({ timeout: 8_000 });
  });

  test('toggle alert active status', async ({ page }) => {
    await page.goto('/telegram/alerts');

    // If there are no alerts the toggle test is skipped
    const firstRow = page.locator('table tbody tr').first();
    const noAlerts = page.getByText(/No alerts found/i);

    if (await noAlerts.isVisible({ timeout: 3_000 }).catch(() => false)) {
      test.skip();
      return;
    }

    // Get current status chip text
    const statusChip = firstRow.locator('[class*="MuiChip"]').first();
    const initialStatus = await statusChip.textContent();

    // Click the toggle button (first icon button in actions column)
    await firstRow.getByRole('button').first().click();

    // Status chip should change
    await expect(page.getByText(/activated|deactivated/i)).toBeVisible({ timeout: 8_000 });
    const newStatus = await statusChip.textContent();
    expect(newStatus).not.toBe(initialStatus);
  });
});
