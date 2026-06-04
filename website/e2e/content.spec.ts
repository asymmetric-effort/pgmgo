import { test, expect } from '@playwright/test';

test('docs page shows documentation sections', async ({ page }) => {
  await page.goto('/#/docs');
  await expect(page.locator('h2:has-text("Overview")')).toBeVisible();
  await expect(page.locator('h2:has-text("Getting Started")')).toBeVisible();
  await expect(page.locator('h2:has-text("CLI Reference")')).toBeVisible();
});

test('api page shows package types', async ({ page }) => {
  await page.goto('/#/api');
  await expect(page.locator('text=BayesianNetwork').first()).toBeVisible();
  await expect(page.locator('text=VariableElimination').first()).toBeVisible();
});
