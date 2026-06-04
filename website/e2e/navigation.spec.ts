import { test, expect } from '@playwright/test';

test('home page loads', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('h1')).toContainText('pgmgo');
});

test('docs page loads via nav', async ({ page }) => {
  await page.goto('/');
  await page.click('a:has-text("Docs")');
  await expect(page.locator('h1')).toContainText('Documentation');
  // Verify Home content is NOT showing
  await expect(page.locator('.hero')).not.toBeVisible();
});

test('tutorials page loads via nav', async ({ page }) => {
  await page.goto('/');
  await page.click('a:has-text("Tutorials")');
  await expect(page.locator('h1')).toContainText('Tutorials');
  await expect(page.locator('.hero')).not.toBeVisible();
});

test('api page loads via nav', async ({ page }) => {
  await page.goto('/');
  await page.click('a:has-text("API")');
  await expect(page.locator('h1')).toContainText('API Reference');
  await expect(page.locator('.hero')).not.toBeVisible();
});
