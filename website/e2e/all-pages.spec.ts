import { test, expect } from '@playwright/test';

const pages = ['/', '/#/docs', '/#/tutorials', '/#/api', '/#/cli'];

for (const pagePath of pages) {
  test(`${pagePath} has no broken links`, async ({ page }) => {
    await page.goto(pagePath);
    await page.waitForLoadState('networkidle');

    // Get all links
    const links = await page.$$eval('a[href]', els =>
      els.map(el => el.getAttribute('href')).filter(Boolean)
    );

    // Check internal links resolve
    for (const href of links) {
      if (!href) continue;
      if (href.startsWith('#/') || href.startsWith('/#/')) {
        // Navigate and verify page loads
        const url = href.startsWith('#') ? '/' + href : href;
        await page.goto(url);
        const h1 = await page.$('h1');
        expect(h1, `Internal link ${href} leads to page with no h1`).toBeTruthy();
        // Go back
        await page.goto(pagePath);
        await page.waitForLoadState('networkidle');
      }
    }
  });
}
