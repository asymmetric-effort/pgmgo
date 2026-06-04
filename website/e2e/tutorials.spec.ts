import { test, expect } from '@playwright/test';

test('tutorials page has no broken internal links', async ({ page }) => {
  await page.goto('/#/tutorials');
  await page.waitForSelector('h1');

  // Get all links on the page
  const links = await page.$$eval('a[href]', els =>
    els.map(el => ({ href: el.getAttribute('href'), text: el.textContent }))
  );

  for (const link of links) {
    if (!link.href) continue;

    // Internal hash links (within the tutorials page)
    if (link.href.startsWith('#tutorial-')) {
      // These are anchor links to sections on the same page
      const id = link.href.slice(1);
      const target = await page.$(`#${id}`);
      expect(target, `Anchor target missing: ${link.href} (${link.text})`).toBeTruthy();
    }

    // External links -- verify HTTP response is not a client/server error
    // Allow 429 (rate limiting) since external sites may throttle automated requests
    if (link.href.startsWith('http')) {
      const response = await page.request.get(link.href);
      const status = response.status();
      const ok = status < 400 || status === 429;
      expect(ok, `Broken link: ${link.href} (${link.text}) returned ${status}`).toBeTruthy();
    }
  }
});

test('each tutorial section has content', async ({ page }) => {
  await page.goto('/#/tutorials');

  const tutorials = [
    'Building Your First Bayesian Network',
    'Probabilistic Inference',
    'Learning from Data',
    'Causal Inference',
    'Working with File Formats',
    'Sampling and Monte Carlo',
    'Advanced Models',
    'Using the Internal Libraries',
  ];

  for (const title of tutorials) {
    await expect(page.locator(`text=${title}`).first()).toBeVisible();
  }
});

test('tutorial code examples are present', async ({ page }) => {
  await page.goto('/#/tutorials');

  // Verify code blocks exist
  const codeBlocks = await page.$$('pre code');
  expect(codeBlocks.length).toBeGreaterThan(10);
});
