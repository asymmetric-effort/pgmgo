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

    // ScrollLink anchor links use javascript:void(0) and onClick to scroll.
    // Verify their target elements exist by checking the onClick handler's target id.
    if (link.href === 'javascript:void(0)') {
      // ScrollLink rendered -- the target id is embedded in the onClick handler.
      // We verify targets separately below.
      continue;
    }

    // Legacy hash links (should no longer appear, but check just in case)
    if (link.href.startsWith('#tutorial-')) {
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

  // Verify all ScrollLink targets exist on the page
  const tutorialIds = [
    'tutorial-1', 'tutorial-2', 'tutorial-3', 'tutorial-4',
    'tutorial-5', 'tutorial-6', 'tutorial-7', 'tutorial-8',
  ];
  for (const id of tutorialIds) {
    const target = await page.$(`#${id}`);
    expect(target, `ScrollLink target missing: #${id}`).toBeTruthy();
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

test('ScrollLink clicks scroll target section into view', async ({ page }) => {
  await page.goto('/#/tutorials');
  await page.waitForSelector('h1');

  // Click the ScrollLink for Tutorial 8 (last tutorial, should require scrolling)
  const scrollLink = page.locator('nav.page-toc a[href="javascript:void(0)"]').filter({ hasText: '8. Internal Libraries' });
  await scrollLink.click();

  // Wait briefly for smooth scroll
  await page.waitForTimeout(500);

  // Verify the target section is now visible in the viewport
  const section = page.locator('#tutorial-8');
  await expect(section).toBeInViewport();
});
