import { test, expect } from '@playwright/test';

test('Full RAG query cycle with source rendering', async ({ page }) => {
  test.setTimeout(45000);

  await page.goto('/');

  await expect(page.locator('.message.assistant')).toBeVisible();

  const input = page.locator('#message-input');
  await input.fill('Jakie są warunki przyznania stypendium rektora na AGH?');
  
  const sendBtn = page.locator('#send-button');
  await sendBtn.click();

  await expect(input).toBeDisabled();

  const loadingIndicator = page.locator('#loading-indicator');
  await expect(loadingIndicator).toBeVisible();

  await expect(loadingIndicator).toBeHidden({ timeout: 30000 });
  
  const botMessages = page.locator('.message.assistant');
  await expect(botMessages).toHaveCount(2);

  await expect(input).toBeEnabled({ timeout: 40000 });

  const sources = page.locator('.source-card');
  await expect(sources.first()).toBeAttached();
});