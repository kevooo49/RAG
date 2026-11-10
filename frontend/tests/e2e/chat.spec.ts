import { test, expect, type Page } from '@playwright/test';

const frame = (type: string, content: unknown) => `data: ${JSON.stringify({ type, content })}\n\n`;
const DONE = 'data: [DONE]\n\n';
const sse = (...parts: string[]) => parts.join('') + DONE;

const SOURCES = [
  { id: 1, text: 'Akademik kosztuje 500 zł miesięcznie.', score: 0.0327,
    meta: { title: 'Cennik DS', section: 'Ceny', source: 'Miasteczko AGH', url: 'https://www.miasteczko.agh.edu.pl/cennik' } },
  { id: 2, text: 'Opłaty wnosi się do 10. dnia miesiąca.', score: 0.0164,
    meta: { title: 'Regulamin', section: 'Opłaty', source: 'Miasteczko AGH', url: '' } },
];

type Reply = { body?: string; status?: number; json?: unknown; delayMs?: number };

async function mockChat(page: Page, reply: (body: any, index: number) => Reply) {
  const requests: any[] = [];
  await page.route('**/api/chat', async (route) => {
    const body = route.request().postDataJSON();
    requests.push(body);
    const r = reply(body, requests.length - 1);
    if (r.delayMs) await new Promise((resolve) => setTimeout(resolve, r.delayMs));
    await route.fulfill(r.json !== undefined
      ? { status: r.status ?? 500, contentType: 'application/json', body: JSON.stringify(r.json) }
      : { status: r.status ?? 200, contentType: 'text/event-stream; charset=utf-8', headers: { 'cache-control': 'no-cache' }, body: r.body ?? DONE });
  });
  return requests;
}

const ui = (page: Page) => ({
  input: page.locator('#message-input'),
  send: page.locator('#send-button'),
  loading: page.locator('#loading-indicator'),
  user: page.locator('.message.user'),
  assistant: page.locator('.message.assistant'),
  cards: page.locator('.source-card'),
  toggle: page.locator('.sources-toggle'),
});

async function ask(page: Page, text: string) {
  const { input, send } = ui(page);
  await input.fill(text);
  await send.click();
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => { sessionStorage.clear(); localStorage.clear(); });
  await page.reload();
});

test('shows the greeting and an enabled input on load', async ({ page }) => {
  const { assistant, input, send, loading } = ui(page);
  await expect(assistant).toHaveCount(1);
  await expect(assistant.first()).toContainText('Jestem asystentem Akademii Górniczo-Hutniczej');
  await expect(input).toBeEnabled();
  await expect(send).toBeEnabled();
  await expect(loading).toBeHidden();
});

test('full cycle: question, loading state, streamed answer, citations and sources', async ({ page }) => {
  await mockChat(page, () => ({
    delayMs: 400,
    body: sse(frame('status', 'Przeszukuję bazę wiedzy AGH...'), frame('sources', SOURCES),
              frame('token', 'Akademik kosztuje '), frame('token', '**500 zł** [1].')),
  }));
  const { input, send, loading, user, assistant, cards, toggle } = ui(page);

  await ask(page, 'Ile kosztuje akademik?');

  await expect(user.last()).toHaveText('Ile kosztuje akademik?');
  await expect(input).toBeDisabled();
  await expect(send).toBeDisabled();
  await expect(loading).toBeVisible();

  await expect(loading).toBeHidden();
  await expect(assistant).toHaveCount(2);
  await expect(assistant.last()).toContainText('Akademik kosztuje 500 zł [1].');
  await expect(assistant.last().locator('strong')).toHaveText('500 zł');
  await expect(assistant.last().locator('.citation')).toHaveText('[1]');

  await expect(input).toBeEnabled();
  await expect(input).toBeFocused();
  await expect(input).toHaveValue('');

  await expect(cards).toHaveCount(2);
  await expect(toggle).toContainText('(2)');
  await expect(cards.first()).toBeHidden();
  await toggle.click();
  await expect(cards.first()).toBeVisible();
  await expect(cards.first()).toContainText('Cennik DS');
  await expect(cards.first().locator('a')).toHaveAttribute('href', 'https://www.miasteczko.agh.edu.pl/cennik');
  await expect(cards.first().locator('a')).toHaveAttribute('target', '_blank');
  await toggle.click();
  await expect(cards.first()).toBeHidden();
});

test('sends the question with the selected model and the earlier conversation', async ({ page }) => {
  const requests = await mockChat(page, (_body, i) => ({ body: sse(frame('token', `Odpowiedź ${i + 1}`)) }));
  const { input, assistant } = ui(page);

  await page.locator('#model-switcher').selectOption('bielik');
  await ask(page, 'Pierwsze pytanie');
  await expect(assistant).toHaveCount(2);
  await expect(input).toBeEnabled();
  await ask(page, 'Drugie pytanie');
  await expect(assistant).toHaveCount(3);

  expect(requests).toHaveLength(2);
  expect(requests[0]).toMatchObject({ query: 'Pierwsze pytanie', model: 'bielik' });
  expect(requests[1].query).toBe('Drugie pytanie');
  expect(requests[1].model).toBe('bielik');
  expect(requests[1].history.map((m: any) => [m.role, m.content]).slice(1)).toEqual([
    ['user', 'Pierwsze pytanie'], ['assistant', 'Odpowiedź 1'],
  ]);
});

test('submits with the Enter key, and an empty message sends nothing', async ({ page }) => {
  const requests = await mockChat(page, () => ({ body: sse(frame('token', 'ok')) }));
  const { input, assistant } = ui(page);

  await input.press('Enter');
  await expect(assistant).toHaveCount(1);
  expect(requests).toHaveLength(0);

  await input.fill('Cześć');
  await input.press('Enter');
  await expect(assistant).toHaveCount(2);
  expect(requests).toHaveLength(1);
});

test('shows the backend message when rate-limited, then recovers', async ({ page }) => {
  await mockChat(page, (_b, i) => i === 0
    ? { status: 429, json: { detail: 'Zbyt wiele zapytań. Odczekaj chwilę przed zadaniem kolejnego pytania.' } }
    : { body: sse(frame('token', 'Już działa')) });
  const { input, assistant } = ui(page);

  await ask(page, 'Pytanie');
  await expect(page.locator('.error-msg')).toHaveText('Zbyt wiele zapytań. Odczekaj chwilę przed zadaniem kolejnego pytania.');
  await expect(input).toBeEnabled();

  await ask(page, 'Jeszcze raz');
  await expect(assistant.last()).toContainText('Już działa');
});

test('shows a message when the server cannot be reached', async ({ page }) => {
  await page.route('**/api/chat', (route) => route.abort('connectionrefused'));
  await ask(page, 'Pytanie');
  await expect(page.locator('.error-msg')).toBeVisible();
  await expect(ui(page).input).toBeEnabled();
});

test('keeps the conversation after a page reload', async ({ page }) => {
  await mockChat(page, () => ({ body: sse(frame('token', 'Zapamiętana odpowiedź')) }));
  await ask(page, 'Zapamiętane pytanie');
  await expect(ui(page).assistant).toHaveCount(2);
  await expect(ui(page).input).toBeEnabled();

  await page.reload();
  await expect(ui(page).user).toHaveText('Zapamiętane pytanie');
  await expect(ui(page).assistant.last()).toContainText('Zapamiętana odpowiedź');
});

test('recovers from corrupted saved data instead of showing a blank page', async ({ page }) => {
  for (const junk of ['{broken', 'null', '{"a":1}']) {
    await page.evaluate((v) => sessionStorage.setItem('chatHistory', v), junk);
    await page.reload();
    await expect(ui(page).assistant.first()).toContainText('Jestem asystentem');
    await expect(ui(page).input).toBeEnabled();
  }
});

test('never executes scripts from questions, answers, errors or sources', async ({ page }) => {
  const dialogs: string[] = [];
  page.on('dialog', async (d) => { dialogs.push(d.message()); await d.dismiss(); });
  const attack = '<img src=x onerror="window.__xss=1"><svg onload="window.__xss=2"><script>window.__xss=3</script>';

  await mockChat(page, (_b, i) => i === 0
    ? { body: sse(frame('sources', [{ id: 1, text: attack, score: 0.03, meta: { title: attack, section: attack, url: 'javascript:window.__xss=4' } }]),
                  frame('token', `${attack} odpowiedź`)) }
    : { status: 500, json: { detail: attack } });

  await ask(page, attack);
  await expect(ui(page).assistant).toHaveCount(2);
  await expect(ui(page).input).toBeEnabled();
  await ui(page).toggle.click();
  await ask(page, 'Drugie');
  await expect(page.locator('.error-msg')).toBeVisible();

  await page.waitForTimeout(300);
  expect(await page.evaluate(() => (window as any).__xss)).toBeUndefined();
  expect(dialogs).toEqual([]);
  await expect(page.locator('.message-content img, .message-content svg, .source-card img, .source-card svg, #chat-messages script')).toHaveCount(0);
  for (const a of await page.locator('.source-card a').all()) {
    expect(await a.getAttribute('href')).not.toMatch(/^javascript:/i);
  }
});

test('stays usable on a phone-sized screen', async ({ page }) => {
  await mockChat(page, () => ({ body: sse(frame('sources', SOURCES), frame('token', 'Odpowiedź '.repeat(40))) }));
  await page.setViewportSize({ width: 375, height: 667 });
  await ask(page, 'Pytanie');
  await expect(ui(page).assistant).toHaveCount(2);
  await expect(ui(page).input).toBeVisible();
  await expect(ui(page).send).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

// ---------------------------------------------------------------------------------------------------------
// Optional: one smoke test against a REAL backend (docker compose up) with real retrieval and a real model.
// Costs money and counts against the rate limit, so it only runs on request:  E2E_LIVE=1 npx playwright test
// ---------------------------------------------------------------------------------------------------------
test.describe('live backend', () => {
  test.skip(!process.env.E2E_LIVE, 'set E2E_LIVE=1 to run against a real backend');

  test('answers a real question and shows sources', async ({ page }) => {
    test.setTimeout(60_000);
    await ask(page, 'Jakie są warunki przyznania stypendium rektora na AGH?');
    await expect(ui(page).loading).toBeHidden({ timeout: 30_000 });
    await expect(ui(page).assistant).toHaveCount(2);
    await expect(ui(page).input).toBeEnabled({ timeout: 40_000 });
    await expect(ui(page).cards.first()).toBeAttached();
  });
});
