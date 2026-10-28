import { test, expect } from '@playwright/test';

test('Pełny cykl zapytania RAG z renderowaniem źródeł', async ({ page }) => {
  // Wydłużamy limit czasu na ten jeden test, bo LLM czasem myśli kilkanaście sekund
  test.setTimeout(45000);

  // 1. Użytkownik wchodzi na stronę
  await page.goto('/');

  // 2. Weryfikacja, czy strona się załadowała i bot wyświetlił wiadomość powitalną
  await expect(page.locator('.message.assistant')).toBeVisible();

  // 3. Wpisanie pytania i wysłanie go formularzem
  const input = page.locator('#message-input');
  await input.fill('Jakie są warunki przyznania stypendium rektora na AGH?');
  
  const sendBtn = page.locator('#send-button');
  await sendBtn.click();

  // 4. Weryfikacja, czy pole input zablokowało się po wysłaniu (zapobieganie spamowi)
  await expect(input).toBeDisabled();

  // 5. Weryfikacja czy kółko ładowania (loader) pokazało się na ekranie
  const loadingIndicator = page.locator('#loading-indicator');
  await expect(loadingIndicator).toBeVisible();

  // 6. Oczekiwanie na finalną odpowiedź. 
  // Nasz kod w main.ts ukrywa loadingIndicator, gdy spłynie pierwszy event z SSE
  await expect(loadingIndicator).toBeHidden({ timeout: 30000 });
  
  // 7. Sprawdzamy, czy na ekranie są teraz DWA dymki asystenta
  const botMessages = page.locator('.message.assistant');
  await expect(botMessages).toHaveCount(2);

  // 8. Oczekujemy na zakończenie całego strumienia SSE!
  // Wiemy, że się skończył, gdy nasz main.ts odblokowuje pole tekstowe.
  await expect(input).toBeEnabled({ timeout: 40000 });

  // 9. Sprawdzamy źródła
  // Używamy toBeAttached(), ponieważ w przypadku zamkniętego akordeonu <details>
  // lub animacji, Playwright mógłby uznać element za tymczasowo "niewidoczny".
  const sources = page.locator('.source-card');
  await expect(sources.first()).toBeAttached();
});