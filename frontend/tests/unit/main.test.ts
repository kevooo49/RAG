import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
// @ts-ignore
import indexHtml from '../../index.html?raw';
import { DONE, controlledStream, errorResponse, frame, okResponse, sseResponse } from '../helpers/sse';

const STORAGE_KEY = 'chatHistory';
const MODEL_KEY = 'rag_selected_model';
const MODELS = ['gpt-4o', 'gpt-4o-mini', 'bielik'];

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const bodyMarkup = () => indexHtml
  .replace(/^[\s\S]*?<body[^>]*>/i, '')
  .replace(/<\/body>[\s\S]*$/i, '')
  .replace(/<script[\s\S]*?<\/script>/gi, '')
  .replace(/<img[^>]*>/gi, '');

async function mountApp(opts: { session?: string; model?: string } = {}) {
  vi.resetModules();
  document.body.innerHTML = bodyMarkup();
  sessionStorage.clear();
  localStorage.clear();
  if (opts.session !== undefined) sessionStorage.setItem(STORAGE_KEY, opts.session);
  if (opts.model !== undefined) localStorage.setItem(MODEL_KEY, opts.model);
  await import('../../src/main');
}

const $ = <T extends Element = HTMLElement>(selector: string) => document.querySelector(selector) as T;
const input = () => $<HTMLInputElement>('#message-input');
const sendButton = () => $<HTMLButtonElement>('#send-button');
const loading = () => $('#loading-indicator');
const loadingText = () => $('#loading-text').textContent;
const bubbles = () => Array.from(document.querySelectorAll<HTMLElement>('.message'));
const bubbleText = (el: Element) => el.querySelector('.message-content')!.textContent!;
const lastAssistant = () => Array.from(document.querySelectorAll<HTMLElement>('.message.assistant')).pop()!;
const stored = () => JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? 'null');
const sentRequest = (call = 0) => JSON.parse(fetchMock.mock.calls[call][1].body);

function ask(text: string) {
  input().value = text;
  $<HTMLFormElement>('#chat-form').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
}
const settled = () => vi.waitFor(() => expect(input().disabled).toBe(false));
const answerWith = (...frames: string[]) => fetchMock.mockResolvedValue(sseResponse([...frames, DONE]));

async function askAndWait(text: string) {
  ask(text);
  await settled();
}

const SOURCES = [
  { id: 1, text: 'Akademik kosztuje 500 zł.', score: 0.0327, meta: { title: 'Cennik DS', section: 'Ceny', url: 'https://miasteczko.agh.edu.pl/cennik' } },
  { id: 2, text: 'Opłaty za pokój.', score: 0.0164, meta: { title: 'Regulamin', section: '', url: '' } },
];

const longConversation = (messages: number, assistantLength = 40) => JSON.stringify(
  Array.from({ length: messages }, (_, i) => ({
    id: String(1000 + i), role: i % 2 === 0 ? 'assistant' : 'user',
    content: i % 2 === 0 ? 'A'.repeat(assistantLength) : `Pytanie ${i}`,
  })));

describe('first render', () => {

  it('greets the user and is ready for input', async () => {
    await mountApp();
    expect(bubbles()).toHaveLength(1);
    expect(bubbles()[0].classList.contains('assistant')).toBe(true);
    expect(bubbleText(bubbles()[0])).toContain('Cześć! Jestem asystentem Akademii Górniczo-Hutniczej');
    expect(input().disabled).toBe(false);
    expect(sendButton().disabled).toBe(false);
    expect(loading().classList.contains('hidden')).toBe(true);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('offers the three models', async () => {
    await mountApp();
    const options = Array.from($<HTMLSelectElement>('#model-switcher').options).map((o) => o.value);
    expect(options).toEqual(MODELS);
  });
});

describe('asking a question', () => {

  it('shows the question at once, blocks input while waiting, then streams the answer into one bubble', async () => {
    await mountApp();
    const stream = controlledStream();
    fetchMock.mockResolvedValue(okResponse(stream.body));

    ask('Ile kosztuje akademik?');

    expect(bubbles()).toHaveLength(2);
    expect(bubbles()[1].classList.contains('user')).toBe(true);
    expect(bubbleText(bubbles()[1])).toBe('Ile kosztuje akademik?');
    expect(input().value).toBe('');
    expect(input().disabled).toBe(true);
    expect(sendButton().disabled).toBe(true);
    expect(loading().classList.contains('hidden')).toBe(false);
    expect(loadingText()).toBe('Analiza zapytania...');

    stream.push(frame('status', 'Przeszukuję bazę wiedzy AGH...'));
    await vi.waitFor(() => expect(loadingText()).toBe('Przeszukuję bazę wiedzy AGH...'));
    expect(bubbles()).toHaveLength(2);

    stream.push(frame('token', 'Od '));
    await vi.waitFor(() => expect(bubbles()).toHaveLength(3));
    expect(loading().classList.contains('hidden')).toBe(true);
    expect(bubbleText(lastAssistant())).toBe('Od ');
    expect(input().disabled).toBe(true);

    stream.push(frame('token', '500 zł'));
    await vi.waitFor(() => expect(bubbleText(lastAssistant())).toBe('Od 500 zł'));
    expect(bubbles()).toHaveLength(3);

    stream.push(DONE);
    stream.close();
    await settled();
    expect(sendButton().disabled).toBe(false);
    expect(document.activeElement).toBe(input());
    expect(input().value).toBe('');
  });

  it('renders markdown in the answer: bold, list items and citation markers', async () => {
    await mountApp();
    answerWith(frame('token', 'Wymagane:\n- **Dowód** [1]\n- Zdjęcie [2]'));
    await askAndWait('Co jest potrzebne?');
    const answer = lastAssistant();
    expect(answer.querySelectorAll('li')).toHaveLength(2);
    expect(answer.querySelector('strong')!.textContent).toBe('Dowód');
    expect(Array.from(answer.querySelectorAll('.citation')).map((c) => c.textContent)).toEqual(['[1]', '[2]']);
  });

  it('shows the sources under the answer, with a working toggle', async () => {
    await mountApp();
    answerWith(frame('sources', SOURCES), frame('token', 'Od 500 zł [1]'));
    await askAndWait('Ile kosztuje akademik?');

    const box = lastAssistant().querySelector('.sources-container') as HTMLElement;
    expect(box).not.toBeNull();
    expect(box.querySelectorAll('.source-card')).toHaveLength(2);
    expect(box.querySelector('.sources-toggle')!.textContent).toContain('(2)');
    expect(box.querySelector('.source-title')!.textContent).toBe('#1 Cennik DS');

    const list = box.querySelector('.sources-list')!;
    expect(list.classList.contains('open')).toBe(false);
    (box.querySelector('.sources-toggle') as HTMLElement).click();
    expect(list.classList.contains('open')).toBe(true);
  });

  it('ignores empty and whitespace-only messages', async () => {
    await mountApp();
    for (const text of ['', '   ', '\n\t ']) ask(text);
    await Promise.resolve();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(bubbles()).toHaveLength(1);
    expect(input().disabled).toBe(false);
  });

  it('trims the question in the bubble and in the request', async () => {
    await mountApp();
    answerWith(frame('token', 'ok'));
    await askAndWait('   Ile kosztuje akademik?   ');
    expect(bubbleText(bubbles()[1])).toBe('Ile kosztuje akademik?');
    expect(sentRequest().query).toBe('Ile kosztuje akademik?');
  });

  it('can ask several questions in a row, each answer in its own bubble', async () => {
    await mountApp();
    answerWith(frame('token', 'Odpowiedź 1'));
    await askAndWait('Pytanie 1');
    answerWith(frame('token', 'Odpowiedź 2'));
    await askAndWait('Pytanie 2');
    expect(bubbles().map(bubbleText)).toEqual([expect.stringContaining('Cześć!'), 'Pytanie 1', 'Odpowiedź 1', 'Pytanie 2', 'Odpowiedź 2']);
  });
});

describe('request contents', () => {

  it('sends the selected model', async () => {
    await mountApp();
    const select = $<HTMLSelectElement>('#model-switcher');
    select.value = 'bielik';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    answerWith(frame('token', 'ok'));
    await askAndWait('Cześć');
    expect(sentRequest().model).toBe('bielik');
  });

  it('remembers the chosen model across page loads', async () => {
    await mountApp();
    const select = $<HTMLSelectElement>('#model-switcher');
    select.value = 'gpt-4o-mini';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    expect(localStorage.getItem(MODEL_KEY)).toBe('gpt-4o-mini');

    await mountApp({ model: 'bielik' });
    expect($<HTMLSelectElement>('#model-switcher').value).toBe('bielik');
  });

  it('always sends a model the backend accepts, even if the saved one is unknown', async () => {
    await mountApp({ model: 'model-that-no-longer-exists' });
    answerWith(frame('token', 'ok'));
    await askAndWait('Cześć');
    expect(MODELS).toContain(sentRequest().model);
  });

  it('sends the earlier conversation, but not the question being asked', async () => {
    await mountApp();
    answerWith(frame('token', 'Odpowiedź 1'));
    await askAndWait('Pierwsze pytanie');
    answerWith(frame('token', 'Odpowiedź 2'));
    await askAndWait('Drugie pytanie');

    expect(sentRequest(0).history).toEqual([{ role: 'assistant', content: expect.stringContaining('Cześć!') }]);
    expect(sentRequest(1).query).toBe('Drugie pytanie');
    expect(sentRequest(1).history).toEqual([
      { role: 'assistant', content: expect.stringContaining('Cześć!') },
      { role: 'user', content: 'Pierwsze pytanie' },
      { role: 'assistant', content: 'Odpowiedź 1' },
    ]);
  });

  it('keeps requests within the backend limits in a very long conversation', async () => {
    await mountApp({ session: longConversation(240, 5000) });
    answerWith(frame('token', 'ok'));
    await askAndWait('Kolejne pytanie');
    const { history } = sentRequest();
    expect(history.length).toBeLessThanOrEqual(100);
    for (const m of history) expect(m.content.length).toBeLessThanOrEqual(3000);
  });
});

describe('conversation persistence (sessionStorage)', () => {

  it('stores id, role and content of every message - and nothing else', async () => {
    await mountApp();
    answerWith(frame('sources', SOURCES), frame('token', 'Odpowiedź'));
    await askAndWait('Pytanie');
    const saved = stored();
    expect(saved).toHaveLength(3);
    expect(saved.map((m: { role: string }) => m.role)).toEqual(['assistant', 'user', 'assistant']);
    expect(saved[2].content).toBe('Odpowiedź');
    for (const m of saved) expect(Object.keys(m).sort()).toEqual(['content', 'id', 'role']);
  });

  it('restores the conversation after a reload', async () => {
    await mountApp();
    answerWith(frame('token', 'Odpowiedź'));
    await askAndWait('Pytanie');
    const saved = sessionStorage.getItem(STORAGE_KEY)!;

    await mountApp({ session: saved });
    expect(bubbles().map(bubbleText)).toEqual([expect.stringContaining('Cześć!'), 'Pytanie', 'Odpowiedź']);
  });

  it('starts a fresh conversation, and clears the bad data, when the stored text is not valid JSON', async () => {
    await mountApp({ session: '{this is not json' });
    expect(bubbles()).toHaveLength(1);
    expect(bubbleText(bubbles()[0])).toContain('Cześć!');
    expect(sessionStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it.each(['null', '{"a":1}', '"tekst"', '42', 'true'])('still starts when the stored history is valid JSON but not a list (%s)', async (junk) => {
    await mountApp({ session: junk });
    expect(bubbles().length).toBeGreaterThanOrEqual(1);
    expect(input().disabled).toBe(false);
    answerWith(frame('token', 'ok'));
    await askAndWait('Czy działa?');
    expect(bubbleText(lastAssistant())).toBe('ok');
  });

  it('does not store an empty assistant message after a failed request (it would take a history slot later)', async () => {
    await mountApp();
    fetchMock.mockResolvedValue(errorResponse(500, { detail: 'Awaria' }));
    await askAndWait('Pytanie');
    expect(stored().filter((m: { role: string; content: string }) => m.role === 'assistant' && m.content === '')).toHaveLength(0);
  });
});

describe('error handling', () => {

  it('shows the backend\'s message and lets the user try again', async () => {
    await mountApp();
    fetchMock.mockResolvedValue(errorResponse(429, { detail: 'Zbyt wiele zapytań. Odczekaj chwilę.' }));
    await askAndWait('Pytanie');
    expect(lastAssistant().querySelector('.error-msg')!.textContent).toBe('Zbyt wiele zapytań. Odczekaj chwilę.');
    expect(loading().classList.contains('hidden')).toBe(true);
    expect(input().disabled).toBe(false);

    answerWith(frame('token', 'Teraz działa'));
    await askAndWait('Jeszcze raz');
    expect(bubbleText(lastAssistant())).toBe('Teraz działa');
    expect(lastAssistant().querySelector('.error-msg')).toBeNull();
  });

  it('reports a network failure inside the conversation', async () => {
    await mountApp();
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));
    await askAndWait('Pytanie');
    expect(lastAssistant().querySelector('.error-msg')!.textContent).toBe('Failed to fetch');
    expect(input().disabled).toBe(false);
  });

  it('keeps the part of the answer that arrived before a server-side error', async () => {
    await mountApp();
    answerWith(frame('token', 'Częściowa odpowiedź'), frame('error', 'Wystąpił błąd podczas generowania odpowiedzi.'));
    await askAndWait('Pytanie');
    expect(lastAssistant().textContent).toContain('Częściowa odpowiedź');
    expect(lastAssistant().querySelector('.error-msg')!.textContent).toBe('Wystąpił błąd podczas generowania odpowiedzi.');
  });

  it('shows error text as text, never as HTML', async () => {
    await mountApp();
    fetchMock.mockResolvedValue(errorResponse(500, { detail: '<img src=x onerror="window.__xss=1"><b>pogrubione</b>' }));
    await askAndWait('Pytanie');
    const message = lastAssistant().querySelector('.error-msg')!;
    expect(message.querySelectorAll('img, b').length).toBe(0);
    expect(message.textContent).toContain('<img src=x onerror="window.__xss=1">');
  });
});

describe('untrusted content', () => {

  const attack = '<img src=x onerror="window.__xss=1"><script>window.__xss=2</script><b>x</b>';

  it('shows what the user typed as text', async () => {
    await mountApp();
    answerWith(frame('token', 'ok'));
    await askAndWait(attack);
    const userBubble = bubbles()[1];
    expect(userBubble.querySelectorAll('img, script, b').length).toBe(0);
    expect(bubbleText(userBubble)).toBe(attack);
  });

  it('shows model output as text, while still formatting the safe markdown in it', async () => {
    await mountApp();
    answerWith(frame('token', `${attack} **ważne**`));
    await askAndWait('Pytanie');
    const answer = lastAssistant();
    expect(answer.querySelectorAll('img, script, b').length).toBe(0);
    expect(answer.querySelector('strong')!.textContent).toBe('ważne');
    expect(bubbleText(answer)).toContain(attack);
  });

  it('shows scraped source metadata as text', async () => {
    await mountApp();
    const evil = { id: 1, text: attack, score: 0.03, meta: { title: attack, section: attack, url: 'javascript:alert(1)' } };
    answerWith(frame('sources', [evil]), frame('token', 'ok'));
    await askAndWait('Pytanie');
    const card = lastAssistant().querySelector('.source-card')!;
    expect(card.querySelectorAll('img, script, b').length).toBe(0);
    for (const a of Array.from(card.querySelectorAll('a'))) expect(a.getAttribute('href')).not.toMatch(/^javascript:/i);
  });
});
