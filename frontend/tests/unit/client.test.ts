import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { sendChatMessage } from '../../src/api/client';
import type { ChatRequestPayload, Message } from '../../src/types';
import { DONE, controlledStream, cutAt, errorResponse, frame, okResponse, recorder, sseResponse, toBytes } from '../helpers/sse';

const BACKEND_HISTORY_LIMIT = 100;      // backend rejects bigger histories with 422
const BACKEND_MESSAGE_LIMIT = 3000;     // ... and messages longer than this

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
  vi.spyOn(console, 'error').mockImplementation(() => {});     // the client logs failures; keep test output clean
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const payload = (over: Partial<ChatRequestPayload> = {}): ChatRequestPayload =>
  ({ query: 'Test', model: 'gpt-4o-mini', history: [], ...over });

/** Run one request whose response body is `chunks`; returns the recorded UI events. */
async function run(chunks: Array<string | Uint8Array>, p: ChatRequestPayload = payload()) {
  fetchMock.mockResolvedValue(sseResponse(chunks));
  const rec = recorder(vi);
  await sendChatMessage(p, rec.onChunk, rec.onSources, rec.onStatus, rec.onError);
  return rec;
}
const sentBody = () => JSON.parse(fetchMock.mock.calls[0][1].body);

// =====================================================================
// 1. What is sent to the backend
// =====================================================================
describe('request', () => {

  it('POSTs JSON to /api/chat with query, model and history', async () => {
    await run([DONE], payload({ query: 'Ile kosztuje akademik?', model: 'bielik', history: [{ role: 'user', content: 'Cześć' }] }));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/chat');
    expect(init.method).toBe('POST');
    expect(init.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(sentBody()).toEqual({ query: 'Ile kosztuje akademik?', model: 'bielik', history: [{ role: 'user', content: 'Cześć' }] });
  });

  describe('history limits (the backend answers 422 above 100 messages or 3000 characters per message)', () => {

    const conversation = (n: number): Message[] =>
      Array.from({ length: n }, (_, i) => ({ id: String(i), role: i % 2 === 0 ? 'user' : 'assistant', content: `Message ${i + 1}` }));

    it('sends only the most recent 6 messages, oldest first', async () => {
      await run([DONE], payload({ history: conversation(15) }));
      const history = sentBody().history;
      expect(history).toHaveLength(6);
      expect(history[0].content).toBe('Message 10');
      expect(history[5].content).toBe('Message 15');
    });

    it('sends short histories unchanged', async () => {
      await run([DONE], payload({ history: conversation(3) }));
      expect(sentBody().history.map((m: { content: string }) => m.content)).toEqual(['Message 1', 'Message 2', 'Message 3']);
    });

    it('stays below the backend limit however long the chat gets', async () => {
      await run([DONE], payload({ history: conversation(500) }));
      expect(sentBody().history.length).toBeLessThanOrEqual(BACKEND_HISTORY_LIMIT);
    });

    it('cuts over-long messages so one long answer cannot make every later request fail', async () => {
      await run([DONE], payload({ history: [{ role: 'assistant', content: 'x'.repeat(5000) }] }));
      expect(sentBody().history[0].content.length).toBeLessThanOrEqual(BACKEND_MESSAGE_LIMIT);
    });

    it('sends only role and content of each message (no ids, no stored sources)', async () => {
      const withExtras = conversation(2).map((m) => ({ ...m, sources: [{ text: 'big', score: 1, meta: {} }] }));
      await run([DONE], payload({ history: withExtras }));
      for (const m of sentBody().history) expect(Object.keys(m).sort()).toEqual(['content', 'role']);
    });

    it('does not modify the caller\'s payload', async () => {
      const p = payload({ history: [...conversation(12), { id: 'long', role: 'assistant', content: 'y'.repeat(4000) }] });
      const before = structuredClone(p);
      await run([DONE], p);
      expect(p).toEqual(before);
    });
  });
});

// =====================================================================
// 2. Parsing the event stream
// =====================================================================
describe('SSE stream parsing', () => {

  it('reports status, sources and tokens in the order they arrive', async () => {
    const rec = await run([
      frame('status', 'Przeszukuję bazę...'),
      frame('sources', [{ id: 1, text: 'Dokument' }]),
      frame('token', 'A'), frame('token', 'B'), DONE,
    ]);
    expect(rec.log).toEqual([
      ['status', 'Przeszukuję bazę...'],
      ['sources', [{ id: 1, text: 'Dokument' }]],
      ['token', 'A'], ['token', 'B'],
    ]);
  });

  it('ignores [DONE], unknown event types, comment/keep-alive frames and empty frames', async () => {
    const rec = await run([
      ': keep-alive\n\n', '\n\n', frame('mystery', 'x'), frame('token', 'A'), DONE, 'event: ping\n\n',
    ]);
    expect(rec.log).toEqual([['token', 'A']]);
    expect(rec.onError).not.toHaveBeenCalled();
  });

  it('survives a malformed frame and keeps delivering later events', async () => {
    const rec = await run([frame('token', 'A'), 'data: {oops not json\n\n', frame('token', 'B'), DONE]);
    expect(rec.log).toEqual([['token', 'A'], ['token', 'B']]);
    expect(console.warn).toHaveBeenCalled();
  });

  it('works without a status handler', async () => {
    fetchMock.mockResolvedValue(sseResponse([frame('status', 's'), frame('token', 'A'), DONE]));
    const onChunk = vi.fn();
    await sendChatMessage(payload(), onChunk, vi.fn());
    expect(onChunk).toHaveBeenCalledWith('A');
  });

  it('delivers token text intact, even if it contains blank lines, "data:" or quotes', async () => {
    const tricky = 'linia 1\n\nlinia 2 data: {"type":"token"} "cytat" \\ koniec';
    const rec = await run([frame('token', tricky), DONE]);
    expect(rec.log).toEqual([['token', tricky]]);
  });

  it('handles many events in a single network chunk', async () => {
    const tokens = Array.from({ length: 50 }, (_, i) => `t${i}`);
    const rec = await run([toBytes(...tokens.map((t) => frame('token', t)), DONE)]);
    expect(rec.log.map(([, v]) => v)).toEqual(tokens);
  });

  describe('network fragmentation', () => {
    // "Zażółć gęślą jaźń" + emoji contain 2-, 3- and 4-byte UTF-8 characters.
    const frames = [frame('status', 'Szukam…'), frame('sources', [{ id: 1, text: 'Zażółć gęślą jaźń' }]),
                    frame('token', 'Zażółć 🎓 gęślą'), frame('token', ' jaźń – ł'), DONE];
    const bytes = toBytes(...frames);

    it('gives identical results wherever the stream is cut into two chunks (including inside a character)', async () => {
      const reference = (await run([bytes])).log;
      expect(reference).toHaveLength(4);
      for (let cut = 1; cut < bytes.length; cut++) {
        const rec = await run(cutAt(bytes, cut));
        expect(rec.log).toEqual(reference);
      }
    });

    it('gives identical results when bytes arrive one at a time', async () => {
      const reference = (await run([bytes])).log;
      const rec = await run(Array.from(bytes, (b) => new Uint8Array([b])));
      expect(rec.log).toEqual(reference);
    });
  });
});

// =====================================================================
// 3. Errors
// =====================================================================
describe('errors', () => {

  it('shows the backend\'s message for HTTP errors through onError (e.g. 429) and delivers no tokens', async () => {
    fetchMock.mockResolvedValue(errorResponse(429, { detail: 'Zbyt wiele zapytań. Odczekaj chwilę.' }));
    const rec = recorder(vi);
    await sendChatMessage(payload(), rec.onChunk, rec.onSources, rec.onStatus, rec.onError);
    expect(rec.log).toEqual([['error', 'Zbyt wiele zapytań. Odczekaj chwilę.']]);
  });

  it('falls back to the answer channel when no error handler is given', async () => {
    fetchMock.mockResolvedValue(errorResponse(429, { detail: 'Too many requests. Please wait a moment.' }));
    const onChunk = vi.fn();
    await sendChatMessage(payload(), onChunk, vi.fn(), vi.fn());
    expect(onChunk).toHaveBeenCalledWith(expect.stringContaining('Too many requests. Please wait a moment.'));
  });

  it.each([
    ['non-JSON error page (502)', errorResponse(502), 'Server error: 502'],
    ['JSON without a detail field', errorResponse(503, { message: 'x' }), 'Server error: 503'],
    ['empty detail', errorResponse(500, { detail: '' }), 'Server error: 500'],
  ])('uses a generic message for a %s', async (_name, response, expected) => {
    fetchMock.mockResolvedValue(response);
    const rec = recorder(vi);
    await sendChatMessage(payload(), rec.onChunk, rec.onSources, rec.onStatus, rec.onError);
    expect(rec.log).toEqual([['error', expected]]);
  });

  it('reports a response without a body', async () => {
    fetchMock.mockResolvedValue(okResponse(null));
    const rec = recorder(vi);
    await sendChatMessage(payload(), rec.onChunk, rec.onSources, rec.onStatus, rec.onError);
    expect(rec.log).toEqual([['error', 'No stream in the response.']]);
  });

  it('reports a network failure', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));
    const rec = recorder(vi);
    await sendChatMessage(payload(), rec.onChunk, rec.onSources, rec.onStatus, rec.onError);
    expect(rec.log).toEqual([['error', 'Failed to fetch']]);
  });

  it('uses a friendly Polish message when the error carries no text', async () => {
    fetchMock.mockRejectedValue(new Error(''));
    const rec = recorder(vi);
    await sendChatMessage(payload(), rec.onChunk, rec.onSources, rec.onStatus, rec.onError);
    expect(rec.log).toEqual([['error', 'Wystąpił problem z połączeniem do serwera.']]);
  });

  it('keeps the tokens already received when the connection drops mid-answer', async () => {
    const stream = controlledStream();
    stream.push(frame('token', 'Od '));
    stream.push(frame('token', '500 '));
    stream.fail(new Error('network reset'));
    fetchMock.mockResolvedValue(okResponse(stream.body));
    const rec = recorder(vi);
    await sendChatMessage(payload(), rec.onChunk, rec.onSources, rec.onStatus, rec.onError);
    expect(rec.log).toEqual([['token', 'Od '], ['token', '500 '], ['error', 'network reset']]);
  });

  it('passes a server-side "error" event to onError', async () => {
    const rec = await run([frame('token', 'Częściowa'), frame('error', 'Wyszukiwarka jest chwilowo niedostępna.'), DONE]);
    expect(rec.log).toEqual([['token', 'Częściowa'], ['error', 'Wyszukiwarka jest chwilowo niedostępna.']]);
  });

  it('writes a server-side "error" event into the answer when there is no error handler', async () => {
    fetchMock.mockResolvedValue(sseResponse([frame('error', 'Awaria'), DONE]));
    const onChunk = vi.fn();
    await sendChatMessage(payload(), onChunk, vi.fn(), vi.fn());
    expect(onChunk).toHaveBeenCalledWith(expect.stringContaining('Awaria'));
  });

  it.each([
    ['HTTP 500', () => errorResponse(500, { detail: 'x' })],
    ['no body', () => okResponse(null)],
    ['broken stream', () => { const s = controlledStream(); s.fail(new Error('boom')); return okResponse(s.body); }],
  ])('never rejects its promise (%s), so the UI can always re-enable the input', async (_name, make) => {
    fetchMock.mockResolvedValue(make());
    await expect(sendChatMessage(payload(), vi.fn(), vi.fn(), vi.fn(), vi.fn())).resolves.toBeUndefined();
  });

  it('never rejects when fetch itself throws', async () => {
    fetchMock.mockRejectedValue(new Error('offline'));
    await expect(sendChatMessage(payload(), vi.fn(), vi.fn())).resolves.toBeUndefined();
  });
});
