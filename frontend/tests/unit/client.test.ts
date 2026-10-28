import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { sendChatMessage } from '../../src/api/client';
import type { ChatRequestPayload, Message } from '../../src/types';

// Tworzymy pomocniczą funkcję do symulowania natywnego strumienia bajtów z backendu
function createMockStream(chunks: string[]) {
  const encoder = new TextEncoder();
  let index = 0;
  return {
    getReader: () => ({
      read: async () => {
        if (index < chunks.length) {
          return { done: false, value: encoder.encode(chunks[index++]) };
        }
        return { done: true, value: undefined };
      }
    })
  };
}

describe('Klient API i SSE (client.ts)', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    // Resetujemy mocki przed każdym testem używając globalThis
    globalThis.fetch = vi.fn();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.clearAllMocks();
  });

  it('powinien ucinać historię do 6 ostatnich wiadomości (Sliding Window)', async () => {
    // Generujemy fałszywą długą historię (15 wiadomości)
    const fakeHistory: Message[] = Array.from({ length: 15 }).map((_, i) => ({
      id: String(i),
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: `Wiadomość ${i + 1}`
    }));

    const payload: ChatRequestPayload = { query: 'Test', model: 'bielik', history: fakeHistory };

    // Symulacja prawidłowej, pustej odpowiedzi z serwera
    (globalThis.fetch as any).mockResolvedValue({
      ok: true,
      body: createMockStream(['data: [DONE]\n\n'])
    });

    await sendChatMessage(payload, vi.fn(), vi.fn(), vi.fn());

    // Weryfikujemy przechwycone żądanie
    const fetchCallArgs = (globalThis.fetch as any).mock.calls[0][1];
    const bodySent = JSON.parse(fetchCallArgs.body);

    // Zamiast 15 wiadomości, serwer musi dostać tylko 6 najnowszych (od 10 do 15)
    expect(bodySent.history).toHaveLength(6);
    expect(bodySent.history[5].content).toBe('Wiadomość 15'); 
  });

  it('powinien poprawnie obsługiwać ludzkie komunikaty błędów z backendu (np. 429)', async () => {
    // Mockujemy odrzucenie zapytania przez SlowAPI
    (globalThis.fetch as any).mockResolvedValue({
      ok: false,
      status: 429,
      json: async () => ({ detail: 'Zbyt wiele zapytań. Odczekaj chwilę.' })
    });

    const onChunkMock = vi.fn();
    const payload: ChatRequestPayload = { query: 'Test', model: 'bielik', history: [] };

    await sendChatMessage(payload, onChunkMock, vi.fn(), vi.fn());

    // Upewniamy się, że klient wrzucił do UI sformatowany error, a nie padł
    expect(onChunkMock).toHaveBeenCalledWith(
      expect.stringContaining('Zbyt wiele zapytań. Odczekaj chwilę.')
    );
  });

  it('powinien parsować poprawne zdarzenia strumienia SSE', async () => {
    // Symulacja prawidłowego potoku generowania odpowiedzi w RAG
    const sseChunks = [
      'data: {"type": "status", "content": "Przeszukuję..."}\n\n',
      'data: {"type": "sources", "content": [{"id": 1, "text": "Dokument"}]}\n\n',
      'data: {"type": "token", "content": "A"}\n\n',
      'data: {"type": "token", "content": "B"}\n\n',
      'data: [DONE]\n\n'
    ];

    (globalThis.fetch as any).mockResolvedValue({
      ok: true,
      body: createMockStream(sseChunks)
    });

    const onChunkMock = vi.fn();
    const onSourcesMock = vi.fn();
    const onStatusMock = vi.fn();

    await sendChatMessage({ query: 'Test' } as any, onChunkMock, onSourcesMock, onStatusMock);

    expect(onStatusMock).toHaveBeenCalledWith('Przeszukuję...');
    expect(onSourcesMock).toHaveBeenCalledWith([{ id: 1, text: 'Dokument' }]);
    expect(onChunkMock).toHaveBeenCalledTimes(2);
    expect(onChunkMock).toHaveBeenNthCalledWith(1, 'A');
    expect(onChunkMock).toHaveBeenNthCalledWith(2, 'B');
  });

  it('powinien radzić sobie z pofragmentowanymi pakietami sieciowymi', async () => {
    // Rozrywamy jeden pakiet JSON na dwa osobne kawałki bajtów 
    const sseChunks = [
      'data: {"type": "token", "con',
      'tent": "Całość"}\n\ndata: [DONE]\n\n'
    ];

    (globalThis.fetch as any).mockResolvedValue({
      ok: true,
      body: createMockStream(sseChunks)
    });

    const onChunkMock = vi.fn();

    await sendChatMessage({ query: 'Test' } as any, onChunkMock, vi.fn(), vi.fn());

    // Mimo przecięcia, wewnętrzny bufor klienta (buffer +=) powinien poprawnie to zlepić i sparsować
    expect(onChunkMock).toHaveBeenCalledWith('Całość');
  });
});