import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { sendChatMessage } from '../../src/api/client';
import type { ChatRequestPayload, Message } from '../../src/types';

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

describe('API and SSE Client (client.ts)', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.clearAllMocks();
  });

  it('should truncate history to the last 6 messages (Sliding Window)', async () => {
    const fakeHistory: Message[] = Array.from({ length: 15 }).map((_, i) => ({
      id: String(i),
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: `Message ${i + 1}`
    }));

    const payload: ChatRequestPayload = { query: 'Test', model: 'bielik', history: fakeHistory };
    (globalThis.fetch as any).mockResolvedValue({
      ok: true,
      body: createMockStream(['data: [DONE]\n\n'])
    });

    await sendChatMessage(payload, vi.fn(), vi.fn(), vi.fn());

    const fetchCallArgs = (globalThis.fetch as any).mock.calls[0][1];
    const bodySent = JSON.parse(fetchCallArgs.body);
    expect(bodySent.history).toHaveLength(6);
    expect(bodySent.history[5].content).toBe('Message 15'); 
  });

  it('should properly handle human-readable error messages from the backend (e.g., 429)', async () => {
    (globalThis.fetch as any).mockResolvedValue({
      ok: false,
      status: 429,
      json: async () => ({ detail: 'Too many requests. Please wait a moment.' })
    });

    const onChunkMock = vi.fn();
    const payload: ChatRequestPayload = { query: 'Test', model: 'bielik', history: [] };

    await sendChatMessage(payload, onChunkMock, vi.fn(), vi.fn());

    expect(onChunkMock).toHaveBeenCalledWith(
      expect.stringContaining('Too many requests. Please wait a moment.')
    );
  });

  it('should parse valid SSE stream events', async () => {
    const sseChunks = [
      'data: {"type": "status", "content": "Searching..."}\n\n',
      'data: {"type": "sources", "content": [{"id": 1, "text": "Document"}]}\n\n',
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

    expect(onStatusMock).toHaveBeenCalledWith('Searching...');
    expect(onSourcesMock).toHaveBeenCalledWith([{ id: 1, text: 'Document' }]);
    expect(onChunkMock).toHaveBeenCalledTimes(2);
    expect(onChunkMock).toHaveBeenNthCalledWith(1, 'A');
    expect(onChunkMock).toHaveBeenNthCalledWith(2, 'B');
  });

  it('should handle fragmented network packets', async () => {
    const sseChunks = [
      'data: {"type": "token", "con',
      'tent": "Whole"}\n\ndata: [DONE]\n\n'
    ];

    (globalThis.fetch as any).mockResolvedValue({
      ok: true,
      body: createMockStream(sseChunks)
    });

    const onChunkMock = vi.fn();

    await sendChatMessage({ query: 'Test' } as any, onChunkMock, vi.fn(), vi.fn());

    expect(onChunkMock).toHaveBeenCalledWith('Whole');
  });
});