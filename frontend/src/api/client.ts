import type { ChatRequestPayload, SourceDocument } from '../types';

// Adres naszego przyszłego serwera FastAPI
const API_URL = 'http://localhost:8000/api/chat';

export async function sendChatMessage(
  payload: ChatRequestPayload,
  onChunk: (text: string) => void,
  onSources: (sources: SourceDocument[]) => void
): Promise<void> {
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Błąd serwera: ${response.status}`);
    }

    if (!response.body) throw new Error('Brak strumienia w odpowiedzi.');

    // Odbieranie strumienia danych na żywo
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let done = false;

    while (!done) {
      const { value, done: readerDone } = await reader.read();
      done = readerDone;
      if (value) {
        const chunk = decoder.decode(value, { stream: true });
        // FastAPI będzie wysyłać linie zaczynające się od "data: " (standard SSE)
        const lines = chunk.split('\n').filter((line) => line.trim().startsWith('data: '));
        
        for (const line of lines) {
          const dataStr = line.replace('data: ', '').trim();
          if (dataStr === '[DONE]') continue; // Koniec strumienia
          
          try {
            const data = JSON.parse(dataStr);
            if (data.type === 'token') {
              onChunk(data.content);
            } else if (data.type === 'sources') {
              onSources(data.content);
            }
          } catch (e) {
            console.warn('Błąd parsowania fragmentu JSON:', dataStr);
          }
        }
      }
    }
  } catch (error) {
    console.error('Błąd komunikacji z API:', error);
    onChunk('\n\n**[Błąd systemu]:** Nie udało się nawiązać połączenia z serwerem wiedzy AGH.');
  }
}