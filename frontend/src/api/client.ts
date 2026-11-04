import type { ChatRequestPayload, SourceDocument } from '../types';

const API_URL = 'http://localhost:8000/api/chat';

export async function sendChatMessage(
  payload: ChatRequestPayload,
  onChunk: (text: string) => void,
  onSources: (sources: SourceDocument[]) => void,
  onStatus?: (status: string) => void
): Promise<void> {
  try {
    const safePayload = {
      ...payload,
      history: payload.history ? payload.history.slice(-6) : []
    };

    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(safePayload),
    });

    if (!response.ok) {
      let errorMessage = `Server error: ${response.status}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = errorData.detail;
        }
      } catch (e) {
      }
      throw new Error(errorMessage);
    }

    if (!response.body) throw new Error('No stream in the response.');

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let done = false;
    let buffer = '';

    while (!done) {
      const { value, done: readerDone } = await reader.read();
      done = readerDone;
      if (value) {
        buffer += decoder.decode(value, { stream: true });
        
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const event of events) {
          if (event.trim().startsWith('data:')) {
            const dataStr = event.replace(/^data:\s*/m, '').trim();
            if (dataStr === '[DONE]') continue; 
            
            try {
              const data = JSON.parse(dataStr);
              if (data.type === 'token') {
                onChunk(data.content);
              } else if (data.type === 'sources') {
                onSources(data.content);
              } else if (data.type === 'status' && onStatus) {
                onStatus(data.content); 
              } else if (data.type === 'error') {
                onChunk(`\n\n**[Error]:** ${data.content}`);
              }
            } catch (e) {
              console.warn('Error parsing JSON fragment:', dataStr);
            }
          }
        }
      }
    }
  } catch (error: any) {
    console.error('Failed to communicate with API', error);
    onChunk(`\n\n**[Error]:** ${error.message || 'Failed to connect with the server'}`);
  }
}