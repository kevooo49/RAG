// 1. Zdefiniowanie ról (kto jest nadawcą wiadomości)
export type Role = 'user' | 'assistant' | 'system';

// 2. Metadane pochodzące z bazy danych Elasticsearch / ChromaDB
export interface SourceMeta {
  tytul?: string;
  url?: string;
  sekcja?: string;
  zrodlo?: string;
}

// 3. Pełny dokument źródłowy wyciągnięty przez algorytm RRF
export interface SourceDocument {
  text: string;
  score: number;      // Wynik z Reciprocal Rank Fusion
  meta: SourceMeta;
}

// 4. Struktura pojedynczej wiadomości wyświetlanej w UI
export interface Message {
  id: string;         // Unikalny ID (np. Date.now().toString())
  role: Role;
  content: string;
  sources?: SourceDocument[]; // Źródła pojawiają się tylko przy odpowiedziach asystenta
}

// 5. Struktura paczki (JSON) wysyłanej do naszego backendu (FastAPI)
export interface ChatRequestPayload {
  query: string;
  model: string;
  history: {
    role: 'user' | 'assistant' | 'system';
    content: string;
  }[];
}