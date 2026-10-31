export type Role = 'user' | 'assistant' | 'system';

export interface SourceMeta {
  tytul?: string;
  url?: string;
  sekcja?: string;
  zrodlo?: string;
}

export interface SourceDocument {
  text: string;
  score: number;
  meta: SourceMeta;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  sources?: SourceDocument[];
}

export interface ChatRequestPayload {
  query: string;
  model: string;
  history: {
    role: 'user' | 'assistant' | 'system';
    content: string;
  }[];
}