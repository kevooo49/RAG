export type Role = 'user' | 'assistant' | 'system';

export interface SourceMeta {
  title?: string;
  url?: string;
  section?: string;
  source?: string;
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