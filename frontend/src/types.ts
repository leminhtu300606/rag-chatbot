// Kiểu dữ liệu khớp API FastAPI (backend/api/app.py) và localStorage cũ.

export interface Msg {
  role: "user" | "assistant";
  content: string;
}

export interface ChatSource {
  filename?: string;
  section?: string;
  page?: number | null;
  block_type?: string;
  table_index?: number | null;
  paragraph_index?: number | null;
  block_index?: number | null;
  category?: string;
  subcategory?: string;
  source?: string;
}

export interface ChatContextItem {
  text: string;
  score?: number;
  metadata?: {
    filename?: string;
    section?: string;
    page?: number | null;
    block_index?: number | null;
    block_type?: string;
  };
}

export interface ChatRequest {
  question: string;
  category?: string;
  top_k?: number;
  use_rerank: boolean;
  show_context: boolean;
  use_history: boolean;
  style: string;
  history?: Msg[];
  session_id?: string;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  context?: ChatContextItem[];
  session_id: string;
  standalone_question?: string;
  history_used?: Msg[];
  style?: string;
  math_result?: string | null;
  math_expression?: string;
  is_social?: boolean;
  needs_clarification?: boolean;
  clarify_reason?: string;
  summary?: string;
}

export interface HealthResponse {
  status: string;
  chroma_count?: number;
  vector_count?: number;
  processed_files?: number;
  error?: string;
}

export interface StatsResponse {
  llm_backend?: string;
  llm_model?: string;
  device?: string;
  embed_model?: string;
  chroma_count?: number;
  vector_count?: number;
  [k: string]: unknown;
}

export interface ServerSession {
  session_id: string;
  title?: string;
  style?: string;
  created_at: number; // giây
  updated_at: number; // giây
}

export interface SessionDetail {
  title?: string;
  style?: string;
  created_at: number;
  updated_at: number;
  history: Msg[];
}

export interface UploadFileInfo {
  filename: string;
  chunks: number;
  page?: number | null;
  category?: string;
  source?: string;
}

export interface UploadsResponse {
  session_id: string;
  files: UploadFileInfo[];
  total_chunks?: number;
  disk_files: { filename: string; size: number }[];
}

export interface UploadResponse {
  filename: string;
  chunks?: number;
  pages?: number;
  auto_message?: string;
  is_auto_answered?: boolean;
  sources?: ChatSource[];
  detail?: string;
}

export interface LocalSession {
  title: string;
  createdAt: number; // ms
  updatedAt: number; // ms
  conversation: Msg[]; // tối đa 60
  style: string;
}

export type SessionMap = Record<string, LocalSession>;

/** Meta hiển thị kèm bubble assistant (không lưu localStorage). */
export interface AssistantMeta {
  badge: string;
  sources: ChatSource[];
  rewrite?: string;
  summary?: string;
}

export interface DisplayMsg extends Msg {
  key: string;
  meta?: AssistantMeta;
  category?: string;
}
