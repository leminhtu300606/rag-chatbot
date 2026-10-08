// Wrapper fetch cho backend FastAPI: gắn Bearer token, báo 401 ra toàn app.
import type {
  ChatRequest,
  ChatResponse,
  ChatSource,
  HealthResponse,
  SessionDetail,
  ServerSession,
  StatsResponse,
  UploadResponse,
  UploadsResponse,
} from "../types";

const TOKEN_KEY = "rag_token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(t: string | null): void {
  try {
    if (t) localStorage.setItem(TOKEN_KEY, t);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* bỏ qua */
  }
}

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail || `HTTP ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

async function apiFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {};
  if (!(init?.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const token = getToken();
  // Mọi /api* gắn token, trừ /api/auth/* (giữ đúng hành vi cũ)
  if (token && url.startsWith("/api") && !url.startsWith("/api/auth/")) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  let res: Response;
  try {
    res = await fetch(url, { ...init, headers: { ...headers, ...(init?.headers as Record<string, string>) } });
  } catch (e) {
    throw new ApiError(0, e instanceof Error ? e.message : String(e));
  }
  if (res.status === 401) {
    window.dispatchEvent(new CustomEvent("rag:unauthorized"));
    let detail = "Unauthorized";
    try {
      const j = await res.json();
      detail = (j as { detail?: string }).detail || detail;
    } catch {
      /* bỏ qua */
    }
    throw new ApiError(401, detail);
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      detail = (j as { detail?: string }).detail || detail;
    } catch {
      try {
        detail = (await res.text()).slice(0, 1200) || detail;
      } catch {
        /* bỏ qua */
      }
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export interface StreamHandlers {
  onToken: (t: string) => void;
  onStatus?: (s: string) => void;
  signal?: AbortSignal;
}

export interface StreamDone {
  answer: string;
  sources: ChatSource[];
  standalone_question?: string;
  summary?: string;
  session_id?: string;
  style?: string;
  math_result?: string | null;
  math_expression?: string;
  is_social?: boolean;
  needs_clarification?: boolean;
  clarify_reason?: string;
}

async function streamFetch<T extends { answer: string }>(
  url: string,
  body: ChatRequest,
  ev: StreamHandlers
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  };
  const token = getToken();
  if (token && url.startsWith("/api") && !url.startsWith("/api/auth/")) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: ev.signal,
    });
  } catch (e) {
    throw e instanceof Error ? e : new Error(String(e));
  }
  const fail = async (status: number, fallback: string): Promise<never> => {
    if (status === 401) {
      window.dispatchEvent(new CustomEvent("rag:unauthorized"));
      let detail = "Unauthorized";
      try {
        detail = ((await res.json()) as { detail?: string }).detail || detail;
      } catch {
        /* bỏ qua */
      }
      throw new ApiError(401, detail);
    }
    let detail = fallback;
    try {
      detail = (((await res.json()) as { detail?: string }).detail || detail).toString().slice(0, 1200);
    } catch {
      try {
        detail = ((await res.text()) || detail).slice(0, 1200);
      } catch {
        /* bỏ qua */
      }
    }
    throw new ApiError(status, detail);
  };
  if (res.status === 401) await fail(401, "Unauthorized");
  if (!res.ok || !res.body) await fail(res.status, `HTTP ${res.status}`);
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let done: T | null = null;
  const handleFrame = (frame: string): void => {
    for (const line of frame.split("\n")) {
      const t = line.trim();
      if (!t.startsWith("data:")) continue;
      const payload = t.slice(5).trim();
      if (!payload) continue;
      let j: Record<string, unknown>;
      try {
        j = JSON.parse(payload) as Record<string, unknown>;
      } catch {
        continue;
      }
      if (typeof j.token === "string" && j.token) ev.onToken(j.token);
      else if (typeof j.status === "string" && j.status) ev.onStatus?.(j.status);
      else if (j.done) done = j as unknown as T;
      else if (typeof j.error === "string" && j.error) throw new ApiError(res.status, j.error.slice(0, 1200));
    }
  };
  try {
    for (;;) {
      const { done: rDone, value } = await reader.read();
      if (value) {
        buf += decoder.decode(value, { stream: true });
        let idx: number;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          handleFrame(frame);
        }
      }
      if (rDone) break;
    }
    if (buf.trim()) handleFrame(buf);
    if (!done) throw new ApiError(res.status, "Stream kết thúc mà không có kết quả");
    return done;
  } finally {
    try {
      await reader.cancel();
    } catch {
      /* bỏ qua */
    }
  }
}

export const api = {
  health: () => apiFetch<HealthResponse>("/api/health"),
  stats: () => apiFetch<StatsResponse>("/api/stats"),
  listSessions: () => apiFetch<{ sessions: ServerSession[] }>("/api/sessions"),
  createSession: (session_id: string, title: string, style: string) =>
    apiFetch("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ session_id, title, style }),
    }).catch(() => undefined),
  getSession: (id: string) =>
    apiFetch<SessionDetail>(`/api/sessions/${encodeURIComponent(id)}`),
  patchSessionTitle: (id: string, title: string) =>
    apiFetch(`/api/sessions/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }).catch(() => undefined),
  deleteSession: (id: string) =>
    apiFetch(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" }),
  getUploads: (id: string) =>
    apiFetch<UploadsResponse>(`/api/sessions/${encodeURIComponent(id)}/uploads`),
  deleteUpload: (id: string, filename: string) =>
    apiFetch(
      `/api/sessions/${encodeURIComponent(id)}/uploads/${encodeURIComponent(filename)}`,
      { method: "DELETE" }
    ),
  uploadFile: (id: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return apiFetch<UploadResponse>(`/api/sessions/${encodeURIComponent(id)}/upload`, {
      method: "POST",
      body: fd,
    });
  },
  postChat: (body: ChatRequest) =>
    apiFetch<ChatResponse>("/api/chat", { method: "POST", body: JSON.stringify(body) }),
  /**
   * Chat streaming qua SSE (POST /api/chat/stream).
   * Server gửi các event: {status} tiến trình, {token} từng đoạn chữ,
   * cuối cùng {done:true, answer, sources, standalone_question, summary, ...}
   * hoặc {error}. Hỗ trợ AbortSignal để hủy giữa chừng.
   */
  postChatStream: (body: ChatRequest, ev: StreamHandlers): Promise<StreamDone> =>
    streamFetch<StreamDone>("/api/chat/stream", body, ev),
  login: (username: string, password: string) =>
    apiFetch<{ token: string; username: string }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  me: () => apiFetch<{ username: string }>("/api/auth/me"),
  logout: () =>
    apiFetch("/api/auth/logout", { method: "POST" }).catch(() => undefined),
};
