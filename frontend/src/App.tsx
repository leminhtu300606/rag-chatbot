import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, getToken, setToken } from "./api/client";
import { ApiError } from "./api/client";
import {
  ChatMessage,
  Composer,
  LoginModal,
  Sidebar,
  SourcesPanel,
  StatsModal,
  UploadsBar,
  Welcome,
} from "./components";
import { isNetworkError, offlineGreeting, offlineMath, parseCategory, shortId } from "./utils";
import type {
  AssistantMeta,
  ChatSource,
  DisplayMsg,
  LocalSession,
  Msg,
  SessionMap,
  StatsResponse,
  UploadFileInfo,
} from "./types";
import type { StreamDone } from "./api/client";

const LS_SESSIONS = "rag_sessions_v2";
const LS_ACTIVE = "rag_active_session_v2";
const LS_SESSION_LEGACY = "rag_session_id";
const LS_CONV_LEGACY = "rag_conversation_v2";
const LS_STYLE = "rag_style";
const DEFAULT_STYLE = "casual";
const MAX_KEEP = 60;

const uid = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `s-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;

let msgSeq = 0;
const mkey = () => `m-${Date.now()}-${msgSeq++}`;

function loadMap(): SessionMap {
  try {
    const raw = localStorage.getItem(LS_SESSIONS);
    if (raw) return JSON.parse(raw) as SessionMap;
  } catch {
    /* bỏ qua */
  }
  // Migrate legacy đơn phiên
  try {
    const legacySess = localStorage.getItem(LS_SESSION_LEGACY);
    const legacyConv = localStorage.getItem(LS_CONV_LEGACY);
    if (legacySess && legacyConv) {
      const conv = JSON.parse(legacyConv) as Msg[];
      const now = Date.now();
      return {
        [legacySess]: {
          title: conv.find((m) => m.role === "user")?.content.slice(0, 50) || "Cuộc trò chuyện cũ",
          createdAt: now,
          updatedAt: now,
          conversation: conv.slice(-MAX_KEEP),
          style: DEFAULT_STYLE,
        },
      };
    }
  } catch {
    /* bỏ qua */
  }
  return {};
}

function persistMap(map: SessionMap) {
  try {
    const slim: SessionMap = {};
    for (const [id, s] of Object.entries(map)) {
      slim[id] = { ...s, conversation: s.conversation.slice(-MAX_KEEP) };
    }
    localStorage.setItem(LS_SESSIONS, JSON.stringify(slim));
  } catch {
    /* bỏ qua */
  }
}

function querySessionParam(): string | null {
  try {
    const u = new URL(window.location.href);
    return u.searchParams.get("c") || u.searchParams.get("session") || u.searchParams.get("s");
  } catch {
    return null;
  }
}

function pushSessionParam(id: string | null) {
  try {
    const u = new URL(window.location.href);
    if (id) u.searchParams.set("c", id);
    else u.searchParams.delete("c");
    window.history.pushState({}, "", u.toString());
  } catch {
    /* bỏ qua */
  }
}

export default function App() {
  const [sessions, setSessions] = useState<SessionMap>(() => loadMap());
  const [activeId, setActiveId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(LS_ACTIVE) || querySessionParam();
    } catch {
      return querySessionParam();
    }
  });
  const [style] = useState<string>(() => {
    try {
      return localStorage.getItem(LS_STYLE) || DEFAULT_STYLE;
    } catch {
      return DEFAULT_STYLE;
    }
  });
  const [display, setDisplay] = useState<DisplayMsg[]>([]);
  const [panelMeta, setPanelMeta] = useState<AssistantMeta | null>(null);
  const [typingKey, setTypingKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [healthText, setHealthText] = useState("Đang kiểm tra hệ thống");
  const [healthOk, setHealthOk] = useState<boolean | null>(null);
  const [llmInfo, setLlmInfo] = useState("Đang kết nối mô hình");
  const [accountName, setAccountName] = useState("Học viên KTMM");
  const [authed, setAuthed] = useState(() => !!getToken());
  const [statsData, setStatsData] = useState<StatsResponse | null>(null);
  const [loginOpen, setLoginOpen] = useState(false);
  const [uploads, setUploads] = useState<UploadFileInfo[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [rightOpen, setRightOpen] = useState(false);
  const [sideOpen, setSideOpen] = useState(false);

  const sessionsRef = useRef(sessions);
  sessionsRef.current = sessions;
  const activeRef = useRef(activeId);
  activeRef.current = activeId;
  const msgBoxRef = useRef<HTMLDivElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);

  const active = activeId ? sessions[activeId] : undefined;
  const turns = useMemo(() => Math.floor(display.length / 2), [display]);
  const sessionInfo = useMemo(
    () => (activeId ? `${shortId(activeId)} • ${turns} lượt` : ""),
    [activeId, turns]
  );

  /* ---------- persist + active ---------- */
  useEffect(() => persistMap(sessions), [sessions]);
  useEffect(() => {
    try {
      if (activeId) localStorage.setItem(LS_ACTIVE, activeId);
      else localStorage.removeItem(LS_ACTIVE);
    } catch {
      /* bỏ qua */
    }
  }, [activeId]);

  /* ---------- autoscroll ---------- */
  useEffect(() => {
    const el = msgBoxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [display, typingKey]);

  /* ---------- health + stats ---------- */
  const loadHealth = useCallback(async () => {
    try {
      const [h, s] = await Promise.allSettled([api.health(), api.stats()]);
      if (h.status === "fulfilled") {
        const j = h.value;
        if (j.status === "ready") {
          const n = j.vector_count ?? j.chroma_count ?? 0;
          setHealthText(`${n} vectors • ${j.processed_files ?? 0} files`);
          setHealthOk(true);
        } else if (j.status === "not_ready") {
          setHealthText("Chưa có dữ liệu");
          setHealthOk(false);
        } else {
          setHealthText(j.error ? `Lỗi: ${String(j.error).slice(0, 80)}` : "Không rõ trạng thái");
          setHealthOk(false);
        }
      } else {
        setHealthText("Mất kết nối máy chủ");
        setHealthOk(false);
      }
      if (s.status === "fulfilled") {
        const st = s.value;
        const model = String(st.llm_model || "?");
        const backend = String(st.llm_backend || "?");
        const device = String(st.device || st.embed_model || "?");
        setLlmInfo(`${model} • ${backend} (${device})`);
      }
    } catch {
      setHealthText("Mất kết nối máy chủ");
      setHealthOk(false);
    }
  }, []);

  /* ---------- sessions: server merge ---------- */
  const mergeServerSessions = useCallback(async () => {
    try {
      const j = await api.listSessions();
      const list = j.sessions || [];
      for (const s of list.slice(0, 20)) {
        const serverUpdated = (s.updated_at || 0) * 1000;
        const local = sessionsRef.current[s.session_id];
        if (!local || serverUpdated > (local.updatedAt || 0)) {
          try {
            const d = await api.getSession(s.session_id);
            const conv = (d.history || []).slice(-MAX_KEEP);
            setSessions((prev) => ({
              ...prev,
              [s.session_id]: {
                title:
                  d.title ||
                  conv.find((m) => m.role === "user")?.content.slice(0, 50) ||
                  "Cuộc trò chuyện mới",
                createdAt: (d.created_at || Date.now() / 1000) * 1000,
                updatedAt: (d.updated_at || Date.now() / 1000) * 1000,
                conversation: conv,
                style: d.style || style,
              },
            }));
          } catch {
            /* bỏ qua phiên lỗi */
          }
        }
      }
    } catch {
      /* offline: giữ local */
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadSessionView = useCallback(
    async (id: string) => {
      const local = sessionsRef.current[id];
      // Ưu tiên server nếu có mạng
      try {
        const d = await api.getSession(id);
        const conv = (d.history || []).slice(-MAX_KEEP);
        setSessions((prev) => ({
          ...prev,
          [id]: {
            title:
              d.title ||
              local?.title ||
              conv.find((m) => m.role === "user")?.content.slice(0, 50) ||
              "Cuộc trò chuyện mới",
            createdAt: (d.created_at || Date.now() / 1000) * 1000,
            updatedAt: (d.updated_at || Date.now() / 1000) * 1000,
            conversation: conv,
            style: d.style || local?.style || style,
          },
        }));
        setDisplay(conv.map((m) => ({ ...m, key: mkey() })));
      } catch {
        const conv = local?.conversation || [];
        setDisplay(conv.map((m) => ({ ...m, key: mkey() })));
      }
      setPanelMeta(null);
      setTypingKey(null);
    },
    [style]
  );

  /* ---------- uploads ---------- */
  const loadUploads = useCallback(async (sid: string | null, silent = true) => {
    if (!sid) {
      setUploads([]);
      setTotalChunks(0);
      return;
    }
    try {
      const j = await api.getUploads(sid);
      setUploads(j.files || []);
      setTotalChunks(
        j.total_chunks ?? (j.files || []).reduce((a, f) => a + (f.chunks || 0), 0)
      );
    } catch {
      if (!silent) setStatus("Không tải được danh sách file");
    }
  }, []);

  /* ---------- boot ---------- */
  useEffect(() => {
    const q = querySessionParam();
    const startId = activeRef.current || q;
    if (startId && sessionsRef.current[startId]) {
      setActiveId(startId);
      loadSessionView(startId);
      loadUploads(startId);
    } else if (startId) {
      // id lạ (?c=): tạo vỏ rồi kéo server
      setActiveId(startId);
      loadSessionView(startId);
      loadUploads(startId);
    } else {
      setDisplay([]);
    }
    loadHealth();
    mergeServerSessions();
    api
      .me()
      .then((j) => {
        if (j.username && j.username !== "anonymous") setAccountName(j.username);
      })
      .catch(() => undefined);
    const onPop = () => {
      const id = querySessionParam();
      if (id && id !== activeRef.current) {
        setActiveId(id);
        loadSessionView(id);
        loadUploads(id);
      }
    };
    window.addEventListener("popstate", onPop);
    const onAuth = () => setLoginOpen(true);
    window.addEventListener("rag:unauthorized", onAuth);
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        newChat();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("popstate", onPop);
      window.removeEventListener("rag:unauthorized", onAuth);
      window.removeEventListener("keydown", onKey);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ---------- polling ---------- */
  useEffect(() => {
    const t = window.setInterval(() => {
      if (document.visibilityState !== "visible") return;
      loadHealth();
      mergeServerSessions();
    }, 30000);
    return () => window.clearInterval(t);
  }, [loadHealth, mergeServerSessions]);

  useEffect(() => {
    if (!activeId) return;
    loadUploads(activeId);
    const t = window.setInterval(() => {
      if (document.visibilityState === "visible") loadUploads(activeRef.current);
    }, 15000);
    return () => window.clearInterval(t);
  }, [activeId, loadUploads]);

  /* ---------- new/select/delete ---------- */
  const newChat = useCallback(() => {
    try {
      streamAbortRef.current?.abort();
    } catch {
      /* bỏ qua */
    }
    streamAbortRef.current = null;    const id = uid();
    const now = Date.now();
    const sess: LocalSession = {
      title: "Cuộc trò chuyện mới",
      createdAt: now,
      updatedAt: now,
      conversation: [],
      style,
    };
    setSessions((prev) => ({ ...prev, [id]: sess }));
    setActiveId(id);
    setDisplay([]);
    setPanelMeta(null);
    setTypingKey(null);
    setUploads([]);
    setTotalChunks(0);
    setSideOpen(false);
    pushSessionParam(id);
    api.createSession(id, sess.title, style);
  }, [style]);

  const selectSession = useCallback(
    (id: string) => {
      if (id === activeRef.current) return;
      try {
        streamAbortRef.current?.abort();
      } catch {
        /* bỏ qua */
      }
      streamAbortRef.current = null;
      setActiveId(id);
      pushSessionParam(id);
      loadSessionView(id);
      loadUploads(id);
      setSideOpen(false);
    },
    [loadSessionView, loadUploads]
  );

  const deleteSession = useCallback(
    async (id: string) => {
      if (!window.confirm("Xóa cuộc trò chuyện này?")) return;
      try {
        streamAbortRef.current?.abort();
      } catch {
        /* bỏ qua */
      }
      streamAbortRef.current = null;
      api.deleteSession(id).catch(() => undefined);
      setSessions((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      if (id === activeRef.current) {
        setActiveId(null);
        setDisplay([]);
        setPanelMeta(null);
        pushSessionParam(null);
      }
    },
    []
  );

  /* ---------- send ---------- */
  const pushAssistant = useCallback(
    (sid: string, text: string, meta?: AssistantMeta) => {
      const key = mkey();
      const msg = { role: "assistant" as const, content: text, key, meta };
      setDisplay((prev) => [...prev, msg]);
      setTypingKey(key);
      setSessions((prev) => {
        const s = prev[sid];
        if (!s) return prev;
        return {
          ...prev,
          [sid]: {
            ...s,
            updatedAt: Date.now(),
            conversation: [...s.conversation, { role: "assistant" as const, content: text }].slice(-MAX_KEEP),
          },
        };
      });
      if (meta) setPanelMeta(meta);
    },
    []
  );

  const send = useCallback(
    async (raw: string) => {
      const text = raw.trim();
      if (!text || loading) return;
      // Đảm bảo có phiên
      let sid = activeRef.current;
      if (!sid) {
        sid = uid();
        const now = Date.now();
        setSessions((prev) => ({
          ...prev,
          [sid as string]: {
            title: "Cuộc trò chuyện mới",
            createdAt: now,
            updatedAt: now,
            conversation: [],
            style,
          },
        }));
        setActiveId(sid);
        pushSessionParam(sid);
        api.createSession(sid, "Cuộc trò chuyện mới", style);
      }
      const cur = sessionsRef.current[sid];
      const history: Msg[] = [...(cur?.conversation || [])];
      const { category, question } = parseCategory(text);
      const userMsg = { role: "user" as const, content: text, key: mkey(), category };
      setDisplay((prev) => [...prev, userMsg]);
      const newConv: Msg[] = [...history, { role: "user" as const, content: text }].slice(-MAX_KEEP);
      setSessions((prev) => {
        const s = prev[sid as string];
        const title =
          !s || s.title === "Cuộc trò chuyện mới"
            ? text.slice(0, 50)
            : s.title;
        return {
          ...prev,
          [sid as string]: {
            title,
            createdAt: s?.createdAt || Date.now(),
            updatedAt: Date.now(),
            conversation: newConv,
            style: s?.style || style,
          },
        };
      });
      setLoading(true);
      setStatus("Đang kết nối...");
      // Stream-first: bubble rỗng để hứng chữ dần (không bật typewriter vì
      // nội dung đã hiện dần; typewriter chỉ dùng cho đường fallback blocking).
      const akey = mkey();
      setDisplay((prev) => [...prev, { role: "assistant" as const, content: "", key: akey }]);
      const ctrl = new AbortController();
      streamAbortRef.current = ctrl;
      let streamed = "";
      const applyStream = (snapshot: string) => {
        setDisplay((prev) => prev.map((m) => (m.key === akey ? { ...m, content: snapshot } : m)));
      };
      const finishStream = (text: string, meta?: AssistantMeta) => {
        setDisplay((prev) => prev.map((m) => (m.key === akey ? { ...m, content: text, meta } : m)));
        setSessions((prev) => {
          const s = prev[sid as string];
          if (!s) return prev;
          return {
            ...prev,
            [sid as string]: {
              ...s,
              updatedAt: Date.now(),
              conversation: [...s.conversation, { role: "assistant" as const, content: text }].slice(-MAX_KEEP),
            },
          };
        });
        if (meta) setPanelMeta(meta);
      };
      const dropPlaceholder = () => {
        setDisplay((prev) => prev.filter((m) => m.key !== akey));
      };
      const buildMeta = (j: StreamDone): AssistantMeta => {
        let badge = `✅ ${(j.sources || []).length} nguồn`;
        if (j.math_result != null) badge = "🧮 Tính toán";
        else if (j.is_social) badge = "💬 Xã giao";
        else if (j.needs_clarification) badge = `❓ ${j.clarify_reason || "Cần làm rõ"}`;
        return {
          badge,
          sources: j.sources || [],
          rewrite: j.standalone_question
            ? `Rewrite: "${question}" → "${j.standalone_question}"`
            : undefined,
          summary: j.summary,
        };
      };
      const chatBody = {
        question,
        category,
        use_rerank: true,
        show_context: false,
        use_history: true,
        style,
        history,
        session_id: sid,
      };
      try {
        const j = await api.postChatStream(chatBody, {
          signal: ctrl.signal,
          onStatus: (s) => setStatus(s),
          onToken: (t) => {
            streamed += t;
            applyStream(streamed);
          },
        });
        const answer = j.answer || streamed || "(không có trả lời)";
        finishStream(answer, buildMeta(j));
        api.patchSessionTitle(sid, text.slice(0, 50));
      } catch (e) {
        if (ctrl.signal.aborted) return; // chuyển/xóa phiên giữa chừng: view mới đã reset
        if (e instanceof ApiError && e.status === 401) {
          dropPlaceholder();
          setDisplay((prev) => prev.slice(0, -1)); // gỡ user msg như hành vi cũ
          return; // login modal đã mở qua event
        }
        if (!streamed) {
          // Stream hỏng ngay từ đầu (chưa có chữ nào): fallback /api/chat blocking 1 lần
          try {
            setStatus("Stream gián đoạn, đang thử lại...");
            const j = await api.postChat(chatBody);
            const answer = j.answer || "(không có trả lời)";
            finishStream(answer, buildMeta(j as StreamDone));
            api.patchSessionTitle(sid, text.slice(0, 50));
            return;
          } catch (e2) {
            if (e2 instanceof ApiError && e2.status === 401) {
              dropPlaceholder();
              setDisplay((prev) => prev.slice(0, -1));
              return;
            }
            e = e2;
          }
        }
        if (streamed) {
          // Lỗi giữa chừng: giữ phần đã nhận được
          finishStream(streamed, { badge: "⚠️ Stream gián đoạn (phần đã nhận)", sources: [] });
        } else {
          dropPlaceholder();
        }
        if (streamed) return; // đã giữ phần nhận được + badge cảnh báo ở trên
        if (isNetworkError(e)) {
          const g = offlineGreeting(text) || offlineMath(question);
          if (g) {
            pushAssistant(sid, g, { badge: "📴 Offline", sources: [] });
          } else {
            pushAssistant(
              sid,
              `❌ **Lỗi kết nối:** Không kết nối được máy chủ ở \`http://localhost:8000\`. Hãy kiểm tra bạn đang chạy \`uvicorn backend.api:app --port 8000\` và Ollama đang chạy.\n\n\`${String(
                e instanceof Error ? e.message : e
              ).slice(0, 800)}\``,
              { badge: "❌ Lỗi", sources: [] }
            );
          }
        } else {
          const detail = e instanceof ApiError ? e.detail : String(e);
          pushAssistant(sid, `❌ Lỗi: ${detail.slice(0, 1200)}`, { badge: "❌ Lỗi", sources: [] });
        }
      } finally {
        if (streamAbortRef.current === ctrl) streamAbortRef.current = null;
        setLoading(false);
        setStatus("");
      }
    },
    [loading, style, pushAssistant]
  );

  /* ---------- upload ---------- */
  const attach = useCallback(
    async (file: File) => {
      const sid = activeRef.current;
      if (!sid) {
        setStatus("Hãy tạo phiên trước khi tải file");
        return;
      }
      const ext = ("." + (file.name.split(".").pop() || "").toLowerCase());
      if (![".pdf", ".docx", ".txt", ".md", ".csv"].includes(ext)) {
        window.alert("Chỉ hỗ trợ: .pdf, .docx, .txt, .md, .csv");
        return;
      }
      if (file.size > 100 * 1024 * 1024) {
        window.alert("File vượt quá 100MB");
        return;
      }
      setBusy(true);
      setStatus("⏳ Đang tải file...");
      const note = { role: "user" as const, content: `📎 Đã tải file: ${file.name}`, key: mkey() };
      setDisplay((prev) => [...prev, note]);
      try {
        const j = await api.uploadFile(sid, file);
        const msg = j.auto_message || `Đã lập chỉ mục ${j.chunks || 0} đoạn từ ${j.filename}.`;
        pushAssistant(sid, msg, { badge: `📎 ${j.filename}`, sources: j.sources || [] });
        setSessions((prev) => {
          const s = prev[sid];
          if (!s) return prev;
          return {
            ...prev,
            [sid]: {
              ...s,
              updatedAt: Date.now(),
              conversation: [
                ...s.conversation,
                { role: "user" as const, content: note.content },
                { role: "assistant" as const, content: msg },
              ].slice(-MAX_KEEP),
            },
          };
        });
        loadUploads(sid);
      } catch (e) {
        const detail = e instanceof ApiError ? e.detail : String(e);
        pushAssistant(sid, `❌ Tải file thất bại: ${detail.slice(0, 800)}`, {
          badge: "❌ Lỗi",
          sources: [],
        });
      } finally {
        setBusy(false);
        setStatus("");
      }
    },
    [pushAssistant, loadUploads]
  );

  const deleteUpload = useCallback(
    async (filename: string) => {
      const sid = activeRef.current;
      if (!sid) return;
      try {
        await api.deleteUpload(sid, filename);
        loadUploads(sid);
      } catch (e) {
        window.alert(`Xóa thất bại: ${e instanceof ApiError ? e.detail : e}`);
      }
    },
    [loadUploads]
  );

  const clearUploads = useCallback(async () => {
    const sid = activeRef.current;
    if (!sid) return;
    for (const f of uploads) {
      try {
        await api.deleteUpload(sid, f.filename);
      } catch {
        /* tiếp tục file khác */
      }
    }
    loadUploads(sid);
  }, [uploads, loadUploads]);

  /* ---------- auth ---------- */
  const doLogin = useCallback(async (u: string, p: string): Promise<string | null> => {
    try {
      const j = await api.login(u, p);
      setToken(j.token);
      try {
        localStorage.removeItem(LS_SESSIONS);
        localStorage.removeItem(LS_ACTIVE);
      } catch {
        /* bỏ qua */
      }
      window.location.reload();
      return null;
    } catch (e) {
      return e instanceof ApiError ? e.detail : "Đăng nhập thất bại";
    }
  }, []);

  const loginClick = useCallback(() => {
    if (getToken()) {
      api.logout().catch(() => undefined);
      setToken(null);
      try {
        localStorage.removeItem(LS_SESSIONS);
        localStorage.removeItem(LS_ACTIVE);
      } catch {
        /* bỏ qua */
      }
      window.location.reload();
    } else {
      setLoginOpen(true);
    }
  }, []);

  const openStats = useCallback(async () => {
    try {
      const s = await api.stats();
      setStatsData(s);
    } catch (e) {
      setStatsData({ error: e instanceof ApiError ? e.detail : String(e) } as StatsResponse);
    }
  }, []);

  const showWelcome = display.length === 0;

  return (
    <div className="app-shell">
      <aside className={`sidebar${sideOpen ? " open" : ""}`}>
        <Sidebar
          sessions={sessions}
          activeId={activeId}
          turns={turns}
          healthText={healthText}
          healthOk={healthOk}
          llmInfo={llmInfo}
          accountName={accountName}
          authed={authed}
          sessionInfo={sessionInfo}
          onNewChat={newChat}
          onSelect={selectSession}
          onDelete={deleteSession}
          onLoginClick={loginClick}
          onStatsClick={openStats}
        />
      </aside>
      <main className="main-panel">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-btn mobile" title="Mở thanh bên" onClick={() => setSideOpen(true)}>
              <i className="fa-solid fa-bars"></i>
            </button>
            <div className="workspace-title">
              <span className="eyebrow">Không gian hỏi đáp</span>
              <strong>Trợ lý học viện</strong>
            </div>
          </div>
          <div className="topbar-actions">
            <span className="secure-label">
              <i className="fa-solid fa-shield-halved"></i> Dữ liệu nội bộ
            </span>
            <a className="icon-btn" href="/docs" target="_blank" title="API Docs">
              <i className="fa-solid fa-code"></i>
            </a>
            <a className="icon-btn" href="/api/health" target="_blank" title="Trạng thái hệ thống">
              <i className="fa-solid fa-heart-pulse"></i>
            </a>
            <button
              className="icon-btn"
              title="Nguồn tham khảo"
              onClick={() => setRightOpen((v) => !v)}
            >
              <i className="fa-solid fa-book-open"></i>
            </button>
          </div>
        </header>
        <section className="chat-stage">
          <div ref={msgBoxRef} className="messages">
            {showWelcome && <Welcome onAsk={send} />}
            {!showWelcome &&
              display.map((m) => (
                <ChatMessage key={m.key} msg={m} typing={m.key === typingKey} />
              ))}
            {loading && <div className="hint">Đang suy nghĩ...</div>}
          </div>
          {uploads.length > 0 && (
            <UploadsBar
              files={uploads}
              totalChunks={totalChunks}
              onDelete={deleteUpload}
              onClearAll={clearUploads}
            />
          )}
          <Composer loading={loading} busy={busy} status={status} onSend={send} onAttach={attach} />
        </section>
      </main>
      <aside className={`sources-panel${rightOpen ? " open" : ""}`}>
        <div className="panel-header">
          <div>
            <span className="eyebrow">Tài liệu tham chiếu</span>
            <h2>
              <i className="fa-solid fa-link"></i> Nguồn trả lời
            </h2>
          </div>
          <button className="icon-btn" title="Đóng nguồn" onClick={() => setRightOpen(false)}>
            <i className="fa-solid fa-xmark"></i>
          </button>
        </div>
        <SourcesPanel meta={panelMeta} />
      </aside>
      <div
        className={`sources-backdrop${rightOpen ? "" : " hidden"}`}
        onClick={() => setRightOpen(false)}
      ></div>
      {sideOpen && <div className="sources-backdrop" onClick={() => setSideOpen(false)}></div>}
      <StatsModal stats={statsData} onClose={() => setStatsData(null)} />
      <LoginModal open={loginOpen} onClose={() => setLoginOpen(false)} onLogin={doLogin} />
    </div>
  );
}
