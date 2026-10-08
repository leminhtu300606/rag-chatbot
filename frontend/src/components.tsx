import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";
import { useTypewriter } from "./hooks";
import { fmtTime, formatLoc, shortId } from "./utils";
import type {
  AssistantMeta,
  DisplayMsg,
  LocalSession,
  StatsResponse,
  UploadFileInfo,
} from "./types";

/* ---------- Markdown an toàn (thay marked + DOMPurify) ---------- */
export function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]}>
      {text}
    </ReactMarkdown>
  );
}

/* ---------- Sidebar ---------- */
interface SidebarProps {
  sessions: Record<string, LocalSession>;
  activeId: string | null;
  turns: number;
  healthText: string;
  healthOk: boolean | null;
  llmInfo: string;
  accountName: string;
  authed: boolean;
  sessionInfo: string;
  onNewChat: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onLoginClick: () => void;
  onStatsClick: () => void;
}

export function Sidebar(p: SidebarProps) {
  const list = Object.entries(p.sessions)
    .sort((a, b) => b[1].updatedAt - a[1].updatedAt)
    .slice(0, 20);
  return (
    <>
      <div className="sidebar-header">
        <a className="brand" href="/" aria-label="Trang chủ">
          <span className="brand-mark">
            <i className="fa-solid fa-sparkles"></i>
          </span>
          <span>
            <strong>Học viện</strong>
            <small>Trợ lý tài liệu</small>
          </span>
        </a>
      </div>
      <button className="new-chat" onClick={p.onNewChat}>
        <i className="fa-solid fa-plus"></i>
        <span>Cuộc trò chuyện mới</span>
        <kbd>Ctrl K</kbd>
      </button>
      <div className="sidebar-label">Các cuộc trò chuyện</div>
      <nav className="history-list" aria-label="Lịch sử hội thoại">
        {list.map(([id, s]) => {
          const turns = Math.floor(s.conversation.length / 2);
          return (
            <div
              key={id}
              className={`history-item${id === p.activeId ? " active" : ""}`}
              onClick={() => p.onSelect(id)}
              title={fmtTime(s.updatedAt)}
            >
              <div className="history-main">
                <div className="history-title">{s.title}</div>
                <div className="hint">
                  {turns} lượt{turns >= 12 ? " • sắp tóm tắt" : ""} • {shortId(id)}
                </div>
              </div>
              <button
                className="icon-btn del-btn"
                title="Xóa phiên"
                onClick={(e) => {
                  e.stopPropagation();
                  p.onDelete(id);
                }}
              >
                <i className="fa-solid fa-trash"></i>
              </button>
            </div>
          );
        })}
      </nav>
      <div className="sidebar-footer">
        <div className="system-status">
          <span id="health-badge" className={`dot ${p.healthOk === null ? "warn" : p.healthOk ? "ok" : "bad"}`}></span>
          <div>
            <strong>{p.healthText}</strong>
            <span>
              {p.turns} lượt • {p.activeId ? "phiên đang mở" : "phiên mới"}
            </span>
          </div>
        </div>
        <div className="account-row">
          <span className="account-avatar">HV</span>
          <span className="account-copy">
            <strong>{p.accountName}</strong>
            <small>{p.llmInfo}</small>
          </span>
          <button className="icon-btn" onClick={p.onLoginClick} title="Đăng nhập/đăng xuất">
            <i className={`fa-solid ${p.authed ? "fa-right-from-bracket" : "fa-user"}`}></i>
          </button>
          <button className="icon-btn" onClick={p.onStatsClick} title="Thống kê hệ thống">
            <i className="fa-solid fa-ellipsis"></i>
          </button>
        </div>
        <div className="session-info">{p.sessionInfo}</div>
      </div>
    </>
  );
}

/* ---------- Welcome + examples ---------- */
const EXAMPLES: { q: string; icon: string; color: string; title: string; sub: string }[] = [
  { q: "Quy chế đào tạo là gì?", icon: "fa-book-open", color: "mint", title: "Quy chế đào tạo", sub: "Điều kiện, quy trình và quyền lợi" },
  { q: "Quy định khảo thí như thế nào?", icon: "fa-clipboard-check", color: "blue", title: "Quy định khảo thí", sub: "Thi, kiểm tra và phúc khảo" },
  { q: "Học bổng KKHT xét thế nào?", icon: "fa-award", color: "gold", title: "Học bổng", sub: "Tiêu chí xét học bổng KKHT" },
  { q: "Điều kiện tốt nghiệp?", icon: "fa-graduation-cap", color: "coral", title: "Tốt nghiệp", sub: "Điều kiện và hồ sơ cần chuẩn bị" },
];

export function Welcome({ onAsk }: { onAsk: (q: string) => void }) {
  return (
    <div className="welcome">
      <div className="welcome-kicker">
        <span className="pulse-dot"></span> Sẵn sàng hỗ trợ
      </div>
      <h1>
        Hôm nay bạn muốn
        <br />
        <em>tìm hiểu điều gì?</em>
      </h1>
      <p>Tra cứu quy chế, quy định và thông báo của Học viện bằng ngôn ngữ tự nhiên.</p>
      <div className="examples">
        {EXAMPLES.map((e) => (
          <button key={e.q} className="example" onClick={() => onAsk(e.q)}>
            <span className={`example-icon ${e.color}`}>
              <i className={`fa-solid ${e.icon}`}></i>
            </span>
            <span>
              <b>{e.title}</b>
              <small>{e.sub}</small>
            </span>
            <i className="fa-solid fa-arrow-up-right"></i>
          </button>
        ))}
      </div>
      <div className="welcome-note">
        <i className="fa-solid fa-circle-info"></i> Câu trả lời được tạo từ tài liệu đã lập chỉ mục và luôn kèm nguồn tham khảo.
      </div>
    </div>
  );
}

/* ---------- Bubble chat ---------- */
export function ChatMessage({ msg, typing }: { msg: DisplayMsg; typing: boolean }) {
  const [shown, done, skip] = useTypewriter(msg.content, typing && msg.role === "assistant");
  const isUser = msg.role === "user";
  return (
    <div className={`msg-row ${isUser ? "user" : "assistant"}`}>
      <div className="msg-inner">
        <div className="bubble" onClick={() => !done && skip()}>
          {isUser ? (
            <>
              <div className="bubble-text">{msg.content}</div>
              {msg.category && <div className="meta">#{msg.category}</div>}
            </>
          ) : typing && !done ? (
            <div className="bubble-text typing">
              {shown}
              <span className="cursor">▍</span>
            </div>
          ) : (
            <div className="bubble-text">
              <Markdown text={msg.content} />
            </div>
          )}
          {msg.meta && done && (
            <div className="meta">
              {msg.meta.badge} {msg.meta.rewrite && <span> • {msg.meta.rewrite}</span>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------- Composer ---------- */
interface ComposerProps {
  loading: boolean;
  busy: boolean;
  status: string;
  onSend: (text: string) => void;
  onAttach: (f: File) => void;
}

export function Composer({ loading, busy, status, onSend, onAttach }: ComposerProps) {
  const [text, setText] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const ta = taRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
    }
  }, [text]);

  const submit = () => {
    const t = text.trim();
    if (!t || loading || busy) return;
    setText("");
    onSend(t);
  };

  return (
    <div className="composer-area">
      <div className="composer">
        <div className="composer-inner">
          <div className="input-box">
            <button
              className="attach-btn"
              title="Đính kèm PDF, DOCX, TXT hoặc MD"
              disabled={busy}
              onClick={() => fileRef.current?.click()}
            >
              <i className="fa-solid fa-paperclip"></i>
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".pdf,.docx,.txt,.md,.csv"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                e.target.value = "";
                if (f) onAttach(f);
              }}
            />
            <textarea
              ref={taRef}
              rows={1}
              placeholder="Hỏi bất cứ điều gì về tài liệu học viện..."
              value={text}
              disabled={loading || busy}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
            />
            <button className="send-btn" aria-label="Gửi câu hỏi" disabled={loading || busy} onClick={submit}>
              <i className="fa-solid fa-arrow-up"></i>
            </button>
          </div>
          <div className="composer-foot">
            <span className="hint">{status}</span>
            <span className="hint center">Enter để gửi · Shift + Enter để xuống dòng</span>
            <span className="hint">
              <i className="fa-solid fa-wand-magic-sparkles"></i> RAG
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- Panel nguồn ---------- */
export function SourcesPanel({ meta }: { meta: AssistantMeta | null }) {
  if (!meta) {
    return (
      <div className="sources-list">
        <div className="empty-panel">
          <i className="fa-regular fa-file-lines"></i>
          <p>Nguồn tham khảo sẽ xuất hiện ở đây sau mỗi câu hỏi.</p>
        </div>
      </div>
    );
  }
  return (
    <>
      <div className="sources-list">
        {meta.sources.length === 0 && <div className="empty-panel"><p>{meta.badge}</p></div>}
        {meta.sources.map((s, i) => (
          <div key={i} className="source-item">
            <b>{s.filename || "(không rõ file)"}</b>
            <div className="hint">
              {formatLoc(s)}
              {s.category ? ` • ${s.category}` : ""}
              {s.subcategory ? `/${s.subcategory}` : ""}
            </div>
          </div>
        ))}
      </div>
      {meta.rewrite && <div className="hint">{meta.rewrite}</div>}
      {meta.summary && (
        <div className="hint">
          <b>📝 Tóm tắt hội thoại cũ</b> {meta.summary}
        </div>
      )}
    </>
  );
}

/* ---------- Uploads bar ---------- */
interface UploadsBarProps {
  files: UploadFileInfo[];
  totalChunks: number;
  onDelete: (filename: string) => void;
  onClearAll: () => void;
}

export function UploadsBar({ files, totalChunks, onDelete, onClearAll }: UploadsBarProps) {
  if (files.length === 0) return null;
  return (
    <div className="uploads-bar">
      <div className="uploads-head">
        <span>
          <i className="fa-solid fa-paperclip"></i> Tài liệu trong phiên
        </span>
        <span className="hint">
          {files.length} file • {totalChunks} đoạn
        </span>
        <button
          onClick={() => {
            if (window.confirm("Xóa tất cả tài liệu trong phiên?")) onClearAll();
          }}
          title="Xóa tài liệu"
        >
          Xóa tất cả
        </button>
      </div>
      <div className="uploads-list">
        {files.map((f) => (
          <div key={f.filename} className="upload-item">
            <span>
              {f.filename} <span className="hint">({f.chunks} đoạn)</span>
            </span>
            <button
              className="icon-btn del-btn"
              title="Xóa file"
              onClick={() => {
                if (window.confirm(`Xóa file ${f.filename}?`)) onDelete(f.filename);
              }}
            >
              <i className="fa-solid fa-xmark"></i>
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ---------- Modal đăng nhập ---------- */
export function LoginModal({
  open,
  onClose,
  onLogin,
}: {
  open: boolean;
  onClose: () => void;
  onLogin: (u: string, p: string) => Promise<string | null>;
}) {
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-title">
          <span className="example-icon mint">
            <i className="fa-solid fa-user"></i>
          </span>
          <h3>Đăng nhập</h3>
        </div>
        <div className="login-fields">
          <input autoComplete="username" placeholder="Tên đăng nhập" value={user} onChange={(e) => setUser(e.target.value)} />
          <input
            type="password"
            autoComplete="current-password"
            placeholder="Mật khẩu"
            value={pass}
            onChange={(e) => setPass(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                (async () => {
                  setBusy(true);
                  setErr("");
                  const m = await onLogin(user, pass);
                  setBusy(false);
                  if (m) setErr(m);
                })();
              }
            }}
          />
          <div className="hint">{err}</div>
        </div>
        <div className="login-actions">
          <button
            className="btn"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              setErr("");
              const m = await onLogin(user, pass);
              setBusy(false);
              if (m) setErr(m);
            }}
          >
            Đăng nhập
          </button>
          <button className="btn ghost" onClick={onClose}>
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
}

/* ---------- Modal thống kê ---------- */
export function StatsModal({ stats, onClose }: { stats: StatsResponse | null; onClose: () => void }) {
  if (!stats) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-title">
          <span className="example-icon blue">
            <i className="fa-solid fa-chart-simple"></i>
          </span>
          <h3>Thống kê hệ thống</h3>
        </div>
        <pre>{JSON.stringify(stats, null, 2)}</pre>
        <button className="btn" onClick={onClose}>
          Đóng
        </button>
      </div>
    </div>
  );
}
