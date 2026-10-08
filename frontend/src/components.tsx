import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";
import { useTypewriter } from "./hooks";
import { formatLoc } from "./utils";
import type {
  AssistantMeta,
  DisplayMsg,
  UploadFileInfo,
} from "./types";

/* ---------- Markdown an toàn ---------- */
export function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]}>
      {text}
    </ReactMarkdown>
  );
}

/* ---------- Header gọn: nút sidebar + bot ---------- */
interface WidgetHeaderProps {
  online: boolean | null;
  healthText: string;
  onToggleSidebar: () => void;
}

export function WidgetHeader(p: WidgetHeaderProps) {
  return (
    <header className="widget-header">
      <button className="hbtn" title="Đóng/mở lịch sử" onClick={p.onToggleSidebar}>
        <i className="fa-solid fa-bars"></i>
      </button>
      <span className="bot-avatar">
        <i className="fa-solid fa-robot"></i>
      </span>
      <div className="bot-id">
        <strong>Trợ lý Học viện</strong>
        <span className="bot-status" title={p.healthText}>
          <span className={`status-dot${p.online ? " on" : ""}`}></span>
          {p.online ? "Đang trực tuyến" : p.online === false ? "Mất kết nối" : "Đang kết nối..."}
        </span>
      </div>
    </header>
  );
}

/* ---------- Sidebar lịch sử kiểu ChatGPT ---------- */
function dayGroup(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  const startOf = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = startOf(now) - startOf(d);
  if (diff <= 0) return "Hôm nay";
  if (diff <= 86400000) return "Hôm qua";
  if (diff <= 7 * 86400000) return "7 ngày qua";
  return "Cũ hơn";
}

interface SideHistoryProps {
  open: boolean;
  sessions: Record<string, { title: string; updatedAt: number; conversation: { role: string }[] }>;
  activeId: string | null;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export function SideHistory(p: SideHistoryProps) {
  if (!p.open) return null;
  const groups = new Map<string, [string, { title: string; updatedAt: number; conversation: { role: string }[] }][]>();
  for (const [id, s] of Object.entries(p.sessions).sort((a, b) => b[1].updatedAt - a[1].updatedAt).slice(0, 50)) {
    const g = dayGroup(s.updatedAt || Date.now());
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g)!.push([id, s]);
  }
  return (
    <aside className="side-history">
      <button className="side-new" onClick={p.onNew}>
        <i className="fa-solid fa-pen-to-square"></i> Đoạn chat mới
      </button>
      <nav className="side-list" aria-label="Lịch sử hội thoại">
        {groups.size === 0 && (
          <div className="hint" style={{ padding: "12px 10px" }}>Chưa có cuộc trò chuyện nào.</div>
        )}
        {[...groups.entries()].map(([g, items]) => (
          <div key={g}>
            <div className="side-group">{g}</div>
            {items.map(([id, s]) => (
              <div
                key={id}
                className={`side-item${id === p.activeId ? " active" : ""}`}
                onClick={() => p.onSelect(id)}
                title={s.title}
              >
                <span className="side-item-title">{s.title}</span>
                <button
                  className="tool-btn side-del"
                  title="Xóa đoạn chat"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (window.confirm("Xóa đoạn chat này?")) p.onDelete(id);
                  }}
                >
                  <i className="fa-solid fa-trash"></i>
                </button>
              </div>
            ))}
          </div>
        ))}
      </nav>
      <div className="side-foot hint">Trợ lý tài liệu • Học viện KTMM</div>
    </aside>
  );
}

/* ---------- Chào mừng: bubble bot + quick reply ---------- */
const QUICK_REPLIES: { q: string; solid?: boolean }[] = [
  { q: "Quy chế đào tạo là gì?", solid: true },
  { q: "Học bổng KKHT xét thế nào?" },
  { q: "Điều kiện tốt nghiệp?" },
];

export function Welcome({ onAsk }: { onAsk: (q: string) => void }) {
  return (
    <>
      <div className="msg-row assistant">
        <div className="msg-inner">
          <div className="bubble">
            <div className="bubble-text">
              Xin chào! 👋 Tôi là trợ lý tài liệu của Học viện Kỹ thuật Mật mã.
              Bạn muốn hỏi về quy chế, quy định hay thông báo nào?
            </div>
          </div>
        </div>
      </div>
      <div className="quick-replies">
        {QUICK_REPLIES.map((r) => (
          <button key={r.q} className={`qr${r.solid ? " solid" : " line"}`} onClick={() => onAsk(r.q)}>
            {r.q}
          </button>
        ))}
      </div>
    </>
  );
}

/* ---------- Bubble chat + nguồn mở rộng inline ---------- */
function SourcesInline({ meta }: { meta: AssistantMeta }) {
  if (!meta.sources || meta.sources.length === 0) return null;
  return (
    <details className="src-details">
      <summary>📚 {meta.sources.length} nguồn tham khảo</summary>
      {meta.sources.map((s, i) => (
        <div key={i} className="src-line">
          <b>{s.filename || "(không rõ file)"}</b>
          <div className="hint">
            {formatLoc(s)}
            {s.category ? ` • ${s.category}` : ""}
          </div>
        </div>
      ))}
    </details>
  );
}

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
            <>
              <div className="meta" title={msg.meta.rewrite || ""}>
                {msg.meta.badge}
              </div>
              <SourcesInline meta={msg.meta} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------- Composer: ô nhập + gửi | emoji + đính kèm ---------- */
const EMOJIS = ["😀", "😁", "😂", "🥰", "😮", "😢", "😡", "👍", "👏", "🙏", "🎓", "📚", "📄", "✅", "❌", "❓", "💡", "🎉", "🔥", "⭐", "❤️", "👋", "🤔", "👌"];

interface ComposerProps {
  loading: boolean;
  busy: boolean;
  status: string;
  onSend: (text: string) => void;
  onAttach: (f: File) => void;
}

export function Composer({ loading, busy, status, onSend, onAttach }: ComposerProps) {
  const [text, setText] = useState("");
  const [emojiOpen, setEmojiOpen] = useState(false);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const ta = taRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 110) + "px";
    }
  }, [text]);

  const submit = () => {
    const t = text.trim();
    if (!t || loading || busy) return;
    setText("");
    setEmojiOpen(false);
    onSend(t);
  };

  const insertEmoji = (e: string) => {
    setText((prev) => prev + e);
    taRef.current?.focus();
  };

  return (
    <div className="composer">
      <div className="input-pill">
        <textarea
          ref={taRef}
          rows={1}
          placeholder="Nhập tin nhắn..."
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
        <button className="send-btn" aria-label="Gửi tin nhắn" disabled={loading || busy} onClick={submit}>
          <i className="fa-solid fa-paper-plane"></i>
        </button>
      </div>
      <div className="tools-row">
        <button
          className="tool-btn"
          title="Chọn emoji"
          onClick={() => setEmojiOpen((v) => !v)}
        >
          <i className="fa-regular fa-face-smile"></i>
        </button>
        <button
          className="tool-btn"
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
        <span className="hint tools-hint">{status}</span>
      </div>
      {emojiOpen && (
        <div className="emoji-pop">
          {EMOJIS.map((e) => (
            <button key={e} onClick={() => insertEmoji(e)}>
              {e}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---------- Uploads bar gọn ---------- */
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
