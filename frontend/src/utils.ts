// Tiện ích thuần: parse category, vị trí nguồn, fallback offline.
import type { ChatSource } from "./types";

export const KNOWN_CATEGORIES = [
  "quyet_dinh",
  "quy_che",
  "quy_dinh",
  "tai_lieu_huong_dan",
  "thong_bao",
];

const CAT_RE = /^(\w+):\s*(.+)$/s;

/** "quy_che: câu hỏi" -> {category, question}. Giữ đúng hành vi cũ. */
export function parseCategory(raw: string): { category?: string; question: string } {
  const m = raw.match(CAT_RE);
  if (m && KNOWN_CATEGORIES.includes(m[1])) return { category: m[1], question: m[2] };
  return { question: raw };
}

/** Chuỗi vị trí nguồn: ưu tiên section -> trang -> khối. */
export function formatLoc(s: ChatSource): string {
  if (s.section) {
    let extra = "";
    if (s.table_index !== undefined && s.table_index !== null) extra = ` (bảng ${s.table_index})`;
    else if (s.paragraph_index !== undefined && s.paragraph_index !== null)
      extra = ` (đoạn ${s.paragraph_index})`;
    return `${s.section}${extra}`;
  }
  if (s.page !== undefined && s.page !== null) return `trang ${s.page}`;
  if (s.block_index !== undefined && s.block_index !== null)
    return `khối ${s.block_index}${s.block_type ? ` ${s.block_type}` : ""}`;
  return s.source || "";
}

const GREETINGS = [
  "xin chào",
  "xin chao",
  "chào",
  "chao",
  "hello",
  "hi",
  "chào bạn",
  "hey",
];

/** Fallback offline khi mất mạng: câu chào cố định. */
export function offlineGreeting(q: string): string | null {
  const t = q.trim().toLowerCase();
  if (GREETINGS.includes(t))
    return "Xin chào! Tôi là trợ lý tài liệu của Học viện. Mạng đang gián đoạn nên tôi chưa tra cứu được tài liệu — bạn thử lại sau nhé.";
  return null;
}

const SAFE_MATH_RE = /^[0-9.+\-*/%() ]+$/;

/** Fallback offline: phép tính số học an toàn (chỉ khi khớp regex ký tự). */
export function offlineMath(q: string): string | null {
  const t = q.trim();
  if (!SAFE_MATH_RE.test(t) || !/[0-9]/.test(t) || t.length > 60) return null;
  try {
    // eslint-disable-next-line no-new-func
    const val = Function(`"use strict"; return (${t})`)() as number;
    if (typeof val !== "number" || !isFinite(val)) return null;
    return `Kết quả (tính offline): ${t} = ${val}`;
  } catch {
    return null;
  }
}

export function isNetworkError(e: unknown): boolean {
  const msg = e instanceof Error ? e.message : String(e);
  return /failed to fetch|networkerror|load failed|network request failed/i.test(msg);
}

export function shortId(id: string): string {
  return id.length > 8 ? `${id.slice(0, 8)}...` : id;
}

export function fmtTime(ms: number): string {
  try {
    return new Date(ms).toLocaleString("vi-VN");
  } catch {
    return "";
  }
}
