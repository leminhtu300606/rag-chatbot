import { useCallback, useEffect, useRef, useState } from "react";

export const TYPEWRITER_SPEED = 12; // ms/ký tự, giữ như bản cũ

function stepDelay(ch: string): number {
  if (ch === " ") return TYPEWRITER_SPEED * 0.4;
  if (ch === "\n" || ".!?".includes(ch)) return TYPEWRITER_SPEED * 7;
  if (",;:".includes(ch)) return TYPEWRITER_SPEED * 3;
  return TYPEWRITER_SPEED;
}

/**
 * Hiệu ứng gõ chữ. Trả về [textĐãHiện, done, skip].
 * Click vào bubble gọi skip() để hiện ngay toàn bộ (hành vi cũ).
 */
export function useTypewriter(full: string, active: boolean): [string, boolean, () => void] {
  const [shown, setShown] = useState(active ? "" : full);
  const [done, setDone] = useState(!active);
  const idx = useRef(0);
  const timer = useRef<number | null>(null);

  const clear = () => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  };

  const skip = useCallback(() => {
    clear();
    setShown(full);
    setDone(true);
  }, [full]);

  useEffect(() => {
    if (!active) {
      setShown(full);
      setDone(true);
      return;
    }
    idx.current = 0;
    setShown("");
    setDone(false);
    const tick = () => {
      idx.current += 1;
      setShown(full.slice(0, idx.current));
      if (idx.current >= full.length) {
        setDone(true);
        timer.current = null;
        return;
      }
      timer.current = window.setTimeout(tick, stepDelay(full[idx.current - 1] || ""));
    };
    timer.current = window.setTimeout(tick, stepDelay(full[0] || ""));
    return clear;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [full, active]);

  useEffect(() => clear, []);
  return [shown, done, skip];
}

/** Trạng thái "đang gõ..." cho composer. */
export function useNow(): number {
  const [, setTick] = useState(0);
  useEffect(() => {
    const t = window.setInterval(() => setTick((x) => x + 1), 1000);
    return () => window.clearInterval(t);
  }, []);
  return Date.now();
}
