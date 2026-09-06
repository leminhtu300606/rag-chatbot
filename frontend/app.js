const $ = s => document.querySelector(s);
const messages = $("#messages");
const welcome = $("#welcome");
const input = $("#input");
const sendBtn = $("#send");
const statusEl = $("#status");
const sourcesEl = $("#sources");
const contextEl = $("#context");
const contextList = $("#context-list");
const healthDot = $("#health-badge");
const healthText = $("#health-text");
const llmInfo = $("#llm-info");
const modelFoot = $("#model-foot");
const historyBadge = $("#history-badge");
const sessionInfo = $("#session-info");
const rewriteInfo = $("#rewrite-info");
const summaryInfo = $("#summary-info");
const useHistoryEl = $("#useHistory");

let isLoading = false;

// ── Typewriter: đặt sớm để tránh lỗi TDZ khi new-chat gọi trước ──
const TYPEWRITER_SPEED = 18;
let _typingAbort = null;

 // Quản lý hội thoại: lưu session và lịch sử trên localStorage
const LS_SESSION = "rag_session_id";
const LS_CONV = "rag_conversation_v2";
let sessionId = localStorage.getItem(LS_SESSION) || null;
let conversation = []; // [{role:"user"|"assistant", content:"..."}]
try {
  const raw = localStorage.getItem(LS_CONV);
  if (raw) {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) conversation = parsed;
  }
} catch(e) { conversation = []; }

function saveConversation() {
  try {
    // Giu toi da 60 messages (30 turns) de tranh qua localStorage limit
    const toSave = conversation.slice(-60);
    localStorage.setItem(LS_CONV, JSON.stringify(toSave));
    if (sessionId) localStorage.setItem(LS_SESSION, sessionId);
  } catch(e) {}
  updateHistoryBadge();
}

function updateHistoryBadge() {
  const turns = Math.floor(conversation.length / 2);
  const sumHint = turns >= 12 ? " • sắp tóm tắt" : "";
  if (historyBadge) historyBadge.textContent = `💬 ${turns} lượt${sumHint} • ${sessionId ? "đã lưu" : "chưa có session"}`;
  if (sessionInfo) sessionInfo.textContent = sessionId ? `session: ${sessionId.slice(0,8)}...` : "";
}

function clearConversationLocal() {
  conversation = [];
  localStorage.removeItem(LS_CONV);
  updateHistoryBadge();
  if (rewriteInfo) { rewriteInfo.classList.add("hidden"); rewriteInfo.textContent = ""; }
  if (summaryInfo) { summaryInfo.classList.add("hidden"); summaryInfo.textContent = ""; }
}

function renderConversationFromStorage() {
  if (!conversation.length) return;
  if (!welcome || !messages) return;
  welcome.style.display = "none";
  // Clear existing msg rows (giữ welcome)
  messages.querySelectorAll(".msg-row").forEach(el => el.remove());
  conversation.forEach(msg => {
    const role = msg.role === "user" ? "user" : "assistant";
    addMsg(role, msg.content, "");
  });
  // Rebuild sidebar history list tu conversation
  const list = $("#history-list");
  if (list) {
    list.innerHTML = "";
    const userQs = conversation.filter(m => m.role === "user").map(m => m.content);
    if (userQs.length === 0) {
      list.innerHTML = '<div class="history-item active"><i class="fa-regular fa-message"></i><span>Cuộc trò chuyện mới</span></div>';
    } else {
      userQs.slice(-10).reverse().forEach((q, idx) => {
        const div = document.createElement("div");
        div.className = "history-item" + (idx===0 ? " active" : "");
        div.innerHTML = `<i class="fa-regular fa-message"></i><span>${escapeHtml(q.slice(0,32))}</span>`;
        list.appendChild(div);
      });
    }
  }
}

// ── Hàm dùng chung để bắt đầu phiên mới (sửa lỗi nút không ấn được) ──
async function startNewChat(opts = {}) {
  const { skipServerDelete = false, silent = false } = opts;
  try {
    if(_typingAbort) try{ _typingAbort(); }catch(e){}
    // Xoa server session neu co (bỏ qua lỗi 404/offline)
    if (sessionId && !skipServerDelete) {
      try { await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {method:"DELETE"}); } catch(e) {}
    }
  } catch(e) {}

  // Hủy session cũ ở client
  sessionId = null;
  try { localStorage.removeItem(LS_SESSION); } catch(e) {}
  clearConversationLocal();
  // Tạo session mới ngay lập tức để UI hiển thị "đã lưu" (phiên mới) - mặc định sang phiên mới khi chạy dự án
  try {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      sessionId = crypto.randomUUID();
    } else {
      sessionId = 'sess-' + Date.now() + '-' + Math.random().toString(36).slice(2,9);
    }
    localStorage.setItem(LS_SESSION, sessionId);
  } catch(e) {
    sessionId = null;
  }
  if (messages) messages.querySelectorAll(".msg-row").forEach(el=>el.remove());
  if (welcome) welcome.style.display = "block";
  if (sourcesEl) sourcesEl.innerHTML = '<p class="hint">Chưa có truy vấn.</p>';
  if (contextEl) contextEl.classList.add("hidden");
  if (rewriteInfo) { rewriteInfo.classList.add("hidden"); rewriteInfo.textContent=""; }
  if (summaryInfo) { summaryInfo.classList.add("hidden"); summaryInfo.textContent=""; }
  const list = $("#history-list");
  if (list) list.innerHTML = '<div class="history-item active"><i class="fa-regular fa-message"></i><span>Cuộc trò chuyện mới</span></div>';
  updateHistoryBadge();
  if (input) { input.value=""; input.style.height="auto"; input.focus(); }
  if (statusEl && !silent) setStatus("");
  // Đóng sidebar trên mobile sau khi tạo mới
  $("#sidebar")?.classList.remove("open");
  if (!silent) console.log("[RAG] Đã tạo phiên chat mới");
}

function shouldAutoStartNewSession() {
  try {
    const params = new URLSearchParams(window.location.search);
    if (params.has("restore") || params.has("keep")) return false;
    if (localStorage.getItem("rag_keep_session")==="1") return false;
  } catch(e) {}
  // Mặc định: luôn sang phiên mới khi chạy dự án (theo yêu cầu)
  return true;
}

// Khởi tạo giao diện: mặc định sang phiên mới thay vì khôi phục cũ
function initConversationOnLoad() {
  if (shouldAutoStartNewSession()) {
    console.log("[RAG] Mặc định khởi động -> tạo phiên chat mới");
    // Luôn tạo phiên mới khi chạy dự án (theo yêu cầu), xóa lịch sử cũ nếu có
    startNewChat({skipServerDelete: true, silent: true});
    return;
  }
  updateHistoryBadge();
  renderConversationFromStorage();
}
initConversationOnLoad();

// Sidebar toggle (ChatGPT mobile)
$("#open-sidebar")?.addEventListener("click", ()=> $("#sidebar").classList.add("open"));
$("#close-sidebar")?.addEventListener("click", ()=> $("#sidebar").classList.remove("open"));
$("#close-right")?.addEventListener("click", ()=> { const rp=$("#right-panel"); if(rp) rp.style.display="none"; });

// New chat - xoa ca server session va local (bản sửa lỗi)
// 1) Gắn trực tiếp
$("#new-chat")?.addEventListener("click", async (e)=>{
  e.preventDefault();
  await startNewChat();
});
// 2) Delegation dự phòng nếu DOM thay đổi hoặc element bị replace
document.addEventListener("click", (e)=>{
  const btn = e.target.closest("#new-chat");
  if(btn){
    e.preventDefault();
    // Tránh double-call nếu đã xử lý ở trên
    if(e.__newChatHandled) return;
    e.__newChatHandled = true;
    startNewChat();
  }
});

// Auto resize
input?.addEventListener("input", ()=>{
  input.style.height="auto";
  input.style.height=Math.min(input.scrollHeight,200)+"px";
});

// Chips / examples
document.querySelectorAll(".example, .chip").forEach(ch=>{
  ch.addEventListener("click", ()=>{
    const q = ch.dataset.q;
    if(q){ input.value=q; input.dispatchEvent(new Event("input")); input.focus(); }
  });
});

function escapeHtml(s){return s.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]))}
function setStatus(t){ if(statusEl) statusEl.textContent=t; }

// ── Typewriter effect: hiển thị từng chữ ──
function typeWriterEffect(targetEl, text, speed = TYPEWRITER_SPEED){
  return new Promise((resolve)=>{
    let i = 0;
    let aborted = false;
    const bubble = targetEl.closest(".bubble");
    const metaEl = bubble ? bubble.querySelector(".meta") : null;
    if(metaEl) metaEl.style.display = "none";
    if(bubble) bubble.classList.add("typing");
    bubble && bubble.setAttribute("title","Nhấp để hiện toàn bộ");

    function finish(){
      targetEl.innerHTML = escapeHtml(text).replace(/\n/g,"<br>");
      if(metaEl) metaEl.style.display = "";
      if(bubble){ bubble.classList.remove("typing"); bubble.removeAttribute("title"); }
      if(messages) messages.scrollTop = messages.scrollHeight;
      _typingAbort = null;
      // gỡ listener skip
      if(bubble) bubble.removeEventListener("click", onSkip);
      targetEl.removeEventListener("click", onSkip);
      resolve();
    }
    function onSkip(){
      if(aborted) return;
      aborted = true;
      finish();
    }
    _typingAbort = onSkip;
    // cho phép click để skip
    if(bubble) bubble.addEventListener("click", onSkip, {once:true});
    targetEl.addEventListener("click", onSkip, {once:true});

    function tick(){
      if(aborted) return;
      if(i <= text.length){
        const partial = text.slice(0, i);
        const cursor = i < text.length ? '<span class="typing-cursor"></span>' : '';
        targetEl.innerHTML = escapeHtml(partial).replace(/\n/g,"<br>") + cursor;
        if(messages) messages.scrollTop = messages.scrollHeight;
        if(i < text.length){
          i++;
          let delay = speed;
          const ch = text[i-1];
          if(ch === " ") delay = speed * 0.4;
          else if(/[.!?。！？\n]/.test(ch)) delay = speed * 7;
          else if(/[,;:，；：]/.test(ch)) delay = speed * 3;
          setTimeout(tick, delay);
        } else {
          finish();
        }
      }
    }
    tick();
  });
}
function setHealth(state, msg){
  if (healthDot) healthDot.className="dot "+state;
  if (healthText) healthText.textContent=msg;
}
function addHistory(q){
  const list=$("#history-list");
  if(!list) return;
  const first=list.querySelector(".history-item");
  if(first) first.classList.remove("active");
  // Nếu list đang chứa "Cuộc trò chuyện mới" thì xóa nó khi có q thật
  if(list.children.length===1 && list.children[0].textContent.includes("Cuộc trò chuyện mới") && q){
    list.innerHTML="";
  }
  const div=document.createElement("div");
  div.className="history-item active";
  div.innerHTML=`<i class="fa-regular fa-message"></i><span>${escapeHtml(q.slice(0,32))}</span>`;
  list.prepend(div);
  // Giới hạn 10 items
  while(list.children.length>10) list.removeChild(list.lastChild);
}
function addMsg(role, text, meta=""){
  if(welcome) welcome.style.display="none";
  const row=document.createElement("div");
  row.className=`msg-row ${role}`;
  const inner=document.createElement("div");
  inner.className="msg-inner";
  const avatar=role==="user"?'<div class="avatar user">U</div>':'<div class="avatar assistant"><i class="fa-solid fa-robot"></i></div>';
  inner.innerHTML=`${avatar}<div class="bubble">${escapeHtml(text)}${meta?`<div class="meta">${meta}</div>`:""}</div>`;
  row.appendChild(inner);
  if(messages) messages.appendChild(row);
  if(messages) messages.scrollTop=messages.scrollHeight;
  return row;
}

async function loadHealth(){
  try{
    const r=await fetch("/api/health"); const j=await r.json();
    if(j.status==="ready") setHealth("ok", `${j.chroma_count} vectors • ${j.processed_files} files`);
    else if(j.status==="not_ready") setHealth("warn", `not ready • ${j.chroma_count||0}`);
    else setHealth("err", j.error||"error");
    const s=await fetch("/api/stats").then(r=>r.json());
    if(s.llm_backend){ if(llmInfo) llmInfo.textContent=`${s.llm_model} • ${s.llm_backend}`; if(modelFoot) modelFoot.textContent=`${s.chroma_count} chunks • ${s.embed_model} • ${s.llm_model} • ${conversation.length/2|0} turns`; }
  }catch(e){ setHealth("err","offline"); }
}
loadHealth(); setInterval(loadHealth,15000);

$("#stats-btn")?.addEventListener("click", async()=>{
  const dlg=$("#stats-modal"); const pre=$("#stats-pre");
  if(!dlg || !pre) return;
  pre.textContent="Đang tải..."; dlg.showModal();
  try{ const j=await fetch("/api/stats").then(r=>r.json()); pre.textContent=JSON.stringify(j,null,2);}catch(e){pre.textContent=String(e);}
});

async function send(){
  if(isLoading) return;
  let q=input.value.trim(); if(!q) return;
  // Cấu hình RAG đã bị xóa khỏi giao diện -> dùng giá trị mặc định, vẫn hỗ trợ prefix "quy_che: câu hỏi"
  let category=undefined;
  const m=q.match(/^(\w+):\s*(.+)/); const known=["quyet_dinh","quy_che","quy_dinh","tai_lieu_huong_dan","thong_bao"];
  if(m && known.includes(m[1])){ category=m[1]; q=m[2]; }
  const top_k=undefined; // mặc định backend sẽ dùng 3
  const use_rerank=true; // luôn bật rerank để tăng độ chính xác
  const show_context=false; // không hiển thị context chi tiết mặc định
  const use_history = true; // luôn bật nhớ hội thoại

  // Chuan bi history truoc khi gui (khong bao gom cau hoi hien tai)
  const historyToSend = use_history && conversation.length ? [...conversation] : undefined;

  addHistory(q);
  addMsg("user", q, `${category?`#${category} • `:``}rerank:${use_rerank} • top_k:${top_k||3}${use_history && historyToSend ? ` • hist:${historyToSend.length/2|0}`:``}`);
  input.value=""; input.style.height="auto"; input.focus();
  isLoading=true; if(sendBtn) sendBtn.disabled=true; setStatus(use_history && historyToSend ? "Đang rewrite & truy xuất..." : "Đang truy xuất...");
  const placeholder=addMsg("assistant", "⏳ Đang suy nghĩ...");

  try{
    const body = {question:q, category, top_k, use_rerank, show_context, use_history};
    if (historyToSend) body.history = historyToSend.map(m=>({role:m.role, content:m.content}));
    if (sessionId) body.session_id = sessionId;

    const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const j=await r.json();
    if(!r.ok) throw new Error(j.detail||JSON.stringify(j));

    // Đồng bộ sessionId từ phản hồi server
    if (j.session_id && j.session_id !== sessionId) {
      sessionId = j.session_id;
      localStorage.setItem(LS_SESSION, sessionId);
      updateHistoryBadge();
    }

    const bubble=placeholder.querySelector(".bubble");
    // Chuẩn bị khung cho typewriter: answer-text riêng + meta ẩn khi đang gõ
    const histInfo = j.history_used ? ` • hist:${j.history_used.length/2|0}` : "";
    const rewriteBadge = j.standalone_question && j.standalone_question !== q ? `<span class="tool" title="${escapeHtml(j.standalone_question)}">🔁 đã rewrite</span>` : "";
    bubble.innerHTML=`<span class="answer-text"></span><div class="meta"><span class="tool">✅ ${j.sources.length} nguồn</span><span class="tool">parquet • rerank top3${histInfo}</span>${rewriteBadge}</div>`;
    // Render nguồn/context ngay (không đợi gõ xong) để người dùng thấy tham khảo
     if(j.sources && j.sources.length){
      sourcesEl.innerHTML=j.sources.map(s=>{
        const sec=(s.section||"").trim();
        const page=s.page;
        let loc="";
        if(sec){
          loc=escapeHtml(sec);
          if(s.block_type==="table" && s.table_index!=null) loc+=` (bảng ${s.table_index})`;
          else if(s.paragraph_index) loc+=` (đoạn ${s.paragraph_index})`;
        } else if(page!=null){
          loc=`trang ${page}`;
        } else {
          loc=`khối ${s.block_index||0} ${escapeHtml(s.block_type||"")}`;
          if(s.paragraph_index) loc+=` para${s.paragraph_index}`;
          if(s.table_index) loc+=` tbl${s.table_index}`;
        }
        return `<div class="source"><b>${escapeHtml(s.filename||"")}</b> ${loc} • ${escapeHtml(s.category||"")}/${escapeHtml(s.subcategory||"")}<br><span class="hint">${escapeHtml(s.source||"")}</span></div>`;
      }).join("");
      const rp=$("#right-panel"); if(rp) rp.style.display="flex";
    }else sourcesEl.innerHTML='<p class="hint">Không có nguồn.</p>';

    // Hiển thị thông tin câu hỏi đã được viết lại
    if (j.standalone_question && j.standalone_question !== q) {
      rewriteInfo.textContent = `🔁 Rewrite: "${q}" → "${j.standalone_question}"`;
      rewriteInfo.classList.remove("hidden");
    } else {
      rewriteInfo.classList.add("hidden");
    }
    // Hiển thị tóm tắt hội thoại nếu có
    if (j.summary) {
      summaryInfo.innerHTML = `<b>📝 Tóm tắt hội thoại cũ:</b><br>${escapeHtml(j.summary)}`;
      summaryInfo.classList.remove("hidden");
    } else {
      // Giua summary cu neu van con gia tri tu server session
      if (j.summary === null && summaryInfo.textContent) {
        // keep
      } else if (!j.summary) {
        // neu khong co summary moi thi an di (nhung neu conversation dai thi van hien)
      }
    }

    if(show_context && j.context){
      contextEl.classList.remove("hidden");
      contextList.innerHTML=j.context.map((c,i)=>{
        const m=c.metadata||{};
        const sec=(m.section||"").trim();
        const page=m.page;
        let loc="";
        if(sec) loc=escapeHtml(sec);
        else if(page!=null) loc=`trang ${page}`;
        else loc=`khối ${m.block_index||0} ${escapeHtml(m.block_type||"")}`;
        return `<div class="context"><div><b>[${i+1}]</b> <span class="score">score=${Number(c.score).toFixed(4)}</span> • ${escapeHtml(m.filename||"")} ${loc}</div><div class="hint" style="margin-top:6px">${escapeHtml(c.text.slice(0,480))}...</div></div>`;
      }).join("");
    }else contextEl.classList.add("hidden");

    // Lưu hội thoại vào localStorage
    if (use_history) {
      conversation.push({role:"user", content:q});
      conversation.push({role:"assistant", content:j.answer});
      saveConversation();
      // Cập nhật thông tin footer
      if(modelFoot) modelFoot.textContent = `${j.sources.length} nguồn • ${historyBadge.textContent}`;
    } else {
      // single-turn thi khong luu
    }

    // Hiệu ứng gõ từng chữ
    if(_typingAbort) try{ _typingAbort(); }catch(e){}
    setStatus("Đang gõ...");
    const answerEl = bubble.querySelector(".answer-text");
    await typeWriterEffect(answerEl, j.answer, TYPEWRITER_SPEED);
    setStatus("");
  }catch(e){
    // Nếu đang gõ mà lỗi thì abort typing trước
    if(_typingAbort) try{ _typingAbort(); }catch(_){}
    const bubble=placeholder.querySelector(".bubble");
    bubble.innerHTML=`❌ <b>Lỗi:</b> ${escapeHtml(String(e.message||e))}<div class="meta">Kiểm tra Ollama/backend.</div>`;
    setStatus("Lỗi");
  }finally{
    isLoading=false; if(sendBtn) sendBtn.disabled=false; if(messages) messages.scrollTop=messages.scrollHeight;
    updateHistoryBadge();
  }
}
sendBtn?.addEventListener("click", send);
input?.addEventListener("keydown", e=>{ if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); send(); }});
input?.focus();

// Demo: enter to send, shift+enter for newline already handled
