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

let isLoading = false;

const TYPEWRITER_SPEED = 12; // giảm từ 18 để hiện nhanh hơn
let _typingAbort = null;
const ENABLE_STREAM = true; // bật streaming nếu backend hỗ trợ

// ── Multi-session storage (persistent + keep old sessions) ──
const LS_SESSIONS = "rag_sessions_v2";
const LS_ACTIVE = "rag_active_session_v2";
// legacy keys for migration
const LS_SESSION_LEGACY = "rag_session_id";
const LS_CONV_LEGACY = "rag_conversation_v2";
const LS_STYLE = "rag_style";
const DEFAULT_STYLE = "casual";
let currentStyle = (()=>{ try{ return localStorage.getItem(LS_STYLE) || DEFAULT_STYLE; }catch(e){ return DEFAULT_STYLE; } })();
function getActiveStyle(){
  if(activeSessionId && sessionsMap && sessionsMap[activeSessionId] && sessionsMap[activeSessionId].style) return sessionsMap[activeSessionId].style;
  return currentStyle;
}
function setActiveStyle(style){
  if(!style) return;
  currentStyle = style;
  try{ localStorage.setItem(LS_STYLE, style); }catch(e){}
  if(activeSessionId && sessionsMap && sessionsMap[activeSessionId]){
    sessionsMap[activeSessionId].style = style;
    sessionsMap[activeSessionId].updatedAt = Date.now();
    saveSessionsMap(sessionsMap);
  }
  updateHistoryBadge();
}
function isSocialFrontend(q){
  if(!q) return false;
  const low = q.toLowerCase();
  const prof = ["quy chế","quy định","quyết định","thông báo","tài liệu","học viện","kỹ thuật mật mã","đào tạo","khảo thí","học bổng","tín chỉ"];
  for(const h of prof) if(low.includes(h)) return false;
  // Nếu chứa đại từ tham chiếu như "nó", "cái đó" -> không phải xã giao, cần nhớ ngữ cảnh
  const pronouns = [" nó ", " nó?", " nó.", "cái đó", "cái này", "trong đó", "ở trên", "như trên", "của nó", "về nó"];
  for(const p of pronouns) if(low.includes(p)) return false;
  if(low.includes(" nó") || low.split(/\s+/).includes("nó")) return false;
  // Câu recall lịch sử cũng không phải xã giao
  const recall = ["vừa hỏi", "câu đầu", "cau dau", "vua hoi", "đã hỏi", "da hoi", "nhớ lại", "nho lai", "tóm tắt", "lich su"];
  for(const kw of recall) if(low.includes(kw)) return false;
  const socialKeywords = ["xin chào","chào bạn","chào","hello","hi","hey","khỏe không","khoe khong","khỏe","khoe","cảm ơn","cam on","tạm biệt","bye","bạn là ai","ban la ai","kể chuyện","hôm nay","thời tiết","bạn khỏe không"];
  for(const kw of socialKeywords) if(low.includes(kw)) return true;
  // Thắt chặt: không coi mọi câu ngắn là xã giao (tránh nhầm "nó là gì?" hoặc "đồ súc vật")
  // Chỉ xã giao nếu khớp allowlist ở trên, không tự động true cho câu ngắn
  return false;
}
function isMathFrontend(q){
  if(!q) return false;
  const low = q.toLowerCase();
  // So sánh số thực: lớn hơn/nhỏ hơn/bằng/so sánh + số
  const cmpWords = ["lớn hơn","lon hon","nhỏ hơn","nho hon","bé hơn","be hon","bằng nhau","bang nhau","bằng","bang","không bằng","khong bang","khác","khac","so sánh","so sanh","so với","so voi"];
  for(const w of cmpWords) if(low.includes(w) && /\d/.test(low)) return true;
  // Ký hiệu so sánh
  if(/\d\s*(>=|<=|==|!=|>|<|=)\s*\d/.test(low)) return true;
  if(/≥|≤|≠/.test(q) && /\d/.test(q)) return true;
  // VN chữ + so sánh
  const vnNum = ["không","khong","một","mot","hai","ba","bốn","bon","năm","nam","sáu","sau","bảy","bay","tám","tam","chín","chin","mười","muoi","mươi","trăm","tram","nghìn","nghin","triệu","trieu","tỷ","ty"];
  let hasVnNum = false;
  for(const w of vnNum) if(low.includes(w)) { hasVnNum = true; break; }
  if(hasVnNum && cmpWords.some(w=>low.includes(w))) return true;
  // Nếu là xã giao/recall thì không phải toán
  // Check toán học: chữ -> số, ký tự toán học toàn bộ
  const mathWords = ["cộng","cong","trừ","tru","nhân","nhan","chia","mũ","mu","lũy thừa","luy thua","căn","can","bình phương","binh phuong","lập phương","lap phuong","phần trăm","phan tram","giai thừa","giai thua","log","ln","sin","cos","tan","sqrt","cbrt","exp","phẩy","phay","chấm","cham"];
  for(const w of mathWords) if(low.includes(w) && /\d/.test(low)) return true;
  for(const w of mathWords) if(low.includes(w) && low.includes("kết quả") || low.includes("ket qua")) return true;
  // Ký tự toán học đầy đủ (bao gồm × ÷ ! và dấu phẩy thập phân VN)
  if(/\d\s*[\+\-\*/%×÷^\.]+\s*\d/.test(low)) return true;
  if(/\d\s*[\+\-\*/%×÷]\s*\d/.test(q)) return true;
  if(/\d\s*!\s*($|[^\d])/.test(q)) return true;
  if(/\(\s*\d/.test(q) && /\d\s*\)/.test(q)) return true;
  if(/sqrt|cbrt|log|sin|cos|tan|factorial|pi|e/.test(low)) return true;
  if(hasVnNum && mathWords.some(w=>low.includes(w))) return true;
  // "tính" + số
  if((low.includes("tính")||low.includes("tinh")) && /\d/.test(low)) return true;
  if(hasVnNum && (low.includes("tính")||low.includes("tinh"))) return true;
  return false;
}

function generateId(){
  try{
    if(typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  }catch(e){}
  return 'sess-' + Date.now() + '-' + Math.random().toString(36).slice(2,9);
}
function getIdFromUrl(){
  try{
    const p = new URLSearchParams(window.location.search);
    return p.get("c") || p.get("session") || p.get("s") || null;
  }catch(e){ return null; }
}
function updateUrlForSession(id, replace=false){
  try{
    const url = new URL(window.location.href);
    if(id) url.searchParams.set("c", id);
    else url.searchParams.delete("c");
    // keep other params, remove legacy restore flags
    if(replace) window.history.replaceState({sessionId:id}, "", url.toString());
    else window.history.pushState({sessionId:id}, "", url.toString());
  }catch(e){}
}
function getSessionsMap(){
  try{
    const raw = localStorage.getItem(LS_SESSIONS);
    if(raw){
      const parsed = JSON.parse(raw);
      if(parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
    }
  }catch(e){}
  return {};
}
function saveSessionsMap(map){
  try{
    localStorage.setItem(LS_SESSIONS, JSON.stringify(map));
  }catch(e){}
}
function getActiveId(){
  try{ return localStorage.getItem(LS_ACTIVE) || null; }catch(e){ return null; }
}
function setActiveId(id){
  try{
    if(id) localStorage.setItem(LS_ACTIVE, id);
    else localStorage.removeItem(LS_ACTIVE);
  }catch(e){}
}

// migrate legacy single conversation -> multi-session
(function migrateLegacy(){
  try{
    const hasMap = localStorage.getItem(LS_SESSIONS);
    if(hasMap) return;
    const legacySess = localStorage.getItem(LS_SESSION_LEGACY);
    const legacyConvRaw = localStorage.getItem(LS_CONV_LEGACY);
    if(legacySess || legacyConvRaw){
      let conv = [];
      try{
        if(legacyConvRaw){
          const p = JSON.parse(legacyConvRaw);
          if(Array.isArray(p)) conv = p;
        }
      }catch(e){}
      const map = {};
      const id = legacySess || generateId();
      const title = conv.find(m=>m.role==="user")?.content?.slice(0,50) || "Cuộc trò chuyện trước";
      map[id] = { title, createdAt: Date.now(), updatedAt: Date.now(), conversation: conv.slice(-60) };
      saveSessionsMap(map);
      setActiveId(id);
      // keep legacy for now, will be cleared on next createNewChat if needed
      console.log("[RAG] Migrated legacy session", id);
    }
  }catch(e){}
})();

let sessionsMap = getSessionsMap();
let activeSessionId = getActiveId();
// ensure active exists in map if present
if(activeSessionId && !sessionsMap[activeSessionId]){
  // will be fetched from server later, keep as is for now
}
let conversation = []; // alias to active session's conversation for compatibility with send()
function syncConversationRef(){
  if(activeSessionId && sessionsMap[activeSessionId]){
    conversation = sessionsMap[activeSessionId].conversation || [];
  }else{
    conversation = [];
  }
}
syncConversationRef();

function saveActiveConversation(){
  try{
    if(!activeSessionId) return;
    if(!sessionsMap[activeSessionId]){
      sessionsMap[activeSessionId] = { title: "Cuộc trò chuyện mới", createdAt: Date.now(), updatedAt: Date.now(), conversation: [] };
    }
    // keep max 60 messages
    sessionsMap[activeSessionId].conversation = conversation.slice(-60);
    sessionsMap[activeSessionId].updatedAt = Date.now();
    // auto title from first user message if still default
    if((!sessionsMap[activeSessionId].title || sessionsMap[activeSessionId].title==="Cuộc trò chuyện mới") && conversation.length){
      const firstUser = conversation.find(m=>m.role==="user");
      if(firstUser && firstUser.content) sessionsMap[activeSessionId].title = firstUser.content.slice(0,50);
    }
    saveSessionsMap(sessionsMap);
  }catch(e){}
  updateHistoryBadge();
  // after save, refresh sidebar without fetching server
  renderSessionListLocal();
}

function updateHistoryBadge(){
  const turns = Math.floor(conversation.length / 2);
  const sumHint = turns >= 12 ? " • sắp tóm tắt" : "";
  const title = activeSessionId && sessionsMap[activeSessionId] ? sessionsMap[activeSessionId].title : null;
  const style = getActiveStyle();
  const styleLabel = style && style !== DEFAULT_STYLE ? ` • ${style}` : "";
  if (historyBadge) historyBadge.textContent = `💬 ${turns} lượt${sumHint} • ${activeSessionId ? "đang ở phiên" : "chưa có phiên"}${title ? " • "+title.slice(0,20) : ""}${styleLabel}`;
  if (sessionInfo) sessionInfo.textContent = activeSessionId ? `session: ${activeSessionId.slice(0,8)}... • style:${style}` : `style:${style}`;
}

function clearChatView(){
  if(messages) messages.querySelectorAll(".msg-row").forEach(el=>el.remove());
  if(welcome) welcome.style.display = "block";
  if(sourcesEl) sourcesEl.innerHTML = '<p class="hint">Chưa có truy vấn.</p>';
  if(contextEl) contextEl.classList.add("hidden");
  if(rewriteInfo){ rewriteInfo.classList.add("hidden"); rewriteInfo.textContent=""; }
  if(summaryInfo){ summaryInfo.classList.add("hidden"); summaryInfo.textContent=""; }
  if(statusEl) setStatus("");
  if(input){ input.value=""; input.style.height="auto"; }
  updateHistoryBadge();
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

function renderMessagesForActive(){
  if(!welcome || !messages) return;
  messages.querySelectorAll(".msg-row").forEach(el=>el.remove());
  if(!conversation.length){
    welcome.style.display = "block";
    return;
  }
  welcome.style.display = "none";
  conversation.forEach(msg=>{
    const role = msg.role === "user" ? "user" : "assistant";
    addMsg(role, msg.content, "");
  });
}

// ── Sidebar list: hiển thị danh sách phiên (không xóa phiên cũ khi tạo mới) ──
function renderSessionListLocal(){
  const list = $("#history-list");
  if(!list) return;
  const ids = Object.keys(sessionsMap);
  if(ids.length===0){
    list.innerHTML = '<div class="history-item active"><i class="fa-regular fa-message"></i><span>Cuộc trò chuyện mới</span></div>';
    return;
  }
  // sort by updatedAt desc
  const sorted = ids.map(id=>({id, ...sessionsMap[id]})).sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0));
  list.innerHTML = "";
  sorted.slice(0,20).forEach(sess=>{
    const div = document.createElement("div");
    const isActive = sess.id === activeSessionId;
    div.className = "history-item" + (isActive ? " active" : "");
    div.dataset.sessionId = sess.id;
    const title = sess.title || "Cuộc trò chuyện mới";
    const turns = Math.floor((sess.conversation?.length||0)/2);
    const style = sess.style || DEFAULT_STYLE;
    const styleHint = style !== DEFAULT_STYLE ? ` • ${style}` : "";
    div.innerHTML = `<i class="fa-regular fa-message"></i><span title="${escapeHtml(title)}">${escapeHtml(title.slice(0,28))}</span><span class="hint" style="margin-left:auto;font-size:10px">${turns?turns+" lượt":""}${styleHint}</span><button class="icon-btn del-btn" title="Xóa phiên" style="width:22px;height:22px;margin-left:4px;flex-shrink:0"><i class="fa-solid fa-xmark" style="font-size:11px"></i></button>`;
    // click to load session
    div.addEventListener("click", (e)=>{
      // if click delete, not load
      if(e.target.closest(".del-btn")) return;
      loadSession(sess.id);
      $("#sidebar")?.classList.remove("open");
    });
    const delBtn = div.querySelector(".del-btn");
    if(delBtn){
      delBtn.addEventListener("click", async (e)=>{
        e.stopPropagation();
        if(!confirm(`Xóa phiên "${title}"?`)) return;
        await deleteSession(sess.id);
      });
    }
    list.appendChild(div);
  });
}

async function fetchAndMergeServerSessions(){
  try{
    const r = await fetch("/api/sessions");
    if(!r.ok) return;
    const j = await r.json();
    const serverSessions = j.sessions || [];
    let changed = false;
    serverSessions.forEach(s=>{
      const id = s.session_id;
      const serverUpdated = s.updated_at ? s.updated_at*1000 : 0; // server is seconds, local is ms
      const local = sessionsMap[id];
      // if server has more recent or local missing, merge
      if(!local){
        sessionsMap[id] = {
          title: s.title || "Cuộc trò chuyện",
          style: s.style || DEFAULT_STYLE,
          createdAt: s.created_at ? s.created_at*1000 : Date.now(),
          updatedAt: serverUpdated || Date.now(),
          conversation: [] // lazy
        };
        changed = true;
      }else{
        if(serverUpdated && serverUpdated > (local.updatedAt||0)){
          if(s.title && s.title !== local.title) { local.title = s.title; changed = true; }
          if(s.style && s.style !== local.style) { local.style = s.style; changed = true; }
          local.updatedAt = serverUpdated;
          changed = true;
        }
        if(s.title && !local.title){ local.title = s.title; changed = true; }
        if(s.style && !local.style){ local.style = s.style; changed = true; }
      }
    });
    if(changed) saveSessionsMap(sessionsMap);
    renderSessionListLocal();
  }catch(e){
    renderSessionListLocal();
  }
}

async function deleteSession(id){
  try{
    await fetch(`/api/sessions/${encodeURIComponent(id)}`, {method:"DELETE"});
  }catch(e){}
  // remove local
  delete sessionsMap[id];
  saveSessionsMap(sessionsMap);
  if(activeSessionId === id){
    // switch to most recent remaining or create new
    const remaining = Object.keys(sessionsMap).sort((a,b)=>(sessionsMap[b].updatedAt||0)-(sessionsMap[a].updatedAt||0));
    if(remaining.length){
      await loadSession(remaining[0]);
    }else{
      await createNewSession(true);
    }
  }else{
    renderSessionListLocal();
  }
}

// ── Tạo phiên mới ở cùng tab, không xóa phiên cũ, lưu bền vững ──
async function createNewSession(pushUrl=true){
  try{ if(_typingAbort) try{ _typingAbort(); }catch(e){} }catch(e){}
  const newId = generateId();
  const now = Date.now();
  // cập nhật UI ngay lập tức trước khi chờ server (để click không có cảm giác "không có gì")
  const initStyle = currentStyle || DEFAULT_STYLE;
  sessionsMap[newId] = { title: "Cuộc trò chuyện mới", createdAt: now, updatedAt: now, conversation: [], style: initStyle };
  activeSessionId = newId;
  setActiveId(newId);
  saveSessionsMap(sessionsMap);
  syncConversationRef();
  // cập nhật URL và giao diện ngay
  try{
    if(pushUrl) updateUrlForSession(newId, false);
    else updateUrlForSession(newId, true);
  }catch(e){}
  clearChatView();
  renderSessionListLocal();
  updateHistoryBadge();
  if(input){ input.focus(); }
  $("#sidebar")?.classList.remove("open");
  console.log("[RAG] Đã tạo phiên mới", newId, "style", initStyle);
  // tạo trên server nền (không chặn UI) để lưu bền vững ngay cả khi chưa gửi câu hỏi
  try{
    fetch("/api/sessions", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({session_id: newId, title: "Cuộc trò chuyện mới", style: initStyle})}).catch(()=>{});
  }catch(e){}
  return newId;
}

async function loadSession(id){
  if(!id) return;
  // if not in local map, try fetch from server
  if(!sessionsMap[id]){
    try{
      const r = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
      if(r.ok){
        const j = await r.json();
        sessionsMap[id] = {
          title: j.title || (j.history?.find(m=>m.role==="user")?.content?.slice(0,50) || "Cuộc trò chuyện"),
          style: j.style || DEFAULT_STYLE,
          createdAt: j.created_at ? j.created_at*1000 : Date.now(),
          updatedAt: j.updated_at ? j.updated_at*1000 : Date.now(),
          conversation: j.history || []
        };
        saveSessionsMap(sessionsMap);
      }else{
        sessionsMap[id] = { title: "Cuộc trò chuyện mới", style: DEFAULT_STYLE, createdAt: Date.now(), updatedAt: Date.now(), conversation: [] };
        saveSessionsMap(sessionsMap);
      }
    }catch(e){
      sessionsMap[id] = { title: "Cuộc trò chuyện mới", style: DEFAULT_STYLE, createdAt: Date.now(), updatedAt: Date.now(), conversation: [] };
      saveSessionsMap(sessionsMap);
    }
  } else {
    if((sessionsMap[id].conversation||[]).length===0){
      try{
        const r = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
        if(r.ok){
          const j = await r.json();
          if(j.history && j.history.length){
            sessionsMap[id].conversation = j.history;
            if(j.title) sessionsMap[id].title = j.title;
            if(j.style) sessionsMap[id].style = j.style;
            saveSessionsMap(sessionsMap);
          } else if(j.style && !sessionsMap[id].style){
            sessionsMap[id].style = j.style;
            saveSessionsMap(sessionsMap);
          }
        }
      }catch(e){}
    } else {
      // nếu server có style mới hơn, đồng bộ
      try{
        const r = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
        if(r.ok){
          const j = await r.json();
          if(j.style && j.style !== sessionsMap[id].style){
            sessionsMap[id].style = j.style;
            saveSessionsMap(sessionsMap);
          }
        }
      }catch(e){}
    }
  }
  activeSessionId = id;
  setActiveId(id);
  // đồng bộ style hiện tại
  if(sessionsMap[id] && sessionsMap[id].style){
    currentStyle = sessionsMap[id].style;
    try{ localStorage.setItem(LS_STYLE, currentStyle); }catch(e){}
  }
  sessionsMap[id].updatedAt = Date.now();
  saveSessionsMap(sessionsMap);
  syncConversationRef();
  updateUrlForSession(id, false);
  // abort typing
  try{ if(_typingAbort) try{ _typingAbort(); }catch(e){} }catch(e){}
  renderMessagesForActive();
  renderSessionListLocal();
  updateHistoryBadge();
  if(rewriteInfo){ rewriteInfo.classList.add("hidden"); rewriteInfo.textContent=""; }
  if(summaryInfo){ summaryInfo.classList.add("hidden"); summaryInfo.textContent=""; }
  if(sourcesEl) sourcesEl.innerHTML = '<p class="hint">Chưa có truy vấn.</p>';
  if(contextEl) contextEl.classList.add("hidden");
  $("#sidebar")?.classList.remove("open");
}

// Khởi tạo: khi chạy code luôn mở phiên mới hoàn toàn, giữ phiên cũ trong danh sách
async function initConversationOnLoad(){
  sessionsMap = getSessionsMap();
  activeSessionId = getActiveId();
  const urlId = getIdFromUrl();
  // Nếu URL chỉ định phiên cụ thể (?c=...) thì mở đúng phiên đó (để chia sẻ link)
  if(urlId){
    activeSessionId = urlId;
    setActiveId(urlId);
    syncConversationRef();
    await loadSession(urlId);
    fetchAndMergeServerSessions();
    return;
  }
  // Kiểm tra ?keep/?restore để giữ phiên cũ (tùy chọn), mặc định luôn tạo mới
  try{
    const p = new URLSearchParams(window.location.search);
    if(p.has("keep") || p.has("restore")){
      const ids = Object.keys(sessionsMap);
      if(ids.length && activeSessionId && sessionsMap[activeSessionId]){
        syncConversationRef();
        renderMessagesForActive();
        updateHistoryBadge();
        renderSessionListLocal();
        updateUrlForSession(activeSessionId, true);
        fetchAndMergeServerSessions();
        return;
      }
    }
  }catch(e){}
  // Mặc định: khi chạy code luôn mở phiên mới hoàn toàn (yêu cầu)
  // Tạo phiên mới ngay để hiện trang trắng, fetch danh sách cũ nền để sidebar vẫn có phiên cũ
  await createNewSession(false);
  fetchAndMergeServerSessions();
}
initConversationOnLoad();

// handle back/forward
window.addEventListener("popstate", ()=>{
  const id = getIdFromUrl();
  if(id && id !== activeSessionId){
    loadSession(id);
  } else if(!id){
    // no id in url -> show active or welcome
    if(activeSessionId) loadSession(activeSessionId);
  }
});

// Sidebar toggle (ChatGPT mobile)
$("#open-sidebar")?.addEventListener("click", ()=> $("#sidebar").classList.add("open"));
$("#close-sidebar")?.addEventListener("click", ()=> $("#sidebar").classList.remove("open"));
$("#close-right")?.addEventListener("click", ()=> { const rp=$("#right-panel"); if(rp) rp.style.display="none"; });

// New chat - cùng tab, không xóa phiên cũ, lưu bền vững (đảm bảo click luôn có phản hồi)
async function createNewChatSameTab(){
  console.log("[RAG] Click new chat");
  try{
    await createNewSession(true);
  }catch(e){
    console.error("[RAG] createNewSession failed", e);
    // fallback: vẫn clear view
    try{ clearChatView(); }catch(_){}
  }
}
// Gắn trực tiếp + delegation để tránh lỗi khi element bị replace
$("#new-chat")?.addEventListener("click", (e)=>{
  e.preventDefault();
  e.stopPropagation();
  if(e.__newChatHandled) return;
  e.__newChatHandled = true;
  createNewChatSameTab();
});
document.addEventListener("click", (e)=>{
  const btn = e.target.closest("#new-chat");
  if(!btn) return;
  e.preventDefault();
  e.stopPropagation();
  if(e.__newChatHandled) return;
  e.__newChatHandled = true;
  createNewChatSameTab();
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
  // legacy helper kept for compatibility: now sidebar is session-based, not per-question
  // just ensure active session title updates
  if(activeSessionId && sessionsMap[activeSessionId]){
    if(sessionsMap[activeSessionId].title==="Cuộc trò chuyện mới" && q){
      sessionsMap[activeSessionId].title = q.slice(0,50);
      saveSessionsMap(sessionsMap);
      renderSessionListLocal();
    }
  }
}
function addMsgSimple(role, text, meta=""){
  return addMsg(role, text, meta);
}

async function loadHealth(){
  try{
    // Gọi song song để nhanh hơn thay vì đợi từng cái
    const [hRes, sRes] = await Promise.allSettled([fetch("/api/health"), fetch("/api/stats")]);
    if(hRes.status==="fulfilled" && hRes.value.ok){
      const j=await hRes.value.json();
      if(j.status==="ready") setHealth("ok", `${j.chroma_count} vectors • ${j.processed_files} files`);
      else if(j.status==="not_ready") setHealth("warn", `not ready • ${j.chroma_count||0}`);
      else setHealth("err", j.error||"error");
    }
    if(sRes.status==="fulfilled" && sRes.value.ok){
      const s=await sRes.value.json();
      if(s.llm_backend){ if(llmInfo) llmInfo.textContent=`${s.llm_model} • ${s.llm_backend} (${s.device||s.embed_model})`; if(modelFoot) modelFoot.textContent=`${s.chroma_count} chunks • ${s.embed_model} • ${s.llm_model} • ${conversation.length/2|0} turns`; }
    }
  }catch(e){ setHealth("err","offline"); }
}
loadHealth();
// Giảm tần suất polling để nhẹ máy hơn: health 30s, sessions 30s và chỉ khi tab đang mở
setInterval(()=>{ if(document.visibilityState==="visible") loadHealth(); },30000);
setInterval(()=>{ if(document.visibilityState==="visible") fetchAndMergeServerSessions(); },30000);

$("#stats-btn")?.addEventListener("click", async()=>{
  const dlg=$("#stats-modal"); const pre=$("#stats-pre");
  if(!dlg || !pre) return;
  pre.textContent="Đang tải..."; dlg.showModal();
  try{ const j=await fetch("/api/stats").then(r=>r.json()); pre.textContent=JSON.stringify(j,null,2);}catch(e){pre.textContent=String(e);}
});

async function send(){
  if(isLoading) return;
  let q=input.value.trim(); if(!q) return;
  let category=undefined;
  const m=q.match(/^(\w+):\s*(.+)/); const known=["quyet_dinh","quy_che","quy_dinh","tai_lieu_huong_dan","thong_bao"];
  if(m && known.includes(m[1])){ category=m[1]; q=m[2]; }
  const top_k=undefined;
  const use_rerank=true;
  const show_context=false;
  const use_history = true;

  // đảm bảo có phiên active (nếu chưa có thì tạo mới)
  if(!activeSessionId){
    await createNewSession(true);
  }
  syncConversationRef();
  const historyToSend = use_history && conversation.length ? [...conversation] : undefined;

  // update title if first message
  if(activeSessionId && sessionsMap[activeSessionId] && sessionsMap[activeSessionId].title==="Cuộc trò chuyện mới"){
    sessionsMap[activeSessionId].title = q.slice(0,50);
    saveSessionsMap(sessionsMap);
    renderSessionListLocal();
  }

  addMsg("user", q, `${category?`#${category} • `:``}rerank:${use_rerank} • top_k:${top_k||3}${use_history && historyToSend ? ` • hist:${historyToSend.length/2|0}`:``}`);
  input.value=""; input.style.height="auto"; input.focus();
  isLoading=true; if(sendBtn) sendBtn.disabled=true; setStatus(use_history && historyToSend ? "Đang rewrite & truy xuất..." : "Đang truy xuất...");
  const placeholder=addMsg("assistant", "⏳ Đang suy nghĩ...");

  // Helper streaming parser
  async function fetchStream(body, onToken){
    const r = await fetch("/api/chat/stream",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    if(!r.ok || !r.body){
      const j = await r.json().catch(()=>({detail:r.statusText}));
      throw new Error(j.detail||JSON.stringify(j));
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let full = "";
    let meta = null;
    while(true){
      const {done, value} = await reader.read();
      if(done) break;
      buf += decoder.decode(value, {stream:true});
      const lines = buf.split("\n\n");
      buf = lines.pop();
      for(const line of lines){
        if(!line.startsWith("data: ")) continue;
        try{
          const data = JSON.parse(line.slice(6));
          if(data.token){
            full += data.token;
            if(onToken) onToken(data.token, full);
          }
          if(data.done) meta = data;
          if(data.error) throw new Error(data.error);
        }catch(e){ if(e.message && e.message.includes("error")) throw e; }
      }
    }
    return {answer: full, meta};
  }

  try{
    const body = {question:q, category, top_k, use_rerank, show_context, use_history, style: getActiveStyle()};
    if (historyToSend) body.history = historyToSend.map(m=>({role:m.role, content:m.content}));
    if (activeSessionId) body.session_id = activeSessionId;

    let j = null;
    let streamed = false;
    // Thử streaming cho câu RAG thông thường (không phải toán/xã giao) để thấy chữ hiện dần
    const canStream = ENABLE_STREAM && !isMathFrontend(q) && !isSocialFrontend(q) && typeof ReadableStream !== "undefined";
    if(canStream){
      try{
        const bubbleTmp = placeholder.querySelector(".bubble");
        bubbleTmp.innerHTML=`<span class="answer-text"></span><div class="meta"><span class="tool">🔄 đang tạo...</span></div>`;
        const answerElTmp = bubbleTmp.querySelector(".answer-text");
        let streamingText = "";
        setStatus("Đang tạo trả lời...");
        const streamRes = await fetchStream(body, (tok, full)=>{
          streamingText = full;
          answerElTmp.textContent = full;
          if(messages) messages.scrollTop = messages.scrollHeight;
        });
        // Sau stream xong, lấy thêm thông tin từ /api/chat để có sources/rewrite (không gọi LLM lại)
        // Dùng kết quả stream làm answer, bổ sung meta từ stream
        j = {
          answer: streamRes.answer,
          sources: [],
          context: [],
          session_id: streamRes.meta?.session_id || activeSessionId,
          standalone_question: q,
          history_used: historyToSend,
          style: getActiveStyle(),
        };
        // Nếu backend stream đã lưu session thì đồng bộ
        streamed = true;
        // Để lấy sources chính xác, gọi nhanh /api/chat với cùng body nhưng backend sẽ trả cache nếu có? Tạm bỏ qua sources cho stream
        // Nếu cần sources, có thể fetch song song nhưng giữ đơn giản
      }catch(e){
        console.warn("[stream] fallback to normal", e);
        streamed = false;
      }
    }
    if(!streamed){
      const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
      j=await r.json();
      if(!r.ok) throw new Error(j.detail||JSON.stringify(j));
    }

    // Đồng bộ style nếu server phát hiện đổi phong cách qua câu tự nhiên
    if(j.style && j.style !== getActiveStyle()){
      setActiveStyle(j.style);
      // hiển thị thông báo nhẹ
      if(rewriteInfo){
        rewriteInfo.textContent = `🎨 Đã đổi sang phong cách ${j.style}`;
        rewriteInfo.classList.remove("hidden");
        setTimeout(()=>{ try{ rewriteInfo.classList.add("hidden"); }catch(e){} }, 3000);
      }
    }

    // Đồng bộ sessionId từ phản hồi server (trường hợp server tạo mới)
    if (j.session_id && j.session_id !== activeSessionId) {
      const oldId = activeSessionId;
      activeSessionId = j.session_id;
      setActiveId(activeSessionId);
      if(sessionsMap[oldId] && !sessionsMap[activeSessionId]){
        sessionsMap[activeSessionId] = sessionsMap[oldId];
        delete sessionsMap[oldId];
      }
      if(!sessionsMap[activeSessionId]) sessionsMap[activeSessionId] = {title: q.slice(0,50), style: j.style || getActiveStyle(), createdAt: Date.now(), updatedAt: Date.now(), conversation: []};
      // đảm bảo style đồng bộ
      if(j.style) sessionsMap[activeSessionId].style = j.style;
      saveSessionsMap(sessionsMap);
      updateUrlForSession(activeSessionId, true);
      syncConversationRef();
      renderSessionListLocal();
    } else if(j.style && activeSessionId && sessionsMap[activeSessionId]){
      // cập nhật style cho phiên hiện tại nếu server trả về style mới
      if(sessionsMap[activeSessionId].style !== j.style){
        sessionsMap[activeSessionId].style = j.style;
        sessionsMap[activeSessionId].updatedAt = Date.now();
        saveSessionsMap(sessionsMap);
        renderSessionListLocal();
        updateHistoryBadge();
      }
    }

    const bubble=placeholder.querySelector(".bubble");
    const histInfo = j.history_used ? ` • hist:${j.history_used.length/2|0}` : "";
    const rewriteBadge = j.standalone_question && j.standalone_question !== q ? `<span class="tool" title="${escapeHtml(j.standalone_question)}">🔁 đã rewrite</span>` : "";
    const isMath = j.math_result !== undefined && j.math_result !== null;
    const mathBadge = isMath ? `<span class="tool" title="${escapeHtml(j.math_expression||j.standalone_question||"")} = ${escapeHtml(j.math_result)}">🧮 ${escapeHtml(j.math_expression||j.standalone_question||"")} = ${escapeHtml(j.math_result)}</span>` : "";
    const sourceBadge = isMath ? `<span class="tool">🧮 calculator • Decimal prec=50</span>` : `<span class="tool">✅ ${j.sources.length} nguồn</span>`;
    bubble.innerHTML=`<span class="answer-text"></span><div class="meta">${sourceBadge}<span class="tool">parquet • rerank top3${histInfo}</span>${mathBadge}${rewriteBadge}</div>`;
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

    if (j.math_result !== undefined && j.math_result !== null) {
      rewriteInfo.textContent = `🧮 ${j.math_expression||j.standalone_question} = ${j.math_result} • Decimal prec=50`;
      rewriteInfo.classList.remove("hidden");
    } else if (j.standalone_question && j.standalone_question !== q) {
      rewriteInfo.textContent = `🔁 Rewrite: "${q}" → "${j.standalone_question}"`;
      rewriteInfo.classList.remove("hidden");
    } else {
      rewriteInfo.classList.add("hidden");
    }
    if (j.summary) {
      summaryInfo.innerHTML = `<b>📝 Tóm tắt hội thoại cũ:</b><br>${escapeHtml(j.summary)}`;
      summaryInfo.classList.remove("hidden");
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

    // Lưu hội thoại vào map phiên active (cùng tab, giữ phiên cũ)
    if (use_history) {
      conversation.push({role:"user", content:q});
      conversation.push({role:"assistant", content:j.answer});
      // ensure map ref sync
      if(activeSessionId && sessionsMap[activeSessionId]){
        sessionsMap[activeSessionId].conversation = conversation.slice(-60);
        sessionsMap[activeSessionId].updatedAt = Date.now();
        // title đã set ở trên
        saveSessionsMap(sessionsMap);
        renderSessionListLocal();
      } else {
        saveActiveConversation();
      }
      if(modelFoot) modelFoot.textContent = `${j.sources.length} nguồn • ${historyBadge.textContent}`;
      // persist title to server
      try{
        if(activeSessionId && sessionsMap[activeSessionId] && sessionsMap[activeSessionId].title){
          await fetch(`/api/sessions/${encodeURIComponent(activeSessionId)}`, {method:"PATCH", headers:{"Content-Type":"application/json"}, body: JSON.stringify({title: sessionsMap[activeSessionId].title})});
        }
      }catch(e){}
    }

    if(_typingAbort) try{ _typingAbort(); }catch(e){}
    setStatus("Đang gõ...");
    const answerEl = bubble.querySelector(".answer-text");
    await typeWriterEffect(answerEl, j.answer, TYPEWRITER_SPEED);
    setStatus("");
  }catch(e){
    if(_typingAbort) try{ _typingAbort(); }catch(_){}
    const msg = String(e.message||e);
    const isNetErr = msg.includes("Failed to fetch") || msg.includes("NetworkError") || msg.includes("load failed");
    const bubble=placeholder.querySelector(".bubble");
    // Nếu là câu xã giao và lỗi mạng, dùng fallback tại chỗ để không hiện Failed to fetch
    if(isNetErr && isSocialFrontend(q)){
      const fallback = "Xin chào! Mình khỏe, cảm ơn bạn đã hỏi thăm. Bạn cần mình hỗ trợ gì về quy chế, quy định hay thông tin nào khác không?";
      bubble.innerHTML=`<span class="answer-text"></span><div class="meta"><span class="tool">💬 xã giao</span><span class="tool">offline fallback</span></div>`;
      // lưu vào lịch sử để nhớ
      conversation.push({role:"user", content:q});
      conversation.push({role:"assistant", content:fallback});
      if(activeSessionId && sessionsMap[activeSessionId]){
        sessionsMap[activeSessionId].conversation = conversation.slice(-60);
        sessionsMap[activeSessionId].updatedAt = Date.now();
        saveSessionsMap(sessionsMap);
        renderSessionListLocal();
      }
      setStatus("Đang gõ...");
      const answerEl = bubble.querySelector(".answer-text");
      await typeWriterEffect(answerEl, fallback, TYPEWRITER_SPEED);
      setStatus("");
      isLoading=false; if(sendBtn) sendBtn.disabled=false; if(messages) messages.scrollTop=messages.scrollHeight;
      updateHistoryBadge();
      return;
    }
    if(isNetErr && isMathFrontend(q)){
      // Offline calculator fallback đơn giản (chỉ +-*/ và ngoặc)
      let fallback = null;
      try{
        // Chuẩn hoá cơ bản cho offline: giữ digits và + - * / % ( ) .
        let expr = q.replace(/[×]/g,"*").replace(/[÷]/g,"/").replace(/[,]/g,".").replace(/[^0-9\.\+\-\*/%\(\) ]/g," ").replace(/\s+/g," ").trim();
        // Thử eval an toàn: chỉ cho phép chars số và +-*/%()
        if(/^[0-9\.\+\-\*/%\(\) ]+$/.test(expr) && /\d/.test(expr) && /[\+\-\*/%]/.test(expr)){
          // Chặn --, ** thừa
          const val = Function('"use strict"; return ('+expr+')')();
          if(typeof val === 'number' && isFinite(val)){
            fallback = `${expr} = ${val}`;
          }
        }
      }catch(e){}
      if(fallback){
        bubble.innerHTML=`<span class="answer-text"></span><div class="meta"><span class="tool">🧮 calculator</span><span class="tool">offline fallback</span></div>`;
        conversation.push({role:"user", content:q});
        conversation.push({role:"assistant", content:fallback});
        if(activeSessionId && sessionsMap[activeSessionId]){
          sessionsMap[activeSessionId].conversation = conversation.slice(-60);
          sessionsMap[activeSessionId].updatedAt = Date.now();
          saveSessionsMap(sessionsMap);
          renderSessionListLocal();
        }
        if(rewriteInfo){ rewriteInfo.textContent = `🧮 Offline: ${fallback}`; rewriteInfo.classList.remove("hidden"); }
        setStatus("Đang gõ...");
        const answerEl = bubble.querySelector(".answer-text");
        await typeWriterEffect(answerEl, fallback, TYPEWRITER_SPEED);
        setStatus("");
        isLoading=false; if(sendBtn) sendBtn.disabled=false; if(messages) messages.scrollTop=messages.scrollHeight;
        updateHistoryBadge();
        return;
      }
    }
    if(isNetErr){
      bubble.innerHTML=`❌ <b>Lỗi kết nối:</b> Không kết nối được máy chủ ở <code>http://localhost:8000</code>. Hãy kiểm tra bạn đang chạy <code>uvicorn backend.app:app --port 8000</code> và Ollama đang chạy.<div class="meta">${escapeHtml(msg)}</div>`;
    } else {
      bubble.innerHTML=`❌ <b>Lỗi:</b> ${escapeHtml(msg)}<div class="meta">Kiểm tra Ollama/backend.</div>`;
    }
    setStatus("Lỗi");
  }finally{
    isLoading=false; if(sendBtn) sendBtn.disabled=false; if(messages) messages.scrollTop=messages.scrollHeight;
    updateHistoryBadge();
  }
}
sendBtn?.addEventListener("click", send);
input?.addEventListener("keydown", e=>{ if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); send(); }});
input?.focus();
