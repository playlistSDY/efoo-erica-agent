'use strict';
const el = Object.fromEntries([
  'new-chat','chat-scroll','welcome','messages','chat-form','message',
  'send-button','notice','mode-label','mode-badge','composer-note',
  'history-button','history-dialog','close-history','history-status','history-list'
].map(id => [id.replaceAll('-', '_'), document.getElementById(id)]));
const state = {conversation:null, ready:false, switching:true, busy:false,
  controller:null, viewController:null, listController:null, generation:0, listGeneration:0};

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}
function readableTime(value) {
  const date = new Date(value);
  if (!value || Number.isNaN(date.getTime())) return '수집 시각 미확인';
  return new Intl.DateTimeFormat('ko-KR', {timeZone:'Asia/Seoul', month:'numeric', day:'numeric', hour:'2-digit', minute:'2-digit'}).format(date);
}
function safeURL(value) {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) ? url.href : null;
  } catch { return null; }
}
function showNotice(text) { el.notice.textContent = text || ''; el.notice.hidden = !text; }
function updateControls() {
  el.message.disabled = !state.ready || state.switching;
  el.send_button.disabled = !state.ready || state.switching || state.busy;
  el.new_chat.disabled = state.switching;
  el.history_button.disabled = state.switching;
  el.chat_form.setAttribute('aria-busy', String(state.busy || state.switching));
}
function scrollToBottom() { el.chat_scroll.scrollTop = el.chat_scroll.scrollHeight; }
function resizeComposer() {
  el.message.style.height = 'auto';
  el.message.style.height = `${Math.min(el.message.scrollHeight, 150)}px`;
}
async function fetchJSON(path, options = {}) {
  const response = await fetch(path, {credentials:'same-origin', ...options, headers: {'Content-Type':'application/json', ...options.headers}});
  let data;
  try { data = await response.json(); }
  catch { throw new Error('서버 응답을 읽을 수 없어요. 잠시 후 다시 시도해 주세요.'); }
  if (!response.ok) {
    const detail = data.detail || data.error;
    throw new Error(typeof detail === 'string' ? detail : '질문을 확인하고 다시 시도해 주세요.');
  }
  return data;
}
async function loadStatus() {
  try {
    const data = await fetchJSON('/api/status');
    const ai = data.agentic_enabled === true || ['openai', 'conversation', 'agent'].includes(data.mode);
    el.mode_label.textContent = ai ? 'AI 대화와 학교 안내' : '자료 검색 미리보기';
    el.mode_badge.title = ai ? `생성 모델 ${data.model || ''}` : 'AI 생성 없이 수집한 자료를 보여드려요.';
    el.composer_note.textContent = ai ? '학교 정보를 확인할 때는 사용한 공식 자료를 함께 보여드려요.'
      : '자료 검색 미리보기 · AI 생성 없이 수집한 자료를 보여드려요.';
  } catch (error) {
    el.mode_label.textContent = '연결 확인 필요';
    if (state.ready && !state.switching) showNotice(error.message);
  }
}
function assistantShell() {
  const article = node('article', 'message assistant');
  article.setAttribute('aria-label', 'EFOO 답변');
  const heading = node('div', 'assistant-heading');
  heading.append(node('span', 'assistant-avatar', 'E'), node('span', 'assistant-name', 'EFOO'));
  article.append(heading);
  el.messages.append(article);
  return {article, heading};
}
function renderSources(article, sources) {
  if (!Array.isArray(sources) || !sources.length) return;
  const details = node('details', 'sources');
  details.append(node('summary', '', `참고한 공식 자료 ${sources.length}건`));
  const list = node('div', 'source-list');
  sources.forEach((source, index) => {
    if (!source || typeof source !== 'object') return;
    const href = safeURL(source.url);
    const card = node(href ? 'a' : 'div', 'source-card');
    if (href) Object.assign(card, {href, target:'_blank', rel:'noopener noreferrer'});
    card.append(node('span', 'source-card-title', `${source.id || index + 1} · ${source.title || '공식 학교 자료'}`));
    card.append(node('span', 'source-card-meta', `수집 ${readableTime(source.fetched_at)} ↗`));
    if (source.text) card.title = String(source.text);
    list.append(card);
  });
  details.append(list);
  article.append(details);
}
function renderImages(article, images) {
  if (!Array.isArray(images) || !images.length) return;
  const grid = node('div', 'answer-images');
  grid.setAttribute('aria-label', '공식 자료에서 수집한 사진');
  const seen = new Set();
  for (const item of images) {
    if (grid.children.length >= 6) break;
    if (!item || typeof item.url !== 'string') continue;
    // Only collected local bitmap files are displayed. Never load a remote
    // URL, executable image format, arbitrary route or query parameter.
    const local = /^\/media\/([a-f0-9]{64})\.(?:jpg|jpeg|png|gif|webp|avif)$/.exec(item.url);
    if (!local || seen.has(local[1])) continue;
    seen.add(local[1]);
    const figure = node('figure', 'answer-image');
    const image = node('img');
    image.src = item.url;
    image.alt = String(item.alt || item.title || '공식 자료 사진');
    image.loading = 'lazy';
    image.decoding = 'async';
    image.addEventListener('load', () => { if (state.busy) scrollToBottom(); }, {once:true});
    image.addEventListener('error', () => {
      image.replaceWith(node('p', 'image-unavailable', `${image.alt} · 사진을 불러오지 못했어요.`));
    }, {once:true});
    figure.append(image);
    const caption = node('figcaption', 'answer-image-caption', item.title || item.alt || '공식 자료 사진');
    const sourceURL = safeURL(item.source_url);
    if (sourceURL) {
      const link = node('a', '', '공식 출처 ↗');
      Object.assign(link, {href:sourceURL, target:'_blank', rel:'noopener noreferrer'});
      caption.append(link);
    }
    figure.append(caption);
    grid.append(figure);
  }
  if (grid.children.length) article.append(grid);
}
function imagesForAnswer(data, text) {
  // New responses explicitly choose their gallery. Older saved turns can
  // recover local photos only from sources actually cited in their answer.
  if (Array.isArray(data.images) && data.images.length) return data.images;
  const cited = new Set([...String(text).matchAll(/\[(S\d+)\]/g)].map(match => match[1]));
  const images = [];
  for (const source of Array.isArray(data.sources) ? data.sources : []) {
    if (!source || typeof source !== 'object') continue;
    const id = /^\d+$/.test(String(source.id)) ? `S${source.id}` : String(source.id || '');
    if (!cited.has(id) || /메뉴가 등록되지 않았습니다|등록된 메뉴가 없습니다/.test(String(source.text || ''))) continue;
    for (const image of Array.isArray(source.images) ? source.images : []) {
      if (!image || !/^\/media\/[a-f0-9]{64}\.(?:jpg|jpeg|png|gif|webp|avif)$/.test(image.url || '')) continue;
      images.push({...image, title:image.title || source.title || '공식 자료 사진',
        alt:image.alt || source.title || '공식 자료 사진', source_url:source.url});
    }
  }
  return images;
}
function renderAssistant(result, data, fallbackContent='') {
  const labels = {openai:'AI 답변', retrieval:'자료 검색 결과', insufficient:'근거 부족',
    clarification:'추가 확인', conversation:'AI 대화', agent:'AI 에이전트'};
  result.heading.append(node('span', 'answer-mode', labels[data.mode] || '답변'));
  const text = data.answer || fallbackContent || '확인된 자료가 부족해 답변하기 어려워요.';
  const content = node('div', 'message-text');
  if (window.EfooMarkdown?.render) window.EfooMarkdown.render(content, text, {sources:data.sources, images:data.images});
  else {content.textContent = text; content.classList.add('markdown-fallback');}
  result.article.append(content);
  renderImages(result.article, imagesForAnswer(data, text));
  renderSources(result.article, data.sources);
}
function renderConversation(conversation) {
  el.messages.replaceChildren();
  for (const message of conversation.messages) {
    if (!message || typeof message.content !== 'string') continue;
    if (message.role === 'user') {
      const article = node('article', 'message user', message.content);
      article.setAttribute('aria-label', '내 질문');
      el.messages.append(article);
    } else if (message.role === 'assistant') {
      const payload = message.payload && typeof message.payload === 'object' && !Array.isArray(message.payload) ? message.payload : {};
      renderAssistant(assistantShell(), payload, message.content);
    }
  }
  el.welcome.hidden = el.messages.children.length > 0;
  scrollToBottom();
}
async function loadConversation(path='/api/conversations/current', options={}) {
  const generation = ++state.generation;
  state.controller?.abort();
  state.viewController?.abort();
  state.controller = null;
  state.busy = false;
  state.switching = true;
  state.ready = false;
  state.conversation = null;
  state.viewController = new AbortController();
  el.messages.replaceChildren();
  el.welcome.hidden = false;
  el.message.value = '';
  resizeComposer();
  updateControls();
  showNotice('대화 불러오는 중…');
  try {
    const data = await fetchJSON(path, {...options, signal:state.viewController.signal});
    if (generation !== state.generation) return;
    if (!data || typeof data.id !== 'string' || !Array.isArray(data.messages)) throw new Error('저장된 대화 응답을 확인할 수 없어요.');
    state.conversation = data;
    state.ready = true;
    renderConversation(data);
    showNotice('');
  } catch (error) {
    if (error.name === 'AbortError' || generation !== state.generation) return;
    state.ready = false;
    state.conversation = null;
    showNotice(`${error.message} 대화를 다시 불러오거나 새로고침해 주세요.`);
    const retry = node('button', 'notice-retry', '다시 불러오기');
    retry.type = 'button';
    retry.addEventListener('click', () => loadConversation());
    el.notice.append(retry);
  } finally {
    if (generation === state.generation) {
      state.switching = false;
      state.viewController = null;
      updateControls();
      if (state.ready) el.message.focus({preventScroll:true});
    }
  }
}
async function sendMessage(text) {
  text = text.trim();
  if (!text || state.busy || state.switching || !state.ready || !state.conversation) return;
  const conversationId = state.conversation.id;
  const turnId = crypto.randomUUID();
  showNotice('');
  el.welcome.hidden = true;
  const user = node('article', 'message user', text);
  user.setAttribute('aria-label', '내 질문');
  el.messages.append(user);
  const result = assistantShell();
  const loading = node('div', 'typing', '질문을 살펴보고 있어요. 필요한 정보가 있으면 자료를 확인할게요…');
  result.article.append(loading);
  const generation = state.generation;
  state.busy = true;
  state.controller = new AbortController();
  updateControls();
  el.message.value = '';
  resizeComposer();
  scrollToBottom();
  try {
    const data = await fetchJSON('/api/chat', {
      method:'POST', signal:state.controller.signal,
      body:JSON.stringify({message:text, conversation_id:conversationId, turn_id:turnId})
    });
    if (generation !== state.generation || state.conversation?.id !== conversationId) return;
    if (data.conversation_id && data.conversation_id !== conversationId) throw new Error('답변의 대화 정보를 확인할 수 없어요. 저장된 대화를 다시 불러와 주세요.');
    loading.remove();
    renderAssistant(result, data);
  } catch (error) {
    if (error.name === 'AbortError' || generation !== state.generation) return;
    loading.remove();
    result.article.append(node('p', 'message-error', error.message));
    const recover = node('button', 'notice-retry', '저장된 대화 다시 불러오기');
    recover.type = 'button';
    recover.addEventListener('click', () => loadConversation(`/api/conversations/${encodeURIComponent(conversationId)}`));
    result.article.append(recover);
  } finally {
    if (generation === state.generation) {
      state.busy = false;
      state.controller = null;
      updateControls();
      scrollToBottom();
      el.message.focus({preventScroll:true});
    }
  }
}
function newConversation() {
  if (el.history_dialog.open) el.history_dialog.close();
  return loadConversation('/api/conversations', {method:'POST', body:JSON.stringify({})});
}
function conversationTime(value) {
  const date = new Date(value);
  if (!value || Number.isNaN(date.getTime())) return '저장 시각 미확인';
  return new Intl.DateTimeFormat('ko-KR', {timeZone:'Asia/Seoul', year:'numeric', month:'long', day:'numeric', hour:'2-digit', minute:'2-digit'}).format(date);
}
async function openHistory() {
  if (!el.history_dialog.open) el.history_dialog.showModal();
  state.listController?.abort();
  state.listController = new AbortController();
  const generation = ++state.listGeneration;
  el.history_list.replaceChildren();
  el.history_status.textContent = '저장된 대화 불러오는 중…';
  try {
    const data = await fetchJSON('/api/conversations', {signal:state.listController.signal});
    if (generation !== state.listGeneration || !el.history_dialog.open) return;
    if (!Array.isArray(data.conversations)) throw new Error('대화 목록을 확인할 수 없어요.');
    el.history_status.textContent = data.conversations.length ? '' : '아직 저장된 대화가 없어요.';
    for (const conversation of data.conversations.slice(0, 50)) {
      if (!conversation || typeof conversation.id !== 'string') continue;
      const button = node('button', 'conversation-item');
      button.type = 'button';
      const current = conversation.id === state.conversation?.id;
      if (current) button.setAttribute('aria-current', 'true');
      button.append(node('span', 'conversation-title', conversation.title || '새 대화'));
      button.append(node('span', 'conversation-date', `${conversationTime(conversation.updated_at || conversation.created_at)}${current ? ' · 현재 대화' : ''}`));
      button.addEventListener('click', () => {
        el.history_dialog.close();
        loadConversation(`/api/conversations/${encodeURIComponent(conversation.id)}`);
      });
      el.history_list.append(button);
    }
  } catch (error) {
    if (error.name === 'AbortError' || generation !== state.listGeneration || !el.history_dialog.open) return;
    el.history_status.textContent = error.message;
    const retry = node('button', 'notice-retry', '목록 다시 불러오기');
    retry.type = 'button';
    retry.addEventListener('click', openHistory);
    el.history_status.append(retry);
  }
}
el.chat_form.addEventListener('submit', event => {event.preventDefault(); sendMessage(el.message.value);});
el.message.addEventListener('input', resizeComposer);
el.message.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault(); sendMessage(el.message.value);}
});
el.new_chat.addEventListener('click', newConversation);
el.history_button.addEventListener('click', openHistory);
el.close_history.addEventListener('click', () => el.history_dialog.close());
el.history_dialog.addEventListener('close', () => {state.listGeneration++; state.listController?.abort(); state.listController = null;});
updateControls();
loadStatus();
loadConversation();
