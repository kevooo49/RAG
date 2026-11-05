import { sendChatMessage } from './api/client';
import { createSourcesComponent } from './components/sources';
import type { Message, ChatRequestPayload } from './types';
import { formatText } from './utils/formatters';

const DEV_MODE_SOURCES = true;
const DEV_MODE_SWITCHER = true;

const form = document.getElementById('chat-form') as HTMLFormElement;
const input = document.getElementById('message-input') as HTMLInputElement;
const sendBtn = document.getElementById('send-button') as HTMLButtonElement;
const messagesArea = document.getElementById('chat-messages') as HTMLDivElement;
const loadingIndicator = document.getElementById('loading-indicator') as HTMLDivElement;
const loadingText = document.getElementById('loading-text') as HTMLSpanElement;

if (DEV_MODE_SWITCHER) {
  const header = document.querySelector('.header-titles') as HTMLDivElement;
  const switcherHtml = `
    <select id="model-switcher" style="margin-top: 8px; padding: 4px; border-radius: 4px; border: 1px solid var(--border-color); font-size: 0.8rem; background: var(--bg-main); color: var(--text-primary); outline: none;">
      <option value="gpt-4o">GPT-4o</option>
      <option value="gpt-4o-mini">GPT-4o-mini</option>
      <option value="bielik">Bielik 11B</option>
    </select>
  `;
  header.insertAdjacentHTML('beforeend', switcherHtml);
  
  const modelSwitcher = document.getElementById('model-switcher') as HTMLSelectElement;
  if (modelSwitcher) {
    const savedModel = localStorage.getItem('rag_selected_model');
    if (savedModel) {
        modelSwitcher.value = savedModel;
    }
    modelSwitcher.addEventListener('change', (event) => {
        const target = event.target as HTMLSelectElement;
        localStorage.setItem('rag_selected_model', target.value);
    });
  }
}

let chatHistory: Message[] = [
  { id: Date.now().toString(), role: 'assistant', content: 'Cześć! Jestem asystentem Akademii Górniczo-Hutniczej. W czym mogę Ci pomóc?' }
];
const savedHistory = sessionStorage.getItem('chatHistory');
if (savedHistory) {
  try {
    chatHistory = JSON.parse(savedHistory);
  } catch (e) {
    console.error("Uszkodzona historia w sessionStorage, resetuję.", e);
    sessionStorage.removeItem('chatHistory');
  }
}

function saveHistory() {
  const historyToSave = chatHistory.map(m => ({ id: m.id, role: m.role, content: m.content }));
  sessionStorage.setItem('chatHistory', JSON.stringify(historyToSave));
}

function appendMessage(message: Message): HTMLElement {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${message.role}`;
  msgDiv.id = message.id;

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.innerHTML = formatText(message.content);
  
  msgDiv.appendChild(contentDiv);
  messagesArea.appendChild(msgDiv);
  messagesArea.scrollTop = messagesArea.scrollHeight;
  return contentDiv;
}

chatHistory.forEach(msg => {
  const contentDiv = appendMessage(msg);
  if (msg.sources && DEV_MODE_SOURCES) {
    const sourcesEl = createSourcesComponent(msg.sources);
    contentDiv.parentElement!.appendChild(sourcesEl);
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  const userMsg: Message = { id: Date.now().toString(), role: 'user', content: query };
  chatHistory.push(userMsg);
  appendMessage(userMsg);
  saveHistory();

  input.value = '';
  input.disabled = true;
  sendBtn.disabled = true;

  loadingIndicator.classList.remove('hidden');
  loadingText.textContent = 'Analiza zapytania...';
  messagesArea.appendChild(loadingIndicator);
  messagesArea.scrollTop = messagesArea.scrollHeight;

  const payload: ChatRequestPayload = {
    query: query,
    model: (document.getElementById('model-switcher') as HTMLSelectElement)?.value || 'gpt-4o-mini',
    history: chatHistory.slice(0, -1).map(m => ({ role: m.role, content: m.content }))
  };

  const botMsg: Message = { id: (Date.now() + 1).toString(), role: 'assistant', content: '' };
  let botContentDiv: HTMLElement | null = null;
  let isFirstEvent = true;

  const handleFirstEvent = () => {
    if (isFirstEvent) {
      loadingIndicator.classList.add('hidden');
      botContentDiv = appendMessage(botMsg);
      isFirstEvent = false;
    }
  };

  await sendChatMessage(
    payload,
    (chunk) => {
      handleFirstEvent();
      botMsg.content += chunk;
      if (botContentDiv) botContentDiv.innerHTML = formatText(botMsg.content);
      messagesArea.scrollTop = messagesArea.scrollHeight;
    },
    (sources) => {
      handleFirstEvent();
      botMsg.sources = sources;
      
      if (DEV_MODE_SOURCES && botContentDiv) {
        const sourcesEl = createSourcesComponent(sources);
        botContentDiv.parentElement!.appendChild(sourcesEl);
        messagesArea.scrollTop = messagesArea.scrollHeight;
      }
    },
    (status) => {
      loadingText.textContent = status;
    },
    (err) => {
      handleFirstEvent();
      if (botContentDiv) {
        botContentDiv.innerHTML += `<div class="error-msg" style="color: red; margin-top: 10px;">${err}</div>`;
      }
    }
  );

  if (isFirstEvent) loadingIndicator.classList.add('hidden');

  chatHistory.push(botMsg);
  saveHistory();

  input.disabled = false;
  sendBtn.disabled = false;
  input.focus();
});