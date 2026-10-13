import { sendChatMessage } from './api/client';
import { createSourcesComponent } from './components/sources';
import type { Message, ChatRequestPayload } from './types';


const DEV_MODE_SOURCES = true;   // Wyświetla listę źródeł (karty RRF) pod odpowiedzią
const DEV_MODE_SWITCHER = true;  // Wyświetla przełącznik modeli w nagłówku

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
    // 1. ODCZYT: Podczas ładowania strony sprawdź, czy mamy już zapisany model
    const savedModel = localStorage.getItem('rag_selected_model');
    if (savedModel) {
        modelSwitcher.value = savedModel;
    }

    // 2. ZAPIS: Gdy użytkownik zmieni model, zapisz to w pamięci przeglądarki
    modelSwitcher.addEventListener('change', (event) => {
        const target = event.target as HTMLSelectElement;
        localStorage.setItem('rag_selected_model', target.value);
        console.log("Zapisano model:", target.value); // Opcjonalnie do debugowania
    });
}
}

const savedHistory = sessionStorage.getItem('chatHistory');
let chatHistory: Message[] = savedHistory ? JSON.parse(savedHistory) : [
  { id: Date.now().toString(), role: 'assistant', content: 'Cześć! Jestem asystentem Akademii Górniczo-Hutniczej. W czym mogę Ci pomóc?' }
];

function saveHistory() {
  sessionStorage.setItem('chatHistory', JSON.stringify(chatHistory));
}

function appendMessage(message: Message): HTMLElement {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${message.role}`;
  msgDiv.id = message.id;

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.innerHTML = `<p>${message.content}</p>`;
  
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
  loadingText.textContent = 'Analiza historii i wyszukiwanie w bazach...';
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
      if (botContentDiv) botContentDiv.innerHTML = `<p>${botMsg.content}</p>`;
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
    }
  );

  if (isFirstEvent) loadingIndicator.classList.add('hidden');

  chatHistory.push(botMsg);
  saveHistory();

  input.disabled = false;
  sendBtn.disabled = false;
  input.focus();
});