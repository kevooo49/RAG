import type { SourceDocument } from '../types';
import { escapeHTML } from '../utils/formatters';

export function createSourcesComponent(sources: SourceDocument[]): HTMLElement {
  const container = document.createElement('div');
  container.className = 'sources-container';

  const toggleBtn = document.createElement('button');
  toggleBtn.className = 'sources-toggle';
  toggleBtn.innerHTML = `
    <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none">
      <polyline points="6 9 12 15 18 9"></polyline>
    </svg>
    Zobacz źródła i cytowane fragmenty (${sources.length})
  `;

  const list = document.createElement('div');
  list.className = 'sources-list';

sources.forEach((source, index) => {
    const card = document.createElement('div');
    card.className = 'source-card';

    const title = escapeHTML(source.meta?.tytul || 'Dokument AGH');
    const section = escapeHTML(source.meta?.sekcja || 'Baza główna');
    const safeText = escapeHTML(source.text);
    const score = source.score ? source.score.toFixed(5) : 'N/A';

    let url = source.meta?.url || '#';
    if (url !== '#' && !url.startsWith('http://') && !url.startsWith('https://')) {
      url = '#';
    }

    card.innerHTML = `
      <div class="source-header">
        <span class="source-title">#${index + 1} ${title}</span>
        <span class="rrf-badge">RRF: ${score}</span>
      </div>
      <div class="source-meta">
        Sekcja: <i>${section}</i> | <span class="link-placeholder"></span>
      </div>
      <div class="source-text">${safeText}</div>
    `;
    
    const linkEl = document.createElement('a');
    linkEl.href = url;
    linkEl.target = '_blank';
    linkEl.rel = 'noopener noreferrer';
    linkEl.textContent = '🔗 Otwórz stronę';
    
    const placeholder = card.querySelector('.link-placeholder');
    if (placeholder) {
      placeholder.replaceWith(linkEl);
    }
    
    list.appendChild(card);
  });

  toggleBtn.addEventListener('click', () => {
    const isOpen = list.classList.contains('open');
    if (isOpen) {
      list.classList.remove('open');
      toggleBtn.classList.remove('open');
    } else {
      list.classList.add('open');
      toggleBtn.classList.add('open');
    }
  });

  container.appendChild(toggleBtn);
  container.appendChild(list);

  return container;
}