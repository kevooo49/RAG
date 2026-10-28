import type { SourceDocument } from '../types';

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
    
    const title = source.meta?.tytul || 'Dokument AGH';
    const section = source.meta?.sekcja || 'Baza główna';
    const url = source.meta?.url || '#';
    const score = source.score ? source.score.toFixed(5) : 'N/A';

    card.innerHTML = `
      <div class="source-header">
        <span class="source-title">#${index + 1} ${title}</span>
        <span class="rrf-badge">RRF: ${score}</span>
      </div>
      <div class="source-meta">
        Sekcja: <i>${section}</i> | <a href="${url}" target="_blank">🔗 Otwórz stronę</a>
      </div>
      <div class="source-text">${source.text}</div>
    `;
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