/**
 * @vitest-environment happy-dom
 */
import { describe, it, expect } from 'vitest';
import { createSourcesComponent } from '../../src/components/sources';
import type { SourceDocument } from '../../src/types';

/** A source exactly as the backend sends it (English meta keys, see rag_service `_META_KEYS`). */
const src = (over: Partial<SourceDocument> & { meta?: SourceDocument['meta'] } = {}): SourceDocument => ({
  text: 'Akademik kosztuje 500 zł miesięcznie.',
  score: 0.0327868,
  ...over,
  meta: { title: 'Cennik DS', section: 'Ceny', source: 'Miasteczko AGH', url: 'https://miasteczko.agh.edu.pl/cennik', ...over.meta },
});
const cards = (el: HTMLElement) => Array.from(el.querySelectorAll<HTMLElement>('.source-card'));
const text = (el: Element | null) => el?.textContent?.replace(/\s+/g, ' ').trim();

// =====================================================================
// Rendering
// =====================================================================
describe('createSourcesComponent: rendering', () => {

  it('builds a container with a toggle button and a list', () => {
    const el = createSourcesComponent([src()]);
    expect(el.className).toBe('sources-container');
    expect(el.querySelector('button.sources-toggle')).not.toBeNull();
    expect(el.querySelector('.sources-list')).not.toBeNull();
  });

  it('renders no cards and a zero count for no sources', () => {
    const el = createSourcesComponent([]);
    expect(cards(el)).toHaveLength(0);
    expect(el.querySelector('.sources-toggle')!.textContent).toContain('(0)');
  });

  it('renders one card per source and shows the count on the toggle', () => {
    const el = createSourcesComponent([src(), src(), src()]);
    expect(cards(el)).toHaveLength(3);
    expect(el.querySelector('.sources-toggle')!.textContent).toContain('(3)');
  });

  it('numbers cards from #1 in the order received', () => {
    const el = createSourcesComponent([src({ meta: { title: 'Pierwszy' } }), src({ meta: { title: 'Drugi' } })]);
    expect(cards(el).map((c) => text(c.querySelector('.source-title')))).toEqual(['#1 Pierwszy', '#2 Drugi']);
  });

  it('shows title, section, full text and a 5-decimal RRF score', () => {
    const [card] = cards(createSourcesComponent([src()]));
    expect(text(card.querySelector('.source-title'))).toBe('#1 Cennik DS');
    expect(text(card.querySelector('.source-meta i'))).toBe('Ceny');
    expect(card.querySelector('.source-text')!.textContent).toBe('Akademik kosztuje 500 zł miesięcznie.');
    expect(text(card.querySelector('.rrf-badge'))).toBe('RRF: 0.03279');
  });

  it('shows N/A when there is no score', () => {
    const [card] = cards(createSourcesComponent([{ ...src(), score: undefined as unknown as number }]));
    expect(text(card.querySelector('.rrf-badge'))).toBe('RRF: N/A');
  });

  it('falls back to a default title and section when the metadata is empty', () => {
    const [card] = cards(createSourcesComponent([src({ meta: { title: '', section: '', url: '' } })]));
    expect(text(card.querySelector('.source-title'))).toBe('#1 Dokument AGH');
    expect(text(card.querySelector('.source-meta i'))).toBe('Baza główna');
  });

  it('does not crash when a source has no metadata at all', () => {
    const bare = { text: 'tekst', score: 0.1 } as unknown as SourceDocument;
    const [card] = cards(createSourcesComponent([bare]));
    expect(text(card.querySelector('.source-title'))).toBe('#1 Dokument AGH');
  });

  it('renders hundreds of sources', () => {
    expect(cards(createSourcesComponent(Array.from({ length: 300 }, () => src())))).toHaveLength(300);
  });
});

// =====================================================================
// Links
// =====================================================================
describe('createSourcesComponent: links', () => {

  it('opens a valid http(s) source in a new tab without leaking the opener', () => {
    for (const url of ['https://css.agh.edu.pl/stypendia', 'http://stare.agh.edu.pl/x']) {
      const [card] = cards(createSourcesComponent([src({ meta: { url } })]));
      const link = card.querySelector('a')!;
      expect(link.getAttribute('href')).toBe(url);
      expect(link.getAttribute('target')).toBe('_blank');
      expect(link.getAttribute('rel')).toContain('noopener');
      expect(link.getAttribute('rel')).toContain('noreferrer');
    }
  });

  it.each([
    'javascript:alert(1)', 'JaVaScRiPt:alert(1)', 'data:text/html,<script>alert(1)</script>',
    'vbscript:msgbox(1)', 'ftp://x.pl/a', '//evil.example/x', '/wzgledny/adres', 'mailto:a@b.pl',
  ])('never links to %s', (url) => {
    const el = createSourcesComponent([src({ meta: { url } })]);
    for (const a of Array.from(el.querySelectorAll('a'))) {
      expect(a.getAttribute('href')).not.toMatch(/^(javascript|data|vbscript|ftp|mailto):/i);
      expect(a.getAttribute('href')).toMatch(/^(https?:\/\/|#$)/);
    }
  });

  // A "link" to "#" that opens a NEW TAB just reloads the chat app. Documents (PDF/DOCX) have no URL,
  // so this affects real sources: with no usable URL there should simply be no link.
  it.each(['', undefined, 'javascript:alert(1)', '/wzgledny'])('renders no link at all when the URL is %j', (url) => {
    const [card] = cards(createSourcesComponent([src({ meta: { url } })]));
    expect(card.querySelector('a')).toBeNull();
  });

  it('cannot be tricked into adding attributes through a crafted URL', () => {
    const url = 'https://x.pl/" onmouseover="alert(1)" data-x="';
    const link = cards(createSourcesComponent([src({ meta: { url } })]))[0].querySelector('a');
    if (link) {
      expect(link.getAttributeNames().sort()).toEqual(['href', 'rel', 'target']);
      expect(link.getAttribute('href')).toBe(url);
    }
  });
});

// =====================================================================
// XSS: titles, sections and texts come from scraped web pages and PDFs
// =====================================================================
describe('createSourcesComponent: XSS safety', () => {

  const payloads = [
    '<img src=x onerror=alert(1)>', '<script>alert(1)</script>', '"><svg onload=alert(1)>',
    '<iframe src="javascript:alert(1)"></iframe>', '</div><b id=pwned>x</b>', "'; alert(1); //",
  ];

  it.each(payloads)('shows %s as plain text in title, section and body', (payload) => {
    const el = createSourcesComponent([src({ text: payload, meta: { title: payload, section: payload } })]);
    const [card] = cards(el);
    // the toggle button has its own chevron <svg>, so look inside the card only
    expect(card.querySelectorAll('img, script, svg, iframe, b, style').length).toBe(0);
    for (const e of Array.from(card.querySelectorAll('*'))) {
      for (const name of e.getAttributeNames()) expect(name.startsWith('on')).toBe(false);
    }
    expect(card.querySelector('.source-text')!.textContent).toBe(payload);
    expect(text(card.querySelector('.source-title'))).toBe(`#1 ${payload}`);
    expect(text(card.querySelector('.source-meta i'))).toBe(payload);
  });

  it('keeps ampersands and angle brackets in ordinary text readable', () => {
    const [card] = cards(createSourcesComponent([src({ text: 'a < b && c > d', meta: { title: 'R&D "Lab"' } })]));
    expect(card.querySelector('.source-text')!.textContent).toBe('a < b && c > d');
    expect(text(card.querySelector('.source-title'))).toBe('#1 R&D "Lab"');
  });
});

// =====================================================================
// Toggle
// =====================================================================
describe('createSourcesComponent: toggle', () => {

  const parts = (el: HTMLElement) => ({
    button: el.querySelector('button.sources-toggle') as HTMLButtonElement,
    list: el.querySelector('.sources-list') as HTMLElement,
  });

  it('is collapsed at first, opens on click and closes on the next click', () => {
    const { button, list } = parts(createSourcesComponent([src()]));
    expect(list.classList.contains('open')).toBe(false);
    expect(button.classList.contains('open')).toBe(false);

    button.click();
    expect(list.classList.contains('open')).toBe(true);
    expect(button.classList.contains('open')).toBe(true);

    button.click();
    expect(list.classList.contains('open')).toBe(false);
    expect(button.classList.contains('open')).toBe(false);
  });

  it('works independently for several answers on the page', () => {
    const a = parts(createSourcesComponent([src()]));
    const b = parts(createSourcesComponent([src()]));
    a.button.click();
    expect(a.list.classList.contains('open')).toBe(true);
    expect(b.list.classList.contains('open')).toBe(false);
  });

  it('uses a real <button> so it is focusable and works from the keyboard', () => {
    expect(parts(createSourcesComponent([src()])).button.tagName).toBe('BUTTON');
  });
});
