// @vitest-environment happy-dom

import { describe, it, expect } from 'vitest';
import { escapeHTML, formatText } from '../../src/utils/formatters';

function render(text: string): HTMLElement {
  const box = document.createElement('div');
  box.innerHTML = formatText(text);
  return box;
}
const tags = (el: HTMLElement) => new Set(Array.from(el.querySelectorAll('*')).map((e) => e.tagName));
const texts = (nodes: NodeListOf<Element>) => Array.from(nodes).map((n) => n.textContent);

// =====================================================================
// escapeHTML
// =====================================================================
describe('escapeHTML', () => {

  it.each([
    ['&', '&amp;'], ['<', '&lt;'], ['>', '&gt;'], ['"', '&quot;'], ["'", '&#39;'],
  ])('escapes %s', (raw, escaped) => {
    expect(escapeHTML(raw)).toBe(escaped);
    expect(escapeHTML(`a${raw}b`)).toBe(`a${escaped}b`);
  });

  it('neutralises a script tag with quotes and ampersands', () => {
    expect(escapeHTML(`<script>alert("XSS" & 'Hacked')</script>`))
      .toBe('&lt;script&gt;alert(&quot;XSS&quot; &amp; &#39;Hacked&#39;)&lt;/script&gt;');
  });

  it('escapes the ampersand first, so nothing is escaped twice within one pass', () => {
    expect(escapeHTML('<')).toBe('&lt;');
    expect(escapeHTML('&lt;')).toBe('&amp;lt;');
  });

  it('turns missing values (undefined / null) into an empty string instead of throwing', () => {
    expect(escapeHTML(undefined as unknown as string)).toBe('');
    expect(escapeHTML(null as unknown as string)).toBe('');
  });

  it('leaves text without special characters, unicode, emoji and newlines untouched', () => {
    for (const s of ['', 'Zwykły tekst 123', 'Zażółć gęślą jaźń', '🎓 AGH – Kraków', 'line1\nline2\ttab']) {
      expect(escapeHTML(s)).toBe(s);
    }
  });

  it('never leaves a raw <, >, " or \' in its output, and is reversible (property check)', () => {
    const alphabet = ['<', '>', '"', "'", '&', 'a', 'Z', ' ', '\n', 'ł', '🎓', ';', '#', 'l', 't'];
    let seed = 42;
    const rand = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
    const unescape = (s: string) => s.replace(/&lt;|&gt;|&quot;|&#39;|&amp;/g, (m) =>
      ({ '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'", '&amp;': '&' }[m] as string));
    for (let i = 0; i < 300; i++) {
      const s = Array.from({ length: Math.floor(rand() * 40) }, () => alphabet[Math.floor(rand() * alphabet.length)]).join('');
      const out = escapeHTML(s);
      expect(out).not.toMatch(/[<>"']/);
      expect(unescape(out)).toBe(s);
    }
  });
});

// =====================================================================
// formatText: structure
// =====================================================================
describe('formatText: basic formatting', () => {

  it('returns an empty string for empty input', () => {
    expect(formatText('')).toBe('');
  });

  it('wraps plain text in a paragraph', () => {
    expect(formatText('Dziekanat jest czynny.')).toBe('<p>Dziekanat jest czynny.</p>');
  });

  it('renders **bold** segments, several per line', () => {
    expect(formatText('a **b** c **d**')).toBe('<p>a <strong>b</strong> c <strong>d</strong></p>');
  });

  it('keeps an unfinished **bold** literal while the answer is still streaming', () => {
    expect(formatText('To jest **ważn')).toBe('<p>To jest **ważn</p>');
  });

  it('does not bold across a line break', () => {
    expect(formatText('**a\nb**')).toBe('<p>**a<br/>b**</p>');
  });

  it('turns every newline into <br/>', () => {
    expect(formatText('a\nb')).toBe('<p>a<br/>b</p>');
    expect(formatText('a\n\nb')).toBe('<p>a<br/><br/>b</p>');
  });

  it('renders "- " and "* " lines as list items', () => {
    for (const text of ['- A\n- B', '* A\n* B', '- A\n* B']) {
      const box = render(text);
      expect(texts(box.querySelectorAll('ul > li'))).toEqual(['A', 'B']);
    }
  });

  it('keeps the text before and after a list, and formats inside list items', () => {
    const box = render('Wymagane:\n- **Dowód** [1]\n- Zdjęcie\nPozdrawiam');
    expect(texts(box.querySelectorAll('li'))).toEqual(['Dowód [1]', 'Zdjęcie']);
    expect(box.querySelector('li strong')?.textContent).toBe('Dowód');
    expect(box.querySelector('li .citation')?.textContent).toBe('[1]');
    expect(box.textContent).toContain('Wymagane:');
    expect(box.textContent).toContain('Pozdrawiam');
  });

  it('does not treat hyphens inside a line, negative numbers or bold headings as list items', () => {
    for (const text of ['to - nie lista', '-5 stopni', '**Tytuł:**\nTreść', 'a-b', '---']) {
      expect(render(text).querySelectorAll('li').length).toBe(0);
    }
  });

  it('keeps numbered lists as plain text lines (not supported)', () => {
    const box = render('1. Pierwszy\n2. Drugi');
    expect(box.querySelectorAll('li').length).toBe(0);
    expect(box.textContent).toBe('1. Pierwszy2. Drugi');
    expect(box.querySelectorAll('br').length).toBe(1);
  });

  // KNOWN BUG - `it.fails` passes while the bug exists and starts failing once it is fixed (then remove `.fails`).
  // The greedy /(<li>.*<\/li>)/s wraps everything from the first to the last item in ONE <ul>, so a paragraph
  // between two separate lists ends up inside the list.
  it('keeps text between two separate lists out of the lists', () => {
    const box = render('- a\n\nOdstęp\n\n- b');
    const inside = Array.from(box.querySelectorAll('ul')).some((ul) => ul.textContent!.includes('Odstęp'));
    expect(inside).toBe(false);
  });
});

describe('formatText: citations', () => {

  it('turns [n] into a citation marker with its number in data-id', () => {
    const box = render('Tak [1] i [2][3] oraz [10].');
    const marks = box.querySelectorAll('span.citation');
    expect(texts(marks)).toEqual(['[1]', '[2]', '[3]', '[10]']);
    expect(Array.from(marks).map((m) => m.getAttribute('data-id'))).toEqual(['1', '2', '3', '10']);
  });

  it.each(['[abc]', '[1a]', '[]', '[ 1 ]', '[1.5]', '[-1]'])('leaves %s alone', (s) => {
    const box = render(`tekst ${s} koniec`);
    expect(box.querySelectorAll('.citation').length).toBe(0);
    expect(box.textContent).toBe(`tekst ${s} koniec`);
  });

});

// =====================================================================
// formatText: links (the model writes [label](https://...) and bare addresses)
// =====================================================================
describe('formatText: links', () => {

  const linkAttrs = (a: Element) => Array.from(a.attributes).map((x) => x.name).sort();

  it('renders a markdown link as a safe anchor that opens in a new tab', () => {
    const box = render('Więcej na stronie [Niepełnoletni](https://rekrutacja.agh.edu.pl/wpis-niestandardowy/#niepelnoletni).');
    const a = box.querySelector('a') as HTMLAnchorElement;
    expect(a.textContent).toBe('Niepełnoletni');
    expect(a.getAttribute('href')).toBe('https://rekrutacja.agh.edu.pl/wpis-niestandardowy/#niepelnoletni');
    expect(a.getAttribute('target')).toBe('_blank');
    expect(a.getAttribute('rel')).toBe('noopener noreferrer nofollow');
    expect(linkAttrs(a)).toEqual(['href', 'rel', 'target']);
    expect(box.textContent).toBe('Więcej na stronie Niepełnoletni.');
  });

  it('links bare http(s) addresses without the trailing punctuation', () => {
    const box = render('Zobacz https://www.agh.edu.pl/studenci. Albo http://example.com?a=1&b=2, dalej.');
    const hrefs = Array.from(box.querySelectorAll('a')).map((a) => a.getAttribute('href'));
    expect(hrefs).toEqual(['https://www.agh.edu.pl/studenci', 'http://example.com/?a=1&b=2']);
    expect(box.textContent).toBe('Zobacz https://www.agh.edu.pl/studenci. Albo http://example.com?a=1&b=2, dalej.');
  });

  it.each([
    '[klik](javascript:alert(1))',
    '[klik](data:text/html,<script>alert(1)</script>)',
    '[klik](vbscript:msgbox(1))',
    '[klik](//evil.example.com)',
    'javascript:alert(1)',
  ])('does not make %s clickable', (payload) => {
    const box = render(payload);
    expect(box.querySelectorAll('a').length).toBe(0);
    expect(box.textContent).toBe(payload);
  });

  it('cannot break out of the href attribute or inject handlers', () => {
    const box = render('[x](https://a.pl/"onmouseover="alert(1)) i [y](https://a.pl/\'><img src=x onerror=alert(1)>)');
    expect(box.querySelectorAll('img').length).toBe(0);
    for (const a of Array.from(box.querySelectorAll('a'))) {
      expect(linkAttrs(a)).toEqual(['href', 'rel', 'target']);
      expect(a.getAttribute('href')).toMatch(/^https:\/\/a\.pl\//);
      expect(a.getAttribute('href')).not.toMatch(/["'<> ]/);
    }
  });

  it('escapes HTML in the link label', () => {
    const box = render('[<img src=x onerror=alert(1)>](https://a.pl)');
    expect(box.querySelectorAll('img').length).toBe(0);
    expect(box.querySelector('a')?.textContent).toBe('<img src=x onerror=alert(1)>');
  });

  it('keeps links working inside lists, next to bold text and citations', () => {
    const box = render('Dokumenty:\n- **Regulamin** [Regulamin studiów](https://www.agh.edu.pl/regulamin.pdf) [2]\n- formularz https://a.pl/f.pdf [3]');
    expect(box.querySelectorAll('li').length).toBe(2);
    expect(texts(box.querySelectorAll('a'))).toEqual(['Regulamin studiów', 'https://a.pl/f.pdf']);
    expect(texts(box.querySelectorAll('.citation'))).toEqual(['[2]', '[3]']);
    expect(box.querySelector('strong')?.textContent).toBe('Regulamin');
  });

  it('does not turn citation numbers or bold markers inside an address into formatting', () => {
    const box = render('[strona](https://a.pl/x**y**/[1]/z)');
    expect(box.querySelectorAll('strong, .citation').length).toBe(0);
    expect(box.querySelector('a')?.getAttribute('href')).toBe('https://a.pl/x**y**/[1]/z');
  });

  it('shows a half-streamed markdown link as text until it is complete', () => {
    const full = 'Info: [Niepełnoletni](https://rekrutacja.agh.edu.pl/x) koniec';
    for (let i = 1; i <= full.length; i++) {
      const box = render(full.slice(0, i));
      for (const tag of tags(box)) expect(['P', 'UL', 'LI', 'STRONG', 'SPAN', 'BR', 'A']).toContain(tag);
    }
    expect(render(full).querySelector('a')?.textContent).toBe('Niepełnoletni');
  });
});

// =====================================================================
// formatText: security - it renders text that came from an LLM / scraped pages
// =====================================================================
describe('formatText: XSS safety', () => {

  const payloads: Array<[string, string]> = [
    ['<script>alert(1)</script>', '<script>alert(1)</script>'],
    ['<img src=x onerror=alert(1)>', '<img src=x onerror=alert(1)>'],
    ['<svg onload=alert(1)>', '<svg onload=alert(1)>'],
    ['<a href="javascript:alert(1)">klik</a>', '<a href="javascript:alert(1)">klik</a>'],
    ['"><img src=x onerror=alert(1)>', '"><img src=x onerror=alert(1)>'],
    ["' onmouseover='alert(1)", "' onmouseover='alert(1)"],
    ['<iframe src="javascript:alert(1)"></iframe>', '<iframe src="javascript:alert(1)"></iframe>'],
    ['<style>body{display:none}</style>', '<style>body{display:none}</style>'],
    ['<<script>script>alert(1)<</script>/script>', '<<script>script>alert(1)<</script>/script>'],
    ['**<img src=x onerror=alert(1)>**', '<img src=x onerror=alert(1)>'],
    ['- <img src=x onerror=alert(1)>', '<img src=x onerror=alert(1)>'],
    ['[1]<script>alert(1)</script>', '<script>alert(1)</script>'],
    ['&lt;script&gt;alert(1)&lt;/script&gt;', '&lt;script&gt;alert(1)&lt;/script&gt;'],
  ];

  it.each(payloads)('renders %s as harmless text', (payload, visible) => {
    const box = render(payload);
    for (const tag of tags(box)) expect(['P', 'UL', 'LI', 'STRONG', 'SPAN', 'BR']).toContain(tag);
    for (const el of Array.from(box.querySelectorAll('*'))) {
      for (const attr of Array.from(el.attributes)) expect(['class', 'data-id']).toContain(attr.name);
    }
    expect(box.textContent).toContain(visible);
  });

  it('only ever puts digits into the citation attribute', () => {
    const box = render('[1] [22] [a"onmouseover="x] [3]');
    for (const m of Array.from(box.querySelectorAll('.citation'))) expect(m.getAttribute('data-id')).toMatch(/^\d+$/);
  });
});

// =====================================================================
// formatText: robustness while streaming
// =====================================================================
describe('formatText: robustness', () => {

  const answer = 'Aby złożyć **wniosek** [1] należy:\n- wypełnić formularz [2]\n- dołączyć <zdjęcie> & "dowód"\nTerminy: 1-15 X; koszt 5 zł.';

  it('produces a valid, whitelisted paragraph for every prefix of a streamed answer', () => {
    for (let i = 1; i <= answer.length; i++) {
      const html = formatText(answer.slice(0, i));
      expect(html.startsWith('<p>')).toBe(true);
      expect(html.endsWith('</p>')).toBe(true);
      const box = document.createElement('div');
      box.innerHTML = html;
      for (const tag of tags(box)) expect(['P', 'UL', 'LI', 'STRONG', 'SPAN', 'BR']).toContain(tag);
    }
  });

  it('is fast on long answers (it is re-run on the whole text for every streamed token)', () => {
    const long = 'Zdanie **ważne** [1]\n- punkt\n'.repeat(3000);
    const started = Date.now();
    formatText(long);
    expect(Date.now() - started).toBeLessThan(2000);
  });

  it.each(['*', '- ', '[', '**', '\n', '[1]'])('does not hang on pathological repetition of %j', (unit) => {
    const started = Date.now();
    formatText(unit.repeat(20000));
    expect(Date.now() - started).toBeLessThan(2000);
  });
});