export function escapeHTML(str: string): string {
  if (!str) return '';
  return str.replace(/[&<>'"]/g, (tag) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;'
  }[tag] || tag));
}

const MD_LINK = /\[([^\]\n]{1,300})\]\((https?:\/\/[^\s)]{1,2000})\)/g;
const BARE_URL = /\bhttps?:\/\/[^\s<>"'()[\]]{1,2000}/g;
const TRAILING_PUNCT = /[.,;:!?]+$/;
const SLOT = /\u0000(\d+)\u0000/g;

function safeLink(label: string, url: string): string | null {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null;
  return `<a href="${escapeHTML(parsed.href)}" target="_blank" rel="noopener noreferrer nofollow">${escapeHTML(label)}</a>`;
}

export function formatText(text: string): string {
  if (!text) return '';
  const links: string[] = [];
  const keep = (html: string | null, original: string) => {
    if (!html) return original;
    links.push(html);
    return `\u0000${links.length - 1}\u0000`;
  };
  let raw = text.replace(/\u0000/g, '');
  raw = raw.replace(MD_LINK, (m, label: string, url: string) => keep(safeLink(label, url), m));
  raw = raw.replace(BARE_URL, (m) => {
    const url = m.replace(TRAILING_PUNCT, '');
    return keep(safeLink(url, url), url) + m.slice(url.length);
  });

  let safeText = escapeHTML(raw);
  safeText = safeText.replace(/^(?:-|\*)\s+(.*)$/gm, '<li>$1</li>');
  safeText = safeText.replace(/(<li>.*?<\/li>(?:[\r\n]*<li>.*?<\/li>)*)/g, '<ul>$1</ul>');
  safeText = safeText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  safeText = safeText.replace(/\[(\d+)\]/g, `<span class="citation" data-id="$1">[$1]</span>`);
  safeText = safeText.replace(/\n(?!<li|<\/ul>|<ul>)/g, '<br/>');

  safeText = safeText.replace(SLOT, (_, i: string) => links[Number(i)] ?? '');
  return `<p>${safeText}</p>`;
}