export function escapeHTML(str: string): string {
  return str.replace(/[&<>'"]/g, (tag) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;'
  }[tag] || tag));
}

export function formatText(text: string): string {
  if (!text) return '';
  let safeText = escapeHTML(text);
  safeText = safeText.replace(/^(?:-|\*)\s+(.*)$/gm, '<li>$1</li>');
  safeText = safeText.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
  safeText = safeText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  safeText = safeText.replace(/\[(\d+)\]/g, `<span class="citation" data-id="$1">[$1]</span>`);
  safeText = safeText.replace(/\n(?!<li|<\/ul>|<ul>)/g, '<br/>');
  return `<p>${safeText}</p>`;
}