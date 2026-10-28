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
  let safeText = escapeHTML(text);
  safeText = safeText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  safeText = safeText.replace(/\n/g, '<br>');
  return `<p>${safeText}</p>`;
}