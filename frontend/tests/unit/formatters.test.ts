import { describe, it, expect } from 'vitest';
import { escapeHTML, formatText } from '../../src/utils/formatters';

describe('Text Logic and Security (formatters.ts)', () => {
  
  describe('escapeHTML function', () => {
    
    it('should neutralize malicious HTML tags (XSS Protection)', () => {
      const maliciousCode = `<script>alert("XSS" & 'Hacked')</script>`;
      const expected = `&lt;script&gt;alert(&quot;XSS&quot; &amp; &#39;Hacked&#39;)&lt;/script&gt;`;
      
      expect(escapeHTML(maliciousCode)).toBe(expected);
    });

    it('should leave safe text alone', () => {
      const safeText = `The Dean's office is open from 8 to 15.`;
      
      expect(escapeHTML(safeText)).toBe(safeText);
    });
  });

  describe('formatText function', () => {
    
    it('should properly parse Markdown-style bolding', () => {
      const text = `This is **extremely** important.`;
      const expected = `<p>This is <strong>extremely</strong> important.</p>`;
      
      expect(formatText(text)).toBe(expected);
    });

    it('should convert newline characters into <br> tags', () => {
      const text = `Point 1\nPoint 2\nPoint 3`;
      const expected = `<p>Point 1<br>Point 2<br>Point 3</p>`;
      
      expect(formatText(text)).toBe(expected);
    });

    it('hybrid scenario: XSS escaping occurring alongside valid tags', () => {
      const complexText = `**Title:**\nWatch out for the <iframe src="virus"> tag.`;
      const expected = `<p><strong>Title:</strong><br>Watch out for the &lt;iframe src=&quot;virus&quot;&gt; tag.</p>`;
      
      expect(formatText(complexText)).toBe(expected);
    });
  });
});