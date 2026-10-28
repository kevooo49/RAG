import { describe, it, expect } from 'vitest';
import { escapeHTML, formatText } from '../../src/utils/formatters';

describe('Logika Tekstowa i Bezpieczeństwo (formatters.ts)', () => {
  
  describe('Funkcja escapeHTML', () => {
    
    it('powinna neutralizować złośliwe tagi HTML (Ochrona przed XSS)', () => {
      const zlosliwyKod = `<script>alert("XSS" & 'Hacked')</script>`;
      const oczekiwany = `&lt;script&gt;alert(&quot;XSS&quot; &amp; &#39;Hacked&#39;)&lt;/script&gt;`;
      
      expect(escapeHTML(zlosliwyKod)).toBe(oczekiwany);
    });

    it('powinna zostawiać bezpieczny tekst w spokoju', () => {
      const bezpiecznyTekst = `Dziekanat czynny od 8 do 15.`;
      
      expect(escapeHTML(bezpiecznyTekst)).toBe(bezpiecznyTekst);
    });
  });

  describe('Funkcja formatText', () => {
    
    it('powinna poprawnie parsować pogrubienia w stylu Markdown', () => {
      const tekst = `To jest **niezwykle** ważne.`;
      const oczekiwany = `<p>To jest <strong>niezwykle</strong> ważne.</p>`;
      
      expect(formatText(tekst)).toBe(oczekiwany);
    });

    it('powinna przekształcać znaki nowej linii w tagi <br>', () => {
      const tekst = `Punkt 1\nPunkt 2\nPunkt 3`;
      const oczekiwany = `<p>Punkt 1<br>Punkt 2<br>Punkt 3</p>`;
      
      expect(formatText(tekst)).toBe(oczekiwany);
    });

    it('scenariusz hybrydowy: ucieczka XSS zachodząca razem z prawidłowymi tagami', () => {
      const skomplikowanyTekst = `**Tytuł:**\nUwaga na tag <iframe src="virus">.`;
      const oczekiwany = `<p><strong>Tytuł:</strong><br>Uwaga na tag &lt;iframe src=&quot;virus&quot;&gt;.</p>`;
      
      expect(formatText(skomplikowanyTekst)).toBe(oczekiwany);
    });
  });
});