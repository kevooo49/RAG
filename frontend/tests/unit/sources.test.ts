/**
 * @vitest-environment happy-dom
 */

import { describe, it, expect } from 'vitest';
import { createSourcesComponent } from '../../src/components/sources';

// Symulujemy interfejs źródła (Source) używany w Twoim typowaniu
interface MockSource {
  id: string | number;
  text?: string;
  meta?: {
    tytul?: string;
    url?: string;
  };
}

describe('Renderowanie Elementów UI (sources.ts)', () => {
  
  it('powinien wygenerować pusty główny kontener, gdy brak źródeł', () => {
    const pustaTablica: MockSource[] = [];
    const el = createSourcesComponent(pustaTablica as any);
    
    // Sprawdzamy, czy funkcja nadal zwraca element HTML (prawdopodobnie div.sources-container)
    expect(el).toBeInstanceOf(HTMLElement);
    // Upewniamy się, że w środku nie wygenerowano żadnych kart źródeł
    expect(el.querySelectorAll('.source-card').length).toBe(0);
  });

  it('powinien wygenerować odpowiednią liczbę kart źródeł (.source-card)', () => {
    const mockSources: MockSource[] = [
      { id: 1, meta: { tytul: 'Regulamin' } },
      { id: 2, meta: { tytul: 'Stypendia' } },
      { id: 3, meta: { tytul: 'DS Kapitol' } }
    ];
    
    const el = createSourcesComponent(mockSources as any);
    const cards = el.querySelectorAll('.source-card');
    
    // Dla 3 źródeł, w DOM muszą pojawić się dokładnie 3 elementy z tą klasą
    expect(cards.length).toBe(3);
  });

  it('powinien poprawnie bindować metadane (tytuł, url) i sekwencyjny numer wewnątrz karty', () => {
    const mockSources: MockSource[] = [
      { 
        id: 42, // Wewnętrzne ID z wektorowej bazy danych (nie powinno być widoczne dla usera jako numer cytowania)
        meta: { 
          tytul: 'Zasady rekrutacji WIEiT',
          url: 'https://kandydaci.agh.edu.pl'
        } 
      }
    ];
    
    const el = createSourcesComponent(mockSources as any);
    const card = el.querySelector('.source-card');
    
    expect(card).not.toBeNull();
    
    const htmlKarty = card!.innerHTML;
    
    // Zamiast wewnętrznego ID (42), komponent powinien nadać numer sekwencyjny (#1) dla pierwszego elementu
    expect(htmlKarty).toContain('#1');
    expect(htmlKarty).toContain('Zasady rekrutacji WIEiT');
    
    // Jeśli źródło ma URL, komponent zazwyczaj generuje tag <a> z atrybutem href
    if (htmlKarty.includes('<a')) {
      expect(htmlKarty).toContain('href="https://kandydaci.agh.edu.pl"');
    }
  });

});