/**
 * @vitest-environment happy-dom
 */

import { describe, it, expect } from 'vitest';
import { createSourcesComponent } from '../../src/components/sources';

interface MockSource {
  id: string | number;
  text?: string;
  meta?: {
    title?: string;
    url?: string;
  };
}

describe('UI Element Rendering (sources.ts)', () => {
  
  it('should generate an empty main container when there are no sources', () => {
    const emptyArray: MockSource[] = [];
    const el = createSourcesComponent(emptyArray as any);

    expect(el).toBeInstanceOf(HTMLElement);
    expect(el.querySelectorAll('.source-card').length).toBe(0);
  });

  it('should generate the appropriate number of source cards (.source-card)', () => {
    const mockSources: MockSource[] = [
      { id: 1, meta: { title: 'Regulations' } },
      { id: 2, meta: { title: 'Scholarships' } },
      { id: 3, meta: { title: 'Kapitol Dormitory' } }
    ];
    
    const el = createSourcesComponent(mockSources as any);
    const cards = el.querySelectorAll('.source-card');

    expect(cards.length).toBe(3);
  });

  it('should properly bind metadata (title, url) and sequential number inside the card', () => {
    const mockSources: MockSource[] = [
      { 
        id: 42,
        meta: { 
          title: 'WIEiT Admission Rules',
          url: 'https://kandydaci.agh.edu.pl'
        } 
      }
    ];
    
    const el = createSourcesComponent(mockSources as any);
    const card = el.querySelector('.source-card');
    
    expect(card).not.toBeNull();
    
    const cardHtml = card!.innerHTML;
    
    expect(cardHtml).toContain('#1');
    expect(cardHtml).toContain('WIEiT Admission Rules');
    
    if (cardHtml.includes('<a')) {
      expect(cardHtml).toContain('href="https://kandydaci.agh.edu.pl"');
    }
  });

});