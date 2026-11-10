import { defineConfig } from 'vitest/config';

export default defineConfig({
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
  test: {
    include: ['tests/unit/**/*.test.ts'],
    environment: 'node',
    restoreMocks: true,
    unstubGlobals: true,
  },
});
