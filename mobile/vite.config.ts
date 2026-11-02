import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: 'www',
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    host: '127.0.0.1',
  },
});
