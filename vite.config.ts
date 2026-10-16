import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

// The website calls the FastAPI backend (main.py) as relative /api/... URLs.
// In dev and preview, Vite forwards them to the backend, so no CORS or hard-coded host is needed.
// Point it elsewhere with EMISSIONX_API_URL, e.g. EMISSIONX_API_URL=http://192.168.1.20:8000
const apiProxy = {
  '/api': {
    target: process.env.EMISSIONX_API_URL || 'http://127.0.0.1:8000',
    changeOrigin: true,
  },
};

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
      proxy: apiProxy,
    },
    preview: {
      proxy: apiProxy,
    },
  };
});
