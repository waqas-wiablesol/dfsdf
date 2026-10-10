import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Two ways to run the interface:
//
//   npm run dev    -> this dev server, with hot reload. Nothing to rebuild, so a
//                     stale page is impossible by construction.
//   npm run build  -> writes into the Python package, so `python -m
//                     wilcom_pipeline.api` serves it with no second process.
//
// The dev server is the better default while the interface is still changing: it
// removes a whole class of confusion where an edit appears to do nothing because
// the served bundle is older than the source.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../wilcom_pipeline/webui',
    emptyOutDir: true,
  },
  server: {
    // Bound to IPv4 explicitly. Vite's default is `localhost`, which on Windows
    // resolves to `::1` only -- so the server is reachable at
    // http://localhost:5173 but NOT at http://127.0.0.1:5173, and anything that
    // prefers IPv4 (a script, a tool, a second browser profile) cannot connect.
    // Pinning it makes one address work for everything.
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    // The API is a separate process on its own port, so the same relative `/api`
    // paths work in development and after the build. `changeOrigin` keeps the Host
    // header sane.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
