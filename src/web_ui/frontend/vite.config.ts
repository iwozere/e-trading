import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { readFileSync } from 'fs'

// App version + build timestamp, baked in at build time so the UI can display
// exactly which build is loaded (useful for confirming a deploy refreshed).
const pkg = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf-8')
)

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
  },
  server: {
    port: 5002, // TRADING_WEBGUI_PORT
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:5003', // TRADING_API_PORT
        changeOrigin: true,
      },
      '/auth': {
        target: 'http://localhost:5003', // TRADING_API_PORT
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:5003', // TRADING_API_PORT WebSocket
        ws: true,
        changeOrigin: true,
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  // Move node_modules to .venv directory
  cacheDir: path.resolve(__dirname, '../../../.venv/frontend-cache'),
})