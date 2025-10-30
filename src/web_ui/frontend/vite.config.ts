import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
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