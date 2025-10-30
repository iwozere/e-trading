/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_WS_URL: string
  readonly VITE_API_URL: string
  // Add other VITE_ environment variables here as needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}