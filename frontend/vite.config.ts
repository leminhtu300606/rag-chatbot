import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` (proxy /api -> FastAPI :8000).
// Prod: `npm run build` -> dist/, được FastAPI serve tại /.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
