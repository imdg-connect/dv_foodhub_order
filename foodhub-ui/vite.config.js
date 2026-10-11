import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // agents_api: /api/ask, /api/orders, /api/ws (websocket)  ->  http://127.0.0.1:8003
      "/api":  { target: "http://127.0.0.1:8003", changeOrigin: true, ws: true, rewrite: (p) => p.replace(/^\/api/, "") },
      // dashboard_api: /dash/start, /dash/stop, /dash/status  ->  http://127.0.0.1:8002
      "/dash": { target: "http://127.0.0.1:8002", changeOrigin: true, rewrite: (p) => p.replace(/^\/dash/, "") },
    },
  },
});