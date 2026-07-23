import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI backend restricts CORS to its own origin, so instead of calling
// it cross-origin we proxy /api and /uploads through the Vite dev server.
// Override the target with VITE_API_TARGET if the backend runs elsewhere.
const API_TARGET = process.env.VITE_API_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
      "/uploads": { target: API_TARGET, changeOrigin: true },
    },
  },
});
