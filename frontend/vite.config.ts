import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const headers = {
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "no-referrer",
  "X-Frame-Options": "DENY",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
};

export default defineConfig({
  plugins: [react()],
  build: { manifest: true },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    headers,
    fs: {
      strict: true,
      deny: [".env", ".env.*", "*.{crt,pem,key}", "**/.git/**", "**/.venv/**", "**/data/**"]
    }
  },
  preview: {
    host: "127.0.0.1",
    strictPort: true,
    headers
  }
});
