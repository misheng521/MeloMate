import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const projectRoot = path.dirname(fileURLToPath(import.meta.url));

// Vite is intentionally build-only. The authenticated runtime APIs and isolated
// workspace origin have one implementation in server.mjs.
export default defineConfig({
  root: projectRoot,
  publicDir: path.resolve(projectRoot, "public"),
  base: "./",
  optimizeDeps: {
    entries: ["src/main.ts"],
  },
  build: {
    outDir: path.resolve(projectRoot, "dist"),
    emptyOutDir: true,
    rollupOptions: {
      input: path.resolve(projectRoot, "index.html"),
    },
  },
});
