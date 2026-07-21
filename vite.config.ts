import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const contentRoots = {
  "/backgrounds": path.resolve(__dirname, "backgrounds"),
  "/models": path.resolve(__dirname, "models/live2d"),
  "/reference_sounds": path.resolve(__dirname, "reference_sounds"),
};

function isInside(basePath: string, filePath: string) {
  const child = path.relative(basePath, filePath);
  return child === "" || (!child.startsWith("..") && !path.isAbsolute(child));
}

function walkFiles(basePath: string) {
  if (!fs.existsSync(basePath)) return [];
  const files: string[] = [];
  const stack = [basePath];

  while (stack.length) {
    const current = stack.pop();
    if (!current) continue;
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
      } else if (entry.isFile()) {
        files.push(fullPath);
      }
    }
  }

  return files;
}

function assetName(filePath: string) {
  return path.basename(filePath, path.extname(filePath)).replace(/[_-]+/g, " ");
}

function listBackgrounds() {
  const supported = new Set([".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"]);
  return walkFiles(contentRoots["/backgrounds"])
    .filter((filePath) => supported.has(path.extname(filePath).toLowerCase()))
    .map((filePath) => {
      const assetPath = path.relative(contentRoots["/backgrounds"], filePath).replace(/\\/g, "/");
      return { name: assetName(filePath), url: `/backgrounds/${assetPath}` };
    })
    .sort((a, b) => a.name.localeCompare(b.name, "zh-CN"));
}

function listLive2DModels() {
  return walkFiles(contentRoots["/models"])
    .filter((filePath) => filePath.toLowerCase().endsWith(".model3.json"))
    .map((filePath) => {
      const relativeModelFile = path.relative(contentRoots["/models"], filePath).replace(/\\/g, "/");
      const fileName = path.basename(filePath, ".model3.json");
      return {
        id: relativeModelFile.split("/")[0] || fileName,
        name: fileName.replace(/[_-]+/g, " "),
        directory: path.dirname(relativeModelFile).replace(/\\/g, "/"),
        fileName,
        scale: 0.9,
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name, "zh-CN"));
}

function contentPlugin() {
  return {
    name: "melomate-content",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const url = new URL(request.url || "/", "http://localhost");

        if (url.pathname === "/api/backgrounds") {
          response.setHeader("Content-Type", "application/json; charset=utf-8");
          response.end(JSON.stringify({ backgrounds: listBackgrounds() }));
          return;
        }

        if (url.pathname === "/api/live2d-models") {
          response.setHeader("Content-Type", "application/json; charset=utf-8");
          response.end(JSON.stringify({ models: listLive2DModels() }));
          return;
        }

        for (const [prefix, basePath] of Object.entries(contentRoots)) {
          if (url.pathname === prefix || url.pathname.startsWith(`${prefix}/`)) {
            const filePath = path.resolve(basePath, decodeURIComponent(url.pathname.slice(prefix.length)));
            if (!isInside(basePath, filePath) || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
              next();
              return;
            }
            response.setHeader("Cache-Control", "no-store");
            fs.createReadStream(filePath).pipe(response);
            return;
          }
        }

        next();
      });
    },
  };
}

export default {
  root: __dirname,
  publicDir: path.resolve(__dirname, "public"),
  base: "./",
  resolve: {
    alias: {
      "@framework": path.resolve(__dirname, "WebSDK/Framework/src"),
      "@cubismsdksamples": path.resolve(__dirname, "WebSDK/src"),
    },
  },
  optimizeDeps: {
    entries: ["src/main.ts"],
  },
  plugins: [contentPlugin()],
  build: {
    outDir: path.resolve(__dirname, "dist"),
    emptyOutDir: true,
    rollupOptions: {
      input: path.resolve(__dirname, "index.html"),
    },
  },
  server: {},
};
