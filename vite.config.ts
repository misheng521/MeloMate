import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const contentRoots = {
  "/backgrounds": path.resolve(__dirname, "backgrounds"),
  "/models": path.resolve(__dirname, "models/live2d"),
  "/reference_sounds": path.resolve(__dirname, "reference_sounds"),
};
const workspaceRoot = path.resolve(__dirname, "workspace");
const responseTypes: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
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

function safeName(value: string, fallback = "default") {
  const cleaned = String(value || "")
    .trim()
    .replace(/\.(ya?ml)$/i, "")
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, "_")
    .replace(/\s+/g, "_")
    .replace(/^[ .]+|[ .]+$/g, "");
  return cleaned || fallback;
}

function safeWorkspaceFolder(value: string) {
  return String(value || "")
    .split(/[\\/]+/)
    .filter((part) => part && part !== "." && part !== "..")
    .map((part) => safeName(part, ""))
    .filter(Boolean)
    .join("/");
}

function listWorkspace(persona: string, folder: string) {
  const safePersona = safeName(persona);
  const safeFolder = safeWorkspaceFolder(folder);
  const personaRoot = path.resolve(workspaceRoot, safePersona);
  const target = path.resolve(personaRoot, safeFolder);

  if (!isInside(personaRoot, target)) {
    return { persona: safePersona, folder: "", entries: [] };
  }

  fs.mkdirSync(target, { recursive: true });
  const entries = fs
    .readdirSync(target, { withFileTypes: true })
    .filter((entry) => !entry.name.startsWith("."))
    .sort((a, b) => Number(a.isFile()) - Number(b.isFile()) || a.name.localeCompare(b.name, "zh-CN"))
    .map((entry) => {
      const entryPath = path.relative(personaRoot, path.join(target, entry.name)).replace(/\\/g, "/");
      return {
        name: entry.name,
        path: entryPath,
        type: entry.isDirectory() ? "directory" : "file",
      };
    });

  return { persona: safePersona, folder: safeFolder, entries };
}

function resolveWorkspaceFile(pathname: string) {
  const prefix = "/workspace-files/";
  if (!pathname.startsWith(prefix)) return null;

  const parts = pathname
    .slice(prefix.length)
    .split("/")
    .filter(Boolean)
    .map((part) => decodeURIComponent(part));
  const persona = safeName(parts.shift() || "");
  if (!persona || !parts.length) return null;

  const personaRoot = path.resolve(workspaceRoot, persona);
  const filePath = path.resolve(personaRoot, safeWorkspaceFolder(parts.join("/")));
  if (!isInside(personaRoot, filePath) || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
    return null;
  }
  return filePath;
}

function workspacePersonaFromFile(filePath: string) {
  if (!isInside(workspaceRoot, filePath)) return "";
  const relativePath = path.relative(workspaceRoot, filePath).replace(/\\/g, "/");
  return relativePath.split("/").filter(Boolean)[0] || "";
}

function readWorkspaceCommands(persona: string, sinceMs: number) {
  const safePersona = safeName(persona);
  if (!safePersona) return [];

  const commandFile = path.resolve(workspaceRoot, safePersona, ".control", "commands.jsonl");
  if (!isInside(path.resolve(workspaceRoot, safePersona), commandFile) || !fs.existsSync(commandFile) || !fs.statSync(commandFile).isFile()) {
    return [];
  }

  const minCreatedMs = Number.isFinite(sinceMs) ? sinceMs : 0;
  return fs
    .readFileSync(commandFile, "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .slice(-200)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter((command) => command && Number(command.created_ms || 0) > minCreatedMs);
}

function workspaceStatePath(persona: string) {
  const safePersona = safeName(persona);
  if (!safePersona) return null;
  const personaRoot = path.resolve(workspaceRoot, safePersona);
  const controlDir = path.resolve(personaRoot, ".control");
  if (!isInside(personaRoot, controlDir)) return null;
  fs.mkdirSync(controlDir, { recursive: true });
  return path.resolve(controlDir, "state.json");
}

function writeWorkspaceState(persona: string, state: unknown) {
  const target = workspaceStatePath(persona);
  if (!target) return false;
  fs.writeFileSync(
    target,
    JSON.stringify(
      {
        updated_ms: Date.now(),
        state,
      },
      null,
      2,
    ),
    "utf8",
  );
  return true;
}

function readRequestBody(request: import("http").IncomingMessage) {
  return new Promise<string>((resolveBody, rejectBody) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1024 * 1024) {
        rejectBody(new Error("Request body is too large."));
        request.destroy();
      }
    });
    request.on("end", () => resolveBody(body));
    request.on("error", rejectBody);
  });
}

function workspaceControlScript(persona: string) {
  return `<script>
(() => {
  const persona = ${JSON.stringify(persona)};
  let since = Date.now();
  const seen = new Set();
  let lastStateJson = "";
  const codeByKey = {
    " ": "Space",
    Space: "Space",
    Enter: "Enter",
    ArrowLeft: "ArrowLeft",
    ArrowRight: "ArrowRight",
    ArrowUp: "ArrowUp",
    ArrowDown: "ArrowDown",
    Escape: "Escape"
  };

  function dispatchKey(type, command) {
    const key = command.key === "Space" ? " " : command.key;
    const code = command.code || codeByKey[command.key] || (/^[a-z]$/i.test(command.key) ? "Key" + command.key.toUpperCase() : command.key);
    const event = new KeyboardEvent(type, {
      key,
      code,
      bubbles: true,
      cancelable: true
    });
    const target = document.activeElement && document.activeElement !== document.body ? document.activeElement : document;
    target.dispatchEvent(event);
    window.dispatchEvent(event);
  }

  function runCommand(command) {
    const repeat = Math.max(1, Math.min(Number(command.repeat || 1), 20));
    const duration = Math.max(20, Math.min(Number(command.duration_ms || 80), 2000));
    for (let index = 0; index < repeat; index += 1) {
      window.setTimeout(() => {
        dispatchKey("keydown", command);
        window.setTimeout(() => dispatchKey("keyup", command), duration);
      }, index * (duration + 35));
    }
  }

  function runAction(command) {
    const detail = {
      action: command.action,
      payload: command.payload || {},
      id: command.id
    };
    if (typeof window.MeloMateGameAction === "function") {
      window.MeloMateGameAction(detail.action, detail.payload, detail);
    }
    window.dispatchEvent(new CustomEvent("melomate-action", { detail }));
    document.dispatchEvent(new CustomEvent("melomate-action", { detail }));
  }

  function currentState() {
    if (typeof window.MeloMateGameState === "function") {
      return window.MeloMateGameState();
    }
    if (window.MeloMateGameState && typeof window.MeloMateGameState === "object") {
      return window.MeloMateGameState;
    }
    return null;
  }

  async function publishState(nextState) {
    if (nextState == null) return;
    const stateJson = JSON.stringify(nextState);
    if (stateJson === lastStateJson) return;
    lastStateJson = stateJson;
    await fetch("/api/workspace-state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ persona, state: nextState })
    });
  }

  async function poll() {
    try {
      const params = new URLSearchParams({ persona, since: String(since) });
      const response = await fetch("/api/workspace-control?" + params.toString(), { cache: "no-store" });
      if (!response.ok) return;
      const payload = await response.json();
      for (const command of payload.commands || []) {
        if (!command || seen.has(command.id)) continue;
        seen.add(command.id);
        since = Math.max(since, Number(command.created_ms || since));
        if (command.type === "key") runCommand(command);
        if (command.type === "action") runAction(command);
      }
      await publishState(currentState());
    } catch {
      // Workspace control is optional; games still run normally without it.
    }
  }

  window.MeloMateWorkspaceControl = {
    runCommand,
    runAction,
    setState: publishState,
    updateState: publishState
  };
  window.setInterval(poll, 180);
})();
</script>`;
}

function workspaceHtml(filePath: string) {
  const html = fs.readFileSync(filePath, "utf8");
  const script = workspaceControlScript(workspacePersonaFromFile(filePath));
  return /<\/body\s*>/i.test(html) ? html.replace(/<\/body\s*>/i, `${script}</body>`) : `${html}\n${script}`;
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
      server.middlewares.use(async (request, response, next) => {
        const url = new URL(request.url || "/", "http://localhost");

        if (url.pathname === "/api/workspace-state" && request.method === "POST") {
          response.setHeader("Content-Type", "application/json; charset=utf-8");
          try {
            const body = await readRequestBody(request);
            const payload = JSON.parse(body || "{}");
            response.end(JSON.stringify({ ok: writeWorkspaceState(payload.persona || "", payload.state ?? null) }));
          } catch (error) {
            response.statusCode = 400;
            response.end(JSON.stringify({ ok: false, message: error instanceof Error ? error.message : "Invalid state payload." }));
          }
          return;
        }

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

        if (url.pathname === "/api/workspace") {
          response.setHeader("Content-Type", "application/json; charset=utf-8");
          response.end(JSON.stringify(listWorkspace(url.searchParams.get("persona") || "", url.searchParams.get("folder") || "")));
          return;
        }

        if (url.pathname === "/api/workspace-control") {
          response.setHeader("Content-Type", "application/json; charset=utf-8");
          response.end(
            JSON.stringify({
              ok: true,
              commands: readWorkspaceCommands(url.searchParams.get("persona") || "", Number(url.searchParams.get("since") || 0)),
            }),
          );
          return;
        }

        const workspaceFile = resolveWorkspaceFile(url.pathname);
        if (workspaceFile) {
          response.setHeader("Content-Type", responseTypes[path.extname(workspaceFile).toLowerCase()] || "application/octet-stream");
          response.setHeader("Cache-Control", "no-store");
          if (path.extname(workspaceFile).toLowerCase() === ".html") {
            response.end(workspaceHtml(workspaceFile));
            return;
          }
          fs.createReadStream(workspaceFile).pipe(response);
          return;
        }

        for (const [prefix, basePath] of Object.entries(contentRoots)) {
          if (url.pathname === prefix || url.pathname.startsWith(`${prefix}/`)) {
            const filePath = path.resolve(basePath, decodeURIComponent(url.pathname.slice(prefix.length)));
            if (!isInside(basePath, filePath) || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
              next();
              return;
            }
            response.setHeader("Content-Type", responseTypes[path.extname(filePath).toLowerCase()] || "application/octet-stream");
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
