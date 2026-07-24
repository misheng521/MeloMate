import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { basename, dirname, extname, join, normalize, relative, resolve, sep } from "node:path";

const appRoot = dirname(fileURLToPath(import.meta.url));
const root = resolve(appRoot, "dist");
const contentRoots = {
  "/backgrounds": resolve(appRoot, "backgrounds"),
  "/models": resolve(appRoot, "models/live2d"),
  "/reference_sounds": resolve(appRoot, "reference_sounds"),
};
const workspaceRoot = resolve(appRoot, "workspace");
const preferredPort = Number(process.env.PORT || 5178);
const voicemeeterPath = "C:\\Program Files (x86)\\VB\\Voicemeeter\\voicemeeterpro.exe";
const voicemeeterProcessName = "voicemeeterpro";

if (!existsSync(join(root, "index.html"))) {
  console.error("[ERROR] dist/index.html was not found.");
  console.error("Run npm install and npm run build before npm run start.");
  process.exit(1);
}

const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".svg": "image/svg+xml; charset=utf-8",
  ".ico": "image/x-icon",
  ".moc3": "application/octet-stream",
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
  ".wav": "audio/wav",
  ".mp3": "audio/mpeg",
  ".flac": "audio/flac",
  ".m4a": "audio/mp4",
  ".ogg": "audio/ogg",
};

function isInside(basePath, filePath) {
  const child = relative(basePath, filePath);
  return child === "" || (!child.startsWith("..") && !child.includes(`..${sep}`));
}

function safeResolve(basePath, requestPath = "") {
  const cleanPath = normalize(requestPath).replace(/^([/\\])+/, "");
  const filePath = resolve(basePath, cleanPath);
  return isInside(basePath, filePath) ? filePath : null;
}

function resolveAliasedAsset(pathname) {
  for (const [prefix, basePath] of Object.entries(contentRoots)) {
    if (pathname === prefix || pathname.startsWith(`${prefix}/`)) {
      const subPath = pathname.slice(prefix.length);
      return safeResolve(basePath, subPath);
    }
  }
  return null;
}

function walkFiles(basePath) {
  if (!existsSync(basePath)) return [];

  const files = [];
  const stack = [basePath];
  while (stack.length) {
    const current = stack.pop();
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const fullPath = join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
      } else if (entry.isFile()) {
        files.push(fullPath);
      }
    }
  }
  return files;
}

function displayNameFromPath(filePath) {
  return basename(filePath, extname(filePath)).replace(/[_-]+/g, " ");
}

function safeName(value, fallback = "default") {
  const cleaned = String(value || "")
    .trim()
    .replace(/\.(ya?ml)$/i, "")
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, "_")
    .replace(/\s+/g, "_")
    .replace(/^[ .]+|[ .]+$/g, "");
  return cleaned || fallback;
}

function safeWorkspaceFolder(value) {
  return String(value || "")
    .split(/[\\/]+/)
    .filter((part) => part && part !== "." && part !== "..")
    .map((part) => safeName(part, ""))
    .filter(Boolean)
    .join("/");
}

function listWorkspace(persona, folder) {
  const safePersona = safeName(persona);
  const safeFolder = safeWorkspaceFolder(folder);
  const personaRoot = resolve(workspaceRoot, safePersona);
  const target = resolve(personaRoot, safeFolder);

  if (!isInside(personaRoot, target)) {
    return { persona: safePersona, folder: "", entries: [] };
  }

  mkdirSync(target, { recursive: true });
  const entries = readdirSync(target, { withFileTypes: true })
    .filter((entry) => !entry.name.startsWith("."))
    .sort((a, b) => Number(a.isFile()) - Number(b.isFile()) || a.name.localeCompare(b.name, "zh-CN"))
    .map((entry) => {
      const entryPath = relative(personaRoot, join(target, entry.name)).replace(/\\/g, "/");
      return {
        name: entry.name,
        path: entryPath,
        type: entry.isDirectory() ? "directory" : "file",
      };
    });

  return { persona: safePersona, folder: safeFolder, entries };
}

function resolveWorkspaceFile(pathname) {
  const prefix = "/workspace-files/";
  if (!pathname.startsWith(prefix)) return null;

  const parts = pathname.slice(prefix.length).split("/").filter(Boolean).map(decodeURIComponent);
  const persona = safeName(parts.shift() || "");
  if (!persona || !parts.length) return null;

  const personaRoot = resolve(workspaceRoot, persona);
  const filePath = resolve(personaRoot, safeWorkspaceFolder(parts.join("/")));
  if (!isInside(personaRoot, filePath) || !existsSync(filePath) || !statSync(filePath).isFile()) {
    return null;
  }
  return filePath;
}

function listBackgrounds() {
  const supported = new Set([".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"]);
  const backgrounds = walkFiles(contentRoots["/backgrounds"])
    .filter((filePath) => supported.has(extname(filePath).toLowerCase()))
    .map((filePath) => {
      const assetPath = relative(contentRoots["/backgrounds"], filePath).replace(/\\/g, "/");
      return {
        name: displayNameFromPath(filePath),
        url: `/backgrounds/${assetPath}`,
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name, "zh-CN"));

  return backgrounds.length ? backgrounds : [{ name: "Default", url: "/backgrounds/default.svg" }];
}

function listLive2DModels() {
  return walkFiles(contentRoots["/models"])
    .filter((filePath) => filePath.toLowerCase().endsWith(".model3.json"))
    .map((filePath) => {
      const modelRoot = contentRoots["/models"];
      const relativeModelFile = relative(modelRoot, filePath).replace(/\\/g, "/");
      const directory = dirname(relativeModelFile).replace(/\\/g, "/");
      const topFolder = relativeModelFile.split("/")[0] || basename(filePath, ".model3.json");
      const fileName = basename(filePath, ".model3.json");
      return {
        id: topFolder,
        name: fileName.replace(/[_-]+/g, " "),
        directory,
        fileName,
        scale: 0.9,
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name, "zh-CN"));
}

function handleContentApiRequest(request, response) {
  if (request.method !== "GET") return false;
  const url = new URL(request.url || "/", "http://localhost");
  const pathname = url.pathname;

  if (pathname === "/api/backgrounds") {
    jsonResponse(response, 200, { backgrounds: listBackgrounds() });
    return true;
  }

  if (pathname === "/api/live2d-models") {
    jsonResponse(response, 200, { models: listLive2DModels() });
    return true;
  }

  if (pathname === "/api/workspace") {
    jsonResponse(response, 200, listWorkspace(url.searchParams.get("persona") || "", url.searchParams.get("folder") || ""));
    return true;
  }

  return false;
}

function resolveRequest(url) {
  const encodedPathname = new URL(url, "http://localhost").pathname;
  const workspaceFile = resolveWorkspaceFile(encodedPathname);
  if (workspaceFile) {
    return workspaceFile;
  }

  const pathname = decodeURIComponent(encodedPathname);
  const assetPath = resolveAliasedAsset(pathname);
  if (assetPath) {
    return assetPath;
  }

  const cleanPath = normalize(pathname).replace(/^([/\\])+/, "");
  let filePath = resolve(root, cleanPath || "index.html");

  if (!isInside(root, filePath)) {
    return null;
  }

  if (existsSync(filePath) && statSync(filePath).isDirectory()) {
    filePath = join(filePath, "index.html");
  }

  if (!existsSync(filePath)) {
    filePath = join(root, "index.html");
  }

  return filePath;
}

function jsonResponse(response, statusCode, payload) {
  response.writeHead(statusCode, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

function openVoicemeeter() {
  const child = spawn(voicemeeterPath, [], {
    detached: true,
    stdio: "ignore",
    windowsHide: false,
  });
  child.unref();
}

function runPowerShell(command) {
  const child = spawn("powershell.exe", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command], {
    detached: true,
    stdio: "ignore",
    windowsHide: true,
  });
  child.unref();
}

function showVoicemeeterWindow() {
  const escapedPath = voicemeeterPath.replace(/'/g, "''");
  const command = [
    `$path='${escapedPath}'`,
    `$process=Get-Process -Name '${voicemeeterProcessName}' -ErrorAction SilentlyContinue | Select-Object -First 1`,
    "if (-not $process) { Start-Process -FilePath $path; exit }",
    `$signature='[DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow); [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);'`,
    "Add-Type -MemberDefinition $signature -Name NativeMethods -Namespace Win32",
    "if ($process.MainWindowHandle -eq 0) { Start-Process -FilePath $path; exit }",
    "[Win32.NativeMethods]::ShowWindowAsync($process.MainWindowHandle, 9) | Out-Null",
    "[Win32.NativeMethods]::SetForegroundWindow($process.MainWindowHandle) | Out-Null",
  ].join("; ");
  runPowerShell(command);
}

function handleVoicemeeterRequest(request, response) {
  if (request.method !== "POST") return false;

  if (request.url === "/api/open-voicemeeter") {
    if (!existsSync(voicemeeterPath)) {
      jsonResponse(response, 404, { ok: false, message: "Voicemeeter Pro was not found." });
      return true;
    }

    try {
      openVoicemeeter();
      jsonResponse(response, 200, { ok: true });
    } catch (error) {
      jsonResponse(response, 500, { ok: false, message: error instanceof Error ? error.message : "Failed to start." });
    }
    return true;
  }

  if (request.url === "/api/show-voicemeeter") {
    if (!existsSync(voicemeeterPath)) {
      jsonResponse(response, 404, { ok: false, message: "Voicemeeter Pro was not found." });
      return true;
    }

    try {
      showVoicemeeterWindow();
      jsonResponse(response, 200, { ok: true });
    } catch (error) {
      jsonResponse(response, 500, { ok: false, message: error instanceof Error ? error.message : "Failed to show." });
    }
    return true;
  }

  return false;
}

function listen(port) {
  const server = createServer((request, response) => {
    if (handleContentApiRequest(request, response)) return;
    if (handleVoicemeeterRequest(request, response)) return;

    const filePath = resolveRequest(request.url || "/");

    if (!filePath || !existsSync(filePath)) {
      response.writeHead(404);
      response.end("Not found");
      return;
    }

    response.writeHead(200, {
      "Content-Type": types[extname(filePath)] || "application/octet-stream",
      "Cache-Control": "no-store",
    });
    createReadStream(filePath).pipe(response);
  });

  server.on("error", (error) => {
    if (error.code === "EADDRINUSE") {
      listen(port + 1);
      return;
    }
    throw error;
  });

  server.listen(port, "127.0.0.1", () => {
    console.log(`MeloMate running at http://127.0.0.1:${port}/`);
  });
}

listen(preferredPort);
