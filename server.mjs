import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { createHmac, randomBytes, randomUUID, timingSafeEqual } from "node:crypto";
import { createReadStream, existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, statSync, unlinkSync, writeFileSync } from "node:fs";
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

function configuredPort(value) {
  const port = Number(value);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    console.error("[ERROR] PORT must be an integer from 1 to 65535.");
    process.exit(1);
  }
  return port;
}

function configuredBackendWsUrl(value) {
  try {
    const url = new URL(value || "ws://127.0.0.1:12393/client-ws");
    if (url.protocol !== "ws:" && url.protocol !== "wss:") throw new Error("invalid protocol");
    return url.toString();
  } catch {
    console.error("[ERROR] MELOMATE_BACKEND_WS_URL must be a valid ws:// or wss:// URL.");
    process.exit(1);
  }
}

const preferredPort = configuredPort(process.env.PORT || 5178);
const workspacePort = configuredPort(process.env.MELOMATE_WORKSPACE_PORT || 5179);
const backendWsUrl = configuredBackendWsUrl(process.env.MELOMATE_BACKEND_WS_URL);
const launchToken = String(process.env.MELOMATE_LAUNCH_TOKEN || "standalone");
const sessionToken = String(process.env.MELOMATE_SESSION_TOKEN || randomBytes(32).toString("base64url"));
const voicemeeterPath = "C:\\Program Files (x86)\\VB\\Voicemeeter\\voicemeeterpro.exe";
const voicemeeterProcessName = "voicemeeterpro";
const workspaceEventLimit = 200;
const workspaceEventWaiters = new Map();
const mainOrigins = new Set([
  `http://127.0.0.1:${preferredPort}`,
  `http://localhost:${preferredPort}`,
]);
const workspaceOrigins = new Set([
  `http://127.0.0.1:${workspacePort}`,
  `http://localhost:${workspacePort}`,
]);
const workspaceBaseUrl = `http://127.0.0.1:${workspacePort}`;

if (workspacePort === preferredPort) {
  console.error("[ERROR] MELOMATE_WORKSPACE_PORT must differ from the frontend port.");
  process.exit(1);
}

function constantTimeEqual(actual, expected) {
  const actualBuffer = Buffer.from(String(actual || ""));
  const expectedBuffer = Buffer.from(String(expected || ""));
  return actualBuffer.length === expectedBuffer.length && timingSafeEqual(actualBuffer, expectedBuffer);
}

function allowedHost(request, port) {
  const host = String(request.headers.host || "").toLowerCase();
  return host === `127.0.0.1:${port}` || host === `localhost:${port}`;
}

function requestComesFrom(request, allowedOrigins) {
  const origin = String(request.headers.origin || "");
  if (origin) return allowedOrigins.has(origin);

  const fetchSite = String(request.headers["sec-fetch-site"] || "").toLowerCase();
  if (fetchSite) return fetchSite === "same-origin";

  const referer = String(request.headers.referer || "");
  if (!referer) return false;
  try {
    return allowedOrigins.has(new URL(referer).origin);
  } catch {
    return false;
  }
}

function hasSessionToken(request) {
  return constantTimeEqual(request.headers["x-melomate-session"], sessionToken);
}

function workspaceAccessToken(persona) {
  const safePersona = safeName(persona, "");
  if (!safePersona) return "";
  return createHmac("sha256", sessionToken)
    .update(`melomate-workspace:${safePersona}`, "utf8")
    .digest("base64url");
}

function hasWorkspaceAccess(request, persona) {
  const expected = workspaceAccessToken(persona);
  return Boolean(expected) && constantTimeEqual(
    request.headers["x-melomate-workspace-access"],
    expected,
  );
}

function authorizeMainApi(request) {
  return allowedHost(request, preferredPort) && requestComesFrom(request, mainOrigins) && hasSessionToken(request);
}

function reject(response, statusCode = 403, message = "Forbidden", headers = mainSecurityHeaders) {
  response.writeHead(statusCode, headers({ "Content-Type": "text/plain; charset=utf-8" }));
  response.end(message);
}

function mainSecurityHeaders(extra = {}) {
  return {
    "Cache-Control": "no-store",
    "Content-Security-Policy": [
      "default-src 'self'",
      "base-uri 'none'",
      "object-src 'none'",
      "frame-ancestors 'none'",
      "form-action 'self'",
      "script-src 'self' 'wasm-unsafe-eval'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "font-src 'self' data:",
      "media-src 'self' data: blob:",
      "worker-src 'self' blob:",
      "connect-src 'self' ws://127.0.0.1:* ws://localhost:* wss://127.0.0.1:* wss://localhost:* http://127.0.0.1:* http://localhost:* https://127.0.0.1:* https://localhost:*",
      "frame-src 'none'",
    ].join("; "),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": [
      "camera=()",
      "display-capture=(self)",
      "geolocation=()",
      "microphone=(self)",
      "payment=()",
      "serial=()",
      "usb=()",
    ].join(", "),
    "Referrer-Policy": "no-referrer",
    "X-DNS-Prefetch-Control": "off",
    "X-Download-Options": "noopen",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-Permitted-Cross-Domain-Policies": "none",
    "X-XSS-Protection": "0",
    ...extra,
  };
}

function workspaceSecurityHeaders(extra = {}) {
  return {
    "Cache-Control": "no-store",
    "Content-Security-Policy":
      "sandbox allow-scripts allow-same-origin allow-forms allow-modals allow-popups " +
      "allow-popups-to-escape-sandbox allow-downloads allow-top-navigation-by-user-activation " +
      "allow-pointer-lock allow-presentation; frame-ancestors 'self'",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), display-capture=(), geolocation=(), microphone=(), payment=(), serial=(), usb=()",
    "Referrer-Policy": "no-referrer",
    "X-DNS-Prefetch-Control": "off",
    "X-Download-Options": "noopen",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "X-Permitted-Cross-Domain-Policies": "none",
    "X-XSS-Protection": "0",
    ...extra,
  };
}

function newWorkspacePageId() {
  return `${Date.now()}${String(Math.floor(Math.random() * 1_000_000)).padStart(6, "0")}`;
}

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
  ".txt": "text/plain; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".csv": "text/csv; charset=utf-8",
  ".xml": "application/xml; charset=utf-8",
  ".yaml": "application/yaml; charset=utf-8",
  ".yml": "application/yaml; charset=utf-8",
  ".pdf": "application/pdf",
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
  ".mp4": "video/mp4",
  ".webm": "video/webm",
};

function isInside(basePath, filePath) {
  const child = relative(basePath, filePath);
  return child === "" || (!child.startsWith("..") && !child.includes(`..${sep}`));
}

function existingFileInside(basePath, filePath) {
  try {
    const realBase = realpathSync(basePath);
    const realFile = realpathSync(filePath);
    return isInside(realBase, realFile) && statSync(realFile).isFile() ? realFile : null;
  } catch {
    return null;
  }
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

  let parts;
  try {
    parts = pathname.slice(prefix.length).split("/").filter(Boolean).map(decodeURIComponent);
  } catch {
    return null;
  }
  const persona = safeName(parts.shift() || "");
  const accessToken = parts.shift() || "";
  if (!persona || !constantTimeEqual(accessToken, workspaceAccessToken(persona)) || !parts.length) return null;

  const personaRoot = resolve(workspaceRoot, persona);
  const filePath = resolve(personaRoot, safeWorkspaceFolder(parts.join("/")));
  return isInside(personaRoot, filePath) ? existingFileInside(personaRoot, filePath) : null;
}

function workspaceFileUrl(persona, requestedPath) {
  const safePersona = safeName(persona, "");
  const safePath = safeWorkspaceFolder(requestedPath);
  if (!safePersona || !safePath) return "";
  const personaRoot = resolve(workspaceRoot, safePersona);
  const filePath = resolve(personaRoot, safePath);
  if (!isInside(personaRoot, filePath) || !existingFileInside(personaRoot, filePath)) return "";
  const encodedPath = safePath.split("/").map(encodeURIComponent).join("/");
  return `${workspaceBaseUrl}/workspace-files/${encodeURIComponent(safePersona)}/${workspaceAccessToken(safePersona)}/${encodedPath}`;
}

function workspacePersonaFromFile(filePath) {
  if (!isInside(workspaceRoot, filePath)) return "";
  const relativePath = relative(workspaceRoot, filePath).replace(/\\/g, "/");
  return relativePath.split("/").filter(Boolean)[0] || "";
}

function readWorkspaceCommands(persona, sinceMs) {
  const safePersona = safeName(persona);
  if (!safePersona) return [];

  const commandFile = safeResolve(workspaceRoot, `${safePersona}/.control/commands.jsonl`);
  if (!commandFile || !existsSync(commandFile) || !statSync(commandFile).isFile()) return [];

  const minCreatedMs = Number.isFinite(sinceMs) ? sinceMs : 0;
  return readFileSync(commandFile, "utf8")
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

function workspaceStatePath(persona) {
  const safePersona = safeName(persona);
  if (!safePersona) return null;
  const personaRoot = resolve(workspaceRoot, safePersona);
  const controlDir = safeResolve(workspaceRoot, `${safePersona}/.control`);
  if (!controlDir || !isInside(personaRoot, controlDir)) return null;
  mkdirSync(controlDir, { recursive: true });
  return safeResolve(controlDir, "state.json");
}

function workspaceEventsPath(persona) {
  const safePersona = safeName(persona);
  if (!safePersona) return null;
  const personaRoot = resolve(workspaceRoot, safePersona);
  const controlDir = safeResolve(workspaceRoot, `${safePersona}/.control`);
  if (!controlDir || !isInside(personaRoot, controlDir)) return null;
  mkdirSync(controlDir, { recursive: true });
  return safeResolve(controlDir, "events.jsonl");
}

function stableEventState(state) {
  if (state?.closed) return JSON.stringify({ pageId: state.page?.id || "", closed: true });
  if (!state?.protocolAvailable) return "";
  return JSON.stringify({
    pageId: state.page?.id || "",
    path: state.page?.path || "",
    appState: state.appState,
    lastAction: state.lastAction || null,
  });
}

function appendWorkspaceEvent(persona, previous, nextState) {
  const nextEventState = stableEventState(nextState);
  if (!nextEventState || stableEventState(previous?.state) === nextEventState) return;

  const target = workspaceEventsPath(persona);
  if (!target) return;
  const lines = existsSync(target) ? readFileSync(target, "utf8").split(/\r?\n/).filter(Boolean) : [];
  let previousCreatedMs = 0;
  if (lines.length) {
    try {
      previousCreatedMs = Number(JSON.parse(lines[lines.length - 1]).created_ms || 0);
    } catch {
      previousCreatedMs = 0;
    }
  }
  const previousActionId = previous?.state?.lastAction?.id || "";
  const nextActionId = nextState.lastAction?.id || "";
  const event = {
    id: randomUUID(),
    type: nextState.closed ? "workspace-page-closed" : "workspace-state-changed",
    created_ms: Math.max(Date.now(), previousCreatedMs + 1),
    state_version: Number(nextState.state_version || 0),
    persona: safeName(persona),
    page: nextState.page,
    appState: nextState.appState,
    lastAction: nextState.lastAction || null,
    actionEvent: Boolean(nextActionId && nextActionId !== previousActionId),
    summary: nextState.closed
      ? "Workspace page was closed."
      : previous
        ? "Workspace page state changed."
        : "Workspace page was opened.",
  };
  writeFileSync(target, [...lines.slice(-workspaceEventLimit + 1), JSON.stringify(event)].join("\n") + "\n", "utf8");
  notifyWorkspaceEventWaiters(persona);
}

function notifyWorkspaceEventWaiters(persona) {
  const safePersona = safeName(persona);
  const waiters = workspaceEventWaiters.get(safePersona);
  if (!waiters) return;
  workspaceEventWaiters.delete(safePersona);
  for (const resolveWaiter of waiters) resolveWaiter();
}

async function waitForWorkspaceEvents(persona, sinceMs, waitMs) {
  const safePersona = safeName(persona);
  const existing = readWorkspaceEvents(safePersona, sinceMs);
  if (existing.length || waitMs <= 0) return existing;

  await new Promise((resolveWaiter) => {
    const waiters = workspaceEventWaiters.get(safePersona) || new Set();
    let timeoutId;
    const finish = () => {
      clearTimeout(timeoutId);
      waiters.delete(finish);
      if (!waiters.size) workspaceEventWaiters.delete(safePersona);
      resolveWaiter();
    };
    waiters.add(finish);
    workspaceEventWaiters.set(safePersona, waiters);
    timeoutId = setTimeout(finish, Math.max(100, Math.min(waitMs, 20_000)));
    if (readWorkspaceEvents(safePersona, sinceMs).length) finish();
  });
  return readWorkspaceEvents(safePersona, sinceMs);
}

function writeWorkspaceState(persona, state) {
  const target = workspaceStatePath(persona);
  if (!target) return false;
  const previous = readWorkspaceState(persona);
  appendWorkspaceEvent(persona, previous, state);
  if (state?.closed) {
    if (existsSync(target) && previous?.state?.page?.id === state.page?.id) {
      unlinkSync(target);
    }
    return true;
  }
  writeFileSync(
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

function readWorkspaceState(persona) {
  const target = workspaceStatePath(persona);
  if (!target || !existsSync(target) || !statSync(target).isFile()) return null;
  try {
    return JSON.parse(readFileSync(target, "utf8"));
  } catch {
    return null;
  }
}

function readWorkspaceEvents(persona, sinceMs) {
  const target = workspaceEventsPath(persona);
  if (!target || !existsSync(target) || !statSync(target).isFile()) return [];
  const minCreatedMs = Number.isFinite(sinceMs) ? sinceMs : 0;
  return readFileSync(target, "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .slice(-workspaceEventLimit)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter((event) => event && Number(event.created_ms || 0) > minCreatedMs);
}

function readRequestBody(request) {
  return new Promise((resolveBody, rejectBody) => {
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

function workspaceControlScript(persona, pageId, accessToken, logicalPath) {
  return `<script>
(() => {
  const persona = ${JSON.stringify(persona)};
  const pageId = ${JSON.stringify(pageId)};
  const accessToken = ${JSON.stringify(accessToken)};
  const logicalPath = ${JSON.stringify(logicalPath)};
  const accessHeaders = Object.freeze({ "X-MeloMate-Workspace-Access": accessToken });
  const bridgeFetch = window.fetch.bind(window);
  const openedAtMs = Date.now();
  let since = Date.now();
  const seen = new Set();
  let lastStateSignature = "";
  let stateVersion = 0;
  const actions = [];
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

  function safeJson(value) {
    if (value === undefined) return null;
    try {
      return JSON.parse(JSON.stringify(value));
    } catch {
      return String(value);
    }
  }

  function applyActionResult(detail, nextResult) {
    if (nextResult === undefined) return;
    detail.result = nextResult;
    if (nextResult === false) detail.accepted = false;
    if (nextResult && typeof nextResult === "object") {
      if (nextResult.accepted === false) detail.accepted = false;
      if (nextResult.handled === true) detail.handled = true;
      if (Object.prototype.hasOwnProperty.call(nextResult, "result")) detail.result = nextResult.result;
      if (nextResult.error) detail.error = String(nextResult.error);
    }
  }

  async function runAction(command) {
    const detail = {
      action: command.action,
      payload: command.payload || {},
      id: command.id,
      pageId,
      handled: false,
      accepted: true,
      result: null,
      error: ""
    };
    if (typeof window.MeloMateGameAction === "function") {
      detail.handled = true;
      try {
        applyActionResult(detail, await window.MeloMateGameAction(detail.action, detail.payload, detail));
      } catch (exception) {
        detail.accepted = false;
        detail.error = exception && exception.message ? exception.message : String(exception);
      }
    }
    const windowEvent = new CustomEvent("melomate-action", { detail, bubbles: true, cancelable: true });
    window.dispatchEvent(windowEvent);
    if (windowEvent.defaultPrevented) detail.handled = true;
    const documentEvent = new CustomEvent("melomate-action", { detail, bubbles: true, cancelable: true });
    document.dispatchEvent(documentEvent);
    if (documentEvent.defaultPrevented) detail.handled = true;
    const actionResult = {
      id: detail.id,
      action: detail.action,
      payload: detail.payload,
      handled: detail.handled === true,
      accepted: detail.handled === true && detail.accepted !== false,
      result: safeJson(detail.result),
      error: detail.error,
      at_ms: Date.now()
    };
    actions.push(actionResult);
    while (actions.length > 20) actions.shift();
    await publishState(currentState(), true);
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

  async function publishState(nextState, force = false) {
    const latestAction = actions.length ? actions[actions.length - 1] : null;
    const stateSignature = JSON.stringify({
      protocolAvailable: nextState != null,
      appState: nextState,
      lastAction: latestAction,
      title: document.title,
      path: logicalPath
    });
    const changed = stateSignature !== lastStateSignature;
    if (changed) {
      lastStateSignature = stateSignature;
      stateVersion += 1;
    }
    if (!force && !changed) return;
    const report = {
      protocolAvailable: nextState != null,
      appState: nextState,
      lastAction: latestAction,
      actions,
      state_version: stateVersion,
      page: {
        id: pageId,
        title: document.title,
        path: logicalPath,
        href: logicalPath,
        opened_at_ms: openedAtMs
      },
      reported_ms: Date.now()
    };
    await bridgeFetch("/api/workspace-state", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...accessHeaders },
      body: JSON.stringify({ persona, state: report })
    });
  }

  function publishClosed() {
    const report = {
      protocolAvailable: false,
      appState: null,
      lastAction: actions.length ? actions[actions.length - 1] : null,
      actions,
      state_version: stateVersion + 1,
      page: {
        id: pageId,
        title: document.title,
        path: logicalPath,
        href: logicalPath,
        opened_at_ms: openedAtMs,
        closed: true
      },
      closed: true,
      reported_ms: Date.now()
    };
    void bridgeFetch("/api/workspace-state", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...accessHeaders },
      body: JSON.stringify({ persona, state: report }),
      keepalive: true
    });
  }

  async function poll() {
    try {
      const params = new URLSearchParams({ persona, since: String(since), page_id: pageId });
      const response = await bridgeFetch("/api/workspace-control?" + params.toString(), {
        cache: "no-store",
        headers: accessHeaders
      });
      if (!response.ok) return;
      const payload = await response.json();
      for (const command of payload.commands || []) {
        if (!command || seen.has(command.id)) continue;
        if (command.page_id && command.page_id !== pageId) continue;
        seen.add(command.id);
        since = Math.max(since, Number(command.created_ms || since));
        if (command.type === "key") runCommand(command);
        if (command.type === "action") await runAction(command);
      }
      await publishState(currentState());
    } catch {
      // Workspace control is optional; games still run normally without it.
    }
  }

  window.MeloMateWorkspaceControl = {
    pageId,
    runCommand,
    runAction,
    setState: publishState,
    updateState: publishState
  };
  window.setInterval(poll, 180);
  window.setInterval(() => publishState(currentState(), true), 1000);
  window.addEventListener("pagehide", publishClosed);
  window.addEventListener("beforeunload", publishClosed);
  publishState(currentState(), true);
})();
</script>`;
}

function sendWorkspaceHtml(filePath, response, headOnly = false) {
  if (headOnly) {
    response.writeHead(200, workspaceSecurityHeaders({ "Content-Type": types[".html"] }));
    response.end();
    return;
  }
  const persona = workspacePersonaFromFile(filePath);
  const html = readFileSync(filePath, "utf8");
  const personaRoot = resolve(workspaceRoot, persona);
  const logicalPath = relative(personaRoot, filePath).replace(/\\/g, "/");
  const script = workspaceControlScript(
    persona,
    newWorkspacePageId(),
    workspaceAccessToken(persona),
    logicalPath,
  );
  const headOpen = /<head(?:\s[^>]*)?>/i;
  const content = headOpen.test(html)
    ? html.replace(headOpen, (match) => `${match}${script}`)
    : `${script}\n${html}`;
  response.writeHead(200, workspaceSecurityHeaders({ "Content-Type": types[".html"] }));
  response.end(content);
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

async function handleWorkspaceStateRequest(request, response, requireWorkspaceAccess = false) {
  if (request.method !== "POST" || request.url !== "/api/workspace-state") return false;
  try {
    const body = await readRequestBody(request);
    const payload = JSON.parse(body || "{}");
    if (requireWorkspaceAccess && !hasWorkspaceAccess(request, payload.persona || "")) {
      reject(response);
      return true;
    }
    const ok = writeWorkspaceState(payload.persona || "", payload.state ?? null);
    jsonResponse(response, ok ? 200 : 400, { ok });
  } catch (error) {
    jsonResponse(response, 400, { ok: false, message: error instanceof Error ? error.message : "Invalid state payload." });
  }
  return true;
}

async function handleContentApiRequest(request, response, requireWorkspaceAccess = false) {
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

  if (pathname === "/api/workspace-control") {
    const persona = url.searchParams.get("persona") || "";
    if (requireWorkspaceAccess && !hasWorkspaceAccess(request, persona)) {
      reject(response);
      return true;
    }
    const pageId = String(url.searchParams.get("page_id") || "").slice(0, 128);
    if (requireWorkspaceAccess && !pageId) {
      reject(response, 400, "page_id is required");
      return true;
    }
    const since = Number(url.searchParams.get("since") || 0);
    const commands = readWorkspaceCommands(persona, since);
    jsonResponse(response, 200, {
      ok: true,
      commands: requireWorkspaceAccess
        ? commands.filter((command) => !command.page_id || command.page_id === pageId)
        : commands,
    });
    return true;
  }

  if (pathname === "/api/workspace-open-url") {
    const openedUrl = workspaceFileUrl(
      url.searchParams.get("persona") || "",
      url.searchParams.get("path") || "",
    );
    jsonResponse(response, openedUrl ? 200 : 404, {
      ok: Boolean(openedUrl),
      url: openedUrl,
    });
    return true;
  }

  if (pathname === "/api/workspace-state") {
    const state = readWorkspaceState(url.searchParams.get("persona") || "");
    jsonResponse(response, 200, { ok: true, state });
    return true;
  }

  if (pathname === "/api/workspace-events") {
    const since = Number(url.searchParams.get("since") || 0);
    const waitMs = Number(url.searchParams.get("wait_ms") || 0);
    jsonResponse(response, 200, {
      ok: true,
      events: await waitForWorkspaceEvents(
        url.searchParams.get("persona") || "",
        since,
        Number.isFinite(waitMs) ? waitMs : 0,
      ),
    });
    return true;
  }

  return false;
}

function resolveMainRequest(url) {
  let pathname;
  try {
    const encodedPathname = new URL(url, "http://localhost").pathname;
    pathname = decodeURIComponent(encodedPathname);
  } catch {
    return null;
  }
  const assetPath = resolveAliasedAsset(pathname);
  if (assetPath) {
    const matchingRoot = Object.entries(contentRoots).find(
      ([prefix]) => pathname === prefix || pathname.startsWith(`${prefix}/`),
    )?.[1];
    return matchingRoot ? existingFileInside(matchingRoot, assetPath) : null;
  }

  const cleanPath = normalize(pathname).replace(/^([/\\])+/, "");
  let filePath = resolve(root, cleanPath || "index.html");

  if (!isInside(root, filePath)) {
    return null;
  }

  try {
    if (existsSync(filePath) && statSync(filePath).isDirectory()) {
      filePath = join(filePath, "index.html");
    }
  } catch {
    return null;
  }

  return existingFileInside(root, filePath);
}

function jsonResponse(response, statusCode, payload) {
  response.writeHead(statusCode, mainSecurityHeaders({ "Content-Type": "application/json; charset=utf-8" }));
  response.end(JSON.stringify(payload));
}

function handleRuntimeRequest(request, response) {
  if (request.method !== "GET") return false;
  const pathname = new URL(request.url || "/", "http://localhost").pathname;
  if (pathname === "/api/health") {
    if (!allowedHost(request, preferredPort) || !constantTimeEqual(request.headers["x-melomate-launch"], launchToken)) {
      reject(response);
      return true;
    }
    response.writeHead(200, mainSecurityHeaders({ "Content-Type": "application/json; charset=utf-8" }));
    response.end(
      JSON.stringify({
        ok: true,
        app: "MeloMate",
        service: "frontend",
        port: preferredPort,
      }),
    );
    return true;
  }
  if (pathname === "/runtime-config.js") {
    if (!allowedHost(request, preferredPort) || !requestComesFrom(request, mainOrigins)) {
      reject(response);
      return true;
    }
    response.writeHead(200, mainSecurityHeaders({ "Content-Type": "text/javascript; charset=utf-8" }));
    response.end(
      `window.__MELOMATE_RUNTIME_CONFIG__ = Object.freeze(${JSON.stringify({
        backendWsUrl,
        sessionToken,
        workspaceBaseUrl,
      })});\n`,
    );
    return true;
  }
  return false;
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

function sendFile(filePath, response, headers, headOnly = false) {
  const responseHeaders = headers({
    "Content-Type": types[extname(filePath).toLowerCase()] || "application/octet-stream",
  });
  if (headOnly) {
    response.writeHead(200, responseHeaders);
    response.end();
    return;
  }
  const stream = createReadStream(filePath);
  stream.once("open", () => {
    response.writeHead(200, responseHeaders);
    stream.pipe(response);
  });
  stream.once("error", (error) => {
    if (!response.headersSent) {
      reject(response, error.code === "ENOENT" ? 404 : 500, error.code === "ENOENT" ? "Not found" : "Read failed", headers);
    } else {
      response.destroy(error);
    }
  });
}

function workspaceFileHeaders(filePath) {
  const extension = extname(filePath).toLowerCase();
  const contentType = types[extension] || "application/octet-stream";
  const extra = { "Content-Type": contentType };
  if (contentType === "application/octet-stream") {
    extra["Content-Disposition"] = `attachment; filename*=UTF-8''${encodeURIComponent(basename(filePath))}`;
  }
  return workspaceSecurityHeaders(extra);
}

function listen(mainPort, isolatedWorkspacePort) {
  const server = createServer(async (request, response) => {
    try {
    if (!allowedHost(request, mainPort)) {
      reject(response, 421, "Misdirected request");
      return;
    }
    if (handleRuntimeRequest(request, response)) return;

    const pathname = new URL(request.url || "/", "http://localhost").pathname;
    if (pathname.startsWith("/workspace-files/")) {
      reject(response, 404, "Workspace content is served from an isolated origin.");
      return;
    }
    if (pathname.startsWith("/api/") && !authorizeMainApi(request)) {
      reject(response);
      return;
    }
    if (await handleWorkspaceStateRequest(request, response)) return;
    if (await handleContentApiRequest(request, response)) return;
    if (handleVoicemeeterRequest(request, response)) return;

    if (request.method !== "GET" && request.method !== "HEAD") {
      response.setHeader("Allow", "GET, HEAD");
      reject(response, 405, "Method not allowed");
      return;
    }

    const filePath = resolveMainRequest(request.url || "/");

    if (!filePath || !existsSync(filePath)) {
      reject(response, 404, "Not found");
      return;
    }

    sendFile(filePath, response, mainSecurityHeaders, request.method === "HEAD");
    } catch (error) {
      console.error(`[ERROR] Main request failed: ${error instanceof Error ? error.message : String(error)}`);
      if (!response.headersSent) reject(response, 500, "Internal server error");
      else response.destroy();
    }
  });

  const workspaceServer = createServer(async (request, response) => {
    try {
    if (!allowedHost(request, isolatedWorkspacePort)) {
      reject(response, 421, "Misdirected request", workspaceSecurityHeaders);
      return;
    }

    const pathname = new URL(request.url || "/", "http://localhost").pathname;
    if (pathname.startsWith("/api/")) {
      if (!requestComesFrom(request, workspaceOrigins)) {
        reject(response, 403, "Forbidden", workspaceSecurityHeaders);
        return;
      }
      if (await handleWorkspaceStateRequest(request, response, true)) return;
      if (pathname === "/api/workspace-control" && await handleContentApiRequest(request, response, true)) return;
      reject(response, 404, "Not found", workspaceSecurityHeaders);
      return;
    }

    if (request.method !== "GET" && request.method !== "HEAD") {
      response.setHeader("Allow", "GET, HEAD");
      reject(response, 405, "Method not allowed", workspaceSecurityHeaders);
      return;
    }

    const filePath = resolveWorkspaceFile(pathname);
    if (!filePath) {
      reject(response, 404, "Not found", workspaceSecurityHeaders);
      return;
    }
    if (extname(filePath).toLowerCase() === ".html") {
      sendWorkspaceHtml(filePath, response, request.method === "HEAD");
      return;
    }
    sendFile(
      filePath,
      response,
      () => workspaceFileHeaders(filePath),
      request.method === "HEAD",
    );
    } catch (error) {
      console.error(`[ERROR] Workspace request failed: ${error instanceof Error ? error.message : String(error)}`);
      if (!response.headersSent) reject(response, 500, "Internal server error", workspaceSecurityHeaders);
      else response.destroy();
    }
  });

  server.on("error", (error) => {
    if (error.code === "EADDRINUSE") {
      console.error(
        `[ERROR] Frontend port 127.0.0.1:${mainPort} is already in use. ` +
          "MeloMate did not stop or reuse the owning process.",
      );
      workspaceServer.close();
      process.exitCode = 1;
      return;
    }
    console.error(`[ERROR] Frontend server failed: ${error.message}`);
    workspaceServer.close();
    process.exitCode = 1;
  });

  server.listen(mainPort, "127.0.0.1", () => {
    console.log(`MeloMate running at http://127.0.0.1:${mainPort}/`);
  });

  workspaceServer.on("error", (error) => {
    if (error.code === "EADDRINUSE") {
      console.error(`[ERROR] Workspace port 127.0.0.1:${isolatedWorkspacePort} is already in use.`);
    } else {
      console.error(`[ERROR] Workspace server failed: ${error.message}`);
    }
    server.close();
    process.exitCode = 1;
  });

  workspaceServer.listen(isolatedWorkspacePort, "127.0.0.1", () => {
    console.log(`MeloMate isolated workspace at ${workspaceBaseUrl}/`);
  });
}

listen(preferredPort, workspacePort);
