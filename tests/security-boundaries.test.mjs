import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash, createHmac } from "node:crypto";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

test("settings remain actionable while the backend reconnects", () => {
  const mainSource = readFileSync(resolve(projectRoot, "src", "main.ts"), "utf8");

  assert.match(mainSource, /const shouldDisable = isSettingsReadOnly \|\| isApplyingSettings;/);
  assert.match(mainSource, /async function waitForWebSocketReady\(/);
  assert.match(mainSource, /后端未连接，点击重试/);
});

test("workspace bridge dispatches each control exactly once", () => {
  const serverSource = readFileSync(resolve(projectRoot, "server.mjs"), "utf8");
  const bridgeStart = serverSource.indexOf("function workspaceControlScript(");
  const bridgeEnd = serverSource.indexOf("function sendWorkspaceHtml(", bridgeStart);
  const bridgeSource = serverSource.slice(bridgeStart, bridgeEnd);

  assert.match(
    bridgeSource,
    /typeof window\.MeloMateWorkspaceAction === "function"[\s\S]*?await actionHandler\([\s\S]*?new CustomEvent\("melomate-workspace-action"/,
  );
  assert.match(bridgeSource, /if \(!detail\.handled\) {[\s\S]*?new CustomEvent\("melomate-action"/);
  assert.equal(
    (bridgeSource.match(/new CustomEvent\("melomate-workspace-action"/g) || []).length,
    1,
  );
  assert.equal(
    (bridgeSource.match(/new CustomEvent\("melomate-action"/g) || []).length,
    1,
  );
  assert.match(bridgeSource, /window\.MeloMateWorkspaceState/);
  assert.doesNotMatch(bridgeSource, /window\.dispatchEvent\(windowEvent\)/);
  assert.doesNotMatch(bridgeSource, /window\.dispatchEvent\(event\)/);
  assert.doesNotMatch(bridgeSource, /runCommand|command\.type === "key"|KeyboardEvent/);
  assert.match(
    bridgeSource,
    /nextResult\.confirmed === false \|\| nextResult\.ok === false/,
  );
});

test("Vite is build-only and cannot expose a second workspace runtime", () => {
  const viteSource = readFileSync(resolve(projectRoot, "vite.config.ts"), "utf8");
  const packageData = JSON.parse(readFileSync(resolve(projectRoot, "package.json"), "utf8"));

  assert.doesNotMatch(viteSource, /configureServer|workspaceControlScript|createServer/);
  assert.equal(packageData.scripts.dev, undefined);
  assert.equal(packageData.scripts.preview, undefined);
});

async function freePort() {
  const probe = createServer();
  await new Promise((resolveListen, rejectListen) => {
    probe.once("error", rejectListen);
    probe.listen(0, "127.0.0.1", resolveListen);
  });
  const address = probe.address();
  const port = typeof address === "object" && address ? address.port : 0;
  await new Promise((resolveClose) => probe.close(resolveClose));
  return port;
}

async function status(url, headers = {}) {
  return (await fetch(url, { headers })).status;
}

test("local services enforce origin, host and per-launch authentication boundaries", async (context) => {
  const frontendPort = await freePort();
  let workspacePort = await freePort();
  while (workspacePort === frontendPort) workspacePort = await freePort();
  const launchToken = "security-test-launch-token";
  const sessionToken = "security-test-session-token";
  const frontendOrigin = `http://127.0.0.1:${frontendPort}`;
  const workspaceOrigin = `http://127.0.0.1:${workspacePort}`;
  const fixtureRoot = resolve(projectRoot, "workspace", "security-test-persona");
  mkdirSync(fixtureRoot, { recursive: true });
  mkdirSync(resolve(fixtureRoot, ".trash", "private-item"), { recursive: true });
  writeFileSync(
    resolve(fixtureRoot, "index.html"),
    "<!doctype html><title>Security fixture</title>",
    "utf8",
  );
  writeFileSync(
    resolve(fixtureRoot, ".trash", "private-item", "payload"),
    "deleted private content",
    "utf8",
  );
  context.after(() => rmSync(fixtureRoot, { recursive: true, force: true }));
  const fixtureWorkspaceToken = createHmac("sha256", sessionToken)
    .update("melomate-workspace:security-test-persona", "utf8")
    .digest("base64url");

  const child = spawn(process.execPath, ["server.mjs"], {
    cwd: projectRoot,
    env: {
      ...process.env,
      PORT: String(frontendPort),
      MELOMATE_WORKSPACE_PORT: String(workspacePort),
      MELOMATE_BACKEND_WS_URL: "ws://127.0.0.1:65534/client-ws",
      MELOMATE_LAUNCH_TOKEN: launchToken,
      MELOMATE_SESSION_TOKEN: sessionToken,
    },
    stdio: "ignore",
    windowsHide: true,
  });
  context.after(() => {
    if (!child.killed) child.kill();
  });

  let ready = false;
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      if (
        (await status(`${frontendOrigin}/api/health`, {
          "X-MeloMate-Launch": launchToken,
        })) === 200
      ) {
        ready = true;
        break;
      }
    } catch {
      // The child is still starting.
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 50));
  }
  assert.equal(ready, true, "test frontend did not start");

  const mainDocumentResponse = await fetch(`${frontendOrigin}/`);
  assert.equal(mainDocumentResponse.status, 200);
  assert.match(mainDocumentResponse.headers.get("content-security-policy") || "", /object-src 'none'/);
  assert.equal(mainDocumentResponse.headers.get("x-content-type-options"), "nosniff");
  assert.equal(mainDocumentResponse.headers.get("x-frame-options"), "DENY");
  assert.equal(mainDocumentResponse.headers.get("x-permitted-cross-domain-policies"), "none");
  assert.match(mainDocumentResponse.headers.get("permissions-policy") || "", /microphone=\(self\)/);

  const missingAssetResponse = await fetch(`${frontendOrigin}/assets/definitely-missing.js`);
  assert.equal(missingAssetResponse.status, 404);
  assert.equal(missingAssetResponse.headers.get("x-content-type-options"), "nosniff");
  assert.match(missingAssetResponse.headers.get("content-security-policy") || "", /default-src 'self'/);
  assert.doesNotMatch(await missingAssetResponse.text(), /<html/i);
  assert.equal(await status(`${frontendOrigin}/not-a-real-client-route`), 404);
  assert.equal(await status(`${frontendOrigin}/backgrounds/definitely-missing.png`), 404);
  assert.equal(await status(`${frontendOrigin}/%E0%A4%A`), 404);
  assert.equal(
    (await fetch(`${frontendOrigin}/index.html`, { method: "POST" })).status,
    405,
  );
  const headResponse = await fetch(`${frontendOrigin}/index.html`, { method: "HEAD" });
  assert.equal(headResponse.status, 200);
  assert.equal(await headResponse.text(), "");

  assert.equal(await status(`${frontendOrigin}/api/health`), 403);
  assert.equal(await status(`${frontendOrigin}/api/backgrounds`, { Origin: frontendOrigin }), 403);
  assert.equal(
    await status(`${frontendOrigin}/api/backgrounds`, {
      Origin: "https://attacker.invalid",
      "X-MeloMate-Session": sessionToken,
    }),
    403,
  );
  assert.equal(
    await status(`${frontendOrigin}/api/backgrounds`, {
      Origin: frontendOrigin,
      "X-MeloMate-Session": sessionToken,
    }),
    200,
  );
  const vrmManifestResponse = await fetch(`${frontendOrigin}/api/vrm-models`, {
    headers: {
      Origin: frontendOrigin,
      "X-MeloMate-Session": sessionToken,
    },
  });
  assert.equal(vrmManifestResponse.status, 200);
  const vrmManifest = await vrmManifestResponse.json();
  assert.ok(vrmManifest.models.some((model) => model.fileName === "melomate_test_avatar.vrm"));
  const vrmHeadResponse = await fetch(`${frontendOrigin}/vrm-models/melomate_test_avatar.vrm`, {
    method: "HEAD",
  });
  assert.equal(vrmHeadResponse.status, 200);
  assert.equal(vrmHeadResponse.headers.get("content-type"), "model/gltf-binary");
  assert.equal(await status(`${frontendOrigin}/vrm-models/BACKEND_MODELS.md`), 404);
  assert.equal(await status(`${frontendOrigin}/vrm-models/backend/private-model.bin`), 404);
  assert.equal(
    await status(`${frontendOrigin}/api/workspace-events?persona=security-test-persona&since=0&wait_ms=100`, {
      Origin: frontendOrigin,
    }),
    403,
  );
  assert.equal(
    await status(`${frontendOrigin}/api/workspace-events?persona=security-test-persona&since=0&wait_ms=100`, {
      Origin: frontendOrigin,
      "X-MeloMate-Session": sessionToken,
    }),
    200,
  );
  assert.equal(
    await status(`${frontendOrigin}/runtime-config.js`, { "Sec-Fetch-Site": "cross-site" }),
    403,
  );
  assert.equal(
    await status(`${frontendOrigin}/runtime-config.js`, { "Sec-Fetch-Site": "same-origin" }),
    200,
  );
  assert.equal(await status(`${frontendOrigin}/workspace-files/demo/index.html`), 404);
  const openedResponse = await fetch(
    `${frontendOrigin}/api/workspace-open-url?persona=security-test-persona&path=index.html`,
    {
      headers: {
        Origin: frontendOrigin,
        "X-MeloMate-Session": sessionToken,
      },
    },
  );
  assert.equal(openedResponse.status, 200);
  const openedPayload = await openedResponse.json();
  assert.match(openedPayload.url, /\/workspace-files\/security-test-persona\/[A-Za-z0-9_-]+\/index\.html$/);
  const workspaceHtmlResponse = await fetch(openedPayload.url);
  assert.equal(workspaceHtmlResponse.status, 200);
  assert.match(workspaceHtmlResponse.headers.get("content-security-policy") || "", /^sandbox /);
  assert.match(workspaceHtmlResponse.headers.get("content-security-policy") || "", /connect-src 'self'/);
  assert.match(workspaceHtmlResponse.headers.get("content-security-policy") || "", /object-src 'none'/);
  assert.doesNotMatch(workspaceHtmlResponse.headers.get("content-security-policy") || "", /allow-top-navigation|allow-popups/);
  assert.match(workspaceHtmlResponse.headers.get("permissions-policy") || "", /microphone=\(\)/);
  assert.equal(workspaceHtmlResponse.headers.get("x-content-type-options"), "nosniff");
  assert.match(await workspaceHtmlResponse.text(), /X-MeloMate-Workspace-Access/);
  const missingWorkspaceResponse = await fetch(`${openedPayload.url}.missing`);
  assert.equal(missingWorkspaceResponse.status, 404);
  assert.match(missingWorkspaceResponse.headers.get("content-security-policy") || "", /^sandbox /);
  assert.equal(
    await status(`${workspaceOrigin}/workspace-files/security-test-persona/index.html`),
    404,
  );
  assert.equal(
    await status(`${frontendOrigin}/api/workspace-open-url?persona=security-test-persona&path=.control/state.json`, {
      Origin: frontendOrigin,
      "X-MeloMate-Session": sessionToken,
    }),
    404,
  );
  assert.equal(
    await status(`${frontendOrigin}/api/workspace-open-url?persona=security-test-persona&path=.trash/private-item/payload`, {
      Origin: frontendOrigin,
      "X-MeloMate-Session": sessionToken,
    }),
    404,
  );
  assert.equal(
    await status(`${workspaceOrigin}/workspace-files/security-test-persona/${fixtureWorkspaceToken}/.trash/private-item/payload`),
    404,
  );
  assert.equal(
    (await fetch(`${frontendOrigin}/api/workspace-state`, {
      method: "POST",
      headers: {
        Origin: frontendOrigin,
        "Content-Type": "application/json",
        "X-MeloMate-Session": sessionToken,
      },
      body: JSON.stringify({ persona: "security-test-persona", state: {} }),
    })).status,
    405,
  );
  assert.equal(
    await status(`${frontendOrigin}/api/workspace-control?persona=security-test-persona`, {
      Origin: frontendOrigin,
      "X-MeloMate-Session": sessionToken,
    }),
    404,
  );
  const reportedState = {
    protocolAvailable: true,
    state_version: 1,
    page: { id: "test-page", path: "index.html" },
    appState: {
      agentShouldAct: false,
      availableActions: [],
      oversized: "x".repeat(10_000),
      ["__proto__"]: { polluted: true },
    },
  };
  const matchingStateResponse = await fetch(`${workspaceOrigin}/api/workspace-state`, {
    method: "POST",
    headers: {
      Origin: workspaceOrigin,
      "Content-Type": "application/json",
      "X-MeloMate-Workspace-Access": fixtureWorkspaceToken,
    },
    body: JSON.stringify({ persona: "security-test-persona", state: reportedState }),
  });
  assert.equal(matchingStateResponse.status, 200);
  const updatedState = {
    ...reportedState,
    state_version: 2,
    appState: {
      agentShouldAct: true,
      availableActions: [
        { id: "move-7-8", action: "place-piece", payload: { row: 7, col: 8 } },
      ],
      oversized: "x".repeat(10_000),
      ["__proto__"]: { polluted: true },
    },
  };
  const updatedStateResponse = await fetch(`${workspaceOrigin}/api/workspace-state`, {
    method: "POST",
    headers: {
      Origin: workspaceOrigin,
      "Content-Type": "application/json",
      "X-MeloMate-Workspace-Access": fixtureWorkspaceToken,
    },
    body: JSON.stringify({ persona: "security-test-persona", state: updatedState }),
  });
  assert.equal(updatedStateResponse.status, 200);
  const persistedState = JSON.parse(
    readFileSync(resolve(fixtureRoot, ".control", "state.json"), "utf8"),
  );
  assert.equal(persistedState.state.state_version, 2);
  assert.equal(persistedState.state.appState.oversized.length, 2000);
  assert.equal(Object.hasOwn(persistedState.state.appState, "__proto__"), false);
  const firstPageKey = createHash("sha256").update("test-page", "utf8").digest("hex");
  assert.equal(
    JSON.parse(readFileSync(resolve(fixtureRoot, ".control", "pages", `${firstPageKey}.json`), "utf8")).state.page.id,
    "test-page",
  );

  const secondPageState = {
    protocolAvailable: true,
    state_version: 1,
    page: { id: "second-page", path: "index.html" },
    appState: { agentShouldAct: false, marker: "second", availableActions: [] },
  };
  const secondPageResponse = await fetch(`${workspaceOrigin}/api/workspace-state`, {
    method: "POST",
    headers: {
      Origin: workspaceOrigin,
      "Content-Type": "application/json",
      "X-MeloMate-Workspace-Access": fixtureWorkspaceToken,
    },
    body: JSON.stringify({ persona: "security-test-persona", state: secondPageState }),
  });
  assert.equal(secondPageResponse.status, 200);
  const firstPageRead = await fetch(
    `${frontendOrigin}/api/workspace-state?persona=security-test-persona&page_id=test-page`,
    {
      headers: {
        Origin: frontendOrigin,
        "X-MeloMate-Session": sessionToken,
      },
    },
  );
  assert.equal((await firstPageRead.json()).state.state.page.id, "test-page");
  const secondPageRead = await fetch(
    `${frontendOrigin}/api/workspace-state?persona=security-test-persona&page_id=second-page`,
    {
      headers: {
        Origin: frontendOrigin,
        "X-MeloMate-Session": sessionToken,
      },
    },
  );
  assert.equal((await secondPageRead.json()).state.state.appState.marker, "second");
  const eventLines = readFileSync(
    resolve(fixtureRoot, ".control", "events.jsonl"),
    "utf8",
  )
    .trim()
    .split(/\r?\n/)
    .map((line) => JSON.parse(line));
  assert.equal(
    eventLines.findLast((event) => event.page?.id === "test-page")?.state_version,
    2,
  );
  const crossPersonaStateResponse = await fetch(`${workspaceOrigin}/api/workspace-state`, {
    method: "POST",
    headers: {
      Origin: workspaceOrigin,
      "Content-Type": "application/json",
      "X-MeloMate-Workspace-Access": fixtureWorkspaceToken,
    },
    body: JSON.stringify({ persona: "other", state: reportedState }),
  });
  assert.equal(crossPersonaStateResponse.status, 403);
  assert.equal(
    await status(`${workspaceOrigin}/api/workspace-control?persona=security-test-persona&since=0`, {
      Origin: "https://attacker.invalid",
    }),
    403,
  );
  assert.equal(
    await status(`${workspaceOrigin}/api/workspace-control?persona=security-test-persona&since=0`, {
      Origin: workspaceOrigin,
    }),
    403,
  );
  assert.equal(
    await status(`${workspaceOrigin}/api/workspace-control?persona=security-test-persona&since=0&page_id=test-page`, {
      Origin: workspaceOrigin,
      "X-MeloMate-Workspace-Access": fixtureWorkspaceToken,
    }),
    200,
  );
  assert.equal(
    await status(`${workspaceOrigin}/api/workspace-control?persona=other&since=0&page_id=test-page`, {
      Origin: workspaceOrigin,
      "X-MeloMate-Workspace-Access": fixtureWorkspaceToken,
    }),
    403,
  );
});
