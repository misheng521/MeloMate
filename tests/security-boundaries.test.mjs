import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHmac } from "node:crypto";
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
  writeFileSync(
    resolve(fixtureRoot, "index.html"),
    "<!doctype html><title>Security fixture</title>",
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
  const reportedState = {
    protocolAvailable: true,
    state_version: 1,
    page: { id: "test-page", path: "index.html" },
    appState: { availableActions: [] },
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
