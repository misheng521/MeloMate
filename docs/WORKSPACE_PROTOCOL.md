# MeloMate Workspace Control Protocol

Interactive workspace HTML files are controlled through a small page protocol injected by `server.mjs`.

Every time an HTML file is opened through MeloMate, the server injects a unique page instance id. Workspace commands are routed to that exact page id, so two open games cannot accidentally receive each other's moves.

## Required Page API

Expose fresh app state for the whole session:

```js
window.MeloMateGameState = () => ({
  screen: "game",
  currentTurn: "MeloMate",
  agentShouldAct: currentTurn === "MeloMate" && !winner,
  board,
  legalMoves,
  winner,
  availableActions: currentTurn === "MeloMate" && !winner
    ? [{ id: "place-0-0", action: "place-piece", payload: { row: 0, col: 0 } }]
    : []
});
```

Handle semantic actions:

```js
window.MeloMateGameAction = (action, payload) => {
  if (action !== "place-piece") return { handled: false, accepted: false };
  if (!isLegal(payload)) return { handled: true, accepted: false };
  applyMove(payload);
  return { handled: true, accepted: true, result: { move: payload } };
};
```

## Rules

- Keep `MeloMateGameState` available after every user action and every MeloMate action.
- Set `agentShouldAct: true` only when the character is expected to make one autonomous decision. At every other time set it to `false` or return no `availableActions`.
- Give every `availableActions` entry a stable, unique `id`. MeloMate chooses an id and the backend revalidates its exact action and payload against the same open page and state revision.
- Return `handled: true` only when the app recognized the action.
- Return `accepted: true` only after the app actually applied the action.
- Do not use a built-in AI opponent when the user asked to play with the character. MeloMate runtime control is the only character-side operator; this internal detail must not appear in character replies.
- Do not synthesize keyboard input. Use semantic actions so the page can validate the operation and report whether it actually happened.
- After a confirmed action, the Agent's natural response is delivered through the same chat subtitle and TTS completion protocol as ordinary conversation. A user message interrupts that speech normally.
- When the page is closed, the injected script reports a close event and clears the current page state. MeloMate should treat the app as disconnected until a new HTML page reports state.
- Each open HTML page has an isolated page id and its own state snapshot, so multiple open apps do not overwrite one another's control state.
- Runtime control files under `.control` are private to the runtime, inaccessible to workspace file tools, and bounded so command/event logs do not grow without limit.
- Workspace HTML runs on a separate local origin with no access to the main application's credentials. Its network, navigation, device, framing, and executable-content permissions are restricted by response headers.
