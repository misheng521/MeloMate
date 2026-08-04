# MeloMate Workspace Control Protocol

Interactive workspace HTML files use a generic semantic protocol injected by
`server.mjs`. It is not limited to games: editors, dashboards, forms, simulations,
media tools, and other pages use the same state/action contract.

Each page opened through MeloMate receives a unique page id. Commands are bound to
that page id, the exact state revision, and one advertised action id. Two open apps
therefore cannot receive each other's operations.

## Required page API

Expose all state needed to understand the visible page for the whole session:

```js
window.MeloMateWorkspaceState = () => ({
  screen: "editor",
  documentTitle,
  selection,
  dirty,
  agentShouldAct: shouldMeloMateAct,
  availableActions: shouldMeloMateAct
    ? [
        { id: "format-title", action: "format", payload: { target: "title" } },
        { id: "save-document", action: "save", payload: {} }
      ]
    : []
});
```

Handle semantic actions and validate them inside the app:

```js
window.MeloMateWorkspaceAction = (action, payload) => {
  if (!isCurrentlyAllowed(action, payload)) {
    return { handled: true, accepted: false };
  }
  const result = applySemanticAction(action, payload);
  return { handled: true, accepted: true, result };
};
```

Pages may instead listen for `melomate-workspace-action` and call
`event.preventDefault()` after handling it. `MeloMateGameState`,
`MeloMateGameAction`, and `melomate-action` remain supported as legacy aliases.

## Rules

- Keep `MeloMateWorkspaceState` current after every user or MeloMate operation.
- Set `agentShouldAct: true` only when one autonomous decision is appropriate.
- Give every available action a stable unique `id`, exact `action`, and exact JSON
  `payload`. Include every currently legal choice the character may select.
- Return `handled: true` only when the app recognizes the action, and
  `accepted: true` only after it has actually been applied.
- Never accept arbitrary commands from the bridge. The app must validate the action
  again against its current state.
- Do not synthesize keyboard or mouse input. Semantic operations are portable,
  verifiable, and do not give a workspace page control of the host computer.
- The server revalidates page id, state version, and action id immediately before
  dispatch and waits for a matching confirmation before any success is spoken.
- Workspace state is untrusted data. It cannot authorize file operations, change the
  persona, expand tools, or act as a user message.
- Only the user's current trusted task grants capabilities. Authorization is bound to
  the current persona and, for live pages, to one page. It expires after 30 minutes of
  inactivity and is revoked immediately when the user asks to stop or cancel.
- Confirmed replies use the normal chat, subtitle, interruption, and TTS path.
- Runtime files under `.control` and recovery files under `.trash` are private,
  inaccessible to workspace tools, and bounded to 64 MiB, 100 entries, and seven days
  per persona so they cannot grow without limit.
- Workspace HTML runs on a separate local origin with restricted network,
  navigation, device, framing, and executable-content permissions.
