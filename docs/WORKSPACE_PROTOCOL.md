# MeloMate Workspace Control Protocol

Interactive workspace HTML files are controlled through a small page protocol injected by `server.mjs`.

## Required Page API

Expose fresh app state for the whole session:

```js
window.MeloMateGameState = () => ({
  screen: "game",
  currentTurn: "MeloMate",
  board,
  legalMoves,
  winner,
  availableActions: [{ action: "place-piece", payload: { row: 0, col: 0 } }]
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
- Return `handled: true` only when the app recognized the action.
- Return `accepted: true` only after the app actually applied the action.
- Do not use a built-in AI opponent when the user asked to play with the character. The character must act through `read_workspace_state` and `send_workspace_action`.
- Keyboard control can dispatch keys, but it cannot prove the app changed. Use semantic actions for turn-based tools and games.

