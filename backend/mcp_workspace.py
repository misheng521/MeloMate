from mcp.server.fastmcp import FastMCP

import workspace_core


mcp = FastMCP("workspace")


def safe_call(fn, *args, **kwargs) -> str:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return workspace_core.response({"ok": False, "message": str(exc)})


@mcp.tool()
def create_workspace_folder(persona: str, folder: str) -> str:
    """Create a folder under workspace/{persona}. Use the current character_name or conf_name as persona."""
    return safe_call(workspace_core.create_workspace_folder, persona, folder)


@mcp.tool()
def write_workspace_file(persona: str, folder: str, filename: str, content: str) -> str:
    """Write a UTF-8 text file under workspace/{persona}/{folder}. Use for any reusable artifact: notes, diary, SVG, HTML/CSS/JS, JSON, lists, plans, drafts, records, data, and user-requested files."""
    return safe_call(workspace_core.write_workspace_file, persona, folder, filename, content)


@mcp.tool()
def append_workspace_file(persona: str, folder: str, filename: str, content: str, reset: bool = False) -> str:
    """Append a UTF-8 text chunk to a file under workspace/{persona}/{folder}. Use reset=True for the first chunk. Prefer this for long code or long documents so tool arguments stay small and valid."""
    return safe_call(workspace_core.append_workspace_file, persona, folder, filename, content, reset)


@mcp.tool()
def write_workspace_project(persona: str, folder: str, files: list[dict]) -> str:
    """Write a multi-file project under workspace/{persona}/{folder}. files must be a list of objects like {"path":"index.html","content":"..."}. Prefer this for games and mini apps, split into index.html, style.css, and main.js."""
    return safe_call(workspace_core.write_workspace_project, persona, folder, files)


@mcp.tool()
def read_workspace_file(persona: str, path: str) -> str:
    """Read a UTF-8 text file from workspace/{persona}. Never read another persona's workspace."""
    return safe_call(workspace_core.read_workspace_file, persona, path)


@mcp.tool()
def list_workspace(persona: str, folder: str = "") -> str:
    """List files and folders under workspace/{persona}/{folder}."""
    return safe_call(workspace_core.list_workspace, persona, folder)


@mcp.tool()
def schedule_reminder(persona: str, message: str, delay_minutes: float = 0, due_at: str = "") -> str:
    """Create a reminder record under workspace/{persona}/reminders/pending. Use delay_minutes for relative times and due_at for exact times. Time is based on the local device clock."""
    return safe_call(workspace_core.schedule_reminder, persona, message, delay_minutes, due_at)


@mcp.tool()
def send_workspace_key(persona: str, key: str, code: str = "", duration_ms: int = 80, repeat: int = 1) -> str:
    """Send keyboard input to an open workspace HTML game or mini app for this persona. Use keys such as ArrowLeft, ArrowRight, ArrowUp, ArrowDown, Space, Enter, w, a, s, or d. Use repeat for repeated taps and duration_ms for how long each key is held."""
    return safe_call(workspace_core.send_workspace_key, persona, key, code, duration_ms, repeat)


@mcp.tool()
def send_workspace_action(persona: str, action: str, payload: dict | None = None) -> str:
    """Send a semantic action to an open workspace HTML app for this persona. Prefer this for board games and turn-based apps, for example action="place-piece" with payload={"row": 7, "col": 7}, or action="select-cell" with coordinates. The app must handle the melomate-action event or window.MeloMateGameAction."""
    return safe_call(workspace_core.send_workspace_action, persona, action, payload)


@mcp.tool()
def read_workspace_state(persona: str) -> str:
    """Read the latest state reported by an open workspace HTML app for this persona. Use this before playing a turn in board games, puzzles, or interactive apps. State is app-defined JSON such as board, turn, score, gameOver, legalMoves, or selected item."""
    return safe_call(workspace_core.read_workspace_state, persona)


@mcp.tool()
def open_workspace_item(persona: str, path: str) -> str:
    """Open a file or folder from workspace/{persona} with the user's default local app. Use after the user says they want to see, open, view, play, or try a generated workspace item."""
    return safe_call(workspace_core.open_workspace_item, persona, path)


if __name__ == "__main__":
    mcp.run()
