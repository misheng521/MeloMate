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
    """Write a multi-file project under workspace/{persona}/{folder}. files must be a list of objects like {"path":"index.html","content":"..."}. Prefer this for games, tools, and mini apps, split into index.html, style.css, and main.js. For anything the user expects the character to operate, expose continuous MeloMateGameState with agentShouldAct and exact availableActions, and handle MeloMateGameAction/melomate-action. The independent workspace Agent then controls it instead of a fake built-in AI."""
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
def replace_workspace_text(
    persona: str,
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> str:
    """Edit an existing UTF-8 workspace file by replacing exact text. Read the file first and provide enough surrounding text for a unique match. Use replace_all only when every exact occurrence should change."""
    return safe_call(
        workspace_core.replace_workspace_text,
        persona,
        path,
        old_text,
        new_text,
        replace_all,
    )


@mcp.tool()
def move_workspace_item(persona: str, source: str, destination: str) -> str:
    """Move or rename one file or folder inside workspace/{persona}. The destination must not already exist."""
    return safe_call(workspace_core.move_workspace_item, persona, source, destination)


@mcp.tool()
def delete_workspace_item(persona: str, path: str, recursive: bool = False) -> str:
    """Delete one item inside workspace/{persona}. Set recursive=true only when the user explicitly authorized deleting a non-empty folder. The persona root and runtime control data cannot be deleted."""
    return safe_call(workspace_core.delete_workspace_item, persona, path, recursive)


@mcp.tool()
def search_workspace(
    persona: str,
    query: str,
    folder: str = "",
    max_results: int = 50,
) -> str:
    """Search UTF-8 workspace files for text and return bounded path, line, and snippet matches."""
    return safe_call(
        workspace_core.search_workspace,
        persona,
        query,
        folder,
        max_results,
    )


@mcp.tool()
def read_workspace_state(persona: str, page_id: str = "") -> str:
    """Read verified state reported by an open workspace HTML app for this persona. page_id may select one exact open page; otherwise the most recently reporting page is returned. This is read-only: the independent workspace Agent owns semantic actions. If available=false, do not invent app state."""
    return safe_call(workspace_core.read_workspace_state, persona, page_id)


@mcp.tool()
def open_workspace_item(persona: str, path: str) -> str:
    """Open a file or folder from workspace/{persona} with the user's default local app. Use after the user says they want to see, open, view, play, or try a generated workspace item."""
    return safe_call(workspace_core.open_workspace_item, persona, path)


if __name__ == "__main__":
    mcp.run()
