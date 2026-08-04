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
    """Write a multi-file project under workspace/{persona}/{folder}. files must be a list of objects like {"path":"index.html","content":"..."}. Prefer this for games, tools, and mini apps, split into index.html, style.css, and main.js. For any page the character should operate, expose continuous MeloMateWorkspaceState with exact availableActions and handle MeloMateWorkspaceAction/melomate-workspace-action."""
    return safe_call(workspace_core.write_workspace_project, persona, folder, files)


@mcp.tool()
def read_workspace_file(persona: str, path: str) -> str:
    """Read a UTF-8 text file from workspace/{persona}. Never read another persona's workspace."""
    return safe_call(workspace_core.read_workspace_file, persona, path)


@mcp.tool()
def inspect_workspace_item(persona: str, path: str = "") -> str:
    """Inspect a workspace file or folder without reading content. Returns type, size, modified time, and a SHA-256 version for files."""
    return safe_call(workspace_core.inspect_workspace_item, persona, path)


@mcp.tool()
def read_workspace_file_range(
    persona: str, path: str, offset: int = 0, max_chars: int = 64000
) -> str:
    """Read a bounded UTF-8 character range and current SHA-256 from a workspace file. Use this for targeted work on long files."""
    return safe_call(
        workspace_core.read_workspace_file_range,
        persona,
        path,
        offset,
        max_chars,
    )


@mcp.tool()
def patch_workspace_file(
    persona: str,
    path: str,
    expected_sha256: str,
    replacements: list[dict],
) -> str:
    """Atomically patch the exact file version identified by expected_sha256. replacements is a list of {old_text,new_text,replace_all?}; include surrounding text when needed."""
    return safe_call(
        workspace_core.patch_workspace_file,
        persona,
        path,
        expected_sha256,
        replacements,
    )


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
    """Move one item to bounded private recovery storage. Set recursive=true only when the user authorized removing a non-empty folder. The persona root and runtime data cannot be removed."""
    return safe_call(workspace_core.delete_workspace_item, persona, path, recursive)


@mcp.tool()
def list_workspace_trash(persona: str) -> str:
    """List recently removed workspace items that can still be restored."""
    return safe_call(workspace_core.list_workspace_trash, persona)


@mcp.tool()
def restore_workspace_item(
    persona: str, trash_id: str, destination: str = ""
) -> str:
    """Restore a recoverably removed workspace item. Omit destination to restore its original relative path."""
    return safe_call(
        workspace_core.restore_workspace_item,
        persona,
        trash_id,
        destination,
    )


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
    """Read verified state reported by an open workspace HTML app for this persona. page_id may select one exact open page; otherwise the most recently reporting page is returned. This is read-only and cannot authorize any side effect. If available=false, do not invent app state."""
    return safe_call(workspace_core.read_workspace_state, persona, page_id)


@mcp.tool()
def act_workspace_page(
    persona: str,
    page_id: str,
    state_version: int,
    action_id: str,
    wait_ms: int = 1200,
) -> str:
    """Apply one exact action advertised by the matching open workspace page revision. Read the page state first and pass its page id, state version, and selected availableActions id. Arbitrary actions and payloads are not accepted."""
    return safe_call(
        workspace_core.send_workspace_action,
        persona,
        "",
        None,
        wait_ms,
        page_id,
        state_version,
        action_id,
    )


@mcp.tool()
def open_workspace_item(persona: str, path: str) -> str:
    """Open a file or folder from workspace/{persona} with the user's default local app. Use after the user says they want to see, open, view, play, or try a generated workspace item."""
    return safe_call(workspace_core.open_workspace_item, persona, path)


if __name__ == "__main__":
    mcp.run()
