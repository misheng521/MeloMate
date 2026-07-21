import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "mcp_local_launcher_targets.json"

mcp = FastMCP("local-launcher")


def normalize_name(value: str) -> str:
    return "".join(value.strip().lower().split())


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def iter_shortcut_dirs() -> list[Path]:
    appdata = os.environ.get("APPDATA", "")
    programdata = os.environ.get("PROGRAMDATA", "")
    shortcut_dirs: list[Path] = []
    if appdata:
        shortcut_dirs.append(Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    if programdata:
        shortcut_dirs.append(Path(programdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return shortcut_dirs


def find_shortcut(keywords: list[str]) -> Path | None:
    normalized_keywords = [normalize_name(keyword) for keyword in keywords if keyword]
    for shortcut_dir in iter_shortcut_dirs():
        if not shortcut_dir.exists():
            continue
        for shortcut in shortcut_dir.rglob("*.lnk"):
            normalized_name = normalize_name(shortcut.stem)
            if any(keyword in normalized_name for keyword in normalized_keywords):
                return shortcut
    return None


def start_uri(uri: str) -> None:
    if sys.platform == "win32":
        os.startfile(uri)  # type: ignore[attr-defined]
        return
    subprocess.Popen(["xdg-open", uri], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_file(path: str | Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    subprocess.Popen([str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_process(path: str | Path, args: list[str] | None = None) -> None:
    subprocess.Popen(
        [str(path), *(args or [])],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


def find_target_path(target: dict[str, Any]) -> Path | None:
    for raw_path in target.get("path_candidates", []):
        path = Path(raw_path)
        if path.exists():
            return path
    return None


def resolve_target(name: str) -> tuple[str, dict[str, Any]] | None:
    config = load_config()
    key = normalize_name(name)

    for target in config.get("targets", []):
        names = [target.get("id", ""), target.get("display_name", ""), *target.get("aliases", [])]
        if key in {normalize_name(item) for item in names if item}:
            return "target", target

    for game in config.get("steam_games", []):
        names = [game.get("id", ""), game.get("display_name", ""), *game.get("aliases", [])]
        if key in {normalize_name(item) for item in names if item}:
            return "steam_game", game

    return None


def launch_battlenet_then_overwatch(target: dict[str, Any]) -> str:
    launcher_path = find_target_path(target)
    launcher_started = False

    if launcher_path:
        start_file(launcher_path)
        launcher_started = True
    else:
        shortcut = find_shortcut(target.get("shortcut_keywords", []))
        if shortcut:
            start_file(shortcut)
            launcher_started = True

    if launcher_started:
        delay = float(target.get("post_launcher_delay_seconds", 4))
        time.sleep(max(0, min(delay, 15)))

    exec_args = target.get("battlenet_exec_args", [])
    if launcher_path and exec_args:
        start_process(launcher_path, exec_args)
        time.sleep(1)

    launched_by_uri = False
    for uri in target.get("uri_candidates", []):
        try:
            start_uri(uri)
            launched_by_uri = True
            time.sleep(1)
        except OSError:
            continue

    if launched_by_uri or launcher_path:
        return f"Started Battle.net and sent Overwatch launch command for {target['display_name']}."

    raise FileNotFoundError(f"Could not find a Battle.net launch method for {target['display_name']}.")


def launch_target_entry(target: dict[str, Any]) -> str:
    if target.get("launch_battlenet_first"):
        return launch_battlenet_then_overwatch(target)

    uri = target.get("uri")
    if uri:
        try:
            start_uri(uri)
            return f"Opened {target['display_name']} via launch protocol."
        except OSError:
            pass

    path = find_target_path(target)
    if path:
        start_file(path)
        return f"Started {target['display_name']}."

    shortcut = find_shortcut(target.get("shortcut_keywords", []))
    if shortcut:
        start_file(shortcut)
        return f"Started {target['display_name']} from the Start Menu."

    raise FileNotFoundError(f"Could not find install path or Start Menu shortcut for {target['display_name']}.")


def launch_steam_game_entry(game: dict[str, Any]) -> str:
    app_id = str(game.get("app_id", ""))
    if not app_id.isdigit():
        raise ValueError("Steam AppID must be numeric.")
    start_uri(f"steam://run/{app_id}")
    return f"Started {game['display_name']} through Steam."


@mcp.tool()
def list_launch_targets() -> str:
    """List the local app/game launch whitelist that this MCP server is allowed to open."""
    config = load_config()
    payload = {
        "apps": [
            {"id": item["id"], "display_name": item["display_name"], "aliases": item.get("aliases", [])}
            for item in config.get("targets", [])
        ],
        "steam_games": [
            {
                "id": item["id"],
                "display_name": item["display_name"],
                "aliases": item.get("aliases", []),
                "app_id": item["app_id"],
            }
            for item in config.get("steam_games", [])
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


@mcp.tool()
def launch_local_target(name: str) -> str:
    """Open a whitelisted local launcher, app, or Steam game by name. Use this when the user asks to open Steam, a configured Steam game, Overwatch, Perfect World Arena, 5E, or Valorant."""
    resolved = resolve_target(name)
    if not resolved:
        allowed = json.loads(list_launch_targets())
        return json.dumps(
            {
                "ok": False,
                "message": f"Not in launch whitelist: {name}",
                "allowed": allowed,
            },
            ensure_ascii=False,
        )

    kind, entry = resolved
    try:
        message = (
            launch_steam_game_entry(entry)
            if kind == "steam_game"
            else launch_target_entry(entry)
        )
        return json.dumps(
            {"ok": True, "target": entry["id"], "message": message},
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps(
            {
                "ok": False,
                "target": entry.get("id"),
                "message": str(exc),
            },
            ensure_ascii=False,
        )


if __name__ == "__main__":
    mcp.run()
