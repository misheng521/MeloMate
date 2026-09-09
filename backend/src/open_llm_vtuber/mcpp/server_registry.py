"""MCP Server Manager for Open-LLM-Vtuber."""

import shutil
import json
import sys
from datetime import timedelta

from pathlib import Path
from typing import Dict, Optional, Union, Any
from loguru import logger

from .types import MCPServer
from .utils.path import validate_file

DEFAULT_CONFIG_PATH = "mcp_servers.json"


class ServerRegistry:
    """MCP Server Manager for managing server files."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        """Initialize the MCP Server Manager."""
        if str(config_path) == DEFAULT_CONFIG_PATH:
            config_path = Path(__file__).resolve().parents[3] / DEFAULT_CONFIG_PATH
        try:
            config_path = validate_file(config_path, ".json")
        except ValueError:
            logger.error(
                f"MCPSR: File '{config_path}' does not exist, or is not a json file."
            )
            if Path(config_path).name != DEFAULT_CONFIG_PATH:
                raise

        config_path = Path(config_path)
        self.config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {"mcp_servers": {}}
        backend = Path(__file__).resolve().parents[3]
        entries = self.config.setdefault("mcp_servers", {})
        for name, file in (("workspace", "mcp_workspace.py"), ("daily-tools", "mcp_daily_tools.py")):
            entries.setdefault(name, {"command": sys.executable, "args": [str(backend / file)], "cwd": str(backend)})

        self.servers: Dict[str, MCPServer] = {}

        self.npx_available = self._detect_runtime("npx")
        self.uvx_available = self._detect_runtime("uvx")
        self.node_available = self._detect_runtime("node")

        self.load_servers()

    def _detect_runtime(self, target: str) -> bool:
        """Check if a runtime is available in the system PATH."""
        founded = shutil.which(target)
        return True if founded else False

    def load_servers(self) -> None:
        """Load servers from the config file."""
        servers_config: Dict[str, Dict[str, Any]] = self.config.get("mcp_servers", {})
        if servers_config == {}:
            logger.warning("MCPSR: No servers found in the config file.")
            return

        for server_name, server_details in servers_config.items():
            if server_details.get("enabled") is False:
                continue
            timeout = timedelta(seconds=max(5, min(600, float(server_details.get("timeout") or 30))))
            if server_details.get("url"):
                from urllib.parse import urlsplit
                parsed = urlsplit(server_details["url"])
                if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                    raise ValueError("MCP URL must be HTTP(S) without embedded credentials")
                self.servers[server_name] = MCPServer(name=server_name, url=server_details["url"],
                    headers=server_details.get("headers"), timeout=timeout)
                continue
            if "command" not in server_details or "args" not in server_details:
                logger.warning(
                    f"MCPSR: Invalid server details for '{server_name}'. Ignoring."
                )
                continue

            command = server_details["command"]
            if command == "npx":
                if not self.npx_available:
                    logger.warning(
                        f"MCPSR: npx is not available. Cannot load server '{server_name}'."
                    )
                    continue
            elif command == "uvx":
                if not self.uvx_available:
                    logger.warning(
                        f"MCPSR: uvx is not available. Cannot load server '{server_name}'."
                    )
                    continue

            elif command == "node":
                if not self.node_available:
                    logger.warning(
                        f"MCPSR: node is not available. Cannot load server '{server_name}'."
                    )
                    continue

            self.servers[server_name] = MCPServer(
                name=server_name,
                command=command,
                args=server_details["args"],
                env=server_details.get("env", None),
                cwd=server_details.get("cwd", None),
                timeout=timeout,
            )
            logger.debug(f"MCPSR: Loaded server: '{server_name}'.")

    def remove_server(self, server_name: str) -> None:
        """Remove a server from the available servers."""
        try:
            self.servers.pop(server_name)
            logger.info(f"MCPSR: Removed server: {server_name}")
        except KeyError:
            logger.warning(f"MCPSR: Server '{server_name}' not found. Cannot remove.")

    def get_server(self, server_name: str) -> Optional[MCPServer]:
        """Get the server by name."""
        return self.servers.get(server_name, None)
