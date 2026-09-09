"""MCP Client for Open-LLM-Vtuber."""

from contextlib import AsyncExitStack
import asyncio
from typing import Dict, Any, List, Callable
from loguru import logger
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.types import Tool
from mcp.client.stdio import stdio_client

from .server_registry import ServerRegistry

DEFAULT_TIMEOUT = timedelta(seconds=30)


class MCPClient:
    """MCP Client for Open-LLM-Vtuber.
    Manages persistent connections to multiple MCP servers.
    """

    def __init__(
        self,
        server_registery: ServerRegistry,
        send_text: Callable = None,
        client_uid: str = None,
    ) -> None:
        """Initialize the MCP Client."""
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.active_sessions: Dict[str, ClientSession] = {}
        self._list_tools_cache: Dict[str, List[Tool]] = {}  # Cache for list_tools
        self._connections = {}
        self._send_text: Callable = send_text
        self._client_uid: str = client_uid

        if isinstance(server_registery, ServerRegistry):
            self.server_registery = server_registery
        else:
            raise TypeError(
                "MCPC: Invalid server manager. Must be an instance of ServerRegistry."
            )
        logger.info("MCPC: Initialized MCPClient instance.")

    async def _ensure_server_running_and_get_session(
        self, server_name: str
    ) -> ClientSession:
        """Gets the existing session or creates a new one."""
        if server_name in self.active_sessions:
            return self.active_sessions[server_name]
        existing = self._connections.get(server_name)
        if existing and existing[0].done():
            self._connections.pop(server_name)
        if server_name not in self._connections:
            ready = asyncio.get_running_loop().create_future()
            stop = asyncio.Event()
            task = asyncio.create_task(self._own_connection(server_name, ready, stop))
            self._connections[server_name] = (task, ready, stop)
        # One transport per server even when independent tools start together.
        return await asyncio.shield(self._connections[server_name][1])

    async def _own_connection(self, server_name, ready, stop):
        # MCP transports use task-local cancellation scopes. The task entering
        # those scopes also owns their exit, across all conversation turns.
        try:
            async with AsyncExitStack() as stack:
                session = await self._connect(server_name, stack)
                ready.set_result(session)
                await stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                logger.warning(f"MCP transport closed: {type(exc).__name__}")
        finally:
            self.active_sessions.pop(server_name, None)

    async def _connect(self, server_name, stack):

        logger.info(f"MCPC: Starting and connecting to server '{server_name}'...")
        server = self.server_registery.get_server(server_name)
        if not server:
            raise ValueError(
                f"MCPC: Server '{server_name}' not found in available servers."
            )

        timeout = server.timeout if server.timeout else DEFAULT_TIMEOUT

        try:
            if server.url:
                from mcp.client.streamable_http import streamablehttp_client
                transport = await stack.enter_async_context(
                    streamablehttp_client(server.url, headers=server.headers, timeout=timeout))
                read, write = transport[:2]
            else:
                server_params = StdioServerParameters(
                    command=server.command, args=server.args, env=server.env, cwd=server.cwd)
                read, write = await stack.enter_async_context(stdio_client(server_params))

            session = await stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=timeout)
            )
            await session.initialize()

            self.active_sessions[server_name] = session
            logger.info(f"MCPC: Successfully connected to server '{server_name}'.")
            return session
        except Exception as e:
            logger.exception(f"MCPC: Failed to connect to server '{server_name}': {e}")
            raise RuntimeError(
                f"MCPC: Failed to connect to server '{server_name}'."
            ) from e

    async def list_tools(self, server_name: str) -> List[Tool]:
        """List all available tools on the specified server."""
        # Check cache first
        if server_name in self._list_tools_cache:
            logger.debug(f"MCPC: Cache hit for list_tools on server '{server_name}'.")
            return self._list_tools_cache[server_name]

        logger.debug(
            f"MCPC: Cache miss for list_tools on server '{server_name}'. Fetching..."
        )
        session = await self._ensure_server_running_and_get_session(server_name)
        response = await session.list_tools()
        all_tools = list(response.tools)
        seen_cursors = set()
        while getattr(response, "nextCursor", None):
            cursor = response.nextCursor
            if cursor in seen_cursors or len(all_tools) > 1000:
                raise ValueError("Invalid or oversized MCP tool pagination")
            seen_cursors.add(cursor)
            response = await session.list_tools(cursor=cursor)
            all_tools.extend(response.tools)

        # Store in cache before returning
        if len(all_tools) > 1000:
            raise ValueError("Oversized MCP tool list")
        self._list_tools_cache[server_name] = all_tools
        logger.debug(f"MCPC: Cached list_tools result for server '{server_name}'.")
        return all_tools

    async def call_tool(
        self, server_name: str, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool on the specified server.

        Returns:
            Dict containing the metadata and content_items from the tool response.
        """
        session = await self._ensure_server_running_and_get_session(server_name)
        logger.info(f"MCPC: Calling tool '{tool_name}' on server '{server_name}'...")
        if server_name == "workspace" and tool_name == "run_workspace_command":
            # The SDK's per-request read timeout must match the outer executor;
            # otherwise its 30-second default interrupts legitimate test runs.
            command_timeout = max(1, min(600, int(tool_args.get("timeout_seconds", 120))))
            response = await session.call_tool(tool_name, tool_args,
                read_timeout_seconds=timedelta(seconds=command_timeout + 45))
        else:
            response = await session.call_tool(tool_name, tool_args)

        content_items = []
        if response.content:
            for item in response.content:
                item_dict = {"type": getattr(item, "type", "text")}
                # Extract available attributes from content item
                for attr in [
                    "text",
                    "data",
                    "mimeType",
                    "url",
                    "altText",
                    "uri",
                    "name",
                    "description",
                    "resource",
                ]:  # Added url and altText
                    if (
                        hasattr(item, attr) and getattr(item, attr) is not None
                    ):  # Check for None
                        value = getattr(item, attr)
                        item_dict[attr] = value.model_dump() if hasattr(value, "model_dump") else value
                content_items.append(item_dict)
        else:
            logger.warning(
                f"MCPC: Tool '{tool_name}' returned no content. Returning empty content_items."
            )
            content_items.append(
                {"type": "text", "text": ""}
            )  # Ensure content_items is not empty

        result = {
            "is_error": bool(response.isError),
            "structured_content": getattr(response, "structuredContent", None),
            "metadata": getattr(response, "metadata", {}),
            "content_items": content_items,
        }
        return result

    async def aclose(self) -> None:
        """Closes all active server connections."""
        logger.info(
            f"MCPC: Closing client instance and {len(self.active_sessions)} active connections..."
        )
        connections, self._connections = self._connections, {}
        for task, ready, stop in connections.values():
            stop.set()
            if not ready.done():
                task.cancel()
        try:
            await asyncio.gather(*(task for task, _, _ in connections.values()), return_exceptions=True)
        finally:
            self.active_sessions.clear()
            self._list_tools_cache.clear()
        logger.info("MCPC: Client instance closed.")

    async def __aenter__(self) -> "MCPClient":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the async context manager."""
        await self.aclose()
        if exc_type:
            logger.error(f"MCPC: Exception in async context: {exc_value}")


# if __name__ == "__main__":
#     # Test the MCPClient.
#     async def main():
#         server_registery = ServerRegistry()
#         async with MCPClient(server_registery) as client:
#             # Assuming 'example' server and 'example_tool' exist
#             # The old call used: await client.call_tool("example_tool", {"arg1": "value1"})
#             # The new call needs server name:
#             try:
#                 result = await client.call_tool("example", "example_tool", {"arg1": "value1"})
#                 print(f"Tool result: {result}")
#                 # Test error handling by calling a non-existent tool
#                 await client.call_tool("example", "non_existent_tool", {})
#             except ValueError as e:
#                 print(f"Caught expected error: {e}")
#             except Exception as e:
#                 print(f"Caught unexpected error: {e}")

#     asyncio.run(main())
