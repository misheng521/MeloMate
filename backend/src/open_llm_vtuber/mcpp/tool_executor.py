import json
import datetime
import asyncio
from uuid import uuid4
from loguru import logger
from typing import (
    Dict,
    Any,
    List,
    Literal,
    Union,
    AsyncIterator,
)

from .types import ToolCallObject
from .mcp_client import MCPClient
from .tool_manager import ToolManager
from .results import collect_result, image_messages
from ..runtime_control import PROJECT_TOOLS, SAFE_READS, scope_arguments, redact
from ..pc_tools import FILE_TOOLS, BROWSER_TOOLS
from ..workspace_security import (
    harden_workspace_tool_result,
)
from ..workspace_intent import WORKSPACE_SIDE_EFFECT_TOOLS
from ..daily_tool_policy import (
    DAILY_PERSONA_TOOLS,
    DAILY_READ_TOOLS,
    DAILY_SIDE_EFFECT_TOOLS,
    DAILY_TOOL_NAMES,
)
from ..network_security import (
    READ_ONLY_NETWORK_TOOLS,
    harden_network_tool_result,
)


WORKSPACE_TOOL_NAMES = {
    *FILE_TOOLS,
    "create_workspace_artifact_bundle",
    "create_workspace_folder",
    "write_workspace_file",
    "append_workspace_file",
    "write_workspace_project",
    "read_workspace_file",
    "list_workspace",
    "replace_workspace_text",
    "move_workspace_item",
    "delete_workspace_item",
    "search_workspace",
    "read_workspace_state",
    "open_workspace_item",
    "inspect_workspace_item",
    "read_workspace_file_range",
    "patch_workspace_file",
    "list_workspace_trash",
    "restore_workspace_item",
    "act_workspace_page",
    "validate_workspace_project",
    "run_workspace_command",
}
TOOL_EXECUTION_TIMEOUT_SECONDS = 30
MAX_TOOL_ARGUMENT_CHARS = 256_000
MAX_TOOL_RESULT_TEXT_CHARS = 64_000
MAX_TOOL_CONTENT_ITEMS = 16
MAX_TOOL_BINARY_CHARS = 12_000_000


class ToolExecutor:
    def __init__(
        self,
        mcp_client: MCPClient,
        tool_manager: ToolManager,
        pc_tools=None,
    ):
        self._mcp_client = mcp_client
        self._tool_manager = tool_manager
        self._pc_tools = pc_tools

    def _tool_timeout(self, name, info, arguments):
        if name == "run_workspace_command":
            return min(650, max(45, int((arguments or {}).get("timeout_seconds", 120)) + 45))
        return max(5, min(600, float(getattr(info, "timeout_seconds", 30))))

    def parse_tool_call(self, call: Union[Dict[str, Any], ToolCallObject]) -> tuple:
        """Parse tool call from different formats.

        Returns:
            tuple: (tool_name, tool_id, tool_input, is_error, result_content, parse_error)
        """
        tool_name: str = ""
        tool_id: str = ""
        tool_input: Any = None
        is_error: bool = False
        result_content: str | dict = ""
        parse_error: bool = False

        if isinstance(call, ToolCallObject):
            tool_name = call.function.name
            tool_id = call.id
            try:
                tool_input = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to decode OpenAI tool arguments for '{tool_name}'"
                )
                result_content = (
                    f"Error: Invalid arguments format for tool '{tool_name}'. "
                    "The tool arguments were not valid JSON, often because long code "
                    "was truncated or contained unescaped quotes. Retry with compact "
                    "valid JSON. For games, mini apps, web pages, or long files, split "
                    "the work into smaller files with write_workspace_project or write "
                    "small chunks with append_workspace_file."
                )
                is_error = True
                parse_error = True
        elif isinstance(call, dict):
            tool_id = call.get("id")
            tool_name = call.get("name")
            tool_input = call.get("input", call.get("args"))

            if tool_input is None:
                logger.warning(
                    f"Empty input for tool '{tool_name}' (ID: {tool_id}). Using empty object."
                )
                tool_input = {}

            if not tool_id or not tool_name:
                logger.error("Invalid dictionary tool-call structure")
                result_content = "Error: Invalid tool call structure from LLM."
                is_error = True
                parse_error = True
        else:
            logger.error(f"Unsupported tool call type: {type(call)}")
            result_content = "Error: Unsupported tool call type."
            is_error = True
            parse_error = True

        return tool_name, tool_id, tool_input, is_error, result_content, parse_error

    def format_tool_result(
        self,
        caller_mode: Literal["Claude", "OpenAI", "Prompt"],
        tool_id: str,
        result_content: str,
        is_error: bool,
    ) -> Dict[str, Any] | None:
        """Format tool result for LLM API."""
        if caller_mode == "Claude":
            # Claude expects content as a list of blocks or a simple string
            # We will return a list if there are multiple items or non-text items
            if isinstance(result_content, list):
                # Already formatted as list of blocks
                content_to_send = result_content
            elif isinstance(result_content, str) and result_content:
                # Simple text result
                content_to_send = result_content
            elif not result_content and is_error:
                # Error case, send error message as string
                content_to_send = "Error occurred during tool execution."
            else:
                # Fallback for empty or unexpected content
                content_to_send = ""

            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content_to_send,
                "is_error": is_error,
            }
        elif caller_mode == "OpenAI":
            # OpenAI expects content as a string
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": str(result_content),
            }
        elif caller_mode == "Prompt":
            # Prompt mode also expects a string content for now
            return {
                "tool_id": tool_id,
                "content": str(result_content),
                "is_error": is_error,
            }
        return None

    def harden_workspace_result(
        self, tool_name: str, is_error: bool, text_content: str
    ) -> tuple[bool, str]:
        result_error, hardened_content = harden_workspace_tool_result(
            tool_name, text_content
        )
        return is_error or result_error, hardened_content

    @staticmethod
    def harden_network_result(tool_name: str, text_content: str) -> str:
        return harden_network_tool_result(tool_name, text_content)

    def apply_tool_policy(
        self,
        tool_name: str,
        tool_input: Any,
        tool_policy: Dict[str, Any] | None,
        consume: bool = False,
    ) -> tuple[Any, str | None]:
        """Enforce the current turn's server-owned workspace capability policy."""
        if tool_name in DAILY_TOOL_NAMES and tool_policy is not None:
            if not isinstance(tool_input, dict):
                return tool_input, "TOOL_POLICY_DENIED: tool arguments must be an object."
            if tool_name in DAILY_PERSONA_TOOLS:
                expected_persona = str(tool_policy.get("workspace_persona") or "")
                supplied_persona = str(tool_input.get("persona") or "")
                if expected_persona and supplied_persona != expected_persona:
                    return tool_input, (
                        "TOOL_POLICY_DENIED: reminder tools may only access the "
                        "current client persona."
                    )
            if (
                tool_name in DAILY_SIDE_EFFECT_TOOLS
                and tool_name
                not in set(tool_policy.get("user_authorized_daily_tools") or ())
            ):
                return tool_input, (
                    "TOOL_POLICY_DENIED: this reminder change was not authorized "
                    "by the user's message for this turn."
                )
        if tool_name in WORKSPACE_TOOL_NAMES and tool_policy is not None:
            if not isinstance(tool_input, dict):
                return tool_input, "TOOL_POLICY_DENIED: tool arguments must be an object."
            expected_persona = str(tool_policy.get("workspace_persona") or "")
            supplied_persona = str(tool_input.get("persona") or "")
            if expected_persona and supplied_persona != expected_persona:
                return tool_input, (
                    "TOOL_POLICY_DENIED: workspace tools may only access the current "
                    "client persona."
                )
            if (
                tool_policy.get("source") == "user_turn"
                and tool_name in WORKSPACE_SIDE_EFFECT_TOOLS
                and tool_name
                not in set(tool_policy.get("user_authorized_workspace_tools") or ())
            ):
                return tool_input, (
                    "TOOL_POLICY_DENIED: this workspace side effect was not "
                    "authorized by the user's message for this turn."
                )
            if tool_name == "act_workspace_page":
                page_id = str(tool_input.get("page_id") or "").strip()[:128]
                action_id = str(tool_input.get("action_id") or "").strip()[:128]
                try:
                    state_version = max(0, int(tool_input.get("state_version") or 0))
                    wait_ms = max(0, min(int(tool_input.get("wait_ms") or 1200), 5000))
                except (TypeError, ValueError, OverflowError):
                    return tool_input, "TOOL_POLICY_DENIED: invalid page action revision."
                if not page_id or not action_id or state_version <= 0:
                    return tool_input, (
                        "TOOL_POLICY_DENIED: page_id, positive state_version, and "
                        "one advertised action_id are required."
                    )
                if tool_policy.get("source") == "workspace_runtime":
                    expected_page_id = str(
                        tool_policy.get("expected_page_id") or ""
                    )[:128]
                    try:
                        expected_state_version = max(
                            0, int(tool_policy.get("expected_state_version") or 0)
                        )
                    except (TypeError, ValueError, OverflowError):
                        expected_state_version = 0
                    allowed_action_ids = {
                        str(value)[:128]
                        for value in tool_policy.get("allowed_action_ids") or ()
                    }
                    if (
                        page_id != expected_page_id
                        or state_version != expected_state_version
                        or action_id not in allowed_action_ids
                    ):
                        return tool_input, (
                            "TOOL_POLICY_DENIED: the page action must match the exact "
                            "verified runtime page revision and advertised action id."
                        )
                scoped_folder = tool_input.get("folder")
                tool_input = {
                    "persona": expected_persona or supplied_persona,
                    "page_id": page_id,
                    "state_version": state_version,
                    "action_id": action_id,
                    "wait_ms": wait_ms,
                }
                if tool_policy.get("project_mode") and scoped_folder:
                    tool_input["folder"] = scoped_folder
        if tool_policy is None or tool_policy.get("enforce") is not True:
            return tool_input, None
        allowed = set(tool_policy.get("allowed_tool_names") or ())
        if tool_name not in allowed:
            return tool_input, (
                f"TOOL_POLICY_DENIED: '{tool_name}' is not permitted for "
                f"{tool_policy.get('source') or 'this turn'}."
            )
        if not isinstance(tool_input, dict):
            return tool_input, "TOOL_POLICY_DENIED: tool arguments must be an object."
        if tool_name in WORKSPACE_TOOL_NAMES or tool_name in DAILY_PERSONA_TOOLS:
            expected_persona = str(tool_policy.get("workspace_persona") or "")
            supplied_persona = str(tool_input.get("persona") or "")
            if not expected_persona or supplied_persona != expected_persona:
                return tool_input, (
                    "TOOL_POLICY_DENIED: scoped tools may only access their "
                    "matching persona."
                )
        if tool_name == "read_workspace_state":
            normalized_input = {"persona": expected_persona}
            page_id = str(tool_input.get("page_id") or "").strip()[:128]
            if page_id:
                normalized_input["page_id"] = page_id
            return normalized_input, self._consume_tool_policy_call(
                tool_name, tool_policy, consume
            )
        return tool_input, self._consume_tool_policy_call(
            tool_name, tool_policy, consume
        )

    @staticmethod
    def _restrict_after_network_result(
        tool_name: str,
        tool_policy: Dict[str, Any] | None,
    ) -> None:
        """A webpage or search result can never authorize a follow-up mutation."""
        if tool_policy is None or tool_name not in READ_ONLY_NETWORK_TOOLS:
            return
        if tool_policy.get("project_mode"):
            tool_policy["network_state_tainted"] = True
            return
        preauthorized_daily = set(
            tool_policy.get("user_authorized_daily_tools") or ()
        )
        preauthorized_workspace = set(
            tool_policy.get("user_authorized_workspace_tools") or ()
        )
        allowed = frozenset(
            set(READ_ONLY_NETWORK_TOOLS)
            | set(DAILY_READ_TOOLS)
            | preauthorized_daily
            | {
                name
                for name in preauthorized_workspace
                if name in WORKSPACE_SIDE_EFFECT_TOOLS
            }
        )
        tool_policy.update(
            {
                "enforce": True,
                "allowed_tool_names": allowed,
                "remaining_tool_calls": {name: 16 for name in allowed},
                "network_state_tainted": True,
            }
        )

    @staticmethod
    def _consume_tool_policy_call(
        tool_name: str, tool_policy: Dict[str, Any], consume: bool
    ) -> str | None:
        if not consume:
            return None
        remaining = tool_policy.get("remaining_tool_calls")
        if not isinstance(remaining, dict):
            return None
        try:
            available = int(remaining.get(tool_name, 0))
        except (TypeError, ValueError):
            available = 0
        if available <= 0:
            return (
                f"TOOL_POLICY_DENIED: '{tool_name}' exceeded the per-turn call limit."
            )
        remaining[tool_name] = available - 1
        return None

    @staticmethod
    def _restrict_after_workspace_state(
        tool_name: str,
        tool_input: Any,
        text_content: str,
        tool_policy: Dict[str, Any] | None,
    ) -> None:
        """Prevent untrusted page state from authorizing unrelated follow-up tools."""
        if tool_policy is None or tool_name != "read_workspace_state":
            return
        if tool_policy.get("project_mode"):
            tool_policy["workspace_state_tainted"] = True
            return
        if tool_policy.get("source") != "user_turn":
            return
        persona = ""
        if isinstance(tool_input, dict):
            persona = str(tool_input.get("persona") or "")
        preauthorized = frozenset(
            str(name)
            for name in tool_policy.get("user_authorized_workspace_tools") or ()
            if str(name) in WORKSPACE_TOOL_NAMES
        )
        allowed = frozenset({"read_workspace_state", *preauthorized})
        tool_policy.update(
            {
                "enforce": True,
                "allowed_tool_names": allowed,
                "workspace_persona": persona,
                "workspace_state_tainted": True,
                "remaining_tool_calls": {name: 64 for name in allowed},
            }
        )

    def process_tool_from_prompt_json(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process tool data from JSON in prompt mode."""
        parsed_tools = []
        tools = self._tool_manager.tools if self._tool_manager else {}
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("tool")
            if not isinstance(name, str) or name not in tools:
                continue
            info = tools[name]
            if item.get("mcp_server", info.related_server) != info.related_server:
                continue
            args = item.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    continue
            if isinstance(args, dict):
                parsed_tools.append({"name": name, "server": info.related_server,
                                     "args": args, "id": "prompt_" + uuid4().hex})
        return parsed_tools

    async def execute_tools(
        self, tool_calls, caller_mode, tool_policy=None,
    ):
        runtime = (tool_policy or {}).get("runtime_control")
        async for event in self._execute_tools(tool_calls, caller_mode, tool_policy):
            if runtime and event.get("type") == "tool_call_status" and event.get("status") in {"completed", "error"}:
                runtime.record(event)
            yield event

    async def _execute_tools(
        self,
        tool_calls: Union[List[Dict[str, Any]], List[ToolCallObject]],
        caller_mode: Literal["Claude", "OpenAI", "Prompt"],
        tool_policy: Dict[str, Any] | None = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute tools and yield status updates."""
        if len(tool_calls) > 1 and (tool_policy or {}).get("project_mode"):
            parsed = [self.parse_tool_call(call) for call in tool_calls]
            if all(name in SAFE_READS and name != "read_workspace_state" and not err
                   for name, _, _, err, _, _ in parsed):
                queue = asyncio.Queue()
                semaphore = asyncio.Semaphore(4)
                async def run(index, call):
                    try:
                        async with semaphore:
                            async for event in self._execute_tools([call], caller_mode, dict(tool_policy)):
                                await queue.put((index, event))
                    finally:
                        await queue.put((index, None))
                tasks = [asyncio.create_task(run(i, call)) for i, call in enumerate(tool_calls)]
                finished, results = 0, {}
                try:
                    while finished < len(tasks):
                        index, event = await queue.get()
                        if event is None:
                            finished += 1
                        elif event.get("type") == "final_tool_results":
                            results[index] = event["results"]
                        else:
                            yield event
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done(): task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                for name, *_ in parsed:
                    self._restrict_after_network_result(name, tool_policy)
                combined = [item for i in range(len(tasks)) for item in results.get(i, [])]
                if caller_mode == "OpenAI": combined.sort(key=lambda item: item.get("role") != "tool")
                yield {"type": "final_tool_results", "results": combined}
                return
        tool_results_for_llm = []

        logger.info(f"Executing {len(tool_calls)} tool(s) for {caller_mode} caller.")
        for call in tool_calls:
            (
                tool_name,
                tool_id,
                tool_input,
                is_error,
                result_content,
                parse_error,
            ) = self.parse_tool_call(call)

            logger.info(f"Executing tool request: {tool_name or 'unknown'}")

            if parse_error:
                logger.warning(
                    f"Skipping tool call due to parsing error: {result_content}"
                )
                status_update = {
                    "type": "tool_call_status",
                    "tool_id": tool_id
                    or f"parse_error_{datetime.datetime.now(datetime.timezone.utc).isoformat()}",
                    "tool_name": tool_name or "Unknown Tool",
                    "status": "error",
                    "content": result_content,
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                    + "Z",
                }
                yield status_update
                # Even on parse error, we might need to format a result for the LLM
                # Use dummy values or the error message
                formatted_result = self.format_tool_result(
                    caller_mode,
                    tool_id
                    or f"parse_error_{datetime.datetime.now(datetime.timezone.utc).isoformat()}",
                    result_content,
                    True,  # is_error
                )
                if formatted_result:
                    tool_results_for_llm.append(formatted_result)
                continue  # Skip execution logic for this call

            tool_input, policy_error = self.apply_tool_policy(
                tool_name, tool_input, tool_policy, consume=True
            )
            runtime = (tool_policy or {}).get("runtime_control")
            if not policy_error and runtime:
                if not isinstance(tool_input, dict):
                    policy_error = "Tool arguments must be a JSON object."
                else:
                    try:
                        tool_input = scope_arguments(tool_name, tool_input, tool_policy)
                    except ValueError as exc:
                        policy_error = str(exc)
                    if not policy_error and not await runtime.authorize(tool_name, tool_input, tool_policy):
                        policy_error = "TOOL_PERMISSION_DENIED: user denied, permission expired, or tool is disabled. Do not retry without new permission."
            if policy_error:
                logger.warning(policy_error)
                yield {
                    "type": "tool_call_status",
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "status": "error",
                    "content": policy_error,
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                    + "Z",
                }
                formatted_result = self.format_tool_result(
                    caller_mode, tool_id, policy_error, True
                )
                if formatted_result:
                    tool_results_for_llm.append(formatted_result)
                continue

            try:
                serialized_input = json.dumps(tool_input, ensure_ascii=False)
            except (TypeError, ValueError):
                serialized_input = None
            if serialized_input is None or len(serialized_input) > MAX_TOOL_ARGUMENT_CHARS:
                policy_error = (
                    "TOOL_POLICY_DENIED: tool arguments are invalid or exceed the size limit."
                )
                logger.warning(f"Oversized arguments rejected for tool '{tool_name}'")
                yield {
                    "type": "tool_call_status",
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "status": "error",
                    "content": policy_error,
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                    + "Z",
                }
                formatted_result = self.format_tool_result(
                    caller_mode, tool_id, policy_error, True
                )
                if formatted_result:
                    tool_results_for_llm.append(formatted_result)
                continue

            # Yield 'running' status before execution
            input_preview = redact(serialized_input)
            if len(input_preview) > 1000:
                input_preview = f"{input_preview[:1000]}... [truncated]"

            yield {
                "type": "tool_call_status",
                "tool_id": tool_id,
                "tool_name": tool_name,
                "status": "running",
                "content": f"Input: {input_preview}",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                + "Z",
            }

            # Execute the tool
            (
                is_error,
                text_content,
                metadata,
                content_items,
            ) = await self.run_single_tool(
                tool_name, tool_id, tool_input, tool_policy=tool_policy
            )
            self._restrict_after_workspace_state(
                tool_name, tool_input, text_content, tool_policy
            )
            self._restrict_after_network_result(tool_name, tool_policy)

            # Guard text separately; never stringify image blocks.
            is_error, guarded = self.harden_workspace_result(tool_name, is_error, text_content)
            guarded = self.harden_network_result(tool_name, guarded)
            _, _, image_items = collect_result({"content_items": content_items})
            status_content = guarded + (f"\n[Tool returned {len(image_items)} image(s)]" if image_items else "")
            llm_formatted_content = guarded
            if image_items and caller_mode == "Claude":
                llm_formatted_content = image_messages(image_items, guarded, "Claude", tool_id)
            text_content = guarded

            # Prepare and yield tool call status update
            status_update = {
                "type": "tool_call_status",
                "tool_id": tool_id,
                "tool_name": tool_name,
                "status": "error" if is_error else "completed",
                "content": status_content
                if not is_error
                else f"Error: {text_content}",  # Use descriptive content or error message
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                + "Z",
            }

            # For stagehand_navigate tool, include browser view links if available
            if tool_name in BROWSER_TOOLS and image_items:
                first = image_items[0]
                status_update["preview_image"] = f"data:{first['mimeType']};base64,{first['data']}"
            if tool_name == "stagehand_navigate" and not is_error:
                live_view_data = metadata.get("liveViewData", {})
                if live_view_data:
                    logger.info("Found live view data for stagehand_navigate")
                    status_update["browser_view"] = live_view_data

            yield status_update

            # Format result for LLM and add to list
            formatted_result = self.format_tool_result(
                caller_mode, tool_id, llm_formatted_content, is_error
            )
            if formatted_result:
                tool_results_for_llm.append(formatted_result)
            if image_items and caller_mode == "OpenAI":
                tool_results_for_llm.extend(image_messages(image_items, guarded, "OpenAI", tool_id))

        # OpenAI requires all tool replies before a subsequent user/image message.
        if caller_mode == "OpenAI":
            tool_results_for_llm.sort(key=lambda item: item.get("role") != "tool")
        logger.info(
            f"Finished executing tools with {len(tool_results_for_llm)} results."
        )
        yield {"type": "final_tool_results", "results": tool_results_for_llm}

    async def run_single_tool(
        self,
        tool_name: str,
        tool_id: str,
        tool_input: Any,
        tool_policy: Dict[str, Any] | None = None,
    ) -> tuple[bool, str, Dict[str, Any], List[Dict[str, Any]]]:
        """Run a single tool using MCPClient.

        Returns:
            tuple: (is_error, text_content, metadata, content_items)
        """
        logger.info(f"Executing tool: {tool_name} (ID: {tool_id})")

        is_error = False
        text_content = ""
        metadata = {}
        content_items = []

        if tool_input is None:
            tool_input = {}

        tool_input, policy_error = self.apply_tool_policy(
            tool_name, tool_input, tool_policy
        )
        if policy_error:
            return True, policy_error, {}, [{"type": "error", "text": policy_error}]

        tool_info = self._tool_manager.get_tool(tool_name)
        if not tool_info:
            logger.error(f"Tool '{tool_name}' not found in ToolManager.")
            text_content = f"Error: Tool '{tool_name}' is not available."
            content_items = [{"type": "error", "text": text_content}]
            is_error = True
        elif not tool_info.related_server:
            logger.error(f"Tool '{tool_name}' does not have a related server defined.")
            text_content = f"Error: Configuration error for tool '{tool_name}'. No server specified."
            content_items = [{"type": "error", "text": text_content}]
            is_error = True
        else:
            try:
                if tool_info.related_server == "__pc__":
                    if not self._pc_tools or not (tool_policy or {}).get("runtime_control"):
                        raise ValueError("PC tools require a trusted current conversation policy")
                    invocation = self._pc_tools.call(tool_name, tool_input)
                else:
                    invocation = self._mcp_client.call_tool(
                        server_name=tool_info.related_server,
                        tool_name=getattr(tool_info, "original_name", "") or tool_name,
                        tool_args=tool_input)
                result_dict = await asyncio.wait_for(
                    invocation,
                    timeout=self._tool_timeout(tool_name, tool_info, tool_input),
                )

                metadata = result_dict.get("metadata", {})
                content_items = result_dict.get("content_items", [])
                if not isinstance(content_items, list):
                    content_items = []
                content_items = content_items[:MAX_TOOL_CONTENT_ITEMS]
                for item in content_items:
                    if not isinstance(item, dict):
                        continue
                    text_value = item.get("text")
                    if (
                        isinstance(text_value, str)
                        and len(text_value) > MAX_TOOL_RESULT_TEXT_CHARS
                    ):
                        item["text"] = (
                            text_value[:MAX_TOOL_RESULT_TEXT_CHARS]
                            + "\n[Tool result truncated by MeloMate]"
                        )
                    binary_value = item.get("data")
                    if (
                        isinstance(binary_value, str)
                        and len(binary_value) > MAX_TOOL_BINARY_CHARS
                    ):
                        item.pop("data", None)
                        item["type"] = "error"
                        item["text"] = "Tool binary result exceeded the size limit."

                result_error, text_content, _ = collect_result({**result_dict, "content_items": content_items})
                is_error = is_error or result_error

                if not is_error:
                    logger.info(f"Tool '{tool_name}' executed successfully.")
                    if content_items:
                        item_types = [
                            str(item.get("type", "unknown"))
                            for item in content_items
                            if isinstance(item, dict)
                        ]
                        logger.info(
                            f"Tool '{tool_name}' returned {len(content_items)} "
                            f"item(s), types={item_types}"
                        )

            except asyncio.TimeoutError:
                logger.warning(f"Tool '{tool_name}' reached the execution time limit")
                text_content = f"Error: Tool '{tool_name}' timed out; its side effects are unknown. Inspect current state before retrying."
                content_items = [{"type": "error", "text": text_content}]
                is_error = True
            except (ValueError, RuntimeError, ConnectionError) as e:
                logger.error(
                    f"Error executing tool '{tool_name}': {type(e).__name__}"
                )
                text_content = f"Error executing tool '{tool_name}': {redact(e)}. Correct the arguments or choose another available tool."
                content_items = [{"type": "error", "text": text_content}]
                is_error = True
            except Exception as e:
                logger.error(
                    f"Unexpected tool error for '{tool_name}': {type(e).__name__}"
                )
                text_content = f"Unexpected error executing tool '{tool_name}': {redact(e)}. Do not claim success."
                content_items = [{"type": "error", "text": text_content}]
                is_error = True

        return is_error, text_content, metadata, content_items
