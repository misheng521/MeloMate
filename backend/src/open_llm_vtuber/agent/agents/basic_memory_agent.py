from typing import (
    AsyncIterator,
    List,
    Dict,
    Any,
    Callable,
    Literal,
    Union,
    Optional,
)
import datetime
import json
import asyncio
from loguru import logger
from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ..stateless_llm.claude_llm import AsyncLLM as ClaudeAsyncLLM
from ..stateless_llm.openai_compatible_llm import AsyncLLM as OpenAICompatibleAsyncLLM
from ...chat_history_manager import (
    commit_core_memory_review,
    get_history,
    prepare_core_memory_review,
    record_core_memory_review_failure,
)
from ...memory_consolidator import (
    MAX_REVIEW_RESPONSE_CHARS,
    build_memory_review_request,
    parse_memory_review_response,
)
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource
from ...mcpp.tool_manager import ToolManager
from ...mcpp.json_detector import StreamJSONDetector
from ...mcpp.types import ToolCallObject
from ...mcpp.tool_executor import ToolExecutor
from ...workspace_security import (
    WORKSPACE_STATE_RESULT_SYSTEM_GUARD,
)
from ...workspace_intent import (
    workspace_fast_ack_text,
    workspace_user_authorized_tools,
)


WORKSPACE_TOOL_NAMES = {
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
    "act_workspace_page",
    "restore_workspace_item",
    "read_workspace_file_range",
    "patch_workspace_file",
    "list_workspace_trash",
}

SILENT_WORKSPACE_TOOL_NAMES = {
    "read_workspace_state",
}

WORKSPACE_WRITE_TOOL_NAMES = {
    "create_workspace_folder",
    "write_workspace_file",
    "append_workspace_file",
    "write_workspace_project",
    "replace_workspace_text",
    "move_workspace_item",
    "delete_workspace_item",
    "restore_workspace_item",
    "patch_workspace_file",
}

DEFAULT_MAX_TOOL_ROUNDS = 8
WORKSPACE_MAX_TOOL_ROUNDS = 24
MAX_TOOL_CALLS_PER_ROUND = 8
MAX_TOOL_CALLS_PER_TURN = 16
MAX_WORKSPACE_TOOL_CALLS_PER_TURN = 64
TOOL_TURN_TIMEOUT_SECONDS = 600
TOOL_LIMIT_MESSAGE = "这次工作量比较大，我已经保留了完成的部分，你说继续我就接着做。"


class BasicMemoryAgent(AgentInterface):
    """Agent with basic chat memory and tool calling support."""

    _system: str = "You are a helpful assistant."

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = False,
        segment_method: str = "regex",
        use_mcpp: bool = False,
        interrupt_method: Literal["system", "user"] = "user",
        tool_prompts: Dict[str, str] = None,
        tool_manager: Optional[ToolManager] = None,
        tool_executor: Optional[ToolExecutor] = None,
        mcp_prompt_string: str = "",
        memory_conf_uid: str = "",
        memory_character_name: str = "",
    ):
        """Initialize agent with LLM and configuration."""
        super().__init__()
        self._memory = []
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self._use_mcpp = use_mcpp
        self.interrupt_method = interrupt_method
        self._tool_prompts = tool_prompts or {}
        self._interrupt_handled = False
        self.prompt_mode_flag = False

        self._tool_manager = tool_manager
        self._tool_executor = tool_executor
        self._mcp_prompt_string = mcp_prompt_string
        self._json_detector = StreamJSONDetector()
        self._memory_conf_uid = str(memory_conf_uid or "")
        self._memory_character_name = str(memory_character_name or "角色")
        self._memory_review_task: asyncio.Task | None = None

        self._formatted_tools_openai = []
        self._formatted_tools_claude = []
        if self._tool_manager:
            self._formatted_tools_openai = self._tool_manager.get_formatted_tools(
                "OpenAI"
            )
            self._formatted_tools_claude = self._tool_manager.get_formatted_tools(
                "Claude"
            )
            logger.debug(
                f"Agent received pre-formatted tools - OpenAI: {len(self._formatted_tools_openai)}, Claude: {len(self._formatted_tools_claude)}"
            )
        else:
            logger.debug(
                "ToolManager not provided, agent will not have pre-formatted tools."
            )

        self._set_llm(llm)
        self.set_system(system if system else self._system)

        if self._use_mcpp and not all(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is True, but some MCP components are missing in the agent. Tool calling might not work as expected."
            )
        elif not self._use_mcpp and any(
            [
                self._tool_manager,
                self._tool_executor,
            ]
        ):
            logger.warning(
                "use_mcpp is False, but some MCP components were passed to the agent."
            )

        logger.info("BasicMemoryAgent initialized.")

    def schedule_core_memory_review(self) -> bool:
        """Start one bounded background consolidation when its turn threshold is due."""
        if not self._memory_conf_uid:
            return False
        if self._memory_review_task and not self._memory_review_task.done():
            return False

        snapshot = prepare_core_memory_review(self._memory_conf_uid)
        if snapshot is None:
            return False
        self._memory_review_task = asyncio.create_task(
            self._run_core_memory_review(snapshot),
            name=f"memory-review-{self._memory_conf_uid}",
        )
        return True

    async def _run_core_memory_review(self, snapshot: dict) -> None:
        try:
            messages, system = build_memory_review_request(
                snapshot, self._memory_character_name
            )
            response_parts: list[str] = []
            response_size = 0
            async with asyncio.timeout(60):
                stream = self._llm.chat_completion(messages=messages, system=system)
                async for event in stream:
                    text = ""
                    if isinstance(event, str):
                        text = event
                    elif isinstance(event, dict) and event.get("type") == "text_delta":
                        text = str(event.get("text") or "")
                    if not text:
                        continue
                    response_size += len(text)
                    if response_size > MAX_REVIEW_RESPONSE_CHARS:
                        raise ValueError("Memory review response exceeded its limit")
                    response_parts.append(text)

            candidate = parse_memory_review_response("".join(response_parts))
            committed = commit_core_memory_review(
                self._memory_conf_uid,
                str(snapshot.get("snapshot_message_id") or ""),
                candidate,
                base_core_memory=(
                    snapshot.get("core_memory")
                    if isinstance(snapshot.get("core_memory"), dict)
                    else None
                ),
            )
            if not committed:
                raise ValueError("Memory review snapshot was no longer valid")
            logger.info("Core memory review completed.")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            record_core_memory_review_failure(self._memory_conf_uid)
            logger.warning(
                "Core memory review failed safely ({}).", type(error).__name__
            )

    async def close(self) -> None:
        """Give an in-flight review a short grace period, then cancel it cleanly."""
        task = self._memory_review_task
        self._memory_review_task = None
        if task is None:
            return
        if not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()
            except asyncio.CancelledError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
        await asyncio.gather(task, return_exceptions=True)

    def _set_llm(self, llm: StatelessLLMInterface):
        """Set the LLM for chat completion."""
        self._llm = llm
        self.chat = self._chat_function_factory()

    def set_system(self, system: str):
        """Set the system prompt."""
        logger.debug(f"Memory Agent: setting system prompt (chars={len(system)})")

        if self.interrupt_method == "user":
            system = f"{system}\n\nIf you received `[interrupted by user]` signal, you were interrupted."

        self._system = system

    def _add_message(
        self,
        message: Union[str, List[Dict[str, Any]]],
        role: str,
        display_text: DisplayText | None = None,
        skip_memory: bool = False,
    ):
        """Add message to memory."""
        if skip_memory:
            return

        text_content = ""
        if isinstance(message, list):
            for item in message:
                if item.get("type") == "text":
                    text_content += item["text"] + " "
            text_content = text_content.strip()
        elif isinstance(message, str):
            text_content = message
        else:
            logger.warning(
                f"_add_message received unexpected message type: {type(message)}"
            )
            text_content = str(message)

        if not text_content and role == "assistant":
            return

        message_data = {
            "role": role,
            "content": text_content,
        }

        if display_text:
            if display_text.name:
                message_data["name"] = display_text.name

        if (
            self._memory
            and self._memory[-1]["role"] == role
            and self._memory[-1]["content"] == text_content
        ):
            return

        self._memory.append(message_data)

    def add_external_assistant_message(self, message: str) -> None:
        """Record a verified workspace reply in the shared chat memory."""
        self._add_message(str(message or "").strip(), "assistant")

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load memory from chat history."""
        messages = get_history(conf_uid, history_uid)

        self._memory = []
        for index, msg in enumerate(messages):
            role = "user" if msg["role"] == "human" else "assistant"
            content = msg["content"]
            if isinstance(content, str) and content:
                self._memory.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )
            else:
                logger.warning(f"Skipping invalid message from history at index {index}")
        logger.info(f"Loaded {len(self._memory)} messages from history.")

    def handle_interrupt(self, heard_response: str) -> None:
        """Handle user interruption."""
        if self._interrupt_handled:
            return

        self._interrupt_handled = True

        if self._memory and self._memory[-1]["role"] == "assistant":
            if not self._memory[-1]["content"].endswith("..."):
                self._memory[-1]["content"] = heard_response + "..."
            else:
                self._memory[-1]["content"] = heard_response + "..."
        else:
            if heard_response:
                self._memory.append(
                    {
                        "role": "assistant",
                        "content": heard_response + "...",
                    }
                )

        interrupt_role = "system" if self.interrupt_method == "system" else "user"
        self._memory.append(
            {
                "role": interrupt_role,
                "content": "[Interrupted by user]",
            }
        )
        logger.info(f"Handled interrupt with role '{interrupt_role}'.")

    def _tool_call_name(self, call: Union[Dict[str, Any], ToolCallObject]) -> str:
        if isinstance(call, ToolCallObject):
            return call.function.name
        return str(call.get("name") or call.get("function", {}).get("name") or "")

    def _tool_call_id(self, call: Union[Dict[str, Any], ToolCallObject]) -> str:
        if isinstance(call, ToolCallObject):
            return str(call.id or "")
        return str(call.get("id") or "")

    @staticmethod
    def _formatted_tool_name(tool: Dict[str, Any], mode: str) -> str:
        if mode == "OpenAI":
            return str(tool.get("function", {}).get("name") or "")
        return str(tool.get("name") or "")

    def _filter_tools_for_policy(
        self,
        tools: List[Dict[str, Any]],
        mode: str,
        tool_policy: Dict[str, Any] | None,
    ) -> List[Dict[str, Any]]:
        if tool_policy is None:
            return tools
        if tool_policy.get("enforce") is True:
            allowed = set(tool_policy.get("allowed_tool_names") or ())
            return [
                tool
                for tool in tools
                if self._formatted_tool_name(tool, mode) in allowed
            ]
        if tool_policy.get("filter_workspace_tools") is True:
            allowed_workspace = set(
                tool_policy.get("user_authorized_workspace_tools") or ()
            )
            return [
                tool
                for tool in tools
                if self._formatted_tool_name(tool, mode) not in WORKSPACE_TOOL_NAMES
                or self._formatted_tool_name(tool, mode) in allowed_workspace
            ]
        return tools

    @staticmethod
    def _secure_system_prompt_for_policy(
        system_prompt: str, tool_policy: Dict[str, Any] | None
    ) -> str:
        if not tool_policy:
            return system_prompt
        secured_prompt = system_prompt
        if tool_policy.get("workspace_fast_ack_sent") is True:
            secured_prompt += (
                "\n\nThe application has already given the user a short spoken "
                "acknowledgement for this workspace task. Start the required tool "
                "work immediately. Do not repeat a promise to start, and do not say "
                "the work is complete until the tools confirm it."
            )
        if tool_policy.get("workspace_state_tainted") is True:
            return f"{secured_prompt}\n\n{WORKSPACE_STATE_RESULT_SYSTEM_GUARD}"
        return secured_prompt

    @staticmethod
    def _user_input_text(input_data: BatchInput) -> str:
        return "\n".join(
            text.content
            for text in input_data.texts
            if text.source == TextSource.INPUT and text.content
        )

    def _workspace_write_tools_available(
        self, tools: List[Dict[str, Any]] | None, mode: str | None
    ) -> bool:
        if not tools or mode not in {"OpenAI", "Claude"}:
            return False
        return any(
            self._formatted_tool_name(tool, mode) in WORKSPACE_WRITE_TOOL_NAMES
            for tool in tools
        )

    def _contains_workspace_tool(
        self, tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]]
    ) -> bool:
        return any(self._tool_call_name(call) in WORKSPACE_TOOL_NAMES for call in tool_calls)

    def _needs_visible_workspace_ack(
        self, tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]]
    ) -> bool:
        names = {self._tool_call_name(call) for call in tool_calls}
        return bool(names & WORKSPACE_TOOL_NAMES) and not names <= SILENT_WORKSPACE_TOOL_NAMES

    @staticmethod
    def _consume_tool_call_budget(
        total_calls: int,
        batch_size: int,
        maximum_calls: int = MAX_TOOL_CALLS_PER_TURN,
    ) -> int:
        if batch_size < 1 or batch_size > MAX_TOOL_CALLS_PER_ROUND:
            raise RuntimeError(TOOL_LIMIT_MESSAGE)
        updated = total_calls + batch_size
        if updated > maximum_calls:
            raise RuntimeError(TOOL_LIMIT_MESSAGE)
        return updated

    def _workspace_ack_text(
        self, tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]]
    ) -> str:
        names = {self._tool_call_name(call) for call in tool_calls}
        if "open_workspace_item" in names:
            return "好，我打开给你。"
        if names <= SILENT_WORKSPACE_TOOL_NAMES:
            return ""
        if names & {
            "write_workspace_file",
            "append_workspace_file",
            "write_workspace_project",
            "create_workspace_folder",
        }:
            return "好，我开始做，做好后告诉你。"
        return "好，我去工作区看看。"

    def _workspace_started_status(
        self, tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]]
    ) -> Dict[str, Any] | None:
        for call in tool_calls:
            tool_name = self._tool_call_name(call)
            if tool_name in WORKSPACE_TOOL_NAMES:
                return {
                    "type": "tool_call_status",
                    "tool_id": self._tool_call_id(call) or "workspace_pending",
                    "tool_name": tool_name,
                    "status": "running",
                    "content": "Workspace work accepted.",
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                    + "Z",
                }
        return None

    def _to_text_prompt(
        self, input_data: BatchInput, include_workspace_context: bool = True
    ) -> str:
        """Format input data to text prompt."""
        message_parts = []

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(
                    f"[User shared content from clipboard: {text_data.content}]"
                )

        if input_data.images:
            message_parts.append("\n[User has also provided images]")

        awareness = (
            input_data.metadata.get("workspace_awareness")
            if isinstance(input_data.metadata, dict)
            else None
        )
        if include_workspace_context and isinstance(awareness, dict):
            encoded = json.dumps(
                awareness, ensure_ascii=False, separators=(",", ":")
            )
            message_parts.append(
                "\n<LIVE_WORKSPACE_CONTEXT_UNTRUSTED_DATA>\n"
                f"{encoded[:20_000]}\n"
                "</LIVE_WORKSPACE_CONTEXT_UNTRUSTED_DATA>"
            )

        return "\n".join(message_parts).strip()

    def _to_messages(
        self, input_data: BatchInput, include_memory: bool = True
    ) -> List[Dict[str, Any]]:
        """Prepare messages for LLM API call."""
        messages = self._memory.copy() if include_memory else []
        user_content = []
        text_prompt = self._to_text_prompt(input_data)
        memory_text_prompt = self._to_text_prompt(
            input_data, include_workspace_context=False
        )
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})

        if input_data.images:
            image_added = False
            for img_data in input_data.images:
                if isinstance(img_data.data, str) and img_data.data.startswith(
                    "data:image"
                ):
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img_data.data, "detail": "auto"},
                        }
                    )
                    image_added = True
                else:
                    logger.error(
                        f"Invalid image data format: {type(img_data.data)}. Skipping image."
                    )

            if not image_added and not text_prompt:
                logger.warning(
                    "User input contains images but none could be processed."
                )

        if user_content:
            user_message = {"role": "user", "content": user_content}
            messages.append(user_message)

            skip_memory = False
            if input_data.metadata and input_data.metadata.get("skip_memory", False):
                skip_memory = True

            if not skip_memory:
                self._add_message(
                    memory_text_prompt
                    if memory_text_prompt
                    else "[User provided image(s)]",
                    "user",
                )
        else:
            logger.warning("No content generated for user message.")

        return messages

    async def _claude_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system_prompt: str,
        tool_policy: Dict[str, Any] | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
        remember_turn: bool = True,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle Claude interaction loop with tool support."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls = []
        current_assistant_message_content = []
        workspace_ack_sent = bool(
            tool_policy and tool_policy.get("workspace_fast_ack_sent") is True
        )
        tool_rounds = 0
        total_tool_calls = 0

        while True:
            tools_for_api = self._filter_tools_for_policy(
                tools, "Claude", tool_policy
            )
            active_system_prompt = self._secure_system_prompt_for_policy(
                system_prompt, tool_policy
            )
            stream = self._llm.chat_completion(
                messages, active_system_prompt, tools=tools_for_api
            )
            pending_tool_calls.clear()
            current_assistant_message_content.clear()
            current_turn_text = ""

            async for event in stream:
                if event["type"] == "text_delta":
                    text = event["text"]
                    current_turn_text += text
                    if (
                        not current_assistant_message_content
                        or current_assistant_message_content[-1]["type"] != "text"
                    ):
                        current_assistant_message_content.append(
                            {"type": "text", "text": text}
                        )
                    else:
                        current_assistant_message_content[-1]["text"] += text
                elif event["type"] == "tool_use_complete":
                    tool_call_data = event["data"]
                    logger.info(
                        f"Tool request: {tool_call_data['name']} (ID: {tool_call_data['id']})"
                    )
                    pending_tool_calls.append(tool_call_data)
                    current_assistant_message_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call_data["id"],
                            "name": tool_call_data["name"],
                            "input": tool_call_data["input"],
                        }
                    )
                # elif event["type"] == "message_delta":
                #     if event["data"]["delta"].get("stop_reason"):
                #         stop_reason = event["data"]["delta"].get("stop_reason")
                elif event["type"] == "message_stop":
                    break
                elif event["type"] == "error":
                    logger.error(f"LLM API Error: {event['message']}")
                    yield f"[Error from LLM: {event['message']}]"
                    return

            if pending_tool_calls:
                tool_rounds += 1
                if tool_rounds > max_tool_rounds:
                    yield TOOL_LIMIT_MESSAGE
                    return
                try:
                    total_tool_calls = self._consume_tool_call_budget(
                        total_tool_calls, len(pending_tool_calls), max_tool_calls
                    )
                except RuntimeError:
                    yield TOOL_LIMIT_MESSAGE
                    return
                if (
                    not workspace_ack_sent
                    and self._contains_workspace_tool(pending_tool_calls)
                    and self._needs_visible_workspace_ack(pending_tool_calls)
                ):
                    workspace_ack_sent = True
                    started_status = self._workspace_started_status(pending_tool_calls)
                    if started_status:
                        yield started_status
                    ack_text = current_turn_text.strip() or self._workspace_ack_text(
                        pending_tool_calls
                    )
                    yield ack_text
                    if remember_turn:
                        self._add_message(ack_text, "assistant")

                filtered_assistant_content = [
                    block
                    for block in current_assistant_message_content
                    if not (
                        block.get("type") == "text"
                        and not block.get("text", "").strip()
                    )
                ]

                if filtered_assistant_content:
                    messages.append(
                        {"role": "assistant", "content": filtered_assistant_content}
                    )
                    assistant_text_for_memory = "".join(
                        [
                            c["text"]
                            for c in filtered_assistant_content
                            if c["type"] == "text"
                        ]
                    ).strip()
                    if assistant_text_for_memory and remember_turn:
                        self._add_message(assistant_text_for_memory, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "Claude Tool interaction requested but ToolExecutor is not available."
                    )
                    yield "[Error: ToolExecutor not configured]"
                    return

                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="Claude",
                    tool_policy=tool_policy,
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            break
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "Tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.append({"role": "user", "content": tool_results_for_llm})
                # stop_reason = None
                continue
            else:
                if current_turn_text:
                    yield current_turn_text
                    if remember_turn:
                        self._add_message(current_turn_text, "assistant")
                return

    async def _openai_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system_prompt: str,
        tool_policy: Dict[str, Any] | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
        remember_turn: bool = True,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle OpenAI interaction with tool support."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]] = []
        current_system_prompt = system_prompt
        workspace_ack_sent = bool(
            tool_policy and tool_policy.get("workspace_fast_ack_sent") is True
        )
        tool_rounds = 0
        total_tool_calls = 0

        while True:
            if self.prompt_mode_flag:
                if self._mcp_prompt_string:
                    current_system_prompt = (
                        f"{system_prompt}\n\n{self._mcp_prompt_string}"
                    )
                    current_system_prompt = self._secure_system_prompt_for_policy(
                        current_system_prompt, tool_policy
                    )
                else:
                    logger.warning("Prompt mode active but mcp_prompt_string is empty!")
                    current_system_prompt = self._secure_system_prompt_for_policy(
                        system_prompt, tool_policy
                    )
                tools_for_api = None
            else:
                current_system_prompt = self._secure_system_prompt_for_policy(
                    system_prompt, tool_policy
                )
                tools_for_api = self._filter_tools_for_policy(
                    tools, "OpenAI", tool_policy
                )

            stream = self._llm.chat_completion(
                messages, current_system_prompt, tools=tools_for_api
            )
            pending_tool_calls.clear()
            current_turn_text = ""
            assistant_message_for_api = None
            detected_prompt_json = None
            goto_next_while_iteration = False

            async for event in stream:
                if self.prompt_mode_flag:
                    if isinstance(event, str):
                        current_turn_text += event
                        if self._json_detector:
                            potential_json = self._json_detector.process_chunk(event)
                            if potential_json:
                                try:
                                    if isinstance(potential_json, list):
                                        detected_prompt_json = potential_json
                                    elif isinstance(potential_json, dict):
                                        detected_prompt_json = [potential_json]

                                    if detected_prompt_json:
                                        break
                                except Exception as e:
                                    logger.error(f"Error parsing detected JSON: {e}")
                                    if self._json_detector:
                                        self._json_detector.reset()
                                    yield f"[Error parsing tool JSON: {e}]"
                                    goto_next_while_iteration = True
                                    break
                else:
                    if isinstance(event, str):
                        current_turn_text += event
                    elif isinstance(event, list) and all(
                        isinstance(tc, ToolCallObject) for tc in event
                    ):
                        pending_tool_calls = event
                        assistant_message_for_api = {
                            "role": "assistant",
                            "content": current_turn_text or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in pending_tool_calls
                            ],
                        }
                        break
                    elif event == "__API_NOT_SUPPORT_TOOLS__":
                        logger.warning(
                            f"LLM {getattr(self._llm, 'model', '')} has no native tool support. Switching to prompt mode."
                        )
                        self.prompt_mode_flag = True
                        if self._tool_manager:
                            self._tool_manager.disable()
                        if self._json_detector:
                            self._json_detector.reset()
                        goto_next_while_iteration = True
                        break
            if goto_next_while_iteration:
                continue

            if detected_prompt_json:
                logger.info("Processing tools detected via prompt mode JSON.")
                if remember_turn:
                    self._add_message(current_turn_text, "assistant")

                parsed_tools = self._tool_executor.process_tool_from_prompt_json(
                    detected_prompt_json
                )
                if parsed_tools:
                    tool_rounds += 1
                    if tool_rounds > max_tool_rounds:
                        yield TOOL_LIMIT_MESSAGE
                        return
                    try:
                        total_tool_calls = self._consume_tool_call_budget(
                            total_tool_calls, len(parsed_tools), max_tool_calls
                        )
                    except RuntimeError:
                        yield TOOL_LIMIT_MESSAGE
                        return
                    tool_results_for_llm = []
                    if not self._tool_executor:
                        logger.error(
                            "Prompt Tool interaction requested but ToolExecutor/MCPClient is not available."
                        )
                        yield "[Error: ToolExecutor/MCPClient not configured for prompt mode]"
                        continue

                    tool_executor_iterator = self._tool_executor.execute_tools(
                        tool_calls=parsed_tools,
                        caller_mode="Prompt",
                        tool_policy=tool_policy,
                    )
                    try:
                        while True:
                            update = await anext(tool_executor_iterator)
                            if update.get("type") == "final_tool_results":
                                tool_results_for_llm = update.get("results", [])
                                break
                            else:
                                yield update
                    except StopAsyncIteration:
                        logger.warning(
                            "Prompt mode tool executor finished without final results marker."
                        )

                    if tool_results_for_llm:
                        result_strings = [
                            res.get("content", "Error: Malformed result")
                            for res in tool_results_for_llm
                        ]
                        combined_results_str = "\n".join(result_strings)
                        messages.append(
                            {"role": "user", "content": combined_results_str}
                        )
                continue

            elif pending_tool_calls and assistant_message_for_api:
                tool_rounds += 1
                if tool_rounds > max_tool_rounds:
                    yield TOOL_LIMIT_MESSAGE
                    return
                try:
                    total_tool_calls = self._consume_tool_call_budget(
                        total_tool_calls, len(pending_tool_calls), max_tool_calls
                    )
                except RuntimeError:
                    yield TOOL_LIMIT_MESSAGE
                    return
                if (
                    not workspace_ack_sent
                    and self._contains_workspace_tool(pending_tool_calls)
                    and self._needs_visible_workspace_ack(pending_tool_calls)
                ):
                    workspace_ack_sent = True
                    started_status = self._workspace_started_status(pending_tool_calls)
                    if started_status:
                        yield started_status
                    ack_text = current_turn_text.strip() or self._workspace_ack_text(
                        pending_tool_calls
                    )
                    yield ack_text
                    if remember_turn:
                        self._add_message(ack_text, "assistant")

                messages.append(assistant_message_for_api)
                if current_turn_text and remember_turn:
                    self._add_message(current_turn_text, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "OpenAI Tool interaction requested but ToolExecutor/MCPClient is not available."
                    )
                    yield "[Error: ToolExecutor/MCPClient not configured for OpenAI mode]"
                    continue

                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="OpenAI",
                    tool_policy=tool_policy,
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            break
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "OpenAI tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.extend(tool_results_for_llm)
                continue

            else:
                if current_turn_text:
                    yield current_turn_text
                    if remember_turn:
                        self._add_message(current_turn_text, "assistant")
                return

    def _chat_function_factory(
        self,
    ) -> Callable[[BatchInput], AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]]:
        """Create the chat pipeline function."""

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def chat_with_memory(
            input_data: BatchInput,
        ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
            """Process chat with memory and tools."""
            self.reset_interrupt()
            self.prompt_mode_flag = False

            metadata = input_data.metadata if isinstance(input_data.metadata, dict) else {}
            user_text = self._user_input_text(input_data)
            provided_policy = metadata.get("workspace_tool_policy")
            tool_policy = provided_policy if isinstance(provided_policy, dict) else {
                "source": "user_turn",
                "enforce": False,
                "filter_workspace_tools": True,
                "workspace_persona": str(metadata.get("workspace_persona") or ""),
                "user_authorized_workspace_tools": workspace_user_authorized_tools(
                    user_text
                ),
            }
            messages = self._to_messages(input_data, include_memory=True)
            tools = None
            tool_mode = None
            llm_supports_native_tools = False
            remember_turn = not bool(
                input_data.metadata and input_data.metadata.get("skip_memory", False)
            )
            system_prompt = self._system
            max_tool_rounds = DEFAULT_MAX_TOOL_ROUNDS
            max_tool_calls = MAX_TOOL_CALLS_PER_TURN
            if self._use_mcpp and self._tool_manager:
                tools = None
                if isinstance(self._llm, ClaudeAsyncLLM):
                    tool_mode = "Claude"
                    tools = self._filter_tools_for_policy(
                        self._formatted_tools_claude, "Claude", tool_policy
                    )
                    llm_supports_native_tools = True
                elif isinstance(self._llm, OpenAICompatibleAsyncLLM):
                    tool_mode = "OpenAI"
                    tools = self._filter_tools_for_policy(
                        self._formatted_tools_openai, "OpenAI", tool_policy
                    )
                    llm_supports_native_tools = True
                else:
                    logger.warning(
                        f"LLM type {type(self._llm)} not explicitly handled for tool mode determination."
                    )

                if llm_supports_native_tools and not tools:
                    logger.warning(
                        f"No tools available/formatted for '{tool_mode}' mode, despite MCP being enabled."
                    )

            if (
                remember_turn
                and self._workspace_write_tools_available(tools, tool_mode)
            ):
                fast_ack = workspace_fast_ack_text(user_text)
                if fast_ack:
                    tool_policy["workspace_fast_ack_sent"] = True
                    max_tool_rounds = WORKSPACE_MAX_TOOL_ROUNDS
                    max_tool_calls = MAX_WORKSPACE_TOOL_CALLS_PER_TURN
                    yield fast_ack
                    self._add_message(fast_ack, "assistant")

            if self._use_mcpp and tool_mode == "Claude":
                logger.debug(
                    f"Starting Claude tool interaction loop with {len(tools)} tools."
                )
                try:
                    async with asyncio.timeout(TOOL_TURN_TIMEOUT_SECONDS):
                        async for output in self._claude_tool_interaction_loop(
                            messages,
                            tools if tools else [],
                            system_prompt,
                            tool_policy=tool_policy,
                            max_tool_rounds=max_tool_rounds,
                            max_tool_calls=max_tool_calls,
                            remember_turn=remember_turn,
                        ):
                            yield output
                except TimeoutError:
                    logger.warning("Claude tool turn reached its time limit")
                    yield TOOL_LIMIT_MESSAGE
                return
            elif self._use_mcpp and tool_mode == "OpenAI":
                logger.debug(
                    f"Starting OpenAI tool interaction loop with {len(tools)} tools."
                )
                try:
                    async with asyncio.timeout(TOOL_TURN_TIMEOUT_SECONDS):
                        async for output in self._openai_tool_interaction_loop(
                            messages,
                            tools if tools else [],
                            system_prompt,
                            tool_policy=tool_policy,
                            max_tool_rounds=max_tool_rounds,
                            max_tool_calls=max_tool_calls,
                            remember_turn=remember_turn,
                        ):
                            yield output
                except TimeoutError:
                    logger.warning("OpenAI tool turn reached its time limit")
                    yield TOOL_LIMIT_MESSAGE
                return
            else:
                logger.info("Starting simple chat completion.")
                token_stream = self._llm.chat_completion(
                    messages,
                    self._secure_system_prompt_for_policy(
                        system_prompt, tool_policy
                    ),
                )
                complete_response = ""
                async for event in token_stream:
                    text_chunk = ""
                    if isinstance(event, dict) and event.get("type") == "text_delta":
                        text_chunk = event.get("text", "")
                    elif isinstance(event, str):
                        text_chunk = event
                    else:
                        continue
                    if text_chunk:
                        yield text_chunk
                        complete_response += text_chunk
                if complete_response and remember_turn:
                    self._add_message(complete_response, "assistant")

        return chat_with_memory

    async def chat(
        self,
        input_data: BatchInput,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """Run chat pipeline."""
        chat_func_decorated = self._chat_function_factory()
        async for output in chat_func_decorated(input_data):
            yield output

    def reset_interrupt(self) -> None:
        """Reset interrupt flag."""
        self._interrupt_handled = False
