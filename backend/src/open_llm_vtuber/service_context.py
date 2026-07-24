import asyncio
import os
import json
from typing import Callable, TYPE_CHECKING
from loguru import logger
from fastapi import WebSocket

from prompts import prompt_loader
from .live2d_model import Live2dModel
from .asr.asr_interface import ASRInterface
from .tts.tts_interface import TTSInterface
from .vad.vad_interface import VADInterface
from .agent.agents.agent_interface import AgentInterface
from .translate.translate_interface import TranslateInterface

from .mcpp.server_registry import ServerRegistry
from .mcpp.tool_manager import ToolManager

from .asr.asr_factory import ASRFactory
from .tts.tts_factory import TTSFactory
from .vad.vad_factory import VADFactory
from .agent.agent_factory import AgentFactory
from .translate.translate_factory import TranslateFactory

if TYPE_CHECKING:
    from .mcpp.mcp_client import MCPClient
    from .mcpp.tool_executor import ToolExecutor
    from .mcpp.tool_adapter import ToolAdapter
    from .tts.omnivoice_clone_tts import TTSEngine as OmniVoiceCloneTTSEngine

from .config_manager import (
    Config,
    AgentConfig,
    CharacterConfig,
    SystemConfig,
    ASRConfig,
    TTSConfig,
    VADConfig,
    TranslatorConfig,
    read_yaml,
    validate_config,
)
from .config_manager.stateless_llm import DeepseekConfig
from .chat_history_manager import SINGLE_HISTORY_UID, get_core_memory_prompt


class ServiceContext:
    """Initializes, stores, and updates the asr, tts, and llm instances and other
    configurations for a connected client."""

    def __init__(self):
        self.config: Config = None
        self.system_config: SystemConfig = None
        self.character_config: CharacterConfig = None

        self.live2d_model: Live2dModel = None
        self.asr_engine: ASRInterface = None
        self.tts_engine: TTSInterface = None
        self.voice_clone_tts: "OmniVoiceCloneTTSEngine | None" = None
        self.agent_engine: AgentInterface = None
        # translate_engine can be none if translation is disabled
        self.vad_engine: VADInterface | None = None
        self.translate_engine: TranslateInterface | None = None

        self.mcp_server_registery: ServerRegistry | None = None
        self.tool_adapter: "ToolAdapter | None" = None
        self.tool_manager: ToolManager | None = None
        self.mcp_client: "MCPClient | None" = None
        self.tool_executor: "ToolExecutor | None" = None

        # the system prompt is a combination of the persona prompt and live2d expression prompt
        self.system_prompt: str = None

        # Store the generated MCP prompt string (if MCP enabled)
        self.mcp_prompt: str = ""

        self.history_uid: str = ""  # Add history_uid field
        self.send_text: Callable = None
        self.client_uid: str = None

    def _load_short_memory_into_agent(self) -> None:
        if not (
            self.agent_engine
            and self.character_config
            and hasattr(self.agent_engine, "set_memory_from_history")
        ):
            return

        self.history_uid = self.history_uid or SINGLE_HISTORY_UID
        try:
            self.agent_engine.set_memory_from_history(
                conf_uid=self.character_config.conf_uid,
                history_uid=self.history_uid,
            )
        except Exception as exc:
            logger.warning(f"Failed to load short memory into agent: {exc}")

    def __str__(self):
        return (
            f"ServiceContext:\n"
            f"  System Config: {'Loaded' if self.system_config else 'Not Loaded'}\n"
            f"    Details: {json.dumps(self.system_config.model_dump(), indent=6) if self.system_config else 'None'}\n"
            f"  Live2D Model: {self.live2d_model.model_info if self.live2d_model else 'Not Loaded'}\n"
            f"  ASR Engine: {type(self.asr_engine).__name__ if self.asr_engine else 'Not Loaded'}\n"
            f"    Config: {json.dumps(self.character_config.asr_config.model_dump(), indent=6) if self.character_config.asr_config else 'None'}\n"
            f"  TTS Engine: {type(self.tts_engine).__name__ if self.tts_engine else 'Not Loaded'}\n"
            f"    Config: {json.dumps(self.character_config.tts_config.model_dump(), indent=6) if self.character_config.tts_config else 'None'}\n"
            f"  LLM Engine: {type(self.agent_engine).__name__ if self.agent_engine else 'Not Loaded'}\n"
            f"    Agent Config: {json.dumps(self.character_config.agent_config.model_dump(), indent=6) if self.character_config.agent_config else 'None'}\n"
            f"  VAD Engine: {type(self.vad_engine).__name__ if self.vad_engine else 'Not Loaded'}\n"
            f"    Agent Config: {json.dumps(self.character_config.vad_config.model_dump(), indent=6) if self.character_config.vad_config else 'None'}\n"
            f"  System Prompt: {self.system_prompt or 'Not Set'}\n"
            f"  MCP Enabled: {'Yes' if self.mcp_client else 'No'}"
        )

    def _enable_workspace_mcp(self, config: Config) -> None:
        """Enable the local workspace tool for every persona without editing each YAML profile."""
        basic_agent = (
            config.character_config.agent_config.agent_settings.basic_memory_agent
            if config and config.character_config and config.character_config.agent_config
            else None
        )
        if not basic_agent:
            return

        basic_agent.use_mcpp = True
        enabled_servers = list(basic_agent.mcp_enabled_servers or [])
        if "workspace" not in enabled_servers:
            enabled_servers.append("workspace")
        basic_agent.mcp_enabled_servers = enabled_servers

    # ==== Initializers

    async def _init_mcp_components(self, use_mcpp, enabled_servers):
        """Initializes MCP components based on configuration, dynamically fetching tool info."""
        logger.debug(
            f"Initializing MCP components: use_mcpp={use_mcpp}, enabled_servers={enabled_servers}"
        )

        # Reset MCP components first
        self.mcp_server_registery = None
        self.tool_manager = None
        self.mcp_client = None
        self.tool_executor = None
        self.json_detector = None
        self.mcp_prompt = ""

        if use_mcpp and enabled_servers:
            from .mcpp.mcp_client import MCPClient
            from .mcpp.tool_executor import ToolExecutor

            # 1. Initialize ServerRegistry
            self.mcp_server_registery = ServerRegistry()
            logger.info("ServerRegistry initialized or referenced.")

            # 2. Use ToolAdapter to get the MCP prompt and tools
            if not self.tool_adapter:
                logger.error(
                    "ToolAdapter not initialized before calling _init_mcp_components."
                )
                self.mcp_prompt = "[Error: ToolAdapter not initialized]"
                return  # Exit if ToolAdapter is mandatory and not initialized

            try:
                (
                    mcp_prompt_string,
                    openai_tools,
                    claude_tools,
                ) = await self.tool_adapter.get_tools(enabled_servers)
                # Store the generated prompt string
                self.mcp_prompt = mcp_prompt_string
                logger.info(
                    f"Dynamically generated MCP prompt string (length: {len(self.mcp_prompt)})."
                )
                logger.info(
                    f"Dynamically formatted tools - OpenAI: {len(openai_tools)}, Claude: {len(claude_tools)}."
                )

                # 3. Initialize ToolManager with the fetched formatted tools

                _, raw_tools_dict = await self.tool_adapter.get_server_and_tool_info(
                    enabled_servers
                )
                self.tool_manager = ToolManager(
                    formatted_tools_openai=openai_tools,
                    formatted_tools_claude=claude_tools,
                    initial_tools_dict=raw_tools_dict,
                )
                logger.info("ToolManager initialized with dynamically fetched tools.")

            except Exception as e:
                logger.error(
                    f"Failed during dynamic MCP tool construction: {e}", exc_info=True
                )
                # Ensure dependent components are not created if construction fails
                self.tool_manager = None
                self.mcp_prompt = "[Error constructing MCP tools/prompt]"

            # 4. Initialize MCPClient
            if self.mcp_server_registery:
                self.mcp_client = MCPClient(
                    self.mcp_server_registery, self.send_text, self.client_uid
                )
                logger.info("MCPClient initialized for this session.")
            else:
                logger.error(
                    "MCP enabled but ServerRegistry not available. MCPClient not created."
                )
                self.mcp_client = None  # Ensure it's None

            # 5. Initialize ToolExecutor
            if self.mcp_client and self.tool_manager:
                self.tool_executor = ToolExecutor(self.mcp_client, self.tool_manager)
                logger.info("ToolExecutor initialized for this session.")
            else:
                logger.warning(
                    "MCPClient or ToolManager not available. ToolExecutor not created."
                )
                self.tool_executor = None  # Ensure it's None

            logger.info("StreamJSONDetector initialized for this session.")

        elif use_mcpp and not enabled_servers:
            logger.warning(
                "use_mcpp is True, but mcp_enabled_servers list is empty. MCP components not initialized."
            )
        else:
            logger.debug(
                "MCP components not initialized (use_mcpp is False or no enabled servers)."
            )

    async def close(self):
        """Clean up resources, especially the MCPClient."""
        logger.info("Closing ServiceContext resources...")
        if self.mcp_client:
            logger.info(f"Closing MCPClient for context instance {id(self)}...")
            await self.mcp_client.aclose()
            self.mcp_client = None
        if self.agent_engine and hasattr(self.agent_engine, "close"):
            await self.agent_engine.close()  # Ensure agent resources are also closed
        logger.info("ServiceContext closed.")

    async def load_cache(
        self,
        config: Config,
        system_config: SystemConfig,
        character_config: CharacterConfig,
        live2d_model: Live2dModel,
        asr_engine: ASRInterface,
        tts_engine: TTSInterface,
        vad_engine: VADInterface,
        agent_engine: AgentInterface,
        translate_engine: TranslateInterface | None,
        mcp_server_registery: ServerRegistry | None = None,
        tool_adapter: "ToolAdapter | None" = None,
        send_text: Callable = None,
        client_uid: str = None,
    ) -> None:
        """
        Load the ServiceContext with the reference of the provided instances.
        Pass by reference so no reinitialization will be done.
        """
        if not character_config:
            raise ValueError("character_config cannot be None")
        if not system_config:
            raise ValueError("system_config cannot be None")

        self._enable_workspace_mcp(config)
        self.config = config
        self.system_config = system_config
        self.character_config = character_config
        self.live2d_model = live2d_model
        self.asr_engine = asr_engine
        self.tts_engine = tts_engine
        self.vad_engine = vad_engine
        self.agent_engine = agent_engine
        self.translate_engine = translate_engine
        # Load potentially shared components by reference
        self.mcp_server_registery = mcp_server_registery
        self.tool_adapter = tool_adapter
        self.send_text = send_text
        self.client_uid = client_uid

        # Initialize session-specific MCP components
        await self._init_mcp_components(
            self.character_config.agent_config.agent_settings.basic_memory_agent.use_mcpp,
            self.character_config.agent_config.agent_settings.basic_memory_agent.mcp_enabled_servers,
        )
        self._load_short_memory_into_agent()

        logger.debug(f"Loaded service context with cache: {character_config}")

    async def load_from_config(self, config: Config) -> None:
        """
        Load the ServiceContext with the config.
        Reinitialize the instances if the config is different.

        Parameters:
        - config (Dict): The configuration dictionary.
        """
        self._enable_workspace_mcp(config)

        if not self.config:
            self.config = config

        if not self.system_config:
            self.system_config = config.system_config

        if not self.character_config:
            self.character_config = config.character_config

        # update all sub-configs

        # init live2d from character config
        self.init_live2d(config.character_config.live2d_model_name)

        # init asr from character config
        self.init_asr(config.character_config.asr_config)

        # init tts from character config
        self.init_tts(config.character_config.tts_config)

        # init vad from character config
        self.init_vad(config.character_config.vad_config)

        # Initialize shared ToolAdapter if it doesn't exist yet
        if (
            not self.tool_adapter
            and config.character_config.agent_config.agent_settings.basic_memory_agent.use_mcpp
        ):
            from .mcpp.tool_adapter import ToolAdapter

            if not self.mcp_server_registery:
                logger.info(
                    "Initializing shared ServerRegistry within load_from_config."
                )
                self.mcp_server_registery = ServerRegistry()
            logger.info("Initializing shared ToolAdapter within load_from_config.")
            self.tool_adapter = ToolAdapter(server_registery=self.mcp_server_registery)

        # Initialize MCP Components before initializing Agent
        await self._init_mcp_components(
            config.character_config.agent_config.agent_settings.basic_memory_agent.use_mcpp,
            config.character_config.agent_config.agent_settings.basic_memory_agent.mcp_enabled_servers,
        )

        # init agent from character config
        await self.init_agent(
            config.character_config.agent_config,
            config.character_config.persona_prompt,
        )
        self._load_short_memory_into_agent()

        self.init_translate(
            config.character_config.tts_preprocessor_config.translator_config
        )

        # store typed config references
        self.config = config
        self.system_config = config.system_config or self.system_config
        self.character_config = config.character_config

    def init_live2d(self, live2d_model_name: str) -> None:
        logger.info(f"Initializing Live2D: {live2d_model_name}")
        try:
            self.live2d_model = Live2dModel(live2d_model_name)
            self.character_config.live2d_model_name = live2d_model_name
        except Exception as e:
            logger.critical(f"Error initializing Live2D: {e}")
            logger.critical("Try to proceed without Live2D...")

    def init_asr(self, asr_config: ASRConfig) -> None:
        if not self.asr_engine or (self.character_config.asr_config != asr_config):
            logger.info(f"Initializing ASR: {asr_config.asr_model}")
            self.asr_engine = ASRFactory.get_asr_system(
                asr_config.asr_model,
                **getattr(asr_config, asr_config.asr_model).model_dump(),
            )
            # saving config should be done after successful initialization
            self.character_config.asr_config = asr_config
        else:
            logger.info("ASR already initialized with the same config.")

    def init_tts(self, tts_config: TTSConfig) -> None:
        if not self.tts_engine or (self.character_config.tts_config != tts_config):
            logger.info(f"Initializing TTS: {tts_config.tts_model}")
            self.tts_engine = TTSFactory.get_tts_engine(
                tts_config.tts_model,
                **getattr(tts_config, tts_config.tts_model.lower()).model_dump(),
            )
            # saving config should be done after successful initialization
            self.character_config.tts_config = tts_config
        else:
            logger.info("TTS already initialized with the same config.")

    def init_voice_clone_tts(self) -> None:
        if self.voice_clone_tts is None:
            from .tts.omnivoice_clone_tts import TTSEngine as OmniVoiceCloneTTSEngine

            self.voice_clone_tts = OmniVoiceCloneTTSEngine()

    def get_current_tts_engine(self) -> TTSInterface:
        if self.voice_clone_tts and self.voice_clone_tts.is_ready():
            return self.voice_clone_tts
        return self.tts_engine

    async def apply_client_voice_clone_config(
        self,
        enabled: bool,
        ref_audio_path: str = "",
        ref_text: str | None = None,
        language: str | None = None,
    ) -> None:
        """Apply per-client OmniVoice clone settings without changing normal TTS config."""
        self.init_voice_clone_tts()
        self.voice_clone_tts.configure(
            enabled=enabled,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            language=language,
        )
        if enabled:
            await asyncio.to_thread(self.voice_clone_tts.prepare_model_cache)
        logger.info(
            f"Voice clone {'enabled' if enabled else 'disabled'} for {self.client_uid}; "
            f"reference={ref_audio_path or 'none'}"
        )

    def init_vad(self, vad_config: VADConfig) -> None:
        if vad_config.vad_model is None:
            logger.info("VAD is disabled.")
            self.vad_engine = None
            return

        if not self.vad_engine or (self.character_config.vad_config != vad_config):
            logger.info(f"Initializing VAD: {vad_config.vad_model}")
            self.vad_engine = VADFactory.get_vad_engine(
                vad_config.vad_model,
                **getattr(vad_config, vad_config.vad_model.lower()).model_dump(),
            )
            # saving config should be done after successful initialization
            self.character_config.vad_config = vad_config
        else:
            logger.info("VAD already initialized with the same config.")

    async def init_agent(self, agent_config: AgentConfig, persona_prompt: str) -> None:
        """Initialize or update the LLM engine based on agent configuration."""
        logger.info(f"Initializing Agent: {agent_config.conversation_agent_choice}")

        if (
            self.agent_engine is not None
            and agent_config == self.character_config.agent_config
            and persona_prompt == self.character_config.persona_prompt
        ):
            logger.debug("Agent already initialized with the same config.")
            return

        system_prompt = await self.construct_system_prompt(persona_prompt)

        try:
            self.agent_engine = AgentFactory.create_agent(
                conversation_agent_choice=agent_config.conversation_agent_choice,
                agent_settings=agent_config.agent_settings.model_dump(),
                llm_configs=agent_config.llm_configs.model_dump(),
                system_prompt=system_prompt,
                live2d_model=self.live2d_model,
                tts_preprocessor_config=self.character_config.tts_preprocessor_config,
                system_config=self.system_config.model_dump(),
                tool_manager=self.tool_manager,
                tool_executor=self.tool_executor,
                mcp_prompt_string=self.mcp_prompt,
            )

            logger.debug(f"Agent choice: {agent_config.conversation_agent_choice}")
            logger.debug(f"System prompt: {system_prompt}")

            # Save the current configuration
            self.character_config.agent_config = agent_config
            self.system_prompt = system_prompt

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise

    async def apply_client_api_config(
        self,
        base_url: str,
        api_key: str,
        model: str,
    ) -> None:
        """Apply a per-client OpenAI-compatible API config without writing it to disk."""
        if not self.character_config or not self.character_config.agent_config:
            raise ValueError("Character agent config is not loaded")

        if not base_url or not api_key or not model:
            raise ValueError("base_url, api_key and model are required")

        agent_config = self.character_config.agent_config
        basic_memory_config = agent_config.agent_settings.basic_memory_agent
        if basic_memory_config is None:
            raise ValueError("client API config currently requires basic_memory_agent")

        logger.info(
            f"Applying client API config for {self.client_uid}: base_url={base_url}, model={model}"
        )

        basic_memory_config.llm_provider = "deepseek_llm"
        agent_config.llm_configs.deepseek_llm = DeepseekConfig(
            base_url=base_url,
            llm_api_key=api_key,
            model=model,
            temperature=0.7,
            interrupt_method="user",
        )

        if self.agent_engine and hasattr(self.agent_engine, "close"):
            await self.agent_engine.close()
        self.agent_engine = None
        await self.init_agent(agent_config, self.character_config.persona_prompt)
        self._load_short_memory_into_agent()

    def init_translate(self, translator_config: TranslatorConfig) -> None:
        """Initialize or update the translation engine based on the configuration."""

        if not translator_config.translate_audio:
            logger.debug("Translation is disabled.")
            return

        if (
            not self.translate_engine
            or self.character_config.tts_preprocessor_config.translator_config
            != translator_config
        ):
            logger.info(
                f"Initializing Translator: {translator_config.translate_provider}"
            )
            self.translate_engine = TranslateFactory.get_translator(
                translator_config.translate_provider,
                getattr(
                    translator_config, translator_config.translate_provider
                ).model_dump(),
            )
            self.character_config.tts_preprocessor_config.translator_config = (
                translator_config
            )
        else:
            logger.info("Translation already initialized with the same config.")

    # ==== utils

    async def construct_system_prompt(self, persona_prompt: str) -> str:
        """
        Append tool prompts to persona prompt.

        Parameters:
        - persona_prompt (str): The persona prompt.

        Returns:
        - str: The system prompt with all tool prompts appended.
        """
        logger.debug(f"constructing persona_prompt: '''{persona_prompt}'''")

        if self.character_config and self.character_config.conf_uid:
            core_memory_prompt = get_core_memory_prompt(self.character_config.conf_uid)
            if core_memory_prompt:
                persona_prompt += f"\n\n{core_memory_prompt}\n"

        persona_prompt += (
            "\n\n# 记忆续接规则\n"
            "- 每次重新进入角色时，系统会读取核心记忆和最近短期对话。\n"
            "- 最近短期对话是上一次聊天上下文；用户继续追问、说“刚才”、"
            "“继续”、或直接接话时，要自然接上最后一轮聊天，而不是当成全新对话。\n"
        )

        for prompt_name, prompt_file in self.system_config.tool_prompts.items():
            if (
                prompt_name == "group_conversation_prompt"
                or prompt_name == "proactive_speak_prompt"
            ):
                continue

            prompt_content = prompt_loader.load_util(prompt_file)

            if prompt_name == "live2d_expression_prompt":
                prompt_content = prompt_content.replace(
                    "[<insert_emomap_keys>]", self.live2d_model.emo_str
                )

            if prompt_name == "mcp_prompt":
                continue

            persona_prompt += prompt_content

        character_name = (
            self.character_config.character_name
            or self.character_config.conf_name
            or "default"
        )
        persona_prompt += f"""

Workspace file rules:
- When the user asks you to create, save, record, write, draw, generate a file, make an SVG, keep a diary, create study notes, or build a small code project, use the workspace MCP tools instead of only replying in chat.
- When the user asks for a reminder, such as "remind me in 10 minutes" or "提醒我十分钟后喝水", use schedule_reminder. Use delay_minutes for relative times and due_at for exact times. Reminder time is based on the user's device/local machine time.
- Always use persona="{character_name}" when calling workspace tools.
- Files must be created under workspace/{character_name}/. Create a fitting folder first, such as diary, drawings, study, notes, mini-apps, reminders, or a user-requested folder.
- Do not create a folder named "{character_name}" inside workspace/{character_name}/. The persona argument already selects that root folder.
- Never read or write another persona's workspace.
- For games, mini apps, web pages, or code projects, create a branch folder under mini-apps or another fitting folder and prefer write_workspace_project with separate files such as index.html, style.css, and main.js.
- When the user asks you to control or play an open workspace HTML game or mini app yourself, use send_workspace_key with keys such as ArrowLeft, ArrowRight, ArrowUp, ArrowDown, Space, Enter, w, a, s, or d. The open workspace page will receive these as keyboard input.
- For turn-based games, board games, chess-like games, puzzles, and apps where you should truly participate, design the app around the workspace control protocol instead of building a fake built-in AI opponent. The page should expose window.MeloMateGameState or a window.MeloMateGameState() function with JSON state such as board, currentTurn, players, legalMoves, score, winner, and gameOver, and should handle window.MeloMateGameAction(action, payload) or the melomate-action event for semantic actions such as place-piece, select-cell, move, pass, or restart.
- When playing such an app yourself, first use read_workspace_state, decide your move from the reported state, then use send_workspace_action for semantic actions. Use send_workspace_key only for real-time keyboard games that do not expose semantic actions.
- If any generated file is long, use append_workspace_file in small chunks instead of putting a whole long file into one tool call.
- Keep each tool call argument compact and valid JSON. Do not put a large complete HTML/CSS/JS app into one write_workspace_file call.
- After a successful file write, reply briefly without mentioning the exact file name unless the user asks.
"""

        persona_prompt += f"""

General workspace judgment rules:
- The workspace is this persona's private working area, not a fixed feature list.
- Do not limit workspace use to diary, drawings, study notes, or mini apps. Those are examples only.
- When the user's request can produce a reusable artifact, record, file, plan, draft, code project, list, dataset, configuration, reminder, note, creative work, or other durable output, decide whether it should be created or updated in the workspace.
- Choose a suitable folder and file type based on the user's intent. Examples include writing, recipes, travel, fitness, music, lists, reviews, budget, data, prompts, configs, plans, logs, and user-requested folders.
- If the user explicitly asks to save, remember in a file, create, generate, draw, write down, make a plan, build, export, or remind, use a workspace tool before replying normally.
- For workspace tasks, it is okay to acknowledge briefly first, then use the required workspace tools and continue working. Keep the first acknowledgement short.
- For games, mini apps, web pages, and code projects, prefer write_workspace_project. Split larger work into multiple files and use append_workspace_file for long files so the tool arguments do not become too large or invalid.
- For open workspace HTML games, you can play by sending keyboard input through send_workspace_key when the user asks you to join, control, play, test, or take a turn.
- For board games or turn-based apps, play as yourself through read_workspace_state plus send_workspace_action. Do not tell the user they are playing against the computer unless the user explicitly asks for a computer opponent.
- If the request is casual chat, emotional support, flirting, or a one-off answer with no durable output, reply normally without writing a file.
- If a durable output would be useful but the user did not ask to save it, ask briefly before saving unless the intent is obvious.
- Always use persona="{character_name}" and never read or write another persona's workspace.
- If you complete workspace work, tell the user briefly that it is ready in the relevant workspace branch folder. Do not mention the exact file name unless the user asks.
- After completing workspace work, ask whether the user wants to see it, open it, play it, or try it now, depending on the artifact type.
- If the user says they want to see, open, play, or try a generated workspace item, call open_workspace_item with the relevant workspace path before replying normally.
"""

        logger.debug("\n === System Prompt ===")
        logger.debug(persona_prompt)

        return persona_prompt

    async def handle_config_switch(
        self,
        websocket: WebSocket,
        config_file_name: str,
    ) -> None:
        """
        Handle the configuration switch request.
        Change the configuration to a new config and notify the client.

        Parameters:
        - websocket (WebSocket): The WebSocket connection.
        - config_file_name (str): The name of the configuration file.
        """
        try:
            new_character_config_data = None

            characters_dir = os.path.normpath(self.system_config.config_alts_dir)
            if config_file_name == "conf.yaml":
                config_file_name = "小可.yaml"
            elif config_file_name in {"xyu.yaml", "xyua.yaml"}:
                config_file_name = "小可.yaml"

            file_path = os.path.normpath(os.path.join(characters_dir, config_file_name))
            if os.path.commonpath([characters_dir, file_path]) != characters_dir:
                raise ValueError("Invalid configuration file path")

            alt_config_data = read_yaml(file_path).get("character_config")

            # Start with original config data and perform a deep merge
            new_character_config_data = deep_merge(
                self.config.character_config.model_dump(), alt_config_data
            )

            if new_character_config_data:
                new_config = {
                    "system_config": self.system_config.model_dump(),
                    "character_config": new_character_config_data,
                }
                new_config = validate_config(new_config)
                await self.load_from_config(new_config)  # Await the async load
                logger.debug(f"New config: {self}")
                logger.debug(
                    f"New character config: {self.character_config.model_dump()}"
                )

                # Send responses to client
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "set-model-and-conf",
                            "model_info": self.live2d_model.model_info,
                            "conf_name": self.character_config.conf_name,
                            "character_name": self.character_config.character_name,
                            "conf_uid": self.character_config.conf_uid,
                        }
                    )
                )

                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "config-switched",
                            "message": f"Switched to config: {config_file_name}",
                        }
                    )
                )

                logger.info(f"Configuration switched to {config_file_name}")
            else:
                raise ValueError(
                    f"Failed to load configuration from {config_file_name}"
                )

        except Exception as e:
            logger.error(f"Error switching configuration: {e}")
            logger.debug(self)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error switching configuration: {str(e)}",
                    }
                )
            )
            raise e


def deep_merge(dict1, dict2):
    """
    Recursively merges dict2 into dict1, prioritizing values from dict2.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
