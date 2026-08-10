import { VrmAvatar } from "./vrm-avatar";
import "./styles.css";

type LineRole = "user" | "assistant" | "system";

type SavedSettings = {
  micId: string;
  speakerId: string;
  characterConfigFile?: string;
  volume: string;
  endpoint: string;
  model: string;
  apiKey: string;
  backgroundUrl?: string;
  vrmModelId?: string;
  voiceChatOutputEnabled?: boolean;
  voiceChatOutputDeviceId?: string;
  voiceCloneEnabled?: boolean;
  screenVisionEnabled?: boolean;
  screenVisionEndpoint?: string;
  screenVisionModel?: string;
  screenVisionApiKey?: string;
  screenVisionIntervalSec?: string;
  proactiveSpeakEnabled?: boolean;
  proactiveIdleSeconds?: string;
};

type MicVadInstance = {
  start: () => void | Promise<void>;
  pause: () => void;
  destroy: () => void;
};

type VadModule = {
  MicVAD: {
    new: (options: Record<string, unknown>) => Promise<MicVadInstance>;
  };
};

type DisplayText = {
  text?: string;
  name?: string;
};

type ProactiveSpeakStage =
  | "opening"
  | "curious"
  | "warm-concern"
  | "playful-impatience"
  | "fresh-topic";

type ProactiveReturnContext = {
  elapsed_seconds: number;
  unanswered_count: number;
  last_proactive_seconds_ago: number;
};

type WsMessage = {
  type?: string;
  text?: string;
  texts?: string[];
  input_id?: string;
  turn_id?: string;
  request_id?: string;
  message?: string;
  reason?: string;
  status?: string;
  page_id?: string;
  state_version?: number;
  action?: string;
  audio?: string | null;
  volumes?: number[];
  slice_length?: number;
  display_text?: DisplayText;
  actions?: {
    expressions?: Array<number | string>;
  };
  conf_name?: string;
  character_name?: string;
  client_uid?: string;
  success?: boolean;
  enabled?: boolean;
  available?: boolean;
  chat_api_key_saved?: boolean;
  screen_vision_api_key_saved?: boolean;
  chat_config_applied?: boolean;
  cleared?: "chat" | "screen_vision";
  capabilities?: {
    voice_clone?: boolean;
    voice_clone_missing?: string[];
  };
  configs?: CharacterConfigOption[];
};

type BackgroundOption = {
  name: string;
  url: string;
};

type WorkspaceEntry = {
  name: string;
  path: string;
  type: "directory" | "file";
};

type WorkspaceEvent = {
  id: string;
  type: "workspace-state-changed" | "workspace-page-closed";
  created_ms: number;
  state_version?: number;
  persona?: string;
  page?: { id?: string; title?: string; path?: string; closed?: boolean };
  appState?: unknown;
  lastAction?: { id?: string; action?: string; accepted?: boolean } | null;
  actionEvent?: boolean;
  summary?: string;
};

type WorkspaceControlStatus = {
  label: string;
  tone: "ready" | "stale" | "missing";
};

type CharacterConfigOption = {
  filename: string;
  name?: string;
  conf_name?: string;
  character_name?: string;
};

type VrmModelOption = {
  id: string;
  name: string;
  fileName: string;
  url: string;
  size: number;
};

type AssetPanelTab = "background" | "character" | "workspace";

declare global {
  interface Window {
    vad?: VadModule;
    __MELOMATE_RUNTIME_CONFIG__?: {
      backendWsUrl?: string;
      sessionToken?: string;
      workspaceBaseUrl?: string;
    };
  }
}

const settingsStorageKey = "melomate-settings";
const credentialProfileStorageKey = "melomate-credential-profile";
const maxVoiceCloneReferenceBytes = 10 * 1024 * 1024;
const allowedVoiceCloneReferenceExtensions = new Set([".wav", ".mp3", ".flac", ".ogg"]);
const backgroundManifestUrl = "/api/backgrounds";
const vrmModelManifestUrl = "/api/vrm-models";
const workspaceManifestUrl = "/api/workspace";
const workspaceStateUrl = "/api/workspace-state";
const workspaceEventsUrl = "/api/workspace-events";
const fallbackBackgrounds: BackgroundOption[] = [{ name: "Default", url: "/backgrounds/default.svg" }];
const defaultCharacterConfigFile = "小可.yaml";
const defaultCharacterOption: CharacterConfigOption = { filename: defaultCharacterConfigFile };
let vrmModelOptions: VrmModelOption[] = [];
const referenceAudioDbName = "melomate-reference-audio";
const moonshotApiEndpoint = "https://api.moonshot.cn/v1";
const defaultApiEndpoint = "https://api.deepseek.com";
const defaultModel = "deepseek-chat";
const defaultScreenVisionEndpoint = moonshotApiEndpoint;
const defaultScreenVisionModel = "moonshot-v1-8k-vision-preview";
const appSessionToken = window.__MELOMATE_RUNTIME_CONFIG__?.sessionToken?.trim() || "";
const workspaceBaseUrl =
  window.__MELOMATE_RUNTIME_CONFIG__?.workspaceBaseUrl?.trim().replace(/\/$/, "") || "http://127.0.0.1:5179";

function authenticatedFetch(input: RequestInfo | URL, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  if (appSessionToken) headers.set("X-MeloMate-Session", appSessionToken);
  return fetch(input, { ...init, headers });
}

function backendWebSocketUrl() {
  const fallback = "ws://127.0.0.1:12393/client-ws";
  const configured = window.__MELOMATE_RUNTIME_CONFIG__?.backendWsUrl?.trim();
  if (!configured) return fallback;
  try {
    const url = new URL(configured);
    return url.protocol === "ws:" || url.protocol === "wss:" ? url.toString() : fallback;
  } catch {
    return fallback;
  }
}

const openLlmWsUrl = backendWebSocketUrl();
const backendWebSocketProtocols = appSessionToken ? [`melomate.session.${appSessionToken}`] : [];
const websocketReconnectDelays = [400, 800, 1200, 2000, 3000];
const vadChunkSize = 4096;
const vadSampleRate = 16000;
const shortSpeechTargetSamples = Math.round(vadSampleRate * 0.9);
const shortSpeechNormalizePeak = 0.45;
const speechPeakGate = 0.025;
const speechRmsGate = 0.008;
const screenVisionMaxWidth = 1024;
const screenVisionJpegQuality = 0.85;
const defaultProactiveIdleSeconds = "120";
const proactiveSpeakCheckIntervalMs = 1_000;
const workspaceEventLongPollMs = 15_000;
const preferredVoiceChatOutputDevicePattern = /^voicemeeter\s+input\b/i;
const voiceChatOutputDevicePattern = /voicemeeter\s+(input|in\s*\d+|aux\s+input|vaio3\s+input)|vb-audio\s+voicemeeter\s+vaio/i;
const voiceChatMicDevicePattern = /voicemeeter\s+out\s*b2|out\s*b2.*voicemeeter|voicemeeter.*b2/i;
const physicalMicDevicePattern = /麦克风.*3-\s*usb|3-\s*usb\s+audio\s+device|usb\s+audio\s+device/i;

const transcriptLog = document.querySelector<HTMLDivElement>("#transcriptLog")!;
const micSelect = document.querySelector<HTMLSelectElement>("#micSelect")!;
const speakerSelect = document.querySelector<HTMLSelectElement>("#speakerSelect")!;
const characterSelect = document.querySelector<HTMLSelectElement>("#characterSelect")!;
const voiceChatOutputToggle = document.querySelector<HTMLInputElement>("#voiceChatOutputToggle")!;
const voiceChatOutputSelect = document.querySelector<HTMLSelectElement>("#voiceChatOutputSelect")!;
const testVoiceChatOutput = document.querySelector<HTMLButtonElement>("#testVoiceChatOutput")!;
const showVoicemeeter = document.querySelector<HTMLButtonElement>("#showVoicemeeter")!;
const voiceChatOutputHint = document.querySelector<HTMLParagraphElement>("#voiceChatOutputHint")!;
const volumeMuteToggle = document.querySelector<HTMLButtonElement>("#volumeMuteToggle")!;
const volumeRange = document.querySelector<HTMLInputElement>("#volumeRange")!;
const volumeNumber = volumeRange;
const endpointInput = document.querySelector<HTMLInputElement>("#endpoint")!;
const modelInput = document.querySelector<HTMLInputElement>("#model")!;
const apiKeyInput = document.querySelector<HTMLInputElement>("#apiKey")!;
const toggleApiKey = document.querySelector<HTMLButtonElement>("#toggleApiKey")!;
const clearApiKey = document.querySelector<HTMLButtonElement>("#clearApiKey")!;
const apiKeyHint = document.querySelector<HTMLParagraphElement>("#apiKeyHint")!;
const screenVisionToggle = document.querySelector<HTMLInputElement>("#screenVisionToggle")!;
const screenVisionEndpointInput = document.querySelector<HTMLInputElement>("#screenVisionEndpoint")!;
const screenVisionModelInput = document.querySelector<HTMLInputElement>("#screenVisionModel")!;
const screenVisionApiKeyInput = document.querySelector<HTMLInputElement>("#screenVisionApiKey")!;
const toggleScreenVisionApiKey = document.querySelector<HTMLButtonElement>("#toggleScreenVisionApiKey")!;
const clearScreenVisionApiKey = document.querySelector<HTMLButtonElement>("#clearScreenVisionApiKey")!;
const screenVisionApiKeyHint = document.querySelector<HTMLParagraphElement>("#screenVisionApiKeyHint")!;
const screenVisionIntervalInput = document.querySelector<HTMLInputElement>("#screenVisionInterval")!;
const proactiveSpeakToggle = document.querySelector<HTMLInputElement>("#proactiveSpeakToggle")!;
const proactiveIdleSecondsInput = document.querySelector<HTMLInputElement>("#proactiveIdleSeconds")!;
const voiceCloneToggle = document.querySelector<HTMLInputElement>("#voiceCloneToggle")!;
const voiceCloneAvailabilityHint = document.querySelector<HTMLParagraphElement>("#voiceCloneAvailabilityHint")!;
const referenceAudioInput = document.querySelector<HTMLInputElement>("#referenceAudioInput")!;
const referenceAudioPlayer = document.querySelector<HTMLAudioElement>("#referenceAudioPlayer")!;
const referenceAudioName = document.querySelector<HTMLSpanElement>("#referenceAudioName")!;
const applySettings = document.querySelector<HTMLButtonElement>("#applySettings")!;
const applySettingsDefaultText = applySettings.textContent?.trim() || "应用配置";
const applySettingsOfflineText = "后端未连接，点击重试";
const applySettingsApplyingText = "正在连接并应用…";
const subtitle = document.querySelector<HTMLDivElement>("#subtitle")!;
const status = document.querySelector<HTMLSpanElement>("#status")!;
const appShell = document.querySelector<HTMLElement>(".app-shell")!;
const settingsButton = document.querySelector<HTMLButtonElement>("#settingsButton")!;
const settingsPanel = document.querySelector<HTMLDivElement>("#settingsPanel")!;
const textPanel = document.querySelector<HTMLElement>(".text-panel")!;
const startButton = document.querySelector<HTMLButtonElement>("#startCapture")!;
const stopButton = document.querySelector<HTMLButtonElement>("#stopCapture")!;
const proactiveSpeakButton = document.querySelector<HTMLButtonElement>("#proactiveSpeakButton")!;
const videoFullscreenButton = document.querySelector<HTMLButtonElement>("#videoFullscreenButton")!;
const videoFrame = document.querySelector<HTMLDivElement>("#videoFrame")!;
const videoBackground = document.querySelector<HTMLImageElement>("#videoBackground")!;
const backgroundSidebar = document.querySelector<HTMLElement>("#backgroundSidebar")!;
const backgroundSidebarToggle = document.querySelector<HTMLButtonElement>("#backgroundSidebarToggle")!;
const backgroundTab = document.querySelector<HTMLButtonElement>("#backgroundTab")!;
const characterTab = document.querySelector<HTMLButtonElement>("#characterTab")!;
const workspaceTab = document.querySelector<HTMLButtonElement>("#workspaceTab")!;
const backgroundList = document.querySelector<HTMLDivElement>("#backgroundList")!;
const characterList = document.querySelector<HTMLDivElement>("#characterList")!;
const workspaceList = document.querySelector<HTMLDivElement>("#workspaceList")!;
const avatarCanvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const avatarStatus = document.querySelector<HTMLDivElement>("#avatarStatus")!;
const avatarZoomHint = document.querySelector<HTMLDivElement>("#avatarZoomHint")!;

function setAvatarStatus(message: string, tone: "loading" | "ready" | "error" = "loading") {
  avatarStatus.textContent = message;
  avatarStatus.dataset.tone = tone;
  avatarStatus.hidden = !message;
}

const avatarDriver = new VrmAvatar(avatarCanvas, setAvatarStatus, (label) => {
  avatarZoomHint.textContent = label;
});

type SinkAudioElement = HTMLAudioElement & {
  setSinkId?: (sinkId: string) => Promise<void>;
};

let micStream: MediaStream | null = null;
let vadInstance: MicVadInstance | null = null;
let ws: WebSocket | null = null;
let isCapturing = false;
let isCaptureStarting = false;
let isWsReady = false;
let isApplyingSettings = false;
let websocketReconnectTimer = 0;
let websocketReconnectAttempt = 0;
let pendingUserLine: HTMLParagraphElement | null = null;
let pendingUserTranscriptionLines: HTMLParagraphElement[] = [];
let userVoiceInputSequence = 0;
const displayedUserInputIds = new Set<string>();
const recentDisplayedUserTranscriptions = new Map<string, number>();
let lastAssistantLine: HTMLParagraphElement | null = null;
let outputVolume = 1;
let savedSettings: SavedSettings | null = null;
let audioQueue: Promise<void> = Promise.resolve();
let pendingPlaybackCompletion: { requestId: string; turnId?: string; queueVersion: number } | null = null;
const acknowledgedPlaybackRequestIds = new Set<string>();
let lastAssistantText = "";
let heardAssistantText = "";
let audioQueueVersion = 0;
let isAssistantResponding = false;
let isUserSpeaking = false;
let isUserInputPriorityActive = false;
let isUserVoiceTurnSubmitted = false;
let hasSentInterruptForCurrentUserInput = false;
let pendingUserTurnId = "";
let activeAssistantTurnId = "";
let referenceAudioBlob: Blob | null = null;
let referenceAudioStoredName = "";
let referenceAudioObjectUrl = "";
let voiceCloneCapability: boolean | null = null;
const pendingVoiceCloneRequests = new Map<
  string,
  { resolve: (success: boolean) => void; timeoutId: number }
>();
const pendingCredentialRequests = new Map<
  string,
  { resolve: (success: boolean) => void; timeoutId: number }
>();
let savedChatApiKeyAvailable = false;
let savedScreenVisionApiKeyAvailable = false;
let credentialStatusInitialized = false;
let backgroundOptions: BackgroundOption[] = [];
let characterOptions: CharacterConfigOption[] = [];
let activeAssetPanelTab: AssetPanelTab = "background";
let currentWorkspaceFolder = "";
let expandedWorkspaceFolders = new Set<string>();
let workspaceEntriesCache = new Map<string, WorkspaceEntry[]>();
let workspaceControlStatus: WorkspaceControlStatus = { label: "未连接", tone: "missing" };
let workspaceEventAbortController: AbortController | null = null;
let lastWorkspaceEventMs = Date.now();
const handledWorkspaceEventIds = new Set<string>();
let lastAppliedCharacterConfigFile = "";
let currentAssistantName = "小可";
let activeVrmModelId = "";
let pendingVrmModelId = "";
let isVrmModelSwitching = false;
let voiceChatOutputSinkId = "";
let isSettingsReadOnly = false;
let lastAudibleVolume = 100;
let screenStream: MediaStream | null = null;
let screenVideo: HTMLVideoElement | null = null;
let screenCaptureTimer = 0;
let latestScreenImage: string | null = null;
let screenShareWarningShown = false;
let lastUserConversationActivityAt = Date.now();
let proactiveSpeakTimer = 0;
let nextProactiveSpeakAt = Number.POSITIVE_INFINITY;
let proactiveUnansweredCount = 0;
let lastProactiveSpeakAt = 0;
let currentProactiveTurnId = "";
let currentProactiveIsAutomatic = false;
let isVideoFullscreen = false;
let isFallbackVideoFullscreen = false;
const responseAudio = new Audio() as SinkAudioElement;
const voiceChatAudio = new Audio() as SinkAudioElement;
const listeningDisplayText = "正在听";
const recognizingDisplayText = "正在识别...";
const duplicateUserTranscriptionWindowMs = 8000;

function roleLabel(role: LineRole) {
  if (role === "user") return "用户";
  if (role === "assistant") return currentAssistantName;
  return "系统";
}

function currentTime() {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date());
}

function readSavedSettings(): SavedSettings | null {
  try {
    const rawValue = localStorage.getItem(settingsStorageKey);
    if (!rawValue) return null;
    const parsed = JSON.parse(rawValue) as SavedSettings;
    const {
      apiKey: _legacyApiKey,
      screenVisionApiKey: _legacyVisionApiKey,
      referenceAudioName: _legacyReferenceAudioName,
      ...safeSettings
    } = parsed as SavedSettings & { referenceAudioName?: string };
    if (
      Object.prototype.hasOwnProperty.call(parsed, "apiKey") ||
      Object.prototype.hasOwnProperty.call(parsed, "screenVisionApiKey") ||
      Object.prototype.hasOwnProperty.call(parsed, "referenceAudioName")
    ) {
      localStorage.setItem(settingsStorageKey, JSON.stringify(safeSettings));
    }
    return {
      ...safeSettings,
      apiKey: "",
      screenVisionApiKey: "",
    };
  } catch (error) {
    console.warn(error);
    return null;
  }
}

function credentialProfileId() {
  try {
    const existing = localStorage.getItem(credentialProfileStorageKey)?.trim() || "";
    if (/^[A-Za-z0-9_-]{20,128}$/.test(existing)) return existing;
    const generated =
      typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : Array.from(crypto.getRandomValues(new Uint8Array(24)), (value) => value.toString(16).padStart(2, "0")).join("");
    localStorage.setItem(credentialProfileStorageKey, generated);
    return generated;
  } catch {
    return typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `temporary-${Date.now()}-${Math.random().toString(16).slice(2).padEnd(20, "0")}`;
  }
}

const localCredentialProfileId = credentialProfileId();

function persistSavedSettings() {
  if (!savedSettings) return;
  const { apiKey: _apiKey, screenVisionApiKey: _screenVisionApiKey, ...safeSettings } = savedSettings;
  localStorage.setItem(settingsStorageKey, JSON.stringify(safeSettings));
}

function normalizeStartupSettings(settings: SavedSettings | null): SavedSettings | null {
  if (!settings) return null;

  return {
    ...settings,
    characterConfigFile: normalizeCharacterConfigFile(settings.characterConfigFile),
    voiceChatOutputEnabled: false,
    voiceCloneEnabled: false,
    screenVisionEnabled: false,
    proactiveSpeakEnabled: false,
  };
}

function openSettingsPanel() {
  settingsPanel.hidden = false;
  settingsButton.setAttribute("aria-expanded", "true");
  syncSettingsPanelMode();
}

function normalizeEndpoint(value: string) {
  return value.trim() || defaultApiEndpoint;
}

function normalizeModel(value: string) {
  return value.trim() || defaultModel;
}

function normalizeScreenVisionInterval(value: string | undefined) {
  const parsed = Number.parseInt(value || "", 10);
  if (!Number.isFinite(parsed)) return "5";
  return String(Math.max(1, Math.min(parsed, 60)));
}

function normalizeProactiveIdleSeconds(value: string | undefined) {
  const parsed = Number.parseInt(value || "", 10);
  if (!Number.isFinite(parsed)) return defaultProactiveIdleSeconds;
  return String(Math.max(15, Math.min(parsed, 3600)));
}

function normalizeScreenVisionEndpoint(value: string | undefined) {
  return value?.trim() || defaultScreenVisionEndpoint;
}

function normalizeScreenVisionModel(value: string | undefined) {
  return value?.trim() || defaultScreenVisionModel;
}

function syncSecretToggle(input: HTMLInputElement, button: HTMLButtonElement) {
  const shouldShow = input.type === "password";
  input.type = shouldShow ? "text" : "password";
  button.textContent = shouldShow ? "隐藏" : "显示";
  button.setAttribute("aria-label", shouldShow ? "隐藏 API Key" : "显示 API Key");
}

function syncCredentialUi() {
  apiKeyInput.placeholder = savedChatApiKeyAvailable
    ? "已由 Windows 加密保存；留空继续使用"
    : "请输入 API Key";
  screenVisionApiKeyInput.placeholder = savedScreenVisionApiKeyAvailable
    ? "已由 Windows 加密保存；留空继续使用"
    : "请输入识图 API Key";
  apiKeyHint.textContent = savedChatApiKeyAvailable
    ? "已绑定当前 Windows 用户安全保存，不会写入浏览器或工作区。"
    : "Key 不会写入浏览器；应用后由 Windows 当前用户加密保存。";
  screenVisionApiKeyHint.textContent = savedScreenVisionApiKeyAvailable
    ? "已绑定当前 Windows 用户安全保存，不会回传给页面。"
    : "Key 不会写入浏览器；应用后由 Windows 当前用户加密保存。";
  clearApiKey.disabled = isSettingsReadOnly || !savedChatApiKeyAvailable;
  clearScreenVisionApiKey.disabled = isSettingsReadOnly || !savedScreenVisionApiKeyAvailable;
}

function validateChatApiSettings() {
  if (!endpointInput.value.trim()) {
    appendLine("system", "请先填写聊天 API 地址。");
    openSettingsPanel();
    return false;
  }
  if (!modelInput.value.trim()) {
    appendLine("system", "请先填写聊天模型。");
    openSettingsPanel();
    return false;
  }
  if (!apiKeyInput.value.trim() && !savedChatApiKeyAvailable) {
    appendLine("system", "请先填写聊天 API Key。");
    openSettingsPanel();
    return false;
  }
  return true;
}

function currentSettings(): SavedSettings {
  return {
    micId: micSelect.value,
    speakerId: speakerSelect.value,
    characterConfigFile: normalizeCharacterConfigFile(characterSelect.value),
    volume: volumeNumber.value,
    endpoint: normalizeEndpoint(endpointInput.value),
    model: normalizeModel(modelInput.value),
    apiKey: apiKeyInput.value.trim(),
    backgroundUrl: savedSettings?.backgroundUrl || backgroundOptions[0]?.url || "",
    vrmModelId: selectedVrmModelOption()?.id || "",
    voiceChatOutputEnabled: voiceChatOutputToggle.checked && !voiceChatOutputToggle.disabled,
    voiceChatOutputDeviceId: voiceChatOutputSelect.value,
    voiceCloneEnabled: voiceCloneToggle.checked,
    screenVisionEnabled: screenVisionToggle.checked,
    screenVisionEndpoint: normalizeScreenVisionEndpoint(screenVisionEndpointInput.value),
    screenVisionModel: normalizeScreenVisionModel(screenVisionModelInput.value),
    screenVisionApiKey: screenVisionApiKeyInput.value.trim(),
    screenVisionIntervalSec: normalizeScreenVisionInterval(screenVisionIntervalInput.value),
    proactiveSpeakEnabled: proactiveSpeakToggle.checked,
    proactiveIdleSeconds: normalizeProactiveIdleSeconds(proactiveIdleSecondsInput.value),
  };
}

function saveSettings() {
  savedSettings = currentSettings();
  persistSavedSettings();
}

function saveBackground(url: string) {
  savedSettings = {
    ...(savedSettings || currentSettings()),
    backgroundUrl: url,
  };
  persistSavedSettings();
}

function setVideoBackground(url: string) {
  if (!url) return;
  videoFrame.style.setProperty("--video-background-image", `url("${url}")`);
  videoBackground.src = url;
  saveBackground(url);
  backgroundList.querySelectorAll<HTMLButtonElement>(".background-item").forEach((button) => {
    const isActive = button.dataset.url === url;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });
}

function setAssetPanelTab(tab: AssetPanelTab, shouldOpen = true) {
  activeAssetPanelTab = tab;
  if (shouldOpen) {
    backgroundSidebar.classList.add("open");
    backgroundSidebarToggle.setAttribute("aria-expanded", "true");
  }
  const isBackgroundTab = tab === "background";
  const isCharacterTab = tab === "character";
  const isWorkspaceTab = tab === "workspace";

  backgroundTab.classList.toggle("active", isBackgroundTab);
  characterTab.classList.toggle("active", isCharacterTab);
  workspaceTab.classList.toggle("active", isWorkspaceTab);
  backgroundTab.setAttribute("aria-pressed", String(isBackgroundTab));
  characterTab.setAttribute("aria-pressed", String(isCharacterTab));
  workspaceTab.setAttribute("aria-pressed", String(isWorkspaceTab));
  backgroundList.hidden = !isBackgroundTab;
  characterList.hidden = !isCharacterTab;
  workspaceList.hidden = !isWorkspaceTab;
  backgroundSidebarToggle.setAttribute("aria-label", backgroundSidebar.classList.contains("open") ? "收起素材" : "展开素材");

  if (isWorkspaceTab) {
    void refreshWorkspaceList();
  }
}

async function readBackgroundOptions() {
  try {
    const response = await authenticatedFetch(backgroundManifestUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`Background manifest failed: ${response.status}`);
    const data = (await response.json()) as { backgrounds?: BackgroundOption[] };
    return data.backgrounds?.length ? data.backgrounds : fallbackBackgrounds;
  } catch (error) {
    console.warn(error);
    return fallbackBackgrounds;
  }
}

function renderBackgroundOptions(options: BackgroundOption[]) {
  backgroundList.textContent = "";

  options.forEach((option) => {
    const item = document.createElement("button");
    item.className = "background-item";
    item.type = "button";
    item.dataset.url = option.url;
    item.setAttribute("role", "listitem");
    item.setAttribute("aria-pressed", "false");

    const thumb = document.createElement("span");
    thumb.className = "background-thumb";

    const image = document.createElement("img");
    image.src = option.url;
    image.alt = option.name;
    thumb.appendChild(image);

    const label = document.createElement("span");
    label.className = "background-name";
    label.textContent = option.name;

    item.append(thumb, label);
    item.addEventListener("click", () => setVideoBackground(option.url));
    backgroundList.appendChild(item);
  });
}

function normalizeCharacterConfigFile(file: string | undefined | null) {
  if (!file || file === "xyu.yaml" || file === "xyua.yaml") return defaultCharacterConfigFile;
  return file;
}

function selectedCharacterConfigFile() {
  return normalizeCharacterConfigFile(characterSelect.value || savedSettings?.characterConfigFile);
}

function characterDisplayName(filename: string) {
  return filename.replace(/\.(ya?ml)$/i, "");
}

function characterOptionDisplayName(option: CharacterConfigOption) {
  return option.character_name || option.conf_name || option.name || characterDisplayName(option.filename);
}

function selectedCharacterOption() {
  const selectedFile = selectedCharacterConfigFile();
  return characterOptions.find((option) => option.filename === selectedFile);
}

function workspacePersonaName() {
  const option = selectedCharacterOption();
  return characterOptionDisplayName(option || defaultCharacterOption);
}

async function readWorkspaceEntries(folder = currentWorkspaceFolder) {
  const params = new URLSearchParams({
    persona: workspacePersonaName(),
    folder,
  });
  const response = await authenticatedFetch(`${workspaceManifestUrl}?${params.toString()}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`工作区加载失败：${response.status}`);
  return (await response.json()) as { entries?: WorkspaceEntry[]; folder?: string };
}

async function readWorkspaceControlStatus(): Promise<WorkspaceControlStatus> {
  const params = new URLSearchParams({ persona: workspacePersonaName() });
  const response = await authenticatedFetch(`${workspaceStateUrl}?${params.toString()}`, { cache: "no-store" });
  if (!response.ok) return { label: "未连接", tone: "missing" };
  const data = (await response.json()) as { state?: { updated_ms?: number; state?: { protocolAvailable?: boolean } } | null };
  if (!data.state) return { label: "未连接", tone: "missing" };

  const ageMs = Date.now() - Number(data.state.updated_ms || 0);
  if (!Number.isFinite(ageMs) || ageMs > 5000) return { label: "状态过期", tone: "stale" };
  if (!data.state.state?.protocolAvailable) return { label: "协议缺失", tone: "missing" };
  return { label: "可控制", tone: "ready" };
}

async function workspaceFileUrl(path: string) {
  const params = new URLSearchParams({
    persona: workspacePersonaName(),
    path,
  });
  const response = await authenticatedFetch(`/api/workspace-open-url?${params.toString()}`, {
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`工作区文件地址创建失败：${response.status}`);
  const payload = (await response.json()) as { url?: string };
  if (!payload.url || !payload.url.startsWith(`${workspaceBaseUrl}/workspace-files/`)) {
    throw new Error("工作区文件地址无效");
  }
  return payload.url;
}

async function openWorkspaceFile(path: string) {
  const target = window.open("about:blank", "_blank");
  if (!target) {
    appendLine("system", "浏览器阻止了工作区窗口，请允许此页面打开弹窗后重试。");
    return;
  }
  target.opener = null;
  try {
    target.location.replace(await workspaceFileUrl(path));
  } catch (error) {
    target.close();
    appendLine("system", "工作区文件打开失败，请刷新工作区列表后重试。");
    console.warn(error);
  }
}

function clearWorkspaceCache() {
  currentWorkspaceFolder = "";
  expandedWorkspaceFolders = new Set<string>();
  workspaceEntriesCache = new Map<string, WorkspaceEntry[]>();
}

function renderWorkspaceMessage(text: string) {
  workspaceList.textContent = "";
  const message = document.createElement("p");
  message.className = "workspace-message";
  message.textContent = text;
  workspaceList.appendChild(message);
}

function renderWorkspaceHeader() {
  const header = document.createElement("div");
  header.className = "workspace-header";

  const title = document.createElement("span");
  title.className = "workspace-title";
  title.textContent = workspacePersonaName();

  const control = document.createElement("span");
  control.className = `workspace-control-status ${workspaceControlStatus.tone}`;
  control.textContent = workspaceControlStatus.label;

  header.append(title, control);
  workspaceList.appendChild(header);
}

function renderWorkspaceEntry(entry: WorkspaceEntry, depth = 0) {
  const section = document.createElement("div");
  section.className = "workspace-entry-section";
  section.style.setProperty("--workspace-depth", String(depth));

  const row = document.createElement("button");
  row.className = entry.type === "directory" ? "workspace-folder-row" : "workspace-file-row";
  row.type = "button";
  row.dataset.path = entry.path;
  row.setAttribute("role", "listitem");

  const arrow = document.createElement("span");
  arrow.className = "workspace-arrow";

  const label = document.createElement("span");
  label.className = "workspace-name";
  label.textContent = entry.name;

  if (entry.type === "directory") {
    const isExpanded = expandedWorkspaceFolders.has(entry.path);
    arrow.textContent = ">";
    arrow.classList.toggle("expanded", isExpanded);
    row.setAttribute("aria-expanded", String(isExpanded));
    row.addEventListener("click", () => {
      sendWs({ type: "workspace-item-viewed", path: entry.path });
      void toggleWorkspaceFolder(entry.path);
    });
  } else {
    arrow.textContent = "";
    row.addEventListener("click", () => {
      sendWs({ type: "workspace-item-viewed", path: entry.path });
      void openWorkspaceFile(entry.path);
    });
  }

  row.append(arrow, label);
  section.appendChild(row);

  if (entry.type === "directory" && expandedWorkspaceFolders.has(entry.path)) {
    const children = workspaceEntriesCache.get(entry.path);
    const childList = document.createElement("div");
    childList.className = "workspace-children";

    if (!children) {
      const loading = document.createElement("p");
      loading.className = "workspace-message compact";
      loading.textContent = "正在读取...";
      childList.appendChild(loading);
    } else if (!children.length) {
      const empty = document.createElement("p");
      empty.className = "workspace-message compact";
      empty.textContent = "无文件";
      childList.appendChild(empty);
    } else {
      children.forEach((child) => childList.appendChild(renderWorkspaceEntry(child, depth + 1)));
    }

    section.appendChild(childList);
  }

  return section;
}

function renderWorkspaceEntries(entries: WorkspaceEntry[]) {
  workspaceList.textContent = "";
  renderWorkspaceHeader();

  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "workspace-message";
    empty.textContent = "无文件";
    workspaceList.appendChild(empty);
    return;
  }

  entries.forEach((entry) => workspaceList.appendChild(renderWorkspaceEntry(entry)));
}

async function toggleWorkspaceFolder(path: string) {
  if (expandedWorkspaceFolders.has(path)) {
    expandedWorkspaceFolders.delete(path);
    renderWorkspaceEntries(workspaceEntriesCache.get("") || []);
    return;
  }

  expandedWorkspaceFolders.add(path);
  renderWorkspaceEntries(workspaceEntriesCache.get("") || []);

  if (!workspaceEntriesCache.has(path)) {
    const data = await readWorkspaceEntries(path);
    workspaceEntriesCache.set(path, data.entries || []);
  }
  renderWorkspaceEntries(workspaceEntriesCache.get("") || []);
}

async function refreshWorkspaceList() {
  renderWorkspaceMessage("正在读取工作区...");

  try {
    const [data, status] = await Promise.all([
      readWorkspaceEntries(),
      readWorkspaceControlStatus().catch(() => ({ label: "未连接", tone: "missing" }) as WorkspaceControlStatus),
    ]);
    workspaceControlStatus = status;
    workspaceEntriesCache.set("", data.entries || []);
    renderWorkspaceEntries(workspaceEntriesCache.get("") || []);
  } catch (error) {
    console.warn(error);
    renderWorkspaceMessage(error instanceof Error ? error.message : "工作区加载失败。");
  }
}

function selectedVrmModelOption() {
  return (
    vrmModelOptions.find((option) => option.id === savedSettings?.vrmModelId) ||
    vrmModelOptions[0] ||
    null
  );
}

function saveVrmModel(id: string) {
  activeVrmModelId = id;
  savedSettings = {
    ...(savedSettings || currentSettings()),
    vrmModelId: id,
  };
  persistSavedSettings();
}

function syncVrmModelActiveState() {
  const selectedId = pendingVrmModelId || activeVrmModelId || selectedVrmModelOption()?.id || "";
  characterList.querySelectorAll<HTMLButtonElement>(".character-item").forEach((button) => {
    const isActive = button.dataset.modelId === selectedId;
    const isLoading = isVrmModelSwitching && button.dataset.modelId === pendingVrmModelId;
    button.classList.toggle("active", isActive);
    button.classList.toggle("loading", isLoading);
    button.setAttribute("aria-pressed", String(isActive));
    button.disabled = isVrmModelSwitching;
    button.setAttribute("aria-busy", String(isLoading));
  });
}

function setCurrentAssistantName(name?: string) {
  const normalizedName = name?.trim();
  currentAssistantName = normalizedName || "小可";
}

function syncAssistantNameFromSelection() {
  setCurrentAssistantName(characterOptionDisplayName(selectedCharacterOption() || defaultCharacterOption));
}

function selectCharacterConfigFile(file: string) {
  const normalizedFile = normalizeCharacterConfigFile(file);
  if (!normalizedFile) return;

  if ([...characterSelect.options].some((option) => option.value === normalizedFile)) {
    characterSelect.value = normalizedFile;
  }

  syncAssistantNameFromSelection();
  saveSettings();
  clearWorkspaceCache();

  if (isWsReady && selectedCharacterConfigFile() !== lastAppliedCharacterConfigFile) {
    sendCharacterConfigSwitch();
  }
  if (activeAssetPanelTab === "workspace") {
    void refreshWorkspaceList();
  }
}

async function readVrmModelOptions() {
  try {
    const response = await authenticatedFetch(vrmModelManifestUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`VRM manifest failed: ${response.status}`);
    const data = (await response.json()) as { models?: VrmModelOption[] };
    return data.models?.filter((option) => option.id && option.fileName && option.url) || [];
  } catch (error) {
    console.warn(error);
    setAvatarStatus("VRM 模型列表加载失败。", "error");
    return [];
  }
}

function formatModelSize(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "VRM";
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function settleVrmModelLayout() {
  refreshAvatarLayout();
  await new Promise((resolve) => window.setTimeout(resolve, 180));
  refreshAvatarLayout();
}

async function selectVrmModel(id: string) {
  const option = vrmModelOptions.find((modelOption) => modelOption.id === id);
  if (!option) return;
  if (isVrmModelSwitching || id === activeVrmModelId) return;

  const previousModelId = activeVrmModelId;
  isVrmModelSwitching = true;
  pendingVrmModelId = option.id;
  syncVrmModelActiveState();

  try {
    await avatarDriver.load(option.url);
    saveVrmModel(option.id);
    await settleVrmModelLayout();
  } catch (error) {
    console.warn(error);
    pendingVrmModelId = previousModelId;
    const message = error instanceof Error ? error.message : "模型切换失败。";
    setAvatarStatus(message, "error");
    appendLine("system", message);
  } finally {
    isVrmModelSwitching = false;
    pendingVrmModelId = "";
    syncVrmModelActiveState();
  }
}

function renderVrmModelOptions() {
  characterList.textContent = "";

  if (!vrmModelOptions.length) {
    const message = document.createElement("p");
    message.className = "workspace-message";
    message.textContent = "未找到 VRM 模型。把 .vrm 文件放进 MeloMate/models 后刷新页面。";
    characterList.appendChild(message);
    setAvatarStatus("请把 .vrm 模型放进 MeloMate/models。", "error");
    return;
  }

  vrmModelOptions.forEach((option) => {
    const item = document.createElement("button");
    item.className = "character-item";
    item.type = "button";
    item.dataset.modelId = option.id;
    item.setAttribute("role", "listitem");
    item.setAttribute("aria-pressed", "false");

    const portrait = document.createElement("span");
    portrait.className = "character-portrait";
    portrait.textContent = option.name.trim().slice(0, 1) || "模";

    const label = document.createElement("span");
    label.className = "character-name";
    label.textContent = option.name;

    const meta = document.createElement("span");
    meta.className = "character-file";
    meta.textContent = `${option.fileName} · ${formatModelSize(option.size)}`;

    item.append(portrait, label, meta);
    item.addEventListener("click", () => {
      void selectVrmModel(option.id);
    });
    characterList.appendChild(item);
  });

  syncVrmModelActiveState();
}

function renderCharacterOptions(options: CharacterConfigOption[]) {
  const previousValue = selectedCharacterConfigFile();
  const normalizedOptions = options.length
    ? options
    : [defaultCharacterOption];

  characterOptions = normalizedOptions;
  characterSelect.textContent = "";

  normalizedOptions.forEach((option) => {
    const item = document.createElement("option");
    item.value = option.filename;
    item.textContent = characterOptionDisplayName(option);
    characterSelect.appendChild(item);
  });

  if ([...characterSelect.options].some((option) => option.value === previousValue)) {
    characterSelect.value = previousValue;
  } else if ([...characterSelect.options].some((option) => option.value === defaultCharacterConfigFile)) {
    characterSelect.value = defaultCharacterConfigFile;
  }

  syncAssistantNameFromSelection();
}

async function setupBackgroundPicker() {
  backgroundOptions = await readBackgroundOptions();
  renderBackgroundOptions(backgroundOptions);
  setVideoBackground(savedSettings?.backgroundUrl || backgroundOptions[0]?.url || "");
}

function readReferenceAudioAsDataUrl(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(String(reader.result || "")));
    reader.addEventListener("error", () => reject(reader.error || new Error("Reference audio read failed.")));
    reader.readAsDataURL(file);
  });
}

function purgeLegacyReferenceAudioStorage(): Promise<void> {
  return new Promise((resolve) => {
    if (!("indexedDB" in window)) {
      resolve();
      return;
    }
    const request = indexedDB.deleteDatabase(referenceAudioDbName);
    request.addEventListener("success", () => resolve(), { once: true });
    request.addEventListener("error", () => {
      console.warn(request.error || new Error("Legacy reference audio cleanup failed."));
      resolve();
    }, { once: true });
    request.addEventListener("blocked", () => {
      console.warn("Legacy reference audio cleanup is waiting for another old MeloMate tab to close.");
      resolve();
    }, { once: true });
  });
}

function setReferenceAudioPreview(blob: Blob | null, name = "") {
  if (referenceAudioObjectUrl) {
    URL.revokeObjectURL(referenceAudioObjectUrl);
    referenceAudioObjectUrl = "";
  }

  referenceAudioBlob = blob;
  referenceAudioStoredName = name;

  if (!blob) {
    referenceAudioPlayer.removeAttribute("src");
    referenceAudioPlayer.load();
    return;
  }

  referenceAudioObjectUrl = URL.createObjectURL(blob);
  referenceAudioPlayer.src = referenceAudioObjectUrl;
  referenceAudioName.textContent = name;
}

function syncVoiceCloneControls() {
  if (voiceCloneCapability === false) {
    voiceCloneToggle.checked = false;
  }
  const enabled = voiceCloneToggle.checked;
  voiceCloneToggle.disabled = isSettingsReadOnly || voiceCloneCapability !== true;
  voiceCloneAvailabilityHint.hidden = voiceCloneCapability === true;
  voiceCloneAvailabilityHint.textContent =
    voiceCloneCapability === null
      ? "正在检查语音克隆组件…"
      : "语音克隆未安装；重新运行 setup-windows.bat 并选择安装即可启用。";
  setCollapsedGroup("voice-clone", !enabled);
  referenceAudioInput.disabled = isSettingsReadOnly || voiceCloneCapability !== true || !enabled;
  referenceAudioPlayer.classList.toggle(
    "disabled-audio",
    isSettingsReadOnly || voiceCloneCapability !== true || !enabled || !referenceAudioPlayer.src,
  );
  if (isSettingsReadOnly) {
    referenceAudioPlayer.pause();
  }
  if (!enabled) {
    referenceAudioInput.value = "";
    setReferenceAudioPreview(null);
    referenceAudioName.textContent = "未选择参考音频";
  } else if (!referenceAudioInput.files?.[0] && !referenceAudioBlob) {
    referenceAudioName.textContent = "请选择 3-10 秒参考音频";
  }
}

function validateVoiceCloneReference(audio: Blob, name: string, announce = true) {
  const normalizedName = name.trim().toLowerCase();
  const dotIndex = normalizedName.lastIndexOf(".");
  const extension = dotIndex >= 0 ? normalizedName.slice(dotIndex) : "";
  let message = "";
  if (!audio.size) {
    message = "参考音频为空，请重新选择。";
  } else if (audio.size > maxVoiceCloneReferenceBytes) {
    message = "参考音频过大，最大允许 10 MB。";
  } else if (!allowedVoiceCloneReferenceExtensions.has(extension)) {
    message = "参考音频格式不支持，请使用 WAV、MP3、FLAC 或 OGG。";
  }
  if (message && announce) appendLine("system", message);
  return !message;
}

function validateVoiceCloneSettings() {
  if (!voiceCloneToggle.checked) return true;
  if (voiceCloneCapability !== true) {
    appendLine("system", "语音克隆组件尚未安装，请重新运行 setup-windows.bat 并选择安装语音克隆。");
    return false;
  }
  const audio = referenceAudioInput.files?.[0] || referenceAudioBlob;
  const audioName = referenceAudioInput.files?.[0]?.name || referenceAudioStoredName;
  if (audio) return validateVoiceCloneReference(audio, audioName);
  appendLine("system", "已开启语音克隆，请先在设置里选择参考音频。");
  return false;
}

function clearInitialLine() {
  const initialLine = transcriptLog.querySelector(".system-line");
  if (initialLine?.textContent === "系统：等待启动麦克风。") {
    initialLine.remove();
  }
}

function isHiddenSystemError(text?: string) {
  return Boolean(text?.trim().startsWith("Conversation error:"));
}

function clearHiddenSystemErrors() {
  transcriptLog.querySelectorAll<HTMLParagraphElement>(".system-line").forEach((line) => {
    if (line.textContent?.includes("Conversation error:")) {
      line.remove();
    }
  });
}

function activeUserInputLines() {
  const seen = new Set<HTMLParagraphElement>();
  return [...pendingUserTranscriptionLines, pendingUserLine]
    .filter((line): line is HTMLParagraphElement => Boolean(line))
    .filter((line) => {
      if (seen.has(line)) return false;
      seen.add(line);
      return true;
    });
}

function keepActiveUserInputLinesAtBottom() {
  const lines = activeUserInputLines();
  if (!lines.length) return;

  lines.forEach((line) => {
    transcriptLog.appendChild(line);
  });
}

function appendLine(role: LineRole, text: string) {
  if (role === "system" && isHiddenSystemError(text)) {
    console.warn(text);
    return null;
  }

  if (role === "user") {
    stopAssistantReplyForUserInput();
  }

  clearInitialLine();
  const line = document.createElement("p");
  line.className = `line ${role}-line`;
  line.dataset.time = currentTime();
  line.dataset.rawText = text;
  line.textContent = `${line.dataset.time} ${roleLabel(role)}：${text}`;
  transcriptLog.appendChild(line);
  if (!activeUserInputLines().includes(line)) {
    keepActiveUserInputLinesAtBottom();
  }
  transcriptLog.scrollTop = transcriptLog.scrollHeight;
  return line;
}

function setPendingUserLine(text: string) {
  stopAssistantReplyForUserInput(true);

  if (!pendingUserLine) {
    pendingUserLine = appendLine("user", text);
    keepActiveUserInputLinesAtBottom();
    transcriptLog.scrollTop = transcriptLog.scrollHeight;
    return;
  }

  pendingUserLine.dataset.rawText = text;
  pendingUserLine.textContent = `${pendingUserLine.dataset.time} 用户：${text}`;
  keepActiveUserInputLinesAtBottom();
  transcriptLog.scrollTop = transcriptLog.scrollHeight;
}

function nextUserVoiceInputId() {
  userVoiceInputSequence += 1;
  return `voice-${Date.now()}-${userVoiceInputSequence}`;
}

function ensurePendingUserInputId() {
  if (!pendingUserLine) return "";
  if (!pendingUserLine.dataset.inputId) {
    pendingUserLine.dataset.inputId = nextUserVoiceInputId();
  }
  return pendingUserLine.dataset.inputId;
}

function takePendingUserLine(inputId?: string) {
  if (inputId) {
    const matchIndex = pendingUserTranscriptionLines.findIndex((line) => line.dataset.inputId === inputId);
    if (matchIndex !== -1) {
      const [line] = pendingUserTranscriptionLines.splice(matchIndex, 1);
      return line;
    }
    if (pendingUserLine?.dataset.inputId === inputId) {
      const line = pendingUserLine;
      pendingUserLine = null;
      return line;
    }
    return null;
  }

  return pendingUserTranscriptionLines.shift() || pendingUserLine;
}

function discardPendingUserLine(inputId?: string) {
  const line = takePendingUserLine(inputId);
  if (!line) return;
  line.remove();
  if (line === pendingUserLine) {
    pendingUserLine = null;
  }
}

function rememberDisplayedUserInput(inputId?: string) {
  if (!inputId) return;
  displayedUserInputIds.add(inputId);
  if (displayedUserInputIds.size > 200) {
    const oldestInputId = displayedUserInputIds.values().next().value as string | undefined;
    if (oldestInputId) displayedUserInputIds.delete(oldestInputId);
  }
}

function pruneRecentDisplayedUserTranscriptions(now = Date.now()) {
  recentDisplayedUserTranscriptions.forEach((displayedAt, text) => {
    if (now - displayedAt > duplicateUserTranscriptionWindowMs) {
      recentDisplayedUserTranscriptions.delete(text);
    }
  });
}

function wasUserTranscriptionRecentlyDisplayed(normalizedText: string) {
  if (!normalizedText) return false;
  const now = Date.now();
  pruneRecentDisplayedUserTranscriptions(now);
  const displayedAt = recentDisplayedUserTranscriptions.get(normalizedText);
  return Boolean(displayedAt && now - displayedAt <= duplicateUserTranscriptionWindowMs);
}

function rememberDisplayedUserTranscription(normalizedText: string) {
  if (!normalizedText) return;
  pruneRecentDisplayedUserTranscriptions();
  recentDisplayedUserTranscriptions.set(normalizedText, Date.now());
  if (recentDisplayedUserTranscriptions.size > 200) {
    const oldestText = recentDisplayedUserTranscriptions.keys().next().value as string | undefined;
    if (oldestText) recentDisplayedUserTranscriptions.delete(oldestText);
  }
}

function finalizePendingUserLine(text: string, inputId?: string) {
  if (inputId && displayedUserInputIds.has(inputId)) {
    discardPendingUserLine(inputId);
    return false;
  }

  const normalizedText = normalizeUserDisplayText(text);
  if (!inputId && wasUserTranscriptionRecentlyDisplayed(normalizedText)) {
    rememberDisplayedUserInput(inputId);
    return false;
  }

  const line = takePendingUserLine(inputId);
  if (line) {
    line.dataset.rawText = text;
    if (inputId) line.dataset.inputId = inputId;
    line.textContent = `${line.dataset.time} 用户：${text}`;
    if (line === pendingUserLine) {
      pendingUserLine = null;
    }
    rememberDisplayedUserInput(inputId || line.dataset.inputId);
    rememberDisplayedUserTranscription(normalizedText);
    keepActiveUserInputLinesAtBottom();
    transcriptLog.scrollTop = transcriptLog.scrollHeight;
    return true;
  }

  const lastUserLine = Array.from(transcriptLog.querySelectorAll<HTMLParagraphElement>(".user-line")).pop();
  if (lastUserLine && normalizeUserDisplayText(lastUserLine.dataset.rawText || "") === normalizedText) {
    return false;
  }

  const appendedLine = appendLine("user", text);
  if (appendedLine && inputId) {
    appendedLine.dataset.inputId = inputId;
  }
  rememberDisplayedUserInput(inputId);
  rememberDisplayedUserTranscription(normalizedText);
  return true;
}

function normalizeUserDisplayText(text: string) {
  return text.trim().replace(/\s+/g, " ");
}

function showMergedUserLine(texts: string[], fallbackText?: string) {
  const cleanTexts = texts.map((text) => text.trim()).filter(Boolean);
  if (!cleanTexts.length && !fallbackText?.trim()) return false;

  if (cleanTexts.length) {
    let displayedAny = false;
    cleanTexts.forEach((text) => {
      displayedAny = finalizePendingUserLine(text) || displayedAny;
    });
    return displayedAny;
  }

  return finalizePendingUserLine(fallbackText!.trim());
}

function appendAssistantLine(text: string, speakerName?: string) {
  if (speakerName) {
    setCurrentAssistantName(speakerName);
  }

  const cleanText = sanitizeAssistantReply(text);
  if (!cleanText || cleanText === lastAssistantText) return;
  if (/^[.。…]+$/.test(cleanText) && lastAssistantLine) {
    lastAssistantText = `${lastAssistantText}${cleanText}`;
    lastAssistantLine.textContent = `${lastAssistantLine.textContent || ""}${cleanText}`;
    subtitle.textContent = `${subtitle.textContent}${cleanText}`;
    return;
  }
  lastAssistantText = cleanText;
  heardAssistantText = [heardAssistantText, cleanText].filter(Boolean).join(" ");
  lastAssistantLine = appendLine("assistant", cleanText);
  subtitle.textContent = cleanText;
}

function setCaptureUi(active: boolean) {
  isCapturing = active;
  avatarDriver.setConversationState(active ? "listening" : "idle");
  startButton.disabled = active || isCaptureStarting;
  stopButton.disabled = !active;
  status.textContent = isCaptureStarting ? "启动中" : active ? "捕捉中" : "已停止";
  status.classList.toggle("active", active);
  syncSettingsPanelMode();
  syncProactiveSpeakButton();
}

function setAssistantStatus(state: "idle" | "thinking" | "answering" | "listening") {
  const isListening = isUserSpeaking || state === "listening";
  const isThinking = !isListening && state === "thinking";
  const isAnswering = !isListening && state === "answering";

  if (isListening) {
    status.textContent = listeningDisplayText;
  } else if (isThinking) {
    status.textContent = "思考中";
  } else if (isAnswering) {
    status.textContent = "回答中";
  } else if (isCapturing) {
    status.textContent = "捕捉中";
  } else {
    status.textContent = "已停止";
  }
  status.classList.toggle("listening", isListening);
  status.classList.toggle("thinking", isThinking);
  status.classList.toggle("answering", isAnswering);
  avatarDriver.setConversationState(
    isListening
      ? "listening"
      : isThinking
        ? "thinking"
        : isAnswering
          ? "speaking"
          : isCapturing
            ? "listening"
            : "idle",
  );
}

function setThinking(isThinking: boolean) {
  setAssistantStatus(isThinking ? "thinking" : "idle");
  syncProactiveSpeakButton();
}

function setAnswering(isAnswering: boolean) {
  setAssistantStatus(isAnswering ? "answering" : "idle");
  syncProactiveSpeakButton();
}

function clampVolume(value: string) {
  const numericValue = Number(value);
  if (Number.isNaN(numericValue)) return 100;
  return Math.min(100, Math.max(0, numericValue));
}

function syncVolume(value: string) {
  const nextValue = clampVolume(value);
  volumeRange.value = String(nextValue);
  volumeNumber.value = String(nextValue);
  outputVolume = nextValue / 100;
  responseAudio.volume = outputVolume;
  voiceChatAudio.volume = outputVolume;
  if (nextValue > 0) {
    lastAudibleVolume = nextValue;
  }
  syncVolumeMuteButton();
}

function saveVolumeSetting() {
  savedSettings = {
    ...(savedSettings || currentSettings()),
    volume: volumeNumber.value,
  };
  persistSavedSettings();
}

function syncVolumeMuteButton() {
  const isMuted = outputVolume <= 0;
  volumeMuteToggle.classList.toggle("muted", isMuted);
  volumeMuteToggle.setAttribute("aria-pressed", String(isMuted));
  volumeMuteToggle.setAttribute("aria-label", isMuted ? "恢复音量" : "静音");
}

function toggleMuteVolume() {
  if (outputVolume > 0) {
    syncVolume("0");
  } else {
    syncVolume(String(lastAudibleVolume || 50));
  }
  saveVolumeSetting();
}

async function askToOpenVoicemeeter() {
  if (!confirm("是否打开 Voicemeeter？")) return;

  try {
    const response = await authenticatedFetch("/api/open-voicemeeter", { method: "POST" });
    if (!response.ok) {
      appendLine("system", "Voicemeeter 启动失败，请确认已安装到默认路径，并重启 start.bat。");
      return;
    }
  } catch (error) {
    appendLine("system", "Voicemeeter 启动失败，请确认当前页面是通过 start.bat 打开的 127.0.0.1:5178。");
    console.warn(error);
  }
}

async function showVoicemeeterWindow() {
  try {
    const response = await authenticatedFetch("/api/show-voicemeeter", { method: "POST" });
    if (!response.ok) {
      appendLine("system", "显示 Voicemeeter 失败，请确认已安装到默认路径。");
      return;
    }
    appendLine("system", "正在显示 Voicemeeter 窗口。");
  } catch (error) {
    appendLine("system", "显示 Voicemeeter 失败，请确认 MeloMate 是通过 start.bat 启动的。");
    console.warn(error);
  }
}

function setCollapsedGroup(group: string, collapsed: boolean) {
  settingsPanel.querySelectorAll<HTMLElement>(`[data-collapse="${group}"]`).forEach((element) => {
    element.hidden = collapsed;
    element.toggleAttribute("hidden", collapsed);
    element.style.display = collapsed ? "none" : "";
  });
  settingsPanel.querySelectorAll<HTMLElement>(`[data-toggle-group="${group}"]`).forEach((element) => {
    element.classList.toggle("is-collapsed", collapsed);
  });
}

function syncCollapsibleSettings() {
  setCollapsedGroup("voice-chat", !voiceChatOutputToggle.checked);
  setCollapsedGroup("voice-clone", !voiceCloneToggle.checked);
  setCollapsedGroup("proactive-speak", !proactiveSpeakToggle.checked);
}

function syncSettingsPanelMode() {
  const isSettingsOpen = !settingsPanel.hidden;
  isSettingsReadOnly = (isCapturing || isCaptureStarting) && isSettingsOpen;
  textPanel.classList.toggle("settings-open", isSettingsOpen);
  textPanel.classList.toggle("settings-readonly", isSettingsReadOnly);

  startButton.hidden = isSettingsOpen;
  stopButton.hidden = isSettingsOpen;
  applySettings.hidden = !isSettingsOpen || isSettingsReadOnly;

  settingsPanel.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>("input, select, button").forEach(
    (control) => {
      control.disabled = isSettingsReadOnly;
    },
  );

  syncVoiceCloneControls();
  syncVoiceChatOutputHint();
  syncScreenVisionControls();
  syncProactiveSpeakControls();
  syncCollapsibleSettings();
  syncCredentialUi();
  syncApplySettingsButtonState();
  syncProactiveSpeakButton();
}

function syncApplySettingsButtonState() {
  const isBackendLoading = !isWsReady;
  const shouldDisable = isSettingsReadOnly || isApplyingSettings;
  applySettings.disabled = shouldDisable;
  applySettings.setAttribute("aria-disabled", String(shouldDisable));
  applySettings.classList.toggle("is-loading", isApplyingSettings);
  applySettings.textContent = isApplyingSettings
    ? applySettingsApplyingText
    : isBackendLoading
      ? applySettingsOfflineText
      : applySettingsDefaultText;
}

function syncScreenVisionControls() {
  setCollapsedGroup("screen-vision", !screenVisionToggle.checked);
  screenVisionEndpointInput.disabled = isSettingsReadOnly || !screenVisionToggle.checked;
  screenVisionModelInput.disabled = isSettingsReadOnly || !screenVisionToggle.checked;
  screenVisionApiKeyInput.disabled = isSettingsReadOnly || !screenVisionToggle.checked;
  screenVisionIntervalInput.disabled = isSettingsReadOnly || !screenVisionToggle.checked;
}

function syncProactiveSpeakControls() {
  setCollapsedGroup("proactive-speak", !proactiveSpeakToggle.checked);
  proactiveIdleSecondsInput.disabled = isSettingsReadOnly || !proactiveSpeakToggle.checked;
}

function syncProactiveSpeakButton() {
  proactiveSpeakButton.disabled = !isCapturing || !isWsReady || isAssistantResponding || isSettingsReadOnly;
}

function refreshAvatarLayout() {
  avatarDriver.resize();
  window.setTimeout(() => {
    avatarDriver.resize();
  }, 80);
}

function syncVideoFullscreenState(active: boolean) {
  isVideoFullscreen = active;
  document.body.classList.toggle("video-fullscreen-active", active);
  appShell.classList.toggle("video-fullscreen-fallback", active && isFallbackVideoFullscreen);
  videoFrame.classList.toggle("video-fullscreen", active);
  videoFullscreenButton.textContent = active ? "×" : "⛶";
  videoFullscreenButton.setAttribute("aria-label", active ? "退出全屏" : "进入全屏");
  refreshAvatarLayout();
}

async function toggleVideoFullscreen() {
  if (isVideoFullscreen) {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      isFallbackVideoFullscreen = false;
      syncVideoFullscreenState(false);
    }
    return;
  }

  try {
    if (videoFrame.requestFullscreen) {
      isFallbackVideoFullscreen = false;
      await videoFrame.requestFullscreen();
    } else {
      isFallbackVideoFullscreen = true;
      syncVideoFullscreenState(true);
    }
  } catch (error) {
    console.warn(error);
    isFallbackVideoFullscreen = true;
    syncVideoFullscreenState(true);
  }
}

function sanitizeAssistantReply(text: string) {
  const gameControlNarrationPatterns = [
    /我先看(?:一下|看)?(?:当前)?(?:局面|棋盘|盘面|情况|后面|画面|状态)?(?:再说|吧)?[，,。.!！~～]*/g,
    /让我看(?:一下|看)?(?:当前)?(?:局面|棋盘|盘面|情况|后面|画面|状态)?(?:再说|吧)?[，,。.!！~～]*/g,
    /我看(?:一下|看)?(?:当前|现在)?(?:局面|棋盘|盘面|情况|后面|画面|状态)[，,。.!！~～]*/g,
    /先看(?:一下|看)?(?:当前)?(?:局面|棋盘|盘面|情况|后面|画面|状态)(?:再说)?[，,。.!！~～]*/g,
    /看(?:一下|看)?(?:当前|现在)?(?:局面|棋盘|盘面|情况|后面|画面|状态)(?:再说)?[，,。.!！~～]*/g,
  ];
  let cleanText = text
    .replace(/\$/g, "")
    .replace(/（[^（）]{1,40}）/g, "")
    .replace(/\([^()]{1,40}\)/g, "")
    .replace(/\[[^\[\]]{1,40}\]/g, "")
    .replace(/【[^【】]{1,40}】/g, "")
    .replace(/\s+/g, " ");
  for (const pattern of gameControlNarrationPatterns) {
    cleanText = cleanText.replace(pattern, "");
  }
  cleanText = cleanText.replace(/\s+/g, " ").replace(/^[，,。.!！~～\s]+|[，,。.!！~～\s]+$/g, "").trim();
  return cleanText;
}

function restoreStaticSettings(settings: SavedSettings) {
  renderCharacterOptions(characterOptions);
  const savedCharacterConfigFile = normalizeCharacterConfigFile(settings.characterConfigFile);
  if ([...characterSelect.options].some((option) => option.value === savedCharacterConfigFile)) {
    characterSelect.value = savedCharacterConfigFile;
  }
  endpointInput.value = normalizeEndpoint(settings.endpoint);
  endpointInput.readOnly = false;
  modelInput.value = normalizeModel(settings.model);
  apiKeyInput.value = settings.apiKey?.trim() || "";
  voiceChatOutputToggle.checked = Boolean(settings.voiceChatOutputEnabled);
  voiceCloneToggle.checked = Boolean(settings.voiceCloneEnabled);
  screenVisionToggle.checked = Boolean(settings.screenVisionEnabled);
  screenVisionEndpointInput.value = normalizeScreenVisionEndpoint(settings.screenVisionEndpoint);
  screenVisionModelInput.value = normalizeScreenVisionModel(settings.screenVisionModel);
  screenVisionApiKeyInput.value = settings.screenVisionApiKey?.trim() || "";
  screenVisionIntervalInput.value = normalizeScreenVisionInterval(settings.screenVisionIntervalSec);
  proactiveSpeakToggle.checked = Boolean(settings.proactiveSpeakEnabled);
  proactiveIdleSecondsInput.value = normalizeProactiveIdleSeconds(settings.proactiveIdleSeconds);
  syncVolume(settings.volume || volumeNumber.value);
  syncVoiceCloneControls();
  syncVoiceChatOutputHint();
  syncScreenVisionControls();
  syncProactiveSpeakControls();
}

function restoreDeviceSelection(settings: SavedSettings | null) {
  if (!settings) return;

  if ([...micSelect.options].some((option) => option.value === settings.micId)) {
    micSelect.value = settings.micId;
  }

  if ([...speakerSelect.options].some((option) => option.value === settings.speakerId)) {
    speakerSelect.value = settings.speakerId;
  }

  if ([...voiceChatOutputSelect.options].some((option) => option.value === settings.voiceChatOutputDeviceId)) {
    voiceChatOutputSelect.value = settings.voiceChatOutputDeviceId || "";
  }
}

function findAudioInputByPattern(pattern: RegExp) {
  return [...micSelect.options].find((option) => pattern.test(option.textContent || ""));
}

function syncVoiceChatMicSelection() {
  if (isSettingsReadOnly) return;

  const targetOption = voiceChatOutputToggle.checked
    ? findAudioInputByPattern(voiceChatMicDevicePattern)
    : findAudioInputByPattern(physicalMicDevicePattern);

  if (targetOption) {
    micSelect.value = targetOption.value;
  }
}

function findVoiceChatAudioOutput(devices: MediaDeviceInfo[]) {
  const audioOutputs = devices.filter((device) => device.kind === "audiooutput");
  return (
    audioOutputs.find((device) => preferredVoiceChatOutputDevicePattern.test(device.label)) ||
    audioOutputs.find((device) => device.label.toLowerCase().includes("voicemeeter input")) ||
    audioOutputs.find((device) => voiceChatOutputDevicePattern.test(device.label))
  );
}

function hasVoiceChatAudioOutput(devices: MediaDeviceInfo[]) {
  return Boolean(findVoiceChatAudioOutput(devices));
}

function findAudioOutputById(devices: MediaDeviceInfo[], deviceId: string) {
  return devices.find((device) => device.kind === "audiooutput" && device.deviceId === deviceId);
}

function syncVoiceChatOutputHint(message?: string) {
  setCollapsedGroup("voice-chat", !voiceChatOutputToggle.checked);
  voiceChatOutputSelect.disabled = isSettingsReadOnly || voiceChatOutputToggle.disabled || !voiceChatOutputToggle.checked;
  testVoiceChatOutput.disabled =
    isSettingsReadOnly || voiceChatOutputToggle.disabled || !voiceChatOutputToggle.checked || !voiceChatOutputSinkId;
  showVoicemeeter.disabled = isSettingsReadOnly || voiceChatOutputToggle.disabled || !voiceChatOutputToggle.checked;
  voiceChatOutputHint.textContent =
    message ||
    (voiceChatOutputToggle.disabled
      ? "没有检测到 Voicemeeter Input，安装并重启 Voicemeeter 后再打开。"
      : voiceChatOutputToggle.checked
      ? "已开启：请选择 Voicemeeter Input / Voicemeeter In 4，然后点测试。语音软件麦克风请选择 Voicemeeter Output / Out B1。"
      : "开启后会额外把 AI 回复输出到 Voicemeeter Input，不会改变你自己的扬声器输出。");
}

function syncVoiceChatAvailability(devices: MediaDeviceInfo[]) {
  const isAvailable = hasVoiceChatAudioOutput(devices);
  voiceChatOutputToggle.disabled = isSettingsReadOnly || !isAvailable;

  if (!isAvailable) {
    voiceChatOutputToggle.checked = false;
    voiceChatOutputSinkId = "";
  }

  syncVoiceChatOutputHint();
  return isAvailable;
}

function applyVoiceChatOutputDevice(devices: MediaDeviceInfo[]) {
  if (!syncVoiceChatAvailability(devices)) {
    return false;
  }

  if (!voiceChatOutputToggle.checked) {
    voiceChatOutputSinkId = "";
    syncVoiceChatOutputHint();
    return false;
  }

  const selectedOutput = voiceChatOutputSelect.value
    ? findAudioOutputById(devices, voiceChatOutputSelect.value)
    : undefined;
  const voiceChatOutput = selectedOutput || findVoiceChatAudioOutput(devices);
  if (!voiceChatOutput) {
    voiceChatOutputSinkId = "";
    syncVoiceChatOutputHint("没有找到 Voicemeeter Input 播放设备。请先启动/安装 Voicemeeter，然后重新打开设置或点击应用配置。");
    return false;
  }

  voiceChatOutputSinkId = voiceChatOutput.deviceId;
  if (!voiceChatOutputSelect.value) {
    voiceChatOutputSelect.value = voiceChatOutput.deviceId;
  }
  syncVoiceChatOutputHint(`已绑定：AI 副路会输出到「${voiceChatOutput.label || "Voicemeeter Input"}」。这是 voicemeeterpro.exe 的 Voicemeeter Input 入口；你的扬声器输出保持不变。`);
  return true;
}

function fillVoiceChatOutputSelect(devices: MediaDeviceInfo[]) {
  const previousValue = voiceChatOutputSelect.value || savedSettings?.voiceChatOutputDeviceId || "";
  voiceChatOutputSelect.textContent = "";

  const autoOption = document.createElement("option");
  autoOption.value = "";
  autoOption.textContent = "自动选择 Voicemeeter Input";
  voiceChatOutputSelect.appendChild(autoOption);

  devices
    .filter((device) => device.kind === "audiooutput")
    .forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent = device.label || `播放设备 ${index + 1}`;
      voiceChatOutputSelect.appendChild(option);
    });

  if ([...voiceChatOutputSelect.options].some((option) => option.value === previousValue)) {
    voiceChatOutputSelect.value = previousValue;
  }
}
async function applyAudioOutput() {
  if (!speakerSelect.value) return;

  if (!responseAudio.setSinkId) {
    appendLine("system", "当前浏览器不支持选择扬声器，请使用 Chrome 或 Edge。");
    return;
  }

  const selectedSinkId = speakerSelect.value;
  try {
    await responseAudio.setSinkId(selectedSinkId);
  } catch (error) {
    console.warn("The selected audio output is unavailable; falling back to the system default output.", error);
    speakerSelect.value = "";
    try {
      await responseAudio.setSinkId("");
      appendLine("system", "所选扬声器不可用，已自动切换到系统默认输出。");
    } catch (fallbackError) {
      console.warn("Falling back to the system default audio output failed.", fallbackError);
      throw fallbackError;
    }
  }
}

async function applyVoiceChatAudioOutput() {
  if (!voiceChatOutputToggle.checked || !voiceChatOutputSinkId) return false;

  if (!voiceChatAudio.setSinkId) {
    syncVoiceChatOutputHint("当前浏览器不支持指定第二路输出。请使用 Chrome 或 Edge。");
    return false;
  }

  try {
    await voiceChatAudio.setSinkId(voiceChatOutputSinkId);
    return true;
  } catch (error) {
    voiceChatOutputSinkId = "";
    syncVoiceChatOutputHint("输出到 Voicemeeter Input 失败，请重新打开设置并点击应用配置。");
    console.warn(error);
    return false;
  }
}

function createTestToneDataUrl() {
  const sampleRate = 48000;
  const durationSeconds = 0.35;
  const sampleCount = Math.floor(sampleRate * durationSeconds);
  const bytesPerSample = 2;
  const dataSize = sampleCount * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  for (let index = 0; index < sampleCount; index += 1) {
    const fadeIn = Math.min(1, index / 1200);
    const fadeOut = Math.min(1, (sampleCount - index) / 1200);
    const envelope = Math.min(fadeIn, fadeOut);
    const sample = Math.sin((2 * Math.PI * 880 * index) / sampleRate) * 0.22 * envelope;
    view.setInt16(44 + index * bytesPerSample, Math.round(sample * 32767), true);
  }

  let binary = "";
  const bytes = new Uint8Array(buffer);
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return `data:audio/wav;base64,${btoa(binary)}`;
}

async function testVoiceChatOutputRoute() {
  const devices = navigator.mediaDevices?.enumerateDevices ? await navigator.mediaDevices.enumerateDevices() : [];
  if (!applyVoiceChatOutputDevice(devices)) return;
  if (!(await applyVoiceChatAudioOutput())) return;

  voiceChatAudio.pause();
  voiceChatAudio.src = createTestToneDataUrl();
  try {
    await playAudioElement(voiceChatAudio);
    syncVoiceChatOutputHint("测试音已发送到当前 Voicemeeter 输入通道。现在看 Voicemeeter 的 Voicemeeter Input 和 MASTER SECTION B1 是否跳动。");
  } catch (error) {
    syncVoiceChatOutputHint("测试音发送失败，请重新选择 Voicemeeter Input / In 4 后再试。");
    console.warn(error);
  }
}

function fillDeviceSelect(select: HTMLSelectElement, devices: MediaDeviceInfo[], fallbackLabel: string) {
  const previousValue = select.value;
  select.textContent = "";

  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = fallbackLabel;
  select.appendChild(defaultOption);

  devices.forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `${fallbackLabel} ${index + 1}`;
    select.appendChild(option);
  });

  if ([...select.options].some((option) => option.value === previousValue)) {
    select.value = previousValue;
  }
}

async function refreshDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = await navigator.mediaDevices.enumerateDevices();
  fillDeviceSelect(
    micSelect,
    devices.filter((device) => device.kind === "audioinput"),
    "默认麦克风",
  );
  fillDeviceSelect(
    speakerSelect,
    devices.filter((device) => device.kind === "audiooutput"),
    "默认扬声器",
  );
  fillVoiceChatOutputSelect(devices);
  restoreDeviceSelection(savedSettings);
  applyVoiceChatOutputDevice(devices);
  syncVoiceChatMicSelection();
}

function sendWs(message: object) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(message));
    return true;
  }
  return false;
}

type WorkspaceSanitizeBudget = { remaining: number; nodes: number; truncated: boolean };

function sanitizeWorkspaceValue(
  value: unknown,
  depth = 0,
  budget: WorkspaceSanitizeBudget = { remaining: 12_000, nodes: 1024, truncated: false },
): unknown {
  if (budget.remaining <= 0 || budget.nodes <= 0) {
    budget.truncated = true;
    return "[truncated]";
  }
  budget.nodes -= 1;
  if (depth > 7) {
    budget.truncated = true;
    return "[maximum depth reached]";
  }
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string") {
    const allowed = Math.max(0, Math.min(600, budget.remaining));
    if (value.length > allowed) budget.truncated = true;
    const result = value.slice(0, allowed);
    budget.remaining -= result.length;
    return result;
  }
  if (Array.isArray(value)) {
    if (value.length > 64) budget.truncated = true;
    const result: unknown[] = [];
    for (const item of value.slice(0, 64)) {
      result.push(sanitizeWorkspaceValue(item, depth + 1, budget));
      if (budget.nodes <= 0) break;
    }
    return result;
  }
  if (typeof value === "object") {
    const result: Record<string, unknown> = {};
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length > 64) budget.truncated = true;
    for (const [rawKey, item] of entries.slice(0, 64)) {
      const key = rawKey.slice(0, 80);
      if (!key || ["__proto__", "prototype", "constructor"].includes(key.toLowerCase())) continue;
      result[key] = sanitizeWorkspaceValue(item, depth + 1, budget);
      if (budget.remaining <= 0 || budget.nodes <= 0) break;
    }
    return result;
  }
  return String(value).slice(0, 600);
}

function normalizedWorkspaceEventData(event: WorkspaceEvent) {
  const budget: WorkspaceSanitizeBudget = { remaining: 12_000, nodes: 1024, truncated: false };
  const createdMs = Number(event.created_ms);
  const value: Record<string, unknown> = {
    id: String(event.id || "").slice(0, 128),
    type: event.type,
    created_ms: Number.isFinite(createdMs) ? createdMs : 0,
    state_version: Math.max(0, Number(event.state_version || 0)),
    persona: String(event.persona || workspacePersonaName()).slice(0, 128),
    page: sanitizeWorkspaceValue(event.page ?? {}, 0, budget),
    appState: sanitizeWorkspaceValue(event.appState ?? null, 0, budget),
    lastAction: sanitizeWorkspaceValue(event.lastAction ?? null, 0, budget),
    actionEvent: Boolean(event.actionEvent),
  };
  if (budget.truncated) value.truncated = true;
  return value;
}

async function readWorkspaceEvents(signal: AbortSignal) {
  const params = new URLSearchParams({
    persona: workspacePersonaName(),
    since: String(lastWorkspaceEventMs),
    wait_ms: String(workspaceEventLongPollMs),
  });
  const response = await authenticatedFetch(`${workspaceEventsUrl}?${params.toString()}`, {
    cache: "no-store",
    signal,
  });
  if (!response.ok) return [];
  const data = (await response.json()) as { events?: WorkspaceEvent[] };
  return data.events || [];
}

async function forwardCurrentWorkspaceState(signal: AbortSignal) {
  const persona = workspacePersonaName();
  const params = new URLSearchParams({ persona });
  const response = await authenticatedFetch(`${workspaceStateUrl}?${params.toString()}`, {
    cache: "no-store",
    signal,
  });
  if (!response.ok) return;
  const payload = (await response.json()) as {
    state?: {
      updated_ms?: number;
      state?: {
        protocolAvailable?: boolean;
        state_version?: number;
        page?: WorkspaceEvent["page"];
        appState?: unknown;
        lastAction?: WorkspaceEvent["lastAction"];
      };
    } | null;
  };
  const updatedMs = Number(payload.state?.updated_ms || 0);
  const report = payload.state?.state;
  if (
    !report?.protocolAvailable
    || !report.page?.id
    || !Number.isFinite(updatedMs)
    || Date.now() - updatedMs > 5000
  ) {
    return;
  }
  const version = Math.max(0, Number(report.state_version || 0));
  forwardWorkspaceEvent({
    id: `workspace-snapshot-${report.page.id}-${version}-${updatedMs}`,
    type: "workspace-state-changed",
    created_ms: updatedMs,
    state_version: version,
    persona,
    page: report.page,
    appState: report.appState,
    lastAction: report.lastAction,
    actionEvent: false,
  });
  lastWorkspaceEventMs = Math.max(lastWorkspaceEventMs, updatedMs);
}

function shouldForwardWorkspaceEvent(event: WorkspaceEvent) {
  if (!event || typeof event !== "object") return false;
  if (!event.id || handledWorkspaceEventIds.has(event.id)) return false;
  if (!["workspace-state-changed", "workspace-page-closed"].includes(event.type)) return false;
  if (event.persona && event.persona !== workspacePersonaName()) return false;
  return true;
}

function forwardWorkspaceEvent(event: WorkspaceEvent) {
  if (!isWsReady || !shouldForwardWorkspaceEvent(event)) return;
  handledWorkspaceEventIds.add(event.id);
  if (handledWorkspaceEventIds.size > 200) {
    const oldestId = handledWorkspaceEventIds.values().next().value;
    if (oldestId) handledWorkspaceEventIds.delete(oldestId);
  }

  sendWs({
    type: "workspace-state-event",
    event: normalizedWorkspaceEventData(event),
  });
}

async function runWorkspaceEventLoop(controller: AbortController) {
  while (isWsReady && workspaceEventAbortController === controller && !controller.signal.aborted) {
    try {
      const events = await readWorkspaceEvents(controller.signal);
      const latestByPage = new Map<string, WorkspaceEvent>();
      for (const event of events) {
        lastWorkspaceEventMs = Math.max(lastWorkspaceEventMs, Number(event.created_ms || 0));
        const pageId = String(event.page?.id || event.id || "");
        const previous = latestByPage.get(pageId);
        if (!previous || Number(event.created_ms || 0) >= Number(previous.created_ms || 0)) {
          latestByPage.set(pageId, event);
        }
      }
      for (const event of latestByPage.values()) {
        forwardWorkspaceEvent(event);
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      console.warn("Workspace event stream failed; reconnecting.", error);
      await new Promise((resolve) => window.setTimeout(resolve, 500));
    }
  }
}

function startWorkspaceEventLoop() {
  if (workspaceEventAbortController) return;
  workspaceEventAbortController = new AbortController();
  const controller = workspaceEventAbortController;
  void (async () => {
    try {
      await forwardCurrentWorkspaceState(controller.signal);
    } catch (error) {
      if (!controller.signal.aborted) console.warn("Workspace state restore failed.", error);
    }
    if (!controller.signal.aborted && workspaceEventAbortController === controller) {
      await runWorkspaceEventLoop(controller);
    }
  })();
}

function stopWorkspaceEventLoop() {
  workspaceEventAbortController?.abort();
  workspaceEventAbortController = null;
}

function stopProactiveSpeakLoop() {
  if (!proactiveSpeakTimer) return;
  window.clearInterval(proactiveSpeakTimer);
  proactiveSpeakTimer = 0;
}

function canTriggerProactiveSpeak() {
  return isWsReady
    && !isAssistantResponding
    && !isUserSpeaking
    && !isUserInputPriorityActive
    && !currentProactiveTurnId;
}

function proactiveBaseIntervalMs() {
  return Number(normalizeProactiveIdleSeconds(proactiveIdleSecondsInput.value)) * 1000;
}

function proactiveStageForCount(unansweredCount: number): ProactiveSpeakStage {
  if (unansweredCount <= 0) return "opening";
  const position = unansweredCount % 4;
  if (position === 0) return "fresh-topic";
  if (position === 1) return "curious";
  if (position === 2) return "warm-concern";
  return "playful-impatience";
}

function scheduleNextProactiveSpeak(now = Date.now()) {
  if (!proactiveSpeakToggle.checked || !isCapturing) {
    nextProactiveSpeakAt = Number.POSITIVE_INFINITY;
    return;
  }
  const position = Math.max(0, proactiveUnansweredCount - 1) % 4;
  const completedCycles = Math.floor(Math.max(0, proactiveUnansweredCount - 1) / 4);
  const baseMultipliers = [1.25, 1.6, 2.1, 3];
  const multiplier = Math.min(4, baseMultipliers[position] + Math.min(completedCycles * 0.2, 1));
  nextProactiveSpeakAt = now + Math.round(proactiveBaseIntervalMs() * multiplier);
}

function resetProactiveSilenceEpisode(now = Date.now()) {
  lastUserConversationActivityAt = now;
  proactiveUnansweredCount = 0;
  lastProactiveSpeakAt = 0;
  nextProactiveSpeakAt = proactiveSpeakToggle.checked && isCapturing
    ? now + proactiveBaseIntervalMs()
    : Number.POSITIVE_INFINITY;
}

function completeProactiveTurn(turnId?: string) {
  if (!currentProactiveTurnId) return false;
  if (turnId && turnId !== currentProactiveTurnId) return false;
  const wasAutomatic = currentProactiveIsAutomatic;
  currentProactiveTurnId = "";
  currentProactiveIsAutomatic = false;
  if (wasAutomatic) {
    scheduleNextProactiveSpeak();
  } else if (proactiveSpeakToggle.checked && isCapturing) {
    nextProactiveSpeakAt = Date.now() + proactiveBaseIntervalMs();
  } else {
    nextProactiveSpeakAt = Number.POSITIVE_INFINITY;
  }
  return true;
}

function proactiveReturnContext(now = Date.now()): ProactiveReturnContext | undefined {
  if (proactiveUnansweredCount <= 0) return undefined;
  return {
    elapsed_seconds: Math.max(0, Math.round((now - lastUserConversationActivityAt) / 1000)),
    unanswered_count: proactiveUnansweredCount,
    last_proactive_seconds_ago: lastProactiveSpeakAt
      ? Math.max(0, Math.round((now - lastProactiveSpeakAt) / 1000))
      : 0,
  };
}

function proactiveTurnId() {
  return typeof crypto.randomUUID === "function"
    ? `proactive-${crypto.randomUUID()}`
    : `proactive-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function requestProactiveSpeak(mode: "manual" | "automatic", announce = false) {
  if (!isCapturing) {
    if (announce) appendLine("system", "请先启动麦克风，再让 ta 说句话。");
    syncProactiveSpeakButton();
    return;
  }

  if (!isWsReady) {
    appendLine("system", "MeloMate 后端还没有连接成功，暂时不能主动说话。");
    syncProactiveSpeakButton();
    return;
  }

  if (!canTriggerProactiveSpeak()) {
    if (announce) appendLine("system", "ta 还在说话，等这一句说完再试。");
    syncProactiveSpeakButton();
    return;
  }

  if (mode === "manual") {
    resetProactiveSilenceEpisode();
  }

  const now = Date.now();
  const unansweredBeforeRequest = mode === "automatic" ? proactiveUnansweredCount : 0;
  const turnId = proactiveTurnId();
  currentProactiveTurnId = turnId;
  currentProactiveIsAutomatic = mode === "automatic";
  nextProactiveSpeakAt = Number.POSITIVE_INFINITY;
  setThinking(true);
  const images = await screenImagesForNextTurn().catch((error) => {
    console.warn("Capturing the proactive screen context failed.", error);
    return [];
  });
  if (
    currentProactiveTurnId !== turnId
    || !isCapturing
    || isUserSpeaking
    || isUserInputPriorityActive
  ) {
    completeProactiveTurn(turnId);
    setThinking(false);
    return;
  }
  const sent = sendWs({
    type: "ai-speak-signal",
    turn_id: turnId,
    images,
    screen_vision: screenVisionConfigPayload(),
    proactive: {
      mode,
      stage: proactiveStageForCount(unansweredBeforeRequest),
      elapsed_seconds: Math.max(0, Math.round((now - lastUserConversationActivityAt) / 1000)),
      unanswered_count: unansweredBeforeRequest,
      cycle_index: Math.floor(unansweredBeforeRequest / 4),
    },
  });
  if (!sent) {
    currentProactiveTurnId = "";
    currentProactiveIsAutomatic = false;
    setThinking(false);
    scheduleNextProactiveSpeak(now);
    return;
  }
  if (mode === "automatic") {
    proactiveUnansweredCount += 1;
    lastProactiveSpeakAt = now;
  }
}

function restartProactiveSpeakLoop() {
  stopProactiveSpeakLoop();

  if (!proactiveSpeakToggle.checked || !isCapturing) {
    nextProactiveSpeakAt = Number.POSITIVE_INFINITY;
    return;
  }

  if (!Number.isFinite(nextProactiveSpeakAt)) {
    nextProactiveSpeakAt = Date.now() + proactiveBaseIntervalMs();
  }

  proactiveSpeakTimer = window.setInterval(() => {
    if (!proactiveSpeakToggle.checked || !isCapturing || !canTriggerProactiveSpeak()) return;
    if (Date.now() < nextProactiveSpeakAt) return;
    void requestProactiveSpeak("automatic");
  }, proactiveSpeakCheckIntervalMs);
}

function requestCredentialStatus() {
  const requestId = typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `credentials-${Date.now()}`;
  sendWs({
    type: "credential-status-request",
    request_id: requestId,
    credential_profile_id: localCredentialProfileId,
  });
}

function cancelPendingCredentialRequests() {
  for (const pending of pendingCredentialRequests.values()) {
    window.clearTimeout(pending.timeoutId);
    pending.resolve(false);
  }
  pendingCredentialRequests.clear();
}

function sendClientApiConfig(): Promise<boolean> {
  const settings = currentSettings();
  if (!settings.apiKey && !savedChatApiKeyAvailable) {
    return Promise.resolve(false);
  }
  const requestId = typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `api-config-${Date.now()}`;
  const response = new Promise<boolean>((resolve) => {
    const timeoutId = window.setTimeout(() => {
      pendingCredentialRequests.delete(requestId);
      resolve(false);
    }, 15_000);
    pendingCredentialRequests.set(requestId, { resolve, timeoutId });
  });
  const sent = sendWs({
    type: "client-api-config",
    request_id: requestId,
    credential_profile_id: localCredentialProfileId,
    api_base_url: settings.endpoint,
    api_key: settings.apiKey,
    screen_vision_api_key: settings.screenVisionApiKey || "",
    model: settings.model,
  });
  if (!sent) {
    const pending = pendingCredentialRequests.get(requestId);
    if (pending) {
      window.clearTimeout(pending.timeoutId);
      pendingCredentialRequests.delete(requestId);
      pending.resolve(false);
    }
  }
  return response;
}

function clearSavedCredential(credential: "chat" | "screen_vision") {
  const label = credential === "chat" ? "聊天 API Key" : "识图 API Key";
  if (!window.confirm(`确定清除已安全保存的${label}吗？`)) return;
  const sent = sendWs({
    type: "clear-saved-credential",
    request_id: typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `clear-${Date.now()}`,
    credential_profile_id: localCredentialProfileId,
    credential,
  });
  if (!sent) appendLine("system", "后端尚未连接，无法清除已保存的 API Key。");
}

function sendCharacterConfigSwitch() {
  const file = selectedCharacterConfigFile();
  if (!file) return;
  sendWs({ type: "switch-config", file });
  lastAppliedCharacterConfigFile = file;
}

function updateVoiceCloneCapability(available: boolean) {
  voiceCloneCapability = available;
  if (!available && savedSettings?.voiceCloneEnabled) {
    savedSettings.voiceCloneEnabled = false;
    persistSavedSettings();
  }
  syncVoiceCloneControls();
}

function persistVoiceCloneDisabled() {
  voiceCloneToggle.checked = false;
  if (savedSettings) {
    savedSettings.voiceCloneEnabled = false;
    persistSavedSettings();
  }
  syncVoiceCloneControls();
}

function createVoiceCloneRequestId() {
  return typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `voice-clone-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function cancelPendingVoiceCloneRequests() {
  for (const request of pendingVoiceCloneRequests.values()) {
    window.clearTimeout(request.timeoutId);
    request.resolve(false);
  }
  pendingVoiceCloneRequests.clear();
}

async function sendClientVoiceCloneConfig(): Promise<boolean> {
  const enabled = voiceCloneToggle.checked;
  if (voiceCloneCapability !== true) {
    return !enabled;
  }

  const audio = enabled ? referenceAudioInput.files?.[0] || referenceAudioBlob : null;
  if (enabled && !audio) return false;
  const audioName = referenceAudioInput.files?.[0]?.name || referenceAudioStoredName;
  if (audio && !validateVoiceCloneReference(audio, audioName)) {
    persistVoiceCloneDisabled();
    return false;
  }

  let audioBase64: string | undefined;
  try {
    audioBase64 = audio ? await readReferenceAudioAsDataUrl(audio) : undefined;
  } catch (error) {
    console.warn(error);
    appendLine("system", "参考音频读取失败，已保持普通语音模式。");
    persistVoiceCloneDisabled();
    return false;
  }
  const requestId = createVoiceCloneRequestId();
  const response = new Promise<boolean>((resolve) => {
    const timeoutId = window.setTimeout(() => {
      pendingVoiceCloneRequests.delete(requestId);
      appendLine("system", "语音克隆配置等待超时，已保持普通语音模式。");
      persistVoiceCloneDisabled();
      resolve(false);
    }, 120_000);
    pendingVoiceCloneRequests.set(requestId, { resolve, timeoutId });
  });

  const sent = sendWs({
    type: "client-voice-clone-config",
    enabled,
    request_id: requestId,
    audio_base64: audioBase64,
    file_name:
      enabled
        ? audioName || "reference.wav"
        : undefined,
  });
  if (!sent) {
    const pending = pendingVoiceCloneRequests.get(requestId);
    if (pending) {
      window.clearTimeout(pending.timeoutId);
      pendingVoiceCloneRequests.delete(requestId);
      pending.resolve(false);
    }
    appendLine("system", "语音克隆配置未发送，已保持普通语音模式。");
    persistVoiceCloneDisabled();
  }
  return response;
}

function scheduleWebSocketReconnect() {
  if (websocketReconnectTimer || ws?.readyState === WebSocket.OPEN || ws?.readyState === WebSocket.CONNECTING) return;

  const delay =
    websocketReconnectDelays[Math.min(websocketReconnectAttempt, websocketReconnectDelays.length - 1)];
  websocketReconnectAttempt += 1;
  websocketReconnectTimer = window.setTimeout(() => {
    websocketReconnectTimer = 0;
    connectWebSocket();
  }, delay);
}

function connectWebSocket() {
  if (ws?.readyState === WebSocket.OPEN || ws?.readyState === WebSocket.CONNECTING) return;

  ws = new WebSocket(openLlmWsUrl, backendWebSocketProtocols);
  isWsReady = false;
  syncApplySettingsButtonState();

  ws.onopen = () => {
    isWsReady = true;
    syncApplySettingsButtonState();
    syncProactiveSpeakButton();
    websocketReconnectAttempt = 0;
    if (websocketReconnectTimer) {
      window.clearTimeout(websocketReconnectTimer);
      websocketReconnectTimer = 0;
    }
    sendWs({ type: "fetch-configs" });
    sendCharacterConfigSwitch();
    sendWs({ type: "fetch-history-list" });
    sendWs({ type: "create-new-history" });
    credentialStatusInitialized = false;
    requestCredentialStatus();
    startWorkspaceEventLoop();
  };

  ws.onmessage = (event) => {
    try {
      handleWsMessage(JSON.parse(event.data) as WsMessage);
    } catch (error) {
      console.warn("WebSocket message parse failed.", error);
    }
  };

  ws.onerror = () => {
    console.warn("MeloMate backend is not ready yet. Retrying...");
  };

  ws.onclose = () => {
    isWsReady = false;
    cancelPendingVoiceCloneRequests();
    cancelPendingCredentialRequests();
    credentialStatusInitialized = false;
    stopWorkspaceEventLoop();
    syncApplySettingsButtonState();
    syncProactiveSpeakButton();
    ws = null;
    if (isCapturing) {
      appendLine("system", "MeloMate 后端连接已断开。");
      stopCaptureInternal(false);
    }
    scheduleWebSocketReconnect();
  };
}

async function waitForWebSocketReady(timeoutMs = 10_000) {
  connectWebSocket();
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (isWsReady && ws?.readyState === WebSocket.OPEN) return true;
    await new Promise<void>((resolveWait) => window.setTimeout(resolveWait, 100));
  }
  return false;
}

function handleControlMessage(text: string, turnId?: string) {
  if (text === "conversation-chain-start") {
    if (!acceptAssistantTurn(turnId)) return;
    lastAssistantText = "";
    heardAssistantText = "";
    isAssistantResponding = true;
    isUserInputPriorityActive = false;
    isUserVoiceTurnSubmitted = false;
    setThinking(true);
    return;
  }

  if (text === "conversation-chain-end") {
    if (!shouldAcceptAssistantOutput({ turn_id: turnId })) return;
    void finishBackendAudio();
    if (turnId && turnId === pendingUserTurnId) {
      pendingUserTurnId = "";
    }
  }
}

function acceptAssistantTurn(turnId?: string) {
  if (pendingUserTurnId) {
    if (!turnId || turnId !== pendingUserTurnId) return false;
    activeAssistantTurnId = turnId;
    return true;
  }

  if (turnId) {
    activeAssistantTurnId = turnId;
    return true;
  }
  return !activeAssistantTurnId;
}

function shouldAcceptAssistantOutput(message: Pick<WsMessage, "turn_id">) {
  if (pendingUserTurnId) {
    return Boolean(message.turn_id && message.turn_id === pendingUserTurnId);
  }
  if (activeAssistantTurnId) {
    return message.turn_id === activeAssistantTurnId;
  }
  return !isUserInputPriorityActive;
}

function releaseUserInputPriorityAfterUserTextDisplayed() {
  if (!isUserInputPriorityActive || !isUserVoiceTurnSubmitted || isUserSpeaking || pendingUserTranscriptionLines.length) {
    return;
  }
  isUserInputPriorityActive = false;
  isUserVoiceTurnSubmitted = false;
}

function cancelUserInputPriority() {
  isUserInputPriorityActive = false;
  isUserVoiceTurnSubmitted = false;
  hasSentInterruptForCurrentUserInput = false;
  pendingUserTurnId = "";
}

function handleWsMessage(message: WsMessage) {
  if (message.type === "credential-status") {
    savedChatApiKeyAvailable = Boolean(message.chat_api_key_saved);
    savedScreenVisionApiKeyAvailable = Boolean(message.screen_vision_api_key_saved);
    syncCredentialUi();

    if (message.success && message.chat_config_applied) {
      apiKeyInput.value = "";
      screenVisionApiKeyInput.value = "";
      if (savedSettings) {
        savedSettings.apiKey = "";
        savedSettings.screenVisionApiKey = "";
        persistSavedSettings();
      }
    }

    const pending = message.request_id
      ? pendingCredentialRequests.get(message.request_id)
      : undefined;
    if (pending && message.request_id) {
      window.clearTimeout(pending.timeoutId);
      pendingCredentialRequests.delete(message.request_id);
      pending.resolve(Boolean(message.success && message.chat_config_applied));
    }

    if (message.cleared) {
      const label = message.cleared === "chat" ? "聊天 API Key" : "识图 API Key";
      appendLine("system", message.success ? `已清除安全保存的${label}。` : message.message || `清除${label}失败。`);
    } else if (message.success === false && message.message) {
      appendLine("system", message.message);
    }

    if (!credentialStatusInitialized) {
      credentialStatusInitialized = true;
      if (savedChatApiKeyAvailable) void sendClientApiConfig();
    }
    return;
  }

  if (message.type === "workspace-event-rejected") {
    console.warn("Workspace state event was rejected:", message.reason);
    return;
  }

  if (message.type === "workspace-control-status") {
    const labels: Record<string, WorkspaceControlStatus> = {
      thinking: { label: "观察中", tone: "ready" },
      acted: { label: "已操作", tone: "ready" },
      paused: { label: "控制暂停", tone: "stale" },
      error: { label: "控制异常", tone: "stale" },
      closed: { label: "未连接", tone: "missing" },
    };
    workspaceControlStatus = labels[message.status || ""] || workspaceControlStatus;
    if (activeAssetPanelTab === "workspace" && workspaceEntriesCache.has("")) {
      renderWorkspaceEntries(workspaceEntriesCache.get("") || []);
    }
    return;
  }

  if (message.type === "control" && message.text) {
    handleControlMessage(message.text, message.turn_id);
    return;
  }

  if (message.type === "full-text") {
    if (!shouldAcceptAssistantOutput(message)) return;
    if (message.text && !["Connection established", "Thinking...", "AI wants to speak something..."].includes(message.text)) {
      subtitle.textContent = sanitizeAssistantReply(message.text) || message.text;
    }
    return;
  }

  if (message.type === "user-input-transcription" && message.text) {
    const displayed = finalizePendingUserLine(message.text, message.input_id);
    if (displayed) {
      subtitle.textContent = message.text;
      stopAssistantReplyForUserInput();
    }
    releaseUserInputPriorityAfterUserTextDisplayed();
    return;
  }

  if (message.type === "user-input-merged") {
    const displayed = showMergedUserLine(message.texts || [], message.text);
    if (displayed && message.text) {
      subtitle.textContent = message.text;
      stopAssistantReplyForUserInput();
    }
    releaseUserInputPriorityAfterUserTextDisplayed();
    return;
  }

  if (message.type === "audio") {
    if (!shouldAcceptAssistantOutput(message)) return;
    queueAudioMessage(message);
    return;
  }

  if (message.type === "backend-synth-complete") {
    if (!shouldAcceptAssistantOutput(message)) return;
    if (!message.request_id) {
      console.warn("Ignoring backend synthesis completion without a request_id.");
      return;
    }
    if (
      acknowledgedPlaybackRequestIds.has(message.request_id)
      || pendingPlaybackCompletion?.requestId === message.request_id
    ) {
      return;
    }
    pendingPlaybackCompletion = {
      requestId: message.request_id,
      turnId: message.turn_id,
      queueVersion: audioQueueVersion,
    };
    void finishBackendAudio();
    return;
  }

  if (message.type === "interrupt-signal") {
    stopCurrentResponsePlayback();
    return;
  }

  if (message.type === "voice-clone-config-applied") {
    if (typeof message.available === "boolean") {
      updateVoiceCloneCapability(message.available);
    }
    const pending = message.request_id
      ? pendingVoiceCloneRequests.get(message.request_id)
      : undefined;
    if (pending && message.request_id) {
      window.clearTimeout(pending.timeoutId);
      pendingVoiceCloneRequests.delete(message.request_id);
      pending.resolve(Boolean(message.success));
    }
    if (!message.success) {
      persistVoiceCloneDisabled();
      appendLine("system", message.message || "语音克隆配置失败，已保持普通语音模式。");
    }
    return;
  }

  if (message.type === "config-files") {
    renderCharacterOptions(message.configs || []);
    return;
  }

  if (message.type === "config-switched") {
    stopWorkspaceEventLoop();
    lastWorkspaceEventMs = Date.now();
    handledWorkspaceEventIds.clear();
    if (isWsReady) startWorkspaceEventLoop();
    void sendClientApiConfig();
    void sendClientVoiceCloneConfig();
    return;
  }

  if (message.type === "error") {
    if (isHiddenSystemError(message.message)) {
      console.warn(message.message);
    } else {
      appendLine("system", message.message || "MeloMate 后端返回错误。");
    }
    if (!message.turn_id || message.turn_id === activeAssistantTurnId || message.turn_id === currentProactiveTurnId) {
      isAssistantResponding = false;
      completeProactiveTurn(message.turn_id);
      if (message.turn_id && message.turn_id === activeAssistantTurnId) {
        activeAssistantTurnId = "";
      }
      syncProactiveSpeakButton();
    }
    setThinking(false);
    return;
  }

  if (message.type === "set-model-and-conf") {
    setCurrentAssistantName(message.character_name || message.conf_name);
    if (typeof message.capabilities?.voice_clone === "boolean") {
      updateVoiceCloneCapability(message.capabilities.voice_clone);
      void sendClientVoiceCloneConfig();
    }
    if (message.conf_name) {
      console.info("MeloMate config:", message.conf_name, message.client_uid);
    }
  }
}

function handleAudioQueueFailure(error: unknown, queueVersion: number) {
  console.warn("An audio queue item failed; subsequent audio items will continue.", error);
  if (queueVersion !== audioQueueVersion) return;

  try {
    appendLine("system", "一段回复音频播放失败，后续音频将继续播放。");
    setAnswering(false);
  } catch (uiError) {
    console.warn("Updating the UI after an audio queue failure failed.", uiError);
  }
}

function queueAudioMessage(message: WsMessage) {
  if (!shouldAcceptAssistantOutput(message)) return;

  const text = message.display_text?.text || "";
  const queueVersion = audioQueueVersion;
  const turnId = message.turn_id;
  audioQueue = audioQueue
    .catch((error) => {
      handleAudioQueueFailure(error, queueVersion);
    })
    .then(async () => {
      if (queueVersion !== audioQueueVersion || !shouldAcceptAssistantOutput({ turn_id: turnId })) return;

      if (text) {
        appendAssistantLine(text, message.display_text?.name);
      }

      if (message.audio) {
        avatarDriver.setExpression(message.actions?.expressions);
        await playBackendAudio(message.audio, queueVersion, text, turnId);
      }
    })
    .catch((error) => {
      handleAudioQueueFailure(error, queueVersion);
    });
}

async function playBackendAudio(
  audioBase64: string,
  queueVersion: number,
  spokenText: string,
  turnId?: string,
) {
  if (queueVersion !== audioQueueVersion || !shouldAcceptAssistantOutput({ turn_id: turnId })) return;

  responseAudio.pause();
  voiceChatAudio.pause();
  setAnswering(true);

  const audioSource = `data:audio/wav;base64,${audioBase64}`;
  let lipSyncTrack = null;
  try {
    lipSyncTrack = await avatarDriver.prepareLipSync(audioBase64, spokenText);
  } catch (error) {
    console.warn("VRM lip-sync analysis failed; audio playback will continue.", error);
  }
  if (queueVersion !== audioQueueVersion || !shouldAcceptAssistantOutput({ turn_id: turnId })) return;

  responseAudio.src = audioSource;
  await applyAudioOutput();
  if (queueVersion !== audioQueueVersion || !shouldAcceptAssistantOutput({ turn_id: turnId })) return;

  avatarDriver.startLipSync(responseAudio, lipSyncTrack);
  const playbackTasks = [playAudioElement(responseAudio)];
  const canUseVoiceChatOutput = await applyVoiceChatAudioOutput();
  if (queueVersion !== audioQueueVersion || !shouldAcceptAssistantOutput({ turn_id: turnId })) return;

  if (canUseVoiceChatOutput) {
    voiceChatAudio.src = audioSource;
    playbackTasks.push(playAudioElement(voiceChatAudio));
  }

  try {
    await Promise.all(playbackTasks);
  } finally {
    avatarDriver.stopLipSync();
  }
}

async function playAudioElement(audio: HTMLAudioElement) {
  audio.volume = outputVolume;

  await new Promise<void>((resolve) => {
    const cleanup = () => {
      audio.removeEventListener("ended", cleanup);
      audio.removeEventListener("error", cleanup);
      audio.removeEventListener("pause", cleanup);
      resolve();
    };

    audio.addEventListener("ended", cleanup);
    audio.addEventListener("error", cleanup);
    audio.addEventListener("pause", cleanup);
    audio.play().catch(cleanup);
  });
}

function screenVisionEnabled() {
  return screenVisionToggle.checked;
}

function screenVisionIntervalMs() {
  return Number(normalizeScreenVisionInterval(screenVisionIntervalInput.value)) * 1000;
}

function screenVisionConfigPayload() {
  if (!screenVisionEnabled()) return null;
  return {
    api_base_url: screenVisionEndpointInput.value.trim(),
    model: screenVisionModelInput.value.trim(),
  };
}

function validateScreenVisionSettings() {
  if (!screenVisionEnabled()) return true;
  if (!screenVisionEndpointInput.value.trim()) {
    appendLine("system", "请先填写识图 API 地址。");
    openSettingsPanel();
    return false;
  }
  if (!screenVisionModelInput.value.trim()) {
    appendLine("system", "请先填写模型。");
    openSettingsPanel();
    return false;
  }
  if (!screenVisionApiKeyInput.value.trim() && !savedScreenVisionApiKeyAvailable) {
    appendLine("system", "请先填写 API Key。");
    openSettingsPanel();
    return false;
  }
  screenVisionEndpointInput.value = screenVisionEndpointInput.value.trim();
  screenVisionModelInput.value = screenVisionModelInput.value.trim();
  return true;
}

async function captureScreenImage() {
  if (!screenVideo || screenVideo.readyState < screenVideo.HAVE_CURRENT_DATA) return null;
  const sourceWidth = screenVideo.videoWidth;
  const sourceHeight = screenVideo.videoHeight;
  if (!sourceWidth || !sourceHeight) return null;

  const scale = Math.min(1, screenVisionMaxWidth / sourceWidth);
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return null;

  context.drawImage(screenVideo, 0, 0, width, height);
  latestScreenImage = canvas.toDataURL("image/jpeg", screenVisionJpegQuality);
  return latestScreenImage;
}

function stopScreenVision() {
  if (screenCaptureTimer) {
    window.clearInterval(screenCaptureTimer);
    screenCaptureTimer = 0;
  }
  screenStream?.getTracks().forEach((track) => {
    try {
      track.stop();
    } catch (error) {
      console.warn("Stopping a screen-capture track failed.", error);
    }
  });
  if (screenVideo) {
    try {
      screenVideo.pause();
      screenVideo.srcObject = null;
    } catch (error) {
      console.warn("Releasing the screen-capture video element failed.", error);
    }
  }
  screenStream = null;
  screenVideo = null;
  latestScreenImage = null;
}

async function startScreenVisionIfNeeded() {
  if (!screenVisionEnabled()) {
    stopScreenVision();
    return true;
  }

  if (!navigator.mediaDevices?.getDisplayMedia) {
    appendLine("system", "当前浏览器不支持屏幕共享，无法开启识别屏幕。");
    return false;
  }

  if (!screenStream) {
    try {
      screenStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: false,
      });
    } catch (error) {
      appendLine("system", "没有拿到屏幕共享权限，识别屏幕已跳过。");
      console.warn(error);
      return false;
    }

    screenStream.getVideoTracks()[0]?.addEventListener("ended", () => {
      stopScreenVision();
      if (!screenShareWarningShown) {
        screenShareWarningShown = true;
        appendLine("system", "屏幕共享已停止，识别屏幕暂不可用。");
      }
    });

    screenVideo = document.createElement("video");
    screenVideo.muted = true;
    screenVideo.playsInline = true;
    screenVideo.srcObject = screenStream;
    await screenVideo.play();
  }

  await captureScreenImage();
  if (screenCaptureTimer) window.clearInterval(screenCaptureTimer);
  screenCaptureTimer = window.setInterval(() => {
    void captureScreenImage();
  }, screenVisionIntervalMs());
  screenShareWarningShown = false;
  return true;
}

async function screenImagesForNextTurn() {
  if (!screenVisionEnabled()) return [];
  const image = latestScreenImage || (await captureScreenImage());
  if (!image) return [];
  return [
    {
      source: "screen",
      data: image,
      mime_type: "image/jpeg",
    },
  ];
}

async function finishBackendAudio() {
  const completion = pendingPlaybackCompletion;
  if (!completion) return;
  await audioQueue.catch(() => undefined);
  if (
    pendingPlaybackCompletion !== completion
    || completion.queueVersion !== audioQueueVersion
  ) {
    return;
  }

  pendingPlaybackCompletion = null;
  acknowledgedPlaybackRequestIds.add(completion.requestId);
  if (acknowledgedPlaybackRequestIds.size > 200) {
    const oldestRequestId = acknowledgedPlaybackRequestIds.values().next().value;
    if (oldestRequestId) acknowledgedPlaybackRequestIds.delete(oldestRequestId);
  }
  sendWs({
    type: "frontend-playback-complete",
    request_id: completion.requestId,
    turn_id: completion.turnId,
  });
  setThinking(false);
  isAssistantResponding = false;
  const completedProactiveTurn = completeProactiveTurn(completion.turnId);
  if (!completedProactiveTurn && proactiveSpeakToggle.checked && isCapturing) {
    nextProactiveSpeakAt = Date.now() + proactiveBaseIntervalMs();
  }
  if (completion.turnId && completion.turnId === activeAssistantTurnId) {
    activeAssistantTurnId = "";
  }
  syncProactiveSpeakButton();
}

function stopCurrentResponsePlayback(force = false) {
  const hasActivePlayback = !responseAudio.paused || !voiceChatAudio.paused;
  if (!force && !isAssistantResponding && !hasActivePlayback) return;

  isAssistantResponding = false;
  completeProactiveTurn();
  pendingPlaybackCompletion = null;
  audioQueueVersion += 1;
  audioQueue = Promise.resolve();

  responseAudio.pause();
  responseAudio.removeAttribute("src");
  responseAudio.load();
  avatarDriver.stopLipSync();
  voiceChatAudio.pause();
  voiceChatAudio.removeAttribute("src");
  voiceChatAudio.load();
  setThinking(false);
  syncProactiveSpeakButton();
}

function interruptCurrentResponse() {
  const hasActivePlayback = !responseAudio.paused || !voiceChatAudio.paused;
  if (!isAssistantResponding && !hasActivePlayback) return;

  const interruptedText = heardAssistantText || lastAssistantText || subtitle.textContent || "";
  stopCurrentResponsePlayback(true);

  sendWs({
    type: "interrupt-signal",
    text: interruptedText,
  });
}

function stopAssistantReplyForUserInput(notifyBackend = false) {
  const shouldNotifyBackend =
    notifyBackend && !hasSentInterruptForCurrentUserInput && (isAssistantResponding || heardAssistantText || lastAssistantText);
  const interruptedText = heardAssistantText || lastAssistantText || subtitle.textContent || "";

  stopCurrentResponsePlayback(true);

  if (shouldNotifyBackend) {
    hasSentInterruptForCurrentUserInput = true;
    sendWs({
      type: "interrupt-signal",
      text: interruptedText,
    });
  }
}

function beginUserVoiceInput() {
  isUserSpeaking = true;
  isUserInputPriorityActive = true;
  isUserVoiceTurnSubmitted = false;
  hasSentInterruptForCurrentUserInput = false;
  pendingUserTurnId = nextUserVoiceInputId();
  activeAssistantTurnId = "";
  stopAssistantReplyForUserInput(true);
  setPendingUserLine(listeningDisplayText);
  ensurePendingUserInputId();
  subtitle.textContent = listeningDisplayText;
  setAssistantStatus("listening");
  syncProactiveSpeakButton();
}

function endUserVoiceInput() {
  if (!isUserSpeaking) return;
  isUserSpeaking = false;
  setAssistantStatus("idle");
  syncProactiveSpeakButton();
}

function markUserVoiceAwaitingTranscription() {
  if (!pendingUserLine) return "";

  const inputId = ensurePendingUserInputId();
  pendingUserLine.dataset.rawText = recognizingDisplayText;
  pendingUserLine.textContent = `${pendingUserLine.dataset.time} 用户：${recognizingDisplayText}`;
  pendingUserTranscriptionLines.push(pendingUserLine);
  pendingUserLine = null;
  subtitle.textContent = recognizingDisplayText;
  stopAssistantReplyForUserInput();
  keepActiveUserInputLinesAtBottom();
  transcriptLog.scrollTop = transcriptLog.scrollHeight;
  return inputId;
}

async function sendAudioPartition(audio: Float32Array) {
  if (!isWsReady) {
    endUserVoiceInput();
    cancelUserInputPriority();
    appendLine("system", "MeloMate 后端还没有连接成功。");
    return;
  }

  const speechAudio = prepareSpeechAudio(audio);
  if (!speechAudio.length) {
    endUserVoiceInput();
    cancelUserInputPriority();
    pendingUserLine?.remove();
    pendingUserLine = null;
    subtitle.textContent = isCapturing ? "麦克风已启动。" : "麦克风已停止。";
    return;
  }

  for (let index = 0; index < speechAudio.length; index += vadChunkSize) {
    const chunk = speechAudio.slice(index, Math.min(index + vadChunkSize, speechAudio.length));
    sendWs({
      type: "mic-audio-data",
      audio: Array.from(chunk),
    });
  }

  const inputId = markUserVoiceAwaitingTranscription();
  const turnId = pendingUserTurnId;
  const returnContext = proactiveReturnContext();
  endUserVoiceInput();
  isUserVoiceTurnSubmitted = true;
  const submitted = sendWs({
    type: "mic-audio-end",
    input_id: inputId,
    turn_id: turnId,
    images: await screenImagesForNextTurn(),
    screen_vision: screenVisionConfigPayload(),
    metadata: returnContext ? { proactive_return: returnContext } : undefined,
  });
  if (submitted) {
    resetProactiveSilenceEpisode();
  }
  setThinking(true);
}

function prepareSpeechAudio(audio: Float32Array) {
  let peak = 0;
  let sumSquares = 0;
  for (let index = 0; index < audio.length; index += 1) {
    const sample = audio[index];
    peak = Math.max(peak, Math.abs(sample));
    sumSquares += sample * sample;
  }

  const rms = Math.sqrt(sumSquares / Math.max(audio.length, 1));
  if (peak < speechPeakGate || rms < speechRmsGate) return new Float32Array();

  const needsShortSpeechHelp = audio.length < shortSpeechTargetSamples;
  const gain = needsShortSpeechHelp && peak < shortSpeechNormalizePeak
    ? Math.min(shortSpeechNormalizePeak / peak, 3)
    : 1;

  if (!needsShortSpeechHelp && gain === 1) return audio;

  const outputLength = Math.max(audio.length, shortSpeechTargetSamples);
  const output = new Float32Array(outputLength);
  output.set(audio, 0);

  if (gain !== 1) {
    for (let index = 0; index < audio.length; index += 1) {
      output[index] = Math.max(-1, Math.min(1, audio[index] * gain));
    }
  }

  return output;
}

async function startOpenLlmVad() {
  if (!micStream) {
    throw new Error("麦克风音频流尚未准备好。");
  }
  if (vadInstance) return;

  if (!window.vad?.MicVAD) {
    throw new Error("MeloMate 语音检测组件没有加载成功。");
  }

  let instance: MicVadInstance | null = null;
  try {
    instance = await window.vad.MicVAD.new({
      model: "v5",
      stream: micStream,
      preSpeechPadFrames: 30,
      positiveSpeechThreshold: 0.4,
      negativeSpeechThreshold: 0.25,
      redemptionFrames: 40,
      minSpeechFrames: 2,
      baseAssetPath: "./libs/",
      onnxWASMBasePath: "./libs/",
      // Interrupt only after VAD confirms real speech; the provisional start callback can misfire on noise.
      onSpeechRealStart: () => {
        beginUserVoiceInput();
      },
      onSpeechEnd: (audio: Float32Array) => {
        void sendAudioPartition(audio);
      },
      onVADMisfire: () => {
        endUserVoiceInput();
        cancelUserInputPriority();
        pendingUserLine?.remove();
        pendingUserLine = null;
        subtitle.textContent = isCapturing ? "麦克风已启动。" : "麦克风已停止。";
      },
    });

    await instance.start();
    vadInstance = instance;
  } catch (error) {
    if (instance) {
      try {
        instance.pause();
      } catch (pauseError) {
        console.warn("Pausing a partially started VAD instance failed.", pauseError);
      }
      try {
        instance.destroy();
      } catch (destroyError) {
        console.warn("Destroying a partially started VAD instance failed.", destroyError);
      }
    }
    throw error;
  }
}

function stopOpenLlmVad() {
  const instance = vadInstance;
  vadInstance = null;
  if (!instance) return;

  try {
    instance.pause();
  } catch (error) {
    console.warn("Pausing VAD failed while stopping capture.", error);
  }
  try {
    instance.destroy();
  } catch (error) {
    console.warn("Destroying VAD failed while stopping capture.", error);
  }
}

async function startCapture() {
  if (isCapturing || isCaptureStarting) return;
  if (!validateVoiceCloneSettings()) return;
  if (!validateScreenVisionSettings()) return;

  isCaptureStarting = true;
  setCaptureUi(false);
  let failureMessage = "语音捕获启动失败，已停止所有采集。";
  try {
    connectWebSocket();
    if (ws?.readyState === WebSocket.OPEN) {
      if (!(await sendClientVoiceCloneConfig())) {
        failureMessage = "语音克隆配置失败，麦克风和屏幕共享均已自动停止。";
        throw new Error("Voice clone configuration was not applied.");
      }
    }

    const audio: MediaTrackConstraints = {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      ...(micSelect.value ? { deviceId: { exact: micSelect.value } } : {}),
    };
    failureMessage = "没有拿到麦克风权限，允许浏览器使用麦克风后再启动。";
    micStream = await navigator.mediaDevices.getUserMedia({ audio });
    await refreshDevices();

    failureMessage = "屏幕共享未能启动，麦克风已自动停止。";
    if (!(await startScreenVisionIfNeeded())) {
      throw new Error("Screen capture did not start.");
    }

    failureMessage = "语音检测初始化失败，麦克风和屏幕共享均已自动停止。";
    await startOpenLlmVad();
  } catch (error) {
    stopCaptureInternal(false);
    appendLine("system", failureMessage);
    console.warn(error);
    return;
  }

  isCaptureStarting = false;
  setCaptureUi(true);
  pendingUserLine = null;
  appendLine("system", "麦克风已启动，正在等待你说话。");
  subtitle.textContent = "麦克风已启动。";
  resetProactiveSilenceEpisode();
  restartProactiveSpeakLoop();
}

function stopCapture() {
  if (!isCapturing) return;
  stopCaptureInternal(true);
}

function stopCaptureInternal(announce: boolean) {
  const activeMicStream = micStream;
  micStream = null;
  isCaptureStarting = false;
  stopOpenLlmVad();
  stopScreenVision();
  activeMicStream?.getTracks().forEach((track) => {
    try {
      track.stop();
    } catch (error) {
      console.warn("Stopping a microphone track failed.", error);
    }
  });

  setCaptureUi(false);
  subtitle.textContent = "麦克风已停止。";
  pendingUserLine = null;
  pendingUserTranscriptionLines = [];
  isUserSpeaking = false;
  cancelUserInputPriority();
  if (announce) {
    appendLine("system", "麦克风已停止。");
  }
  stopProactiveSpeakLoop();
  completeProactiveTurn();
  resetProactiveSilenceEpisode();
  syncProactiveSpeakButton();
}

async function applyCurrentSettings() {
  if (isApplyingSettings) return;
  isApplyingSettings = true;
  syncApplySettingsButtonState();
  try {
    if (!isWsReady) {
      appendLine("system", "正在重新连接 MeloMate 后端…");
      const connected = await waitForWebSocketReady();
      if (!connected) {
        appendLine("system", "无法连接 MeloMate 后端，请确认 start.bat 窗口仍在运行，然后重试。");
        return;
      }
    }
    await applyCurrentSettingsWhenConnected();
  } finally {
    isApplyingSettings = false;
    syncApplySettingsButtonState();
  }
}

async function applyCurrentSettingsWhenConnected() {
  if (!isWsReady) {
    syncApplySettingsButtonState();
    appendLine("system", "MeloMate 后端连接已断开，请重试。");
    return;
  }
  const wasCapturing = isCapturing;
  if (!validateChatApiSettings()) return;
  if (!validateVoiceCloneSettings()) return;
  if (!validateScreenVisionSettings()) return;
  if (wasCapturing) {
    stopCaptureInternal(false);
  }

  endpointInput.value = normalizeEndpoint(endpointInput.value);
  modelInput.value = normalizeModel(modelInput.value);
  apiKeyInput.value = apiKeyInput.value.trim();
  screenVisionEndpointInput.value = screenVisionEndpointInput.value.trim();
  screenVisionModelInput.value = screenVisionModelInput.value.trim();
  screenVisionApiKeyInput.value = screenVisionApiKeyInput.value.trim();
  screenVisionIntervalInput.value = normalizeScreenVisionInterval(screenVisionIntervalInput.value);
  proactiveIdleSecondsInput.value = normalizeProactiveIdleSeconds(proactiveIdleSecondsInput.value);
  syncVolume(volumeNumber.value);
  const devices = navigator.mediaDevices?.enumerateDevices ? await navigator.mediaDevices.enumerateDevices() : [];
  applyVoiceChatOutputDevice(devices);

  saveSettings();
  connectWebSocket();

  try {
    await applyAudioOutput();
  } catch (error) {
    appendLine("system", error instanceof Error ? error.message : "扬声器设置应用失败。");
    console.warn(error);
  }

  let apiConfigApplied = false;
  let voiceCloneApplied = true;
  if (ws?.readyState === WebSocket.OPEN) {
    if (selectedCharacterConfigFile() !== lastAppliedCharacterConfigFile) {
      sendCharacterConfigSwitch();
    }
    apiConfigApplied = await sendClientApiConfig();
    voiceCloneApplied = await sendClientVoiceCloneConfig();
  }

  if (!apiConfigApplied) {
    appendLine("system", "聊天 API 配置未能安全保存或应用，请检查后重试。");
    syncCredentialUi();
    return;
  }

  settingsPanel.hidden = true;
  settingsButton.setAttribute("aria-expanded", "false");
  syncSettingsPanelMode();
  appendLine(
    "system",
    voiceCloneApplied
      ? wasCapturing
        ? "配置已应用，已按新设置重新启动麦克风。"
        : "配置已应用。"
      : "其他配置已应用，但语音克隆未能启用，当前使用普通语音模式。",
  );
  restartProactiveSpeakLoop();

  if (wasCapturing) {
    await startCapture();
  }
}

async function bootVrmAvatar() {
  const modelOption = selectedVrmModelOption();
  if (!modelOption) return;
  await selectVrmModel(modelOption.id);
}

clearHiddenSystemErrors();

settingsButton.addEventListener("click", () => {
  const isHidden = settingsPanel.hidden;
  settingsPanel.hidden = !isHidden;
  settingsButton.setAttribute("aria-expanded", String(isHidden));
  syncSettingsPanelMode();
  if (isHidden) {
    connectWebSocket();
    void refreshDevices();
    sendWs({ type: "fetch-configs" });
  }
});

backgroundSidebarToggle.addEventListener("click", () => {
  const isOpen = backgroundSidebar.classList.toggle("open");
  backgroundSidebarToggle.setAttribute("aria-expanded", String(isOpen));
  backgroundSidebarToggle.setAttribute("aria-label", isOpen ? "收起素材" : "展开素材");
});

backgroundTab.addEventListener("click", () => setAssetPanelTab("background"));
characterTab.addEventListener("click", () => setAssetPanelTab("character"));
workspaceTab.addEventListener("click", () => setAssetPanelTab("workspace"));

toggleApiKey.addEventListener("click", () => syncSecretToggle(apiKeyInput, toggleApiKey));
toggleScreenVisionApiKey.addEventListener("click", () =>
  syncSecretToggle(screenVisionApiKeyInput, toggleScreenVisionApiKey),
);
clearApiKey.addEventListener("click", () => clearSavedCredential("chat"));
clearScreenVisionApiKey.addEventListener("click", () => clearSavedCredential("screen_vision"));

volumeRange.addEventListener("input", () => {
  syncVolume(volumeRange.value);
  saveVolumeSetting();
});
volumeMuteToggle.addEventListener("click", toggleMuteVolume);
voiceChatOutputToggle.addEventListener("change", () => {
  if (voiceChatOutputToggle.checked) {
    void askToOpenVoicemeeter();
  }
  syncVoiceChatOutputHint();
  syncVoiceChatMicSelection();
  void refreshDevices();
});
voiceChatOutputSelect.addEventListener("change", () => {
  voiceChatOutputSinkId = "";
  void refreshDevices();
});
testVoiceChatOutput.addEventListener("click", () => {
  void testVoiceChatOutputRoute();
});
showVoicemeeter.addEventListener("click", () => {
  void showVoicemeeterWindow();
});
videoFullscreenButton.addEventListener("click", () => {
  void toggleVideoFullscreen();
});
document.addEventListener("fullscreenchange", () => {
  isFallbackVideoFullscreen = false;
  syncVideoFullscreenState(document.fullscreenElement === videoFrame);
});
characterSelect.addEventListener("change", () => {
  selectCharacterConfigFile(characterSelect.value);
});
voiceCloneToggle.addEventListener("change", syncVoiceCloneControls);
screenVisionToggle.addEventListener("change", syncScreenVisionControls);
screenVisionIntervalInput.addEventListener("change", () => {
  screenVisionIntervalInput.value = normalizeScreenVisionInterval(screenVisionIntervalInput.value);
});
proactiveSpeakToggle.addEventListener("change", () => {
  syncProactiveSpeakControls();
});
proactiveIdleSecondsInput.addEventListener("change", () => {
  proactiveIdleSecondsInput.value = normalizeProactiveIdleSeconds(proactiveIdleSecondsInput.value);
});
referenceAudioInput.addEventListener("change", () => {
  const file = referenceAudioInput.files?.[0];
  if (!file) {
    syncVoiceCloneControls();
    return;
  }
  if (!validateVoiceCloneReference(file, file.name)) {
    referenceAudioInput.value = "";
    setReferenceAudioPreview(null);
    syncVoiceCloneControls();
    return;
  }

  setReferenceAudioPreview(file, file.name);
  syncVoiceCloneControls();
  if (voiceCloneToggle.checked && isWsReady) {
    void sendClientVoiceCloneConfig();
  }
});
applySettings.addEventListener("click", applyCurrentSettings);

startButton.addEventListener("click", () => {
  void startCapture();
});

stopButton.addEventListener("click", stopCapture);
proactiveSpeakButton.addEventListener("click", () => {
  void requestProactiveSpeak("manual", true);
});

savedSettings = normalizeStartupSettings(readSavedSettings());
if (savedSettings) {
  persistSavedSettings();
  restoreStaticSettings(savedSettings);
} else {
  renderCharacterOptions([]);
  endpointInput.value = defaultApiEndpoint;
  endpointInput.readOnly = false;
  modelInput.value = defaultModel;
  screenVisionEndpointInput.value = defaultScreenVisionEndpoint;
  screenVisionModelInput.value = defaultScreenVisionModel;
  screenVisionIntervalInput.value = "5";
  proactiveIdleSecondsInput.value = defaultProactiveIdleSeconds;
  syncScreenVisionControls();
  syncProactiveSpeakControls();
}

async function startup() {
  openSettingsPanel();
  connectWebSocket();
  setAssetPanelTab(activeAssetPanelTab, false);
  await setupBackgroundPicker();
  vrmModelOptions = await readVrmModelOptions();
  renderVrmModelOptions();
  await purgeLegacyReferenceAudioStorage();
  await refreshDevices();
  setCaptureUi(false);
  syncVolume(volumeNumber.value);
  syncVoiceChatOutputHint();
  syncScreenVisionControls();
  syncProactiveSpeakControls();
  syncProactiveSpeakButton();
  restartProactiveSpeakLoop();
  await bootVrmAvatar();
}

void startup();

