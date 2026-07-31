import { initializeLive2D } from "../WebSDK/src/main";
import { LAppAdapter } from "../WebSDK/src/lappadapter";
import { updateModelConfig } from "../WebSDK/src/lappdefine";
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
  live2dModelId?: string;
  voiceChatOutputEnabled?: boolean;
  voiceChatOutputDeviceId?: string;
  voiceCloneEnabled?: boolean;
  referenceAudioName?: string;
  screenVisionEnabled?: boolean;
  screenVisionEndpoint?: string;
  screenVisionModel?: string;
  screenVisionApiKey?: string;
  screenVisionIntervalSec?: string;
  proactiveSpeakEnabled?: boolean;
  proactiveIdleSeconds?: string;
};

type MicVadInstance = {
  start: () => void;
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

type WsMessage = {
  type?: string;
  text?: string;
  texts?: string[];
  input_id?: string;
  message?: string;
  audio?: string | null;
  volumes?: number[];
  slice_length?: number;
  display_text?: DisplayText;
  forwarded?: boolean;
  conf_name?: string;
  character_name?: string;
  client_uid?: string;
  success?: boolean;
  enabled?: boolean;
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

type Live2DModelOption = {
  id: string;
  name: string;
  directory: string;
  fileName: string;
  scale: number;
};

type AssetPanelTab = "background" | "character" | "workspace";

declare global {
  interface Window {
    vad?: VadModule;
  }
}

const settingsStorageKey = "melomate-settings";
const backgroundManifestUrl = "/api/backgrounds";
const live2DModelManifestUrl = "/api/live2d-models";
const workspaceManifestUrl = "/api/workspace";
const workspaceStateUrl = "/api/workspace-state";
const workspaceEventsUrl = "/api/workspace-events";
const fallbackBackgrounds: BackgroundOption[] = [{ name: "Default", url: "/backgrounds/default.svg" }];
const defaultCharacterConfigFile = "小可.yaml";
const defaultCharacterOption: CharacterConfigOption = { filename: defaultCharacterConfigFile };
const defaultLive2DModelId = "epsilon_free";
const fallbackLive2DModelOptions: Live2DModelOption[] = [
  {
    id: "epsilon_free",
    name: "Epsilon",
    directory: "epsilon_free/runtime",
    fileName: "Epsilon_free",
    scale: 0.9,
  },
  {
    id: "mao_pro",
    name: "Mao Pro",
    directory: "mao_pro/runtime",
    fileName: "mao_pro",
    scale: 0.9,
  },
];
let live2dModelOptions: Live2DModelOption[] = fallbackLive2DModelOptions;
const referenceAudioDbName = "melomate-reference-audio";
const referenceAudioStoreName = "files";
const referenceAudioRecordKey = "last-reference";
const moonshotApiEndpoint = "https://api.moonshot.cn/v1";
const defaultApiEndpoint = "https://api.deepseek.com";
const defaultModel = "deepseek-chat";
const defaultScreenVisionEndpoint = moonshotApiEndpoint;
const defaultScreenVisionModel = "moonshot-v1-8k-vision-preview";
const openLlmWsUrl = "ws://127.0.0.1:12393/client-ws";
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
const proactiveSpeakCheckIntervalMs = 10_000;
const proactiveSpeakChance = 0.35;
const workspaceEventPollMs = 700;
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
const screenVisionToggle = document.querySelector<HTMLInputElement>("#screenVisionToggle")!;
const screenVisionEndpointInput = document.querySelector<HTMLInputElement>("#screenVisionEndpoint")!;
const screenVisionModelInput = document.querySelector<HTMLInputElement>("#screenVisionModel")!;
const screenVisionApiKeyInput = document.querySelector<HTMLInputElement>("#screenVisionApiKey")!;
const toggleScreenVisionApiKey = document.querySelector<HTMLButtonElement>("#toggleScreenVisionApiKey")!;
const screenVisionIntervalInput = document.querySelector<HTMLInputElement>("#screenVisionInterval")!;
const proactiveSpeakToggle = document.querySelector<HTMLInputElement>("#proactiveSpeakToggle")!;
const proactiveIdleSecondsInput = document.querySelector<HTMLInputElement>("#proactiveIdleSeconds")!;
const voiceCloneToggle = document.querySelector<HTMLInputElement>("#voiceCloneToggle")!;
const referenceAudioInput = document.querySelector<HTMLInputElement>("#referenceAudioInput")!;
const referenceAudioPlayer = document.querySelector<HTMLAudioElement>("#referenceAudioPlayer")!;
const referenceAudioName = document.querySelector<HTMLSpanElement>("#referenceAudioName")!;
const applySettings = document.querySelector<HTMLButtonElement>("#applySettings")!;
const applySettingsDefaultText = applySettings.textContent?.trim() || "应用配置";
const applySettingsLoadingText = "正在加载";
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

type SinkAudioElement = HTMLAudioElement & {
  setSinkId?: (sinkId: string) => Promise<void>;
};

let micStream: MediaStream | null = null;
let vadInstance: MicVadInstance | null = null;
let ws: WebSocket | null = null;
let isCapturing = false;
let isWsReady = false;
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
let backendSynthComplete = false;
let lastAssistantText = "";
let heardAssistantText = "";
let audioQueueVersion = 0;
let isAssistantResponding = false;
let isUserSpeaking = false;
let isUserInputPriorityActive = false;
let isUserVoiceTurnSubmitted = false;
let hasSentInterruptForCurrentUserInput = false;
let referenceAudioBlob: Blob | null = null;
let referenceAudioStoredName = "";
let referenceAudioObjectUrl = "";
let backgroundOptions: BackgroundOption[] = [];
let characterOptions: CharacterConfigOption[] = [];
let activeAssetPanelTab: AssetPanelTab = "background";
let currentWorkspaceFolder = "";
let expandedWorkspaceFolders = new Set<string>();
let workspaceEntriesCache = new Map<string, WorkspaceEntry[]>();
let workspaceControlStatus: WorkspaceControlStatus = { label: "未连接", tone: "missing" };
let workspaceEventTimer = 0;
let lastWorkspaceEventMs = Date.now();
let workspaceBroadcastChannel: BroadcastChannel | null = null;
const handledWorkspaceEventIds = new Set<string>();
let lastAppliedCharacterConfigFile = "";
let currentAssistantName = "小可";
let activeLive2DModelId = "";
let pendingLive2DModelId = "";
let isLive2DModelSwitching = false;
let voiceChatOutputSinkId = "";
let isSettingsReadOnly = false;
let lastAudibleVolume = 100;
let screenStream: MediaStream | null = null;
let screenVideo: HTMLVideoElement | null = null;
let screenCaptureTimer = 0;
let latestScreenImage: string | null = null;
let screenShareWarningShown = false;
let lastConversationActivityAt = Date.now();
let proactiveSpeakTimer = 0;
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
    return rawValue ? (JSON.parse(rawValue) as SavedSettings) : null;
  } catch (error) {
    console.warn(error);
    return null;
  }
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
    live2dModelId: selectedLive2DModelOption().id,
    voiceChatOutputEnabled: voiceChatOutputToggle.checked && !voiceChatOutputToggle.disabled,
    voiceChatOutputDeviceId: voiceChatOutputSelect.value,
    voiceCloneEnabled: voiceCloneToggle.checked,
    referenceAudioName:
      referenceAudioInput.files?.[0]?.name || referenceAudioStoredName || savedSettings?.referenceAudioName || "",
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
  localStorage.setItem(settingsStorageKey, JSON.stringify(savedSettings));
}

function saveBackground(url: string) {
  savedSettings = {
    ...(savedSettings || currentSettings()),
    backgroundUrl: url,
  };
  localStorage.setItem(settingsStorageKey, JSON.stringify(savedSettings));
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
    const response = await fetch(backgroundManifestUrl, { cache: "no-store" });
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
  const response = await fetch(`${workspaceManifestUrl}?${params.toString()}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`工作区加载失败：${response.status}`);
  return (await response.json()) as { entries?: WorkspaceEntry[]; folder?: string };
}

async function readWorkspaceControlStatus(): Promise<WorkspaceControlStatus> {
  const params = new URLSearchParams({ persona: workspacePersonaName() });
  const response = await fetch(`${workspaceStateUrl}?${params.toString()}`, { cache: "no-store" });
  if (!response.ok) return { label: "未连接", tone: "missing" };
  const data = (await response.json()) as { state?: { updated_ms?: number; state?: { protocolAvailable?: boolean } } | null };
  if (!data.state) return { label: "未连接", tone: "missing" };

  const ageMs = Date.now() - Number(data.state.updated_ms || 0);
  if (!Number.isFinite(ageMs) || ageMs > 5000) return { label: "状态过期", tone: "stale" };
  if (!data.state.state?.protocolAvailable) return { label: "协议缺失", tone: "missing" };
  return { label: "可控制", tone: "ready" };
}

function workspaceFileUrl(path: string) {
  const persona = encodeURIComponent(workspacePersonaName());
  const filePath = path
    .split("/")
    .filter(Boolean)
    .map((part) => encodeURIComponent(part))
    .join("/");
  return `/workspace-files/${persona}/${filePath}`;
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
      void toggleWorkspaceFolder(entry.path);
    });
  } else {
    arrow.textContent = "";
    row.addEventListener("click", () => {
      window.open(workspaceFileUrl(entry.path), "_blank", "noopener");
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

function selectedLive2DModelOption() {
  return (
    live2dModelOptions.find((option) => option.id === savedSettings?.live2dModelId) ||
    live2dModelOptions.find((option) => option.id === defaultLive2DModelId) ||
    live2dModelOptions[0] ||
    fallbackLive2DModelOptions[0]
  );
}

function saveLive2DModel(id: string) {
  activeLive2DModelId = id;
  savedSettings = {
    ...(savedSettings || currentSettings()),
    live2dModelId: id,
  };
  localStorage.setItem(settingsStorageKey, JSON.stringify(savedSettings));
}

function syncLive2DModelActiveState() {
  const selectedId = pendingLive2DModelId || activeLive2DModelId || selectedLive2DModelOption().id;
  characterList.querySelectorAll<HTMLButtonElement>(".character-item").forEach((button) => {
    const isActive = button.dataset.modelId === selectedId;
    const isLoading = isLive2DModelSwitching && button.dataset.modelId === pendingLive2DModelId;
    button.classList.toggle("active", isActive);
    button.classList.toggle("loading", isLoading);
    button.setAttribute("aria-pressed", String(isActive));
    button.disabled = isLive2DModelSwitching;
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

function live2DModelJsonUrl(option: Live2DModelOption) {
  return `/models/${option.directory}/${option.fileName}.model3.json`;
}

async function readLive2DModelOptions() {
  try {
    const response = await fetch(live2DModelManifestUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`Live2D manifest failed: ${response.status}`);
    const data = (await response.json()) as { models?: Live2DModelOption[] };
    const models = data.models?.filter((option) => option.id && option.directory && option.fileName);
    return models?.length ? models : fallbackLive2DModelOptions;
  } catch (error) {
    console.warn(error);
    return fallbackLive2DModelOptions;
  }
}

async function ensureLive2DModelAvailable(option: Live2DModelOption) {
  const response = await fetch(live2DModelJsonUrl(option), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`模型文件加载失败：${option.name}`);
  }
}

async function settleLive2DModelLayout() {
  refreshLive2DLayout();
  await new Promise((resolve) => window.setTimeout(resolve, 180));
  refreshLive2DLayout();
}

async function selectLive2DModel(id: string) {
  const option = live2dModelOptions.find((modelOption) => modelOption.id === id);
  if (!option) return;
  if (isLive2DModelSwitching || id === activeLive2DModelId) return;

  const previousModelId = activeLive2DModelId || selectedLive2DModelOption().id;
  isLive2DModelSwitching = true;
  pendingLive2DModelId = option.id;
  syncLive2DModelActiveState();

  try {
    await ensureLive2DModelAvailable(option);
    updateModelConfig("/models/", option.directory, option.fileName, option.scale);
    LAppAdapter.getInstance().setChara("/models/" + option.directory, option.fileName);
    saveLive2DModel(option.id);
    await settleLive2DModelLayout();
  } catch (error) {
    console.warn(error);
    pendingLive2DModelId = previousModelId;
    appendLine("system", error instanceof Error ? error.message : "模型切换失败。");
  } finally {
    isLive2DModelSwitching = false;
    pendingLive2DModelId = "";
    syncLive2DModelActiveState();
  }
}

function renderLive2DModelOptions() {
  characterList.textContent = "";

  live2dModelOptions.forEach((option) => {
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
    meta.textContent = option.id;

    item.append(portrait, label, meta);
    item.addEventListener("click", () => {
      void selectLive2DModel(option.id);
    });
    characterList.appendChild(item);
  });

  syncLive2DModelActiveState();
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

function openReferenceAudioDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(referenceAudioDbName, 1);
    request.addEventListener("upgradeneeded", () => {
      request.result.createObjectStore(referenceAudioStoreName);
    });
    request.addEventListener("success", () => resolve(request.result));
    request.addEventListener("error", () => reject(request.error || new Error("Reference audio DB failed.")));
  });
}

async function readStoredReferenceAudio(): Promise<{ name: string; type: string; blob: Blob } | null> {
  const db = await openReferenceAudioDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(referenceAudioStoreName, "readonly");
    const request = tx.objectStore(referenceAudioStoreName).get(referenceAudioRecordKey);
    request.addEventListener("success", () => {
      const value = request.result as { name: string; type: string; blob: Blob } | undefined;
      resolve(value || null);
      db.close();
    });
    request.addEventListener("error", () => {
      reject(request.error || new Error("Reference audio read failed."));
      db.close();
    });
  });
}

async function writeStoredReferenceAudio(file: File) {
  const db = await openReferenceAudioDb();
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(referenceAudioStoreName, "readwrite");
    tx.objectStore(referenceAudioStoreName).put(
      {
        name: file.name,
        type: file.type || "audio/wav",
        blob: file,
      },
      referenceAudioRecordKey,
    );
    tx.addEventListener("complete", () => {
      resolve();
      db.close();
    });
    tx.addEventListener("error", () => {
      reject(tx.error || new Error("Reference audio save failed."));
      db.close();
    });
  });
}

async function deleteStoredReferenceAudio() {
  const db = await openReferenceAudioDb();
  return new Promise<void>((resolve, reject) => {
    const tx = db.transaction(referenceAudioStoreName, "readwrite");
    tx.objectStore(referenceAudioStoreName).delete(referenceAudioRecordKey);
    tx.addEventListener("complete", () => {
      resolve();
      db.close();
    });
    tx.addEventListener("error", () => {
      reject(tx.error || new Error("Reference audio delete failed."));
      db.close();
    });
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

async function restoreReferenceAudio() {
  if (!savedSettings?.voiceCloneEnabled) return;

  try {
    const stored = await readStoredReferenceAudio();
    if (stored) {
      setReferenceAudioPreview(stored.blob, stored.name);
      savedSettings.referenceAudioName = stored.name;
      localStorage.setItem(settingsStorageKey, JSON.stringify(savedSettings));
      syncVoiceCloneControls();
      return;
    }
  } catch (error) {
    console.warn(error);
  }

}

function syncVoiceCloneControls() {
  const enabled = voiceCloneToggle.checked;
  setCollapsedGroup("voice-clone", !enabled);
  referenceAudioInput.disabled = isSettingsReadOnly || !enabled;
  referenceAudioPlayer.classList.toggle("disabled-audio", isSettingsReadOnly || !enabled || !referenceAudioPlayer.src);
  if (isSettingsReadOnly) {
    referenceAudioPlayer.pause();
  }
  if (!enabled) {
    referenceAudioInput.value = "";
    setReferenceAudioPreview(null);
    referenceAudioName.textContent = "未选择参考音频";
  } else if (!referenceAudioInput.files?.[0] && !referenceAudioBlob) {
    referenceAudioName.textContent = savedSettings?.referenceAudioName || "请选择 3-10 秒参考音频";
  }
}

function validateVoiceCloneSettings() {
  if (!voiceCloneToggle.checked) return true;
  if (referenceAudioInput.files?.[0]) return true;
  if (referenceAudioBlob) return true;
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
  markConversationActivity();
  if (inputId && displayedUserInputIds.has(inputId)) {
    discardPendingUserLine(inputId);
    return false;
  }

  const normalizedText = normalizeUserDisplayText(text);
  if (wasUserTranscriptionRecentlyDisplayed(normalizedText)) {
    if (inputId) {
      discardPendingUserLine(inputId);
    }
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
  markConversationActivity();
}

function setCaptureUi(active: boolean) {
  isCapturing = active;
  startButton.disabled = active;
  stopButton.disabled = !active;
  status.textContent = active ? "捕捉中" : "已停止";
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
  localStorage.setItem(settingsStorageKey, JSON.stringify(savedSettings));
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
    const response = await fetch("/api/open-voicemeeter", { method: "POST" });
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
    const response = await fetch("/api/show-voicemeeter", { method: "POST" });
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
  isSettingsReadOnly = isCapturing && isSettingsOpen;
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
  syncApplySettingsButtonState();
  syncProactiveSpeakButton();
}

function syncApplySettingsButtonState() {
  const isBackendLoading = !isWsReady;
  const shouldDisable = isSettingsReadOnly || isBackendLoading;
  applySettings.disabled = shouldDisable;
  applySettings.setAttribute("aria-disabled", String(shouldDisable));
  applySettings.classList.toggle("is-loading", isBackendLoading);
  applySettings.textContent = isBackendLoading ? applySettingsLoadingText : applySettingsDefaultText;
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

function refreshLive2DLayout() {
  window.setTimeout(() => {
    window.dispatchEvent(new Event("resize"));
  }, 80);
}

function syncVideoFullscreenState(active: boolean) {
  isVideoFullscreen = active;
  document.body.classList.toggle("video-fullscreen-active", active);
  appShell.classList.toggle("video-fullscreen-fallback", active && isFallbackVideoFullscreen);
  videoFrame.classList.toggle("video-fullscreen", active);
  videoFullscreenButton.textContent = active ? "×" : "⛶";
  videoFullscreenButton.setAttribute("aria-label", active ? "退出全屏" : "进入全屏");
  refreshLive2DLayout();
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

  await responseAudio.setSinkId(speakerSelect.value);
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
  }
}

function workspaceEventPrompt(event: WorkspaceEvent) {
  const pageName = event.page?.title || event.page?.path || "工作区页面";
  if (event.type === "workspace-page-closed") {
    return `工作区事件：用户关闭了「${pageName}」。请像同桌朋友一样自然回应，不要声称还能继续控制这个已关闭页面。`;
  }

  const stateText = JSON.stringify(event.appState ?? null);
  return [
    `工作区事件：「${pageName}」发生了新变化。`,
    "你正在观察这个固定页面实例。请像现实中同桌互动一样自然回应。",
    "如果这是游戏/棋局，且状态显示轮到你、currentTurn 是你、或 availableActions 非空，不要等待用户提问：立刻先用 read_workspace_state 确认状态，再用 send_workspace_action 下下一步，确认成功后再说话。",
    "如果还没轮到你，只简短回应你看到了变化。没确认行动成功时不要声称已行动。",
    `最新状态：${stateText.slice(0, 1800)}`,
  ].join("\n");
}

async function readWorkspaceEvents() {
  const params = new URLSearchParams({
    persona: workspacePersonaName(),
    since: String(lastWorkspaceEventMs),
  });
  const response = await fetch(`${workspaceEventsUrl}?${params.toString()}`, { cache: "no-store" });
  if (!response.ok) return [];
  const data = (await response.json()) as { events?: WorkspaceEvent[] };
  return data.events || [];
}

function shouldForwardWorkspaceEvent(event: WorkspaceEvent) {
  if (!event.id || handledWorkspaceEventIds.has(event.id)) return false;
  if (event.actionEvent && event.lastAction?.accepted !== false) return false;
  return true;
}

function forwardWorkspaceEvent(event: WorkspaceEvent) {
  if (!isWsReady || !shouldForwardWorkspaceEvent(event)) return;
  handledWorkspaceEventIds.add(event.id);
  if (handledWorkspaceEventIds.size > 200) handledWorkspaceEventIds.clear();
  markConversationActivity();
  setThinking(true);
  sendWs({
    type: "text-input",
    text: workspaceEventPrompt(event),
    metadata: {
      workspace_event: true,
      skip_memory: true,
      skip_history: true,
    },
  });
}

async function pollWorkspaceEvents() {
  if (!isWsReady) return;
  try {
    const events = await readWorkspaceEvents();
    for (const event of events.slice(-4)) {
      lastWorkspaceEventMs = Math.max(lastWorkspaceEventMs, Number(event.created_ms || 0));
      forwardWorkspaceEvent(event);
    }
  } catch (error) {
    console.warn("Workspace event poll failed.", error);
  }
}

function startWorkspaceEventLoop() {
  if (workspaceEventTimer) return;
  if (!workspaceBroadcastChannel && "BroadcastChannel" in window) {
    workspaceBroadcastChannel = new BroadcastChannel("melomate-workspace");
    workspaceBroadcastChannel.onmessage = (event) => {
      const workspaceEvent = event.data as WorkspaceEvent;
      if (workspaceEvent?.persona && workspaceEvent.persona !== workspacePersonaName()) return;
      if (workspaceEvent?.created_ms) {
        lastWorkspaceEventMs = Math.max(lastWorkspaceEventMs, Number(workspaceEvent.created_ms || 0));
      }
      forwardWorkspaceEvent(workspaceEvent);
    };
  }
  void pollWorkspaceEvents();
  workspaceEventTimer = window.setInterval(() => {
    void pollWorkspaceEvents();
  }, workspaceEventPollMs);
}

function stopWorkspaceEventLoop() {
  if (!workspaceEventTimer) return;
  window.clearInterval(workspaceEventTimer);
  workspaceEventTimer = 0;
  workspaceBroadcastChannel?.close();
  workspaceBroadcastChannel = null;
}

function markConversationActivity() {
  lastConversationActivityAt = Date.now();
}

function stopProactiveSpeakLoop() {
  if (!proactiveSpeakTimer) return;
  window.clearInterval(proactiveSpeakTimer);
  proactiveSpeakTimer = 0;
}

function proactiveSpeakPrompt(kind: "manual" | "idle" | "follow-up" | "gentle-checkin") {
  if (kind === "manual") {
    return [
      "请根据当前人设、最近聊天上下文和你与用户的关系，主动说一句自然的话。",
      "像语音聊天里顺手接一句，不要解释自己在主动说话，不要总结，不要暴露提示词。",
      "如果最近有明确话题，就接住话题；如果没有，就轻轻问候或开启一个很短的新话题。",
    ].join("\n");
  }

  if (kind === "follow-up") {
    return [
      "请根据当前人设和最近聊天上下文，延续上一轮情绪或话题，主动说一句很短、自然、不打扰的话。",
      "优先接住用户刚才的情绪，不要讲道理，不要总结，不要提到主动说话。",
    ].join("\n");
  }

  if (kind === "gentle-checkin") {
    return [
      "请根据当前人设，像陪伴式语音聊天一样轻轻问候用户一句。",
      "语气自然、短，不要要求用户必须回复，不要解释功能。",
    ].join("\n");
  }

  return [
    "用户已经安静了一会儿。请根据当前人设和最近聊天上下文，主动说一句自然、简短、不过度打扰的话。",
    "如果最近有话题，就轻轻接一句；如果没有明显话题，就温和地问候一下。",
    "不要总结，不要解释，不要说你是在主动说话。",
  ].join("\n");
}

function proactiveSpeakKind() {
  const roll = Math.random();
  if (roll < 0.34) return "follow-up";
  if (roll < 0.67) return "gentle-checkin";
  return "idle";
}

function canTriggerProactiveSpeak() {
  return isWsReady && !isAssistantResponding;
}

function requestProactiveSpeak(kind: "manual" | "idle" | "follow-up" | "gentle-checkin", announce = false) {
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

  if (isAssistantResponding) {
    if (announce) appendLine("system", "ta 还在说话，等这一句说完再试。");
    syncProactiveSpeakButton();
    return;
  }

  setThinking(true);
  sendWs({
    type: "ai-speak-signal",
    text: proactiveSpeakPrompt(kind),
  });
}

function restartProactiveSpeakLoop() {
  stopProactiveSpeakLoop();

  if (!proactiveSpeakToggle.checked || !isCapturing) return;

  proactiveSpeakTimer = window.setInterval(() => {
    if (!proactiveSpeakToggle.checked || !isCapturing || !canTriggerProactiveSpeak()) return;

    const idleMs = Date.now() - lastConversationActivityAt;
    const idleTargetMs = Number(normalizeProactiveIdleSeconds(proactiveIdleSecondsInput.value)) * 1000;
    if (idleMs < idleTargetMs) return;
    if (Math.random() > proactiveSpeakChance) return;

    requestProactiveSpeak(proactiveSpeakKind());
  }, proactiveSpeakCheckIntervalMs);
}

function sendClientApiConfig() {
  const settings = currentSettings();
  sendWs({
    type: "client-api-config",
    api_base_url: settings.endpoint,
    api_key: settings.apiKey,
    model: settings.model,
  });
}

function sendCharacterConfigSwitch() {
  const file = selectedCharacterConfigFile();
  if (!file) return;
  sendWs({ type: "switch-config", file });
  lastAppliedCharacterConfigFile = file;
}

async function sendClientVoiceCloneConfig() {
  if (!voiceCloneToggle.checked) {
    sendWs({ type: "client-voice-clone-config", enabled: false });
    return;
  }

  const audio = referenceAudioInput.files?.[0] || referenceAudioBlob;
  if (!audio) return;

  const audioBase64 = await readReferenceAudioAsDataUrl(audio);
  sendWs({
    type: "client-voice-clone-config",
    enabled: true,
    audio_base64: audioBase64,
    file_name:
      referenceAudioInput.files?.[0]?.name ||
      referenceAudioStoredName ||
      savedSettings?.referenceAudioName ||
      "reference.wav",
  });
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

  ws = new WebSocket(openLlmWsUrl);
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
    sendClientApiConfig();
    void sendClientVoiceCloneConfig();
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

function handleControlMessage(text: string) {
  if (text === "conversation-chain-start") {
    lastAssistantText = "";
    heardAssistantText = "";
    isAssistantResponding = true;
    setThinking(true);
    return;
  }

  if (text === "conversation-chain-end") {
    if (shouldSuppressAssistantOutput()) return;
    void finishBackendAudio();
  }
}

function shouldSuppressAssistantOutput() {
  return isUserInputPriorityActive;
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
}

function handleWsMessage(message: WsMessage) {
  if (message.type === "control" && message.text) {
    handleControlMessage(message.text);
    return;
  }

  if (message.type === "full-text") {
    if (shouldSuppressAssistantOutput()) return;
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
    if (shouldSuppressAssistantOutput()) return;
    queueAudioMessage(message);
    return;
  }

  if (message.type === "backend-synth-complete") {
    if (shouldSuppressAssistantOutput()) return;
    backendSynthComplete = true;
    void finishBackendAudio();
    return;
  }

  if (message.type === "interrupt-signal") {
    stopCurrentResponsePlayback();
    return;
  }

  if (message.type === "voice-clone-config-applied") {
    return;
  }

  if (message.type === "config-files") {
    renderCharacterOptions(message.configs || []);
    return;
  }

  if (message.type === "config-switched") {
    sendClientApiConfig();
    void sendClientVoiceCloneConfig();
    return;
  }

  if (message.type === "error") {
    if (isHiddenSystemError(message.message)) {
      console.warn(message.message);
    } else {
      appendLine("system", message.message || "MeloMate 后端返回错误。");
    }
    setThinking(false);
    return;
  }

  if (message.type === "set-model-and-conf") {
    setCurrentAssistantName(message.character_name || message.conf_name);
    if (message.conf_name) {
      console.info("MeloMate config:", message.conf_name, message.client_uid);
    }
  }
}

function queueAudioMessage(message: WsMessage) {
  if (shouldSuppressAssistantOutput()) return;

  const text = message.display_text?.text || "";
  const queueVersion = audioQueueVersion;
  audioQueue = audioQueue.then(async () => {
    if (queueVersion !== audioQueueVersion || shouldSuppressAssistantOutput()) return;

    if (text) {
      appendAssistantLine(text, message.display_text?.name);
      if (!message.forwarded) {
        sendWs({
          type: "audio-play-start",
          display_text: message.display_text,
          forwarded: true,
        });
      }
    }

    if (message.audio) {
      await playBackendAudio(message.audio, queueVersion);
    }
  });
}

async function playBackendAudio(audioBase64: string, queueVersion: number) {
  if (queueVersion !== audioQueueVersion || shouldSuppressAssistantOutput()) return;

  responseAudio.pause();
  voiceChatAudio.pause();
  setAnswering(true);

  const audioSource = `data:audio/wav;base64,${audioBase64}`;
  responseAudio.src = audioSource;
  await applyAudioOutput();
  if (queueVersion !== audioQueueVersion || shouldSuppressAssistantOutput()) return;

  const playbackTasks = [playAudioElement(responseAudio)];
  const canUseVoiceChatOutput = await applyVoiceChatAudioOutput();
  if (queueVersion !== audioQueueVersion || shouldSuppressAssistantOutput()) return;

  if (canUseVoiceChatOutput) {
    voiceChatAudio.src = audioSource;
    playbackTasks.push(playAudioElement(voiceChatAudio));
  }

  await Promise.all(playbackTasks);
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
    api_key: screenVisionApiKeyInput.value.trim(),
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
  if (!screenVisionApiKeyInput.value.trim()) {
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
  screenStream?.getTracks().forEach((track) => track.stop());
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
  if (!backendSynthComplete) return;
  backendSynthComplete = false;
  await audioQueue.catch(() => undefined);
  sendWs({ type: "frontend-playback-complete" });
  setThinking(false);
  isAssistantResponding = false;
  markConversationActivity();
  syncProactiveSpeakButton();
}

function stopCurrentResponsePlayback(force = false) {
  const hasActivePlayback = !responseAudio.paused || !voiceChatAudio.paused;
  if (!force && !isAssistantResponding && !hasActivePlayback) return;

  isAssistantResponding = false;
  backendSynthComplete = false;
  audioQueueVersion += 1;
  audioQueue = Promise.resolve();

  responseAudio.pause();
  responseAudio.removeAttribute("src");
  responseAudio.load();
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
  stopAssistantReplyForUserInput(true);
  setPendingUserLine(listeningDisplayText);
  ensurePendingUserInputId();
  subtitle.textContent = listeningDisplayText;
  setAssistantStatus("listening");
  markConversationActivity();
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
  endUserVoiceInput();
  isUserVoiceTurnSubmitted = true;
  sendWs({
    type: "mic-audio-end",
    input_id: inputId,
    images: await screenImagesForNextTurn(),
    screen_vision: screenVisionConfigPayload(),
  });
  markConversationActivity();
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
  if (!micStream || vadInstance) return;

  if (!window.vad?.MicVAD) {
    appendLine("system", "MeloMate 语音检测组件没有加载成功。");
    return;
  }

  vadInstance = await window.vad.MicVAD.new({
    model: "v5",
    stream: micStream,
    preSpeechPadFrames: 30,
    positiveSpeechThreshold: 0.4,
    negativeSpeechThreshold: 0.25,
    redemptionFrames: 40,
    minSpeechFrames: 2,
    baseAssetPath: "./libs/",
    onnxWASMBasePath: "./libs/",
    onSpeechStart: () => {
      beginUserVoiceInput();
    },
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

  vadInstance.start();
}

function stopOpenLlmVad() {
  vadInstance?.pause();
  vadInstance?.destroy();
  vadInstance = null;
}

async function startCapture() {
  if (isCapturing) return;
  if (!validateVoiceCloneSettings()) return;
  if (!validateScreenVisionSettings()) return;
  connectWebSocket();
  if (ws?.readyState === WebSocket.OPEN) {
    await sendClientVoiceCloneConfig();
  }
  if (!(await startScreenVisionIfNeeded())) return;

  try {
    const audio: MediaTrackConstraints = {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      ...(micSelect.value ? { deviceId: { exact: micSelect.value } } : {}),
    };
    micStream = await navigator.mediaDevices.getUserMedia({ audio });
    await refreshDevices();
  } catch (error) {
    appendLine("system", "没有拿到麦克风权限，允许浏览器使用麦克风后再启动。");
    console.warn(error);
    return;
  }

  pendingUserLine = null;
  appendLine("system", "麦克风已启动，正在等待你说话。");
  subtitle.textContent = "麦克风已启动。";
  markConversationActivity();
  setCaptureUi(true);
  restartProactiveSpeakLoop();

  void startOpenLlmVad();
}

function stopCapture() {
  if (!isCapturing) return;
  stopCaptureInternal(true);
}

function stopCaptureInternal(announce: boolean) {
  setCaptureUi(false);
  subtitle.textContent = "麦克风已停止。";
  pendingUserLine = null;
  pendingUserTranscriptionLines = [];
  isUserSpeaking = false;
  cancelUserInputPriority();
  if (announce) {
    appendLine("system", "麦克风已停止。");
  }

  stopOpenLlmVad();
  stopScreenVision();
  micStream?.getTracks().forEach((track) => track.stop());
  micStream = null;
  stopProactiveSpeakLoop();
  markConversationActivity();
  syncProactiveSpeakButton();
}

async function applyCurrentSettings() {
  if (!isWsReady) {
    syncApplySettingsButtonState();
    appendLine("system", "MeloMate 后端还在连接中，请稍后再应用配置。");
    return;
  }
  const wasCapturing = isCapturing;
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

  if (voiceCloneToggle.checked && referenceAudioInput.files?.[0]) {
    try {
      await writeStoredReferenceAudio(referenceAudioInput.files[0]);
      referenceAudioBlob = referenceAudioInput.files[0];
      referenceAudioStoredName = referenceAudioInput.files[0].name;
    } catch (error) {
      appendLine("system", "参考音频保存失败，下次打开可能需要重新选择。");
      console.warn(error);
    }
  } else if (!voiceCloneToggle.checked) {
    try {
      await deleteStoredReferenceAudio();
    } catch (error) {
      console.warn(error);
    }
  }

  saveSettings();
  connectWebSocket();

  try {
    await applyAudioOutput();
  } catch (error) {
    appendLine("system", error instanceof Error ? error.message : "扬声器设置应用失败。");
    console.warn(error);
  }

  if (ws?.readyState === WebSocket.OPEN) {
    if (selectedCharacterConfigFile() !== lastAppliedCharacterConfigFile) {
      sendCharacterConfigSwitch();
    }
    sendClientApiConfig();
    await sendClientVoiceCloneConfig();
  }

  settingsPanel.hidden = true;
  settingsButton.setAttribute("aria-expanded", "false");
  syncSettingsPanelMode();
  appendLine("system", wasCapturing ? "配置已应用，已按新设置重新启动麦克风。" : "配置已应用。");
  restartProactiveSpeakLoop();

  if (wasCapturing) {
    await startCapture();
  }
}

function bootLive2D() {
  const modelOption = selectedLive2DModelOption();
  activeLive2DModelId = modelOption.id;
  updateModelConfig("/models/", modelOption.directory, modelOption.fileName, modelOption.scale);
  initializeLive2D();
  syncLive2DModelActiveState();
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
  requestProactiveSpeak("manual", true);
});

savedSettings = normalizeStartupSettings(readSavedSettings());
if (savedSettings) {
  localStorage.setItem(settingsStorageKey, JSON.stringify(savedSettings));
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
  live2dModelOptions = await readLive2DModelOptions();
  renderLive2DModelOptions();
  await restoreReferenceAudio();
  await refreshDevices();
  setCaptureUi(false);
  syncVolume(volumeNumber.value);
  syncVoiceChatOutputHint();
  syncScreenVisionControls();
  syncProactiveSpeakControls();
  syncProactiveSpeakButton();
  restartProactiveSpeakLoop();
  bootLive2D();
}

void startup();

