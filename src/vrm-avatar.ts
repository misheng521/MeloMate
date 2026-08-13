import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  VRM,
  VRMHumanBoneName,
  VRMLoaderPlugin,
  VRMUtils,
} from "@pixiv/three-vrm";

const mouthNames = ["aa", "ih", "ou", "ee", "oh"] as const;
const emotionNames = ["neutral", "happy", "angry", "sad", "relaxed", "surprised"] as const;
const gestureNames = ["greet", "explain", "emphasize", "agree", "think"] as const;
const analysisFrequencies = [300, 350, 500, 700, 850, 1000, 1200, 1900, 2400] as const;
const targetFrameRate = 30;

type MouthName = (typeof mouthNames)[number];
type MouthWeights = Record<MouthName, number>;
type GestureName = (typeof gestureNames)[number];

type GesturePose = {
  bodyLean: number;
  bodyTurn: number;
  bodyTilt: number;
  headPitch: number;
  headYaw: number;
  headRoll: number;
  leftShoulderX: number;
  leftShoulderY: number;
  leftShoulderZ: number;
  rightShoulderX: number;
  rightShoulderY: number;
  rightShoulderZ: number;
  leftUpperArmX: number;
  leftUpperArmY: number;
  leftUpperArmZ: number;
  rightUpperArmX: number;
  rightUpperArmY: number;
  rightUpperArmZ: number;
  leftLowerArmX: number;
  leftLowerArmY: number;
  leftLowerArmZ: number;
  rightLowerArmX: number;
  rightLowerArmY: number;
  rightLowerArmZ: number;
  leftHandX: number;
  leftHandY: number;
  leftHandZ: number;
  rightHandX: number;
  rightHandY: number;
  rightHandZ: number;
};

type LipSyncFrame = {
  time: number;
  rms: number;
  scores: MouthWeights;
};

type PronouncedSyllable = {
  initial: string;
  final: string;
};

export type LipSyncTrack = {
  duration: number;
  step: number;
  frames: Array<{
    time: number;
    weights: MouthWeights;
    energy: number;
    accent: number;
  }>;
};

type MotionBone = {
  node: THREE.Object3D;
  rest: THREE.Quaternion;
};

export type AvatarConversationState = "idle" | "listening" | "thinking" | "speaking";

export type AvatarLoadInfo = {
  name: string;
  authors: string[];
  mouthExpressions: string[];
};

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value));
}

function damp(current: number, target: number, speed: number, delta: number) {
  return THREE.MathUtils.lerp(current, target, 1 - Math.exp(-speed * delta));
}

function randomBetween(min: number, max: number) {
  return min + Math.random() * (max - min);
}

function emptyMouthWeights(): MouthWeights {
  return { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
}

function emptyGesturePose(): GesturePose {
  return {
    bodyLean: 0,
    bodyTurn: 0,
    bodyTilt: 0,
    headPitch: 0,
    headYaw: 0,
    headRoll: 0,
    leftShoulderX: 0,
    leftShoulderY: 0,
    leftShoulderZ: 0,
    rightShoulderX: 0,
    rightShoulderY: 0,
    rightShoulderZ: 0,
    leftUpperArmX: 0,
    leftUpperArmY: 0,
    leftUpperArmZ: 0,
    rightUpperArmX: 0,
    rightUpperArmY: 0,
    rightUpperArmZ: 0,
    leftLowerArmX: 0,
    leftLowerArmY: 0,
    leftLowerArmZ: 0,
    rightLowerArmX: 0,
    rightLowerArmY: 0,
    rightLowerArmZ: 0,
    leftHandX: 0,
    leftHandY: 0,
    leftHandZ: 0,
    rightHandX: 0,
    rightHandY: 0,
    rightHandZ: 0,
  };
}

function decodeBase64(base64: string) {
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let offset = 0; offset < binary.length; offset += 1) {
    bytes[offset] = binary.charCodeAt(offset);
  }
  return bytes.buffer;
}

function goertzelEnergy(samples: Float32Array, frequency: number, sampleRate: number) {
  const omega = (2 * Math.PI * frequency) / sampleRate;
  const coefficient = 2 * Math.cos(omega);
  let previous = 0;
  let previousPrevious = 0;

  for (let index = 0; index < samples.length; index += 1) {
    const current = samples[index] + coefficient * previous - previousPrevious;
    previousPrevious = previous;
    previous = current;
  }

  return Math.max(
    0,
    (previousPrevious * previousPrevious
      + previous * previous
      - coefficient * previous * previousPrevious) / (samples.length * samples.length),
  );
}

function percentile(values: number[], fraction: number) {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
}

async function pronunciationGuide(text: string): Promise<PronouncedSyllable[]> {
  if (!text.trim()) return [];
  const { pinyin } = await import("pinyin-pro");
  const sharedOptions = {
    toneType: "none",
    type: "array",
    nonZh: "consecutive",
    v: true,
  } as const;
  const initials = pinyin(text, { ...sharedOptions, pattern: "initial" }) as string[];
  const finals = pinyin(text, { ...sharedOptions, pattern: "final" }) as string[];

  return finals.flatMap((value, index) => {
    const final = value.toLowerCase().replace(/[^a-zv]/g, "");
    if (!final) return [];
    return [{
      initial: (initials[index] || "").toLowerCase().replace(/[^a-z]/g, ""),
      final,
    }];
  });
}

function finalVisemeSequence(final: string): MouthName[] {
  const transitions: Record<string, MouthName[]> = {
    ai: ["aa", "ih"],
    ei: ["ee", "ih"],
    ao: ["aa", "oh"],
    ou: ["oh", "ou"],
    ia: ["ih", "aa"],
    ie: ["ih", "ee"],
    iao: ["ih", "aa", "oh"],
    iou: ["ih", "oh", "ou"],
    iu: ["ih", "oh", "ou"],
    ian: ["ih", "ee"],
    iang: ["ih", "aa"],
    iong: ["ih", "oh", "ou"],
    ua: ["ou", "aa"],
    uo: ["ou", "oh"],
    uai: ["ou", "aa", "ih"],
    uei: ["ou", "ee", "ih"],
    ui: ["ou", "ee", "ih"],
    uan: ["ou", "aa"],
    uang: ["ou", "aa"],
    ue: ["ou", "ee"],
    ve: ["ou", "ee"],
    van: ["ou", "ee"],
    vn: ["ou", "ih"],
    ong: ["oh", "ou"],
    er: ["ee", "oh"],
  };
  if (transitions[final]) return transitions[final];

  const vowel = final.replace(/ng?$|r$/g, "")[0] || final[0];
  if (vowel === "a") return ["aa"];
  if (vowel === "o") return ["oh"];
  if (vowel === "e") return ["ee"];
  if (vowel === "i") return ["ih"];
  if (vowel === "u" || vowel === "v") return ["ou"];
  return ["aa"];
}

function pronunciationWeights(syllable: PronouncedSyllable, progress: number) {
  const weights = emptyMouthWeights();
  const sequence = finalVisemeSequence(syllable.final);
  const hasInitial = Boolean(syllable.initial);
  const finalStartsAt = hasInitial ? 0.18 : 0.035;

  if (progress < finalStartsAt) {
    const initialProgress = clamp01(progress / finalStartsAt);
    if (/^(b|p|m)$/.test(syllable.initial)) return weights;
    if (syllable.initial === "f") {
      weights.ih = 0.24;
      weights.ee = 0.08;
      return weights;
    }
    if (/^(j|q|x|z|c|s|zh|ch|sh|r)$/.test(syllable.initial)) {
      weights.ih = 0.3;
      weights.ee = 0.1;
      return weights;
    }
    weights[sequence[0]] = initialProgress * 0.22;
    return weights;
  }

  const finalProgress = clamp01((progress - finalStartsAt) / (1 - finalStartsAt));
  const position = finalProgress * Math.max(0, sequence.length - 1);
  const firstIndex = Math.min(sequence.length - 1, Math.floor(position));
  const secondIndex = Math.min(sequence.length - 1, firstIndex + 1);
  const mix = position - firstIndex;
  weights[sequence[firstIndex]] += 1 - mix;
  weights[sequence[secondIndex]] += mix;

  if (/n(g)?$/.test(syllable.final) && finalProgress > 0.84) {
    const close = 1 - ((finalProgress - 0.84) / 0.16) * 0.28;
    mouthNames.forEach((name) => {
      weights[name] *= close;
    });
  }
  return weights;
}

async function analyzeAudioBuffer(buffer: AudioBuffer, spokenText = ""): Promise<LipSyncTrack> {
  const sampleRate = buffer.sampleRate;
  const channel = buffer.getChannelData(0);
  const step = 1 / targetFrameRate;
  const windowSize = Math.max(256, Math.min(2048, Math.round(sampleRate * 0.032)));
  const hopSize = Math.max(1, Math.round(sampleRate * step));
  const frames: LipSyncFrame[] = [];

  for (let start = 0; start < channel.length; start += hopSize) {
    const samples = new Float32Array(windowSize);
    let squaredSum = 0;
    for (let index = 0; index < windowSize; index += 1) {
      const sourceIndex = start + index;
      const sample = sourceIndex < channel.length ? channel[sourceIndex] : 0;
      squaredSum += sample * sample;
      samples[index] = sample * (0.5 - 0.5 * Math.cos((2 * Math.PI * index) / (windowSize - 1)));
    }

    const bands = new Map<number, number>();
    analysisFrequencies.forEach((frequency) => {
      bands.set(frequency, Math.sqrt(goertzelEnergy(samples, frequency, sampleRate)));
    });
    const band = (frequency: (typeof analysisFrequencies)[number]) => bands.get(frequency) || 0;

    frames.push({
      time: start / sampleRate,
      rms: Math.sqrt(squaredSum / windowSize),
      scores: {
        aa: band(700) * 0.56 + band(1200) * 0.44,
        ih: band(350) * 0.45 + band(1900) * 0.55,
        ou: band(350) * 0.58 + band(850) * 0.42,
        ee: band(300) * 0.42 + band(2400) * 0.58,
        oh: band(500) * 0.55 + band(1000) * 0.45,
      },
    });
  }

  const rmsValues = frames.map((frame) => frame.rms);
  const peak = Math.max(0.0001, percentile(rmsValues, 0.96));
  const noiseFloor = percentile(rmsValues, 0.15);
  const gate = Math.max(0.0025, noiseFloor + (peak - noiseFloor) * 0.06);
  const fullOpen = Math.max(gate + 0.0001, peak * 0.42);
  const syllables = await pronunciationGuide(spokenText);
  const activities = frames.map((frame) => {
    const energy = clamp01((frame.rms - gate) / Math.max(0.0001, peak - gate));
    return frame.rms > gate ? 0.34 + Math.sqrt(energy) * 0.66 : 0;
  });
  const totalActivity = activities.reduce((sum, activity) => sum + activity, 0);
  let accumulatedActivity = 0;

  return {
    duration: buffer.duration,
    step,
    frames: frames.map((frame, index) => {
      const open = Math.sqrt(clamp01((frame.rms - gate) / (fullOpen - gate)));
      const energy = clamp01((frame.rms - gate) / Math.max(0.0001, peak - gate));
      const previousRms = frames[Math.max(0, index - 2)]?.rms || 0;
      const accent = clamp01(((frame.rms - previousRms) / peak) * 5.5) * energy;
      const ranked = mouthNames
        .map((name) => ({ name, score: frame.scores[name] }))
        .sort((left, right) => right.score - left.score);
      const weights = emptyMouthWeights();
      const best = ranked[0];
      const second = ranked[1];
      weights[best.name] = open;
      if (best.score > 0 && second.score > 0) {
        weights[second.name] = open * 0.32 * clamp01(second.score / best.score);
      }
      if (syllables.length) {
        const activity = activities[index];
        const spokenProgress = totalActivity > 0
          ? (accumulatedActivity + activity * 0.5) / totalActivity
          : frame.time / Math.max(buffer.duration, step);
        const syllablePosition = THREE.MathUtils.clamp(
          spokenProgress * syllables.length,
          0,
          Math.max(0, syllables.length - 0.0001),
        );
        const syllableIndex = Math.min(syllables.length - 1, Math.floor(syllablePosition));
        const textWeights = pronunciationWeights(
          syllables[syllableIndex],
          syllablePosition - syllableIndex,
        );
        const textStrength = Math.max(...mouthNames.map((name) => textWeights[name]));
        const spectralBlend = textStrength < 0.05 ? 0.025 : 0.07;
        mouthNames.forEach((name) => {
          weights[name] = clamp01(
            open * textWeights[name] * 0.96 + weights[name] * spectralBlend,
          );
        });
        accumulatedActivity += activity;
      }
      return { time: frame.time, weights, energy, accent };
    }),
  };
}

export class VrmAvatar {
  private readonly canvas: HTMLCanvasElement;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(28, 1, 0.01, 100);
  private readonly clock = new THREE.Clock();
  private readonly loader = new GLTFLoader();
  private readonly resizeObserver: ResizeObserver;
  private readonly motionBones = new Map<VRMHumanBoneName, MotionBone>();
  private readonly mouthState = emptyMouthWeights();
  private readonly rotation = new THREE.Quaternion();
  private readonly poseEuler = new THREE.Euler();
  private readonly gazeTarget = new THREE.Object3D();
  private readonly headWorldPosition = new THREE.Vector3();
  private readonly onStatus: (message: string, tone?: "loading" | "ready" | "error") => void;
  private readonly onZoom: (percentage: number) => void;
  private vrm: VRM | null = null;
  private loadVersion = 0;
  private elapsed = 0;
  private lastRenderAt = 0;
  private nextBlinkAt = 2.2;
  private blinkStartedAt = -1;
  private lipAudio: HTMLAudioElement | null = null;
  private lipTrack: LipSyncTrack | null = null;
  private emotionName = "";
  private emotionStrength = 0;
  private emotionUntil = 0;
  private conversationState: AvatarConversationState = "idle";
  private stateChangedAt = 0;
  private gazeYaw = 0;
  private gazePitch = 0;
  private gazeYawTarget = 0;
  private gazePitchTarget = 0;
  private headGazeYaw = 0;
  private headGazePitch = 0;
  private nextGazeAt = 0.8;
  private nextListeningNodAt = 3.5;
  private listeningNodStartedAt = -1;
  private bodyLean = 0;
  private bodyTurn = 0;
  private bodyTilt = 0;
  private speechEnergy = 0;
  private speechAccent = 0;
  private activeGesture: GestureName | null = null;
  private pendingGesture: GestureName | null = null;
  private gestureStartedAt = -1;
  private gestureDuration = 0;
  private gestureIntensity = 0;
  private nextAutoGestureAt = 0;
  private zoom = 0.68;

  constructor(
    canvas: HTMLCanvasElement,
    onStatus: (message: string, tone?: "loading" | "ready" | "error") => void,
    onZoom: (percentage: number) => void,
  ) {
    this.canvas = canvas;
    this.onStatus = onStatus;
    this.onZoom = onZoom;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: "low-power",
      preserveDrawingBuffer: false,
    });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.88;

    this.scene.add(new THREE.HemisphereLight(0xfff8f3, 0x554a5c, 1.25));
    const keyLight = new THREE.DirectionalLight(0xfff7f0, 1.5);
    keyLight.position.set(1.4, 2.8, 2.6);
    this.scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xffdfe9, 0.38);
    fillLight.position.set(-2, 1.4, 1.2);
    this.scene.add(fillLight);
    this.scene.add(this.gazeTarget);

    this.loader.register((parser) => new VRMLoaderPlugin(parser));
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement || canvas);
    (canvas.parentElement || canvas).addEventListener("wheel", (event) => this.handleWheel(event), {
      passive: false,
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) this.clock.getDelta();
    });
    this.renderer.setAnimationLoop((time) => this.render(time));
    this.resize();
    this.reportZoom();
  }

  async load(url: string): Promise<AvatarLoadInfo> {
    const version = ++this.loadVersion;
    this.stopLipSync();
    this.onStatus("正在加载 VRM 模型…", "loading");

    const gltf = await this.loadWithCompatibleTextures(url);
    const nextVrm = gltf.userData.vrm as VRM | undefined;
    if (!nextVrm) {
      VRMUtils.deepDispose(gltf.scene);
      throw new Error("这个文件不是有效的 VRM 模型。");
    }
    if (version !== this.loadVersion) {
      VRMUtils.deepDispose(nextVrm.scene);
      throw new Error("模型加载已被新的选择取代。");
    }

    if (nextVrm.meta.metaVersion === "0") VRMUtils.rotateVRM0(nextVrm);
    this.removeCurrentModel();
    this.vrm = nextVrm;
    this.scene.add(nextVrm.scene);
    this.configureModel();
    this.frameModel();
    this.resize();

    const expressionMap = nextVrm.expressionManager?.expressionMap || {};
    const missingMouthExpressions = mouthNames.filter((name) => !expressionMap[name]);
    const name = "name" in nextVrm.meta && nextVrm.meta.name
      ? nextVrm.meta.name
      : "title" in nextVrm.meta && nextVrm.meta.title
        ? nextVrm.meta.title
        : "VRM 模型";
    const authors = "authors" in nextVrm.meta
      ? nextVrm.meta.authors || []
      : "author" in nextVrm.meta && nextVrm.meta.author
        ? [nextVrm.meta.author]
        : [];

    if (missingMouthExpressions.length) {
      this.onStatus(`模型缺少口型：${missingMouthExpressions.join(" / ")}`, "error");
    } else {
      this.onStatus("", "ready");
    }

    return {
      name,
      authors,
      mouthExpressions: mouthNames.filter((expression) => Boolean(expressionMap[expression])),
    };
  }

  setZoomPercentage(percentage: number) {
    const nextZoom = clamp01(percentage / 100);
    if (Math.abs(nextZoom - this.zoom) < 0.001) return;
    this.zoom = nextZoom;
    this.frameModel();
    this.reportZoom();
  }

  private async loadWithCompatibleTextures(url: string) {
    const browserWindow = window as Window & {
      createImageBitmap?: typeof createImageBitmap;
    };
    const hadOwnCreateImageBitmap = Object.prototype.hasOwnProperty.call(
      browserWindow,
      "createImageBitmap",
    );
    const originalDescriptor = Object.getOwnPropertyDescriptor(browserWindow, "createImageBitmap");
    const canShadowCreateImageBitmap = !originalDescriptor || originalDescriptor.configurable;

    if (typeof browserWindow.createImageBitmap !== "function" || !canShadowCreateImageBitmap) {
      return this.loader.loadAsync(url);
    }

    // Chromium can expose ImageBitmap while rejecting GLB-internal blob URLs.
    // Shadow it during parsing so GLTFLoader selects its HTMLImageElement path.
    Object.defineProperty(browserWindow, "createImageBitmap", {
      configurable: true,
      value: undefined,
    });

    try {
      return await this.loader.loadAsync(url);
    } finally {
      if (hadOwnCreateImageBitmap && originalDescriptor) {
        Object.defineProperty(browserWindow, "createImageBitmap", originalDescriptor);
      } else {
        delete browserWindow.createImageBitmap;
      }
    }
  }

  async prepareLipSync(audioBase64: string, spokenText = ""): Promise<LipSyncTrack | null> {
    if (!audioBase64 || !this.vrm?.expressionManager) return null;
    const context = new OfflineAudioContext(1, 1, 44_100);
    const buffer = await context.decodeAudioData(decodeBase64(audioBase64));
    return analyzeAudioBuffer(buffer, spokenText);
  }

  startLipSync(audio: HTMLAudioElement, track: LipSyncTrack | null) {
    this.lipAudio = audio;
    this.lipTrack = track;
    this.setConversationState("speaking");
    if (this.pendingGesture) {
      this.beginGesture(this.pendingGesture, 1);
      this.pendingGesture = null;
    } else if (!this.activeGesture) {
      this.nextAutoGestureAt = this.elapsed + randomBetween(0.75, 1.45);
    }
  }

  stopLipSync() {
    this.lipAudio = null;
    this.lipTrack = null;
    this.pendingGesture = null;
    mouthNames.forEach((name) => {
      this.mouthState[name] = 0;
      this.vrm?.expressionManager?.setValue(name, 0);
    });
  }

  setConversationState(state: AvatarConversationState) {
    if (state === this.conversationState) return;
    this.conversationState = state;
    this.stateChangedAt = this.elapsed;
    this.nextGazeAt = this.elapsed;
    if (state === "listening") {
      this.nextListeningNodAt = this.elapsed + randomBetween(2.4, 4.8);
    } else {
      this.listeningNodStartedAt = -1;
    }
  }

  setExpression(expressions?: Array<number | string>) {
    const expression = expressions?.[expressions.length - 1];
    if (expression === undefined || expression === null) return;
    const indexMap: Record<number, (typeof emotionNames)[number]> = {
      0: "neutral",
      1: "sad",
      2: "angry",
      3: "happy",
      4: "relaxed",
      5: "surprised",
    };
    const name = typeof expression === "number" ? indexMap[expression] : expression.toLowerCase();
    if (!emotionNames.includes(name as (typeof emotionNames)[number])) return;

    emotionNames.forEach((emotion) => this.vrm?.expressionManager?.setValue(emotion, 0));
    this.emotionName = name;
    const naturalStrength: Record<(typeof emotionNames)[number], number> = {
      neutral: 0.12,
      happy: 0.34,
      angry: 0.46,
      sad: 0.42,
      relaxed: 0.32,
      surprised: 0.44,
    };
    this.emotionStrength = naturalStrength[name as (typeof emotionNames)[number]];
    this.emotionUntil = performance.now() + 4200;
  }

  setGesture(gestures?: string[], _spokenText = "") {
    const selected = [...(gestures || [])]
      .reverse()
      .find((gesture) => gestureNames.includes(gesture.toLowerCase() as GestureName));
    // Audio analysis happens after the websocket payload is received. Queue the
    // semantic gesture so its attack lands on speech playback, not during prep.
    this.pendingGesture = selected ? selected.toLowerCase() as GestureName : null;
  }

  resize() {
    const width = Math.max(1, this.canvas.clientWidth);
    const height = Math.max(1, this.canvas.clientHeight);
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.35);
    this.renderer.setPixelRatio(pixelRatio);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    if (this.vrm) this.frameModel();
  }

  private removeCurrentModel() {
    if (!this.vrm) return;
    if (this.vrm.lookAt) this.vrm.lookAt.target = null;
    this.scene.remove(this.vrm.scene);
    VRMUtils.deepDispose(this.vrm.scene);
    this.vrm = null;
    this.motionBones.clear();
  }

  private configureModel() {
    if (!this.vrm) return;
    this.vrm.scene.traverse((object) => {
      object.frustumCulled = false;
    });

    const boneNames: VRMHumanBoneName[] = [
      VRMHumanBoneName.Hips,
      VRMHumanBoneName.Spine,
      VRMHumanBoneName.Chest,
      VRMHumanBoneName.UpperChest,
      VRMHumanBoneName.Neck,
      VRMHumanBoneName.Head,
      VRMHumanBoneName.LeftShoulder,
      VRMHumanBoneName.LeftUpperArm,
      VRMHumanBoneName.LeftLowerArm,
      VRMHumanBoneName.LeftHand,
      VRMHumanBoneName.RightShoulder,
      VRMHumanBoneName.RightUpperArm,
      VRMHumanBoneName.RightLowerArm,
      VRMHumanBoneName.RightHand,
    ];
    boneNames.forEach((name) => {
      const node = this.vrm?.humanoid.getNormalizedBoneNode(name);
      if (node) this.motionBones.set(name, { node, rest: node.quaternion.clone() });
    });

    if (this.vrm.lookAt) {
      this.vrm.lookAt.autoUpdate = true;
      this.vrm.lookAt.target = this.gazeTarget;
    }
    this.nextGazeAt = this.elapsed;
    this.nextListeningNodAt = this.elapsed + randomBetween(2.4, 4.8);
    this.updateBodyMotion(1 / targetFrameRate);
    this.vrm.update(0);
  }

  private frameModel() {
    if (!this.vrm) return;
    const box = new THREE.Box3().setFromObject(this.vrm.scene);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    if (!Number.isFinite(size.y) || size.y <= 0) return;

    const verticalFov = THREE.MathUtils.degToRad(this.camera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * this.camera.aspect);
    const visibleHeight = size.y * THREE.MathUtils.lerp(1.05, 0.3, this.zoom);
    const visibleWidthFraction = THREE.MathUtils.lerp(0.52, 0.17, this.zoom);
    const heightDistance = (visibleHeight * 0.5) / Math.tan(verticalFov / 2);
    const widthDistance = (size.x * visibleWidthFraction) / Math.tan(Math.max(0.2, horizontalFov) / 2);
    const distance = Math.max(heightDistance, widthDistance) * 1.08;
    const targetY = box.min.y + size.y * THREE.MathUtils.lerp(0.5, 0.86, this.zoom);

    this.camera.position.set(center.x, targetY, box.max.z + distance);
    this.camera.near = Math.max(0.01, distance / 100);
    this.camera.far = distance + size.y * 12;
    this.camera.lookAt(center.x, targetY, center.z);
    this.camera.updateProjectionMatrix();
  }

  private handleWheel(event: Event) {
    const wheelEvent = event as WheelEvent;
    const target = wheelEvent.target as HTMLElement | null;
    if (target?.closest(".background-sidebar, button, input, select, textarea, audio")) return;
    if (!this.vrm) return;

    wheelEvent.preventDefault();
    const boundedDelta = THREE.MathUtils.clamp(wheelEvent.deltaY, -120, 120);
    const nextZoom = clamp01(this.zoom - boundedDelta * 0.0011);
    if (Math.abs(nextZoom - this.zoom) < 0.001) return;
    this.zoom = nextZoom;
    this.frameModel();
    this.reportZoom();
  }

  private reportZoom() {
    this.onZoom(Math.round(this.zoom * 100));
  }

  private render(time: number) {
    if (document.hidden) return;
    if (time - this.lastRenderAt < 1000 / targetFrameRate) return;
    this.lastRenderAt = time;
    const delta = Math.min(0.05, this.clock.getDelta());
    this.elapsed += delta;

    if (this.vrm) {
      this.updateMouth(delta);
      this.updateEmotion(time);
      this.updateGaze(delta);
      this.updateBodyMotion(delta);
      this.updateBlink();
      this.vrm.update(delta);
    }
    this.renderer.render(this.scene, this.camera);
  }

  private chooseGazeTarget() {
    const stateAge = this.elapsed - this.stateChangedAt;
    if (this.conversationState === "thinking") {
      const direction = Math.random() < 0.5 ? -1 : 1;
      this.gazeYawTarget = direction * randomBetween(0.09, 0.18);
      this.gazePitchTarget = randomBetween(-0.045, 0.07);
      this.nextGazeAt = this.elapsed + randomBetween(1.2, 2.4);
      return;
    }

    const glanceChance = this.conversationState === "idle"
      ? 0.24
      : this.conversationState === "speaking"
        ? 0.045
        : stateAge < 0.8
          ? 0
          : 0.08;
    if (Math.random() < glanceChance) {
      this.gazeYawTarget = (Math.random() < 0.5 ? -1 : 1) * randomBetween(0.055, 0.12);
      this.gazePitchTarget = randomBetween(-0.04, 0.035);
      this.nextGazeAt = this.elapsed + randomBetween(0.45, 1.05);
      return;
    }

    this.gazeYawTarget = randomBetween(-0.012, 0.012);
    this.gazePitchTarget = randomBetween(-0.014, 0.008);
    const minInterval = this.conversationState === "speaking" ? 1.4 : 1.6;
    const maxInterval = this.conversationState === "idle" ? 3.4 : 2.6;
    this.nextGazeAt = this.elapsed + randomBetween(minInterval, maxInterval);
  }

  private updateGaze(delta: number) {
    if (this.elapsed >= this.nextGazeAt) this.chooseGazeTarget();
    this.gazeYaw = damp(this.gazeYaw, this.gazeYawTarget, 18, delta);
    this.gazePitch = damp(this.gazePitch, this.gazePitchTarget, 18, delta);

    const head = this.motionBones.get(VRMHumanBoneName.Head)?.node;
    if (!head) return;
    this.vrm?.scene.updateMatrixWorld(true);
    head.getWorldPosition(this.headWorldPosition);
    const distance = Math.max(0.5, this.headWorldPosition.distanceTo(this.camera.position));
    const horizontalDistance = Math.max(
      0.2,
      Math.hypot(
        this.camera.position.x - this.headWorldPosition.x,
        this.camera.position.z - this.headWorldPosition.z,
      ),
    );
    const viewerPitch = THREE.MathUtils.clamp(
      Math.atan2(this.camera.position.y - this.headWorldPosition.y, horizontalDistance),
      -0.18,
      0.1,
    );
    this.headGazeYaw = damp(this.headGazeYaw, this.gazeYaw * 0.48, 3.8, delta);
    this.headGazePitch = damp(
      this.headGazePitch,
      viewerPitch * 0.78 + this.gazePitch * 0.32,
      3.4,
      delta,
    );

    // The eyes meet a person just behind the display; micro-glances stay relative
    // to that anchor instead of drifting toward a fixed point above the camera.
    this.gazeTarget.position.copy(this.camera.position);
    this.gazeTarget.position.x += Math.tan(this.gazeYaw) * distance;
    this.gazeTarget.position.y += Math.tan(this.gazePitch) * distance;
    this.gazeTarget.updateMatrixWorld(true);
  }

  private listeningNod() {
    if (this.conversationState !== "listening") return 0;
    if (this.listeningNodStartedAt < 0 && this.elapsed >= this.nextListeningNodAt) {
      this.listeningNodStartedAt = this.elapsed;
    }
    if (this.listeningNodStartedAt < 0) return 0;

    const progress = (this.elapsed - this.listeningNodStartedAt) / 0.72;
    if (progress >= 1) {
      this.listeningNodStartedAt = -1;
      this.nextListeningNodAt = this.elapsed + randomBetween(3.8, 7.2);
      return 0;
    }
    return Math.sin(progress * Math.PI) * Math.sin(progress * Math.PI * 2) * 0.035;
  }

  private setBoneRotation(name: VRMHumanBoneName, x: number, y: number, z: number) {
    const bone = this.motionBones.get(name);
    if (!bone) return;
    this.poseEuler.set(x, y, z, "XYZ");
    this.rotation.setFromEuler(this.poseEuler);
    bone.node.quaternion.copy(bone.rest).multiply(this.rotation);
  }

  private beginGesture(name: GestureName, intensity: number) {
    const durations: Record<GestureName, number> = {
      greet: 2.35,
      explain: 2.8,
      emphasize: 1.45,
      agree: 1.25,
      think: 2.7,
    };
    this.activeGesture = name;
    this.gestureStartedAt = this.elapsed;
    this.gestureDuration = durations[name] * randomBetween(0.92, 1.08);
    this.gestureIntensity = clamp01(intensity);
  }

  private maybeStartAutonomousGesture() {
    if (this.conversationState !== "speaking" || this.activeGesture) return;
    if (this.elapsed < this.nextAutoGestureAt) return;
    if (this.speechEnergy < 0.24) {
      this.nextAutoGestureAt = this.elapsed + 0.4;
      return;
    }

    const roll = Math.random();
    const name: GestureName = roll < 0.52
      ? "explain"
      : roll < 0.82
        ? "emphasize"
        : "agree";
    this.beginGesture(name, randomBetween(0.38, 0.52));
  }

  private gesturePose(): GesturePose {
    const pose = emptyGesturePose();
    if (!this.activeGesture || this.gestureStartedAt < 0 || this.gestureDuration <= 0) {
      return pose;
    }

    const progress = (this.elapsed - this.gestureStartedAt) / this.gestureDuration;
    if (progress >= 1) {
      this.activeGesture = null;
      this.gestureStartedAt = -1;
      this.nextAutoGestureAt = this.elapsed + randomBetween(3.8, 6.4);
      return pose;
    }

    const attack = THREE.MathUtils.smoothstep(progress, 0, 0.18);
    const release = 1 - THREE.MathUtils.smoothstep(progress, 0.7, 1);
    const weight = attack * release * this.gestureIntensity;
    const phase = progress * Math.PI * 2;

    if (this.activeGesture === "greet") {
      const wave = Math.sin(phase * 3.2) * weight;
      pose.bodyTurn = -0.018 * weight;
      pose.headRoll = -0.018 * weight;
      pose.rightShoulderZ = -0.08 * weight;
      pose.rightUpperArmX = -0.1 * weight;
      pose.rightUpperArmY = 0.16 * weight;
      pose.rightUpperArmZ = -0.5 * weight;
      pose.rightLowerArmY = 0.18 * weight;
      pose.rightLowerArmZ = 0.48 * weight;
      pose.rightHandX = -0.08 * weight;
      pose.rightHandY = 0.08 * weight;
      pose.rightHandZ = wave * 0.2;
    } else if (this.activeGesture === "explain") {
      const handLead = Math.sin(phase - 0.65) * 0.5 + 0.5;
      pose.bodyLean = -0.012 * weight;
      pose.bodyTurn = Math.sin(phase * 0.5) * 0.012 * weight;
      pose.leftShoulderZ = 0.035 * weight;
      pose.rightShoulderZ = -0.035 * weight;
      pose.leftUpperArmX = 0.055 * weight;
      pose.leftUpperArmY = -0.1 * weight;
      pose.leftUpperArmZ = 0.24 * weight;
      pose.rightUpperArmX = -0.055 * weight;
      pose.rightUpperArmY = 0.1 * weight;
      pose.rightUpperArmZ = -0.24 * weight;
      pose.leftLowerArmY = -0.16 * weight;
      pose.leftLowerArmZ = -0.08 * weight * handLead;
      pose.rightLowerArmY = 0.16 * weight;
      pose.rightLowerArmZ = 0.08 * weight * (1 - handLead);
      pose.leftHandY = -0.08 * weight;
      pose.leftHandZ = -0.06 * weight;
      pose.rightHandY = 0.08 * weight;
      pose.rightHandZ = 0.06 * weight;
    } else if (this.activeGesture === "emphasize") {
      const beat = Math.sin(Math.min(1, progress / 0.58) * Math.PI);
      pose.bodyLean = 0.022 * weight * beat;
      pose.headPitch = 0.026 * weight * beat;
      pose.leftUpperArmZ = 0.15 * weight;
      pose.rightUpperArmZ = -0.15 * weight;
      pose.leftLowerArmY = -0.13 * weight;
      pose.rightLowerArmY = 0.13 * weight;
      pose.leftHandX = 0.07 * weight;
      pose.rightHandX = -0.07 * weight;
    } else if (this.activeGesture === "agree") {
      const nod = Math.sin(progress * Math.PI * 4) * Math.sin(progress * Math.PI);
      pose.headPitch = nod * 0.038 * this.gestureIntensity;
      pose.bodyLean = 0.012 * weight;
      pose.rightUpperArmZ = -0.1 * weight;
      pose.rightLowerArmY = 0.1 * weight;
    } else if (this.activeGesture === "think") {
      pose.bodyTurn = 0.018 * weight;
      pose.bodyTilt = 0.018 * weight;
      pose.headYaw = 0.025 * weight;
      pose.headRoll = 0.035 * weight;
      pose.rightUpperArmX = -0.06 * weight;
      pose.rightUpperArmY = 0.08 * weight;
      pose.rightUpperArmZ = -0.16 * weight;
      pose.rightLowerArmY = 0.18 * weight;
      pose.rightLowerArmZ = 0.12 * weight;
      pose.rightHandX = -0.07 * weight;
    }

    return pose;
  }

  private updateBodyMotion(delta: number) {
    this.maybeStartAutonomousGesture();
    const gesture = this.gesturePose();
    const breath = Math.sin(this.elapsed * 1.36);
    const slowSway = Math.sin(this.elapsed * 0.31);
    const counterSway = Math.sin(this.elapsed * 0.43 + 1.1);
    const speaking = this.conversationState === "speaking";
    const listening = this.conversationState === "listening";
    const thinking = this.conversationState === "thinking";

    const targetLean = speaking
      ? 0.012 + this.speechEnergy * 0.012
      : listening
        ? 0.012
        : thinking
          ? 0.004
          : 0;
    const targetTurn = thinking ? this.gazeYawTarget * 0.34 : slowSway * 0.008;
    const targetTilt = thinking
      ? (this.gazeYawTarget < 0 ? -0.018 : 0.018)
      : counterSway * (listening ? 0.003 : 0.006);
    this.bodyLean = damp(this.bodyLean, targetLean, 3.2, delta);
    this.bodyTurn = damp(this.bodyTurn, targetTurn, 2.8, delta);
    this.bodyTilt = damp(this.bodyTilt, targetTilt, 2.6, delta);

    const nod = this.listeningNod();
    const speechBeat = speaking
      ? (Math.sin(this.elapsed * 4.7) * 0.016 + Math.sin(this.elapsed * 7.6 + 0.7) * 0.0065)
        * (0.18 + this.speechEnergy * 0.92)
      : 0;
    const accentNod = speaking ? this.speechAccent * 0.044 : 0;
    const speechTurn = speaking
      ? Math.sin(this.elapsed * 0.72 + 0.45) * this.speechEnergy * 0.009
      : 0;
    const speechShoulder = speaking
      ? (Math.sin(this.elapsed * 3.35 + 0.3) * this.speechEnergy * 0.007)
        + this.speechAccent * 0.018
      : 0;
    const emotionPitch = this.emotionName === "sad"
      ? 0.035 * this.emotionStrength
      : this.emotionName === "surprised"
        ? -0.025 * this.emotionStrength
        : this.emotionName === "angry"
          ? 0.012 * this.emotionStrength
          : 0;
    const emotionTilt = this.emotionName === "happy"
      ? 0.012 * this.emotionStrength
      : this.emotionName === "relaxed"
        ? -0.008 * this.emotionStrength
        : 0;

    // Keep relaxed arms anchored to the torso. Audio envelopes change quickly and
    // look like tremors when applied to both arms, so speech motion stays above
    // the shoulders while the arms only inherit an almost imperceptible breath.
    const armBreath = breath * 0.0018;
    const speechLift = speaking ? this.speechEnergy * 0.03 + this.speechAccent * 0.068 : 0;

    this.setBoneRotation(
      VRMHumanBoneName.Hips,
      this.bodyLean * 0.18 + gesture.bodyLean * 0.2,
      this.bodyTurn * 0.35 + gesture.bodyTurn * 0.3,
      this.bodyTilt * 0.25 + slowSway * 0.003 + gesture.bodyTilt * 0.28,
    );
    this.setBoneRotation(
      VRMHumanBoneName.Spine,
      breath * 0.004 + this.bodyLean * 0.28 + speechBeat * 0.12 + gesture.bodyLean * 0.3,
      this.bodyTurn * 0.45 + speechTurn * 0.35 + gesture.bodyTurn * 0.45,
      this.bodyTilt * 0.42 + gesture.bodyTilt * 0.4,
    );
    this.setBoneRotation(
      VRMHumanBoneName.Chest,
      breath * 0.009 + this.bodyLean * 0.34 + speechBeat * 0.24 + gesture.bodyLean * 0.42,
      this.bodyTurn * 0.55 + speechTurn * 0.55 + gesture.bodyTurn * 0.58,
      this.bodyTilt * 0.5 + counterSway * 0.0025 + gesture.bodyTilt * 0.5,
    );
    this.setBoneRotation(
      VRMHumanBoneName.UpperChest,
      breath * 0.006 + this.bodyLean * 0.2 + speechBeat * 0.18 + gesture.bodyLean * 0.28,
      this.bodyTurn * 0.38 + speechTurn * 0.42 + gesture.bodyTurn * 0.42,
      this.bodyTilt * 0.3 + gesture.bodyTilt * 0.38,
    );
    this.setBoneRotation(
      VRMHumanBoneName.Neck,
      -this.headGazePitch * 0.38 + nod * 0.28 + speechBeat * 0.22 + gesture.headPitch * 0.35,
      this.headGazeYaw * 0.34 + speechTurn * 0.4 + gesture.headYaw * 0.3,
      -this.headGazeYaw * 0.08 + emotionTilt * 0.35 + gesture.headRoll * 0.28,
    );
    this.setBoneRotation(
      VRMHumanBoneName.Head,
      -this.headGazePitch + nod + speechBeat + accentNod + emotionPitch + gesture.headPitch,
      this.headGazeYaw + (speaking ? Math.sin(this.elapsed * 0.53) * 0.006 : 0)
        + speechTurn + gesture.headYaw,
      -this.headGazeYaw * 0.1 + emotionTilt + slowSway * 0.003 + gesture.headRoll,
    );

    this.setBoneRotation(
      VRMHumanBoneName.LeftShoulder,
      gesture.leftShoulderX,
      -0.025 + gesture.leftShoulderY,
      -0.055 + armBreath + speechShoulder + gesture.leftShoulderZ,
    );
    this.setBoneRotation(
      VRMHumanBoneName.RightShoulder,
      gesture.rightShoulderX,
      0.025 + gesture.rightShoulderY,
      0.055 - armBreath - speechShoulder * 0.62 + gesture.rightShoulderZ,
    );
    this.setBoneRotation(
      VRMHumanBoneName.LeftUpperArm,
      0.035 + gesture.leftUpperArmX,
      -0.075 + gesture.leftUpperArmY,
      -1.19 + armBreath + speechLift + gesture.leftUpperArmZ,
    );
    this.setBoneRotation(
      VRMHumanBoneName.RightUpperArm,
      -0.035 + gesture.rightUpperArmX,
      0.075 + gesture.rightUpperArmY,
      1.19 - armBreath - speechLift + gesture.rightUpperArmZ,
    );
    this.setBoneRotation(
      VRMHumanBoneName.LeftLowerArm,
      -0.02 + gesture.leftLowerArmX,
      -0.11 + gesture.leftLowerArmY,
      -0.16 + gesture.leftLowerArmZ,
    );
    this.setBoneRotation(
      VRMHumanBoneName.RightLowerArm,
      0.02 + gesture.rightLowerArmX,
      0.11 + gesture.rightLowerArmY,
      0.16 + gesture.rightLowerArmZ,
    );
    this.setBoneRotation(
      VRMHumanBoneName.LeftHand,
      0.03 + gesture.leftHandX,
      -0.015 + gesture.leftHandY,
      -0.045 + gesture.leftHandZ,
    );
    this.setBoneRotation(
      VRMHumanBoneName.RightHand,
      -0.03 + gesture.rightHandX,
      0.015 + gesture.rightHandY,
      0.045 + gesture.rightHandZ,
    );
  }

  private updateBlink() {
    const manager = this.vrm?.expressionManager;
    if (!manager?.getExpression("blink")) return;
    if (this.blinkStartedAt < 0 && this.elapsed >= this.nextBlinkAt) {
      this.blinkStartedAt = this.elapsed;
    }
    if (this.blinkStartedAt < 0) return;

    const progress = (this.elapsed - this.blinkStartedAt) / 0.15;
    if (progress >= 1) {
      manager.setValue("blink", 0);
      this.blinkStartedAt = -1;
      this.nextBlinkAt = this.elapsed + 2.2 + Math.random() * 2.8;
      return;
    }
    manager.setValue("blink", progress < 0.46 ? progress / 0.46 : (1 - progress) / 0.54);
  }

  private updateMouth(delta: number) {
    const manager = this.vrm?.expressionManager;
    if (!manager) return;
    const target = emptyMouthWeights();
    let energyTarget = 0;
    let accentTarget = 0;

    if (
      this.lipAudio
      && this.lipTrack?.frames.length
      && !this.lipAudio.paused
      && !this.lipAudio.ended
    ) {
      const framePosition = this.lipAudio.currentTime / this.lipTrack.step;
      const firstIndex = Math.min(this.lipTrack.frames.length - 1, Math.max(0, Math.floor(framePosition)));
      const secondIndex = Math.min(this.lipTrack.frames.length - 1, firstIndex + 1);
      const mix = clamp01(framePosition - firstIndex);
      const first = this.lipTrack.frames[firstIndex];
      const second = this.lipTrack.frames[secondIndex];
      if (first && second) {
        mouthNames.forEach((name) => {
          target[name] = THREE.MathUtils.lerp(first.weights[name], second.weights[name], mix);
        });
        energyTarget = THREE.MathUtils.lerp(first.energy, second.energy, mix);
        accentTarget = THREE.MathUtils.lerp(first.accent, second.accent, mix);
      }
    }

    const smoothing = 1 - Math.exp(-delta * 24);
    mouthNames.forEach((name) => {
      this.mouthState[name] = THREE.MathUtils.lerp(this.mouthState[name], target[name], smoothing);
      manager.setValue(name, this.mouthState[name]);
    });
    this.speechEnergy = damp(this.speechEnergy, energyTarget, energyTarget > this.speechEnergy ? 15 : 8, delta);
    this.speechAccent = damp(this.speechAccent, accentTarget, accentTarget > this.speechAccent ? 20 : 10, delta);
  }

  private updateEmotion(time: number) {
    const manager = this.vrm?.expressionManager;
    if (!manager || !this.emotionName) return;
    if (time >= this.emotionUntil) {
      this.emotionStrength = Math.max(0, this.emotionStrength - 0.035);
      if (this.emotionStrength <= 0) this.emotionName = "";
    }
    if (this.emotionName) manager.setValue(this.emotionName, this.emotionStrength);
  }
}
