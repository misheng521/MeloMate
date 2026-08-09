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
const analysisFrequencies = [300, 350, 500, 700, 850, 1000, 1200, 1900, 2400] as const;
const targetFrameRate = 30;

type MouthName = (typeof mouthNames)[number];
type MouthWeights = Record<MouthName, number>;

type LipSyncFrame = {
  time: number;
  rms: number;
  scores: MouthWeights;
};

export type LipSyncTrack = {
  duration: number;
  step: number;
  frames: Array<{ time: number; weights: MouthWeights }>;
};

type IdleBone = {
  node: THREE.Object3D;
  rest: THREE.Quaternion;
  kind: "chest" | "head" | "spine";
};

export type AvatarLoadInfo = {
  name: string;
  authors: string[];
  mouthExpressions: string[];
};

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value));
}

function emptyMouthWeights(): MouthWeights {
  return { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
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

function analyzeAudioBuffer(buffer: AudioBuffer): LipSyncTrack {
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

  return {
    duration: buffer.duration,
    step,
    frames: frames.map((frame) => {
      const open = Math.sqrt(clamp01((frame.rms - gate) / (fullOpen - gate)));
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
      return { time: frame.time, weights };
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
  private readonly idleBones: IdleBone[] = [];
  private readonly mouthState = emptyMouthWeights();
  private readonly rotation = new THREE.Quaternion();
  private readonly onStatus: (message: string, tone?: "loading" | "ready" | "error") => void;
  private readonly onZoom: (label: string) => void;
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
  private zoom = 0.68;

  constructor(
    canvas: HTMLCanvasElement,
    onStatus: (message: string, tone?: "loading" | "ready" | "error") => void,
    onZoom: (label: string) => void,
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
    this.renderer.toneMappingExposure = 1.05;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x66515d, 2.1));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
    keyLight.position.set(1.4, 2.8, 2.6);
    this.scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xffdfe9, 0.8);
    fillLight.position.set(-2, 1.4, 1.2);
    this.scene.add(fillLight);

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

    const gltf = await this.loader.loadAsync(url);
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

  async prepareLipSync(audioBase64: string): Promise<LipSyncTrack | null> {
    if (!audioBase64 || !this.vrm?.expressionManager) return null;
    const context = new OfflineAudioContext(1, 1, 44_100);
    const buffer = await context.decodeAudioData(decodeBase64(audioBase64));
    return analyzeAudioBuffer(buffer);
  }

  startLipSync(audio: HTMLAudioElement, track: LipSyncTrack | null) {
    this.lipAudio = audio;
    this.lipTrack = track;
  }

  stopLipSync() {
    this.lipAudio = null;
    this.lipTrack = null;
    mouthNames.forEach((name) => {
      this.mouthState[name] = 0;
      this.vrm?.expressionManager?.setValue(name, 0);
    });
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
    this.emotionStrength = name === "neutral" ? 0.25 : 0.68;
    this.emotionUntil = performance.now() + 4200;
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
    this.scene.remove(this.vrm.scene);
    VRMUtils.deepDispose(this.vrm.scene);
    this.vrm = null;
    this.idleBones.length = 0;
  }

  private configureModel() {
    if (!this.vrm) return;
    this.vrm.scene.traverse((object) => {
      object.frustumCulled = false;
    });

    const addBone = (name: string, kind: IdleBone["kind"]) => {
      const node = this.vrm?.humanoid.getNormalizedBoneNode(name as never);
      if (node) this.idleBones.push({ node, rest: node.quaternion.clone(), kind });
    };
    addBone(VRMHumanBoneName.Spine, "spine");
    addBone(VRMHumanBoneName.Chest, "chest");
    addBone(VRMHumanBoneName.Head, "head");
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
    const framing = this.zoom < 0.18 ? "全身" : this.zoom > 0.86 ? "肩部以上" : "半身";
    this.onZoom(`${framing} · 滚轮缩放`);
  }

  private render(time: number) {
    if (document.hidden) return;
    if (time - this.lastRenderAt < 1000 / targetFrameRate) return;
    this.lastRenderAt = time;
    const delta = Math.min(0.05, this.clock.getDelta());
    this.elapsed += delta;

    if (this.vrm) {
      this.updateIdleMotion();
      this.updateBlink();
      this.updateMouth(delta);
      this.updateEmotion(time);
      this.vrm.update(delta);
    }
    this.renderer.render(this.scene, this.camera);
  }

  private updateIdleMotion() {
    const breath = Math.sin(this.elapsed * 1.45);
    const drift = Math.sin(this.elapsed * 0.37);
    this.idleBones.forEach(({ node, rest, kind }) => {
      const euler = kind === "chest"
        ? new THREE.Euler(breath * 0.012, drift * 0.008, drift * 0.006)
        : kind === "head"
          ? new THREE.Euler(breath * 0.004, drift * 0.014, Math.sin(this.elapsed * 0.29) * 0.007)
          : new THREE.Euler(breath * 0.006, 0, drift * 0.004);
      this.rotation.setFromEuler(euler);
      node.quaternion.copy(rest).multiply(this.rotation);
    });
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

    if (this.lipAudio && this.lipTrack && !this.lipAudio.paused && !this.lipAudio.ended) {
      const framePosition = this.lipAudio.currentTime / this.lipTrack.step;
      const firstIndex = Math.min(this.lipTrack.frames.length - 1, Math.max(0, Math.floor(framePosition)));
      const secondIndex = Math.min(this.lipTrack.frames.length - 1, firstIndex + 1);
      const mix = clamp01(framePosition - firstIndex);
      const first = this.lipTrack.frames[firstIndex]?.weights;
      const second = this.lipTrack.frames[secondIndex]?.weights;
      if (first && second) {
        mouthNames.forEach((name) => {
          target[name] = THREE.MathUtils.lerp(first[name], second[name], mix);
        });
      }
    }

    const smoothing = 1 - Math.exp(-delta * 24);
    mouthNames.forEach((name) => {
      this.mouthState[name] = THREE.MathUtils.lerp(this.mouthState[name], target[name], smoothing);
      manager.setValue(name, this.mouthState[name]);
    });
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
