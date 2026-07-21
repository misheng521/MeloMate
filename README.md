# MeloMate

MeloMate is a local AI voice companion app with a Vite frontend, Live2D rendering, microphone input, a Python WebSocket backend, character profiles, memory files, backgrounds, and reference voice assets.

This repository is the **source edition**. It is intended for development, GitHub hosting, and reproducible setup. Generated folders such as `node_modules`, `dist`, `backend/.venv`, caches, logs, and large downloaded backend models are intentionally not part of the source tree.

## Requirements

- Windows 10/11 is recommended for the bundled `start.bat`.
- Node.js 20 or newer.
- Python 3.11.
- A DeepSeek/OpenAI-compatible API key, or another LLM provider configured in `backend/conf.yaml`.

Optional:

- Voicemeeter Pro, if you want the extra virtual audio output controls.
- Backend ASR/TTS models for fully local speech features.

## Quick Start

For Windows users, run the setup script from the project root:

```bash
setup-windows.bat
```

This installs frontend dependencies, creates `backend/.venv`, and installs backend Python dependencies from `backend/requirements.txt`.

Then build and run:

```bash
npm run build
start.bat
```

For NVIDIA GPU voice cloning, use the GPU setup script instead. It installs the
normal dependencies first, then replaces PyTorch with the CUDA 12.1 build:

```bash
setup-windows-gpu.bat
```

If the GPU check prints `cuda available: False`, install the latest NVIDIA driver
or use the normal CPU setup.

If you prefer to install manually, run:

```bash
npm install
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
cd ..
```

The default app URL is:

```text
http://127.0.0.1:5178/
```

## Development

Frontend development server:

```bash
npm run dev
```

Backend development server:

```bash
backend\.venv\Scripts\python backend\mini_backend.py
```

Frontend type check:

```bash
npm run check
```

## Project Layout

- `src` - Main frontend TypeScript code.
- `WebSDK` - Live2D Cubism Web SDK integration used by the frontend.
- `public` - Browser-side runtime libraries and WASM files.
- `backend/src/open_llm_vtuber` - Python backend modules for WebSocket, conversation, ASR, TTS, memory, tools, and configuration.
- `backend/prompts` - Prompt fragments used by the backend.
- `backend/conf.yaml` - Main backend configuration.
- `characters/profiles` - Character YAML profiles.
- `characters/memory` - Default character memory files.
- `models/live2d` - Live2D model assets.
- `backgrounds` - Background images discovered by the frontend.
- `reference_sounds/samples` - Small sample reference voices.

## Source Edition vs Portable Edition

The source edition should stay small and reproducible. Do not commit:

- `node_modules`
- `dist`
- `backend/.venv`
- `backend/cache`
- `backend/logs`
- `models/backend`
- downloaded Hugging Face or ModelScope caches

A portable edition should be built as a release artifact, for example `MeloMate-v0.1.0-windows-portable.zip`. That package may include `dist`, a prebuilt Python environment, and selected backend models, but it should be generated from this source tree instead of committed to Git.

## Backend Models

The source repository does not include GB-scale backend models. If a selected ASR/TTS provider needs local model files, place them under `models/backend` according to `backend/conf.yaml`, or adjust the config to use an online provider.

The default ASR config points to:

```text
models/backend/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/
```

Create that folder only in your local checkout or in a portable release package.

## Notes

- API keys in `backend/conf.yaml` are placeholders. Keep real keys local.
- `server.mjs` only binds to `127.0.0.1` by default.
- Voicemeeter integration is optional and Windows-specific.
- Check `NOTICE.md` before publishing a public release, because SDKs, Live2D models, browser libraries, and audio samples may have separate redistribution terms.
