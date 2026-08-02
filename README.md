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
- OmniVoice voice cloning. The normal microphone, speech recognition, LLM, and
  configured TTS path do not require its PyTorch packages.

## Quick Start

Install 64-bit Python 3.11 and Node.js 20 or newer first. Make sure both
`python` and `node` are available in a new Command Prompt, then run the single
installer from the project root:

```bash
setup-windows.bat
```

The installer always installs and verifies the common application first. It
then asks:

- `Install voice cloning now? [Y/N]`
- Choose `N` for the smaller normal installation. Setup finishes without
  PyTorch, OmniVoice, or the cloning models.
- Choose `Y` to continue in the same installer. If an NVIDIA GPU is detected,
  choose `G` for CUDA 12.8 or `C` for CPU. Without an NVIDIA driver, CPU mode is
  selected automatically.

The optional download is large and can take a while. Do not close the window
until it reports `Setup finished successfully`. The script also builds the
frontend, so no separate build command is needed.

Configure the LLM provider/API key in `backend/conf.yaml` (or in the app's
settings), then start MeloMate:

```bash
start.bat
```

Open the address below if the browser does not open automatically:

```text
http://127.0.0.1:5178/
```

Device choices, API endpoints, and model names are remembered by the local
browser profile. Chat and screen-vision API keys are never written to browser
storage or the project directory. On Windows, applying settings stores only
current-user-bound DPAPI ciphertext in
`%LOCALAPPDATA%\MeloMate\credentials-v1.json`; saved keys are loaded directly
inside the backend and are not returned to the page. Use the **Clear** button
beside either key to remove its saved credential. Feature switches still start
off on every new page load.

`start.bat` never terminates an existing port owner. It checks the configured
frontend and backend ports before launch, starts only its own child processes,
verifies both services with a per-launch identity token, and opens the browser
only after both services are ready. If a port is occupied, close the owning
application or choose another port; MeloMate exits without touching it.

To change the frontend port for one launch:

```bat
set MELOMATE_FRONTEND_PORT=5180
start.bat
```

The launcher opens the selected frontend port automatically. The backend port
comes from `system_config.port` in `backend/conf.yaml`, and the browser receives
the matching WebSocket address at runtime. The two ports must be different.

You can rerun `setup-windows.bat` later and choose `Y` to add voice cloning to
the same installation. `setup-windows-gpu.bat` remains only as a compatibility
shortcut that preselects the NVIDIA option; the main installer is recommended.

`download-omnivoice-model.bat` is a repair/redownload helper. The main installer
already calls it when voice cloning is selected, so it is not a normal extra
installation step.

All runtime models are local-first and pinned to tested versions; MeloMate does
not require the latest Hugging Face revision. To avoid a long Hugging Face
download, download all three `MeloMate-Model-Cache.7z.001/.002/.003` parts from
the [`v1.0.0-models` Release](https://github.com/misheng521/MeloMate/releases/tag/v1.0.0-models),
extract `.001` into `models/backend/`, and then run the normal installer. The
installer verifies and reuses the extracted local OmniVoice and Whisper cache.

The default microphone ASR uses SenseVoice. You can either let MeloMate fetch
its two fixed, SHA-256-verified files, or download the fixed archive from the
[`models-v0.1.0` Release](https://github.com/misheng521/MeloMate/releases/tag/models-v0.1.0),
place it in the project root, and run `install-sensevoice-model.bat`. The script
checks the Release archive hash, rejects unsafe archive paths and links,
extracts into a temporary directory, verifies both model files, and installs
them atomically.

For unattended installation, these environment variables are supported:

```bat
set MELOMATE_VOICE_CLONE=0
set MELOMATE_NO_PAUSE=1
setup-windows.bat
```

Use `MELOMATE_VOICE_CLONE=1` together with
`MELOMATE_VOICE_CLONE_DEVICE=cpu` or `gpu` to install the optional feature
without prompts.

If you prefer to install manually, run:

```bash
npm ci
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
cd ..
npm run build
```

Manual commands above install only the common application. Use the unified
installer for validated voice-cloning installation and model download.

## Development

Frontend development server:

```bash
npm run dev
```

The Vite development server is a developer tool, not the production runtime.
Its custom middleware is implemented separately from `server.mjs`; production-
only APIs such as Voicemeeter control and the complete workspace event behavior
may therefore be unavailable or behave differently under `npm run dev`. Use
`start.bat` for end-to-end acceptance testing and normal use.

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
- `docs/WORKSPACE_PROTOCOL.md` - Protocol for interactive workspace apps that both the user and MeloMate can control.
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
