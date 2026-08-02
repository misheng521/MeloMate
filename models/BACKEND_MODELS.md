# Backend Models

Large ASR, TTS, Whisper, OmniVoice, Hugging Face, and ModelScope caches do not belong in the source repository.

Put local backend models under:

```text
models/backend/
```

The default ASR configuration expects:

```text
models/backend/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx
models/backend/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt
```

MeloMate verifies these files by exact size and SHA-256 before use. A verified
local copy is always reused. The fixed `models-v0.1.0` GitHub Release archive
can be installed with `install-sensevoice-model.bat`.

For voice cloning, an extracted local Hugging Face cache is detected under:

```text
models/backend/hub/models--k2-fsa--OmniVoice/
models/backend/hub/models--openai--whisper-large-v3-turbo/
```

For a portable release, copy only the selected runtime models into `models/backend` during packaging. Do not commit that folder to Git.
