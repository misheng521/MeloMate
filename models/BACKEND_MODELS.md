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

For a portable release, copy only the selected runtime models into `models/backend` during packaging. Do not commit that folder to Git.
