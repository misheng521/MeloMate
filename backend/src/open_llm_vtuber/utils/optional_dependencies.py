import importlib.util

from .model_downloads import ensure_voice_clone_models


VOICE_CLONE_MODULES = (
    "torch",
    "torchaudio",
    "transformers",
    "accelerate",
    "librosa",
    "omnivoice",
    "huggingface_hub",
)
VOICE_CLONE_MODEL_INSTALLATION = "verified local voice-clone model cache"


def missing_voice_clone_dependencies() -> list[str]:
    """Return missing optional modules without importing heavyweight packages."""
    missing = [
        module_name
        for module_name in VOICE_CLONE_MODULES
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        return missing
    try:
        ensure_voice_clone_models(download_missing=False)
    except (OSError, RuntimeError, ValueError):
        missing.append(VOICE_CLONE_MODEL_INSTALLATION)
    return missing


def voice_clone_dependencies_available() -> bool:
    return not missing_voice_clone_dependencies()
