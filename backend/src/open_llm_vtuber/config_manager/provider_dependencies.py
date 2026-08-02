"""Dependency checks for providers that are not part of the core installation."""

from importlib.util import find_spec


_OPTIONAL_DEPENDENCIES: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ("agent", "letta_agent"): (("letta_client", "letta-client"),),
    ("llm", "llama_cpp_llm"): (("llama_cpp", "llama-cpp-python"),),
    ("asr", "azure_asr"): (("azure.cognitiveservices.speech", "azure-cognitiveservices-speech"),),
    ("asr", "faster_whisper"): (("faster_whisper", "faster-whisper"),),
    ("asr", "whisper_cpp"): (("pywhispercpp", "pywhispercpp"),),
    ("asr", "whisper"): (("whisper", "openai-whisper"),),
    ("asr", "fun_asr"): (
        ("funasr", "funasr"),
        ("modelscope", "modelscope"),
        ("torch", "torch"),
    ),
    ("asr", "groq_whisper_asr"): (("groq", "groq"),),
    ("tts", "azure_tts"): (("azure.cognitiveservices.speech", "azure-cognitiveservices-speech"),),
    ("tts", "bark_tts"): (("bark", "bark"), ("scipy", "scipy")),
    ("tts", "cosyvoice_tts"): (("gradio_client", "gradio-client"),),
    ("tts", "cosyvoice2_tts"): (("gradio_client", "gradio-client"),),
    ("tts", "melo_tts"): (("melo", "MeloTTS"), ("nltk", "nltk")),
    ("tts", "coqui_tts"): (("TTS", "coqui-tts"), ("torch", "torch")),
    ("tts", "fish_api_tts"): (("fish_audio_sdk", "fish-audio-sdk"),),
    ("tts", "spark_tts"): (("gradio_client", "gradio-client"),),
    ("tts", "elevenlabs_tts"): (("elevenlabs", "elevenlabs"),),
    ("tts", "cartesia_tts"): (("cartesia", "cartesia"),),
    ("tts", "piper_tts"): (("piper", "piper-tts"),),
    ("vad", "silero_vad"): (("torch", "torch"),),
}


def require_provider_dependencies(category: str, provider: str) -> None:
    """Reject an optional provider at validation time when its package is absent."""
    def is_missing(module: str) -> bool:
        try:
            return find_spec(module) is None
        except (ImportError, ModuleNotFoundError, AttributeError):
            return True

    missing = [
        package
        for module, package in _OPTIONAL_DEPENDENCIES.get((category, provider), ())
        if is_missing(module)
    ]
    if missing:
        packages = ", ".join(sorted(set(missing)))
        raise ValueError(
            f"Provider '{provider}' is selected but its optional dependency is not installed: {packages}"
        )
