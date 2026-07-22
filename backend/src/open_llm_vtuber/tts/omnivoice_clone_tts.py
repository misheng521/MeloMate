import asyncio
import os
import re
import threading
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from loguru import logger

from .tts_interface import TTSInterface


STYLE_SPEED_MULTIPLIERS = {
    "normal": 1.0,
    "happy": 1.08,
    "shy": 0.94,
    "sad": 0.88,
    "excited": 1.16,
}


def _project_models_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "models" / "backend"


def _local_hf_snapshot(repo_id: str) -> Optional[str]:
    cache_root = Path(os.environ.get("HF_HOME") or _project_models_dir())
    repo_cache = cache_root / "hub" / f"models--{repo_id.replace('/', '--')}"
    ref_file = repo_cache / "refs" / "main"
    if ref_file.exists():
        revision = ref_file.read_text(encoding="utf-8").strip()
        snapshot = repo_cache / "snapshots" / revision
        if snapshot.is_dir():
            return str(snapshot)
    snapshots_dir = repo_cache / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if snapshots:
            return str(snapshots[0])
    return None


def _download_hf_snapshot(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    cache_root = _project_models_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    return snapshot_download(repo_id=repo_id, cache_dir=str(cache_root / "hub"))


class TTSEngine(TTSInterface):
    """OmniVoice voice-cloning TTS wrapper for MeloMate."""

    def __init__(
        self,
        model: str = "k2-fsa/OmniVoice",
        device: Optional[str] = None,
        num_step: int = 16,
        guidance_scale: float = 2.0,
        speed: float = 1.0,
    ):
        self.model_name = model
        self.device = device
        self.num_step = num_step
        self.guidance_scale = guidance_scale
        self.speed = speed
        self.enabled = False
        self.ref_audio_path = ""
        self.ref_text: Optional[str] = None
        self.language: Optional[str] = None
        self._model = None
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()

    def configure(
        self,
        enabled: bool,
        ref_audio_path: str = "",
        ref_text: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text or None
        self.language = language or None

    def is_ready(self) -> bool:
        return self.enabled and bool(self.ref_audio_path) and Path(self.ref_audio_path).exists()

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            model_cache = _project_models_dir()
            os.environ.setdefault("HF_HOME", str(model_cache))
            os.environ.setdefault("MODELSCOPE_CACHE", str(model_cache))
            from omnivoice import OmniVoice
            from omnivoice.utils.common import get_best_device
        except Exception as exc:
            raise RuntimeError(
                "OmniVoice dependencies are not installed in MeloMate backend\\.venv. "
                "Install omnivoice, torchaudio, transformers, accelerate and librosa "
                "before enabling voice cloning."
            ) from exc

        device = self.device or get_best_device()
        dtype = torch.float16 if "cuda" in str(device).lower() else torch.float32
        model_path = _local_hf_snapshot(self.model_name)
        if model_path is None:
            logger.info(
                f"OmniVoice model {self.model_name} was not found in local cache. "
                "Downloading to MeloMate models/backend..."
            )
            model_path = _download_hf_snapshot(self.model_name)
        load_asr = self.ref_text is None and _local_hf_snapshot("openai/whisper-large-v3-turbo") is not None
        logger.info(f"Loading OmniVoice model {model_path} on {device}")
        self._model = OmniVoice.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            load_asr=load_asr,
        )
        return self._model

    async def async_generate_audio(
        self,
        text: str,
        file_name_no_ext=None,
        voice_style: Optional[str] = None,
        voice_style_key: str = "normal",
    ) -> str:
        async with self._async_lock:
            return await asyncio.to_thread(
                self.generate_audio,
                text,
                file_name_no_ext,
                voice_style,
                voice_style_key,
            )

    def generate_audio(
        self,
        text: str,
        file_name_no_ext=None,
        voice_style: Optional[str] = None,
        voice_style_key: str = "normal",
    ) -> str:
        if not self.is_ready():
            raise RuntimeError("Voice cloning is enabled, but no valid reference audio is selected.")
        if self.ref_text is None:
            raise RuntimeError(
                "Voice cloning needs the transcript of the reference audio. "
                "Fill in the reference text in Settings, or install the Whisper "
                "model cache openai/whisper-large-v3-turbo for automatic transcription."
            )

        with self._sync_lock:
            model = self._load_model()
            output_path = self.generate_cache_file_name(file_name_no_ext, "wav")
            synth_text = add_emotion_tag(text)
            speed = self.speed * STYLE_SPEED_MULTIPLIERS.get(voice_style_key, 1.0)

            logger.info(
                f"Generating OmniVoice clone audio with style={voice_style_key}, "
                f"speed={speed:.2f}, steps={self.num_step} for: {text[:80]}"
            )
            audios = model.generate(
                text=synth_text,
                language=self.language,
                ref_audio=self.ref_audio_path,
                ref_text=self.ref_text,
                instruct=voice_style,
                num_step=self.num_step,
                guidance_scale=self.guidance_scale,
                speed=speed,
            )
            sf.write(output_path, audios[0], model.sampling_rate)
            return output_path


def add_emotion_tag(text: str) -> str:
    clean = text.strip()
    if not clean or clean.startswith("["):
        return clean

    if re.search(r"(哈哈|笑死|好玩|有趣|开心|高兴|太棒|不错|厉害|cute|funny|happy|great|nice)", clean, re.I):
        return f"[laughter] {clean}"
    if re.search(r"(唉|哎|难过|伤心|可惜|累|抱歉|对不起|遗憾|sad|sorry|tired)", clean, re.I):
        return f"[sigh] {clean}"
    # Avoid question/surprise non-verbal tags here: OmniVoice realizes
    # [question-ah] and [surprise-ah] as an audible "ah" before the sentence.
    if re.search(r"(嗯|好的|没问题|可以|明白|对|是的|ok|okay|yes)", clean, re.I):
        return f"[confirmation-en] {clean}"
    if re.search(r"(不行|不要|讨厌|生气|烦|糟糕|bad|angry|annoy)", clean, re.I):
        return f"[dissatisfaction-hnn] {clean}"
    return clean
