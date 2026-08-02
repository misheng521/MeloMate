import hashlib
import os
from pathlib import Path
from uuid import uuid4

import requests
from loguru import logger
from tqdm import tqdm


SENSEVOICE_REVISION = "2365baeacb507f821a0c8120fcee3d484dba7a07"
SENSEVOICE_FILES = {
    "model.int8.onnx": {
        "expected_sha256": (
            "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"
        ),
        "expected_size": 239_233_841,
    },
    "tokens.txt": {
        "expected_sha256": (
            "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc"
        ),
        "expected_size": 315_894,
    },
}
DOWNLOAD_CONNECT_TIMEOUT_SECONDS = 15
DOWNLOAD_READ_TIMEOUT_SECONDS = 60
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def project_backend_models_dir() -> Path:
    """Return <project-root>/models/backend regardless of the current cwd."""
    return Path(__file__).resolve().parents[4] / "models" / "backend"


def _file_matches(path: Path, expected_sha256: str, expected_size: int) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def download_file(
    url: str,
    file_path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> Path:
    """Download a pinned model file, verify it, then atomically replace the target."""
    if not url.startswith("https://huggingface.co/"):
        raise ValueError("Model downloads must use the trusted Hugging Face HTTPS host")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_name(f".{file_path.name}.{uuid4().hex}.part")
    logger.info(f"Downloading verified model file: {file_path.name}")
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(DOWNLOAD_CONNECT_TIMEOUT_SECONDS, DOWNLOAD_READ_TIMEOUT_SECONDS),
        ) as response:
            response.raise_for_status()
            declared_size = response.headers.get("content-length")
            if declared_size and int(declared_size) != expected_size:
                raise ValueError(
                    f"Unexpected Content-Length for {file_path.name}: {declared_size}"
                )

            digest = hashlib.sha256()
            downloaded = 0
            with (
                temp_path.open("xb") as destination,
                tqdm(
                    desc=file_path.name,
                    total=expected_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > expected_size:
                        raise ValueError(
                            f"Download exceeded expected size for {file_path.name}"
                        )
                    destination.write(chunk)
                    digest.update(chunk)
                    progress.update(len(chunk))
                destination.flush()
                os.fsync(destination.fileno())

        if downloaded != expected_size:
            raise ValueError(
                f"Incomplete download for {file_path.name}: "
                f"{downloaded}/{expected_size} bytes"
            )
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"SHA-256 verification failed for {file_path.name}")

        os.replace(temp_path, file_path)
        logger.info(f"Verified model file installed: {file_path.name}")
        return file_path
    finally:
        temp_path.unlink(missing_ok=True)


def ensure_sense_voice_minimal_model(output_dir: str | Path | None = None) -> Path:
    """Ensure the exact, verified SenseVoice runtime files are available."""
    model_dir_name = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    backend_models_dir = (
        Path(output_dir) if output_dir is not None else project_backend_models_dir()
    )
    if not backend_models_dir.is_absolute():
        backend_models_dir = project_backend_models_dir()
    model_dir = backend_models_dir / model_dir_name
    model_file = model_dir / "model.int8.onnx"
    tokens_file = model_dir / "tokens.txt"

    model_ok = _file_matches(model_file, **SENSEVOICE_FILES[model_file.name])
    tokens_ok = _file_matches(tokens_file, **SENSEVOICE_FILES[tokens_file.name])
    if model_ok and tokens_ok:
        logger.info(f"Verified SenseVoice model is ready: {model_dir}")
        return model_dir

    base_url = (
        f"https://huggingface.co/csukuangfj/{model_dir_name}/resolve/"
        f"{SENSEVOICE_REVISION}"
    )
    for target, is_valid in ((model_file, model_ok), (tokens_file, tokens_ok)):
        if is_valid:
            continue
        if target.exists():
            logger.warning(
                f"Existing SenseVoice file failed integrity verification: {target.name}"
            )
        download_file(
            f"{base_url}/{target.name}",
            target,
            **SENSEVOICE_FILES[target.name],
        )

    return model_dir


if __name__ == "__main__":
    ensure_sense_voice_minimal_model()
