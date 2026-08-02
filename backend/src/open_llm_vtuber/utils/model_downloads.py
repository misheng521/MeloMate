import os
import hashlib
from pathlib import Path
from typing import Final


VOICE_CLONE_MODEL_SPECS: Final = {
    "k2-fsa/OmniVoice": {
        "revision": "c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
        "required_files": {
            "config.json": (
                2_238,
                "5e359117e13b420c5e0c925d4aba650d624767131f1d1746928f8b850d5dc372",
            ),
            "model.safetensors": (
                2_450_344_112,
                "730839316de585f4c8298ec0e1712efc10fb19c6fa4e36eb741cb8d51ebcf6aa",
            ),
            "audio_tokenizer/model.safetensors": (
                805_665_628,
                "fe7c5e8785e0a05833e1bfc3e002ec7f55af21e306b2e7154a448c1f54ccfb0d",
            ),
            "audio_tokenizer/preprocessor_config.json": (
                206,
                "ae61eea88558608ee2fa86d2aec9fce8d99a5ff75d09cb7651ccce21ae1d9084",
            ),
            "tokenizer.json": (
                11_423_986,
                "408f669b7e2b045fdf54201d815bd364e6667dbd845115da81239c40bc6dcfd1",
            ),
        },
    },
    "openai/whisper-large-v3-turbo": {
        "revision": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "required_files": {
            "config.json": (
                1_256,
                "c5b526b3e3cd64cd8940dabb45e8ba726629e22d8ed389c29b552f9140daf04a",
            ),
            "model.safetensors": (
                1_617_824_864,
                "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1",
            ),
            "preprocessor_config.json": (
                340,
                "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711",
            ),
        },
    },
}
HF_ETAG_TIMEOUT_SECONDS: Final = 30
HF_DOWNLOAD_TIMEOUT_SECONDS: Final = 60
_HASH_CACHE: dict[str, tuple[int, int, str]] = {}


def project_backend_models_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "models" / "backend"


def _spec(repo_id: str) -> dict:
    try:
        return VOICE_CLONE_MODEL_SPECS[repo_id]
    except KeyError as exc:
        raise ValueError(f"Unapproved model repository: {repo_id}") from exc


def _repo_cache(cache_root: Path, repo_id: str) -> Path:
    return cache_root / "hub" / f"models--{repo_id.replace('/', '--')}"


def _file_has_sha256(path: Path, expected_size: int, expected_sha256: str) -> bool:
    if not path.is_file():
        return False
    stat_result = path.stat()
    if stat_result.st_size != expected_size:
        return False
    cache_key = str(path.resolve())
    cached = _HASH_CACHE.get(cache_key)
    fingerprint = (stat_result.st_size, stat_result.st_mtime_ns)
    if cached and cached[:2] == fingerprint:
        return cached[2] == expected_sha256
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    _HASH_CACHE[cache_key] = (*fingerprint, actual)
    return actual == expected_sha256


def _validate_snapshot(repo_id: str, snapshot: Path, cache_root: Path) -> str:
    spec = _spec(repo_id)
    repo_cache = _repo_cache(cache_root, repo_id).resolve()
    expected_snapshot = (repo_cache / "snapshots" / spec["revision"]).resolve()
    if snapshot.resolve() != expected_snapshot or not expected_snapshot.is_dir():
        raise RuntimeError(f"Pinned model snapshot is missing for {repo_id}")

    for relative_name, (expected_size, expected_sha256) in spec[
        "required_files"
    ].items():
        model_file = expected_snapshot / relative_name
        resolved_file = model_file.resolve()
        try:
            resolved_file.relative_to(repo_cache)
        except ValueError as exc:
            raise RuntimeError(
                f"Model cache file escapes its repository cache: {relative_name}"
            ) from exc
        if not _file_has_sha256(model_file, expected_size, expected_sha256):
            raise RuntimeError(
                f"Pinned model snapshot failed verification for {repo_id}: {relative_name}"
            )
    return str(expected_snapshot)


def local_hf_snapshot(repo_id: str, cache_root: Path | None = None) -> str | None:
    root = (cache_root or project_backend_models_dir()).resolve()
    spec = _spec(repo_id)
    snapshot = _repo_cache(root, repo_id) / "snapshots" / spec["revision"]
    if not snapshot.is_dir():
        return None
    return _validate_snapshot(repo_id, snapshot, root)


def download_hf_snapshot(repo_id: str, cache_root: Path | None = None) -> str:
    """Download an allowlisted, immutable HF snapshot into its atomic cache."""
    from huggingface_hub import snapshot_download

    root = (cache_root or project_backend_models_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = _spec(repo_id)
    os.environ["HF_HOME"] = str(root)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(HF_ETAG_TIMEOUT_SECONDS))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(HF_DOWNLOAD_TIMEOUT_SECONDS))
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=spec["revision"],
            cache_dir=str(root / "hub"),
            etag_timeout=HF_ETAG_TIMEOUT_SECONDS,
            max_workers=4,
        )
    )
    return _validate_snapshot(repo_id, snapshot, root)


def ensure_voice_clone_models(download_missing: bool) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for repo_id in VOICE_CLONE_MODEL_SPECS:
        snapshot = local_hf_snapshot(repo_id)
        if snapshot is None:
            if not download_missing:
                raise RuntimeError(
                    f"Required voice-clone model is not installed: {repo_id}. "
                    "Run download-omnivoice-model.bat first."
                )
            snapshot = download_hf_snapshot(repo_id)
        resolved[repo_id] = snapshot
    return resolved
