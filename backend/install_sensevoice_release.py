import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR / "src"))

from open_llm_vtuber.asr.utils import SENSEVOICE_FILES, _file_matches


ARCHIVE_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.7z"
ARCHIVE_SHA256 = "104a9435c8458df689a80774354a757445fb88773d30fcd2a13b14d4cce59a63"
ARCHIVE_SIZE = 154_358_274
MODEL_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
MAX_ARCHIVE_ENTRIES = 1_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_member_names(tar_executable: str, archive: Path) -> list[str]:
    result = subprocess.run(
        [tar_executable, "-tf", str(archive)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not names or len(names) > MAX_ARCHIVE_ENTRIES:
        raise RuntimeError("SenseVoice archive has an invalid entry count")
    for raw_name in names:
        normalized = raw_name.replace("\\", "/")
        member = PurePosixPath(normalized)
        if (
            member.is_absolute()
            or re.match(r"^[A-Za-z]:", normalized)
            or ".." in member.parts
            or "\x00" in normalized
        ):
            raise RuntimeError(f"Unsafe archive member rejected: {raw_name}")
    return names


def _reject_links(extracted_root: Path) -> None:
    for path in extracted_root.rglob("*"):
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        reparse_point = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if path.is_symlink() or (reparse_point and attributes & reparse_point):
            raise RuntimeError("SenseVoice archive contains a link or reparse point")
        path.resolve().relative_to(extracted_root.resolve())


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
    try:
        with source.open("rb") as input_file, temporary.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
        if not _file_matches(temporary, **SENSEVOICE_FILES[destination.name]):
            raise RuntimeError(f"Extracted model verification failed: {destination.name}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def install(archive: Path) -> Path:
    archive = archive.resolve(strict=True)
    if archive.name != ARCHIVE_NAME:
        raise ValueError(f"Expected archive name: {ARCHIVE_NAME}")
    if archive.stat().st_size != ARCHIVE_SIZE or _sha256(archive) != ARCHIVE_SHA256:
        raise RuntimeError("SenseVoice Release archive integrity verification failed")

    tar_executable = shutil.which("tar")
    if not tar_executable:
        raise RuntimeError("Windows tar.exe was not found")
    _validated_member_names(tar_executable, archive)

    models_root = (PROJECT_ROOT / "models" / "backend").resolve()
    models_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sensevoice-", dir=models_root) as temp:
        extracted_root = Path(temp).resolve()
        subprocess.run(
            [tar_executable, "-xf", str(archive), "-C", str(extracted_root)],
            check=True,
            timeout=300,
        )
        _reject_links(extracted_root)
        candidates = [
            path
            for path in extracted_root.rglob("model.int8.onnx")
            if path.parent.name == MODEL_DIR_NAME
        ]
        if len(candidates) != 1:
            raise RuntimeError("SenseVoice archive has an unexpected directory layout")
        source_dir = candidates[0].parent
        for file_name, expected in SENSEVOICE_FILES.items():
            if not _file_matches(source_dir / file_name, **expected):
                raise RuntimeError(f"SenseVoice archive file is invalid: {file_name}")

        target_dir = models_root / MODEL_DIR_NAME
        for file_name in SENSEVOICE_FILES:
            _atomic_copy(source_dir / file_name, target_dir / file_name)
    return target_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", nargs="?", default=str(PROJECT_ROOT / ARCHIVE_NAME))
    args = parser.parse_args()
    target = install(Path(args.archive))
    print(f"Verified SenseVoice model installed: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
