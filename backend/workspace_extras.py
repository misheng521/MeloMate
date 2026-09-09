"""Bounded copy and ZIP tools, with atomic publication and no link traversal."""
from __future__ import annotations
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
import workspace_core as ws

MAX_FILES = 2000
MAX_TOTAL = 64 * 1024 * 1024


def files_under(persona, source):
    if source.is_file(): return [(source, source.name)]
    result, total = [], 0
    for current, dirs, names in os.walk(source, followlinks=False):
        if Path(current) != source:
            if len(result) >= MAX_FILES: raise ValueError("Copy/archive item limit reached")
            result.append((Path(current), Path(current).relative_to(source).as_posix() + "/"))
        for name in dirs + names:
            path = Path(current) / name
            ws._reject_reparse_components(ws.persona_root(persona), path)
        dirs[:] = [d for d in dirs if d.casefold() not in ws.RESERVED_WORKSPACE_PARTS]
        for name in names:
            if name.casefold() in ws.RESERVED_WORKSPACE_PARTS: continue
            path = Path(current) / name
            total += path.stat().st_size
            if total > MAX_TOTAL or len(result) >= MAX_FILES: raise ValueError("Copy/archive size limit reached")
            result.append((path, path.relative_to(source).as_posix()))
    return result


@ws._locked_workspace_mutation
def copy_item(persona, source, destination):
    src, dst = ws.workspace_path(persona, source), ws.workspace_path(persona, destination)
    if not src.exists(): raise ValueError("Source does not exist")
    if dst.exists() or src == dst or src in dst.parents: raise ValueError("Destination must be new and outside the source")
    files = files_under(persona, src)
    if sum(path.stat().st_size for path, _ in files) > MAX_TOTAL: raise ValueError("Copy exceeds 64 MiB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".copy-", dir=dst.parent) as temporary:
        stage = Path(temporary) / "item"
        if src.is_file(): shutil.copyfile(src, stage)
        else:
            stage.mkdir()
            for path, rel in files:
                target = stage / rel
                if path.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        if dst.exists(): raise FileExistsError("Destination appeared while copying")
        stage.rename(dst)
    return {"ok": True, "path": dst.relative_to(ws.WORKSPACE_ROOT).as_posix(), "files": len(files)}


@ws._locked_workspace_mutation
def archive(persona, source, destination):
    src, dst = ws.workspace_path(persona, source), ws.workspace_path(persona, destination)
    if not src.exists() or dst.exists() or dst.suffix.lower() != ".zip": raise ValueError("Use an existing source and a new .zip destination")
    files = files_under(persona, src)
    if sum(path.stat().st_size for path, _ in files) > MAX_TOTAL: raise ValueError("Archive exceeds 64 MiB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".zip-", dir=dst.parent) as temporary:
        staged = Path(temporary) / "archive.zip"
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as output:
            for path, rel in files: output.write(path, rel)
        if dst.exists(): raise FileExistsError("Destination appeared while archiving")
        staged.rename(dst)
    return {"ok": True, "path": dst.relative_to(ws.WORKSPACE_ROOT).as_posix(), "files": len(files)}


@ws._locked_workspace_mutation
def extract(persona, path, destination):
    src, dst = ws.workspace_path(persona, path), ws.workspace_path(persona, destination)
    if not src.is_file() or src.stat().st_size > MAX_TOTAL or dst.exists(): raise ValueError("Use a ZIP within 64 MiB and a new destination directory")
    with zipfile.ZipFile(src) as archive:
        members = archive.infolist()
        if len(members) > MAX_FILES or sum(i.file_size for i in members) > MAX_TOTAL: raise ValueError("Expanded archive exceeds limits")
        planned, seen = [], set()
        for item in members:
            # Do not use persona-prefix stripping on archive member names.
            parts = ws.clean_workspace_parts("__archive__", item.filename)
            if not parts: raise ValueError("Invalid archive member")
            key = "/".join(parts).casefold()
            mode = item.external_attr >> 16
            if key in seen or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
                raise ValueError("Archive contains duplicate or special entries")
            seen.add(key)
            planned.append((item, parts))
        dst.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=dst.parent) as temporary:
            stage = Path(temporary) / "contents"
            stage.mkdir()
            total = 0
            for item, parts in planned:
                target = stage.joinpath(*parts)
                if item.is_dir(): target.mkdir(parents=True, exist_ok=True); continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("xb") as output:
                    while chunk := source.read(65536):
                        total += len(chunk)
                        if total > MAX_TOTAL: raise ValueError("Expanded archive exceeds limits")
                        output.write(chunk)
            if dst.exists(): raise FileExistsError("Destination appeared while extracting")
            stage.rename(dst)
    return {"ok": True, "path": dst.relative_to(ws.WORKSPACE_ROOT).as_posix(), "files": len(planned)}


@ws._locked_workspace_mutation
def save_download(persona, path, data):
    dst = ws.workspace_path(persona, path)
    if len(data) > 16 * 1024 * 1024 or dst.exists(): raise ValueError("Download requires a new path and at most 16 MiB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=dst.parent) as temporary:
        stage = Path(temporary) / "file"
        stage.write_bytes(data)
        if dst.exists(): raise FileExistsError("Destination appeared while downloading")
        stage.rename(dst)
    return {"ok": True, "path": dst.relative_to(ws.WORKSPACE_ROOT).as_posix(), "bytes": len(data)}
