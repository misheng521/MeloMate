"""Static project checks and the local background code execution API."""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import workspace_core as workspace
from local_execution import runtime_info, run_command, run_command_async


def validate_project(persona: str, folder: str = "") -> dict:
    root = workspace.workspace_path(persona, folder)
    if not root.is_dir():
        raise ValueError("Project directory does not exist")
    checked, errors, skipped = [], [], []
    deadline = time.monotonic() + 20
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {".git", ".control", ".trash", ".runtime", "node_modules", ".venv", "__pycache__"}
                   and not workspace._is_reparse_point(Path(current) / d)]
        for name in files:
            if len(checked) + len(skipped) + len(errors) >= 500 or time.monotonic() >= deadline:
                return {"ok": not errors, "checked": checked, "errors": errors, "skipped": skipped, "truncated": True, "scope": "syntax only"}
            path = Path(current) / name
            rel = path.relative_to(root).as_posix()
            if workspace._is_reparse_point(path):
                skipped.append(rel + ": link"); continue
            if path.stat().st_size > 1024 * 1024:
                skipped.append(rel + ": too large"); continue
            try:
                if path.suffix == ".py":
                    ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
                elif path.suffix == ".json":
                    json.loads(path.read_text(encoding="utf-8-sig"))
                elif path.suffix in {".js", ".mjs", ".cjs"} and shutil.which("node"):
                    # --check parses stdin without loading project packages or executing code.
                    checked_node = subprocess.run([shutil.which("node"), "--check", "--input-type=" + ("commonjs" if path.suffix == ".cjs" else "module")],
                        input=path.read_text(encoding="utf-8-sig"), capture_output=True, text=True, timeout=max(1, min(15, deadline - time.monotonic())),
                        env={k: v for k, v in os.environ.items() if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP"}},
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    if checked_node.returncode:
                        raise ValueError(checked_node.stderr[:3000])
                else:
                    skipped.append(rel + ": no static checker"); continue
                checked.append(rel)
            except (ValueError, SyntaxError, UnicodeError, OSError, subprocess.TimeoutExpired) as exc:
                errors.append({"path": rel, "error": str(exc)[:3000]})
    return {"ok": not errors, "checked": checked, "errors": errors, "skipped": skipped,
            "scope": "Syntax checks only. This does not prove application behavior; run the project tests with run_workspace_command."}
