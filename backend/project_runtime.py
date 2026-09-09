"""Project checks and resource-limited execution. Never execute project code on host.

Docker is optional and provisioned by the user. No automatic installation or image
pull is performed. Only a verified project directory is mounted into the container.
"""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from uuid import uuid4

import workspace_core as workspace


def runtime_info() -> dict:
    executable = shutil.which("docker")
    image = os.getenv("MELOMATE_RUNNER_IMAGE", "melomate-runner:local")
    if not executable:
        return {"available": False, "reason": "Docker is not installed. Static checks remain available; project code will not be executed on the host.", "image": image}
    try:
        result = subprocess.run([executable, "image", "inspect", image], capture_output=True, timeout=10,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"available": result.returncode == 0, "image": image,
                "reason": "ready" if result.returncode == 0 else "Start Docker and build the supplied project runner image."}
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "image": image, "reason": "Docker is not available or did not respond."}


def validate_project(persona: str, folder: str = "") -> dict:
    root = workspace.workspace_path(persona, folder)
    if not root.is_dir():
        raise ValueError("Project directory does not exist")
    checked, errors, skipped = [], [], []
    deadline = time.monotonic() + 20
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {".git", ".control", ".trash", "node_modules", ".venv", "__pycache__"}
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
            "scope": "Syntax checks only. This does not prove application behavior; run tests in the isolated runner."}


def command_argv(persona: str, cwd: str, argv: list[str], image: str, container: str, network: bool = False) -> list[str]:
    root = workspace.workspace_path(persona, cwd)
    if not root.is_dir() or "," in str(root):
        raise ValueError("Invalid project directory")
    if not isinstance(argv, list) or not argv or len(argv) > 100 or any(not isinstance(v, str) or "\0" in v or len(v) > 16000 for v in argv):
        raise ValueError("argv must be a nonempty list of bounded strings")
    # No host env, Docker socket, device, privileged flag or extra mounts.
    return [shutil.which("docker") or "docker", "run", "--rm", "--pull=never", "--name", container,
            "--network=bridge" if network else "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit=128", "--memory=1g", "--cpus=2", "--user=65534:65534",
            "--tmpfs=/tmp:rw,nosuid,size=256m", "--env=HOME=/tmp", "--workdir=/workspace",
            "--mount", f"type=bind,source={root},target=/workspace",
            "--tmpfs=/workspace/.control:ro,noexec,nosuid,size=1m",
            "--tmpfs=/workspace/.trash:ro,noexec,nosuid,size=1m", image, *argv]


def run_command(persona: str, argv: list[str], cwd: str = "", timeout_seconds: int = 120, network: bool = False) -> dict:
    info = runtime_info()
    if not info["available"]:
        return {"ok": False, "executed": False, "error": info["reason"], "runtime": info}
    container = "melomate-" + uuid4().hex
    command = command_argv(persona, cwd, argv, info["image"], container, network is True)
    timeout = min(600, max(1, int(timeout_seconds)))
    output = bytearray()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    def drain():
        while chunk := process.stdout.read(4096):
            remaining = 64000 - len(output)
            if remaining > 0: output.extend(chunk[:remaining])
    reader = threading.Thread(target=drain, daemon=True); reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        # Kill the named container, not just the docker client (including on timeout).
        try:
            cleanup = subprocess.run([command[0], "rm", "-f", container], capture_output=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            cleanup_failed = cleanup.returncode != 0 and timed_out
        except (OSError, subprocess.TimeoutExpired):
            cleanup_failed = True
        finally:
            if process.poll() is None: process.kill()
            process.wait(timeout=5); reader.join(timeout=5)
            if not reader.is_alive(): process.stdout.close()
    return {"ok": process.returncode == 0 and not timed_out, "executed": True,
            "exit_code": process.returncode, "timed_out": timed_out,
            "cleanup_failed": cleanup_failed, "container": container,
            "network_enabled": network is True,
            "output": output.decode("utf-8", errors="replace"), "output_truncated": len(output) >= 64000}
